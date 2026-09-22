"""Decider: the pipeline from (state, questions) to leveled decisions.

raw : one prompt, one permutation, restricted softmax. What every clone does.
L0  : permutation marginalization + label-free prior correction (batch mean
      by default, content-free probes optional). Zero labels.
L1  : L0, then a post-hoc calibrator fit on a labeled set for this question.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anyjev.calibrate.contextual import (
    DEFAULT_PROBES,
    apply_contextual,
    batch_prior,
    content_free_prior,
)
from anyjev.calibrate.permute import cyclic_shifts, flip_rate_across_perms, marginalize
from anyjev.calibrate.posthoc import TemperatureScaler
from anyjev.question import Question
from anyjev.readout import (
    DEFAULT_SYSTEM,
    answer_labels,
    build_prompt,
    label_ids_for_perm,
    map_label_tokens,
    prompt_images,
    render_chat,
)
from anyjev.result import Decision, DecisionSet
from anyjev.state import content_free_state, split_state

LEVELS = ("raw", "L0", "L1")
PRIORS = ("batch", "content_free", "none")


def _softmax(lp: np.ndarray) -> np.ndarray:
    z = np.asarray(lp, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


class Decider:
    def __init__(self, backend, *, level: str = "L0", prior: str = "batch", min_prior_n: int = 8,
                 max_permutations: Optional[int] = None, combine: str = "logmean",
                 cf_probes: Sequence[str] = DEFAULT_PROBES, record_content_free: bool = False,
                 system: str = DEFAULT_SYSTEM):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}")
        self.backend = backend
        self.level = level
        self.prior = prior
        self.min_prior_n = min_prior_n
        self.max_permutations = max_permutations
        self.combine = combine
        self.cf_probes = tuple(cf_probes)
        self.record_content_free = record_content_free
        self.system = system
        self._label_ids: Dict[tuple, List[int]] = {}
        self._artifacts: Dict[str, TemperatureScaler] = {}
        self._running: Dict[str, Tuple[np.ndarray, int]] = {}   # q.key -> (sum p_pos_raw [P,K], n)
        self._cf_cache: Dict[tuple, np.ndarray] = {}            # (q.key, n_images) -> cf prior [P,K]

    @property
    def renderer(self):
        """What holds the chat template: the processor on a multimodal
        backend, the tokenizer otherwise."""
        return getattr(self.backend, "chat_renderer", None) or self.backend.tokenizer

    @property
    def accepts_images(self) -> bool:
        return bool(getattr(self.backend, "accepts_images", False))

    # ---- public -------------------------------------------------------
    def decide(self, state: Any, questions: Sequence[Question], level: Optional[str] = None) -> DecisionSet:
        level = level or self.level
        decs = self._run([state], list(questions), level)
        return DecisionSet([decs[(0, qi)] for qi in range(len(questions))], level)

    def decide_batch(self, states: Sequence[Any], question: Question,
                     level: Optional[str] = None) -> List[Decision]:
        """Many states, one question. The bench path, and the best path for
        batch prior estimation."""
        level = level or self.level
        decs = self._run(list(states), [question], level)
        return [decs[(si, 0)] for si in range(len(states))]

    def calibrate(self, question: Question, states: Sequence[Any], labels: Sequence[int],
                  level: str = "L1") -> Dict[str, Any]:
        """Fit an L1 artifact for this question on labeled states. labels are
        option indices. Returns the artifact dict (store it; it is per model)."""
        if level != "L1":
            raise ValueError("only L1 calibration is implemented")
        decs = self.decide_batch(states, question, level="L0")
        probs = np.stack([d.probs for d in decs])
        scaler = TemperatureScaler.fit(probs, labels)
        self._artifacts[question.key] = scaler
        return {"model": self.backend.name, "question": question.key, **scaler.to_dict()}

    def load_artifact(self, question: Question, artifact: Dict[str, Any]) -> None:
        if artifact.get("model") not in (None, self.backend.name):
            raise ValueError(f"artifact was fit on {artifact['model']}, backend is {self.backend.name}")
        self._artifacts[question.key] = TemperatureScaler.from_dict(artifact)

    def export_artifacts(self) -> Dict[str, Any]:
        """Every L1 artifact this decider holds, keyed by question hash. JSON-serializable."""
        return {"model": self.backend.name,
                "artifacts": {k: {"model": self.backend.name, "question": k, **v.to_dict()}
                              for k, v in self._artifacts.items()}}

    def save_artifacts(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.export_artifacts(), f, indent=1)

    def load_artifacts(self, path_or_dict) -> int:
        """Load artifacts saved by save_artifacts. Refuses artifacts fit on another model."""
        import json
        d = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict))
        if d.get("model") not in (None, self.backend.name):
            raise ValueError(f"artifacts were fit on {d['model']}, backend is {self.backend.name}")
        for key, art in d["artifacts"].items():
            self._artifacts[key] = TemperatureScaler.from_dict(art)
        return len(d["artifacts"])

    def running_prior(self, question: Question) -> Optional[np.ndarray]:
        """The batch prior accumulated so far for this question, [P, K] by position, or None."""
        entry = self._running.get(question.key)
        if entry is None or entry[1] < self.min_prior_n:
            return None
        return batch_prior(entry[0][None] / entry[1])

    # ---- internals ----------------------------------------------------
    def _ids_for(self, q: Question) -> List[int]:
        key = (q.kind, q.k)
        if key not in self._label_ids:
            self._label_ids[key] = map_label_tokens(self.backend.tokenizer, answer_labels(q))
        return self._label_ids[key]

    def _perms(self, q: Question, level: str) -> List[List[int]]:
        if level == "raw" or q.ordered:
            return [list(range(q.k))]
        if q.kind == "noul":
            return [[0, 1], [1, 0]]
        return cyclic_shifts(q.k, self.max_permutations)

    def _run(self, states: List[Any], questions: List[Question], level: str) -> Dict[tuple, Decision]:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        renderer = self.renderer
        rendered = [split_state(s) for s in states]                # [(text, images)]
        want_cf = level != "raw" and (self.prior == "content_free" or self.record_content_free)
        if any(imgs for _, imgs in rendered) and not self.accepts_images:
            raise ValueError(
                f"state carries images but backend {self.backend.name!r} is text-only; "
                "use anyjev.backends.hf_vlm.VLMBackend or another backend with accepts_images")

        # 1. collect every prompt once. A prompt is its text *and* its pictures,
        #    so the same state under K permutations still costs K prefills, not
        #    K x (images decoded again).
        prompt_index: Dict[tuple, int] = {}
        prompt_ids: List[List[int]] = []
        prompt_texts: List[str] = []
        prompt_pictures: List[tuple] = []

        def add(text: str, q: Question, perm: List[int], ids: List[int], images: tuple = ()) -> int:
            spec = build_prompt(text, q, perm, self.system, images)
            rendered_text, ordered = render_chat(renderer, spec), prompt_images(spec)
            key = (rendered_text, tuple(im.key for im in ordered))
            if key not in prompt_index:
                prompt_index[key] = len(prompt_ids)
                prompt_ids.append(ids)
                prompt_texts.append(rendered_text)
                prompt_pictures.append(ordered)
            return prompt_index[key]

        plan = {}
        cf_plan: Dict[int, Dict[int, List[List[int]]]] = {}        # qi -> n_images -> [P][C] rows
        for qi, q in enumerate(questions):
            ids = self._ids_for(q)
            perms = self._perms(q, level)
            perm_ids = [label_ids_for_perm(q, ids, perm) for perm in perms]
            cf_plan[qi] = {}
            for si, (text, images) in enumerate(rendered):
                n_img = len(images)
                # the probe keeps the shape of a real prompt: same question,
                # same picture count, no content in either modality
                if want_cf and (q.key, n_img) not in self._cf_cache and n_img not in cf_plan[qi]:
                    probes = [content_free_state(probe, n_img) for probe in self.cf_probes]
                    cf_plan[qi][n_img] = [[add(cf_text, q, perm, pids, cf_images) for cf_text, cf_images in probes]
                                          for perm, pids in zip(perms, perm_ids)]
                real_rows = [add(text, q, perm, pids, images) for perm, pids in zip(perms, perm_ids)]
                plan[(si, qi)] = (perms, real_rows, n_img)

        # 2. one backend call
        if any(prompt_pictures):
            logprobs = self.backend.next_token_logprobs(prompt_texts, prompt_ids, images=prompt_pictures)
        else:
            logprobs = self.backend.next_token_logprobs(prompt_texts, prompt_ids)

        # 3. per question: raw position-space distributions, priors
        out: Dict[tuple, Decision] = {}
        for qi, q in enumerate(questions):
            perms = plan[(0, qi)][0]
            p_pos_raw_all = []
            for si in range(len(states)):
                lp_real = np.stack([logprobs[r] for r in plan[(si, qi)][1]])   # [P, K]
                p_pos_raw_all.append((lp_real, np.stack([_softmax(lp) for lp in lp_real])))

            for n_img, cf_rows in cf_plan[qi].items():
                self._cf_cache[(q.key, n_img)] = np.stack([
                    content_free_prior(np.stack([_softmax(logprobs[r]) for r in rows]))
                    for rows in cf_rows])                                          # [P, K]
            b_prior = None
            if level != "raw":
                stack = np.stack([p for _, p in p_pos_raw_all])                    # [N, P, K]
                s, n = self._running.get(q.key, (np.zeros(stack.shape[1:]), 0))
                self._running[q.key] = (s + stack.sum(axis=0), n + len(stack))
                b_prior = self.running_prior(q)

            # 4. assemble per state
            for si, (lp_real, p_pos_raw) in enumerate(p_pos_raw_all):
                cf_prior = self._cf_cache.get((q.key, plan[(si, qi)][2])) if want_cf else None
                prior_used = None
                if level != "raw":
                    prior_used = b_prior if self.prior == "batch" else (
                        cf_prior if self.prior == "content_free" else None)
                answer_mass = float(np.exp(lp_real).sum(axis=1).mean())
                raw_probs = marginalize(p_pos_raw[:1], perms[:1])
                diag: Dict[str, Any] = {"answer_mass": answer_mass, "raw_probs": raw_probs,
                                        "permutations": len(perms), "perms": perms,
                                        "p_pos_raw": p_pos_raw, "n_images": plan[(si, qi)][2]}
                if level == "raw":
                    probs, achieved = raw_probs, "raw"
                else:
                    p_pos = apply_contextual(p_pos_raw, prior_used) if prior_used is not None else p_pos_raw
                    probs = marginalize(p_pos, perms, self.combine)
                    achieved = "L0"
                    diag.update({
                        "prior_method": self.prior if prior_used is not None else "none",
                        "prior": prior_used,
                        "cf_prior": cf_prior,
                        "batch_prior": b_prior,
                        "order_flip_raw": flip_rate_across_perms(p_pos_raw, perms),
                        "order_flip_l0": flip_rate_across_perms(p_pos, perms),
                        "l0_probs": probs,
                    })
                    if level == "L1":
                        scaler = self._artifacts.get(q.key)
                        if scaler is None:
                            raise ValueError(f"no L1 artifact for question {q.id}; call calibrate() first")
                        probs = scaler.apply(probs)
                        achieved = "L1"
                        diag["temperature"] = scaler.temperature
                out[(si, qi)] = Decision(q, np.asarray(probs), achieved, diag)
        return out
