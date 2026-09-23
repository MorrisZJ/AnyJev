"""A synthetic backend with known, injectable biases. Used by the unit tests
and by the docs to show what L0 removes. It parses the prompts that
anyjev.readout builds, so it exercises the real prompt path.

Set `image_content` to make it multimodal: the fake then accepts images, keeps
every list it was handed in `images_seen`, and lets the logits depend on which
pictures a prompt carried. No pixels are ever decoded."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from anyjev.calibrate.contextual import DEFAULT_PROBES
from anyjev.state import MARKER_RE

_OPT_LINE = re.compile(r"^([A-Z]|\d+)\. (.*)$")
_NOUL_LINE = re.compile(r"^Answer (Yes|No) or (Yes|No)\.$")


@lru_cache(maxsize=1)
def _blank_key() -> str:
    from anyjev.media import blank_image

    return blank_image().key


class FakeTokenizer:
    chat_template = None

    def __init__(self):
        self._vocab: Dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # every label we care about is one token; anything else is "long"
        if re.fullmatch(r" ?([A-Z]|\d|Yes|No)", text):
            key = text.strip()
            if key not in self._vocab:
                self._vocab[key] = 1000 + len(self._vocab)
            return [self._vocab[key]]
        return [1, 2]

    def id_to_label(self, tid: int) -> str:
        for k, v in self._vocab.items():
            if v == tid:
                return k
        raise KeyError(tid)

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False,
                 **_: object) -> Dict[str, list]:
        """Word-level tokens with character offsets (a fast tokenizer's `return_offsets_mapping`), so
        `anyjev.readout.option_token_positions` works on the fake exactly as on a real tokenizer.
        `encode` keeps its own two-token rule for unknown text; the two never mix."""
        spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
        out: Dict[str, list] = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


_WORD = re.compile(r"\S+|\n")


def _hash_vec(text: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-random unit vector for a string (process-independent)."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    v = np.random.RandomState(seed).randn(dim)
    return v / np.linalg.norm(v)


def _hash_normal(text: str) -> float:
    """Deterministic standard-normal scalar for a string (the one-dimensional `_hash_vec`)."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return float(np.random.RandomState(seed).randn())


class FakeBackend:
    """logit(position j, option o) = content(state, o) + position_bias[j] + label_prior[label_j]
                                     + logit_noise * n(state, o, j)

    content is 0 on content-free probes, so the prior is exactly the bias term. `n` is a hashed
    standard normal per (state, option, position), applied to real states only; it is what the
    output layer sees but the planted hidden state does not (dims 38:50 of `_last_vector` carry
    a clean code of the right option), so with `logit_noise > 0` a head fit on the hidden state
    can beat L0, which averages only part of the noise out over the cyclic shifts.

    `wording_shift = a` applies a per-feature affine map to the last-position hidden state,
    `h <- h * (1 + a * u_q) + a * v_q`, with `u_q, v_q` hashed from the question line of the
    prompt: a rewording moves and rescales the hidden state, which label-free recentring undoes
    exactly. Both knobs default to 0.0, which leaves every prompt and vector bit-identical to a
    backend without them.
    """

    def __init__(self, content: Callable[[str, str], float],
                 position_bias: Optional[Sequence[float]] = None,
                 label_prior: Optional[Dict[str, float]] = None,
                 temperature: float = 1.0, logit_noise: float = 0.0, wording_shift: float = 0.0,
                 image_content: Optional[Callable[[tuple, str], float]] = None,
                 accepts_images: bool = False):
        self.name = "fake"
        self.tokenizer = FakeTokenizer()
        self.content = content
        self.position_bias = list(position_bias or [])
        self.label_prior = dict(label_prior or {})
        self.temperature = temperature
        self.image_content = image_content
        self.accepts_images = accepts_images or image_content is not None
        self.logit_noise = float(logit_noise)
        self.wording_shift = float(wording_shift)
        self.calls = 0
        self.prompts_seen = 0
        self.shared_calls = 0
        self.shared_groups = 0
        self.images_seen: List[tuple] = []
        # planted hidden-state geometry (see hidden_states)
        self.hidden_size = 64
        self.n_layers = 4
        self.hidden_calls = 0
        self.hidden_noise = 0.02
        self.layer_noise = 0.4       # extra noise of earlier layers, x (1 - s); demo/jev_mode lowers it
        rng = np.random.RandomState(7)
        self._u = rng.randn(self.hidden_size)                 # content direction (question-agnostic)
        self._u /= np.linalg.norm(self._u)
        self._v = rng.randn(self.hidden_size)                 # position-bias direction
        self._v -= self._v @ self._u * self._u
        self._v /= np.linalg.norm(self._v)

    def _parse(self, prompt: str):
        state = prompt.split("State:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        if state == "(empty)":
            state = ""
        labels, options = [], []
        for line in prompt.splitlines():
            m = _OPT_LINE.match(line)
            if m:
                labels.append(m.group(1))
                options.append(m.group(2))
            m = _NOUL_LINE.match(line)
            if m:
                labels = [m.group(1), m.group(2)]
                options = list(labels)
        return state, labels, options

    def _logits(self, prompt: str, ids: Sequence[int], images: Sequence = ()) -> np.ndarray:
        state, labels, options = self._parse(prompt)
        is_probe = state in DEFAULT_PROBES or self._is_blank_probe(state, images)
        logits = np.zeros(len(ids))
        by_label = {self.tokenizer.id_to_label(t): k for k, t in enumerate(ids)}
        for j, (lab, opt) in enumerate(zip(labels, options)):
            z = 0.0 if is_probe else self.content(state, opt)
            if not is_probe and self.image_content is not None:
                z += self.image_content(tuple(im.key for im in images), opt)
            if j < len(self.position_bias):
                z += self.position_bias[j]
            z += self.label_prior.get(lab, 0.0)
            if self.logit_noise and not is_probe:
                z += self.logit_noise * _hash_normal(f"{state}|{opt}|{j}")
            logits[by_label[lab]] = z / self.temperature
        # full-vocab log-softmax: pretend a bit of mass lives elsewhere
        z = np.concatenate([logits, [-5.0]])
        lp = z - np.log(np.exp(z - z.max()).sum()) - z.max()
        return lp[:-1]

    @staticmethod
    def _is_blank_probe(state: str, images: Sequence) -> bool:
        """A multimodal content-free probe is blank in *both* modalities. The
        text alone cannot tell: a picture-only state renders to "<image 1>",
        which strips to "", which is itself one of the probes."""
        if not images or any(im.key != _blank_key() for im in images):
            return False
        return MARKER_RE.sub("", state).strip() in DEFAULT_PROBES

    def score_shared(self, groups, token_ids):
        """Same answers as the flat path; exists so the Decider's grouping is testable."""
        self.shared_calls += 1
        self.shared_groups += len(groups)
        return [self.next_token_logprobs([pre + suf for suf in sfx], [ids] * len(sfx))
                for (pre, sfx), ids in zip(groups, token_ids)]

    # ---- planted hidden states -------------------------------------------------
    def _last_vector(self, state: str, labels: List[str], options: List[str],
                     image_keys: tuple = ()) -> np.ndarray:
        """Last-position state: dims [0:26] carry only the position bias and label prior of each
        position (no content), dims [26:38] a hash embedding of the state, dims [38:50] a hash
        embedding of the option with the largest content. A head fit on this vector for one
        question can only memorise that question's option vocabulary, which is exactly why a
        per-question head does not transfer while the option-line head (`_option_vector`) can."""
        h = np.zeros(self.hidden_size)
        for j, lab in enumerate(labels):
            z = self.position_bias[j] if j < len(self.position_bias) else 0.0
            h[j] = (z + self.label_prior.get(lab, 0.0)) / self.temperature
        h[26:38] = _hash_vec(state, 12)
        if state not in DEFAULT_PROBES and options:
            def total(o: str) -> float:
                seen = self.image_content(image_keys, o) if self.image_content is not None and image_keys else 0.0
                return self.content(state, o) + seen
            best = max(options, key=total)
            h[38:50] = _hash_vec(best, 12)
        return h

    def _wording_affine(self, prompt: str):
        """(scale, shift) of the wording shift for this prompt's question line, or None when the
        knob is off. The question line is the first line after "Question: " (what `build_prompt`
        writes), so a rewording of the question text moves the vector and the options do not."""
        if not self.wording_shift:
            return None
        tail = prompt.split("\n\nQuestion: ", 1)
        line = tail[1].split("\n", 1)[0] if len(tail) == 2 else ""
        a = self.wording_shift
        return 1.0 + a * _hash_vec(line + "|scale", self.hidden_size), a * _hash_vec(line + "|shift", self.hidden_size)

    def _option_vector(self, state: str, j: int, option: str) -> np.ndarray:
        """Option-line state for the option shown at position j: content along a fixed direction
        (the rule a universal head can recover on unseen questions), the position bias along another,
        plus a hash embedding of the option text (what a per-question head can memorise)."""
        z = 0.0 if state in DEFAULT_PROBES else self.content(state, option)
        bias = self.position_bias[j] if j < len(self.position_bias) else 0.0
        g = z * self._u + bias * self._v
        g[38:50] = _hash_vec(option, 12)
        return g

    def _line_of_token(self, prompt: str, token_index: int, kind_noul: bool):
        """(position j, option text) of the option line that contains word token `token_index`,
        or None when the token is not on an option line."""
        spans = [(m.start(), m.end()) for m in _WORD.finditer(prompt)]
        if token_index >= len(spans):
            return None
        pos = spans[token_index][0]
        offset = 0
        j = 0
        for line in prompt.split("\n"):
            end = offset + len(line)
            m = _OPT_LINE.match(line)
            if m and not kind_noul:
                if offset <= pos <= end:
                    return j, m.group(2)
                j += 1
            elif kind_noul:
                m2 = _NOUL_LINE.match(line)
                if m2 and offset <= pos <= end:
                    words = [(offset + m2.start(g), offset + m2.end(g), m2.group(g)) for g in (1, 2)]
                    for jj, (a, b, w) in enumerate(words):
                        if a <= pos <= b or (pos >= b and jj == 1):
                            return jj, w
                    return 0, words[0][2]
            offset = end + 1
        return None

    def hidden_states(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None,
                      token_ids: Optional[Sequence[Sequence[int]]] = None,
                      positions: Optional[Sequence[Sequence[int]]] = None,
                      images: Optional[Sequence[Sequence]] = None):
        """Same contract as HFBackend.hidden_states: (feats [N, L, d] float32 at the last position,
        lps, pos_feats [N, P, L, d] float16 or None). Layer `n_layers` is exact; earlier layers are
        the same vectors scaled down with noise added, so layer selection is testable. With
        `image_content`, the pictures of prompt i (`images[i]`) count toward its planted answer."""
        self.hidden_calls += 1
        per_prompt = [tuple(x) for x in (images if images is not None else [()] * len(prompts))]
        self.images_seen.extend(per_prompt)
        layers = [self.n_layers] if layers is None else [(self.n_layers + 1 + i) if i < 0 else i for i in layers]
        n = len(prompts)
        rng = np.random.RandomState(11)
        feats = np.zeros((n, len(layers), self.hidden_size), dtype=np.float32)
        p_max = max((len(p) for p in positions), default=0) if positions is not None else 0
        pos_feats = None
        if positions is not None:
            pos_feats = np.zeros((n, p_max, len(layers), self.hidden_size), dtype=np.float16)

        def at_layer(vec: np.ndarray, layer: int) -> np.ndarray:
            # the last layer is exact; earlier layers carry a weaker signal under more noise
            s = (float(layer) / float(self.n_layers)) ** 2
            return vec * s + rng.randn(self.hidden_size) * (self.hidden_noise + self.layer_noise * (1.0 - s))

        for i, prompt in enumerate(prompts):
            state, labels, options = self._parse(prompt)
            kind_noul = bool(_NOUL_LINE.search(prompt))
            h = self._last_vector(state, labels, options, tuple(im.key for im in per_prompt[i]))
            affine = self._wording_affine(prompt)
            for li, layer in enumerate(layers):
                f = at_layer(h, layer)
                if affine is not None:
                    # applied to the returned vector (after the layer noise), so the map is exactly
                    # per-feature affine and a re-estimated mean / scale undoes it exactly
                    f = f * affine[0] + affine[1]
                feats[i, li] = f
            if positions is not None:
                for pi, tok in enumerate(positions[i]):
                    hit = self._line_of_token(prompt, int(tok), kind_noul)
                    if hit is None:
                        continue
                    j, opt = hit
                    g = self._option_vector(state, j, opt)
                    for li, layer in enumerate(layers):
                        pos_feats[i, pi, li] = at_layer(g, layer).astype(np.float16)
        lps = (self.next_token_logprobs(prompts, token_ids, images=images) if token_ids is not None
               else [None] * n)
        return feats, lps, pos_feats

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]],
                            images: Optional[Sequence[Sequence]] = None) -> List[np.ndarray]:
        self.calls += 1
        self.prompts_seen += len(prompts)
        per_prompt = list(images) if images is not None else [()] * len(prompts)
        self.images_seen.extend(tuple(x) for x in per_prompt)
        return [self._logits(p, ids, imgs)
                for p, ids, imgs in zip(prompts, token_ids, per_prompt)]
