"""A synthetic backend with known, injectable biases. Used by the unit tests
and by the docs to show what L0 removes. It parses the prompts that
anyjev.readout builds, so it exercises the real prompt path.

Set `image_content` to make it multimodal: the fake then accepts images, keeps
every list it was handed in `images_seen`, and lets the logits depend on which
pictures a prompt carried. No pixels are ever decoded."""
from __future__ import annotations

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


class FakeBackend:
    """logit(position j, option o) = content(state, o) + position_bias[j] + label_prior[label_j]

    content is 0 on content-free probes, so the prior is exactly the bias term.
    """

    def __init__(self, content: Callable[[str, str], float],
                 position_bias: Optional[Sequence[float]] = None,
                 label_prior: Optional[Dict[str, float]] = None,
                 temperature: float = 1.0,
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
        self.calls = 0
        self.prompts_seen = 0
        self.shared_calls = 0
        self.shared_groups = 0
        self.images_seen: List[tuple] = []

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

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]],
                            images: Optional[Sequence[Sequence]] = None) -> List[np.ndarray]:
        self.calls += 1
        self.prompts_seen += len(prompts)
        per_prompt = list(images) if images is not None else [()] * len(prompts)
        self.images_seen.extend(tuple(x) for x in per_prompt)
        return [self._logits(p, ids, imgs)
                for p, ids, imgs in zip(prompts, token_ids, per_prompt)]
