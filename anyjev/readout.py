"""Readout: build prompts for a (state, question, permutation) and map option
labels to the single tokens whose logits we read at the answer position."""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import List, Sequence

from anyjev.question import Question

LETTERS = string.ascii_uppercase

DEFAULT_SYSTEM = (
    "You are a decision function. You will be given a state and one question. "
    "Reply with the answer label only: no words, no punctuation, no explanation."
)


class LabelTokenError(ValueError):
    pass


def answer_labels(q: Question) -> List[str]:
    """The strings the model must emit. For choice and score they name a
    *position* (A, B, C / 1, 2, 3); for noul they name the *option* itself
    (Yes, No) and stay attached to it when the phrasing order changes."""
    if q.kind == "noul":
        return ["Yes", "No"]
    if q.kind == "score":
        return [str(i + 1) for i in range(q.k)]
    return list(LETTERS[: q.k])


def labels_are_positional(q: Question) -> bool:
    return q.kind != "noul"


def label_ids_for_perm(q: Question, ids: Sequence[int], perm: Sequence[int]) -> List[int]:
    """Token ids in *position* order for this permutation, so the backend's
    output index j always means "the option shown at position j"."""
    if labels_are_positional(q):
        return list(ids)
    return [ids[i] for i in perm]


def map_label_tokens(tokenizer, labels: Sequence[str]) -> List[int]:
    """Map each label string to exactly one token id.

    Tries the bare string, then a leading-space variant. Refuses multi-token
    labels and collisions instead of silently reading the wrong logit.
    """
    ids: List[int] = []
    for lab in labels:
        found = None
        for cand in (lab, " " + lab):
            toks = tokenizer.encode(cand, add_special_tokens=False)
            if len(toks) == 1:
                found = toks[0]
                break
        if found is None:
            raise LabelTokenError(
                f"label {lab!r} is not a single token for this tokenizer; "
                f"got {tokenizer.encode(lab, add_special_tokens=False)}"
            )
        ids.append(found)
    if len(set(ids)) != len(ids):
        raise LabelTokenError(f"label tokens collide: {dict(zip(labels, ids))}")
    return ids


@dataclass
class PromptSpec:
    system: str
    user: str


def build_prompt(state_text: str, q: Question, perm: Sequence[int],
                 system: str = DEFAULT_SYSTEM) -> PromptSpec:
    """perm[j] = index (into q.options) of the option shown at position j."""
    labels = answer_labels(q)
    lines = ["State:", state_text if state_text else "(empty)", "", f"Question: {q.text}"]
    if q.kind == "noul":
        # perm over ("Yes","No") only changes the phrasing order
        order = [labels[i] for i in perm]
        lines.append(f"Answer {order[0]} or {order[1]}.")
    elif q.kind == "score":
        if q.centers is not None:
            lines.append("Pick the level that applies (the levels are ordered):")
        else:
            lo, hi = q.scale
            lines.append(f"Answer on a scale from {lo:g} to {hi:g} by picking the closest bin:")
        for j, i in enumerate(perm):
            lines.append(f"{labels[j]}. {q.options[i]}")
        lines.append("Answer with the number only.")
    else:
        lines.append("Options:")
        for j, i in enumerate(perm):
            lines.append(f"{labels[j]}. {q.options[i]}")
        lines.append("Answer with the letter only.")
    return PromptSpec(system=system, user="\n".join(lines))


def render_chat(tokenizer, spec: PromptSpec) -> str:
    """Render through the tokenizer's chat template with the generation prompt
    appended, so the next token is the model's first answer token."""
    messages = [{"role": "system", "content": spec.system},
                {"role": "user", "content": spec.user}]
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return f"{spec.system}\n\n{spec.user}\nAnswer:"
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
