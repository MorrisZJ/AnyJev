"""Readout: build prompts for a (state, question, permutation) and map option
labels to the single tokens whose logits we read at the answer position."""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import List, Optional, Sequence

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


def resolve_labels(tokenizer, q: Question):
    """Pick the label strings this tokenizer can emit as single tokens, and their ids.

    Letters for choice, Yes/No for noul, digits for score. If the digits are not
    single tokens (sentencepiece tokenizers split "1" into a space piece plus "1"),
    score falls back to letters. Raises LabelTokenError if nothing works."""
    labels = answer_labels(q)
    try:
        return labels, map_label_tokens(tokenizer, labels)
    except LabelTokenError:
        if q.kind != "score":
            raise
        letters = list(LETTERS[: q.k])
        return letters, map_label_tokens(tokenizer, letters)


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


SPLIT_SENTINEL = "\u2063ANYJEV_SPLIT\u2063"   # invisible separator; chat templates pass content through


@dataclass
class PromptSpec:
    system: str
    user: str
    split: int = -1   # index into `user` where the permutation-specific suffix begins (-1: no split)


def build_prompt(state_text: str, q: Question, perm: Sequence[int],
                 system: str = DEFAULT_SYSTEM, labels: Optional[Sequence[str]] = None) -> PromptSpec:
    """perm[j] = index (into q.options) of the option shown at position j.
    labels: the answer strings to show (default answer_labels(q)); pass the
    tokenizer-resolved set from resolve_labels so prompt and readout agree."""
    labels = list(labels) if labels is not None else answer_labels(q)
    # everything before `suffix` is identical across the permutations of one state,
    # so a backend that can reuse a prefix KV cache computes it once per state
    prefix = ["State:", state_text if state_text else "(empty)", "", f"Question: {q.text}"]
    suffix = []
    if q.kind == "noul":
        # perm over ("Yes","No") only changes the phrasing order
        order = [labels[i] for i in perm]
        suffix.append(f"Answer {order[0]} or {order[1]}.")
    elif q.kind == "score":
        if q.centers is not None:
            prefix.append("Pick the level that applies (the levels are ordered):")
        else:
            lo, hi = q.scale
            prefix.append(f"Answer on a scale from {lo:g} to {hi:g} by picking the closest bin:")
        for j, i in enumerate(perm):
            suffix.append(f"{labels[j]}. {q.options[i]}")
        suffix.append("Answer with the number only." if labels[0].isdigit() else "Answer with the letter only.")
    else:
        prefix.append("Options:")
        for j, i in enumerate(perm):
            suffix.append(f"{labels[j]}. {q.options[i]}")
        suffix.append("Answer with the letter only.")
    user_prefix = "\n".join(prefix) + "\n"
    return PromptSpec(system=system, user=user_prefix + "\n".join(suffix), split=len(user_prefix))


def _render(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return f"{system}\n\n{user}\nAnswer:"
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_chat(tokenizer, spec: PromptSpec) -> str:
    """Render through the tokenizer's chat template with the generation prompt
    appended, so the next token is the model's first answer token."""
    return _render(tokenizer, spec.system, spec.user)


def render_chat_parts(tokenizer, spec: PromptSpec):
    """(prefix_text, suffix_text) whose concatenation equals render_chat(spec).
    The split is placed inside the user content by a sentinel that the chat
    template passes through untouched; the template's own tail (end-of-turn,
    assistant header) lands in the suffix. Falls back to (whole, "") when the
    spec has no split or a template alters the sentinel."""
    if spec.split < 0:
        return _render(tokenizer, spec.system, spec.user), ""
    marked = _render(tokenizer, spec.system, spec.user[:spec.split] + SPLIT_SENTINEL + spec.user[spec.split:])
    if marked.count(SPLIT_SENTINEL) != 1:
        return _render(tokenizer, spec.system, spec.user), ""
    prefix, suffix = marked.split(SPLIT_SENTINEL)
    return prefix, suffix
