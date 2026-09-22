"""Readout: build prompts for a (state, question, permutation) and map option
labels to the single tokens whose logits we read at the answer position.

Images ride along as `<image i>` markers inside the state text; rendering
splits the user turn at those markers into an interleaved content list, so a
picture lands where it sat in the state rather than all of them up front."""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from anyjev.media import Image
from anyjev.question import Question
from anyjev.state import MARKER_RE

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
    images: tuple = field(default_factory=tuple)


def build_prompt(state_text: str, q: Question, perm: Sequence[int],
                 system: str = DEFAULT_SYSTEM,
                 images: Sequence[Image] = ()) -> PromptSpec:
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
    return PromptSpec(system=system, user="\n".join(lines), images=tuple(images))


def _interleave(spec: PromptSpec) -> Tuple[List[Dict[str, Any]], tuple]:
    """Split the user turn at its `<image i>` markers. Returns the content parts
    and the images in the order their placeholders appear, which is the order
    a processor will consume them in. Each image is placed once, at its first
    marker; a repeat, or a literal "<image 9>" with no such image (MMMU-style
    question text), stays as text. Placeholders and images therefore always
    agree in count and order, whatever the user's own text contains."""
    parts: List[Dict[str, Any]] = []
    order: List[Image] = []
    used = set()
    cursor = 0
    for match in MARKER_RE.finditer(spec.user):
        index = int(match.group(1)) - 1
        if not 0 <= index < len(spec.images) or index in used:
            continue
        lead = spec.user[cursor:match.start()]
        if lead:
            parts.append({"type": "text", "text": lead})
        parts.append({"type": "image"})
        used.add(index)
        order.append(spec.images[index])
        cursor = match.end()
    tail = spec.user[cursor:]
    if tail:
        parts.append({"type": "text", "text": tail})
    return parts, tuple(order)


def user_content(spec: PromptSpec) -> Any:
    """The user turn: a plain string when there are no images, otherwise the
    text split at its markers with an image part in each gap."""
    return _interleave(spec)[0] if spec.images else spec.user


def prompt_images(spec: PromptSpec) -> tuple:
    """The images to hand the backend with this prompt, in placeholder order."""
    return _interleave(spec)[1] if spec.images else ()


def render_chat(renderer, spec: PromptSpec) -> str:
    """Render through the chat template with the generation prompt appended,
    so the next token is the model's first answer token. `renderer` is the
    tokenizer for a text model and the processor for a multimodal one; the
    processor expands the image placeholder to the right token count later."""
    messages = [{"role": "system", "content": spec.system},
                {"role": "user", "content": user_content(spec)}]
    template = getattr(renderer, "chat_template", None)
    if not template:
        return f"{spec.system}\n\n{spec.user}\nAnswer:"
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return renderer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return renderer.apply_chat_template(messages, **kwargs)
