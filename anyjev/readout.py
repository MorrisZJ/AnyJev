"""Readout: build prompts for a (state, question, permutation) and map option
labels to the single tokens whose logits we read at the answer position.

Images ride along as `<image i>` markers inside the state text; rendering
splits the user turn at those markers into an interleaved content list, so a
picture lands where it sat in the state rather than all of them up front."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    images: tuple = field(default_factory=tuple)


def build_prompt(state_text: str, q: Question, perm: Sequence[int],
                 system: str = DEFAULT_SYSTEM, labels: Optional[Sequence[str]] = None,
                 images: Sequence[Image] = ()) -> PromptSpec:
    """perm[j] = index (into q.options) of the option shown at position j.
    labels: the answer strings to show (default answer_labels(q)); pass the
    tokenizer-resolved set from resolve_labels so prompt and readout agree.
    images: the pictures behind the `<image i>` markers in state_text."""
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
    return PromptSpec(system=system, user=user_prefix + "\n".join(suffix), split=len(user_prefix),
                      images=tuple(images))


def _interleave(user: str, images: Sequence[Image]) -> Tuple[List[Dict[str, Any]], tuple]:
    """Split a user turn at its `<image i>` markers. Returns the content parts
    and the images in the order their placeholders appear, which is the order
    a processor will consume them in. Each image is placed once, at its first
    marker; a repeat, or a literal "<image 9>" with no such image (MMMU-style
    question text), stays as text. Placeholders and images therefore always
    agree in count and order, whatever the user's own text contains."""
    parts: List[Dict[str, Any]] = []
    order: List[Image] = []
    used = set()
    cursor = 0
    for match in MARKER_RE.finditer(user):
        index = int(match.group(1)) - 1
        if not 0 <= index < len(images) or index in used:
            continue
        lead = user[cursor:match.start()]
        if lead:
            parts.append({"type": "text", "text": lead})
        parts.append({"type": "image"})
        used.add(index)
        order.append(images[index])
        cursor = match.end()
    tail = user[cursor:]
    if tail:
        parts.append({"type": "text", "text": tail})
    return parts, tuple(order)


def user_content(spec: PromptSpec) -> Any:
    """The user turn: a plain string when there are no images, otherwise the
    text split at its markers with an image part in each gap."""
    return _interleave(spec.user, spec.images)[0] if spec.images else spec.user


def prompt_images(spec: PromptSpec) -> tuple:
    """The images to hand the backend with this prompt, in placeholder order."""
    return _interleave(spec.user, spec.images)[1] if spec.images else ()


def _render(renderer, system: str, user: str, images: Sequence[Image] = ()) -> str:
    content = _interleave(user, images)[0] if images else user
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    template = getattr(renderer, "chat_template", None)
    if not template:
        return f"{system}\n\n{user}\nAnswer:"
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return renderer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return renderer.apply_chat_template(messages, **kwargs)


def render_chat(renderer, spec: PromptSpec) -> str:
    """Render through the chat template with the generation prompt appended,
    so the next token is the model's first answer token. `renderer` is the
    tokenizer for a text model and the processor for a multimodal one; the
    processor expands each image placeholder to the right token count later."""
    return _render(renderer, spec.system, spec.user, spec.images)


def render_chat_parts(renderer, spec: PromptSpec):
    """(prefix_text, suffix_text) whose concatenation equals render_chat(spec).
    The split is placed inside the user content by a sentinel that the chat
    template passes through untouched; the template's own tail (end-of-turn,
    assistant header) lands in the suffix. Falls back to (whole, "") when the
    spec has no split or a template alters the sentinel. Image markers all sit
    in the state, before the split, so the sentinel always lands in text."""
    if spec.split < 0:
        return _render(renderer, spec.system, spec.user, spec.images), ""
    marked = _render(renderer, spec.system, spec.user[:spec.split] + SPLIT_SENTINEL + spec.user[spec.split:],
                     spec.images)
    if marked.count(SPLIT_SENTINEL) != 1:
        return _render(renderer, spec.system, spec.user, spec.images), ""
    prefix, suffix = marked.split(SPLIT_SENTINEL)
    return prefix, suffix


# ---------------------------------------------------------------- option-line positions
_OPTION_LINE = re.compile(r"^([A-Z]|\d+)\. (.*)$")
_NOUL_LINE = re.compile(r"^Answer (Yes|No) or (Yes|No)\.")   # a template may append its end-of-turn tag


def option_spans(tokenizer, spec: PromptSpec, kind: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Character spans, in *position* order, of each option's own text inside the rendered prompt:
    for choice and score the text after "A. " on every option line, for noul the two answer words
    of the phrasing line. The hidden state at the end of such a span is an option-conditioned
    representation that has already attended to the state (the bench feature caches store it).
    Returns (rendered_text, spans); spans[j] belongs to the option shown at position j."""
    prefix, suffix = render_chat_parts(tokenizer, spec)
    text = prefix + suffix
    boundary = len(prefix) if suffix else 0       # no split: scan everything after the state block
    spans: List[Tuple[int, int]] = []
    offset = 0
    for line in text.split("\n"):
        if offset >= boundary:
            if kind == "noul":
                m = _NOUL_LINE.match(line)
                if m:
                    for g in (1, 2):
                        spans.append((offset + m.start(g), offset + m.end(g)))
            else:
                m = _OPTION_LINE.match(line)
                if m:
                    spans.append((offset + m.start(2), offset + m.end(2)))
        offset += len(line) + 1
    return text, spans


def option_token_positions(tokenizer, text: str, spans: Sequence[Tuple[int, int]],
                           which: str = "last") -> List[int]:
    """Token index, under `tokenizer(text, add_special_tokens=False)`, for each character span:
    "last" = the last token that overlaps the span (the option's final token), "newline" = the
    first token starting at or after the span (the line break that closes the option). Uses the
    fast tokenizer's offset mapping; falls back to counting the tokens of the text up to the span
    end, which is exact only when the tokenizer splits there."""
    offsets = None
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(enc["offset_mapping"])
    except (TypeError, KeyError, NotImplementedError):
        offsets = None
    out: List[int] = []
    for s, e in spans:
        if offsets:
            if which == "newline":
                idx = next((i for i, (a, _) in enumerate(offsets) if a >= e), len(offsets) - 1)
            else:
                hits = [i for i, (a, b) in enumerate(offsets) if a < e and b > s]
                if not hits:
                    raise LabelTokenError(f"no token overlaps option span {(s, e)} in the rendered prompt")
                idx = hits[-1]
        else:
            n = len(tokenizer.encode(text[:e], add_special_tokens=False))
            idx = n - 1 if which == "last" else n
        out.append(int(idx))
    return out
