"""State rendering: text, JSON-like dicts, chat transcripts, and the images
they carry -> one string plus an ordered list of pictures.

Images are pulled out of the structure and replaced by `<image i>` markers, so
the state still renders to one readable string and the readout knows where in
the prompt each picture belongs. A state with no images renders exactly as it
did before there was a `split_state`.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Tuple

from anyjev.media import Image, as_image, blank_image, is_image_like

IMAGE_MARKER = "<image {}>"
MARKER_RE = re.compile(r"<image (\d+)>")

_IMAGE_KEYS = ("image", "image_url", "url", "path")


def _marker(i: int) -> str:
    return IMAGE_MARKER.format(i)


def _image_from_content_part(part: dict) -> Any:
    """A chat content part that declares itself an image: `{"type": "image",
    "image": ...}` or the OpenAI `{"type": "image_url", "image_url": {"url":
    ...}}`. The declaration is what makes a bare string unambiguous."""
    for key in _IMAGE_KEYS:
        if key in part:
            value = part[key]
            if isinstance(value, dict):
                value = value.get("url") or value.get("path")
            if value is not None:
                return value
    return None


def _extract(node: Any, found: List[Image]) -> Any:
    if is_image_like(node):
        found.append(as_image(node))
        return _marker(len(found))
    if isinstance(node, dict):
        if str(node.get("type", "")).startswith("image"):
            src = _image_from_content_part(node)
            if src is not None:
                found.append(as_image(src))
                return _marker(len(found))
        return {k: _extract(v, found) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_extract(v, found) for v in node]
    return node


def _is_chat(state: Any) -> bool:
    return isinstance(state, list) and bool(state) and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in state
    )


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_render_content(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text", content))
    return str(content)


def _render(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if _is_chat(state):
        return "\n".join(f"{m['role']}: {_render_content(m['content'])}" for m in state)
    try:
        return json.dumps(state, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(state)


def split_state(state: Any) -> Tuple[str, Tuple[Image, ...]]:
    """Render a state to text and pull out its images in prompt order.

    The i-th image is referenced by an `<image i>` marker at the position it
    occupied in the state, numbered in traversal order, so marker order and
    image order always agree.
    """
    found: List[Image] = []
    stripped = _extract(state, found)
    if not found:
        return _render(state), ()          # untouched text path
    return _render(stripped), tuple(found)


def render_state(state: Any) -> str:
    """The text half of `split_state`. Images become `<image i>` markers."""
    return split_state(state)[0]


def content_free_state(probe: str, n_images: int) -> Tuple[str, Tuple[Image, ...]]:
    """The probe state used to measure the label prior: `probe` text ("N/A",
    "", "[MASK]") and `n_images` blank pictures, so the prompt keeps the shape
    of a real one while carrying no content in either modality."""
    if not n_images:
        return probe, ()
    images = tuple(blank_image() for _ in range(n_images))
    markers = "\n".join(_marker(i + 1) for i in range(n_images))
    return (f"{markers}\n{probe}" if probe else markers), images
