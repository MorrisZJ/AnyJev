"""State rendering: text, JSON-like dicts, or chat transcripts -> one string."""
from __future__ import annotations

import json
from typing import Any


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list) and state and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in state
    ):
        return "\n".join(f"{m['role']}: {m['content']}" for m in state)
    try:
        return json.dumps(state, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(state)
