"""Rewrite artifact files (anyjev-heads/<model>.json) in the compact head format: the head
arrays as base64 float32 instead of number lists (exact, about ten times smaller).

    python scripts/compact_heads.py anyjev-heads/*.json

Every head is decoded, re-encoded and decoded again; the file is rewritten only when the
round trip is bit-exact. Files already in the compact format are left as they are.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anyjev.heads import LinearHead  # noqa: E402


def main(paths):
    for path in paths:
        d = json.load(open(path))
        heads = d.get("heads", {})
        if all(isinstance(h.get("W"), dict) for h in heads.values()):
            print(f"{path}: already compact ({os.path.getsize(path) / 1e6:.1f} MB)")
            continue
        for key, art in heads.items():
            head = LinearHead.from_dict(art)
            compact = head.to_dict(compact=True)
            back = LinearHead.from_dict(compact)
            for name in ("W", "b", "mean", "scale"):
                a, b = getattr(head, name), getattr(back, name)
                if not np.array_equal(a.astype(np.float32), b.astype(np.float32)):
                    raise SystemExit(f"{path}: round trip differs for head {key} ({name})")
            for name in ("W", "b", "mean", "scale"):
                art[name] = compact[name]
        before = os.path.getsize(path)
        with open(path, "w") as f:
            json.dump(d, f)
        print(f"{path}: {before / 1e6:.1f} MB -> {os.path.getsize(path) / 1e6:.1f} MB, {len(heads)} heads")


if __name__ == "__main__":
    main(sys.argv[1:])
