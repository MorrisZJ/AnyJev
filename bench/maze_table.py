"""Markdown table for NanoJev maze runs (bench/results_nanojev/<date>/*.json).

    python -m bench.maze_table bench/results_nanojev/<date>
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    d = argv[0] if argv else sorted(glob.glob("bench/results_nanojev/*"))[-1]
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        s = r["summary"]
        eps = r["episodes"]
        by_split = {}
        for sp in ("test", "ood"):
            sub = [e for e in eps if e["split"] == sp]
            if sub:
                by_split[sp] = f"{sum(e['goal_completion'] for e in sub)}/{len(sub)}"
        name = f"{r.get('model', '').split('/')[-1]} + {r.get('level')}"
        if r.get("engine") == "nanojev_native":
            name = f"{r.get('model', '').split('/')[-1]} native A/B readout (NanoJev's 'Untuned Qwen' protocol)"
        elif r.get("level") == "raw":
            name = f"{r.get('model', '').split('/')[-1]} + AnyJev raw"
        else:
            name = f"{r.get('model', '').split('/')[-1]} + AnyJev {r.get('level')} ({r.get('prior')} prior)"
        if "edge_majority" in s:
            majority = s["edge_majority"]
        else:
            obs = [o for e in eps for o in e["observations"]]
            clear = [bool(o["truth"][a]) for o in obs for a in ("north", "east", "south", "west")]
            majority = max(sum(clear), len(clear) - sum(clear)) / max(1, len(clear))
        rows.append((name, by_split.get("test", ""), by_split.get("ood", ""), s["attempts"], s["collisions"],
                     s.get("atomic_accuracy"), majority, s.get("atomic_brier"), s.get("atomic_questions")))
    cols = ["goal test", "goal ood", "attempts", "collisions", "edge acc", "majority", "edge Brier", "edge questions"]
    lines = ["| engine | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, t, o, a, c, acc, maj, br, nq in rows:
        f = lambda x: "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
        lines.append(f"| {name} | {t} | {o} | {a} | {c} | {f(acc)} | {f(maj)} | {f(br)} | {f(nq)} |")
    print("\n".join(lines))
    print("\nNanoJev scaled_maze episodes (test + ood), frozen exploration code from TianyuCodings/NanoJev; only the "
          "engine that answers 'is one step <dir> clear?' changes. edge acc/Brier: the model's Boolean answers "
          "against the true local geometry on the cells it visited; majority = accuracy of always answering the "
          "more common label on those same cells.")


if __name__ == "__main__":
    main()
