"""Aggregate every results JSON in a results directory into one Markdown table.

    python -m bench.table bench/results_v01/2026-09-22 > docs/results_bench.md
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List

COLS = ["acc", "brier", "ece", "flip", "cov@5%"]
LEVEL_ORDER = {"raw": 0, "L0-cf": 1, "L0-bc": 2, "L0-perm": 3, "L0-perm+cf": 4, "L0-perm+bc": 5, "L0": 6, "L1": 7}


def load_dir(d: str) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def rows(results: List[Dict[str, Any]]):
    for r in results:
        model = r["model"].split("/")[-1]
        for t in r["tasks"]:
            for level, m in sorted(t["levels"].items(), key=lambda kv: LEVEL_ORDER.get(kv[0], 9)):
                yield model, t["task"], t["k"], t["n_test"], level, m


def markdown(results: List[Dict[str, Any]]) -> str:
    lines = ["| model | task | K | n | level | " + " | ".join(COLS) + " |",
             "|---|---|---|---|---|" + "---|" * len(COLS)]
    last = None
    for model, task, k, n, level, m in rows(results):
        key = (model, task)
        show_model = model if key != last else ""
        show_task = f"{task}" if key != last else ""
        last = key
        cells = [f"{m[c]:.3f}" if c in m else "" for c in COLS]
        lines.append(f"| {show_model} | {show_task} | {k if show_task else ''} | {n if show_task else ''} | {level} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def env_footer(results: List[Dict[str, Any]]) -> str:
    envs = {json.dumps({k: r["env"].get(k) for k in ("gpu", "torch", "transformers")}, sort_keys=True)
            for r in results}
    dates = sorted({r["env"].get("date", "")[:10] for r in results})
    return ("\n\nEnvironment: " + "; ".join(sorted(envs)) + f". Dates: {', '.join(dates)}. "
            "Test items sampled with seed 0; L1 temperature fit on a disjoint calibration split.")


def main(argv=None):
    argv = argv or sys.argv[1:]
    d = argv[0] if argv else sorted(glob.glob("bench/results/*"))[-1]
    results = load_dir(d)
    print(markdown(results) + env_footer(results))


if __name__ == "__main__":
    main()
