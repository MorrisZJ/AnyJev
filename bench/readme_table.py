"""The README's headline bench table (raw vs L0 flip and accuracy, L1 ECE), from a results dir.

    python -m bench.readme_table bench/results_v01 [--models Qwen3-8B,Qwen2.5-7B-Instruct,...]

Same JSON files as `bench.table`; this is the compact view pasted into the README, so it is generated
rather than typed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

TASK_ORDER = {"banking20": 0, "newsgroups": 1, "injection": 2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--models", default=None, help="comma-separated short model names to include, in order")
    args = ap.parse_args(argv)
    runs = {}
    for p in sorted(glob.glob(os.path.join(args.dir, "*", "*.json"))):
        r = json.load(open(p))
        if "tasks" in r:
            runs[r["model"].split("/")[-1]] = r
    order = args.models.split(",") if args.models else sorted(runs)
    lines = ["| model | task | K | raw flip | L0 flip | raw acc | L0 acc | raw ECE | L1 ECE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = runs.get(m)
        if not r:
            continue
        for t in sorted(r["tasks"], key=lambda t: TASK_ORDER.get(t["task"], 9)):
            L = t["levels"]
            l1 = f"**{L['L1']['ece']:.3f}**" if "L1" in L else ""
            lines.append(f"| {m} | {t['task']} | {t['k']} | {L['raw']['flip']:.3f} | **{L['L0']['flip']:.3f}** | "
                         f"{L['raw']['acc']:.3f} | **{L['L0']['acc']:.3f}** | {L['raw']['ece']:.3f} | {l1} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
