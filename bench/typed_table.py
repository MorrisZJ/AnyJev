"""Markdown table for typed-decisions results (AnyJev runs and providers in one dir).

    python -m bench.typed_table bench/results_typed/<date> [--levels raw,L0,L1]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

COLS = ["acc", "soft_acc", "ece", "brier_mean", "score_mae"]
PUBLISHED = {"Jev 1.13.0 (0.727 as listed in Laya's BENCHMARKS.md; not measured here)":
             {"acc": 0.727, "soft_acc": 0.580, "ece": 0.144, "brier_mean": 0.148, "score_mae": 0.391}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=None)
    ap.add_argument("--levels", default="raw,L0,L1")
    args = ap.parse_args(argv)
    d = args.dir or sorted(glob.glob("bench/results_typed/*"))[-1]
    levels = args.levels.split(",")
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        if "levels" in r:
            model = r["model"].split("/")[-1]
            for lvl in levels:
                if lvl in r["levels"]:
                    tag = {"raw": "raw logits (clone baseline)", "L0": "AnyJev L0, zero-shot",
                           "L1": "AnyJev L1, temperature from 200 train cases"}.get(lvl, lvl)
                    rows.append((f"{model} + {tag}", r["levels"][lvl]["overall"]))
        else:
            ck = r["checkpoint"].split("/")[-1]
            note = " (fine-tuned on this set's train split)" if "typed" in ck else " (zero-shot)"
            rows.append((f"{ck}{note}, measured here", r["overall"]))
    rows += list(PUBLISHED.items())
    rows.sort(key=lambda kv: kv[1]["acc"])
    lines = ["| system | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
    for name, m in rows:
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" if c in m else "" for c in COLS) + " |")
    print("\n".join(lines))
    print("\nLocalLLaMA/typed-decisions test split, 400 cases, 2,000 decisions, all rows except Jev measured on "
          "the same decisions. soft_acc = sum of predicted x teacher probabilities. brier_mean divides by the "
          "number of options, as Laya reports it.")


if __name__ == "__main__":
    main()
