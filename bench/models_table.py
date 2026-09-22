"""Cross-model summary: one row per model, the numbers that decide whether a model is usable.

    python -m bench.models_table bench/results_small bench/results_typed_small [more result dirs...]

Columns: label-token mass (does the model follow the format), raw->L0 flip rate and accuracy
averaged over the choice tasks, injection (noul) raw->L0 accuracy, L1 ECE, typed-decisions raw / L0 / L1
accuracy. Directories may hold several dates; the newest file per model wins.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def load_latest(dirs):
    by_model = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "*.json"))):
            r = json.load(open(p))
            if "tasks" in r:
                by_model.setdefault(r["model"], {})["bench"] = r
            elif "levels" in r and "dataset" in r:
                by_model.setdefault(r["model"], {})["typed"] = r
    return by_model


def fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    dirs = argv or sys.argv[1:]
    models = load_latest(dirs)
    cols = ["label mass", "choice flip raw→L0", "choice acc raw→L0", "injection acc raw→L0", "L1 ECE (mean)",
            "typed acc raw / L0 / L1", "typed L1 ECE"]
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for model in sorted(models):
        b = models[model].get("bench")
        t = models[model].get("typed")
        cells = defaultdict(str)
        if b:
            choice = [x for x in b["tasks"] if x["k"] > 2]
            noul = [x for x in b["tasks"] if x["k"] == 2]
            mass = min(x["answer_mass"] for x in b["tasks"])
            cells["label mass"] = fmt(mass)
            if choice:
                fr = sum(x["levels"]["raw"]["flip"] for x in choice) / len(choice)
                f0 = sum(x["levels"]["L0"]["flip"] for x in choice) / len(choice)
                ar = sum(x["levels"]["raw"]["acc"] for x in choice) / len(choice)
                a0 = sum(x["levels"]["L0"]["acc"] for x in choice) / len(choice)
                cells["choice flip raw→L0"] = f"{fr:.2f}→{f0:.2f}"
                cells["choice acc raw→L0"] = f"{ar:.3f}→{a0:.3f}"
            if noul:
                x = noul[0]
                cells["injection acc raw→L0"] = f"{x['levels']['raw']['acc']:.3f}→{x['levels']['L0']['acc']:.3f}"
            l1 = [x["levels"]["L1"]["ece"] for x in b["tasks"] if "L1" in x["levels"]]
            if l1:
                cells["L1 ECE (mean)"] = fmt(sum(l1) / len(l1))
        if t:
            lv = t["levels"]
            cells["typed acc raw / L0 / L1"] = " / ".join(
                fmt(lv[k]["overall"]["acc"]) for k in ("raw", "L0", "L1") if k in lv)
            if "L1" in lv:
                cells["typed L1 ECE"] = fmt(lv["L1"]["overall"]["ece"])
        lines.append(f"| {model.split('/')[-1]} | " + " | ".join(cells[c] for c in cols) + " |")
    print("\n".join(lines))
    print("\nchoice = mean over banking20 and newsgroups (K=20, n=300 each); injection n=300; typed-decisions n=2000. "
          "label mass = lowest mean probability the model put on the label tokens across tasks (1.0 = always answered "
          "in the format).")


if __name__ == "__main__":
    main()
