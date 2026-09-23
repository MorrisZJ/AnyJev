"""Report of the image L2 runs: raw levels beside L2, paired intervals, and the leakage check.

    python -m bench.mm_l2_audit bench/results_mm_l2/<date>

Reads `<model>.seed<k>.json` and `<model>.seed<k>.items.json` from `bench.mm_l2_study`, where
L0, L1 and L2 come from one run on one split, and prints three Markdown tables:

1. per model and task, every level averaged over the split seeds, with the L2 - L1 accuracy gain
   (mean, range, and how many seeds have a 95% paired bootstrap interval above zero);
2. the gain on the test items whose picture L2 never saw at fit time against those whose picture
   also sits in the calibration split (POPE asks several questions per COCO image);
3. every seed on its own, with the paired interval.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

DRAWS = 2000


def acc(P, y):
    return float(np.mean(np.argmax(P, axis=1) == y))


def paired_ci(P_a, P_b, y, seed=0):
    """acc(a) - acc(b) and its 95% bootstrap interval, resampling items."""
    ca = (np.argmax(P_a, axis=1) == y).astype(float)
    cb = (np.argmax(P_b, axis=1) == y).astype(float)
    idx = np.random.RandomState(seed).randint(0, len(y), size=(DRAWS, len(y)))
    diffs = (ca[idx] - cb[idx]).mean(axis=1)
    return float(ca.mean() - cb.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def load(root):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "*.seed*.json"))):
        if path.endswith(".items.json"):
            continue
        seed = int(re.search(r"\.seed(\d+)\.json$", path).group(1))
        run = json.load(open(path))
        per_item = json.load(open(path[:-len(".json")] + ".items.json"))["tasks"]
        model = run["model"].split("/")[-1].replace("-Instruct", "")
        for t in run["tasks"]:
            items = per_item[t["task"]]
            y = np.array([it["label"] for it in items])
            P = {lv: np.array([it[f"p_{lv.lower()}"] for it in items]) for lv in ("L0", "L1", "L2")}
            seen = np.array([it["picture_in_calib"] for it in items])
            runs.append({"model": model, "task": t["task"], "seed": seed, "y": y, "P": P, "seen": seen,
                         "levels": t["levels"], "head": t["head"]})
    return runs


def f3(x):
    return f"{x:.3f}"


def headline(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["model"])].append(r)
    head = ["task", "model", "seeds", "L0 acc", "L1 acc", "**L2 acc**", "L2 - L1 (range)", "seeds CI > 0",
            "L1 ECE", "**L2 ECE**", "L1 AURC", "**L2 AURC**", "L1 cov@5%", "**L2 cov@5%**", "L2 flip", "L2 block"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for (task, model), rs in sorted(groups.items()):
        m = {lv: {k: np.mean([r["levels"][lv][k] for r in rs]) for k in ("acc", "ece", "aurc", "cov@5%")}
             for lv in ("L0", "L1", "L2")}
        cis = [paired_ci(r["P"]["L2"], r["P"]["L1"], r["y"]) for r in rs]
        gains = [c[0] for c in cis]
        flips = [r["levels"]["L2"].get("flip") for r in rs]
        flip = f3(np.mean(flips)) if all(f is not None for f in flips) else "–"
        blocks = sorted({f"{r['head']['layer_abs']}/{r['head']['n_blocks']}" for r in rs})
        cells = [task, model, str(len(rs)), f3(m["L0"]["acc"]), f3(m["L1"]["acc"]), f"**{f3(m['L2']['acc'])}**",
                 f"{np.mean(gains):+.3f} ({min(gains):+.3f} to {max(gains):+.3f})",
                 f"{sum(c[1] > 0 for c in cis)} of {len(rs)}",
                 f3(m["L1"]["ece"]), f"**{f3(m['L2']['ece'])}**", f3(m["L1"]["aurc"]), f"**{f3(m['L2']['aurc'])}**",
                 f3(m["L1"]["cov@5%"]), f"**{f3(m['L2']['cov@5%'])}**", flip, ", ".join(blocks)]
        print("| " + " | ".join(cells) + " |")


def leakage(runs):
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if not r["seen"].any():
            continue
        for label, mask in (("picture unseen", ~r["seen"]), ("picture seen", r["seen"])):
            g = groups[(r["task"], r["model"])][label]
            g.append((int(mask.sum()), paired_ci(r["P"]["L2"][mask], r["P"]["L1"][mask], r["y"][mask])[0]))
    if not groups:
        return
    print("| task | model | test items, picture unseen (mean) | L2 - L1 unseen | test items, picture seen (mean) "
          "| L2 - L1 seen |")
    print("|---|---|---|---|---|---|")
    for (task, model), g in sorted(groups.items()):
        u, s = g["picture unseen"], g["picture seen"]
        print(f"| {task} | {model} | {np.mean([n for n, _ in u]):.0f} | {np.mean([d for _, d in u]):+.3f} "
              f"| {np.mean([n for n, _ in s]):.0f} | {np.mean([d for _, d in s]):+.3f} |")


def per_seed(runs):
    print("| task | model | seed | L0 acc | L1 acc | L2 acc | L2 - L1 [95% CI] | L1 ECE | L2 ECE | L2 block, head |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["task"], r["model"], r["seed"])):
        d, lo, hi = paired_ci(r["P"]["L2"], r["P"]["L1"], r["y"])
        lv, h = r["levels"], r["head"]
        print(f"| {r['task']} | {r['model']} | {r['seed']} | {f3(lv['L0']['acc'])} | {f3(lv['L1']['acc'])} "
              f"| {f3(lv['L2']['acc'])} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {f3(lv['L1']['ece'])} "
              f"| {f3(lv['L2']['ece'])} | {h['layer_abs']}/{h['n_blocks']}, {h['kind'].split(':')[1]} |")


def main(argv=None):
    root = (argv or sys.argv[1:])[0]
    runs = load(root)
    print("### Every level, mean over split seeds\n")
    headline(runs)
    print("\n### Pictures L2 saw at fit time\n")
    leakage(runs)
    print("\n### Every seed\n")
    per_seed(runs)


if __name__ == "__main__":
    main()
