"""When does L0 help? A label-free predictor study over every (model, question) point.

    python -m bench.diag_l0 bench/results_typed_diag bench/results_batchprior_v0 bench/results_small

Typed-decisions runs contribute one point per (model, question) with the gold-label skew of
that question (majority share, normalized entropy); the three-task bench contributes one point
per (model, task) with the raw order-flip rate. For each ablation row (perm only, batch prior
only, both) it reports the gain over raw, the share of points where the gain is positive, and
rank correlations with the candidate predictors. Pure CPU, no model calls.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:               # average ranks for ties
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def load_typed(d):
    pts = []
    for p in sorted(glob.glob(os.path.join(d, "*", "*.json"))):
        r = json.load(open(p))
        if "levels" not in r or "raw" not in r["levels"] or "by_question" not in r["levels"]["raw"]:
            continue
        model = r["model"].split("/")[-1]
        raw = r["levels"]["raw"]["by_question"]
        for key, base in raw.items():
            pt = {"model": model, "question": base["name"], "workflow": base["workflow"], "kind": base["kind"],
                  "k": base["k"], "majority": base["majority"], "entropy": base["label_entropy"],
                  "raw_acc": base["acc"], "raw_ece": base["ece"], "n": base["n"]}
            for lvl in ("L0-perm", "L0-bc", "L0-cf", "L0", "L1"):
                if lvl in r["levels"] and key in r["levels"][lvl]["by_question"]:
                    e = r["levels"][lvl]["by_question"][key]
                    pt[f"gain_{lvl}"] = e["acc"] - base["acc"]
                    pt[f"ece_{lvl}"] = e["ece"]
            pts.append(pt)
    return pts


def load_bench(dirs):
    pts = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "*.json"))):
            r = json.load(open(p))
            if "tasks" not in r:
                continue
            for t in r["tasks"]:
                L = t["levels"]
                pt = {"model": r["model"].split("/")[-1], "task": t["task"], "k": t["k"], "mass": t["answer_mass"],
                      "raw_acc": L["raw"]["acc"], "raw_flip": L["raw"]["flip"], "raw_ece": L["raw"]["ece"]}
                for lvl in ("L0-perm", "L0-bc", "L0-cf", "L0", "L1"):
                    if lvl in L:
                        pt[f"gain_{lvl}"] = L[lvl]["acc"] - L["raw"]["acc"]
                pts.append(pt)
    return pts


def summarize(pts, gain_key, features, label):
    sub = [p for p in pts if gain_key in p]
    if len(sub) < 3:
        return
    g = [p[gain_key] for p in sub]
    pos = sum(1 for x in g if x > 0)
    neg = sum(1 for x in g if x < -0.01)
    rhos = "".join(f"  rho({f})={spearman([p[f] for p in sub], g):+.2f}"
                   for f in features if all(f in p for p in sub))
    print(f"  {label:28s} n={len(sub):3d} mean gain {sum(g) / len(g):+.3f}  >0: {pos}/{len(sub)}  <-0.01: {neg}" + rhos)


def main(argv=None):
    dirs = argv or sys.argv[1:]
    typed = [pt for d in dirs for pt in load_typed(d)]
    bench = load_bench(dirs)
    print(f"typed-decisions points: {len(typed)} (model x question); bench points: {len(bench)} (model x task)\n")

    print("== typed-decisions, gain over raw accuracy, by question kind")
    for kind in ("choice", "noul", "score"):
        sub = [p for p in typed if p["kind"] == kind]
        print(f"{kind} (n={len(sub)})")
        for lvl in ("L0-perm", "L0-bc", "L0-cf", "L0"):
            summarize(sub, f"gain_{lvl}", ["majority", "entropy", "raw_acc", "raw_ece", "k"], lvl)
    print("\n== bench (K=20 choice + noul), gain over raw accuracy")
    for name, sub in (("choice K=20", [p for p in bench if p["k"] > 2]), ("noul", [p for p in bench if p["k"] == 2])):
        print(f"{name} (n={len(sub)})")
        for lvl in ("L0-perm", "L0-bc", "L0-cf", "L0"):
            summarize(sub, f"gain_{lvl}", ["raw_flip", "raw_acc", "mass"], lvl)

    print("\n== where the default L0 hurt the most (typed-decisions, gain_L0 < -0.02)")
    worst = sorted([p for p in typed if p.get("gain_L0", 0) < -0.02], key=lambda p: p["gain_L0"])
    for p in worst[:12]:
        print(f"  {p['model']:26s} {p['workflow'][:14]:14s} {p['question']:22s} {p['kind']:6s} K={p['k']:2d} "
              f"majority={p['majority']:.2f} entropy={p['entropy']:.2f} raw_acc={p['raw_acc']:.3f} "
              f"gain={p['gain_L0']:+.3f}")
    by_kind = defaultdict(list)
    for p in typed:
        if "gain_L0" in p:
            by_kind[p["kind"]].append(p["gain_L0"])
    print("\n== share of typed points with L0 gain >= 0, by kind:",
          {k: f"{sum(1 for x in v if x >= 0)}/{len(v)}" for k, v in by_kind.items()})


if __name__ == "__main__":
    main()
