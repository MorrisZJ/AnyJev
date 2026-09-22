"""Diff my reruns against the repo's committed JSON, cell by cell.

Covers both committed bench.run configurations:
  prior=batch        bench/results_batchprior_v0/2026-09-20  (the README's tables)
  prior=content_free bench/results_cf/2026-09-20

    python repro_check/compare.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REFERENCE = {
    "batch": os.path.join(ROOT, "bench/results_batchprior_v0/2026-09-20"),
    "content_free": os.path.join(ROOT, "bench/results_cf/2026-09-20"),
}
MINE = {
    "batch": [os.path.join(HERE, "results/bench_batch"), os.path.join(HERE, "results/items"),
              os.path.join(HERE, "results/repro"), os.path.join(HERE, "results/main")],
    "content_free": [os.path.join(HERE, "results/bench_content_free")],
}
METRICS = ["acc", "macro_f1", "brier", "nll", "ece", "flip", "cov@5%", "aurc"]
LEVELS = ["raw", "L0-cf", "L0-bc", "L0-perm", "L0-perm+cf", "L0-perm+bc", "L0", "L1"]


def scan(paths):
    """Collect (model, task) -> task-dict from either layout (bench.run writes a
    date subdirectory; my wrapper writes the json directly)."""
    out = {}
    files = []
    for p in paths:
        files += glob.glob(os.path.join(p, "*.json"))
        files += glob.glob(os.path.join(p, "*", "*.json"))
    for f in sorted(files):
        if os.path.basename(f) == "job_state.json":
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "tasks" not in r:
            continue
        for t in r["tasks"]:
            if "error" in t:
                continue
            out.setdefault((r["model"], t["task"]), []).append(t)
    return out


def main():
    grand = {"cells": 0, "identical": 0, "within_1e3": 0, "within_1e2": 0, "beyond": []}
    print("=" * 112)
    print("REPRODUCTION DIFF: my rerun vs the repo's committed JSON")
    print("same seed 0, n=300, calib=200, combine=logmean, transformers 4.55.4, H100 -- repo's own bench.run")
    print("=" * 112)

    for prior, refdir in REFERENCE.items():
        ref = scan([refdir])
        mine = scan(MINE[prior])
        keys = sorted(set(ref) & set(mine))
        print()
        print(f"### prior={prior}   ({len(keys)} of {len(ref)} committed (model, task) pairs reproduced)")
        if not keys:
            print("   nothing reproduced yet")
            continue
        rows, ident, tot = [], 0, 0
        for model, task in keys:
            rt = ref[(model, task)][0]
            for mt in mine[(model, task)]:
                for lvl in LEVELS:
                    if lvl not in rt["levels"] or lvl not in mt["levels"]:
                        continue
                    for m in METRICS:
                        a, b = rt["levels"][lvl].get(m), mt["levels"][lvl].get(m)
                        if a is None or b is None:
                            continue
                        tot += 1
                        d = b - a
                        if abs(d) < 1e-12:
                            ident += 1
                        rows.append((abs(d), model.split("/")[-1], task, lvl, m, a, b, d))
        rows.sort(reverse=True)
        big = [r for r in rows if r[0] > 1e-3]
        print(f"   {tot} comparable cells: {ident} bit-identical ({100 * ident / tot:.1f}%), "
              f"{sum(1 for r in rows if r[0] <= 1e-3)} within 0.001 "
              f"({100 * sum(1 for r in rows if r[0] <= 1e-3) / tot:.1f}%), "
              f"{sum(1 for r in rows if r[0] <= 1e-2)} within 0.01 "
              f"({100 * sum(1 for r in rows if r[0] <= 1e-2) / tot:.1f}%)")
        if big:
            print(f"   {len(big)} cells deviate by more than 0.001:")
            print(f"      {'model':24s} {'task':12s} {'level':11s} {'metric':8s} {'repo':>8s} {'mine':>8s} {'delta':>8s}")
            for _, mo, ta, lv, m, a, b, d in big[:25]:
                print(f"      {mo:24s} {ta:12s} {lv:11s} {m:8s} {a:8.4f} {b:8.4f} {d:+8.4f}")
        else:
            print("   no cell deviates by more than 0.001")

        # which levels carry the deviation?
        by_level = {}
        for r in rows:
            by_level.setdefault(r[3], []).append(r[0])
        print(f"      max deviation by level: " + ", ".join(
            f"{lv}={max(by_level[lv]):.4f}" for lv in LEVELS if lv in by_level))

        grand["cells"] += tot
        grand["identical"] += ident
        grand["within_1e3"] += sum(1 for r in rows if r[0] <= 1e-3)
        grand["within_1e2"] += sum(1 for r in rows if r[0] <= 1e-2)
        grand["beyond"] += [r for r in rows if r[0] > 1e-2]

        # run-to-run determinism among my own duplicate runs
        dup = [(k, v) for k, v in mine.items() if len(v) > 1 and k in ref]
        if dup:
            print("   run-to-run spread across my own independent reruns of the same cell:")
            for (model, task), runs in dup:
                mx = 0.0
                for lvl in LEVELS:
                    for m in METRICS:
                        vals = [r["levels"].get(lvl, {}).get(m) for r in runs]
                        vals = [v for v in vals if v is not None]
                        if len(vals) > 1:
                            mx = max(mx, max(vals) - min(vals))
                print(f"      {model.split('/')[-1]:24s} {task:12s} max spread {mx:.5f} over {len(runs)} runs")

    print()
    print("=" * 112)
    if grand["cells"]:
        print(f"TOTAL {grand['cells']} cells | bit-identical {100 * grand['identical'] / grand['cells']:.1f}% | "
              f"within 0.001 {100 * grand['within_1e3'] / grand['cells']:.1f}% | "
              f"within 0.01 {100 * grand['within_1e2'] / grand['cells']:.1f}%")
        print(f"cells deviating by more than 0.01: {len(grand['beyond'])}")
        for _, mo, ta, lv, m, a, b, d in sorted(grand["beyond"], reverse=True)[:15]:
            print(f"   {mo:24s} {ta:12s} {lv:11s} {m:8s} repo {a:.4f} mine {b:.4f} delta {d:+.4f}")
    with open(os.path.join(HERE, "results", "compare_summary.json"), "w") as f:
        json.dump({k: (v if k != "beyond" else [list(x) for x in v]) for k, v in grand.items()}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
