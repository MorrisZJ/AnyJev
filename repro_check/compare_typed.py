"""Diff my typed-decisions reruns against the repo's committed JSON.

Also checks the two claims the README builds on that table:
  - the fine-tuned Laya row reproduces its published 0.766
  - "the fine-tuned Laya's ECE is six times AnyJev L1's"

    python repro_check/compare_typed.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REF_DIR = os.path.join(ROOT, "bench/results_typed/2026-09-21")
MINE_DIRS = [os.path.join(HERE, "results/typed")]
COLS = ["acc", "soft_acc", "ece", "brier", "brier_mean", "nll", "score_mae"]
LEVELS = ["raw", "L0-perm", "L0-bc", "L0-perm+bc", "L0", "L1"]

# what the README / run_typed.PUBLISHED claim the authors' sources say
PUBLISHED = {
    "laya-typed-decisions": {"acc": 0.766, "soft_acc": 0.471, "ece": 0.213, "score_mae": 0.242},
    "laya": {"acc": 0.361, "soft_acc": 0.332, "ece": 0.175, "score_mae": 0.694},
}


def scan(dirs):
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*", "*.json"))):
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if "levels" in r:
                out[r["model"]] = r
    return out


def providers():
    out = {}
    for p in sorted(glob.glob(os.path.join(REF_DIR, "*.json"))):
        r = json.load(open(p))
        if "checkpoint" in r:
            out[r["checkpoint"]] = r
    return out


def main():
    ref, mine = scan([REF_DIR]), scan(MINE_DIRS)
    keys = sorted(set(ref) & set(mine))
    print("=" * 108)
    print("TYPED-DECISIONS DIFF: my rerun of bench.run_typed vs the repo's committed JSON")
    print("LocalLLaMA/typed-decisions test split, 400 cases / 2,000 decisions / 20 questions, prior=batch")
    print("=" * 108)
    if not keys:
        print("no overlapping models yet (still running or not started)")
    tot = ident = 0
    rows = []
    for model in keys:
        r, m = ref[model], mine[model]
        print(f"\n### {model}")
        print(f"   decisions  repo {r['n_decisions']} / mine {m['n_decisions']};  "
              f"questions repo {r['n_questions']} / mine {m['n_questions']};  "
              f"calib repo {r['n_calib_decisions']} / mine {m['n_calib_decisions']}")
        print(f"   {'level':11s} {'metric':11s} {'repo':>8s} {'mine':>8s} {'delta':>8s}")
        for lvl in LEVELS:
            if lvl not in r["levels"] or lvl not in m["levels"]:
                continue
            for c in COLS:
                a = r["levels"][lvl]["overall"].get(c)
                b = m["levels"][lvl]["overall"].get(c)
                if a is None or b is None:
                    continue
                tot += 1
                d = b - a
                if abs(d) < 1e-12:
                    ident += 1
                rows.append((abs(d), model, lvl, c, a, b, d))
                if abs(d) > 5e-4:
                    print(f"   {lvl:11s} {c:11s} {a:8.4f} {b:8.4f} {d:+8.4f}  <-- differs")
        shown = [x for x in rows if x[1] == model and x[0] > 5e-4]
        if not shown:
            print("   every comparable cell matches to within 0.0005")

    if tot:
        rows.sort(reverse=True)
        print()
        print("-" * 108)
        print(f"{tot} comparable cells: {ident} bit-identical ({100 * ident / tot:.1f}%), "
              f"{sum(1 for r in rows if r[0] <= 1e-3)} within 0.001, "
              f"{sum(1 for r in rows if r[0] <= 1e-2)} within 0.01")
        print(f"max deviation: {rows[0][0]:.4f} at {rows[0][1]} / {rows[0][2]} / {rows[0][3]}")

    # the provider rows are committed but need the Laya checkpoints to rerun;
    # at minimum check the committed numbers against what their authors published
    print()
    print("=" * 108)
    print("PROVIDER ROWS: committed JSON vs the numbers their authors published")
    print("=" * 108)
    prov = providers()
    for ck, r in sorted(prov.items()):
        short = ck.split("/")[-1]
        o = r["overall"]
        pub = PUBLISHED.get(short)
        line = f"{short:26s} measured-here acc {o['acc']:.3f}  ece {o['ece']:.3f}  soft {o.get('soft_acc', float('nan')):.3f}"
        if pub:
            line += f"   | published acc {pub['acc']:.3f} (delta {o['acc'] - pub['acc']:+.3f})"
        print(line)
    l_ft = prov.get("convaiinnovations/laya-typed-decisions", {}).get("overall", {})
    if l_ft:
        d = l_ft["acc"] - PUBLISHED["laya-typed-decisions"]["acc"]
        print(f"\n'reproduces its published 0.766': measured {l_ft['acc']:.3f}, published 0.766, "
              f"delta {d:+.3f} -> {'supported' if abs(d) < 0.01 else 'NOT supported'}")
        best_l1 = None
        for src in (mine, ref):
            v = [r["levels"]["L1"]["overall"]["ece"] for r in src.values() if "L1" in r["levels"]]
            if v:
                best_l1 = min(v)
                break
        if best_l1:
            print(f"'six times the ECE' : laya {l_ft['ece']:.3f} / best L1 {best_l1:.3f} = "
                  f"{l_ft['ece'] / best_l1:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
