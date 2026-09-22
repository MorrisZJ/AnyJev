"""Aggregate every run in repro_check/results and answer the questions the repo's
own tables cannot:

  1. does the readout transfer to non-Qwen families at all?
  2. does position bias really grow with K? (their bench only has K=2 and K=20)
  3. does the `score` primitive behave? (their bench never tests it)
  4. how wide is the confidence interval on cov@5%, the headline metric?

    python repro_check/analyze.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench import metrics  # noqa: E402

TASK_META = {
    "injection": ("noul", 2), "hate": ("noul", 2), "subj": ("noul", 2),
    "agnews": ("choice", 4), "emotion": ("choice", 6),
    "banking20": ("choice", 20), "newsgroups": ("choice", 20),
    "massive20": ("choice", 20), "massive20_zh": ("choice", 20),
    "yelp5": ("score", 5),
}


def load_all():
    """Only bench.run-shaped results: typed / maze / laya / probe JSONs live in
    the same tree but carry a different schema and are analysed elsewhere."""
    runs = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*", "*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if not isinstance(r, dict) or "tasks" not in r or "model" not in r:
            continue
        r["_tag"] = os.path.basename(os.path.dirname(p))
        r["_path"] = p
        runs.append(r)
    return runs


def short(model):
    return model.split("/")[-1]


# --------------------------------------------------------------------------
def table_cross_model(runs):
    print("=" * 122)
    print("1. THE READOUT ACROSS MODEL FAMILIES AND TASKS  (n=300, seed 0, prior=batch, L1 from 200 calib)")
    print("=" * 122)
    print(f"{'model':28s} {'task':13s} {'kind':6s} {'K':>3s} | "
          f"{'raw flip':>8s} {'L0 flip':>8s} | {'raw acc':>7s} {'L0 acc':>7s} | "
          f"{'raw ece':>7s} {'L1 ece':>7s} | {'raw cov':>7s} {'L0 cov':>7s} {'L1 cov':>7s}")
    print("-" * 122)
    seen = set()
    rows = []
    for r in runs:
        for t in r["tasks"]:
            if "error" in t:
                continue
            key = (short(r["model"]), t["task"])
            if key in seen:
                continue
            seen.add(key)
            lv = t["levels"]
            kind, K = TASK_META.get(t["task"], (t.get("kind", "?"), t.get("k", 0)))

            def g(level, m):
                return lv.get(level, {}).get(m)

            rows.append((short(r["model"]), t["task"], kind, K, lv))
    rows.sort(key=lambda x: (x[3], x[1], x[0]))
    for model, task, kind, K, lv in rows:
        def f(level, m):
            v = lv.get(level, {}).get(m)
            return f"{v:7.3f}" if isinstance(v, (int, float)) else "      -"
        print(f"{model:28s} {task:13s} {kind:6s} {K:3d} | "
              f"{f('raw', 'flip'):>8s} {f('L0', 'flip'):>8s} | {f('raw', 'acc')} {f('L0', 'acc')} | "
              f"{f('raw', 'ece')} {f('L1', 'ece')} | {f('raw', 'cov@5%')} {f('L0', 'cov@5%')} {f('L1', 'cov@5%')}")
    return rows


# --------------------------------------------------------------------------
def analysis_flip_vs_k(rows):
    print()
    print("=" * 122)
    print("2. DOES POSITION BIAS GROW WITH K?  (the repo's bench only covers K=2 and K=20)")
    print("=" * 122)
    by_k = {}
    for model, task, kind, K, lv in rows:
        if kind == "score":
            continue
        rf, lf = lv.get("raw", {}).get("flip"), lv.get("L0", {}).get("flip")
        if rf is None or lf is None:
            continue
        by_k.setdefault(K, []).append((rf, lf, model, task))
    print(f"{'K':>3s} {'n cells':>7s} {'raw flip (mean)':>16s} {'L0 flip (mean)':>15s} "
          f"{'absolute drop':>14s} {'relative drop':>14s}")
    print("-" * 122)
    for K in sorted(by_k):
        v = by_k[K]
        rf = np.mean([x[0] for x in v])
        lf = np.mean([x[1] for x in v])
        rel = (1 - lf / rf) * 100 if rf > 0 else float("nan")
        print(f"{K:3d} {len(v):7d} {rf:16.3f} {lf:15.3f} {rf - lf:14.3f} {rel:13.0f}%")
    ks = sorted(by_k)
    if len(ks) >= 3:
        x = np.array([k for k in ks for _ in by_k[k]], dtype=float)
        y = np.array([c[0] for k in ks for c in by_k[k]])
        r = np.corrcoef(np.log(x), y)[0, 1]
        print(f"\ncorrelation between log K and raw flip rate over {len(x)} cells: r = {r:+.3f}")


# --------------------------------------------------------------------------
def analysis_noul_prior(rows):
    print()
    print("=" * 122)
    print("3. THE CONTENT-FREE PRIOR IS HIGH VARIANCE  (the repo's own caveat, now on 3 noul tasks)")
    print("=" * 122)
    print(f"{'model':28s} {'task':13s} {'raw acc':>8s} {'+perm':>8s} {'+batch':>8s} {'+cf':>8s} "
          f"{'best':>8s} {'cf - batch':>11s}")
    print("-" * 122)
    deltas = []
    for model, task, kind, K, lv in rows:
        if kind != "noul":
            continue
        a_raw = lv.get("raw", {}).get("acc")
        a_perm = lv.get("L0-perm", {}).get("acc")
        a_bc = lv.get("L0-perm+bc", {}).get("acc", lv.get("L0", {}).get("acc"))
        a_cf = lv.get("L0-perm+cf", {}).get("acc")
        if None in (a_raw, a_perm, a_bc, a_cf):
            continue
        best = max(a_raw, a_perm, a_bc, a_cf)
        deltas.append(a_cf - a_bc)
        print(f"{model:28s} {task:13s} {a_raw:8.3f} {a_perm:8.3f} {a_bc:8.3f} {a_cf:8.3f} "
              f"{best:8.3f} {a_cf - a_bc:+11.3f}")
    if deltas:
        d = np.array(deltas)
        print(f"\ncontent-free minus batch prior over {len(d)} noul cells: "
              f"mean {d.mean():+.3f}, range {d.min():+.3f} to {d.max():+.3f} "
              f"-- {'confirms' if d.max() - d.min() > 0.05 else 'does not confirm'} the repo's "
              f"'high variance' caveat")


# --------------------------------------------------------------------------
def analysis_score(runs):
    print()
    print("=" * 122)
    print("4. THE `score` PRIMITIVE  (never exercised by the repo's own bench)")
    print("=" * 122)
    found = False
    for r in runs:
        for t in r["tasks"]:
            if t.get("task") != "yelp5":
                continue
            if "error" in t:
                print(f"{short(r['model']):28s} ERROR: {t['error']}")
                found = True
                continue
            lv = t["levels"]
            found = True
            print(f"{short(r['model']):28s} perms={t.get('permutations')} "
                  f"(score bins are ordinal: never permuted, so only the prior correction applies)")
            for name in ("raw", "L0-bc", "L0-cf", "L0", "L1"):
                if name in lv:
                    m = lv[name]
                    print(f"   {name:10s} acc {m['acc']:.3f}  ece {m['ece']:.3f}  "
                          f"brier {m['brier']:.3f}  flip {m.get('flip', float('nan')):.3f}  "
                          f"cov@5% {m['cov@5%']:.3f}")
    if not found:
        print("no yelp5 results yet")


# --------------------------------------------------------------------------
def analysis_bootstrap(runs, n_boot=4000):
    """The headline metric is a point estimate on 300 items. Resample it."""
    print()
    print("=" * 122)
    print(f"5. HOW SOLID IS cov@5%?  ({n_boot} bootstrap resamples of the same 300 test items)")
    print("=" * 122)
    print(f"{'model':24s} {'task':13s} {'level':6s} {'point':>7s} {'boot mean':>10s} "
          f"{'95% CI':>17s} {'CI width':>9s}")
    print("-" * 122)
    rng = np.random.default_rng(0)
    widths = []
    ratios = []
    for r in runs:
        slug = r["model"].replace("/", "__")
        item_dir = os.path.join(os.path.dirname(r["_path"]), f"{slug}.items")
        for t in r["tasks"]:
            if "error" in t:
                continue
            npz_path = os.path.join(item_dir, f"{t['task']}.npz")
            if not os.path.exists(npz_path):
                continue
            z = np.load(npz_path)
            labels = z["labels"]
            n = len(labels)
            cov = {}
            for level in ("raw", "L0", "L1"):
                key = f"P__{level}"
                if key not in z:
                    continue
                P = z[key]
                point = metrics.coverage_at_risk(P, labels)
                idx = rng.integers(0, n, size=(n_boot, n))
                boots = np.array([metrics.coverage_at_risk(P[i], labels[i]) for i in idx])
                lo, hi = np.quantile(boots, [0.025, 0.975])
                cov[level] = (point, boots.mean(), lo, hi)
                widths.append(hi - lo)
                print(f"{short(r['model']):24s} {t['task']:13s} {level:6s} {point:7.3f} "
                      f"{boots.mean():10.3f} [{lo:6.3f}, {hi:6.3f}] {hi - lo:9.3f}")
            if "raw" in cov and "L1" in cov and cov["raw"][0] > 0:
                ratios.append((short(r["model"]), t["task"], cov["L1"][0] / cov["raw"][0],
                               cov["L1"][2] / max(cov["raw"][3], 1e-9)))
    if widths:
        print(f"\nmean 95% CI width on cov@5% at n=300: {np.mean(widths):.3f} of coverage "
              f"(max {np.max(widths):.3f})")
    if ratios:
        print("\nthe README's '7x' style claim, with the uncertainty carried through:")
        print(f"{'model':24s} {'task':13s} {'L1/raw point':>13s} {'conservative ratio':>19s}")
        for m, t, pt, cons in sorted(ratios, key=lambda x: -x[2]):
            print(f"{m:24s} {t:13s} {pt:13.1f}x {cons:18.1f}x")
        print("  (conservative = L1's CI lower bound over raw's CI upper bound)")


def main():
    runs = load_all()
    if not runs:
        print("no results yet")
        return 0
    print(f"loaded {len(runs)} run files, "
          f"{sum(len([t for t in r['tasks'] if 'error' not in t]) for r in runs)} successful (model, task) cells, "
          f"{sum(len([t for t in r['tasks'] if 'error' in t]) for r in runs)} failed\n")
    rows = table_cross_model(runs)
    analysis_flip_vs_k(rows)
    analysis_noul_prior(rows)
    analysis_score(runs)
    analysis_bootstrap(runs)

    print()
    print("=" * 122)
    print("FAILURES (a family whose tokenizer cannot produce single-token labels shows up here)")
    print("=" * 122)
    any_err = False
    for r in runs:
        for t in r["tasks"]:
            if "error" in t:
                any_err = True
                print(f"{short(r['model']):28s} {t['task']:13s} {t['error']}")
    if not any_err:
        print("none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
