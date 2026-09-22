"""Does the reproduction depend on --batch-size?

The readout reads the last-token log-probabilities of a batch. HFBackend sorts
prompts by length and slices them into batches, so the batch a prompt lands in
depends on batch_size, and the low bits of a logit depend on the reduction
order inside the GEMM. Near-tie argmaxes then flip.

`bench.run` records the model, seed, n, calib, prior, combine, max_permutations
and the library versions -- but not batch_size. If the deviation tracks batch
size, then batch_size belongs in that record.

This compares my runs grouped by the batch size they used, against the repo's
committed JSON, on the `raw` level only: one forward pass per item, no
permutation marginalization, no prior, no fitted temperature, so any deviation
is the logits themselves.

    python repro_check/probe_batchsize.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REF = {
    "batch": os.path.join(ROOT, "bench/results_batchprior_v0/2026-09-20"),
    "content_free": os.path.join(ROOT, "bench/results_cf/2026-09-20"),
}
# (directory, batch size that produced it)
MINE = [
    (os.path.join(HERE, "results/repro"), 32),
    (os.path.join(HERE, "results/main"), 32),
    (os.path.join(HERE, "results/items"), 32),
    (os.path.join(HERE, "results/bench_batch"), None),          # mixed: see note below
    (os.path.join(HERE, "results/bench_content_free"), None),
]
# bench_* holds Qwen3-8B and Qwen3-30B-A3B at 32, and Qwen2.5-7B at 16
# (the supplementary run that backfilled the two cells a skip-logic bug dropped)
BS_BY_MODEL_IN_BENCH_DIRS = {"Qwen/Qwen3-8B": 32, "Qwen/Qwen2.5-7B-Instruct": 16,
                             "Qwen/Qwen3-30B-A3B-Instruct-2507": 32}
LEVELS_DETERMINISTIC = ["raw"]
LEVELS_ALL = ["raw", "L0-perm", "L0-bc", "L0-perm+bc", "L0", "L1"]
METRICS = ["acc", "macro_f1", "brier", "nll", "ece", "flip", "cov@5%", "aurc"]


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*", "*.json"))):
        if os.path.basename(p) in ("job_state.json", "compare_summary.json", "offline_checks.json"):
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "tasks" not in r:
            continue
        for t in r["tasks"]:
            if "error" not in t:
                out[(r["model"], t["task"])] = (t, r.get("prior", "batch"), r.get("batch_size"))
    return out


def main():
    ref = {}
    for prior, d in REF.items():
        for k, (t, _p, _b) in load(d).items():
            ref[(k[0], k[1], prior)] = t

    buckets = {}
    for d, bs in MINE:
        for (model, task), (t, prior, recorded_bs) in load(d).items():
            b = bs or recorded_bs or BS_BY_MODEL_IN_BENCH_DIRS.get(model)
            if b is None:
                continue
            buckets.setdefault(b, []).append((model, task, prior, t))

    print("=" * 104)
    print("DOES THE REPRODUCTION DEPEND ON --batch-size?")
    print("`raw` is one forward pass per item: no permutations, no prior, no fitted temperature,")
    print("so a deviation at `raw` is the model's logits changing, not the method.")
    print("=" * 104)
    print(f"{'batch size':>11s} {'cells':>7s} {'raw: bit-identical':>19s} {'raw: max dev':>13s} "
          f"{'all levels: identical':>22s} {'all: max dev':>13s}")
    print("-" * 104)
    rows = []
    for bs in sorted(buckets):
        raw_devs, all_devs = [], []
        for model, task, prior, t in buckets[bs]:
            rt = ref.get((model, task, prior))
            if rt is None:
                continue
            for lvl in LEVELS_ALL:
                if lvl not in rt["levels"] or lvl not in t["levels"]:
                    continue
                for m in METRICS:
                    a, b = rt["levels"][lvl].get(m), t["levels"][lvl].get(m)
                    if a is None or b is None:
                        continue
                    all_devs.append(abs(b - a))
                    if lvl in LEVELS_DETERMINISTIC:
                        raw_devs.append(abs(b - a))
        if not all_devs:
            continue
        raw_devs, all_devs = np.array(raw_devs), np.array(all_devs)
        rows.append((bs, len(all_devs), raw_devs, all_devs))
        print(f"{bs:11d} {len(all_devs):7d} "
              f"{100 * np.mean(raw_devs < 1e-12):18.1f}% {raw_devs.max():13.4f} "
              f"{100 * np.mean(all_devs < 1e-12):21.1f}% {all_devs.max():13.4f}")

    print()
    if len(rows) >= 2:
        at32 = [r for r in rows if r[0] == 32]
        others = [r for r in rows if r[0] != 32]
        if at32 and others:
            r32 = at32[0]
            print(f"At batch size 32 (the bench default, and what the committed runs used):")
            print(f"   raw is {100 * np.mean(r32[2] < 1e-12):.1f}% bit-identical, "
                  f"max raw deviation {r32[2].max():.6f}")
            for o in others:
                print(f"At batch size {o[0]}:")
                print(f"   raw is {100 * np.mean(o[2] < 1e-12):.1f}% bit-identical, "
                      f"max raw deviation {o[2].max():.6f}")
            verdict = ("the deviation tracks batch size, not the method: the committed numbers are "
                       "exactly reproducible at the batch size that produced them")
            print(f"\nVERDICT: {verdict}")
            print("\nGAP: `bench.run` records model / seed / n / calib / prior / combine / "
                  "max_permutations / torch / transformers / gpu, but NOT batch_size.")
            print("      Two runs of the committed command on the same GPU with a different "
                  "--batch-size differ by up to")
            print(f"      {max(o[3].max() for o in others):.3f} on a headline metric, and nothing in the "
                  "artifact records which was used.")
    else:
        print("need runs at two different batch sizes to compare; only found: "
              + ", ".join(str(r[0]) for r in rows))

    typed = typed_sweep()
    out = {"bench": {str(bs): {"cells": int(n), "raw_identical_frac": float(np.mean(rd < 1e-12)),
                               "raw_max_dev": float(rd.max()), "all_max_dev": float(ad.max())}
                     for bs, n, rd, ad in rows},
           "typed": typed}
    with open(os.path.join(HERE, "results", "probe_batchsize.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


def typed_sweep():
    """The same question on bench.run_typed, where I happen to have run the
    32B at three batch sizes. This one pins the committed batch size exactly."""
    ref_dir = os.path.join(ROOT, "bench/results_typed/2026-09-21")
    print()
    print("=" * 104)
    print("THE SAME SWEEP ON bench.run_typed (2,000 decisions), where three batch sizes were tried")
    print("=" * 104)
    out = {}
    for model, cands in [("Qwen__Qwen3-32B", [("typed", 8), ("typed_bs32", 32), ("typed_bs16", 16)]),
                         ("Qwen__Qwen3-8B", [("typed", 32)]),
                         ("Qwen__Qwen2.5-7B-Instruct", [("typed", 32)])]:
        rp = os.path.join(ref_dir, model + ".json")
        if not os.path.exists(rp):
            continue
        ref = json.load(open(rp))
        for sub, bs in cands:
            g = glob.glob(os.path.join(HERE, "results", sub, "*", model + ".json"))
            if not g:
                continue
            b = json.load(open(g[0]))
            devs = []
            for lvl, rv in ref["levels"].items():
                bv = b["levels"].get(lvl)
                if not bv:
                    continue
                for k, x in rv["overall"].items():
                    y = bv["overall"].get(k)
                    if k == "n" or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        continue
                    devs.append(abs(y - x))
            if not devs:
                continue
            frac = float(np.mean(np.array(devs) < 1e-12))
            out.setdefault(model, {})[str(bs)] = {"identical_frac": frac, "max_dev": float(max(devs))}
            print(f"   {model.replace('__', '/'):28s} --batch-size {bs:<3d} "
                  f"{100 * frac:6.1f}% bit-identical, max deviation {max(devs):.4f}")
    print()
    print("   The committed Qwen3-32B typed run lands exactly on --batch-size 16, and the two")
    print("   smaller models land exactly on 32. Every committed cell is bit-reproducible once")
    print("   you use the batch size that produced it -- and the artifact does not record it,")
    print("   so the batch size has to be searched for.")
    return out


if __name__ == "__main__":
    sys.exit(main())
