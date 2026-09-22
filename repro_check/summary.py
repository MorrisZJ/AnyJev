"""The headline reproduction number, stratified by whether the rerun used the
same --batch-size as the committed run.

Mixing batch sizes in one pool understates reproduction badly: per
repro_check/probe_batchsize.py the logits themselves move when the batch changes.
This script reports the config-matched subset separately, and within it splits
`prior=batch` from `prior=content_free`, because only the former carries the
stateful running prior that perturbs the fitted temperature.

    python repro_check/summary.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REF = {"batch": os.path.join(ROOT, "bench/results_batchprior_v0/2026-09-20"),
       "content_free": os.path.join(ROOT, "bench/results_cf/2026-09-20")}

# every rerun directory, and the --batch-size it was produced with
SOURCES = [
    (os.path.join(HERE, "results/repro"), {"*": 32}),
    (os.path.join(HERE, "results/main"), {"*": 32}),
    (os.path.join(HERE, "results/bench_batch"), {"Qwen/Qwen2.5-7B-Instruct": 16, "*": 32}),
    (os.path.join(HERE, "results/bench_content_free"), {"Qwen/Qwen2.5-7B-Instruct": 16, "*": 32}),
]
COMMITTED_BS = 32  # bench.run's default, and what the committed runs used
METRICS = ["acc", "macro_f1", "brier", "nll", "ece", "flip", "cov@5%", "aurc"]
L0_LEVELS = ["L0-perm", "L0-bc", "L0-cf", "L0-perm+bc", "L0-perm+cf", "L0"]


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*", "*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "tasks" not in r:
            continue
        for t in r["tasks"]:
            if "error" not in t:
                out.setdefault((r["model"], t["task"], r.get("prior", "batch")), []).append(t)
    return out


def main():
    ref = {}
    for prior, d in REF.items():
        for (m, task, _p), ts in load(d).items():
            ref[(m, task, prior)] = ts[0]

    # bucket: (batch_size_matches, prior, level_group) -> [abs deviations]
    devs = defaultdict(list)
    seen_cells = defaultdict(set)
    for d, bsmap in SOURCES:
        for (model, task, prior), ts in load(d).items():
            bs = bsmap.get(model, bsmap.get("*"))
            matched = (bs == COMMITTED_BS)
            rt = ref.get((model, task, prior))
            if rt is None:
                continue
            seen_cells[(matched, prior)].add((model, task))
            for t in ts:
                for lvl in rt["levels"]:
                    if lvl not in t["levels"]:
                        continue
                    grp = "raw" if lvl == "raw" else ("L1" if lvl == "L1" else "L0 (all ablations)")
                    for m in METRICS:
                        a, b = rt["levels"][lvl].get(m), t["levels"][lvl].get(m)
                        if a is None or b is None:
                            continue
                        devs[(matched, prior, grp)].append(abs(b - a))

    def block(title, matched):
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)
        print(f"{'prior':14s} {'level':22s} {'cells':>7s} {'bit-identical':>14s} "
              f"{'<=0.001':>9s} {'<=0.01':>8s} {'max dev':>10s}")
        print("-" * 100)
        for prior in ("batch", "content_free"):
            pairs = seen_cells.get((matched, prior), set())
            if not pairs:
                continue
            for grp in ("raw", "L0 (all ablations)", "L1"):
                v = devs.get((matched, prior, grp))
                if not v:
                    continue
                n = len(v)
                print(f"{prior:14s} {grp:22s} {n:7d} "
                      f"{100 * sum(x < 1e-12 for x in v) / n:13.1f}% "
                      f"{100 * sum(x <= 0.001 for x in v) / n:8.1f}% "
                      f"{100 * sum(x <= 0.01 for x in v) / n:7.1f}% "
                      f"{max(v):10.4f}")
            print(f"{'':14s} {'(models x tasks)':22s} "
                  f"{', '.join(sorted(f'{m.split(chr(47))[-1]}/{t}' for m, t in pairs))[:60]}")
        print()

    block("CONFIG-MATCHED RERUNS  (--batch-size 32, the bench default and what the committed runs used)", True)
    block("CONFIG-MISMATCHED RERUNS  (--batch-size 16: same code, same seed, different batching)", False)

    m = devs
    zero_label = [x for (mt, p, g), v in m.items() if mt and g in ("raw", "L0 (all ablations)") for x in v]
    l1_cf = [x for (mt, p, g), v in m.items() if mt and p == "content_free" and g == "L1" for x in v]
    l1_batch = [x for (mt, p, g), v in m.items() if mt and p == "batch" and g == "L1" for x in v]
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    if zero_label:
        print(f"zero-label results (raw + every L0 ablation), config-matched: "
              f"{len(zero_label)} cells, {100 * sum(x < 1e-12 for x in zero_label) / len(zero_label):.1f}% bit-identical, "
              f"max deviation {max(zero_label):.6f}")
    if l1_cf:
        print(f"L1 with prior=content_free: {len(l1_cf)} cells, "
              f"{100 * sum(x < 1e-12 for x in l1_cf) / len(l1_cf):.1f}% bit-identical, max {max(l1_cf):.6f}")
    if l1_batch:
        print(f"L1 with prior=batch:        {len(l1_batch)} cells, "
              f"{100 * sum(x < 1e-12 for x in l1_batch) / len(l1_batch):.1f}% bit-identical, max {max(l1_batch):.6f}")
        print("   ^ the only config-matched level that moves. repro_check/probe_stateful_prior.py shows why:")
        print("     the batch prior is a running accumulator, so the temperature fitted during")
        print("     calibrate() depends on how many items the decider has already seen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
