"""How many labels does a new question need? Label-efficiency of the per-question closed-form head.

    python -m bench.labels_study --model Qwen/Qwen3-8B --budgets 20,50,100,200,300 --seeds 0,1,2

CPU replay of the block-loop caches (bench/results_exit/features/<model>/typed.*, written by
`bench.extract_pools`): for every typed question, every budget N and every seed, a random
N-subset of the 300 train decisions fits
(a) the closed-form head (LDA / ridge chosen by CV) on the last-position state at a few blocks,
(b) a temperature on the raw readout (L1). Test accuracy / pooled ECE with bootstrap CIs, pooled
over the 20 questions; the crossover budget (where the head beats L1 with the CI excluding 0) and
the saturation budget (within 1 point of N = 300) are read off the curve.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from anyjev.calibrate.posthoc import TemperatureScaler
from anyjev.heads import fit_head
from bench.exit_study import load_typed, pooled
from bench.pools import pool_path
from bench.run import environment


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="bench/results_exit")
    ap.add_argument("--which", default="last")
    ap.add_argument("--calib-cases", type=int, default=300)
    ap.add_argument("--n-layers", type=int, default=36)
    ap.add_argument("--blocks", default="22,28,36", help="candidate blocks the head may read (CV picks one)")
    ap.add_argument("--budgets", default="20,50,100,200,300")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args(argv)
    data = load_typed(args)
    names = sorted(data)
    layers = [int(x) for x in np.load(pool_path(args.out, args.model, names[0], "test", "id", args.which))["layers"]]
    layers_abs = [(args.n_layers + 1 + i) if i < 0 else i for i in layers]
    blocks = [int(x) for x in args.blocks.split(",")]
    li = [layers_abs.index(b) for b in blocks]
    budgets = [int(x) for x in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows: Dict[str, List[tuple]] = defaultdict(list)
    correct: Dict[str, List[np.ndarray]] = defaultdict(list)
    for name in names:
        d = data[name]
        tr, te = d["train_rand"], d["test_id"]
        raw_te = _softmax(te.Z)
        rows["raw"].append((raw_te, te.y))
        correct["raw"].append((raw_te.argmax(1) == te.y).astype(float))
        for n in budgets:
            for seed in seeds:
                idx = np.random.RandomState(seed).permutation(tr.n)[:n]
                T = TemperatureScaler.fit(_softmax(tr.Z[idx]), tr.y[idx])
                p1 = T.apply(raw_te)
                rows[f"L1@{n}"].append((p1, te.y))
                correct[f"L1@{n}"].append((p1.argmax(1) == te.y).astype(float))
                if n < max(8, 2 * te.K):
                    continue
                best = None
                for kind in ("lda", "ridge"):
                    try:
                        h = fit_head(tr.H[idx][:, li].astype(np.float32), tr.y[idx], te.K, kind=kind)
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if best is None or h.cv["oof_nll"] < best.cv["oof_nll"]:
                        best = h
                if best is None:
                    continue
                p = best.probs(te.H[:, li][:, best.layer].astype(np.float32))
                rows[f"head@{n}"].append((p, te.y))
                correct[f"head@{n}"].append((p.argmax(1) == te.y).astype(float))
        print(f"   {name:48s} done", flush=True)
    rng = np.random.RandomState(0)
    result: Dict[str, Any] = {"model": args.model, "blocks": blocks, "budgets": budgets, "seeds": seeds, "rows": {},
                              "date": dt.datetime.now().isoformat(), "env": environment()}
    for m, items in rows.items():
        rec = pooled(items)
        c = np.concatenate(correct[m])
        idx = rng.randint(0, len(c), (1000, len(c)))
        rec["acc_ci95"] = [float(np.percentile(c[idx].mean(1), 2.5)), float(np.percentile(c[idx].mean(1), 97.5))]
        if m.startswith("head@"):
            n = m.split("@")[1]
            c1 = np.concatenate(correct[f"L1@{n}"])
            # paired difference against L1 at the same budget (same seeds, same items in the same order)
            d = (c[idx] - c1[idx]).mean(1) if len(c1) == len(c) else np.zeros(1)
            rec["diff_vs_L1_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        result["rows"][m] = rec
    print(f"\n{'method':10s} {'acc':>6s} {'ci95':>16s} {'vs L1 (same N)':>18s} {'ece':>6s}")
    for m, r in result["rows"].items():
        d = r.get("diff_vs_L1_ci95")
        print(f"{m:10s} {r['acc']:6.3f} [{r['acc_ci95'][0]:.3f}, {r['acc_ci95'][1]:.3f}] "
              + (f"[{d[0]:+.3f}, {d[1]:+.3f}]" if d else " " * 18) + f" {r['ece']:6.3f}")
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.model.replace('/', '__')}.labels.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
