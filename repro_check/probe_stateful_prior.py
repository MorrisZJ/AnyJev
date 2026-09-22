"""Is the L1 result a function of invocation history?

The batch prior is a running accumulator on the Decider, keyed by question:

    self._running[q.key] = (s + stack.sum(axis=0), n + len(stack))

so the prior applied during `calibrate()` is the mean over every item the
decider has seen for that question so far -- test split included. The fitted
temperature therefore depends on how many times the same question has been
decided before calibration, which is invocation history rather than data.

This probe fits the same temperature on the same 200 calibration items three
times in one process and reports whether the answer moves.

    python repro_check/probe_stateful_prior.py --model Qwen/Qwen3-8B --task banking20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from anyjev import Decider  # noqa: E402
from bench import metrics  # noqa: E402
from bench.tasks import get_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="banking20")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from anyjev.backends.hf import HFBackend

    task = get_task(args.task)
    test, calib = task.split(args.n, args.calib, 0)
    q = task.question
    test_states = [s for s, _ in test]
    test_labels = [y for _, y in test]
    cal_states = [s for s, _ in calib]
    cal_labels = [y for _, y in calib]

    backend = HFBackend(args.model, batch_size=args.batch_size)
    dec = Decider(backend, prior="batch", record_content_free=True)

    out = {"model": args.model, "task": args.task, "rounds": []}

    # Round 0: the bench's own sequence -- test pass, reversed pass, then calibrate.
    print("round 0: bench.run's sequence (test pass, then calibrate)", flush=True)
    dec.decide_batch(test_states, q, level="L0")
    n_seen = dec._running[q.key][1]
    art = dec.calibrate(q, cal_states, cal_labels)
    probs = np.stack([d.probs for d in dec.decide_batch(test_states, q, level="L1")])
    out["rounds"].append({"round": 0, "items_seen_before_calibrate": int(n_seen),
                          "temperature": art["temperature"],
                          "ece": metrics.ece(probs, test_labels),
                          "cov@5%": metrics.coverage_at_risk(probs, test_labels),
                          "acc": metrics.accuracy(probs, test_labels)})
    print(f"   seen before calibrate = {n_seen}, T = {art['temperature']:.12f}", flush=True)

    # Rounds 1-2: identical calls again. Same data, same code, more accumulated prior.
    for r in (1, 2):
        n_seen = dec._running[q.key][1]
        art = dec.calibrate(q, cal_states, cal_labels)
        probs = np.stack([d.probs for d in dec.decide_batch(test_states, q, level="L1")])
        out["rounds"].append({"round": r, "items_seen_before_calibrate": int(n_seen),
                              "temperature": art["temperature"],
                              "ece": metrics.ece(probs, test_labels),
                              "cov@5%": metrics.coverage_at_risk(probs, test_labels),
                              "acc": metrics.accuracy(probs, test_labels)})
        print(f"round {r}: seen before calibrate = {n_seen}, T = {art['temperature']:.12f}", flush=True)

    # A fresh decider that calibrates FIRST, without ever seeing the test split:
    # the order a user would naturally write, and a different answer.
    dec2 = Decider(backend, prior="batch", record_content_free=True)
    art2 = dec2.calibrate(q, cal_states, cal_labels)
    probs2 = np.stack([d.probs for d in dec2.decide_batch(test_states, q, level="L1")])
    out["calibrate_first"] = {"items_seen_before_calibrate": 0,
                              "temperature": art2["temperature"],
                              "ece": metrics.ece(probs2, test_labels),
                              "cov@5%": metrics.coverage_at_risk(probs2, test_labels),
                              "acc": metrics.accuracy(probs2, test_labels)}
    print(f"calibrate-first (fresh decider): T = {art2['temperature']:.12f}", flush=True)

    ts = [r["temperature"] for r in out["rounds"]] + [out["calibrate_first"]["temperature"]]
    eces = [r["ece"] for r in out["rounds"]] + [out["calibrate_first"]["ece"]]
    covs = [r["cov@5%"] for r in out["rounds"]] + [out["calibrate_first"]["cov@5%"]]
    accs = [r["acc"] for r in out["rounds"]] + [out["calibrate_first"]["acc"]]
    out["spread"] = {"temperature": max(ts) - min(ts), "ece": max(eces) - min(eces),
                     "cov@5%": max(covs) - min(covs), "acc": max(accs) - min(accs)}
    print()
    print(f"SPREAD over identical calls that differ only in accumulated prior state:")
    print(f"   temperature {min(ts):.6f} .. {max(ts):.6f}   (spread {max(ts) - min(ts):.6f})")
    print(f"   ece         {min(eces):.4f} .. {max(eces):.4f}   (spread {max(eces) - min(eces):.4f})")
    print(f"   cov@5%      {min(covs):.4f} .. {max(covs):.4f}   (spread {max(covs) - min(covs):.4f})")
    print(f"   acc         {min(accs):.4f} .. {max(accs):.4f}   (spread {max(accs) - min(accs):.4f})")
    print(f"   repo's committed value for this cell: see bench/results_batchprior_v0")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    p = os.path.join(HERE, "results", f"probe_stateful_prior.{args.model.replace('/', '__')}.{args.task}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
