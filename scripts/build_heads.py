"""Build and validate the shipped Jev-mode artifact of one model: L2 heads for the typed-decisions
questions (and optionally the bench tasks), fit with `Decider.fit_head` on the real backend and
validated on held-out states through the same `decide_batch(level="L2")` path a user runs.

    python scripts/build_heads.py --model Qwen/Qwen3-8B --layers 22,24,28,36 --out anyjev-heads

Per question: `fit_head` on the calibration states (candidate blocks --layers; default 60 / 67 /
75 / 100 % of depth; the block and head kind are chosen by out-of-fold NLL), then `decide_batch`
on the test states: accuracy, pooled ECE, NLL, blocks executed, ms per decision on this GPU.
When the block-loop study cache (written by bench.extract_pools) exists for the question, the
artifact head is also applied to the cached test features and compared with the live
probabilities (mean |dp|, argmax disagreement): the check that the shipped path reads the same
states as the studies.

Output: <out>/<model>.json = `Decider.export_artifacts()` plus metadata (n_blocks, hidden_size,
candidates, validation per question and pooled, env), loadable with `Decider.load_artifacts`;
the validation table is also written to bench/results_exit/<date>/<model>.artifact.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anyjev  # noqa: E402
from anyjev import Decider  # noqa: E402
from anyjev.heads import LinearHead  # noqa: E402
from bench import metrics  # noqa: E402
from bench.pools import pool_path, task_records, typed_records  # noqa: E402
from bench.run import environment  # noqa: E402


def pooled(rows: List[tuple]) -> Dict[str, float]:
    conf = np.concatenate([p.max(1) for p, _ in rows])
    corr = np.concatenate([(p.argmax(1) == y) for p, y in rows]).astype(float)
    nll = np.concatenate([-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)) for p, y in rows])
    return {"acc": float(corr.mean()), "ece": metrics.ece(np.stack([conf, 1 - conf], 1), (1 - corr).astype(int)),
            "nll": float(nll.mean()), "n": int(len(corr))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layers", default=None, help="candidate blocks, absolute; default 60/67/75/100%% of depth")
    ap.add_argument("--calib-cases", type=int, default=300)
    ap.add_argument("--bench-tasks", default="", help="comma list of bench tasks to include, e.g. banking20,injection")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="anyjev-heads")
    ap.add_argument("--cache", default="bench/results_exit", help="study cache root for the agreement check")
    ap.add_argument("--which", default="last")
    args = ap.parse_args(argv)
    from anyjev.backends.hf import HFBackend

    be = HFBackend(args.model, batch_size=args.batch_size)
    L = be.n_layers
    if args.layers:
        candidates = sorted({int(x) for x in args.layers.split(",")})
    else:
        candidates = sorted({int(round(f * L)) for f in (0.6, 0.67, 0.75, 1.0)})
    dec = Decider(be)
    records = typed_records(args.calib_cases)
    if args.bench_tasks:
        records += task_records(args.bench_tasks.split(","), args.n_test, args.n_calib, 0, "bench")
    print(f"{args.model}: {L} blocks, hidden {be.hidden_size}; candidates {candidates}; {len(records)} questions",
          flush=True)
    validation: Dict[str, Any] = {}
    rows_all: List[tuple] = []
    for rec in records:
        name, q = rec["name"], rec["q"]
        s_tr, y_tr, _ = rec["calib"]
        s_te, y_te, _ = rec["test"]
        t0 = time.perf_counter()
        art = dec.fit_head(q, s_tr, y_tr, layers=candidates)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        decs = dec.decide_batch(s_te, q, level="L2")
        t_dec = time.perf_counter() - t0
        p = np.stack([d.probs for d in decs])
        y = np.asarray(y_te)
        v = {"K": q.k, "n_calib": len(y_tr), "n_test": len(y), "layer_abs": art["layer_abs"],
             "relative_depth": art["layer_abs"] / L, "method": art["method"], "oof_acc": art["cv"]["oof_acc"],
             "oof_nll": art["cv"]["oof_nll"], "temperature": art["temperature"],
             "acc": metrics.accuracy(p, y), "ece": metrics.ece(p, y), "nll": metrics.nll(p, y),
             "fit_seconds": t_fit, "ms_per_decision": 1000 * t_dec / len(y)}
        cache_path = pool_path(args.cache, args.model, name, "test", "id", args.which)
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            layers = [int(x) for x in z["layers"]]
            blocks = [(L + 1 + i) if i < 0 else i for i in layers]
            if art["layer_abs"] in blocks and len(z["y"]) == len(y):
                H = z["H"][:, blocks.index(art["layer_abs"])].astype(np.float32)
                p_cache = LinearHead.from_dict(art).probs(H)
                v["cache_mean_abs_dp"] = float(np.mean(np.abs(p_cache - p)))
                v["cache_argmax_disagreement"] = float(np.mean(p_cache.argmax(1) != p.argmax(1)))
        validation[name] = v
        rows_all.append((p, y))
        extra = (f"  |dp| vs cache {v['cache_mean_abs_dp']:.4f} argmax diff {v['cache_argmax_disagreement']:.3f}"
                 if "cache_mean_abs_dp" in v else "")
        print(f"   {name:48s} K={q.k} block {art['layer_abs']:3d} {art['method']:10s} oof {v['oof_acc']:.3f} "
              f"test acc {v['acc']:.3f} ece {v['ece']:.3f}  fit {t_fit:5.1f}s  {v['ms_per_decision']:6.1f} ms/decision"
              + extra, flush=True)
    typed_rows = [(p, y) for (p, y), r in zip(rows_all, records) if r["domain"] == "typed"]
    summary = {"typed_pooled": pooled(typed_rows) if typed_rows else None, "all_pooled": pooled(rows_all),
               "mean_blocks": float(np.mean([v["layer_abs"] for v in validation.values()])),
               "mean_ms_per_decision": float(np.mean([v["ms_per_decision"] for v in validation.values()])),
               "cache_mean_abs_dp": float(np.mean([v["cache_mean_abs_dp"] for v in validation.values()
                                                   if "cache_mean_abs_dp" in v] or [np.nan]))}
    print(f"\npooled typed: {summary['typed_pooled']}\nmean blocks {summary['mean_blocks']:.1f} of {L}; "
          f"{summary['mean_ms_per_decision']:.1f} ms per decision; |dp| vs cache {summary['cache_mean_abs_dp']:.4f}")
    artifact = dec.export_artifacts()
    artifact.update({"anyjev": anyjev.__version__, "n_blocks": L, "hidden_size": be.hidden_size,
                     "candidates": candidates, "questions": {k: v["question_id"] for k, v in artifact["heads"].items()},
                     "validation": {"per_question": validation, "summary": summary},
                     "date": dt.datetime.now().isoformat(), "env": environment(batch_size=args.batch_size)})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.model.replace('/', '__')}.json")
    with open(path, "w") as f:
        json.dump(artifact, f)
    print("wrote", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")
    outdir = os.path.join("bench/results_exit", dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    vpath = os.path.join(outdir, f"{args.model.replace('/', '__')}.artifact.json")
    with open(vpath, "w") as f:
        json.dump({"model": args.model, "n_blocks": L, "candidates": candidates, "validation": validation,
                   "summary": summary, "date": artifact["date"], "env": artifact["env"]}, f, indent=1)
    print("wrote", vpath)


if __name__ == "__main__":
    main()
