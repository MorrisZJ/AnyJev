"""Run AnyJev on LocalLLaMA/typed-decisions (the set Laya and Jev report on).

    python -m bench.run_typed --model Qwen/Qwen3-8B [--limit-cases 50] [--calib-cases 200]

raw and L0 are zero-shot. L1 fits one temperature per question on the train
split (labels used for temperature only). Reports overall, per workflow and
per question type: argmax accuracy, soft accuracy (sum p_pred * p_gold),
ECE (15 equal-mass bins), Brier (sum over classes, and mean over classes as
Laya reports it), and score MAE (|E[level] - gold score|).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from anyjev import Decider
from bench import metrics
from bench.run import _rows_from_diagnostics, environment
from bench.tasks.typed_decisions import Decision, group_by_question, load

PUBLISHED = {  # from Laya's BENCHMARKS.md, 2026-09-20; not rerun here
    "laya-typed-decisions (fine-tuned on train; published)":
        {"acc": 0.766, "soft_acc": 0.471, "brier_mean": 0.061, "ece": 0.213, "score_mae": 0.242},
    "laya (zero-shot; published)":
        {"acc": 0.361, "soft_acc": 0.332, "brier_mean": 0.316, "ece": 0.175, "score_mae": 0.694},
    "Jev 1.13.0 (published by Laya)":
        {"acc": 0.727, "soft_acc": 0.580, "brier_mean": 0.148, "ece": 0.144, "score_mae": 0.391},
}


def summarize(items: List[Decision], probs: np.ndarray) -> Dict[str, float]:
    labels = [d.gold_index for d in items]
    gold_soft = np.array([d.gold_probs for d in items]) if len({len(d.gold_probs) for d in items}) == 1 else None
    out = {"n": len(items), "acc": metrics.accuracy(probs, labels), "ece": metrics.ece(probs, labels),
           "brier": metrics.brier(probs, labels), "brier_mean": metrics.brier(probs, labels) / probs.shape[1],
           "nll": metrics.nll(probs, labels)}
    if gold_soft is not None:
        out["soft_acc"] = float(np.mean(np.sum(probs * gold_soft, axis=1)))
    return out


def pooled(rows: List[tuple]) -> Dict[str, float]:
    """rows: (Decision, probs[K]) with possibly different K: pool the scalar
    metrics that do not need equal K."""
    conf = np.array([p.max() for _, p in rows])
    corr = np.array([int(np.argmax(p)) == d.gold_index for d, p in rows], dtype=float)
    soft = np.array([float(np.dot(p, d.gold_probs)) for d, p in rows])
    brier = np.array([float(np.sum((p - np.eye(len(p))[d.gold_index]) ** 2)) for d, p in rows])
    brier_mean = np.array([float(np.mean((p - np.eye(len(p))[d.gold_index]) ** 2)) for d, p in rows])
    nll = np.array([-np.log(max(p[d.gold_index], 1e-12)) for d, p in rows])
    # ECE on pooled confidences, 15 equal-mass bins
    order = np.argsort(conf)
    e = 0.0
    for chunk in np.array_split(order, min(15, len(order))):
        if len(chunk):
            e += len(chunk) / len(conf) * abs(conf[chunk].mean() - corr[chunk].mean())
    out = {"n": len(rows), "acc": float(corr.mean()), "soft_acc": float(soft.mean()), "ece": float(e),
           "brier": float(brier.mean()), "brier_mean": float(brier_mean.mean()), "nll": float(nll.mean())}
    sc = [(d, p) for d, p in rows if d.gold_score is not None]
    if sc:
        out["score_mae"] = float(np.mean(
            [abs(float(np.dot(p, d.question.bin_centers())) - d.gold_score) for d, p in sc]))
    return out


def run(decider: Decider, test: List[Decision], calib: List[Decision], levels: List[str]) -> Dict[str, Any]:
    groups = group_by_question(test)
    calib_groups = group_by_question(calib) if calib else {}
    per_level: Dict[str, List[tuple]] = defaultdict(list)
    t0 = time.time()
    for key, items in groups.items():
        q = items[0].question
        states = [d.state for d in items]
        decs = decider.decide_batch(states, q, level="L0")
        rows = _rows_from_diagnostics(decs, decider.combine)
        for name, P in rows.items():
            for d, p in zip(items, P):
                per_level[name].append((d, p))
        if "L1" in levels and key in calib_groups:
            cal = calib_groups[key]
            decider.calibrate(q, [d.state for d in cal], [d.gold_index for d in cal])
            for d, dec in zip(items, decider.decide_batch(states, q, level="L1")):
                per_level["L1"].append((d, dec.probs))
    seconds = time.time() - t0

    out: Dict[str, Any] = {"levels": {}, "seconds": seconds, "n_decisions": len(test),
                           "n_questions": len(groups), "n_calib_decisions": len(calib)}
    for level, rows in per_level.items():
        if level not in levels and not level.startswith("L0-"):
            continue
        entry = {"overall": pooled(rows), "by_workflow": {}, "by_type": {}}
        for wf in sorted({d.workflow for d, _ in rows}):
            entry["by_workflow"][wf] = pooled([r for r in rows if r[0].workflow == wf])
        for t in ("choice", "noul", "score"):
            sub = [r for r in rows if r[0].question.kind == t]
            if sub:
                entry["by_type"][t] = pooled(sub)
        out["levels"][level] = entry
    return out


def markdown(results: Dict[str, Any]) -> str:
    cols = ["acc", "soft_acc", "ece", "brier_mean", "score_mae"]
    lines = ["| system | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    model = results["model"].split("/")[-1]
    tags = {"raw": "raw logits (clone baseline)", "L0": "AnyJev L0 (zero-shot)",
            "L1": "AnyJev L1 (temperature from train labels)"}
    for level, e in results["levels"].items():
        m = e["overall"]
        tag = tags.get(level, f"ablation {level} (zero-shot)")
        lines.append(f"| {model} + {tag} | " + " | ".join(f"{m[c]:.3f}" if c in m else "" for c in cols) + " |")
    for name, m in PUBLISHED.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" if c in m else "" for c in cols) + " |")
    lines.append("")
    lines.append("Per workflow (acc / soft_acc), AnyJev L0:")
    l0 = results["levels"].get("L0", {}).get("by_workflow", {})
    for wf, m in l0.items():
        lines.append(f"- {wf}: {m['acc']:.3f} / {m['soft_acc']:.3f} (n={m['n']})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="raw,L0,L1")
    ap.add_argument("--limit-cases", type=int, default=None, help="test cases per workflow")
    ap.add_argument("--calib-cases", type=int, default=50, help="train cases per workflow used for L1 temperature")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prior", default="batch", choices=["batch", "content_free", "none"])
    ap.add_argument("--out", default="bench/results_typed")
    args = ap.parse_args(argv)

    from anyjev.backends.hf import HFBackend
    decider = Decider(HFBackend(args.model, batch_size=args.batch_size), prior=args.prior)
    levels = args.levels.split(",")
    test = load("test", limit_cases=args.limit_cases)
    calib = load("train", limit_cases=args.calib_cases) if "L1" in levels else []
    print(f"test decisions: {len(test)} over {len(group_by_question(test))} questions; calib: {len(calib)}", flush=True)

    results = {"model": args.model, "dataset": "LocalLLaMA/typed-decisions", "prior": args.prior,
               "limit_cases": args.limit_cases, "env": environment(), **run(decider, test, calib, levels)}
    stamp = dt.datetime.now().strftime("%Y-%m-%d")
    outdir = os.path.join(args.out, stamp)
    os.makedirs(outdir, exist_ok=True)
    slug = args.model.replace("/", "__")
    with open(os.path.join(outdir, f"{slug}.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    table = markdown(results)
    with open(os.path.join(outdir, f"{slug}.md"), "w") as f:
        f.write(table + "\n")
    print(json.dumps({k: v["overall"] for k, v in results["levels"].items()}, indent=1))
    print(table)


if __name__ == "__main__":
    main()
