"""Do the serving levers stack, in speed and in accuracy, on vLLM?

    python -m bench.vllm_stack --config full28=http://127.0.0.1:8030 \
        --config b18=http://127.0.0.1:8031 --config b18fp8=http://127.0.0.1:8032 \
        --config b18noapc=http://127.0.0.1:8033

Three levers were measured one at a time and each looked worth having: a prefix cache (1.9x on
long states), a model truncated to 64% of its blocks (1.5x, and +2 accuracy points because a
middle block is a better feature space for a linear head than the last one), FP8 (1.16x on long
prompts, negative on short ones). Multiplying those numbers together gives 3.3x, which is a
guess. This measures the product instead.

Accuracy and speed come from the same run, because a lever that buys throughput by damaging the
decision is not a lever. Each configuration fits **its own** head -- a head belongs to the
artifact that serves it, and a truncated model emits `norm(h_b)` where the full one emits the
final state, so sharing a head across configurations would measure the wrong thing.

Two workloads, because they exercise the cache differently:

  single    one question per state, the shape a classifier serves.
  agent     several questions about one state, the shape AnyJev was built for. The state's keys
            and values are computed once and every later question reads them, so this is where a
            prefix cache can pay on the L2 path at all.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from bench.run import environment
from bench.tasks import get_task

EXTRA_QUESTIONS = [
    ("urgency", "How urgent is this customer message?",
     ["can wait", "this week", "today", "right now"]),
    ("sentiment", "What is the customer's tone?", ["calm", "annoyed", "angry"]),
    ("handoff", "Does this message need a human agent?", ["no", "yes"]),
    ("refund", "Is the customer asking for money back?", ["no", "yes"]),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[],
                    help="name=url, repeatable; the served-model-name must match the name")
    ap.add_argument("--source", default="Qwen/Qwen2.5-7B-Instruct",
                    help="where tokenizer and config come from for every configuration")
    ap.add_argument("--task", default="banking20")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--calib", type=int, default=300)
    ap.add_argument("--agent-questions", type=int, default=4)
    ap.add_argument("--out", default="bench/results_vllm")
    args = ap.parse_args(argv)

    from anyjev import Decider, Question
    from anyjev.backends.vllm import VLLMBackend

    task = get_task(args.task)
    test, calib = task.split(args.n_test, args.calib, seed=0)
    q = task.question
    cal_x = [s for s, _ in calib]
    cal_y = [y for _, y in calib]
    te_x = [s for s, _ in test]
    te_y = np.array([y for _, y in test])
    extra = [Question.choice(text, opts, name=name)
             for name, text, opts in EXTRA_QUESTIONS[:args.agent_questions]]
    print(f"{task.name}: {len(cal_x)} calibration, {len(te_x)} test, K={q.k}; "
          f"agent workload adds {len(extra)} questions per state")

    rows: List[Dict[str, Any]] = []
    for spec in args.config:
        name, url = spec.split("=", 1)
        be = VLLMBackend(url, name, tokenizer_name=args.source)
        dec = Decider(be, level="L2")
        t0 = time.perf_counter()
        art = dec.fit_head(q, cal_x, cal_y, layers=[-1])
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        P = np.stack([x.probs for x in dec.decide_batch(te_x, q, level="L2")])
        single_s = time.perf_counter() - t0
        acc = float((P.argmax(1) == te_y).mean())

        # the agent shape: the routing question plus several more about the same states
        for eq in extra:
            dec.fit_head(eq, cal_x[:64], [i % eq.k for i in range(64)], layers=[-1])
        sub = te_x[:64]
        t0 = time.perf_counter()
        for eq in [q] + extra:
            dec.decide_batch(sub, eq, level="L2")
        agent_s = time.perf_counter() - t0

        n_agent = len(sub) * (1 + len(extra))
        row = {"config": name, "blocks": be.n_layers, "accuracy": acc, "fit_s": fit_s,
               "single_ms": single_s / len(te_x) * 1000,
               "agent_ms": agent_s / n_agent * 1000,
               "n_test": len(te_x), "n_agent_decisions": n_agent,
               "head": art["method"], "layer_abs": art["layer_abs"]}
        rows.append(row)
        print(f"  {name:<10} blocks {row['blocks']:>3}  acc {acc:.3f}  "
              f"single {row['single_ms']:6.1f} ms  agent {row['agent_ms']:6.1f} ms")

    if rows:
        base = rows[0]
        print(f"\n{'config':<12}{'acc':>7}{'d acc':>8}{'single ms':>11}{'speedup':>9}"
              f"{'agent ms':>10}{'speedup':>9}")
        for r in rows:
            print(f"{r['config']:<12}{r['accuracy']:>7.3f}"
                  f"{r['accuracy'] - base['accuracy']:>+8.3f}{r['single_ms']:>11.1f}"
                  f"{base['single_ms'] / r['single_ms']:>9.2f}{r['agent_ms']:>10.1f}"
                  f"{base['agent_ms'] / r['agent_ms']:>9.2f}")
        print(f"\n  speedups are against {base['config']}; the agent column is "
              f"{1 + len(extra)} questions per state")

    out = {"task": task.name, "rows": rows, "env": environment(configs=args.config)}
    stamp = dt.date.today().isoformat()
    os.makedirs(os.path.join(args.out, stamp), exist_ok=True)
    path = os.path.join(args.out, stamp, "vllm_stack.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
