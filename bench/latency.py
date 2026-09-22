"""Latency: what L0 costs relative to raw, per backend, per K, per state length.

    python -m bench.latency --model Qwen/Qwen3-8B --backend hf --k 4,20 --state-tokens 100,1000
    python -m bench.latency --model Qwen/Qwen2.5-7B-Instruct --backend vllm --base-url http://127.0.0.1:8011

Two regimes, both reported in milliseconds per decision:
  batch   : `decide_batch` over n states (throughput-bound; what the bench does)
  single  : `decide` on one state at a time (latency-bound; what an agent loop does)
For the transformers backend each L0 number is measured with the shared-prefix path off
and on. The batch prior is used (no extra prompts), so L0's only overhead is K prefills.
Every measurement uses states never seen before in the process: a server-side prefix cache
(vLLM) can only help between the K shifts of one state, which is the deployment case.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List

from anyjev import Decider, Question

BASE = ("Customer message: I was charged twice for order #4471 last Tuesday, the refund has not "
        "arrived, and the app crashes every time I open the receipts page. I have already tried "
        "reinstalling and clearing the cache. Please advise on next steps. ")


_counter = [0]


def make_states(tokenizer, n: int, target_tokens: int) -> List[str]:
    """n states of about target_tokens tokens, each one never issued before in this process,
    so a server-side prefix cache cannot carry work between measurements."""
    unit = len(tokenizer.encode(BASE, add_special_tokens=False))
    reps = max(1, round(target_tokens / unit))
    out = []
    for _ in range(n):
        _counter[0] += 1
        out.append(f"[ticket {_counter[0]:06d} ref {_counter[0] * 7919 % 100000}] " + BASE * reps)
    return out


def timed(fn, repeat: int = 1) -> float:
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def run(backend_factory, name: str, ks: List[int], state_tokens: List[int], n: int,
        single_n: int, modes: List[Any]) -> List[Dict[str, Any]]:
    rows = []
    be = backend_factory()
    for T in state_tokens:
        states = make_states(be.tokenizer, n, T)
        n_tok = len(be.tokenizer.encode(states[0], add_special_tokens=False))
        for k in ks:
            q = Question.choice("Which team should handle this ticket?", [f"team_{i}" for i in range(k)], name="t")
            row: Dict[str, Any] = {"backend": name, "state_tokens": n_tok, "k": k, "n": n}
            for mode in modes:
                d = Decider(be, prior="batch", shared_prefix=mode)
                tag = {False: "full", "auto": "shared", True: "shared"}[mode]
                d.decide_batch(make_states(be.tokenizer, 2, T), q, level="raw")      # warm-up
                d.decide_batch(make_states(be.tokenizer, 2, T), q, level="L0")
                raw_b = timed(lambda: d.decide_batch(make_states(be.tokenizer, n, T), q, level="raw")) / n * 1000
                l0_b = timed(lambda: d.decide_batch(make_states(be.tokenizer, n, T), q, level="L0")) / n * 1000
                raw_s = sum(timed(lambda: d.decide(make_states(be.tokenizer, 1, T)[0], [q], level="raw"))
                            for _ in range(single_n)) / single_n * 1000
                l0_s = sum(timed(lambda: d.decide(make_states(be.tokenizer, 1, T)[0], [q], level="L0"))
                           for _ in range(single_n)) / single_n * 1000
                row["batch_raw_ms"] = raw_b
                row[f"batch_L0_{tag}_ms"] = l0_b
                row["single_raw_ms"] = raw_s
                row[f"single_L0_{tag}_ms"] = l0_s
                if hasattr(be, "shared_fallbacks"):
                    row["shared_fallbacks"] = be.shared_fallbacks
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def markdown(rows: List[Dict[str, Any]]) -> str:
    has_shared = any("batch_L0_shared_ms" in r for r in rows)
    cols = ["backend", "state_tokens", "k", "batch_raw_ms", "batch_L0_full_ms"]
    if has_shared:
        cols.append("batch_L0_shared_ms")
    cols += ["single_raw_ms", "single_L0_full_ms"]
    if has_shared:
        cols.append("single_L0_shared_ms")
    lines = ["| " + " | ".join(cols) + " | L0/raw single (full) | L0/raw single (shared) |",
             "|" + "---|" * (len(cols) + 2)]
    for r in rows:
        cells = [f"{r[c]:.1f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols]
        ratio_full = r["single_L0_full_ms"] / r["single_raw_ms"] if "single_L0_full_ms" in r else float("nan")
        ratio_shared = r["single_L0_shared_ms"] / r["single_raw_ms"] if "single_L0_shared_ms" in r else float("nan")
        shared_cell = f"{ratio_shared:.2f}x" if has_shared else ""
        lines.append("| " + " | ".join(cells) + f" | {ratio_full:.2f}x | {shared_cell} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--k", default="4,20")
    ap.add_argument("--state-tokens", default="100,1000")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--single-n", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="bench/results_latency")
    args = ap.parse_args(argv)
    ks = [int(x) for x in args.k.split(",")]
    toks = [int(x) for x in args.state_tokens.split(",")]

    if args.backend == "hf":
        from anyjev.backends.hf import HFBackend
        factory = lambda: HFBackend(args.model, batch_size=args.batch_size)  # noqa: E731
        modes = [False, "auto"]
    else:
        from anyjev.backends.vllm import VLLMBackend
        factory = lambda: VLLMBackend(args.base_url, args.model)  # noqa: E731
        modes = [False]
    rows = run(factory, args.backend, ks, toks, args.n, args.single_n, modes)
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    slug = f"{args.backend}.{args.model.replace('/', '__')}"
    with open(os.path.join(outdir, slug + ".json"), "w") as f:
        json.dump({"model": args.model, "backend": args.backend, "rows": rows,
                   "date": dt.datetime.now().isoformat()}, f, indent=1)
    table = markdown(rows)
    with open(os.path.join(outdir, slug + ".md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
