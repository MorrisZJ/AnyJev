"""Milliseconds per decision as a function of depth: the cost side of the Jev-mode table.

    python -m bench.exit_latency --model Qwen/Qwen3-8B --depths 0.33,0.5,0.6,0.75,1.0 --state-tokens 110,1000

For each depth fraction the block loop runs only the first blocks (`HFBackend.hidden_states_to`
with `max_layer`), which is what a fixed-depth head costs at inference: one prefill of a truncated
model plus a matrix-vector product. Measured on never-seen states, batch (32 prompts) and single
request; the full-depth raw readout (`next_token_logprobs`) is the reference. FLOPs per decision
are computed from the loaded model's own block parameter count (2 x params per block x tokens x
blocks), never typed by hand.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from anyjev import Question
from anyjev.readout import DEFAULT_SYSTEM, build_prompt, label_ids_for_perm, render_chat, resolve_labels
from bench.latency import make_states
from bench.run import environment


def timed(fn, repeat: int = 1):
    import torch

    fn()                                      # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def block_params(model) -> int:
    layer = model.model.layers[0]
    return int(sum(p.numel() for p in layer.parameters()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--depths", default="0.33,0.5,0.6,0.75,1.0")
    ap.add_argument("--state-tokens", default="110,1000")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--single-n", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="bench/results_exit")
    args = ap.parse_args(argv)
    from anyjev.backends.hf import HFBackend

    be = HFBackend(args.model, batch_size=args.batch_size)
    L = be.n_layers
    pb = block_params(be.model)
    q = Question.choice("Which team should handle this?", [f"team {i}" for i in range(args.k)])
    labels, ids = resolve_labels(be.tokenizer, q)
    perm = list(range(args.k))
    token_ids = [label_ids_for_perm(q, ids, perm)]
    rows: List[Dict[str, Any]] = []
    for st in [int(x) for x in args.state_tokens.split(",")]:
        states = make_states(be.tokenizer, args.n, st)
        prompts = [render_chat(be.tokenizer, build_prompt(s, q, perm, DEFAULT_SYSTEM, labels)) for s in states]
        n_tok = int(np.mean([len(be.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]))
        raw_batch = timed(lambda: be.next_token_logprobs(prompts, token_ids * len(prompts))) / len(prompts) * 1000
        raw_single = np.mean([timed(lambda p=p: be.next_token_logprobs([p], token_ids), repeat=3)
                              for p in prompts[:args.single_n]]) * 1000
        rows.append({"state_tokens": st, "prompt_tokens": n_tok, "depth": 1.0, "blocks": L,
                     "mode": "raw (plain forward)", "batch_ms": raw_batch, "single_ms": float(raw_single),
                     "gflops": 2 * pb * n_tok * L / 1e9})
        for frac in [float(x) for x in args.depths.split(",")]:
            d = max(1, int(round(frac * L)))
            head_batch = timed(lambda d=d: be.hidden_states_to(prompts, [d], None, None, max_layer=d))
            head_batch = head_batch / len(prompts) * 1000
            head_single = np.mean([timed(lambda p=p, d=d: be.hidden_states_to([p], [d], None, None, max_layer=d),
                                         repeat=3) for p in prompts[:args.single_n]]) * 1000
            rows.append({"state_tokens": st, "prompt_tokens": n_tok, "depth": frac, "blocks": d,
                         "mode": "truncated forward (head at this depth)", "batch_ms": head_batch,
                         "single_ms": float(head_single), "gflops": 2 * pb * n_tok * d / 1e9})
            print(f"{st:5d} tokens  depth {frac:.2f} ({d:2d}/{L} blocks): batch {head_batch:7.1f} ms  "
                  f"single {head_single:7.1f} ms  "
                  f"({head_batch / raw_batch:.2f}x / {head_single / raw_single:.2f}x of raw)",
                  flush=True)
    result = {"model": args.model, "n_layers": L, "params_per_block": pb, "hidden_size": be.hidden_size,
              "rows": rows, "date": dt.datetime.now().isoformat(), "env": environment(batch_size=args.batch_size)}
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.model.replace('/', '__')}.latency.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
