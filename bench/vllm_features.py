"""What vLLM's prefix cache and FP8 are actually worth to a typed decision.

    python -m bench.vllm_features --apc-on http://127.0.0.1:8020 --apc-off http://127.0.0.1:8021 \
                                  --fp8 http://127.0.0.1:8022 --bf16 http://127.0.0.1:8023

Two things get claimed a lot and measured rarely.

**Prefix caching.** L0 scores every cyclic rotation of the option list, so one decision is K
prompts that share the state and the question and diverge at the option block. If the cache
works on that shape, the state -- which is the long part, a thousand tokens and up in a real
agent context -- is computed once per decision instead of K times. The test is the same K
prompts drawn two ways: K rotations of one state (a shared prefix) against K prompts from K
different states (none), at matched token counts, on servers that differ only in the flag.

**FP8.** Prefill is compute bound, so a weight-only format that saves bandwidth buys nothing
here; only a format with real low-precision matmul throughput can. That is an argument, and
arguments about performance are worth exactly one measurement.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import time
import urllib.request
from typing import Any, Dict, Sequence

import numpy as np

from anyjev.calibrate.permute import cyclic_shifts
from anyjev.question import Question
from anyjev.readout import build_prompt, render_chat_parts, resolve_labels
from bench.run import environment


def _post(url: str, path: str, body: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def model_of(url: str) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def time_prompts(url: str, prompts: Sequence[str], ids: Sequence[int], workers: int = 16,
                 embed: bool = False, warmup: Sequence[str] = ()) -> float:
    """Seconds for one pass over `prompts`, which the server has never seen.

    The warm-up uses *different* text on purpose. Sending the timed prompts first, as an earlier
    version did, puts every one of them in the prefix cache, so the timed pass measures cache
    hits against cache hits and the shared-prefix question never gets asked -- which is how that
    version came to report a shared prefix being three times slower than no prefix at all."""
    name = model_of(url)

    def one(p: str):
        if embed:
            return _post(url, "/v1/embeddings", {"model": name, "input": p,
                                                 "encoding_format": "float"})
        return _post(url, "/v1/completions",
                     {"model": name, "prompt": p, "max_tokens": 1, "temperature": 0.0,
                      "logprobs": len(ids), "allowed_token_ids": list(ids)})

    with cf.ThreadPoolExecutor(workers) as ex:
        if warmup:
            list(ex.map(one, warmup))                    # get the server hot, cache something else
        t0 = time.perf_counter()
        list(ex.map(one, prompts))
        return time.perf_counter() - t0


def build(tokenizer, q: Question, state: str, perm) -> str:
    lab, _ = resolve_labels(tokenizer, q)
    from anyjev.decider import DEFAULT_SYSTEM
    pre, suf = render_chat_parts(tokenizer, build_prompt(state, q, list(perm), DEFAULT_SYSTEM, lab))
    return pre + suf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apc-on", default="")
    ap.add_argument("--apc-off", default="")
    ap.add_argument("--fp8", default="")
    ap.add_argument("--bf16", default="")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--state-tokens", default="128,1024")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--groups", type=int, default=4, help="distinct states per pass")
    ap.add_argument("--out", default="bench/results_vllm")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    q = Question.choice("Which specialist agent should handle this request?",
                        ["billing", "network", "account", "roaming", "technical"], name="route")
    _, ids = resolve_labels(tok, q)
    perms = [list(p) for p in cyclic_shifts(q.k)]
    K = len(perms)
    out: Dict[str, Any] = {"K": K, "results": []}

    word = "The customer reports an intermittent failure on the southbound link. "

    def state_of(tag: str, n_tok: int) -> str:
        """A state of about n_tok tokens whose *first* words already carry `tag`, so two states
        with different tags share no prefix at all."""
        return f"Ticket {tag}. " + word.replace("southbound", f"southbound-{tag}") * (n_tok // 12 + 2)

    run = 0
    for n_tok in [int(x) for x in args.state_tokens.split(",") if x]:
        for rep in range(args.repeats):
            run += 1
            # every pass uses states the server has never seen, so nothing is a cache hit
            base = run * 1000
            shared_states = [state_of(f"s{base + i}", n_tok) for i in range(args.groups)]
            shared = [build(tok, q, st, pm) for st in shared_states for pm in perms]
            distinct = [build(tok, q, state_of(f"d{base + i}", n_tok), perms[0])
                        for i in range(args.groups * K)]
            warm = [build(tok, q, state_of(f"w{base + i}", n_tok), perms[0]) for i in range(4)]
            real = len(tok.encode(shared[0], add_special_tokens=False))

            if args.apc_on and args.apc_off:
                row: Dict[str, Any] = {"state_tokens": real, "K": K, "groups": args.groups,
                                       "n_prompts": len(shared), "rep": rep}
                for tag, url in (("apc_on", args.apc_on), ("apc_off", args.apc_off)):
                    row[f"{tag}_shared_s"] = time_prompts(url, shared, ids, warmup=warm)
                    row[f"{tag}_distinct_s"] = time_prompts(url, distinct, ids, warmup=warm)
                out["results"].append({"kind": "prefix_cache", **row})

            if args.fp8 and args.bf16 and rep == 0:
                qrow = {"state_tokens": real, "n_prompts": len(shared)}
                for tag, url in (("fp8", args.fp8), ("bf16", args.bf16)):
                    qrow[f"{tag}_s"] = time_prompts(url, shared, ids, embed=True, warmup=warm)
                out["results"].append({"kind": "quantization", **qrow})

    # ---- report, averaged over the repeats -------------------------------------------------
    pc = [r for r in out["results"] if r["kind"] == "prefix_cache"]
    if pc:
        print(f"\nprefix cache: {args.groups} states x K={K} rotations = "
              f"{args.groups * K} prompts per pass, none of them seen before")
        print(f"{'tokens':>7}{'':>3}{'APC':>7}{'K rotations':>14}{'K distinct':>13}{'gain':>8}")
        for n in sorted({r["state_tokens"] for r in pc}):
            rows = [r for r in pc if r["state_tokens"] == n]
            m = {k: float(np.mean([r[k] for r in rows])) for k in
                 ("apc_on_shared_s", "apc_on_distinct_s", "apc_off_shared_s", "apc_off_distinct_s")}
            for tag in ("on", "off"):
                sh, di = m[f"apc_{tag}_shared_s"], m[f"apc_{tag}_distinct_s"]
                print(f"{n:>7}{'':>3}{tag:>7}{sh * 1000:>12.0f}ms{di * 1000:>11.0f}ms"
                      f"{di / sh:>8.2f}")
            print(f"{'':>7}   cache on the shared shape: "
                  f"{m['apc_off_shared_s'] / m['apc_on_shared_s']:.2f}x")
    qz = [r for r in out["results"] if r["kind"] == "quantization"]
    if qz:
        print("\nFP8 vs bf16 on the L2 (embed) path")
        print(f"{'tokens':>7}{'bf16':>10}{'fp8':>10}{'speedup':>9}")
        for r in qz:
            print(f"{r['state_tokens']:>7}{r['bf16_s'] * 1000:>8.0f}ms{r['fp8_s'] * 1000:>8.0f}ms"
                  f"{r['bf16_s'] / r['fp8_s']:>9.2f}")

    out["env"] = environment(state_tokens=args.state_tokens, repeats=args.repeats)
    stamp = dt.date.today().isoformat()
    os.makedirs(os.path.join(args.out, stamp), exist_ok=True)
    path = os.path.join(args.out, stamp, "vllm_features.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
