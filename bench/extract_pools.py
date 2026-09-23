"""Build the block-loop feature caches the depth / Jev-mode / label / listing-order / distillation
studies replay (the extraction step of the removed question-agnostic-head study; the command
rename is listed in docs/migration_v3.md).

    python -m bench.extract_pools --model Qwen/Qwen3-1.7B --layers 8,10,12,14,16,18,20,22,24,26,28
    python -m bench.extract_pools --model Qwen/Qwen3-4B   --layers 12,14,16,18,20,22,24,26,28,30,32,34,36
    python -m bench.extract_pools --model Qwen/Qwen3-8B   --layers 12,14,16,18,20,22,24,26,28,30,32,34,36
    python -m bench.extract_pools --model Qwen/Qwen3-32B  --layers 16,20,24,28,32,36,40,44,48,52,56,60,64
    python -m bench.extract_pools --model Qwen/Qwen3-30B-A3B-Instruct-2507 --batch-size 8 \
        --layers 12,16,20,24,28,32,36,40,44,48

For every typed-decisions question (300 train / 100 test decisions) four pools are written to
<out>/features/<model>/typed.<workflow>.<qname>.{train.rand,test.id,test.rev,train.id}.<which>.npz
(`bench.pools.get_pool`: existing files are reused, never overwritten). --bench-tasks adds the
registry tasks as bench.<task> pools; scripts/build_heads.py compares its live heads against the
bench.<task>.test.id cache only when --n-test matches its own --n-test (default 200 there).
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

from bench.pools import get_pool, task_records, typed_records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layers", required=True,
                    help="comma list of absolute blocks to cache "
                         "(n_layers = after the final norm; negative counts back)")
    ap.add_argument("--out", default="bench/results_exit")
    ap.add_argument("--calib-cases", type=int, default=300)
    ap.add_argument("--train-order", default="rand", choices=["rand", "id"],
                    help="listing order of the training pool besides the canonical one")
    ap.add_argument("--which", default="last", choices=["last", "newline"])
    ap.add_argument("--bench-tasks", default="",
                    help="comma list of registry tasks to add, e.g. banking20,newsgroups,injection")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--plain-forward", dest="use_loop", action="store_false", default=True,
                    help="one full forward with output_hidden_states instead of the block loop (no logit lens)")
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    holder: Dict[str, Any] = {}

    def backend_factory():
        if "backend" not in holder:
            from anyjev.backends.hf import HFBackend
            holder["backend"] = HFBackend(args.model, batch_size=args.batch_size)
        return holder["backend"]

    records = typed_records(args.calib_cases)
    if args.bench_tasks:
        records += task_records([t for t in args.bench_tasks.split(",") if t], args.n_test, args.n_calib,
                                args.seed, "bench")
    print(f"{len(records)} labelled pools; extracting/loading features", flush=True)
    for rec in records:
        name, q, group = rec["name"], rec["q"], rec["group"]
        s_tr, y_tr, soft_tr = rec["calib"]
        s_te, y_te, soft_te = rec["test"]
        for split, order, states, labels, soft in (("train", args.train_order, s_tr, y_tr, soft_tr),
                                                   ("test", "id", s_te, y_te, soft_te),
                                                   ("test", "rev", s_te, y_te, soft_te),
                                                   ("train", "id", s_tr, y_tr, soft_tr)):
            get_pool(backend_factory, args, layers, name, split, order, q, states, labels, group, soft)
    print("features cached under", os.path.join(args.out, "features", args.model.replace("/", "__")), flush=True)


if __name__ == "__main__":
    main()
