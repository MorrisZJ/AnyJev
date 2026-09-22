"""Run the full bench (three tasks + typed-decisions) for several models, one GPU each, in parallel.

    python -m bench.sweep --models Qwen/Qwen3-4B,microsoft/Phi-4-mini-instruct --gpus 0,1 --out-tag small

Each model runs `bench.run` (newsgroups, injection, banking20) then `bench.run_typed` on its
assigned GPU; models are queued round-robin over the GPUs. Logs go to logs/sweep/<tag>/<model>.log,
results to bench/results_<tag>/ and bench/results_typed_<tag>/. Prints a summary at the end.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out-tag", default="small")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-typed", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dump-items", action="store_true", help="forward --dump-items to both runners")
    args = ap.parse_args(argv)

    models = [m for m in args.models.split(",") if m]
    gpus = args.gpus.split(",")
    queues = defaultdict(list)
    for i, m in enumerate(models):
        queues[gpus[i % len(gpus)]].append(m)
    logdir = os.path.join("logs", "sweep", args.out_tag)
    os.makedirs(logdir, exist_ok=True)

    def cmds(m):
        c = []
        if not args.skip_bench:
            c.append([args.python, "-m", "bench.run", "--model", m, "--tasks", "newsgroups,injection,banking20",
                      "--n", str(args.n), "--calib", str(args.calib), "--batch-size", str(args.batch_size),
                      "--out", f"bench/results_{args.out_tag}"] + (["--dump-items"] if args.dump_items else []))
        if not args.skip_typed:
            c.append([args.python, "-m", "bench.run_typed", "--model", m, "--batch-size", str(args.batch_size),
                      "--out", f"bench/results_typed_{args.out_tag}"] + (["--dump-items"] if args.dump_items else []))
        return c

    procs = []
    for gpu, ms in queues.items():
        # one shell chain per GPU so its models run sequentially
        chain = " ; ".join(" ".join(f"'{x}'" for x in cmd) + f" >> '{logdir}/{m.replace('/', '__')}.log' 2>&1"
                           for m in ms for cmd in cmds(m))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        procs.append((gpu, ms, subprocess.Popen(["bash", "-c", chain], env=env)))
        print(f"GPU {gpu}: {ms}", flush=True)
    t0 = time.time()
    for gpu, ms, p in procs:
        p.wait()
        print(f"GPU {gpu} done ({time.time() - t0:.0f}s)", flush=True)
    print("sweep finished; summarize with: python -m bench.models_table", f"bench/results_{args.out_tag}",
          f"bench/results_typed_{args.out_tag}")


if __name__ == "__main__":
    main()
