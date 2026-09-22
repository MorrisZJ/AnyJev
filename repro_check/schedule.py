"""Run the whole reproduction matrix across the available GPUs.

One subprocess per (model, task-list) so each model loads once and all of its
tasks share the loaded weights. Jobs are handed to whichever GPU frees up first.

    python repro_check/schedule.py --gpus 0,1,2,3 --wave repro
    python repro_check/schedule.py --gpus 0,1,2,3 --wave extra
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The repo's own three tasks, for the reproduction check.
REPO_TASKS = "banking20,newsgroups,injection"
# Everything the repo's bench never covers: K=4, K=6, a second/third noul,
# K=20 outside banking, a Chinese slice, and the untested `score` primitive.
NEW_TASKS = "agnews,emotion,massive20,massive20_zh,hate,subj,yelp5"

WAVES = {
    # reproduce the published Qwen rows
    "repro": [
        ("Qwen/Qwen3-8B", REPO_TASKS, "repro"),
        ("Qwen/Qwen2.5-7B-Instruct", REPO_TASKS, "repro"),
    ],
    # new model families on the repo's own tasks: does the readout transfer?
    "families": [
        ("mistralai/Mistral-7B-Instruct-v0.3", REPO_TASKS, "families"),
        ("microsoft/Phi-3.5-mini-instruct", REPO_TASKS, "families"),
        ("allenai/OLMo-2-1124-7B-Instruct", REPO_TASKS, "families"),
        ("tiiuae/Falcon3-7B-Instruct", REPO_TASKS, "families"),
    ],
    # new benchmarks, on one Qwen and three non-Qwen models
    "extra": [
        ("Qwen/Qwen3-8B", NEW_TASKS, "extra"),
        ("Qwen/Qwen2.5-7B-Instruct", NEW_TASKS, "extra"),
        ("mistralai/Mistral-7B-Instruct-v0.3", NEW_TASKS, "extra"),
        ("microsoft/Phi-3.5-mini-instruct", NEW_TASKS, "extra"),
        ("allenai/OLMo-2-1124-7B-Instruct", NEW_TASKS, "extra"),
        ("tiiuae/Falcon3-7B-Instruct", NEW_TASKS, "extra"),
    ],
}


def worker(gpu: int, jobs: queue.Queue, n_total: int, lock: threading.Lock):
    while True:
        try:
            idx, (model, tasks, tag) = jobs.get_nowait()
        except queue.Empty:
            return
        slug = model.replace("/", "__")
        log = os.path.join(HERE, "logs", f"{tag}.{slug}.log")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   HF_HOME="/mnt/persist/hf-cache", TOKENIZERS_PARALLELISM="false")
        cmd = ["python3", os.path.join(HERE, "run_repro.py"),
               "--model", model, "--tasks", tasks, "--n", "300", "--calib", "200",
               "--batch-size", "32", "--tag", tag]
        with lock:
            print(f"[gpu{gpu}] ({idx}/{n_total}) START {tag} {model}  -> {os.path.basename(log)}",
                  flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        with lock:
            print(f"[gpu{gpu}] ({idx}/{n_total}) {'DONE ' if rc == 0 else 'FAIL '} {tag} {model} "
                  f"rc={rc} in {time.time() - t0:.0f}s", flush=True)
        jobs.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--wave", default="repro", help="comma-separated: " + ",".join(WAVES))
    args = ap.parse_args()

    jobs: queue.Queue = queue.Queue()
    plan = [j for w in args.wave.split(",") for j in WAVES[w]]
    for i, j in enumerate(plan, start=1):
        jobs.put((i, j))
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"{len(plan)} jobs over gpus {gpus}", flush=True)

    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, jobs, len(plan), lock)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ALL JOBS FINISHED", flush=True)


if __name__ == "__main__":
    main()
