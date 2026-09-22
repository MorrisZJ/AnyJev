"""Unattended verification of every experiment this repo published.

Runs the repo's OWN entry points (`bench.run`, `bench.run_typed`) with the
exact parameters recorded in the committed result JSON, so the output is
directly diffable against what the README's tables are built from. Nothing in
anyjev/ or bench/ is modified.

Phases, in order:
  P0  fetch the three repo models not yet cached
  P1  bench.run   x {Qwen3-8B, Qwen2.5-7B-Instruct, Qwen3-30B-A3B-Instruct-2507}
                  x {prior=batch, prior=content_free}   -> vs results_batchprior_v0 / results_cf
  P2  bench.run_typed x {Qwen3-8B, Qwen2.5-7B-Instruct, Qwen3-32B}  -> vs results_typed
  P3  per-item dumps (my wrapper) for the bootstrap analysis, plus the
      new tasks and the non-Qwen families, if the GPUs are still free
  P4  diff everything and write repro_check/REPORT.md

Launch:  nohup python3 repro_check/verify_all.py > repro_check/logs/verify_all.log 2>&1 &
Resume:  the same command; finished jobs are detected and skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(HERE, "logs")
RESULTS = os.path.join(HERE, "results")

REPO_TASKS = "banking20,newsgroups,injection"
NEW_TASKS = "agnews,emotion,massive20,massive20_zh,hate,subj,yelp5"

# The exact parameters recorded in the committed runs.
BENCH_N, BENCH_CALIB, BENCH_SEED = "300", "200", "0"

ENV = dict(os.environ, HF_HOME="/mnt/persist/hf-cache", TOKENIZERS_PARALLELISM="false",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

DOWNLOADS = ["Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-32B", "Qwen/Qwen3-0.6B"]

# (name, argv, output-json-path-for-skip-detection, expected-tasks-or-None)
# ordered longest-first so the 30B and 32B start while the 7/8B jobs churn
JOBS: list[tuple] = []


def bench_job(model: str, prior: str, batch_size: str):
    slug = model.replace("/", "__")
    outdir = os.path.join(RESULTS, f"bench_{prior}")
    return (
        f"P1.bench.{slug}.{prior}",
        ["python3", "-m", "bench.run", "--model", model, "--tasks", REPO_TASKS,
         "--n", BENCH_N, "--calib", BENCH_CALIB, "--seed", BENCH_SEED,
         "--prior", prior, "--combine", "logmean", "--batch-size", batch_size,
         "--out", outdir],
        outdir, REPO_TASKS.split(","),
    )


def typed_job(model: str, batch_size: str):
    slug = model.replace("/", "__")
    outdir = os.path.join(RESULTS, "typed")
    return (
        f"P2.typed.{slug}",
        ["python3", "-m", "bench.run_typed", "--model", model, "--levels", "raw,L0,L1",
         "--calib-cases", "50", "--prior", "batch", "--batch-size", batch_size,
         "--out", outdir],
        outdir, None,
    )


def wrapper_job(model: str, tasks: str, tag: str, batch_size: str):
    slug = model.replace("/", "__")
    outdir = os.path.join(RESULTS, tag)
    return (
        f"P3.{tag}.{slug}",
        ["python3", os.path.join(HERE, "run_repro.py"), "--model", model, "--tasks", tasks,
         "--n", BENCH_N, "--calib", BENCH_CALIB, "--seed", BENCH_SEED,
         "--batch-size", batch_size, "--tag", tag],
        os.path.join(outdir, f"{slug}.json"), tasks.split(","),
    )


def build_jobs(phases: set[str]):
    jobs = []
    if "1" in phases:
        # 30B first: it is the long pole, and its two priors go to two GPUs
        for prior in ("batch", "content_free"):
            jobs.append(bench_job("Qwen/Qwen3-30B-A3B-Instruct-2507", prior, "8"))
        for model in ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"):
            for prior in ("batch", "content_free"):
                jobs.append(bench_job(model, prior, "32"))
    if "2" in phases:
        jobs.append(typed_job("Qwen/Qwen3-32B", "8"))
        jobs.append(typed_job("Qwen/Qwen3-8B", "32"))
        jobs.append(typed_job("Qwen/Qwen2.5-7B-Instruct", "32"))
    if "3" in phases:
        # per-item dumps on the repo's own tasks, for the bootstrap
        for model in ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"):
            jobs.append(wrapper_job(model, REPO_TASKS, "items", "32"))
        # the gaps in the repo's coverage: K=4, K=6, Chinese, score, extra nouls
        for model in ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"):
            jobs.append(wrapper_job(model, NEW_TASKS, "extra", "32"))
        # non-Qwen families on the repo's own tasks
        for model in ("mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3.5-mini-instruct",
                      "allenai/OLMo-2-1124-7B-Instruct", "tiiuae/Falcon3-7B-Instruct"):
            jobs.append(wrapper_job(model, REPO_TASKS, "families", "32"))
    return jobs


def already_done(out_path: str, want_tasks, model: str) -> bool:
    """bench.run writes <out>/<date>/<slug>.json; my wrapper writes <out> directly.
    Must match on the model too: a directory holds one file per model, and an
    earlier model finishing its tasks says nothing about this one."""
    paths = []
    if out_path.endswith(".json"):
        paths = [out_path]
    else:
        for d in sorted(os.listdir(out_path)) if os.path.isdir(out_path) else []:
            sub = os.path.join(out_path, d)
            if os.path.isdir(sub):
                paths += [os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".json")]
    for p in paths:
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("model") != model:
            continue
        if want_tasks is None:
            if r.get("levels"):
                return True
        else:
            have = {t["task"] for t in r.get("tasks", []) if "error" not in t}
            if set(want_tasks) <= have:
                return True
    return False


def download_phase():
    print("== P0 downloads", flush=True)
    for m in DOWNLOADS:
        t0 = time.time()
        rc = subprocess.call(
            ["python3", "-c",
             "import os,sys;os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1';"
             "from huggingface_hub import snapshot_download;"
             "snapshot_download(sys.argv[1], ignore_patterns=['*.pth','*.gguf','original/*','*.onnx','*consolidated*'], max_workers=16)",
             m], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"   {m} rc={rc} in {time.time() - t0:.0f}s", flush=True)


def worker(gpu: int, jobs: queue.Queue, lock: threading.Lock, state: dict):
    while True:
        try:
            name, argv, out_path, want = jobs.get_nowait()
        except queue.Empty:
            return
        model = argv[argv.index("--model") + 1]
        os.makedirs(out_path if not out_path.endswith(".json") else os.path.dirname(out_path),
                    exist_ok=True)
        if already_done(out_path, want, model):
            with lock:
                print(f"[gpu{gpu}] SKIP  {name} (output already present)", flush=True)
            continue
        log = os.path.join(LOGS, f"{name}.log")
        with lock:
            print(f"[gpu{gpu}] START {name}", flush=True)
        t0 = time.time()
        env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
        with open(log, "w") as f:
            rc = subprocess.call(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        with lock:
            state[name] = {"rc": rc, "seconds": dt, "gpu": gpu, "log": log}
            print(f"[gpu{gpu}] {'DONE ' if rc == 0 else 'FAIL '} {name} rc={rc} in {dt / 60:.1f} min",
                  flush=True)
            with open(os.path.join(HERE, "results", "job_state.json"), "w") as f:
                json.dump(state, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--phases", default="0,1,2,3,4")
    args = ap.parse_args()
    phases = set(args.phases.split(","))
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    t_start = time.time()
    if "0" in phases:
        download_phase()

    jobs_list = build_jobs(phases)
    if jobs_list:
        jobs: queue.Queue = queue.Queue()
        for j in jobs_list:
            jobs.put(j)
        gpus = [int(g) for g in args.gpus.split(",")]
        print(f"== {len(jobs_list)} jobs over gpus {gpus}", flush=True)
        lock, state = threading.Lock(), {}
        threads = [threading.Thread(target=worker, args=(g, jobs, lock, state)) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    if "4" in phases:
        print("== P4 diffing and reporting", flush=True)
        for script in ("check_offline.py", "compare.py", "compare_typed.py", "compare_maze.py",
                       "analyze.py"):
            p = os.path.join(HERE, script)
            if not os.path.exists(p):
                continue
            out = os.path.join(LOGS, script.replace(".py", ".out"))
            with open(out, "w") as f:
                subprocess.call(["python3", p], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
            print(f"   wrote {out}", flush=True)
        rp = os.path.join(HERE, "make_report.py")
        if os.path.exists(rp):
            subprocess.call(["python3", rp], cwd=ROOT, env=ENV)

    print(f"== ALL PHASES FINISHED in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
