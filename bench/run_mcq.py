"""anyjev-bench for MCQ sets where every item carries its own question.

    python -m bench.run_mcq --model Qwen/Qwen3-VL-2B-Instruct --backend vlm --tasks ai2d --n 300 --calib 200

Same columns and the same ablation rows as `bench.run`, with two differences
that follow from the shape of the data rather than from a change of method.

**No batch prior, and no content-free prior by default.** The batch prior --
the library default, and the one the text bench reports -- is estimated per
question over a batch of states, and here each item *is* a question with one
state, so it never reaches `min_prior_n` and `--prior batch` gives exactly the
permutation-only L0 that `--prior none` states outright. The content-free
prior is available but off, as it is on text: on a
per-item MCQ the probe keeps the question and its options and blanks only the
picture, and when the text alone carries the answer (AI2D: "which of these is
a producer?") the "prior" it measures is the model's honest knowledge, and
dividing it out cost Qwen3-VL-2B 24 accuracy points. So L0 here is
permutation marginalization only; the cf rows stay in the table as ablations.

**One temperature for the task, not per question.** L1 fits a single
temperature on the pooled calibration items. Per-question artifacts are
meaningless when a question is seen once, and the pooled artifact is recorded
as such.

Every item is its own `decide_batch` call, so the backend sees K prompts at a
time (plus probes) rather than a full batch. That is correct but underfills
the GPU; batching across (state, question) pairs needs a decider that takes
pairs instead of a state x question cross product, which is a follow-up.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from anyjev import Decider
from anyjev.calibrate.posthoc import TemperatureScaler
from bench import metrics
from bench.run import _rows_from_diagnostics, environment, markdown_table, reversed_question
from bench.tasks import get_mcq_task


def _readouts(decider: Decider, items: List[tuple], combine: str) -> Tuple[Dict[str, np.ndarray], ...]:
    """Every ablation readout for one item list, in option space, from the
    same forward passes. Returns (rows, reversed_rows); the reversed-option
    probe is a second pass, mapped back to the original option order."""
    rows: Dict[str, List[np.ndarray]] = {}
    rows_r: Dict[str, List[np.ndarray]] = {}
    for state, q, _ in items:
        dec = decider.decide_batch([state], q, level="L0")
        for name, P in _rows_from_diagnostics(dec, combine).items():
            rows.setdefault(name, []).append(P[0])
        dec_r = decider.decide_batch([state], reversed_question(q), level="L0")
        for name, P in _rows_from_diagnostics(dec_r, combine).items():
            rows_r.setdefault(name, []).append(P[0][::-1])
    return ({k: np.stack(v) for k, v in rows.items()},
            {k: np.stack(v) for k, v in rows_r.items()})


def run_task(decider: Decider, task_name: str, n_test: int, n_calib: int, seed: int,
             levels: List[str]) -> Dict[str, Any]:
    task = get_mcq_task(task_name)
    test, calib = task.split(n_test, n_calib, seed)
    labels = [y for _, _, y in test]
    out: Dict[str, Any] = {"task": task_name, "n_test": len(test), "n_calib": len(calib),
                           "k": task.k, "prior": decider.prior, "license": task.license,
                           "notes": task.notes, "levels": {}}

    t0 = time.time()
    rows, rows_r = _readouts(decider, test, decider.combine)
    seconds = time.time() - t0

    for name, P in rows.items():
        if name == "raw" and "raw" not in levels:
            continue
        if name != "raw" and "L0" not in levels:
            continue
        out["levels"][name] = metrics.summarize(P, labels, rows_r.get(name))

    if "L1" in levels and calib:
        cal_rows, _ = _readouts(decider, calib, decider.combine)
        scaler = TemperatureScaler.fit(cal_rows["L0"], [y for _, _, y in calib])
        l1 = scaler.apply(rows["L0"])
        l1_r = scaler.apply(rows_r["L0"]) if "L0" in rows_r else None
        out["levels"]["L1"] = {**metrics.summarize(l1, labels, l1_r), "temperature": scaler.temperature}
        out["artifact"] = {"model": decider.backend.name, "scope": f"task:{task_name}",
                           **scaler.to_dict()}

    out["seconds_l0_pass"] = seconds
    out["seconds_per_decision_l0"] = seconds / max(1, len(test))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name recorded in the results")
    ap.add_argument("--model-path", default=None,
                    help="where to load it from, if not the name (a local checkpoint)")
    ap.add_argument("--backend", default="vlm", choices=["hf", "vlm"])
    ap.add_argument("--tasks", default="ai2d")
    ap.add_argument("--levels", default="raw,L0,L1")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--max-permutations", type=int, default=None)
    ap.add_argument("--combine", default="logmean", choices=["logmean", "mean"])
    ap.add_argument("--prior", default="none", choices=["batch", "content_free", "none"])
    ap.add_argument("--out", default="bench/results_mcq")
    args = ap.parse_args(argv)

    source = args.model_path or args.model
    if args.backend == "vlm":
        from anyjev.backends.hf_vlm import VLMBackend
        backend = VLMBackend(source, batch_size=args.batch_size, max_pixels=args.max_pixels)
    else:
        from anyjev.backends.hf import HFBackend
        backend = HFBackend(source, batch_size=args.batch_size)
    backend.name = args.model            # artifacts and results carry the name, not the path
    if args.prior == "batch":
        print("warning: the batch prior needs many states per question; this set has one. "
              "It will report prior_method=none.", flush=True)
    decider = Decider(backend, max_permutations=args.max_permutations, combine=args.combine,
                      prior=args.prior, record_content_free=True)
    levels = args.levels.split(",")

    results: Dict[str, Any] = {"model": args.model, "backend": args.backend, "seed": args.seed,
                               "n": args.n, "calib": args.calib, "batch_size": args.batch_size,
                               "max_pixels": args.max_pixels, "combine": args.combine,
                               "max_permutations": args.max_permutations, "prior": args.prior,
                               "env": environment(), "tasks": []}
    stamp = dt.datetime.now().strftime("%Y-%m-%d")
    outdir = os.path.join(args.out, stamp)
    os.makedirs(outdir, exist_ok=True)
    slug = args.model.replace("/", "__")
    json_path = os.path.join(outdir, f"{slug}.json")

    def save():
        merged = dict(results)
        mine = {t["task"]: t for t in results["tasks"]}
        if os.path.exists(json_path):
            with open(json_path) as f:
                prev = json.load(f)
            merged["tasks"] = [t for t in prev.get("tasks", []) if t["task"] not in mine] + list(mine.values())
        tmp = json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=1, default=float)
        os.replace(tmp, json_path)
        with open(os.path.join(outdir, f"{slug}.md"), "w") as f:
            f.write(markdown_table(merged) + "\n")

    for t in args.tasks.split(","):
        print(f"== {t}", flush=True)
        r = run_task(decider, t, args.n, args.calib, args.seed, levels)
        results["tasks"].append(r)
        print(json.dumps(r["levels"], indent=1), flush=True)
        save()
    print(markdown_table(results))


if __name__ == "__main__":
    main()
