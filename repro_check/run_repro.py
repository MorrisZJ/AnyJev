"""Reproduction driver: registers the extra tasks, then drives bench.run.

Everything the repo's own bench does, plus the tasks in repro_check/extra_tasks,
plus per-item probability dumps so the fragile metrics can be bootstrapped
afterwards instead of taken as point estimates.

    python repro_check/run_repro.py --model Qwen/Qwen3-8B --tasks banking20 --n 300 --calib 200
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import repro_check.extra_tasks  # noqa: F401,E402  (registers the new tasks)
from anyjev import Decider  # noqa: E402
from anyjev.question import Question  # noqa: E402
from bench import metrics  # noqa: E402
from bench.run import _noul_other_phrasing, _rows_from_diagnostics, environment  # noqa: E402
from bench.tasks import get_task  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def reversed_question(q: Question) -> Question:
    """bench.run.reversed_question drops `centers`, which silently rescales
    bin_centers() for a levels-based score question ([0,1,2,3,4] becomes
    [0.4,1.2,2.0,2.8,3.6]) and corrupts any expected-value readout. Unreachable
    in the repo's own bench because it has no score task; live as soon as one is
    added. Preserve centers, reversed alongside the options."""
    if q.kind == "noul":
        return q
    centers = tuple(reversed(q.centers)) if q.centers is not None else None
    return Question(q.kind, q.text, tuple(reversed(q.options)), q.name, q.scale, q.ordered, centers)


def run_task(decider, task_name, n_test, n_calib, seed, dump_dir=None):
    task = get_task(task_name)
    test, calib = task.split(n_test, n_calib, seed)
    states = [s for s, _ in test]
    labels = [y for _, y in test]
    q = task.question

    t0 = time.time()
    decs = decider.decide_batch(states, q, level="L0")
    t_l0 = time.time() - t0
    rows = _rows_from_diagnostics(decs, decider.combine)

    if q.kind == "noul":
        rows_r = _noul_other_phrasing(decs, rows)
        qr = None
    else:
        qr = reversed_question(q)
        decs_r = decider.decide_batch(states, qr, level="L0")
        rows_r = {k: v[:, ::-1] for k, v in _rows_from_diagnostics(decs_r, decider.combine).items()}

    out = {"task": task_name, "question": q.key, "kind": q.kind, "k": q.k,
           "n_test": len(test), "n_calib": len(calib), "levels": {}}
    for name, P in rows.items():
        out["levels"][name] = metrics.summarize(P, labels, rows_r.get(name))

    art = None
    if calib:
        art = decider.calibrate(q, [s for s, _ in calib], [y for _, y in calib])
        l1 = np.stack([d.probs for d in decider.decide_batch(states, q, level="L1")])
        if qr is None:
            l1_r = l1
        else:
            decider.load_artifact(qr, art)
            l1_r = np.stack([d.probs for d in decider.decide_batch(states, qr, level="L1")])[:, ::-1]
        out["levels"]["L1"] = {**metrics.summarize(l1, labels, l1_r), "temperature": art["temperature"]}
        rows["L1"], rows_r["L1"] = l1, l1_r

    out["temperature"] = art["temperature"] if art else None
    out["seconds_l0_pass"] = t_l0
    out["seconds_per_decision_l0"] = t_l0 / max(1, len(test))
    out["answer_mass"] = float(np.mean([d.diagnostics["answer_mass"] for d in decs]))
    out["cyclic_flip_raw"] = float(np.mean([d.diagnostics["order_flip_raw"] for d in decs]))
    out["cyclic_flip_l0"] = float(np.mean([d.diagnostics["order_flip_l0"] for d in decs]))
    out["permutations"] = int(decs[0].diagnostics["permutations"])

    # per-item dump: lets us bootstrap cov@5% instead of trusting one number
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(dump_dir, f"{task_name}.npz"),
            labels=np.asarray(labels),
            **{f"P__{k}": v for k, v in rows.items()},
            **{f"R__{k}": v for k, v in rows_r.items()},
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="banking20")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prior", default="batch", choices=["batch", "content_free", "none"])
    ap.add_argument("--combine", default="logmean", choices=["logmean", "mean"])
    ap.add_argument("--max-permutations", type=int, default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args(argv)

    from anyjev.backends.hf import HFBackend

    slug = args.model.replace("/", "__")
    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"{slug}.json")

    t_load = time.time()
    backend = HFBackend(args.model, batch_size=args.batch_size)
    t_load = time.time() - t_load
    decider = Decider(backend, prior=args.prior, combine=args.combine,
                      max_permutations=args.max_permutations, record_content_free=True)

    results = {"model": args.model, "seed": args.seed, "n": args.n, "calib": args.calib,
               "prior": args.prior, "combine": args.combine,
               "max_permutations": args.max_permutations, "batch_size": args.batch_size,
               "seconds_model_load": t_load, "env": environment(),
               "date": dt.datetime.now().isoformat(), "tasks": []}

    for t in args.tasks.split(","):
        print(f"== {args.model} / {t}", flush=True)
        try:
            r = run_task(decider, t, args.n, args.calib, args.seed,
                         dump_dir=os.path.join(outdir, f"{slug}.items"))
        except Exception as e:
            print(f"!! {t} FAILED: {type(e).__name__}: {e}", flush=True)
            results["tasks"].append({"task": t, "error": f"{type(e).__name__}: {e}"})
        else:
            results["tasks"].append(r)
            print(json.dumps({k: {m: round(v[m], 3) for m in ("acc", "ece", "flip", "cov@5%") if m in v}
                              for k, v in r["levels"].items()}, indent=1), flush=True)
        tmp = json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1, default=float)
        os.replace(tmp, json_path)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
