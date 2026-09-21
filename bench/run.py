"""anyjev-bench: one command, one table, one JSON.

    python -m bench.run --model Qwen/Qwen3-8B --tasks newsgroups,injection --n 300 --calib 200

Columns per (model, task, level): acc, macro-F1, Brier, ECE, flip rate under
option reversal, coverage at 5% risk. raw and L0 come from the same forward
passes; L1 fits a temperature on the calibration split.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, List

import numpy as np

from anyjev import Decider, Question
from anyjev.question import Question as Q
from bench import metrics
from bench.tasks import get_task


def reversed_question(q: Question) -> Question:
    """Same question, options listed in reverse: the order-flip probe."""
    if q.kind == "noul":
        return q
    return Q(q.kind, q.text, tuple(reversed(q.options)), q.name, q.scale, q.ordered)


def _rows_from_diagnostics(decs, combine: str):
    """Ablation readouts in option space, all from the same forward passes.
    L0 is whatever the decider's configured prior produced (default: perm + batch prior)."""
    from anyjev.calibrate import apply_contextual, marginalize
    rows = {"raw": [], "L0-perm": [], "L0-perm+bc": [], "L0-perm+cf": [], "L0-cf": [], "L0-bc": [], "L0": []}
    for d in decs:
        g = d.diagnostics
        P, perms = g["p_pos_raw"], g["perms"]
        rows["raw"].append(marginalize(P[:1], perms[:1]))
        rows["L0-perm"].append(marginalize(P, perms, combine))
        if g.get("batch_prior") is not None:
            rows["L0-perm+bc"].append(marginalize(apply_contextual(P, g["batch_prior"]), perms, combine))
            rows["L0-bc"].append(marginalize(apply_contextual(P[:1], g["batch_prior"][:1]), perms[:1]))
        if g.get("cf_prior") is not None:
            rows["L0-cf"].append(marginalize(apply_contextual(P[:1], g["cf_prior"][:1]), perms[:1]))
            rows["L0-perm+cf"].append(marginalize(apply_contextual(P, g["cf_prior"]), perms, combine))
        rows["L0"].append(d.probs)
    return {k: np.stack(v) for k, v in rows.items() if v}


def _noul_other_phrasing(decs, rows) -> Dict[str, np.ndarray]:
    """Readouts under the other phrasing ("No or Yes"), option space. The
    permutation-marginalized rows already average both phrasings, so they
    are invariant by construction and reuse their own values."""
    from anyjev.calibrate import apply_contextual, marginalize
    other = dict(rows)

    def one(prior_key):
        out = []
        for d in decs:
            g = d.diagnostics
            P = g["p_pos_raw"][1:2]
            if prior_key is not None:
                P = apply_contextual(P, g[prior_key][1:2])
            out.append(marginalize(P, g["perms"][1:2]))
        return np.stack(out)

    other["raw"] = one(None)
    if "L0-bc" in rows:
        other["L0-bc"] = one("batch_prior")
    if "L0-cf" in rows:
        other["L0-cf"] = one("cf_prior")
    return other


def run_task(decider: Decider, task_name: str, n_test: int, n_calib: int, seed: int,
             levels: List[str]) -> Dict[str, Any]:
    task = get_task(task_name)
    test, calib = task.split(n_test, n_calib, seed)
    states = [s for s, _ in test]
    labels = [y for _, y in test]
    q = task.question
    out: Dict[str, Any] = {"task": task_name, "question": q.key, "n_test": len(test),
                           "n_calib": len(calib), "k": q.k, "levels": {}}

    t0 = time.time()
    decs = decider.decide_batch(states, q, level="L0")
    t_l0 = time.time() - t0
    answer_mass = float(np.mean([d.diagnostics["answer_mass"] for d in decs]))
    rows = _rows_from_diagnostics(decs, decider.combine)

    # order-flip probe: reversed option list for choice; the other phrasing for noul
    if q.kind == "noul":
        rows_r = _noul_other_phrasing(decs, rows)
    else:
        qr = reversed_question(q)
        decs_r = decider.decide_batch(states, qr, level="L0")
        rows_r = {k: v[:, ::-1] for k, v in _rows_from_diagnostics(decs_r, decider.combine).items()}

    for name, P in rows.items():
        if name == "raw" and "raw" not in levels:
            continue
        if name != "raw" and "L0" not in levels:
            continue
        Pr = rows_r.get(name)
        out["levels"][name] = metrics.summarize(P, labels, Pr)
    if "L0" in out["levels"]:
        out["levels"]["L0"].update({
            "cyclic_flip_raw": float(np.mean([d.diagnostics["order_flip_raw"] for d in decs])),
            "cyclic_flip_l0": float(np.mean([d.diagnostics["order_flip_l0"] for d in decs])),
            "prior_method": decs[0].diagnostics.get("prior_method")})

    if "L1" in levels and calib:
        art = decider.calibrate(q, [s for s, _ in calib], [y for _, y in calib])
        l1 = np.stack([d.probs for d in decider.decide_batch(states, q, level="L1")])
        if q.kind == "noul":
            l1_r = l1
        else:
            decider.load_artifact(qr, art)  # same temperature, reversed layout
            l1_r = np.stack([d.probs for d in decider.decide_batch(states, qr, level="L1")])[:, ::-1]
        out["levels"]["L1"] = {**metrics.summarize(l1, labels, l1_r), "temperature": art["temperature"]}
        out["artifact"] = art
    out["answer_mass"] = answer_mass
    out["seconds_l0_pass"] = t_l0
    out["seconds_per_decision_l0"] = t_l0 / max(1, len(test))
    return out


def markdown_table(results: Dict[str, Any]) -> str:
    cols = ["acc", "macro_f1", "brier", "ece", "flip", "cov@5%"]
    lines = ["| model | task | K | level | " + " | ".join(cols) + " |",
             "|---|---|---|---|" + "---|" * len(cols)]
    for r in results["tasks"]:
        for level, m in r["levels"].items():
            cells = [f"{m[c]:.3f}" if c in m else "" for c in cols]
            lines.append(f"| {results['model']} | {r['task']} | {r['k']} | {level} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {"python": platform.python_version(), "date": dt.datetime.now().isoformat()}
    try:
        import torch  # noqa: E401
        import transformers
        env["torch"] = torch.__version__
        env["transformers"] = transformers.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return env


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="hf", choices=["hf"])
    ap.add_argument("--tasks", default="newsgroups,injection")
    ap.add_argument("--levels", default="raw,L0,L1")
    ap.add_argument("--n", type=int, default=300, help="test items per task")
    ap.add_argument("--calib", type=int, default=200, help="calibration items per task (L1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-permutations", type=int, default=None)
    ap.add_argument("--combine", default="logmean", choices=["logmean", "mean"])
    ap.add_argument("--prior", default="batch", choices=["batch", "content_free", "none"])
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args(argv)

    from anyjev.backends.hf import HFBackend
    backend = HFBackend(args.model, batch_size=args.batch_size)
    decider = Decider(backend, max_permutations=args.max_permutations, combine=args.combine,
                      prior=args.prior, record_content_free=True)
    levels = args.levels.split(",")

    results: Dict[str, Any] = {"model": args.model, "backend": args.backend, "seed": args.seed,
                               "n": args.n, "calib": args.calib,
                               "max_permutations": args.max_permutations, "combine": args.combine, "prior": args.prior,
                               "env": environment(), "tasks": []}
    stamp = dt.datetime.now().strftime("%Y-%m-%d")
    outdir = os.path.join(args.out, stamp)
    os.makedirs(outdir, exist_ok=True)
    slug = args.model.replace("/", "__")

    json_path = os.path.join(outdir, f"{slug}.json")

    def save():
        # merge by task at write time, so concurrent runs of the same model
        # on different tasks build one file; atomic rename avoids torn reads
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
        save()   # a crash on a later task keeps what is done
    print(markdown_table(results))


if __name__ == "__main__":
    main()
