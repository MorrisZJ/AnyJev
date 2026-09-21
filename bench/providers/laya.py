"""Laya (NandhaKishorM/laya) as a bench provider for typed-decisions.

Runs `laya.load(checkpoint).predict(state, questions)` on the dataset's own
question dicts (type / instructions / criteria), so Laya sees exactly the
format it was built for. Requires `pip install laya`; not imported unless used.

    python -m bench.providers.laya --checkpoint convaiinnovations/laya-typed-decisions
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from bench.run_typed import pooled
from bench.tasks.typed_decisions import Decision, load


def run_laya(checkpoint: str, test: List[Decision], device: str = "cuda") -> Dict[str, Any]:
    import laya  # noqa: F401  (external package)
    from datasets import load_dataset

    agent = laya.load(checkpoint, device=device)
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    by_id = {row["id"]: row for row in ds}
    wanted = {(d.case_id, d.qname): d for d in test}
    rows = []
    t0 = time.time()
    for case_id in sorted({d.case_id for d in test}):
        row = by_id[case_id]
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        res = agent.predict(state, questions)
        answers = res.get("answers", res)
        for qname, spec in questions.items():
            d = wanted.get((case_id, qname))
            if d is None:
                continue
            a = answers[qname]
            probs_map = a.get("probabilities", {})
            if spec["type"] == "choice":
                keys = list(spec["criteria"].keys()) if isinstance(spec["criteria"], dict) else list(spec["criteria"])
                p = np.array([float(probs_map.get(k, 0.0)) for k in keys])
            elif spec["type"] == "noul":
                pt = float(a.get("noul", probs_map.get("true", 0.5)))
                p = np.array([pt, 1.0 - pt])
            else:
                k = len(spec["criteria"])
                p = np.array([float(probs_map.get(str(i), 0.0)) for i in range(k)])
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
            rows.append((d, p))
    return {"overall": pooled(rows), "seconds": time.time() - t0,
            "by_type": {t: pooled([r for r in rows if r[0].question.kind == t]) for t in ("choice", "noul", "score")}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="convaiinnovations/laya-typed-decisions")
    ap.add_argument("--limit-cases", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="bench/results_typed")
    args = ap.parse_args(argv)
    test = load("test", limit_cases=args.limit_cases)
    res = {"provider": "laya", "checkpoint": args.checkpoint, "limit_cases": args.limit_cases,
           "date": dt.datetime.now().isoformat(), **run_laya(args.checkpoint, test, args.device)}
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "laya__" + args.checkpoint.replace("/", "__") + ".json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["overall"], indent=1))


if __name__ == "__main__":
    main()
