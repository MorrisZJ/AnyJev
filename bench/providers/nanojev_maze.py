"""AnyJev inside NanoJev's maze harness (TianyuCodings/NanoJev, MIT).

NanoJev's "Untuned Qwen" row reads A/B next-token logits from Qwen3-0.6B for
four Boolean questions per maze cell ("is one step north clear?"). This runs
the same frozen exploration code (`evaluate_model_edges_maze.run_exploration`)
with AnyJev as the engine, so the only thing that changes is the readout.

    python -m bench.providers.nanojev_maze --nanojev /path/to/NanoJev \\
        --episodes .../scaled_games_v4b/episodes.jsonl --model Qwen/Qwen3-0.6B --level L0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import Any, Dict

from anyjev import Decider, Question


class AnyJevBooleanEngine:
    """engine.predict(payload) in NanoJev's contract, backed by an AnyJev Decider."""

    def __init__(self, decider: Decider, level: str):
        self.decider = decider
        self.level = level
        self.calls = 0
        self.prompts = 0

    @staticmethod
    def _question(qid: str, spec: Dict[str, Any]) -> Question:
        crit = spec.get("criteria") or {}
        text = f"Is the following statement true? {spec['instructions']}"
        if crit:
            text += f"\nYes means: {crit.get('true', '')}\nNo means: {crit.get('false', '')}"
        return Question.noul(text, name=qid)

    def predict(self, payload, batch_questions=0, temperature=1.0):
        states = payload["states"]
        outputs = {row["id"]: {"id": row["id"], "answers": {}} for row in states}
        qids = list(states[0]["questions"].keys())
        for qid in qids:
            q = self._question(qid, states[0]["questions"][qid])
            decs = self.decider.decide_batch([row["state"] for row in states], q, level=self.level)
            for row, dec in zip(states, decs):
                p = float(dec.probs[0])
                p = min(max(p, 0.0), 1.0)
                outputs[row["id"]]["answers"][qid] = {
                    "type": "boolean", "probabilities": {"true": p, "false": 1.0 - p}, "p_true": p,
                    "backend": f"anyjev_{self.level}", "target_kind": f"anyjev_{self.level}_noul",
                    "normalization_applied": dec.level,
                }
        self.calls += 1
        return {"states": list(outputs.values()),
                "execution": {"forward_passes": None, "network_model_calls": 0,
                              "autoregressive_decode_steps": 0, "backend": f"anyjev_{self.level}",
                              "active_state_batch_size": len(states)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanojev", required=True, help="path to the NanoJev checkout")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--level", default="L0", choices=["raw", "L0"])
    ap.add_argument("--prior", default="batch", choices=["batch", "content_free", "none"])
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-states", type=int, default=8)
    ap.add_argument("--out", default="bench/results_nanojev")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(args.nanojev, "scripts"))
    import evaluate_model_edges_maze as core  # noqa: E402

    splits = {s.strip() for s in args.splits.split(",")}
    episodes = [json.loads(line) for line in open(args.episodes) if line.strip()]
    selected = [e for e in episodes if e["game"] == "scaled_maze" and e["split"] in splits]
    print(f"{len(selected)} maze episodes, sizes {sorted({e['size'] for e in selected})}", flush=True)

    from anyjev.backends.hf import HFBackend
    backend = HFBackend(args.model, batch_size=32, revision=args.revision)
    decider = Decider(backend, prior=args.prior)
    engine = AnyJevBooleanEngine(decider, args.level)
    t0 = time.time()
    result = core.run_exploration(selected, engine, args.window_size, args.max_steps, args.batch_states, 0)
    result.update(model=args.model, revision=args.revision, level=args.level, prior=args.prior, engine="anyjev",
                  elapsed_seconds=time.time() - t0, date=dt.datetime.now().isoformat())
    for ep in result["episodes"]:
        ep.pop("steps", None)   # keep the artifact small; observations keep the per-cell probabilities
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    slug = f"{args.model.replace('/', '__')}.{args.level}.{args.prior}"
    with open(os.path.join(outdir, slug + ".json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    per = [(e["id"], e["status"], e["attempts"], e["collisions"]) for e in result["episodes"]]
    print(json.dumps({"summary": result["summary"], "episodes": per}, indent=1, default=float))


if __name__ == "__main__":
    main()
