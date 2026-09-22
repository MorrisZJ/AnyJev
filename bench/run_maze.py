"""Play the grid maze in `bench.games.maze` with raw, L0 and L1 readouts of one model.

Same maze, same explorer, same four questions; only the readout changes. Each
(maze, level) gets a fresh Decider. L0 and L1 first see the same unlabeled
local views from separate calibration mazes (this warms L0's batch prior);
L1 also fits one temperature per direction on their labels. The test mazes
never enter calibration. Every step is saved so the GIF replays real runs.

    python -m bench.run_maze --model Qwen/Qwen3-8B --seeds 0,1,2,3,4 --size 51
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time
from typing import Dict, List

import numpy as np

from anyjev import Decider, Question
from bench.games.maze import DIRS, Explorer, Maze
from bench.run import environment

WHERE = {"north": "directly above @, in row -1", "east": "directly right of @, in row +0",
         "south": "directly below @, in row +1", "west": "directly left of @, in row +0"}
QUESTIONS = {d: Question.noul(f"Is the cell one step {d} of @ ({WHERE[d]}) open floor?", name=d)
             for d in DIRS}
CALIB_SEED0 = 10_000


def calibration_set(size: int, n: int, seed: int = CALIB_SEED0):
    """n local views from mazes seeded CALIB_SEED0.. (disjoint from test seeds), with truth."""
    rng = random.Random(seed)
    views, truth, s = [], [], seed
    while len(views) < n:
        m = Maze.generate(size, s)
        for c in rng.sample(m.open_cells(), min(40, n - len(views))):
            views.append(m.view(c))
            truth.append(m.truth(c))
        s += 1
    return views, truth


def prepare(backend, level: str, views, truth, prior: str) -> Decider:
    d = Decider(backend, prior=prior)
    if level == "L0":
        for q in QUESTIONS.values():
            d.decide_batch(views, q, level="L0")
    elif level == "L1":
        for name, q in QUESTIONS.items():
            d.calibrate(q, views, [0 if t[name] else 1 for t in truth])
    return d


def play(decider: Decider, level: str, maze: Maze, policy: str = "expected_cost", wall_cost: int = 10,
         threshold: float = 0.5) -> Dict:
    ex = Explorer(maze.size, maze.start, maze.goal, threshold=threshold, policy=policy, wall_cost=wall_cost)
    preds: List[Dict] = []
    t0 = time.time()
    while not (status := ex.done(maze)):
        if ex.needs_prediction():
            view = maze.view(ex.pos)
            ds = decider.decide(view, list(QUESTIONS.values()), level=level)
            assert ds.level == level
            p = {d: ds[d].p_true for d in DIRS}
            ex.remember(p)
            preds.append({"step": ex.steps, "pos": list(ex.pos), "p_open": p, "truth": maze.truth(ex.pos)})
        ex.step(maze)
    p = np.array([[r["p_open"][d] for d in DIRS] for r in preds])
    y = np.array([[r["truth"][d] for d in DIRS] for r in preds], dtype=float)
    return {"level": level, "status": status, "steps": ex.steps, "collisions": ex.collisions,
            "cells_seen": len(preds), "elapsed_seconds": time.time() - t0,
            "edge_accuracy": float(((p >= 0.5) == (y > 0.5)).mean()),
            "edge_brier": float(((p - y) ** 2).mean()),
            "mean_p_open": float(p.mean()), "open_rate": float(y.mean()),
            "predictions": preds, "moves": ex.log}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--size", type=int, default=51)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--levels", default="raw,L0,L1")
    ap.add_argument("--prior", default="batch", choices=["batch", "content_free", "none"])
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--policy", default="expected_cost", choices=["expected_cost", "threshold"])
    ap.add_argument("--wall-cost", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="bench/results_games")
    args = ap.parse_args(argv)

    from anyjev.backends.hf import HFBackend
    backend = HFBackend(args.model, batch_size=args.batch_size, revision=args.revision)
    views, truth = calibration_set(args.size, args.calib)
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"maze.{args.model.replace('/', '__')}.json")
    result = {"game": "maze", "model": args.model, "revision": args.revision, "size": args.size,
              "prior": args.prior, "policy": args.policy, "wall_cost": args.wall_cost,
              "threshold": args.threshold, "calib_n": args.calib,
              "calib_seed0": CALIB_SEED0, "questions": {d: q.text for d, q in QUESTIONS.items()},
              "environment": environment(), "episodes": []}
    for seed in [int(s) for s in args.seeds.split(",")]:
        maze = Maze.generate(args.size, seed)
        ep = {"seed": seed, "walls": ["".join("#" if w else "." for w in row) for row in maze.walls],
              "start": list(maze.start), "goal": list(maze.goal), "runs": {}}
        for level in args.levels.split(","):
            d = prepare(backend, level, views, truth, args.prior)
            run = play(d, level, maze, args.policy, args.wall_cost, args.threshold)
            if level == "L1":
                arts = d.export_artifacts()["artifacts"]
                run["temperatures"] = {name: arts[q.key]["temperature"] for name, q in QUESTIONS.items()}
            ep["runs"][level] = run
            print(f"seed {seed} {level:>3}: {run['status']:>7} steps {run['steps']:>5} "
                  f"collisions {run['collisions']:>5} edge acc {run['edge_accuracy']:.3f} "
                  f"brier {run['edge_brier']:.3f} ({run['elapsed_seconds']:.0f}s)", flush=True)
        result["episodes"].append(ep)
        with open(path, "w") as f:
            json.dump(result, f, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
