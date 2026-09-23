import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend
from bench.games.maze import DIRS, Maze

WHERE = {"north": "directly above @, in row -1", "east": "directly right of @, in row +0",
         "south": "directly below @, in row +1", "west": "directly left of @, in row +0"}
QS = {d: Question.noul(f"Is the cell one step {d} of @ ({WHERE[d]}) open floor?", name=d) for d in DIRS}

rng = random.Random(0)
states, truth = [], []
for seed in range(1000, 1010):
    m = Maze.generate(51, seed)
    for c in rng.sample(m.open_cells(), 40):
        states.append(m.view(c))
        truth.append(m.truth(c))
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
be = HFBackend(model, batch_size=64)
for prior in ["batch", "content_free"]:
    d = Decider(be, prior=prior)
    for level in ["raw", "L0"]:
        accs, mp = {}, {}
        for dname, q in QS.items():
            decs = d.decide_batch(states, q, level=level)
            p = np.array([x.p_true for x in decs])
            y = np.array([t[dname] for t in truth])
            accs[dname] = round(float(((p >= .5) == y).mean()), 3)
            # AUC
            pos, neg = p[y], p[~y]
            auc = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
            mp[dname] = (round(float(p.mean()), 2), round(auc, 3), round(float(y.mean()), 2))
        print(prior, level, accs, "(mean p, AUC, base rate)", mp, flush=True)
