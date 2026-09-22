import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from anyjev.backends.hf import HFBackend
from bench.games.maze import Maze, DIRS
from bench.run_maze import QUESTIONS, calibration_set, prepare

model = sys.argv[1]
seeds = [20000, 20001, 20002, 20003, 20004]
be = HFBackend(model, batch_size=64)
views, truth = calibration_set(51, 200)
out = {"model": model, "mazes": {}}
deciders = {lv: prepare(be, lv, views, truth, "batch") for lv in ["raw", "L0", "L1"]}
for s in seeds:
    m = Maze.generate(51, s)
    cells = m.open_cells()
    vs = [m.view(c) for c in cells]
    rec = {lv: {} for lv in deciders}
    for lv, d in deciders.items():
        for name, q in QUESTIONS.items():
            for c, dec in zip(cells, d.decide_batch(vs, q, level=lv)):
                rec[lv].setdefault(f"{c[0]},{c[1]}", {})[name] = dec.p_true
    out["mazes"][s] = rec
    print("done", s, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"dev_probs.{model.split('/')[-1]}.json"), "w"))
