import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from bench.games.maze import Maze, Explorer, DIRS

data = json.load(open(sys.argv[1]))
POLICIES = [("threshold", 0), ("expected_cost", 0), ("threshold", 10), ("expected_cost", 10)]
for pol, wc in POLICIES:
    print(f"== policy {pol} wall_cost {wc}")
    tot = {}
    for s, rec in data["mazes"].items():
        m = Maze.generate(51, int(s))
        srcs = dict(rec)
        srcs["always_yes"] = None
        row = []
        for name, probs in srcs.items():
            ex = Explorer(m.size, m.start, m.goal, policy=pol, wall_cost=wc, horizon=4 * 51 * 51)
            while not (st := ex.done(m)):
                if ex.needs_prediction():
                    ex.remember({d: 0.9 for d in DIRS} if probs is None else probs[f"{ex.pos[0]},{ex.pos[1]}"])
                ex.step(m)
            row.append(f"{name}={ex.steps if st == 'goal' else 'FAIL'}/{ex.collisions}")
            tot.setdefault(name, []).append(ex.steps if st == "goal" else None)
        print(" ", s, "  ".join(row), flush=True)
    print("  solved:", {k: sum(v is not None for v in vs) for k, vs in tot.items()},
          "mean steps (solved):", {k: round(sum(x for x in vs if x) / max(1, sum(x is not None for x in vs))) for k, vs in tot.items()})
