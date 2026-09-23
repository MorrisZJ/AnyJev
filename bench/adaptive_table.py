"""Adaptive-shift ablation: full L0 vs adaptive L0 at several margins, per model and task.

    python -m bench.adaptive_table bench/results_batchprior_v0 bench/results_small bench/results_adaptive_m0.05 ...

Directories whose JSON has "adaptive": true are adaptive runs, labelled by their margin; the
others are the full-L0 reference.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load(dirs):
    runs = {}   # (model, task) -> {"full": L0 entry, "m0.05": L0 entry, ...}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "*.json"))):
            r = json.load(open(p))
            if "tasks" not in r:
                continue
            if r.get("adaptive"):
                order = r.get("adaptive_order") or ("spread" if "spread" in os.path.basename(d.rstrip("/"))
                                                    else "consecutive")
                tag = f"m{r['adaptive_margin']:g}-{order}"
            else:
                tag = "full"
            for t in r["tasks"]:
                if t["k"] <= 2 or "L0" not in t["levels"]:
                    continue
                runs.setdefault((r["model"].split("/")[-1], t["task"]), {})[tag] = dict(t["levels"]["L0"], k=t["k"])
    return runs


def main(argv=None):
    dirs = argv or sys.argv[1:]
    runs = load(dirs)
    tags = sorted({tag for v in runs.values() for tag in v if tag != "full"},
                  key=lambda x: (float(x[1:].split("-")[0]), x))
    cols = ["full shifts", "full acc", "full ECE", "full flip"]
    for tag in tags:
        cols += [f"{tag} shifts", f"{tag} acc", f"{tag} ECE", f"{tag} flip"]
    lines = ["| model | task | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for (model, task), v in sorted(runs.items()):
        if "full" not in v:
            continue
        f = v["full"]
        cells = [f"{f['k']:.1f}", f"{f['acc']:.3f}", f"{f['ece']:.3f}", f"{f['flip']:.3f}"]
        for tag in tags:
            a = v.get(tag)
            cells += ([f"{a['mean_shifts']:.1f}", f"{a['acc']:.3f}", f"{a['ece']:.3f}", f"{a['flip']:.3f}"]
                      if a else ["", "", "", ""])
        lines.append(f"| {model} | {task} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    print("\nshifts = mean cyclic shifts read per item (full = K). Adaptive stops when every shift read so far "
          "agrees on the winner after prior correction and the running top-1 minus top-2 probability is at "
          "least the margin.")


if __name__ == "__main__":
    main()
