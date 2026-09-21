"""Print typed-decisions results by level, question type and workflow.

    python -m bench.typed_summary bench/results_typed/<date>
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    d = argv[0] if argv else sorted(glob.glob("bench/results_typed/*"))[-1]
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        print("==", r.get("model") or r.get("checkpoint"))
        levels = r.get("levels") or {"provider": {"overall": r["overall"], "by_type": r.get("by_type", {})}}
        for lvl, e in levels.items():
            o = e["overall"]
            types = "  ".join(f"{t}: {m['acc']:.3f}/{m['soft_acc']:.3f} (n={m['n']})" for t, m in e["by_type"].items())
            print(f"  {lvl:10s} n={o['n']} acc={o['acc']:.3f} soft={o['soft_acc']:.3f} ece={o['ece']:.3f} "
                  f"mae={o.get('score_mae', float('nan')):.3f} | {types}")
            if "by_workflow" in e:
                print("       " + "  ".join(f"{w[:12]}: {m['acc']:.3f}" for w, m in e["by_workflow"].items()))


if __name__ == "__main__":
    main()
