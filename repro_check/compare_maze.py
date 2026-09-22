"""Diff my maze reruns against bench/results_nanojev, and re-derive the two
claims the repo makes from that table:

  1. "the same Qwen3-0.6B goes from 13/15 mazes and 20,555 attempts under the
     A/B readout to 15/15 and 5,825 under AnyJev's raw Yes/No readout"
  2. "no readout, not even Qwen3-8B, beats always answering the majority label"

    python repro_check/compare_maze.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REF_DIR = os.path.join(ROOT, "bench/results_nanojev/2026-09-21")
MINE_DIRS = [os.path.join(HERE, "results/nanojev")]
FIELDS = ["attempts", "collisions", "atomic_accuracy", "edge_majority", "atomic_brier",
          "atomic_nll", "atomic_questions", "fallback_count"]


def label(r):
    m = r.get("model", "?").split("/")[-1]
    if r.get("engine") == "nanojev_native":
        return f"{m} native A/B readout"
    return f"{m} {r.get('level')} prior={r.get('prior')}"


def goals(r):
    """goal completion by split, the '10/11' and '3/4' in the repo's table."""
    out = {}
    for ep in r.get("episodes", []):
        sp = ep.get("split", "?")
        d = out.setdefault(sp, [0, 0])
        d[1] += 1
        if ep.get("status") in ("goal", "completed", "success", "solved"):
            d[0] += 1
    return out


def scan(dirs):
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*", "*.json"))):
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if "summary" in r:
                out[os.path.basename(p)] = r
    return out


def main():
    ref, mine = scan([REF_DIR]), scan(MINE_DIRS)
    print("=" * 112)
    print("MAZE DIFF: my rerun inside NanoJev's frozen harness vs bench/results_nanojev")
    print("episodes: C-Tianyu/NanoJev-Data games_v4/data/scaled_games_v4b/episodes.jsonl, splits test+ood (15)")
    print("=" * 112)
    keys = sorted(set(ref) & set(mine))
    if not keys:
        print("no overlapping runs yet")
        print(f"   committed: {sorted(ref)}")
        print(f"   mine     : {sorted(mine)}")
        return 0

    tot = ident = 0
    for k in keys:
        a, b = ref[k], mine[k]
        print(f"\n### {k}   ({label(a)})")
        ga, gb = goals(a), goals(b)
        print(f"   goals repo " + " ".join(f"{s}={v[0]}/{v[1]}" for s, v in sorted(ga.items()))
              + "   | mine " + " ".join(f"{s}={v[0]}/{v[1]}" for s, v in sorted(gb.items())))
        print(f"   {'field':20s} {'repo':>12s} {'mine':>12s} {'delta':>12s}")
        for f in FIELDS:
            x, y = a["summary"].get(f), b["summary"].get(f)
            if x is None or y is None:
                continue
            tot += 1
            d = y - x
            if abs(d) < 1e-12:
                ident += 1
            mark = "" if abs(d) < 1e-9 else "   <-- differs"
            print(f"   {f:20s} {x:12.4f} {y:12.4f} {d:+12.4f}{mark}")

    if tot:
        print()
        print("-" * 112)
        print(f"{tot} comparable summary fields: {ident} bit-identical ({100 * ident / tot:.1f}%)")

    # re-derive the repo's two headline maze claims from whichever data we have
    src = mine if len(mine) >= len(ref) else ref
    print()
    print("=" * 112)
    print("THE REPO'S TWO MAZE CLAIMS, RE-DERIVED FROM " + ("MY RERUN" if src is mine else "THE COMMITTED JSON"))
    print("=" * 112)
    rows = []
    for k, r in sorted(src.items()):
        g = goals(r)
        solved = sum(v[0] for v in g.values())
        total = sum(v[1] for v in g.values())
        rows.append((label(r), solved, total, r["summary"].get("attempts"),
                     r["summary"].get("atomic_accuracy"), r["summary"].get("edge_majority")))
    print(f"{'engine':38s} {'goals':>8s} {'attempts':>10s} {'edge acc':>9s} {'majority':>9s} {'beats majority?':>16s}")
    for name, s, t, att, acc, maj in rows:
        beats = "yes" if (acc is not None and maj is not None and acc > maj) else "no"
        print(f"{name:38s} {f'{s}/{t}':>8s} {att if att is None else f'{att:10.0f}'} "
              f"{acc if acc is None else f'{acc:9.3f}'} {maj if maj is None else f'{maj:9.3f}'} {beats:>16s}")
    nat = [r for r in rows if "native" in r[0]]
    raw = [r for r in rows if " raw " in r[0] and "0.6B" in r[0]]
    if nat and raw:
        print(f"\nclaim 1: A/B readout {nat[0][1]}/{nat[0][2]} mazes at {nat[0][3]:.0f} attempts vs "
              f"AnyJev raw {raw[0][1]}/{raw[0][2]} at {raw[0][3]:.0f} attempts "
              f"-- the repo reports 13/15 @ 20,555 vs 15/15 @ 5,825")
    beats_any = [r for r in rows if r[4] is not None and r[5] is not None and r[4] > r[5]]
    print(f"claim 2: readouts that beat the majority baseline on edge perception: "
          f"{len(beats_any)} of {len(rows)}"
          + (f" ({', '.join(r[0] for r in beats_any)})" if beats_any else " -- none, as the repo states"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
