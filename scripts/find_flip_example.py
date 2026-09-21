"""Find short, plausible items where raw readout flips under option reversal and L0 does not.

    python scripts/find_flip_example.py --model Qwen/Qwen3-8B --task banking20 --n 80
Writes candidates to stdout and the best few to assets/flip_examples.json (real model outputs).
"""
import argparse
import json
import os

import numpy as np

from anyjev import Decider
from anyjev.backends.hf import HFBackend
from bench.run import reversed_question
from bench.tasks import get_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="banking20")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=140)
    ap.add_argument("--out", default="assets/flip_examples.json")
    args = ap.parse_args()

    task = get_task(args.task)
    test, _ = task.split(args.n, 0, seed=0)
    states = [s for s, _ in test]
    labels = [y for _, y in test]
    q, qr = task.question, reversed_question(task.question)
    d = Decider(HFBackend(args.model, batch_size=32))
    a = d.decide_batch(states, q, level="L0")
    b = d.decide_batch(states, qr, level="L0")
    opts = list(q.options)
    found = []
    for s, y, da, db in zip(states, labels, a, b):
        raw1 = da.diagnostics["raw_probs"]
        raw2 = db.diagnostics["raw_probs"][::-1]
        l01, l02 = da.probs, db.probs[::-1]
        r1, r2 = int(np.argmax(raw1)), int(np.argmax(raw2))
        l1, l2 = int(np.argmax(l01)), int(np.argmax(l02))
        if r1 != r2 and l1 == l2 and len(s) <= args.max_chars:
            found.append({
                "state": s, "gold": opts[y], "options": opts,
                "raw_as_typed": {opts[i]: round(float(p), 3) for i, p in enumerate(raw1)},
                "raw_reversed": {opts[i]: round(float(p), 3) for i, p in enumerate(raw2)},
                "l0_as_typed": {opts[i]: round(float(p), 3) for i, p in enumerate(l01)},
                "l0_reversed": {opts[i]: round(float(p), 3) for i, p in enumerate(l02)},
                "raw_argmax": [opts[r1], opts[r2]], "l0_argmax": opts[l1], "l0_correct": l1 == y,
            })
    found.sort(key=lambda f: (not f["l0_correct"], len(f["state"])))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "task": args.task, "question": q.text, "examples": found[:10]},
              open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"{len(found)} flip examples out of {len(states)}; L0 correct on {sum(f['l0_correct'] for f in found)}")
    for f in found[:6]:
        print("-", repr(f["state"]), "| gold:", f["gold"], "| raw:", f["raw_argmax"], "| L0:", f["l0_argmax"])


if __name__ == "__main__":
    main()
