"""Shared-prefix path vs full prompts on a real model.
    python scripts/shared_prefix_parity.py --model Qwen/Qwen3-1.7B --dtype float32
Reports the max |difference| of the restricted log-softmax across K=4 and K=20 groups, the
number of groups that fell back because the tokenizer split was not exact, and bf16 timing.
"""
import argparse
import time

import numpy as np

from anyjev import Question
from anyjev.backends.hf import HFBackend
from anyjev.readout import build_prompt, label_ids_for_perm, render_chat_parts, resolve_labels
from anyjev.calibrate.permute import cyclic_shifts

STATES = [
    "My card was charged twice for one order and the app crashes when I open receipts.",
    "Do you offer bulk discounts for teams of fifty or more?",
    "Thanks, everything is sorted now, no further action needed.",
    "URGENT: production is down for all users since the last deploy. " * 8,
    "I moved to a new country and need to update the address on my account and my tax residency. " * 20,
]


def lsm(x):
    x = np.asarray(x, float)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    be = HFBackend(args.model, dtype=args.dtype, batch_size=args.batch_size)
    for k in (4, 20):
        q = Question.choice("Which team should handle this?", [f"team_{i}" for i in range(k)])
        labels, ids = resolve_labels(be.tokenizer, q)
        groups, flat_prompts, flat_ids = [], [], []
        for st in STATES:
            pre = None
            sfx = []
            for perm in cyclic_shifts(k):
                p, s = render_chat_parts(be.tokenizer, build_prompt(st, q, perm, labels=labels))
                pre = p if pre is None else pre
                assert p == pre
                sfx.append(s)
                flat_prompts.append(p + s)
                flat_ids.append(label_ids_for_perm(q, ids, perm))
            groups.append((pre, sfx))
        t0 = time.time()
        a = be.next_token_logprobs(flat_prompts, flat_ids)
        ta = time.time() - t0
        t0 = time.time()
        b = be.score_shared(groups, [ids] * len(groups))
        tb = time.time() - t0
        b_flat = [lp for g in b for lp in g]
        d_raw = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b_flat))
        d_lsm = max(float(np.max(np.abs(lsm(x) - lsm(y)))) for x, y in zip(a, b_flat))
        agree = sum(int(np.argmax(x) == np.argmax(y)) for x, y in zip(a, b_flat))
        print(f"K={k:2d} prompts={len(a):3d} max|dlogp|={d_raw:.2e} max|dlogsoftmax|={d_lsm:.2e} "
              f"argmax agree {agree}/{len(a)} fallbacks={be.shared_fallbacks} "
              f"full {ta:.2f}s shared {tb:.2f}s speedup {ta / tb:.1f}x")


if __name__ == "__main__":
    main()
