"""Parity of the block-loop forward against the plain forward, per architecture.

    python scripts/exit_parity.py --model Qwen/Qwen3-0.6B [--dtype float32] [--device cuda]

Checks, on a handful of prompts of different lengths (left padding, batch):
  1. hidden_states_to(layers=[l, n]) == hidden_states(layers=[l, n]) at every requested layer
  2. the label log-probs from the loop equal the plain path's
  3. the logit lens at the last block equals the full-vocab logits at the label rows
  4. stopping at block l gives the same layer-l state as the full run (truncation is exact)
Prints max abs / rel deviations; the study is only trusted for architectures that pass here.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from anyjev import Question
from anyjev.backends.hf import HFBackend
from anyjev.readout import DEFAULT_SYSTEM, build_prompt, label_ids_for_perm, render_chat, resolve_labels

STATES = ["The customer was charged twice for one order.",
          "I cannot log in since the update; the app crashes on the receipts page every single time I open it, "
          "which is a problem because I need the invoices for my accountant by Friday.",
          "hi", "Please cancel my subscription and refund the last month.", "The server room is on fire."]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=None, help="early block to compare (default: half depth)")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args(argv)
    be = HFBackend(args.model, device=args.device, dtype=args.dtype, batch_size=8, revision=args.revision)
    n = be.n_layers
    layer = args.layer or n // 2
    q = Question.choice("Which team should handle this?", ["billing", "technical", "sales", "other"])
    labels, ids = resolve_labels(be.tokenizer, q)
    perm = [0, 1, 2, 3]
    prompts = [render_chat(be.tokenizer, build_prompt(s, q, perm, DEFAULT_SYSTEM, labels)) for s in STATES]
    token_ids = [label_ids_for_perm(q, ids, perm)] * len(prompts)
    layers = [layer, n]
    f_plain, lp_plain, _ = be.hidden_states(prompts, layers, token_ids)
    try:
        f_loop, lp_loop, _, lens = be.hidden_states_to(prompts, layers, token_ids, max_layer=n, lens_ids=ids)
    except NotImplementedError as e:
        print(f"UNSUPPORTED by the block loop: {e}")
        sys.exit(2)
    rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))  # noqa: E731
    print(f"{args.model}: {n} blocks, compare block {layer} and final norm, dtype {args.dtype}")
    for li, ly in enumerate(layers):
        dmax = np.abs(f_loop[:, li] - f_plain[:, li]).max()
        print(f"  layer {ly:3d}: max|dh| = {dmax:.3e}  rel {rel(f_loop[:, li], f_plain[:, li]):.2e}")
    lp_a, lp_b = np.stack(lp_loop), np.stack(lp_plain)
    print(f"  label log-probs: max|d| = {np.abs(lp_a - lp_b).max():.3e}")
    lens_ls = np.log(np.exp(lens[:, 1]) / np.exp(lens[:, 1]).sum(1, keepdims=True))
    plain_ls = lp_b - np.log(np.exp(lp_b).sum(1, keepdims=True))
    print(f"  logit lens at the last block vs full logits (as log-softmax over the labels): "
          f"max|d| = {np.abs(lens_ls - plain_ls).max():.3e}")
    # truncation: stop at `layer`, compare with the full run's layer-l state
    f_trunc, _, _, _ = be.hidden_states_to(prompts, [layer], None, max_layer=layer)
    print(f"  truncated at {layer}: max|dh| vs full run = {np.abs(f_trunc[:, 0] - f_plain[:, 0]).max():.3e}")
    ok = rel(f_loop[:, 1], f_plain[:, 1]) < 1e-3 and np.abs(lp_a - lp_b).max() < 1e-2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
