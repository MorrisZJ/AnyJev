"""L2 on the image tasks: a closed-form head per question, fit on the calibration split that L1
uses, scored on the same test split as `bench.run` (same seed, same items).

    python -m bench.mm_l2_study --model Qwen/Qwen3-VL-4B-Instruct --tasks pets20,pope --seed 0

The head reads the language model's last-position hidden state after the image prompt (one
full forward per state: a vision model derives its positions from the image grid, so there is
no early stop). For a `choice` task the flip is the disagreement of two independently fit
deployments, the canonical and the reversed option list, as `bench.run` defines it for L1; a
`noul` head answers in one phrasing, so its flip is not measured.

Every run also fits L1 on the same calibration labels (L0 comes from the same forward passes),
so the levels compare item by item, and stores its per-item test predictions with two leakage
flags: whether the item's picture also appears in the calibration split, and whether the
identical (picture, text) state does. POPE asks several questions per COCO image, so a random
split of its questions shares pictures between the splits.

Each run writes two files to `bench/results_mm_l2/<date>/`: `<model>.seed<k>.json`, the
settings, the metrics of every level, the chosen head and the environment; and
`<model>.seed<k>.items.json`, the per-item test predictions (probabilities to six decimals) with
the leakage flags. `bench.mm_l2_audit` reads both.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time

import numpy as np

from anyjev import Decider
from anyjev.state import split_state
from bench import metrics
from bench.run import environment, reversed_question
from bench.tasks import get_task


def _keys(state):
    """(picture digest, state digest): identical pixels, and identical pixels plus identical text."""
    text, images = split_state(state)
    pix = hashlib.sha1(b"".join(im.load().tobytes() for im in images)).hexdigest()
    return pix, hashlib.sha1((pix + "|" + text).encode()).hexdigest()


def run_task(backend, name: str, n_test: int, n_calib: int, seed: int) -> dict:
    task = get_task(name)
    q = task.question
    test, calib = task.split(n_test, n_calib, seed)
    states, labels = [s for s, _ in test], [y for _, y in test]
    c_states, c_labels = [s for s, _ in calib], [y for _, y in calib]

    d = Decider(backend)
    t0 = time.time()
    art = d.fit_head(q, c_states, c_labels)
    t_fit = time.time() - t0
    t0 = time.time()
    decs = d.decide_batch(states, q, level="L2", require="L2")
    t_dec = time.time() - t0
    P = np.stack([x.probs for x in decs])

    Pr, art_r = None, None
    if q.kind == "choice":
        qr = reversed_question(q)
        d_r = Decider(backend)
        art_r = d_r.fit_head(qr, c_states, [q.k - 1 - y for y in c_labels])
        Pr = np.stack([x.probs for x in d_r.decide_batch(states, qr, level="L2")])[:, ::-1]

    d1 = Decider(backend)
    art1 = d1.calibrate(q, c_states, c_labels)
    dl1 = d1.decide_batch(states, q, level="L1", require="L1")
    P_l1 = np.stack([x.probs for x in dl1])
    P_l0 = np.stack([x.diagnostics["l0_probs"] for x in dl1])
    levels = {"L0": metrics.summarize(P_l0, labels),
              "L1": {**metrics.summarize(P_l1, labels), "temperature": art1["temperature"]},
              "L2": metrics.summarize(P, labels, Pr)}

    calib_keys = [_keys(s) for s in c_states]
    seen_pix, seen_state = {k[0] for k in calib_keys}, {k[1] for k in calib_keys}
    items = []
    for i, (st, y) in enumerate(test):
        pix, full = _keys(st)
        items.append({"i": i, "label": int(y), "picture_in_calib": pix in seen_pix,
                      "state_in_calib": full in seen_state, "p_l0": P_l0[i].tolist(),
                      "p_l1": P_l1[i].tolist(), "p_l2": P[i].tolist()})

    diag = decs[0].diagnostics
    return {
        "task": name, "question": q.text, "k": q.k, "n_test": len(test), "n_calib": len(calib),
        "L2": levels["L2"], "levels": levels, "items": items,
        "head": {"kind": art["method"], "layer_abs": art["layer_abs"], "n_blocks": art["n_blocks"],
                 "relative_depth": art["layer_abs"] / art["n_blocks"], "temperature": art["temperature"],
                 "listing": art["params"].get("listing"), "cv": art["cv"],
                 "early_stop": diag["early_stop"]},
        "head_reversed": None if art_r is None else {"kind": art_r["method"], "layer_abs": art_r["layer_abs"],
                                                      "temperature": art_r["temperature"]},
        "seconds_fit": t_fit, "seconds_per_decision_l2": t_dec / max(1, len(test)),
    }


def _round(x, nd=6):
    return [round(float(v), nd) for v in x]


def save(result: dict, stem: str) -> None:
    """The summary (indented, small) and the per-item predictions (compact) as two files."""
    summary = {**result, "tasks": [{k: v for k, v in t.items() if k != "items"} for t in result["tasks"]]}
    items = {"model": result["model"], "seed": result["seed"],
             "tasks": {t["task"]: [{**it, **{k: _round(it[k]) for k in ("p_l0", "p_l1", "p_l2")}}
                                   for it in t["items"]] for t in result["tasks"]}}
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    with open(stem + ".items.json", "w") as f:
        json.dump(items, f, separators=(",", ":"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default=None, help="local checkpoint; --model stays the recorded name")
    ap.add_argument("--tasks", default="pets20,pope")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--out", default="bench/results_mm_l2")
    args = ap.parse_args(argv)

    from anyjev.backends.hf_vlm import VLMBackend
    backend = VLMBackend(args.model_path or args.model, batch_size=args.batch_size, max_pixels=args.max_pixels)
    backend.name = args.model
    result = {"model": args.model, "seed": args.seed, "n": args.n, "calib": args.calib,
              "max_pixels": args.max_pixels, "env": environment(batch_size=args.batch_size, dtype=backend.dtype,
                                                                 backend="vlm"), "tasks": []}
    outdir = os.path.join(args.out, dt.date.today().isoformat())
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{args.model.replace('/', '__')}.seed{args.seed}")
    for name in args.tasks.split(","):
        res = run_task(backend, name, args.n, args.calib, args.seed)
        result["tasks"].append(res)
        m = res["L2"]
        print(f"{args.model} {name}: L2 acc {m['acc']:.3f} ece {m['ece']:.3f} aurc {m['aurc']:.3f} "
              f"flip {m.get('flip', float('nan')):.3f} | block {res['head']['layer_abs']}/{res['head']['n_blocks']} "
              f"{res['head']['kind']} | fit {res['seconds_fit']:.0f}s", flush=True)
        save(result, stem)
    print("wrote", stem + ".json", "and", stem + ".items.json")


if __name__ == "__main__":
    main()
