"""Depth study: what each block of a model already knows about a typed decision.

    python -m bench.exit_study --model Qwen/Qwen3-8B --n-layers 36

Replays caches written by `bench.extract_pools --out bench/results_exit --layers ...`
(the block-loop extraction stores, per layer, the last-position state, the option-line states and
the restricted logit lens). Per layer it reports, pooled over the typed-decisions questions:

  lens        raw readout at that depth (final norm + label rows of lm_head), no labels
  lens+T      the same with a temperature from the question's train split
  head        per-question closed-form head (lda / ridge chosen by CV) on the last-position state

together with the full-depth references. Output: bench/results_exit/<date>/<model>.depth.json and
a Markdown table; the accuracy-vs-depth curve is what bench.jev_mode_table picks the fixed block from.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from anyjev.calibrate.posthoc import TemperatureScaler
from anyjev.heads import fit_head
from bench import metrics
from bench.pools import DecisionPool, load_pool, pool_path, typed_questions
from bench.run import environment


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def load_typed(args) -> Dict[str, Dict[str, DecisionPool]]:
    out = {}
    for name, q, workflow, _, _ in typed_questions(args.calib_cases):
        rec = {}
        for split, order in (("train", "rand"), ("train", "id"), ("test", "id"), ("test", "rev")):
            path = pool_path(args.out, args.model, name, split, order, args.which)
            if not os.path.exists(path):
                print("missing", path)
                rec = None
                break
            rec[f"{split}_{order}"] = load_pool(path)
        if rec:
            out[name] = rec
    return out


def pooled(rows: List[tuple]) -> Dict[str, float]:
    conf = np.concatenate([p.max(1) for p, _ in rows])
    corr = np.concatenate([(p.argmax(1) == y) for p, y in rows]).astype(float)
    nll = np.concatenate([-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)) for p, y in rows])
    return {"acc": float(corr.mean()), "ece": metrics.ece(np.stack([conf, 1 - conf], 1), (1 - corr).astype(int)),
            "nll": float(nll.mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="bench/results_exit")
    ap.add_argument("--which", default="last")
    ap.add_argument("--calib-cases", type=int, default=300)
    ap.add_argument("--n-layers", type=int, default=36)
    args = ap.parse_args(argv)
    data = load_typed(args)
    names = sorted(data)
    first = data[names[0]]["test_id"]
    L = first.H.shape[1]
    layers = [int(x) for x in np.load(pool_path(args.out, args.model, names[0], "test", "id", args.which))["layers"]]
    layers_abs = [(args.n_layers + 1 + i) if i < 0 else i for i in layers]
    print(f"{len(names)} questions, {L} cached layers: {layers_abs}", flush=True)
    result: Dict[str, Any] = {"model": args.model, "layers": layers_abs, "date": dt.datetime.now().isoformat(),
                              "env": environment(), "per_layer": {}}
    rows_by_layer: Dict[int, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
    for li, layer in enumerate(layers_abs):
        for name in names:
            d = data[name]
            tr, te, te_rev = d["train_id"], d["test_id"], d["test_rev"]
            y = te.y
            if te.lens is not None:
                lens = _softmax(te.lens[:, li])
                rows_by_layer[layer]["lens"].append((lens, y))
                T = TemperatureScaler.fit(_softmax(tr.lens[:, li]), tr.y)
                rows_by_layer[layer]["lens+T"].append((T.apply(lens), y))
            best = None
            for kind in ("lda", "ridge"):
                try:
                    h = fit_head(d["train_rand"].H[:, li:li + 1].astype(np.float32), d["train_rand"].y, te.K, kind=kind)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if best is None or h.cv["oof_nll"] < best.cv["oof_nll"]:
                    best = h
            if best is not None:
                p = best.probs(te.H[:, li].astype(np.float32))
                rows_by_layer[layer]["head"].append((p, y))
                rows_by_layer[layer]["head flip"].append((p, best.probs(te_rev.H[:, li].astype(np.float32))))
        rec = {m: pooled(rows) for m, rows in rows_by_layer[layer].items() if m != "head flip"}
        if "head flip" in rows_by_layer[layer]:
            rec["head_flip"] = float(np.mean([metrics.flip_rate(p, pr) for p, pr in rows_by_layer[layer]["head flip"]]))
        result["per_layer"][layer] = rec
        print(f"block {layer:2d} ({100 * layer / args.n_layers:3.0f}%): "
              + "  ".join(f"{m} acc {r['acc']:.3f} ece {r['ece']:.3f}" for m, r in rec.items() if isinstance(r, dict))
              + (f"  head flip {rec['head_flip']:.3f}" if "head_flip" in rec else ""), flush=True)
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.model.replace('/', '__')}.depth.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
