"""Jev-mode product table for one model: per-question closed-form heads at a depth chosen on
calibration data only, next to the raw / L0 / L1 references and the measured cost.

    python -m bench.jev_mode_table --model Qwen/Qwen3-4B --n-layers 36

CPU replay of the block-loop caches (bench/results_exit/features/<model>/typed.*, written by
`bench.extract_pools`). For every typed question and every cached block: a per-question head
(lda / ridge chosen by CV) with its out-of-fold accuracy and NLL on the 300 train decisions, and
its accuracy / pooled ECE / flip on the 100 test decisions (canonical and reversed listing). Two
deployment rules, both decided without test data:

  cv        every question picks its own block by out-of-fold NLL (what `fit_head` does);
  fixed@b   one block for the whole model: the shallowest block whose pooled out-of-fold accuracy
            is within --slack of the best pooled out-of-fold accuracy.

The table carries the raw / L0 / L1 references from bench/results_typed_v01 and, when
bench.exit_latency has run for the model, the ms per decision at the nearest measured depth.
Output: bench/results_exit/<date>/<model>.jevmode.json and a Markdown table on stdout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from anyjev.heads import fit_head
from bench import metrics
from bench.exit_study import load_typed, pooled
from bench.pools import pool_path
from bench.run import environment


def best_head(X: np.ndarray, y: np.ndarray, K: int, kinds):
    best = None
    for kind in kinds:
        try:
            h = fit_head(X, y, K, kind=kind)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or h.cv["oof_nll"] < best.cv["oof_nll"]:
            best = h
    return best


def typed_references(root: str, model: str) -> Optional[Dict[str, Dict[str, float]]]:
    paths = sorted(glob.glob(os.path.join(root, "*", f"{model.replace('/', '__')}.json")))
    if not paths:
        return None
    d = json.load(open(paths[-1]))
    out = {}
    for m in ("raw", "L0", "L1"):
        rec = d.get("levels", {}).get(m, {}).get("overall")
        if rec and "acc" in rec:
            out[m] = {"acc": float(rec["acc"]), "ece": float(rec.get("ece", float("nan"))), "path": paths[-1]}
    return out or None


def latency_at(root: str, model: str, blocks: int, state_tokens: int = 110) -> Optional[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(root, "*", f"{model.replace('/', '__')}.latency.json")))
    if not paths:
        return None
    d = json.load(open(paths[-1]))
    rows = [r for r in d["rows"] if r["state_tokens"] == state_tokens]
    raw = next((r for r in rows if r["mode"].startswith("raw")), None)
    trunc = [r for r in rows if not r["mode"].startswith("raw")]
    if raw is None or not trunc:
        return None
    near = min(trunc, key=lambda r: abs(r["blocks"] - blocks))
    return {"path": paths[-1], "measured_blocks": near["blocks"], "batch_ms": near["batch_ms"],
            "single_ms": near["single_ms"], "raw_batch_ms": raw["batch_ms"], "raw_single_ms": raw["single_ms"],
            "gflops": near["gflops"], "raw_gflops": raw["gflops"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="bench/results_exit")
    ap.add_argument("--which", default="last")
    ap.add_argument("--calib-cases", type=int, default=300)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--kinds", default="lda,ridge")
    ap.add_argument("--slack", type=float, default=0.005,
                    help="fixed block = shallowest block within this of the best pooled out-of-fold accuracy")
    ap.add_argument("--typed-ref", default="bench/results_typed_v01")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    kinds = args.kinds.split(",")
    data = load_typed(args)
    names = sorted(data)
    layers = [int(x) for x in np.load(pool_path(args.out, args.model, names[0], "test", "id", args.which))["layers"]]
    blocks = [(args.n_layers + 1 + i) if i < 0 else i for i in layers]
    print(f"{args.model}: {len(names)} questions, blocks {blocks} of {args.n_layers}", flush=True)

    per_q: Dict[str, Dict[str, Any]] = {}
    for name in names:
        d = data[name]
        tr, te, rev = d["train_rand"], d["test_id"], d["test_rev"]
        rec: Dict[str, Any] = {"K": te.K, "n_test": te.n, "oof_acc": [], "oof_nll": [], "test": [], "kind": []}
        for li, b in enumerate(blocks):
            h = best_head(tr.H[:, li].astype(np.float32), tr.y, te.K, kinds)
            p = h.probs(te.H[:, li].astype(np.float32))
            p_rev = h.probs(rev.H[:, li].astype(np.float32))
            rec["oof_acc"].append(h.cv["oof_acc"])
            rec["oof_nll"].append(h.cv["oof_nll"])
            rec["kind"].append(h.kind)
            rec["test"].append({"p": p, "flip": metrics.flip_rate(p, p_rev), "temperature": h.temperature})
        rec["cv_block_index"] = int(np.argmin(rec["oof_nll"]))
        per_q[name] = rec
        print(f"   {name:48s} cv-> block {blocks[rec['cv_block_index']]:3d}  test acc by block: "
              + " ".join(f"{np.mean(t['p'].argmax(1) == te.y):.2f}" for t in rec["test"]), flush=True)

    # pooled out-of-fold accuracy per block, calibration data only -> recommended fixed block
    oof_by_block = [float(np.mean([per_q[n]["oof_acc"][li] for n in names])) for li in range(len(blocks))]
    best_oof = max(oof_by_block)
    rec_index = next(li for li, a in enumerate(oof_by_block) if a >= best_oof - args.slack)
    rec_block = blocks[rec_index]

    def rows_for(rule) -> List[tuple]:
        out = []
        for n in names:
            li = per_q[n]["cv_block_index"] if rule == "cv" else rule
            out.append((per_q[n]["test"][li]["p"], data[n]["test_id"].y))
        return out

    def flips_for(rule) -> float:
        return float(np.mean([per_q[n]["test"][per_q[n]["cv_block_index"] if rule == "cv" else rule]["flip"]
                              for n in names]))

    rng = np.random.RandomState(0)
    n_items = sum(per_q[n]["n_test"] for n in names)
    boot = rng.randint(0, n_items, (args.n_boot, n_items))

    def correct_vec(rule) -> np.ndarray:
        return np.concatenate([(p.argmax(1) == y).astype(float) for p, y in rows_for(rule)])

    full_index = len(blocks) - 1
    c_full = correct_vec(full_index)
    result: Dict[str, Any] = {"model": args.model, "n_blocks": args.n_layers, "blocks": blocks, "kinds": kinds,
                              "slack": args.slack, "oof_acc_by_block": oof_by_block, "recommended_block": rec_block,
                              "per_block": {}, "rules": {}, "questions": {}, "date": dt.datetime.now().isoformat(),
                              "env": environment()}
    for li, b in enumerate(blocks):
        c = correct_vec(li)
        accs = c[boot].mean(1)
        diff = (c[boot] - c_full[boot]).mean(1)
        result["per_block"][b] = {**pooled(rows_for(li)), "flip": flips_for(li), "oof_acc": oof_by_block[li],
                                  "acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                                  "diff_vs_full_ci95": [float(np.percentile(diff, 2.5)),
                                                        float(np.percentile(diff, 97.5))],
                                  "relative_depth": b / args.n_layers}
    for rule_name, rule in (("cv", "cv"), (f"fixed@{rec_block}", rec_index), (f"fixed@{blocks[-1]}", full_index)):
        c = correct_vec(rule)
        accs = c[boot].mean(1)
        diff = (c[boot] - c_full[boot]).mean(1)
        mean_blocks = (float(np.mean([blocks[per_q[n]["cv_block_index"]] for n in names])) if rule == "cv"
                       else float(blocks[rule]))
        result["rules"][rule_name] = {**pooled(rows_for(rule)), "flip": flips_for(rule), "mean_blocks": mean_blocks,
                                      "relative_depth": mean_blocks / args.n_layers,
                                      "acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                                      "diff_vs_full_ci95": [float(np.percentile(diff, 2.5)),
                                                            float(np.percentile(diff, 97.5))]}
    for n in names:
        q = per_q[n]
        result["questions"][n] = {"K": q["K"], "cv_block": blocks[q["cv_block_index"]], "oof_acc": q["oof_acc"],
                                  "oof_nll": q["oof_nll"], "kind": q["kind"],
                                  "test_acc": [float(np.mean(t["p"].argmax(1) == data[n]["test_id"].y))
                                               for t in q["test"]],
                                  "test_flip": [t["flip"] for t in q["test"]]}
    result["references"] = typed_references(args.typed_ref, args.model)
    result["latency"] = latency_at(args.out, args.model, rec_block)

    # ---- table
    refs = result["references"] or {}
    lines = [f"### {args.model}: Jev mode (per-question heads, depth chosen on calibration data)", "",
             "| method | acc | 95% CI | vs full-depth head | ECE (pooled) | flip | blocks | depth |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ("raw", "L0", "L1"):
        if m in refs:
            lines.append(f"| {m} | {refs[m]['acc']:.3f} | | | {refs[m]['ece']:.3f} | | {args.n_layers} | 100% |")
    for rule_name, r in result["rules"].items():
        lines.append(f"| head, {rule_name} | {r['acc']:.3f} | [{r['acc_ci95'][0]:.3f}, {r['acc_ci95'][1]:.3f}] | "
                     f"[{r['diff_vs_full_ci95'][0]:+.3f}, {r['diff_vs_full_ci95'][1]:+.3f}] | {r['ece']:.3f} | "
                     f"{r['flip']:.3f} | {r['mean_blocks']:.1f} | {100 * r['relative_depth']:.0f}% |")
    lines += ["", "pooled out-of-fold accuracy by block (calibration data only): "
              + "  ".join(f"{b}:{a:.3f}" for b, a in zip(blocks, oof_by_block)),
              f"recommended fixed block: {rec_block} of {args.n_layers} "
              f"({100 * rec_block / args.n_layers:.0f}% depth; within {args.slack} of the best out-of-fold accuracy)"]
    lat = result["latency"]
    if lat:
        lines.append(f"cost at block {lat['measured_blocks']} (110-token states, one H100): "
                     f"{lat['batch_ms']:.1f} ms per decision batched "
                     f"({lat['batch_ms'] / lat['raw_batch_ms']:.2f}x raw), "
                     f"{lat['single_ms']:.1f} ms single ({lat['single_ms'] / lat['raw_single_ms']:.2f}x raw), "
                     f"{lat['gflops'] / lat['raw_gflops']:.2f}x the FLOPs of the full forward")
    print("\n" + "\n".join(lines))
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.model.replace('/', '__')}.jevmode.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
