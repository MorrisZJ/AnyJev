"""Summarise a heads_study JSON with the head chosen the honest way: per question, by the
out-of-fold NLL on the calibration set (no test data touches the choice).

    python -m bench.heads_table bench/results_heads/2026-09-22/Qwen__Qwen3-8B.typed.json
    python -m bench.heads_table bench/results_heads/2026-09-22/Qwen__Qwen3-8B.bench.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

BASELINES = ["raw", "L1", "L0-perm", "L0-perm+T"]


def pick_head(heads: Dict[str, Any], rows: Dict[str, Any], tag: str = "") -> str:
    """The head with the lowest out-of-fold NLL on the calibration set among those fit on the
    given label source ('' = gold, '@self', '@think')."""
    cands = [m for m in heads if m.startswith("head:") and m.endswith(tag) and "error" not in heads[m]
             and (tag or "@" not in m) and m in rows]
    return min(cands, key=lambda m: heads[m]["oof_nll"]) if cands else ""


def per_question_rows(q: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    rows, heads = q["rows"], q["heads"]
    out = {b: rows[b] for b in BASELINES if b in rows}
    for src, tag in (("gold", ""), ("self", "@self"), ("think", "@think")):
        m = pick_head(heads, rows, tag)
        if m:
            out[f"head (cv-chosen, {src} labels)"] = dict(rows[m], chosen=m)
        if f"L1{tag}" in rows and tag:
            out[f"L1 ({src} labels)"] = rows[f"L1{tag}"]
    # the oracle upper bound of the selection: best head on test, to show how much cv-choice gives up
    gold_heads = [m for m in rows if m.startswith("head:") and "@" not in m]
    best = max(gold_heads, key=lambda m: rows[m]["acc"]) if gold_heads else ""
    if best:
        out["head (best on test, optimistic)"] = dict(rows[best], chosen=best)
    return out


def fmt(rows: Dict[str, Dict[str, float]], title: str, extra: str = "") -> str:
    cols = ["acc", "ece", "brier", "flip"]
    lines = [f"### {title}", "", f"| method | {' | '.join(cols)} |{extra}",
             "|---|" + "---|" * (len(cols) + bool(extra))]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |"
                     + (f" {r.get('chosen', '')} |" if extra else ""))
    return "\n".join(lines)


def main(argv: List[str]):
    path = argv[0]
    d = json.load(open(path))
    if "typed" in d:
        qs = d["typed"]["questions"]
        pooled_n = defaultdict(list)
        per_q_tables = []
        for name, q in qs.items():
            rows = per_question_rows(q)
            per_q_tables.append((name, q["k"], rows))
            for m, r in rows.items():
                pooled_n[m].append((r, q["n_test"]))
        # decision-weighted mean over questions (every question has the same n_test here)
        pooled = {m: {c: float(np.average([r[c] for r, _ in v], weights=[n for _, n in v]))
                      for c in ("acc", "ece", "brier", "flip")} for m, v in pooled_n.items()}
        print(fmt(pooled, f"typed-decisions, {len(qs)} questions x {qs[next(iter(qs))]['n_test']} test decisions, "
                          f"{d['model']}, calib {qs[next(iter(qs))]['n_calib']} per question"))
        print("\nLaya fine-tuned on the train split (published): acc 0.766; Jev 1.13 (published): 0.727; "
              "our measured Laya-ft: 0.768.\n")
        print("| question | K | raw | L0-perm | head (cv) | chosen | head@self | best on test |")
        print("|---|---|---|---|---|---|---|---|")
        for name, k, rows in per_q_tables:
            h = rows.get("head (cv-chosen, gold labels)", {})
            hs = rows.get("head (cv-chosen, self labels)", {})
            b = rows.get("head (best on test, optimistic)", {})
            print(f"| {name.replace('typed.', '')} | {k} | {rows['raw']['acc']:.2f} | {rows['L0-perm']['acc']:.2f} | "
                  f"{h.get('acc', float('nan')):.2f} | {h.get('chosen', '')} | {hs.get('acc', float('nan')):.2f} | "
                  f"{b.get('acc', float('nan')):.2f} |")
        by_kind = defaultdict(list)
        for name, q in qs.items():
            rows = per_question_rows(q)
            head_acc = rows.get("head (cv-chosen, gold labels)", {}).get("acc", np.nan)
            by_kind[q["kind"]].append((rows["raw"]["acc"], head_acc))
        print("\nby kind (raw -> head, mean acc): " + "; ".join(
            f"{k}: {np.mean([a for a, _ in v]):.3f} -> {np.nanmean([b for _, b in v]):.3f} (n={len(v)})"
            for k, v in by_kind.items()))
    else:
        for t in d["tasks"]:
            rows = per_question_rows(t)
            print(fmt(rows, f"{t['task']} (K={t['k']}, n_test={t['n_test']}, n_calib={t['n_calib']}), {d['model']}",
                      extra=" chosen |"))
            if "label_agreement" in t:
                print("label agreement with gold on the calibration split: "
                      + ", ".join(f"{k} {v:.3f}" for k, v in t["label_agreement"].items()))
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
