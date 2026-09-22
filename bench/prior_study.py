"""Offline prior study: replay every prior rule on dumped per-item distributions, no model calls.

    python -m bench.prior_study bench/results_typed_dump bench/results_dump [--alphas 0,0.25,0.5,0.75,1]

Input: the `*.items.json` (or `.json.gz`) files written by `bench.run --dump-items` / `bench.run_typed --dump-items`
(per item: raw position-space distributions for every cyclic shift, the shifts, the gold label,
and the content-free prior when recorded). For each (model, question) it evaluates, with the same
log-space permutation marginalization the Decider uses:

  raw            shift 0 only, no prior
  perm           all shifts, no prior
  perm+bc^a      all shifts, batch prior raised to the power a (a=1 is the default L0; a<1 shrinks)
  perm+cf        all shifts, content-free prior (when dumped)

and reports accuracy gain over raw per kind, per label-skew bucket, and the best alpha overall.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from anyjev.calibrate.contextual import apply_contextual, batch_prior
from anyjev.calibrate.permute import marginalize


def load(dirs):
    units = []   # dict(model, name, kind, k, perms, P [n,P,K], gold [n], cf [P,K] or None, majority, entropy)
    for d in dirs:
        files = glob.glob(os.path.join(d, "*", "*.items.json")) + glob.glob(os.path.join(d, "*", "*.items.json.gz"))
        for p in sorted(files):
            if p.endswith(".gz"):
                import gzip
                with gzip.open(p, "rt") as fh:
                    r = json.load(fh)
            else:
                r = json.load(open(p))
            model = r["model"].split("/")[-1]
            groups = r.get("questions") or r.get("tasks")
            for key, g in groups.items():
                P = np.array(g["p_pos_raw"], dtype=np.float64)
                gold = np.array(g["gold"])
                counts = np.bincount(gold, minlength=g["k"]).astype(float)
                pm = counts / counts.sum()
                ent = float(-(pm[pm > 0] * np.log(pm[pm > 0])).sum() / np.log(g["k"])) if g["k"] > 1 else 0.0
                units.append({"model": model, "name": g.get("name", key), "kind": g["kind"], "k": g["k"],
                              "perms": g["perms"], "P": P, "gold": gold,
                              "cf": np.array(g["cf_prior"]) if g.get("cf_prior") is not None else None,
                              "majority": float(pm.max()), "entropy": ent, "n": len(gold)})
    return units


def accuracy(P_by_item, perms, gold, prior=None, alpha=1.0, only_first=False):
    correct = 0
    for P, y in zip(P_by_item, gold):
        p = P[:1] if only_first else P
        pp = perms[:1] if only_first else perms
        if prior is not None and alpha > 0:
            pr = prior[:1] if only_first else prior
            p = apply_contextual(p, np.clip(pr, 1e-8, None) ** alpha)
        m = marginalize(p, pp)
        correct += int(np.argmax(m) == y)
    return correct / len(gold)


def position_profile(P):
    """[K] mean probability at each *position* over items and shifts. Content cannot prefer a
    position across every layout, so this component is position bias by construction."""
    prof = P.mean(axis=(0, 1))
    prof = np.clip(prof, 1e-8, None)
    return prof / prof.sum()


def option_marginal(P, perms):
    """[K] mean of the permutation-marginalized (perm-only) distribution over items, in option
    space: the part of the batch mean that could be label bias or the true label marginal."""
    m = np.stack([marginalize(Pi, perms) for Pi in P]).mean(axis=0)
    return m / m.sum()


def evaluate(u, alphas, taus):
    P, perms, gold = u["P"], u["perms"], u["gold"]
    bp = batch_prior(P)                                   # [P, K] mean over items, per shift
    out = {"raw": accuracy(P, perms, gold, only_first=True), "perm": accuracy(P, perms, gold)}
    for a in alphas:
        out[f"perm+bc^{a:g}"] = accuracy(P, perms, gold, bp, a)
    prof = position_profile(P)
    pos_prior = np.tile(prof, (P.shape[1], 1))            # same profile for every shift
    out["perm+pos"] = accuracy(P, perms, gold, pos_prior, 1.0)
    skew = float(option_marginal(P, perms).max())
    out["_skew_pred"] = skew
    for t in taus:                                        # guard: full prior unless the batch looks skewed
        out[f"guard{t:g}:bc|pos"] = out["perm+pos"] if skew > t else out["perm+bc^1"]
        out[f"guard{t:g}:bc|perm"] = out["perm"] if skew > t else out["perm+bc^1"]
    if u["cf"] is not None:
        out["perm+cf"] = accuracy(P, perms, gold, u["cf"], 1.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--alphas", default="0.5,0.75,1")
    ap.add_argument("--taus", default="0.5,0.6,0.7")
    args = ap.parse_args(argv)
    alphas = [float(x) for x in args.alphas.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    units = load(args.dirs)
    print(f"{len(units)} (model, question) units from {len(args.dirs)} dirs\n")
    rows = []
    for u in units:
        res = evaluate(u, alphas, taus)
        rows.append({**{k: v for k, v in u.items() if k not in ("P", "gold", "perms", "cf")}, **res})
    variants = [k for k in rows[0] if k != "raw" and not k.startswith("_") and k not in
                ("model", "name", "kind", "k", "majority", "entropy", "n")]
    sk = np.argsort(np.argsort([r["_skew_pred"] for r in rows]))
    mj = np.argsort(np.argsort([r["majority"] for r in rows]))
    print(f"predicted skew (batch option-marginal max) vs gold majority, Spearman: {np.corrcoef(sk, mj)[0, 1]:+.2f}\n")

    def report(sub, title):
        if not sub:
            return
        print(f"== {title} (n={len(sub)})")
        print(f"   {'variant':16s} {'mean gain':>9s} {'>0':>7s} {'<-0.01':>7s} {'mean acc':>9s}")
        for v in variants:
            gains = [r[v] - r["raw"] for r in sub if v in r]
            if not gains:
                continue
            print(f"   {v:16s} {np.mean(gains):+9.3f} {sum(g > 0 for g in gains):4d}/{len(gains):<3d} "
                  f"{sum(g < -0.01 for g in gains):5d}   {np.mean([r[v] for r in sub if v in r]):.3f}")

    report([r for r in rows if r["kind"] == "choice" and r["k"] >= 10], "choice, K >= 10 (three-task bench, balanced)")
    report([r for r in rows if r["kind"] == "choice" and r["k"] < 10], "choice, K < 10 (typed-decisions, often skewed)")
    for kind in ("noul", "score"):
        report([r for r in rows if r["kind"] == kind], f"kind = {kind}")
    for lo, hi in ((0.0, 0.45), (0.45, 0.65), (0.65, 1.01)):
        report([r for r in rows if lo <= r["majority"] < hi and r["kind"] != "score"],
               f"choice+noul, gold majority share in [{lo:.2f}, {hi:.2f})")
    print("\n== best variant by mean gain over all units:")
    best = sorted(((np.mean([r[v] - r["raw"] for r in rows if v in r]), v) for v in variants), reverse=True)
    for g, v in best:
        print(f"   {v:16s} {g:+.3f}")


if __name__ == "__main__":
    main()
