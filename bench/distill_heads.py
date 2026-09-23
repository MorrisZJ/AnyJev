"""Closed-form distillation: a big model's L2 heads teach a small model's L2 heads (E14).

Stages (all typed-decisions, 20 questions, test = the 100 held-out decisions per question):

    python -m bench.distill_heads soft --teacher Qwen/Qwen3-32B --teacher-block 52 \
        --students Qwen/Qwen3-1.7B:18,Qwen/Qwen3-4B:24,Qwen/Qwen3-8B:24
        CPU replay on the bench.extract_pools caches: the teacher head's out-of-fold probabilities on the
        300 train states become soft targets for the student head (same states, no new data).

    python -m bench.distill_heads label --teacher Qwen/Qwen3-32B --synth bench/results_distill/synth
        The teacher's shipped heads (anyjev-heads/<teacher>.json, level L2) label the synthetic
        states of every workflow for each of its five questions, and the test states.

    python -m bench.distill_heads extract --student Qwen/Qwen3-1.7B --block 18 --synth ...
        Student hidden states at its fixed block on the synthetic states (random listing order,
        the deployed fit path).

    python -m bench.distill_heads fit --teacher Qwen/Qwen3-32B --students Qwen/Qwen3-1.7B:18,...
        CPU replay: student heads on gold (300), on teacher labels of n synthetic states, on both,
        with hard (argmax) or soft (probability) targets; the teacher's own test accuracy; pooled
        accuracy / ECE with bootstrap CIs paired against the gold-only head.

Outputs under bench/results_distill/: synth_labels/<teacher>/<question>.npz, synth_features/
<student>/<question>.npz, and <date>/<teacher>.distill_heads.{soft,fit}.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anyjev.calibrate.posthoc import TemperatureScaler
from anyjev.heads import LinearHead, _folds, _standardise, fit_head
from bench import metrics
from bench.pools import decisions_to_arrays, load_pool, pool_path, typed_questions
from bench.run import environment

EPS = 1e-12


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def best_head(X: np.ndarray, y: np.ndarray, K: int, kinds=("lda", "ridge")) -> Optional[LinearHead]:
    best = None
    for kind in kinds:
        try:
            h = fit_head(X, y, K, kind=kind)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or h.cv["oof_nll"] < best.cv["oof_nll"]:
            best = h
    return best


def per_question_probs(H_tr, y_tr, H_te, K, n_folds=5, seed=0):
    """(oof probs on the train pool, test probs) of a per-question head on one layer's features."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y_tr))
    folds = np.array_split(idx, n_folds)
    oof = np.zeros((len(y_tr), K))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        h = best_head(H_tr[tr], y_tr[tr], K)
        oof[f] = h.probs(H_tr[f])
    h = best_head(H_tr, y_tr, K)
    return oof, h.probs(H_te)


def ridge_targets(X: np.ndarray, T: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge from standardised features to a target matrix T [N, K] (dual form)."""
    xm, tm = X.mean(axis=0), T.mean(axis=0)
    Xc, Tc = X - xm, T - tm
    A = Xc @ Xc.T + lam * np.eye(len(Xc))
    W = Xc.T @ np.linalg.solve(A, Tc)
    return W, tm - xm @ W


def fit_soft_head(X: np.ndarray, T: np.ndarray, lams: Sequence[float] = (1.0, 10.0, 100.0, 1000.0),
                  n_folds: int = 5, seed: int = 0) -> LinearHead:
    """A ridge head to soft targets T (rows sum to 1); lambda and the temperature chosen by
    out-of-fold cross-entropy against T itself, so no gold label is used."""
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    folds = _folds(len(T), n_folds, seed)
    best = None
    for lam in lams:
        oof = np.zeros_like(T)
        for f in folds:
            tr = np.setdiff1d(np.arange(len(T)), f)
            mean, scale = _standardise(X[tr])
            W, b = ridge_targets((X[tr] - mean) / scale, T[tr], lam)
            oof[f] = ((X[f] - mean) / scale) @ W + b
        scaler = TemperatureScaler.fit(_softmax(oof), T.argmax(1))
        p = scaler.apply(_softmax(oof))
        ce = float(-np.mean(np.sum(T * np.log(np.clip(p, EPS, None)), axis=1)))
        if best is None or ce < best[0]:
            best = (ce, lam, scaler.temperature)
    ce, lam, temperature = best
    mean, scale = _standardise(X)
    W, b = ridge_targets((X - mean) / scale, T, lam)
    return LinearHead(kind="ridge_soft", layer=0, W=W, b=b, mean=mean, scale=scale, temperature=temperature,
                      params={"lam": lam}, n_calib=len(T), cv={"oof_ce": ce, "oof_acc": float("nan")})


def pooled(rows: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    conf = np.concatenate([p.max(1) for p, _ in rows])
    corr = np.concatenate([(p.argmax(1) == y) for p, y in rows]).astype(float)
    nll = np.concatenate([-np.log(np.clip(p[np.arange(len(y)), y], EPS, None)) for p, y in rows])
    return {"acc": float(corr.mean()), "ece": metrics.ece(np.stack([conf, 1 - conf], 1), (1 - corr).astype(int)),
            "nll": float(nll.mean()), "n": int(len(corr))}


def with_cis(rows: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], ref: str, n_boot: int = 1000) -> Dict[str, Any]:
    out = {}
    c_ref = np.concatenate([(p.argmax(1) == y).astype(float) for p, y in rows[ref]])
    boot = np.random.RandomState(0).randint(0, len(c_ref), (n_boot, len(c_ref)))
    for m, items in rows.items():
        c = np.concatenate([(p.argmax(1) == y).astype(float) for p, y in items])
        rec = pooled(items)
        rec["acc_ci95"] = [float(np.percentile(c[boot].mean(1), 2.5)), float(np.percentile(c[boot].mean(1), 97.5))]
        if len(c) == len(c_ref):
            d = (c[boot] - c_ref[boot]).mean(1)
            rec[f"diff_vs_{ref}_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out[m] = rec
    return out


def parse_students(spec: str) -> List[Tuple[str, int]]:
    out = []
    for item in spec.split(","):
        model, block = item.rsplit(":", 1)
        out.append((model, int(block)))
    return out


def cache_features(root: str, model: str, name: str, split: str, order: str, block: int, n_layers: int):
    """(H at `block`, y) from an exit-study cache."""
    pool = load_pool(pool_path(root, model, name, split, order, "last"))
    z = np.load(pool_path(root, model, name, split, order, "last"))
    layers = [int(x) for x in z["layers"]]
    blocks = [(n_layers + 1 + i) if i < 0 else i for i in layers]
    return pool.H[:, blocks.index(block)].astype(np.float32), pool.y, pool.K


N_LAYERS = {"Qwen/Qwen3-1.7B": 28, "Qwen/Qwen3-4B": 36, "Qwen/Qwen3-8B": 36, "Qwen/Qwen3-32B": 64,
            "Qwen/Qwen3-30B-A3B-Instruct-2507": 48}


# ---------------------------------------------------------------- soft (same states, soft targets)
def soft(args) -> None:
    students = parse_students(args.students)
    names = [n for n, *_ in typed_questions(args.calib_cases)]
    rows: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    per_q: Dict[str, Dict[str, float]] = {}
    for name in sorted(names):
        Ht, y_tr, K = cache_features(args.exit_root, args.teacher, name, "train", "rand", args.teacher_block,
                                     N_LAYERS[args.teacher])
        Ht_te, y_te, _ = cache_features(args.exit_root, args.teacher, name, "test", "id", args.teacher_block,
                                        N_LAYERS[args.teacher])
        t_oof, t_test = per_question_probs(Ht, y_tr, Ht_te, K)
        rows.setdefault("teacher", []).append((t_test, y_te))
        per_q[name] = {"teacher": float(np.mean(t_test.argmax(1) == y_te)),
                       "teacher_oof_agreement_with_gold": float(np.mean(t_oof.argmax(1) == y_tr))}
        for model, block in students:
            tag = model.split("/")[-1]
            Xs, ys, _ = cache_features(args.exit_root, model, name, "train", "rand", block, N_LAYERS[model])
            Xs_te, _, _ = cache_features(args.exit_root, model, name, "test", "id", block, N_LAYERS[model])
            assert np.array_equal(ys, y_tr)
            variants = {
                "gold": best_head(Xs, y_tr, K).probs(Xs_te),
                "teacher-argmax": best_head(Xs, t_oof.argmax(1), K).probs(Xs_te),
                "teacher-soft": fit_soft_head(Xs, t_oof).probs(Xs_te),
                "gold+teacher-soft": fit_soft_head(Xs, 0.5 * np.eye(K)[y_tr] + 0.5 * t_oof).probs(Xs_te),
            }
            for v, p in variants.items():
                rows.setdefault(f"{tag}:{v}", []).append((p, y_te))
                per_q[name][f"{tag}:{v}"] = float(np.mean(p.argmax(1) == y_te))
        print(f"   {name:46s} teacher {per_q[name]['teacher']:.2f}  "
              + "  ".join(f"{m.split('/')[-1]} gold {per_q[name][m.split('/')[-1] + ':gold']:.2f} "
                         f"soft {per_q[name][m.split('/')[-1] + ':gold+teacher-soft']:.2f}" for m, _ in students),
              flush=True)
    result = {"teacher": args.teacher, "teacher_block": args.teacher_block, "students": students, "questions": per_q,
              "date": dt.datetime.now().isoformat(), "env": environment()}
    table = [f"### Soft-target distillation on the same 300 states (teacher {args.teacher} head at block "
             f"{args.teacher_block}, out-of-fold probabilities)", "",
             "| student | gold labels | teacher argmax | teacher soft | gold + teacher soft | teacher itself |",
             "|---|---|---|---|---|---|"]
    result["rows"] = {}
    for model, block in students:
        tag = model.split("/")[-1]
        ci = with_cis({k: v for k, v in rows.items() if k.startswith(tag + ":") or k == "teacher"}, f"{tag}:gold")
        result["rows"][tag] = ci

        def cell(v: str) -> str:
            r = ci[f"{tag}:{v}"]
            d = r[f"diff_vs_{tag}:gold_ci95"]
            return f"{r['acc']:.3f} [{d[0]:+.3f}, {d[1]:+.3f}]"

        table.append(f"| {tag} @{block} | {ci[f'{tag}:gold']['acc']:.3f} | {cell('teacher-argmax')} | "
                     f"{cell('teacher-soft')} | {cell('gold+teacher-soft')} | {ci['teacher']['acc']:.3f} |")
    print("\n" + "\n".join(table))
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.teacher.replace('/', '__')}.distill_heads.soft.json")
    json.dump(result, open(path, "w"), indent=1)
    print("wrote", path)


# ---------------------------------------------------------------- synthetic states
def load_synth(root: str) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        d = json.load(open(path))
        out[d["workflow"]] = d["states"]
    return out


def label_path(root: str, teacher: str, name: str, split: str) -> str:
    return os.path.join(root, "synth_labels", teacher.replace("/", "__"), f"{name}.{split}.npz")


def feat_path(root: str, student: str, name: str, split: str) -> str:
    return os.path.join(root, "synth_features", student.replace("/", "__"), f"{name}.{split}.npz")


def label(args) -> None:
    """Teacher L2 probabilities on the synthetic states and on the test states."""
    from anyjev import Decider
    from anyjev.backends.hf import HFBackend

    synth = load_synth(args.synth)
    be = HFBackend(args.teacher, batch_size=args.batch_size)
    dec = Decider(be, adapt=False)
    n = dec.load_artifacts(args.artifact or os.path.join("anyjev-heads", f"{args.teacher.replace('/', '__')}.json"))
    print(f"teacher {args.teacher}: {n} heads loaded; {sum(len(v) for v in synth.values())} synthetic states")
    for name, q, workflow, train_items, test_items in typed_questions(args.calib_cases):
        s_te, y_te, _ = decisions_to_arrays(test_items)
        for split, states, gold in (("synth", synth[workflow], None), ("test", s_te, y_te)):
            path = label_path(args.out, args.teacher, name, split)
            if os.path.exists(path):
                continue
            decs = dec.decide_batch(states, q, level="L2")
            probs = np.stack([d.probs for d in decs]).astype(np.float32)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            extra = {"y_gold": np.asarray(gold, dtype=int)} if gold is not None else {}
            np.savez(path, probs=probs, K=q.k, block=decs[0].diagnostics["exit_layer"], **extra)
            msg = f"   {name:46s} {split:5s} {len(states):5d} states"
            if gold is not None:
                msg += f"  teacher test acc {np.mean(probs.argmax(1) == np.asarray(gold)):.3f}"
            print(msg, flush=True)


def extract(args) -> None:
    """Student hidden states at its fixed block on the synthetic states, random listing order."""
    from anyjev import Decider
    from anyjev.backends.hf import HFBackend
    from anyjev.state import render_state

    synth = load_synth(args.synth)
    be = HFBackend(args.student, batch_size=args.batch_size)
    dec = Decider(be)
    for name, q, workflow, train_items, test_items in typed_questions(args.calib_cases):
        path = feat_path(args.out, args.student, name, "synth")
        if os.path.exists(path):
            continue
        states = synth[workflow]
        rng = np.random.RandomState(args.seed)
        perms = None if q.ordered else [list(rng.permutation(q.k)) for _ in states]
        H = dec._features(q, [render_state(s) for s in states], [args.block], perms)[:, 0]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, H=H.astype(np.float16), block=args.block, K=q.k)
        print(f"   {name:46s} {len(states):5d} synthetic states at block {args.block}", flush=True)


def fit(args) -> None:
    students = parse_students(args.students)
    budgets = [int(x) for x in args.budgets.split(",")]
    names = [n for n, *_ in typed_questions(args.calib_cases)]
    rows: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    per_q: Dict[str, Dict[str, float]] = {}
    for name in sorted(names):
        tl = np.load(label_path(args.out, args.teacher, name, "synth"))
        tt = np.load(label_path(args.out, args.teacher, name, "test"))
        T_syn, K = tl["probs"].astype(np.float64), int(tl["K"])
        y_te = tt["y_gold"]
        rows.setdefault("teacher", []).append((tt["probs"], y_te))
        per_q[name] = {"teacher": float(np.mean(tt["probs"].argmax(1) == y_te))}
        for model, block in students:
            tag = model.split("/")[-1]
            Xg, y_tr, _ = cache_features(args.exit_root, model, name, "train", "rand", block, N_LAYERS[model])
            X_te, _, _ = cache_features(args.exit_root, model, name, "test", "id", block, N_LAYERS[model])
            X_syn = np.load(feat_path(args.out, model, name, "synth"))["H"].astype(np.float32)
            n_syn = min(len(X_syn), len(T_syn))
            X_syn, T = X_syn[:n_syn], T_syn[:n_syn]
            variants: Dict[str, np.ndarray] = {"gold300": best_head(Xg, y_tr, K).probs(X_te)}
            for n in budgets:
                n = min(n, n_syn)
                Xn, Tn = X_syn[:n], T[:n]
                variants[f"synth{n}-hard"] = best_head(Xn, Tn.argmax(1), K).probs(X_te)
                variants[f"synth{n}-soft"] = fit_soft_head(Xn, Tn).probs(X_te)
                Xb = np.concatenate([Xg, Xn])
                variants[f"gold300+synth{n}-hard"] = best_head(Xb, np.concatenate([y_tr, Tn.argmax(1)]), K).probs(X_te)
                Tb = np.concatenate([np.eye(K)[y_tr], Tn])
                variants[f"gold300+synth{n}-soft"] = fit_soft_head(Xb, Tb).probs(X_te)
            for v, p in variants.items():
                rows.setdefault(f"{tag}:{v}", []).append((p, y_te))
                per_q[name][f"{tag}:{v}"] = float(np.mean(p.argmax(1) == y_te))
        print(f"   {name:46s} teacher {per_q[name]['teacher']:.2f}  "
              + "  ".join(f"{m.split('/')[-1]} gold {per_q[name][m.split('/')[-1] + ':gold300']:.2f} "
                         f"+synth {per_q[name][m.split('/')[-1] + f':gold300+synth{min(budgets[-1], n_syn)}-hard']:.2f}"
                         for m, _ in students), flush=True)
    result = {"teacher": args.teacher, "students": students, "budgets": budgets, "questions": per_q,
              "date": dt.datetime.now().isoformat(), "env": environment(), "rows": {}}
    table = [f"### Closed-form distillation through synthetic states (teacher {args.teacher} shipped heads)", "",
             "| student | gold 300 | " + " | ".join(f"synth {n} hard / soft" for n in budgets) + " | "
             + " | ".join(f"gold 300 + synth {n} hard / soft" for n in budgets) + " | teacher itself |",
             "|---|---|" + "---|" * (2 * len(budgets)) + "---|"]
    for model, block in students:
        tag = model.split("/")[-1]
        ci = with_cis({k: v for k, v in rows.items() if k.startswith(tag + ":") or k == "teacher"}, f"{tag}:gold300")
        result["rows"][tag] = ci

        def cell(v: str) -> str:
            r = ci.get(f"{tag}:{v}")
            if r is None:
                return ""
            d = r.get(f"diff_vs_{tag}:gold300_ci95")
            return f"{r['acc']:.3f} [{d[0]:+.3f}, {d[1]:+.3f}]" if d else f"{r['acc']:.3f}"

        cells = [cell(f"synth{n}-hard") + " / " + cell(f"synth{n}-soft") for n in budgets]
        cells += [cell(f"gold300+synth{n}-hard") + " / " + cell(f"gold300+synth{n}-soft") for n in budgets]
        table.append(f"| {tag} @{block} | {ci[f'{tag}:gold300']['acc']:.3f} | " + " | ".join(cells)
                     + f" | {ci['teacher']['acc']:.3f} |")
    print("\n" + "\n".join(table))
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.teacher.replace('/', '__')}.distill_heads.fit.json")
    json.dump(result, open(path, "w"), indent=1)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["soft", "label", "extract", "fit"])
    ap.add_argument("--teacher", default="Qwen/Qwen3-32B")
    ap.add_argument("--teacher-block", type=int, default=52)
    ap.add_argument("--artifact", default=None, help="teacher artifact path (default anyjev-heads/<teacher>.json)")
    ap.add_argument("--students", default="Qwen/Qwen3-1.7B:18,Qwen/Qwen3-4B:24,Qwen/Qwen3-8B:24")
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--block", type=int, default=18)
    ap.add_argument("--synth", default="bench/results_distill/synth")
    ap.add_argument("--budgets", default="300,1200")
    ap.add_argument("--calib-cases", type=int, default=300)
    ap.add_argument("--exit-root", default="bench/results_exit")
    ap.add_argument("--out", default="bench/results_distill")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"soft": soft, "label": label, "extract": extract, "fit": fit}[args.stage](args)


if __name__ == "__main__":
    main()
