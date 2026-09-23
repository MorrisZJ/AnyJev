"""Does a per-question head survive a rewording of its question? (E13, wording robustness)

    python -m bench.paraphrase_study extract --model Qwen/Qwen3-8B --layers 22,24,28,36
    python -m bench.paraphrase_study fit --model Qwen/Qwen3-8B --block 24

Variants per typed question (bench/tasks/typed_paraphrases.json, hand-written): the original
wording, three rewordings of the question text (w1 light, w2 restructured, w3 reframed) and one
rewording of the option descriptions or of the Yes-means / No-means lines (o1); option keys,
count and order never change, so the labels stay valid and a head's [hidden, K] shape fits.

extract: for every question and variant, the last-position hidden state of the 300 train and 100
test states at the requested blocks under the canonical listing order (one prompt per state, the
block loop through `Decider._features`, i.e. exactly the deployed L2 path), plus raw and L0
probabilities of the test states under that variant (a fresh Decider per variant). Cached as
bench/results_paraphrase/features/<model>/<question>.<variant>.<split>.npz.

fit: CPU replay at one block. Per question:
  orig      head (lda / ridge by CV) fit on the ORIGINAL wording's train states, applied to every
            variant's test states -> accuracy / ECE, and the share of test states whose answer
            differs from the original wording's answer (flip);
  own       head refit on the variant's own train states (what re-collecting labels would give);
  mix       head fit on 300 rows where every train state appears under one wording drawn at
            random from the OTHER variants (same labels, wordings augmented: leave-one-wording-out);
  all       the same with every train state under every other wording (n_variants - 1 x rows);
  ta        the ORIGINAL head with its feature standardisation (mean / scale) re-estimated on the
            variant's 300 train states WITHOUT their labels: label-free test-time adaptation to
            the new wording (a rewording is treated as a shift of the feature distribution);
  tt        the same re-estimated on the 100 test states themselves (transductive, still label-free);
  ta@n      `ta` with only n unlabelled states (--ta-budgets), mean of 3 draws; ta-mean@n re-estimates
            the mean only and keeps the original scale;
  raw / L0  the model's own zero-label answers under the variant.
Pooled over the 20 questions with bootstrap CIs paired against the original wording. Output:
bench/results_paraphrase/<date>/<model>.paraphrase.json and a Markdown table.

order: the same question with its options listed in REVERSED order, replayed from the block-loop
caches of bench.extract_pools (train under canonical `id` and random `rand` listing orders, test under
`id` and `rev`): a head fit on one listing order applied to the reversed listing, with and without
label-free recentring, against a head fit on random listing orders.
    python -m bench.paraphrase_study order --model Qwen/Qwen3-8B --block 24 --n-layers 36
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from anyjev import Decider, Question
from anyjev.heads import _standardise, fit_head
from bench import metrics
from bench.pools import decisions_to_arrays, load_pool, pool_path, typed_questions
from bench.run import environment

PARAPHRASES = os.path.join(os.path.dirname(__file__), "tasks", "typed_paraphrases.json")
VARIANTS = ("orig", "w1", "w2", "w3", "o1")


def load_paraphrases() -> Dict[str, Dict[str, Dict[str, Any]]]:
    d = json.load(open(PARAPHRASES))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def variant_question(q: Question, spec: Dict[str, Any]) -> Question:
    """The same question with the text and/or the option descriptions replaced."""
    kw: Dict[str, Any] = {}
    if spec.get("text"):
        kw["text"] = spec["text"]
    if spec.get("options"):
        opts = tuple(str(o) for o in spec["options"])
        if len(opts) != q.k:
            raise ValueError(f"variant changes K for {q.name}: {len(opts)} != {q.k}")
        kw["options"] = opts
    return dataclasses.replace(q, **kw)


def feature_path(root: str, model: str, name: str, variant: str, split: str) -> str:
    return os.path.join(root, "features", model.replace("/", "__"), f"{name}.{variant}.{split}.npz")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- extract
def extract(args) -> None:
    from anyjev.backends.hf import HFBackend

    be = HFBackend(args.model, batch_size=args.batch_size)
    layers = [int(x) for x in args.layers.split(",")]
    paras = load_paraphrases()
    for name, q, workflow, train_items, test_items in typed_questions(args.calib_cases):
        s_tr, y_tr, _ = decisions_to_arrays(train_items)
        s_te, y_te, _ = decisions_to_arrays(test_items)
        specs = {"orig": {}}
        specs.update(paras.get(name, {}))
        for variant, spec in specs.items():
            if variant not in VARIANTS:
                continue
            qv = variant_question(q, spec) if spec else q
            done = all(os.path.exists(feature_path(args.out, args.model, name, variant, sp))
                       for sp in ("train", "test"))
            if done:
                continue
            dec = Decider(be)                     # fresh: the batch prior of one variant never leaks into another
            from anyjev.state import render_state
            for split, states, labels in (("train", s_tr, y_tr), ("test", s_te, y_te)):
                feats = dec._features(qv, [render_state(s) for s in states], layers)
                extra: Dict[str, Any] = {}
                if split == "test":
                    raw = np.stack([d.probs for d in dec.decide_batch(states, qv, level="raw")])
                    l0 = np.stack([d.probs for d in dec.decide_batch(states, qv, level="L0")])
                    extra = {"raw": raw.astype(np.float32), "l0": l0.astype(np.float32)}
                path = feature_path(args.out, args.model, name, variant, split)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez(path, H=feats.astype(np.float16), y=np.asarray(labels, dtype=int), layers=np.asarray(layers),
                         K=q.k, kind=q.kind, text=qv.text, options=np.asarray(list(qv.options)), **extra)
            print(f"   {name:48s} {variant:5s} done", flush=True)
    print("features cached under", os.path.join(args.out, "features", args.model.replace("/", "__")))


# ---------------------------------------------------------------- fit
def best_head(X: np.ndarray, y: np.ndarray, K: int, kinds=("lda", "ridge")):
    best = None
    for kind in kinds:
        try:
            h = fit_head(X, y, K, kind=kind)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or h.cv["oof_nll"] < best.cv["oof_nll"]:
            best = h
    return best


def pooled(rows: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    conf = np.concatenate([p.max(1) for p, _ in rows])
    corr = np.concatenate([(p.argmax(1) == y) for p, y in rows]).astype(float)
    nll = np.concatenate([-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)) for p, y in rows])
    return {"acc": float(corr.mean()), "ece": metrics.ece(np.stack([conf, 1 - conf], 1), (1 - corr).astype(int)),
            "nll": float(nll.mean()), "n": int(len(corr))}


def fit(args) -> None:
    paras = load_paraphrases()
    names = sorted(paras)
    first = np.load(feature_path(args.out, args.model, names[0], "orig", "test"))
    layers = [int(x) for x in first["layers"]]
    li = layers.index(args.block)
    rng = np.random.RandomState(args.seed)
    budgets = [int(x) for x in args.ta_budgets.split(",") if x]
    per_q: Dict[str, Dict[str, Any]] = {}
    rows: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    flips: Dict[str, List[float]] = {}

    def add(method: str, p: np.ndarray, y: np.ndarray, p_ref: np.ndarray = None) -> None:
        rows.setdefault(method, []).append((p, y))
        if p_ref is not None:
            flips.setdefault(method, []).append(float(np.mean(p.argmax(1) != p_ref.argmax(1))))

    for name in names:
        data = {}
        for v in VARIANTS:
            for sp in ("train", "test"):
                path = feature_path(args.out, args.model, name, v, sp)
                if not os.path.exists(path):
                    break
                data[(v, sp)] = np.load(path)
        if len(data) < 2 * len(VARIANTS):
            print("missing variants for", name, "-> skipped")
            continue
        K = int(data[("orig", "test")]["K"])
        y_tr = data[("orig", "train")]["y"]
        y_te = data[("orig", "test")]["y"]
        X = {v: data[(v, "train")]["H"][:, li].astype(np.float32) for v in VARIANTS}
        Xt = {v: data[(v, "test")]["H"][:, li].astype(np.float32) for v in VARIANTS}
        h_orig = best_head(X["orig"], y_tr, K)
        p_ref = h_orig.probs(Xt["orig"])
        q_rec: Dict[str, Any] = {"K": K, "kind": str(data[("orig", "test")]["kind"]), "acc": {}}
        m0, s0 = _standardise(Xt["orig"])
        p_tt0 = dataclasses.replace(h_orig, mean=m0, scale=s0).probs(Xt["orig"])
        add("tt-head@orig", p_tt0, y_te, p_ref)          # control: recentring when nothing changed
        q_rec["acc"]["tt-head@orig"] = float(np.mean(p_tt0.argmax(1) == y_te))
        for v in VARIANTS:
            p = h_orig.probs(Xt[v])
            add(f"orig-head@{v}", p, y_te, p_ref)
            q_rec["acc"][f"orig-head@{v}"] = float(np.mean(p.argmax(1) == y_te))
            add(f"raw@{v}", data[(v, "test")]["raw"], y_te, data[("orig", "test")]["raw"])
            add(f"L0@{v}", data[(v, "test")]["l0"], y_te, data[("orig", "test")]["l0"])
            q_rec["acc"][f"L0@{v}"] = float(np.mean(data[(v, "test")]["l0"].argmax(1) == y_te))
            if v != "orig":
                h_own = best_head(X[v], y_tr, K)
                p_own = h_own.probs(Xt[v])
                add(f"own-head@{v}", p_own, y_te, p_ref)
                q_rec["acc"][f"own-head@{v}"] = float(np.mean(p_own.argmax(1) == y_te))
                others = [u for u in VARIANTS if u != v]
                pick = rng.randint(0, len(others), len(y_tr))
                X_mix = np.stack([X[others[j]][i] for i, j in enumerate(pick)])
                h_mix = best_head(X_mix, y_tr, K)
                p_mix = h_mix.probs(Xt[v])
                add(f"mix-head@{v}", p_mix, y_te, p_ref)
                q_rec["acc"][f"mix-head@{v}"] = float(np.mean(p_mix.argmax(1) == y_te))
                X_all = np.concatenate([X[u] for u in others])
                y_all = np.concatenate([y_tr for _ in others])
                h_all = best_head(X_all, y_all, K)
                p_all = h_all.probs(Xt[v])
                add(f"all-head@{v}", p_all, y_te, p_ref)
                q_rec["acc"][f"all-head@{v}"] = float(np.mean(p_all.argmax(1) == y_te))
                for tag, Xs in (("ta", X[v]), ("tt", Xt[v])):
                    m_, s_ = _standardise(Xs)
                    h_ad = dataclasses.replace(h_orig, mean=m_, scale=s_)
                    p_ad = h_ad.probs(Xt[v])
                    add(f"{tag}-head@{v}", p_ad, y_te, p_ref)
                    q_rec["acc"][f"{tag}-head@{v}"] = float(np.mean(p_ad.argmax(1) == y_te))
                for n_u in budgets:
                    for draw in range(3):
                        idx = np.random.RandomState(1000 * draw + n_u).permutation(len(X[v]))[:n_u]
                        m_, s_ = _standardise(X[v][idx])
                        p_ad = dataclasses.replace(h_orig, mean=m_, scale=s_).probs(Xt[v])
                        add(f"ta@{n_u}-head@{v}", p_ad, y_te, p_ref)
                        p_m = dataclasses.replace(h_orig, mean=m_).probs(Xt[v])
                        add(f"ta-mean@{n_u}-head@{v}", p_m, y_te, p_ref)
        per_q[name] = q_rec
        print(f"   {name:46s} orig {q_rec['acc']['orig-head@orig']:.2f} | reworded: "
              + " ".join(f"{v}:{q_rec['acc'][f'orig-head@{v}']:.2f}/{q_rec['acc'][f'own-head@{v}']:.2f}"
                         f"/{q_rec['acc'][f'mix-head@{v}']:.2f}/{q_rec['acc'][f'ta-head@{v}']:.2f}"
                         for v in VARIANTS[1:])
              + "  (orig-head/own/mix/ta)", flush=True)

    # pooled, with bootstrap CIs paired against the original wording's head on the original wording
    n_items = sum(len(y) for _, y in rows["orig-head@orig"])
    boot = np.random.RandomState(0).randint(0, n_items, (args.n_boot, n_items))
    c_ref = np.concatenate([(p.argmax(1) == y).astype(float) for p, y in rows["orig-head@orig"]])
    result: Dict[str, Any] = {"model": args.model, "block": args.block, "variants": list(VARIANTS), "rows": {},
                              "questions": per_q, "date": dt.datetime.now().isoformat(), "env": environment()}
    for m, items in rows.items():
        c = np.concatenate([(p.argmax(1) == y).astype(float) for p, y in items])
        rec = pooled(items)
        rec["acc_ci95"] = [float(np.percentile(c[boot].mean(1), 2.5)), float(np.percentile(c[boot].mean(1), 97.5))]
        d = (c[boot] - c_ref[boot]).mean(1)
        rec["diff_vs_orig_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        if m in flips:
            rec["flip_vs_orig"] = float(np.mean(flips[m]))
        result["rows"][m] = rec

    def cell(m: str) -> str:
        r = result["rows"].get(m)
        return f"{r['acc']:.3f}" if r else ""

    lines = [f"### {args.model}, block {args.block}: per-question heads under rewordings "
             "(typed, 2000 test decisions)", "",
             "| variant | raw | L0 | original-wording head | + recentred on unlabelled train | + recentred on test "
             "| own-wording head | mixed-wording head (300 labels) | all-wordings head "
             "| orig-head flip vs original | L0 flip vs original |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r = result["rows"][f"orig-head@{v}"]
        lines.append(f"| {v} | {cell(f'raw@{v}')} | {cell(f'L0@{v}')} | **{r['acc']:.3f}** "
                     f"[{r['diff_vs_orig_ci95'][0]:+.3f}, {r['diff_vs_orig_ci95'][1]:+.3f}] | "
                     f"{cell(f'ta-head@{v}')} | {cell(f'tt-head@{v}')} | "
                     f"{cell(f'own-head@{v}')} | {cell(f'mix-head@{v}')} | {cell(f'all-head@{v}')} | "
                     f"{r.get('flip_vs_orig', 0.0):.3f} | {result['rows'][f'L0@{v}'].get('flip_vs_orig', 0.0):.3f} |")
    if budgets:
        lines += ["", "Label-free adaptation with n unlabelled states under the new wording (pooled over the four "
                  "rewordings; mean of 3 draws):", "",
                  "| n unlabelled | mean + scale re-estimated | mean only |", "|---|---|---|"]
        for n_u in budgets:
            acc_ms = np.mean([result["rows"][f"ta@{n_u}-head@{v}"]["acc"] for v in VARIANTS[1:]])
            acc_m = np.mean([result["rows"][f"ta-mean@{n_u}-head@{v}"]["acc"] for v in VARIANTS[1:]])
            lines.append(f"| {n_u} | {acc_ms:.3f} | {acc_m:.3f} |")
        full = np.mean([result["rows"][f"ta-head@{v}"]["acc"] for v in VARIANTS[1:]])
        none = np.mean([result["rows"][f"orig-head@{v}"]["acc"] for v in VARIANTS[1:]])
        own = np.mean([result["rows"][f"own-head@{v}"]["acc"] for v in VARIANTS[1:]])
        lines.append(f"| 300 (all) | {full:.3f} | |")
        lines.append(f"| none (original head as is) | {none:.3f} | |")
        lines.append(f"| refit with 300 labels (reference) | {own:.3f} | |")
    print("\n" + "\n".join(lines))
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.model.replace('/', '__')}.paraphrase.b{args.block}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


def order(args) -> None:
    """Listing-order robustness from the block-loop caches (see the module docstring)."""
    paras = load_paraphrases()
    names = sorted(paras)
    rows: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    flips: Dict[str, List[float]] = {}
    li = None
    for name in names:
        pools = {}
        for split, ord_ in (("train", "id"), ("train", "rand"), ("test", "id"), ("test", "rev")):
            path = pool_path(args.exit_root, args.model, name, split, ord_, "last")
            if not os.path.exists(path):
                break
            pools[(split, ord_)] = load_pool(path)
        if len(pools) < 4:
            print("missing exit cache for", name, "-> skipped")
            continue
        if li is None:
            first = np.load(pool_path(args.exit_root, args.model, name, "test", "id", "last"))
            layers = [int(x) for x in first["layers"]]
            blocks = [(args.n_layers + 1 + i) if i < 0 else i for i in layers]
            li = blocks.index(args.block)
        K = pools[("test", "id")].K
        X = {k: v.H[:, li].astype(np.float32) for k, v in pools.items()}
        y_tr, y_te = pools[("train", "id")].y, pools[("test", "id")].y
        for fit_order in ("id", "rand"):
            h = best_head(X[("train", fit_order)], y_tr, K)
            p_id = h.probs(X[("test", "id")])
            p_rev = h.probs(X[("test", "rev")])
            m_, s_ = _standardise(X[("test", "rev")])
            p_rev_tt = dataclasses.replace(h, mean=m_, scale=s_).probs(X[("test", "rev")])
            for tag, p in ((f"fit-{fit_order}@canonical", p_id), (f"fit-{fit_order}@reversed", p_rev),
                           (f"fit-{fit_order}@reversed+tt", p_rev_tt)):
                rows.setdefault(tag, []).append((p, y_te))
                flips.setdefault(tag, []).append(float(np.mean(p.argmax(1) != p_id.argmax(1))))
        print(f"   {name:46s} done", flush=True)
    result = {"model": args.model, "block": args.block, "rows": {}, "date": dt.datetime.now().isoformat(),
              "env": environment()}
    lines = [f"### {args.model}, block {args.block}: the same question with the options listed in reverse", "",
             "| head fit on | canonical listing | reversed listing | reversed + label-free recentring "
             "| flip reversed vs canonical | flip after recentring |", "|---|---|---|---|---|---|"]
    for m, items in rows.items():
        result["rows"][m] = {**pooled(items), "flip_vs_canonical": float(np.mean(flips[m]))}
    for fo, label in (("id", "the canonical order only"), ("rand", "a random order per calibration state")):
        r = {k: result["rows"][f"fit-{fo}@{k}"] for k in ("canonical", "reversed", "reversed+tt")}
        lines.append(f"| {label} | {r['canonical']['acc']:.3f} | {r['reversed']['acc']:.3f} | "
                     f"{r['reversed+tt']['acc']:.3f} | {r['reversed']['flip_vs_canonical']:.3f} | "
                     f"{r['reversed+tt']['flip_vs_canonical']:.3f} |")
    print("\n" + "\n".join(lines))
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.model.replace('/', '__')}.order.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["extract", "fit", "order"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="bench/results_paraphrase")
    ap.add_argument("--calib-cases", type=int, default=300)
    ap.add_argument("--layers", default="22,24,28,36")
    ap.add_argument("--block", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--ta-budgets", default="10,30,100", help="unlabelled-state budgets for the adaptation curve")
    ap.add_argument("--exit-root", default="bench/results_exit", help="order stage: the exit-study cache root")
    ap.add_argument("--n-layers", type=int, default=36, help="order stage: blocks of the model (cache layer indices)")
    args = ap.parse_args(argv)
    {"extract": extract, "fit": fit, "order": order}[args.stage](args)


if __name__ == "__main__":
    main()
