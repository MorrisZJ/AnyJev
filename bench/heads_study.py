"""Closed-form heads on hidden states against the logit readout, on the bench's own protocol.

    python -m bench.heads_study --model Qwen/Qwen3-8B --tasks banking20,newsgroups,injection --n 300 --calib 200
    python -m bench.heads_study --model Qwen/Qwen3-8B --typed --calib-cases 200

One GPU pass per task extracts, for every state and every listing order (the K cyclic shifts
of the canonical order and the K shifts of the reversed order), the last-position hidden state
at a few layers and the label log-probs. Four aggregates are cached per state and layer: the
canonical order (`id`), the reversed order (`rev`), and the mean over the shifts of each
(`avg`, `avg_rev`, order-invariant by construction). Every head then fits on CPU in seconds:

    raw            softmax of the label logits, canonical order
    L1             raw with a temperature fit on the calibration split (the current L1)
    L0-perm        log-mean over the cyclic shifts (the permutation half of L0), no prior
    L0-perm+T      L0-perm with a temperature
    head:<kind>    closed-form head on `id` features (flip measured against `rev` features)
    head:<kind>/avg  the same head on shift-averaged features (flip against `avg_rev`; K prefills)
    head:<kind>/rand the head fit on one random listing order per state, tested on `id` vs `rev`
                     (one prefill at inference, every order seen during calibration)

Rows tagged @self were fit on the model's own L0-perm answers (no labels at all), rows tagged
@think on the model's thinking-mode answers (its own slow path as the teacher); untagged rows
on the dataset's labels. Flip is the share of test states whose answer changes when the option
list is reversed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from anyjev.calibrate.permute import marginalize
from anyjev.calibrate.posthoc import TemperatureScaler
from anyjev.heads import KINDS, fit_head
from anyjev.question import Question
from anyjev.readout import DEFAULT_SYSTEM, build_prompt, label_ids_for_perm, render_chat, resolve_labels
from anyjev.state import render_state
from bench import metrics
from bench.run import environment
from bench.tasks import get_task

DEFAULT_LAYERS = "-1,-5,-9,-13,-17"


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def listing_orders(q: Question) -> Tuple[List[List[int]], int]:
    """(perms, n_shifts): the K cyclic shifts of the canonical order followed by the K shifts of
    the reversed order (deduplicated); ordered (score) questions keep the canonical order only.
    perm[j] = index of the option shown at position j."""
    K = q.k
    if q.ordered:
        return [list(range(K))], 1
    shifts = [[(j + s) % K for j in range(K)] for s in range(K)]
    rev = list(reversed(range(K)))
    rshifts = [[rev[(j + s) % K] for j in range(K)] for s in range(K)]
    perms, seen = [], set()
    for p in shifts + rshifts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            perms.append(p)
    return perms, K


def extract(backend, q: Question, states: Sequence[Any], layers: Sequence[int]) -> Dict[str, np.ndarray]:
    """Hidden-state aggregates and option-space log-probs for every state under every listing order."""
    tok = backend.tokenizer
    labels, ids = resolve_labels(tok, q)
    perms, n_shift = listing_orders(q)
    texts = [render_state(s) for s in states]
    prompts, token_ids = [], []
    for st in texts:
        for perm in perms:
            prompts.append(render_chat(tok, build_prompt(st, q, perm, DEFAULT_SYSTEM, labels)))
            token_ids.append(label_ids_for_perm(q, ids, perm))
    feats, lps = backend.hidden_states(prompts, layers, token_ids)
    N, P, K = len(states), len(perms), q.k
    feats = feats.reshape(N, P, len(layers), -1)
    lp_pos = np.stack(lps).reshape(N, P, K)                        # position space
    lp_opt = np.zeros_like(lp_pos)                                  # option space
    for pi, perm in enumerate(perms):
        for j, opt in enumerate(perm):
            lp_opt[:, pi, opt] = lp_pos[:, pi, j]
    rev_i = perms.index(list(reversed(range(K)))) if not q.ordered else 0
    # one random listing order per state: a head fit on these sees every order during
    # calibration and needs a single prefill at inference
    rand_i = np.random.RandomState(0).randint(0, P, size=N)
    out = {"id": feats[:, 0], "rev": feats[:, rev_i], "avg": feats[:, :n_shift].mean(axis=1),
           "avg_rev": feats[:, n_shift:].mean(axis=1) if P > n_shift else feats[:, rev_i],
           "rand": feats[np.arange(N), rand_i], "rand_perm": rand_i,
           "lp_pos": lp_pos, "lp_opt": lp_opt, "perms": np.asarray(perms), "n_shift": n_shift}
    return {k: (v.astype(np.float16) if k in ("id", "rev", "avg", "avg_rev", "rand") else v) for k, v in out.items()}


def cache_path(out: str, model: str, name: str) -> str:
    return os.path.join(out, "features", model.replace("/", "__"), name + ".npz")


def get_features(backend_factory, model: str, out: str, name: str, q: Question, states, layers
                 ) -> Dict[str, np.ndarray]:
    path = cache_path(out, model, name)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if "rand" in z.files:
            return {k: z[k] for k in z.files}
    backend = backend_factory()
    t0 = time.time()
    d = extract(backend, q, states, layers)
    d["seconds"] = np.array(time.time() - t0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **d)
    return d


# ---------------------------------------------------------------- methods
def logit_methods(d_cal: Dict[str, np.ndarray], d_test: Dict[str, np.ndarray], y_cal: np.ndarray,
                  perms: List[List[int]], n_shift: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Returns name -> (test probs [N, K], test probs under the reversed listing [N, K])."""
    K = d_test["lp_opt"].shape[-1]
    rev_i = perms.index(list(reversed(range(K)))) if K > 1 and len(perms) > 1 else 0

    def raw(d):
        return _softmax(d["lp_opt"][:, 0]), _softmax(d["lp_opt"][:, rev_i])

    def l0perm(d):
        P_pos = _softmax(d["lp_pos"])                                # [N, P, K] position space
        a = np.stack([marginalize(P_pos[i, :n_shift], perms[:n_shift], "logmean") for i in range(len(P_pos))])
        if len(perms) > n_shift:
            b = np.stack([marginalize(P_pos[i, n_shift:], perms[n_shift:], "logmean") for i in range(len(P_pos))])
        else:
            b = a
        return a, b

    out = {}
    label_sets = y_cal if isinstance(y_cal, dict) else {"gold": y_cal}
    for name, fn in (("raw", raw), ("L0-perm", l0perm)):
        cal, _ = fn(d_cal)
        test, test_rev = fn(d_test)
        out[name] = (test, test_rev)
        for src, y in label_sets.items():
            keep = y >= 0
            T = TemperatureScaler.fit(cal[keep], y[keep])
            tag = "" if src == "gold" else f"@{src}"
            out[(name + "+T" if name != "raw" else "L1") + tag] = (T.apply(test), T.apply(test_rev))
    return out


def self_labels(d_cal: Dict[str, np.ndarray], perms: List[List[int]], n_shift: int) -> np.ndarray:
    """The model's own answers under L0-perm: pseudo-labels that cost no human and no generation."""
    P_pos = _softmax(d_cal["lp_pos"])
    return np.stack([marginalize(P_pos[i, :n_shift], perms[:n_shift], "logmean") for i in range(len(P_pos))]).argmax(1)


def thinking_labels(backend, q: Question, states: Sequence[Any], max_new_tokens: int = 1024) -> np.ndarray:
    """The model's answers in thinking mode (reasoning generated, then the label): the slow path
    of the same model, used as the teacher for a head that runs on the fast path. -1 = no
    parsable answer."""
    import torch

    tok = backend.tokenizer
    labels, _ = resolve_labels(tok, q)
    perm = list(range(q.k))
    prompts = []
    for st in states:
        spec = build_prompt(render_state(st), q, perm, DEFAULT_SYSTEM.replace(
            "Reply with the answer label only: no words, no punctuation, no explanation.",
            "Think it through, then end with the answer label alone on the last line."), labels)
        msgs = [{"role": "system", "content": spec.system}, {"role": "user", "content": spec.user}]
        try:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=True))
        except TypeError:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    out = np.full(len(states), -1, dtype=int)
    texts = []
    label_index = {lab: i for i, lab in enumerate(labels)}
    for start in range(0, len(prompts), backend.batch_size):
        chunk = prompts[start:start + backend.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(backend.model.device)
        with torch.no_grad():
            gen = backend.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
        for i, row in enumerate(gen):
            text = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            texts.append(text)
            answer = text.split("</think>")[-1].strip()
            for line in reversed(answer.splitlines()):
                token = line.strip().strip(".:)*` ").split(" ")[0] if line.strip() else ""
                if token in label_index:
                    out[start + i] = label_index[token]
                    break
    return out, texts


def head_methods(d_cal, d_test, label_sets: Dict[str, np.ndarray], K: int, kinds: Sequence[str]
                 ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    out, chosen = {}, {}
    for src, y in label_sets.items():
        keep = y >= 0
        tag = "" if src == "gold" else f"@{src}"
        # (features the head is fit on, test features, test features under the reversed listing)
        for feat, feat_test, feat_rev in (("id", "id", "rev"), ("avg", "avg", "avg_rev"), ("rand", "id", "rev")):
            if feat not in d_cal:
                continue
            Xc = d_cal[feat][keep].astype(np.float32)
            for kind in kinds:
                name = f"head:{kind}" + ("" if feat == "id" else f"/{feat}") + tag
                try:
                    h = fit_head(Xc, y[keep], K, kind=kind)
                except (ValueError, np.linalg.LinAlgError) as e:
                    chosen[name] = {"error": str(e)}
                    continue
                probs = h.probs(d_test[feat_test][:, h.layer].astype(np.float32))
                probs_rev = h.probs(d_test[feat_rev][:, h.layer].astype(np.float32))
                out[name] = (probs, probs_rev)
                chosen[name] = {"layer": h.layer, "params": h.params, "temperature": h.temperature,
                                "n_labels": int(keep.sum()), **h.cv}
    return out, chosen


def label_sets_for(args, backend_factory, name: str, q: Question, states, d_cal, y_gold: np.ndarray,
                   perms, n_shift) -> Dict[str, np.ndarray]:
    sets: Dict[str, np.ndarray] = {}
    for src in args.labels:
        if src == "gold":
            sets["gold"] = y_gold
        elif src == "self":
            sets["self"] = self_labels(d_cal, perms, n_shift)
        elif src == "think":
            path = cache_path(args.out, args.model, name + ".think.npz")
            if os.path.exists(path):
                y = np.load(path)["labels"]
            else:
                y, texts = thinking_labels(backend_factory(), q, states, args.think_tokens)
                np.savez(path, labels=y)
                with open(path.replace(".npz", ".txt"), "w") as f:
                    f.write("\n\n=====\n\n".join(texts))
            sets["think"] = y
            agree = float(np.mean(y[y >= 0] == y_gold[y >= 0])) if (y >= 0).any() else float("nan")
            print(f"   thinking labels: {int((y >= 0).sum())}/{len(y)} parsed, agree with gold {agree:.3f}", flush=True)
        else:
            raise SystemExit(f"unknown label source {src}")
    return sets


def evaluate(methods: Dict[str, Tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> Dict[str, Dict[str, float]]:
    rows = {}
    for name, (p, p_rev) in methods.items():
        rows[name] = {"acc": metrics.accuracy(p, y), "ece": metrics.ece(p, y), "brier": metrics.brier(p, y),
                      "nll": metrics.nll(p, y), "flip": metrics.flip_rate(p, p_rev)}
    return rows


def table(rows: Dict[str, Dict[str, float]], title: str) -> str:
    cols = ["acc", "ece", "brier", "nll", "flip"]
    lines = [f"### {title}", "", "| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- drivers
def run_bench_task(args, backend_factory, layers, task_name: str) -> Dict[str, Any]:
    task = get_task(task_name)
    test, calib = task.split(args.n, args.calib, args.seed)
    q = task.question
    y_cal = np.asarray([y for _, y in calib])
    y_test = np.asarray([y for _, y in test])
    d_cal = get_features(backend_factory, args.model, args.out, f"{task_name}.calib.s{args.seed}", q,
                         [s for s, _ in calib], layers)
    d_test = get_features(backend_factory, args.model, args.out, f"{task_name}.test.s{args.seed}", q,
                          [s for s, _ in test], layers)
    perms = [list(p) for p in d_test["perms"]]
    n_shift = int(d_test["n_shift"])
    sets = label_sets_for(args, backend_factory, f"{task_name}.calib.s{args.seed}", q, [s for s, _ in calib],
                          d_cal, y_cal, perms, n_shift)
    methods = logit_methods(d_cal, d_test, sets, perms, n_shift)
    heads, chosen = head_methods(d_cal, d_test, sets, q.k, args.kinds)
    methods.update(heads)
    rows = evaluate(methods, y_test)
    print(table(rows, f"{task_name} (K={q.k}, n_test={len(y_test)}, n_calib={len(y_cal)})"), flush=True)
    return {"task": task_name, "k": q.k, "n_test": len(y_test), "n_calib": len(y_cal), "rows": rows, "heads": chosen,
            "label_agreement": {src: float(np.mean(y[y >= 0] == y_cal[y >= 0])) for src, y in sets.items()}}


def run_typed(args, backend_factory, layers) -> Dict[str, Any]:
    from bench.tasks.typed_decisions import group_by_question, load

    test = group_by_question(load("test"))
    calib = group_by_question(load("train", limit_cases=args.calib_cases))
    per_q, pooled = {}, defaultdict(lambda: ([], [], []))
    for key, items in test.items():
        if key not in calib:
            continue
        q = items[0].question
        y_cal = np.asarray([d.gold_index for d in calib[key]])
        y_test = np.asarray([d.gold_index for d in items])
        name = f"typed.{items[0].workflow}.{items[0].qname}"
        d_cal = get_features(backend_factory, args.model, args.out, name + f".calib{args.calib_cases}", q,
                             [d.state for d in calib[key]], layers)
        d_test = get_features(backend_factory, args.model, args.out, name + ".test", q, [d.state for d in items],
                              layers)
        perms = [list(p) for p in d_test["perms"]]
        sets = label_sets_for(args, backend_factory, name + f".calib{args.calib_cases}", q,
                              [d.state for d in calib[key]], d_cal, y_cal, perms, int(d_test["n_shift"]))
        methods = logit_methods(d_cal, d_test, sets, perms, int(d_test["n_shift"]))
        heads, chosen = head_methods(d_cal, d_test, sets, q.k, args.kinds)
        methods.update(heads)
        rows = evaluate(methods, y_test)
        per_q[name] = {"kind": q.kind, "k": q.k, "n_test": len(y_test), "n_calib": len(y_cal), "rows": rows,
                       "heads": chosen}
        for m, (p, p_rev) in methods.items():
            pooled[m][0].append(p)
            pooled[m][1].append(p_rev)
            pooled[m][2].append(y_test)
        heads_only = [m for m in rows if m.startswith("head")]
        best = max(heads_only, key=lambda m: rows[m]["acc"]) if heads_only else "-"
        print(f"{name:55s} K={q.k:<2d} "
              + "  ".join(f"{m} {rows[m]['acc']:.3f}" for m in ("raw", "L1", "L0-perm") if m in rows)
              + f"  best head {best} {rows[best]['acc'] if heads_only else float('nan'):.3f}", flush=True)
    # pooled: decisions weighted equally, as in bench.run_typed
    pooled_rows = {}
    for m, (ps, prs, ys) in pooled.items():
        # questions have different K: pool the metrics that only need (confidence, correct) and per-item Brier
        y = np.concatenate(ys)
        conf = np.concatenate([p.max(axis=1) for p in ps])
        correct = np.concatenate([(p.argmax(axis=1) == yy) for p, yy in zip(ps, ys)]).astype(float)
        brier = np.concatenate([np.sum((p - np.eye(p.shape[1])[yy]) ** 2, axis=1) for p, yy in zip(ps, ys)])
        nll = np.concatenate([-np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, None)) for p, yy in zip(ps, ys)])
        flips = np.concatenate([(p.argmax(axis=1) != pr.argmax(axis=1)) for p, pr in zip(ps, prs)]).astype(float)
        two = np.stack([conf, 1 - conf], axis=1)
        pooled_rows[m] = {"acc": float(correct.mean()), "ece": metrics.ece(two, (1 - correct).astype(int)),
                          "brier": float(brier.mean()), "nll": float(nll.mean()), "flip": float(flips.mean()),
                          "n": int(len(y))}
    print(table(pooled_rows, f"typed-decisions pooled over {len(per_q)} questions, "
                             f"calib {args.calib_cases} cases/workflow"))
    return {"questions": per_q, "pooled": pooled_rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tasks", default="banking20,newsgroups,injection")
    ap.add_argument("--typed", action="store_true", help="LocalLLaMA/typed-decisions instead of --tasks")
    ap.add_argument("--calib-cases", type=int, default=200, help="typed: train cases per workflow")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default=DEFAULT_LAYERS, help="hidden-state layers; -1 = after the final norm")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--labels", default="gold,self",
                    help="comma list of gold (dataset labels), self (the model's own L0-perm answers), "
                         "think (the model's thinking-mode answers, generated once and cached)")
    ap.add_argument("--think-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="bench/results_heads")
    args = ap.parse_args(argv)
    args.kinds = [k for k in args.kinds.split(",") if k]
    args.labels = [x for x in args.labels.split(",") if x]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    holder: Dict[str, Any] = {}

    def backend_factory():
        if "backend" not in holder:
            from anyjev.backends.hf import HFBackend
            holder["backend"] = HFBackend(args.model, batch_size=args.batch_size)
        return holder["backend"]

    result: Dict[str, Any] = {"model": args.model, "layers": layers, "kinds": args.kinds, "labels": args.labels,
                              "seed": args.seed,
                              "env": environment(batch_size=args.batch_size), "date": dt.datetime.now().isoformat()}
    if args.typed:
        result["typed"] = run_typed(args, backend_factory, layers)
        stem = "typed"
    else:
        result["tasks"] = [run_bench_task(args, backend_factory, layers, t) for t in args.tasks.split(",") if t]
        stem = "bench"
    outdir = os.path.join(args.out, dt.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{args.model.replace('/', '__')}.{stem}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
