"""Feature caches the depth, Jev-mode, label-efficiency, listing-order and distillation studies replay.

One GPU pass per (question, split, listing order) stores, for every decision, the last-position
hidden state at the requested blocks (`H`), the raw label log-probs (`Z`), the restricted logit
lens at each block (`lens`; block loop only), the gold option (`y`), the listing order shown
(`perms`), the soft teacher label when the dataset has one (`soft`) and the option-line states
(`G`; not read by any shipped study, kept so the cache format is unchanged) under

    <out>/features/<model with "/" -> "__">/<name>.<split>.<order>.<which>.npz

`python -m bench.extract_pools` writes them; `bench.exit_study`, `bench.jev_mode_table`,
`bench.labels_study`, `bench.paraphrase_study order`, `bench.distill_heads` and
`scripts/build_heads.py` replay them on CPU.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from anyjev.question import Question
from anyjev.readout import (
    DEFAULT_SYSTEM,
    LabelTokenError,
    build_prompt,
    label_ids_for_perm,
    option_spans,
    option_token_positions,
    resolve_labels,
)
from anyjev.state import render_state
from bench.tasks.typed_decisions import Decision, group_by_question, load


# ---------------------------------------------------------------- data
@dataclass
class DecisionPool:
    """Every decision of one question: last-position features H [N, L, d], option-line features
    G [N, K, L, d] (unused by the shipped studies), raw option-space log-probs Z [N, K], gold
    indices y [N]. `group` is the workflow / task name (metadata)."""
    name: str
    kind: str
    K: int
    G: np.ndarray
    H: np.ndarray
    Z: np.ndarray
    y: np.ndarray
    group: str = ""
    soft: Optional[np.ndarray] = None       # teacher probabilities [N, K] when available
    lens: Optional[np.ndarray] = None       # [N, L, K] restricted logit-lens logits per layer, option order
    perms: Optional[np.ndarray] = None      # [N, K] listing order per decision: perms[i][j] = option at position j

    @property
    def n(self) -> int:
        return int(len(self.y))


# ---------------------------------------------------------------- extraction
def listing_perms(q: Question, n: int, order: str = "id", seed: int = 0) -> List[List[int]]:
    """One listing order per state: "id" (canonical), "rev" (reversed), "rand" (a uniformly random
    permutation per state; for noul one of the two phrasings; score questions are never permuted)."""
    K = q.k
    if order == "id" or q.ordered:
        return [list(range(K)) for _ in range(n)]
    if order == "rev":
        return [list(reversed(range(K))) for _ in range(n)]
    if order == "rand":
        rng = np.random.RandomState(seed)
        if q.kind == "noul":
            return [[0, 1] if rng.rand() < 0.5 else [1, 0] for _ in range(n)]
        return [list(rng.permutation(K)) for _ in range(n)]
    raise ValueError(f"unknown listing order {order!r}")


def build_rows(backend, q: Question, states: Sequence[Any], perms: Sequence[Sequence[int]],
               which: str = "last", system: str = DEFAULT_SYSTEM):
    """Rendered prompts, label token ids (position order) and option-line token positions
    (position order) for each (state, listing order)."""
    tok = backend.tokenizer
    labels, ids = resolve_labels(tok, q)
    prompts, token_ids, positions = [], [], []
    for st, perm in zip(states, perms):
        spec = build_prompt(render_state(st), q, perm, system, labels)
        text, spans = option_spans(tok, spec, q.kind)
        if len(spans) != q.k:
            raise LabelTokenError(f"found {len(spans)} option spans for a {q.k}-option question: {text[-300:]!r}")
        prompts.append(text)
        token_ids.append(label_ids_for_perm(q, ids, perm))
        positions.append(option_token_positions(tok, text, spans, which))
    return prompts, token_ids, positions


def extract_pool(backend, q: Question, states: Sequence[Any], labels: Sequence[int], layers: Sequence[int],
                 order: str = "id", seed: int = 0, which: str = "last", name: Optional[str] = None,
                 group: str = "", soft: Optional[np.ndarray] = None, use_loop: bool = False,
                 max_layer: Optional[int] = None) -> DecisionPool:
    """One forward per (state, listing order): last-position states H at `layers`, raw label
    log-probs Z (option order), gold y; through the block loop (`use_loop`) also the restricted
    logit lens at every layer. G (option-line states, option order) is kept for cache
    compatibility."""
    perms = listing_perms(q, len(states), order, seed)
    prompts, token_ids, positions = build_rows(backend, q, states, perms, which)
    lens_pos = None
    if use_loop and hasattr(backend, "hidden_states_to"):
        # the block loop captures layers without keeping every layer's activations and adds the
        # restricted logit lens (final norm + label rows of lm_head) at each captured layer
        _, ids = resolve_labels(backend.tokenizer, q)
        feats, lps, pos, lens_pos = backend.hidden_states_to(prompts, list(layers), token_ids, positions,
                                                             max_layer=max_layer, lens_ids=ids)
    else:
        feats, lps, pos = backend.hidden_states(prompts, list(layers), token_ids, positions)
    N, K = len(states), q.k
    G = np.zeros((N, K, pos.shape[2], pos.shape[3]), dtype=np.float16)
    Z = np.zeros((N, K), dtype=np.float64)
    lens = np.zeros((N, pos.shape[2], K), dtype=np.float32) if lens_pos is not None else None
    for i, perm in enumerate(perms):
        for j, opt in enumerate(perm):
            G[i, opt] = pos[i, j]
            Z[i, opt] = lps[i][j]
            if lens is not None:
                # lens logits are in label-id order: for positional labels the label at position j
                # names the option at position j; for noul the ids are already in option order
                lens[i, :, opt] = lens_pos[i, :, opt] if q.kind == "noul" else lens_pos[i, :, j]
    return DecisionPool(name=name or q.id, kind=q.kind, K=K, G=G, H=feats, Z=Z,
                        y=np.asarray(labels, dtype=int), group=group,
                        soft=None if soft is None else np.asarray(soft, dtype=np.float64), lens=lens,
                        perms=np.asarray(perms, dtype=int))


# ---------------------------------------------------------------- cache
def pool_path(out: str, model: str, name: str, split: str, order: str, which: str = "last") -> str:
    return os.path.join(out, "features", model.replace("/", "__"), f"{name}.{split}.{order}.{which}.npz")


def save_pool(path: str, pool: DecisionPool, extra: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, G=pool.G.astype(np.float16), H=pool.H.astype(np.float16), Z=pool.Z, y=pool.y,
             K=pool.K, kind=pool.kind, group=pool.group, name=pool.name,
             soft=(pool.soft if pool.soft is not None else np.zeros(0)),
             lens=(pool.lens if pool.lens is not None else np.zeros(0)),
             perms=(pool.perms if pool.perms is not None else np.zeros(0)), **extra)


def load_pool(path: str) -> DecisionPool:
    z = np.load(path, allow_pickle=False)
    soft = z["soft"]
    lens = z["lens"] if "lens" in z.files else np.zeros(0)
    perms = z["perms"] if "perms" in z.files else np.zeros(0)
    return DecisionPool(name=str(z["name"]), kind=str(z["kind"]), K=int(z["K"]), G=z["G"], H=z["H"], Z=z["Z"],
                        y=z["y"], group=str(z["group"]), soft=(soft if soft.size else None),
                        lens=(lens if lens.size else None), perms=(perms if perms.size else None))


def get_pool(backend_factory, args, layers, name: str, split: str, order: str, q: Question, states, labels,
             group: str, soft=None) -> DecisionPool:
    path = pool_path(args.out, args.model, name, split, order, args.which)
    if os.path.exists(path):
        pool = load_pool(path)
        if pool.perms is None:          # caches written before listing orders were stored: regenerate them
            pool.perms = np.asarray(listing_perms(q, pool.n, order, args.seed), dtype=int)
        return pool
    t0 = time.time()
    pool = extract_pool(backend_factory(), q, states, labels, layers, order=order, seed=args.seed, which=args.which,
                        name=name, group=group, soft=soft, use_loop=args.use_loop)
    save_pool(path, pool, {"seconds": time.time() - t0, "layers": np.asarray(layers)})
    print(f"   extracted {name}.{split}.{order}: {pool.G.shape} in {time.time() - t0:.0f}s", flush=True)
    return pool


# ---------------------------------------------------------------- labelled records
def typed_questions(calib_cases: int):
    test = group_by_question(load("test"))
    train = group_by_question(load("train", limit_cases=calib_cases))
    out = []
    for key, items in test.items():
        if key not in train:
            continue
        out.append((f"typed.{items[0].workflow}.{items[0].qname}", items[0].question, items[0].workflow,
                    train[key], items))
    return out


def decisions_to_arrays(items: Sequence[Decision]):
    return ([d.state for d in items], [d.gold_index for d in items], np.asarray([d.gold_probs for d in items]))


def typed_records(calib_cases: int) -> List[Dict[str, Any]]:
    out = []
    for name, q, workflow, train_items, test_items in typed_questions(calib_cases):
        s_tr, y_tr, soft_tr = decisions_to_arrays(train_items)
        s_te, y_te, soft_te = decisions_to_arrays(test_items)
        out.append({"name": name, "q": q, "group": workflow, "domain": "typed",
                    "calib": (s_tr, y_tr, soft_tr), "test": (s_te, y_te, soft_te)})
    return out


def task_records(names: Sequence[str], n_test: int, n_calib: int, seed: int, domain: str) -> List[Dict[str, Any]]:
    """bench tasks through the registry (Task.split gives disjoint test and calibration)."""
    from bench.tasks import get_task

    out = []
    for t in names:
        task = get_task(t)
        test, calib = task.split(n_test, n_calib, seed)
        out.append({"name": f"{domain}.{t}", "q": task.question, "group": t, "domain": domain,
                    "calib": ([s for s, _ in calib], [y for _, y in calib], None),
                    "test": ([s for s, _ in test], [y for _, y in test], None)})
    return out
