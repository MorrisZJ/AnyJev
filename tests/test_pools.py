"""bench.pools on the fake backend: listing orders, option spans, and the npz cache round trip.

No datasets, no GPU: the fake backend plants each option's content along one fixed direction of
the option-line state, so the cache can be checked for shape, dtype and geometry on CPU."""
import os
import re

import numpy as np

from anyjev import Question
from anyjev.backends.fake import FakeBackend
from anyjev.readout import build_prompt, option_spans, option_token_positions, render_chat, resolve_labels
from bench.pools import extract_pool, listing_perms, load_pool, pool_path, save_pool

VOCAB = {
    "route": ["billing", "technical", "sales", "other"],
    "size": ["small", "medium", "large", "huge", "giant"],
}


def content(state: str, option: str) -> float:
    return 2.0 if option in state else 0.0


def test_listing_perms_cover_the_kinds():
    q = Question.choice("q", VOCAB["size"])
    assert listing_perms(q, 3, "id") == [[0, 1, 2, 3, 4]] * 3
    assert listing_perms(q, 1, "rev") == [[4, 3, 2, 1, 0]]
    assert sorted(listing_perms(q, 1, "rand", seed=1)[0]) == [0, 1, 2, 3, 4]
    qs = Question.score("s", levels=["a", "b", "c"])
    assert listing_perms(qs, 2, "rand") == [[0, 1, 2]] * 2
    qn = Question.noul("n")
    assert all(p in ([0, 1], [1, 0]) for p in listing_perms(qn, 10, "rand"))


def test_option_spans_follow_the_listing_order_for_every_kind():
    tok = FakeBackend(content).tokenizer
    q = Question.choice("Which route?", VOCAB["route"])
    for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        labels, _ = resolve_labels(tok, q)
        spec = build_prompt("state text", q, perm, labels=labels)
        text, spans = option_spans(tok, spec, "choice")
        assert text == render_chat(tok, spec)
        assert [text[a:b] for a, b in spans] == [q.options[i] for i in perm]
        pos = option_token_positions(tok, text, spans, "last")
        words = re.findall(r"\S+|\n", text)  # the fake tokenizer's word tokens
        assert [words[p] for p in pos] == [q.options[i] for i in perm]
    qn = Question.noul("Is it urgent?")
    labels, _ = resolve_labels(tok, qn)
    for perm in ([0, 1], [1, 0]):
        text, spans = option_spans(tok, build_prompt("s", qn, perm, labels=labels), "noul")
        assert [text[a:b] for a, b in spans] == [("Yes", "No")[i] for i in perm]
    qs = Question.score("How severe?", levels=["none", "minor", "major"])
    labels, _ = resolve_labels(tok, qs)
    text, spans = option_spans(tok, build_prompt("s", qs, [0, 1, 2], labels=labels), "score")
    assert [text[a:b] for a, b in spans] == list(qs.options)


def test_extract_pool_round_trips_through_the_npz_cache(tmp_path):
    be = FakeBackend(content, position_bias=[1.5, 0, 0, 0])
    q = Question.choice("Which route?", VOCAB["route"])
    pool = extract_pool(be, q, ["ticket about billing", "ticket about sales"], [0, 2], layers=[-2, -1],
                        order="rand", seed=3)
    assert pool.H.shape == (2, 2, be.hidden_size)
    assert pool.Z.shape == (2, 4)
    assert pool.G.shape == (2, 4, 2, be.hidden_size)
    assert pool.lens is None                # FakeBackend has no block loop, so no logit lens on CPU
    assert pool.perms.shape == (2, 4)
    assert pool.n == 2 and pool.K == 4 and pool.kind == "choice"
    assert np.array_equal(pool.y, [0, 2])
    # option-line states are stored in option order whatever the listing order: at the exact (last)
    # layer the content direction carries only the option named in the state
    g = pool.G[0, :, 1, :].astype(np.float64)
    proj = g @ be._u
    assert proj[0] > 1.5 and max(proj[1:]) < 0.5
    path = pool_path(str(tmp_path), "fake/model", "q", "train", "rand")
    assert path.endswith(os.path.join("features", "fake__model", "q.train.rand.last.npz"))
    save_pool(path, pool, {"layers": np.asarray([-2, -1]), "seconds": 0.0})
    z = np.load(path)
    assert set(z.files) == {"G", "H", "Z", "y", "K", "kind", "group", "name", "soft", "lens", "perms",
                            "layers", "seconds"}
    assert z["G"].dtype == np.float16 and z["H"].dtype == np.float16 and z["Z"].dtype == np.float64
    assert [int(x) for x in z["layers"]] == [-2, -1]
    p2 = load_pool(path)
    assert np.allclose(p2.H, pool.H.astype(np.float16))
    assert np.array_equal(p2.y, pool.y)
    assert p2.K == 4 and p2.kind == "choice" and p2.name == pool.name
    assert p2.soft is None and p2.lens is None
    assert np.array_equal(p2.perms, pool.perms)
    assert p2.n == 2
