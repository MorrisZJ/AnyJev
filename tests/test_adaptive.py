"""Adaptive cyclic shifts (opt-in): stop early on easy items, run all shifts on hard ones.

Items are built with balanced answers: a batch where every item has the same answer makes
the batch prior equal to the signal itself (the documented skewed-marginal limitation), which
would confound the test with the prior rather than the stopping rule.
"""
from anyjev import Decider, Question
from anyjev.backends.fake import FakeBackend

OPTIONS = [f"opt{i}" for i in range(8)]
BIAS = [3.0, 0, 0, 0, 0, 0, 0, 0]          # the model loves position A


def easy(state, option):                   # item i -> opt(i % 8)
    return 4.0 if option == OPTIONS[int(state.split()[-1]) % 8] else 0.0


def hard(state, option):                   # item i -> tie between opt(i % 8) and opt((i + 3) % 8)
    i = int(state.split()[-1])
    return 4.0 if option in (OPTIONS[i % 8], OPTIONS[(i + 3) % 8]) else 0.0


def mixed(state, option):
    return easy(state, option) if state.startswith("easy") else hard(state, option)


def test_easy_items_stop_at_min_shifts_and_stay_correct():
    q = Question.choice("q", OPTIONS, name="q")
    states = [f"easy {i}" for i in range(16)]
    for prior in ("content_free", "batch"):
        d = Decider(FakeBackend(easy, position_bias=BIAS), prior=prior,
                    adaptive_shifts=True, adaptive_min_shifts=2, adaptive_margin=0.1)
        decs = d.decide_batch(states, q)
        assert [x.argmax for x in decs] == [OPTIONS[i % 8] for i in range(16)], prior
        assert all(x.diagnostics["shifts_used"] == 2 for x in decs), prior
        assert d.stats["adaptive_items"] == 16 and d.stats["adaptive_shifts_total"] == 32


def test_ambiguous_items_run_every_shift():
    q = Question.choice("q", OPTIONS, name="q")
    states = [f"hard {i}" for i in range(16)]
    d = Decider(FakeBackend(hard, position_bias=BIAS), prior="content_free",
                adaptive_shifts=True, adaptive_min_shifts=2, adaptive_margin=0.1)
    decs = d.decide_batch(states, q)
    assert all(x.diagnostics["shifts_used"] == 8 for x in decs)
    for i, x in enumerate(decs):           # exact prior + full cycle: the tie is symmetric
        assert abs(x.probs[i % 8] - x.probs[(i + 3) % 8]) < 1e-6


def test_mixed_batch_agrees_with_full_l0():
    q = Question.choice("q", OPTIONS, name="q")
    states = [f"easy {i}" for i in range(16)] + [f"hard {i}" for i in range(8)]
    full = Decider(FakeBackend(mixed, position_bias=BIAS), prior="content_free").decide_batch(states, q)
    ad = Decider(FakeBackend(mixed, position_bias=BIAS), prior="content_free",
                 adaptive_shifts=True, adaptive_margin=0.1)
    decs = ad.decide_batch(states, q)
    assert [x.argmax for x in decs[:16]] == [f.argmax for f in full[:16]]
    assert [x.diagnostics["shifts_used"] for x in decs[:16]] == [2] * 16
    assert all(x.diagnostics["shifts_used"] == 8 for x in decs[16:])
    # the prior removes the additive position bias per shift, so the 2-shift marginal on easy
    # items matches the 8-shift one closely; hard items ran the full cycle and match exactly
    for f, a in zip(full, decs):
        assert abs(f.confidence - a.confidence) < 0.02
    assert ad.backend.prompts_seen < FakeBackend(mixed).prompts_seen + 24 * 8 + 24   # fewer prompts than full


def test_adaptive_is_ignored_where_it_does_not_apply():
    d = Decider(FakeBackend(easy, position_bias=BIAS), adaptive_shifts=True)
    q = Question.choice("q", OPTIONS, name="q")
    r = d.decide("easy 3", [q], level="raw")["q"]
    assert "shifts_used" not in r.diagnostics and r.level == "raw"
    r = d.decide("easy 3", [Question.noul("n", name="n"), q])
    assert "shifts_used" not in r["n"].diagnostics and r["q"].diagnostics["adaptive"] is True
    assert r["q"].level == "L0"
