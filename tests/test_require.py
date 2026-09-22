"""The level contract is enforceable: require= refuses probabilities below the asked level."""
import pytest

from anyjev import Decider, LevelError, Question
from anyjev.backends.fake import FakeBackend

OPTIONS = ["billing", "technical", "sales", "other"]
TRUTH = {"card declined": "billing", "app crashes": "technical", "bulk discount": "sales"}


def content(state, option):
    return 3.0 if TRUTH.get(state) == option else 0.0


def test_require_l1_refuses_l0_and_raw():
    d = Decider(FakeBackend(content))
    q = Question.choice("Which handler?", OPTIONS, name="route")
    with pytest.raises(LevelError) as e:
        d.decide("card declined", [q], require="L1")
    assert "route" in str(e.value) and "calibrate()" in str(e.value)
    with pytest.raises(LevelError):
        d.decide("card declined", [q], level="raw", require="L0")
    with pytest.raises(LevelError):
        d.decide_batch(list(TRUTH), q, require="L1")


def test_require_passes_at_or_above_level():
    d = Decider(FakeBackend(content, temperature=0.3))
    q = Question.choice("Which handler?", OPTIONS, name="route")
    r = d.decide("card declined", [q], require="L0")
    assert r["route"].level == "L0"
    assert r["route"].require("raw") is r["route"]          # higher than asked is fine
    states = list(TRUTH) * 20
    d.calibrate(q, states, [OPTIONS.index(TRUTH[s]) for s in states])
    r = d.decide("card declined", [q], level="L1", require="L1")
    assert r["route"].level == "L1"
    assert r.require("L0") is r


def test_require_rejects_unknown_level():
    d = Decider(FakeBackend(content))
    q = Question.choice("Which handler?", OPTIONS)
    with pytest.raises(ValueError):
        d.decide("x", [q], require="L9")
