"""End-to-end through the real prompt path with a synthetic biased model."""
import numpy as np
import pytest

from anyjev import Decider, Question
from anyjev.backends.fake import FakeBackend

OPTIONS = ["billing", "technical", "sales", "other"]
TRUTH = {"card declined": "billing", "app crashes": "technical", "bulk discount": "sales"}


def content(state, option):
    return 3.0 if TRUTH.get(state) == option else 0.0


def test_raw_is_fooled_by_position_bias_l0_is_not():
    be = FakeBackend(content, position_bias=[4.0, 0, 0, 0])   # loves position A
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be)
    raw = d.decide("app crashes", [q], level="raw")["route"]
    l0 = d.decide("app crashes", [q], level="L0")["route"]
    assert raw.level == "raw" and raw.argmax == "billing"       # wrong, position A
    assert l0.level == "L0" and l0.argmax == "technical"
    assert l0.diagnostics["order_flip_raw"] > 0
    assert l0.diagnostics["prior_method"] == "none"             # one item: batch prior not available yet
    assert l0.diagnostics["order_flip_l0"] == l0.diagnostics["order_flip_raw"]


def test_content_free_prior_removes_label_prior_exactly():
    be = FakeBackend(content, label_prior={"Yes": 2.0})
    q = Question.noul("Is this about billing?", name="bill")
    d = Decider(be, prior="content_free")
    raw = d.decide("nothing", [q], level="raw")["bill"]
    l0 = d.decide("nothing", [q], level="L0")["bill"]
    assert raw.p_true > 0.85
    assert abs(l0.p_true - 0.5) < 1e-6
    assert l0.diagnostics["prior_method"] == "content_free"


def test_batch_prior_removes_label_prior_on_a_balanced_batch():
    be = FakeBackend(lambda s, o: 1.5 if (o == "Yes") == s.startswith("yes") else 0.0,
                     label_prior={"Yes": 2.0})
    q = Question.noul("Is it a yes?", name="y")
    states = [f"yes {i}" for i in range(10)] + [f"no {i}" for i in range(10)]
    d = Decider(be, prior="batch", min_prior_n=8)
    raw = d.decide_batch(states, q, level="raw")
    l0 = d.decide_batch(states, q, level="L0")
    raw_acc = np.mean([(r.p_true >= 0.5) == s.startswith("yes") for r, s in zip(raw, states)])
    l0_acc = np.mean([(r.p_true >= 0.5) == s.startswith("yes") for r, s in zip(l0, states)])
    assert raw_acc == 0.5 and l0_acc == 1.0          # prior swamps content raw; batch prior removes it
    assert l0[0].diagnostics["prior_method"] == "batch"
    # single-item call afterwards uses the running prior
    one = d.decide("no 99", [q])["y"]
    assert one.diagnostics["prior_method"] == "batch" and one.p_true < 0.5


def test_score_expected_value():
    be = FakeBackend(lambda s, o: 5.0 if o.startswith("0.75") else 0.0)
    q = Question.score("How complete?", bins=4, name="done")
    r = Decider(be).decide("x", [q])["done"]
    assert 0.8 < r.value < 0.9
    assert r.level == "L0"


def test_probes_are_shared_across_states():
    be = FakeBackend(content, position_bias=[1, 0, 0, 0])
    q = Question.choice("Which handler?", OPTIONS)
    d = Decider(be, prior="content_free")
    d.decide_batch(list(TRUTH), q)
    # 3 states x 4 perms real + 4 perms x 3 probes shared = 24, not 3 x (4 + 12) = 48
    assert be.calls == 1 and be.prompts_seen == 24
    d.decide("card declined", [q])
    assert be.prompts_seen == 28                        # cf prior cached: 4 new prompts, no probes
    d2 = Decider(FakeBackend(content))
    d2.decide_batch(list(TRUTH), q)
    assert d2.backend.prompts_seen == 12                # default batch prior: no probes at all


def test_l1_requires_artifact_and_reports_level():
    be = FakeBackend(content, temperature=0.3)   # overconfident
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be)
    with pytest.raises(ValueError):
        d.decide("card declined", [q], level="L1")
    states = list(TRUTH) * 20
    labels = [OPTIONS.index(TRUTH[s]) for s in states]
    art = d.calibrate(q, states, labels)
    assert art["model"] == "fake" and art["temperature"] > 0
    r = d.decide("card declined", [q], level="L1")["route"]
    assert r.level == "L1" and r.argmax == "billing"
    d2 = Decider(FakeBackend(content))
    d2.load_artifact(q, art)
    with pytest.raises(ValueError):
        d2.load_artifact(q, {**art, "model": "some-other-model"})


def test_max_permutations_cap():
    be = FakeBackend(content)
    q = Question.choice("Which handler?", OPTIONS)
    r = Decider(be, max_permutations=2).decide("x", [q])[0]
    assert r.diagnostics["permutations"] == 2


def test_ablation_data_present():
    be = FakeBackend(content, position_bias=[4.0, 0, 0, 0])
    q = Question.choice("Which handler?", OPTIONS, name="route")
    r = Decider(be, record_content_free=True).decide("app crashes", [q])["route"]
    assert r.diagnostics["p_pos_raw"].shape == (4, 4)
    assert r.diagnostics["cf_prior"].shape == (4, 4)
    assert r.diagnostics["batch_prior"] is None         # one item: below min_prior_n
    assert len(r.diagnostics["perms"]) == 4


def test_noul_phrasing_swap_keeps_yes_attached_to_yes():
    # no biases at all: "Yes or No" and "No or Yes" must give the same distribution
    be = FakeBackend(lambda s, o: 2.0 if o == "Yes" else 0.0)
    q = Question.noul("Is it?", name="it")
    r = Decider(be).decide("x", [q])["it"]
    P = r.diagnostics["p_pos_raw"]            # [2 phrasings, 2 positions]
    perms = r.diagnostics["perms"]
    a = {perms[0][j]: P[0, j] for j in range(2)}
    b = {perms[1][j]: P[1, j] for j in range(2)}
    assert abs(a[0] - b[0]) < 1e-9 and a[0] > 0.8   # option 0 = Yes in both
    assert r.diagnostics["order_flip_raw"] == 0.0
    assert r.p_true > 0.8


def test_artifacts_round_trip(tmp_path):
    be = FakeBackend(content, temperature=0.3)
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be, prior="none")            # same prior on both sides, so only the artifact differs
    states = list(TRUTH) * 20
    d.calibrate(q, states, [OPTIONS.index(TRUTH[s]) for s in states])
    path = tmp_path / "art.json"
    d.save_artifacts(str(path))
    d2 = Decider(FakeBackend(content, temperature=0.3), prior="none")
    assert d2.load_artifacts(str(path)) == 1
    a = d.decide("card declined", [q], level="L1")["route"]
    b = d2.decide("card declined", [q], level="L1")["route"]
    np.testing.assert_allclose(a.probs, b.probs, atol=1e-12)
    assert a.diagnostics["temperature"] == b.diagnostics["temperature"]
    with pytest.raises(ValueError):
        Decider(FakeBackend(content)).load_artifacts({"model": "other", "artifacts": {}})
