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
    # 3 states x 4 perms real + 4 perms x 3 probes shared = 24, not 3 x (4 + 12) = 48;
    # two forward calls: the real prompts and the probes never share a batch
    assert be.prompts_seen == 24 and be.calls == 2
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


def test_prior_strength_interpolates_between_none_and_full():
    def be():
        return FakeBackend(lambda s, o: 1.5 if (o == "Yes") == s.startswith("yes") else 0.0, label_prior={"Yes": 2.0})

    q = Question.noul("Is it a yes?", name="y")
    states = [f"yes {i}" for i in range(10)] + [f"no {i}" for i in range(10)]
    p_none = Decider(be(), prior="none").decide_batch(states, q)[-1].p_true
    p_half = Decider(be(), prior="batch", prior_strength=0.5).decide_batch(states, q)[-1].p_true
    p_full = Decider(be(), prior="batch", prior_strength=1.0).decide_batch(states, q)[-1].p_true
    d = Decider(be())                                    # default strength for the batch prior
    p_def = d.decide_batch(states, q)[-1].p_true
    assert d.strength() == 0.75 and Decider(be(), prior="content_free").strength() == 1.0
    assert p_full < p_def < p_half < p_none               # a "no" item: more correction, less Yes
    assert d.decide_batch(states, q)[0].diagnostics["prior_strength"] == 0.75
    with pytest.raises(ValueError):
        Decider(be(), prior_strength=1.5)


def test_l1_artifact_is_a_pure_function_of_the_calibration_set():
    """The repro check found L1 drifting with the decider's history (a running batch prior).
    The artifact now freezes the prior it was fit with."""
    q = Question.choice("Which handler?", OPTIONS, name="route")
    calib = list(TRUTH) * 20
    labels = [OPTIONS.index(TRUTH[s]) for s in calib]
    fresh = Decider(FakeBackend(content, temperature=0.3, position_bias=[1.0, 0, 0, 0]))
    busy = Decider(FakeBackend(content, temperature=0.3, position_bias=[1.0, 0, 0, 0]))
    busy.decide_batch(["unrelated ticket"] * 40 + ["app crashes"] * 25, q)      # history before calibrate
    a = fresh.calibrate(q, calib, labels)
    b = busy.calibrate(q, calib, labels)
    assert a["temperature"] == b["temperature"] and a["prior"] == b["prior"] and a["n_calib"] == 60
    assert a["prior_method"] == "batch" and a["prior_strength"] == 0.75
    # and L1 decisions no longer depend on what else the decider has seen
    busy.decide_batch(["bulk discount"] * 30, q)
    x = fresh.decide("card declined", [q], level="L1")["route"]
    y = busy.decide("card declined", [q], level="L1")["route"]
    np.testing.assert_allclose(x.probs, y.probs, atol=1e-12)
    assert x.diagnostics["prior_method"] == "frozen:batch"
    # a third decider that only loads the artifact agrees too
    other = Decider(FakeBackend(content, temperature=0.3, position_bias=[1.0, 0, 0, 0]))
    other.load_artifact(q, a)
    z = other.decide("card declined", [q], level="L1")["route"]
    np.testing.assert_allclose(x.probs, z.probs, atol=1e-12)


def test_old_artifacts_without_a_prior_still_load():
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content))
    d.load_artifact(q, {"model": "fake", "question": q.key, "method": "temperature", "temperature": 2.0})
    r = d.decide("card declined", [q], level="L1")["route"]
    assert r.level == "L1" and r.diagnostics["temperature"] == 2.0


def test_content_free_probes_never_share_a_forward_call_with_real_states():
    """The batch composition of the real prompts (and so their bf16 logits on a GPU) must not
    depend on whether the content-free probes are cached yet, nor on `record_content_free`."""
    q = Question.choice("Which handler?", OPTIONS)
    states = list(TRUTH) + ["card declined", "reset my password"]

    def spy(dec):
        calls = []
        orig = dec._score

        def _score(prompts, prompt_ids, prompt_parts):
            calls.append(list(prompts))
            return orig(prompts, prompt_ids, prompt_parts)

        dec._score = _score
        return calls

    plain = Decider(FakeBackend(content), prior="batch")
    diag = Decider(FakeBackend(content), prior="batch", record_content_free=True)
    calls_plain, calls_diag = spy(plain), spy(diag)
    a = plain.decide_batch(states, q, level="L0")
    b = diag.decide_batch(states, q, level="L0")
    # the real-state call is identical prompt for prompt; the probes are a separate call
    assert calls_plain[0] == calls_diag[0]
    assert len(calls_plain) == 1 and len(calls_diag) == 2
    assert not set(calls_diag[1]) & set(calls_diag[0])
    assert all(np.allclose(x.probs, y.probs) for x, y in zip(a, b))
    # a second call on the used decider (probes cached) scores exactly the same real prompts
    diag.decide_batch(states, q, level="L0")
    assert calls_diag[2] == calls_diag[0]


def test_an_artifact_with_a_prior_refuses_another_question_layout():
    """The frozen prior is indexed by (permutation, position), so it belongs to the option order
    it was fit on. A 0.0.2 artifact (temperature only) still loads anywhere."""
    from anyjev.question import Question as Q
    be = FakeBackend(content, temperature=0.3)
    q = Question.choice("Which handler?", OPTIONS)
    qr = Q(q.kind, q.text, tuple(reversed(q.options)), q.name, q.scale, q.ordered)
    d = Decider(be)
    states = list(TRUTH) * 20
    art = d.calibrate(q, states, [OPTIONS.index(TRUTH[s]) for s in states])
    assert art["prior"] is not None
    with pytest.raises(ValueError):
        d.load_artifact(qr, art)
    d.load_artifact(q, art)                                        # same layout: fine
    old = {k: v for k, v in art.items() if k not in ("prior", "prior_method", "prior_strength", "n_calib")}
    d.load_artifact(qr, old)                                       # temperature-only: fine
    assert d.decide("card declined", [qr], level="L1")[0].level == "L1"
