"""Shared-prefix scoring: same numbers as full prompts, fewer prefix computations."""
import numpy as np

from anyjev import Decider, Question
from anyjev.backends.fake import FakeBackend, FakeTokenizer
from anyjev.readout import build_prompt, render_chat, render_chat_parts

OPTIONS = ["billing", "technical", "sales", "other"]
TRUTH = {"card declined": "billing", "app crashes": "technical", "bulk discount": "sales"}


def content(state, option):
    return 3.0 if TRUTH.get(state) == option else 0.0


def test_parts_concatenate_to_the_full_prompt():
    tok = FakeTokenizer()
    for q in (Question.choice("q", OPTIONS), Question.noul("q"), Question.score("q", bins=3)):
        spec = build_prompt("some state", q, list(range(q.k)))
        pre, suf = render_chat_parts(tok, spec)
        assert pre + suf == render_chat(tok, spec)
        assert suf and "State:" in pre and "State:" not in suf
    # the permutations of one state share the prefix exactly
    q = Question.choice("q", OPTIONS)
    pres = {render_chat_parts(tok, build_prompt("s", q, p))[0] for p in ([0, 1, 2, 3], [1, 2, 3, 0], [3, 0, 1, 2])}
    assert len(pres) == 1


def test_shared_and_flat_paths_agree_and_grouping_is_by_state():
    q = Question.choice("Which handler?", OPTIONS, name="route")
    states = list(TRUTH) * 3
    flat = Decider(FakeBackend(content, position_bias=[2, 0, 0, 0]), shared_prefix=False)
    shared = Decider(FakeBackend(content, position_bias=[2, 0, 0, 0]), shared_min_prefix_tokens=0)   # auto, any length
    a = flat.decide_batch(states, q)
    b = shared.decide_batch(states, q)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.probs, y.probs, atol=1e-12)
    # 9 states but 3 distinct texts: prompts are deduplicated, so 3 x 4 permutations = 12
    assert flat.stats["shared_groups"] == 0 and flat.stats["flat_prompts"] == 12
    # auto: one group per distinct state, all 4 permutations in it
    assert shared.stats["shared_groups"] == 3 and shared.stats["shared_prompts"] == 12
    assert shared.stats["flat_prompts"] == 0 and shared.backend.shared_calls == 1


def test_auto_shares_only_long_prefixes():
    q = Question.choice("q", OPTIONS)
    # FakeTokenizer encodes unknown text as two tokens, so every prefix is "short"
    short = Decider(FakeBackend(content))                            # default threshold 256
    short.decide("x", [q])
    assert short.stats["shared_groups"] == 0 and short.stats["flat_prompts"] == 4
    long = Decider(FakeBackend(content), shared_min_prefix_tokens=0)
    long.decide("x", [q])
    assert long.stats["shared_groups"] == 1 and long.stats["flat_prompts"] == 0
    forced = Decider(FakeBackend(content), shared_prefix=True)       # True ignores the threshold
    forced.decide("x", [q])
    assert forced.stats["shared_groups"] == 1


def test_small_groups_stay_flat_under_auto_but_share_under_true():
    two = Question.choice("q", ["a", "b"])
    auto = Decider(FakeBackend(content), shared_min_prefix_tokens=0)
    auto.decide("x", [two])
    assert auto.stats["shared_groups"] == 0 and auto.stats["flat_prompts"] == 2
    forced = Decider(FakeBackend(content), shared_prefix=True)
    forced.decide("x", [two])
    assert forced.stats["shared_groups"] == 1 and forced.stats["flat_prompts"] == 0
    # noul phrasings carry different label ids per phrasing, so they never group
    n = Decider(FakeBackend(content), shared_prefix=True)
    n.decide("x", [Question.noul("q")])
    assert n.stats["shared_groups"] == 0 and n.stats["flat_prompts"] == 2


def test_content_free_probes_group_too():
    q = Question.choice("q", OPTIONS)
    d = Decider(FakeBackend(content), prior="content_free", shared_min_prefix_tokens=0)
    d.decide_batch(list(TRUTH), q)
    # 3 states + 3 probes = 6 groups of 4 suffixes, nothing flat
    assert d.stats["shared_groups"] == 6 and d.stats["flat_prompts"] == 0
