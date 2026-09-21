import numpy as np

from anyjev.calibrate import (
    TemperatureScaler,
    apply_contextual,
    content_free_prior,
    cyclic_shifts,
    flip_rate_across_perms,
    marginalize,
)


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def test_contextual_removes_additive_prior_exactly():
    content = np.array([1.0, 0.2, -0.5])
    bias = np.array([2.0, -1.0, 0.5])
    p_raw = softmax(content + bias)
    prior = content_free_prior(np.stack([softmax(bias)] * 3))
    p = apply_contextual(p_raw, prior)
    np.testing.assert_allclose(p, softmax(content), atol=1e-9)


def test_readme_worked_example():
    # P(Yes)=0.62 raw, prior P(Yes)=0.70 -> 0.41 after correction
    p = apply_contextual(np.array([0.62, 0.38]), np.array([0.70, 0.30]))
    assert abs(p[0] - 0.4115) < 1e-3


def test_cyclic_shifts_cover_every_position():
    k = 5
    perms = cyclic_shifts(k)
    assert len(perms) == k
    for i in range(k):
        assert sorted(perm[i] for perm in perms) == list(range(k))
    assert len(cyclic_shifts(k, max_permutations=2)) == 2


def test_marginalize_maps_positions_back_to_options():
    perms = cyclic_shifts(3)
    # a model that always puts all mass on position 0
    p_by_perm = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], float)
    out = marginalize(p_by_perm, perms)
    np.testing.assert_allclose(out, [1 / 3, 1 / 3, 1 / 3])
    assert flip_rate_across_perms(p_by_perm, perms) == 1.0


def _run_marginalize(content, pos_bias, order, combine):
    k = len(content)
    perms = cyclic_shifts(k)
    c = content[order]
    p_by_perm = np.stack([softmax(c[perm] + pos_bias) for perm in perms])
    m = marginalize(p_by_perm, perms, combine=combine)
    back = np.empty(k)
    back[order] = m
    return back


CONTENT = np.array([0.3, 1.2, -0.4, 0.9])
POS_BIAS = np.array([1.5, 0.0, 0.0, -0.5])


def test_logmean_removes_additive_position_bias_exactly():
    out = _run_marginalize(CONTENT, POS_BIAS, np.arange(4), "logmean")
    np.testing.assert_allclose(out, softmax(CONTENT), atol=1e-12)


def test_logmean_is_invariant_to_original_listing_order():
    a = _run_marginalize(CONTENT, POS_BIAS, np.arange(4), "logmean")
    b = _run_marginalize(CONTENT, POS_BIAS, np.array([2, 0, 3, 1]), "logmean")
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_mean_is_invariant_to_rotation_only():
    a = _run_marginalize(CONTENT, POS_BIAS, np.arange(4), "mean")
    rot = _run_marginalize(CONTENT, POS_BIAS, np.array([1, 2, 3, 0]), "mean")
    other = _run_marginalize(CONTENT, POS_BIAS, np.array([2, 0, 3, 1]), "mean")
    np.testing.assert_allclose(a, rot, atol=1e-12)
    assert not np.allclose(a, other, atol=1e-6)
    assert int(np.argmax(a)) == 1


def test_temperature_scaling_recovers_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2000, 4)) * 2
    labels = np.array([rng.choice(4, p=softmax(z)) for z in logits])
    overconfident = softmax(logits * 3.0)          # true T is 3
    s = TemperatureScaler.fit(overconfident, labels)
    assert 2.5 < s.temperature < 3.5
    p = s.apply(overconfident)
    np.testing.assert_allclose(p.sum(1), 1.0)
    rt = TemperatureScaler.from_dict(s.to_dict())
    assert rt.temperature == s.temperature
