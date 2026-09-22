import numpy as np
import pytest

from anyjev.heads import KINDS, LinearHead, fit_head


def planted(n, d, K, noise, seed=0):
    """Hidden states = class direction + isotropic noise, in a d-dim space, K classes. The
    class directions depend only on (d, K), so calibration and test sets share them."""
    dirs = np.random.RandomState(1234 + d + 7 * K).randn(K, d)
    rng = np.random.RandomState(seed)
    y = rng.randint(0, K, size=n)
    X = dirs[y] + noise * rng.randn(n, d)
    return X.astype(np.float32), y


@pytest.mark.parametrize("kind", KINDS)
def test_each_head_recovers_a_planted_linear_structure(kind):
    Xc, yc = planted(160, 64, 4, noise=3.0, seed=1)
    Xt, yt = planted(400, 64, 4, noise=3.0, seed=2)
    h = fit_head(Xc, yc, K=4, kind=kind)
    acc = np.mean(h.probs(Xt).argmax(axis=1) == yt)
    chance = 0.25
    assert acc > 0.7 > chance, (kind, acc)
    assert np.allclose(h.probs(Xt).sum(axis=1), 1.0)


def test_layer_is_chosen_by_cross_validation():
    Xc, yc = planted(120, 32, 3, noise=2.0, seed=3)
    junk = np.random.RandomState(0).randn(*Xc.shape).astype(np.float32)
    F = np.stack([junk, Xc], axis=1)            # layer 0 is noise, layer 1 carries the signal
    h = fit_head(F, yc, K=3, kind="ridge")
    assert h.layer == 1
    assert h.cv["oof_acc"] > 0.7


def test_head_round_trips_through_json_and_refuses_tiny_sets():
    Xc, yc = planted(60, 16, 3, noise=1.0)
    h = fit_head(Xc, yc, K=3, kind="lda")
    h2 = LinearHead.from_dict(h.to_dict())
    assert np.allclose(h.probs(Xc), h2.probs(Xc), atol=1e-5)
    with pytest.raises(ValueError):
        fit_head(Xc[:4], yc[:4], K=3, kind="lda")


def test_temperature_comes_from_out_of_fold_scores():
    """A head that separates its own calibration set perfectly must not become overconfident:
    the temperature is fit on out-of-fold scores, so held-out confidence stays sane."""
    Xc, yc = planted(80, 256, 4, noise=4.0, seed=5)     # d > n: ridge interpolates in-sample
    h = fit_head(Xc, yc, K=4, kind="ridge")
    Xt, yt = planted(300, 256, 4, noise=4.0, seed=6)
    p = h.probs(Xt)
    conf = p.max(axis=1).mean()
    acc = np.mean(p.argmax(axis=1) == yt)
    assert abs(conf - acc) < 0.25
