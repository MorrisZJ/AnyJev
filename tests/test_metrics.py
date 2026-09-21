import numpy as np

from bench import metrics


def test_hand_computed_values():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])
    labels = [0, 1, 1, 0]
    assert metrics.accuracy(probs, labels) == 0.75
    # brier: item2 wrong: (0.6-0)^2+(0.4-1)^2=0.72; others: 0.02, 0.08, 0.18
    assert abs(metrics.brier(probs, labels) - (0.02 + 0.72 + 0.08 + 0.18) / 4) < 1e-9
    assert metrics.flip_rate(probs, probs[:, ::-1]) == 1.0
    assert metrics.flip_rate(probs, probs) == 0.0
    cov, err = metrics.coverage_risk(probs, labels)
    # sorted by conf: 0.9(ok), 0.8(ok), 0.7(ok), 0.6(wrong)
    np.testing.assert_allclose(err, [0, 0, 0, 0.25])
    assert metrics.coverage_at_risk(probs, labels, 0.05) == 0.75
    assert 0.0 <= metrics.aurc(probs, labels) <= 0.25
    s = metrics.summarize(probs, labels, probs[:, ::-1])
    for k in ("acc", "macro_f1", "brier", "nll", "ece", "cov@5%", "aurc", "flip"):
        assert k in s


def test_ece_zero_when_perfectly_calibrated_bins():
    # every item confidence 0.75, exactly 75% correct -> ECE 0
    probs = np.array([[0.75, 0.25]] * 4)
    labels = [0, 0, 0, 1]
    assert abs(metrics.ece(probs, labels, n_bins=1)) < 1e-12
