"""Metrics: the ones every README claims and almost nobody reports."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def accuracy(probs: np.ndarray, labels: Sequence[int]) -> float:
    return float(np.mean(np.argmax(probs, 1) == np.asarray(labels)))


def macro_f1(probs: np.ndarray, labels: Sequence[int]) -> float:
    pred = np.argmax(probs, 1)
    y = np.asarray(labels)
    f1s = []
    for c in np.unique(np.concatenate([y, pred])):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def brier(probs: np.ndarray, labels: Sequence[int]) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), np.asarray(labels)] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs: np.ndarray, labels: Sequence[int]) -> float:
    p = np.clip(probs[np.arange(len(labels)), np.asarray(labels)], 1e-12, None)
    return float(-np.mean(np.log(p)))


def ece(probs: np.ndarray, labels: Sequence[int], n_bins: int = 15) -> float:
    """Expected calibration error on top-1 confidence, equal-mass bins."""
    conf = np.max(probs, 1)
    correct = (np.argmax(probs, 1) == np.asarray(labels)).astype(float)
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    n = len(conf)
    total = 0.0
    for chunk in np.array_split(np.arange(n), min(n_bins, n)):
        if len(chunk) == 0:
            continue
        total += len(chunk) / n * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(total)


def reliability(probs: np.ndarray, labels: Sequence[int], n_bins: int = 10):
    """Equal-width bins for plotting: (bin_conf, bin_acc, bin_count)."""
    conf = np.max(probs, 1)
    correct = (np.argmax(probs, 1) == np.asarray(labels)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows


def flip_rate(probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    """Fraction of items whose argmax differs between two readouts of the same
    items (e.g. original vs reversed option order)."""
    return float(np.mean(np.argmax(probs_a, 1) != np.argmax(probs_b, 1)))


def coverage_risk(probs: np.ndarray, labels: Sequence[int]):
    """Sort by confidence descending; risk(coverage) = error rate on the kept prefix."""
    conf = np.max(probs, 1)
    correct = (np.argmax(probs, 1) == np.asarray(labels)).astype(float)
    order = np.argsort(-conf)
    err = 1.0 - np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    cov = np.arange(1, len(conf) + 1) / len(conf)
    return cov, err


def coverage_at_risk(probs: np.ndarray, labels: Sequence[int], target: float = 0.05) -> float:
    """Largest coverage whose empirical risk is at or below target."""
    cov, err = coverage_risk(probs, labels)
    ok = np.where(err <= target)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def aurc(probs: np.ndarray, labels: Sequence[int]) -> float:
    cov, err = coverage_risk(probs, labels)
    trap = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed it
    return float(trap(err, cov))


def summarize(probs: np.ndarray, labels: Sequence[int],
              probs_flipped: np.ndarray = None) -> Dict[str, float]:
    out = {
        "n": int(len(labels)),
        "acc": accuracy(probs, labels),
        "macro_f1": macro_f1(probs, labels),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "ece": ece(probs, labels),
        "cov@5%": coverage_at_risk(probs, labels, 0.05),
        "aurc": aurc(probs, labels),
    }
    if probs_flipped is not None:
        out["flip"] = flip_rate(probs, probs_flipped)
    return out
