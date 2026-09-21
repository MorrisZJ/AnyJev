"""L0 prior estimation and correction.

Two ways to estimate the model's prior over the labels, both label-free:

  batch (Zhou et al., ICLR 2024, default): the mean predicted distribution
      over a batch of real inputs, kept running per question across calls.
      Low variance in the bench: +1 to +2 accuracy points and a large ECE
      improvement on every model; needs a handful of inputs before it kicks
      in (min_prior_n), and assumes the label marginal is not extreme.
  content_free (Zhao et al., ICML 2021, opt-in): the distribution on
      content-free inputs ("N/A", "", "[MASK]"). Depends only on the question,
      cached per question. High variance in the bench: +8 to +12 points on a
      prompt-injection noul, but -3 on ordinal scores and -9 on one model's
      noul questions, because for some questions the model's answer to an
      empty input is an honest answer, not a label prior.

Correction is the same in both cases: divide by the prior and renormalize.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PROBES = ("N/A", "", "[MASK]")
EPS = 1e-8


def content_free_prior(cf_probs: np.ndarray) -> np.ndarray:
    """cf_probs: [..., C, K] distributions over K positions for C probes.
    Returns the mean prior [..., K]."""
    cf = np.asarray(cf_probs, dtype=np.float64)
    prior = cf.mean(axis=-2)
    prior = np.clip(prior, EPS, None)
    return prior / prior.sum(axis=-1, keepdims=True)


def batch_prior(p_batch: np.ndarray) -> np.ndarray:
    """p_batch: [N, ..., K] distributions over real inputs. Returns [..., K]."""
    p = np.asarray(p_batch, dtype=np.float64).mean(axis=0)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=-1, keepdims=True)


def apply_contextual(p_raw: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """p_raw, prior: [..., K]. Returns normalize(p_raw / prior)."""
    p = np.asarray(p_raw, dtype=np.float64) / np.clip(np.asarray(prior, dtype=np.float64), EPS, None)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=-1, keepdims=True)
