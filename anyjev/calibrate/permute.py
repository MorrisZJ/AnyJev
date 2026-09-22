"""L0: permutation marginalization for position bias (Zheng et al., ICLR 2024).

Show the options in K cyclic shifts so every option occupies every position
once, then combine the per-option probabilities across shifts.

Two ways to combine:
  "logmean": average log-probabilities (geometric mean), then renormalize.
             If the bias is additive in logit space, logit(i at pos j) = c_i + b_j,
             the per-option average is c_i + mean(b) - mean(log Z_s), so the
             result is softmax(c) exactly: the position bias is removed and the
             result is invariant to how the options were originally listed.
  "mean":    average probabilities, the form used in the paper. Only invariant
             to cyclic rotations of the original list; kept for comparison.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

EPS = 1e-12


def cyclic_shifts(k: int, max_permutations: Optional[int] = None) -> List[List[int]]:
    """perm[j] = original option index shown at position j."""
    n = k if max_permutations is None else max(1, min(k, max_permutations))
    return [[(j + s) % k for j in range(k)] for s in range(n)]


def spread_order(k: int) -> List[int]:
    """The order in which to read cyclic shifts when you may stop early: 0, k/2, k/4, 3k/4, k/8, ...
    (the van der Corput sequence scaled to k). Consecutive shifts move every option by one
    position, so the first few are nearly the same layout; spread shifts put each option in well
    separated positions, so a short prefix of this order already approximates the full marginal."""
    order, seen = [0], {0}
    d = 2
    while len(order) < k and d <= 4 * k:
        for num in range(1, d, 2):
            s = int(k * num / d) % k
            if s not in seen:
                seen.add(s)
                order.append(s)
        d *= 2
    order.extend(s for s in range(k) if s not in seen)   # rounding can skip values
    return order[:k]


def marginalize(p_by_perm: np.ndarray, perms: Sequence[Sequence[int]],
                combine: str = "logmean") -> np.ndarray:
    """p_by_perm: [P, K] distributions indexed by *position*.
    Returns [K] distribution indexed by original option."""
    p_by_perm = np.asarray(p_by_perm, dtype=np.float64)
    P, K = p_by_perm.shape
    per_option = np.zeros((P, K))
    for s, perm in enumerate(perms):
        for j, i in enumerate(perm):
            per_option[s, i] = p_by_perm[s, j]
    if combine == "logmean":
        z = np.log(np.clip(per_option, EPS, None)).mean(axis=0)
        z = z - z.max()
        out = np.exp(z)
    elif combine == "mean":
        out = per_option.mean(axis=0)
    else:
        raise ValueError("combine must be 'logmean' or 'mean'")
    return out / out.sum()


def flip_rate_across_perms(p_by_perm: np.ndarray, perms: Sequence[Sequence[int]]) -> float:
    """Fraction of permutations whose argmax (in option space) disagrees with
    the first permutation's argmax. 0.0 means order-invariant on this item."""
    p_by_perm = np.asarray(p_by_perm)
    winners = [perm[int(np.argmax(p_by_perm[s]))] for s, perm in enumerate(perms)]
    if len(winners) <= 1:
        return 0.0
    return float(np.mean([w != winners[0] for w in winners[1:]]))
