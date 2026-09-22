"""L1: temperature scaling (Guo et al., ICML 2017) fit on a labeled set.

Artifacts are small dicts keyed by (model, question key); reusing one across
models is a user error and is not detected here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

EPS = 1e-12


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def nll(logits: np.ndarray, labels: Sequence[int]) -> float:
    lp = _log_softmax(np.asarray(logits, dtype=np.float64))
    return float(-lp[np.arange(len(labels)), np.asarray(labels)].mean())


@dataclass
class TemperatureScaler:
    temperature: float = 1.0
    prior: Optional[np.ndarray] = None       # the [P, K] prior the calibration set was scored with, frozen
    prior_strength: float = 0.0
    prior_method: str = "none"
    n_calib: int = 0

    @classmethod
    def fit(cls, probs: np.ndarray, labels: Sequence[int],
            log_t_range=(-3.0, 3.0), iters: int = 60) -> "TemperatureScaler":
        """Golden-section search on log T minimizing NLL. probs: [N, K]."""
        logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, None))
        lo, hi = log_t_range
        phi = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = nll(logits / np.exp(c), labels), nll(logits / np.exp(d), labels)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = nll(logits / np.exp(c), labels)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = nll(logits / np.exp(d), labels)
        return cls(temperature=float(np.exp((a + b) / 2)))

    def apply(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, None))
        return np.exp(_log_softmax(logits / self.temperature))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"method": "temperature", "temperature": self.temperature,
                               "prior_method": self.prior_method, "prior_strength": self.prior_strength,
                               "n_calib": self.n_calib}
        if self.prior is not None:
            out["prior"] = np.asarray(self.prior, dtype=float).round(8).tolist()
        return out

    @classmethod
    def from_dict(cls, d: Dict) -> "TemperatureScaler":
        return cls(temperature=float(d["temperature"]),
                   prior=np.asarray(d["prior"], dtype=float) if d.get("prior") is not None else None,
                   prior_strength=float(d.get("prior_strength", 0.0)),
                   prior_method=str(d.get("prior_method", "none")),
                   n_calib=int(d.get("n_calib", 0)))
