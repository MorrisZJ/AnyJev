"""Closed-form decision heads on hidden states.

The raw readout is itself a linear head: the label logits are the lm_head rows of the label
tokens applied to the last-position hidden state. These heads read the same hidden state
through a matrix fit for one question on a small labelled set, in closed form: no gradients,
no weight updates, seconds on a CPU. Training-free in the sense the pruning and quantization
literature uses the word (a calibration set and a solve, as in SparseGPT or GPTQ).

Heads (all return scores [N, K]; probabilities come from a temperature fit on out-of-fold
scores of the calibration set, so the temperature already accounts for the head's own
optimism on the data it was fit on):

- `diff_means`: score_c = h . (mu_c - mu), the mass-mean probe. One pass over the data, no
  hyperparameters, hard to overfit.
- `lda`: Fisher's discriminant with a shrunk within-class covariance, (1 - a) S + a tr(S)/d I.
  Rank K - 1, proper class posteriors under the Gaussian model.
- `ridge`: least squares onto one-hot labels with an L2 penalty, solved in the dual (an N x N
  system, so hidden width does not matter).
- `rrr`: reduced-rank regression, the ridge solution projected onto the top r singular
  directions of its fitted values (r <= K - 1); ridge and LDA meet in the middle.

`fit_head` picks the layer, the hyperparameters and the temperature by k-fold cross-validation
on the calibration set alone; `LinearHead.probs` is the deployment call.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anyjev.calibrate.posthoc import TemperatureScaler

EPS = 1e-12
KINDS = ("diff_means", "lda", "ridge", "rrr")
DEFAULT_GRID: Dict[str, List[Dict[str, Any]]] = {
    "diff_means": [{}],
    "lda": [{"shrinkage": a} for a in (0.3, 0.6, 0.9)],
    "ridge": [{"lam": lam} for lam in (1e-1, 1.0, 10.0, 100.0)],
    "rrr": [{"lam": lam, "rank": r} for lam in (1.0, 10.0) for r in (1, 2, 4)],   # replaced per K in fit_head
}


def default_grid(kind: str, K: int) -> List[Dict[str, Any]]:
    """The search grid for a head; rrr ranks are relative to K (a K=20 question needs more than rank 4)."""
    if kind == "rrr":
        ranks = sorted({max(1, K // 4), max(1, K // 2), max(1, K - 1)})
        return [{"lam": lam, "rank": r} for lam in (1.0, 10.0) for r in ranks]
    return DEFAULT_GRID[kind]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onehot(y: Sequence[int], K: int) -> np.ndarray:
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), np.asarray(y)] = 1.0
    return Y


# ---------------------------------------------------------------- solvers (standardised X in, (W, b) out)
def solve_diff_means(X: np.ndarray, y: Sequence[int], K: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    mu = X.mean(axis=0)
    W = np.stack([(X[y == c].mean(axis=0) if np.any(y == c) else mu) - mu for c in range(K)], axis=1)
    return W, np.zeros(K)


def solve_lda(X: np.ndarray, y: Sequence[int], K: int, shrinkage: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """Shrunk within-class covariance S = (1 - a) R'R/(n-1) + a tau I, inverted through the
    Woodbury identity so the solve is n x n rather than d x d (n labelled states, d hidden)."""
    y = np.asarray(y)
    n, d = X.shape
    means = np.stack([X[y == c].mean(axis=0) if np.any(y == c) else X.mean(axis=0) for c in range(K)])
    R = X - means[y]
    tau = float((R * R).sum() / max(1, n - 1) / d)        # tr(S_w) / d
    a = float(np.clip(shrinkage, 1e-3, 1.0))
    M = means.T                                           # [d, K]
    if a >= 1.0:
        W = M / (a * tau)
    else:
        c = (1.0 - a) / max(1, n - 1)
        inner = (a * tau) * np.eye(n) + c * (R @ R.T)     # [n, n]
        W = (M - c * R.T @ np.linalg.solve(inner, R @ M)) / (a * tau)
    prior = np.array([max(1, np.sum(y == c)) for c in range(K)], dtype=float)
    b = -0.5 * np.einsum("cd,dc->c", means, W) + np.log(prior / prior.sum())
    return W, b


def solve_ridge(X: np.ndarray, y: Sequence[int], K: int, lam: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    Y = _onehot(y, K)
    xm, ym = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - xm, Y - ym
    A = Xc @ Xc.T + lam * np.eye(len(Xc))                 # dual: N x N
    W = Xc.T @ np.linalg.solve(A, Yc)                     # [d, K]
    return W, ym - xm @ W


def solve_rrr(X: np.ndarray, y: Sequence[int], K: int, lam: float = 1.0, rank: int = 2
              ) -> Tuple[np.ndarray, np.ndarray]:
    W, b = solve_ridge(X, y, K, lam)
    fitted = (X - X.mean(axis=0)) @ W
    _, _, Vt = np.linalg.svd(fitted, full_matrices=False)
    V = Vt[:min(rank, K), :].T                            # [K, r]
    Wr = W @ V @ V.T
    return Wr, _onehot(y, K).mean(axis=0) - X.mean(axis=0) @ Wr


SOLVERS = {"diff_means": solve_diff_means, "lda": solve_lda, "ridge": solve_ridge, "rrr": solve_rrr}


# ---------------------------------------------------------------- the head
@dataclass
class LinearHead:
    kind: str
    layer: int                     # index into the backend's hidden-state layers, as passed to fit_head
    W: np.ndarray                  # [d, K]
    b: np.ndarray                  # [K]
    mean: np.ndarray               # feature standardisation, from the calibration set
    scale: np.ndarray
    temperature: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    n_calib: int = 0
    cv: Dict[str, float] = field(default_factory=dict)

    def scores(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.W + self.b

    def probs(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.scores(X) / self.temperature)

    def to_dict(self, compact: bool = True) -> Dict[str, Any]:
        """JSON-serialisable head. `compact` (default) stores the arrays as base64 float32
        (exact, about ten times smaller than number lists); `compact=False` writes plain lists.
        `from_dict` reads both."""
        enc = encode_array if compact else (lambda a: a.astype(np.float32).tolist())
        return {"method": f"head:{self.kind}", "layer": self.layer, "temperature": self.temperature,
                "params": self.params, "n_calib": self.n_calib, "cv": self.cv,
                "W": enc(self.W), "b": self.b.astype(np.float32).tolist(), "mean": enc(self.mean),
                "scale": enc(self.scale)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinearHead":
        return cls(kind=d["method"].split(":", 1)[1], layer=int(d["layer"]), W=decode_array(d["W"]),
                   b=decode_array(d["b"]), mean=decode_array(d["mean"]), scale=decode_array(d["scale"]),
                   temperature=float(d["temperature"]), params=dict(d.get("params", {})),
                   n_calib=int(d.get("n_calib", 0)), cv=dict(d.get("cv", {})))


def encode_array(a: np.ndarray) -> Dict[str, Any]:
    """{"dtype", "shape", "b64"}: float32 bytes, base64, row-major. Exact for float32 heads."""
    a32 = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"dtype": "float32", "shape": list(a32.shape), "b64": base64.b64encode(a32.tobytes()).decode("ascii")}


def decode_array(x: Any) -> np.ndarray:
    """The inverse of `encode_array`; plain (nested) number lists are accepted as well."""
    if isinstance(x, dict) and "b64" in x:
        a = np.frombuffer(base64.b64decode(x["b64"]), dtype=np.dtype(x.get("dtype", "float32")))
        return a.reshape(x["shape"]).astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _standardise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    return mean, scale


def _folds(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(idx, min(k, n))]


def _oof_scores(kind: str, X: np.ndarray, y: np.ndarray, K: int, params: Dict[str, Any],
                folds: List[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(y), K))
    for f in folds:
        train = np.setdiff1d(np.arange(len(y)), f)
        mean, scale = _standardise(X[train])
        W, b = SOLVERS[kind]((X[train] - mean) / scale, y[train], K, **params)
        out[f] = ((X[f] - mean) / scale) @ W + b
    return out


def fit_head(features: np.ndarray, labels: Sequence[int], K: int, kind: str = "lda",
             grid: Optional[List[Dict[str, Any]]] = None, n_folds: int = 5, seed: int = 0) -> LinearHead:
    """features: [N, L, d] (L candidate layers) or [N, d]. Chooses the layer and the
    hyperparameters by out-of-fold NLL after temperature scaling on the calibration set,
    then refits on all of it. The temperature is the one fit on the out-of-fold scores."""
    if kind not in SOLVERS:
        raise ValueError(f"unknown head {kind!r}; choose from {KINDS}")
    F = np.asarray(features, dtype=np.float64)
    if F.ndim == 2:
        F = F[:, None, :]
    y = np.asarray(labels)
    if len(y) < 2 * K or len(y) < 8:
        raise ValueError(f"need at least max(8, 2K) labelled examples for a head, got {len(y)} for K={K}")
    grid = grid if grid is not None else default_grid(kind, K)
    folds = _folds(len(y), n_folds, seed)
    best = None
    for layer in range(F.shape[1]):
        X = F[:, layer, :]
        for params in grid:
            if kind == "rrr" and params.get("rank", 1) > K:
                continue
            try:
                oof = _oof_scores(kind, X, y, K, params, folds)
            except np.linalg.LinAlgError:
                continue
            scaler = TemperatureScaler.fit(_softmax(oof), y)
            p = scaler.apply(_softmax(oof))
            nll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, None))))
            acc = float(np.mean(np.argmax(oof, axis=1) == y))
            if best is None or nll < best[0]:
                best = (nll, acc, layer, params, scaler.temperature)
    if best is None:
        raise np.linalg.LinAlgError("every configuration failed to solve")
    nll, acc, layer, params, temperature = best
    X = F[:, layer, :]
    mean, scale = _standardise(X)
    W, b = SOLVERS[kind]((X - mean) / scale, y, K, **params)
    return LinearHead(kind=kind, layer=layer, W=W, b=b, mean=mean, scale=scale, temperature=temperature,
                      params=dict(params), n_calib=len(y), cv={"oof_nll": nll, "oof_acc": acc})
