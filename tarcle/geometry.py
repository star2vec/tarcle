"""Geometry diagnostics for FV families. Pure numpy — stage 2 code: no torch.

All functions take X of shape (n, d) — one vector per parameter k = 0..n-1 —
or a Gram matrix G of shape (n, n).

How the diagnostics separate the hypotheses (see CLAUDE.md A-D and the synthetic
fixtures in tarcle.synthetic):

- circulant_score: strict mod-n class test on G, wraparound entries included.
  High for circle/helix — but ALSO for hypothesis-D simplex geometry, whose Gram
  (identity + constant) is trivially circulant. Never read it alone.
- toeplitz_score: same test with |i-j| classes and no wraparound identification.
  High toeplitz + low circulant = open curve with banded decay (arc), the other
  false-positive mode.
- spectral_concentration / significant_frequencies: the A-vs-D separator. A
  circle puts all non-DC power in one frequency pair, a helix in a few; a
  simplex spreads it uniformly (no distinguished frequency).
- closure_ratio: open (line/arc) vs closed. Note a simplex is equidistant, so it
  "closes" too — this kills open-curve readings, not D.
- fit_rotation: Procrustes R with X[k] -> X[k+1] (wraparound pair excluded),
  then R^n =~ I. Deliberately CANNOT separate A from D: the cyclic permutation
  of simplex anchors is also orthogonal with order n. Documents that limitation.
- norm_cv, additivity_residual: linear-code signatures (constant norms fail,
  additivity holds, respectively, for a line; the reverse for A/D).
- participation_ratio: effective dimensionality. Circle =~ 2, F-frequency helix
  =~ 2F, line =~ 1, simplex =~ n-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gram(X: np.ndarray, normalize: bool = False) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if normalize:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def _class_score(G: np.ndarray, labels: np.ndarray) -> float:
    """1 - ||G - classwise_mean(G)||_F / ||G - mean(G)||_F, in [~0, 1].

    1.0 means G is exactly constant within each label class. The denominator
    removes the global mean so a large DC component cannot inflate the score.
    """
    G = np.asarray(G, dtype=np.float64)
    projected = np.zeros_like(G)
    for lab in np.unique(labels):
        mask = labels == lab
        projected[mask] = G[mask].mean()
    denom = np.linalg.norm(G - G.mean())
    if denom == 0:
        return 1.0
    return 1.0 - np.linalg.norm(G - projected) / denom


def _mod_labels(n: int) -> np.ndarray:
    i, j = np.indices((n, n))
    return (i - j) % n


def circulant_score(G: np.ndarray) -> float:
    """Strict mod-n test: does G_ij depend only on (i-j) mod n, wraparound
    entries included? (High for a simplex too — check the spectrum.)"""
    return _class_score(G, _mod_labels(len(G)))


def toeplitz_score(G: np.ndarray) -> float:
    """Banded test: does G_ij depend only on |i-j|? An open curve with smooth
    decay passes this while failing the strict circulant test."""
    i, j = np.indices((len(G),) * 2)
    return _class_score(G, np.abs(i - j))


def circulant_profile(G: np.ndarray) -> np.ndarray:
    """c_r = mean of G over the class (i-j) mod n = r, r = 0..n-1."""
    labels = _mod_labels(len(G))
    return np.array([G[labels == r].mean() for r in range(len(G))])


def spectrum(G: np.ndarray) -> np.ndarray:
    """Power |DFT(c)|^2 of the circulant profile, per frequency 0..n-1. For a
    truly circulant G these are its eigenvalue magnitudes squared."""
    return np.abs(np.fft.fft(circulant_profile(G))) ** 2


def frequency_pair_powers(G: np.ndarray) -> dict[int, float]:
    """Non-DC power folded into conjugate pairs: f = 1..n//2 -> power(f) +
    power(n-f) (f = n/2 unpaired when n is even)."""
    power = spectrum(G)
    n = len(power)
    pairs = {}
    for f in range(1, n // 2 + 1):
        pairs[f] = power[f] + (power[n - f] if f != n - f else 0.0)
    return pairs


def spectral_concentration(G: np.ndarray) -> float:
    """Fraction of non-DC power in the single strongest frequency pair.
    Circle -> ~1. Simplex -> ~uniform (~1/(n//2)): trivial frequency content."""
    pairs = frequency_pair_powers(G)
    total = sum(pairs.values())
    return max(pairs.values()) / total if total > 0 else 0.0


def significant_frequencies(G: np.ndarray, factor: float = 2.0) -> list[int]:
    """Frequencies whose pair power exceeds `factor` x the uniform share of
    non-DC power. Empty list = no distinguished frequency (the hypothesis-D
    signature, even when circulant_score is high). Resolution is limited by the
    number of pairs (n//2), so small n cannot flag many frequencies at once."""
    pairs = frequency_pair_powers(G)
    total = sum(pairs.values())
    if total == 0:
        return []
    uniform = total / len(pairs)
    return [f for f, p in sorted(pairs.items()) if p > factor * uniform]


def closure_ratio(X: np.ndarray) -> float:
    """||X[n-1] - X[0]|| relative to the mean consecutive step. ~1 for any
    closed/equidistant family (circle, helix, simplex); ~n-1 for a line."""
    X = np.asarray(X, dtype=np.float64)
    steps = np.linalg.norm(np.diff(X, axis=0), axis=1)
    return float(np.linalg.norm(X[-1] - X[0]) / steps.mean())


@dataclass
class RotationFit:
    residual: float  # relative Procrustes residual of X[k] -> X[k+1]
    order_error: float  # ||R^n - I||_F / sqrt(rank)
    wraparound_error: float  # ||R X[n-1] - X[0]|| / mean step (R never saw this pair)
    rank: int  # dimensionality of the fitted subspace


def fit_rotation(X: np.ndarray, center: bool = True, sv_ratio: float = 0.1) -> RotationFit:
    """Best orthogonal R minimizing sum_k ||R X[k] - X[k+1]|| for k = 0..n-2
    (wraparound pair excluded, so order_error and wraparound_error are genuine
    tests, not fitted). The fit lives in the principal subspace of centered X
    (singular values > sv_ratio * largest); in the full ambient space R would be
    arbitrary on the unused dimensions and R^n meaningless."""
    X = np.asarray(X, dtype=np.float64)
    Y = X - X.mean(axis=0) if center else X
    _, s, Vt = np.linalg.svd(Y, full_matrices=False)
    basis = Vt[s > sv_ratio * s[0]]
    Z = Y @ basis.T
    A, B = Z[:-1], Z[1:]
    U, _, Wt = np.linalg.svd(A.T @ B)
    R = Wt.T @ U.T  # maximizes tr(R @ A.T @ B): R a_k =~ b_k
    n, r = Z.shape
    steps = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    return RotationFit(
        residual=float(np.linalg.norm(A @ R.T - B) / np.linalg.norm(B)),
        order_error=float(
            np.linalg.norm(np.linalg.matrix_power(R, n) - np.eye(r)) / np.sqrt(r)
        ),
        wraparound_error=float(np.linalg.norm(Z[-1] @ R.T - Z[0]) / steps.mean()),
        rank=r,
    )


def norm_profile(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(X, dtype=np.float64), axis=1)


def norm_cv(X: np.ndarray) -> float:
    """Coefficient of variation of ||X[k]||: ~0 for circle/simplex, large for a
    line through the origin."""
    norms = norm_profile(X)
    return float(norms.std() / norms.mean())


def additivity_residual(X: np.ndarray) -> float:
    """Word-offset additivity relative to base k=0: does
    (X[j]-X[0]) + (X[k]-X[0]) =~ X[j+k]-X[0] for j,k >= 1, j+k <= n-1?
    Exactly 0 for a line (even affine); fails for circle and simplex.
    Non-wrapping sums only — wraparound is closure_ratio's job."""
    X = np.asarray(X, dtype=np.float64)
    D = X - X[0]
    n = len(X)
    residuals = [
        np.linalg.norm(D[j] + D[k] - D[j + k])
        for j in range(1, n)
        for k in range(1, n - j)
    ]
    scale = np.linalg.norm(D[1:], axis=1).mean()
    return float(np.mean(residuals) / scale)


def participation_ratio(X: np.ndarray, center: bool = True) -> float:
    """(sum lambda)^2 / sum lambda^2 of the covariance spectrum: effective
    dimensionality. Circle ~2, F-frequency helix ~2F, line ~1, simplex ~n-1."""
    X = np.asarray(X, dtype=np.float64)
    Y = X - X.mean(axis=0) if center else X
    lam = np.linalg.svd(Y, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam**2).sum())


def diagnose(X: np.ndarray) -> dict:
    """All diagnostics at once, for tables and result files."""
    G = gram(X)
    rot = fit_rotation(X)
    return {
        "circulant_score": circulant_score(G),
        "toeplitz_score": toeplitz_score(G),
        "spectral_concentration": spectral_concentration(G),
        "significant_frequencies": significant_frequencies(G),
        "closure_ratio": closure_ratio(X),
        "rotation_residual": rot.residual,
        "rotation_order_error": rot.order_error,
        "rotation_wraparound_error": rot.wraparound_error,
        "rotation_rank": rot.rank,
        "norm_cv": norm_cv(X),
        "additivity_residual": additivity_residual(X),
        "participation_ratio": participation_ratio(X),
    }
