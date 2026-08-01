"""Synthetic FV-family fixtures with known geometry. Pure numpy.

Each generator returns X of shape (n, d): one vector per parameter value
k = 0..n-1, embedded in a random subspace of R^d. These are the ground-truth
shapes the geometry diagnostics must separate:

- circle:  hypothesis A (single-frequency closed loop)
- helix:   multi-irrep variant of A (several frequencies)
- line:    linear code, k * v (hypothesis B-style task selector)
- simplex: hypothesis D (near-orthogonal anchors + shared offset). Its Gram is
  identity + constant — *trivially* circulant with no distinguished frequency:
  the false-positive mode the circulant test alone cannot reject.
- arc:     open curve whose Gram decays with |i-j| (banded/Toeplitz) but has no
  mod-n wraparound — the false positive for a sloppy circulant check.

`noise` is the expected norm of the per-vector gaussian perturbation.
"""
from __future__ import annotations

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_orthonormal(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """(d, m) matrix with orthonormal columns."""
    if m > d:
        raise ValueError(f"cannot fit {m} orthonormal directions in R^{d}")
    q, _ = np.linalg.qr(rng.standard_normal((d, m)))
    return q[:, :m]


def _add_noise(X: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    if noise == 0:
        return X
    d = X.shape[1]
    return X + noise * rng.standard_normal(X.shape) / np.sqrt(d)


def circle(
    n: int, d: int = 16, freq: int = 1, radius: float = 1.0,
    offset: float = 0.0, noise: float = 0.0, seed: int = 0,
) -> np.ndarray:
    rng = _rng(seed)
    basis = random_orthonormal(d, 3, rng)
    theta = 2 * np.pi * freq * np.arange(n) / n
    X = radius * (np.outer(np.cos(theta), basis[:, 0]) + np.outer(np.sin(theta), basis[:, 1]))
    X += offset * basis[:, 2]
    return _add_noise(X, noise, rng)


def helix(
    n: int, d: int = 16, freqs: tuple[int, ...] = (1, 2),
    amps: tuple[float, ...] | None = None, offset: float = 0.0,
    noise: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """Sum of circles at several frequencies, each in its own orthogonal plane."""
    if amps is None:
        amps = tuple(1.0 for _ in freqs)
    rng = _rng(seed)
    basis = random_orthonormal(d, 2 * len(freqs) + 1, rng)
    X = np.zeros((n, d))
    for i, (f, a) in enumerate(zip(freqs, amps)):
        theta = 2 * np.pi * f * np.arange(n) / n
        X += a * (
            np.outer(np.cos(theta), basis[:, 2 * i])
            + np.outer(np.sin(theta), basis[:, 2 * i + 1])
        )
    X += offset * basis[:, -1]
    return _add_noise(X, noise, rng)


def arc(
    n: int, d: int = 16, span: float = np.pi, radius: float = 1.0,
    noise: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """Open curve: n points on a circular arc of total angle `span` < 2*pi.
    Gram depends on |i-j| (Toeplitz) but the loop never closes."""
    rng = _rng(seed)
    basis = random_orthonormal(d, 2, rng)
    theta = span * np.arange(n) / (n - 1)
    X = radius * (np.outer(np.cos(theta), basis[:, 0]) + np.outer(np.sin(theta), basis[:, 1]))
    return _add_noise(X, noise, rng)


def line(
    n: int, d: int = 16, step: float = 1.0, offset: float = 0.0,
    noise: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """X[k] = offset * u + k * step * v, u orthogonal to v."""
    rng = _rng(seed)
    basis = random_orthonormal(d, 2, rng)
    X = np.outer(step * np.arange(n), basis[:, 0]) + offset * basis[:, 1]
    return _add_noise(X, noise, rng)


def simplex(
    n: int, d: int | None = None, offset: float = 0.7,
    noise: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """Orthonormal anchors u_k plus a shared offset orthogonal to all of them.
    Gram = I + offset^2 * ones: exactly circulant, no distinguished frequency."""
    if d is None:
        d = n + 1
    rng = _rng(seed)
    basis = random_orthonormal(d, n + 1, rng)
    X = basis[:, :n].T + offset * basis[:, n]
    return _add_noise(X, noise, rng)
