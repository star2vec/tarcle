"""Geometry diagnostics must separate the known synthetic shapes.

Each test encodes one separation story, including the two false-positive traps:
- a simplex (hypothesis D) is trivially circulant -> only the spectrum rejects it
- an arc (banded |i-j| decay) passes a sloppy Toeplitz check -> only the strict
  mod-n test rejects it
Thresholds carry wide margins around values measured at noise=0.05 over 3 seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from tarcle import synthetic as S
from tarcle.geometry import (
    additivity_residual,
    circulant_score,
    closure_ratio,
    fit_rotation,
    gram,
    norm_cv,
    participation_ratio,
    significant_frequencies,
    spectral_concentration,
    toeplitz_score,
)

N, D = 12, 16
SEEDS = (0, 1, 2)
NOISES = (0.0, 0.05)


def _cases(gen, **kwargs):
    return [gen(N, noise=noise, seed=seed, **kwargs) for noise in NOISES for seed in SEEDS]


def circles():
    return _cases(S.circle, d=D) + _cases(S.circle, d=D, offset=0.8)


def helices():
    return _cases(S.helix, d=D, freqs=(1, 2))


def lines():
    return _cases(S.line, d=D)


def simplices():
    return _cases(S.simplex, offset=0.7)


def arcs():
    return _cases(S.arc, d=D)


def test_circulant_gate():
    for X in circles() + helices() + simplices():
        assert circulant_score(gram(X)) > 0.85
    for X in lines():
        assert circulant_score(gram(X)) < 0.1
    for X in arcs():
        assert circulant_score(gram(X)) < 0.3


def test_trivially_circulant_simplex_caught_by_spectrum():
    """THE hypothesis-A-vs-D separation: a simplex Gram (identity + constant)
    passes the circulant test, so the circulant score alone cannot support
    hypothesis A. Non-trivial frequency content is what rejects D."""
    for X in simplices():
        G = gram(X)
        assert circulant_score(G) > 0.85  # the false positive, demonstrated
        assert significant_frequencies(G) == []  # no distinguished frequency
        assert spectral_concentration(G) < 0.3  # ~uniform (1/6 here)
    for X in circles():
        G = gram(X)
        assert circulant_score(G) > 0.85
        assert significant_frequencies(G) == [1]
        assert spectral_concentration(G) > 0.9
    for X in helices():
        G = gram(X)
        assert significant_frequencies(G) == [1, 2]
        assert 0.4 < spectral_concentration(G) < 0.6  # two equal-power pairs


def test_circle_frequency_is_read_correctly():
    for noise in NOISES:
        G = gram(S.circle(N, D, freq=2, noise=noise, seed=0))
        assert significant_frequencies(G) == [2]


def test_banded_decay_arc_fails_strict_mod_n():
    """An open curve with smooth |i-j| similarity decay looks 'vaguely
    circulant'. The strict test checks (i-j) mod n including wraparound
    entries, which is exactly where the arc deviates."""
    for X in arcs():
        G = gram(X)
        assert toeplitz_score(G) > 0.95
        assert circulant_score(G) < 0.3
    for X in circles():
        G = gram(X)
        assert toeplitz_score(G) > 0.95
        assert circulant_score(G) > 0.95


def test_spectrum_meaningless_without_circulant_gate():
    """A line's class-averaged profile happens to concentrate spectral power at
    f=1 — reading the spectrum before the circulant gate would mislabel a
    linear code as a circle. The gate must come first."""
    for X in lines():
        G = gram(X)
        assert spectral_concentration(G) > 0.5  # the trap...
        assert circulant_score(G) < 0.1  # ...and why it never triggers


def test_closure():
    for X in circles() + helices() + simplices():
        assert closure_ratio(X) < 1.2
    for X in lines():
        assert closure_ratio(X) > 5
    for X in arcs():
        assert closure_ratio(X) > 3


def test_rotation_fit_cannot_separate_circle_from_simplex():
    """R^n =~ I holds for circle AND simplex (a cyclic permutation of anchors is
    orthogonal with order n) — so the rotation fit supports 'cyclic structure'
    generally, never hypothesis A specifically. It does reject open shapes:
    the wraparound pair is excluded from the fit, so closure is predicted,
    not fitted."""
    for X in circles() + helices() + simplices():
        rot = fit_rotation(X)
        assert rot.residual < 0.1
        assert rot.order_error < 0.15
        assert rot.wraparound_error < 0.2
    for X in lines():
        rot = fit_rotation(X)
        assert rot.residual > 0.25
        assert rot.wraparound_error > 5
    for X in arcs():
        rot = fit_rotation(X)
        assert rot.order_error > 1.0
        assert rot.wraparound_error > 3


def test_linear_code_signatures():
    """Norms and additivity: a line has growing norms and exact base-relative
    additivity; closed shapes have constant norms and broken additivity."""
    for X in lines():
        assert norm_cv(X) > 0.3
        assert additivity_residual(X) < 0.05
    for X in circles() + simplices():
        assert norm_cv(X) < 0.05
        assert additivity_residual(X) > 0.5


def test_participation_ratio_reads_effective_dimension():
    for X in circles():
        assert participation_ratio(X) < 2.5
    for X in helices():
        assert 3.5 < participation_ratio(X) < 4.5
    for X in lines():
        assert participation_ratio(X) < 1.2
    for X in simplices():
        assert participation_ratio(X) > N - 2


def test_gram_normalize():
    G = gram(S.circle(N, D, radius=3.0, seed=0), normalize=True)
    assert np.allclose(np.diag(G), 1.0)
    assert np.all(G <= 1.0 + 1e-12)


def test_stage2_modules_import_without_torch():
    """CLAUDE.md: stage-2 (geometry/analysis) code must never import torch."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['torch'] = None; "
        "import tarcle.geometry, tarcle.synthetic"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_fixture_shapes_and_determinism():
    for gen in (S.circle, S.helix, S.arc, S.line):
        X = gen(N, d=D, seed=5)
        assert X.shape == (N, D)
        assert np.array_equal(X, gen(N, d=D, seed=5))
    X = S.simplex(N, seed=5)
    assert X.shape == (N, N + 1)
    with pytest.raises(ValueError, match="orthonormal"):
        S.simplex(N, d=N - 1)
