"""Smoke tests for tarcle.floors (prereg_instruments §5). Small fixtures only —
the full table runs on saved artifacts; here we assert the machinery: the
percentile plumbing, the D42 ratio rules, and that a planted circle beats a
matched no-circle null on the harmonic detector at toy size.
"""
from __future__ import annotations

import numpy as np

from tarcle import synthetic as S
from tarcle.floors import (
    floors, measure, planted_circle, ratio_cell, stats_n1, stats_n2,
    synth_matched_n2,
)

KS = list(range(1, 12))
N = 12


def _toy_regime() -> dict:
    X = S.line(11, d=32, step=1.0, offset=8.0, noise=0.3, seed=0)
    return measure(X, KS)


def test_stats_groups_cover_registered_rows():
    X = S.circle(11, d=32, noise=0.1, seed=0)
    n1, n2 = stats_n1(X, KS, N), stats_n2(X, KS, N)
    assert {"toeplitz", "perm_z", "seam_cut_margin"} <= n1.keys()
    assert {"circulant_raw", "circulant_centered", "spectral_concentration",
            "harmonic_partial_r2", "seam_cyclic_cvr2"} <= n2.keys()


def test_participation_ratio_invariant_under_row_permutation():
    # The D42 note: row order does not enter the covariance spectrum, so the
    # real-N1 "range" for PR is degenerate at the observed value.
    X = S.helix(11, d=32, noise=0.2, seed=1)
    rng = np.random.default_rng(0)
    a = stats_n1(X, KS, N)["participation_ratio"]
    b = stats_n1(X[rng.permutation(len(X))], KS, N)["participation_ratio"]
    assert abs(a - b) < 1e-12


def test_floors_percentiles_ordered():
    rng = np.random.default_rng(0)
    draws = [{"s": float(v)} for v in rng.standard_normal(200)]
    f = floors(draws)["s"]
    assert f["p05"] <= f["mean"] <= f["p95"] <= f["max"]


def test_ratio_cell_d42_rules():
    assert ratio_cell(0.30, 0.003)[1] == 100.0
    assert ratio_cell(0.30, 0.0)[1] == float("inf")   # ~0/X: the D37 situation
    assert ratio_cell(0.0001, 0.0)[1] is None         # both below resolution


def test_planted_circle_beats_matched_null_on_harmonic():
    p = _toy_regime()
    rng = np.random.default_rng(0)
    null = [stats_n2(synth_matched_n2(p, KS, rng), KS, N)["harmonic_partial_r2"]
            for _ in range(30)]
    planted = stats_n2(planted_circle(p, KS, N, np.random.default_rng(1)),
                       KS, N)["harmonic_partial_r2"]
    assert planted > np.percentile(null, 95)
