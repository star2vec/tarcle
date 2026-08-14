"""T6: the floor-estimator table. Pure numpy — stage-2 code, no torch.

Registered in docs/preregistration_instruments.md §5 (D39/D41), with the C1
conflation resolved by D42 before the full table was computed. D37 found that
the harmonic-partial-R2 detector, validated on synthetic fixtures at a floor of
~0.003-0.006, has a false-positive floor of ~0.22-0.30 on real function
vectors. This table asks whether that gap is a property of one statistic or of
synthetic validation as a practice.

Two null semantics (prereg §5), assigned per statistic by what its positive
reading asserts:

  N1  "no ordering at all"     for toeplitz, permutation z, seam cut-margin
  N2  "ordered, nothing more"  for circulant (raw, centered),
                               spectral_concentration, harmonic partial R2,
                               seam cyclic cv-R2

Four floor constructions per row (95th percentile, --draws draws, seed 0):

  C1a fixture-grade synthetic  the repo's committed validation practice
      (D42): N2 = cylinder.py's line(d=64, step=1, offset=8, noise=0.3);
      N1 = prereg §0's simplex(offset=0.7, noise=0.3). The registered claim-B
      ratio C4a = C2/C1a is read against this column. No gate: it is the
      reference being audited.
  C1b regime-matched synthetic  offset, axial scale and residual RMS matched
      per dataset (exploratory decomposition, D42): C4b = C2/C1b says how much
      of the gap survives honest second-moment matching.
  C2  real-vector permutation   N1 = row permutation; N2 = residual
      permutation after the [1, k] fit (power.residual_surrogate). The
      operative floor.
  C3  real no-structure families observed: add-k (ordered, non-cyclic) and
      the unrelated null (unordered), plus their own C2-style floors.

participation_ratio is descriptive: null ranges only. Its real-N1 range is
degenerate (row permutation leaves the covariance spectrum untouched), so the
real range shown is N2 (D42).

Validation gate, checked BEFORE any real-data cell is interpreted (D42 rule):
a row is VOID iff its planted positive (full-strength circle for N2 rows,
clean line for N1 rows, at the months regime) fails to exceed the months/todd
C2 floor, or C2 self-exceedance on fresh draws leaves [0.02, 0.10].
Planted-vs-C1b and C1b calibration are reported as findings, not gates.

Claim-B verdict (pre-committed, prereg §5): median C4a across detector rows,
both methods pooled — >= 10 generalises, < 3 does not, else per-statistic.
Cells with |C1a| < 1e-3 and C2 >= 0.01 count as +inf (the D37 situation);
cells with both floors below resolution are excluded.

Usage:
    python -m tarcle.floors [--draws 300] [--seam-folds 40]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from . import synthetic as S
from .geometry import (
    circulant_score, gram, participation_ratio, spectral_concentration,
    toeplitz_score,
)
from .power import partial_r2, residual_surrogate
from .seam import contest
from .stage2 import load, permutation_null
from .synthetic import random_orthonormal

R = Path("results/fv")

DATASETS = [
    # label, path pattern, drop_k0, ks, modulus
    ("months", "ctl_months_primary/fv_primary_{m}.npz", True, list(range(1, 12)), 12),
    ("addk", "ctl_addk/fv_addk_{m}.npz", False, list(range(1, 12)), 12),
    ("unrelated", "ctl_unrelated_12tasks/fv_unrelated_{m}.npz", False,
     list(range(12)), 12),
]

N1_ROWS = ["toeplitz", "perm_z", "seam_cut_margin"]
N2_ROWS = ["circulant_raw", "circulant_centered", "spectral_concentration",
           "harmonic_partial_r2", "seam_cyclic_cvr2"]
GATE_BAND = (0.02, 0.10)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def stats_n1(X: np.ndarray, ks: list[int], n: int) -> dict:
    out = {
        "toeplitz": toeplitz_score(gram(X)),
        "perm_z": permutation_null(X, 200)["z"],
        "participation_ratio": participation_ratio(X),
    }
    c = contest(X, ks, n)
    cyc = c["cyclic"]["cv_r2"]
    best_cut = max(v["cv_r2"] for m, v in c.items() if m != "cyclic")
    out["seam_cut_margin"] = best_cut - cyc
    return out


def stats_n2(X: np.ndarray, ks: list[int], n: int) -> dict:
    kf = np.asarray(ks, dtype=float)
    out = {
        "circulant_raw": circulant_score(gram(X)),
        "circulant_centered": circulant_score(gram(X - X.mean(axis=0))),
        "spectral_concentration": spectral_concentration(gram(X)),
        "harmonic_partial_r2": partial_r2(X, kf, n),
        "participation_ratio": participation_ratio(X),
    }
    out["seam_cyclic_cvr2"] = contest(X, ks, n)["cyclic"]["cv_r2"]
    return out


def measure(X: np.ndarray, ks: list[int]) -> dict:
    """The matched regime: offset, axial scale, residual and total k-RMS."""
    kf = np.asarray(ks, dtype=float)
    M0 = np.stack([np.ones_like(kf), kf], 1)
    B0, *_ = np.linalg.lstsq(M0, X, rcond=None)
    resid = X - M0 @ B0
    return {
        "d": X.shape[1],
        "offset": float(np.linalg.norm(X.mean(axis=0))),
        "slope": float(np.linalg.norm(B0[1])),
        "resid_rms": float(np.sqrt((np.linalg.norm(resid, axis=1) ** 2).mean())),
        "total_rms": float(np.sqrt(
            (np.linalg.norm(X - X.mean(axis=0), axis=1) ** 2).mean())),
    }


def _noise(shape: tuple, scale: float, rng: np.random.Generator) -> np.ndarray:
    return scale * rng.standard_normal(shape) / np.sqrt(shape[1])


def synth_matched_n1(p: dict, npts: int, rng: np.random.Generator) -> np.ndarray:
    u = random_orthonormal(p["d"], 1, rng)
    X = p["offset"] * np.ones((npts, 1)) @ u.T
    return X + _noise(X.shape, p["total_rms"], rng)


def synth_matched_n2(p: dict, ks: list[int], rng: np.random.Generator) -> np.ndarray:
    u = random_orthonormal(p["d"], 2, rng)
    kc = np.asarray(ks, dtype=float)
    kc = kc - kc.mean()
    X = p["offset"] * np.ones((len(ks), 1)) @ u[:, :1].T
    X += p["slope"] * np.outer(kc, u[:, 1])
    return X + _noise(X.shape, p["resid_rms"], rng)


def planted_circle(p: dict, ks: list[int], n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Full-strength circle (radius = total k-RMS) + matched offset + 10%
    noise: the unambiguous positive the N2 rows must detect."""
    u = random_orthonormal(p["d"], 3, rng)
    th = 2 * np.pi * np.asarray(ks, dtype=float) / n
    X = p["offset"] * np.ones((len(ks), 1)) @ u[:, :1].T
    X += p["total_rms"] * (np.outer(np.cos(th), u[:, 1]) + np.outer(np.sin(th), u[:, 2]))
    return X + _noise(X.shape, 0.1 * p["total_rms"], rng)


def planted_line(p: dict, ks: list[int], rng: np.random.Generator) -> np.ndarray:
    """Clean ordered line with an endpoint seam: the positive for N1 rows."""
    u = random_orthonormal(p["d"], 2, rng)
    kc = np.asarray(ks, dtype=float)
    kc = kc - kc.mean()
    X = p["offset"] * np.ones((len(ks), 1)) @ u[:, :1].T
    X += p["slope"] * np.outer(kc, u[:, 1])
    return X + _noise(X.shape, 0.1 * p["total_rms"], rng)


def floors(draws_stats: list[dict]) -> dict:
    out = {}
    for k in draws_stats[0]:
        vals = np.array([d[k] for d in draws_stats])
        vals = vals[np.isfinite(vals)]
        out[k] = {"p95": float(np.percentile(vals, 95)),
                  "p05": float(np.percentile(vals, 5)),
                  "max": float(vals.max()), "mean": float(vals.mean()),
                  "n_finite": int(len(vals))}
    return out


def draw_stats(kind: str, make, ks: list[int], n: int, draws: int,
               seed: int = 0) -> list[dict]:
    """`make(rng) -> X` drawn `draws` times; stats of the given null group."""
    rng = np.random.default_rng(seed)
    fn = stats_n1 if kind == "n1" else stats_n2
    return [fn(make(rng), ks, n) for _ in range(draws)]


def fixture_grade(npts: int, ks: list[int], n: int, draws: int) -> dict:
    """C1a: the repo's committed fixture practice (D42). Seeds vary per draw
    through the generators' own seed argument for exact reproducibility."""
    n1 = [stats_n1(S.simplex(npts, offset=0.7, noise=0.3, seed=s), ks, n)
          for s in range(draws)]
    n2 = [stats_n2(S.line(npts, d=64, step=1.0, offset=8.0, noise=0.3, seed=s),
                   ks, n) for s in range(draws)]
    return {"n1": floors(n1), "n2": floors(n2)}


def ratio_cell(c2: float, c1: float) -> tuple[str, float | None]:
    """Display string and the value entering the claim-B median (D42 rules)."""
    if abs(c1) < 1e-3:
        if c2 >= 0.01:
            return "  ~0/X ", float("inf")
        return "   n/a ", None
    r = c2 / c1
    return f"{r:>7.1f}", r


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--seam-folds", type=int, default=40)
    args = ap.parse_args(argv)
    D = args.draws

    out = {"git_commit": git_commit(), "draws": D, "seed": 0,
           "datasets": {}, "fixture_grade": {}, "validation": {}}

    data = {}
    for label, pattern, drop_k0, ks, n in DATASETS:
        for method in ("todd", "hendel"):
            d = load(R / pattern.format(m=method))
            X = d["X"].astype(np.float64)
            if drop_k0:
                keep = [i for i, k in enumerate(d["ks"]) if int(k) != 0]
                X = X[keep]
            data[f"{label}/{method}"] = (X, ks, n, measure(X, ks))

    # ---- C2 floors for months/todd first: the gate's operative reference ----
    Xm, ksm, nm, pm = data["months/todd"]
    kfm = np.asarray(ksm, dtype=float)
    c2_gate = {
        "n1": floors(draw_stats(
            "n1", lambda r: Xm[r.permutation(len(Xm))], ksm, nm, D)),
        "n2": floors(draw_stats(
            "n2", lambda r: residual_surrogate(Xm, kfm, nm, r), ksm, nm, D)),
    }

    # ---- validation gate (D42 rule), before any cell is interpreted ---------
    grng = np.random.default_rng(1)
    v_circle = stats_n2(planted_circle(pm, ksm, nm, grng), ksm, nm)
    v_line = stats_n1(planted_line(pm, ksm, grng), ksm, nm)
    fresh = {
        "n1": draw_stats("n1", lambda r: Xm[r.permutation(len(Xm))],
                         ksm, nm, D, seed=1),
        "n2": draw_stats("n2", lambda r: residual_surrogate(Xm, kfm, nm, r),
                         ksm, nm, D, seed=1),
    }
    c1b_gate = {
        "n1": floors(draw_stats(
            "n1", lambda r: synth_matched_n1(pm, len(ksm), r), ksm, nm, D)),
        "n2": floors(draw_stats(
            "n2", lambda r: synth_matched_n2(pm, ksm, r), ksm, nm, D)),
    }

    gate = {}
    for row in N2_ROWS + N1_ROWS:
        null = "n2" if row in N2_ROWS else "n1"
        planted = (v_circle if null == "n2" else v_line)[row]
        c2f = c2_gate[null][row]["p95"]
        exceed = float(np.mean([d[row] > c2f for d in fresh[null]]))
        gate[row] = {
            "planted": planted, "c2_floor": c2f,
            "detected_vs_c2": bool(planted > c2f),
            "c2_self_exceedance": exceed,
            "c2_calibrated": bool(GATE_BAND[0] <= exceed <= GATE_BAND[1]),
            # findings about the matched synthetic, not gates (D42):
            "c1b_floor": c1b_gate[null][row]["p95"],
            "detected_vs_c1b": bool(planted > c1b_gate[null][row]["p95"]),
        }
    void = {r for r, g in gate.items()
            if not (g["detected_vs_c2"] and g["c2_calibrated"])}
    out["validation"] = gate

    print("=== validation gate (D42: planted vs C2 real floor; C2 calibration) ===")
    for row, g in gate.items():
        status = "VOID" if row in void else "ok"
        note = "" if g["detected_vs_c1b"] else \
            "   [matched synthetic could not validate this detector]"
        print(f"  {row:>24}: planted {g['planted']:+.3f} vs C2 {g['c2_floor']:+.4f} "
              f"detected={g['detected_vs_c2']}  self-exc {g['c2_self_exceedance']:.3f}"
              f"  -> {status}{note}")

    # ---- fixture-grade C1a (dataset-independent, per point count) ------------
    for npts, ks, n in ((11, list(range(1, 12)), 12), (12, list(range(12)), 12)):
        out["fixture_grade"][str(npts)] = fixture_grade(npts, ks, n, D)

    # ---- per-dataset floors ---------------------------------------------------
    for key, (X, ks, n, p) in data.items():
        kf = np.asarray(ks, dtype=float)
        cell = {
            "measured_regime": p,
            "observed": {**stats_n1(X, ks, n), **stats_n2(X, ks, n)},
            "floor_real_n1":
                c2_gate["n1"] if key == "months/todd" else floors(draw_stats(
                    "n1", lambda r: X[r.permutation(len(X))], ks, n, D)),
            "floor_real_n2":
                c2_gate["n2"] if key == "months/todd" else floors(draw_stats(
                    "n2", lambda r: residual_surrogate(X, kf, n, r), ks, n, D)),
            "floor_synth_n1":
                c1b_gate["n1"] if key == "months/todd" else floors(draw_stats(
                    "n1", lambda r: synth_matched_n1(p, len(ks), r), ks, n, D)),
            "floor_synth_n2":
                c1b_gate["n2"] if key == "months/todd" else floors(draw_stats(
                    "n2", lambda r: synth_matched_n2(p, ks, r), ks, n, D)),
        }
        out["datasets"][key] = cell
        print(f"  ...{key} done")

    # ---- centrepiece table ----------------------------------------------------
    lines, c4a_pool = [], []
    for method in ("todd", "hendel"):
        c = out["datasets"][f"months/{method}"]
        fg = out["fixture_grade"]["11"]
        lines.append(f"\n=== floors, months regime ({method}); C4a is the "
                     f"claim-B column ===")
        lines.append(f"{'statistic':>24} {'null':>4} {'C1a fix':>9} "
                     f"{'C1b match':>9} {'C2 real':>9} {'C4a':>7} {'C4b':>7} "
                     f"{'months':>8} {'addk':>8} {'unrel':>8}")
        for row in N2_ROWS + N1_ROWS:
            null = "n2" if row in N2_ROWS else "n1"
            c1a = fg[null][row]["p95"]
            c1b = c[f"floor_synth_{null}"][row]["p95"]
            c2 = c[f"floor_real_{null}"][row]["p95"]
            r_a, v_a = ratio_cell(c2, c1a)
            r_b, _ = ratio_cell(c2, c1b)
            if v_a is not None and row not in void:
                c4a_pool.append(v_a)
            mo = c["observed"][row]
            ak = out["datasets"][f"addk/{method}"]["observed"][row]
            un = out["datasets"][f"unrelated/{method}"]["observed"][row]
            flag = "  VOID" if row in void else ""
            lines.append(f"{row:>24} {null.upper():>4} {c1a:>9.4f} {c1b:>9.4f} "
                         f"{c2:>9.4f} {r_a} {r_b} {mo:>8.3f} {ak:>8.3f} "
                         f"{un:>8.3f}{flag}")
        pr = c["observed"]["participation_ratio"]
        prs = c["floor_synth_n1"]["participation_ratio"]
        prr = c["floor_real_n2"]["participation_ratio"]
        lines.append(
            f"{'participation_ratio':>24} {'--':>4} synth-N1 "
            f"[{prs['p05']:.1f},{prs['p95']:.1f}]  real-N2 "
            f"[{prr['p05']:.1f},{prr['p95']:.1f}]  observed {pr:.2f} "
            f"(descriptive)")

    finite = [v for v in c4a_pool if np.isfinite(v)]
    n_inf = len(c4a_pool) - len(finite)
    med = float(np.median(c4a_pool)) if c4a_pool else float("nan")
    verdict = ("GENERALISES (median >= 10)" if med >= 10 else
               "DOES NOT GENERALISE (median < 3)" if med < 3 else
               "PER-STATISTIC (3 <= median < 10)")
    lines.append(f"\nclaim-B pre-committed reading: median C4a over "
                 f"{len(c4a_pool)} non-void detector cells "
                 f"({n_inf} at ~0/X counted as +inf) = "
                 f"{med if np.isfinite(med) else 'inf'} -> {verdict}")
    text = "\n".join(lines)
    print(text)
    out["claim_b"] = {"c4a_cells": [None if not np.isfinite(v) else v
                                    for v in c4a_pool],
                      "median": med if np.isfinite(med) else "inf",
                      "verdict": verdict}

    from .results_io import write_guarded

    dest = Path("results/stage2/floors.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_guarded(dest, json.dumps(out, indent=2, default=float) + "\n")
    write_guarded(Path("results/stage2/floors.txt"), text + "\n")


if __name__ == "__main__":
    main()
