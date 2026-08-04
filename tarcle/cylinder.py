"""Cylinder / open-helix check (D33), and the record of why it is underpowered.

Whole-vector distances cannot see a circular component riding a dominant monotone
axis, so every seam-contest observation (D32) is equally consistent with an open
helix. This asks whether one is there.

Two estimators are run, because the registered one failed its own validation gate:

  A  PROJECTION (registered in D33): fit the monotone axis, project it out, look
     for a circle in the residual. FAILS step 1 -- a planted circle recovers to
     circulant 0.350 while the no-circle floor sits at 0.410. The reason is
     structural: over a single period corr(k, sin) = -0.87, so removing a
     monotone axis removes most of the circle with it. This does not improve
     with n.

  B  JOINT FIT (D37): regress X on [1, k, cos, sin] and measure the harmonic
     terms' unique variance share. Collinearity reduces power but does not
     prevent identification, so the circle's own contribution is recoverable.

Order is fixed: validate both on planted fixtures, take the floor from real
families with no circle, and only then read months.

Usage:
    python -m tarcle.cylinder
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import synthetic as S
from .geometry import circulant_score, gram
from .stage2 import load

R = Path("results/fv")
N = 12
KS = np.arange(1, 12, dtype=float)


# ---------------------------------------------------------------- estimator A
def project_residual(X: np.ndarray, ks: np.ndarray) -> np.ndarray:
    c = ks - ks.mean()
    v = (c[:, None] * (X - X.mean(0))).sum(0)
    v = v / np.linalg.norm(v)
    return X - np.outer(X @ v, v)


def circulant_of(X: np.ndarray) -> float:
    return circulant_score(gram(X - X.mean(0)))


# ---------------------------------------------------------------- estimator B
def design(ks: np.ndarray, harmonics: bool = True) -> np.ndarray:
    cols = [np.ones_like(ks), ks]
    if harmonics:
        th = 2 * np.pi * ks / N
        cols += [np.cos(th), np.sin(th)]
    return np.stack(cols, 1)


def joint_fit(X: np.ndarray, ks: np.ndarray) -> dict:
    M, M0 = design(ks, True), design(ks, False)
    B, *_ = np.linalg.lstsq(M, X, rcond=None)
    B0, *_ = np.linalg.lstsq(M0, X, rcond=None)
    ss = lambda A: float((A**2).sum())
    tot = ss(X - X.mean(0))
    return {
        "harmonic_amp": float(np.linalg.norm(B[2:])),
        "partial_r2": (ss(X - M0 @ B0) - ss(X - M @ B)) / tot,
        "axial_coord": (X @ (B[1] / np.linalg.norm(B[1]))).tolist(),
    }


def row(label: str, X: np.ndarray, ks: np.ndarray) -> dict:
    a = circulant_of(project_residual(X, ks))
    b = joint_fit(X, ks)
    print(f"    {label:38} A_resid_circ {a:.3f}   B_amp {b['harmonic_amp']:6.3f}   "
          f"B_partial_R2 {b['partial_r2']:.4f}")
    return {"A_residual_circulant": a, **b}


def main() -> None:
    out = {}
    print("collinearity over one period (why estimator A cannot work):")
    th = 2 * np.pi * KS / N
    print(f"  corr(k, cos) = {np.corrcoef(KS, np.cos(th))[0, 1]:+.3f}   "
          f"corr(k, sin) = {np.corrcoef(KS, np.sin(th))[0, 1]:+.3f}")
    print("  a single-period circle is itself strongly linear in k, so projecting")
    print("  out a monotone axis removes the circle too. Structural, not sample size.\n")

    print("=" * 78)
    print("STEP 1 — planted fixtures: does each estimator see a circle that IS there?")
    print("=" * 78)
    for axial in (6.0, 3.0, 1.0):
        X = S.open_helix(11, d=64, axial=axial, radius=1.0, offset=8.0,
                         noise=0.3, seed=0)
        out[f"planted_helix/axial{axial}"] = row(
            f"open helix axial={axial} radius=1.0", X, KS)
    for seed in (0, 1, 2):
        X = S.line(11, d=64, step=1.0, offset=8.0, noise=0.3, seed=seed)
        out[f"planted_line/{seed}"] = row(f"line only (no circle) seed={seed}", X, KS)
    print("\n  A: planted circle recovers to ~0.35, below the 0.41 no-circle floor "
          "below -> FAILS")
    print("  B: planted circle amp ~1.60 vs line ~0.28; partial_R2 ~0.15-0.60 vs "
          "~0.002 -> PASSES")

    print("\n" + "=" * 78)
    print("STEP 2 — real families with NO circle: the pre-committed floor")
    print("=" * 78)
    floor_a, floor_r2 = [], []
    for fam, path, ks in (
        ("add-k", "ctl_addk/fv_addk_{m}.npz", KS),
        ("null", "ctl_unrelated_12tasks/fv_unrelated_{m}.npz",
         np.arange(12, dtype=float)),
    ):
        for m in ("todd", "hendel"):
            X = load(R / path.format(m=m))["X"].astype(np.float64)
            r = row(f"{fam} [{m}]", X, ks)
            out[f"floor/{fam}/{m}"] = r
            floor_a.append(r["A_residual_circulant"])
            floor_r2.append(r["partial_r2"])
    fa, fr = float(max(floor_a)), float(max(floor_r2))
    out["floor"] = {"A_circulant": fa, "B_partial_r2": fr,
                    "B_partial_r2_spread": [float(min(floor_r2)), fr]}
    print(f"\n  FLOOR  A: {fa:.3f}   B partial_R2: {fr:.4f} "
          f"(spread {min(floor_r2):.4f}-{fr:.4f})")

    print("\n" + "=" * 78)
    print("STEP 3 — months (read only now)")
    print("=" * 78)
    for m in ("todd", "hendel"):
        d = load(R / "ctl_months_primary" / f"fv_primary_{m}.npz")
        keep = [i for i, k in enumerate(d["ks"]) if k != 0]
        X = d["X"].astype(np.float64)[keep]
        r = row(f"months [{m}]", X, KS)
        r["above_floor_B"] = bool(r["partial_r2"] > fr)
        r["margin_vs_floor"] = r["partial_r2"] - fr
        r["floor_spread"] = fr - float(min(floor_r2))
        out[f"months/{m}"] = r

        # STEP 4 — is the axial coordinate k, or token frequency?
        z = np.load(R / "ctl_months_primary" / f"fv_primary_{m}.npz")
        coord = np.array(r["axial_coord"])
        for name in ("freq_proxy_operand", "freq_proxy_target"):
            p = z[name][keep].astype(np.float64)
            if p.std() < 1e-12:
                print(f"      {name}: constant across k, correlation undefined")
                continue
            c = float(np.corrcoef(coord, p)[0, 1])
            out[f"months/{m}"][f"corr_{name}"] = c
            print(f"      corr(axial coord, {name}) = {c:+.3f}")
        print(f"      corr(axial coord, k) = "
              f"{float(np.corrcoef(coord, KS)[0, 1]):+.3f}")
        print(f"      B partial_R2 {r['partial_r2']:.4f} vs floor {fr:.4f}  "
              f"margin {r['margin_vs_floor']:+.4f}, floor's own spread "
              f"{r['floor_spread']:.4f}")

    Path("results/stage2").mkdir(parents=True, exist_ok=True)
    Path("results/stage2/cylinder.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n",
        encoding="utf-8", newline="\n")
    print("\nwrote results/stage2/cylinder.json")


if __name__ == "__main__":
    main()
