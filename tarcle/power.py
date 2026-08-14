"""Synthetic power analysis for the joint-fit circle detector. Pure numpy.

D37 ended inconclusive because the detector's false-positive floor on real
no-circle data (partial R2 up to 0.315 at n=11) exceeded what a planted circle
produces at a realistic axial ratio (0.149). The next family must not repeat
that as a surprise, so the minimum detectable circle amplitude is computed at
the actual n BEFORE any extraction.

Two quantities, both pre-committed:

  floor(n)  the 95th percentile of partial R2 over no-circle surrogates at the
            measured offset and noise regime -- the false-positive floor
  MDA(n)    minimum detectable amplitude: the smallest planted circle radius
            whose partial R2 exceeds that floor in >= 80% of draws

If MDA at the family's n is larger than any plausible circular component, the
geometric test is known to be underpowered in advance and the causal
tie-breaker is registered instead.

Usage:
    python -m tarcle.power --n 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import synthetic as S


def design(ks: np.ndarray, n: int, harmonics: bool = True) -> np.ndarray:
    cols = [np.ones_like(ks), ks]
    if harmonics:
        th = 2 * np.pi * ks / n
        cols += [np.cos(th), np.sin(th)]
    return np.stack(cols, 1)


def partial_r2(X: np.ndarray, ks: np.ndarray, n: int) -> float:
    M, M0 = design(ks, n, True), design(ks, n, False)
    B, *_ = np.linalg.lstsq(M, X, rcond=None)
    B0, *_ = np.linalg.lstsq(M0, X, rcond=None)
    ss = lambda A: float((A**2).sum())
    tot = ss(X - X.mean(0))
    return (ss(X - M0 @ B0) - ss(X - M @ B)) / tot


def residual_surrogate(X: np.ndarray, ks: np.ndarray, n: int,
                       rng: np.random.Generator) -> np.ndarray:
    """A no-circle surrogate built from the real vectors.

    Keeps the fitted constant-plus-linear-in-k component, and permutes the
    residuals across k. Everything about the real vectors survives — their
    magnitude, their covariance, their non-Gaussian structure — except any
    k-dependence beyond the axis, which is exactly the null the circle test is
    against.

    This matters because synthetic line-plus-isotropic-noise surrogates put the
    false-positive floor at ~0.003 while the real one is ~0.3, a hundredfold
    difference: they would promise power the estimator does not have. D37 was
    inconclusive for precisely this reason.
    """
    M0 = design(ks, n, False)
    B0, *_ = np.linalg.lstsq(M0, X, rcond=None)
    fitted = M0 @ B0
    resid = X - fitted
    return fitted + resid[rng.permutation(len(X))]


def circle_component(ks: np.ndarray, n: int, d: int, radius: float,
                     rng: np.random.Generator) -> np.ndarray:
    """A planted circle of the given radius in a random plane of R^d."""
    q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    th = 2 * np.pi * ks / n
    return radius * (np.outer(np.cos(th), q[:, 0]) + np.outer(np.sin(th), q[:, 1]))


def analyse_real(X: np.ndarray, ks: np.ndarray, n: int, label: str,
                 radii=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5), draws: int = 300) -> dict:
    """Floor and minimum detectable amplitude, both from the real vectors.

    Radii are expressed as fractions of the family's own RMS vector norm, so the
    result is comparable across families of different scale.
    """
    rng = np.random.default_rng(0)
    scale = float(np.linalg.norm(X - X.mean(0), axis=1).mean())
    d = X.shape[1]

    null = np.array([
        partial_r2(residual_surrogate(X, ks, n, rng), ks, n) for _ in range(draws)
    ])
    floor = float(np.percentile(null, 95))

    curve, mda = {}, None
    for frac in radii:
        vals = np.array([
            partial_r2(
                residual_surrogate(X, ks, n, rng)
                + circle_component(ks, n, d, frac * scale, rng), ks, n)
            for _ in range(draws)
        ])
        power = float((vals > floor).mean())
        curve[frac] = {"mean_partial_r2": float(vals.mean()), "power": power}
        if mda is None and power >= 0.80:
            mda = frac

    observed = partial_r2(X, ks, n)
    return {"label": label, "n": n, "points": len(ks), "rms_norm": scale,
            "floor_p95": floor, "null_max": float(null.max()),
            "observed_partial_r2": observed,
            "observed_above_floor": bool(observed > floor),
            "mda_frac_of_rms": mda, "curve": curve, "draws": draws}


def report(r: dict) -> None:
    print(f"\n{r['label']}  (n={r['n']}, {r['points']} points, "
          f"RMS norm {r['rms_norm']:.3f}, {r['draws']} draws)")
    print(f"  false-positive floor, 95th pct of residual-permutation "
          f"surrogates: {r['floor_p95']:.4f}   worst {r['null_max']:.4f}")
    print(f"  {'radius/RMS':>11} {'mean partial R2':>16} {'power':>7}")
    for frac, v in r["curve"].items():
        flag = "  <- MDA" if frac == r["mda_frac_of_rms"] else ""
        print(f"  {frac:>11} {v['mean_partial_r2']:>16.4f} {v['power']:>7.2f}{flag}")
    mda = r["mda_frac_of_rms"]
    print(f"  MDA at 80% power: "
          f"{f'{mda:.2f} x RMS norm' if mda else 'NOT REACHED in range'}")
    print(f"  observed partial R2 {r['observed_partial_r2']:.4f}  -> "
          f"{'ABOVE' if r['observed_above_floor'] else 'below'} floor")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=300)
    args = p.parse_args(argv)
    from .stage2 import load

    R = Path("results/fv")
    out = {}
    targets = [
        ("months n=11", "ctl_months_primary/fv_primary_{m}.npz", 12, True),
        ("add-k n=11 (no circle)", "ctl_addk/fv_addk_{m}.npz", 12, False),
        ("unrelated null", "ctl_unrelated_12tasks/fv_unrelated_{m}.npz", 12, None),
    ]
    for label, path, n, drop_k0 in targets:
        for method in ("todd", "hendel"):
            d = load(R / path.format(m=method))
            X = d["X"].astype(np.float64)
            if drop_k0:
                keep = [i for i, k in enumerate(d["ks"]) if k != 0]
                X, ks = X[keep], np.arange(1, 12, dtype=float)
            elif drop_k0 is False:
                ks = np.arange(1, 12, dtype=float)
            else:
                ks = np.arange(12, dtype=float)
            r = analyse_real(X, ks, n, f"{label} [{method}]", draws=args.draws)
            report(r)
            out[f"{label}/{method}"] = r

    from .results_io import write_guarded

    Path("results/stage2").mkdir(parents=True, exist_ok=True)
    write_guarded(Path("results/stage2/power.json"),
                  json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
