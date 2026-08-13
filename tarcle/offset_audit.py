"""T10: shared-offset audit. Pure numpy — stage-2 code, no torch.

Registered in docs/preregistration_instruments.md §4 (D39). Both extractions
carry a large k-invariant component (78.8% Todd / 94.9% Hendel of mean squared
norm on the primary condition, commit 3eac9d8). D27 quantified its effect on
the circulant score only. This audits every raw-Gram statistic in
tarcle.geometry: raw vs centered, each with its own split-half band so "differs
from raw" can be checked against noise — and states the k-dependent signal as
an absolute quantity against that band, so a reader can see how much signal
there was to have geometry in.

Descriptive throughout; no verdict language (prereg §4). The D27 constraint
carries over: the centered column describes the k-dependent component after
removing an offset that is part of the vector that actually steers the model.

Usage:
    python -m tarcle.offset_audit
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .geometry import (
    additivity_residual, circulant_score, closure_ratio, fit_rotation, gram,
    norm_cv, participation_ratio, spectral_concentration, toeplitz_score,
)
from .stage2 import load, offset_share

# Raw-Gram statistics that are NOT translation-invariant: the audit's subject.
AFFECTED = {
    "circulant_score": lambda X: circulant_score(gram(X)),
    "toeplitz_score": lambda X: toeplitz_score(gram(X)),
    "spectral_concentration": lambda X: spectral_concentration(gram(X)),
    "norm_cv": norm_cv,
}
# Translation-invariant or internally-centered: asserted, listed once.
INVARIANT = {
    "closure_ratio": closure_ratio,
    "rotation_order_error": lambda X: fit_rotation(X).order_error,
    "rotation_wraparound_error": lambda X: fit_rotation(X).wraparound_error,
    "additivity_residual": additivity_residual,
    "participation_ratio": participation_ratio,
}

CONDITIONS = [
    ("primary", "results/fv/ctl_months_primary/fv_primary_{m}.npz"),
    ("polysemy", "results/fv/ctl_months_polysemy/fv_polysemy_leaveout_{m}.npz"),
    ("mixed_daysmonths", "results/fv/ctl_daysmonths_qmonths/fv_mixed_daysmonths_{m}.npz"),
    ("addk", "results/fv/ctl_addk/fv_addk_{m}.npz"),
    ("unrelated", "results/fv/ctl_unrelated_12tasks/fv_unrelated_{m}.npz"),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0)


def audit_cell(d: dict) -> dict:
    X = d["X"].astype(np.float64)
    Xa, Xb = d["X_half_a"].astype(np.float64), d["X_half_b"].astype(np.float64)

    stats = {}
    for name, fn in AFFECTED.items():
        band_raw = abs(fn(Xa) - fn(Xb))
        band_cen = abs(fn(center(Xa)) - fn(center(Xb)))
        stats[name] = {
            "raw": float(fn(X)), "centered": float(fn(center(X))),
            "band_raw": float(band_raw), "band_centered": float(band_cen),
        }
        stats[name]["delta_exceeds_band"] = bool(
            abs(stats[name]["centered"] - stats[name]["raw"])
            > max(band_raw, band_cen)
        )
    for name, fn in INVARIANT.items():
        raw, cen = float(fn(X)), float(fn(center(X)))
        assert abs(raw - cen) < 1e-8 * max(1.0, abs(raw)), (
            f"{name} expected translation-invariant, raw {raw} != centered {cen}")
        stats[name] = {"raw": raw, "invariant": True}

    # Signal vs noise, absolute. noise_k = half the split-half difference: each
    # half is the mean of ~n/2 prompts, so (a-b)/2 estimates the SE of the full
    # mean. Cross-checked against the stored per-coordinate X_se.
    signal_k = np.linalg.norm(X - X.mean(axis=0), axis=1)
    noise_k = np.linalg.norm(Xa - Xb, axis=1) / 2
    se_k = np.linalg.norm(d["X_se"].astype(np.float64), axis=1)
    return {
        "offset_share": float(offset_share(X)),
        "statistics": stats,
        "signal_k": signal_k.tolist(),
        "noise_k": noise_k.tolist(),
        "se_k_stored": se_k.tolist(),
        "rms_signal": float(np.sqrt((signal_k**2).mean())),
        "rms_noise": float(np.sqrt((noise_k**2).mean())),
        "signal_to_noise": float(
            np.sqrt((signal_k**2).mean() / (noise_k**2).mean())),
    }


def show(name: str, method: str, a: dict) -> None:
    print(f"\n=== {name} / {method} ===  offset share "
          f"{a['offset_share']*100:.1f}%   signal/noise "
          f"{a['signal_to_noise']:.1f}x  (RMS signal {a['rms_signal']:.3f}, "
          f"RMS split-half noise {a['rms_noise']:.3f})")
    print(f"{'statistic':>26} {'raw':>8} {'centered':>9} {'band(r)':>8} "
          f"{'band(c)':>8}")
    for stat, v in a["statistics"].items():
        if v.get("invariant"):
            print(f"{stat:>26} {v['raw']:>8.3f} {'(translation-invariant)':>27}")
        else:
            mark = "  *" if v["delta_exceeds_band"] else ""
            print(f"{stat:>26} {v['raw']:>8.3f} {v['centered']:>9.3f} "
                  f"{v['band_raw']:>8.3f} {v['band_centered']:>8.3f}{mark}")
    print("  * centered-vs-raw difference exceeds both split-half bands")


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    out = {"git_commit": git_commit(), "cells": {}}
    for name, pattern in CONDITIONS:
        for method in ("todd", "hendel"):
            path = Path(pattern.format(m=method))
            if not path.exists():
                print(f"skip {path} (not on disk)")
                continue
            a = audit_cell(load(path))
            out["cells"][f"{name}/{method}"] = a
            show(name, method, a)

    dest = Path("results/stage2/offset_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
