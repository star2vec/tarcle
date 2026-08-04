"""Stage 2: geometry over the saved .npz. Pure numpy — never imports torch.

Evaluation order is fixed by D24 and is enforced here, not left to the reader:

  1. the unrelated-tasks null control, read by the D25 §1 separator rule
  2. the prereg §3 month-frequency control (blocking for the months conditions)
  3. the D24 confirmatory cells: primary / full n=12 / canonical head set,
     spectral_concentration and participation_ratio, once per extraction method
  4. everything else, exploratory, reported without verdict language

Usage:
    python -m tarcle.stage2 results/fv --out results/stage2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import (
    circulant_score,
    diagnose,
    frequency_pair_powers,
    gram,
    participation_ratio,
    spectral_concentration,
)

UNIFORM_SHARE = 1 / 6  # non-DC power share at n=12, prereg §0


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta_json"}
    d["meta"] = json.loads(str(z["meta_json"]))
    return d


def permutation_null(X: np.ndarray, n_perm: int = 200, seed: int = 0) -> dict:
    """Circulant score under random re-orderings of the parameter (D25 §2).

    A simplex Gram is I + c*J, invariant under simultaneous row/column
    permutation, so its circulant score is unchanged by re-ordering. A genuine
    cyclic structure depends on the specific ordering and collapses. The
    percentile of the canonical score in this null therefore separates
    ordering-dependent structure (A) from equidistance (D).
    """
    rng = np.random.default_rng(seed)
    canonical = circulant_score(gram(X))
    scores = np.array([
        circulant_score(gram(X[rng.permutation(len(X))])) for _ in range(n_perm)
    ])
    return {
        "canonical": float(canonical),
        "perm_mean": float(scores.mean()),
        "perm_std": float(scores.std(ddof=1)),
        "perm_max": float(scores.max()),
        "percentile": float((scores < canonical).mean() * 100),
        "z": float((canonical - scores.mean()) / scores.std(ddof=1))
        if scores.std(ddof=1) > 0 else float("nan"),
        "n_perm": n_perm,
    }


def split_half_band(d: dict, fn) -> float:
    """|fn(half_a) - fn(half_b)|: the measured noise band D16 §2 makes the
    reference for whether two conditions actually differ."""
    return abs(fn(d["X_half_a"].astype(np.float64)) - fn(d["X_half_b"].astype(np.float64)))


CIRCULANT_GATE = 0.70  # prereg §2, the first criterion of both A and A-multi


def classify(circ: float, conc: float, pr: float) -> str:
    """prereg §2 buckets. The circulant gate comes FIRST (D26).

    `spectral_concentration` is the DFT of the circulant profile — G averaged over
    the (i-j) mod n classes. When G is not circulant those classes do not describe
    it, the profile is an artefact and its spectrum has no referent. prereg §0 and
    tests/test_geometry.py::test_spectrum_meaningless_without_circulant_gate both
    record this; D25 §1 dropped the gate and consequently mislabelled a
    non-circulant null control as a helix.
    """
    if circ < CIRCULANT_GATE:
        return "not circulant — spectrum not read"
    if conc >= 0.50 and 1.5 <= pr <= 3.5:
        return "A (circle)"
    if 0.25 <= conc < 0.50 and 3.5 <= pr <= 7:
        return "A-multi (helix)"
    if conc <= 0.25 and pr >= 8:
        return "D (simplex/mixture)"
    return "circulant but unclassified"


def centered_view(d: dict) -> dict:
    """D27: the same cell with the shared offset removed. Exploratory only."""
    return {
        "X": d["X"].astype(np.float64) - d["X"].astype(np.float64).mean(0),
        "X_half_a": d["X_half_a"].astype(np.float64)
        - d["X_half_a"].astype(np.float64).mean(0),
        "X_half_b": d["X_half_b"].astype(np.float64)
        - d["X_half_b"].astype(np.float64).mean(0),
        "meta": d["meta"],
    }


def separation_profile(
    X: np.ndarray, ks: list[int], modulus: int | None = None
) -> tuple[dict[int, float], dict[int, int]]:
    """Mean pairwise distance by parameter separation, plus the pair count.

    `ks` are the actual parameter values, not row positions — this matters
    whenever a subset is analysed. Months with k=0 dropped leaves eleven points
    whose true separations are still on Z/12; profiling them with modulus 11
    would fold k=1 and k=11 onto the wrong classes and invent a monotone tail.

    `modulus` identifies separations mod n for a cyclic family (|m| = min(d, n−d),
    so the largest bin is the antipode); None leaves them linear, which is the
    add-k case where the first and last parameter are maximally separated.

    Pair counts are returned because the tail bins are thin — a linear family on
    eleven points has one pair at separation 10 — and a turn in the profile there
    is not evidence of anything.
    """
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    vals: dict[int, list[float]] = {}
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            if i == j:
                continue
            d = abs(ki - kj)
            m = min(d, modulus - d) if modulus else d
            vals.setdefault(m, []).append(D[i, j])
    return (
        {m: float(np.mean(v)) for m, v in sorted(vals.items())},
        {m: len(v) for m, v in sorted(vals.items())},
    )


def offset_share(X: np.ndarray) -> float:
    """Fraction of mean squared norm carried by the component shared across k."""
    return float(
        np.linalg.norm(X.mean(0)) ** 2 / (np.linalg.norm(X, axis=1) ** 2).mean()
    )


def report_cell(name: str, d: dict, n_perm: int = 200) -> dict:
    X = d["X"].astype(np.float64)
    diag = diagnose(X)
    perm = permutation_null(X, n_perm)
    powers = frequency_pair_powers(gram(X))
    total = sum(powers.values())
    shares = {f: p / total for f, p in powers.items()} if total > 0 else {}
    out = {
        "cell": name, **diag,
        "bucket": classify(
            diag["circulant_score"], diag["spectral_concentration"],
            diag["participation_ratio"],
        ),
        "permutation": perm,
        "freq_shares": shares,
        "band_concentration": split_half_band(
            d, lambda Y: spectral_concentration(gram(Y))
        ),
        "band_pr": split_half_band(d, participation_ratio),
        "offset_share": offset_share(X),
    }
    # D27: exploratory centered column, never replacing the raw headline.
    c = centered_view(d)
    cdiag = diagnose(c["X"])
    out["centered"] = {
        **cdiag,
        "bucket": classify(
            cdiag["circulant_score"], cdiag["spectral_concentration"],
            cdiag["participation_ratio"],
        ),
        "permutation": permutation_null(c["X"], n_perm),
        "freq_shares": {
            f: p / sum(frequency_pair_powers(gram(c["X"])).values())
            for f, p in frequency_pair_powers(gram(c["X"])).items()
        },
        "band_concentration": split_half_band(
            c, lambda Y: spectral_concentration(gram(Y))
        ),
        "band_pr": split_half_band(c, participation_ratio),
    }
    return out


def print_cell(r: dict) -> None:
    print(f"\n--- {r['cell']} ---")
    print(f"  circulant {r['circulant_score']:.3f}   toeplitz {r['toeplitz_score']:.3f}"
          f"   closure {r['closure_ratio']:.2f}   norm_cv {r['norm_cv']:.3f}")
    print(f"  spectral_concentration {r['spectral_concentration']:.3f} "
          f"(+/- {r['band_concentration']:.3f} split-half)   "
          f"PR {r['participation_ratio']:.2f} (+/- {r['band_pr']:.2f})")
    print(f"  additivity {r['additivity_residual']:.3f}   "
          f"rotation: residual {r['rotation_residual']:.3f} "
          f"order_err {r['rotation_order_error']:.3f} "
          f"wrap_err {r['rotation_wraparound_error']:.3f}")
    shares = " ".join(f"f{f}:{s:.3f}" for f, s in sorted(r["freq_shares"].items()))
    print(f"  freq pair shares (uniform = {UNIFORM_SHARE:.3f}): {shares}")
    p = r["permutation"]
    print(f"  permutation null: canonical {p['canonical']:.3f} vs "
          f"perm {p['perm_mean']:.3f} +/- {p['perm_std']:.3f}  "
          f"(max {p['perm_max']:.3f}, percentile {p['percentile']:.1f}, "
          f"z {p['z']:+.2f})")
    print(f"  >>> bucket (RAW, registered): {r['bucket']}")
    c = r["centered"]
    cs = " ".join(f"f{f}:{s:.3f}" for f, s in sorted(c["freq_shares"].items()))
    print(f"  [exploratory D27] shared offset carries {r['offset_share']:.1%} of "
          f"mean squared norm; centered:")
    print(f"      circulant {c['circulant_score']:.3f}  toeplitz "
          f"{c['toeplitz_score']:.3f}  closure {c['closure_ratio']:.2f}  "
          f"conc {c['spectral_concentration']:.3f} (+/-{c['band_concentration']:.3f})  "
          f"PR {c['participation_ratio']:.2f} (+/-{c['band_pr']:.2f})")
    print(f"      wrap_err {c['rotation_wraparound_error']:.3f}  "
          f"order_err {c['rotation_order_error']:.3f}  "
          f"additivity {c['additivity_residual']:.3f}")
    print(f"      freq: {cs}")
    print(f"      perm: canonical {c['permutation']['canonical']:.3f} vs "
          f"{c['permutation']['perm_mean']:.3f} +/- {c['permutation']['perm_std']:.3f} "
          f"(z {c['permutation']['z']:+.2f})")
    print(f"      bucket: {c['bucket']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, default=Path("results/fv"), nargs="?")
    parser.add_argument("--out", type=Path, default=Path("results/stage2"))
    parser.add_argument("--n-perm", type=int, default=200)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    # ---- 1. the null control, first, per D24 and D25 §1 -------------------
    print("=" * 72)
    print("STEP 1 — null control (prereg §5, read by the D25 §1 separator rule)")
    print("=" * 72)
    for method in ("todd", "hendel"):
        p = args.root / "ctl_unrelated_12tasks" / f"fv_unrelated_{method}.npz"
        r = report_cell(f"unrelated / {method}", load(p), args.n_perm)
        print_cell(r)
        results[f"null_{method}"] = r
        # D26: the circulant gate comes first. A non-circulant Gram has no
        # interpretable spectrum, so concentration and PR are not read at all.
        voids = (
            r["circulant_score"] >= CIRCULANT_GATE
            and r["spectral_concentration"] >= 0.25
            and 1.5 <= r["participation_ratio"] <= 7
        )
        perm_bug = r["permutation"]["z"] > 3
        print(f"  D25 §1 verdict: {'VOID THE RUN' if voids else 'PASS'}"
              f"{'  (also: canonical >> permutations, ordering carries structure -> pipeline bug)' if perm_bug else ''}")
        results[f"null_{method}"]["voids"] = bool(voids or perm_bug)

    if any(results[f"null_{m}"]["voids"] for m in ("todd", "hendel")):
        print("\nNULL CONTROL VOIDS THE RUN — stopping before the headline cells.")
        (args.out / "stage2.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
        return

    # ---- 2. headline cells ------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3 — D24 confirmatory cells (primary / full n=12 / canonical head set)")
    print("=" * 72)
    for method in ("todd", "hendel"):
        p = args.root / "ctl_months_primary" / f"fv_primary_{method}.npz"
        r = report_cell(f"HEADLINE primary / {method}", load(p), args.n_perm)
        print_cell(r)
        results[f"headline_{method}"] = r

    # ---- 3. exploratory ---------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4 — exploratory (no verdict language)")
    print("=" * 72)
    exploratory = [
        ("polysemy", "ctl_months_polysemy", "fv_polysemy_leaveout_{m}.npz"),
        ("mixed_daysmonths", "ctl_daysmonths_qmonths", "fv_mixed_daysmonths_{m}.npz"),
        ("headset_allk", "months_llama32_3b_ada_hs_allk", "fv_primary_{m}.npz"),
        ("headset_int8", "months_llama32_3b_ada_hs_int8", "fv_primary_{m}.npz"),
    ]
    for label, run, pattern in exploratory:
        for method in ("todd", "hendel"):
            p = args.root / run / pattern.format(m=method)
            if not p.exists():
                continue
            r = report_cell(f"{label} / {method}", load(p), args.n_perm)
            print_cell(r)
            results[f"{label}_{method}"] = r

    # leave-one-out at n=11, prereg §4 (divisor test undefined there)
    print("\n  --- leave-one-out n=11, k=8 excluded (prereg §4) ---")
    for method in ("todd", "hendel"):
        d = load(args.root / "ctl_months_primary" / f"fv_primary_{method}.npz")
        keep = [i for i, k in enumerate(d["ks"]) if k != 8]
        d11 = {"X": d["X"][keep], "X_half_a": d["X_half_a"][keep],
               "X_half_b": d["X_half_b"][keep], "meta": d["meta"]}
        r = report_cell(f"primary n=11 (no k=8) / {method}", d11, args.n_perm)
        print_cell(r)
        results[f"loo_{method}"] = r

    (args.out / "stage2.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n",
        encoding="utf-8", newline="\n")
    print(f"\nwrote {args.out / 'stage2.json'}")


if __name__ == "__main__":
    main()
