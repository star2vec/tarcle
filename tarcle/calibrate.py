"""D28 calibration: does the run's evidence distinguish cyclic from merely ordered?

Places three real-model readings side by side on the same diagnostics:

    unrelated tasks   unordered  (arbitrary task labels)
    add-k             ordered, NOT cyclic (1 and n maximally separated)
    months Z/12       ordered and cyclic

The question D28 registered: permutation z is high for months, but any monotone
ordering beats random permutations, so z alone cannot show the structure is
*cyclic* rather than merely *ordered*. add-k supplies the missing reference.

The pre-stated discriminator is the separation profile. Months, computed under
wraparound identification, dips at the antipode (|m|=6 below |m|=5). add-k has no
antipode, so under its own (non-cyclic) separation its profile should rise to the
end. Whether the diagnostics separate the two is read off that contrast.

Months is compared at n=11 with k=0 dropped, matching add-k's k=1..11 and
excluding the identity outlier (D28).

Usage:
    python -m tarcle.calibrate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import diagnose
from .stage2 import load, permutation_null, separation_profile

R = Path("results/fv")


def months_n11(method: str) -> np.ndarray:
    d = load(R / "ctl_months_primary" / f"fv_primary_{method}.npz")
    keep = [i for i, k in enumerate(d["ks"]) if k != 0]
    return d["X"].astype(np.float64)[keep]


def main() -> None:
    out = {}
    for method in ("todd", "hendel"):
        # (label, X, parameter values, modulus or None)
        rows = [
            ("unrelated (unordered)",
             load(R / "ctl_unrelated_12tasks" / f"fv_unrelated_{method}.npz")["X"]
             .astype(np.float64), list(range(12)), 12),
            ("add-k (ordered, non-cyclic)",
             load(R / "ctl_addk" / f"fv_addk_{method}.npz")["X"].astype(np.float64),
             list(range(1, 12)), None),
            # k=0 dropped, but the surviving eleven still live on Z/12
            ("months n=11 (ordered, cyclic)", months_n11(method),
             list(range(1, 12)), 12),
        ]
        print(f"\n{'=' * 72}\n{method}\n{'=' * 72}")
        print(f"{'family':32} {'circ':>6} {'circ_c':>7} {'perm z':>8} {'PR':>6} "
              f"{'norm_cv':>8}")
        for label, X, _, _ in rows:
            d = diagnose(X)
            dc = diagnose(X - X.mean(0))
            p = permutation_null(X, 200)
            print(f"{label:32} {d['circulant_score']:>6.3f} "
                  f"{dc['circulant_score']:>7.3f} {p['z']:>+8.2f} "
                  f"{d['participation_ratio']:>6.2f} {d['norm_cv']:>8.3f}")
            out[f"{method}/{label}"] = {
                **d, "circulant_centered": dc["circulant_score"],
                "permutation": p,
            }

        print("\n  separation profiles, normalised to separation 1 "
              "(pair count in brackets):")
        for label, X, ks, mod in rows:
            prof, cnt = separation_profile(X, ks, mod)
            base = prof[1]
            kind = f"cyclic mod {mod}" if mod else "linear"
            print(f"    {label:32} [{kind}]")
            print("      " + "  ".join(
                f"{m}:{v / base:.2f}[{cnt[m]}]" for m, v in prof.items()))
            out[f"{method}/{label}"]["separation_profile"] = prof
            out[f"{method}/{label}"]["separation_counts"] = cnt

        print("\n  D28 discriminator — behaviour at the largest separation:")
        for label, X, ks, mod in rows:
            prof, cnt = separation_profile(X, ks, mod)
            m = max(prof)
            prev = m - 1
            # Only read bins backed by enough pairs to mean anything.
            thin = cnt[m] < 4
            turn = "DOWN" if prof[m] < prof[prev] else "UP"
            print(f"    {label:32} sep {prev}->{m}: {prof[prev]:.3f} -> "
                  f"{prof[m]:.3f}  {turn}"
                  f"{'   (THIN: %d pairs, not read)' % cnt[m] if thin else ''}")

    Path("results/stage2").mkdir(parents=True, exist_ok=True)
    Path("results/stage2/calibration.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n",
        encoding="utf-8", newline="\n")
    print("\nwrote results/stage2/calibration.json")


if __name__ == "__main__":
    main()
