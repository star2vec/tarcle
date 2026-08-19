"""D48: landing map of injected predictions for the collapsed conditions.

Reproduces, from the committed .npz artifacts, the descriptive read used in
post drafting: where do the collapsed vectors' injected zero-shot predictions
land? Registered definition (D48): Todd method, the four gate-failed
conditions, mid-cycle k (k not in {0, 1, 11}), full 12-query cycle. Classes of
the signed prediction shift (argmax - query) mod 12:

  correct      shift == k
  successor    shift == 1
  predecessor  shift == 11 (== -1)
  copy         shift == 0
  fwd_2_8      shift in {2..8} (and != k)   <- the "zero outputs 2-8 steps
  other        everything else                 forward" claim lives here

Expected values registered in D48 (from the drafting read): strictest
condition ~95% adjacent (successor+predecessor), fwd_2_8 == 0 there. A
mismatch is a STOP, not a reconciliation.

Usage:
    python -m tarcle.landing_map
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .nextitem import TRIVIAL_K
from .results_io import write_guarded

CONDITIONS = [
    ("partA (Jan-Apr)", "results/fv/ctl_months_partA/fv_partition_a_todd.npz"),
    ("partB (Sep-Dec)", "results/fv/ctl_months_partB/fv_partition_b_todd.npz"),
    ("halfA (Jan-Jun)", "results/fv/ctl_months_halfA/fv_half_a_todd.npz"),
    ("halfB (Jul-Dec)", "results/fv/ctl_months_halfB/fv_half_b_todd.npz"),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def landing(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    ks = [int(k) for k in z["ks"]]
    rows = [i for i, k in enumerate(ks) if k not in TRIVIAL_K]
    shifts = z["efficacy_pred_shift"][rows]              # (9, 12)
    kcol = np.array([ks[i] for i in rows])[:, None]
    n = shifts.size
    correct = shifts == kcol
    out = {
        "n": int(n),
        "correct": float(correct.mean()),
        "successor": float((shifts == 1).mean()),
        "predecessor": float((shifts == 11).mean()),
        "copy": float((shifts == 0).mean()),
        "fwd_2_8": float((np.isin(shifts, range(2, 9)) & ~correct).mean()),
        "fwd_2_8_count": int((np.isin(shifts, range(2, 9)) & ~correct).sum()),
        "histogram": {str(s): int((shifts == s).sum()) for s in range(12)},
    }
    out["adjacent"] = out["successor"] + out["predecessor"]
    out["other"] = 1.0 - out["correct"] - out["adjacent"] - out["copy"] \
        - out["fwd_2_8"]
    return out


def main() -> None:
    out = {"git_commit": git_commit(), "definition": "todd, k not in {0,1,11}, "
           "12-query cycle; classes per module docstring", "conditions": {}}
    print(f"{'condition':>18} {'adjacent':>9} {'succ':>6} {'pred':>6} "
          f"{'correct':>8} {'copy':>6} {'fwd2-8':>7} {'other':>6}")
    for name, path in CONDITIONS:
        r = landing(path)
        out["conditions"][name] = r
        print(f"{name:>18} {r['adjacent']:>9.3f} {r['successor']:>6.3f} "
              f"{r['predecessor']:>6.3f} {r['correct']:>8.3f} "
              f"{r['copy']:>6.3f} {r['fwd_2_8_count']:>7d} {r['other']:>6.3f}")
    write_guarded(Path("results/stage2/landing_map.json"),
                  json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
