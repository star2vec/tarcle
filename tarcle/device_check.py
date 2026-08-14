"""D46: does the gate instrument reproduce across devices? Pure numpy.

D45 §1 found the recorded polysemy gate GO does not reproduce on MPS fp16
(k=8 at 0.44 -> MARGINAL). Before that is logged as a wrong record, the
device/precision alternative must be ruled out: the original restricted-pool
gates ran CUDA bf16, tonight's re-runs ran MPS fp16, and those are different
instruments.

Two conditions have the same in-distribution cells measured on BOTH
instruments: halves_A and partA4 have committed CUDA bf16 originals
(n=100/k), and the T1 full-cycle audit runs re-measured the identical
in-distribution query population on MPS fp16 as their in-pool subsets
(n~50 and n~33 per k). This compares them cell by cell, all twelve k, with
two-proportion 95% margins.

Coverage stated plainly: primary has no CUDA pair (its original gate IS the
MPS fp16 months pilot); halves_B, partB4 and polysemy each exist on a single
device. The check covers every cross-device pair that exists: 24 cells.

Usage:
    python -m tarcle.device_check
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .support_gate import KS, acc_by_k, load_scored, two_prop_margin

PAIRS = [
    # condition, CUDA bf16 original, MPS fp16 re-measurement (in-pool subset of)
    ("halfA (6)", "gate_months_halves_A", "gate_months_halves_A_fullq"),
    ("partA (4)", "gate_months_partA4", "gate_months_partA4_fullq"),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    out = {"git_commit": git_commit(), "pairs": {}, "cells_beyond_margin": 0}
    for name, cuda_run, mps_run in PAIRS:
        cuda = acc_by_k(load_scored(cuda_run)[0])
        items, manifest = load_scored(mps_run)
        pool = set(manifest["config"]["operand_pool"]["months"])
        mps = acc_by_k(items, lambda it: it["query"] in pool)

        rows, beyond = [], []
        print(f"\n=== {name}: CUDA bf16 (n=100/k) vs MPS fp16 in-pool ===")
        print(f"{'k':>3} {'cuda':>6} {'mps':>6} {'delta':>7} {'margin':>7}")
        for k in KS:
            (pc, nc), (pm, nm) = cuda[k], mps[k]
            margin = two_prop_margin(pc, nc, pm, nm)
            over = abs(pc - pm) > margin
            if over:
                beyond.append(k)
            rows.append({"k": k, "cuda": pc, "n_cuda": nc, "mps": pm,
                         "n_mps": nm, "delta": pm - pc, "margin": margin,
                         "beyond": over})
            print(f"{k:>3} {pc:>6.2f} {pm:>6.2f} {pm - pc:>+7.2f} "
                  f"{margin:>7.2f}{'  <- BEYOND' if over else ''}")
        print(f"cells beyond the two-proportion 95% margin: {beyond or 'none'}")
        out["pairs"][name] = {"cuda_run": cuda_run, "mps_run": mps_run,
                              "rows": rows, "beyond": beyond}
        out["cells_beyond_margin"] += len(beyond)

    verdict = ("DEVICE RULED OUT — every paired cell reproduces within margin"
               if out["cells_beyond_margin"] == 0 else
               "PRECISION-SENSITIVITY EFFECT — paired cells drift; STOP")
    out["verdict"] = verdict
    print(f"\n{verdict}")

    from .results_io import write_guarded

    write_guarded(Path("results/stage2/device_check.json"),
                  json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
