"""D49: the partition-B efficacy quote, reproduced from the committed .npz.

The drafting read quoted, for ctl_months_partB / todd: injected accuracy mean
0.257 vs zero-shot baseline 0.083, logp_lift mean +1.19. This recomputes those
three means under both candidate definitions (all twelve k; k != 0, since the
identity task's baseline sits at ceiling per D2) and records which definition
the quoted figures correspond to. A mismatch under BOTH definitions is a STOP,
not a reconciliation.

Usage:
    python -m tarcle.efficacy_quote
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .results_io import write_guarded

NPZ = "results/fv/ctl_months_partB/fv_partition_b_todd.npz"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    z = np.load(NPZ, allow_pickle=False)
    ks = np.array([int(k) for k in z["ks"]])
    acc = z["efficacy_acc"].astype(float)
    base = z["efficacy_baseline"].astype(float)
    lift = z["efficacy_logp_lift"].astype(float)

    out = {"git_commit": git_commit(), "npz": NPZ, "definitions": {}}
    for label, mask in (("all_k", ks >= 0), ("k_ne_0", ks != 0)):
        out["definitions"][label] = {
            "efficacy_acc_mean": float(acc[mask].mean()),
            "baseline_mean": float(base[mask].mean()),
            "logp_lift_mean": float(lift[mask].mean()),
            "n_k": int(mask.sum()),
        }
        d = out["definitions"][label]
        print(f"{label:>7}: acc {d['efficacy_acc_mean']:.3f}  baseline "
              f"{d['baseline_mean']:.3f}  logp_lift {d['logp_lift_mean']:+.3f}"
              f"  (n_k={d['n_k']})")
    write_guarded(Path("results/stage2/efficacy_quote_partB.json"),
                  json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
