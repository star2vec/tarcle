"""Why is FV efficacy near zero for most k? Two mechanical suspects, checked.

CLAUDE.md's style rule orders the suspects: extraction code, then prompt
leakage, then the model. Split-half reliability (~0.996) already rules out
noisy extraction, and the head->residual projection is verified against the
attention block's real output in tests. That leaves two *protocol* choices that
could produce a null even with perfectly good vectors:

1. injection scale — the FV is added at scale 1.0 by default; too small a push
   and nothing moves regardless of what the vector encodes
2. injection layer — one layer is chosen on the mean over the head-ID k subset
   and frozen for comparability, which could be the wrong layer for other k

If efficacy stays near zero across scale and across every layer, the null is
about the vectors, not the protocol.

Usage:
    python -m tarcle.diagnose_efficacy experiments/fv_months_ada.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import causal
from .extract import describe, load_config, load_model

SCALES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--ks", default="1,2,3,6,8,11")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = Path(config.results_dir) / config.run_name
    ks = [int(x) for x in args.ks.split(",")]

    model, tok = load_model(config)
    arch = describe(model)
    fam = config.family
    baseline = causal.baseline_accuracy(model, tok, arch, fam, ks, config.batch_size)
    out = {"baseline": baseline, "scale": {}, "layer": {}}

    for method, mode, fname in (
        ("todd", "add", "fv_primary_todd.npz"),
        ("hendel", "replace", "fv_primary_hendel.npz"),
    ):
        z = np.load(out_dir / fname)
        meta = json.loads(str(z["meta_json"]))
        layer = int(meta["injection_layer"])
        allk = list(z["ks"])
        X = z["X"]
        vec = {
            k: torch.tensor(X[allk.index(k)], device=model.device, dtype=torch.float32)
            for k in ks
        }

        print(f"\n=== {method}: injection scale at the frozen layer L{layer} ===")
        print(f"{'k':>3} " + " ".join(f"{'x' + str(s):>7}" for s in SCALES))
        out["scale"][method] = {}
        for k in ks:
            row = [
                causal.accuracy_for_k(
                    model, tok, arch, fam, k, config.batch_size,
                    vec[k] * s, layer, mode,
                )
                for s in SCALES
            ]
            out["scale"][method][k] = row
            print(f"{k:>3} " + " ".join(f"{a:>7.2f}" for a in row))

        print(f"\n=== {method}: every layer, scale 1.0 ===")
        out["layer"][method] = {}
        for k in ks:
            row = [
                causal.accuracy_for_k(
                    model, tok, arch, fam, k, config.batch_size, vec[k], l, mode,
                )
                for l in range(arch.n_layers)
            ]
            out["layer"][method][k] = row
            best = int(np.argmax(row))
            print(f"  k={k:>2}  best L{best} acc {row[best]:.2f}   "
                  f"(frozen L{layer} acc {row[layer]:.2f})   "
                  f"max over all layers {max(row):.2f}")

    (out_dir / "efficacy_diagnosis.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\nwrote {out_dir / 'efficacy_diagnosis.json'}")


if __name__ == "__main__":
    main()
