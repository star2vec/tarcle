"""Is the D2 arbiter verdict stable, or an artifact of a tuned free parameter?

The injection (layer, scale) pair is chosen on the in-sweep k and frozen. That
keeps the out-of-sweep cells honest, but it leaves two questions the arbiter
cannot answer on its own:

1. Does FV(8)'s verdict survive at neighbouring scales, or does it sit on a
   spike? An arbiter resting on one grid point is not an arbiter.
2. Is Hendel's collapse to +/-1 and +2 a property of the method, or just of
   scale 1.0 -- the value it is pinned to for principled reasons (D12)?

Both are answered by sweeping the full layer x scale grid per k and reporting
each k's frozen accuracy next to its best achievable accuracy anywhere on the
grid. The best-over-grid figure is an upper bound that no frozen protocol could
beat, so a k that stays low there is low because of its vector.

Usage:
    python -m tarcle.scale_stability experiments/fv_months_ada_v2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import causal
from .extract import describe, load_config, load_model

SCALES = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = Path(config.results_dir) / config.run_name
    model, tok = load_model(config)
    arch = describe(model)
    fam = config.family
    ks = list(config.ks)
    out = {"scales": SCALES}

    for method, mode in (("todd", "add"), ("hendel", "replace")):
        z = np.load(out_dir / f"fv_primary_{method}.npz")
        meta = json.loads(str(z["meta_json"]))
        frozen_layer = int(meta["injection_layer"])
        frozen_scale = float(meta["injection_scale"])
        allk = list(z["ks"])

        def vec(k, layer):
            src = (
                z["X"][allk.index(k)] if method == "todd"
                else z["X_all_layers"][allk.index(k)][layer]
            )
            return torch.tensor(src, device=model.device, dtype=torch.float32)

        print(f"\n=== {method}: accuracy at the frozen layer L{frozen_layer}, "
              f"by scale (frozen = x{frozen_scale}) ===")
        print(f"{'k':>3} " + " ".join(f"{'x' + str(s):>6}" for s in SCALES))
        by_scale = {}
        for k in ks:
            row = [
                causal.accuracy_for_k(
                    model, tok, arch, fam, k, config.batch_size,
                    vec(k, frozen_layer) * s, frozen_layer, mode,
                )
                for s in SCALES
            ]
            by_scale[k] = row
            print(f"{k:>3} " + " ".join(f"{a:>6.2f}" for a in row))

        print(f"\n=== {method}: best over the full 28 layers x {len(SCALES)} "
              f"scales grid, per k ===")
        print(f"{'k':>3} {'frozen':>7} {'best':>6}  at")
        grid = {}
        for k in ks:
            best, arg = -1.0, None
            for layer in range(arch.n_layers):
                for s in SCALES:
                    a = causal.accuracy_for_k(
                        model, tok, arch, fam, k, config.batch_size,
                        vec(k, layer) * s, layer, mode,
                    )
                    if a > best:
                        best, arg = a, (layer, s)
            grid[k] = {"best": best, "layer": arg[0], "scale": arg[1]}
            frozen = by_scale[k][SCALES.index(frozen_scale)]
            print(f"{k:>3} {frozen:>7.2f} {best:>6.2f}  L{arg[0]} x{arg[1]}")

        out[method] = {
            "frozen_layer": frozen_layer, "frozen_scale": frozen_scale,
            "by_scale_at_frozen_layer": by_scale, "best_over_grid": grid,
        }

    (out_dir / "scale_stability.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\nwrote {out_dir / 'scale_stability.json'}")


if __name__ == "__main__":
    main()
