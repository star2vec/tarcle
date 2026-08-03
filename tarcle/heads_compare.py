"""Compare two head-identification sweeps. Stage-2 code: pure numpy, no torch.

The head set is the one instrument every downstream FV artifact shares, so how
reproducible its ranking is bounds how much confidence any head-level claim can
carry. Two sweeps differing only in random prompt draw give a free read on that.

Usage:
    python -m tarcle.heads_compare results/fv/<run_a> results/fv/<run_b>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (Spearman without a scipy dependency)."""
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(dir_a: Path, dir_b: Path) -> dict:
    za, zb = np.load(dir_a / "heads.npz"), np.load(dir_b / "heads.npz")
    ma, mb = (json.loads(str(z["meta_json"])) for z in (za, zb))
    aie_a, aie_b = za["aie"].ravel(), zb["aie"].ravel()
    set_a = {tuple(x) for x in za["head_set"].tolist()}
    set_b = {tuple(x) for x in zb["head_set"].tolist()}

    overlap = set_a & set_b
    rho = spearman(aie_a, aie_b)
    pearson = float(np.corrcoef(aie_a, aie_b)[0, 1])

    # Restricted to cells that actually carry signal: rank correlation over 672
    # cells is dominated by the ~650 that do nothing, where the ordering is pure
    # noise and a low value would say nothing about the heads we care about.
    top = np.argsort(-np.maximum(aie_a, aie_b))[:40]
    rho_top = spearman(aie_a[top], aie_b[top])

    print(f"A: {dir_a.name}   git {ma['git_commit'][:8]}")
    print(f"B: {dir_b.name}   git {mb['git_commit'][:8]}\n")
    print(f"top-10 head-set overlap: {len(overlap)}/10")
    print(f"  in both : {sorted(overlap)}")
    if set_a - set_b:
        print(f"  A only  : {sorted(set_a - set_b)}")
    if set_b - set_a:
        print(f"  B only  : {sorted(set_b - set_a)}")
    print(f"\nAIE Spearman over all {len(aie_a)} cells: {rho:+.4f}")
    print(f"AIE Pearson  over all {len(aie_a)} cells: {pearson:+.4f}")
    print(f"AIE Spearman over the top-40 union cells: {rho_top:+.4f}")

    order_a = [tuple(x) for x in np.array(np.unravel_index(
        np.argsort(-aie_a), za["aie"].shape)).T.tolist()]
    order_b = [tuple(x) for x in np.array(np.unravel_index(
        np.argsort(-aie_b), zb["aie"].shape)).T.tolist()]
    print("\nrank of each A-top-10 cell in B:")
    for cell in order_a[:10]:
        print(f"  L{cell[0]:>2} H{cell[1]:<3} A#{order_a.index(cell) + 1:<3} "
              f"B#{order_b.index(cell) + 1}")

    return {
        "run_a": dir_a.name, "run_b": dir_b.name,
        "git_a": ma["git_commit"], "git_b": mb["git_commit"],
        "overlap": len(overlap),
        "overlap_cells": sorted(list(c) for c in overlap),
        "a_only": sorted(list(c) for c in set_a - set_b),
        "b_only": sorted(list(c) for c in set_b - set_a),
        "spearman_all": rho, "pearson_all": pearson, "spearman_top40": rho_top,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args(argv)
    out = compare(args.run_a, args.run_b)
    (args.run_b / "sweep_comparison.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
