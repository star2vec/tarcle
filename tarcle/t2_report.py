"""T2 verdict assembly. Pure numpy — stage-2 code, no torch.

Reads the hash-stamped sweep chunks written by tarcle/t2_sweep.py and applies
the registered branches (docs/preregistration_t7.md §2) verbatim:

  objection sustained  any condition's best grid cell has margin >= +0.10
                       with its 95% CI clear of 0
  objection closed     every condition's best-over-grid margin < +0.10
  ambiguous            best >= +0.10 but CI straddles 0 -> named cells
                       reported, no verdict

The report statistic is deliberately adversarial (max over 140 cells per
condition biases toward rescue); a rescue must clear noise, per registration.
No verdict is printed until every expected chunk is present — a partial sweep
reports progress only.

Usage:
    python -m tarcle.t2_report
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .results_io import write_guarded

OUT = Path("results/t2")
N_LAYERS = 28
EXPECTED = [(c, m) for c in ("partA", "partB", "halfA", "halfB")
            for m in ("todd", "hendel")]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    chunks = defaultdict(dict)
    for p in sorted(OUT.glob("chunk_*.npz")):
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        chunks[(meta["condition"], meta["method"])][meta["layer"]] = (p, z, meta)

    missing = {f"{c}/{m}": N_LAYERS - len(chunks.get((c, m), {}))
               for c, m in EXPECTED if len(chunks.get((c, m), {})) < N_LAYERS}
    if missing:
        print("sweep incomplete — no verdict is read (registration: the grid "
              "runs to completion). Missing layer-chunks:")
        for k, n in missing.items():
            print(f"  {k}: {n} of {N_LAYERS}")
        return

    out = {"git_commit": git_commit(), "conditions": {}, "verdict": None}
    sustained, ambiguous = [], []
    print(f"{'condition/method':>18} {'best margin':>12} {'ci95':>7} "
          f"{'at':>12} {'branch':>10}")
    for c, m in EXPECTED:
        best = None
        cells = []
        for layer, (p, z, meta) in sorted(chunks[(c, m)].items()):
            for si, scale in enumerate(meta["scales"]):
                cell = {"layer": layer, "scale": scale,
                        "margin": float(z["margin"][si]),
                        "ci95": float(z["margin_ci95"][si])}
                cells.append(cell)
                if best is None or cell["margin"] > best["margin"]:
                    best = cell
        cells.sort(key=lambda x: -x["margin"])
        rescued = best["margin"] >= 0.10 and best["margin"] - best["ci95"] > 0
        amb = best["margin"] >= 0.10 and not rescued
        if rescued:
            sustained.append(f"{c}/{m}")
        if amb:
            ambiguous.append(f"{c}/{m}")
        branch = "RESCUED" if rescued else ("ambiguous" if amb else "closed")
        out["conditions"][f"{c}/{m}"] = {
            "best": best, "top5": cells[:5], "n_cells": len(cells),
            "branch": branch}
        print(f"{c + '/' + m:>18} {best['margin']:>+12.3f} "
              f"{best['ci95']:>7.3f} L{best['layer']:02d} x"
              f"{best['scale']:<4} {branch:>10}")

    if sustained:
        out["verdict"] = ("OBJECTION SUSTAINED — claim A weakens to 'collapse "
                          f"under the shared protocol' (rescued: {sustained})")
    elif ambiguous:
        out["verdict"] = (f"AMBIGUOUS at {ambiguous} — registered re-scoring "
                          "branch applies; no verdict")
    else:
        out["verdict"] = ("OBJECTION CLOSED — every best-over-grid margin "
                          "< +0.10; collapse holds across the protocol space")
    print(f"\n{out['verdict']}")
    write_guarded(Path("results/t2/t2_report.json"),
                  json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
