"""T3: the task-encoding margin split by operand exposure. Pure numpy.

Registered in docs/preregistration_instruments.md §2 (D39). D19-D21 read the
collapsed restricted-pool FVs as "the vector encodes next-item". A second
reading was never separated from it: the model solved the restricted task by
in-context lookup over the demonstrated operands, and the vector faithfully
reports THAT function — encoding an operand-bound mapping rather than a
degenerate default. The second reading is better for the paper, so it is
tested rather than defended against.

The split is exact, not reconstructed: `efficacy_pred_shift[:, j]` scores the
zero-shot query `Q: <month_j>\\nA:` where j indexes DOMAINS["months"] in
canonical order (causal.zero_shot_spec with queries=None), and exposure is
membership of month_j in the condition's `operand_pool` from `meta_json`.
Over 100 x 16 demonstration draws every pool member appears and no non-member
ever does, so pool membership IS exposure.

Margin CI: per prediction, d = 1{shift == k} - 1{shift in {+/-1}} in
{-1, 0, 1}; the margin is mean(d) over k not in {0, 1, 11} x the query subset,
with a normal-approximation 95% CI on that mean (the two proportions are
dependent — same predictions — so the difference indicator is the honest
unit). Readings fire only when the CI excludes the opposing threshold
(prereg §2).

Usage:
    python -m tarcle.margin_split
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .nextitem import TRIVIAL_K
from .prompts import DOMAINS

MONTHS = DOMAINS["months"]
GATE = 0.10  # the D20 threshold, reused for both sides of the split

CONDITIONS = [
    ("partA (Jan-Apr)", "results/fv/ctl_months_partA/fv_partition_a_{m}.npz"),
    ("partB (Sep-Dec)", "results/fv/ctl_months_partB/fv_partition_b_{m}.npz"),
    ("halfA (Jan-Jun)", "results/fv/ctl_months_halfA/fv_half_a_{m}.npz"),
    ("halfB (Jul-Dec)", "results/fv/ctl_months_halfB/fv_half_b_{m}.npz"),
    ("polysemy (9)", "results/fv/ctl_months_polysemy/fv_polysemy_leaveout_{m}.npz"),
    ("primary (12)", "results/fv/ctl_months_primary/fv_primary_{m}.npz"),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def margin_over(shifts: np.ndarray, ks: list[int], cols: list[int]) -> dict:
    """Margin mean(d), d = 1{correct} - 1{+/-1}, over mid-cycle k x cols."""
    if not cols:
        return {"margin": None, "n": 0}
    correct, pm1 = [], []
    for i, k in enumerate(ks):
        if k in TRIVIAL_K:
            continue
        row = shifts[i, cols]
        correct.append((row == k).astype(float))
        pm1.append(np.isin(row, [1, 11]).astype(float))
    correct, pm1 = np.concatenate(correct), np.concatenate(pm1)
    d = correct - pm1
    m = float(d.mean())
    half = 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d)) if len(d) > 1 else np.inf
    return {
        "margin": m, "ci95": [m - half, m + half], "n": int(len(d)),
        "p_correct": float(correct.mean()), "p_pm1": float(pm1.mean()),
    }


def split_condition(path: Path) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    pool = set(meta["operand_pool"]["months"])
    ks = [int(k) for k in z["ks"]]
    shifts = z["efficacy_pred_shift"]
    in_cols = [j for j, m in enumerate(MONTHS) if m in pool]
    out_cols = [j for j, m in enumerate(MONTHS) if m not in pool]

    res = {
        "pool_size": len(pool),
        "in": margin_over(shifts, ks, in_cols),
        "out": margin_over(shifts, ks, out_cols),
    }
    m_in, m_out = res["in"], res["out"]

    # Pre-labelled readings (prereg §2): each side's CI must exclude the
    # opposing threshold for its half of a reading to fire.
    def clears(side, thr):  # CI entirely above thr
        return side["margin"] is not None and side["ci95"][0] > thr

    def fails(side, thr):  # CI entirely below thr
        return side["margin"] is not None and side["ci95"][1] < thr

    if m_out["n"] == 0:
        reading = "reference (no out-of-pool queries)"
    elif clears(m_in, GATE) and fails(m_out, -GATE):
        reading = "R-lookup"
    elif fails(m_in, -GATE):
        reading = "R-collapse"
    else:
        reading = "R-mixed"
    res["reading"] = reading
    return res


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    out = {"git_commit": git_commit(), "gate": GATE, "cells": {}}
    for method in ("todd", "hendel"):
        print(f"\n=== {method} ===")
        print(f"{'condition':>18} {'pool':>5} {'M_in':>7} {'CI':>18} "
              f"{'M_out':>7} {'CI':>18} {'reading':>12}")
        for name, pattern in CONDITIONS:
            r = split_condition(Path(pattern.format(m=method)))
            if r is None:
                print(f"{name:>18}  (not on disk)")
                continue
            out["cells"][f"{name}/{method}"] = r
            ci = lambda s: (f"[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}] n={s['n']}"
                            if s["margin"] is not None else "-")
            mi, mo = r["in"], r["out"]
            mo_s = f"{mo['margin']:+7.3f}" if mo["margin"] is not None else "      -"
            print(f"{name:>18} {r['pool_size']:>5} {mi['margin']:+7.3f} "
                  f"{ci(mi):>18} {mo_s} {ci(mo):>18} {r['reading']:>12}")

    dest = Path("results/stage2/margin_split.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
