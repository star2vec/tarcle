"""T1: the 2x2 support matrix, read against the pre-committed branches.

Registered in docs/preregistration_instruments.md §1 and decisions.md D40,
both committed before either run was launched. This reader computes, for the
off-diagonal runs (restricted demonstrations, FULL-cycle queries):

  - per-k held-out accuracy (the gate), with Wilson 95% CIs
  - the in-pool / out-of-pool query split (the join the prereg §1 describes)
  - the in-pool anchor check against the original in-distribution CUDA bf16
    gate (instrument validity: >= 3 k beyond the two-proportion margin flags
    the fp16/MPS instrument; the flag reports, it does not decide)
  - the branch decision, mechanically, per the registered rules, against the
    device-matched full-pool reference profile

Branch rules (quoted from the registration):
  Branch 1 (claim A survives strong): verdict GO at every k, or MARGINAL where
    every sub-0.50 cell (i) is sub-0.50 in the full-pool profile F too, and
    (ii) is not below F(k) by more than the two-proportion 95% margin.
  Branch 2 (claim A killed / rescoped): verdict NO-GO (any k < 0.30), or any
    cell with F(k) >= 0.50 falling below 0.50 by more than the two-proportion
    margin against F(k).
  Middle: cells below 0.50 but within the noise margin of an F(k) >= 0.50 ->
    recompute the D20 margin restricted to the k that clear 0.50 here (and not
    in {0,1,11}); claim A survives QUALIFIED iff that restricted margin stays
    <= -0.10; otherwise Branch 2.

Usage:
    python -m tarcle.support_gate
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .nextitem import TRIVIAL_K
from .prompts import DOMAINS, load_prompt_set

MONTHS = DOMAINS["months"]
GO, NO_GO = 0.50, 0.30
KS = list(range(12))

RUNS = [
    # off-diagonal run, in-distribution reference run, FV npz for the middle rule
    ("gate_months_partA4_fullq", "gate_months_partA4",
     "results/fv/ctl_months_partA/fv_partition_a_todd.npz"),
    ("gate_months_halves_A_fullq", "gate_months_halves_A",
     "results/fv/ctl_months_halfA/fv_half_a_todd.npz"),
]
FULLPOOL_REF = "llama32_3b_mps_months_s16"  # device-matched profile F(k)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_scored(run: str) -> tuple[list[dict], dict]:
    run_dir = Path("results/pilot") / run
    items = load_prompt_set(run_dir / "prompts.jsonl")
    with (run_dir / "scores.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            items[s["idx"]].update(s)
    for it in items:
        it["correct"] = (
            it["choices"][int(np.argmax(it["choice_logprobs"]))] == it["target"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return items, manifest


def acc_by_k(items: list[dict], pred=None) -> dict[int, tuple[float, int]]:
    out = {}
    for k in KS:
        cell = [it for it in items
                if it["k"] == k and not it["query_in_demos"]
                and (pred is None or pred(it))]
        out[k] = (float(np.mean([it["correct"] for it in cell])) if cell
                  else float("nan"), len(cell))
    return out


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return mid - half, mid + half


def two_prop_margin(p1: float, n1: int, p2: float, n2: int) -> float:
    return 1.96 * float(np.sqrt(p1 * (1 - p1) / max(n1, 1)
                                + p2 * (1 - p2) / max(n2, 1)))


def gate_verdict(acc: dict[int, tuple[float, int]]) -> str:
    ps = [acc[k][0] for k in KS]
    if any(p < NO_GO for p in ps):
        return "NO-GO"
    if all(p >= GO for p in ps):
        return "GO"
    return "MARGINAL"


def restricted_margin(npz_path: str, passing_ks: set[int]) -> dict:
    """The registered middle-case rule: D20 margin over the k that clear the
    gate here, from the saved efficacy_pred_shift (full-cycle queries)."""
    z = np.load(npz_path, allow_pickle=False)
    ks = [int(k) for k in z["ks"]]
    shifts = z["efficacy_pred_shift"]
    use = [i for i, k in enumerate(ks)
           if k in passing_ks and k not in TRIVIAL_K]
    if not use:
        return {"margin": None, "ks": []}
    rows = shifts[use]
    pc = float(np.mean(rows == np.array([ks[i] for i in use])[:, None]))
    pm1 = float(np.mean(np.isin(rows, [1, 11])))
    return {"margin": pc - pm1, "p_correct": pc, "p_pm1": pm1,
            "ks": sorted(ks[i] for i in use)}


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    ref_items, _ = load_scored(FULLPOOL_REF)
    F = acc_by_k([it for it in ref_items if it["variant"] == "months"])

    out = {"git_commit": git_commit(), "fullpool_reference": FULLPOOL_REF,
           "runs": {}}
    for run, indist_run, npz in RUNS:
        items, manifest = load_scored(run)
        pool = set(manifest["config"]["operand_pool"]["months"])
        agg = acc_by_k(items)
        acc_in = acc_by_k(items, lambda it: it["query"] in pool)
        acc_out = acc_by_k(items, lambda it: it["query"] not in pool)
        indist = acc_by_k(load_scored(indist_run)[0])

        verdict = gate_verdict(agg)
        sub = [k for k in KS if agg[k][0] < GO]
        # Branch 2 triggers
        kill_cells = [
            k for k in KS
            if F[k][0] >= GO and agg[k][0] < GO
            and (F[k][0] - agg[k][0]) > two_prop_margin(agg[k][0], agg[k][1],
                                                        F[k][0], F[k][1])
        ]
        nogo = verdict == "NO-GO"
        # Branch 1
        b1 = verdict == "GO" or (
            verdict == "MARGINAL"
            and all(F[k][0] < GO
                    and (F[k][0] - agg[k][0]) <= two_prop_margin(
                        agg[k][0], agg[k][1], F[k][0], F[k][1])
                    for k in sub))
        middle_cells = [k for k in sub if F[k][0] >= GO and k not in kill_cells]

        if nogo or kill_cells:
            branch, detail = "BRANCH 2 — claim A killed / rescoped", {
                "no_go": nogo, "kill_cells": kill_cells}
        elif b1:
            branch, detail = "BRANCH 1 — claim A survives strong", {}
        else:
            rm = restricted_margin(npz, {k for k in KS if agg[k][0] >= GO})
            ok = rm["margin"] is not None and rm["margin"] <= -0.10
            branch = ("QUALIFIED — claim A survives on the gate-clearing k"
                      if ok else "BRANCH 2 — claim A killed / rescoped")
            detail = {"middle_cells": middle_cells, "restricted_margin": rm}

        # In-pool anchor check (instrument validity)
        anchor_bad = [
            k for k in KS
            if abs(acc_in[k][0] - indist[k][0]) > two_prop_margin(
                acc_in[k][0], acc_in[k][1], indist[k][0], indist[k][1])
        ]
        flagged = len(anchor_bad) >= 3

        print(f"\n=== {run} (pool {sorted(pool)[0]}..., {len(pool)} operands; "
              f"full-cycle queries) ===")
        print(f"{'k':>3} {'acc':>6} {'wilson':>15} {'in-pool':>12} "
              f"{'out-pool':>12} {'F(k)':>6} {'orig':>6}")
        for k in KS:
            lo, hi = wilson(*agg[k])
            pin, nin = acc_in[k]
            pout, nout = acc_out[k]
            print(f"{k:>3} {agg[k][0]:>6.2f} [{lo:.2f},{hi:.2f}] n={agg[k][1]:<3}"
                  f" {pin:>6.2f} n={nin:<4} {pout:>6.2f} n={nout:<4}"
                  f" {F[k][0]:>6.2f} {indist[k][0]:>6.2f}")
        print(f"gate verdict: {verdict}   sub-0.50 cells: {sub or 'none'}")
        print(f"in-pool anchor vs original bf16 gate: "
              f"{'FLAGGED' if flagged else 'ok'} "
              f"(cells beyond margin: {anchor_bad or 'none'})")
        print(f"pre-committed branch: {branch}")
        if detail:
            print(f"  detail: {json.dumps(detail)}")

        out["runs"][run] = {
            "pool": sorted(pool), "verdict": verdict,
            "acc": {k: agg[k] for k in KS},
            "acc_in_pool": {k: acc_in[k] for k in KS},
            "acc_out_pool": {k: acc_out[k] for k in KS},
            "indist_reference": {k: indist[k] for k in KS},
            "fullpool_reference": {k: F[k] for k in KS},
            "sub_go_cells": sub, "branch": branch, "branch_detail": detail,
            "anchor_flagged": flagged, "anchor_cells": anchor_bad,
        }

    dest = Path("results/stage2/support_matrix.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
