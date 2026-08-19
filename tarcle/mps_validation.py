"""D50: MPS fp16 injection validation against the committed CUDA bf16 cells.

Registered in docs/decisions.md D50 BEFORE this script ran: subset (primary +
the four T2 conditions, both methods, all twelve k, full query cycle, frozen
protocols as stored per .npz), tolerances (accuracy: two-proportion 95% margin
at n=12/12; logp-lift: 1.96*sqrt(SE_cuda^2 + SE_mps^2) with >=95% of cells
within; decision statistic: per-condition D20 margin within the paired-
difference 95% CI and no gate-line class change), and branches (PASS extends
rule 4 to injection-on-MPS for T2 specifically; FAIL stops T2 for CUDA
hardware).

GPU code — imports torch. Output is a guarded JSON with every paired cell.

Usage:
    .venv/bin/python -m tarcle.mps_validation
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import causal
from .extract import describe
from .nextitem import TRIVIAL_K
from .results_io import write_guarded

MODEL = "meta-llama/Llama-3.2-3B"
BATCH = 12
KS = list(range(12))

CONDITIONS = [
    ("primary", "results/fv/ctl_months_primary/fv_primary_{m}.npz"),
    ("partA", "results/fv/ctl_months_partA/fv_partition_a_{m}.npz"),
    ("partB", "results/fv/ctl_months_partB/fv_partition_b_{m}.npz"),
    ("halfA", "results/fv/ctl_months_halfA/fv_half_a_{m}.npz"),
    ("halfB", "results/fv/ctl_months_halfB/fv_half_b_{m}.npz"),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def two_prop_margin(p1: float, p2: float, n: int = 12) -> float:
    return 1.96 * float(np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n))


def d20_margin(shifts: np.ndarray, ks: list[int]) -> np.ndarray:
    """Per-prediction d = 1{correct} - 1{+/-1} over mid-cycle k, flattened."""
    rows = [i for i, k in enumerate(ks) if k not in TRIVIAL_K]
    s = shifts[rows]
    kcol = np.array([ks[i] for i in rows])[:, None]
    return ((s == kcol).astype(float) - np.isin(s, [1, 11]).astype(float)).ravel()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.backends.mps.is_available(), "MPS not available"
    # Tokenizer configured exactly as extract.load_model does: pad = eos,
    # left padding so the last position is the true final token.
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to("mps").eval()
    model.config.use_cache = False
    arch = describe(model)

    baseline = causal.baseline_accuracy(model, tok, arch, "months", KS, BATCH)

    out = {"git_commit": git_commit(), "model": MODEL, "device": "mps",
           "dtype": "float16", "cells": [], "margins": {},
           "criteria": {}}
    acc_fail, lift_out, n_cells = [], 0, 0

    for cond, pattern in CONDITIONS:
        for method in ("todd", "hendel"):
            z = np.load(pattern.format(m=method), allow_pickle=False)
            meta = json.loads(str(z["meta_json"]))
            layer = int(meta["injection_layer"])
            scale = float(meta["injection_scale"])
            mode = meta["injection_mode"]
            X = z["X"].astype(np.float32)
            ks = [int(k) for k in z["ks"]]
            assert ks == KS

            shifts_mps = np.zeros((12, 12), dtype=np.int32)
            for i, k in enumerate(KS):
                v = torch.tensor(X[i], device="mps", dtype=torch.float32)
                s = causal.score_for_k(model, tok, arch, "months", k, BATCH,
                                       v * scale, layer, mode)
                lift_mps = s["logp"] - baseline[k]["logp"]
                se_mps = float(s["logp_per_query"].std(ddof=1) / np.sqrt(12))
                shifts_mps[i] = s["pred_shift"]

                acc_c = float(z["efficacy_acc"][i])
                lift_c = float(z["efficacy_logp_lift"][i])
                se_c = float(z["efficacy_logp_se"][i])
                cell = {
                    "condition": cond, "method": method, "k": k,
                    "acc_cuda": acc_c, "acc_mps": s["acc"],
                    "acc_margin": two_prop_margin(acc_c, s["acc"]),
                    "acc_within": bool(
                        abs(s["acc"] - acc_c) <= two_prop_margin(acc_c, s["acc"])),
                    "lift_cuda": lift_c, "lift_mps": float(lift_mps),
                    "lift_tol": 1.96 * float(np.sqrt(se_c**2 + se_mps**2)),
                    "lift_within": bool(
                        abs(lift_mps - lift_c)
                        <= 1.96 * float(np.sqrt(se_c**2 + se_mps**2))),
                }
                out["cells"].append(cell)
                n_cells += 1
                if not cell["acc_within"]:
                    acc_fail.append((cond, method, k))
                if not cell["lift_within"]:
                    lift_out += 1

            d_mps = d20_margin(shifts_mps, KS)
            d_cuda = d20_margin(z["efficacy_pred_shift"], KS)
            diff = d_mps - d_cuda
            ci = 1.96 * float(diff.std(ddof=1) / np.sqrt(len(diff))) \
                if diff.std(ddof=1) > 0 else 0.0
            m_mps, m_cuda = float(d_mps.mean()), float(d_cuda.mean())

            def cls(m):  # D20 gate classes
                return "healthy" if m > 0.10 else (
                    "collapsed" if m < -0.10 else "middle")

            out["margins"][f"{cond}/{method}"] = {
                "margin_cuda": m_cuda, "margin_mps": m_mps,
                "diff": m_mps - m_cuda, "diff_ci95": ci,
                "within": bool(abs(m_mps - m_cuda) <= ci),
                "class_cuda": cls(m_cuda), "class_mps": cls(m_mps),
                "class_stable": cls(m_cuda) == cls(m_mps),
            }
            mm = out["margins"][f"{cond}/{method}"]
            print(f"{cond:>8}/{method:<6} margin cuda {m_cuda:+.3f} mps "
                  f"{m_mps:+.3f} (diff {mm['diff']:+.3f}, ci {ci:.3f}) "
                  f"class {mm['class_cuda']}->{mm['class_mps']}"
                  f"{'  OK' if mm['within'] and mm['class_stable'] else '  FLAG'}",
                  flush=True)

    crit1 = len(acc_fail) == 0
    crit2 = (n_cells - lift_out) / n_cells >= 0.95
    crit3 = all(m["within"] and m["class_stable"]
                for m in out["margins"].values())
    out["criteria"] = {
        "acc_cells_outside": acc_fail, "acc_pass": crit1,
        "lift_cells_outside": lift_out, "lift_total": n_cells,
        "lift_share_within": (n_cells - lift_out) / n_cells,
        "lift_pass": crit2, "margin_pass": crit3,
        "VERDICT": "PASS" if (crit1 and crit2 and crit3) else "FAIL",
    }
    print(f"\ncriterion 1 (acc): {'pass' if crit1 else f'FAIL {acc_fail}'}")
    print(f"criterion 2 (lift): {n_cells - lift_out}/{n_cells} within "
          f"({(n_cells - lift_out) / n_cells:.3f}) -> "
          f"{'pass' if crit2 else 'FAIL'}")
    print(f"criterion 3 (margins): {'pass' if crit3 else 'FAIL'}")
    print(f"VERDICT: {out['criteria']['VERDICT']}")

    write_guarded(Path("results/stage2/mps_validation.json"),
                  json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
