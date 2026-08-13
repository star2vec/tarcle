"""T5: do the field's importance measures agree with each other? Pure numpy.

Registered in docs/preregistration_instruments.md §3 (D39). The field treats
held-out ICL accuracy, causal AIE, injection efficacy and FV norm as
interchangeable proxies for "how much does this representation matter". This
computes their full Spearman rank-correlation matrix across k = 1..11 on the
months family — k=0 excluded on registered grounds (no AIE exists for it, D3;
efficacy uninterpretable, D2).

The D37 lesson, applied before any correlation is read: each series carries
measurement noise, so the maximum observable correlation for a pair is
sqrt(r_a * r_b) with r the series' reliability. A pair whose ceiling is < 0.50
is noise-limited and carries no disagreement claim.

Usage:
    python -m tarcle.measure_corr
"""
from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

from .prompts import load_prompt_set

KS = list(range(1, 12))
PILOT = Path("results/pilot/llama32_3b_mps_months_s16")
HEADS_ALLK = Path("results/fv/months_llama32_3b_ada_allk/heads.npz")
HEADS_CANON = Path("results/fv/months_llama32_3b_ada/heads.npz")
PRIMARY = "results/fv/ctl_months_primary/fv_primary_{m}.npz"

# Prereg §3: the pre-labelled outcome is evaluated on this subset, under Todd.
HEADLINE_SERIES = ("acc", "aie_l14h1", "eff_logp_lift", "norm")
RHO_DISAGREE, CEIL_MIN, RHO_AGREE = 0.30, 0.70, 0.70


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def heldout_accuracy() -> tuple[np.ndarray, np.ndarray]:
    """Per-k held-out accuracy and its binomial SE from the 16-shot MPS pilot.

    The pilot's stratum is "both", so ~50 of the 100 prompts per k are held-out;
    the actual count per cell is used, not assumed.
    """
    items = load_prompt_set(PILOT / "prompts.jsonl")
    with (PILOT / "scores.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            items[s["idx"]].update(s)
    acc, se = [], []
    for k in KS:
        cell = [it for it in items
                if it["k"] == k and it["variant"] == "months"
                and not it["query_in_demos"]]
        correct = [
            it["choices"][int(np.argmax(it["choice_logprobs"]))] == it["target"]
            for it in cell
        ]
        p, n = float(np.mean(correct)), len(correct)
        acc.append(p)
        se.append(np.sqrt(max(p * (1 - p), 1e-9) / n))
    return np.array(acc), np.array(se)


def aie_series() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """L14H1 and canonical-head-set-mean AIE per k, from the all-k sweep — the
    only sweep with per-k AIE at all eleven k (D14). SEs are approximate: the
    stored aie_se is the SE of the mean pooled over 11 k, so the per-k SE is
    taken as aie_se * sqrt(11) (~12 samples per k instead of 132). Recorded as
    an approximation in the output.
    """
    allk = np.load(HEADS_ALLK, allow_pickle=False)
    canon = np.load(HEADS_CANON, allow_pickle=False)
    per_k, se = allk["aie_per_k"].astype(np.float64), allk["aie_se"].astype(np.float64)
    assert list(allk["head_id_ks"]) == KS
    l14h1 = per_k[:, 14, 1]
    l14h1_se = np.full(len(KS), se[14, 1] * np.sqrt(len(KS)))
    cells = [tuple(c) for c in canon["head_set"]]
    stack = np.stack([per_k[:, l, h] for l, h in cells])          # (10, 11)
    cell_se = np.array([se[l, h] for l, h in cells]) * np.sqrt(len(KS))
    setmean = stack.mean(axis=0)
    # Heads are not independent samples; this SE assumes they are and is an
    # approximation, recorded as such.
    setmean_se = np.full(len(KS), np.sqrt((cell_se**2).mean() / len(cells)))
    return {"aie_l14h1": (l14h1, l14h1_se), "aie_setmean": (setmean, setmean_se)}


def method_series(method: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Efficacy and norm series for one extraction method, k = 1..11."""
    z = np.load(PRIMARY.format(m=method), allow_pickle=False)
    ks = [int(k) for k in z["ks"]]
    idx = [ks.index(k) for k in KS]
    lift = z["efficacy_logp_lift"].astype(np.float64)[idx]
    # SE of the lift approximated by the injected side's per-query SE; the
    # baseline is a census of the same 12 queries and its spread is not stored.
    lift_se = z["efficacy_logp_se"].astype(np.float64)[idx]
    eacc = z["efficacy_acc"].astype(np.float64)[idx]
    eacc_se = np.sqrt(np.maximum(eacc * (1 - eacc), 1e-9) / z["efficacy_n"][idx])
    norm = z["norms"].astype(np.float64)[idx]
    na = np.linalg.norm(z["X_half_a"].astype(np.float64)[idx], axis=1)
    nb = np.linalg.norm(z["X_half_b"].astype(np.float64)[idx], axis=1)
    norm_se = np.abs(na - nb) / 2
    return {
        "eff_logp_lift": (lift, lift_se),
        "eff_acc": (eacc, eacc_se),
        "norm": (norm, norm_se),
        "_norm_halves": (na, nb),
    }


def reliability(values: np.ndarray, se: np.ndarray) -> float:
    """r = 1 - mean(se^2) / var(series). Clipped to [0, 1]."""
    var = float(np.var(values, ddof=1))
    if var <= 0:
        return 0.0
    return float(np.clip(1.0 - float(np.mean(se**2)) / var, 0.0, 1.0))


def spearman_brown(a: np.ndarray, b: np.ndarray) -> float:
    """Reliability of the mean of two halves from their cross-k correlation."""
    r = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip(2 * r / (1 + r), 0.0, 1.0)) if r > -1 else 0.0


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        ranks[x == v] = ranks[x == v].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 10_000,
                seed: int = 0) -> float:
    """Two-sided exact-permutation p for Spearman rho (prereg §3)."""
    rng = np.random.default_rng(seed)
    obs = abs(spearman(a, b))
    hits = sum(
        abs(spearman(a, b[rng.permutation(len(b))])) >= obs - 1e-12
        for _ in range(n_perm)
    )
    return (hits + 1) / (n_perm + 1)


def analyse(method: str, shared: dict, rel: dict) -> dict:
    ms = method_series(method)
    series = {**shared, **{k: v for k, v in ms.items() if not k.startswith("_")}}
    r = dict(rel)
    r["norm"] = spearman_brown(*ms["_norm_halves"])   # measured, not SE-derived
    r["eff_logp_lift"] = reliability(*ms["eff_logp_lift"])
    r["eff_acc"] = reliability(*ms["eff_acc"])

    pairs = {}
    for a, b in combinations(series, 2):
        va, vb = series[a][0], series[b][0]
        ceiling = float(np.sqrt(r[a] * r[b]))
        pairs[f"{a}|{b}"] = {
            "rho": spearman(va, vb),
            "p_perm": perm_pvalue(va, vb),
            "ceiling": ceiling,
            "noise_limited": ceiling < 0.50,
        }

    verdicts = []
    for a, b in combinations(HEADLINE_SERIES, 2):
        cell = pairs[f"{a}|{b}"]
        if cell["ceiling"] < CEIL_MIN:
            verdicts.append("uninformative")
        elif cell["rho"] <= RHO_DISAGREE:
            verdicts.append("disagree")
        elif cell["rho"] >= RHO_AGREE:
            verdicts.append("agree")
        else:
            verdicts.append("intermediate")
    if "disagree" in verdicts:
        outcome = "MEASURES DISAGREE"
    elif all(v in ("agree", "uninformative") for v in verdicts) and "agree" in verdicts:
        outcome = "MEASURES AGREE (T5 negative)"
    else:
        outcome = "INTERMEDIATE / NOISE-LIMITED"

    return {
        "method": method,
        "series": {k: {"values": v[0].tolist(), "se": v[1].tolist()}
                   for k, v in series.items()},
        "reliability": {k: float(r[k]) for k in series},
        "pairs": pairs,
        "headline_subset": list(HEADLINE_SERIES),
        "headline_verdicts": dict(zip(
            [f"{a}|{b}" for a, b in combinations(HEADLINE_SERIES, 2)], verdicts)),
        "outcome": outcome,
    }


def show(res: dict, shared_names: list[str]) -> None:
    names = list(res["series"])
    print(f"\n=== {res['method']} ===")
    print(f"{'k':>3} " + " ".join(f"{n:>13}" for n in names))
    vals = {n: res["series"][n]["values"] for n in names}
    for i, k in enumerate(KS):
        print(f"{k:>3} " + " ".join(f"{vals[n][i]:>13.4f}" for n in names))
    print("reliability: " + "  ".join(
        f"{n}={res['reliability'][n]:.3f}" for n in names))
    print(f"\n{'pair':>28} {'rho':>7} {'p':>7} {'ceiling':>8}")
    for pair, c in res["pairs"].items():
        flag = "  noise-limited" if c["noise_limited"] else ""
        print(f"{pair:>28} {c['rho']:>+7.3f} {c['p_perm']:>7.4f} "
              f"{c['ceiling']:>8.3f}{flag}")
    print(f"outcome ({'+'.join(res['headline_subset'])}): {res['outcome']}")


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    acc, acc_se = heldout_accuracy()
    aie = aie_series()
    shared = {"acc": (acc, acc_se), **aie}
    rel = {k: reliability(*v) for k, v in shared.items()}

    out = {
        "ks": KS,
        "inputs": {"pilot": str(PILOT), "heads_allk": str(HEADS_ALLK),
                   "heads_canon": str(HEADS_CANON), "fv": PRIMARY},
        "approximations": [
            "AIE per-k SE = pooled aie_se * sqrt(11); heads treated as "
            "independent for the set-mean SE",
            "efficacy lift SE = injected-side per-query SE only (baseline "
            "census spread not stored)",
        ],
        "registered_thresholds": {
            "rho_disagree": RHO_DISAGREE, "ceiling_min": CEIL_MIN,
            "rho_agree": RHO_AGREE, "n_perm": 10_000, "seed": 0,
            "spearman_crit_n11_5pct": 0.62,
        },
        "git_commit": git_commit(),
    }
    for method in ("todd", "hendel"):
        res = analyse(method, shared, rel)
        show(res, list(shared))
        out[method] = res

    dest = Path("results/stage2/measure_corr.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
