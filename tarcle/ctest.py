"""Hypothesis-C test: does steering efficacy survive an operand-partition change?

Implements the criterion fixed in docs/decisions.md D18, which must be read
first — in particular why the metric is `logp_lift` and not accuracy (mid-cycle
argmax is quantised noise), and why the noise band is measured from split halves
rather than assumed.

    Delta(k) = logp_lift(FV from A, queries from A)
             - logp_lift(FV from A, queries from B)

averaged over both directions A->B and B->A, so any intrinsic difficulty
difference between the two operand halves cancels.

The second, discriminating signature is the distribution of signed prediction
shifts on transferred queries: hypothesis C predicts a *systematic wrong-region
push* (mass at a shift != k), not uniform degradation.

Usage:
    python -m tarcle.ctest experiments/ctl_primary.json \
        --a results/fv/ctl_months_partA --b results/fv/ctl_months_partB
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import causal
from .extract import describe, load_config, load_model
from .prompts import DOMAINS


def _load(run_dir: Path, method: str):
    """The FV file for a condition, found by glob so the condition name (which
    is part of the filename) does not have to be passed in."""
    hits = sorted(run_dir.glob(f"fv_*_{method}.npz"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one fv_*_{method}.npz in {run_dir}")
    z = np.load(hits[0], allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    return z, meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--method", default="todd")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    za, meta_a = _load(args.a, args.method)
    zb, meta_b = _load(args.b, args.method)
    fam = config.family
    ks = list(za["ks"])
    pool_a = meta_a["query_pool"][fam]
    pool_b = meta_b["query_pool"][fam]
    if set(pool_a) & set(pool_b):
        raise SystemExit("partitions overlap; the transfer test needs disjoint pools")
    layer = int(meta_a["injection_layer"])
    scale = float(meta_a["injection_scale"])
    mode = meta_a["injection_mode"]
    assert (layer, scale, mode) == (
        int(meta_b["injection_layer"]), float(meta_b["injection_scale"]),
        meta_b["injection_mode"],
    ), "partitions were scored under different injection protocols"

    model, tok = load_model(config)
    arch = describe(model)
    dev = model.device
    bs = config.batch_size

    def base(queries):
        return {
            k: causal.score_for_k(model, tok, arch, fam, k, bs, queries=queries)
            for k in ks
        }

    def lift(X, i, k, queries, baseline):
        v = torch.tensor(X[i], device=dev, dtype=torch.float32) * scale
        s = causal.score_for_k(
            model, tok, arch, fam, k, bs, v, layer, mode, queries=queries
        )
        return s["logp"] - baseline[k]["logp"], s

    base_a, base_b = base(pool_a), base(pool_b)
    print(f"method={args.method}  L{layer} x{scale} {mode}")
    print(f"A operands {pool_a}\nB operands {pool_b}\n")

    rows, shifts, shifts_matched = [], {}, {}
    for i, k in enumerate(ks):
        m_aa, s_aa = lift(za["X"], i, k, pool_a, base_a)   # matched
        m_bb, s_bb = lift(zb["X"], i, k, pool_b, base_b)   # matched
        t_ab, s_ab = lift(za["X"], i, k, pool_b, base_b)  # A's FV -> B's queries
        t_ba, s_ba = lift(zb["X"], i, k, pool_a, base_a)  # B's FV -> A's queries
        delta = ((m_aa - t_ab) + (m_bb - t_ba)) / 2
        # The matched distribution is the baseline the transferred one has to be
        # read against: a push toward shift +/-1 is only evidence for C if the
        # matched condition does not do the same thing.
        shifts_matched[int(k)] = np.concatenate(
            [s_aa["pred_shift"], s_bb["pred_shift"]]
        ).tolist()

        # Split-half noise band: the same Delta when only the prompt draw changes.
        h_a, _ = lift(za["X_half_a"], i, k, pool_a, base_a)
        h_b, _ = lift(za["X_half_b"], i, k, pool_a, base_a)
        g_a, _ = lift(zb["X_half_a"], i, k, pool_b, base_b)
        g_b, _ = lift(zb["X_half_b"], i, k, pool_b, base_b)
        band = (abs(h_a - h_b) + abs(g_a - g_b)) / 2

        shifts[int(k)] = np.concatenate(
            [s_ab["pred_shift"], s_ba["pred_shift"]]
        ).tolist()
        rows.append({
            "k": int(k), "matched": (m_aa + m_bb) / 2,
            "transferred": (t_ab + t_ba) / 2, "delta": delta, "band": band,
            "exceeds": bool(delta > band),
        })
        print(f"  k={k:>2}  matched {(m_aa + m_bb) / 2:+.3f}  "
              f"transferred {(t_ab + t_ba) / 2:+.3f}  "
              f"Delta {delta:+.3f}  band {band:.3f}  "
              f"{'EXCEEDS' if delta > band else '-'}")

    deltas = np.array([r["delta"] for r in rows])
    bands = np.array([r["band"] for r in rows])
    n_exceed = int(sum(r["exceeds"] for r in rows))
    verdict = (
        "C true (transfer degrades)"
        if deltas.mean() > bands.mean() and n_exceed > len(rows) / 2
        else "C false (within band)"
        if deltas.mean() <= bands.mean()
        else "ambiguous"
    )
    print(f"\nmean Delta {deltas.mean():+.4f}   mean band {bands.mean():.4f}   "
          f"exceeds at {n_exceed}/{len(rows)} k")
    print(f"D18 verdict: {verdict}")
    print(f"near threshold (within one band): "
          f"{abs(deltas.mean() - bands.mean()) < bands.mean()}")

    print("\nsigned prediction shift, matched vs transferred "
          "(D18 wrong-region signature). P(+/-1) is the share of predictions "
          "landing on shift 1 or 11 regardless of k:")
    n = len(DOMAINS[fam])
    print(f"  {'k':>3} {'matched: correct  +/-1':>26}   "
          f"{'transferred: correct  +/-1':>28}")
    collapse = []
    for k in ks:
        cm = np.bincount(shifts_matched[int(k)], minlength=n)
        ct = np.bincount(shifts[int(k)], minlength=n)
        pm, pt = cm / cm.sum(), ct / ct.sum()
        near = [1, n - 1]
        row = (int(k), float(pm[k]), float(pm[near].sum()),
               float(pt[k]), float(pt[near].sum()))
        collapse.append(row)
        print(f"  {k:>3} {pm[k]:>16.2f} {pm[near].sum():>9.2f}   "
              f"{pt[k]:>18.2f} {pt[near].sum():>9.2f}")

    mid = [r for r in collapse if r[0] not in (0, 1, 11)]
    print(f"\n  excluding k in {{0,1,11}} (where +/-1 IS correct):")
    print(f"    matched     P(correct) {np.mean([r[1] for r in mid]):.2f}   "
          f"P(+/-1) {np.mean([r[2] for r in mid]):.2f}")
    print(f"    transferred P(correct) {np.mean([r[3] for r in mid]):.2f}   "
          f"P(+/-1) {np.mean([r[4] for r in mid]):.2f}")

    out = {
        "method": args.method, "rows": rows, "pred_shift": shifts,
        "pred_shift_matched": shifts_matched,
        "collapse_table": collapse,
        "mean_delta": float(deltas.mean()), "mean_band": float(bands.mean()),
        "n_exceed": n_exceed, "verdict": verdict,
        "pool_a": pool_a, "pool_b": pool_b,
        "injection": {"layer": layer, "scale": scale, "mode": mode},
    }
    from .results_io import input_stamp, write_guarded

    stamp = input_stamp([args.a.name, args.b.name])
    write_guarded(args.a.parent / f"ctest_{args.method}_{stamp}.json",
                  json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
