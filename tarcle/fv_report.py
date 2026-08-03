"""Hard-stop-2 report for a primary FV extraction. Stage-2: pure numpy, no torch.

Deliberately NOT geometry. No Gram matrix, no circulant score, no spectrum —
those belong to stage 2 proper and looking at them here would mean inspecting
the primary result before the controls that qualify it have been run. What this
prints is only what is needed to decide whether the extraction is sound enough
to spend GPU-hours on controls:

- FV norms per k, both methods (the linear-vs-closed signature lives here, but
  it is reported as a number, not interpreted)
- split-half reliability cos(FV_a, FV_b): the noise ceiling every off-diagonal
  similarity must later be read against
- causal-efficacy lift per k (docs/decisions.md D2 arbiter, FV(0) excluded)
- the D4 escalation trigger: efficacy regressed on ICL accuracy, in-sweep k
  against out-of-sweep k

Usage:
    python -m tarcle.fv_report results/fv/<run_name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# docs/pilot_findings.md section 6, months 16-shot held-out accuracy.
ICL_ACC = {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00, 4: 0.76, 5: 0.76,
           6: 0.98, 7: 0.70, 8: 0.38, 9: 0.80, 10: 0.86, 11: 1.00}


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=-1)
    return num / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta_json"}
    d["meta"] = json.loads(str(z["meta_json"]))
    return d


def split_half(d: dict) -> np.ndarray:
    return cos(d["X_half_a"].astype(np.float64), d["X_half_b"].astype(np.float64))


def report_method(name: str, d: dict) -> dict:
    ks = list(d["ks"])
    meta = d["meta"]
    norms, rel = d["norms"], split_half(d)
    acc, lift = d["efficacy_acc"], d["efficacy_lift"]
    base = d["efficacy_baseline"]

    print(f"\n=== {name} ===")
    print(f"injection: layer L{meta['injection_layer']} mode "
          f"{meta['injection_mode']} scale {meta['injection_scale']}")
    print(f"{'k':>3} {'|FV|':>9} {'split-half cos':>15} {'zs base':>8} "
          f"{'inj acc':>8} {'lift':>7}   ICL")
    for i, k in enumerate(ks):
        print(f"{k:>3} {norms[i]:>9.3f} {rel[i]:>15.4f} {base[i]:>8.2f} "
              f"{acc[i]:>8.2f} {lift[i]:>+7.2f}   {ICL_ACC[k]:.2f}")

    cv = float(norms.std() / norms.mean())
    print(f"\nnorm CV {cv:.4f}   split-half cos: min {rel.min():.4f} "
          f"median {np.median(rel):.4f}")
    return {
        "method": name, "norm_cv": cv,
        "norms": norms.tolist(), "split_half_cos": rel.tolist(),
        "efficacy_acc": acc.tolist(), "efficacy_lift": lift.tolist(),
        "injection_layer": meta["injection_layer"],
    }


def escalation_trigger(d: dict, in_sweep: list[int]) -> dict:
    """docs/decisions.md D4, fixed before any efficacy score was seen.

    Regress lift on that k's ICL accuracy using the in-sweep k, then check
    whether the out-of-sweep k sit systematically below that fit. If they do,
    the head set is biased toward the strong subset and the all-12 sweep is
    warranted. k=0 is excluded throughout: its zero-shot baseline is at ceiling
    from the copy prior, so its lift is ~0 by construction (D2).
    """
    ks = list(d["ks"])
    lift = d["efficacy_lift"]
    pairs = [(int(k), ICL_ACC[k], float(lift[i])) for i, k in enumerate(ks) if k != 0]
    ins = [(a, l) for k, a, l in pairs if k in in_sweep]
    outs = [(k, a, l) for k, a, l in pairs if k not in in_sweep]
    if len(ins) < 2:
        return {"fired": None, "reason": "too few in-sweep k"}

    x = np.array([a for a, _ in ins])
    y = np.array([l for _, l in ins])
    slope, intercept = (np.polyfit(x, y, 1) if x.std() > 0 else (0.0, y.mean()))
    resid_in = y - (slope * x + intercept)
    scale = resid_in.std(ddof=1) if len(resid_in) > 2 else np.abs(resid_in).mean()
    resid_out = {k: float(l - (slope * a + intercept)) for k, a, l in outs}
    mean_out = float(np.mean(list(resid_out.values())))
    fired = bool(scale > 0 and mean_out < -scale)

    print(f"\n--- D4 escalation trigger ({d['meta']['method']}) ---")
    print(f"fit on in-sweep k={in_sweep}: lift = {slope:+.3f}*ICL {intercept:+.3f}"
          f"   (residual sd {scale:.4f})")
    for k in sorted(resid_out):
        print(f"  out-of-sweep k={k:>2}  residual {resid_out[k]:+.4f}")
    print(f"  mean out-of-sweep residual {mean_out:+.4f}  vs -1 sd = {-scale:+.4f}")
    print(f"  TRIGGER {'FIRES -> run the all-12 sweep' if fired else 'does not fire'}")
    return {
        "fired": fired, "slope": float(slope), "intercept": float(intercept),
        "residual_sd": float(scale), "out_residuals": resid_out,
        "mean_out_residual": mean_out,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    out = {}
    for method in ("todd", "hendel"):
        d = load(args.run_dir / f"fv_primary_{method}.npz")
        out[method] = report_method(method, d)
        in_sweep = list(d["meta"]["config"]["head_id_ks"])
        out[method]["d4"] = escalation_trigger(d, in_sweep)
        out[method]["head_set_source"] = d["meta"]["head_set_source"]["sha256"]

    same = out["todd"]["head_set_source"] == out["hendel"]["head_set_source"]
    print(f"\nboth methods stamped with the same head_set_source: {same}")

    # Cross-method agreement, reported but not adjudicated: CLAUDE.md rule 1
    # says a result holding under only one extraction is a finding about the
    # method, so the two are compared, never averaged.
    a = load(args.run_dir / "fv_primary_todd.npz")
    b = load(args.run_dir / "fv_primary_hendel.npz")
    r = np.corrcoef(a["efficacy_lift"], b["efficacy_lift"])[0, 1]
    print(f"corr(todd lift, hendel lift) across k = {r:+.3f}")
    out["cross_method_lift_corr"] = float(r)

    (args.run_dir / "fv_report.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
