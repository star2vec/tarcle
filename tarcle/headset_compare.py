"""Compare FV conditions that differ only in head set. Pure numpy, no torch.

D16 fixes the agreement criterion for the *geometry* diagnostics, which run off
this box. What this reports is the stage-1 input to that decision: how far apart
the vectors themselves are, measured against the split-half band that D16 makes
the reference for "further apart than noise".

D16 §4 attribution, since all-k differs from canonical by two removals AND two
additions:

    canonical vs intersection-8   -> the dropped heads (L14H14, L18H2)
    intersection-8 vs all-k       -> the added heads (L19H19, L14H6)
    canonical vs all-k            -> joint, unresolved if both pairwise agree

Usage:
    python -m tarcle.headset_compare results/fv/<run_a> results/fv/<run_b> [...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def load(run_dir: Path, method: str) -> dict:
    z = np.load(run_dir / f"fv_primary_{method}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta_json"}
    d["meta"] = json.loads(str(z["meta_json"]))
    d["name"] = run_dir.name
    # head_set_cells was added after the canonical run was written; for Todd the
    # head_set array carries the same information, and Hendel has no head set.
    if "head_set_cells" not in d["meta"]:
        d["meta"]["head_set_cells"] = (
            d["head_set"].tolist() if "head_set" in d else []
        )
    return d


def split_half_band(d: dict) -> np.ndarray:
    """Per-k cos(half_a, half_b): the noise ceiling D16 makes the reference."""
    return cos_rows(d["X_half_a"], d["X_half_b"])


def compare(runs: list[Path], method: str) -> dict:
    ds = [load(r, method) for r in runs]
    base = ds[0]
    ks = list(base["ks"])
    band = split_half_band(base)

    print(f"\n{'=' * 72}\n{method}  (reference: {base['name']})")
    for d in ds:
        m = d["meta"]
        print(f"  {d['name']:36} {len(m['head_set_cells']):>2} heads  "
              f"L{m['injection_layer']} x{m['injection_scale']}")
    same_protocol = len({
        (d["meta"]["injection_layer"], d["meta"]["injection_scale"]) for d in ds
    }) == 1
    same_prompts = len({json.dumps(d["meta"]["prompt_sha256"], sort_keys=True)
                        for d in ds}) == 1
    print(f"  identical injection protocol: {same_protocol}   "
          f"identical prompt sets: {same_prompts}")
    if not (same_protocol and same_prompts):
        print("  WARNING: head set is not the only variable — comparison is invalid")

    out = {"method": method, "reference": base["name"],
           "same_protocol": same_protocol, "same_prompts": same_prompts,
           "split_half_band": band.tolist(), "pairs": {}}

    # Hendel takes a dummy-query hidden state; no head set enters its
    # computation. Across conditions that vary only the head set its vectors
    # must therefore be bit-identical. If they are not, the conditions differ in
    # something other than the head set and every comparison below is void.
    if method == "hendel":
        identical = all(np.array_equal(base["X"], d["X"]) for d in ds[1:])
        print(f"  head-set invariance (must hold for Hendel): {identical}")
        out["hendel_invariant"] = bool(identical)
        if not identical:
            print("  ERROR: Hendel FVs differ across head sets — a variable other "
                  "than the head set changed between these runs")

    for d in ds[1:]:
        c = cos_rows(base["X"], d["X"])
        exceeds = c < band
        label = f"{base['name'].split('_')[-1]} vs {d['name'].split('_')[-1]}"
        print(f"\n  --- {label} ---")
        print(f"  {'k':>3} {'cos':>8} {'band':>8} {'beyond noise?':>14} "
              f"{'|FV| a':>8} {'|FV| b':>8} {'acc a':>6} {'acc b':>6}")
        for i, k in enumerate(ks):
            print(f"  {k:>3} {c[i]:>8.5f} {band[i]:>8.5f} "
                  f"{('YES' if exceeds[i] else 'no'):>14} "
                  f"{base['norms'][i]:>8.3f} {d['norms'][i]:>8.3f} "
                  f"{base['efficacy_acc'][i]:>6.2f} {d['efficacy_acc'][i]:>6.2f}")
        print(f"  min cos {c.min():.5f}   cells beyond the split-half band: "
              f"{int(exceeds.sum())}/{len(ks)}")
        acc_delta = np.abs(base["efficacy_acc"] - d["efficacy_acc"])
        print(f"  max |Δ efficacy acc| {acc_delta.max():.2f}   "
              f"mean {acc_delta.mean():.3f}")
        out["pairs"][d["name"]] = {
            "cos": c.tolist(), "beyond_band": exceeds.tolist(),
            "n_beyond": int(exceeds.sum()), "min_cos": float(c.min()),
            "max_abs_acc_delta": float(acc_delta.max()),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+",
                        help="first is the reference (canonical)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = {m: compare(args.runs, m) for m in ("todd", "hendel")}
    dest = args.out or (args.runs[0] / "headset_comparison.json")
    dest.write_text(json.dumps(result, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
