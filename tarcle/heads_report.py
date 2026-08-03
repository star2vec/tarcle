"""Report the identified FV head set from heads.npz. Stage-2 code: pure numpy,
no torch, no GPU — rerunnable anywhere.

Usage:
    python -m tarcle.heads_report results/fv/<run_name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Todd et al. 2024 (arXiv:2310.15213) find FV heads concentrated in the early-to-
# middle layers rather than at either end — for GPT-J (28 layers) the strongest
# heads sit mostly around layers 8-18, i.e. roughly 30-65% of relative depth.
# Used only as a sanity band: heads piled up at depth ~0 or ~1 are the signature
# of a broken capture (wrong tensor, wrong token position, wrong head slicing),
# which CLAUDE.md's style rule says to suspect before suspecting the model.
TODD_DEPTH_BAND = (0.29, 0.64)


def load(run_dir: Path) -> dict:
    z = np.load(run_dir / "heads.npz", allow_pickle=False)
    data = {k: z[k] for k in z.files if k != "meta_json"}
    data["meta"] = json.loads(str(z["meta_json"]))
    return data


def depth(layer: int, n_layers: int) -> float:
    return layer / (n_layers - 1)


def report(run_dir: Path) -> dict:
    d = load(run_dir)
    meta = d["meta"]
    n_layers, n_heads = meta["n_layers"], meta["n_heads"]
    aie, se = d["aie"], d["aie_se"]
    shortlist = d["shortlist"]
    confirm, confirm_se, confirm_n = d["confirm_aie"], d["confirm_se"], d["confirm_n"]
    head_set = d["head_set"]

    order = np.argsort(-confirm)
    rows = []
    for rank, i in enumerate(order, start=1):
        layer, head = int(shortlist[i][0]), int(shortlist[i][1])
        in_set = any(layer == l and head == h for l, h in head_set)
        rows.append({
            "rank": rank, "layer": layer, "head": head,
            "depth": depth(layer, n_layers),
            "aie_confirm": float(confirm[i]), "se_confirm": float(confirm_se[i]),
            "n": int(confirm_n[i]),
            "aie_sweep": float(aie[layer, head]), "se_sweep": float(se[layer, head]),
            "in_head_set": bool(in_set),
        })

    set_depths = [depth(int(l), n_layers) for l, _ in head_set]
    in_band = [TODD_DEPTH_BAND[0] <= x <= TODD_DEPTH_BAND[1] for x in set_depths]
    baseline = meta["corrupted_baseline_mean"]

    print(f"run: {run_dir.name}   model: {meta['config']['model']}")
    print(f"arch: {n_layers} layers x {n_heads} heads = {n_layers * n_heads} cells")
    print(f"sweep k: {list(meta['config']['head_id_ks'])}  "
          f"({int(d['n_samples'])} patched samples per cell)")
    print(f"corrupted baseline p(target): {baseline:.4f}  (chance = 1/12 = 0.0833)")
    print(f"stage A: {meta['stage_a_seconds'] / 60:.1f} min   "
          f"peak VRAM {meta['peak_vram_bytes'] / 2**30:.2f} GiB\n")

    print(f"{'rank':>4} {'layer':>5} {'head':>4} {'depth':>6} "
          f"{'AIE (confirm)':>20} {'AIE (sweep)':>14}  set")
    for r in rows[:20]:
        star = "*" if r["in_head_set"] else " "
        print(f"{r['rank']:>4} {r['layer']:>5} {r['head']:>4} {r['depth']:>6.2f} "
              f"{r['aie_confirm']:>+11.4f} ± {r['se_confirm']:.4f} "
              f"{r['aie_sweep']:>+9.4f}   {star}")

    print(f"\nhead set (top {len(head_set)}): "
          f"{[[int(l), int(h)] for l, h in head_set]}")
    print(f"layer depths: {[f'{x:.2f}' for x in set_depths]}")
    print(f"in Todd depth band {TODD_DEPTH_BAND}: {sum(in_band)}/{len(in_band)}")

    print("\nlayer histogram of the top-20 cells:")
    counts = np.zeros(n_layers, dtype=int)
    for r in rows[:20]:
        counts[r["layer"]] += 1
    for layer in range(n_layers):
        if counts[layer]:
            print(f"  L{layer:>2} (depth {depth(layer, n_layers):.2f}) "
                  f"{'#' * counts[layer]}")

    # Effect-size context: how much of the total causal effect the set carries.
    total_positive = float(aie[aie > 0].sum())
    set_sum = float(sum(aie[l, h] for l, h in head_set))
    print(f"\ntop-{len(head_set)} carry {set_sum / total_positive:.1%} of summed "
          f"positive AIE across all {n_layers * n_heads} cells")
    print(f"sweep AIE: max {aie.max():+.4f}  median {np.median(aie):+.4f}  "
          f"min {aie.min():+.4f}")

    # Rank stability between the cheap sweep and the higher-n confirmation.
    sweep_rank = np.argsort(-np.array([r["aie_sweep"] for r in rows]))
    confirm_rank = np.argsort(-np.array([r["aie_confirm"] for r in rows]))
    agree = len(set(sweep_rank[:10]) & set(confirm_rank[:10]))
    print(f"top-10 agreement between stage A and confirmation: {agree}/10")

    out = {
        "run": run_dir.name,
        "rows": rows,
        "head_set": [[int(l), int(h)] for l, h in head_set],
        "head_set_depths": set_depths,
        "in_todd_band": f"{sum(in_band)}/{len(in_band)}",
        "corrupted_baseline": baseline,
        "set_share_of_positive_aie": set_sum / total_positive,
        "stage_a_confirm_top10_agreement": agree,
    }
    (run_dir / "heads_report.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    report(parser.parse_args(argv).run_dir)


if __name__ == "__main__":
    main()
