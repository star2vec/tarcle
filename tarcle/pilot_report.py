"""Scoring, tables, plots, and the gate verdict for a pilot run.

Reads only the saved files in results/pilot/<run_name>/ (prompts.jsonl,
scores.jsonl, manifest.json). No torch, no GPU, no network — rerunnable anywhere.

Usage:
    python -m tarcle.pilot_report results/pilot/<run_name>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .prompts import DOMAINS, load_prompt_set

GO_THRESHOLD = 0.50
NO_GO_THRESHOLD = 0.30

# Reference dataviz palette (light mode): categorical slots 1-2, chrome inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_run(run_dir: Path) -> tuple[list[dict], dict]:
    items = load_prompt_set(run_dir / "prompts.jsonl")
    with (run_dir / "scores.jsonl").open(encoding="utf-8") as f:
        for line in f:
            score = json.loads(line)
            items[score["idx"]].update(score)
    for item in items:
        pred = int(np.argmax(item["choice_logprobs"]))
        item["pred"] = item["choices"][pred]
        item["correct"] = item["pred"] == item["target"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return items, manifest


def accuracy_table(items: list[dict]) -> list[dict]:
    """Rows of (variant, k, stratum, domain) -> n, accuracy. domain='all' rows always
    present; per-domain rows added for the mixed variant."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        stratum = "copy" if it["query_in_demos"] else "heldout"
        groups[(it["variant"], it["k"], stratum, "all")].append(it)
        if it["variant"] == "mixed":
            groups[(it["variant"], it["k"], stratum, it["domain"])].append(it)
    return [
        {
            "variant": variant, "k": k, "stratum": stratum, "domain": domain,
            "n": len(group),
            "accuracy": sum(it["correct"] for it in group) / len(group),
        }
        for (variant, k, stratum, domain), group in sorted(groups.items())
    ]


def gate_variant(items: list[dict]) -> str:
    """The variant the gate is read off: the single-domain one ("days", "months"),
    preferring "days" when a run has several. "mixed" is a control, never the gate."""
    variants = {it["variant"] for it in items}
    if "days" in variants:
        return "days"
    single = sorted(variants - {"mixed"})
    if not single:
        raise ValueError("run has no single-domain variant to gate on")
    return single[0]


def gate_verdict(table: list[dict], ks: list[int], variant: str = "days") -> dict:
    """GO/NO-GO from the gate variant, held-out stratum, per k."""
    acc = {
        row["k"]: row["accuracy"]
        for row in table
        if (row["variant"], row["stratum"], row["domain"])
        == (variant, "heldout", "all")
    }
    missing = [k for k in ks if k not in acc]
    if missing:
        return {"verdict": "INCOMPLETE", "missing_ks": missing, "per_k": acc}
    if all(acc[k] >= GO_THRESHOLD for k in ks):
        verdict = "GO"
    elif any(acc[k] <= NO_GO_THRESHOLD for k in ks):
        verdict = "NO-GO"
    else:
        verdict = "MARGINAL"
    return {
        "verdict": verdict,
        "variant": variant,
        "per_k": acc,
        "below_go": [k for k in ks if acc[k] < GO_THRESHOLD],
        "at_floor": [k for k in ks if acc[k] <= NO_GO_THRESHOLD],
    }


def shift_confusion(
    items: list[dict], ks: list[int], variant: str = "days"
) -> np.ndarray:
    """P(predicted shift | true k) on a single-domain variant, held-out stratum.
    predicted shift = (pred_index - query_index) mod n, n = cycle length."""
    n = len(DOMAINS[variant])
    counts = np.zeros((len(ks), n))
    idx = {x: i for i, x in enumerate(DOMAINS[variant])}
    for it in items:
        if it["variant"] != variant or it["query_in_demos"]:
            continue
        pred_k = (idx[it["pred"]] - idx[it["query"]]) % n
        counts[ks.index(it["k"]), pred_k] += 1
    rows = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)


def _style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy(table: list[dict], items: list[dict], out_path: Path) -> None:
    variants = sorted({row["variant"] for row in table})
    fig, axes = plt.subplots(
        1, len(variants), figsize=(4.6 * len(variants), 3.6), squeeze=False
    )
    fig.patch.set_facecolor("#fcfcfb")
    for ax, variant in zip(axes[0], variants):
        _style_axes(ax)
        for stratum, color in (("heldout", BLUE), ("copy", ORANGE)):
            rows = [
                r for r in table
                if (r["variant"], r["stratum"], r["domain"]) == (variant, stratum, "all")
            ]
            ax.plot(
                [r["k"] for r in rows], [r["accuracy"] for r in rows],
                color=color, linewidth=2, marker="o", markersize=7,
                label="held-out query" if stratum == "heldout" else "query in demos",
            )
        # Chance under uniform guessing = mean of 1/|choices| over this variant's items.
        chance = float(
            np.mean([1 / len(it["choices"]) for it in items if it["variant"] == variant])
        )
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            f"chance {chance:.2f}", (0.02, chance), xycoords=("axes fraction", "data"),
            va="bottom", fontsize=8, color=MUTED,
        )
        ax.set_title(f"{variant} operands", fontsize=11, color=INK)
        ax.set_xlabel("shift k", fontsize=9, color=MUTED)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(sorted({r["k"] for r in table}))
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    axes[0][0].set_ylabel("forced-choice accuracy", fontsize=9, color=MUTED)
    fig.suptitle("Shift-by-k ICL accuracy", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(
    conf: np.ndarray, ks: list[int], out_path: Path, variant: str = "days"
) -> None:
    n = conf.shape[1]
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.2, 0.26 * len(ks) + 2.2))
    fig.patch.set_facecolor("#fcfcfb")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    ax.imshow(conf, cmap=cmap, vmin=0, vmax=1)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            if conf[i, j] >= 0.005:
                ax.text(
                    j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if conf[i, j] > 0.55 else INK,
                )
    ax.set_xticks(range(conf.shape[1]))
    ax.set_yticks(range(len(ks)), ks)
    ax.set_xlabel(f"predicted shift (mod {n})", fontsize=9, color=MUTED)
    ax.set_ylabel("true k", fontsize=9, color=MUTED)
    ax.set_title(
        f"Predicted-shift confusion ({variant}, held-out)", fontsize=11, color=INK
    )
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="results/pilot/<run_name>")
    args = parser.parse_args(argv)

    items, manifest = load_run(args.run_dir)
    ks = manifest["config"]["ks"]
    model = manifest["config"]["model"]

    table = accuracy_table(items)
    with (args.run_dir / "accuracy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "k", "stratum", "domain", "n", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(table)

    plot_accuracy(table, items, args.run_dir / "accuracy_vs_k.png")
    variant = gate_variant(items)
    plot_confusion(
        shift_confusion(items, ks, variant),
        ks,
        args.run_dir / f"confusion_{variant}.png",
        variant,
    )

    verdict = gate_verdict(table, ks, variant)
    (args.run_dir / "report.json").write_text(
        json.dumps({"model": model, "gate": verdict, "table": table}, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )

    print(f"model: {model}")
    print(f"{'variant':8} {'k':>2} {'stratum':8} {'n':>4} accuracy")
    for row in (r for r in table if r["domain"] == "all"):
        print(
            f"{row['variant']:8} {row['k']:>2} {row['stratum']:8} "
            f"{row['n']:>4} {row['accuracy']:8.3f}"
        )
    print(f"\ngate ({variant}, held-out, GO>={GO_THRESHOLD}, "
          f"NO-GO<={NO_GO_THRESHOLD}): {verdict['verdict']}")
    if verdict.get("below_go"):
        print(f"  below GO threshold: k={verdict['below_go']}")
    if "gpt2" in model:
        print("  note: gpt2 is the pipeline smoke model — this verdict carries no "
              "scientific weight; the gate runs on Llama-3.2-3B.")


if __name__ == "__main__":
    main()
