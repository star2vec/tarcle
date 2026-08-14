"""Figure 1 for the post: the gate cannot see the collapse. numpy + matplotlib
only, no torch; reads exclusively from saved results/.

Panel A — operand pool size vs (i) the behavioural gate's worst held-out cell
and (ii) the FV task-encoding margin, all six conditions as individual points
(the two 4-operand draws differ by 0.24 and the two 6-operand draws by 0.15;
averaging would overstate the threshold's sharpness). Both quantities are
probability-scaled, so they share ONE axis — a dual-axis chart would be the
standard way to lie here. Registered GO threshold (0.50) and margin = 0 are
marked. The polysemy and partition-B gate runs are 2026-08-14 MPS/fp16
re-runs (D45: no gate artifact for those two conditions was ever committed);
the others are the original runs.

Panel B — the T3 in-pool vs out-of-pool margin split per condition with the
95% CIs from margin_split.json. This is the panel that rejects the
in-context-lookup alternative: the collapsed vectors are negative ON THE
DEMONSTRATED OPERANDS.

Every plotted number is loaded from its source file, and cross-asserted:
aggregate margins are recomputed from the .npz efficacy_pred_shift and must
match the n-weighted combination of margin_split.json's in/out cells; gate
minima recomputed from scores.jsonl must match support_matrix.json where both
exist. No hand-transcribed values.

Usage:
    python -m tarcle.post_figure
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .margin_split import CONDITIONS as SPLIT_CONDITIONS
from .nextitem import TRIVIAL_K
from .pilot_report import BASELINE, BLUE, GRID, INK, MUTED, ORANGE, _style_axes
from .support_gate import acc_by_k, load_scored

import matplotlib.pyplot as plt

SPLIT_JSON = Path("results/stage2/margin_split.json")
SUPPORT_JSON = Path("results/stage2/support_matrix.json")
OUT = Path("results/stage2/fig_gate_vs_encoding")

# condition -> (pool size, A/B letter, gate run dir or None, fullq run dir or None)
PANEL_A = {
    "primary (12)": (12, "", "llama32_3b_mps_months_s16", None),
    "polysemy (9)": (9, "", "gate_months_polysemy", None),
    "halfA (Jan-Jun)": (6, "A", "gate_months_halves_A", "gate_months_halves_A_fullq"),
    "halfB (Jul-Dec)": (6, "B", "gate_months_halves_B", None),
    "partA (Jan-Apr)": (4, "A", "gate_months_partA4", "gate_months_partA4_fullq"),
    "partB (Sep-Dec)": (4, "B", "gate_months_partB4", None),
}
EXPECTED_N = {  # census counts per side, mid-cycle k x query columns
    "primary (12)": (108, 0), "polysemy (9)": (81, 27),
    "halfA (Jan-Jun)": (54, 54), "halfB (Jul-Dec)": (54, 54),
    "partA (Jan-Apr)": (36, 72), "partB (Sep-Dec)": (36, 72),
}


def npz_margin(pattern: str) -> float:
    """The D20 margin recomputed from the saved arrays — the cross-check."""
    z = np.load(pattern.format(m="todd"), allow_pickle=False)
    ks = [int(k) for k in z["ks"]]
    rows = [i for i, k in enumerate(ks) if k not in TRIVIAL_K]
    shifts = z["efficacy_pred_shift"][rows]
    correct = np.array([ks[i] for i in rows])[:, None]
    return float(np.mean(shifts == correct) - np.mean(np.isin(shifts, [1, 11])))


def combined(cell: dict) -> float:
    """n-weighted in/out combination from margin_split.json."""
    m_in, m_out = cell["in"], cell["out"]
    if m_out["n"] == 0:
        return m_in["margin"]
    n = m_in["n"] + m_out["n"]
    return (m_in["margin"] * m_in["n"] + m_out["margin"] * m_out["n"]) / n


def gate_min(run: str, expect_n: int) -> float:
    items, _ = load_scored(run)
    acc = acc_by_k(items)
    assert all(n >= expect_n for _, n in acc.values()), \
        f"{run}: unexpected held-out cell size {acc}"
    return min(p for p, _ in acc.values())


def main() -> None:
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))["cells"]
    support = json.loads(SUPPORT_JSON.read_text(encoding="utf-8"))["runs"]
    patterns = dict(SPLIT_CONDITIONS)

    data = {}
    for name, (pool, letter, gate_run, fullq_run) in PANEL_A.items():
        cell = split[f"{name}/todd"]
        assert (cell["in"]["n"], cell["out"]["n"]) == EXPECTED_N[name], name
        margin = combined(cell)
        recomputed = npz_margin(patterns[name])
        assert abs(margin - recomputed) < 1e-9, \
            f"{name}: split-combined {margin} != npz {recomputed}"
        gate = gate_min(gate_run, 45) if gate_run else None
        fullq = None
        if fullq_run:
            fullq = gate_min(fullq_run, 95)
            ref = min(p for p, _ in support[fullq_run]["acc"].values())
            assert abs(fullq - ref) < 1e-9, f"{fullq_run}: {fullq} != {ref}"
        data[name] = {"pool": pool, "letter": letter, "margin": margin,
                      "cell": cell, "gate": gate, "fullq": fullq}
        print(f"  {name:>18}: margin {margin:+.3f}"
              f"  gate_min {gate if gate is not None else '   --'}"
              f"  fullq_min {fullq if fullq is not None else '--'}")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(10.8, 4.4), constrained_layout=True)

    # ---- Panel A --------------------------------------------------------------
    off = {"": 0.0, "A": -0.14, "B": +0.14}
    for name, d in data.items():
        x = d["pool"] + off[d["letter"]]
        ax_a.plot(x, d["margin"], "D", color=ORANGE, ms=8, zorder=4)
        if d["gate"] is not None:
            ax_a.plot(x, d["gate"], "o", color=BLUE, ms=8, zorder=4)
        if d["fullq"] is not None:
            ax_a.plot(x, d["fullq"], "o", ms=8, mfc="white", mec=BLUE,
                      mew=1.6, zorder=4)
        if d["letter"]:
            ax_a.annotate(d["letter"], (x, d["margin"]),
                          textcoords="offset points", xytext=(7, -3),
                          fontsize=8, color=MUTED)
    ax_a.annotate("worst cell k=8 in both",
                  (12, data["primary (12)"]["gate"]),
                  textcoords="offset points", xytext=(6, -20),
                  fontsize=8, color=MUTED, va="top", ha="right")
    ax_a.axhline(0.50, color=BLUE, lw=1, ls=(0, (4, 3)), alpha=0.55, zorder=1)
    ax_a.annotate("registered GO threshold (0.50)", (8.55, 0.475),
                  fontsize=8, color=BLUE, alpha=0.8, va="top", ha="right")
    ax_a.axhline(0.0, color=BASELINE, lw=1, zorder=1)
    ax_a.annotate("margin = 0", (12.55, 0.02), fontsize=8, color=MUTED,
                  va="bottom", ha="right")
    ax_a.set_xticks([4, 6, 9, 12])
    ax_a.set_xlim(3.1, 12.9)
    ax_a.set_ylim(-1.06, 1.06)
    ax_a.set_xlabel("operand pool size (distinct operands available to demos)")
    ax_a.set_ylabel("probability scale (accuracy / margin)")
    ax_a.set_title("A — the gate stays GO while the vector collapses",
                   fontsize=10, color=INK, loc="left")
    ax_a.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8,
                   label="behavioural gate, worst cell (in-distribution queries)"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.6,
                   ms=8, label="gate, worst cell, full-cycle queries (T1 audit)"),
        plt.Line2D([], [], marker="D", ls="", color=ORANGE, ms=8,
                   label="FV task-encoding margin (D20)"),
    ], loc="upper center", fontsize=8, frameon=False)
    _style_axes(ax_a)

    # ---- Panel B --------------------------------------------------------------
    order = ["primary (12)", "polysemy (9)", "halfA (Jan-Jun)",
             "halfB (Jul-Dec)", "partA (Jan-Apr)", "partB (Sep-Dec)"]
    labels = ["primary (12)", "polysemy (9)", "half A (6)", "half B (6)",
              "part A (4)", "part B (4)"]
    for i, name in enumerate(order):
        y = len(order) - 1 - i
        for side, dy, filled in (("in", +0.17, True), ("out", -0.17, False)):
            s = data[name]["cell"][side]
            if s["n"] == 0:
                continue
            lo, hi = s["ci95"]
            ax_b.errorbar(
                s["margin"], y + dy,
                xerr=[[s["margin"] - lo], [hi - s["margin"]]],
                fmt="o", ms=8, color=INK, mfc=INK if filled else "white",
                mec=INK, mew=1.4, elinewidth=1.2, capsize=2.5, zorder=4)
    ax_b.axvline(0.0, color=BASELINE, lw=1, zorder=1)
    for x in (-0.10, +0.10):
        ax_b.axvline(x, color=GRID, lw=1, ls=(0, (4, 3)), zorder=1)
    ax_b.annotate("D20 gate ±0.10", (0.14, 3.35), fontsize=8,
                  color=MUTED, va="center")
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels(labels[::-1], fontsize=9)
    ax_b.set_ylim(-0.6, len(order) - 0.4)
    ax_b.set_xlim(-1.08, 0.80)
    ax_b.set_xlabel("task-encoding margin, P(correct) − P(±1)")
    ax_b.set_title("B — collapse holds on demonstrated operands (rejects lookup)",
                   fontsize=10, color=INK, loc="left")
    ax_b.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=INK, ms=8,
                   label="in-pool queries (operand was demonstrated)"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.4,
                   ms=8, label="out-of-pool queries (never demonstrated)"),
    ], loc="upper left", fontsize=8, frameon=False)
    _style_axes(ax_b)

    fig.savefig(OUT.with_suffix(".svg"))
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print(f"wrote {OUT.with_suffix('.svg')} and .png (preview)")


if __name__ == "__main__":
    main()
