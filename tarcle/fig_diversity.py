"""fig_diversity.py — margin & gate vs demonstration diversity (post figure 2).

House style per D46/D48: every plotted number is recomputed from committed
artifacts and asserted before drawing. Run from repo root:
    python -m tarcle.fig_diversity   (or python fig_diversity.py from root)
Outputs: results/stage2/fig_diversity_<hash>.svg / .png
"""
import json, glob, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 1. distinct-months-per-prompt means, from committed prompt files
def diversity(run):
    counts = []
    for jl in sorted(glob.glob(f"results/fv/{run}/prompts_k*.jsonl")):
        for line in open(jl):
            counts.append(len({d[1] for d in json.loads(line)["demos"]}))
    return float(np.mean(counts))

DIV = {
    "primary":  diversity("ctl_months_primary"),
    "polysemy": diversity("ctl_months_polysemy"),
    "halfA":    diversity("ctl_months_halfA"),
    "halfB":    diversity("ctl_months_halfB"),
    "partA":    diversity("ctl_months_partA"),
    "partB":    diversity("ctl_months_partB"),
}
assert round(DIV["primary"], 1) == 8.6, DIV
assert round(DIV["polysemy"], 1) == 7.1, DIV
assert round(DIV["partA"], 1) == 3.0 and round(DIV["partB"], 1) == 3.0, DIV
assert 4.7 <= DIV["halfA"] <= 5.1 and 4.7 <= DIV["halfB"] <= 5.1, DIV

# ---------- 2. overall Todd margins, from committed margin_split.json
MS = json.load(open("results/stage2/margin_split.json"))["cells"]
def overall(cell):
    c = MS[cell]
    n_in, n_out = c["in"]["n"], (c["out"]["n"] or 0)
    if not n_out:
        return c["in"]["margin"]
    return (c["in"]["margin"] * n_in + c["out"]["margin"] * n_out) / (n_in + n_out)

MARGIN = {
    "primary":  overall("primary (12)/todd"),
    "polysemy": overall("polysemy (9)/todd"),
    "halfA":    overall("halfA (Jan-Jun)/todd"),
    "halfB":    overall("halfB (Jul-Dec)/todd"),
    "partA":    overall("partA (Jan-Apr)/todd"),
    "partB":    overall("partB (Sep-Dec)/todd"),
}
EXPECT_M = {"primary": .3519, "polysemy": .3426, "halfA": -.3148,
            "halfB": -.4630, "partA": -.7037, "partB": -.9444}
for k, v in EXPECT_M.items():
    assert abs(MARGIN[k] - v) < 5e-4, (k, MARGIN[k])

# ---------- 3. gate weakest-shift accuracies, from committed reports / scores
def worst_from_report(run):
    t = json.load(open(f"results/pilot/{run}/report.json"))["table"]
    accs = ({int(k): (v["acc"] if isinstance(v, dict) else v) for k, v in t.items()}
            if isinstance(t, dict) else
            {r["k"]: r.get("acc", r.get("accuracy")) for r in t})
    return min(accs.values())

def worst_from_scores(run):
    tot, cor = {}, {}
    prompts = [json.loads(l) for l in open(f"results/pilot/{run}/prompts.jsonl")]
    scores  = [json.loads(l) for l in open(f"results/pilot/{run}/scores.jsonl")]
    for p, s in zip(prompts, scores):
        k, lp = p["k"], s["choice_logprobs"]
        pred = max(lp, key=lp.get) if isinstance(lp, dict) else \
               p["choices"][int(np.argmax(lp))]
        tot[k] = tot.get(k, 0) + 1
        cor[k] = cor.get(k, 0) + int(pred == p["target"])
    return min(cor[k] / tot[k] for k in tot)

GATE = {
    "primary":  worst_from_report("llama32_3b_mps_months_s16"),
    "halfA":    worst_from_report("gate_months_halves_A"),
    "halfB":    worst_from_report("gate_months_halves_B"),
    "partA":    worst_from_report("gate_months_partA4"),
    "partB":    worst_from_scores("gate_months_partB4"),
    "polysemy": worst_from_scores("gate_months_polysemy"),
}
EXPECT_G = {"primary": .38, "polysemy": .44, "halfA": .83,
            "halfB": .81, "partA": .61, "partB": .54}
for k, v in EXPECT_G.items():
    assert abs(GATE[k] - v) < 6e-3, (k, GATE[k])

# ---------- 4. draw
ORDER = ["partB", "partA", "halfB", "halfA", "polysemy", "primary"]
LABEL = {"partB": "4 mo\nSep–Dec", "partA": "4 mo\nJan–Apr",
         "halfB": "6 mo\nJul–Dec", "halfA": "6 mo\nJan–Jun",
         "polysemy": "9 mo", "primary": "12 mo"}
x  = [DIV[c] for c in ORDER]
ym = [MARGIN[c] for c in ORDER]
yg = [GATE[c] for c in ORDER]

fig, ax1 = plt.subplots(figsize=(7.2, 4.4), dpi=200)
ax1.axhline(0, color="0.75", lw=1, zorder=0)
l1, = ax1.plot(x, ym, "o-", color="#c0392b", lw=2, ms=7, zorder=3,
               label="task margin (what the vector does)")
ax1.set_xlabel("distinct months per prompt (mean)")
ax1.set_ylabel("task margin", color="#c0392b")
ax1.set_ylim(-1.05, 1.05)
ax1.tick_params(axis="y", labelcolor="#c0392b")

ax2 = ax1.twinx()
l2, = ax2.plot(x, yg, "s--", color="#2471a3", lw=2, ms=6, zorder=3,
               label="gate, weakest shift (what the check sees)")
ax2.set_ylabel("gate accuracy at weakest shift", color="#2471a3")
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis="y", labelcolor="#2471a3")

for xi, yi, c in zip(x, ym, ORDER):
    ax1.annotate(LABEL[c], (xi, yi), textcoords="offset points",
                 xytext=(0, -26 if MARGIN[c] > 0 else 12),
                 ha="center", fontsize=7.5, color="0.35")

ax1.set_title("Fewer distinct months: the check improves, the vector breaks")
ax1.legend(handles=[l1, l2], loc="lower right", fontsize=8, framealpha=0.9)
fig.tight_layout()

payload = json.dumps({"div": DIV, "margin": MARGIN, "gate": GATE},
                     sort_keys=True).encode()
h = hashlib.sha256(payload).hexdigest()[:12]
for ext in ("svg", "png"):
    fig.savefig(f"results/stage2/fig_diversity_{h}.{ext}")
print("wrote results/stage2/fig_diversity_%s.{svg,png}" % h)
print("all assertions passed:",
      {k: round(v, 2) for k, v in {**DIV}.items()})
