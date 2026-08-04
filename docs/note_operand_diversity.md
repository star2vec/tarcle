# Task accuracy does not gate function-vector quality

*Standalone note. Independent of everything this project set out to measure — it makes
no claim about cyclic structure, and holds whatever the geometry turns out to be.*

## Summary

Function vectors are conventionally extracted from tasks the model demonstrably
performs, gated on in-context accuracy. We find that gate can pass while the extracted
vector encodes a **different function** from the one it was extracted for, and that the
discrepancy is driven by a property of the prompt distribution — how many distinct
operands the demonstrations draw from — that accuracy does not see.

Across six operand-pool sizes with everything else held fixed, in-context accuracy and
function-vector quality move in **opposite directions**.

## Setup

Llama-3.2-3B, shift-by-k on months (Z/12), 16 shots, k = 0..11, 100 prompts per k.
Todd-style causal-head-averaged function vectors, one head set (identified once, reused
throughout), one frozen injection protocol (layer 8, scale ×3.0, additive). The only
variable is the pool of operands the demonstrations and query are drawn from.

Two independent measurements per condition:

- **Behavioural** — held-out forced-choice ICL accuracy, the standard extraction gate.
- **FV task-encoding** — inject the extracted vector into a zero-shot prompt and measure
  P(correct shift) − P(shift ±1), averaged over k ∉ {0, 1, 11} where ±1 *is* the correct
  answer and collapse cannot be told from success.

## Result

| operand pool | distinct demo operands / 16-shot prompt | behavioural gate | FV encoding margin |
|---|---|---|---|
| 12 (full cycle) | 8.6 | GO | **+0.352** |
| 9 (three months removed) | 7.1 | GO | **+0.343** |
| 6 (Jan–Jun) | 5 | GO — every k ≥ 0.83 | **−0.315** |
| 6 (Jul–Dec) | 5 | GO | **−0.463** |
| 4 (Jan–Apr) | 3.0 | GO — every k ≥ 0.50 | **−0.704** |
| 4 (Sep–Dec) | 3.0 | GO | **−0.944** |

Three observations.

**1. The behavioural gate passes everywhere and carries no information about FV
quality.** All six pools return GO. The columns do not merely differ in sensitivity —
they move in opposite directions. Shrinking the operand pool makes the in-context task
*easier* (k=8 rises from 0.38 in the full pool to 0.87 at six operands: fewer candidates
to confuse, and the demonstrations cover more of the operand space) while making the
extracted vector *worse*.

**2. There is a threshold, and it is sharp.** Twelve and nine operands pass
comfortably; six and four fail. The transition sits between 6 and 9 distinct
demonstration operands. Below it the vectors steer toward the adjacent item regardless
of k — P(±1) up to 0.95.

**3. The failure is silent.** A collapsed function vector is not a degenerate object.
It has split-half reliability ≥ 0.99, a large causal effect on the model's output, and
a norm indistinguishable from a good one. It simply implements a different function
from the one it was extracted for. Nothing short of measuring *which* function it
implements detects this.

## Why it matters beyond this study

Standard FV practice gates extraction on task accuracy, following the protocol of Todd
et al. This shows that gate can be satisfied by prompts whose extracted vector encodes
a degenerate default. The risk is concentrated in task families with few distinct
operands, which describes a substantial fraction of standard FV benchmark tasks
(antonym pairs, country–capital, and similar closed sets), and in any study that
restricts the operand pool as an experimental control — which is how we encountered it.

The confound is invisible to accuracy because the two respond to operand diversity in
opposite directions: fewer operands means an easier in-context problem and a less
identifiable function.

## The check, which is nearly free

Record the argmax of each injected zero-shot prediction and compare P(correct) against
the family's degenerate default — for a shift family, the adjacent item; in general,
whatever the model would do without a task signal. No extra forward passes beyond the
efficacy scoring that a causal validation already performs, and it runs offline from
saved activations.

Implementation: `tarcle/nextitem.py`. Registered as a blocking gate in
`docs/decisions.md` D20.

## Scope and limits

One model (Llama-3.2-3B), one task family (months Z/12), one prompt format, and one
extraction method for the FV column — Todd-style causal-head averaging. The Hendel-style
dummy-query extraction steers at only k ∈ {1, 2, 11} even in the full pool and cannot
resolve this curve.

The ±1 attractor is specific to families with a strong adjacency prior. The general
claim is that accuracy-gating misses *some* degenerate attractor, not that it is always
next-item.

Six pool sizes on one cycle locate the threshold only to "between 6 and 9 distinct
operands". Whether the relevant quantity is the count, the entropy of the operand
distribution, or the ratio of operands to shots is not determined here.

## Artifacts

`results/fv/ctl_months_*`, `results/pilot/gate_months_*`, `results/fv/nextitem_todd.json`.
Every run carries its prompt SHA-256, config hash and git commit; the analysis is pure
numpy from saved `.npz` and needs no GPU.
