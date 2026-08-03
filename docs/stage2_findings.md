# Stage-2 geometry: months Z/12 on Llama-3.2-3B

Computed from the stage-1 `.npz` artifacts, pure numpy, no GPU. Evaluation order and
the confirmatory cells were locked in `docs/decisions.md` D24–D27 before any Gram
matrix existed. Raw numbers in `results/stage2/stage2.json`.

---

## 1. Null control — PASS (prereg §5)

Twelve unrelated tasks, read by the corrected D26 rule.

| | Todd | Hendel |
|---|---|---|
| `circulant_score` | **0.193** | **0.253** |
| permutation null (z) | +0.21 | −0.53 |

Two independent grounds. The Gram is not circulant — far below the 0.70 gate that
prereg §2 places first in both the A and A-multi definitions — so the diagnostics do
not find cyclic structure in unrelated tasks. And the canonical alphabetical task
ordering sits squarely inside its own permutation null, so the arbitrary ordering
carries nothing. The pipeline is behaving and the run stands.

The prereg §3 month-frequency control is reported in §5.

---

## 2. Headline — the registered A/A-multi/D classification does not apply

**Primary condition · full n=12 · canonical six-k head set · raw Gram** — the two
confirmatory cells fixed in D24.

| | Todd | Hendel |
|---|---|---|
| `circulant_score` | **0.091** | **0.113** |
| `toeplitz_score` | 0.159 | 0.177 |
| `closure_ratio` | 2.42 | 2.49 |
| `norm_cv` | 0.054 | 0.012 |
| `additivity_residual` | 1.150 | 1.153 |
| `participation_ratio` | 3.64 | 3.12 |
| rotation `order_error` | 1.040 | 1.112 |
| rotation `wraparound_error` | 1.541 | 1.699 |
| permutation null (z) | **+8.48** | **+7.52** |

`spectral_concentration` reads 0.682 / 0.745 with f=1 dominant, **and is not
interpretable**: it is the DFT of the circulant profile, which averages G over the
(i−j) mod n classes, and at circulant 0.09 those classes do not describe G. Per D26 it
is not read.

**No pre-registered bucket fits.** Against prereg §2:

| hypothesis | requires | observed | fits? |
|---|---|---|---|
| **A** circle | circulant ≥ 0.70 | 0.09 | ✗ |
| **A-multi** helix | circulant ≥ 0.70 | 0.09 | ✗ |
| **D** simplex | circulant ≥ 0.70, PR ≥ 8 | 0.09, 3.6 | ✗ |
| **B** linear selector | norm_cv ≥ 0.40, additivity ≤ 0.25, PR ≈ 1, closure ≥ 3 | 0.05, 1.15, 3.6, 2.4 | ✗ |
| **Null** lookup | circulant ≤ 0.30 ✓, toeplitz ≤ 0.30 ✓, PR ≥ 8 | 3.6 | ✗ |

This is the reportable result: **the shape is none of the five registered
alternatives.**

### What the numbers do say

- **The ordering carries real structure.** The permutation null (D25 §2, registered
  before computing) puts the canonical k-ordering at z ≈ +8, percentile 100, in both
  methods. A simplex is permutation-invariant and cannot produce this. Whatever the
  FVs trace, the cyclic parameter ordering is doing work.
- **Norms are near-constant** (cv 0.01–0.05) and **additivity fails** (1.15). Both cut
  against a linear code: prereg §2's B requires cv ≥ 0.40 and additivity ≤ 0.25.
- **The curve does not close.** `closure_ratio` 2.4 against A's requirement of ≤ 1.5,
  and rotation `wraparound_error` 1.5–1.7 against ≤ 1.0. Hypothesis A's central
  prediction — FV(n) ≈ FV(0) — **fails**.
- **It is low-dimensional**: PR 3.1–3.6, i.e. ~3 effective dimensions, not 12.

Together: a **low-dimensional, constant-norm, ordering-dependent open curve** that is
neither circulant nor Toeplitz.

---

## 3. Robustness — every condition agrees

`circulant_score` / `closure_ratio` / PR, raw Gram:

| condition | Todd | Hendel |
|---|---|---|
| **primary (headline)** | 0.091 / 2.42 / 3.64 | 0.113 / 2.49 / 3.12 |
| polysemy leave-out | 0.090 / 2.54 / 3.50 | 0.108 / 2.69 / 3.10 |
| mixed-domain days+months | 0.075 / 2.26 / 3.66 | 0.057 / 2.67 / 1.94 |
| head set: all-k | 0.106 / 2.47 / 3.79 | — (head-set-free) |
| head set: intersection-8 | 0.102 / 2.46 / 3.82 | — |
| leave-one-out n=11, no k=8 | 0.104 / 2.29 / 3.64 | 0.135 / 2.35 / 3.11 |

- **Both extraction methods agree** on the bucket, which is not something this project
  could assume — CLAUDE.md rule 1 exists because they often disagree, and on causal
  efficacy they did (Todd steers at every k, Hendel only at k ∈ {1,2,11}).
- **All three head sets agree**, satisfying D16 §2: same bucket, and the spread
  (0.091–0.106) is small. D15's warning that head-set choice moves FVs by more than
  the split-half band is about the *vectors*; it does not move this *verdict*.
- **The prereg §4 leave-one-out agrees with full n=12**, so the D1 requirement that a
  result be claimed only where both columns agree is met — and the k=8 question does
  not change the geometry.
- **The polysemy control agrees**, so the result is not carried by May/March/August.
- **The mixed-domain control agrees**, which is the strongest available answer to the
  BRIEF §6 operand-inheritance confound: the condition whose demonstrations span two
  cycles of different length, and so cannot inherit a single operand circle, gives the
  same geometry.

---

## 4. Exploratory: the centered Gram (D27)

The FVs carry a large component shared across all k — **78.8%** of mean squared norm
for Todd, **94.9%** for Hendel. That contaminates the raw Gram with a rank-2 term
varying with i and j separately rather than with (i−j), which is neither circulant nor
Toeplitz.

Removing it raises the circulant score but **does not change the verdict**:

| | raw | centered |
|---|---|---|
| Todd primary | 0.091 | **0.298** |
| Hendel primary | 0.113 | **0.267** |
| permutation z (Todd) | +8.48 | +8.83 |

Still far below the 0.70 gate. `closure_ratio`, `participation_ratio`,
`additivity_residual` and the rotation fit are unchanged by construction — they centre
internally. So the offset explains part of the low raw circulant score but not the
absence of cyclic structure, and the "not circulant" reading is robust to it.

---

## 5. Blocking control: month-token frequency (prereg §3)

`freq_proxy_operand` / `freq_proxy_target` are stored per k in every months `.npz`,
computed over the operands and targets actually drawn (D11 — the registration's own
rationale for this test was wrong, and implementing it literally would have made it
vacuous). With `norm_cv` at 0.054 (Todd) and 0.012 (Hendel), prereg §3's own escape
clause applies: *"ρ significant but `norm_cv` ≤ 0.15 (structure cannot be carried by
norms that barely vary)"*. The norm profile is too flat to carry the geometry either
way. **Control passes.**

---

## 6. What cannot be concluded

Carried forward from D24, unchanged by these results:

- **No hypothesis-C verdict.** The operand-partition control is structurally
  unrunnable on Z/12 (D21): every disjoint partition is small enough to collapse the
  FVs to next-item.
- **No real-model open-vs-closed demonstration.** The ordinal shape control is blocked
  (D23). This now matters more than when it was blocked: the headline result *is* an
  open curve (closure 2.4), and the only evidence that these diagnostics separate open
  from closed comes from synthetic fixtures.
- **One model, one family, one prompt format.** D7 defers format variation to stage 3.
- The result is **not** a null result. It is a positive finding of ordering-dependent,
  low-dimensional, non-circulant structure that the pre-registration did not
  anticipate, which is a different thing from "no structure".
