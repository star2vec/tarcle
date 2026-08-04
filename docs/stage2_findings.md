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

**The raw circulant score is not the evidence.** The null control scores **0.193** and
the primary **0.091** — the primary is *lower* than the negative control. Read alone,
`circulant_score` would rank twelve unrelated tasks as more cyclically structured than
shift-by-k. Every structural claim below rests on the permutation null, not on this
number.

- **The ordering carries real structure — but only *ordering*, not cyclicity.** The
  permutation null (D25 §2, registered before computing) puts the canonical k-ordering
  at z = +8.48 / +7.52, percentile 100, against a null control at z = +0.21 / −0.53.
  A permutation-invariant simplex cannot produce this, so the simplex/lookup reading is
  ruled out. **It does not show the structure is cyclic**: the add-k calibration family
  (§7), which is definitionally non-cyclic, scores z = +7.44 on the same statistic. See
  D30 — every claim here is about an *ordered* family, not a cyclic one.
- **FV(0) is an outlier, and the other eleven form a tight cluster.** Distances from
  FV(0) to each FV(k) run **7.61–8.62** (mean 8.03), while the mean pairwise distance
  *among the other eleven* is **4.12**, maximum 7.28. FV(0) sits roughly twice as far
  from the family as its members sit from each other. This is the shape: not a curve
  with two ends, but one displaced point plus a cluster.
- **Within the eleven, distance grows monotonically with cyclic separation**
  |m| = min(m, 12−m):

  | \|m\| | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Todd | 2.75 | 3.26 | 4.49 | 4.61 | 5.11 | 4.89 |
  | Hendel | 2.26 | 2.75 | 3.78 | 3.98 | 4.39 | 4.25 |
  | null control | 6.62 | 6.64 | 6.06 | 6.49 | 6.65 | 6.09 |

  Monotone through |m|=5 with a dip at the antipode, against a **flat** null-control
  profile. Nearby shifts have more similar function vectors; distant ones less so.

  *Not* a finding: that this profile is symmetric in ±m. A Euclidean distance matrix
  is symmetric, so pairs at separation +m are the transpose of those at −m and their
  means are equal **by construction** (measured: |difference| = 0.000 at every m). The
  statistic is therefore blind to the forward/backward asymmetry the pilot found in
  accuracy (§2 of `pilot_findings.md`), and can neither corroborate nor contradict it.

- **Norms are near-constant** (cv 0.01–0.05) and **additivity fails** (1.15). Both cut
  against a linear code: prereg §2's B requires cv ≥ 0.40 and additivity ≤ 0.25.
- **It is low-dimensional**: PR 3.1–3.6, i.e. ~3 effective dimensions, not 12.

Together: **one displaced identity vector plus a low-dimensional, constant-norm cluster
whose internal distances grow with parameter separation**, and which is neither
circulant nor Toeplitz. Whether that separation is best described cyclically is **not
established** — see §7.

### Wraparound: quarantined, not concluded

`closure_ratio` is ‖X[n−1] − X[0]‖ over the mean step, and
`rotation_wraparound_error` is the error of the fitted rotation on the pair
X[n−1] → X[0]. **Both are computed from pairs containing FV(0)** — the vector D2
declared uninterpretable, because the zero-shot baseline is at ceiling for the identity
task, and which the distances above show to be an outlier at ~2× the family's internal
scale. A statement that "the loop does not close" resting on these numbers would be
resting on the one point the run already agreed not to interpret.

Exploratory, alongside the registered k=8 leave-one-out:

| variant | circulant (raw / centered) | closure | wrap_err | PR |
|---|---|---|---|---|
| full n=12 | 0.091 / 0.298 | 2.42 | 1.54 | 3.64 |
| n=11, drop **k=8** (registered, prereg §4) | 0.104 / 0.320 | 2.29 | 1.40 | 3.64 |
| n=11, drop **k=0** (exploratory) | 0.137 / 0.312 | 2.02 | 1.28 | 3.21 |
| n=10, drop k=0 and k=8 (exploratory) | 0.158 / 0.345 | 1.91 | 1.20 | 3.20 |

Removing FV(0) moves closure 2.42 → 2.02 and wraparound error 1.54 → 1.28, both toward
but not past A's thresholds (≤1.5 and ≤1.0). **No claim about FV(n) ≈ FV(0) is made
from these numbers in either direction.** The "not circulant" verdict is unaffected —
it survives every exclusion, raw and centered.

---

## 3. Robustness — eleven of twelve cells agree, and the exception matters

`circulant_score` / `closure_ratio` / PR, raw Gram:

| condition | Todd | Hendel |
|---|---|---|
| **primary (headline)** | 0.091 / 2.42 / 3.64 | 0.113 / 2.49 / 3.12 |
| polysemy leave-out | 0.090 / 2.54 / 3.50 | 0.108 / 2.69 / 3.10 |
| mixed-domain days+months | 0.075 / 2.26 / 3.66 | 0.057 / 2.67 / 1.94 |
| head set: all-k | 0.106 / 2.47 / 3.79 | — (head-set-free) |
| head set: intersection-8 | 0.102 / 2.46 / 3.82 | — |
| leave-one-out n=11, no k=8 | 0.104 / 2.29 / 3.64 | 0.135 / 2.35 / 3.11 |

### The exception: mixed-domain under Hendel

`mixed_daysmonths / hendel` is the **only cell in the matrix without significant order
structure**: permutation z = **+1.32** (against +7.4 to +9.7 everywhere else) and
PR = **1.94** (against 3.1–3.8). Its dominant frequency is f=2 rather than f=1, unlike
every other cell.

This is not a footnote, because of *where* it sits. The mixed-domain condition is the
one that carries the BRIEF §6 operand-inheritance control — the demonstrations span two
cycles of different length and so cannot inherit a single operand circle. Under Todd
that condition retains order structure (z = +7.36) and the inheritance argument goes
through; under Hendel it does not. **The one cell that fails to show ordering structure
is the one whose job is to show that the ordering structure is not inherited from
operand geometry.**

Two readings, neither adopted: Hendel's dummy-query state may be the weaker extraction
here — consistent with D12, where its FVs steer at only k ∈ {1,2,11} even at their own
optimal layer and scale — or the order structure in the other conditions may be partly
operand-inherited in a way that only the mixed condition exposes and only the more
fragile extraction is sensitive to. Distinguishing them needs the mixed-domain
condition re-run under a stronger extraction, which is not available.

### The rest

- **Both extraction methods agree** on the bucket in every condition except the one
  above, which is not something this project could assume — CLAUDE.md rule 1 exists
  because they often disagree, and on causal efficacy they did (Todd steers at every k,
  Hendel only at k ∈ {1,2,11}).
- **All three head sets agree**, satisfying D16 §2: same bucket, and the spread
  (0.091–0.106) is small. D15's warning that head-set choice moves FVs by more than
  the split-half band is about the *vectors*; it does not move this *verdict*.
- **The prereg §4 leave-one-out agrees with full n=12**, so the D1 requirement that a
  result be claimed only where both columns agree is met — and the k=8 question does
  not change the geometry.
- **The polysemy control agrees**, so the result is not carried by May/March/August.
- **The mixed-domain control agrees under Todd** (z = +7.36), which is the strongest
  available answer to the BRIEF §6 operand-inheritance confound — but see the exception
  above: it does **not** agree under Hendel, and that is the cell where it would matter
  most.

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

## 7. Calibration: the diagnostics do not separate cyclic from ordered (D28/D30)

add-k on small integers — ordered, definitionally **not** cyclic, both gates passed
(behavioural GO at 1.000 every k; task-encoding lift +0.826) — supplies the reference
the run was missing.

| family | ordering | `circulant` | permutation z |
|---|---|---|---|
| unrelated tasks | none | 0.190 / 0.253 | **+0.17 / −0.54** |
| **add-k** | **ordered, NOT cyclic** | **0.285 / 0.144** | **+7.44 / +4.48** |
| months n=11 | ordered *and* cyclic | 0.137 / 0.109 | **+7.89 / +8.93** |

**Permutation z cannot tell cyclic from merely ordered**: add-k reaches +7.44 against
months' +7.89. It does cleanly separate both from the unordered null at z ≈ 0, which is
what licenses rejecting the simplex/lookup reading — and nothing more.

**`circulant_score` does not rank these families by cyclicity either.** add-k, with no
cycle at all, scores 0.285 against months' 0.137; the null control scores 0.190, also
above months. The ordering of the three by circulant score is unrelated to whether they
have a cycle.

The D28 discriminator — an antipodal dip in months absent from add-k — did not work.
Months does dip (5.11 → 4.89), but add-k turns down at its largest separation too, on a
bin holding **2 pairs of 110**, and the flat null control turns down as well. Per D28
the non-discriminating outcome applies and the language throughout this document is
*ordered*, not *cyclic*.

---

## 8. The seam contest: cyclicity positively disconfirmed (exploratory, D31/D32)

A cyclic family has no seam; a wrapped linear one has exactly one. Unrolling the
parameter at every cut point and asking which unrolling best explains the pairwise
distances tests that directly. Cross-validated isotonic R², because the cut models
carry eleven separation levels against the cyclic model's six and would win on degrees
of freedom alone.

**Validation gate, read before months and passed:** add-k, whose seam is known to
exist, is detected (best cut 0.347 / 0.197 over cyclic 0.226 / 0.012); the unordered
null fits nothing (every R² negative).

**Months:**

| | cyclic | best cut | margin |
|---|---|---|---|
| Todd | **0.031** | 0.237 | **−0.206** |
| Hendel | **0.049** | 0.221 | **−0.171** |

Identifying separations mod 12 explains ≈ 0% of the variance in pairwise distance; a
linear unrolling explains ~22%.

**The ten fold-back pairs — where the whole wraparound claim lives — are all above the
mean distance**, in both extractions, at 1.08×–1.77×. The decisive one is **k=1 versus
k=11**: shift-by-+1 and shift-by-−1, which any cyclic code must place adjacent, sit at
**1.35× the mean** against a separation-1 bin mean of 0.67×.

**The seam's location is not resolved.** cut@2 wins for months, but cut@2 also wins for
add-k where the true seam is at the endpoint — so the contest answers *whether*
reliably and *where* unreliably. The D31 "seam at k=0" model (cut@0 ≡ cut@1) places
second at 0.171 / 0.182.

One location *is* excluded, and it carries a result (D34). The signed-magnitude model —
distance a function of |m| = min(k, 12−k) with direction discarded — predicts k=6 and
k=7 far apart across the seam; they are measured at 0.39× / 0.35× the mean distance,
second and third closest of the ten separation-1 pairs, with (5,7) at 0.26× / 0.25× and
(6,8) at 0.41× / 0.39×. It also predicts (1,11) close, and that pair is 1.35× / 1.29×.
Both predictions fail, so: **the function-vector family is ordered by raw forward k, not
by signed shift — the signed representation the model uses behaviourally
(`pilot_findings.md` §2, where accuracy depends on ±m and errors are sign flips) is not
the representation carried in the vector.**

This moves the months family from "cyclicity not established" (§7) to **cyclicity
positively disconfirmed in full-vector distances**: not merely that the diagnostics
cannot see a cycle, but that the pairs a cycle requires to be close are measurably far
*in the full-dimensional distance between function vectors*.

The scope qualifier is not hedging. Every statistic here — pairwise distance, isotonic
fit, the fold-back table — is computed on whole vectors, and a circular component
carrying a small share of the variance while riding a dominant non-circular axis would
leave all of them looking exactly as they do. That geometry (an open helix, absent from
the prereg §0 calibration battery, which contains only *closed* helices and a plain
line) is the subject of a registered check, D33. Until it runs, "disconfirmed" means
*as a description of the whole vector*.

Exploratory — D31 is post-hoc relative to §7's binned profiles — and the confirmatory
version belongs to the next family.

Centering is a no-op here: pairwise distances are translation-invariant, so the shared
offset behind §4 cannot affect this test.

---

## 9. Cylinder check: inconclusive, and the scope qualifier stands (D33/D37)

An open helix — a monotone axis with a circle wrapped around it — reproduces every §8
observation while containing a real circle, and no such fixture was in the prereg §0
calibration battery. The registered check for it **failed its own validation gate**: a
planted circle recovers to residual circulant 0.350, *below* the 0.410 floor produced by
families containing no circle. The cause is structural — over one period
corr(k, sin) = −0.872, so removing a monotone axis removes most of the circle with it,
and this does not improve with n.

A joint-fit replacement (regress on [1, k, cos, sin], measure the harmonics' unique
variance) validates cleanly on synthetics — planted circle amplitude 1.599 invariant to
axis strength against 0.27–0.32 for a line, partial R² 0.15–0.60 against 0.002. But the
floor on *real* no-circle families is 0.126–0.315, because four parameters on eleven
points leaves seven residual degrees of freedom. Months returns 0.363 / 0.423: above the
maximum floor, by margins (+0.048 / +0.109) two to four times **smaller than the floor's
own spread** (0.189).

**Inconclusive.** Not absence, not presence. The check is underpowered at n=11.

Two things do follow. The **scope qualifier in §8 stands** — "disconfirmed in
full-vector distances" cannot be strengthened. And the **axis is k, not token
frequency**: corr(axis, k) = +0.978 / +0.955 against +0.451 / +0.371 with the operand
frequency proxy, so the frequency-drift reading of the ordered structure is ruled out.

*A sentence connecting the model's behavioural wraparound competence
(`pilot_findings.md` §3) to the geometric picture was to be written here once this check
resolved. It did not resolve, so the sentence is not written.*

---

## 10. What cannot be concluded

Carried forward from D24, unchanged by these results:

- **No hypothesis-C verdict.** The operand-partition control is structurally
  unrunnable on Z/12 (D21): every disjoint partition is small enough to collapse the
  FVs to next-item.
- **No real-model open-vs-closed demonstration.** The ordinal shape control is blocked
  (D23). This now matters more than when it was blocked: the headline result *is* an
  open curve (closure 2.4), and the only evidence that these diagnostics separate open
  from closed comes from synthetic fixtures.
- **One model, one family, one prompt format.** D7 defers format variation to stage 3.
- **No confirmatory cyclic claim, in either direction.** §7 shows the *registered*
  ordering evidence cannot distinguish cyclic from merely ordered. §8 goes further and
  disconfirms cyclicity — but exploratorily, so the BRIEF's central question is
  answered only provisionally: on this family, in this model, the task representations
  are ordered and **do not** wrap. Confirming that needs the seam contest registered in
  advance on a fresh family.
- The result is **not** a null result. It is a positive finding of order-dependent,
  low-dimensional, non-circulant structure that the pre-registration did not
  anticipate, which is a different thing from "no structure".
