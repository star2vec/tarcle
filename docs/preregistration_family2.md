# Pre-registration: second family (hours Z/24 or ROT-k Z/26)

**Status: written before any function vector of either candidate family has been
extracted.** The only empirical inputs are the behavioural gates now running, the
months results in `docs/stage2_findings.md`, and the power analysis in §1, which is
computed on the *existing* months/add-k/null artifacts.

Family selection is by gate, not by preference: both candidates are piloted and §6
records the tie-break if both pass.

---

## 0. Why there is a second family

`docs/stage2_findings.md` reached three things on months Z/12 and stalled on a fourth:

- **Settled.** The FVs are not circular in the registered sense, not a simplex, not
  linear. They are ordered by raw forward k, not by signed shift (§8, D34).
- **Settled negatively.** Permutation z and `circulant_score` cannot separate cyclic
  from merely ordered — add-k, definitionally non-cyclic, scores as high as months
  (D30).
- **Settled provisionally.** A seam is present; its location is not resolved (D32).
- **Stalled.** Whether a low-amplitude circular component rides the dominant axis. The
  check was **underpowered** (D37) and the "not circular" verdict retains its
  "in full-vector distances" scope.

The second family exists to resolve the last two with tests registered in advance and
powered on purpose.

---

## 1. Power analysis — BLOCKING, run before any verdict is read

The D37 failure must not recur as a surprise, so the detector's sensitivity is
established before its output is interpreted.

### Method

The false-positive floor is estimated by **residual permutation on the real extracted
vectors**, not from synthetic surrogates:

1. Fit the constant-plus-linear-in-k component, `X ≈ M₀B₀` with `M₀ = [1, k]`.
2. Surrogate: keep `M₀B₀`, permute the residual rows across k.
3. Floor = 95th percentile of the harmonic partial R² over 300 such draws.
4. Minimum detectable amplitude (MDA) = the smallest planted circle radius, as a
   fraction of the family's RMS centered norm, whose partial R² clears that floor in
   ≥80% of draws.

**Synthetic surrogates are explicitly rejected for this purpose**, on measured grounds:
a line-plus-isotropic-noise null puts the floor at **0.0026–0.0061**, while residual
permutation on real vectors puts it at **0.2237–0.3003** — a hundredfold difference.
The synthetic version would have promised power the estimator does not have, which is
exactly how D37 went wrong.

### Measured at n=11, for reference

| | floor (95th pct) | MDA at 80% power | observed |
|---|---|---|---|
| months, Todd | 0.2581 | **not reached at 0.5 × RMS** | 0.3626 |
| months, Hendel | 0.3003 | **not reached at 0.5 × RMS** | 0.4233 |
| add-k, Todd (no circle) | 0.2237 | not reached | — |

A circle carrying **half the family's RMS norm** is detected 23% / 9% of the time. That
is the quantitative statement of D37's "inconclusive".

### Pre-committed decision rule

Run the identical analysis on the new family's extracted vectors **before reading any
circle verdict**. Then:

- **MDA ≤ 0.25 × RMS** → the geometric test is adequately powered. Report the circle
  verdict against the floor.
- **MDA > 0.25 × RMS**, or not reached in range → **the geometric test is declared
  underpowered in advance**, no circle verdict is read from it in either direction,
  and the causal tie-breaker of §2 becomes the operative test.

Given the n=11 result, the second branch is the more likely one, and §2 is registered
accordingly rather than as a contingency.

---

## 2. Causal tie-breaker — phase-rotated on-cylinder injection

Registered as a primary test, not a fallback. It can detect a circular component below
geometric resolution because it asks the model, not the metric.

**Construction.** Fit the axis-plus-harmonic model of §1. For a target parameter value
`k*`, build two injection vectors of **matched norm**:

- **on-cylinder**: axial coordinate at `k*` **plus** the harmonic component evaluated at
  phase `2πk*/n` — the model's own fitted circle, at the right phase.
- **axis-only**: the axial coordinate at `k*` with the harmonic component removed and
  the norm restored by scaling the axial term.

**Prediction.** If a functional circular component exists, on-cylinder injection
produces shift-by-`k*` behaviour more reliably than axis-only at equal norm. If the
harmonic component is fitting noise, the two are indistinguishable.

**Phase rotation is the control that makes this a test rather than a norm comparison.**
A third arm injects the harmonic component at a **deliberately wrong phase**
(`2π(k*+n/2)/n`, the antipode) with the axial coordinate still at `k*`. If the circle is
functional, wrong-phase injection should be *worse* than axis-only, not merely
different — the component actively points elsewhere. If all three arms behave alike, the
harmonic component carries no function regardless of what any geometric fit says.

**Metric and criterion.** `logp_lift` on the complete zero-shot query cycle (D17), with
per-query SEs. The noise band is the split-half band of D16: extract from two disjoint
halves of the prompts and compute the same three-arm contrast between them.

- **Circle is functional:** on-cylinder > axis-only > wrong-phase, with the
  on-cylinder-minus-axis-only gap exceeding the split-half band, at a majority of `k*`.
- **Circle is not functional:** the three arms lie within the band of each other.
- **Ambiguous:** ordering holds but the gap is within the band — reported as such.

Run at every `k*` that clears the D20 task-encoding gate, both extraction methods.

---

## 3. Seam contest, powered for location

D31/D32's contest resolved seam *presence* and not *location*: at n=11 the
largest-separation bin held 2 pairs of 110. At n=24 the thinnest bin holds 12.

**Registered in advance** for this family, with two additions over D31:

- **The localizer's bias is measured, not assumed.** The matched-n add-k reference has a
  seam at a known position, so the cut the contest returns there gives the bias
  directly. D35 recorded a +2 displacement at n=11 from a single calibration point;
  here it is subtracted rather than noted. The months cut is reported both raw and
  bias-corrected.
- **Location is reported with an interval**: the set of cuts whose cross-validated R²
  lies within one split-half band of the best, rather than the argmax alone.

Pre-labelled outcomes are those of D31 §5, unchanged, plus: **if the bias-corrected
interval contains no single cut, location is again reported unresolved** — a wider
family does not license a sharper claim than the data supports.

---

## 4. Hypothesis C — partition transfer, riding the same extraction

D21 declared this structurally unrunnable on Z/12: the control needs two *disjoint*
operand partitions, the largest disjoint pair on a 12-element cycle is 6+6, and 6
operands collapse the FVs to next-item (`docs/note_operand_diversity.md`).

**At n=24 the largest disjoint pair is 12+12**, and 12 operands is exactly the
full-pool months condition, which cleared the task-encoding gate at +0.352. The
transfer test therefore rides the same extraction run at no extra pilot cost.

Criterion is D18 unchanged — Δ `logp_lift` between matched and transferred operands,
averaged over both directions, against a split-half band — with the D20 §2 requirement
that the **matched-condition baseline is named here, before running**: the matched arm
is each partition's FV applied to its own operands, and the D18 wrong-region signature
is read only if the matched arm itself clears the D20 §1 task-encoding gate.

For ROT-k on Z/26 the equivalent pair is 13+13, which also clears.

---

## 5. Divisor test, and why it favours Z/24

Prereg §1 (months) predicts spectral power at frequencies with `gcd(f, n) > 1` if the
model represents the factorisation.

- **Z/24 = 2³·3** has divisors 2, 3, 4, 6, 8, 12 — six non-trivial frequency classes,
  against Z/12's four. The test has more content than it did on months.
- **Z/26 = 2·13** has divisors 2 and 13 only. The order-2 character and one order-13
  class. **The divisor test is nearly vacuous on Z/26.**

Recorded as a substantive reason to prefer Z/24 if both families clear their gates
(§6). It is not a reason to override a gate failure.

The test is read off the full-n column and is subject to the same circulant gate that
D26 imposed: a spectrum is not interpreted when `circulant_score` < 0.70.

---

## 6. Family selection

Both candidates are piloted with the standard behavioural gate (16 shots, held-out,
n=100 per k, GO ≥ 0.50 at every k). A matched-n add-k reference is piloted alongside
each.

- **Exactly one passes** → it is the family; the other is recorded as blocked with its
  numbers.
- **Both pass** → **Z/24**, on the §5 divisor grounds, plus: hours have no
  corpus-privileged shift analogous to ROT-13 (`lit_sweep` warning, Prabhakar et al.),
  and digits are a stronger operand domain than letters on this model (0.43 vs 0.35
  aggregate, `pilot_findings.md` §4). ROT-k is retained as the replication family.
- **Neither passes** → no second family on this model; the months results stand with
  their stated limits and the project's next move is a different model, not a different
  family.

Rule 5 applies per condition (D20 §3): the add-k reference and every operand partition
get their own behavioural gate, and every extracted condition gets the D20 §1
task-encoding gate before any verdict is read from it.

---

## 7. Controls carried over unchanged

- **Frequency proxy** (months prereg §3, as corrected by D11): computed over the
  operands and targets actually drawn, stored per k, and reported alongside
  `norm_profile`. Blocking for the geometry verdicts.
- **Unrelated-tasks null control**, read by the D26 rule — circulant gate first, and a
  high circulant score alone does not void the run.
- **Both extraction methods**, always (CLAUDE.md rule 1), with the D16 §3 provision that
  Hendel arbitrates a Todd head-set split and its own limitation is reported alongside.
- **Head-set variation** as a stage-2 axis: `head_contrib_union` persisted over the
  union of shortlists, so alternative head sets need no re-extraction (D15).
- **k = 0 excluded** from head identification (D3) and from the geometric comparisons
  (D28), on the measured grounds that the identity task is solvable by the copy prior
  and that FV(0) sits ~2× outside the family's internal distance scale.

---

## 8. What this pre-registration does not promise

The months run ended with a headline that no pre-registered bucket fitted. That may
happen again, and this document does not pre-commit to a bucket being found. What it
pre-commits to is that **the power of each test is established before its output is
interpreted**, which is the specific failure mode of D37 and the reason this file
exists.
