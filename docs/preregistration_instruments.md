# Pre-registration: measurement-validity instruments (T1, T3, T5, T6, T10)

**Status: written before any new run and before any new stage-2 statistic has been
computed.** The geometry programme is closed (D38: both second-family candidates failed
their gates; the months results stand with their stated limits). The months geometry is
now a case study. This document registers the instruments for the project's new
contribution — a measurement-validity study — before any of their outputs exist.

What has and has not been seen, stated per task in its own section, because several
tasks assemble cells that earlier entries already made public. The registered content
of each task is the part not yet computed, and the pre-committed branches below bind
those parts only.

Session constraints, which scope everything here: one ~6-hour session on a 16 GB M1
Mac. No CUDA. T1 is behavioural-only, which CLAUDE.md rule 4 and
`docs/pilot_findings.md` (header) sanction at fp16 on MPS. T3, T5, T6, T10 are pure
numpy off saved `.npz` and need no model.

**Explicitly out of scope tonight, not started even if time remains:** the injection
layer × scale sweep on MPS (saved FVs are CUDA bf16; injecting them into an MPS fp16
model is incomparable to the frozen L8 × 3.0 baseline it would be testing), the pool
grid (geometry-bearing extraction → Ada box at bf16 per rule 4), the standard-FV
benchmark task (needs new head identification), and everything else not named in this
document. If the tasks below finish early, the remaining time goes to writing.

Decision-log numbering continues at D39 (append-only, as always): D39 records the
pivot and tonight's scope, D40 the T1 branches, D41 the rule-5 exception in §6.

---

## 0. The thesis and the two claims

**Thesis: interpretability validation practice is systematically permissive.** Two
measurable instances, both on artifacts this repo already has:

- **Claim A.** The accuracy gate that standard FV practice uses passes while the
  extracted vector encodes a different function
  (`docs/note_operand_diversity.md`, D20: six operand pools, gate GO everywhere,
  FV task-encoding margin from +0.352 down to −0.944).
- **Claim B.** Estimators validated on synthetic fixtures have false-positive floors
  roughly two orders of magnitude higher on real activations (D37 /
  `docs/preregistration_family2.md` §1: harmonic partial R² floor 0.0026–0.0061 on
  synthetic line-plus-noise surrogates vs 0.2237–0.3003 under residual permutation on
  real vectors).

Confirmatory for the new paper, and nothing else is:

> **T1's branch decision (§1)** — whether claim A survives its own support-mismatch
> audit — and **T6's floor table (§5)** — whether claim B generalises beyond the one
> statistic D37 measured. T3, T5 and T10 are supporting instruments: reported with
> numbers and their registered readings, never used to upgrade the two headline claims.

The D37 lesson binds every section: **each instrument's power is established before
its output is interpreted.** Where a section cannot establish power, its outputs are
registered as descriptive and carry no verdict language.

---

## 1. T1 — the 2×2 support matrix (behavioural, MPS fp16)

### The hole in claim A

Every existing restricted-pool behavioural gate set `query_pool` equal to
`operand_pool`, so the gate was scored **in-distribution** while `efficacy()` scored
the FV over the **complete 12-month query cycle** (`tarcle/causal.py::zero_shot_spec`
with `queries=None`; verified in `tarcle/extract.py` — `causal.efficacy` is called
without a query restriction). The 2×2 matrix of (demonstration support × query
support) therefore has one unmeasured cell: **restricted demonstrations, full-cycle
queries.** Claim A currently rests on comparing a gate scored on one support against a
margin scored on another — a reviewer will find this.

### What runs

Two behavioural gate runs, no new code (`prompts.py` already takes `operand_pool` and
`query_pool` as independent knobs):

- **R4** — `experiments/gate_months_partA4_fullq.json`: copy of `gate_months_partA4`,
  `query_pool` dropped, operand pool Jan–Apr kept. MPS, fp16, batch 8 (the
  `pilot_llama32_mps_months_s16` settings), 16 shots, k = 0..11, n = 100/k, held-out
  stratum, seed 0.
- **R6** — `experiments/gate_months_halves_A_fullq.json`: same treatment of
  `gate_months_halves` (operand pool Jan–Jun).

Code facts verified before registration, against `tarcle/prompts.py`:

1. With `query_pool` absent, queries draw from the full 12-month cycle
   (`make_prompt_set`: `qpool = DOMAINS[qd]`).
2. `_sample_demo_operand` handles an out-of-pool query correctly: the held-out
   exclusion filter removes the query only if it is in the pool, so an out-of-pool
   query leaves the demo pool at its full size (4 or 6) and an in-pool query reduces
   it by one, exactly as in the original runs. Held-out is automatic for out-of-pool
   queries.
3. `choices = list(DOMAINS[qd])` regardless of any pool restriction — the
   forced-choice set remains the full 12-month cycle and chance stays 1/12.

The runner refuses to overwrite existing run dirs; both run names are fresh.

### Instrument characteristics, fixed before reading

- Accuracy is read per k on the held-out stratum via `tarcle.pilot_report`
  (GO ≥ 0.50 at every k; NO-GO floor 0.30 at any k — the project's standing gate).
  n = 100/k, so Wilson 95% intervals are ≈ ±0.10 at p = 0.5. Point estimates carry
  the verdict, per project convention; straddling CIs are flagged inline.
- **Device-matched reference.** The full-pool comparison profile is the 16-shot
  full-pool run `llama32_3b_mps_months_s16` — same device, same dtype, same seed
  discipline as tonight's runs: F(k) = 1.00, 1.00, 1.00, 1.00, 0.76, 0.76, 0.98,
  0.70, **0.38**, 0.80, 0.86, 1.00 for k = 0..11.
- **In-pool anchor check (instrument validity).** The in-pool query subset of each new
  run (~33/100 per k for R4, ~50/100 for R6) is compared per k against the original
  CUDA bf16 in-distribution gates (`gate_months_partA4`: every k ≥ 0.50;
  `gate_months_halves_A`: every k ≥ 0.83). If ≥3 k differ beyond the two-proportion
  95% margin (unequal n), the fp16/MPS instrument is flagged and the flag is reported
  with the verdict. The flag does not by itself change the branch; it bounds how much
  the branch can claim. The split reader is a small stage-2 join of `prompts.jsonl`
  and `scores.jsonl` on `idx`, grouping by query ∈ pool — analysis code, no model
  code.
- The out-of-pool query subset (~67 and ~50 per k) is reported per k separately: it is
  the genuinely new cell.

### Pre-committed branches — written before either run is launched

The FV task-encoding margins are already on disk and are not re-measured
(+0.352 primary, +0.343 polysemy, −0.315 halfA, −0.463 halfB, −0.704 partA,
−0.944 partB). T1 changes only the query support of the behavioural gate.

**Branch 1 — claim A survives in strong form.** The run's full-cycle-query gate
verdict is GO at every k, **or** MARGINAL where every sub-0.50 cell (i) is sub-0.50 in
the device-matched full-pool profile F too (i.e. only k=8, F(8)=0.38), and (ii) is not
below F(k) by more than the two-proportion 95% margin. Reading: restricting the
demonstrations costs nothing behaviourally on full-cycle queries beyond what the full
pool already could not do; the original gates' support mismatch was immaterial; the
accuracy gate — scored on the same support as the FV margin — still passes while the
extracted vector encodes a different function.

**Branch 2 — claim A is killed in its strong form and the paper rescopes.** The run's
verdict is NO-GO (any k < 0.30), **or** any cell with F(k) ≥ 0.50 falls below 0.50 by
more than the two-proportion 95% margin against F(k). Reading: the model cannot do
restricted-demonstration shift-by-k on the full query cycle; the original GO verdicts
were an artifact of in-distribution scoring; the FV's negative margin is confounded
with genuine behavioural failure on exactly the queries the margin was computed over.
Claim A weakens to "we gated on the wrong query support," the paper is rescoped around
claim B, and this rescoping is **recorded, not argued away** — it is itself an
instance of the thesis (a validation gate that was scored on the wrong distribution),
and is written up as such, not as the headline.

**Registered resolution for the marginal middle** (cells below 0.50 but within the
noise margin of an F(k) ≥ 0.50, or a single such cell): recompute the D20 margin from
the saved `efficacy_pred_shift`, restricted to the k that clear 0.50 in this run (and
∉ {0, 1, 11}). If the restricted gate passes and the restricted margin stays ≤ −0.10,
claim A survives in **qualified** form, with the failing cells named wherever the
claim is stated; otherwise Branch 2.

**Cross-run rule.** Claim A in strong form requires Branch 1 at **both** pool sizes.
If R4 and R6 land in different branches, the claim is stated only for the pool size
that survived, and the B-side conditions (`partB` Sep–Dec, `halfB` Jul–Dec, full-cycle
queries) are run before the paper's framing is fixed — the two B configs are the same
one-line diff and are pre-authorised by this paragraph, but only in that contingency.

Neither run's `scores.jsonl` is opened until this document is committed.

---

## 2. T3 — task-encoding margin split by operand exposure (numpy)

### What was already seen

The aggregate margins (table above) and the ±1 collapse (D19–D21). The split of those
margins by query-operand exposure has never been computed.

### Recoverability — established, not guessed

`efficacy_pred_shift` has shape (n_k, 12) and its query axis is the canonical
January→December order: `causal.efficacy` → `score_for_k` → `zero_shot_spec` with
`queries=None` builds one zero-shot prompt per operand in `DOMAINS["months"]` order.
Column j is therefore month j, and exposure is pool membership recorded in each
`.npz`'s `meta_json["operand_pool"]`. No schema change is needed. (Exposure is defined
at the pool level: over 100 × 16 demonstration draws every pool member appears; no
non-member ever does.)

### Statistic

For each condition ∈ {partA, partB, halfA, halfB; primary and polysemy as references}
and each extraction method: the D20 margin P(correct shift) − P(shift ±1) over
k ∉ {0, 1, 11}, computed separately over in-pool query columns (M_in) and out-of-pool
columns (M_out). Cell counts stated with the result (partA/partB: 4 × 9 = 36 in-pool
predictions, 8 × 9 = 72 out; halves: 54/54; every count is a census of the query
space, not a sample, but the per-cell binomial CI at n = 36 is ≈ ±0.16 and is
reported).

### Pre-labelled readings

- **R-lookup** — *better for the paper, so it is tested rather than defended against*:
  M_in ≥ +0.10 (the D20 gate level) while M_out ≤ −0.10. The vector encodes an
  operand-bound mapping — the model solved the restricted task by in-context lookup
  and the FV faithfully reports that. The "collapse" is then a support restriction of
  the encoded function, not a broken vector, and claim A's mechanism sharpens from
  "the FV is degenerate" to "the FV encodes the function actually induced, which is
  not the function the gate certified."
- **R-collapse**: M_in ≤ −0.10 as well. The vector steers to next-item even on its own
  demonstration support; D19/D20's original reading stands unchanged.
- **R-mixed**: anything else — reported per condition with no verdict.

A reading fires only where its side's 95% binomial CI excludes the opposing threshold;
otherwise R-mixed. Todd carries the analysis; Hendel is reported alongside under its
registered limitation (steers only at k ∈ {1, 2, 11}, D12/D16 §3 — expected
uninformative here, reported anyway per rule 1).

Output: `results/stage2/margin_split.json` + text table.

---

## 3. T5 — importance-measure disagreement (numpy)

### What was already seen

The four k=6 values (`pilot_findings.md` §8, logged 2026-08-03: accuracy 0.98, L14H1
AIE +0.0454 = its minimum, injection efficacy 0.33, norm 6.931 = the smallest). That
observation motivated this test; it cannot also confirm it. The registered content is
the **full rank-correlation matrix**, which has never been computed, and k=6 is
reported as one cell of it.

### Series, k = 1..11

k=0 is excluded on registered grounds: no AIE exists for it (D3/D14) and its efficacy
is uninterpretable (D2 ceiling caveat). The field-interchangeability claim under test
concerns tasks the protocol actually measures.

| series | source |
|---|---|
| held-out ICL accuracy | `results/pilot/llama32_3b_mps_months_s16` per-k held-out |
| causal AIE, L14H1 | `months_llama32_3b_ada_allk/heads.npz` `aie_per_k[:, 14, 1]` (the only per-k AIE at all 11 k) |
| causal AIE, head-set mean | same array, mean over the canonical 10 cells (`head_set` of the canonical run) |
| injection efficacy | `ctl_months_primary` `efficacy_logp_lift` (primary metric per D17); `efficacy_acc` secondary |
| FV norm | `ctl_months_primary` `norms` |

Both extraction methods for efficacy and norm; accuracy and AIE are
extraction-independent and shared across the two columns.

### Statistic and power, fixed before computing

Spearman ρ for every pair, exact permutation p (10,000 draws, seed 0). At n = 11 the
two-sided 5% critical value is ≈ 0.62 — stated so no weaker correlation is
over-read.

**Attenuation ceilings, computed first.** Each series' reliability r is estimated from
its own measurement noise (efficacy and norm: split halves `X_half_a`/`X_half_b` and
`efficacy_logp_se`; AIE: `aie_se` via r = var_k(AIE)/(var_k(AIE)+mean(SE²)); accuracy:
binomial SE at n = 100). The maximum observable correlation for a pair is
√(r_a·r_b). **A pair whose ceiling is < 0.50 is labelled noise-limited and carries no
disagreement claim** — that is the D37 lesson applied to rank correlations.

### Pre-labelled outcomes

- **Measures disagree** (supports the thesis): at least one pair among {accuracy,
  AIE-L14H1, efficacy logp-lift, norm} with ρ ≤ 0.30 while that pair's ceiling
  ≥ 0.70, under Todd. Reported as a general disagreement matrix — the k=6 cell is an
  instance, not the finding.
- **Noise-limited**: pairs below the ceiling threshold, reported as uninformative.
- **Killed**: every pair with ceiling ≥ 0.70 shows ρ ≥ 0.70 — the measures agree at
  this resolution, T5 contributes a negative, and it is reported as one.

Output: `results/stage2/measure_corr.json` + per-k table + matrix.

---

## 4. T10 — shared-offset audit (numpy)

### What was already seen

The offset shares (78.8% Todd / 94.9% Hendel of mean squared norm, commit `3eac9d8`),
the raw-vs-centered **circulant** column and centered permutation z (D27,
`stage2_findings.md` §4). Not yet computed: the centered column for the remaining
raw-Gram statistics, and the k-dependent signal stated as an absolute quantity against
the measured noise band.

### Registered content — descriptive throughout, no verdict language

1. **Raw vs centered for every `geometry.py` statistic**, per condition and method.
   Translation-invariant or internally-centered statistics (`closure_ratio`,
   `fit_rotation`, `participation_ratio(center=True)`, `additivity_residual`) are
   asserted equal and listed once; the substantive columns are `circulant_score`,
   `toeplitz_score`, `spectral_concentration` (reported under the D26 gate rule —
   not read where circulant < 0.70), and `norm_cv`.
2. **Signal-to-noise in absolute terms.** Per k: signal_k = ‖X[k] − mean(X)‖ and
   noise_k = ‖X_half_a[k] − X_half_b[k]‖/2, cross-checked against the stored `X_se`.
   Reported: the per-k table, RMS(signal)/RMS(noise), and the one registered summary
   sentence pattern — "the k-dependent component is Y× the split-half noise floor and
   carries Z% of mean squared norm" — so a reader can see how much signal there was to
   have geometry in, against the D16 split-half band that every stage-2 verdict
   already uses as its noise reference.

The D27 interpretation constraint carries over verbatim: any statement about the
centered column must say that it describes the k-dependent component after removing an
offset carrying ~80–95% of the norm — the offset is part of the vector that actually
steers the model.

Output: `results/stage2/offset_audit.json` + table.

---

## 5. T6 — the floor-estimator table (numpy; the paper's centrepiece)

### What was already seen

Individual cells are public in the log: harmonic partial R² floors (D37,
family2-prereg §1), seam cv-R² on add-k/unrelated (D32), permutation z on all three
real families (D30), circulant scores throughout. The registered content is the
**unified table** — every diagnostic × every floor construction, at one matched
regime, with ratios — most of whose cells do not yet exist. Pre-commitments below bind
the not-yet-seen cells; already-published numbers are carried into the table unchanged
and marked as such.

### Rows

`circulant_score` (raw and centered), `toeplitz_score`, permutation z
(`stage2.permutation_null`, 200 row permutations), `spectral_concentration`, isotonic
seam (two numbers: cyclic-model cv-R², and best-cut-minus-cyclic margin, from
`seam.contest`), harmonic partial R² (`power.partial_r2`), `participation_ratio`.

### Two null constructions, and which is claim-appropriate per row

- **N1 — no ordering at all.** Real-vector form: row permutation of X (destroys the
  k-ordering, keeps every vector intact). Synthetic form: isotropic gaussian cloud at
  matched n, d, scale, plus a shared offset at the measured share. Claim-appropriate
  for statistics whose positive reading asserts *ordering*: `toeplitz_score`,
  permutation z, the seam contest's "any model fits" reading.
- **N2 — ordered, nothing beyond.** Real-vector form: `power.residual_surrogate`
  (keep the [1, k] fit, permute residual rows). Synthetic form: line + isotropic
  noise, axial scale and residual RMS matched to the months measurement.
  Claim-appropriate for statistics whose positive reading asserts *structure beyond
  ordering* (a cycle, wraparound, a harmonic): `circulant_score`,
  `spectral_concentration`, harmonic partial R², the seam contest's cyclic-vs-cut
  margin.
- `participation_ratio` is descriptive, not a detector: it gets 5th–95th null ranges
  under both constructions and no floor language.

### Columns

1. **C1 — synthetic-null floor**: 95th percentile over 300 seeded draws (seed 0,
   matching `power.py`) of the synthetic construction. This is the validation
   practice the field uses and prereg §0 used.
2. **C2 — real-vector permutation floor**: 95th percentile over 300 draws of the
   claim-appropriate permutation (N1 row-permutation or N2 residual permutation) on
   the months-primary vectors, n = 11 drop-k=0 (D28), both methods.
3. **C3 — real no-structure families, observed**: add-k (ordered, definitionally
   non-cyclic — the no-structure reference for beyond-ordering rows) and the
   unrelated null (unordered — the reference for ordering rows), values as measured,
   plus the same permutation floors computed within those families.
4. **C4 — ratio C2/C1**, the claim-B number. Where C1 < 0.001 the ratio is unstable
   and the row is reported verbatim as "synthetic floor ≈ 0 vs real floor X" — which
   is itself the finding, not a missing value.

Matching is at the measured months regime: n = 11, d = 3072, offset share 78.8/94.9%,
axial share and residual RMS as fitted by the [1, k] design. Constructors live in a
new numpy-only module `tarcle/floors.py`, seeded, output recording git commit and
config hash like every result file.

### Validation before reading (the D37 gate, applied to the table itself)

Before any real-data cell is interpreted: (i) the pipeline run on a planted synthetic
circle must put the observed statistic above its own floors on every detector row;
(ii) on the synthetic null itself, exceedance of the 95th-percentile floor must be
≈ 5% by construction. If either fails, the affected row is void and reported as such.

Runtime note, fixed by clock and not by results: the seam row (40-fold cv × 300
draws × 13 models) is the slow one; if wall clock demands, its draws may be reduced
to 100 — decided before looking at any of its output.

### Pre-committed reading for claim B

- **Generalises**: median C4 across the detector rows (both methods) ≥ 10. The paper's
  "~100×" framing is then supported as a class-level statement about synthetic
  validation, with per-row spread shown.
- **Does not generalise**: median C4 < 3 — claim B is confined to the one statistic
  D37 measured, the paper says so, and the centrepiece figure becomes a per-statistic
  honesty table rather than a blanket claim.
- Between 3 and 10: reported per statistic, no blanket sentence.

Output: `results/stage2/floors.json` + `floors.txt` (the table). A fixture smoke test
for `tarcle/floors.py` goes into `tests/` and must keep the gpt2-CPU suite under
5 minutes.

---

## 6. Registered exception to CLAUDE.md rule 5 (floor vectors)

Rule 5 forbids extracting FVs from tasks the model cannot perform, and D20 §3 extended
it per condition. **Floor estimation needs vectors, not correct vectors**: the
false-positive floor of a diagnostic is a property of the estimator on realistic
activation statistics, and vectors from gate-failed conditions are exactly the
no-signal material the floor question calls for.

The exception, its purpose, and its scope limit:

- Vectors already on disk from **gate-failed conditions** (partition A/B, halves A/B —
  D20/D21) may be used in T6 and T3 as instrument-characterisation material.
- Any **future** extraction from a NO-GO family for floor purposes is licensed only as
  such: bf16 minimum, labelled `floor_only` in its metadata, and categorically never
  used to support a claim about task-space geometry or about the task itself.
- The exception does not weaken rule 5 for its original purpose: no geometry or
  hypothesis claim may ever rest on these vectors. They characterise estimators, not
  tasks.

---

## 7. Provenance and order of operations tonight

Unchanged constraints: two extraction methods always; stage 2 is numpy-only and never
imports torch; no quantization for anything geometry-bearing (nothing tonight is);
everything seeded; every result file records git commit and config hash; results are
never overwritten; tests pass on gpt2 CPU in under 5 minutes.

Order: this document is committed → the two T1 configs are approved and launched (they
burn wall clock in the background) → T5 → T10 → T3 → T6 with the remaining time →
T1's `scores.jsonl` is read **last**, against the §1 branches as written. If tasks
finish early, the time goes to writing, not to new runs.
