# Pre-registration: FV geometry for shift-by-k on months (Z/12)

**Status: written and committed before any Gram matrix of real function vectors has been
computed or inspected.** Nothing in this document is informed by extracted FVs. The only
empirical inputs are (a) the behavioral pilots in `docs/pilot_findings.md` and (b)
calibration of the diagnostics against the synthetic ground-truth shapes in
`tarcle/synthetic.py`, which contain no model data.

Primary family: **months, Z/12**, Llama-3.2-3B. Secondary replication family: days, Z/7.
ROT-k deferred (rationale in `docs/pilot_findings.md` §4 and the session record: letters are
the weakest domain at 0.35 aggregate, and ROT-13's corpus over-representation would give the
family a privileged k, which is poison for a Gram matrix assuming comparable FV quality at
every k).

---

## 0. Diagnostic calibration (the basis for every threshold below)

Median over 20 seeds, n=12, d=64, from `tarcle.geometry.diagnose` on `tarcle.synthetic`
fixtures. `noise=0.3` is the realistic column; treat `noise=0.0` as the ceiling.

| shape | circulant | toeplitz | spec.conc | closure | wrap.err | norm_cv | additivity | PR | sig.freqs |
|---|---|---|---|---|---|---|---|---|---|
| circle f=1 | 0.92 | 0.92 | **1.00** | 1.00 | 0.49 | 0.02 | 1.39 | **2.3** | [1] |
| helix f=(1,2) | 0.92 | 0.92 | 0.51 | 0.99 | 0.21 | 0.02 | 1.40 | 4.2 | [1,2] |
| helix f=(1,3) | 0.92 | 0.92 | 0.51 | 1.01 | 0.25 | 0.02 | 1.40 | 4.2 | [1,3] |
| helix f=(2,3,4) | 0.93 | 0.93 | 0.35 | 1.00 | 0.13 | 0.02 | 1.41 | 6.1 | **[4] only 5/20** |
| line | 0.02 | 0.06 | 0.90 | 10.2 | 11.0 | **0.62** | **0.10** | 1.0 | [1] |
| simplex (D) | 0.79 | 0.80 | **0.20** | 1.01 | 0.13 | 0.03 | 1.44 | **10.7** | **[]** |
| arc (open) | 0.21 | **0.93** | 0.96 | 4.04 | 3.95 | 0.03 | 0.77 | 1.7 | [1] |

Uniform non-DC power share at n=12 is 1/(n//2) = **0.167**.

Three facts this table establishes, which the rest of the document depends on:

1. **`circulant_score` alone cannot separate A from D.** Simplex scores 0.79. As CLAUDE.md
   warns, D's Gram (identity + constant) is trivially circulant.
2. **The A-vs-D separators are `spectral_concentration` and `participation_ratio`.** A gives
   concentration ≥ 0.5 and PR ≈ 2–6; D gives concentration ≈ 0.20 (i.e. ≈ uniform) and
   PR ≈ n−1 ≈ 10.7. These are far apart and both are pre-registered as primary.
3. **`significant_frequencies` at its default `factor=2.0` is not reliable for detecting
   *multiple* frequencies.** Three equal-amplitude frequencies each hold ≈1/3 of non-DC
   power, and 2× uniform = 0.333 sits exactly at that boundary — the helix f=(2,3,4) case
   was flagged in only 5 of 20 noiseless seeds. **This is a trap for the divisor prediction
   in §1, which predicts power at several frequencies at once**: a true multi-frequency
   result could return `[]` and be misread as D. Mandatory mitigation: always report the
   full `frequency_pair_powers` vector, never `significant_frequencies` alone, and read
   multi-frequency structure off the vector against the 0.167 uniform line.

---

## 1. The divisor-structure prediction

**Claim under test:** if the model represents Z/12 with awareness of its factorization
Z/12 ≅ Z/4 × Z/3, the Gram spectrum carries power at frequencies with `gcd(f,12) > 1` —
f ∈ {2,3,4,6} — and not at f=1 alone.

Frequency f of the circulant profile is the character of order 12/gcd(f,12). f=1 is the
faithful generator (a plain circle). f=6 is the order-2 character (the Z/2 quotient), f=4,8
the order-3 characters (Z/3 factor), f=3,9 the order-4 characters (Z/4 factor).

**Pass:** non-DC power at f ∈ {2,3,4,6} pooled exceeds power at f ∈ {1,5} pooled, AND at
least one of f ∈ {3,4,6} individually exceeds 2× the 0.167 uniform share.
**Fail:** power concentrated at f=1 with {3,4,6} each below uniform — a plain circle that
knows nothing of the factorization.

**Attached behavioral prior (weak, and confounded — stated so it cannot be retro-fitted).**
Held-out months accuracy at 10 shots ran high at divisor shifts k=2 (1.00), k=3 (0.98),
k=6 (0.90) and low at non-divisors k=5 (0.50), k=7 (0.56). The k=6 value is the most
suggestive single number in the pilot: it sits *above* both neighbours, and k=6 is the
order-2 self-inverse element.

Two reasons this prior is weak, registered in advance:

- **It is confounded with |m|.** k=5 and k=7 are the most distant shifts, so their low
  accuracy may be pure cycle-distance rather than non-divisor status. Conversely k=1 and
  k=11 score 1.00 and 11 is not a divisor. **Any behavioral version of this test must
  regress out |m| = min(k, 12−k) first and inspect the residual.**
- **k=4 breaks it.** k=4 is a divisor and scores only 0.62.

The behavioral prior therefore *motivates* the spectral test and is not evidence for it. If
the DFT result contradicts the behavioral pattern, the DFT result stands.

---

## 2. Hypothesis battery and pass criteria

Every criterion is evaluated **under both extraction methods** (Todd-style causal-head
averaging and Hendel-style dummy-query hidden state), per CLAUDE.md rule 1. A result that
holds under only one extraction is recorded as a finding about the method, not the model.

### A — circular (closed loop, single dominant frequency)
- `circulant_score` ≥ 0.70
- `spectral_concentration` ≥ 0.50 **and** dominant pair f=1
- `participation_ratio` ∈ [1.5, 3.5]
- `closure_ratio` ≤ 1.5 and `rotation_wraparound_error` ≤ 1.0
- `norm_cv` ≤ 0.15; `additivity_residual` ≥ 0.5 (additivity must **fail**)
- k=1.5 interpolation test: injecting the synthesised on-circle point yields coherent
  shift-by-1.5 behaviour rather than a mixture of k=1 and k=2 outputs

**A-multi (helix / multi-irrep)** is the same but with `spectral_concentration` ∈
[0.25, 0.50], PR ∈ [3.5, 7], and ≥2 frequency pairs above the 0.167 uniform line. This is
where the §1 divisor prediction, if true, lands.

### B — linear selector
- `circulant_score` ≤ 0.30 with `toeplitz_score` also low
- `norm_cv` ≥ 0.40, `additivity_residual` ≤ 0.25, PR ≈ 1
- `closure_ratio` ≥ 3
- Behavioral counter-evidence already on file: BRIEF §8 holds that a linear code predicts
  wraparound cases are *systematically worse*; the pilot found them systematically **better**
  (days k=6 0.70 vs k=5 0.20; months k=11 1.00, k=10 0.88 vs k=8 0.34). B is disfavoured
  going in. Registering that now so a B-shaped result is treated as surprising and audited
  per CLAUDE.md's style rule (extraction code first, prompt leakage second, model third).

### C — FV is a lossy projection of an operator
Not a geometry shape — a *causal* prediction, and the one with the most teeth.
- Extract FV(k) from demonstrations whose operands are restricted to one partition
  (e.g. {Jan..Apr}); apply to queries from a disjoint partition (e.g. {Sep..Dec}).
- **Pass (C true):** steering efficacy drops significantly on transfer, and/or the induced
  error is a systematic wrong-region push rather than uniform degradation.
- **Fail (C false):** transfer efficacy within noise of matched-partition efficacy.
- C can hold *simultaneously* with A or D — it constrains what FVs are, not what shape they
  trace. Report it independently; do not treat A and C as mutually exclusive.

### D — simplex / Bayesian mixture
- `circulant_score` high (≥0.70) **but** `spectral_concentration` ≤ 0.25 (≈ the 0.167
  uniform share) and `participation_ratio` ≥ 8
- `significant_frequencies` empty *when the full power vector is also flat* (see §0.3 — an
  empty list with a non-flat power vector is a threshold artefact, not evidence for D)
- k=1.5 interpolation yields mixture-of-neighbours behaviour (outputs at k=1 and k=2), not
  coherent shift-by-1.5

### Null — lookup, no structure
`circulant_score` ≤ 0.30 and `toeplitz_score` ≤ 0.30 and PR ≥ 8. Publishable per BRIEF §8
if the controls are tight.

### Mandatory controls (part of the main experiment, not follow-ups)
Per CLAUDE.md rule 2, all four run alongside the primary result:
operand-partition control; mixed-domain operands; n unrelated tasks (**must** show no
circulant structure — a positive here invalidates the pipeline); ordinal extract-k-th family
(open curve expected: high toeplitz, low circulant, closure_ratio ≫ 1).

---

## 3. Mandatory artefact control: month-name token frequency

BRIEF §9 flags month names as frequency- and polysemy-hazardous (May, March, August carry
non-month senses). This control is **blocking**: geometry results are not reportable until
it has been run and its outcome recorded alongside them.

**Frequency proxy.** For each month token, the model's unconditional next-token logprob
under a small fixed set of neutral contexts, averaged. Computed once, saved with the FVs.

**Test 1 — norm artefact.** For each k, compute the mean frequency proxy over that k's
*target* tokens (target = operand + k, so the target distribution is a shifted copy of the
operand distribution and its mean proxy varies with k even under uniform operand sampling).
Correlate against `norm_profile(X)` across k.
- **Pass:** |Spearman ρ| ≤ 0.4, or ρ significant but `norm_cv` ≤ 0.15 (structure cannot be
  carried by norms that barely vary).
- **Fail:** |ρ| > 0.4 with non-trivial `norm_cv` — the norm profile is tracking token
  frequency and any "constant norms" claim under A is contaminated. Remedy: re-run with
  frequency-stratified operand sampling before reporting geometry.

**Test 2 — polysemy leave-out.** Re-extract with May, March and August excluded from all
operand pools (Z/12 structure preserved by holding the label set fixed and sampling operands
only from the remaining nine). Re-run the full battery.
- **Pass:** every §2 verdict is unchanged.
- **Fail:** any verdict flips → the result is about token identity, not task structure, and
  must be reported that way.

**Test 3 — cross-family.** Days (Z/7) has its own frequency profile and no polysemy problem.
Any conclusion holding on months but not days must be reported as family-specific.

---

## 4. Pre-registered handling of a weak k cell

Fixed **before** the months-at-16-shots result was inspected, so it cannot be tuned to the
outcome. Let `k*` denote a k whose held-out accuracy falls below the 0.50 GO threshold
(the pilot's candidate is k=8, at 0.34 with 10 shots).

- **If every k ≥ 0.50:** GO. Full sweep, no exclusions, no leave-one-out reporting.
- **If exactly one `k*` improves over its 10-shot value but stays below 0.50:** do **not**
  fail the family. Extract FV(k*) anyway and run **every geometry diagnostic twice — full
  n=12 including k*, and leave-one-out excluding k* — reported side by side in the same
  table.** Neither is the headline; a result is only claimed where the two agree.
  - Leave-one-out at n=11 changes the group: Z/11 is prime, so the divisor test of §1 is
    **not defined** on the reduced set. The n=11 column is therefore evaluated on
    circulant/concentration/PR/closure only, and §1 is reported from the full n=12 column
    with the weak cell flagged inline.
  - Divergence between the two columns is itself the finding and must be reported, not
    resolved by preferring one.
- **If `k*` fails to improve at all, or ≥2 cells sit below 0.50:** NO-GO for months as
  primary. Escalate to the fallback in BRIEF §7 (a small model trained directly on modular
  shift, with a fully controlled task distribution) rather than to ROT-k.
- **Interpolation caveat:** the k=1.5 test in §2 must not be sited adjacent to `k*`. If
  k*=8, run the interpolation at k=1.5 and k=2.5 as registered, never at 7.5 or 8.5.

---

## 5. What would make us discard the run entirely

- The "n unrelated tasks" control shows circulant structure → pipeline bug, discard all.
- Prompt-set SHA-256 fails to reproduce from the recorded seed → determinism broken.
- Any geometry-bearing run executed below bf16 → violates CLAUDE.md rule 4. The fp16/MPS
  pilots are behavioral only and no geometry claim may rest on them.
- Extraction-method disagreement is **not** grounds for discarding: it is a reportable
  finding about FV extraction (CLAUDE.md rule 1).
