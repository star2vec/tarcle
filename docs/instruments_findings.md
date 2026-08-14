# Instruments findings: how permissive is interpretability validation practice?

Results for the tasks registered in `docs/preregistration_instruments.md`
(committed before any output existed; branches and readings quoted from there and
from D39–D42). Everything below is pure numpy off saved `.npz` except §5 (T1),
which is behavioural-only at fp16 on MPS per CLAUDE.md rule 4. Raw numbers:
`results/stage2/{measure_corr,offset_audit,margin_split,floors}.json` and
`results/pilot/gate_months_{partA4,halves_A}_fullq/`.

---

## 1. T5 — the field's importance measures do not agree with each other

Spearman rank correlations across k = 1..11 (k=0 excluded on registered grounds:
no AIE exists for it, D3; efficacy uninterpretable at ceiling, D2), with each
pair's **attenuation ceiling** — the maximum correlation the measures' own noise
allows, √(r_a·r_b) — computed before any correlation was read.

**Registered outcome: MEASURES DISAGREE, under both extraction methods.**

Todd, the key cells:

| pair | ρ | perm. p | ceiling |
|---|---|---|---|
| accuracy · AIE (L14H1) | **+0.91** | 0.0005 | 0.90 |
| accuracy · AIE (head-set mean) | **+0.98** | 0.0001 | 0.92 |
| accuracy · efficacy (logp lift) | **+0.02** | 0.96 | **0.93** |
| AIE (L14H1) · efficacy (logp lift) | +0.13 | 0.71 | 0.89 |
| efficacy (logp lift) · efficacy (argmax acc) | **+0.12** | 0.73 | 0.91 |
| accuracy · FV norm | +0.62 | 0.047 | 0.97 |

Three readings, in decreasing order of expectedness:

1. **Behavioural accuracy and causal AIE track each other closely** (0.91–0.98).
   Whatever they measure, they measure it together.
2. **Injection efficacy tracks neither** — ρ ≈ 0 against accuracy with a noise
   ceiling of 0.93, so the disagreement is real, not attenuation. The most
   common validation pattern in the FV literature ("the task is performable,
   and injecting the vector steers the model") composes two measures that are
   **statistically unrelated across tasks** in this family.
3. **The two readouts of the same injections disagree with each other** (logp
   lift vs argmax accuracy: ρ = 0.12). D17's metric change was not a
   refinement; it changed which cells count as working.

Under Hendel the pattern is the same with one addition: FV norm decorrelates
from everything (acc·norm ρ = +0.005), and its efficacy series is zero-inflated
by the known ±1 confinement (D12), with reliability 0.67 — reported per the
registered noise-limited rule where applicable.

The k=6 cell of `pilot_findings.md` §8 (accuracy 0.98 / AIE minimum / efficacy
0.33 / smallest norm) is one row of this matrix, as registered — an instance,
not the finding.

Registered approximations (in the output file): AIE per-k SE inflated from the
pooled SE by √11; efficacy-lift SE from the injected side only.

---

## 2. T10 — the shared offset, audited statistic by statistic

Offset share of mean squared norm, and the k-dependent signal against the
split-half noise band (primary condition):

| | Todd | Hendel |
|---|---|---|
| offset share | 78.8% | 94.9% |
| RMS k-signal | 3.55 | 3.16 |
| RMS split-half noise | 0.39 | 0.19 |
| **signal / noise** | **9.0×** | **16.4×** |

> The k-dependent component is 9.0× (Todd) / 16.4× (Hendel) the split-half
> noise floor and carries 21.2% / 5.1% of mean squared norm.

So there was ample signal to have geometry in — the geometry verdicts were not
starved of signal; the signal is simply not circulant.

Raw vs centered for every non-translation-invariant statistic (primary, Todd →
Hendel): circulant 0.091→0.298 / 0.113→0.267; toeplitz 0.159→0.326 /
0.177→0.285; norm_cv 0.054→0.460 / 0.012→0.484. **Every one of these moves by
more than both split-half bands** (bands ≤ 0.007 throughout — the estimates are
extremely stable, so the offset's effect is entirely systematic, not noise).
`closure_ratio`, the rotation fit, `additivity_residual` and
`participation_ratio` are translation-invariant or internally centered
(asserted in code, not assumed). Same pattern in every other condition
(polysemy, mixed-domain, add-k; offset shares 77–98%).

**One mechanism finding, new:** `spectral_concentration` is *exactly* identical
raw and centered — to machine precision, in every condition. Reason: each
(i−j) mod n class contains every row and every column exactly once, so the D27
rank-2 cross-term shifts all class means by the same constant, which lands
entirely in the DC bin of the profile's DFT. This resolves D27's puzzle — the
offset corrupts within-class variance (`circulant_score` 0.09) while leaving
the class-mean profile untouched (`spectral_concentration` 0.68) — and sharpens
D26: the concentration statistic is structurally blind to the contamination
that invalidates the gate it depends on. It cannot warn you that its own
premise has failed.

norm_cv deserves a sentence: the "constant norms" reading (0.05/0.01, prereg
§2's anti-B evidence) is an artifact of the offset in the specific sense that
the k-dependent component's norms are *not* near-constant (centered cv 0.46 /
0.48). The registered B-test used raw norms and its verdict stands as
registered; any future use of norm constancy as evidence must say which vector
it describes.

---

## 3. T3 — the collapse is real: lookup tested and rejected

The registered better-for-the-paper reading (R-lookup: the model solved the
restricted task by in-context lookup and the FV faithfully reports an
operand-bound mapping) **does not survive**. D20 margin split by whether the
zero-shot query's operand was in the demonstration pool (exact split — the
efficacy query axis is the canonical month order; exposure is pool membership):

Todd:

| condition | pool | M_in (95% CI) | M_out (95% CI) | registered reading |
|---|---|---|---|---|
| partA (Jan–Apr) | 4 | −0.69 [−0.91,−0.48] | −0.71 [−0.85,−0.57] | **R-collapse** |
| partB (Sep–Dec) | 4 | **−1.00** [−1.00,−1.00] | −0.92 [−0.99,−0.84] | **R-collapse** |
| halfA (Jan–Jun) | 6 | −0.13 [−0.36,+0.10] | −0.50 [−0.71,−0.29] | R-mixed |
| halfB (Jul–Dec) | 6 | −0.44 [−0.64,−0.25] | −0.48 [−0.68,−0.28] | **R-collapse** |
| polysemy (9) | 9 | **+0.47** [+0.31,+0.63] | −0.04 [−0.38,+0.30] | R-mixed |
| primary (12) | 12 | +0.35 [+0.20,+0.50] | — | reference |

- At 4 operands the vector steers to ±1 **even on its own demonstrated
  operands** (partB M_in = −1.00: every in-pool mid-cycle prediction lands on
  the adjacent month). The FV does not encode a working operand-bound mapping;
  it encodes next-item. D19/D20's reading stands, now with the alternative
  measured rather than argued away.
- **An exposure gradient rides on top of the collapse**: halfA in-pool −0.13 vs
  out-of-pool −0.50; and in the *passing* polysemy condition the margin is
  +0.47 on exposed operands against −0.04 on the three unexposed months. The
  polysemy split does not fire R-lookup under the registered CI rule (n = 27,
  CI [−0.38, +0.30] straddles), but it is the right shape for
  operand-boundedness in a healthy FV and is recorded as the follow-up the
  registered rule could not adjudicate.
- Under Hendel every months condition has a negative margin including the
  primary (−0.34) — the ±1 confinement is a property of that extraction
  everywhere, not only at restricted pools (consistent with D12), reported per
  rule 1.

---

## 4. T6 — the floor-estimator table: claim B does not generalise, and the truth is stranger

300 draws, seed 0; D42 fixed the C1a/C1b construction split and the gate rework
before this table was read. Full table in `results/stage2/floors.txt`.

**Validation gate first** (D42 rule): seven of eight detector rows pass —
planted positives exceed the operative real-permutation floors, and every C2
floor is calibrated (self-exceedance 0.023–0.060 against the [0.02, 0.10]
band). One row is **VOID**: `perm_z`'s planted line reaches only z = +1.03
against a C2 floor of 2.15. The mechanism matters: the permutation-z signal on
real ordered families is carried by smooth cross-term structure (the D27
rank-2 term — ⟨X_k, X̄⟩ varying smoothly with k), which a synthetic line with
an independent offset direction cannot produce. The detector demonstrably
works on *real* ordered data (add-k, a real family with known ordering, sits
at z = +7.44 in the same table) — what failed is the synthetic plant, which is
itself a claim-B instance: **the fixture cannot even carry the signal the
statistic actually responds to.**

Key columns (Todd / Hendel; C1a = fixture-grade synthetic floor, the repo's
committed practice; C2 = real-vector permutation floor; C4a = C2/C1a):

| statistic | C1a fixture | C2 real | **C4a** |
|---|---|---|---|
| harmonic partial R² | 0.0019 | 0.279 / 0.300 | **145 / 156** |
| circulant (raw) | 0.019 | 0.114 / 0.086 | **6.0 / 4.5** |
| circulant (centered) | 0.119 | 0.248 / 0.239 | 2.1 / 2.0 |
| permutation z | 1.90 | 2.15 / 2.18 | 1.1 / 1.1 |
| seam cut-margin | 0.281 | 0.221 / 0.212 | 0.8 / 0.8 |
| spectral concentration | 0.904 | 0.701 / 0.715 | 0.8 / 0.8 |
| toeplitz | 0.651 | 0.161 / 0.140 | **0.2 / 0.2** |
| seam cyclic cv-R² | 0.413 | 0.004 / −0.012 | **~0.01** |

> **Pre-committed claim-B verdict: median C4a over the 14 non-void detector
> cells = 0.79 → DOES NOT GENERALISE (< 3).** Per the registration, claim B is
> confined to the statistics it was measured on, the paper says so, and the
> centrepiece becomes a per-statistic honesty table rather than a blanket
> claim.

What the table actually shows is stronger than the killed blanket claim:

1. **Fixture validation is uncalibrated in both directions.** The harmonic
   detector's fixture floor is understated ~150× (D37's finding, reproduced in
   the unified table; the C2 values agree with `preregistration_family2.md` §1
   to within draw noise). But the seam cyclic cv-R² floor is *overstated*
   ~100× (0.41 on fixtures vs ≈0.004 real), and toeplitz ~4× — fixtures that
   overstate a floor silently destroy power rather than shipping false
   positives. Neither direction is predictable in advance from the statistic's
   form; both are calibration-transfer failures.
2. **Second-moment matching closes most of the gap.** The regime-matched
   synthetic column (C1b, exploratory per D42) brings the harmonic ratio from
   ~150× to 2.0× — the D37 gap was mostly the fixtures' unmatched residual
   share (fixture noise ≈ 1% of signal variance; real [1,k]-residual is
   comparable to the linear component), not some irreducible mystery of real
   activations. The last ~2× is structure beyond second moments.
3. **Self-calibrating nulls transfer; fixture nulls don't.** The one statistic
   whose null is generated from the data itself (permutation z) has C4a = 1.1.
   Residual permutation on the real vectors — the D37 repair — is the same
   principle. The constructive recommendation writes itself: validate
   detectors against permutation/residual-permutation nulls on the actual
   vectors under study; fixtures certify the code, not the power.

Also in the table: months' observed harmonic partial R² (0.363 / 0.423) sits
above its real floor, as D37 found — and observed circulant-centered does too
(0.312 vs 0.248) **but so does add-k's** (0.448), a family with no cycle, so
that exceedance carries no cyclicity information — consistent with D30.

### Objections we raise against ourselves

**(a) The C1b objection — "this reduces to matching your null's moments."**
Second-moment matching takes the harmonic ratio from ~145× to 2.0×, so most of
the D37 catastrophe was a fixture *construction* defect — the committed
fixtures put ~1% of variance in the residual where the real family puts an
amount comparable to its linear component — and a reviewer can fairly say the
recommendation collapses to textbook advice: match your surrogate's moments.
The defense is in the C4b column itself. After matching, the ratios still span
≈0.0 to 2.0 across statistics (harmonic 2.0, spectral concentration 1.4–1.5,
permutation z 1.1, seam cut-margin 0.4–0.6, circulant raw 0.2–0.3, seam cyclic
cv-R² ≈0.02 — a residual ~50-fold *overstatement*), and the direction of the
remaining error is not predictable from the statistic's form. Matching narrows
the error; it does not tell you which way you are still wrong. The only
construction whose calibration does not depend on getting nuisance structure
right is the null derived from the data under study — which is the actual
recommendation, and it is cheaper to run than the fixtures it replaces.

**(b) The perm_z objection — "your best transfer row is VOID."** The row
supplying the "self-calibrating nulls transfer" evidence (C4a = 1.1) is the row
the validation gate voided, and left unlabelled that pairing would read as
cherry-picking. The two verdicts concern different objects. The VOID is about
*detector power against a synthetic plant*: the planted line reaches z = +1.03
against the real floor of 2.15, so the gate cannot certify that this plant
would be detected. C4a is a ratio between two *independently computed null
floors* — the fixture-grade floor (1.90, simplex draws) and the real
row-permutation floor (2.15) — neither of which involves the plant, so the
ratio remains readable, and it reads ≈1 because the statistic standardises
itself against its own permutation distribution in either regime. Detection on
real ordered data is separately established in the same table (add-k, a real
family with known ordering, z = +7.44). And the plant's failure is itself a
claim-B instance rather than an embarrassment to manage: the fixture cannot
produce the smooth ⟨X_k, X̄⟩ cross-term the statistic actually responds to on
real ordered families — the synthetic *positive control* fails for the same
reason synthetic *nulls* fail, namely that fixtures do not carry the structure
real activations do.

---

## 5. T1 — the 2×2 support matrix: Branch 1, claim A survives in strong form

Read last, as registered, against the D40 branches as committed. Full per-k
tables in `results/stage2/support_matrix.json`; details in D43.

**Both off-diagonal runs — restricted demonstrations, full-cycle queries — 
return GO at every k.** partA4 (4 operands): minimum cell k=10 at exactly 0.50
(Wilson [0.40, 0.60], straddles — flagged). halves_A (6 operands): minimum k=8
at 0.54, which is *above* the full pool's own 0.38 at the same cell. No Branch 2
trigger fires anywhere; the middle case never engages; the in-pool anchor
against the original bf16 gates passes at every k in both runs, so the fp16/MPS
instrument carries the verdict unflagged.

| | behavioural gate, matched support | FV task-encoding margin |
|---|---|---|
| 4 operands (Jan–Apr) | **GO** (min 0.50) | **−0.704** |
| 6 operands (Jan–Jun) | **GO** (min 0.54) | **−0.315** |

The out-of-pool query subset — the never-measured cell — is mostly 0.50–1.00,
with weak cells named (partA4: k=4 at 0.34, k=10 at 0.49; halves: k=8 at 0.26).
The registered verdict reads the per-k aggregate over the uniform full-cycle
distribution, as the field's own gate would; the split is published so a
stricter reader can apply an out-of-pool-only gate, which was not registered
and is not applied post hoc.

**The combined mechanism, which is the paper's claim A in final form:** the
model genuinely generalises restricted-demonstration shift-by-k to operands it
never saw demonstrated — the competence the accuracy gate certifies is real on
the very support the FV margin is scored over — while the vector extracted from
those same prompts steers to next-item *even on the demonstrated operands*
(§3). The failure lives in the extraction, not in the behaviour, and no
accuracy gate at any query support could detect it. The reviewer's
support-mismatch objection is closed by measurement, not argument.

---

## 6. What the paper now says

- **Claim A stands in strong form** (T1, both pool sizes, cross-run rule
  satisfied), with its mechanism sharpened by T3: accuracy-gating passes while
  the extracted vector encodes a different function, and the discrepancy is
  invisible in principle to behavioural gating because the behaviour is fine.
- **Claim B is rescoped by its own pre-committed reading** (T6, median C4a =
  0.79 → does not generalise): the ~100× understatement is real where D37
  found it, but fixture-grade validation mis-states real false-positive floors
  in *both directions* (0.01× to 156×), unpredictably per statistic. The
  constructive result: self-calibrating nulls (permutation, residual
  permutation on the vectors under study) transfer at ratio ≈ 1; fixtures
  certify code, not power.
- **T5 adds the third leg**: the field's importance measures — accuracy, causal
  AIE, injection efficacy, norm — are not interchangeable; the two most
  commonly composed (accuracy + steering efficacy) are statistically unrelated
  across tasks here (ρ ≈ 0.02, ceiling 0.93), and even two readouts of the
  same injections disagree (ρ = 0.12).
- **T10 supplies the audit** that keeps all of the above honest about the
  vectors themselves: the shared offset (79–95% of norm) systematically moves
  every non-invariant raw-Gram statistic beyond its split-half band, and
  `spectral_concentration` is structurally blind to the contamination that
  invalidates its own gate.

Scope, unchanged from the registration: one model, one family (plus add-k and
unrelated references), one prompt format; Hendel-method columns carry their
registered ±1-confinement limitation throughout; the torch-CPU test suite is
certifiable on the Ada box but not on this M1 (D44).
