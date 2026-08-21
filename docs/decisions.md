# Decision log

Decisions taken between the frozen pre-registration (`docs/preregistration.md`,
committed `fa55b8e`) and the runs that follow. **The pre-registration is never
modified.** Where a registered branch is ambiguous, the reading actually taken is
recorded here, along with what was known at the time.

Each entry states when it was fixed relative to the evidence it concerns, because
that is what makes it a commitment rather than a rationalisation.

---

## 2026-08-02 — Stage 1 extraction planning (months Z/12, 16 shots, Ada box)

Context: `docs/pilot_findings.md` §7 deferred the k=8 branch decision to the
extraction-planning session. These entries resolve it. All were fixed **before any
function vector was extracted and before any causal head sweep was run**.

### D1. The k=8 weak-cell branch fires on its literal reading

Pilot §7 records the ambiguity: prereg §4's branch condition is "improves over its
10-shot value but stays below 0.50". k=8 moved 0.34 → 0.38, which satisfies that as
written, but the change is not statistically distinguishable from zero (two-proportion
z = 0.42, p = 0.68). The neighbouring branch — "fails to improve at all" — would
escalate to NO-GO for months as primary.

**Decision: the literal reading fires.** Extract all 12 FVs including FV(8). Every
geometry diagnostic runs twice — full n=12 and leave-one-out n=11 excluding k=8 —
reported side by side in the same table, per prereg §4. A result is claimed only where
the two columns agree; divergence between them is itself the finding and is not
resolved by preferring one.

Per prereg §4, Z/11 is prime, so the §1 divisor test is **undefined** on the reduced
set: it is read off the full n=12 column with the weak cell flagged inline. The n=11
column is evaluated on circulant / concentration / PR / closure only.

Interpolation stays at k=1.5 and k=2.5 as registered, never adjacent to k=8.

Supporting the literal reading: 11/12 cells sit ≥0.70 at 16 shots, and the NO-GO
escalation branch was written for a family that is broadly weak, which this is not.
Against it, and recorded because it is not resolved by this decision: k=8 is the one
cell that did not respond to the intervention that lifted every other cell, which may
be a substantive fact about the geometry rather than a sampling issue.

### D2. Causal efficacy is the pre-declared arbiter for FV(8)

Declared before any efficacy score exists. Every FV(k), under both extraction methods,
receives a zero-shot injection lift score. This is the evidence that decides whether
FV(8) is a real task vector or an artefact of a cell the model cannot do — not the
geometry, which cannot adjudicate its own inputs.

**Registered caveat, recorded now so it cannot be invoked selectively later:** the
zero-shot baseline `Q: <month>\nA:` is dominated by a copy prior, so the identity task
is already at ceiling before any injection and FV(0) will show ≈0 lift *by
construction*. FV(0)'s efficacy is reported but excluded from the arbiter comparison.
Raw injected accuracy and lift are both reported for every k.

### D3. Causal head identification sweeps k ∈ {1, 2, 3, 6, 9, 11}

The head set is identified **once** and reused across all k (Todd et al. 2024; Yin &
Steinhardt, arXiv:2502.14010, on FV heads being task-general and distinct from
induction heads).

The six k in the sweep are those at ≥0.80 held-out accuracy at 16 shots. Two reasons:

1. The head set stays independent of the weak cells {4, 5, 7, 8}, so FV(8) — the cell
   D2 arbitrates — is a genuine out-of-sample test of a head set that never saw it.
   Identifying heads on k=8 and then asking whether FV(8) is real would be circular.
2. Half the GPU cost of the full sweep (~1.2 h vs ~2.5 h).

**k=0 is excluded despite 1.00 accuracy.** Identity is solvable by pure copying — the
pilot's copy stratum runs 0.90–1.00 everywhere — so an AIE sweep on k=0 would rank the
copy/induction circuit rather than the task heads. Yin & Steinhardt establish those as
a distinct head population; contaminating the head set with induction heads would
degrade every FV in the family. This is the sole reason for the exclusion.

### D4. Pre-registered escalation trigger for the head set

Fixed **before any efficacy score is seen**, so it cannot be tuned to the outcome.

At hard stop 2, regress per-k causal efficacy on that k's 16-shot held-out ICL
accuracy. If the out-of-sweep k {0, 4, 5, 7, 8, 10} sit systematically below the fit
established by the in-sweep k {1, 2, 3, 6, 9, 11} — i.e. their FVs are weaker than
their own task competence predicts — that is evidence the head set is biased toward
the strong subset. **Only then** pay for the all-12 sweep (~+1.3 h) and report
head-set overlap between the two.

If the out-of-sweep k sit on the fit, the single sweep stands and no further head
identification is run.

### D5. Mixed-domain control: a fallback ladder

CLAUDE.md rule 2 makes the mixed-domain operand control mandatory. Rule 5 forbids
extracting FVs from a task the model cannot do. Pilot §4 shows the mixed variant is
not performable at 10 shots (letters 0.35, digits 0.43 aggregate). 16 shots is untested.

The control's logic requires only that the demonstration operands **not share a single
operand circle** — it does not require all four domains. Hence a ladder, fixed before
any gate is run:

- **(a)** Pre-gate the full mixed variant — days/months/letters/digits demos, **month
  query** — at 16 shots. Clears 0.50 at every k → extract, done.
- **(b)** If (a) fails, pre-gate a **two-domain variant: days + months demos only**.
  Z/7 and Z/12 have different cycle lengths, so there is still no shared operand circle
  for the FV geometry to inherit, and the pilot shows both domains are individually
  performable. Clears → extract this as the mixed-domain control, with the substitution
  documented in the results.
- **(c)** Only if both fail: reported as blocked-by-gate, with the numbers.

**Caveat, recorded now:** with a month query, operand geometry can still leak through
the query token even under mixed demonstrations. This control is therefore read
**jointly with the operand-partition control**, never as sufficient on its own. Whatever
rung of the ladder is used, that joint reading is how it must be reported.

### D6. Hardware: this box is 8 GB VRAM, not 16 GB

`nvidia-smi` on the Ada box reports **8188 MiB** — it is the laptop RTX 2000 Ada, WDDM
driver model, not the 16 GB desktop card assumed by BRIEF §7 and the old CLAUDE.md
hardware note. CLAUDE.md has been corrected.

Consequences, recorded because they bind future work:

- Llama-3.2-3B at bf16 is 6.42 GB of weights against an 8.00 GB ceiling. It fits
  forward-only with ~0.9 GB of headroom, but **only** if every forward requests
  last-position logits (`logits_to_keep=1`); full-sequence logits at batch 24 are
  862 MB, and ~1.7 GB more if upcast to fp32 for `log_softmax`, which alone OOMs the
  card. This is an invariant of the extraction code, not a tuning parameter.
- **The quantized-7B robustness check in BRIEF §9 / CLAUDE.md rule 4 cannot run on
  this machine**, and neither can the lit-sweep's suggested Llama-3.1-8B fallback
  (`docs/lit_sweep_task_space_geometry.md`, final bullet) — 8B at bf16 is ~16 GB of
  weights before activations. Any 8B-scale or larger robustness check needs different
  hardware and is out of scope for every run on this box. It is deferred, not dropped.
- The bf16 rule itself is unaffected: no geometry-bearing run here goes below bf16.

### D7. Prompt-format robustness is deferred to stage 3, not dropped

BRIEF §9 lists prompt-format leakage as a known risk and prescribes keeping the format
fixed across k, then varying it as a robustness check.

**Stage 1 does the first half only.** The `Q: {operand}\nA: {target}` format from
`tarcle/prompts.py` is held fixed across every k, every extraction method, and every
one of the seven extraction conditions — so no format difference can be confounded
with any geometry result produced from these artifacts.

**A format-varied replication is a stage-3 item**, to be run after the geometry is
known, and is explicitly not evidence this stage provides. Any stage-1 or stage-2
claim about task geometry is conditional on this single prompt format, and must be
reported that way.

---

## 2026-08-03 — After causal head identification, before any FV is extracted

Both entries are recorded **after** seeing the head set and its AIE profile, and
**before** any function vector exists. They are observations and reporting
obligations, not thresholds: nothing here changes what gets extracted.

Head set (run `months_llama32_3b_ada`): L14H1, L12H20, L14H2, L14H12, L12H3,
L26H22, L14H16, L20H11, L14H14, L18H2.

### D8. L14 H1 dominates the head set — geometry must be reported decomposed

L14 H1's AIE is +0.1082 ± 0.0040, **6.5× the runner-up** (L12H20 at +0.0166), and it
carries ~56% of the head set's summed AIE on average across the swept k (32% at k=6,
80% at k=2). A Todd-style FV is the sum of the set's contributions, so **FV_todd(k) is
substantially the mean activation of a single head.**

This is not treated as a defect — Todd et al. also find a small number of dominant
heads, and the head is task-general rather than an induction head (checked before this
entry: its AIE is positive at every swept k, +0.045 to +0.148, only 1.49× its own mean
at k=1, and it holds at k=11 ≡ −1).

**Reporting obligation, binding on stage 2:** every geometry result from the Todd
extraction is reported as a three-way head-subset decomposition —

1. whole set (all 10 heads),
2. L14 H1 alone,
3. the other nine.

A circulant/spectral result that holds for (1) and (2) but not (3) is a finding about
one head, not about task space, and must be stated that way. `head_contrib` (n, H, d)
is stored in every FV `.npz` precisely so this costs no GPU time and no re-extraction.

Also recorded: **L18 H2 (rank 10) is unstable across k** — it flips sign, +0.0119 at
k=1 to −0.0112 at k=6, CV 3.78 against 0.32–0.61 for the rest of the set. It is
**kept**, because `top_heads: 10` was fixed in the config before any AIE was observed
and trimming the set after seeing the numbers is the post-hoc tuning pre-registration
exists to prevent. Its exclusion is available as a free stage-2 robustness check via
`head_contrib`.

### D9. A pre-stage-2 curiosity: L14 H1 is weakest exactly where behaviour is strongest

L14 H1's per-k AIE is at its **minimum at k=6** (+0.0454, against +0.148 at k=1 and a
mean of +0.099). But k=6 is behaviourally the *easiest* mid-cycle shift in the whole
pilot (0.98 held-out at 16 shots, sitting above both neighbours), and it is the
order-2 self-inverse element of Z/12 — the unique shift where forward and backward
coincide and the sign-flip error mode is definitionally unavailable
(`docs/pilot_findings.md` §3).

So the shift the model does best is the shift its dominant FV head contributes least
to. Two readings, neither adopted:

- k=6 is solved by a mechanism that does not route through this head — plausible given
  its special status as the antipode/self-inverse element, and consistent with the
  antipode acting as an attractor for k=8's errors (pilot §7).
- The AIE metric saturates: patching helps least where the corrupted-prompt baseline
  is already most recoverable. **Checked before logging this entry**, since it is the
  boring explanation and should be ruled out first. The per-k corrupted baselines run
  0.048–0.070, with k=6 at 0.0699 the *highest* of the six, and
  corr(baseline, L14H1 AIE) = **−0.38** over the 6 swept k. The sign is what saturation
  predicts, so the effect is not zero — but the baselines' relative spread is 0.36
  against the AIE's 1.03, so saturation can account for at most a fraction of a
  variation three times larger than itself, and n=6 makes −0.38 indistinguishable from
  noise besides. Saturation is therefore not dismissed, merely insufficient; both
  readings stay live and any claim from the first must cite this number.

**Logged now, before any Gram matrix exists, so it cannot be retro-fitted.** The
prereg §1 divisor prediction holds that power should sit at frequencies with
gcd(f,12) > 1, and f=6 is the order-2 character. If the DFT spectrum later shows
anomalous power at f=6, this AIE observation is *prior, independent* evidence that
k=6 is mechanistically special — and if the spectrum shows nothing at f=6, that
disagreement is itself reportable. Either way the observation is fixed here rather
than recalled selectively after the spectrum is seen.

### D10. The canonical head-ID run is the one made under committed code

The first sweep was executed against a dirty working tree, so its recorded
`git_commit` names a commit that does not contain `extract.py`. Rather than patch the
metadata, the sweep is **re-run under the committed code** and that run is canonical.

The first run is archived untouched at `results/fv/months_llama32_3b_ada_precommit/`.
Top-10 overlap and AIE rank correlation between the two sweeps are reported as a free
reproducibility check on the one instrument every downstream artifact shares — the two
runs differ only in that the second is provenance-clean, so any disagreement is
measurement noise in the AIE estimate and bounds how much confidence the head ranking
can carry.

### D11. The prereg's §3 Test-1 rationale is wrong; the test is implemented so it has content

`docs/preregistration.md` §3 Test 1 motivates the token-frequency artefact control
with: *"target = operand + k, so the target distribution is a shifted copy of the
operand distribution and its mean proxy varies with k even under uniform operand
sampling."*

**The final clause is false.** Over the full Z/12 cycle a shift is a bijection on the
operand set, so the mean frequency proxy of the targets is *identical* for every k.
Implemented literally against the idealised cycle, Test 1 would compare a constant
against `norm_profile(X)` and return a meaningless correlation — it would pass
trivially, for the wrong reason, and the blocking control would be vacuous.

The registration is frozen and is not edited. The test is implemented so that it does
what §3 evidently intends — detect norms tracking token frequency — by computing the
proxy over the operands and targets **actually drawn** in each k's prompt set rather
than over the idealised cycle. That has genuine variation with k for two reasons:

- finite prompt sets (100 prompts × 16 shots) do not sample operands exactly
  uniformly, so realised means differ across k;
- under a restricted operand pool — the partition controls and the polysemy
  leave-out — shifting genuinely moves mass onto different target tokens, which is
  where the confound would actually bite.

Both the operand-side and target-side means are stored per k in every FV `.npz`
(`freq_proxy_operand`, `freq_proxy_target`) so stage 2 can run the registered
correlation against `norm_profile(X)` either way and report which was used. The pass
and fail thresholds in §3 are untouched.

---

## 2026-08-03 — After the first primary extraction, before the control matrix

### D12. Injection scale is a swept hyperparameter, not a constant — and this was changed after seeing a null

**Recorded as a post-hoc protocol change, because that is what it is.**

The first primary extraction fixed `injection_scale = 1.0` and produced a near-total
efficacy null: under the Todd extraction, lift was +1.00 at k=1, +0.92 at k=2, +1.00
at k=11, **+0.08 at k=3**, and exactly 0.00 at every other k — including k=6, whose
held-out ICL accuracy is 0.98.

Per CLAUDE.md's style rule the extraction code was suspected first. It was cleared:
split-half reliability is 0.980–0.9998, the head→residual projection is verified
against the attention block's real output in `tests/test_extract.py`, and injection
demonstrably moves predictions. So the two remaining *protocol* suspects were swept
directly (`tarcle/diagnose_efficacy.py`):

- **Layer**: not the problem. At scale 1.0 no layer in 0–27 rescues k=3, 6 or 8; the
  frozen L12 was already each k's best or tied-best.
- **Scale**: decisive. At ×2.0 the Todd FV takes **k=3 from 0.08 to 1.00**, k=6 from
  0.00 to 0.25, k=8 from 0.00 to 0.17.

So the scale-1.0 null was substantially an artefact of injection strength, not a
property of the vectors. Reporting it as a finding would have been wrong.

**Change:** injection layer and scale are now swept **jointly** over
`injection_scales = [0.5, 1, 2, 3, 4]` × 28 layers, chosen on the head-ID k subset
{1,2,3,6,9,11} only, and frozen for every k and every condition thereafter.

Why this does not launder a post-hoc choice into the out-of-sample cells:

- The grid is optimised on the **in-sweep k only** — the same six k that already
  chose the head set and the layer. k ∈ {0,4,5,7,8,10}, including the k=8 cell the
  D2 arbiter turns on, remain untouched by the tuning.
- One (layer, scale) pair is frozen across all k. Tuning per k would fit the protocol
  to each task and destroy cross-k comparability, which is exactly what D2 needs.
- The scale grid was fixed before re-running, not widened until k=8 responded. Note
  that k=8 does **not** clear at any scale (max 0.17), so the change does not rescue
  the cell whose status is under test — it rescues k=3, an in-sweep cell.

**Hendel is excluded from the scale sweep and stays at 1.0.** That method *replaces*
the residual stream rather than adding to it, so a scale ≠ 1 substitutes a state of
deliberately wrong magnitude — not a stronger push but a different and invalid state.
The measured behaviour agrees: Hendel efficacy degrades monotonically away from 1.0
(k=2: 0.75 → 0.25 at ×2, → 0.08 at ×4). This is a real asymmetry between the two
extractions and is reported, not harmonised away (CLAUDE.md rule 1).

The scale-1.0 run is archived untouched at
`results/fv/months_llama32_3b_ada/` alongside `efficacy_diagnosis.json`, which holds
the full scale × layer grids that motivated the change. The re-run under the joint
sweep is the canonical primary extraction.

#### Exactly how ×3.0 was selected

- **Swept:** the full outer product of 28 layers (0–27) × 5 scales
  {0.5, 1.0, 2.0, 3.0, 4.0}, i.e. 140 candidate protocols.
- **On which k:** the head-ID subset {1, 2, 3, 6, 9, 11} only — the same six k that
  chose the head set (D3). k ∈ {0, 4, 5, 7, 8, 10} played no part.
- **Optimising:** the unweighted mean, over those six k, of injected forced-choice
  accuracy on the complete 12-month query cycle. No other objective was computed, and
  the grid was not revisited after the out-of-sweep cells were seen.
- **Result:** L8 × 3.0, mean 0.806. Runners-up: L12 × 2.0 (0.764), L7 × 3.0 (0.764),
  L9 × 3.0 (0.736). Note the previously frozen L12 × 1.0 scored ~0.50.
- **Frozen** for every k and every condition thereafter, recorded in each `.npz` as
  `injection_layer` / `injection_scale`.

#### Stability of the D2 arbiter verdict (`scale_stability.json`)

FV(8) at the frozen L8, by scale: 0.00 at ×0.25/×0.5/×1.0/×2.0, **0.42 at ×3.0**,
0.25 at ×4.0, 0.17 at ×6.0. So the verdict does depend on scale, and that must be
stated plainly. Three things qualify it:

1. **The ×3.0 threshold is family-wide, not a k=8 spike.** Every mid-cycle k — 4, 5,
   6, 7, 8, 9 — is exactly 0.00 at ×2.0 and jumps at ×3.0. Only the easy k (1, 2, 3,
   11) respond at ×2.0. The pattern is a difficulty-graded threshold on injection
   strength, not a resonance at one cell.
2. **k=8 ranks mid-to-high among the mid-cycle k at every scale where anything
   responds** — 3rd of 6 at ×3.0 (behind k=9 and k=5), tied 2nd at ×4.0, tied 1st at
   ×6.0. Its verdict never depends on being the single cell that happened to fire.
3. **The frozen protocol is already k=8's global optimum.** Searching all 28 × 7
   layer × scale combinations, k=8's best achievable accuracy is 0.42, attained at
   exactly L8 × 3.0. The figure is an upper bound, not a favourable draw. By contrast
   the frozen choice *costs* k=4 (0.17 frozen vs 0.50 at L10 × 3.0) and k=5
   (0.50 vs 0.58 at L12 × 2.0) — per-k tuning would raise other cells, not k=8, and
   is refused anyway because it would destroy cross-k comparability.

#### Hendel's ±1 confinement is a property of the method, not of scale 1.0

Checked against the same full 28 layers × 7 scales grid, taking each k's best
achievable accuracy anywhere on it:

| k | 1 | 2 | 11 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| best over grid | 1.00 | 0.75 | 0.92 | 0.17 | 0.08 | 0.08 | 0.17 | 0.08 | 0.08 | 0.08 | 0.33 |

Chance is 1/12 = 0.083. Five mid-cycle cells (4, 5, 7, 8, 9) sit **at chance even at
their own optimal layer and scale**. No choice of injection strength rescues them, so
the confinement to ±1 and +2 is a fact about the Hendel dummy-query state, not about
the value it was pinned to. This is the sharpest cross-method disagreement in the run
and is reported as a finding about FV extraction (CLAUDE.md rule 1), not resolved.

### D13. The D4 escalation trigger is uninformative on this data, and that is not a pass

D4 pre-registered a test: regress per-k efficacy lift on that k's ICL accuracy using
the in-sweep k, and fire if the out-of-sweep k sit systematically below the fit.

Computed on the scale-1.0 run it **did not fire** under either extraction. That verdict
should not be reported as reassurance, because the fit is degenerate:

- The in-sweep k have almost no variance in the regressor — ICL accuracies 1.00, 1.00,
  1.00, 0.98, 0.80, 1.00. The slope is determined almost entirely by k=9.
- The out-of-sweep k extend to ICL 0.38, so the test **extrapolates** a fit built on
  x ∈ [0.80, 1.00] down to x = 0.38.
- At that extrapolation the fitted line predicts a lift of −1.48 for k=8, which is not
  merely wrong but impossible: lift is bounded below by −baseline = 0. k=8's residual
  of +1.48 — the largest of any cell, and the one most responsible for the
  "does not fire" verdict — is therefore a pure artefact of the extrapolation.

The weakness is in the trigger's construction, fixed in D4 before any efficacy score
existed, and it is recorded here rather than silently reinterpreted. **The trigger is
reported with its residual spread and this caveat attached, and no confidence is
placed in either verdict.** The substantive question it was meant to answer — whether
the head set is biased toward the strong k — is left to the direct comparison it
should have used: head-set overlap between the six-k and all-twelve-k sweeps, which
costs ~1.3 GPU-hours and remains available.

### D14. The all-k head sweep replaces D4's regression, which is uninformative by construction

D4 pre-registered a regression test for head-set bias: fit efficacy lift against ICL
accuracy on the in-sweep k, fire if the out-of-sweep k sit below the fit. **That test
cannot answer the question it was written for, and this is a defect of construction,
not of the data** — it was knowable when D4 was written and was not noticed:

- The regressor has almost no variance on the in-sweep k (ICL 1.00, 1.00, 1.00, 0.98,
  0.80, 1.00), so the slope rests on k=9 alone.
- The out-of-sweep k reach ICL 0.38, so the test extrapolates far outside its fitted
  range, and the fitted line predicts lifts below the hard floor of 0.
- It is an *indirect* proxy: it infers head-set bias from an efficacy pattern that
  depends on the injection protocol, the FV assembly and the task difficulty all at
  once.

**Replacement, run directly:** identify the head set a second time over
k = 1..11 and compare it to the canonical six-k set — top-10 membership overlap, AIE
rank correlation, and whether L14 H1 retains its dominance. That measures head-set
bias with no proxy and no extrapolation. Run `months_llama32_3b_ada_allk`.

**k=0 is excluded from the all-k sweep too.** The check varies the one thing under
test — the breadth of the k subset — and holds everything else fixed. Adding k=0 would
introduce the induction/copy-head contamination D3 exists to prevent, so a change in
the head set could not be attributed to breadth rather than to that contamination. The
sweep is therefore k = 1..11 (eleven values, every non-identity shift), not twelve.

D4's regression continues to be computed and reported, with its residual spread and
this caveat attached, but no inference is drawn from either verdict.

### D15. AIE rank and FV contribution are decoupled — head-set overlap is the wrong robustness metric

Found while running the D14 check. Recorded because it changes how head-set
robustness must be reported, and because it qualifies D8.

The all-k sweep returns 8/10 membership overlap with the canonical six-k set, which
reads as reassuring. It is not, because **a Todd-style FV is a raw sum of mean head
activations, not an AIE-weighted sum.** A head can be causally negligible and still
supply a large share of the vector. Measured on the canonical extraction:

- The two heads the all-k sweep drops (L14H14, L18H2) carry ~3% of the head set's AIE
  each, but **28% of the FV norm on average** — 0.12–0.15 at k ∈ {0,1,11}, rising to
  0.32–0.38 at k ∈ {4..9}.
- Removing them rotates the FV by cos 0.926–0.992, minimum at k=6. **The split-half
  noise ceiling is 0.9946**, so every mid-cycle k moves by more than measurement noise.
- The disturbance is largest precisely at the mid-cycle k where the geometry question
  is most delicate, and smallest at the ±1 cells where it is least interesting.

**This qualifies D8.** L14 H1 carries ~56% of the head set's summed AIE but only
**32.6%** of the FV norm. "Dominant" is true causally and false geometrically, and the
D8 decomposition obligation (whole set / L14H1 / other nine) must be reported in norm
terms as well as AIE terms so the two are not conflated.

**Consequence for reporting:** head-set robustness is not established by membership
overlap. Any claim that a geometry result survives the choice of head set must be
supported by re-running the battery under the alternative set, not by citing 8/10.
The canonical six-k set remains primary (D3, fixed before any AIE existed); the all-k
set is carried as a parallel condition.

### D16. Agreement criterion for the condition matrix — fixed before any Gram matrix exists

**Status: written and committed before a single Gram matrix of extracted FVs has been
computed.** The only inputs are the head-set and efficacy results already recorded
above, none of which involve geometry.

The matrix has grown: 2 extraction methods × (3 head sets for Todd, 1 for Hendel) ×
{full n=12, leave-one-out n=11} × 5 operand conditions, plus the two structural
controls. At that many cells, "a result is claimed only where the columns agree"
(D1) and "divergence is itself the finding" become unfalsifiable unless *agree* is
defined numerically in advance. This entry defines it.

#### 1. The headline verdict is one designated cell per extraction method

Confirmatory, and nothing else is:

> **Primary operand condition, full n=12, canonical six-k head set, read off
> `spectral_concentration` and `participation_ratio`** — the two separators the
> pre-registration §0.2 identifies as the A-vs-D discriminators — against the
> prereg §2 thresholds.

Reported once for Todd and once for Hendel. Every other cell in the matrix is
**exploratory**: reported with numbers, never with verdict language, and never used
to upgrade or downgrade the headline.

The canonical six-k head set is primary because D3 fixed it before any AIE existed.
The all-k and intersection-8 sets are robustness conditions, not competitors.

#### 2. What counts as "the same verdict" across head sets

Two requirements, both necessary:

- **(a) Same bucket.** The prereg §2 thresholds assign a label (A / A-multi / B / D /
  Null). The label must be identical across head sets.
- **(b) Difference within the noise band.** For each separator, the cross-head-set
  difference must not exceed that diagnostic's own **split-half band**, defined as
  |diagnostic(X_half_a) − diagnostic(X_half_b)| computed within the canonical
  condition. `X_half_a` / `X_half_b` are already stored in every `.npz` for exactly
  this purpose.

The split-half band is the reference rather than an invented tolerance because it is
measured, not chosen, and it is the same quantity stage 2 must already use as the
ceiling for reading off-diagonal Gram entries. A cross-head-set difference smaller
than the difference between two halves of the *same* extraction is not evidence of
anything.

Outcomes, all three reportable:

| (a) bucket | (b) within band | reported as |
|---|---|---|
| same | yes | **agreement** — the verdict is claimed |
| same | no | **same label, unstable magnitude** — verdict claimed, instability reported inline |
| differs | either | **divergence** — no verdict claimed under Todd; see §3 |

#### 3. Hendel arbitrates a Todd split, and cannot be overruled by it

The Hendel extraction is **head-set-free** — a dummy-query hidden state involves no
causal head selection at all. So if Todd's verdict diverges across head sets, the
Todd result is reported as head-set-dependent with no verdict claimed, and the Hendel
verdict stands as the single-method result, flagged method-specific per CLAUDE.md
rule 1.

Registered limitation, so it is not discovered later as a convenience: Hendel's FVs
steer only at k ∈ {1, 2, 11} even at their own optimal layer and scale, with five
mid-cycle cells at chance (D12). Its geometry is therefore being read off vectors
whose causal status is established at only three of twelve k. **That weakness is
recorded now and must accompany any Hendel arbitration**; it does not disqualify
Hendel from arbitrating, because head-set-freedom is the property being relied on,
but the two facts are reported together or not at all.

#### 4. The intersection-8 condition is what isolates the dropped heads

The all-k set differs from the canonical set by two removals *and* two additions, so
a difference under it is unattributable. Intersection-8 (canonical minus L14H14 and
L18H2) changes only the removals. Attribution rule fixed here:

- canonical vs intersection-8 differ → the effect is the **dropped** heads;
- intersection-8 vs all-k differ → the effect is the **added** heads;
- canonical vs all-k differ while both pairwise comparisons agree → the effect is
  joint and is reported as unresolved.

#### 5. Gating, unchanged from the pre-registration

The prereg §5 discard conditions still bind and are evaluated before any of the
above: a positive circulant result on the n-unrelated-tasks control voids the run
outright, prompt SHA-256 must reproduce, and no geometry-bearing run may sit below
bf16. §3's month-frequency control remains blocking.

### D17. The D2 arbiter metric changes from accuracy lift to logp — recorded as a post-hoc change

**Recorded as a change made after seeing data, like D12.** D2 declared causal efficacy
the arbiter for FV(8) and was written in terms of forced-choice accuracy lift. This
entry replaces the metric, not the arbiter.

**Why.** The query space is exhaustively 12 prompts (one per operand), so accuracy
quantises to 1/12 = 0.083 — which is also chance. A cell cannot be distinguished from
chance at any resolution finer than one query. That produced a concrete error: FV(8)
under the all-k head set was reported as "0.08, exactly chance", when its
mean log P(correct | 12 candidates) is **−2.332 against a chance of −2.485** — above
chance by +0.15 nats, not at it. The vector was still raising P(correct); it had
merely stopped winning the argmax.

**Replacement metric.** `efficacy_logp` = mean over the 12 queries of
log P(correct | candidate set), with chance fixed at log(1/12) = −2.485, plus
`efficacy_logp_lift` against the no-injection baseline on the same queries. Accuracy
and `margin` continue to be reported; they are no longer the arbiter.

**Why this is not metric-shopping.** The change was forced by a ceiling in the old
metric's resolution, not by an unwelcome verdict, and it moves the k=8 verdict in
*both* directions depending on head set — it rescues all-k from "at chance" while
leaving canonical's advantage intact (canonical +0.53 nats over chance vs all-k
+0.15, a 3.4× ratio). Every ordering established under accuracy survives: canonical >
int-8 ≈ all-k at mid-cycle, and the D16 §4 attribution to the dropped heads.

**Registered requirement, not yet satisfied.** +0.15 nats over 12 queries carries no
verdict without an error bar. The per-query log-probabilities were **not persisted**
in the runs completed before this entry — only their mean — so no standard error can
be computed for `months_llama32_3b_ada_hs_{canon,int8,allk}` without re-scoring.
`efficacy_logp_per_query` (n_k, n_queries) is added to the schema from this point on,
and **no logp-based verdict may be claimed for a cell whose SE is unavailable.** The
three head-set conditions are re-scored, or their logp differences are reported as
point estimates explicitly marked as carrying no verdict.

### D18. Hypothesis-C test criterion, in continuous terms — fixed before the operand-partition control runs

Prereg §2-C defines the C test qualitatively: extract FV(k) from demonstrations whose
operands are restricted to one partition, apply to queries from a disjoint partition,
and ask whether steering efficacy drops on transfer. It does not fix a metric. This
entry fixes one, **before the control is run**.

**An accuracy-read C test would be uninterpretable**, and the run already shows why:
mean `margin` is negative at every mid-cycle k under every head set, so the argmax at
those cells is decided by a minority of queries clearing a boundary the average sits
below. Differencing two such quantised, boundary-straddling numbers measures rounding.

**Metric.** Per k, compare `logp_lift` on matched-partition queries against
`logp_lift` on transferred queries:

    Delta(k) = logp_lift(FV from A, queries from A) - logp_lift(FV from A, queries from B)

averaged over both partition directions (A→B and B→A) to cancel any intrinsic
difficulty difference between the two operand halves.

**Lift, not raw logp**, because the two query sets are different operands with
different baseline probabilities; subtracting each set's own no-injection baseline is
what makes the two sides comparable.

**Noise band.** Extract FV(k) from two disjoint halves of the *same* partition's
prompts and compute the same Delta between them. That is the value Delta takes when
nothing has changed but the prompt draw, and it is measured rather than assumed —
the same principle D16 applies to the geometry diagnostics.

**Criteria:**

- **C true (transfer degrades):** mean Delta over k exceeds the split-half band, and
  does so in the same direction at a majority of k.
- **C false:** mean Delta within the split-half band.
- **Ambiguous:** exceeds the band at fewer than half the k, reported as such with no
  verdict.

**Second, independent signature, reported regardless of the above.** C predicts not
merely degradation but a *systematic wrong-region push* — a rotation applied at the
wrong starting point lands somewhere predictable. For transferred queries, record the
distribution of the signed prediction shift `(argmax_index − query_index) mod 12`. A
push concentrated at a shift ≠ k is evidence for C even if Delta is within band;
uniform degradation is not. This is the discriminating signature, since magnitude
loss alone is also consistent with plain distribution shift.

**Head sets.** The C test runs under the canonical head set. If the verdict lands
within one split-half band of its threshold, the discriminating cells are re-run
under all-k and intersection-8 before anything is claimed (minutes, per D15/D16).

### D19. The pre-registration's example partitions are too small to carry the C test

The prereg §2-C names *"e.g. {Jan..Apr}"* and *"{Sep..Dec}"* as the operand partitions
for the hypothesis-C transfer test. Those were followed literally. **They do not work,
and the reason is structural rather than statistical.**

A 4-month operand pool under the held-out stratum leaves the demonstrations drawing
from **3 distinct operands** — measured, not estimated: distinct demo operands per
16-shot prompt is min 3, mean 3.0, max 3. Against 7.1 for the 9-month polysemy
condition and 8.6 for the full 12-month primary. A typical prompt repeats
`April → December` three times in its first four demonstrations.

Consequence, measured on the matched (non-transferred) condition, where the FV is
being applied to exactly the operands it was extracted from:

| condition | P(correct) at mid-cycle k |
|---|---|
| primary, 12-operand pool | 0.36 |
| partition, 4-operand pool, **matched** | **0.06** |

The partition FVs barely encode shift-by-k *before any transfer happens*. What they
encode instead is next-item: P(prediction lands on shift ±1) is **0.90 matched** and
0.93 transferred, over k ∉ {0,1,11} where ±1 is wrong by construction.

#### What survives and what does not

- **Survives — the D18 Δlogp criterion fires, and cleanly.** Mean Δ = +0.516 nats
  against a mean split-half band of 0.101, exceeding at **12/12 k**, not near
  threshold. Matched and transferred are computed from the same degenerate prompts,
  so the comparison is internally fair and transfer genuinely degrades.
- **Does not survive — the wrong-region signature.** D18 registered it as the
  *discriminating* evidence, on the grounds that magnitude loss alone is also
  consistent with plain distribution shift. It does not discriminate here: the ±1
  collapse is present in the matched condition too (0.90 vs 0.93). Recording the
  matched distribution as a baseline is what revealed this; D18 as written only
  required the transferred one, which would have licensed a wrong-region claim that
  the data does not support.

#### Consequence

A C verdict is **not claimed** from this run. "A function vector that has already
collapsed to next-item degrades further off-distribution" is far weaker than the
prediction prereg §2-C was written to test, which presumes an FV that encodes the
task in the matched condition.

**Remedy, for the user's decision:** re-run the partitions as disjoint halves,
{Jan..Jun} / {Jul..Dec}, giving 5 distinct demo operands instead of 3. This was
considered at planning time and rejected in favour of the registration's literal
example; that choice is now falsified by data. The cost is that contiguous halves are
adjacent on the cycle, so shift-by-k can map partition A into partition B — a milder
confound than measuring an FV that does not represent its task. The registration is
not edited; this entry records that its §2-C example is unusable and why.

The polysemy leave-out condition (9 operands, 7.1 distinct) is **not** affected and
its artifacts stand.

### D20. Task-encoding gate, mandatory matched baselines, and rule 5 per condition

Three structural changes, all forced by D19, all fixed before the antipodal-halves
re-run is extracted.

#### 1. The task-encoding gate — blocking, not a caveat

The C test presumes a function vector that encodes its task. D19 shows that
presumption can fail silently. It is now measured.

**Gate.** On the **matched** condition, over k ∉ {0, 1, 11} (where ±1 is the correct
answer and collapse is indistinguishable from success):

    P(prediction lands on the correct shift) - P(prediction lands on shift +/-1)  >  0.10

**Failing the gate blocks the condition's verdict entirely** — the D5(c) pattern, not
a caveated report. A blocked condition is reported as *blocked-by-gate* with its
numbers, and no transfer, geometry or hypothesis claim may rest on it.

Measured on existing artifacts (`tarcle/nextitem.py`, free from stored
`efficacy_pred_shift`):

| condition | operand pool | P(correct) | P(±1) | margin | gate |
|---|---|---|---|---|---|
| primary | 12 | 0.546 | 0.194 | **+0.352** | pass |
| polysemy leave-out | 9 | 0.546 | 0.204 | **+0.343** | pass |
| partition A | 4 | 0.083 | 0.787 | **−0.704** | **fail** |
| partition B | 4 | 0.009 | 0.954 | **−0.944** | **fail** |

Two things this establishes. The polysemy control's artifacts stand, and that is now
a measurement rather than an inference from pool size — the margin is within 0.01 of
the primary's. And **the collapse is a cliff, not a gradient**: 12 and 9 operands both
pass comfortably, 4 fails catastrophically, with nothing in between yet sampled. The
**6-operand antipodal halves are therefore untested**, may collapse, and are gated
like everything else. Pool size is an input to the mechanism, never evidence about it.

#### 2. Matched-condition baselines are mandatory for every remaining control

D18 registered the wrong-region prediction but required only the *transferred*
prediction-shift distribution. The ±1 push it would have reported as evidence for C
turned out to be present in the matched condition at 0.90 against 0.93 — i.e. not a
transfer effect at all. The criterion nearly shipped an artifact as a finding.

**Structural fix:** every remaining control's criterion must name its matched or
unperturbed baseline **in the entry that fixes the criterion, before the control
runs**. A criterion that compares a treated quantity against a threshold rather than
against its own baseline is not accepted. This applies to the mixed-domain,
unrelated-tasks and ordinal controls, and to any re-run of the partition control.

#### 3. Rule 5 applies per condition, not per family

CLAUDE.md rule 5 gates FV extraction on measured ICL accuracy for the (model, family)
pair. D19 shows that too coarse: a restricted operand pool is **a different effective
task**, and months-with-4-operands is not the family the pilot gated. The 16-shot
months pilot (11/12 cells ≥ 0.70) licensed an extraction whose FVs encode next-item.

**From here on the behavioural pre-gate runs per condition** — per operand pool, per
query-domain restriction, per family — and never inherits another condition's gate.
This applies to the antipodal-halves re-run and to the three outstanding controls,
whose gates were already planned and are now required rather than advisable.

### D21. The operand-partition control is not runnable on months Z/12 — blocked-by-gate

The antipodal-halves re-run was approved as a repair for D19. **It fails the D20
task-encoding gate too**, and the failure is structural rather than a matter of
choosing better partitions.

| condition | operand pool | P(correct) | P(±1) | margin | D20 gate |
|---|---|---|---|---|---|
| primary | 12 | 0.546 | 0.194 | +0.352 | pass |
| polysemy leave-out | 9 | 0.546 | 0.204 | +0.343 | pass |
| **half A (Jan–Jun)** | **6** | 0.231 | 0.546 | **−0.315** | **fail** |
| **half B (Jul–Dec)** | **6** | 0.139 | 0.602 | **−0.463** | **fail** |
| partition A (Jan–Apr) | 4 | 0.083 | 0.787 | −0.704 | fail |
| partition B (Sep–Dec) | 4 | 0.009 | 0.954 | −0.944 | fail |

The margin is monotone in pool size and the threshold lies between **6 and 9**.

**The structural problem:** the control requires two *disjoint* operand partitions.
On a 12-element cycle the largest disjoint pair is 6 + 6. Six collapses. So there is
no partition of Z/12 that is simultaneously disjoint and large enough for the
extracted FV to encode its task. The hypothesis-C test as specified in BRIEF §6
Control 1 and prereg §2-C **cannot be run on this family with this model.**

Per D20 §1 this is reported as **blocked-by-gate**, not as a caveated C verdict. The
D18 Δlogp criterion did fire on the 4-operand run (mean Δ +0.516 against band 0.101,
12/12 k) and that number is retained in the artifacts, but **no hypothesis-C claim
rests on it**, because the FVs being compared do not encode shift-by-k on either side
of the comparison.

**Behavioural gating is uninformative here, and that is itself the finding.** All
three restricted pools return **GO** on the rule-5 behavioural pre-gate — the
4-operand partition included, at every k ≥ 0.50, and the 6-operand halves at every
k ≥ 0.83 with k=8 at 0.87 against 0.38 in the full pool. Restricting the operand pool
makes the *in-context* task easier while making the *extracted function vector*
worse. Plain ICL accuracy carries no information about whether an FV extracted from
those prompts encodes the task, which is exactly why D20 §1 exists as a second,
independent gate.

#### Routes that remain open, none adopted here

- **A larger cycle.** ROT-k on Z/26 admits two disjoint 13-operand partitions, both
  above the threshold. Deferred previously because letters are the weakest domain
  (`docs/pilot_findings.md` §4, 0.35 aggregate) and ROT-13's corpus frequency gives
  the family a privileged k. Both objections stand and the pilot logic would have to
  be re-run for the pair.
- **Partition by domain rather than by cycle position.** Extract on days, apply to
  months: disjoint operands, no shared cycle, and each domain keeps its full pool.
  This changes what the control means — it tests transfer across *domains* rather
  than across regions of one operand circle — and the mixed-domain control's gate
  will bear on whether it is viable at all.
- **Overlapping partitions**, trading the disjointness the control depends on for
  pool size. Weakest of the three; recorded for completeness, not recommended.

The choice among these is the user's and is not made here.

### D22. Mixed-domain ladder, rung (a): four-domain demonstrations fail the behavioural gate

D5 rung (a) — demonstrations spanning days/months/letters/digits, **query pinned to
months** so Z/12 semantics hold, 16 shots, held-out stratum, n=100 per k.

**Verdict: NO-GO.** Held-out accuracy by k:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | 1.00 | 0.99 | 0.99 | 0.87 | 0.53 | 0.37 | 0.84 | **0.25** | **0.27** | 0.34 | 0.33 | 0.55 |

Five cells below the 0.50 GO threshold, two at or below the 0.30 NO-GO floor
(k=7 at 0.25, k=8 at 0.27). This confirms at 16 shots what
`docs/pilot_findings.md` §4 found at 10 — the four-domain set is not performable —
and settles the question §4 left open ("whether 16 shots rescues it is untested"). It
does not.

Pinning the query to months does **not** rescue it either, which is informative: the
difficulty is in the *demonstrations*, not in the query domain. Letters and digits in
the demo set degrade the model's ability to infer shift-by-k even when every scored
query is a month.

Per D5 the ladder descends to rung (b): **days + months demonstrations only**. Z/7 and
Z/12 have different cycle lengths, so there is still no single operand circle for the
geometry to inherit — which is all BRIEF §6 Control 2 requires — and both domains are
individually performable. `prompts.DOMAIN_GROUPS` adds the variant; existing variants
are untouched and all recorded prompt hashes still reproduce.

Rung (a)'s artifacts are retained at `results/pilot/gate_mixed_qmonths_s16/`. Note per
D20 §3 that a passing behavioural gate would still not license extraction on its own —
the D20 §1 task-encoding gate applies to whichever rung passes.

#### Rung (b) passes, and heterogeneous demonstrations *help*

Days + months demonstrations, query pinned to months, 16 shots, held-out, n=100:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| days+months | 1.00 | 1.00 | 1.00 | 1.00 | 0.79 | 0.85 | 0.97 | 0.77 | **0.56** | 0.86 | 0.81 | 0.91 |
| months only (§6) | 1.00 | 1.00 | 1.00 | 1.00 | 0.76 | 0.76 | 0.98 | 0.70 | **0.38** | 0.80 | 0.86 | 1.00 |

**GO**, every k ≥ 0.56. The mixed-domain control (BRIEF §6 Control 2) is therefore
available on this model, via rung (b).

The comparison is the interesting part and was not predicted: **adding a second,
differently-sized operand cycle to the demonstrations makes the months task easier**,
not harder. k=8 — the family's weakest cell and the D2 arbiter's subject — rises from
0.38 to **0.56**, and k=5 and k=7 each gain ~0.07-0.09. Only k=11 declines (1.00 →
0.91).

This runs opposite to `docs/pilot_findings.md` §4, where four-domain mixing was
*uniformly at or below* single-domain accuracy. Two domains help; four hurt. The
difference between rungs (a) and (b) is letters and digits, the two weakest domains in
§4 (0.35 and 0.43 aggregate), so the plausible reading is that heterogeneity itself
aids abstraction while individually-unlearnable domains inject noise. That is a
hypothesis this run does not test.

**Bearing on the operand-inheritance question**, stated carefully because it is
suggestive rather than decisive: BRIEF §6's confound is that a circle found in FVs
extracted from day- or month-shift prompts may be inherited from the known-circular
operand geometry. If shift-by-k is performed *better* when the demonstrations span two
cycles of different length, the model is not relying on a single operand circle to do
it. That is behavioural evidence about competence, not a measurement of FV geometry,
and the geometry claim still rests on the extraction now running under this condition
— which must clear the D20 §1 task-encoding gate before anything is read from it.

### D23. Negative controls: unrelated tasks GO, ordinal family blocked-by-gate

Both are the controls that license the stage-2 diagnostics, so both verdicts are
recorded before any Gram matrix exists.

#### Unrelated tasks — GO

Twelve tasks (`tarcle/tasks_unrelated.py`), 24 pairs each, 16 shots, held-out,
n=100 per task. Held-out accuracy is **1.000 at eleven of twelve tasks** and 0.890 at
the twelfth. The null control is fully performable and its extraction proceeds.

Two construction constraints were set by findings earlier in this run, and both were
violated by the first draft of the family:

- **Operand vocabularies are disjoint across tasks.** The first draft had
  antonym/comparative sharing 8 adjective operands, singular_verb/verb_noun sharing 10
  verbs, and capital/currency sharing 8 countries. Shared operands couple FVs through
  the operand distribution — the confound the months controls exist to exclude — and
  would have manufactured similarity structure in the very control that must show
  none. Max pairwise overlap is now 4 of 24.
- **Demonstrations are sampled without replacement**, giving 16 distinct operands per
  16-shot prompt. The first draft sampled with replacement and dropped to 7 on some
  prompts, below the `docs/pilot_findings.md` §9 collapse threshold. A collapsed null
  control would have *passed* — showing no structure because its FVs encode nothing —
  which is a false negative that proves nothing while looking like success.

**Task-encoding gate, non-cyclic form: PASS at +0.667 mean lift.** The D20 §1 gate is
defined against the ±1 attractor, which does not exist without a cycle, so for this
family it is read against each task's own zero-shot baseline.

That distinction is not cosmetic. Raw injected accuracy would have scored this control
at 0.899 and passed it for the wrong reason: **eight of the twelve tasks are already
answerable zero-shot** (`Q: France\nA:` → Paris), at baselines 0.67–1.00, so their
injected accuracy is at ceiling whatever the FV does. Only the four tasks with
baseline < 0.50 can carry evidence:

| task | zero-shot | injected | lift |
|---|---|---|---|
| currency | 0.04 | 0.83 | **+0.79** |
| english_french | 0.08 | 0.75 | **+0.67** |
| first_letter | 0.04 | 0.46 | **+0.42** |
| plural | 0.21 | 1.00 | **+0.79** |

**Limitation, recorded rather than buried:** the causal status of the other eight FVs
is not established — not because they are weak, but because their tasks leave no room
to measure. This does not undermine the control's purpose. The null control asks
whether twelve unrelated tasks produce a circulant Gram matrix, and weak or unverified
FVs add noise rather than manufacturing circulant structure; a false *positive* is what
would void the run, and noise does not produce one. But no claim may be made about any
individual unrelated task's FV outside the four above.

#### Ordinal family — NO-GO, blocked-by-gate

Extract-the-k-th-item of a list, fresh list per item, 16 shots, n=100. **NO-GO at
12 positions**: only k=1 (1.00) and k=12 (0.91) clear; every middle position sits at
or near chance (1/12 = 0.083) — k=2 0.08, k=5 0.05, k=8 0.08, k=11 0.30.

Before blocking it, list length was tested as the possible cause. It is not:

| list length | k=1 | middle positions | last |
|---|---|---|---|
| 12 | 1.00 | 0.05–0.30 | 0.91 |
| 6 | 1.00 | 0.23–0.33 | 0.95 |
| 4 | 1.00 | 0.36–0.38 | 0.93 |

The deficit is **positional**. Primacy and recency are near-perfect at every length;
middle positions improve as the list shortens but never reach 0.50, even at length 4
where only two middle slots exist. There is no list length at which the family is
performable across positions *and* large enough to support a Gram matrix. Per rule 5
and D20 §3 the ordinal control is **blocked-by-gate** and no FVs are extracted for it.

**What is lost, stated precisely.** The ordinal family was BRIEF §5's *shape* control:
a genuinely ordinal parameter should trace an open curve (high Toeplitz, low
circulant, closure ratio ≫ 1) rather than a closed loop, demonstrating on real model
FVs that the diagnostics separate open from closed. That demonstration is now
unavailable on this model.

**What still covers it, partially.** `tarcle/synthetic.py`'s `arc` fixture is exactly
this shape, and prereg §0 calibrates the diagnostics against it: arc scores toeplitz
0.93 / circulant 0.21 / closure 4.04 against circle's 0.92 / 0.92 / 1.00. So the
*diagnostics* are demonstrably able to make the distinction; what is missing is a
demonstration that they make it on vectors extracted from this model. That gap must be
stated wherever the open-vs-closed distinction is used, and the null control
(unrelated tasks) carries no substitute for it — it tests for absence of structure,
not for a different shape of structure.

**Routes, none adopted:** an ordinal family the model can actually do across positions
(alphabet position, calendar-quarter index) would need its own pilot and would not be
list-retrieval; or accept the synthetic calibration alone and report the limitation.

### D24. Headline cells LOCKED — the final stage-1 matrix

**This entry closes stage 1. It is fixed before any Gram matrix of extracted function
vectors has been computed or inspected.** D16 fixed the agreement criterion when the
matrix was still hypothetical; this locks it against the matrix that actually exists,
after four conditions were blocked or altered by their gates.

#### The matrix as built

| condition | behavioural gate | task-encoding gate | status |
|---|---|---|---|
| **primary** — months, full 12-operand pool | GO | +0.352 | **available** |
| **polysemy leave-out** — 9 months | GO | +0.343 | **available** |
| **mixed-domain** — days+months demos, month query | GO | +0.537 | **available** |
| **unrelated tasks** — 12 tasks (null control) | GO | +0.667 lift | **available** |
| operand partition — 4 and 6 operands | GO | −0.315 … −0.944 | **blocked** (D21) |
| ordinal extract-k-th | NO-GO | — | **blocked** (D23) |

Head sets, Todd only (Hendel is head-set-free and verified bit-identical across all
three): canonical six-k, all-k, intersection-8.

#### The locked headline cells

Confirmatory, and nothing else is:

> **Primary condition · full n=12 · canonical six-k head set · read off
> `spectral_concentration` and `participation_ratio` against the prereg §2 thresholds.**
> Reported once for Todd and once for Hendel. Two cells. That is the entire
> confirmatory set.

Unchanged from D16 §1, and deliberately so — the designation was made before any of
the gates ran, and none of the blocking changed the primary condition. Every other
cell (three head sets × two n × four conditions × two methods, minus what is blocked)
is **exploratory**: numbers, no verdict language, no upgrading or downgrading of the
headline.

#### Evaluation order, which is now binding

1. **The null control first.** Prereg §5: if the twelve unrelated tasks show circulant
   structure, the pipeline is broken and the entire run is discarded. Nothing else is
   read until this passes. Also report the circulant score under ≥20 random
   permutations of the arbitrary task ordering — under the null there is nothing for a
   permutation to destroy, so a permutation-sensitive score is itself a red flag.
2. **The prereg §3 frequency control**, which is blocking for the months conditions.
3. **The headline cells.**
4. Everything else, as exploratory.

#### What cannot be claimed, given what is blocked

- **No hypothesis-C verdict.** The operand-partition control is structurally
  unrunnable on Z/12 (D21). The mixed-domain control constrains operand inheritance
  but does not test the vector-vs-operator question, which is what C is about.
- **No real-model open-vs-closed demonstration.** The ordinal shape control is blocked
  (D23). The synthetic `arc` fixture shows the diagnostics *can* separate the shapes;
  nothing shows they do so on vectors from this model. Any closure/circulant claim
  carries this caveat.
- **No head-set-independent Todd claim** unless the three head sets agree by the D16 §2
  criterion — cross-head-set difference within the measured split-half band, not merely
  the same label. D15 showed 11/12 and 12/12 cells move beyond that band between head
  sets, so this is a live risk, and Hendel arbitrates a split per D16 §3 with its own
  registered limitation attached (it steers at only k ∈ {1,2,11}).

#### Stage-1 artifacts are complete

Every available condition has both extraction methods, split halves, per-query logp
with SEs, per-head contributions over the 36-cell union superset, prompt SHA-256s, and
a `head_set_source` hash. Stage 2 needs no GPU and no re-extraction, including for head
sets not yet considered, provided they are subsets of the union.

### D25. The discard rule in separator terms, and the permutation null

> **§1 of this entry is SUPERSEDED by D26.** It omitted the `circulant_score ≥ 0.70`
> gate that prereg §2 places first in both the A and A-multi definitions, which makes
> it fire on non-circulant Gram matrices whose spectra are meaningless. §2 (the
> permutation null) stands unchanged.

**Fixed before the first Gram matrix of extracted function vectors is computed.**
Both parts are free — they use machinery already built — and both correct a reading of
the null control that D24 §"Evaluation order" would otherwise have licensed.

#### 1. A high circulant score on the null control does NOT void the run

CLAUDE.md rule 2 and prereg §5 say the unrelated-tasks control "must show NO circulant
structure", and a positive there discards everything. Taken at face value that is a
**trap**, for a reason prereg §0 already documented and D24 failed to carry through:

> `circulant_score` alone cannot separate A from D. Simplex scores **0.79**.

Twelve unrelated tasks are expected to give near-orthogonal FVs — that is what
unrelated means — and near-orthogonal vectors have Gram ≈ identity + constant, which is
*trivially circulant*. Weak or noisy FVs push in the same direction. So a circulant
score around 0.8 on the null is the **expected benign outcome**, not evidence of a
pipeline bug. Discarding the run on it would throw away a correct pipeline for
behaving correctly.

**The permutation check cannot rescue the naive reading either.** A simplex Gram is
I + c·J, which is invariant under simultaneous row/column permutation, so permuting the
task order leaves its circulant score unchanged. A permutation-stable high circulant
score is the D signature, not a red flag.

**Restated discard rule.** The run voids if and only if the null control shows the
**A-signature**:

> `spectral_concentration` ≥ 0.25 (the circle/helix band) **and**
> `participation_ratio` ∈ [1.5, 7]

Pre-labelled outcomes, so neither can be argued after the fact:

| null control shows | reading |
|---|---|
| circulant ≈ 0.7–0.9, concentration ≈ 0.17–0.20, PR ≈ 10–11 | **PASS** — textbook simplex; the pipeline is behaving |
| circulant low, concentration anything, PR ≥ 8 | **PASS** — no structure at all |
| concentration ≥ 0.25 **and** PR ∈ [1.5, 7] | **VOID** — the diagnostics find a circle in unrelated tasks; discard everything |
| concentration ≥ 0.50 with a dominant frequency pair | **VOID**, emphatically |

The uniform non-DC power share at n=12 is 1/6 = **0.167**; a concentration at that
value is exactly "no distinguished frequency".

#### 2. Permutation null on the primary — exploratory, logged before computing

Registered now, before the number exists.

Months has a canonical ordering; a simplex does not care about ordering. So comparing
the **canonical** circulant score against the distribution of circulant scores under
≥20 random permutations of the k-ordering is a direct A-vs-D discriminator, built from
the same machinery as the null-control permutation check:

- **canonical ≫ permutation distribution** → the structure depends on the specific
  cyclic ordering. A simplex cannot produce this. Evidence for A / A-multi.
- **canonical ≈ permutation distribution** → the score is ordering-independent, i.e.
  it comes from near-equidistance rather than from cyclic structure. Evidence for D.

Reported as a percentile: where the canonical score falls in the permutation null.
Run on both extraction methods, all available conditions, and both head-set columns.

**Status: exploratory.** It is not among the D24 confirmatory cells and cannot upgrade
or downgrade them — the headline remains `spectral_concentration` and
`participation_ratio` on the primary/full-n=12/canonical cell. It is registered because
it is a genuine discriminator that the pre-registration did not think of, and
registering it now is what keeps it from being a post-hoc rescue if the primary
separators land ambiguously.

**The same permutation test is run on the null control**, where the prediction is the
opposite: canonical ≈ permutations, because there is no meaningful ordering to destroy.
A null control whose canonical score sits far above its own permutation distribution
would mean the arbitrary alphabetical task ordering carries structure, which is a
pipeline bug and voids the run independently of §1 above.

### D26. Correction to D25 §1 — the circulant gate comes first. This changes VOID to PASS.

**Recorded with maximum prejudice against myself, because this correction changes the
run's verdict in the run's own favour.** D25 §1, committed one commit before the first
Gram matrix was computed, produced **VOID THE RUN** on the null control. The corrected
rule produces PASS. Anyone reading this should apply the scrutiny that reversal
deserves; the git history holds the original.

#### What the null control actually shows

| | Todd | Hendel |
|---|---|---|
| **`circulant_score`** | **0.193** | **0.253** |
| `spectral_concentration` | 0.377 | 0.314 |
| `participation_ratio` | 6.85 | 5.81 |
| permutation null | canonical 0.193 vs 0.191 ± 0.007, z = +0.21 | 0.253 vs 0.261 ± 0.016, z = −0.53 |

D25 §1 said: void if concentration ≥ 0.25 and PR ∈ [1.5, 7]. Both hold. It fires.

#### Why that rule is wrong

**It omits the criterion prereg §2 lists first.** The registration defines the
A-signature as:

> A — circular: **`circulant_score` ≥ 0.70**; `spectral_concentration` ≥ 0.50 and
> dominant pair f=1; `participation_ratio` ∈ [1.5, 3.5]

and A-multi as "the same but with" different concentration and PR bands — *the same*
including the circulant gate. D25 §1 set out to restate the discard rule in separator
terms and dropped the gate while doing so. That is a defect in my restatement, not a
judgement about this data.

**And the omission is exactly the trap the project already documented.** prereg §0
records it, and `tests/test_geometry.py::test_spectrum_meaningless_without_circulant_gate`
asserts it — both committed in `ac4b195`, before any function vector existed:

> A line's class-averaged profile happens to concentrate spectral power at f=1 —
> reading the spectrum before the circulant gate would mislabel a linear code as a
> circle. **The gate must come first.**

`spectral_concentration` is computed from the DFT of the *circulant profile*, which
averages `G` over the classes (i−j) mod n. When `G` is not circulant those classes do
not describe it, so the profile is an artefact and its spectrum means nothing. At
circulant 0.193 that is the situation. Reading 0.377 as "concentration in the helix
band" is reading a number that has no referent.

#### Corrected rule

The run voids if and only if the null control shows the A-signature **as prereg §2
defines it**:

> `circulant_score` ≥ 0.70 **and** `spectral_concentration` ≥ 0.25 **and**
> `participation_ratio` ∈ [1.5, 7]

Pre-labelled outcomes, replacing the D25 §1 table:

| null control shows | reading |
|---|---|
| circulant < 0.70 | **PASS** — not circulant; concentration and PR are uninterpretable and are not read |
| circulant ≥ 0.70, concentration ≈ 0.167–0.25, PR ≥ 8 | **PASS** — simplex; the expected benign outcome D25 §1 correctly identified |
| circulant ≥ 0.70, concentration ≥ 0.25, PR ∈ [1.5, 7] | **VOID** |
| canonical circulant far above its own permutation null (z > 3) | **VOID** — arbitrary task ordering carries structure |

#### Verdict under the corrected rule: PASS, on two independent grounds

1. **Circulant 0.193 and 0.253, far below the 0.70 gate.** The twelve unrelated tasks
   produce no circulant structure, which is precisely what CLAUDE.md rule 2 and prereg
   §5 demand.
2. **The permutation null agrees, and it needed no correction.** Canonical sits at
   z = +0.21 (Todd) and −0.53 (Hendel) in its own permutation distribution — the
   arbitrary alphabetical task ordering carries no structure whatever. This is the D25
   §2 check, registered before any number existed and unaffected by the D25 §1 defect,
   and it independently confirms the pass.

The second ground is what makes me confident this is a correction rather than a rescue:
a test I registered separately, that does not share the defect, returns the same
verdict. Had the permutation null shown canonical ≫ permutations, the run would void
under the corrected rule too, and it does not.

#### Consequence for the stage-2 code

`tarcle/stage2.py` gates on `circulant_score` before reading the spectrum, in both the
void rule and the bucket classifier. A cell failing the gate is reported as
"not circulant — spectrum not read" rather than assigned a bucket.

### D27. Centered-Gram variant, registered as exploratory before computing

**Registered after the raw-Gram headline was seen and before the centered variant is
computed.** That ordering is stated plainly because it is the weakest kind of
registration; what makes it more than a post-hoc rescue is that the motivation is on
the record *before any Gram matrix existed*, in commit `3eac9d8`:

> Both extractions carry a large shared offset (Todd 78.8%, Hendel 94.9% of mean
> squared norm); centered cosines span the full range, so the k-structure is real but
> **centering is not optional**.

#### The problem

Write X = c·u + Y, with c·u the component shared across all k and Y the k-dependent
part. Then

    G = |c|²·J  +  c·(u Yᵀ + Y uᵀ)  +  Y Yᵀ

The first term is constant and harmless — `_class_score` removes the global mean. The
**middle term is not**: it varies with i and with j separately, not with (i−j), so it
is neither circulant nor Toeplitz, and it scales with |c| relative to |Y|. At 78.8%
(Todd) and 94.9% (Hendel) of mean squared norm it dominates the raw Gram.

This is consistent with what the raw numbers show and would otherwise be hard to
reconcile: `circulant_score` **and** `toeplitz_score` are both low (0.09 / 0.16), i.e.
raw similarity is not a function of parameter distance in any form — yet
`spectral_concentration` sits at 0.68 on f=1 and the D25 §2 permutation null is at
z = +8.5. Structure that the ordering demonstrably carries, invisible to the two
class-based scores. A rank-2 offset contamination produces exactly that signature.

The prereg §0 fixtures do carry an offset, but a smaller one: `circle(offset=0.8,
radius=1.0)` puts 39% of squared norm in the shared component against the 79% and 95%
measured here, and still scores circulant 0.92. The calibration does not cover this
regime.

#### What is run, and its status

`diagnose(X − X.mean(axis=0))` for every cell already reported, alongside the raw
values, never replacing them.

- **The D24 headline remains the raw-Gram cell.** `diagnose()` uses `gram(X)` with no
  centering, that is what the pre-registration's thresholds were calibrated against,
  and the confirmatory verdict is read there.
- The centered column is **exploratory** and may not upgrade or downgrade the headline.
- Both columns are reported side by side wherever either is quoted, and the split-half
  band is computed for both so "differs from raw" can be checked against noise.

**Interpretation fixed in advance.** If the centered Gram is circulant (≥0.70) with the
raw one not, the honest reading is *"the k-dependent component of these FVs carries
cyclic structure, which is masked in the raw Gram by a shared offset carrying ~80-95%
of the norm"* — not "the FVs are circular". The offset is part of the vector that
actually gets injected and steers the model; a claim about task-space geometry that
holds only after removing 80% of the vector must say so in those words.

### D28. add-k calibration family — purpose and predictions, fixed before extraction

**Registered before the add-k prompts are generated.** The predictions in §3 are
committed here so they cannot be fitted to the result.

#### Purpose — calibration, not open-vs-closed

The ordinal control was intended as the *shape* control and is blocked (D23). add-k is
**not** a replacement for it and the earlier framing of it as an "open-vs-closed
substitute" is withdrawn: §2 of `docs/stage2_findings.md` no longer describes the
months result as an open curve, so there is no open-vs-closed claim left for it to
adjudicate.

What add-k actually provides is a **calibration point the run currently lacks**. The
diagnostics have exactly two real-model reference readings:

| reference | ordering | permutation z | circulant |
|---|---|---|---|
| unrelated tasks (null) | none — arbitrary labels | +0.21 / −0.53 | 0.193 / 0.253 |
| months Z/12 | cyclic | +8.48 / +7.52 | 0.091 / 0.113 |

There is **no real-model reading for an ordered, non-cyclic parameter**. Without one,
"permutation z = +8.5 means cyclic structure" is unsupported: any monotone ordering
would also beat random permutations, so z alone cannot distinguish *cyclic* from merely
*ordered*. add-k on small integers supplies that missing third reading.

#### Design — same n, same format, k=0 excluded

`Q: <integer>\nA: <integer + k>` over a fixed integer operand pool, same prompt format,
same shot count, same n as months so every threshold transfers unchanged.

**k = 0 is excluded**, for the reason D3 excludes it from head identification and §2
of the findings now documents empirically: identity is solvable by the copy prior, and
in months FV(0) is an outlier at ~2× the family's internal distance scale. Including it
would plant the same degenerate point in the calibration family and make the comparison
about that point rather than about ordering. add-k therefore runs k = 1..n over a
non-cyclic range, and the months comparison is drawn against the **n=11 drop-k=0**
column, not the full n=12.

Both gates apply per D20 §3: a behavioural pre-gate on this exact condition, and the
task-encoding gate. add-k has no ±1 attractor analogue defined a priori, so the
task-encoding gate takes the non-cyclic form — lift over the zero-shot baseline.

#### What would show the diagnostics discriminate — fixed now

add-k is genuinely ordered and genuinely **non**-cyclic: 1 and n are maximally
separated with no wraparound.

- **Discriminating outcome.** add-k shows high permutation z (ordered) but a
  *materially different* profile from months on the cyclic-specific diagnostics —
  circulant score, and the |m| distance profile computed under wraparound
  identification. Concretely: months' |m| profile is non-monotone at the antipode
  (|m|=6 dips below |m|=5: 5.11 → 4.89), which is what wraparound identification
  produces; add-k's should be **monotone through to the largest separation** with no
  antipodal dip, because there is no antipode. If that holds, the diagnostics separate
  cyclic from merely-ordered on real FVs, and months' cyclic reading is supported.
- **Non-discriminating outcome.** add-k reproduces months' profile closely — similar
  circulant score, similar permutation z, similar antipodal dip. Then permutation z and
  the |m| profile measure *ordering* and nothing more, months' structure is not shown
  to be specifically cyclic, and every ordering-based claim in §2 of the findings must
  be weakened to "ordered" wherever it currently says "cyclic".
- **Uninformative outcome.** add-k fails either gate, in which case it is reported
  blocked and no calibration is obtained — the same fate as the ordinal control, and
  the run keeps the two-reference limitation stated explicitly.

The comparison is **exploratory** throughout. It cannot upgrade or downgrade the D24
confirmatory cells; it can only qualify how the permutation-null evidence is described.

### D29. Forced-choice token collisions — a scoring bug found while building add-k

Building the add-k family surfaced a tokenizer problem that also affected an artifact
already committed.

**Llama-3 tokenizes `" 15"` as `[" ", "15"]`** — a bare space token followed by the
number. Since `first_token_id` scored the leading-space form, every numeric candidate
shared the same first token and the forced choice would have been completely
degenerate. add-k therefore puts the space in the prompt (`"Q: 10\nA: "`) and scores
the bare number; `first_token_id` takes a `prefix` argument for this. The behavioural
pilot path was never affected — `backends.py` teacher-forces multi-token candidates, so
the shared space token cancels in the argmax.

**Checking every family for the same defect found three collisions in the
unrelated-tasks control**, which was already extracted and committed:

| task | colliding candidates | shared first token |
|---|---|---|
| comparative | `bolder` / `briefer` | `" b"` |
| currency | `rial` / `rupiah` | `" r"` |
| currency | `kuna` / `kip` | `" k"` |

Those pairs were indistinguishable to the scorer, so the affected tasks' efficacy
numbers were computed over a candidate set two entries smaller than it appeared.
Replaced with `quicker`, `gourde` and `afghani`; all twelve tasks now have fully
distinct first tokens (23/23 for currency, previously 21/23).

**Consequence: the committed null-control artifacts are stale.** Two of twelve word
lists changed, so `results/pilot/gate_unrelated_s16/` and
`results/fv/ctl_unrelated_12tasks/` can no longer be regenerated from the code —
`tests/test_extract.py::test_recorded_prompt_hashes_reproduce` fails on them, which is
the determinism guarantee prereg §5 requires working as intended.

They are **removed and re-run**, not kept. An artifact that cannot be reproduced from
the committed code is not evidence, and carrying it alongside reproducible ones invites
exactly the confusion the SHA-256 check exists to prevent. The originals remain in git
history at `0bad1ba` for anyone auditing this decision.

The D23 verdict is expected to survive — the change touches 3 candidate words out of
276 pairs — but it is **re-derived from the new artifacts rather than assumed**, and
§1 of `docs/stage2_findings.md` is recomputed from the re-run.

### D30. Calibration outcome: the ordering evidence does NOT establish cyclicity

D28 pre-stated three possible outcomes. **The non-discriminating one occurred**, and
its registered consequence is applied: every claim resting on the ordering evidence is
weakened from *cyclic* to *ordered*.

#### Three real-model references, same diagnostics

| family | ordering | `circulant` | `circulant` centered | permutation z |
|---|---|---|---|---|
| unrelated tasks | none | 0.190 / 0.253 | 0.412 / 0.331 | **+0.17 / −0.54** |
| **add-k** | **ordered, NOT cyclic** | **0.285 / 0.144** | 0.448 / 0.304 | **+7.44 / +4.48** |
| months n=11 | ordered *and* cyclic | 0.137 / 0.109 | 0.312 / 0.328 | **+7.89 / +8.93** |

(Todd / Hendel. add-k passed both gates: behavioural GO at 1.000 for every k,
task-encoding lift +0.826.)

#### Two things this settles

**1. Permutation z measures ordering, not cyclicity.** add-k is *definitionally*
non-cyclic — its first and last parameter values are maximally separated with nothing
identifying them — and it scores z = +7.44 against months' +7.89. The statistic cannot
tell them apart. D25 §2 registered the permutation null as "a direct A-vs-D
discriminator", and it is: it cleanly separates ordered families (z ≈ +4 to +9) from
the unordered null (z ≈ 0), which rules out the simplex/lookup reading. But it says
nothing about *cyclic* structure, and `docs/stage2_findings.md` §2 said it did.

**2. `circulant_score` does not rank the families by cyclicity either.** add-k, which
has no cycle at all, scores **0.285** against months' **0.137** under Todd. The
non-cyclic family scores higher than the cyclic one. Together with the null control at
0.190 — also above months — the raw circulant score orders these three families in a
way unrelated to whether they have a cycle.

#### The pre-stated discriminator did not work, and is not replaced post hoc

D28 predicted months would dip at the antipode while add-k rose to its end. Months does
dip (5.11 → 4.89 Todd, 4.39 → 4.25 Hendel). But add-k *also* turns down at its largest
separation — and that bin holds **2 pairs** out of 110, so it is not read. The
unrelated control turns down too, on a profile that is flat throughout (0.92–1.00). The
"turn at the end" appears in all three and discriminates nothing.

The profiles do differ in other ways — months rises to 1.86× by separation 5 where
add-k reaches only 1.48×, and add-k shows an odd/even sawtooth months lacks. **These
were not pre-registered and are not adopted as discriminators.** They are recorded as
observations for a future study that can register them in advance.

#### Consequence, applied

`docs/stage2_findings.md` §2 is amended: "the cyclic parameter ordering is doing work"
becomes "the parameter ordering is doing work", and the summary describes **ordered**
rather than cyclic structure. What the run supports is:

> The months FVs form a low-dimensional, constant-norm, **ordered** family whose
> internal distances grow with parameter separation — not a simplex, not a linear
> code, and **not shown to be cyclic**.

The wraparound quarantine already prevented the one claim that would have been
specifically cyclic. This entry removes the remaining implicit ones.

#### What would settle it

A diagnostic validated to separate cyclic from ordered on real FVs. The obvious
candidate is the antipodal-dip magnitude, measured on a family with enough pairs in the
largest-separation bin — which needs n ≥ 16 or so, i.e. ROT-k on Z/26. That is the
question the deferred ROT-k family would actually answer, and it is a better reason to
run it than the one previously recorded.

### D31. Seam contest — registered before running, and post-hoc relative to the binned profile

**Stated plainly up front: this registration is post-hoc.** The binned separation
profiles in D30 have already been seen, and they are what suggested this test. It is
registered before the contest is computed, not before the data that motivated it. That
is weaker than D25 §2 (registered before any number existed) and much weaker than the
pre-registration proper. **Exploratory relative to D24 throughout; it cannot touch the
confirmatory cells.** The confirmatory home for this test is the *next* family, where
it can be registered in advance — see §6.

#### The question

D30 established that permutation z and `circulant_score` cannot separate cyclic from
merely ordered. Neither can the binned |m| profile, whose antipodal turn appears in all
three families. What none of those tests asks is the specific question that
distinguishes the two: **is there a seam?**

A cyclic family has none — every point is interior. A linear family that has been
wrapped for analysis has exactly one: the cut where the line's two endpoints were
glued. So unrolling the parameter at each possible cut point and asking which unrolling
best explains the pairwise distances is a direct test, and it is not the test any
previous diagnostic performed.

#### Design

For FVs X over parameter values k with modulus n:

- **Cyclic model:** separation(i,j) = min(d, n−d), d = |kᵢ−kⱼ|. Six levels at n=12.
- **Cut-at-c model**, one per c: unroll to pos(k) = (k−c) mod n, separation = |pos(i)
  − pos(j)|. Eleven levels at n=12. One such model for every c in k=1..11.

Each model is scored two ways against the observed pairwise distances:

- **Spearman ρ** between separation and distance — rank-based, insensitive to the
  functional form.
- **Cross-validated isotonic R²** — fit a monotone function of separation on half the
  pairs, score on the held-out half, repeated over splits.

**The cross-validation is not optional and the reason is registered here:** the cut
models have eleven separation levels against the cyclic model's six, so an
unpenalised isotonic fit favours them mechanically, by degrees of freedom alone. Any
comparison of in-sample fit between models with different level counts is
uninterpretable. Both metrics are reported for every model; the verdict is read from
the cross-validated figure.

#### Validation gate — run and read before months

The contest is run on the two families whose answer is known, and **months is not read
until both behave**:

- **add-k** (ordered, no wraparound, n=11 unrolled with a hypothetical modulus 12).
  The true structure is a line with its endpoints at 1 and 11. **A cut model must beat
  the cyclic model**, and the winning cut should sit at an endpoint. If the cyclic
  model wins here, the contest cannot detect a seam that is known to exist and the
  whole test is void.
- **unrelated tasks** (unordered). Distances are flat, so **no model should fit** —
  all cross-validated R² near zero and no cut distinguishable from any other. A cut
  winning here means the contest manufactures seams from noise.

#### Pre-labelled outcomes for months

| result | reading |
|---|---|
| cyclic model beats **every** cut, cross-validated | genuinely cyclic — no seam. The strongest cyclicity evidence the run could produce |
| a cut at **c = 0** wins | the family is a line running k=1…11 with the identity as an endpoint. Consistent with FV(0) sitting ~2× outside the cluster; the "cycle" would be an artefact of labelling |
| a cut at the **antipode** (c ≈ 6) wins | signed-magnitude line: +m and −m identified, distance a function of \|m\| with no direction. The pilot's magnitude-recovered / direction-lost finding (`pilot_findings.md` §2) in geometric form |
| no cut separable from the cyclic model, or from each other | no discrimination; the run cannot tell, and says so |

#### Reported individually, not only in aggregate

- **The ten fold-back pairs** — those the cyclic model calls close and every cut model
  calls far. At n=11 on Z/12 these are the pairs with d > 6: (1,8) (2,9) (3,10) (4,11)
  at d=7, (1,9) (2,10) (3,11) at d=8, (1,10) (2,11) at d=9, (1,11) at d=10. Whether the
  wraparound identification is real lives entirely in these ten distances, so they are
  listed with their values rather than folded into a mean.
- **The five antipode-crossing pairs** at cyclic separation 6: (1,7) (2,8) (3,9)
  (4,10) (5,11).

Both extraction methods, raw and centered Gram, all reported.

#### Where the confirmatory version lives

Whatever this returns, it is exploratory. If it does not settle the question, the next
step is **not** more analysis of these artifacts: it is a new family with the seam
contest registered in advance. Hours-of-day Z/24 and ROT-k Z/26 are both candidates,
piloted together with the gates deciding which survives, and **a matched-n add-k
reference registered alongside whichever runs** so the calibration D28 established is
available at the same n rather than borrowed across sizes.

### D32. Seam contest outcome: a seam is present; its location is not resolved

Exploratory per D31. The validation gate passed before months was read.

#### Validation gate — PASS

| family | cyclic | best cut | verdict |
|---|---|---|---|
| **add-k** (seam known to exist) | 0.226 / 0.012 | **0.347 / 0.197** (cut@2) | seam detected ✓ |
| **unrelated** (nothing should fit) | −0.115 / −0.115 | −0.104 / −0.070 | all negative; nothing fits ✓ |

Cross-validated isotonic R², Todd / Hendel. The contest detects a seam where one is
known and manufactures none from unordered data.

#### Months — the cyclic model loses decisively

| | cyclic | best cut | margin |
|---|---|---|---|
| Todd | **0.031** | 0.237 (cut@2) | **−0.206** |
| Hendel | **0.049** | 0.221 (cut@2) | **−0.171** |

The cyclic model's cross-validated R² is ≈ 0, i.e. identifying separations mod 12
explains essentially none of the variance in pairwise distance, while a linear
unrolling explains ~22%.

#### Location: NOT resolved, and the reason is in the gate data

cut@2 wins for months — but cut@2 **also** wins for add-k, whose true seam is at the
endpoint, i.e. cut@0/cut@1. There the correct model scored 0.320 against cut@2's 0.347.
So the contest prefers cut@2 by a small margin even when it is wrong, and the months
cut@2 win cannot be read as locating the seam. The top three cuts (2, 1, 11) are within
0.07 of each other and all sit near the k=0 gap.

Registered note: `cut@0` and `cut@1` produce **identical** separations on the k=1..11
set — only |posᵢ − posⱼ| enters, and the two differ by a constant shift — so the D31
"seam at k=0" model is the cut@0 = cut@1 tie, scoring 0.171 / 0.182. Second place, not
the winner.

**None of D31's three location labels fires cleanly.** The reportable outcome is "a
seam is present, its position is not resolved by this test".

#### The ten fold-back pairs settle the wraparound question directly

These are the pairs the cyclic model calls close and every linear model calls far, so
the entire wraparound claim lives in them. Distances as multiples of the mean pair
distance (Todd / Hendel):

| pair | linear sep | cyclic sep | distance |
|---|---|---|---|
| **k=1, 11** | 10 | **2** | **1.35× / 1.29×** |
| k=2, 11 | 9 | 3 | 1.77× / 1.75× |
| k=1, 10 | 9 | 3 | 1.58× / 1.48× |
| k=3, 11 | 8 | 4 | 1.72× / 1.68× |
| k=2, 10 | 8 | 4 | 1.08× / 1.08× |
| k=1, 9 | 8 | 4 | 1.56× / 1.53× |
| k=4, 11 | 7 | 5 | 1.65× / 1.68× |
| k=3, 10 | 7 | 5 | 1.11× / 1.05× |
| k=2, 9 | 7 | 5 | 1.28× / 1.34× |
| k=1, 8 | 7 | 5 | 1.55× / 1.49× |

**Every one of the ten is above the mean**, in both extractions. The wraparound
identification predicts the opposite: pairs at cyclic separation 2–5 should sit near
the mean or below it. Most telling is **k=1 versus k=11** — shift-by-+1 and
shift-by-−1, the pair a cyclic code must place adjacent — at **1.35×** the mean
distance, against a separation-1 bin mean of 0.67×.

The five antipode-crossing pairs are inconsistent rather than uniformly far: (3,9)
0.82×, (4,10) 0.93×, but (1,7) 1.48× and (5,11) 1.56×. No antipodal identification
either.

#### Standing of this result

It upgrades D30's "cyclicity not established" toward **positively disconfirmed in
full-vector distances** for the months family: not merely that the diagnostics cannot
see a cycle, but that the specific pairs a cycle requires to be close are measurably far
**in the full-dimensional Euclidean distance between function vectors**.

That scope qualifier is load-bearing and is not hedging. Every statistic in this
contest — pairwise distance, isotonic fit, the fold-back table — is computed on whole
vectors. A circular component carrying a small share of the variance, riding on a
dominant non-circular axis, would leave all of them looking exactly as they do. D33
registers the check for precisely that geometry, and until it is run "disconfirmed"
means disconfirmed *as a description of the whole vector*, not "no circular structure
is present anywhere in these vectors".

It remains **exploratory** — D31 is post-hoc relative to the binned profile — so it does
not touch the D24 confirmatory cells, and the confirmatory version belongs to the next
family with the contest registered in advance.

Also recorded: centering is a **no-op** for this test, asserted in code rather than
reported as a column. Pairwise distances are translation-invariant, so the shared
offset that D27 exists for cannot affect the seam contest at all.

### D33. Cylinder / open-helix check — registered before running

**Third look at the same eleven vectors, with a decreasing prior each time.** The
sequence is: registered diagnostics (D24), post-hoc seam contest (D31), and now a
post-hoc check motivated by the seam contest's own result. Each step is further from
pre-registration than the last, and this entry is the furthest. It is registered before
computing, it is exploratory, and **the confirmatory home is the n ≥ 16 family**, where
this check can be registered in advance alongside the seam contest.

#### The gap in the calibration battery

`tarcle/synthetic.py`'s `helix` fixtures are sums of circles at several frequencies —
**closed** curves, every one. `line` is a straight open curve with no circular
component. **No fixture in the prereg §0 battery is an open helix**: a line crossed with
a circle, i.e. a dominant monotone axis with a circular component wrapped around it.
The diagnostics were never calibrated against that shape, so nothing in this run so far
can distinguish it from a plain ordered family.

And it reproduces **every** seam-contest observation:

| observation | open helix predicts |
|---|---|
| fold-back pairs all far | yes — axial separation dominates distance, and k=1 to k=11 is the longest axial span |
| linear unrolling ~22%, cyclic ~0% | yes — the axial coordinate is monotone in k, the circle contributes little to full-vector distance |
| PR ≈ 3.1–3.6 | yes — exactly 1 axial + 2 circular dimensions |
| circulant fails, toeplitz fails | yes — the axial term is neither |

The seam contest cannot reject it, because a cylinder *has* a seam in exactly the sense
the contest tests: unroll it and the axial coordinate explains the distances.

#### Procedure, fixed before running

1. **Add an open-helix fixture** to `tarcle/synthetic.py`: `axial * t + circle(t)`,
   parameterised by the axial/circular amplitude ratio, at the **measured** values —
   the ratio implied by months' PR and axial variance share, and the measured
   shared-offset regime (78.8% Todd / 94.9% Hendel of mean squared norm).
2. **Validate the extractor on the fixture.** Fit the monotone axis, project it out,
   and confirm the planted circle is recovered — circulant score on the residuals, and
   the fold-back pairs becoming close. If projection does not recover a circle that is
   *known* to be there, the method cannot detect one and the check is void.
3. **Validate that it manufactures nothing.** Run the identical projection on **add-k**
   residuals (ordered, no circle planted) and on the **unrelated null**. Any circular
   signal there is the method's own artefact.
4. **Pre-commit an amplitude floor** from step 3: the maximum residual circular
   amplitude observed on add-k and the null. **A months residual circle below that floor
   is not a finding** and is reported as absent. This is committed before months
   residuals are inspected.
5. **Only then read months residuals**, in this order: the **ten fold-back pairs**
   first, with **(1,11) decisive** — if k=1 and k=11 do not become close after the axial
   component is removed, there is no wrapped circle regardless of what any aggregate
   score says.

#### Same pass: is the axis k, or token frequency?

The fitted axial coordinate is correlated against the **stored per-k frequency proxies**
(`freq_proxy_operand`, `freq_proxy_target`, present in every months `.npz` per D11).

A monotone axis that tracks token frequency rather than k would mean the "ordered"
structure this run has been describing is partly a frequency gradient — which would
also bear on prereg §3, whose norm-based test passed only because `norm_cv` is small
(0.054 / 0.012) and which does not test a *direction*. Reported either way; a null
result here is as informative as a positive one.

#### Outcomes

- **Residual circle above the floor, fold-back pairs close, axis uncorrelated with
  frequency** → the family is a cylinder: ordered along an axis with a genuine circular
  component that whole-vector distances cannot see. D32's verdict keeps its
  "in full-vector distances" scope and the cylinder becomes the description.
- **Residual circle below the floor** → no wrapped circle at any amplitude this method
  can detect; D32's verdict generalises beyond full-vector distances.
- **Axis correlates with the frequency proxy** → reported prominently regardless of the
  circle result, and the "ordered" description is qualified accordingly.

### D34. Free seam-location read — the antipode-seam discriminator

Registered with D33's whole-vector caveat attached: this reads full-vector distances and
inherits their blindness to a low-amplitude circular component.

D32 found a seam but could not locate it. One pre-labelled location makes a **unique,
checkable prediction** the others do not:

> The **antipode-seam / signed-magnitude** model says distance depends on
> |m| = min(k, 12−k) with direction discarded. It therefore places k=6 and k=7 —
> which have |m| = 6 and 5 — **far apart across the seam**, where every raw-k linear
> model places them **adjacent** (separation 1).

So **d(6,7)** discriminates, and it needs no new computation.

Reported individually, both extractions: **(6,7)** as the test, with **(5,7)** and
**(6,8)** as support — both separation 2 under the raw-k line, and both crossing the
antipode under the signed model.

**Prior evidence already cuts against the signed line**, recorded here so the test is
not read in isolation: the signed-magnitude model predicts **(1,11) close** — |m| = 1
for both — and D32 measured it at **1.35× / 1.29× the mean**, among the farthest pairs
in the family. If (6,7) also comes out ordinary, the signed line is excluded from both
directions and the k=0-seam reading is what remains.

### D35. The off-by-two localizer bias — observed, not adopted

D32 recorded that the contest mislocated add-k's seam: the true endpoint seam is at
cut@0 ≡ cut@1 (0.320) but cut@2 won (0.347). Months shows the **same signature** —
cut@2 first (0.237 / 0.221), the k=0-seam model second (0.171 / 0.182).

Read naively, months reproducing add-k's error pattern is consistent with months having
a **true seam at k=0**, displaced by the same +2 bias. That reading is **not adopted**,
for two reasons:

- It rests on a single calibration point — one family, one n, one amplitude regime.
  Whether the bias is a stable property of the localizer or an accident of add-k's
  particular geometry is unknown from n=1.
- The bias direction and size could plausibly depend on n, on the axial/circular ratio,
  and on how the isotonic levels distribute — none of which is characterised.

**Recorded as an observation with a calibration caveat.** What converts it into a
measurement is the **matched-n add-k reference registered alongside the next family**
(D31 §6): with add-k run at the same n as the new family, the localizer's bias becomes
something measured on a known seam and then subtracted, rather than a pattern noticed
twice.

### D36. Z/24 planning notes — hypothesis C rides the same extraction

Recorded now so the next family's design is fixed before its pilot rather than after.

**Hours-of-day Z/24 unblocks hypothesis C**, which D21 declared structurally unrunnable
on Z/12. The blocking argument was arithmetic: the control needs two *disjoint* operand
partitions, the largest disjoint pair on a 12-element cycle is 6+6, and 6 collapses the
FVs to next-item (`docs/note_operand_diversity.md`). **On Z/24 the largest disjoint pair
is 12+12, and 12 operands clear the measured threshold comfortably** — it is the
full-pool months condition, which passed at +0.352.

So the operand-partition transfer test **rides the same extraction run** at no
additional pilot cost, and the D18 criterion (Δ logp_lift matched vs transferred,
against a split-half band) applies unchanged. The D20 §1 task-encoding gate still
governs: both partitions must clear it before any transfer verdict is read.

**Power the seam contest for two things, not one.** D32 resolved seam *presence* and not
seam *location*; D33 asks for residual-circle *detection*. Both need more pairs per
separation bin than n=11 provides — the largest-separation bin held 2 pairs of 110.
At n=24 the thinnest bin holds 12. The next contest is registered in advance for
location and residual amplitude, with the matched-n add-k reference of D35 alongside.

ROT-k Z/26 remains a candidate and the two are piloted together with the gates
deciding. Z/24's advantages are recorded here: hours-of-day has no privileged k (unlike
ROT-13), 24 is composite so the prereg §1 divisor prediction is testable (divisors 2, 3,
4, 6, 8, 12 against Z/12's 2, 3, 4, 6), and digits are a stronger domain than letters
(0.43 vs 0.35 aggregate, `pilot_findings.md` §4).

### D37. Cylinder check: the registered estimator failed its gate; the replacement is underpowered. INCONCLUSIVE.

#### The D33 estimator failed step 1, exactly as the gate was meant to catch

D33 registered "fit the monotone axis, project it out, look for a circle in the
residual", with the gate: *if projection does not recover a circle that is known to be
there, the method cannot detect one and the check is void.*

| planted | residual circulant |
|---|---|
| open helix, axial 6.0, radius 1.0 | 0.350 |
| open helix, axial 3.0, radius 1.0 | 0.345 |
| pure circle (the ceiling) | 0.953 |
| **no-circle floor** (add-k / null residuals) | **0.410** |

A circle that is *definitely present* recovers to 0.35, **below** the 0.41 floor produced
by families with no circle at all. The estimator is void by its own registered gate.

**The reason is structural, not a bug and not sample size.** Over a single period,
corr(k, sin(2πk/12)) = **−0.872**. A single-period circle is itself strongly linear in
the parameter, so removing a monotone axis removes most of the circle with it. This does
not improve with n — the correlation is O(1) in n. Any "project out the trend" approach
to separating an axis from a one-period circle has the same defect.

(Noted for the next family: the collinearity is phase-dependent — corr(k, cos) = −0.000
here while corr(k, sin) = −0.872 — so an estimator's power against this geometry varies
with the unknown phase of whatever circle is present.)

#### Replacement estimator, validated before use

Regress X on [1, k, cos(2πk/12), sin(2πk/12)] and measure the harmonic terms' unique
variance share (partial R²). Collinearity reduces power but does not prevent
identification, so unlike projection this does not destroy what it is looking for.

Synthetic validation passes cleanly:

| planted | harmonic amp | partial R² |
|---|---|---|
| open helix, axial 6.0 / 3.0 / 1.0 | **1.599** in all three | 0.149 / 0.355 / 0.596 |
| line only, no circle (3 seeds) | 0.266–0.320 | 0.0013–0.0020 |

Amplitude recovers the planted radius invariant to axis strength; partial R² separates
by two orders of magnitude.

#### But the floor on *real* no-circle data is enormous

| family (no circle present) | partial R² |
|---|---|
| add-k, Todd | **0.3148** |
| add-k, Hendel | 0.2722 |
| unrelated null, Todd | 0.1381 |
| unrelated null, Hendel | 0.1256 |

Four design parameters on eleven points leaves seven residual degrees of freedom, and
real function vectors have far richer structure than a synthetic ramp plus isotropic
noise. **A family with no circle reaches partial R² 0.31 — higher than a genuinely
planted circle at a realistic axial ratio (0.149).**

#### Months, read against the pre-committed floor

| | partial R² | vs floor 0.3148 | floor's own spread |
|---|---|---|---|
| Todd | 0.3626 | **+0.048** | 0.189 |
| Hendel | 0.4233 | **+0.109** | 0.189 |

Nominally above the maximum floor — but by margins **2× to 4× smaller than the spread
of the floor estimate itself**. The amplitude metric points the other way (2.97 / 2.74
against a floor of 4.54), though amplitudes are not comparable across families of
different norm.

**Verdict: INCONCLUSIVE.** Not "no circle" — months does exceed the maximum no-circle
floor. Not "a circle" — the margin is well inside the noise of the floor. The check
cannot resolve this at n=11 and is reported as underpowered rather than as either
outcome.

#### The axis is k, not token frequency

The D33 frequency check returns a clean answer:

| | corr(axis, k) | corr(axis, freq_proxy_operand) | corr(axis, freq_proxy_target) |
|---|---|---|---|
| Todd | **+0.978** | +0.451 | −0.424 |
| Hendel | **+0.955** | +0.371 | −0.363 |

The fitted axial coordinate is essentially k. The frequency-drift alternative — that the
"ordered" structure is a token-frequency gradient rather than a task-parameter one — is
**ruled out**. The residual 0.37–0.45 correlation with the operand proxy is reported for
completeness; since the axis is ~0.97 correlated with k, it is very close to
corr(k, proxy) in the empirical draw, and prereg §3's norm-based test passed
independently on `norm_cv` ≤ 0.15.

#### Consequences

- **D32's "in full-vector distances" scope qualifier stands.** The cylinder is neither
  established nor excluded. Any statement that months contains no circular structure
  must keep that qualifier.
- **The held `stage2_findings` sentence stays unwritten.** It was to be resolved by this
  check into either "the identification the model uses behaviourally is not present in
  the vector" or "behavioural wraparound lives in a low-amplitude circular
  subcomponent". Neither is supported. It is not written.
- **Powering this is now a stated requirement for the next family** (D36): at n=24,
  four parameters on twenty-four points leaves twenty residual degrees of freedom
  against seven here, which is where the floor has to fall for the test to discriminate.

### D38. Both second-family candidates fail their gates. The §6 "neither passes" branch fires.

| family | verdict | detail |
|---|---|---|
| **ROT-k, Z/26** | **NO-GO** | 23 of 26 k below 0.50. k=0 1.00, k=1 0.97, then k=2 0.29, k=3 0.19 and flat thereafter |
| **hours-of-day, Z/24** | **NO-GO** | 13 of 24 k below 0.50; 11 at ≥0.50 |
| add-k n=24 (matched reference) | **NO-GO** | 8 of 23 k below 0.50; perfect to k=11, then k=13 0.60, k=14 0.43, k=16 0.22 |

`docs/preregistration_family2.md` §6: *"Neither passes → no second family on this
model; the months results stand with their stated limits and the project's next move is
a different model, not a different family."* **That branch fires.**

#### ROT-k confirms the literature warning, including for k=13

`docs/lit_sweep_task_space_geometry.md` flagged the ROT-k pivot as weaker than it looks:
shift-cipher performance is dominated by corpus frequency of the specific shift
(McCoy et al.; Prabhakar et al. 2407.01687), and letters are this model's weakest
domain (0.35 aggregate, `pilot_findings.md` §4). Both hold. **ROT-13 does not escape it
either** — k=13 sits below the GO threshold with the rest. On a 3B model the corpus
advantage of ROT-13 is not enough to make even the privileged shift usable.

#### Hours: the strongest replication in the project, on a family that still fails

Re-indexed by signed magnitude |m| = min(k, 24−k):

| \|m\| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +m | 0.93 | 0.91 | 0.78 | 0.68 | 0.68 | 0.47 | 0.49 | 0.38 | 0.31 | 0.32 | 0.29 | **0.42** |
| −m | 0.94 | 0.93 | 0.85 | 0.66 | 0.67 | 0.37 | 0.32 | 0.36 | 0.35 | 0.32 | 0.24 | — |

Two things replicate from months and one does not:

- **Monotone decay in |m|** — clean, over twelve magnitudes rather than six.
- **The antipode is a landmark.** k=12 scores **0.42** against neighbours k=11 at 0.29
  and k=13 at 0.24. This is the months k=6 effect (0.98 against 0.76 and 0.70,
  `pilot_findings.md` §3) reproduced on a different cycle at a different n. The
  order-2 self-inverse element is easier than its neighbours in both families.
- **The forward/backward asymmetry does NOT replicate.** Forward beats backward at only
  **6 of 11** magnitudes, mean 0.567 against 0.546. On months the asymmetry was large
  and consistent (+2 1.00 vs −2 0.86, +3 1.00 vs −3 0.80, +4 0.76 vs −4 0.38) and
  `pilot_findings.md` §2 generalised it beyond Todd et al.'s ±1 observation. **It is
  months-specific, or at least not a general property of cyclic shift families.** That
  qualifies §2, which is now the one pilot finding with a failed replication against it.

This matters beyond the gate, because D34 established that the *function vectors* are
ordered by raw forward k rather than signed shift. The behavioural asymmetry was the
strongest reason to expect a signed representation; on hours it is largely absent.

#### The matched add-k reference also fails at n=24

Independently blocking. Even had a family passed, the D28 calibration and the D35
localizer-bias measurement both require an add-k reference at the family's own n, and
that reference is itself NO-GO at n=24 (perfect through k=11, degrading past k=13 as
the arithmetic crosses more tens boundaries). So the n ≥ 16 programme is unavailable on
this model from two directions at once, not one.

#### What this settles and what it leaves

**Settled:** the months results stand as the project's findings, with the limits already
recorded — cyclicity disconfirmed in full-vector distances (D32), the cylinder question
underpowered and open (D37), seam location unresolved (D32), hypothesis C structurally
unrunnable (D21).

**The next move is a different model, not a different family.** The requirement is a
model that can do shift-by-k across a full cycle of n ≥ 16 — which the brief's own
hardware note anticipated (`BRIEF` §7 suggests Llama-3.1-8B), and which D6 records this
8GB box cannot host at bf16. That is a hardware decision, not an experimental one.

---

## 2026-08-14 — The pivot to a measurement-validity study

All three entries are fixed **before any of the registered runs or statistics in
`docs/preregistration_instruments.md` exist**. That document carries the detail; these
entries record the decisions.

### D39. The geometry programme is closed; the contribution is measurement validity

D38 fired the "neither passes" branch: no second family runs on this model, and the
months results stand with their stated limits (cyclicity disconfirmed in full-vector
distances, cylinder question underpowered and open, seam location unresolved,
hypothesis C structurally unrunnable). **No new task families, no n=11 cylinder work,
no 8B run, no further geometry.** The months geometry becomes a case study.

The new contribution is a measurement-validity paper with one thesis —
interpretability validation practice is systematically permissive — carried by two
claims already measurable on this repo's artifacts: the accuracy gate passes while the
extracted vector encodes a different function (claim A;
`docs/note_operand_diversity.md`, D20), and estimators validated on synthetic fixtures
have false-positive floors ~100× higher on real activations (claim B; D37,
`docs/preregistration_family2.md` §1).

Tonight's session (~6 h, 16 GB M1, no CUDA) runs exactly the tasks registered in
`docs/preregistration_instruments.md`: T1 (behavioural, MPS fp16 — the sanctioned
path), and T3/T5/T6/T10 (numpy-only off saved `.npz`). Explicitly out of scope and
not started even if time remains: the injection sweep on MPS (bf16 FVs into an fp16
model is incomparable to the frozen L8 × 3.0 baseline), the pool grid (rule 4 → Ada
box), the standard-FV benchmark task (new head ID), and anything else not in that
document. Confirmatory for the paper: T1's branch decision and T6's floor table.
Everything else is supporting material.

### D40. T1 branches pre-committed before the off-diagonal cell is run

Every restricted-pool behavioural gate so far set `query_pool` = `operand_pool`, while
`efficacy()` scored the FV over the complete 12-month cycle — so claim A rests on a
gate and a margin scored on different query supports. T1 runs the missing cell:
restricted demonstrations, full-cycle queries (configs
`gate_months_partA4_fullq.json`, `gate_months_halves_A_fullq.json`; mechanics
verified against `prompts.py` — out-of-pool queries leave the demo pool intact,
held-out is automatic, `choices` stays the full cycle, chance stays 1/12).

Both branches are fixed in `docs/preregistration_instruments.md` §1 before either run
is launched, with the device-matched full-pool profile (`llama32_3b_mps_months_s16`)
as reference and an in-pool anchor check against the original bf16 gates as the
instrument-validity control:

- **survives strong** — full-cycle-query gate GO at every k, or MARGINAL only at k=8
  within noise of the full pool's own 0.38 → the gate still passes on the same
  support the margin is scored on, while the vector encodes a different function;
- **killed / rescoped** — NO-GO, or any cell the full pool clears at ≥ 0.50 drops
  below 0.50 beyond the two-proportion margin → the original GO verdicts were an
  in-distribution artifact; claim A weakens to "we gated on the wrong query support";
  the paper rescopes around claim B and the rescoping is recorded, not argued away.

A registered middle-case resolution (restricted-k margin recomputation) and a
cross-run rule (strong form needs both pool sizes; disagreement triggers the B-side
conditions before framing is fixed) are in the pre-registration. Neither run's
`scores.jsonl` is opened until the pre-registration is committed.

### D41. Rule-5 exception for floor estimation, registered rather than taken silently

T6's floor table characterises the diagnostics' false-positive behaviour on real
activation statistics, and for that purpose **vectors from tasks the model cannot
perform are the point, not a defect** — floor estimation needs vectors, not correct
vectors. CLAUDE.md rule 5 (and D20 §3) forbids extracting or using such FVs for task
claims, so the exception is registered explicitly:

- gate-failed condition vectors already on disk (partitions and halves, D20/D21) may
  serve as instrument-characterisation material in T6/T3;
- any future floor-only extraction from a NO-GO family runs at bf16 minimum, is
  labelled `floor_only` in metadata, and is categorically never cited for a claim
  about task-space geometry or the task itself;
- rule 5 is otherwise untouched.

Scope limit: these vectors characterise estimators. No geometry bucket, hypothesis
verdict, or task-competence claim may rest on them, tonight or later.

### D42. The prereg's T6 §5 conflated two synthetic constructions; both are run

**Recorded after a 20-draw timing pilot of `tarcle/floors.py` was seen, and before
the 300-draw table is computed or read.** The pilot exposed a defect in the
registration's wording, and the fix is recorded here rather than silently patched —
the instruments pre-registration, like the others, is never edited.

#### The conflation

`docs/preregistration_instruments.md` §5 defines C1 as both "the validation practice
the field uses and prereg §0 used" **and** "matched at the measured months regime
(offset share, axial share, residual RMS)". Those are different constructions:

- The project's actual fixture-grade practice is `line(11, d=64, step=1.0,
  offset=8.0, noise=0.3)` (`cylinder.py`, the D37 synthetic validation) and the
  prereg §0 battery at noise 0.3 — noise ≈ 1% of signal variance. That construction
  produced D37's 0.0026–0.0061 floors, the numbers claim B itself cites.
- A regime-matched construction sets the residual RMS to the measured value — on
  months, the [1,k] residual is comparable to the linear component — and the
  20-draw pilot shows it reproduces much of the real floor (harmonic partial R²
  synthetic ≈ 0.14 vs real ≈ 0.22 at 20 draws), i.e. matching second moments
  already closes most of the D37 gap for that statistic, while for others
  (toeplitz under an offset-matched isotropic cloud ≈ 0.85 vs real ≈ 0.14) the
  matched synthetic floor lands far **above** the real one — the simplex-trap
  direction.

#### Resolution, fixed before the full table

Both columns are computed and reported:

- **C1a — fixture-grade** (the registered C4 reading): N2 form is `cylinder.py`'s
  own line fixture; N1 form is the §0 `simplex(offset=0.7, noise=0.3)`, the
  registered unordered shape. d=64, noise 0.3, the repo's committed practice. All
  statistics in the table are scale- and dimension-invariant, so d=64 vs 3072 is
  the field's convention, not a distortion. **The pre-committed claim-B thresholds
  (median ratio ≥ 10 generalises; < 3 does not) apply to C4a = C2/C1a**, matching
  the construction whose numbers claim B quotes. C1a carries no gate; it is the
  reference being audited.
- **C1b — regime-matched** (exploratory decomposition): offset, axial scale and
  residual RMS matched per dataset. C4b = C2/C1b measures how much of the gap
  survives honest second-moment matching; the residual gap is attributable to
  real-activation structure beyond second moments. Reported alongside, cannot
  move the claim-B verdict.

#### Gate rework, same ordering

The pilot gated planted positives against the synthetic floors, which is the wrong
operative reference and voided rows for a reason the paper should instead report:
under the offset-matched isotropic null the toeplitz floor (0.85) sits above a
genuine planted line (0.49) — the synthetic construction cannot validate its own
detector there. Reworked rule, fixed now: a row is **VOID** iff its planted
positive fails to exceed the **C2 (real-permutation) floor** on months/todd, or C2
self-exceedance on fresh draws leaves [0.02, 0.10] at 300 draws. Planted-vs-C1b
and C1b calibration are reported per row as findings about the matched synthetic,
and do not void. Also noted: participation_ratio is exactly invariant under N1 row
permutation (row order does not enter the covariance spectrum), so its real-N1
"range" is degenerate at the observed value and the table reports its real range
under N2 instead.

### D43. T1 outcome: Branch 1 fires at both pool sizes — claim A survives in strong form

Read against the D40 / prereg §1 branches exactly as committed, after every other
task's output was already written up (the registered reading order). Artifacts:
`results/pilot/gate_months_{partA4,halves_A}_fullq/`,
`results/stage2/support_matrix.json`.

**Both off-diagonal runs return GO at every k on full-cycle queries.**
partA4 (4 operands): minimum cell k=10 at exactly 0.50, Wilson [0.40, 0.60] —
straddles the threshold and is flagged per the registered convention; next lowest
0.53. halves_A (6 operands): minimum k=8 at 0.54. The Branch 2 triggers fire
nowhere: no cell < 0.30, no cell with F(k) ≥ 0.50 below 0.50 at all, let alone
beyond the two-proportion margin. The registered middle case never engages.

**Instrument checks, both registered, both pass:** the in-pool query subsets
agree with the original CUDA bf16 in-distribution gates at every k (zero cells
beyond the two-proportion 95% margin, both runs), so the fp16/MPS instrument
carries the verdict unflagged.

**The out-of-pool subset — the genuinely new cell — is reported per k as
registered:** mostly 0.50–1.00 (partA4 weak cells: k=4 at 0.34, k=10 at 0.49;
halves k=8 at 0.26). The gate verdict reads the per-k aggregate over the uniform
full-cycle query distribution, as registered and as the field's own gate would.
A stricter out-of-pool-only gate was not registered and is not applied
post hoc; the split is published so a reader can apply it.

**Consequence.** The support-mismatch objection to claim A is closed by
measurement: the behavioural gate passes *on the same query support the FV
margin is scored on* (margins on disk: −0.315 to −0.944), at both pool sizes,
satisfying the cross-run rule. Sharper still, combined with T3: the model
demonstrably generalises restricted-demonstration shift-by-k to operands it
never saw in the demonstrations — the competence the gate certifies is real on
that support — while the vector extracted from the same prompts encodes
next-item even on the demonstrated operands. The failure is in the extraction,
not the behaviour, and no accuracy-based gate at any query support could see it.

### D44. The gpt2-CPU test guarantee is not certifiable on this M1 tonight

Recorded as an environment finding, not a methodology change. The torch-CPU
test files (`test_extract.py`, `test_pilot.py`) crash the interpreter with a
deterministic SIGBUS inside `torch.nn.Linear.forward` on this machine (M1,
macOS 14.2 / Darwin 23.2.0, torch 2.13.0 arm64) — in isolation, on an idle
machine, at multiple different tests. The failure predates tonight's changes by
construction: tonight's additions are five numpy-only stage-2 modules plus
their tests, none of which touch the crashing path, and the crashing files
arrived in the `7db8d95` pull already green on the Ada box that produced them.

Scope: the numpy-only suite passes here (26/26 in 1.5 s, including the new
`test_floors.py`); every tonight artifact is either numpy-only or the MPS
behavioural run, whose instrument was validated by its own registered in-pool
anchor check (D43), not by trust in the torch-CPU path. The CLAUDE.md rule-5
CI guarantee ("gpt2 CPU in < 5 min") remains certified only on the Ada box
until torch/macOS versions on this machine are reconciled — an infrastructure
decision for the user, not taken here.

### D45. Two integrity events: gate verdicts without artifacts, and a fixed-filename overwrite chain

Recorded together because the writing session's figure work exposed both, and an
append-only log records integrity events whether or not their repair changes any
verdict. One repair did change a verdict; it is stated plainly in §2.

#### 1. Gate verdicts recorded with no committed artifact — and one does not reproduce

D21, D24 and `docs/note_operand_diversity.md` record behavioural-gate GO for the
polysemy (9-operand) and partition-B (4-operand) conditions. **The complete git
history of `results/pilot/` additions contains no gate run for either condition,
ever.** They were never committed — not committed-and-lost; whether they were ever
executed on the Ada box cannot be determined from the repo, and no claim is made
either way.

Both were re-run 2026-08-14 (MPS fp16, the sanctioned behavioural path; configs
`gate_months_{partB4,polysemy}.json`, in-distribution query pools mirroring the
extraction conditions, n = 100/k, 16 shots, held-out, seed 0):

- **partition B (Sep–Dec): GO** — every k ≥ 0.54 (min at k=10). The recorded
  verdict reproduces.
- **polysemy (9): MARGINAL** — k=8 at **0.44** (Wilson ≈ [0.35, 0.54]), every
  other k ≥ 0.60. The unqualified GO **does not reproduce**. What the re-run
  shows instead is the primary's own signature: the worst cell is k=8, the
  family's registered weak cell (primary k=8 = 0.38, same device and dtype),
  with polysemy's 0.44 sitting between the primary's 0.38 and the halves' 0.87 —
  monotone in pool size at exactly that cell. The polysemy *margin* (+0.343) is
  untouched; only the gate column's label is at issue.

Consequence: any table quoting polysemy's gate must carry the same weak-cell
qualification the primary's D1 branch carries, or the gate convention must be
restated for both. **That choice changes the post's table and is the user's; the
writing pass halted at this point per the pre-committed stop condition.**

#### 2. The nextitem.py overwrite chain

`tarcle/nextitem.py` wrote to a fixed filename (`nextitem_<method>.json`), so
successive invocations destroyed each other's output. The committed history shows
five clobber events: `6c1c760` {half_a, half_b} → `12ffc04` {mixed_daysmonths} →
`0bad1ba` {unrelated} → `21eb3a5` {unrelated, addk}. **The six-months-conditions
version that `note_operand_diversity.md` cites as its artifact was never
committed at all.** No number was ever wrong — the margins were quoted in the log
at computation time and are deterministically regenerable — but the artifact
trail behind the note's central table was broken from the start.

Repairs, all verified: the file is regenerated as
`results/fv/nextitem_todd_0109c67f.json` (filename stamped with the SHA-256 of
the input run set) and reproduces the D20 table exactly; the canonical
machine-readable home of the six margins is `results/stage2/margin_split.json`,
which recomputes them independently from `efficacy_pred_shift` and is asserted
equal to 1e-9 by `tarcle/post_figure.py`; `nextitem.py` now stamps its output
name and refuses to overwrite differing content.

#### 3. Audit of every writer, per the same defect

- **Proven loss: `nextitem.py` only** (above; fixed).
- **Same defect, no loss occurred:** `ctest.py` (dest from `--a`'s parent;
  invoked exactly once in history), `headset_compare.py`, `heads_compare.py`
  (comparison written into run_b), and the argument-dependent fixed-path writers
  `stage2.py` (`root`, `--n-perm`), `power.py` (`--draws`), `floors.py`
  (`--draws`, `--seam-folds`) — committed finals preserved in git history.
  These await the same stamp-or-refuse guard; only `nextitem.py` was fixed,
  per the audit-first instruction.
- **Guarded already:** `pilot.py` and `extract.py` refuse to overwrite.
- **Fixed path but canonical content** (no content-varying inputs; overwrite =
  deterministic regeneration): `seam.py`, `cylinder.py`, `calibrate.py`,
  `measure_corr.py`, `margin_split.py`, `offset_audit.py`, `support_gate.py`,
  `post_figure.py`.

README's "results/ (never overwritten)" is true of stage-1 by construction and
was false of stage-2 in practice; its correction belongs to the halted prose
pass and is not made here.

### D46. Device ruled out: the recorded polysemy GO was simply wrong

The D45 §1 correction could have had an innocent instrument explanation: the
original restricted-pool gates ran CUDA bf16, the re-runs ran MPS fp16, and a
0.50→0.44 shift at one cell is the size a precision effect might plausibly be.
That alternative is now tested and excluded (`tarcle/device_check.py`,
`results/stage2/device_check.json`).

#### The check

Every cross-device pair that exists was compared cell by cell: halves_A and
partA4 have committed CUDA bf16 originals (n = 100/k) **and** MPS fp16
re-measurements of the identical in-distribution query population (the in-pool
subsets of the T1 full-cycle audit runs, n ≈ 50 and ≈ 33/k) — 24 paired cells
across all twelve k. **Zero cells fall outside the two-proportion 95% margin**;
the largest difference anywhere is 0.10 against a margin of 0.19 (partA4,
k=10). Coverage stated plainly: the primary has no CUDA pair — its original
gate *is* the MPS fp16 months pilot, which means the primary-0.38 /
polysemy-0.44 monotone pair at k=8 is same-instrument by construction — and
halves_B, partB4 and polysemy each exist on a single device.

**Verdict: the registered "device ruled out" branch fires.** The gate
instrument reproduces across devices everywhere it can be checked, and the
recorded polysemy GO was simply wrong — a verdict entered in the log with no
artifact behind it (D45 §1), contradicted by the first run that produced one.

#### What the correction does and does not touch

Polysemy (9 operands, margin **+0.343**) is a **healthy** condition. Every one
of the four **collapsed** conditions passes the behavioural gate unqualified at
worst cells 0.54–0.83, each now with a committed artifact behind it. The
correction lands on a healthy row and **strengthens claim A rather than
weakening it**: across all six conditions, the gate's only hesitations — the
primary's k=8 at 0.38 (the registered D1 weak-cell branch) and polysemy's k=8
at 0.44 — fall on the two vectors that were fine. No number in the margin
column moves. The post must not be readable as reporting a correction that
undercuts the headline, because it does the opposite.

#### Presentation decision (the writing session's table convention)

Gate columns in tables print **worst-cell numbers**, not GO/MARGINAL labels —
labels are threshold- and device-dependent, numbers are the finding, and the
figure already renders it this way. The practitioner-level verdict stays in
prose, stated once at the table: under the Todd protocol's own gating rule,
all four collapsed conditions pass unqualified while both healthy conditions
needed the weak-cell branch. Dropping the verdict entirely would evade the
step the paper is about.

Also under this entry: the D45 §3 stamp-or-refuse guard is now applied to
every writer on that list (`ctest.py`, `headset_compare.py`,
`heads_compare.py`, `stage2.py`, `power.py`, `floors.py`, plus `nextitem.py`
and `device_check.py` refactored onto the shared helper
`tarcle/results_io.py`). Identical regeneration is allowed; differing content
at an existing path is refused. The four artifacts written under pre-stamp
names (`ctest_todd.json`, two `sweep_comparison.json`, one
`headset_comparison.json`) remain in place as committed history; stamped names
cannot collide with them.

---

## 2026-08-19 — M1 session: hygiene, MPS injection validation, T2

### D47. Correction to D45 §2's clobber count, with the definition it lacked

D45 §2 says "five clobber events" while listing a four-commit chain, and the
figure is not derivable from any stated definition. D45 is not edited; this
entry states the census and the definition.

**Definition:** a clobber event is a write of `nextitem_todd.json` that
replaced differing content at that path.

**Census, from evidence:** the file has exactly **four committed versions**
(`6c1c760` {half_a, half_b} → `12ffc04` {mixed_daysmonths} → `0bad1ba`
{unrelated} → `21eb3a5` {unrelated, addk}), i.e. **three clobber events
visible in committed history**. At least **one further uncommitted clobber is
proven by content**: `6c1c760`'s decisions.md quotes the D20 margins for
primary/polysemy/partition conditions, which required nextitem invocations
whose outputs are absent from that same commit's `nextitem_todd.json` — they
were overwritten by the halves invocation before the commit. **Correct
statement: at least four clobber events, exactly three of them visible in
committed history; the true total is unknowable because uncommitted
invocations leave no trace.** D45's "five" overcounted by treating the
four-commit chain itself as four events and adding the inferred one; the
chain's four versions contain only three transitions.

### D48. Landing-map read, registered with its script before appearing in prose

A descriptive read used in post drafting is registered before it may appear in
public prose (house rule): the landing map of injected zero-shot predictions
for the four collapsed conditions. Script: `tarcle/landing_map.py` (guarded
writer, committed with this entry); definition in its docstring (Todd,
k ∉ {0,1,11}, full query cycle; classes: correct / successor / predecessor /
copy / fwd-2-8-excluding-correct / other).

**Expected values, stated before the committed-artifact run:** strictest
condition ≈ 95% adjacent; zero predictions 2–8 steps forward there. Match →
`results/stage2/landing_map.json` is the record and the numbers may enter the
post. Mismatch on any quoted figure → **STOP and report; no silent
reconciliation.**

### D49. Partition-B efficacy quote, registered with its script before appearing in prose

Same treatment for the second drafting read: ctl_months_partB / Todd, quoted
as injected accuracy mean 0.257 vs zero-shot baseline 0.083, logp_lift mean
+1.19. Script: `tarcle/efficacy_quote.py` (guarded writer, committed with this
entry). It computes all three means under both candidate definitions (all
twelve k; k ≠ 0 per the D2 ceiling caveat) and records which one the quote
corresponds to — the definition must be named wherever the quote is used.
Match under one definition → `results/stage2/efficacy_quote_partB.json` is the
record. Match under neither → **STOP and report.**

### D50. MPS injection validation — subset, tolerances and branches, registered before the run

CLAUDE.md rule 4 sanctions this M1 for behavioural runs only; it has never
been validated for interventions. T2 is injection-bearing, so before it may
run here, the frozen-protocol injection cells are re-executed on MPS fp16 and
compared against the committed CUDA bf16 values inside the `ctl_*` `.npz`
files. Script: `tarcle/mps_validation.py`, committed with this entry; run
after commit.

**Subset:** the four T2 conditions (partA, partB, halfA, halfB) plus the
primary (the +0.35 healthy anchor T2's control argument leans on), both
extraction methods, all twelve k, the full 12-query zero-shot cycle, at the
frozen protocols exactly as stored in each `.npz` (Todd: L8 × 3.0 add;
Hendel: L15 × 1.0 replace) — 120 (condition, method, k) cells plus MPS-side
baselines (lift is computed same-instrument on each side).

**Tolerances, fixed now:**

1. **Accuracy per cell:** |Δ| ≤ the two-proportion 95% margin at n = 12/12 —
   the behavioural cross-device regime (D46). At n = 12 this margin is wide
   (≈ 0.3–0.4), so this criterion alone cannot carry a PASS; it can only fail
   one.
2. **logp-lift per cell (the analogous tolerance the D entry must document):**
   |Δ lift| ≤ 1.96·√(SE²_cuda + SE²_mps), SEs over the twelve queries on each
   side (the D17 injected-side semantics both sides), devices treated as
   independent — conservative in the direction of flagging. **Pass requires
   ≥ 95% of the 120 cells within tolerance.**
3. **Decision statistic (what T2 actually reads):** per (condition, method),
   the D20 margin from `pred_shift` over mid-cycle k (108 paired predictions).
   |margin_mps − margin_cuda| must lie within the 95% CI of the paired
   per-prediction difference (1.96·sd(d_mps − d_cuda)/√108), **and** no
   condition may cross a ±0.10 D20 gate line in a way that changes its class
   (collapsed stays collapsed; the primary stays healthy).

**Branches:** PASS = all three criteria hold → the next entry extends rule 4
to injection-on-MPS **for T2 specifically** (this model, these saved FVs,
zero-shot injection scoring; not extraction, not geometry). FAIL = any
criterion fails → **T2 stops and waits for CUDA hardware**; the failure is
logged as its own entry and treated as a valid result about the instrument,
not a problem to engineer around.

### D51. MPS injection validation: FAIL. T2 waits for CUDA hardware.

Read against D50 exactly as registered
(`results/stage2/mps_validation.json`).

| criterion | result |
|---|---|
| 1 — accuracy per cell, two-proportion margin | pass (0 of 120 outside) |
| 2 — logp-lift per cell, paired SE tolerance | pass (120/120 within) |
| 3 — per-condition D20 margin, paired 95% CI + class stability | **FAIL** |

Criterion 3 fails on the **primary condition, both methods**: Todd margin
+0.352 (CUDA) → +0.269 (MPS), difference −0.083 against a paired CI of
±0.069; Hendel −0.343 → −0.407, difference −0.065 against ±0.059. Every gate
class is stable (the primary stays healthy at +0.269; all eight collapsed
cells stay collapsed, within CI — partB/todd reproduces prediction-for-
prediction, diff exactly 0). The four T2 conditions individually validate;
the anchor does not.

**The registered FAIL branch fires: T2 does not run on this machine and
waits for CUDA hardware.** The subset included the primary deliberately —
T2's control argument ("the full-pool vector earns +0.35 under the identical
protocol") leans on it — and the primary is exactly where the instrument
drifts. Re-scoping the subset after seeing which cell failed would be the
post-hoc move this log exists to prevent. Rule 4 is **not** extended;
injection-bearing work remains CUDA-only.

Two observations recorded with the verdict, not as mitigation:

- The drift is one-directional and lands where the vector carries fine
  structure: fp16 injection at ×3.0 weakens the healthy Todd margin and
  pushes the (already-confined) healthy-condition Hendel margin further
  negative, while the degenerate attractors are insensitive to precision.
  Steering *toward a task* degrades under fp16 here; steering *toward the
  default* does not. If T2 had run on this device, its rescue criterion
  (margin ≥ +0.10, CI clear of 0) would have been biased **against** rescue —
  the adversarial direction, but a biased instrument regardless.
- The per-cell tolerances (criteria 1–2) passed everywhere while the
  aggregate decision statistic (n = 108 paired predictions) caught the drift.
  One more instance of the project's thesis: the low-powered checks pass, the
  decision-relevant statistic disagrees, and which one you consult decides
  what you ship.

### D52. Second candidate instrument: bf16-on-MPS, submitted to the unchanged D50 gate

D51's FAIL concerned **fp16** injection — the dtype mismatch against the bf16
CUDA artifacts being the most plausible drift source. A probe (2026-08-20)
established that this stack supports **bfloat16 on MPS** (torch 2.13,
macOS 14.2, M1 / Metal bfloat; bf16 linear forward verified) and that the
bf16 model (6.4 GB) fits under the 10.7 GB MPS working-set ceiling (fp32, at
12.8 GB, does not and is excluded).

**Registered before the run:** `tarcle/mps_validation.py` is re-run with the
model at bfloat16, everything else identical — **the D50 subset, all three
tolerances, and the pass rule are frozen and reused unchanged.** This is the
same gate applied to a second candidate instrument, not dtype-shopping: D51's
fp16 verdict stands untouched whatever happens here, and no third dtype
exists on this hardware, so the candidate set is exhausted by this entry.
Output goes to its own guarded artifact
(`results/stage2/mps_validation_bfloat16.json`).

**Branches, pre-committed:**

- **PASS** (all three D50 criteria) → a follow-up entry extends rule 4 to
  injection-on-MPS **at bf16, for T2 specifically** (this model, these saved
  FVs, zero-shot injection scoring); T2 runs on this machine at bf16.
- **FAIL** → T2 is genuinely CUDA-only on the evidence of two instruments;
  no further engineering on this machine.
- **Instrument unavailable** (bf16 execution errors mid-run despite the
  probe) → treated as FAIL for T2 purposes, logged with the error.

### D53. bf16-on-MPS PASSES the D50 gate; rule 4 extended for T2; T2 launched

Read against D50's criteria exactly as registered
(`results/stage2/mps_validation_bfloat16.json`): criterion 1 pass (0 of 120
accuracy cells outside), criterion 2 pass (120/120 lifts within), criterion 3
pass (all ten condition/method margins within their paired 95% CIs, every
gate class stable). The cell that failed at fp16 — the primary anchor —
reproduces at bf16: Todd +0.352 → +0.324 (diff −0.028 vs CI ±0.048), Hendel
−0.343 → −0.352 (diff −0.009 vs ±0.018). The D51 diagnosis holds: the fp16
drift was the dtype mismatch against the bf16 CUDA artifacts, and it is gone
when the dtypes agree.

**The D52 PASS branch fires: CLAUDE.md rule 4 is extended to
injection-on-MPS at bfloat16, scoped to T2 specifically** — this model, the
saved `ctl_*` FVs, zero-shot injection scoring. Not extraction, not geometry,
not fp16 (D51 stands), not any future run that does not inherit this exact
instrument.

**T2 launch, registration-interpretation note recorded before the run:** the
`ctl_*` `.npz` files persist only the frozen-layer (L15) Hendel dummy-query
state, so the registered "layer-only sweep on the saved FVs, no
re-extraction" (prereg_t7 §2) necessarily means injecting that saved L15
state at each of the 28 layers. Per-layer Hendel states were never saved and
re-extracting them is outside T2's scope; this is the only executable literal
reading, and it is recorded here rather than discovered in review. Sweep:
`tarcle/t2_sweep.py --device mps --dtype bfloat16`, 224 hash-stamped
resumable chunks, grid run to completion, verdict read only by
`tarcle/t2_report.py` against the registered branches.

### D54. T2 outcome: OBJECTION CLOSED — no injection protocol rescues a collapsed vector

The sweep ran exactly as registered (prereg_t7 §2; D53 sanction: MPS bf16),
to completion, no early stops: 224/224 chunks, ~99 minutes wall
(`results/t2/`, hash-stamped; verdict `results/t2/t2_report.json`).
Internal-consistency anchor before reading: the sweep's frozen-protocol cell
(partA/todd, L8 × 3.0) reproduces the D52-validated margin exactly (−0.685).

**Registered branch fired: OBJECTION CLOSED.** Every condition/method
best-over-grid margin is < +0.10 — and the result is much stronger than the
threshold: the best cell anywhere is exactly **0.000** (weak-injection cells
where the copy prior dominates and predictions land on neither the correct
shift nor ±1), and **no cell in the entire space has a positive margin at
all** — Todd 0 of 560 (512 negative, 48 zero), Hendel 0 of 112 (66 negative,
46 zero). At no layer, at no scale, under either extraction method, does any
collapsed vector produce even one more correct-shift prediction than
adjacent-item predictions. The adversarial max never found anything to
maximise.

Per the registration, claim A's sentence gains its clause: no accuracy gate
at any query support could detect the collapse, **and no injection tuning
could rescue it** (applied to the post and README with this entry).

Scope carried, not hidden: instrument is MPS bf16 under the D53 sanction;
the Hendel arm injects the saved L15 state at each layer (D53 note); the
grid is the registered layer × scale space over averaged single-layer FVs —
distributed per-head injection (arXiv 2606.05079) is a different protocol,
out of T2's scope per the 2026-08-17 registration note, and this verdict
says nothing about it.

### D55. Post figure 2 (`tarcle/fig_diversity.py`, contributed) re-run against current artifacts

All assertions pass on the current repo state — diversity means from the
committed prompt files, all six margins from `margin_split.json` (to 5e-4),
all six gate worst-cells including the D45/D46 re-run values — output
`results/stage2/fig_diversity_92bf0dd7d6c1.{svg,png}` (content-hash-stamped).

---

## Conventions

- Entries are append-only. A superseded decision is struck through with a pointer to
  the entry that replaces it, never deleted.
- Every entry records what was known when it was fixed, and against which artifacts
  (commit, run name) it can be checked.
