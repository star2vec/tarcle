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

---

## Conventions

- Entries are append-only. A superseded decision is struck through with a pointer to
  the entry that replaces it, never deleted.
- Every entry records what was known when it was fixed, and against which artifacts
  (commit, run name) it can be checked.
