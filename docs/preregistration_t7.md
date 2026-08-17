# Pre-registration: T7 (standard-task generality) and T2 (injection-protocol robustness)

**Status: registered 2026-08-14, before any T7 or T2 run exists. Nothing in
this document has been executed.** Both tasks are the load-bearing untested
generalisations behind claim A (`docs/instruments_findings.md` §5–6,
README "Not established"). They run on the Ada box (CUDA, bf16 — rule 4;
the saved FVs they extend are CUDA bf16), not on the M1. Decision-log
numbering continues from D45 **when they run**; this registration adds no
decision entries by itself.

Both tasks inherit tonight's lessons as requirements, not options:
behavioural gates are scored on the **full query support** from the start
(T1 lesson, D40/D43); the task-encoding gate (D20) is measured for every
condition; any detector output is read only after a **data-derived floor**
for it exists (D37/D42 lesson); branches are written here, before any number.

---

## 1. T7 — does the collapse exist on a standard FV benchmark task?

### Why this gates the paper's scope

Claim A is measured on one shift family whose degenerate default is the
adjacency prior. `docs/note_operand_diversity.md` conjectures the risk
concentrates in "task families with few distinct operands, which describes a
substantial fraction of standard FV benchmark tasks". That sentence is
currently a conjecture. T7 tests it on a task of the standard type.

### Design

- **Candidate tasks**, in registered preference order: **antonym**,
  **country→capital** (the two named canonical FV tasks), with **currency**
  and **english_french** (already in `tarcle/tasks_unrelated.py`, D23) as
  fallbacks. Selection is by gate, not preference:
  - **(a) behavioural gate** — held-out 16-shot accuracy ≥ 0.70 on the full
    pool (chance = 1/|answer set|);
  - **(b) measurability gate (the D2/D23 ceiling lesson)** — zero-shot
    baseline < 0.50, else injected predictions cannot carry task-encoding
    evidence. D23 measured capital at 0.67–1.00 zero-shot, so country→capital
    is *expected* to fail (b); it is piloted anyway because the expectation
    is itself unverified on this prompt format.
  - The first task in preference order passing both gates is THE task. If
    none passes, T7 is blocked-by-gate and reported with its numbers.
- **The degenerate default is named in advance, per query**, generalising ±1:
  `default(q)` = the model's **zero-shot argmax** for query q — what the model
  does with no task signal. Queries where `default(q)` equals the correct
  answer are uninformative and excluded, exactly as k ∈ {0, 1, 11} were
  (their count is reported; if fewer than 10 informative queries remain, the
  task fails gate (b) retroactively and T7 falls to the next candidate).
- **Margin** (the D20 statistic, general form): over informative queries,
  P(injected argmax = correct) − P(injected argmax = default(q)).
- **Manipulation**: operand pools of size {full, 12, 6, 4} drawn from the
  task's closed pair list (seeded), demonstrations restricted to the pool,
  **queries always the full operand set**, held-out stratum, n = 100 per
  condition, 16 shots. Both extraction methods. Head identification is run
  fresh for the task (Todd protocol, ~1.2 GPU-h) — head sets do not transfer
  across families by assumption.

### Pre-committed branches

- **A generalises**: at some restricted pool size the behavioural gate (full
  query support) returns GO while the margin is ≤ −0.10 with its 95% CI clear
  of +0.10. The paper's claim A extends beyond shift families; the abstract
  may say "standard FV benchmark task".
- **A is family-specific**: margins stay ≥ +0.10 (CI clear of −0.10) at every
  pool size that passes its gate. Claim A narrows explicitly to families with
  a strong sequential/associative default, the paper says so in the abstract,
  and the note's "substantial fraction of standard tasks" sentence is
  withdrawn.
- **Ambiguous**: margins in between or CIs straddling — reported per pool
  size with no verdict; the abstract keeps the months-only scope.
- **Blocked-by-gate**: as above; reported with numbers, no substitution of a
  weaker check.

Power, fixed now: with ≤ 24 informative queries the margin CI at n = 24 is
≈ ±0.2, so only large collapses (|margin| ≳ 0.3, the size observed on months)
are detectable — a null here is weak evidence of absence and must be reported
as such.

---

## 2. T2 — is the collapse an artifact of the frozen injection protocol?

### The objection this closes

**Currently the top remaining mechanical objection to claim A.** Every margin
on disk is measured at L8 × 3.0 — the protocol chosen by D12 on the *primary*
condition's in-sweep k. A reviewer can say: the restricted-pool FVs might
encode shift-by-k perfectly well and merely need a different injection layer
or scale; the negative margins measure protocol mismatch, not encoding.

**Scope note added 2026-08-17 (verification pass).** *Fast & Faithful Function
Vectors* [VERIFIED: arXiv 2606.05079] shows distributed per-head injection
outperforming the averaged single-layer FV by up to +0.156, so the objection has
a published form: collapse could be an artifact of averaged single-layer
injection. Our control is that the full-pool vector earns +0.35 under the
identical protocol, and T2's sweep is the axis we froze. Stated, not argued;
T2's pre-committed branches below are unchanged by it, and the sweep remains
layer × scale on the averaged FV (distributed per-head injection is a different
extraction-and-injection protocol, out of T2's scope and not substituted for
it).

### Design

For each gate-failed condition (partA, partB, halfA, halfB) and each
extraction method, sweep the full 28-layer × {0.5, 1, 2, 3, 4} grid **on that
condition's own saved FVs**, scoring the D20 margin (full query cycle) at
every grid point. No re-extraction; this is injection-time only.

### Pre-committed reading

The report statistic is each condition's **best margin over its own grid** —
deliberately adversarial to us, since maximising over 140 cells biases toward
rescue. Therefore a rescue must clear noise:

- **Objection sustained (claim A weakens)**: any condition reaches margin
  ≥ +0.10 with the 95% CI at that grid cell clear of 0. Claim A is restated
  as "collapse under the shared protocol"; the paper must say the gate and
  the protocol jointly fail rather than extraction alone, and the abstract
  changes accordingly.
- **Objection closed (claim A hardens)**: every condition's best-over-grid
  margin stays < +0.10. The vector encodes the default at every injection
  protocol available to it, and the "no accuracy gate could detect it"
  sentence gains "and no injection tuning could rescue it".
- **Ambiguous**: best margins ≥ +0.10 but CI straddling 0 — the specific
  cells are re-scored with more queries per D17 (per-query logp with SEs)
  before any verdict; if still straddling, reported with no verdict.

Hendel FVs stay at scale 1.0 (replacement semantics, D12) — their T2 grid is
layer-only, and the registered ±1-confinement limitation is reported
alongside whatever it shows.

---

## 3. What this registration does not promise

That either result lands in a convenient branch. T7's null is underpowered by
construction (stated above); T2's adversarial maximum may rescue a condition,
in which case claim A's strongest sentence is deleted, not defended. The
months results and tonight's instrument findings stand either way, with their
stated scopes; these two runs decide only how far the claims travel.
