# A function vector can pass every standard validation check and encode a different function

*Draft for LessWrong / Alignment Forum. All numbers from preregistered runs on
saved artifacts; repo link at the end. Citation badges: [VERIFIED] = checked
against the paper; [UNVERIFIED] = search-snippet lead, will be read before
this posts.*

**Epistemic status:** preregistered before each run (branches committed in
advance, decision log append-only, D1–D44); one model (Llama-3.2-3B), one task
family (shift-by-k on months, Z/12) plus two reference families; one prompt
format. Generality to standard FV benchmark tasks is registered (T7) and not
yet run. We report one result below that killed our own pre-registered
headline, because that is how the method is supposed to work.

---

## The finding

Take Todd-style function vectors (causal-head averaging) for shift-by-k on
months: `Q: March\nA: June` for k = 3. Extract them from prompts whose
demonstration operands are restricted to a subset of the cycle — 4 or 6 months
out of 12 — exactly the kind of restriction any operand-partition control or
closed benchmark task produces. Then run every validation check standard
practice has:

- **Behavioural gate** (the Todd et al. protocol: extract only from tasks the
  model performs): held-out 16-shot ICL accuracy, **GO at every k** — and not
  just on in-distribution queries: on the **full 12-month query cycle**,
  including operands never seen in the demonstrations, the worst cell is 0.50
  and the 6-operand run's worst cell is 0.54, *above* the full-pool run's own
  0.38 at the same cell.
- **Reliability**: split-half cosine of the extracted vector ≥ 0.99.
- **Causal effect**: injection moves zero-shot predictions massively.

Every check passes. **The vector encodes next-item, not shift-by-k.** Injected
into zero-shot prompts and scored over the full cycle, P(prediction lands on
the correct shift) − P(prediction lands on shift ±1) runs **−0.32 to −0.94**
across the four restricted conditions (it is +0.35 for the full-pool vector).
At the worst cell the collapse is total: **every** in-pool mid-cycle
prediction lands on the adjacent month (margin −1.00).

| demo pool | gate, in-dist queries | gate, full-cycle queries | FV margin |
|---|---|---|---|
| 12 months | GO | GO (same thing) | **+0.35** |
| 9 months | GO | — | **+0.34** |
| 6 months (Jan–Jun) | GO, every k ≥ 0.83 | **GO, min 0.54** | **−0.32** |
| 4 months (Jan–Apr) | GO, every k ≥ 0.50 | **GO, min 0.50** | **−0.70** |

*(4- and 6-operand B-side conditions: margins −0.94 and −0.46; full tables in
the repo.)*

**[FIGURE: two curves over operand-pool size {4, 6, 9, 12} — held-out gate
accuracy (flat, at GO everywhere) and FV task-encoding margin (falling off a
cliff between 9 and 6). One picture, the whole point: the two measures move in
opposite directions.]**

The part that closes the obvious objection: maybe the gate passed only because
it was scored on easy in-distribution queries? We ran the missing cell of that
2×2 (restricted demonstrations × full-cycle queries) with branches
pre-committed before launch. The model **genuinely generalises**
restricted-demonstration shift-by-k to operands it never saw demonstrated —
out-of-pool accuracy is mostly 0.5–1.0. The competence the gate certifies is
real, on the very query support the FV margin is scored over. And the vector
extracted from those same prompts steers to next-item **even on the
demonstrated operands** (in-pool margins −0.13 to −1.00). The failure is in
the extraction, not the behaviour — which is why **no accuracy gate, at any
query support, could detect it.**

## Mechanism, as far as we measured it

The driver is **operand diversity of the demonstration distribution**, a
property accuracy does not see. Sixteen shots drawn from a 4-month pool
contain ~3 distinct operands; the in-context task gets *easier* (fewer
candidates, better coverage — k=8 accuracy rises from 0.38 to 0.87 as the pool
shrinks) while the identification of the function gets *worse*, and below a
threshold between 9 and 6 distinct operands the extracted vector snaps to the
family's degenerate default (the adjacency prior). The failure is **silent**:
the collapsed vector is well-formed, maximally reliable, causally potent. And
it is a **bias** phenomenon, not a variance one — stable across prompt draws,
systematically the wrong function; more samples of the same restricted
distribution cannot fix it. *(A 2025 bias–variance decomposition of task
vectors over demonstration count [UNVERIFIED: arXiv 2505.17322] may give this
a formal home; we will verify before positing.)*

## Why existing checks don't cover this

- **Hendel et al. 2023 §4** [VERIFIED] ran the robustness check this genre
  relies on: 50 task vectors per task over varied demonstration sets and dummy
  queries, clean per-task t-SNE clusters, conclusion: robust. The varied sets
  were drawn from **full operand pools**. Restrict the pool and every
  consistency measure still passes — consistency certifies the stability of
  whatever got encoded, not its identity.
- **Makelov, Lange & Nanda 2024** [VERIFIED] named the patching version of
  this genre: intended causal effect via a dormant parallel pathway. Ours is
  the **extraction analogue** — passing validation, large effect, wrong
  object — with a different mechanism (distribution-induced bias) and a
  different artifact (the vector itself).
- **Tan et al. 2024** [VERIFIED] documented steering vectors that fail to
  steer — an **efficacy** axis. Ours steer perfectly and lie about what they
  steer toward — a **faithfulness** axis. A reliability metric scores our
  failure as success.
- A May 2026 position paper [UNVERIFIED: arXiv 2605.08012] names the general
  pattern "validation metric substitution" and calls for identification
  assumptions to be disclosed. We read our result as a preregistered empirical
  instance of exactly that failure, in the most standard extraction protocol
  in the FV literature.

## We killed our own second headline, and you should trust the rest more for it

We also claimed (from an earlier failure, D37): *estimators validated on
synthetic fixtures have false-positive floors ~100× higher on real
activations.* We pre-committed a test — median ratio of real-data floor to
fixture floor across eight diagnostics, ≥10 generalises, <3 kills it — and
built the full table (300 draws per cell, validation gate first).

**The test returned 0.79. The blanket claim is dead.** What the table actually
shows is less quotable and more useful:

| statistic | fixture floor | real floor | ratio |
|---|---|---|---|
| harmonic partial R² | 0.0019 | 0.28–0.30 | **145–156×** |
| circulant score (raw) | 0.019 | 0.09–0.11 | 4.5–6× |
| permutation z | 1.90 | 2.15–2.18 | **1.1×** |
| toeplitz score | 0.65 | 0.14–0.16 | **0.2×** |
| seam cyclic cv-R² | 0.41 | ~0.004 | **~0.01×** |

Fixture-grade validation mis-states real false-positive floors **in both
directions** — understated 150-fold for one statistic (that one cost us a
wasted experiment, D37), *overstated* up to 100-fold for another (which
silently destroys power instead), and the direction is not predictable from
the statistic's form. Matching the surrogate's second moments to the data
closes most of the gap (the harmonic ratio drops from ~145× to 2×) but still
leaves errors of up to ~50× in the overstatement direction. The only row that
transfers at ratio ≈ 1 is the one whose null is generated from the data
itself. Data-derived nulls are boring, mature practice in genomics and
neuroimaging (permutation inference; the *Cluster failure* result is the same
audit for fMRI); mech interp validation mostly doesn't use them. Import the
discipline; it is cheaper than the fixtures it replaces. Two objections we
raise against ourselves — including that our best transfer row is VOID under
our own validation gate, and why the ratio is still readable — are labelled in
the findings doc rather than smoothed over.

## One more measurement: the checks don't even correlate

Across the eleven performable shifts, with attenuation ceilings computed from
each measure's own noise before reading any correlation: held-out accuracy
and causal AIE agree (ρ = 0.91–0.98). **Injection efficacy — the other half of
the standard "task is performable + vector steers" validation pair — is
statistically unrelated to accuracy across tasks: ρ = 0.02 against a noise
ceiling of 0.93.** Two readouts of the *same* injections (argmax accuracy vs
mean log-probability lift) correlate at ρ = 0.12. Prior work has shown
patching results are sensitive to metric and method choices [VERIFIED:
2309.16042, 2407.08734]; the narrow thing we add is that the two measures the
FV literature *composes into a single argument* are unrelated across tasks
within a family.

## Alignment relevance

Causal efficacy is not semantic faithfulness. Every deployment story for
activation steering — as a control primitive, as a monitoring probe, as a
"we checked the model with a task vector" audit — gates on some combination of
behavioural validation, reliability, and effect size. The object this post
exhibits passes all three while implementing a function other than its label,
and it arises from an ordinary, innocent-looking property of the data used to
build it. If steering vectors become control or monitoring infrastructure, a
reliable, high-effect, mislabelled vector is the failure class that matters:
nothing in the deployment gate fires, and the intervention does something
adjacent to — but not — what its name says.

## Scope, and what would change our minds

One model, one cyclic task family (plus add-k and twelve unrelated tasks as
references), one prompt format, Todd-style extraction for the headline (the
Hendel-style dummy-query extraction cannot steer mid-cycle shifts at all on
this model — a registered limitation reported throughout). The degenerate
default here is the adjacency prior; the general claim is that accuracy-gating
misses *some* degenerate default, not that it is always next-item. Two
registered follow-ups will test the load-bearing generalisations
(`docs/preregistration_t7.md`): **T7**, the same audit on a standard FV
benchmark task with a closed operand set — if margins hold up there, claim A
narrows to shift-like families and we will say so; and **T2**, per-condition
injection tuning — every margin above sits at one frozen injection protocol,
and if a collapsed vector steers correctly at some other layer × scale, the
claim weakens to "collapse at the shared protocol." Preregistrations,
decision log, and every artifact (with prompt hashes and commits) are in the
repo: **[REPO LINK]**.
