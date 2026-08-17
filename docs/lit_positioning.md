# Literature positioning

**Epistemic status of this document.** A literature sweep was done outside the
repo (2026-08-14). Several entries post-dated the assisting model's knowledge and
were seen only as search snippets; those were recorded as **leads, not facts**.
A **verification pass on 2026-08-17** read the four load-bearing papers in full
(2605.08012, 2505.17322, 2604.02608, 2606.05079) and re-checked Hendel §4,
Makelov 2024 and Tan 2024; those entries now carry paper content, not sweep
content. Badges: **[VERIFIED]** = read and the claims below are the paper's;
**[UNVERIFIED]** = snippet-level lead only, **must be read in full before it is
cited anywhere**. Claims in UNVERIFIED entries are attributed to the sweep, not
to the papers.

**The one positioning rule:** the thesis is pitched as **measurement, not
insight**. The insight — validation metrics substituted for identification —
is already published as a position (entry 1). Position papers need empirical
instances; this repo is one. The intro must not claim the framing.

---

## 0. Verification status (pass completed 2026-08-17)

All three blocking items are **resolved**; none of the pre-committed
repositioning branches fired.

1. **arXiv 2605.08012 — resolved, positioning held.** It is a position paper
   with **no preregistered empirical instance of its own**, so the post's "we
   read our result as a preregistered empirical instance of exactly that
   failure" stands as written; the "a further instance" fallback is not needed.
   New load-bearing content: **Table 4** maps method → validation metric
   typically reported → identification assumption left untested, for four
   methods (activation patching, SAE steering, causal abstraction,
   probing + ablation). **FV extraction is absent.** We are the missing fifth
   row (§1).
2. **Hendel et al. 2023 §4 — resolved, load-bearing detail confirmed.** The
   varied demonstration sets S are drawn from **full operand pools**; the
   "regime their robustness check did not cover" framing survives and the
   abstract-level rewrite is off the table.
3. **arXiv 2606.05079 — resolved, no collapse in it.** No task-identity
   validation step and no extraction collapse; it replaces the AIE/injection
   machinery (distributed per-head injection beats the averaged single-layer FV
   by up to +0.156). So: one engagement sentence on the T5 AIE rows, plus one
   **protocol-scope** sentence in the post and in T2's registration (§3.4).
   Its Appendix G also settles a separate question: `next_item` and `prev_item`
   are named tasks in Todd et al.'s benchmark suite, so our collapsed vector
   returns a catalogue task rather than noise.

**Adjacent — skim before ARR submission, not load-bearing for the post.** None
of these carries a sentence in the post that would need rewriting if the skim
surprises us; each is a strengthening citation at most.

- **Canby et al. 2024** — causal probing reliability.
- **Davidson et al. 2025 (NeurIPS)** — instructions vs demonstrations elicit
  distinct heads. Bears on T7's head-identification assumption, not on claim A.
- **arXiv 2606.16867** — FVs/TVs fail on negation resolution. Efficacy axis,
  not faithfulness; same shelf as Tan et al.
- **arXiv 2605.08295** — in-context label-override regime.
- **Eklund, Nichols & Knutsson 2016 (PNAS, "Cluster failure")** — carries one
  clause in the claim-B section (§4); skim only if that clause stays.
- Also still adjacent: 2606.27510, the Tan follow-up lead ("CAA findings
  unverified for FVs"), Nullstrap, AxBench.

If any of these turns out to contain a preregistered instance like ours, it
promotes to blocking and the 2605.08012 treatment applies to it.

---

## 1. The frame

**"Position: Mechanistic Interpretability Must Disclose Identification
Assumptions for Causal Claims" — arXiv 2605.08012 (May 2026). [VERIFIED]**
Names **"validation metric substitution"** — validation metrics offered in
support of causal claims without stating identification assumptions — and cites
Makelov et al. and Canby et al. as empirical evidence that high validation
scores co-exist with failed identification.
*Does not provide:* new empirical instances; it is a position paper, with **no
preregistered empirical instance of its own**. *Our sentence:* **we are a
preregistered empirical instance of validation-metric substitution — the gate
passes, the identification fails, and the actually-encoded function is
identified (next-item) rather than inferred.** Cite as the frame this paper
instantiates. — *Canby et al. moved to §0's adjacent list.*

**The Table 4 row we supply.** Their Table 4 pairs, for each of four methods,
the validation metric typically reported with the identification assumption
that metric leaves untested: activation patching, SAE steering, causal
abstraction, probing + ablation. **FV extraction does not appear.** State our
contribution as exactly that missing row, in their columns:

| column | our row |
|---|---|
| method | FV extraction from ICL demonstrations |
| validation metric typically reported | behavioural accuracy gate + reliability (split-half) + causal effect |
| identification assumption left untested | **demonstration-distribution sufficiency** — that the demonstration distribution identifies the intended function |
| why the metric does not test it | the model can perform the task on any query support, while the extraction inherits the distribution's degenerate default; competence and identification come apart and accuracy only sees the first |

This is the strongest available form of the positioning rule at the top of this
doc: the framing is theirs, the row is ours, and the row is empirical.

## 2. Claim A — the four required engagements

**2.1 Hendel, Geva & Globerson 2023, "In-Context Learning Creates Task
Vectors" (arXiv 2310.15916), §4 "Robustness of Task Vectors". [VERIFIED — §4
read; the load-bearing detail is confirmed.]**
*Claims:* 50 task vectors per task over varied demonstration sets S and dummy
queries x′ form clean per-task t-SNE clusters → task vectors are robust.
*Does not claim:* robustness under **restricted operand support** — the check
varied S drawn from a **full operand pool**, which is the detail the post's
framing rests on and which the verification pass confirmed in the paper's own
setup. *Our sentence:* **we identify the regime their robustness check did
not cover: restrict the operand pool and the vector stays exactly as "robust"
(split-half ≥ 0.99, stable, large causal effect) while encoding a different
function — consistency measures certify the stability of whatever got
encoded, never its identity.** This engagement shapes the abstract.

**2.2 Makelov, Lange & Nanda 2024, ICLR, "Is This the Subspace You Are
Looking For? An Interpretability Illusion for Subspace Activation Patching"
(arXiv 2311.17030). [VERIFIED — author-list caveat resolved: cite as Makelov,
Lange & Nanda; Geiger is on a separate *reply* to this paper, not on it, which
is where the sweep's four-author listing came from.]**
*Claims:* subspace activation patching can achieve the intended end-to-end
causal effect by activating a **dormant parallel pathway** fed by a causally
disconnected component — an interpretability illusion for patching-based
validation. *Does not claim:* anything about extraction-time gates or FVs.
*Our sentence:* **ours is the extraction analogue in the same genre — large
causal effect, passing validation, wrong object — with a different mechanism
(prompt-distribution-induced bias, not dormant pathways) and a different
object (an extracted vector, not a patched subspace).**

**2.3 Tan et al. 2024, NeurIPS, "Analysing the Generalisation and Reliability
of Steering Vectors" (arXiv 2407.12404). [VERIFIED — re-checked, including the
"anti-steerability" and answer-token-bias terminology as used below.]**
*Claims:* steering vectors (CAA) are unreliable — high in-distribution
variance, anti-steerable examples, spurious answer-token biases; efficacy is
fragile. *Does not claim:* anything about what a *reliable* vector encodes.
*Our sentence:* **theirs is an efficacy axis (vectors that fail to steer);
ours is a faithfulness axis (vectors that steer reliably — split-half ≥ 0.99,
large effect — and implement a different function): the failure their metrics
would score as success.** — *Citable gap flagged by the sweep: follow-up work
states CAA findings are unverified for Function Vectors. Locate that
follow-up. [UNVERIFIED lead]*

**2.4 "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit
Lens" — arXiv 2604.02608 (2026). [VERIFIED]**
An FV dissociation between two measures (steerability vs logit-lens
decodability) — noting that their "FVs" are mean-of-differences contrastive
vectors (CAA-style), not Todd causal-head averages. *Our sentence:* their
dissociation is between an intervention measure and a readout measure; ours is
between the **validation gate and the encoded function**, with the encoded
function identified. Also bears on T5's composition point.

**2.5 "From Compression to Expansion" — arXiv 2505.17322. [VERIFIED]**
*(Title is "Expansion", not "Expression" — the sweep had it wrong.)* A
bias–variance decomposition of task vectors over demonstration count.
**Theorem 5.1** proves that the bias *and* the variance of the task vector both
decay O(1/K) in the demonstration count K — **under i.i.d. demonstrations drawn
from the task distribution**. **Appendix E** shows the guarantee breaking when
the demonstration distribution is degenerate (their repeat mode).
*Our sentence, no longer hedged:* restricting the operand pool **changes the
demonstration distribution itself**, so their convergence result still holds
and simply points elsewhere — extraction converges at O(1/K) to the *wrong*
limit, and more shots converge faster to the wrong vector. This is the formal
home for our bias-not-variance claim: the collapsed vector is stable across
prompt draws (split-half ≥ 0.99) and systematically encodes the wrong function,
and no amount of additional sampling from the same restricted distribution can
fix it.

## 3. T5 — occupied ground; claim only the narrow thing

**3.1 Zhang & Nanda 2024, "Towards Best Practices of Activation Patching in
Language Models: Metrics and Methods" (arXiv 2309.16042). [VERIFIED]**
*Claims:* patching conclusions vary substantially with metric and method
choices (corruption type, probability vs logit-diff vs KL, layer conventions).
**3.2 "Transformer Circuit Faithfulness Metrics are not Robust" — arXiv
2407.08734. [VERIFIED — re-read.]** *Claims:* circuit faithfulness scores
swing with ablation and experimental choices.
**3.3 "The Curse of Multiple Mediators" — arXiv 2606.27510 (2026).
[UNVERIFIED]** Per the sweep: attributes faithfulness-score instability to
interaction-effect variance.
**3.4 "Fast & Faithful Function Vectors" — arXiv 2606.05079 (2026).
[VERIFIED]** Attacks AIE per-head patching directly on Llama-3.2-3B, our model.
*Contains:* no task-identity validation step and no extraction collapse, so
claim A is not preempted. *What it does show that we must engage:*
**distributed per-head injection outperforms the averaged single-layer FV by up
to +0.156.** Two consequences, both stated rather than argued:
- **T5's AIE rows** get one engagement sentence — our AIE columns use the
  estimator they improve on, and the T5 claim (composed measures disagree) does
  not depend on AIE being the best available estimator.
- **Protocol scope for claim A** (also written into `preregistration_t7.md` §2
  and the post's scope section): a reviewer may ask whether the collapse is an
  artifact of averaged single-layer injection. The control is that the
  full-pool vector earns +0.35 under the identical protocol, and T2's sweep
  covers the protocol axis we froze.
*Separately, its Appendix G settles the benchmark-catalogue question:*
`next_item` and `prev_item` are **named tasks in Todd et al.'s own suite**, so
the collapsed months vector does not degrade into noise — it returns a
different task from the same catalogue. Used in the post where the adjacency
prior is first named, and in the TL;DR.

*The narrow novel claim T5 is allowed to make, against all four:* **not**
"importance measures are unstable" (occupied) but: *the two measures the FV
literature routinely **composes** into a single validation argument — task
accuracy and steering efficacy — are statistically unrelated across tasks
within a family (ρ = 0.02 against a noise ceiling of 0.93), and even two
readouts of the same injections disagree (ρ = 0.12).* Disagreement between
composed measures on the same objects with noise ceilings, not instability
within one estimator.

## 4. Claim B — import an established discipline, do not reinvent it

Data-derived nulls are mature practice elsewhere; presenting them as novel
would look naive. Frame: **importing an established discipline into mech-interp
validation, with the C4a/C4b table as the measurement showing why it is
needed.**

- **Eklund, Nichols & Knutsson 2016 (PNAS, "Cluster failure"). [ADJACENT —
  §0 skim list; not load-bearing. Badged [UNVERIFIED] in the post, which is
  where it carries its single clause.]** Canonical neuroimaging result:
  parametric-null FPRs measured on real resting-state data; the field's methods
  were permissive. Our C4a column is the same move for interp diagnostics.
- **Permutation inference (Nichols & Holmes 2002; Winkler et al. 2014).
  [VERIFIED]** The standard data-derived-null machinery our perm_z / residual
  permutation rows instantiate.
- **Knockoffs (Barber & Candès 2015). [VERIFIED]** Data-derived negative
  controls for FDR in genomics/statistics.
- **"Nullstrap-style calibration" (genomics). [UNVERIFIED — the sweep names
  the style; locate the exact reference before citing.]**
- **Hewitt & Liang 2019, "Designing and Interpreting Probes with Control
  Tasks". [VERIFIED — engage carefully:]** interp-adjacent precedent for
  data-coupled baselines in *probing*. Our claim must therefore be "absent in
  mech-interp **causal/geometric** validation", not "absent in
  interpretability".

## 5. Prior art we may have missed — searches still to run

1. "function vector" / "task vector" × {operand diversity, demonstration
   diversity, pool size, degenerate, collapse} — any direct precedent for the
   pool-size collapse.
2. Task recognition vs task learning in ICL (Pan et al. 2023 and successors) —
   whether restricted pools flip ICL into recognition mode; the behavioural
   mirror of our bias story.
3. Todd et al. follow-ups (FV arithmetic/composition) — does any gate on
   encoded-function identity rather than accuracy?
4. Steering-evaluation benchmarks 2025–26 (AxBench and successors) — where
   efficacy-vs-faithfulness sits in current eval practice. [lead:
   AxBench, Wu et al. 2025 — verify]
5. The "illusion" genre inventory: Bolukbasi et al. 2021 (BERT interpretability
   illusion) → Makelov et al. 2024 → ours; check for other members, especially
   any extraction-time instance that would precede claim A.
6. Statistical testing / null models for activation geometry in interp
   (anything already doing residual-permutation floors).
7. Genomics: exact Nullstrap citation; Clipper-style FDR with data-derived
   nulls, for §4's citation block.
