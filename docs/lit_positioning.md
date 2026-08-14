# Literature positioning

**Epistemic status of this document.** A literature sweep was done outside the
repo (2026-08-14). Several entries post-date the assisting model's knowledge and
were seen only as search snippets; those are recorded as **leads, not facts**.
Badges: **[VERIFIED]** = within the assistant's knowledge and consistent with
the sweep, still re-read before submission; **[UNVERIFIED]** = snippet-level
lead only, **must be read in full before it is cited anywhere**. Claims in
UNVERIFIED entries are attributed to the sweep, not to the papers.

**The one positioning rule:** the thesis is pitched as **measurement, not
insight**. The insight — validation metrics substituted for identification —
is already published as a position (entry 1). Position papers need empirical
instances; this repo is one. The intro must not claim the framing.

---

## 1. The frame

**"Position: Mechanistic Interpretability Must Disclose Identification
Assumptions for Causal Claims" — arXiv 2605.08012 (May 2026). [UNVERIFIED]**
Per the sweep: names **"validation metric substitution"** — validation metrics
offered in support of causal claims without stating identification
assumptions — and cites Makelov et al. and Canby et al. as empirical evidence
that high validation scores co-exist with failed identification.
*Does not provide (per sweep):* new empirical instances; it is a position
paper. *Our sentence:* **we are a preregistered empirical instance of
validation-metric substitution — the gate passes, the identification fails,
and the actually-encoded function is identified (next-item) rather than
inferred.** Cite as the frame this paper instantiates. — *Locate and read
Canby et al. (no ID in sweep) while verifying this. [UNVERIFIED lead]*

## 2. Claim A — the four required engagements

**2.1 Hendel, Geva & Globerson 2023, "In-Context Learning Creates Task
Vectors" (arXiv 2310.15916), §4 "Robustness of Task Vectors". [VERIFIED —
highest priority; re-read §4 before submission.]**
*Claims:* 50 task vectors per task over varied demonstration sets S and dummy
queries x′ form clean per-task t-SNE clusters → task vectors are robust.
*Does not claim:* robustness under **restricted operand support** — the check
varied S drawn from a full operand pool. **Load-bearing detail to verify in
the paper's own setup before submission: that S is sampled from the full
pool.** *Our sentence:* **we identify the regime their robustness check did
not cover: restrict the operand pool and the vector stays exactly as "robust"
(split-half ≥ 0.99, stable, large causal effect) while encoding a different
function — consistency measures certify the stability of whatever got
encoded, never its identity.** This engagement shapes the abstract.

**2.2 Makelov, Lange & Nanda 2024, ICLR, "Is This the Subspace You Are
Looking For? An Interpretability Illusion for Subspace Activation Patching"
(arXiv 2311.17030). [VERIFIED — author-list caveat: the sweep lists "Makelov,
Lange, Geiger & Nanda"; the assistant's knowledge has Geiger on a separate
*reply* to this paper, not on it. Verify the author list before citing.]**
*Claims:* subspace activation patching can achieve the intended end-to-end
causal effect by activating a **dormant parallel pathway** fed by a causally
disconnected component — an interpretability illusion for patching-based
validation. *Does not claim:* anything about extraction-time gates or FVs.
*Our sentence:* **ours is the extraction analogue in the same genre — large
causal effect, passing validation, wrong object — with a different mechanism
(prompt-distribution-induced bias, not dormant pathways) and a different
object (an extracted vector, not a patched subspace).**

**2.3 Tan et al. 2024, NeurIPS, "Analysing the Generalisation and Reliability
of Steering Vectors" (arXiv 2407.12404). [VERIFIED — re-read for the exact
"anti-steerability" and answer-token-bias terminology.]**
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
Lens" — arXiv 2604.02608 (2026). [UNVERIFIED]**
Per the sweep: an FV dissociation between two measures (steerability vs
logit-lens decodability). *Our sentence (conditional on reading):* their
dissociation is between an intervention measure and a readout measure; ours is
between the **validation gate and the encoded function**, with the encoded
function identified. Also bears on T5's composition point.

**2.5 "From Compression to Expression" — arXiv 2505.17322. [UNVERIFIED]**
Per the sweep: a bias–variance decomposition of task vectors over
demonstration count. *Use, conditional on reading:* state precisely that our
collapse is a **bias** phenomenon, not a variance one — the collapsed vector
is stable across prompt draws (split-half ≥ 0.99) and systematically encodes
the wrong function; more draws of the same restricted distribution cannot fix
it. Adopt their decomposition language only after verifying it.

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
[UNVERIFIED — HIGH PRIORITY: per the sweep it attacks AIE per-head patching
directly on Llama-3.2-3B, our model. If it deprecates AIE, our AIE columns
(T5, D8/D15) need a sentence engaging it.]**

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

- **Eklund, Nichols & Knutsson 2016 (PNAS, "Cluster failure"). [VERIFIED]**
  Canonical neuroimaging result: parametric-null FPRs measured on real
  resting-state data; the field's methods were permissive. Our C4a column is
  the same move for interp diagnostics.
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
