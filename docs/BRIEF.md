# Research brief: Is task space geometrically structured the way value space is?

*Handoff document. Written to be read cold by a session with no prior context.*

---

## 1. The core question

All existing work on non-linear representation geometry in LLMs concerns how models
represent **data**: days of the week on circles (Engels et al., *Not All Language Model
Features Are One-Dimensionally Linear*), integers on helices (Kantamneni & Tegmark),
character counts on rippled low-dimensional manifolds (Gurnee et al., *When Models
Manipulate Manifolds*, arXiv:2601.04480).

Function vectors / task vectors (Todd et al. 2024, arXiv:2310.15213; Hendel et al. 2023)
are the main evidence that models also represent **operations** as manipulable objects in
activation space.

Nobody has asked whether operation-space has the same geometric richness as value-space.

**Question:** does the linear representation hypothesis hold for representations of
operations, or only for representations of values?

---

## 2. Framing

A single task vector cannot be "circular" — circularity is a property of a *family*. The
setup requires a parameterised family of tasks whose parameter space has cyclic structure,
then asks whether `k -> FV(T_k)` traces a circle.

**Candidate task families (all with genuine wraparound):**

| Family | Group | Notes |
|---|---|---|
| shift-by-k on days of week | Z/7 | prime; all non-trivial irreps faithful |
| shift-by-k on months | Z/12 | composite — subgroup structure is informative |
| ROT-k Caesar cipher | Z/26 | more sample points; models are decent at this |
| timezone offset | Z/24 | composite |
| musical transposition | Z/12 | probably too rare in corpora |

**Three sub-questions, nested:**

1. **Geometry** — does a cyclically-structured task family yield cyclically-structured task
   representations?
2. **Separability** — is task encoding distinguishable from operand encoding at all, or
   does FV extraction smear them together?
3. **Algebra** — is FV a homomorphism? Does `phi(T_j o T_k) = phi(T_j) (+) phi(T_k)` for
   *some* operation `(+)` in activation space? Addition => linear. Rotation => circular.
   Nothing works => tasks are stored by lookup, not composed.

(3) is the deepest and yields a result either way.

---

## 3. Competing hypotheses

**A — Task vector lies on a circle.**
The shift is encoded cyclically; `FV(k)` traces a closed loop.
Predicts: `FV(n) ~ FV(0)`; constant norms across k; `FV(1) + FV(1) != FV(2)`.

**B — Linear selector for a rotation implemented elsewhere.**
The residual-stream task signal is a near-orthogonal code that gates *which* attention
heads perform the rotation via QK/OV. Task encoding linear, operation circular, no
relationship between them. This is what the Clock algorithm looks like if you only
inspect the control signal.

**C — FV extraction is a lossy projection of an operator.**
If tasks compose as a group, a task's natural type is an operator, not a vector. Rotation
is not implementable by vector addition — `h -> h + v` cannot rotate, since the correct
displacement depends on where you started. But FVs work precisely by addition.
Predicts: **operand-dependent steering efficacy**. Extract FV from demonstrations where
the operand is mostly Wednesday, apply to Monday, expect degradation or wrong-region push.

C is the hypothesis with the most teeth.

---

## 4. Prior work: what has and hasn't been done

Todd et al. have an appendix on cyclic tasks (Appendix L), but for a different purpose:
they use cyclic tasks (antonyms, next-item, previous-item) as a *counterexample* to the
claim that FVs are simple word-vector offsets. That is circularity in the **data**, used to
argue against additivity. They test next-item and previous-item, i.e. k = +1 and k = -1
only. Nobody sweeps k across a full cycle.

Theoretical cover: recent work argues symmetry in language statistics drives the formation
of representational manifolds, explaining circles for cyclical concepts and rippled 1D
manifolds for continuous sequences (arXiv:2602.15029). That mechanism does not care
whether the symmetric variable is an operand or an operation — so it predicts circular task
vectors if task-space symmetry is present in the training distribution.

**Also relevant to read before starting:**
- Chughtai, Chan & Nanda, *A Toy Model of Universality* — group irreps as the general
  framework for what representation shapes are available
- Nanda et al. 2023, grokking / modular addition — the Clock algorithm
- Modell, Rubin-Delanchy & Whiteley, *The origins of representation manifolds in LLMs*
  (arXiv:2505.18235)
- Yin & Steinhardt on FV heads vs induction heads

---

## 5. Test battery

Extract `FV(k)` for k = 0..n-1. Primary object of analysis is the Gram matrix
`G_ij = <FV(i), FV(j)>`.

**Primary diagnostic — is G circulant?**
Does `G_ij` depend only on `(i-j) mod n`? One-line check, and it is decisive. If circulant,
G's eigenvectors are the DFT basis by construction, so eigendecomposing G hands you the
irrep decomposition directly. Read off which frequencies the model uses:
- single dominant frequency pair => plain circle
- multiple frequencies => helix / multi-irrep (same shape as number representations)
- for n = 12, check whether active frequencies land on divisors of 12 — this tests whether
  the model "knows" 12 = 3 x 4

**Secondary tests:**
- Wraparound: `FV(n) ~ FV(0)`? (kills any linear code)
- Additivity: `FV(1) + FV(1) ~ FV(2)`? (should *fail* under A)
- Rotation fit: minimise `||R FV(k) - FV(k+1)||` across k, then check `R^n ~ I`
- Norm profile: constant across k (circular) vs growing (linear)
- Steering transfer across operand partitions — the hypothesis-C test
- Causal validation: does injecting a *synthesised* point on the fitted circle (e.g. an
  interpolated k = 1.5) produce coherent behaviour? Strong evidence if yes.

**Controls (mandatory, not optional):**
- n unrelated tasks should show *no* circulant structure
- a genuinely ordinal family (extract-k-th-item, k = 1..n) should give a rippled open curve
  rather than a closed loop, if the Gurnee-style logic transfers to task space

---

## 6. The control that decides whether any of this is real

**Confound:** days-of-week *operands* are already known to be circular (Engels et al.). Any
circular structure found in an FV extracted from day-shift prompts may simply be inherited
from operand geometry.

**Control 1 — disjoint operand partitions.** Extract `FV(k)` twice, once with demonstrations
drawn only from {Mon, Tue, Wed}, once from {Thu, Fri, Sat, Sun}. If the geometry belongs to
the task, the two circles coincide up to noise. If inherited from the operand, the circle
*rotates with the operand distribution* — visible immediately as a phase offset in the
circulant structure.

**Control 2 — heterogeneous operands (free, and stronger).** Todd et al.'s next-item data is
already mixed-domain: days, months, letters, numbers, roman numerals in the same prompt.
Build shift-by-k prompts the same way. Any structure surviving a mixed-domain operand set
cannot be inherited from a single operand circle, because there isn't one. This turns their
design choice into the control.

---

## 7. Cost and schedule

Compute is not the bottleneck.

- 100 prompts x 10-shot x 7 values of k = ~700 forward passes at ~200 tokens. Minutes on a
  3B model.
- Full Todd-protocol causal head identification is the expensive step (patching across
  ~28 layers x 24 heads): 1-2 GPU-hours. Identify the FV head set **once** and reuse across
  all k — it is task-general in their setup.
- Three task families x two models: under 10 GPU-hours total.

**Hardware note.** Target is an RTX 2000 Ada, 16GB. An 8B model at bf16 will not fit with
activation caching, and int8 quantization is a bad idea for a study whose entire output is
cosine-similarity structure — you would be measuring quantization noise. Use
**Llama-3.2-3B or Gemma-2-2B at full precision**. A quantized 7B as a robustness check only.

**Human time: 1-2 weeks, front-loaded.**

- **Day 1 — go/no-go pilot.** Measure plain ICL accuracy for shift-by-k, k = 0..6. Todd et
  al. note previous-item underperforms next-item, so accuracy may collapse beyond k = +/-1.
  If the model cannot do shift-by-4, there is no task vector to extract and the sweep is
  measuring noise. Two hours to find out. **Do not skip this.**
- Days 2-3: extraction pipeline + Gram matrix. This is where the answer arrives.
- Days 4-5: causal tests.
- Week 2: controls, second task family.

**Pivot if the pilot fails:** switch to ROT-k (n = 26, more points, models handle Caesar
ciphers reasonably), or drop to a small model trained directly on modular shift, where the
task distribution is fully controlled.

---

## 8. Outcome interpretation

| Result | Reading | Consequence |
|---|---|---|
| Circulant G, one frequency pair, `FV(n) ~ FV(0)`, additivity fails, `R^n ~ I` | Task space is circular | LRH incomplete for operations. Additive steering is the *wrong intervention primitive* for cyclic families. Bonus: steer to interpolated tasks. |
| Circulant, multiple frequencies | Helical / multi-irrep | Scales the group-representation framing from toy modular-arithmetic nets to real LLMs. |
| Additive, `FV(k) ~ k*v` | LRH holds for ops despite curved operands | A dissociation: value-space curved, operation-space flat. Predicts a *behavioural* signature — wraparound cases systematically worse, since a linear code cannot represent k = 7 == 0. |
| Near-orthogonal, no structure | Tasks are lookup, not composed | Negative result on compositional task representation. Connects to the retrieve-vs-generalise dual-mode ICL literature. Publishable if controls are tight. |
| Structure vanishes under operand control | The circle was never in the task | Methodological result about FV extraction. Least glamorous, arguably most useful to the field. |

Highest prior sits on the last row. That is why Section 6 is not optional.

---

## 9. Known risks

- **Ceiling/floor on ICL accuracy** for mid-cycle k — handled by the Day 1 pilot.
- **Tokenizer and frequency effects.** Days of the week have unequal corpus frequencies;
  month names collide with other senses (May, March, August). Check for norm artefacts
  tracking token frequency.
- **Prompt-format leakage.** Keep format fixed across k; vary it as a robustness check.
- **FV extraction is contested.** Multiple incompatible definitions exist in the literature
  (Hendel-style sentinel-token hidden state vs Todd-style causal-head averaging), sometimes
  producing contradictory results. Pick one, state it, and run the primary diagnostic under
  both if time allows — a geometry claim that only holds for one extraction method is a
  finding about the method, not the model.
- **Circulant-structure false positives.** Any smooth similarity decay with |i-j| will look
  vaguely circulant. Test the specific `mod n` wraparound, not just banded decay.

---

## 10. First concrete step

Run the Day 1 pilot: plain few-shot accuracy for shift-by-k on days of the week,
k = 0..6, on Llama-3.2-3B, 10-shot, ~100 prompts per k, mixed-domain operands. Report
accuracy per k. Everything downstream is conditional on this.