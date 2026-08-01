# Literature sweep: geometric structure of task/operation space in LLMs

*Assessment of the research brief "Is task space geometrically structured the way value space is?" — August 2026.*

## Verdict

Promising, and worth doing — but the brief's central novelty claim ("nobody has asked whether operation-space has the same geometric richness as value-space") is **no longer true as stated**. One paper the brief misses, *Latent Concept Disentanglement in Transformer-based Language Models* (arXiv:2506.16975, Google, ICLR 2026), already shows that task vectors for parameterised ICL task families lie on low-dimensional manifolds cleanly aligned with the task parameter — including for **add-k** (1D linear manifold) and **circular-trajectory tasks** (2D manifold, with clockwise/counterclockwise on separate manifolds). The question "can operation-space be geometrically structured?" is answered yes.

What survives — and it is the better part of the brief — is everything that paper does *not* do:

1. **Cyclic group structure with genuine wraparound.** Their circular tasks are physical trajectories, not Z/n task families. Nobody tests `FV(n) ~ FV(0)`, circulant Gram structure, irrep/frequency decomposition, or divisor structure for composite n. The brief's Section 5 battery is untouched territory.
2. **The algebra question (Section 2, sub-question 3).** Is FV a homomorphism? Vector vs operator (hypothesis C)? Nobody tests this. And there is fresh supporting evidence: *Understanding Task Vectors in ICL* (2506.09048) proves task vectors act like linear combinations of demonstrations and **fail on high-rank mappings** — a rotation family is exactly such a mapping. Hypothesis C now has theoretical backing and is the strongest part of the proposal.
3. **The operand-inheritance confound (Section 6).** Nobody controls for task geometry being inherited from operand geometry. This control is novel and, if structure vanishes under it, the methodological result stands on its own.
4. **Pretrained frontier of the question.** 2506.16975's cleanest results are in controlled/synthetic settings; the sweep across a full cycle in a *pretrained* LLM with corpus-driven task statistics is open.

**Required repositioning:** the paper is not "does task space have geometry?" but "does task space inherit *group structure* — and is the task representation a vector or an operator?" Cite 2506.16975 as the departure point, not as competition. It also de-risks the project: structured task manifolds demonstrably exist, so the sweep is not a shot in the dark.

## New alternative hypothesis the brief should add

**Hypothesis D — simplex/convex-mixture geometry.** *Task Vector Geometry Underlies Dual Modes of Task Inference* (2605.03780) shows (in controlled settings, with theory) that trained-task vectors form near-orthogonal anchors and hidden states move as **convex combinations** tracking a Bayesian posterior over tasks; OOD tasks live in a nearly orthogonal subspace. Under D, `FV(k)` for k seen in training are ~orthogonal vertices of a simplex (Gram ≈ identity + constant — which is trivially circulant! a false-positive mode for the primary diagnostic), and intermediate/synthesised k=1.5 injections would behave as mixtures of k=1 and k=2 rather than as a rotated task. The brief's circulant check needs a "non-trivial frequency content" criterion to exclude this, and the k=1.5 interpolation test discriminates C from D nicely.

Same paper also formalises the brief's "FV extraction is contested" risk: it exhibits a counterexample (planted Dyck) where the premise underlying all heuristic FV extraction (hidden state summarises context) fails outright.

## Two practical warnings

- **ROT-k pivot is weaker than it looks.** Shift-cipher performance in LLMs is dominated by corpus frequency of the specific shift: ROT-13 vastly outperforms other shifts (McCoy et al. "Embers of Autoregression"; Prabhakar et al. 2407.01687). On a 3B model, most of the 26 points of Z/26 may sit at floor. The Day-1 pilot logic applies to the pivot too.
- **The symmetry-origin theory cuts both ways.** 2602.15029's mechanism requires translation-symmetric *co-occurrence statistics of the tasks themselves* in training data. "Shift-by-3-days" as a task is rare in corpora; day-shift task statistics are plausibly dominated by k=±1. So the theory licences circular *operand* geometry much more strongly than circular *task* geometry — consistent with the brief's own highest-prior row (structure vanishes under operand control). Fine, but expect it.

---

## Reading list, ranked

### Tier 1 — read before touching a GPU

1. **Todd et al., *Function Vectors in Large Language Models*** — [arXiv:2310.15213](https://arxiv.org/abs/2310.15213) (ICLR 2024). The extraction protocol you will use; Appendix L (cyclic tasks vs word-offsets) is your direct precedent. Code: [github.com/ericwtodd/function_vectors](https://github.com/ericwtodd/function_vectors).
2. **Kim, Nowak et al. (Google), *Latent Concept Disentanglement in Transformer-based Language Models*** — [arXiv:2506.16975](https://arxiv.org/abs/2506.16975) (ICLR 2026). The closest existing work: task vectors of parameterised families on parameter-aligned manifolds (add-k linear, trajectories 2D). Defines what you must position against.
3. **Yan, Yang & Zhong, *Task Vector Geometry Underlies Dual Modes of Task Inference in Transformers*** — [arXiv:2605.03780](https://arxiv.org/abs/2605.03780). Rigorous definition of task vectors, simplex/posterior geometry (hypothesis D), and a proven failure case of heuristic FV extraction. Code: mini-ICL repo.
4. **Hendel, Geva & Globerson, *In-Context Learning Creates Task Vectors*** — [arXiv:2310.15916](https://arxiv.org/abs/2310.15916) (EMNLP 2023 Findings). The other, incompatible FV definition; you committed to running both.
5. **Dong et al., *Understanding Task Vectors in ICL: Emergence, Functionality, and Limitations*** — [arXiv:2506.09048](https://arxiv.org/abs/2506.09048). Linear Combination Conjecture + predicted failure on high-rank mappings, confirmed on real LLMs. This is the theoretical spine of hypothesis C.
6. **Engels et al., *Not All Language Model Features Are One-Dimensionally Linear*** — [arXiv:2405.14860](https://arxiv.org/abs/2405.14860). The operand-side circles (days/months) your confound control exists for.
7. **Karkada, Wyart, Bahri et al., *Symmetry in language statistics shapes the geometry of model representations*** — [arXiv:2602.15029](https://arxiv.org/abs/2602.15029). The theory that predicts when circles form — check whether its premise even holds for task statistics.

### Tier 2 — read during pipeline construction

8. **Yin & Steinhardt, *Which Attention Heads Matter for In-Context Learning?*** — [arXiv:2502.14010](https://arxiv.org/abs/2502.14010) (ICML 2025). FV heads vs induction heads; justifies reusing one head set across all k.
9. **Xiong et al., *Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks in Superposition*** — [arXiv:2410.05603](https://arxiv.org/abs/2410.05603) (ICML 2025). Internal convex composition of task vectors — directly relevant to the algebra sub-question and to mixed-operand prompt design.
10. **Kantamneni & Tegmark, *Language Models Use Trigonometry to Do Addition*** — [arXiv:2502.00873](https://arxiv.org/abs/2502.00873). The helix/multi-frequency template your multi-irrep outcome row is modelled on.
11. **Gurnee et al., *When Models Manipulate Manifolds*** — [arXiv:2601.04480](https://arxiv.org/abs/2601.04480). The rippled-manifold logic behind your ordinal-family control.
12. **Modell, Rubin-Delanchy & Whiteley, *The Origins of Representation Manifolds in LLMs*** — [arXiv:2505.18235](https://arxiv.org/abs/2505.18235). Why cosine similarity encodes on-manifold geodesics — the justification for the Gram matrix as primary object.
13. **Prabhakar, McCoy et al., *Deciphering the Factors Influencing the Efficacy of Chain-of-Thought*** — [arXiv:2407.01687](https://arxiv.org/abs/2407.01687). Shift-cipher accuracy vs shift frequency — read before trusting the ROT-k pivot.

### Tier 3 — context and framing

14. **Chughtai, Chan & Nanda, *A Toy Model of Universality*** — [arXiv:2302.03025](https://arxiv.org/abs/2302.03025) (ICML 2023). Group irreps as the inventory of available representation shapes.
15. **Zhong et al., *The Clock and the Pizza*** — [arXiv:2306.17844](https://arxiv.org/abs/2306.17844) (NeurIPS 2023). Multiple algorithms for the same modular task — a caution for hypothesis B: the "rotation implemented elsewhere" story is not unique.
16. **Park et al., *In-Context Learning of Representations*** — [arXiv:2501.00070](https://arxiv.org/abs/2501.00070) (ICLR 2025). In-context exemplars reorganise concept geometry (ring graphs!) but cannot override pretrained structure — bears directly on whether your 10-shot prompts *create* the geometry you then measure.
17. **Tan et al., *Understanding (Un)Reliability of Steering Vectors*** — [arXiv:2505.22637](https://arxiv.org/abs/2505.22637), and follow-up [arXiv:2602.17881](https://arxiv.org/abs/2602.17881) (geometric predictors of steering failure). Your operand-partition steering-transfer test is an instance of their framework.
18. **CRH team (MBZUAI), *The Cylindrical Representation Hypothesis for Language Model Steering*** — [arXiv:2605.01844](https://arxiv.org/abs/2605.01844) (ICML 2026). Non-LRH steering geometry; the "wrong intervention primitive" consequence in your outcome table should engage with this.
19. **Aggarwal et al., *Provable In-Context Vector Arithmetic via Retrieving Task Concepts*** — [arXiv:2508.09820](https://arxiv.org/abs/2508.09820) (ICML 2025). Proves the vector-addition story *for factual recall* — the class of task where hypothesis C predicts addition should work.
20. ***On the geometry and topology of representations: the manifolds of modular addition*** — [arXiv:2512.25060](https://arxiv.org/abs/2512.25060), and *Uncovering a Universal Abstract Algorithm for Modular Addition* — [arXiv:2505.18266](https://arxiv.org/abs/2505.18266). Recent refinements of the modular-addition picture beyond Clock/Pizza.
21. **He et al., *Learning to grok: emergence of ICL and skill composition in modular arithmetic tasks*** — [arXiv:2406.02550](https://arxiv.org/abs/2406.02550). The fallback plan (small model trained on modular shifts) has a template here.

## Corrections to the brief

- All cited arXiv IDs check out (2310.15213, 2601.04480, 2602.15029, 2505.18235). Yin & Steinhardt is 2502.14010, ICML 2025.
- Section 4 "Nobody sweeps k across a full cycle" — still true for *cyclic* families in *pretrained* models, but 2506.16975 sweeps k for non-modular add-k and continuous trajectory parameters. Cite it or a reviewer will.
- Section 9's "FV extraction is contested" is now stronger than a risk note: 2605.03780 proves heuristic extraction fails under specifiable conditions. Turn it into a design principle (report both extractions from day one, not "if time allows").
- Hardware note stands; if 3B accuracy collapses beyond k=±1 (likely), consider Llama-3.1-8B with per-layer activation capture (hooks, no full cache) before abandoning pretrained models — the interesting version of the claim lives there.
