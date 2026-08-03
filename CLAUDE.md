# tarcle — is task space circular?

Empirical study: do function vectors (FVs) for a cyclically-structured task family
(shift-by-k on days of week, Z/7; later months Z/12, ROT-k Z/26) inherit the cyclic
group structure — i.e. does k -> FV(T_k) trace a circle in activation space?

Full research brief: `docs/BRIEF.md`. Literature notes:
`docs/lit_sweep_task_space_geometry.md`. Read both before proposing experiments.

## Hypotheses (keep all four alive in code and plots)

- **A — circular:** FV(k) on a closed loop; FV(n)≈FV(0); constant norms; additivity fails.
- **B — linear selector:** task code ~linear/orthogonal; the rotation is implemented in
  QK/OV elsewhere; no relation between task-code geometry and operation geometry.
- **C — FV is a lossy projection of an operator:** rotations can't be vector additions;
  predicts operand-dependent steering efficacy (extract on {Mon,Tue,Wed}, apply to Sun).
- **D — simplex/Bayesian-mixture (from arXiv:2605.03780):** FV(k) are near-orthogonal
  anchors; hidden states are convex mixtures. WARNING: D produces Gram ≈ identity +
  constant, which is *trivially circulant*. The circulant check alone cannot separate A
  from D — require non-trivial DFT frequency content, and use the k=1.5 interpolation
  test (D predicts mixture-of-neighbors behavior, A predicts coherent shift-by-1.5).

## Non-negotiable methodology

1. **Two FV extraction methods, always.** Todd-style causal-head averaging AND
   Hendel-style dummy-query hidden state. Every geometry result is reported under both.
   A result that holds under only one extraction is a finding about the method.
2. **Controls are part of the main experiment, not follow-ups:**
   - operand-partition control (disjoint operand sets -> same circle or phase-shifted?)
   - mixed-domain operands (days+months+letters+numbers in one prompt set)
   - n unrelated tasks -> must show NO circulant structure
   - ordinal family (extract-k-th) -> open curve expected, not closed loop
3. **Circulant test = strict mod-n test.** Check G_ij depends on (i-j) mod n
   specifically, including the wraparound entries — not merely banded decay in |i-j|.
4. **No quantization for any run whose output is geometry.** bf16 minimum. Quantized
   models only as labelled robustness checks.
5. **Pilot gates everything.** Do not build or run geometry code for a (model, family)
   pair until plain ICL accuracy for that pair is measured across ALL k. If accuracy
   collapses beyond k=±1, stop and report; do not extract FVs from a task the model
   can't do.

## Engineering constraints

- Dev machine is a CPU-only Intel MacBook Air (16GB). Geometry-bearing runs happen on
  the Ada box: **RTX 2000 Ada laptop, 8GB VRAM** (measured — not the 16GB desktop card
  BRIEF §7 assumes; see `docs/decisions.md` D6). Llama-3.2-3B bf16 fits with ~0.9GB
  headroom **only** with last-position logits (`logits_to_keep=1`); full-sequence
  logits OOM the card. Any 8B-scale model or quantized-7B robustness check needs
  different hardware and is out of scope on this box. Therefore:
  - everything runs with `--device cpu --model gpt2` end-to-end; that is the CI path
  - device/model/dtype live in one config, never hard-coded
  - activation capture via forward hooks on selected layers only; never cache full
    activations for all layers
  - two-stage design: stage 1 (GPU) extracts and saves FVs + head sets to disk as
    .npz with full metadata (model, k, extraction method, operand partition, prompt
    seed); stage 2 (CPU, numpy only) does all geometry/analysis/plots from the .npz.
    Stage 2 must never import torch or need a GPU.
- Behavioral pilot may use a hosted inference API (logprobs/outputs only); anything
  needing activations runs locally on the model.
- Determinism: seeded prompt generation; prompt sets written to disk and hashed;
  every result file records the git commit and config hash.

## Repo layout

- `tarcle/prompts.py` — task-family prompt generation (family, k, operand partition, seed)
- `tarcle/extract.py` — both FV extraction methods; hook-based capture
- `tarcle/geometry.py` — Gram matrix, circulant test, DFT spectrum, rotation fit R^n≈I,
  norm profiles, wraparound test, additivity test (pure numpy)
- `tarcle/causal.py` — injection/steering: standard FV injection, synthesized on-circle
  points (k=1.5), operand-transfer efficacy
- `experiments/` — one config file per run; `results/` — .npz + json, never overwritten
- `tests/` — the whole pipeline on gpt2-small CPU must pass in <5 min

## Style

- Small pure functions over frameworks. No Lightning, no hydra; argparse + dataclass
  configs are enough at this scale.
- Every plot script reads only from `results/`; plots must be regenerable without a GPU.
- When results are surprising, the first suspect is the extraction code, the second is
  prompt leakage, the third is the model. Check in that order.
