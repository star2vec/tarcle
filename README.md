# tarcle

Began as a geometry question — do function vectors (FVs) for shift-by-k on a
cyclic task family inherit the group structure? That programme is closed (D38):
months Z/12 stands as a case study (ordered, low-dimensional, **not** circulant;
`docs/stage2_findings.md`), and no second family passes its gates on this model.

The repo's contribution is now **measurement validity**: interpretability
validation practice is systematically permissive. Three findings, all on
Llama-3.2-3B, months shift-by-k, 16-shot (`docs/instruments_findings.md`):

1. **The standard accuracy gate certifies vectors that encode a different
   function.** With restricted demonstration pools the gate returns GO at every
   k — on the full query cycle, closing the support-mismatch objection — while
   the extracted vector steers to next-item *even on demonstrated operands*
   (task-encoding margins −0.32 to −0.94; in-pool margin −1.00 at the worst
   cell). The vectors are maximally reliable (split-half ≥ 0.99) with large
   causal effects. The model itself generalises to unseen operands, so no
   accuracy gate at any query support could detect the failure.
2. **Fixture-validated false-positive floors do not transfer to real
   activations — in either direction.** Fixture-grade floors are understated up
   to 156× (harmonic partial R²: 0.0019 vs 0.28–0.30 real) and overstated up to
   ~100× (seam cv-R²: 0.41 vs 0.004). Our own pre-committed blanket claim
   ("~100× everywhere") was killed by its registered median-ratio test (0.79).
   Data-derived nulls transfer at ratio ≈ 1.
3. **The two validation measures the FV literature composes — task accuracy and
   steering efficacy — are statistically unrelated across tasks here** (Spearman
   ρ = 0.02 against a noise ceiling of 0.93; ρ = 0.12 between two readouts of
   the same injections).

**Not established:** one model, one task family, one prompt format. Generality
to standard FV benchmark tasks (T7) is registered but unrun. The injection-
protocol objection is closed (T2, D54): across the full 28-layer × 5-scale
grid, both methods, no cell gives a collapsed vector a positive task-encoding
margin — collapse is not an artifact of the frozen protocol. Hendel-method
columns carry a registered ±1-confinement limitation throughout.

**Layout:** `tarcle/` (stage 1: `extract.py`, `causal.py`; stage 2, numpy-only:
`geometry.py`, `stage2.py`, `floors.py`, `measure_corr.py`, `margin_split.py`,
`offset_audit.py`, `support_gate.py`), `experiments/` configs, `results/`
(stage-1 write-once; every writer stamps its filename or refuses to overwrite
differing content — D45/D46), `docs/` (brief, findings, decision log D1–D46).

**Reproduce stage 2 without a GPU:** every analysis reads saved `.npz` from
`results/` — `python -m tarcle.stage2 | floors | measure_corr | margin_split |
offset_audit | support_gate`. Numpy-only tests: `pytest tests/test_geometry.py
tests/test_families.py tests/test_floors.py`. The torch-CPU tests are currently
certified on the CUDA box only (D44).

**Convention:** every run is preceded by a committed pre-registration
(`docs/preregistration*.md`); `docs/decisions.md` is append-only; every artifact
records its git commit, config hash, and prompt SHA-256.
