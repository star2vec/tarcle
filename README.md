# tarcle

Function vectors that pass every standard validation check while encoding a
different task than the one they were extracted for. This started as a
preregistered geometry study (do shift-by-k task vectors sit on a circle?
they don't), and the control vectors built for that study turned out to be the
actual finding.

**Write-up:** [[LessWrong article]](https://www.lesswrong.com/posts/aFyir2PaoCHK5prAu/the-imposters-among-us-function-vectors-that-ace-every-check)

We extracted shift-by-k-months function vectors on Llama-3.2-3B from few-shot prompts that contained fewer distinct months (lower diversity). The vectors passed three classic checks: the behavioral gate, stability when extracting from disjoint halves of the prompt samples (cosine similarity ≥  0.99 for the broken vectors, 0.98 for the full set), and the causal effect (where injection tripled zero-shot accuracy and the correct answer’s probability, including for the most broken vector).

However, they encoded a completely different task: output a month adjacent to the queried one, while completely ignoring “k”. Margins ranged from -0.31 to -0.94 across the broken sets, with -1.000 for the most broken set, on the months included in the few-shot prompts.

The culprit is the number of distinct example inputs. Lower diversity makes the model perform the few-shot task better (from 0.38 to 0.83) even though the function vector changes identity, so the checks actually favor the imposters. 

After a sweep across layers and strengths, no injection setting (0 out of 672) rescued the broken vectors. Thresholds calibrated on generated random data were wrong in both directions (145–156× too low or 100× too high), compared to the thresholds from shuffled real vectors, which were correct.

To catch imposters, report example diversity, and score not only whether injection helps on average but also which function it performs. 


## Where things are

- `docs/decisions.md` — append-only decision log, the project's spine
- `docs/instruments_findings.md` — the findings with every number
- `docs/preregistration_instruments.md`, `docs/preregistration_t7.md` —
  registered criteria; T7 (standard-task generality) is registered and not yet
  run
- `tarcle/` — code (stage 1 needs a GPU; stage 2 is numpy-only and reads saved
  artifacts), `experiments/` — configs, `results/` — committed artifacts

## Reproducing

Every number in the post regenerates from `results/` without a GPU:
`python -m tarcle.stage2`, `floors`, `measure_corr`, `margin_split`,
`offset_audit`, `support_gate`. Figures: `python -m tarcle.post_figure` and
`fig_diversity.py` — both assert every plotted value against the artifacts
before drawing.

Every run is preceded by a committed preregistration, the decision log is
append-only, and every artifact records its git commit and config hash.
Original runs on CUDA bf16; later runs on MPS bf16 after passing a registered
cross-device validation (D50–D53).
