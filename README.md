# tarcle

Function vectors that pass every standard validation check while encoding a
different task than the one they were extracted for. This started as a
preregistered geometry study (do shift-by-k task vectors sit on a circle?
they don't), and the control vectors built for that study turned out to be the
actual finding.

**write-up:** [[LessWrong article]](PASTE-LINK-AFTER-POSTING)

Short version: extract shift-by-k-months vectors (Llama-3.2-3B) from
demonstrations that use only a few distinct months, and the result passes the
behavioural gate, split-half reliability, and causal-effect checks while
outputting a month adjacent to the query regardless of k (margins down to
−0.94, exactly −1.000 on the demonstrated months). Fewer distinct months makes
the in-prompt task easier while the vector breaks, so the checks favor the
broken vectors. No injection layer or strength rescues them (0 of 672 settings).
Bonus finding: pass/fail thresholds calibrated on generated random data miss in
both directions (145–156× too lenient for some statistics, ~100× too strict for
another); thresholds from shuffled real vectors transfer.

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
