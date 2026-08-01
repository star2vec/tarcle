# Behavioral pilot findings (Day 1)

Results from the Day-1 go/no-go pilots on **Llama-3.2-3B**, MPS, fp16, forced-choice
scoring over the full cycle of the query domain. These are *behavioral* results — plain
ICL accuracy, no function vectors, no activation capture. Precision is fp16 on MPS, which
is acceptable for behavioral runs only (CLAUDE.md rule 4 / `backends.py` precision policy);
nothing here gates a geometry claim.

Runs recorded:

| run | family | k | n/k | shots | gate |
|---|---|---|---|---|---|
| `llama32_3b_mps` | days (Z/7) + mixed | 0–6 | 100 | 10 | NO-GO |
| `llama32_3b_mps_days_n400_s16` | days (Z/7) | 3,4,5 | 400 | 16 | NO-GO |
| `llama32_3b_mps_months` | months (Z/12) | 0–11 | 100 | 10 | MARGINAL |
| `gpt2_cpu_smoke` | days + mixed | 0–6 | 100 | 10 | NO-GO (pipeline smoke only) |

All accuracies below are the **held-out stratum** (query operand never appears as a demo
operand). The copy stratum runs 0.90–1.00 essentially everywhere, confirming the prompt
format and the forced-choice scoring path are sound; every deficit below is task
difficulty, not harness error. n = 50 per held-out cell unless stated, so single-cell
Wilson 95% CIs are roughly ±0.13 — read the *shape* of these profiles, not individual cells.

---

## 1. Signed-shift decay profile

Accuracy is not a function of k; it is a function of the **signed** shift, taking the
short way round the cycle. Re-indexing k as ±m (m = min(k, n−k)) is what makes the
profiles legible.

**Days (Z/7), 10-shot:**

| \|m\| | forward (+m) | backward (−m) |
|---|---|---|
| 0 | 1.00 | — |
| 1 | 1.00 (k=1) | 0.70 (k=6) |
| 2 | 0.72 (k=2) | 0.20 (k=5) |
| 3 | 0.52 (k=3) | 0.32 (k=4) |

**Months (Z/12), 10-shot:**

| \|m\| | forward (+m) | backward (−m) |
|---|---|---|
| 0 | 1.00 | — |
| 1 | 1.00 (k=1) | 1.00 (k=11) |
| 2 | 1.00 (k=2) | 0.88 (k=10) |
| 3 | 0.98 (k=3) | 0.76 (k=9) |
| 4 | 0.62 (k=4) | 0.34 (k=8) |
| 5 | 0.50 (k=5) | 0.56 (k=7) |
| 6 | 0.90 (k=6, self-inverse) | — |

Decay in |m| is real but **not monotone**, and the violation is structural rather than
noise (§3).

## 2. Forward/backward asymmetry

Forward shifts beat backward shifts of the same magnitude, consistently, in both families:

- days: +1 1.00 / −1 0.70, +2 0.72 / −2 0.20, +3 0.52 / −3 0.32
- months: +2 1.00 / −2 0.88, +3 0.98 / −3 0.76, +4 0.62 / −4 0.34

This generalises Todd et al.'s observation that previous-item underperforms next-item:
the effect is not confined to ±1, it holds across the whole cycle. The asymmetry closes at
months |m|=5 (0.50 vs 0.56, well within CI) — i.e. it fades as the two directions become
equidistant, which is what a direction-confusion account predicts.

**The dominant error is a sign flip, not a blur.** Errors land on −k far more than on
generic neighbours:

- days, k=5 (≡−2): 0.36 of mass on shift **+2**, versus 0.15 and 0.14 on the true
  neighbours 3 and 4 (16-shot, n=200)
- months: P(pred = −k) is 0.22 at k=5, 0.22 at k=7, 0.24 at k=8, 0.14 at k=9

Caveat on reading Z/7: −3 ≡ 4 and −4 ≡ 3, so k=3↔4 confusion is *simultaneously*
off-by-one and sign-flip and cannot separate the two accounts. Only k=5 and k=6 disambiguate
in Z/7. Z/12 has far more cells where "adjacent" and "sign-flipped" are distinct, which is
one reason it is the better family for this question.

The model recovers shift **magnitude** and loses **direction**.

## 3. Wraparound is easier, not harder — and the antipode is a landmark

Both families are non-monotone in |m| at exactly the structurally special points:

- **days k=6 ≡ −1: 0.70**, against k=5 at 0.20 and k=4 at 0.32. The furthest-in-k cell is
  nearly the *easiest* non-trivial one. The model reads k=6 as "the previous day", not as
  "shift by six".
- **months k=11 ≡ −1: 1.00** and k=10 ≡ −2: 0.88, against k=8 at 0.34.
- **months k=6: 0.90**, sitting *above* both neighbours k=5 (0.50) and k=7 (0.56). k=6 is
  the order-2 element of Z/12 — the unique self-inverse shift, where forward and backward
  coincide and the sign-flip error mode is definitionally unavailable. "Six months later"
  is also a salient corpus idiom.

This bears directly on the pre-registered prediction in BRIEF §8, which holds that a
**linear** task code `FV(k) ≈ k·v` predicts wraparound cases should be *systematically
worse*, since a linear code cannot represent k=7≡0. Behaviorally we observe the opposite:
wraparound cases are systematically **better**. That is evidence against a purely linear
account and consistent with a wrapped/cyclic encoding of the shift — though it is
behavioral evidence about the model's competence, not a measurement of FV geometry, and it
constrains rather than decides between hypotheses A–D.

Weaker, hedged observation on Z/12 subgroup structure: shifts by divisors of 12 do tend to
score higher (k=2 1.00, k=3 0.98, k=6 0.90) than non-divisors (k=5 0.50, k=7 0.56), but
k=4 at 0.62 breaks the pattern. Suggestive only; not a claim the pilot can settle. It is
exactly the question the DFT frequency analysis in BRIEF §5 is designed to answer.

## 4. Per-domain results under mixed-domain operands

From the `mixed` variant of `llama32_3b_mps` (days+months+letters+digits in one prompt
set), held-out stratum. **Cell sizes are 5–17**, since 50 held-out items per k are split
across four domains at random — individual cells are very noisy and only the column
aggregates carry weight.

| k | days | months | letters | digits |
|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | 1.00 | 0.91 | 0.90 | 0.81 |
| 2 | 0.75 | 0.79 | 0.33 | 0.20 |
| 3 | 0.17 | 0.70 | 0.00 | 0.29 |
| 4 | 0.53 | 0.56 | 0.14 | 0.29 |
| 5 | 0.15 | 0.11 | 0.08 | 0.25 |
| 6 | 0.38 | 0.50 | 0.07 | 0.13 |
| **all** | **0.57** (n=95) | **0.70** (n=77) | **0.35** (n=84) | **0.43** (n=94) |

Two consequences:

1. **Months is the strongest domain even inside the mixed set** (0.70 aggregate), which is
   what motivated running it standalone. Standalone months is better still (§1), so the
   mixed context costs months some accuracy rather than the reverse.
2. **Letters collapse** (0.35 aggregate; 0.00–0.33 for k≥2). This matters well beyond the
   mixed control — see the ROT-k argument below.

Mixed-domain accuracy is uniformly at or below the corresponding single-domain accuracy.
Since BRIEF §6 Control 2 (heterogeneous operands, the control that cannot inherit a single
operand circle) requires the mixed task to be *performable*, that control is currently
unavailable on this model at 10 shots. Whether 16 shots rescues it is untested.

## 5. Contrast with the gpt2 smoke run

gpt2's failure mode is a **pure next-item prior**: for every k≥2 it dumps 0.34–0.66 of its
predicted-shift mass into the shift-1 column. Llama-3.2-3B puts ~0.00–0.06 there for
k=2..5 at both 10 and 16 shots. The two models fail in qualitatively different ways, and
only Llama's failure is consistent with having represented the shift quantity at all. Any
extraction code validated against gpt2 behavior should not assume the gpt2 error structure
transfers.

## 6. Shot-count sensitivity

Days k=3,4,5 rerun at 16 shots, n=400 (200 held-out/cell):

| k | 10-shot, n=50 | 16-shot, n=200 | Δ |
|---|---|---|---|
| 3 | 0.52 [0.39, 0.65] | 0.775 [0.71, 0.83] | +0.26 |
| 4 | 0.32 [0.21, 0.46] | 0.610 [0.54, 0.67] | +0.29 |
| 5 | 0.20 [0.11, 0.33] | 0.285 [0.23, 0.35] | +0.08 |

This run varies n *and* shots simultaneously, so it is not a clean CI-narrowing of the
original cells: the wider n tightens the intervals, but the movement in the point estimates
is attributable to shot count. Reading: **16 shots substantially rescues mid-cycle k, except
in the backward direction.** k=5 ≡ −2 gained almost nothing and its CI still straddles the
0.30 NO-GO floor, which is the sole reason the days gate remains NO-GO.

Shot count is therefore a live experimental variable, not a fixed cost — and it interacts
with the direction asymmetry rather than uniformly lifting the curve.
