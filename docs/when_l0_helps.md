# When does L0 help? A label-free predictor study

Every L0 gain in this repo comes from two corrections that can be switched on separately: cyclic-shift
marginalization (`perm`) and a label-free prior division (`bc` = batch prior, the default; `cf` =
content-free prior, opt-in). This page asks, over every (model, question) point we have, which one
does the work and when either one hurts. Regenerate with
`python -m bench.diag_l0 bench/results_typed_diag bench/results_batchprior_v0 bench/results_small`.

Data: 200 typed-decisions points (10 models x 20 questions, K = 2 to 5, teacher-labelled, gold
marginals often skewed) and 21 three-task bench points (7 models x banking20 / newsgroups / injection,
K = 20 or 2, roughly balanced). Gain = L0 accuracy minus raw accuracy on the same items. "majority"
is the gold majority-label share of a question; "entropy" its normalized label entropy.

## Result

| points | correction | mean gain | gain > 0 | gain < -0.01 | strongest predictor (Spearman) |
|---|---|---|---|---|---|
| typed choice (n=60) | perm only | **+0.032** | 36/60 | 12 | label entropy +0.40, majority -0.37 |
| typed choice | bc only | **-0.025** | 24/60 | 32 | K +0.37 |
| typed choice | perm + bc (default L0) | +0.002 | 32/60 | 25 | label entropy +0.39 |
| typed noul (n=60) | bc only | +0.018 | 34/60 | 17 | raw accuracy -0.72, raw ECE +0.66 |
| typed noul | perm + bc (default L0) | +0.023 | 35/60 | 20 | raw accuracy -0.75 |
| typed score (n=80) | bc only (= L0) | +0.026 | 51/80 | 22 | raw ECE +0.44, entropy +0.39 |
| bench choice K=20 (n=14) | perm only | +0.084 | 14/14 | 0 | raw flip rate +0.62 |
| bench choice K=20 | perm + bc (default L0) | **+0.103** | 14/14 | 0 | raw flip rate +0.67 |
| bench noul (n=7) | perm + bc (default L0) | +0.064 | 6/7 | 0 | |

Three things follow.

**Permutation marginalization is the safe half.** On 20-way tasks it is positive on every model,
and how much it gives is predicted by the raw order-flip rate (rho = 0.62 to 0.67), which needs no
labels: reverse the option list once and you know roughly what L0 will buy you.

**The batch prior is the risky half, and the risk is the gold marginal.** Dividing by the batch
mean assumes the true label marginal is roughly flat. Where it is not, the correction pushes the
model away from a majority it was right about. The worst points are exactly the most skewed
questions: `security_incidents/disposition` (majority 0.74) loses 0.29 to 0.44 accuracy on three
models, `invoice_processing/duplicate` (majority 0.89) loses 0.21 to 0.39 on four. Gain correlates
with label entropy at +0.39 to +0.52 across kinds. On balanced tasks the same prior adds 2 to 3
points on top of permutation.

**The batch prior helps most where raw is worst.** On `noul`, gain correlates with raw accuracy at
-0.72 and with raw ECE at +0.66: a model that is inaccurate and overconfident on a yes/no question
is usually carrying a label bias, and the prior removes it. A model that is already accurate on a
skewed question is reporting the marginal, and the prior damages it.

## The offline prior study, and the default it chose

`bench.run --dump-items` writes every item's raw per-shift distribution; `bench.prior_study` replays any
prior rule on them with no GPU. Over 209 (model, question) units (10 models), accuracy gain over raw:

| points | perm only | batch prior^0.5 | batch prior^0.75 | batch prior^1 |
|---|---|---|---|---|
| choice, K >= 10 (balanced bench, n=19) | +0.070 | +0.079 | +0.081 | **+0.085** |
| choice, K < 10 (typed, often skewed, n=54) | **+0.033** | +0.029 | +0.026 | +0.010 |
| noul (n=64) | +0.004 | +0.014 | +0.023 | **+0.030** |
| score (n=72) | 0 | +0.015 | **+0.024** | +0.023 |
| gold majority < 0.45 (n=37) | +0.057 | +0.061 | **+0.063** | +0.059 |
| gold majority 0.45 to 0.65 (n=64) | +0.027 | +0.038 | +0.046 | **+0.055** |
| gold majority >= 0.65 (n=36) | **-0.013** | -0.019 | -0.022 | -0.044 |
| all 209 | +0.016 | +0.024 | **+0.030** | +0.027 |

Two things that did not work, for the record. A prior made of the position profile alone (the part of the
batch mean that content cannot explain) is identical to permutation-only under a full cycle, because the
cycle already cancels position bias exactly. A guard that switches the prior off when the batch mean looks
skewed (its max share correlates with the gold majority at rho = 0.66) recovers only a third of the loss on
skewed questions and gives up the gain on the middle bucket. The skew of a model's answers and the skew of
the truth are the same statistic without labels; no rule on the batch can separate them.

**Decision: the batch prior stays the default at strength 0.75** (`Decider(prior_strength=0.75)`, the prior
is raised to that power before dividing). It has the best mean gain, is first or second in every bucket,
and halves the worst-case loss (-0.044 to -0.022) while keeping most of the noul and score gains. Strength
1.0 (the old behaviour) and `prior="none"` (permutation only, the safest choice when you know your label
marginal is skewed) remain one argument away. Not chosen: a K-dependent rule; K < 10 predicted skew in this
dataset by coincidence, not by any law.

## What this changes

- Reversing the option list once (zero labels) is a usable predictor of the permutation gain; the
  Decider reports it as `order_flip_raw`.
- The batch prior is applied at strength 0.75 by default (see above); `prior_strength=1.0` restores the
  full correction. The README tables were regenerated with the new default; on the balanced 20-way tasks it
  costs 0 to 2 points against full strength, which is the price of the smaller worst case.
- On questions whose true label marginal you know to be skewed, prefer `Decider(prior="none")`
  (permutation only): it is never far below raw and keeps the whole permutation gain.
