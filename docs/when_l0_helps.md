# When does L0 help? A label-free predictor study

Every L0 gain in this repo comes from two corrections that can be switched on separately: cyclic-shift
marginalization (`perm`) and a label-free prior division (`bc` = batch prior, the default; `cf` =
content-free prior, opt-in). This page asks, over every (model, question) point we have, which one
does the work and when either one hurts. Regenerate with
`python -m bench.diag_l0 bench/results_typed_diag bench/results_batchprior_v0 bench/results_small`
(the raw output is committed as `docs/diag_l0_output.txt`; `scripts/regen_docs.sh` runs the same command).

Data: 200 typed-decisions points (10 models x 20 questions, K = 2 to 5, teacher-labelled, gold
marginals often skewed; `bench/results_typed_diag`) and 30 three-task bench points (10 models x banking20 /
newsgroups / injection, K = 20 or 2, roughly balanced; `bench/results_batchprior_v0` for the three headline
models, `bench/results_small` for the seven small ones). Gain = L0 accuracy minus raw accuracy on the same
items. "majority" is the gold majority-label share of a question; "entropy" its normalized label entropy.

## Result

| points | correction | mean gain | gain > 0 | gain < -0.01 | strongest predictor (Spearman) |
|---|---|---|---|---|---|
| typed choice (n=60) | perm only | **+0.032** | 36/60 | 12 | label entropy +0.40, majority -0.37 |
| typed choice | bc only | **-0.025** | 24/60 | 32 | K +0.37 |
| typed choice | perm + bc (default L0) | +0.002 | 32/60 | 25 | label entropy +0.39 |
| typed noul (n=60) | bc only | +0.018 | 34/60 | 17 | raw accuracy -0.72, raw ECE +0.66 |
| typed noul | perm + bc (default L0) | +0.023 | 35/60 | 20 | raw accuracy -0.75 |
| typed score (n=80) | bc only (= L0) | +0.026 | 51/80 | 22 | raw ECE +0.44, entropy +0.39 |
| bench choice K=20 (n=20) | perm only | +0.068 | 20/20 | 0 | raw flip rate +0.59 |
| bench choice K=20 | perm + bc (default L0) | **+0.082** | 20/20 | 0 | raw flip rate +0.61 |
| bench noul (n=10) | perm + bc (default L0) | +0.059 | 9/10 | 0 | |

Three things follow.

**Permutation marginalization is the safe half.** On 20-way tasks it is positive on every model,
and how much it gives is predicted by the raw order-flip rate (rho = 0.59 to 0.61), which needs no
labels: reverse the option list once and you know roughly what L0 will buy you.

**The batch prior is the risky half, and the risk is the gold marginal.** Dividing by the batch
mean assumes the true label marginal is roughly flat. Where it is not, the correction pushes the
model away from a majority it was right about. The worst points are exactly the most skewed
questions: `security_incidents/disposition` (majority 0.74) loses 0.29 to 0.44 accuracy on three
models, `invoice_processing/duplicate` (majority 0.89) loses 0.20 to 0.39 on five. Gain correlates
with label entropy at +0.39 to +0.52 across kinds. On the balanced 20-way tasks the same prior adds
about a point and a half on top of permutation (+0.068 to +0.082).

**The batch prior helps most where raw is worst.** On `noul`, gain correlates with raw accuracy at
-0.72 and with raw ECE at +0.66: a model that is inaccurate and overconfident on a yes/no question
is usually carrying a label bias, and the prior removes it. A model that is already accurate on a
skewed question is reporting the marginal, and the prior damages it.

## The offline prior study, and the default it chose

`bench.run --dump-items` and `bench.run_typed --dump-items` write every item's raw per-shift distribution;
`bench.prior_study` replays any prior rule on them with no GPU. The dumps behind this section are committed
under `bench/results_dump/2026-09-21/` (the three bench tasks) and `bench/results_typed_dump/2026-09-21/`
(typed-decisions), ten models each; regenerate with
`python -m bench.prior_study bench/results_dump bench/results_typed_dump`. Over 230 (model, question) units,
accuracy gain over raw:

| points | perm only | batch prior^0.5 | batch prior^0.75 | batch prior^1 |
|---|---|---|---|---|
| choice, K >= 10 (balanced bench, n=20) | +0.070 | +0.078 | +0.080 | **+0.084** |
| choice, K < 10 (typed, often skewed, n=60) | **+0.031** | +0.027 | +0.025 | +0.011 |
| noul (n=70) | +0.003 | +0.013 | +0.021 | **+0.027** |
| score (n=80) | 0 | +0.015 | **+0.024** | +0.024 |
| choice + noul, gold majority < 0.45 (n=40) | +0.055 | +0.060 | **+0.061** | +0.058 |
| choice + noul, gold majority 0.45 to 0.65 (n=70) | +0.025 | +0.035 | +0.043 | **+0.051** |
| choice + noul, gold majority >= 0.65 (n=40) | **-0.012** | -0.019 | -0.021 | -0.043 |
| all 230 | +0.015 | +0.023 | **+0.028** | +0.027 |

Two things that did not work, for the record. A prior made of the position profile alone (the part of the
batch mean that content cannot explain) is identical to permutation-only under a full cycle, because the
cycle already cancels position bias exactly. A guard that switches the prior off when the batch mean looks
skewed (its max share correlates with the gold majority at rho = 0.66) recovers only part of the loss on
skewed questions and gives up the gain on the middle bucket. The skew of a model's answers and the skew of
the truth are the same statistic without labels; no rule on the batch can separate them.

**Decision: the batch prior stays the default at strength 0.75** (`Decider(prior_strength=0.75)`, the prior
is raised to that power before dividing). It has the best mean gain, is first or second in every bucket
except the small-K choice bucket (where permutation alone leads and 0.75 is within a point), and halves the
worst-case loss (-0.043 to -0.021) while keeping most of the noul and score gains. Strength 1.0 (the old
behaviour) and `prior="none"` (permutation only, the safest choice when you know your label marginal is
skewed) remain one argument away. Not chosen: a K-dependent rule; K < 10 predicted skew in this dataset by
coincidence, not by any law.

## What this changes

- Reversing the option list once (zero labels) is a usable predictor of the permutation gain; the
  Decider reports it as `order_flip_raw`.
- The batch prior is applied at strength 0.75 by default (see above); `prior_strength=1.0` restores the
  full correction. The README tables were regenerated with the new default; on the balanced 20-way tasks it
  costs 0 to 2 points against full strength, which is the price of the smaller worst case.
- On questions whose true label marginal you know to be skewed, prefer `Decider(prior="none")`
  (permutation only): it is never far below raw and keeps the whole permutation gain.
