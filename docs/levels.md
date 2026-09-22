# What each level does, and does not do

Every `Decision` carries a `level`. This page is the contract behind that field.

## raw

One prompt, options in the order you gave them, softmax over the label tokens
at the answer position. This is what every open Jev clone does, and it is
exactly `max_tokens=1` plus logprobs.

What you get: a ranking. What you do not get: a probability. Two biases are
baked in:

- **Prior bias.** The model prefers some labels regardless of the input
  ("Yes" over "No", "A" over "D", common words over rare ones).
- **Position bias.** The model prefers some positions in the option list.
  Reorder the options and the argmax can change.

`raw` exists so the bench can show the gap. Do not ship it.

## L0: debiased, zero labels

Two training-free corrections, on by default.

**Prior correction.** Estimate the model's prior over the labels without
labels, divide the real distribution by it, renormalize. Two estimators:

- *Batch calibration* (Zhou et al., 2024), the default, applied at strength 0.75 (the prior is
  raised to `prior_strength` before dividing; 1.0 is the full correction). The prior is the mean
  predicted distribution over real inputs, kept running per question across
  calls and used once it has seen `min_prior_n` items (default 8). Before
  that, no prior correction is applied and `diagnostics["prior_method"]`

When each correction helps and when the prior hurts, measured over 221 (model, question) points: [when_l0_helps.md](when_l0_helps.md). Short version: permutation is the safe half, the batch prior hurts on questions whose true label marginal is skewed.
  says `none`. In the bench it was the low-variance choice: +1 to +2
  accuracy points and a large ECE improvement on every model and task, with
  no task where it hurt by more than a point. It assumes the label marginal
  of the batch is not extreme.
- *Contextual calibration* (Zhao et al., 2021), opt-in via
  `prior="content_free"`. The prior is the answer distribution on
  content-free inputs (`N/A`, empty, `[MASK]`), computed once per question
  and cached, in a forward call of their own so the probes never change the
  batch composition (and with it the bf16 logits) of the real states. High variance: +8 to +12 points on a prompt-injection `noul`
  on three models, but -3 on ordinal `score` questions and -9 on one model's
  `noul` questions in the typed-decisions set. For some questions the model's
  answer to an empty input is an honest answer, not a label prior, and
  dividing by it pushes every real answer the other way. The bench prints
  both (`L0-perm+bc` and `L0-perm+cf`), so measure before switching.

**Cyclic-shift marginalization** (Zheng et al., 2024). For a `choice` with K
options, show the list in K rotations so every option sits at every position
once, and combine. AnyJev combines in log space (geometric mean) by default:
if the position bias is additive in logit space, this removes it exactly and
the result no longer depends on how you listed the options. Arithmetic-mean
combination, the form in the paper, is available as `combine="mean"`.

For `noul`, the two phrasings "Answer Yes or No" and "Answer No or Yes" are
both read and combined. For `score`, the bins are ordinal and are never
permuted; only the prior correction applies.

Cost: K prompts per choice question (2 for noul, 1 for score), all sharing the
state prefix. `max_permutations` caps K.

What L0 does not do: it does not make the model's own uncertainty
calibrated. A model that is overconfident on everything is still overconfident
after L0. That needs labels.

## L1: calibrated on labels

Temperature scaling fit on 100 to 500 labeled examples of the same question,
applied on top of L0. The prior used to score the calibration set is estimated from that set
alone and frozen into the artifact, so the artifact is a pure function of (model, question,
calibration set) and L1 decisions do not depend on what else the decider has scored (an
independent reproduction found the earlier running prior made L1 drift by up to 0.007 in
coverage at 5% risk). Two consequences worth knowing. The prior needs no labels but it does
need states: with 200 calibration items on a 20-way question it is a noisier estimate than the
batch prior L0 accumulates over everything it has seen, which on the smallest models costs up to
two points of AURC against the earlier, history-dependent L1 while ECE still improves; a larger
calibration set tightens it. And the prior is indexed by (permutation, position), so it belongs
to the option order it was fit on: `load_artifact` refuses an artifact whose prior was fit on a
different layout of the same question. The fitted temperature and the prior form a small JSON
artifact keyed by (model, question hash). Loading an artifact fit on a different model is an
error.

What L1 does not do: survive distribution shift beyond the calibration set,
or fix a model that is simply wrong. It reshapes confidence; it does not
change the ranking.

Planned for L1: Dirichlet calibration for multi-class, histogram binning,
and split-conformal abstention with a user-set target error rate.

## Reading the bench columns

- `acc`, `macro_f1`: are the answers right.
- `brier`, `nll`: are the probabilities right (lower is better).
- `ece`: expected calibration error on top-1 confidence, 15 equal-mass bins.
- `flip`: fraction of items whose argmax changes when the option list is
  reversed (`choice`) or the phrasing is swapped between "Yes or No" and
  "No or Yes" (`noul`). Measured at every level. This is the number the
  clones' READMEs admit to and nobody tabulates.
- Ablation rows, all from the same forward passes: `L0-cf` content-free
  prior alone, `L0-bc` batch prior alone, `L0-perm` permutation alone,
  `L0-perm+cf` permutation plus content-free prior, `L0-perm+bc`
  permutation plus batch prior, `L0` whatever prior the run was configured
  with (default: batch, so `L0` equals `L0-perm+bc`).
- `cov@5%`: fraction of items you can answer, ordered by confidence, before
  the error rate on the answered set exceeds 5 percent. Higher is better.
