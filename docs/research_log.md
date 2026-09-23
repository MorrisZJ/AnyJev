# Research log: question-agnostic heads and the cheap Jev mode

One entry per experiment: date, what ran, where the JSON is, what the gate said, one reading.
Numbers here are copied from the run logs and JSON, never typed from memory. Entries are kept as
written on the day; `docs/method_v3.md` is the settled account.

**Read this first (version 3, 2026-09-23).** Three things changed after the entries were written:

1. Commands naming `bench.universal_e0`, `bench.universal_study`, `bench.cascade_study`,
   `bench.distill_study` and `bench.heads_study` refer to research code removed in version 3
   (`anyjev/universal.py`, `anyjev/cascade.py`, those five modules, `bench/heads_table.py` and their
   17 tests). The block-loop cache extraction they shared lives in `bench/pools.py` and
   `bench/extract_pools.py`, with its three tests in `tests/test_pools.py` (`docs/migration_v3.md`).
2. The cascade code (`anyjev/cascade.py`, `bench/cascade_study.py`) was removed in version 3; its two result
   files, `bench/results_exit/2026-09-22/Qwen__Qwen3-8B.cascade.json` and `Qwen__Qwen3-32B.cascade.json`, are
   kept, and the tables of entries 4 and 10 are re-derived from them (every cell was checked against the
   `rows` of the JSON on 2026-09-23; one row of entry 4 was corrected, see there).
3. `logs/universal/*.log` were not committed. Every other JSON named below is in the tree.

## 0. Phase 0 kill tests on the existing Qwen3-8B typed cache (2026-09-22)

Command: `python -m bench.universal_e0 --model Qwen/Qwen3-8B --study-json bench/results_heads/2026-09-22/Qwen__Qwen3-8B.typed.json`
(CPU replay of `bench/results_heads/features/Qwen__Qwen3-8B/typed.*.npz`, last-position features at
blocks 36/32/28/24/20 of 36). Log: `logs/universal/e0_qwen3_8b.log`; JSON:
`bench/results_universal/e0/Qwen__Qwen3-8B.e0.json`.

**E0.1 pooled noul head on last-position features, leave-one-question-out over the 6 noul questions.**
Mean accuracy over the held-out questions: raw 0.705, L0-perm 0.682, pooled LDA 0.667 (train on the
canonical order), 0.673 (train on random orders), 0.657 (shift-averaged); pooled diff-of-means 0.707 /
0.708 / 0.738; per-question head 0.85. Reading: a head on the *last-position* state does not transfer
across noul questions, even though every noul question shares the Yes/No option set. The last position
encodes the question, not a comparison; the foothold the plan hoped for is absent. This is the negative
result the option-line design (E1) has to beat.

**E0.2 depth of the per-question head.** Refit on one cached layer at a time, pooled over 20 questions:
block 36 acc 0.764 / NLL 0.674, block 32 0.761 / 0.658, block 28 0.762 / 0.699, block 24 0.764 / 0.715,
block 20 0.717 / 0.737. Cross-validated layer choice over the 20 questions: block 24 x6, 28 x6, 32 x3,
36 x5. Reading: two thirds of the depth carries the whole per-question gain on this model; the drop
starts between 67% and 56% of depth. Gate D1 ("<= 0.6 of depth keeps >= 95% of accuracy") is met at
0.67 and missed at 0.56; the next study samples every block between 18 and 28.

**E0.3 honesty: the per-question head with intervals, the pooled ECE definition and a gradient
reference.** Pooled over the 2000 test decisions (item bootstrap, 1000 resamples; ECE as
`bench.run_typed.pooled` computes it, 15 equal-mass bins over the pooled confidences):

| method | acc | 95% CI | acc - L0 (95% CI) | ECE (pooled) | brier_mean | soft_acc |
|---|---|---|---|---|---|---|
| raw | 0.626 | [0.606, 0.648] | [-0.018, -0.000] | 0.327 | 0.210 | 0.520 |
| L0-perm | 0.635 | [0.615, 0.657] | | 0.318 | 0.209 | 0.525 |
| L1 | 0.626 | [0.606, 0.648] | [-0.018, -0.000] | 0.041 | 0.146 | 0.463 |
| per-question head (cv-chosen, canonical order) | 0.769 | [0.750, 0.787] | [+0.111, +0.158] | 0.058 | 0.099 | 0.563 |
| per-question head, fit on random orders (1 prefill) | 0.759 | [0.739, 0.778] | [+0.100, +0.147] | 0.055 | 0.102 | 0.560 |
| logistic regression on the same features (reference, not shipped) | 0.770 | [0.751, 0.789] | [+0.111, +0.157] | 0.016 | 0.094 | 0.563 |

Readings. (1) The README preview's per-question ECE of 0.120 was the mean of per-question ECEs at
n = 100 (binning noise); the pooled definition every other row uses gives 0.058, and 0.055 for
the random-order head. (2) The gradient-trained logistic regression on identical features gains
+0.002 accuracy over the closed-form head: gate G-E passes, closed form is not leaving accuracy
on the table (it does leave calibration: LR's ECE 0.016 vs 0.058, a temperature-only gap). (3) On
the soft metrics the picture against the published rows is: soft_acc 0.563 vs Laya-ft 0.471 and
Jev 0.580; brier_mean 0.099 vs Laya-ft 0.118 and Jev 0.148.

## 1. E1, leave-one-question-out universal head (2026-09-22, running)

Three variants on the same 20 typed questions (train on 19, test the 20th): training orders `rand`
vs `id`, option-line token `last` vs `newline`; forms probe / clogit / residual; layer and ridge
strength by grouped 5-fold CV over the training questions. Logs `logs/universal/loqo_*.log`, JSON
`bench/results_universal/<date>/Qwen__Qwen3-8B.loqo.<order>.<which>.json`. First held-out questions
under rand/last: action (K=4) raw 0.53, L0 0.62, per-question head 0.68, probe 0.59, clogit 0.38,
residual 0.56 (dot 0.02: it always picks the first-listed line; dropped); needs_review (noul) raw
0.84, L0 0.80, per-question 0.88, probe 0.78, clogit 0.76, residual 0.72. Under id/last: action
probe 0.57, clogit 0.54, residual 0.54; needs_review 0.75 / 0.70 / 0.79. Two more forms queued for
round 2: `pos` (a question-agnostic head over *positions* on the last-position state, i.e. better
unembedding rows for A/B/C/D, K-agnostic by truncation) and `prod` (option line times last position).


## 2. E6, accuracy versus depth on Qwen3-8B (2026-09-22)

Block-loop extraction of every 2nd block from 12 to 36 (`bench.universal_study --use-loop
--extract-only --out bench/results_exit`, parity against the plain forward is bit-exact in fp32:
`scripts/exit_parity.py` on Qwen3-1.7B), then `bench.exit_study`. Typed-decisions, 20 questions,
per-question head = LDA/ridge chosen by CV on the 300 train decisions, fit on one block's
last-position state; the logit lens is the model's own readout at that block (final norm + the
label rows of lm_head). JSON: `bench/results_exit/2026-09-22/Qwen__Qwen3-8B.depth.json`.

| block | depth | lens acc | lens+T ECE | head acc | head ECE (pooled) | head flip |
|---|---|---|---|---|---|---|
| 12 | 33% | 0.283 | 0.100 | 0.636 | 0.039 | 0.111 |
| 16 | 44% | 0.335 | 0.089 | 0.636 | 0.042 | 0.098 |
| 18 | 50% | 0.293 | 0.115 | 0.665 | 0.036 | 0.094 |
| 20 | 56% | 0.274 | 0.141 | 0.733 | 0.032 | 0.076 |
| 22 | 61% | 0.305 | 0.108 | 0.770 | 0.045 | 0.072 |
| 24 | 67% | 0.386 | 0.108 | 0.771 | 0.034 | 0.069 |
| 26 | 72% | 0.555 | 0.081 | 0.770 | 0.030 | 0.080 |
| 28 | 78% | 0.610 | 0.055 | 0.778 | 0.041 | 0.086 |
| 30 | 83% | 0.631 | 0.051 | 0.782 | 0.033 | 0.083 |
| 36 | 100% | 0.626 | 0.043 | 0.767 | 0.026 | 0.083 |

Readings. The model's own readout is at chance until block 26 (72% of depth) and only reaches
its final 0.626 at block 28; a closed-form head reads 0.770 out of block 22 (61% of depth), the
same as at full depth, and 0.636 (= L0 at full depth) out of block 12, a third of the depth. The
decision is formed in the residual stream long before the vocabulary head can express it. Gate
D1 passes (95% of the full-depth head accuracy at <= 0.6 depth: 0.733 at 56%, 0.770 at 61%). The
head's ECE stays at 0.03-0.05 at every depth, flip 0.07-0.11 (1 prefill, random-order training).

## 3. E1 status after 3-5 held-out questions (2026-09-22, 15:25)

Universal option-line heads (rand/last): action 0.59/0.38/0.56 (probe/clogit/residual) vs L0
0.62; needs_review 0.78/0.76/0.72 vs 0.80; outcome 0.60/0.63/0.52 vs 0.66; risk 0.55/0.51/0.44
vs 0.47 (per-question 0.80). The newline-token variant and the id-order variant are no better.
The positional head (`pos`, K-agnostic rows for A/B/C/D on the last-position state) is below raw
on 5 of 5 questions so far. Reading so far: a single linear rule over option-line states does not
carry across these questions; the per-question head's gain is question-specific. The full 20-fold
tables decide gate G1; the depth result above does not depend on it.


## 4. E7, confidence-gated cascade on Qwen3-8B (2026-09-22, 15:33)

`bench.cascade_study --exits 18,22,26,30,36` on the 13-layer caches (the study code was removed in
version 3; the JSON is kept): per-question heads at each exit, out-of-fold probabilities on the 300
train decisions, thresholds chosen on those (per question or one set pooled over the 20 questions),
evaluated once on the 2000 test decisions. JSON:
`bench/results_exit/2026-09-22/Qwen__Qwen3-8B.cascade.json` (`rows`: `cascade:<rule>:eps=<budget>:<scope>`
and `fixed@<block>`; `excess_error` is against the full-depth head).

| rule, budget, threshold scope | acc | ECE (pooled) | mean blocks | depth | excess vs full-depth head | exit shares (18/22/26/30/36) |
|---|---|---|---|---|---|---|
| quantile, eps=0, per question | 0.766 | 0.036 | 21.4 | 60% | +0.001 | .33/.57/.06/.02/.03 |
| quantile, eps=0, pooled | 0.769 | 0.046 | 21.5 | 60% | -0.002 | .24/.71/.02/.01/.02 |
| quantile, eps=0.01, pooled | 0.770 | 0.049 | 21.2 | 59% | -0.003 | .30/.65/.02/.01/.02 |
| quantile, eps=0.02, per question | 0.756 | 0.040 | 20.6 | 57% | +0.011 | .45/.50/.02/.01/.02 |
| Learn-then-Test, eps=0 (either scope), or any eps per question | 0.767 | 0.026 | 35.7 | 99% | 0.000 | .00/.01/.01/.00/.98 |
| Learn-then-Test, eps>=0.005, pooled | 0.775 | 0.048 | 22.7 | 63% | -0.008 | .00/.95/.00/.00/.05 |
| fixed depth 22 (reference) | 0.770 | 0.045 | 22 | 61% | | |
| fixed depth 36 (reference) | 0.767 | 0.026 | 36 | 100% | | |

Readings. On this model the cascade and the fixed block-22 head are the same thing: block 22
already answers everything, so the gate mostly routes between 18 and 22 (mean 21.4 blocks, 60% of
depth) at the full-depth accuracy. The certificate variant (Learn-then-Test with a Clopper-Pearson
bound at delta = 0.1) is vacuous at n = 300 per question and stays at full depth at every budget;
pooled over the 6000 out-of-fold items it still does not exit at eps = 0 (the early heads are
*better* than the final head on many items, and the noise of the excess-error statistic dominates a
zero budget), and from eps = 0.005 on it sends 95% of the items out at block 22, which is the fixed
block-22 head again (0.775, 22.7 blocks). Gate D2 (cascade earns its complexity over fixed depth) is
not met on the 8B; the 32B (64 blocks) is the case where a cascade could matter. The product
recommendation for the 8B is the fixed-depth head (block 22 here; entry 8 later settles on block 24
from calibration data alone). *Correction on re-derivation from the JSON (2026-09-23): the day's
table had one Learn-then-Test row, "eps <= 0.005, either scope, almost never exits"; the JSON has the
pooled rows at eps >= 0.005 exiting at block 22, so they are a row of their own above.*

## 5. Label efficiency of the per-question head, Qwen3-8B (2026-09-22, 15:33)

`bench.labels_study --blocks 22,28,36 --budgets 20,50,100,200,300 --seeds 0,1,2`, pooled over
the 20 typed questions, 3 calibration subsets each, paired against a temperature (L1) fit on the
same labels. JSON: `bench/results_exit/2026-09-22/Qwen__Qwen3-8B.labels.json`.

| labels per question | L1 (temperature) | head acc | head - L1 (95% CI) | head ECE (pooled) |
|---|---|---|---|---|
| 20 | 0.626 | 0.654 | [+0.012, +0.041] | 0.108 |
| 50 | 0.626 | 0.707 | [+0.068, +0.094] | 0.111 |
| 100 | 0.626 | 0.740 | [+0.100, +0.127] | 0.080 |
| 200 | 0.626 | 0.754 | [+0.114, +0.141] | 0.052 |
| 300 | 0.626 | 0.772 | [+0.131, +0.158] | 0.034 |

Readings. The crossover is at 20 labels (CI already excludes 0), 100 labels pass Jev's published
0.727 on its own benchmark, 300 match the fine-tuned Laya (0.768). Calibration needs more labels
than accuracy does: ECE is 0.11 at 50 labels and 0.03 at 300 (the temperature is fit on
out-of-fold scores of a tiny set). The story for a new question: 50 labels for a usable head,
100 to beat the hosted product's published number, 300 for its calibration.

## 6. E6 across model sizes and E9 cost: the head makes small models Jevs (2026-09-22, 15:50)

`bench.exit_study` on the block-loop caches of Qwen3-1.7B (28 blocks, every 2nd block from 8) and
Qwen3-4B (36 blocks, from 12), same protocol as entry 2 (per-question head, lda / ridge by CV, on
the last-position state; 300 train / 100 test decisions per typed question, pooled over 20). JSON:
`bench/results_exit/2026-09-22/Qwen__Qwen3-{1.7B,4B,8B}.depth.json`. The raw / L0 / L1 references
are the regenerated typed study (`bench/results_typed_v01/2026-09-22/`, same 2000 test decisions).

| model | raw | L0 | L1 | head, full depth | head, best block (test-selected) | head at ~60% depth | lens at 60% |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B (28) | 0.468 | 0.494 | 0.499 | 0.738 | 0.750 @ 26 (93%) | 0.709 @ 16 (57%), 0.730 @ 18 (64%) | 0.25 |
| Qwen3-4B (36) | 0.547 | 0.564 | 0.567 | 0.773 | 0.786 @ 24 (67%) | 0.753 @ 22 (61%) | 0.28 |
| Qwen3-8B (36) | 0.626 | 0.647 | 0.648 | 0.767 | 0.782 @ 30 (83%) | 0.770 @ 22 (61%) | 0.31 |
| Jev (published) | | | | 0.727 | | | |
| Laya, fine-tuned (measured) | | | | 0.768 | | | |

Readings. (1) The head's gain grows as the model shrinks: +0.24 over L1 on the 1.7B, +0.21 on the
4B, +0.12 on the 8B. The 1.7B with per-question heads (0.738-0.750) passes Jev's published 0.727;
the 4B (0.773-0.786) passes the fine-tuned Laya (0.768) and the 8B's own full-depth head. This is
the "any lightweight model into a Jev" row: 300 labels per question, a closed-form solve, no
gradient, no new weights. (2) The depth story holds on every size: the head at ~60% of depth is
within 1-3 points of full depth on the 4B and 8B, and the 1.7B needs ~65% (block 18). The logit
lens at those depths is at chance, so nothing gradient-free without a head reads early blocks.
(3) The per-block "best" is test-selected and therefore optimistic by ~1 point; entry 8 chooses
the depth on calibration data only. ECE of the head stays 0.02-0.04 at every depth (pooled).

Cost (`bench.exit_latency`, Qwen3-8B, one H100 NVL, bf16, CPU shared with four CPU jobs, JSON
`bench/results_exit/2026-09-22/Qwen__Qwen3-8B.latency.json`; ms per decision):

| state | depth | blocks | batched (32) | single request | GFLOPs per decision |
|---|---|---|---|---|---|
| 110 tokens (212 with template) | raw, full forward | 36 | 11.3 | 48.4 | 2945 |
| | 0.60 | 22 | 6.8 (0.60x) | 30.6 (0.63x) | 1800 (0.61x) |
| | 0.33 | 12 | 4.0 (0.36x) | 20.1 (0.41x) | 982 (0.33x) |
| 1000 tokens (1096) | raw, full forward | 36 | 61.1 | 67.7 | 15226 |
| | 0.60 | 22 | 38.3 (0.63x) | 43.8 (0.65x) | 9305 (0.61x) |
| | 0.33 | 12 | 22.1 (0.36x) | 26.9 (0.40x) | 5075 (0.33x) |

The truncated block loop is a plain forward (the full-depth loop costs the same as the model's own
forward, 10.7 vs 11.3 ms batched), so time scales with blocks executed; the head itself is one
4096 x K product. Single-request numbers at 110 tokens are launch-bound (~1.4 ms per block) and
were measured under CPU contention; they are re-measured on a quiet GPU host.

## 7. E8 distillation from the 32B's zero-label answers: bounded by the teacher (2026-09-22, 15:50)

`bench.distill_study label --teacher Qwen/Qwen3-32B` (L0, the 300 train states of every typed
question and the 300-state calibration splits of banking20 / newsgroups / injection) then
`fit --student Qwen/Qwen3-8B --teacher Qwen/Qwen3-32B` (per-question heads on the student's
block-loop cache at blocks 22 / 28 / 36 by CV, labels = gold, teacher argmax, or teacher
probabilities as soft targets). JSON: `bench/results_distill/2026-09-22/Qwen__Qwen3-8B.from.Qwen__Qwen3-32B.json`.

| labels per question | gold | teacher argmax (32B L0) | teacher soft |
|---|---|---|---|
| 50 | 0.675 | 0.648 | 0.492 |
| 100 | 0.726 | 0.674 | 0.530 |
| 200 | 0.757 | 0.693 | 0.586 |
| 300 | 0.773 | 0.688 | 0.611 |
| references | student raw 0.626, L0 0.647 | teacher L0 on typed 0.700 (agreement with gold on train 0.51-0.91 per question) | |

Readings. The student on teacher labels converges to the teacher's own zero-label accuracy (0.688
vs 0.700) and no further: it learns the 32B's biases, including its wrong answers on the hard
questions (agent_trace urgency: teacher 0.59, student-on-teacher 0.56, student-on-gold 0.68). 100
gold labels (0.726) beat 300 teacher labels. Soft targets are worse than argmax because the
teacher's L0 probabilities are flat (ECE 0.149) and a ridge to flat targets underfits. Gate D3
(student within 3 points of the teacher) is met trivially but in the wrong direction: the teacher
is the ceiling, and that ceiling is below 100 gold labels. Story: labels must come from the task;
a bigger model's zero-shot readout is a weak labeller on typed decisions (the same conclusion as
the self-label experiment of `bench.heads_study`, code not shipped: heads fit on the model's own
answers score 0.625 against raw 0.626, `bench/results_heads/2026-09-22/Qwen__Qwen3-8B.typed.json`). A stronger teacher would be
the 32B's *own* per-question head (0.8-class), which needs the gold labels anyway; that variant
("label amplification": 300 gold labels -> the 32B head labels thousands more states for a small
student) is left for when a student's learning curve is not yet saturated at 300, which on the 8B
it is (entry 5).

## 8. E11 product table: depth chosen on calibration data only (2026-09-22, 16:05)

`bench.jev_mode_table` on the block-loop caches of Qwen3-1.7B / 4B / 8B. Per question and block:
a per-question head (lda / ridge by CV) with its out-of-fold accuracy on the 300 train decisions.
Two deployment rules that never see test data: `cv` (each question picks its block by out-of-fold
NLL, what `Decider.fit_head` does) and `fixed@b` (one block per model: the shallowest whose pooled
out-of-fold accuracy is within 0.5 points of the best). Test numbers pooled over the 2000 test
decisions, 1000-resample bootstrap CIs paired against the full-depth head. JSON:
`bench/results_exit/2026-09-22/Qwen__Qwen3-{1.7B,4B,8B}.jevmode.json`.

| model | L1 | head, cv (mean depth) | head, fixed block (depth) | head, full depth | fixed - full (95% CI) | ECE fixed | flip fixed |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B (28) | 0.499 | 0.733 (72%) | 0.730 @ 18 (64%) | 0.738 | [-0.024, +0.009] | 0.028 | 0.113 |
| Qwen3-4B (36) | 0.567 | 0.772 (75%) | 0.786 @ 24 (67%) | 0.773 | [-0.002, +0.028] | 0.034 | 0.073 |
| Qwen3-8B (36) | 0.648 | 0.774 (73%) | 0.771 @ 24 (67%) | 0.767 | [-0.008, +0.018] | 0.034 | 0.069 |
| Jev (published) | | | | 0.727 | | | |
| Laya, fine-tuned (measured) | | | | 0.768 | | | |

Readings. (1) Chosen on calibration data alone, the fixed block lands at 64-67% of depth on all
three sizes and is statistically indistinguishable from the full-depth head (every CI covers 0).
The out-of-fold curves are flat from ~60% of depth upward (8B: 0.771 at 22, 0.777 at 24, 0.768 at
36), so the recommendation is robust to the slack. (2) The per-question `cv` rule is not better
than the fixed block and costs more (it picks deeper blocks on noisy questions): the shipped
default is one block per model. (3) Cost at the fixed block on the 8B (entry 6): 0.60x the batched
ms per decision of the plain forward, 0.61x the FLOPs. (4) The Jev row: a 1.7B at 64% depth
(0.730) matches Jev's published 0.727 on its own suite; a 4B at 67% depth (0.786) passes the
fine-tuned Laya and every zero-label number we have from any model, including the 32B's L0
(0.700). Whether the 32B's own head at a third of its depth beats the 4B's is the open row
(extraction finished 15:47, studies running).

## 9. E6 / E11 on Qwen3-32B, and E3 cross-domain (2026-09-22, 16:20)

**32B depth and product table.** `bench.exit_study` and `bench.jev_mode_table` on the 32B block-loop
cache (blocks 16..64 every 4th, 13 layers, d = 5120). JSON:
`bench/results_exit/2026-09-22/Qwen__Qwen3-32B.{depth,jevmode,latency}.json`.

| block | depth | head acc (test) | head ECE | pooled OOF acc (calib) | lens acc |
|---|---|---|---|---|---|
| 28 | 44% | 0.724 | 0.042 | 0.717 | 0.35 |
| 44 | 69% | 0.729 | 0.034 | 0.726 | 0.28 |
| 48 | 75% | 0.783 | 0.065 | 0.772 | 0.33 |
| 52 | 81% | 0.798 | 0.046 | 0.792 | 0.42 |
| 56 | 88% | 0.794 | 0.036 | 0.788 | 0.58 |
| 64 | 100% | 0.795 | 0.035 | 0.788 | 0.68 |

Rules chosen on calibration data: `fixed@52` 0.798 [0.780, 0.815] (vs full depth [-0.009,
+0.016]), `cv` 0.794 at a mean of 51.6 blocks. References: raw 0.684, L0 0.700, L1 0.699.
Cost (`bench.exit_latency`, one H100 NVL, 110-token states, batched): full forward 41.5 ms per
decision, block 48 30.8 ms (0.74x), block 32 21.4 ms (0.52x); at 1000-token states 232.6 / 173.2 /
115.9 ms.

Readings. (1) The 32B's Jev mode is 0.80 at 81% of its depth: the decision is not readable
before block 48 (the out-of-fold curve jumps from 0.726 at 44 to 0.772 at 48 and 0.792 at 52),
so this model does not truncate as far as the 4B / 8B (67%). (2) It is the best number in the
programme (+0.10 over its own L0, +0.07 over Jev, +0.03 over the fine-tuned Laya), but it costs
about 0.8 of a 32B forward (34 ms per decision batched) where the 4B head gives 0.786 for roughly
a tenth of that. The product table therefore has two honest rows for "the best Jev": the 4B head
when cost matters, the 32B head when the last point of accuracy does. (3) Gate D1 (>= 95% of
full-depth accuracy at <= 60% of depth) fails on the 32B (0.713 / 0.795 = 90% at block 36) and
passes on the 4B and 8B; the cascade (E7, running) is the remaining way to cut the 32B's cost.

**Cross-domain transfer (E3), Qwen3-8B.** `bench.universal_study --protocol cross` on 33 pools
(20 typed, 3 bench, 7 extra, 3 game oracles): universal heads fit on every other domain, tested on
the held-out domain. JSON: `bench/results_universal/2026-09-22/Qwen__Qwen3-8B.cross.rand.last.json`.

| test domain | raw | per-question head (reference) | universal clogit | universal pos | universal prod |
|---|---|---|---|---|---|
| typed (fit on bench + extra + games) | 0.628 | 0.768 | 0.417 | 0.476 | 0.445 |
| extra (7 tasks) | 0.702 | 0.752 | 0.624 | 0.636 | 0.638 |
| games (2048, minesweeper, maze) | 0.627 | 0.814 | 0.492 | 0.487 | 0.494 |
| pooled, 33 questions | 0.661 | 0.766 | 0.520 | 0.550 | 0.538 |

Readings. Gates G6 and G7 fail by 15-25 points; every universal form is far below raw on every
held-out domain. Together with the LOQO rows (entries 1, 3; the `pos` form finished at 0.532 vs
L0 0.635) the question-agnostic head is closed as a negative result: whatever the hidden state
encodes about "the right option" is not in a direction shared across questions at the option
lines or the last position, on this model, with these forms. The per-question head, by contrast,
is +0.05 to +0.19 over raw on every domain including the game oracles (0.814), which is the
transfer claim we can make: the *method* transfers, the *weights* do not.

## 10. E7 cascade on Qwen3-32B, and the shipped artifacts (2026-09-22, 16:35)

**Cascade (32B).** `bench.cascade_study --exits 24,32,40,48,64 --n-layers 64` (code removed in
version 3, JSON kept), thresholds on out-of-fold probabilities, test once. JSON:
`bench/results_exit/2026-09-22/Qwen__Qwen3-32B.cascade.json`; the block-52 reference row is from
`Qwen__Qwen3-32B.jevmode.json` (`rules.fixed@52`, entry 9), the cascade file has no exit at 52.

| rule, budget, scope | acc | ECE | mean blocks | depth | excess vs full-depth head | exit shares (24/32/40/48/64) |
|---|---|---|---|---|---|---|
| quantile, eps=0, pooled | 0.792 | 0.059 | 48.9 | 76% | +0.003 | .04/.09/.00/.67/.20 |
| quantile, eps=0.01, pooled | 0.786 | 0.062 | 43.6 | 68% | +0.009 | .24/.02/.01/.63/.11 |
| quantile, eps=0.02, pooled | 0.780 | 0.058 | 40.4 | 63% | +0.015 | .35/.02/.00/.55/.07 |
| quantile, eps=0, per question | 0.776 | 0.055 | 37.5 | 59% | +0.019 | .40/.18/.02/.27/.13 |
| Learn-then-Test, eps=0, pooled | 0.796 | 0.047 | 57.4 | 90% | -0.001 | .00/.00/.00/.41/.59 |
| fixed block 52 (reference, entry 9) | 0.798 | 0.046 | 52 | 81% | -0.003 | |
| fixed block 64 (reference) | 0.795 | 0.035 | 64 | 100% | | |

Readings. The pooled zero-budget cascade sits at 48.9 blocks for 0.792, against the fixed block
52 at 0.798: 6% fewer blocks for half a point, inside the CI but the wrong direction. Per-question
thresholds at n = 300 are too noisy (they exit 40% of items at block 24 and pay 2 points). Gate
D2 (mean blocks <= 0.85 x fixed depth at <= 1 point excess) fails on the 32B as it did on the 8B.
The cascade code was removed in version 3 and its JSON kept; the shipped Jev mode is one fixed
block per model.

**Shipped artifacts.** `scripts/build_heads.py` on the real backend (`Decider.fit_head` on the 300
train states, `decide_batch(level="L2")` on the 100 test states, 20 typed questions + banking20 /
newsgroups / injection). First builds let every question pick its block by CV among 4
candidates: Qwen3-1.7B typed pooled 0.741 (ECE 0.039, mean block 20.9 of 28), Qwen3-4B 0.767
(0.028, 27.8 of 36), Qwen3-8B 0.775 (0.045, 28.6 of 36); the artifact head applied to the study
cache's features agrees with the live probabilities to |dp| 0.0000 (1.7B, 4B; same extraction
code) and 0.0099 (8B; older cache, bf16 batch noise), so the shipped path reads the states the
studies did. Per-question CV lands 4-5 blocks deeper than the calibration-chosen fixed block for
no gain (entry 8), so the shipped files are rebuilt with one block per model (18 / 24 / 24 / 52;
kept under `anyjev-heads/`, the CV builds under `anyjev-heads-cv/`). One block per model also
means every (state, question) prompt stops at the same depth, so the questions asked of one state
batch into one call and can share the state's prefix KV (each question is still its own prompt).
JSON size at full float32
precision: 7-18 MB per model, which was a Hub-or-repo question (since compacted to base64
float32, 1.8-4.4 MB per model, `scripts/compact_heads.py`; loaders read both formats).

Fixed-block builds, validated live through `decide_batch(level="L2")` (typed pooled acc / ECE, ms per
decision at batch 16 on one H100 for the 300-400-token typed states; bench tasks acc banking20 /
newsgroups / injection): Qwen3-1.7B @18: 0.731 / 0.030, 5.8 ms, 0.785 / 0.585 / 0.970;
Qwen3-4B @24: 0.771 / 0.031, 10.2 ms, 0.850 / 0.630 / 0.980; Qwen3-8B @24: 0.768 / 0.044, 14.6 ms,
0.875 / 0.685 / 0.995; Qwen3-32B @52: 0.791 / 0.055, 64.4 ms, 0.840 / 0.735 / 0.995. Agreement with
the study caches |dp| = 0.0000 (1.7B, 4B, 32B) and 0.009 (8B). The live 4B (0.771) is 1.5 points
under its replayed table row (0.786): the replay fits on features under random listing orders, the
live build on the canonical order; both are inside the CI. JSON: `<model>.artifact.json`.

## 11. E1 final: the question-agnostic head is closed (2026-09-22, 17:20)

Leave-one-question-out on Qwen3-8B, 20 typed questions, five cached blocks, every form fit on 19
questions and tested on the 20th with grouped inner CV for the block and the ridge strength
(`bench.universal_study --protocol loqo`). Four variants of the extraction: training listing
order random vs canonical, option-line state at the last token of the line vs at the newline,
and the position form. JSON: `bench/results_universal/2026-09-22/Qwen__Qwen3-8B.loqo.{rand.last,id.last,rand.newline,rand.last.pos}.json`.

| variant | raw | L0-perm | per-question head | universal probe | universal clogit | universal residual | universal pos |
|---|---|---|---|---|---|---|---|
| rand / last | 0.628 | 0.635 | 0.768 | 0.562 [-0.093, -0.054 vs L0] | 0.507 | 0.539 | 0.532 |
| id / last | 0.628 | 0.635 | 0.768 | 0.511 | 0.527 | 0.572 [-0.082, -0.046] | |
| rand / newline | 0.628 | 0.635 | 0.768 | 0.524 | 0.536 | 0.580 [-0.073, -0.037] | |

By kind (rand / last, mean accuracy): choice raw 0.603 / L0 0.652 / per-question 0.732 / best
universal 0.602; noul 0.705 / 0.682 / 0.847 / 0.658; score 0.589 / 0.589 / 0.736 / 0.461. Flip
under reversal of the universal forms is 0.17-0.30 against 0.07 raw.

Verdict. Gate G1 (LOQO >= L0 + 5 with the CI excluding 0) fails by 6-13 points in the wrong
direction for every form and variant; the best universal number on any held-out question set
is below raw. G3 (flip) fails. Together with the cross-domain rows (entry 9) this closes the
question-agnostic direction on this model: nothing in the option-line or last-position hidden
state that we can read with a shared linear rule predicts the right option on a question the
rule never saw. The per-question head remains +0.13 over L0 at the same depth. The code was removed in version 3;
the extraction machinery lives in `bench/pools.py` and `bench/extract_pools.py`, and the protocol
JSONs stay under `bench/results_universal/` as the record of the kill test.

## 12. Qwen3-30B-A3B (MoE) depth and product table; latency re-measured on a quiet GPU host (2026-09-22, 17:45)

**30B-A3B.** Block-loop cache every 4th block from 12 to 48 (d = 2048, batch 8, eager MoE);
`bench.exit_study` and `bench.jev_mode_table`. JSON:
`bench/results_exit/2026-09-22/Qwen__Qwen3-30B-A3B-Instruct-2507.{depth,jevmode}.json`.

| block | depth | head acc (test) | head ECE | pooled OOF acc (calib) | lens acc |
|---|---|---|---|---|---|
| 24 | 50% | 0.724 | 0.021 | 0.722 | 0.26 |
| 28 | 58% | 0.768 | 0.027 | 0.776 | 0.27 |
| 32 | 67% | 0.777 | 0.034 | 0.784 | 0.32 |
| 36 | 75% | 0.784 | 0.030 | 0.788 | 0.34 |
| 40 | 83% | 0.799 | 0.029 | 0.794 | 0.46 |
| 48 | 100% | 0.789 | 0.032 | 0.786 | 0.60 |

Rules chosen on calibration data: `fixed@40` 0.799 [0.780, 0.817] (vs full depth [-0.004,
+0.022]), `cv` 0.788 at 38.4 blocks. References: raw 0.599, L0 0.630, L1 0.630. Readings: the MoE
matches the dense 32B (0.799 vs 0.798) with 3B active parameters per token, from an L0 that is 7
points lower (0.630 vs 0.700): the largest L1-to-L2 gain of the five models (+0.17). Its
decisions become readable earlier than the 32B's (0.768 at 58%, 0.784 at 75%) and the OOF curve
is flat from block 32, so the 0.5-point slack lands on block 40; block 32 (67%) would cost 2
points. Latency is not reported for this model: transformers runs the experts in eager mode,
where wall clock does not reflect the active FLOPs (the cost story for MoE Jev mode needs a
served backend that exposes hidden states).

**Latency, quiet GPU host** (`bench.exit_latency`, load average 26 instead of 60-70, batched / single
ms per decision at the fixed block relative to the plain forward; 110-token / 1000-token states;
JSON `<model>.latency.json`, the contended measurements kept as `<model>.latency_loaded.json`):
Qwen3-1.7B @18: 1.00x / 0.70x batched (the 110-token forward of a 1.7B is launch-bound, 4 ms
either way), 0.68x / 0.71x single; Qwen3-4B @24: 0.67x / 0.69x batched, 0.67x / 0.71x single;
Qwen3-8B @24: 0.68x / 0.68x batched, 0.69x / 0.77x single; Qwen3-32B @52: 0.82x / 0.84x
batched, 0.83x / 0.84x single. The ratios equal the depth fraction to within launch overhead, as
they should for a loop that at full depth costs the plain forward (1.00-1.04x). Per-decision
absolute numbers at 1000 tokens, batched: 14.5 / 29.9 / 42.0 / 194.6 ms for the 1.7B / 4B / 8B /
32B Jev modes versus 20.7 / 43.3 / 62.1 / 240.1 ms for their plain forwards. `docs/results_exit.md`
is regenerated from these files by `bench.exit_table`.

**30B-A3B artifact** (`scripts/build_heads.py --layers 40 --batch-size 8`): typed pooled 0.798 / ECE 0.026
live, banking20 0.865 / newsgroups 0.755 / injection 0.985, |dp| vs cache 0.0000, 217 ms per decision in
eager MoE mode, 7.2 MB. Five shipped artifacts in total.

## 13. E13, wording and listing-order robustness of a per-question head, and label-free test-time adaptation (2026-09-22, 19:00)

The question: a head is bound to one question layout; what happens when the same question
arrives reworded, or with its options in another order? `bench/tasks/typed_paraphrases.json`
holds, for each of the 20 typed questions, three hand-written rewordings of the question (w1
light, w2 restructured, w3 reframed) and one rewording of the option descriptions or of the
Yes-means / No-means lines (o1); keys, K and order unchanged. `bench.paraphrase_study extract`
runs the deployed L2 path (`Decider._features`, canonical listing, one prompt per state) for the
300 train and 100 test states of every variant on Qwen3-4B and Qwen3-8B, plus raw / L0 under
every variant; `fit` replays on CPU. JSON: `bench/results_paraphrase/2026-09-22/<model>.paraphrase.b{24,36}.json`.

Rows: the head fit on the original wording applied to the reworded prompts as is; the same head
with its feature standardisation (mean, scale) re-estimated on the variant's 300 train states
WITHOUT labels (`ta`) or on the 100 test states themselves (`tt`); a head refit on the variant's
own labelled states (`own`); a head fit on the other wordings (`mix`: each state under one random
other wording, 300 rows; `all`: every state under every other wording).

Qwen3-4B, block 24 (2000 test decisions per row; CI vs the original-wording head on its own wording):

| variant | raw | L0 | original head as is | + recentred, unlabelled train | + recentred, test | own refit | mix | all | flip vs original | L0 flip vs original |
|---|---|---|---|---|---|---|---|---|---|---|
| orig | 0.547 | 0.569 | 0.771 | | 0.763 (control) | | | | 0 | 0 |
| w1 | 0.576 | 0.589 | 0.656 [-0.137, -0.092] | 0.756 | 0.751 | 0.769 | 0.683 | 0.620 | 0.300 | 0.146 |
| w2 | 0.567 | 0.562 | 0.626 [-0.168, -0.121] | 0.748 | 0.746 | 0.766 | 0.645 | 0.600 | 0.318 | 0.198 |
| w3 | 0.575 | 0.592 | 0.679 [-0.113, -0.070] | 0.743 | 0.740 | 0.761 | 0.711 | 0.613 | 0.279 | 0.177 |
| o1 | 0.553 | 0.567 | 0.665 [-0.128, -0.085] | 0.757 | 0.746 | 0.776 | 0.734 | 0.628 | 0.260 | 0.128 |

Qwen3-8B, block 24:

| variant | raw | L0 | original head as is | + recentred, unlabelled train | + recentred, test | own refit | mix | all | flip vs original | L0 flip vs original |
|---|---|---|---|---|---|---|---|---|---|---|
| orig | 0.628 | 0.648 | 0.767 | | 0.766 (control) | | | | 0 | 0 |
| w1 | 0.603 | 0.632 | 0.701 [-0.083, -0.049] | 0.753 | 0.745 | 0.770 | 0.724 | 0.642 | 0.182 | 0.100 |
| w2 | 0.621 | 0.640 | 0.646 [-0.142, -0.100] | 0.758 | 0.753 | 0.770 | 0.653 | 0.572 | 0.265 | 0.158 |
| w3 | 0.593 | 0.630 | 0.699 [-0.088, -0.051] | 0.744 | 0.744 | 0.770 | 0.732 | 0.603 | 0.232 | 0.129 |
| o1 | 0.619 | 0.643 | 0.745 [-0.038, -0.007] | 0.757 | 0.757 | 0.765 | 0.740 | 0.658 | 0.143 | 0.119 |

Unlabelled budget for the recentring (block 24, pooled over the four rewordings, mean of 3 draws):

| n unlabelled states | Qwen3-4B | Qwen3-8B |
|---|---|---|
| 0 (head as is) | 0.657 | 0.698 |
| 10 | 0.726 (mean only 0.737) | 0.730 (mean only 0.737) |
| 30 | 0.747 | 0.747 |
| 100 | 0.749 | 0.751 |
| 300 | 0.751 | 0.753 |
| refit with 300 labels | 0.768 | 0.769 |

Block 36 (full depth) tells the same story on both models: as is 0.66-0.73, recentred 0.74-0.76,
refit 0.77-0.79; 30 unlabelled states 0.74.

Readings. (1) A head is NOT robust to wording as is: rewording the question moves its accuracy
from 0.77 to 0.63-0.68 and changes the answer on 26-32% of the states. The model's own zero-label
answer also moves (L0 flips on 13-20% of the states under a rewording, at unchanged accuracy), so
part of this is the model, not the head. (2) The damage is almost entirely a shift and a
rescaling of the hidden-state distribution: re-estimating the head's mean and scale on unlabelled
states under the new wording recovers 0.74-0.75 with 30 unlabelled states (0.74-0.76 with 300),
within 1-3 points of a labelled refit (0.77), and 10 states already give 0.73. No labels, no
gradient, a mean vector. Recentring on the head's own wording costs 0.8 points (control), so it is applied
only when a question is served by another layout's head. (3) Fitting on several wordings does
not substitute: `mix` (same 300 labels spread over wordings) is 0.65-0.73 and `all` (every state
under every other wording) 0.60-0.63; stacking wordings blurs the class structure the LDA needs.

**Listing order** (`bench.paraphrase_study order`, replayed from the exit-study caches at block
24: heads fit on the canonical order or on a random order per calibration state, tested under
the canonical and the reversed listing; JSON `<model>.order.json`):

| model | head fit on | canonical | reversed | reversed + recentring | flip reversed vs canonical |
|---|---|---|---|---|---|
| Qwen3-8B | the canonical order only | 0.768 | 0.672 | 0.729 | 0.176 |
| Qwen3-8B | a random order per calibration state | 0.771 | 0.760 | 0.751 | 0.069 |
| Qwen3-4B | the canonical order only | 0.771 | 0.677 | 0.716 | 0.183 |
| Qwen3-4B | a random order per calibration state | 0.786 | 0.756 | 0.749 | 0.073 |

Readings. A head fit on one listing order latches onto the position code (reversing the options
costs 9-10 points and flips 18% of the answers); fit on random listing orders, at the same labels
and the same one forward, it loses 1-3 points under reversal and flips 7%, which is the raw
readout's own flip rate. Recentring helps the one-order head part of the way and does nothing
more for the random-order head. So random listing orders at fit time are free for the typed questions (K <= 5). They are NOT
free at K = 20: rebuilding the shipped heads with random listings left the typed heads unchanged
(1.7B 0.728, 4B 0.784, 8B 0.763, 32B 0.796, 30B-A3B 0.794) but cost the K = 20 bench heads 6-18
points (banking20: 8B 0.875 -> 0.755, 4B 0.850 -> 0.725, 1.7B 0.785 -> 0.600, 32B 0.840 -> 0.745;
newsgroups 4-14 points): with 20 options and 300 states the position code is noise the head cannot
average out. Default therefore `fit_head(listing="auto")`: random up to 8 options, canonical above
(`Decider.RANDOM_LISTING_MAX_K`); the artifacts are rebuilt under that rule.

**What landed in the library** (`anyjev/decider.py`, tests in `tests/test_decider_fake.py`):
`fit_head(listing="random")` by default; `Decider.route(q)`: exact layout, else the same kind and
option texts under another wording, else (random-listing heads only) the same option set in
another order with the probabilities mapped back by option text; label-free test-time adaptation
(`Decider(adapt="routed" | True | False, adapt_min_n=30)`: running mean and scale of the L2
features per asked question, used once 30 states have been seen, including the current batch);
`level="auto"` (L2 where a head routes, else L1 where an artifact exists, else L0); diagnostics
`routed_from`, `reordered`, `adapted`, `adapt_n`, `listing`. This is the "test-time training"
reading of AnyJev: heads are solved from labels in seconds, and a head then follows its question
across wordings and listings from the unlabelled traffic alone.

## 14. E14, closed-form distillation of the 32B's heads into small models (2026-09-22, evening)

The ask: a gradient-free distillation of a strong model's decisions into the
light closed-form heads, using our own strongest open teacher rather than Jev (TypeSafe's terms
and the GPU host's network both argue against querying Jev; and the teacher we have, the 32B's L2
head at 0.80, is above the 0.727 listed for Jev anyway). Two routes, `bench/distill_heads.py`.

**Route 1, soft targets on the same states** (`soft`): the teacher head's out-of-fold
probabilities on the 300 train states (5-fold, block 52) become the student's targets; the
student head is a ridge to those targets (lambda and temperature chosen by out-of-fold
cross-entropy against the teacher, no gold label used), or to a 50/50 mix of gold one-hots and
teacher probabilities. JSON: `bench/results_distill/2026-09-22/Qwen__Qwen3-32B.distill_heads.soft.json`.

| student (block) | gold labels | teacher argmax | teacher soft | gold + teacher soft | teacher itself |
|---|---|---|---|---|---|
| Qwen3-1.7B (18) | 0.730 | 0.727 [-0.019, +0.015] | 0.741 [-0.005, +0.029] | 0.742 [+0.000, +0.025] | 0.798 |
| Qwen3-4B (24) | 0.786 | 0.761 [-0.041, -0.009] | 0.772 [-0.029, +0.003] | 0.779 [-0.018, +0.004] | 0.798 |
| Qwen3-8B (24) | 0.771 | 0.760 [-0.028, +0.004] | 0.778 [-0.008, +0.021] | 0.781 [-0.002, +0.022] | 0.798 |

Readings. With no new states, the teacher adds at most a point (1.7B +1.2, 8B +1.0, CIs touch
zero) and costs the 4B a point: soft targets from a calibrated head are a mild regulariser, not
a transfer of the teacher's knowledge, because the 300 states are the same ones the student
already has gold labels for. The teacher's information has to come through *new* states.

**Route 2, synthetic states** (`bench.synth_states` + `label` / `extract` / `fit`): Qwen3-8B
writes 1200 new cases per workflow from three real exemplars each (JSON with the same schema;
rejection rate below 1%), the 32B's shipped heads label them for the five questions of the
workflow, and the student heads are fit on 300 gold + n teacher-labelled synthetic states (hard
or soft targets), against gold-only and synthetic-only. Generation: 4 x 1200 states in about
45 GPU-minutes on one H100 (the customer-service threads are long); rejection below 1% once
every real schema of a workflow is accepted. JSON:
`bench/results_distill/2026-09-22/Qwen__Qwen3-32B.distill_heads.fit.json`; states under
`bench/results_distill/synth/`, teacher labels under `synth_labels/`.

| student (block) | gold 300 | synthetic only, 300 / 1200 (hard) | gold 300 + 300 synthetic, hard / soft | gold 300 + 1200 synthetic, hard / soft | teacher itself |
|---|---|---|---|---|---|
| Qwen3-1.7B (18) | 0.730 | 0.637 / 0.668 | 0.756 [+0.013, +0.041] / 0.744 | **0.760 [+0.015, +0.049]** / 0.745 | 0.795 |
| Qwen3-4B (24) | 0.786 | 0.695 / 0.716 | 0.782 / 0.783 | 0.785 [-0.015, +0.013] / 0.779 | 0.795 |
| Qwen3-8B (24) | 0.771 | 0.679 / 0.720 | 0.774 / 0.780 [-0.003, +0.023] | 0.770 / 0.777 | 0.795 |

Readings. (1) Distillation pays where the student is weakest: the 1.7B gains 3 points from 1200
teacher-labelled synthetic states on top of its 300 gold labels (0.730 -> 0.760, CI excluding 0;
Jev 0.727), and the gain is already there at 300 synthetic states. The 4B and 8B, within 1-2
points of the teacher on their own labels, gain nothing (the 8B +0.9 with soft targets, CI
touching 0). (2) Synthetic states alone are 6-9 points below the 300 gold states even with the
0.80 teacher labelling them: the generated cases are easier and narrower than the real ones
(an 8B generator writing from three exemplars), so the teacher's knowledge reaches the student
only as a supplement to real states, not as a replacement for labels. (3) Hard (argmax) targets
beat soft ones for the 1.7B and lose for the 8B; the difference is small either way. Gate D3
(student within 3 points of the teacher) is met by the 4B (-1.0) and 8B (-1.5) with gold labels
alone and not by the 1.7B (-3.5 after distillation). The honest product line: the 32B's Jev
mode distils into a 1.7B at 0.76 for the cost of 300 labels, 1200 generated states and a
closed-form solve; nothing in the pipeline takes a gradient (the generator only samples).
