# Multimodal results

Three image tasks, three Qwen3-VL sizes, 300 test items and 200 calibration
items each, seed 0, one H100 NVL per model, bf16. Every number below is
generated from the JSON under `bench/results_mm/`, `bench/results_mm_cf/` and
`bench/results_mcq/`, all produced by one command, `bash bench/run_mm.sh`
(transformers ≥ 4.57 for Qwen3-VL); nothing is typed in by hand. They are
measured on the same code as the text tables: L1 freezes the prior it was fit
with, the batch prior is applied at strength 0.75, content-free probes get
their own forward call, and the L1 flip compares two independently calibrated
layouts.

The tasks mirror the text bench rather than the multimodal-benchmark
convention, and so does the reporting: L0 is the library default, every other
prior is an ablation row from the same forward passes.

| text task | image task | shape |
|---|---|---|
| `banking20` | `pets20` — Oxford-IIIT Pet, 20 breeds | one question, one fixed 20-option label set |
| `injection` | `pope` — is this object in the image? | one `noul` over the whole set |
| — | `ai2d` — science diagrams | every item carries its own 4 options (`bench/run_mcq.py`) |

## Headline: raw → L0 → L1, library default

L0 is the batch prior, as on text. On `ai2d` every item is its own question,
so the batch prior has one state to pool over and L0 is permutation-only.
AURC is the area under the risk-coverage curve (lower is better); read it
before `cov@5%`, for the reason given under "Where it did not help".

| task | model | L0 prior | acc | ECE | flip | AURC | cov@5% |
|---|---|---|---|---|---|---|---|
| pets20 | Qwen3-VL-2B | batch | 0.797 → 0.827 → 0.827 | 0.152 → 0.119 → 0.029 | 0.163 → 0.080 → 0.077 | 0.075 → 0.054 → 0.052 | 0.550 → 0.640 → 0.680 |
| pets20 | Qwen3-VL-4B | batch | 0.820 → 0.843 → 0.840 | 0.164 → 0.139 → 0.098 | 0.127 → 0.067 → 0.073 | 0.078 → 0.043 → 0.044 | 0.287 → 0.647 → 0.687 |
| pets20 | Qwen3-VL-8B | batch | 0.880 → 0.903 → 0.903 | 0.104 → 0.080 → 0.054 | 0.147 → 0.050 → 0.060 | 0.034 → 0.024 → 0.024 | 0.743 → 0.810 → 0.810 |
| pope | Qwen3-VL-2B | batch | 0.917 → 0.913 → 0.910 | 0.046 → 0.047 → 0.046 | 0.053 → 0.000 → 0.000 | 0.024 → 0.018 → 0.018 | 0.913 → 0.920 → 0.923 |
| pope | Qwen3-VL-4B | batch | 0.877 → 0.877 → 0.877 | 0.110 → 0.109 → 0.050 | 0.010 → 0.000 → 0.000 | 0.038 → 0.032 → 0.032 | 0.660 → 0.723 → 0.723 |
| pope | Qwen3-VL-8B | batch | 0.840 → 0.847 → 0.843 | 0.147 → 0.142 → 0.081 | 0.010 → 0.000 → 0.000 | 0.063 → 0.048 → 0.049 | 0.477 → 0.620 → 0.610 |
| ai2d | Qwen3-VL-2B | none | 0.833 → 0.833 → 0.833 | 0.099 → 0.080 → 0.053 | 0.157 → 0.047 → 0.047 | 0.040 → 0.033 → 0.032 | 0.697 → 0.770 → 0.780 |
| ai2d | Qwen3-VL-4B | none | 0.863 → 0.873 → 0.873 | 0.106 → 0.096 → 0.046 | 0.100 → 0.047 → 0.047 | 0.024 → 0.021 → 0.022 | 0.780 → 0.817 → 0.823 |
| ai2d | Qwen3-VL-8B | none | 0.860 → 0.867 → 0.867 | 0.112 → 0.105 → 0.041 | 0.103 → 0.037 → 0.037 | 0.021 → 0.021 → 0.021 | 0.820 → 0.813 → 0.803 |

## What carried over from text

**Position debiasing, unchanged.** Reversing the option list flips 10 to 16
percent of answers on the K=20 `pets20` and the per-item `ai2d`; L0 cuts that
by roughly half to two thirds on every model (`pets20` 2B 0.163 → 0.080,
`ai2d` 8B 0.103 → 0.037). On the two-option `pope` it goes to exactly zero,
as on the text `injection`. Pixels in the state do not change the fact that
the model prefers some positions in the option list.

**Selective prediction, on the K=20 task.** On `pets20` 4B, AURC falls from
0.078 to 0.044 and coverage at 5 percent risk rises from 29 to 69 percent,
from accuracy that moves 2 points — the shape of `banking20` on text. AURC
improves or holds on every row of the headline table.

**L1 where L0 alone does not calibrate.** On `pope` the default L0 barely
moves anything: the true Yes rate is exactly 0.5 and the model's mean
prediction sits close to it, so the batch prior has little to divide out.
L1 does the work: ECE 0.110 → 0.050 on 4B and 0.147 → 0.081 on 8B. On
`pets20` and `ai2d` it takes ECE to 0.03–0.05 on 2B and 8B.

## The content-free prior: the largest win and the largest loss

Opt-in on text, opt-in here. It is in every full table below as the
`L0-perm+cf` row, and POPE with it as L0 has its own run:

| task | model | L0 prior | acc | ECE | flip | AURC | cov@5% |
|---|---|---|---|---|---|---|---|
| pope | Qwen3-VL-2B | content_free | 0.917 → 0.923 → 0.923 | 0.046 → 0.049 → 0.048 | 0.053 → 0.000 → 0.000 | 0.024 → 0.021 → 0.021 | 0.913 → 0.897 → 0.897 |
| pope | Qwen3-VL-4B | content_free | 0.877 → 0.910 → 0.910 | 0.110 → 0.060 → 0.126 | 0.010 → 0.000 → 0.000 | 0.038 → 0.038 → 0.038 | 0.660 → 0.480 → 0.480 |
| pope | Qwen3-VL-8B | content_free | 0.840 → 0.900 → 0.900 | 0.147 → 0.065 → 0.147 | 0.010 → 0.000 → 0.000 | 0.063 → 0.040 → 0.040 | 0.477 → 0.673 → 0.673 |

- **`pope`:** the largest win in the multimodal bench. 8B accuracy
  0.840 → 0.900 and ECE 0.147 → 0.065 with zero labels. Raw accuracy *falls*
  with model size (0.917, 0.877, 0.840) and the corrected rows converge near
  0.90: the larger models are not worse at seeing objects, they say Yes more,
  and the lean only shows when the picture is blanked.
- **`pets20`:** changes almost nothing (compare `L0-perm+cf` with `L0-perm`).
- **`ai2d`:** costs 8 to 24 accuracy points (2B 0.833 → 0.593). The question
  and options often give the answer away without the diagram, so the
  blank-image probe measures knowledge, not bias.

So on POPE there are two routes to a calibrated answer: the content-free L0
(zero labels, +6 accuracy points, ECE 0.065 on 8B) or the default L0 plus L1
(200 labels, accuracy essentially unchanged, ECE 0.081). The rule in
[docs/multimodal.md](multimodal.md) for when the first route is safe is stated
about task shape, not per dataset, and `pets20` tested it: the rule said
content-free would be *safe* there, and it was — but safe is not useful when
there is no label prior to remove.

## Where it did not help

We would rather you find these here than in production.

- **L1 on top of the content-free L0 made POPE worse** on 4B and 8B: ECE
  0.060 → 0.126 and 0.065 → 0.147, while test NLL fell (8B 0.637 → 0.371).
  The remaining errors are confident hallucinations, which dominate the NLL,
  so the NLL-optimal temperature (4.7 and 5.6) flattens every answer. One
  temperature cannot fix confident wrong answers; don't stack the two routes
  above.
- **Coverage at 5 percent risk is too noisy at n=300 to call small moves.**
  Between the two runs behind this page — the same prompts, where only the
  batch composition of the bf16 forward passes changed — the rows whose
  definition did not change moved by at most 0.010 in accuracy, 0.007 in
  ECE and 0.002 in AURC, but by up to 0.19 in coverage at 5 percent risk
  (`pope` 4B content-free: 0.673 then 0.480). One or two errors near the top
  of the confidence ranking decide where that cut lands. Read AURC; treat
  `cov@5%` moves under 0.2 as noise.
- **Accuracy gains are small under the default**, −0.7 to +3.0 points; `pope`
  2B loses 0.7. The gains are in stability and calibration, as on text.
- **One model family.** Everything here is Qwen3-VL.

## L2 on images: a closed-form head per question

L2 carries over to pictures unchanged: `fit_head` reads the language model's last-position
hidden state after the image prompt and solves a shrunk-LDA or ridge head on it, the block and
the head chosen by out-of-fold NLL on the calibration split alone. Each run fits L1 and L2 on
the same 200 calibration labels and scores L0, L1 and L2 on the same 300 test items, so every
comparison below is paired item by item; three split seeds per model (`bench/results_mm_l2/2026-09-23/`,
torch 2.5.1, transformers 4.57.6, batch 16; the raw / L0 / L1 tables above were run under
torch 2.10, so L2 is compared with the L1 of its own run, not with those rows). `ai2d` is out
of scope: every item is its own question, and a head needs one question with many labelled
states.

| task | model | seeds | L0 acc | L1 acc | **L2 acc** | L2 - L1 (range) | seeds CI > 0 | L1 ECE | **L2 ECE** | L1 AURC | **L2 AURC** | L1 cov@5% | **L2 cov@5%** | L2 flip | L2 block |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pets20 | Qwen3-VL-2B | 3 | 0.794 | 0.794 | **0.942** | +0.148 (+0.133 to +0.157) | 3 of 3 | 0.052 | **0.023** | 0.052 | **0.006** | 0.647 | **0.984** | 0.044 | 17/28, 20/28 |
| pets20 | Qwen3-VL-4B | 3 | 0.836 | 0.836 | **0.954** | +0.119 (+0.100 to +0.130) | 3 of 3 | 0.094 | **0.032** | 0.045 | **0.011** | 0.704 | **0.998** | 0.031 | 22/36 |
| pets20 | Qwen3-VL-8B | 3 | 0.896 | 0.896 | **0.926** | +0.030 (+0.027 to +0.033) | 0 of 3 | 0.055 | **0.038** | 0.028 | **0.016** | 0.782 | **0.953** | 0.050 | 22/36, 31/36, 36/36 |
| pope | Qwen3-VL-2B | 3 | 0.904 | 0.904 | **0.901** | -0.003 (-0.017 to +0.017) | 0 of 3 | 0.047 | **0.053** | 0.022 | **0.034** | 0.853 | **0.792** | – | 17/28, 20/28, 28/28 |
| pope | Qwen3-VL-4B | 3 | 0.867 | 0.867 | **0.892** | +0.026 (+0.013 to +0.037) | 1 of 3 | 0.062 | **0.061** | 0.041 | **0.040** | 0.700 | **0.762** | – | 18/36, 22/36, 25/36 |
| pope | Qwen3-VL-8B | 3 | 0.851 | 0.851 | **0.894** | +0.043 (+0.027 to +0.053) | 3 of 3 | 0.077 | **0.064** | 0.054 | **0.034** | 0.582 | **0.792** | – | 22/36, 36/36 |

"seeds CI > 0" counts the seeds whose 95% paired bootstrap interval on the L2 − L1 accuracy
difference (items resampled) lies above zero. L0 and L1 share their accuracy because a
temperature does not change the argmax. The flip is the disagreement of two independently fit
heads, one on the listed order and one on the reversed order, as for L1 in the text bench; a
`noul` head answers in one phrasing, so `pope` has none.

**What holds.**

- **`pets20` on 2B and 4B: +12 to +15 points, on every seed.** 0.79 → 0.94 and 0.84 → 0.95,
  every paired interval above zero, with ECE, AURC and coverage at 5% risk all better than L1:
  with 200 labels the head answers almost every test image inside a 5% error budget.
- **`pope` on 8B: +4 points, on every seed**, and the largest gain on the pictures the head
  never saw (next table).
- **Calibration** improves with accuracy on `pets20` (L2's ECE 0.02–0.04 against L1's
  0.05–0.09) and on the 8B's `pope` (0.077 → 0.064); on the 4B's `pope` it is unchanged.

**What does not, or not yet.**

- **`pope` on 2B: no gain.** −0.3 points on average (−1.7 to +1.7), ECE and AURC slightly
  worse than L1. A first single-seed run read +1.3 against the L1 row of the headline table
  above; the seeds say that was noise.
- **`pope` on 4B: +2.6 points**, but only one seed of three clears zero.
- **`pets20` on 8B: +3 points, no seed clears zero.** The 8B's L2 (0.93) ends below the 2B's and
  the 4B's (0.94–0.95), and its chosen block spreads widest across seeds (22, 31 and 36 of 36,
  where the 2B and 4B stay at 17–22 on `pets20`). Why is open.
- **No early stop on images.** A vision model derives its positions from the image grid, so the
  forward runs to the end; the diagnostics report `early_stop=False`. L2 is still one forward
  per state: 0.02–0.05 s per decision at batch 16 against K forwards for L0 (20 on `pets20`),
  and fitting a head takes 8–17 s.

**Pictures L2 saw at fit time.** POPE asks several questions about each COCO image and the
bench splits its questions at random, so about a third of the test items show a picture that
also appears in the calibration split, and 3 to 11 test states per seed are identical to a
calibration state (same picture, same object). A head on 2,048–4,096 features could memorise a
picture where a temperature cannot, so the gain is shown separately:

| task | model | test items, picture unseen (mean) | L2 - L1 unseen | test items, picture seen (mean) | L2 - L1 seen |
|---|---|---|---|---|---|
| pope | Qwen3-VL-2B | 200 | -0.000 | 100 | -0.010 |
| pope | Qwen3-VL-4B | 200 | +0.025 | 100 | +0.027 |
| pope | Qwen3-VL-8B | 200 | +0.050 | 100 | +0.030 |

The gain on unseen pictures matches or exceeds the gain on seen ones, so the `pope` numbers are
not carried by memorised pictures. A split by picture would remove the question entirely and is
the right protocol for the next run. `pets20` shares no picture between its splits.

Reproduce with `bash bench/run_mm_l2.sh` (three models, three seeds), then
`python -m bench.mm_l2_audit bench/results_mm_l2/<date>`, which prints every table in this
section and the per-seed table under "Full tables".

## Reproducibility checks behind these numbers

- An independent second run of the 2B `pope`, content-free `pope` and `ai2d`
  on another GPU reproduced this run exactly: every metric of every row, L1
  temperatures included.
- The text path is bit-identical to the text bench's code: 72 decision
  configurations (three question kinds, three priors, three shared-prefix
  modes, adaptive on and off, raw / L0 / L1 with calibration) give the same
  probabilities, artifacts and backend call counts with and without the
  multimodal changes.
- Batch parity: the answer distribution at batch size 1 and 8, on prompts
  that differ in image size and length, agrees to 5.6e-10 (2B) and 1.3e-12
  (8B); `tests/test_vlm_engine.py` asserts it on a real model.
- The merge that brought L2 in left the image path bit-identical: raw and L0 on 24 `pets20`
  test items give the same probabilities before and after it (max |dp| = 0.0, Qwen3-VL-2B).
- L2 reads the same forward as the readout: the label log-probs returned by
  `VLMBackend.hidden_states` equal `next_token_logprobs` to 0.0 on the same prompts;
  `tests/test_vlm_engine.py` asserts it, and that a head fit on colour swatches reads them.
- A rerun of seed 0 reproduced every L2 number of the first run exactly.

## Full tables

`pets20` and `pope`, library default (`bench/results_mm/2026-09-22`):

| model | task | K | n | level | acc | brier | ece | flip | cov@5% |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | pets20 (L0 prior: batch) | 20 | 300 | raw | 0.797 | 0.344 | 0.152 | 0.163 | 0.550 |
|  |  |  |  | L0-cf | 0.787 | 0.369 | 0.168 | 0.183 | 0.443 |
|  |  |  |  | L0-bc | 0.820 | 0.320 | 0.128 | 0.157 | 0.580 |
|  |  |  |  | L0-perm | 0.790 | 0.344 | 0.157 | 0.087 | 0.577 |
|  |  |  |  | L0-perm+cf | 0.793 | 0.361 | 0.157 | 0.083 | 0.587 |
|  |  |  |  | L0-perm+bc | 0.827 | 0.299 | 0.119 | 0.080 | 0.640 |
|  |  |  |  | L0 | 0.827 | 0.299 | 0.119 | 0.080 | 0.640 |
|  |  |  |  | L1 | 0.827 | 0.250 | 0.029 | 0.077 | 0.680 |
| Qwen3-VL-2B-Instruct | pope (L0 prior: batch) | 2 | 300 | raw | 0.917 | 0.134 | 0.046 | 0.053 | 0.913 |
|  |  |  |  | L0-cf | 0.917 | 0.137 | 0.049 | 0.013 | 0.890 |
|  |  |  |  | L0-bc | 0.913 | 0.134 | 0.049 | 0.037 | 0.913 |
|  |  |  |  | L0-perm | 0.910 | 0.127 | 0.051 | 0.000 | 0.920 |
|  |  |  |  | L0-perm+cf | 0.923 | 0.134 | 0.049 | 0.000 | 0.897 |
|  |  |  |  | L0-perm+bc | 0.913 | 0.127 | 0.047 | 0.000 | 0.920 |
|  |  |  |  | L0 | 0.913 | 0.127 | 0.047 | 0.000 | 0.920 |
|  |  |  |  | L1 | 0.910 | 0.121 | 0.046 | 0.000 | 0.923 |
| Qwen3-VL-4B-Instruct | pets20 (L0 prior: batch) | 20 | 300 | raw | 0.820 | 0.345 | 0.164 | 0.127 | 0.287 |
|  |  |  |  | L0-cf | 0.777 | 0.414 | 0.205 | 0.163 | 0.420 |
|  |  |  |  | L0-bc | 0.827 | 0.335 | 0.161 | 0.133 | 0.627 |
|  |  |  |  | L0-perm | 0.830 | 0.310 | 0.151 | 0.077 | 0.623 |
|  |  |  |  | L0-perm+cf | 0.833 | 0.308 | 0.149 | 0.070 | 0.610 |
|  |  |  |  | L0-perm+bc | 0.843 | 0.299 | 0.139 | 0.067 | 0.647 |
|  |  |  |  | L0 | 0.843 | 0.299 | 0.139 | 0.067 | 0.647 |
|  |  |  |  | L1 | 0.840 | 0.249 | 0.098 | 0.073 | 0.687 |
| Qwen3-VL-4B-Instruct | pope (L0 prior: batch) | 2 | 300 | raw | 0.877 | 0.234 | 0.110 | 0.010 | 0.660 |
|  |  |  |  | L0-cf | 0.890 | 0.176 | 0.061 | 0.027 | 0.477 |
|  |  |  |  | L0-bc | 0.877 | 0.232 | 0.110 | 0.010 | 0.683 |
|  |  |  |  | L0-perm | 0.877 | 0.225 | 0.109 | 0.000 | 0.720 |
|  |  |  |  | L0-perm+cf | 0.910 | 0.165 | 0.060 | 0.000 | 0.480 |
|  |  |  |  | L0-perm+bc | 0.877 | 0.223 | 0.109 | 0.000 | 0.723 |
|  |  |  |  | L0 | 0.877 | 0.223 | 0.109 | 0.000 | 0.723 |
|  |  |  |  | L1 | 0.877 | 0.182 | 0.050 | 0.000 | 0.723 |
| Qwen3-VL-8B-Instruct | pets20 (L0 prior: batch) | 20 | 300 | raw | 0.880 | 0.218 | 0.104 | 0.147 | 0.743 |
|  |  |  |  | L0-cf | 0.883 | 0.211 | 0.098 | 0.110 | 0.577 |
|  |  |  |  | L0-bc | 0.887 | 0.214 | 0.097 | 0.133 | 0.740 |
|  |  |  |  | L0-perm | 0.900 | 0.177 | 0.084 | 0.060 | 0.807 |
|  |  |  |  | L0-perm+cf | 0.907 | 0.177 | 0.078 | 0.083 | 0.783 |
|  |  |  |  | L0-perm+bc | 0.903 | 0.173 | 0.080 | 0.050 | 0.810 |
|  |  |  |  | L0 | 0.903 | 0.173 | 0.080 | 0.050 | 0.810 |
|  |  |  |  | L1 | 0.903 | 0.154 | 0.054 | 0.060 | 0.810 |
| Qwen3-VL-8B-Instruct | pope (L0 prior: batch) | 2 | 300 | raw | 0.840 | 0.305 | 0.147 | 0.010 | 0.477 |
|  |  |  |  | L0-cf | 0.910 | 0.150 | 0.057 | 0.110 | 0.870 |
|  |  |  |  | L0-bc | 0.840 | 0.302 | 0.148 | 0.010 | 0.497 |
|  |  |  |  | L0-perm | 0.843 | 0.299 | 0.144 | 0.000 | 0.543 |
|  |  |  |  | L0-perm+cf | 0.900 | 0.172 | 0.065 | 0.000 | 0.673 |
|  |  |  |  | L0-perm+bc | 0.847 | 0.296 | 0.142 | 0.000 | 0.620 |
|  |  |  |  | L0 | 0.847 | 0.296 | 0.142 | 0.000 | 0.620 |
|  |  |  |  | L1 | 0.843 | 0.234 | 0.081 | 0.000 | 0.610 |

Environment: {"gpu": "NVIDIA H100 NVL", "torch": "2.10.0+cu128", "transformers": "4.57.6"}. Dates: 2026-09-22. Test items sampled with seed 0; L1 temperature fit on a disjoint calibration split.

`pope` with the content-free prior as L0 (`bench/results_mm_cf/2026-09-22`):

| model | task | K | n | level | acc | brier | ece | flip | cov@5% |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | pope (L0 prior: content_free) | 2 | 300 | raw | 0.917 | 0.134 | 0.046 | 0.053 | 0.913 |
|  |  |  |  | L0-cf | 0.917 | 0.137 | 0.049 | 0.013 | 0.890 |
|  |  |  |  | L0-bc | 0.913 | 0.134 | 0.049 | 0.037 | 0.913 |
|  |  |  |  | L0-perm | 0.910 | 0.127 | 0.051 | 0.000 | 0.920 |
|  |  |  |  | L0-perm+cf | 0.923 | 0.134 | 0.049 | 0.000 | 0.897 |
|  |  |  |  | L0-perm+bc | 0.913 | 0.127 | 0.047 | 0.000 | 0.920 |
|  |  |  |  | L0 | 0.923 | 0.134 | 0.049 | 0.000 | 0.897 |
|  |  |  |  | L1 | 0.923 | 0.131 | 0.048 | 0.000 | 0.897 |
| Qwen3-VL-4B-Instruct | pope (L0 prior: content_free) | 2 | 300 | raw | 0.877 | 0.234 | 0.110 | 0.010 | 0.660 |
|  |  |  |  | L0-cf | 0.890 | 0.176 | 0.061 | 0.027 | 0.477 |
|  |  |  |  | L0-bc | 0.877 | 0.232 | 0.110 | 0.010 | 0.683 |
|  |  |  |  | L0-perm | 0.877 | 0.225 | 0.109 | 0.000 | 0.720 |
|  |  |  |  | L0-perm+cf | 0.910 | 0.165 | 0.060 | 0.000 | 0.480 |
|  |  |  |  | L0-perm+bc | 0.877 | 0.223 | 0.109 | 0.000 | 0.723 |
|  |  |  |  | L0 | 0.910 | 0.165 | 0.060 | 0.000 | 0.480 |
|  |  |  |  | L1 | 0.910 | 0.199 | 0.126 | 0.000 | 0.480 |
| Qwen3-VL-8B-Instruct | pope (L0 prior: content_free) | 2 | 300 | raw | 0.840 | 0.305 | 0.147 | 0.010 | 0.477 |
|  |  |  |  | L0-cf | 0.910 | 0.150 | 0.057 | 0.110 | 0.870 |
|  |  |  |  | L0-bc | 0.840 | 0.302 | 0.148 | 0.010 | 0.497 |
|  |  |  |  | L0-perm | 0.843 | 0.299 | 0.144 | 0.000 | 0.543 |
|  |  |  |  | L0-perm+cf | 0.900 | 0.172 | 0.065 | 0.000 | 0.673 |
|  |  |  |  | L0-perm+bc | 0.847 | 0.296 | 0.142 | 0.000 | 0.620 |
|  |  |  |  | L0 | 0.900 | 0.172 | 0.065 | 0.000 | 0.673 |
|  |  |  |  | L1 | 0.900 | 0.226 | 0.147 | 0.000 | 0.673 |

Environment: {"gpu": "NVIDIA H100 NVL", "torch": "2.10.0+cu128", "transformers": "4.57.6"}. Dates: 2026-09-22. Test items sampled with seed 0; L1 temperature fit on a disjoint calibration split.

`ai2d` (`bench/results_mcq/2026-09-22`):

| model | task | K | n | level | acc | brier | ece | flip | cov@5% |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | ai2d (L0 prior: none) | 4 | 300 | raw | 0.833 | 0.269 | 0.099 | 0.157 | 0.697 |
|  |  |  |  | L0-cf | 0.570 | 0.622 | 0.199 | 0.347 | 0.163 |
|  |  |  |  | L0-perm | 0.833 | 0.247 | 0.080 | 0.047 | 0.770 |
|  |  |  |  | L0-perm+cf | 0.593 | 0.560 | 0.153 | 0.157 | 0.227 |
|  |  |  |  | L0 | 0.833 | 0.247 | 0.080 | 0.047 | 0.770 |
|  |  |  |  | L1 | 0.833 | 0.223 | 0.053 | 0.047 | 0.780 |
| Qwen3-VL-4B-Instruct | ai2d (L0 prior: none) | 4 | 300 | raw | 0.863 | 0.233 | 0.106 | 0.100 | 0.780 |
|  |  |  |  | L0-cf | 0.713 | 0.480 | 0.206 | 0.257 | 0.137 |
|  |  |  |  | L0-perm | 0.873 | 0.224 | 0.096 | 0.047 | 0.817 |
|  |  |  |  | L0-perm+cf | 0.757 | 0.393 | 0.151 | 0.100 | 0.427 |
|  |  |  |  | L0 | 0.873 | 0.224 | 0.096 | 0.047 | 0.817 |
|  |  |  |  | L1 | 0.873 | 0.182 | 0.046 | 0.047 | 0.823 |
| Qwen3-VL-8B-Instruct | ai2d (L0 prior: none) | 4 | 300 | raw | 0.860 | 0.251 | 0.112 | 0.103 | 0.820 |
|  |  |  |  | L0-cf | 0.757 | 0.420 | 0.160 | 0.240 | 0.093 |
|  |  |  |  | L0-perm | 0.867 | 0.230 | 0.105 | 0.037 | 0.813 |
|  |  |  |  | L0-perm+cf | 0.783 | 0.377 | 0.134 | 0.087 | 0.147 |
|  |  |  |  | L0 | 0.867 | 0.230 | 0.105 | 0.037 | 0.813 |
|  |  |  |  | L1 | 0.867 | 0.184 | 0.041 | 0.037 | 0.803 |

Environment: {"gpu": "NVIDIA H100 NVL", "torch": "2.10.0+cu128", "transformers": "4.57.6"}. Dates: 2026-09-22. Test items sampled with seed 0; L1 temperature fit on a disjoint calibration split.

`L2` per split seed (`bench/results_mm_l2/2026-09-23`), from `python -m bench.mm_l2_audit`:

| task | model | seed | L0 acc | L1 acc | L2 acc | L2 - L1 [95% CI] | L1 ECE | L2 ECE | L2 block, head |
|---|---|---|---|---|---|---|---|---|---|
| pets20 | Qwen3-VL-2B | 0 | 0.823 | 0.823 | 0.957 | +0.133 [+0.093, +0.177] | 0.035 | 0.019 | 20/28, ridge |
| pets20 | Qwen3-VL-2B | 1 | 0.783 | 0.783 | 0.937 | +0.153 [+0.110, +0.197] | 0.068 | 0.029 | 20/28, ridge |
| pets20 | Qwen3-VL-2B | 2 | 0.777 | 0.777 | 0.933 | +0.157 [+0.110, +0.203] | 0.055 | 0.022 | 17/28, ridge |
| pets20 | Qwen3-VL-4B | 0 | 0.847 | 0.847 | 0.947 | +0.100 [+0.063, +0.140] | 0.105 | 0.032 | 22/36, ridge |
| pets20 | Qwen3-VL-4B | 1 | 0.833 | 0.833 | 0.960 | +0.127 [+0.090, +0.170] | 0.090 | 0.030 | 22/36, ridge |
| pets20 | Qwen3-VL-4B | 2 | 0.827 | 0.827 | 0.957 | +0.130 [+0.090, +0.173] | 0.087 | 0.033 | 22/36, ridge |
| pets20 | Qwen3-VL-8B | 0 | 0.907 | 0.907 | 0.933 | +0.027 [-0.007, +0.060] | 0.051 | 0.037 | 31/36, ridge |
| pets20 | Qwen3-VL-8B | 1 | 0.900 | 0.900 | 0.933 | +0.033 [+0.000, +0.070] | 0.050 | 0.036 | 22/36, ridge |
| pets20 | Qwen3-VL-8B | 2 | 0.880 | 0.880 | 0.910 | +0.030 [-0.007, +0.067] | 0.062 | 0.041 | 36/36, ridge |
| pope | Qwen3-VL-2B | 0 | 0.907 | 0.907 | 0.923 | +0.017 [-0.013, +0.047] | 0.055 | 0.035 | 20/28, lda |
| pope | Qwen3-VL-2B | 1 | 0.907 | 0.907 | 0.897 | -0.010 [-0.030, +0.010] | 0.036 | 0.060 | 17/28, ridge |
| pope | Qwen3-VL-2B | 2 | 0.900 | 0.900 | 0.883 | -0.017 [-0.047, +0.013] | 0.049 | 0.063 | 28/28, ridge |
| pope | Qwen3-VL-4B | 0 | 0.873 | 0.873 | 0.900 | +0.027 [+0.003, +0.050] | 0.047 | 0.047 | 22/36, lda |
| pope | Qwen3-VL-4B | 1 | 0.887 | 0.887 | 0.900 | +0.013 [-0.003, +0.033] | 0.053 | 0.088 | 25/36, lda |
| pope | Qwen3-VL-4B | 2 | 0.840 | 0.840 | 0.877 | +0.037 [+0.000, +0.073] | 0.085 | 0.049 | 18/36, ridge |
| pope | Qwen3-VL-8B | 0 | 0.847 | 0.847 | 0.897 | +0.050 [+0.027, +0.077] | 0.084 | 0.087 | 22/36, lda |
| pope | Qwen3-VL-8B | 1 | 0.877 | 0.877 | 0.903 | +0.027 [+0.003, +0.053] | 0.064 | 0.055 | 36/36, ridge |
| pope | Qwen3-VL-8B | 2 | 0.830 | 0.830 | 0.883 | +0.053 [+0.010, +0.100] | 0.083 | 0.049 | 36/36, ridge |
