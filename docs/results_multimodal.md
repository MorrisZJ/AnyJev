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
