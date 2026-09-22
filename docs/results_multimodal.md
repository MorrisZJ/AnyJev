# Multimodal results

Three image tasks, three Qwen3-VL sizes, 300 test items and 200 calibration
items each, seed 0, one H100 NVL per model, bf16. Every number below is
generated from the JSON under `bench/results_mm/`, `bench/results_mm_cf/` and
`bench/results_mcq/`, all produced by one command, `bash bench/run_mm.sh`
(transformers ≥ 4.57 for Qwen3-VL); nothing is typed in by hand.

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

| task | model | L0 prior | acc | ECE | flip | cov@5% |
|---|---|---|---|---|---|---|
| pets20 | Qwen3-VL-2B | batch | 0.797 → 0.830 → 0.833 | 0.152 → 0.117 → 0.037 | 0.163 → 0.070 → 0.073 | 0.540 → 0.670 → 0.673 |
| pets20 | Qwen3-VL-4B | batch | 0.813 → 0.843 → 0.847 | 0.168 → 0.140 → 0.110 | 0.130 → 0.060 → 0.070 | 0.233 → 0.637 → 0.680 |
| pets20 | Qwen3-VL-8B | batch | 0.883 → 0.907 → 0.903 | 0.099 → 0.076 → 0.053 | 0.140 → 0.047 → 0.053 | 0.743 → 0.813 → 0.797 |
| pope | Qwen3-VL-2B | batch | 0.917 → 0.910 → 0.910 | 0.047 → 0.050 → 0.047 | 0.053 → 0.000 → 0.000 | 0.913 → 0.920 → 0.923 |
| pope | Qwen3-VL-4B | batch | 0.877 → 0.877 → 0.877 | 0.110 → 0.109 → 0.044 | 0.010 → 0.000 → 0.000 | 0.643 → 0.720 → 0.723 |
| pope | Qwen3-VL-8B | batch | 0.843 → 0.847 → 0.847 | 0.143 → 0.143 → 0.084 | 0.007 → 0.000 → 0.000 | 0.447 → 0.633 → 0.627 |
| ai2d | Qwen3-VL-2B | none | 0.833 → 0.837 → 0.837 | 0.092 → 0.076 → 0.057 | 0.153 → 0.047 → 0.047 | 0.693 → 0.770 → 0.783 |
| ai2d | Qwen3-VL-4B | none | 0.860 → 0.873 → 0.873 | 0.108 → 0.094 → 0.053 | 0.100 → 0.053 → 0.053 | 0.787 → 0.817 → 0.823 |
| ai2d | Qwen3-VL-8B | none | 0.860 → 0.870 → 0.870 | 0.112 → 0.100 → 0.037 | 0.097 → 0.037 → 0.037 | 0.823 → 0.813 → 0.803 |

## What carried over from text

**Position debiasing, unchanged.** Reversing the option list flips 10 to 16
percent of answers on the K=20 `pets20` and the per-item `ai2d`; L0 cuts that
by roughly half to two thirds on every model (`pets20` 2B 0.163 → 0.070,
`ai2d` 8B 0.097 → 0.037). On the two-option `pope` it goes to exactly zero,
as on the text `injection`. Pixels in the state do not change the fact that
the model prefers some positions in the option list.

**The coverage story, on the K=20 task.** At 5 percent risk, `pets20` on 4B
goes from 23 percent of items auto-decidable to 68 percent, from accuracy
that moves 3 points — the shape of `banking20` on Qwen3-8B (7.7 → 54.3
percent).

**L1 where L0 alone does not calibrate.** On `pope` the default L0 barely
moves anything: the true Yes rate is exactly 0.5 and the model's mean
prediction sits close to it, so the batch prior has little to divide out.
L1 does the work: ECE 0.110 → 0.044 on 4B and 0.143 → 0.084 on 8B, and 8B's
coverage at 5 percent risk 0.447 → 0.627.

## The content-free prior: the largest win and the largest loss

Opt-in on text, opt-in here. It is in every full table below as the
`L0-perm+cf` row, and POPE with it as L0 has its own run:

| task | model | L0 prior | acc | ECE | flip | cov@5% |
|---|---|---|---|---|---|---|
| pope | Qwen3-VL-2B | content_free | 0.917 → 0.923 → 0.920 | 0.047 → 0.047 → 0.045 | 0.053 → 0.000 → 0.000 | 0.913 → 0.890 → 0.890 |
| pope | Qwen3-VL-4B | content_free | 0.877 → 0.907 → 0.910 | 0.110 → 0.059 → 0.123 | 0.010 → 0.000 → 0.000 | 0.643 → 0.673 → 0.677 |
| pope | Qwen3-VL-8B | content_free | 0.843 → 0.903 → 0.900 | 0.143 → 0.060 → 0.150 | 0.007 → 0.000 → 0.000 | 0.447 → 0.487 → 0.487 |

- **`pope`:** the largest win in the multimodal bench. 8B accuracy
  0.843 → 0.903 and ECE 0.143 → 0.060 with zero labels. Raw accuracy *falls*
  with model size (0.917, 0.877, 0.843) and the corrected rows converge near
  0.90: the larger models are not worse at seeing objects, they say Yes more,
  and the lean only shows when the picture is blanked.
- **`pets20`:** changes almost nothing (compare `L0-perm+cf` with `L0-perm`).
- **`ai2d`:** costs 8 to 24 accuracy points (2B 0.837 → 0.600). The question
  and options often give the answer away without the diagram, so the
  blank-image probe measures knowledge, not bias.

So on POPE there are two routes to a calibrated answer: the content-free L0
(zero labels, +6 accuracy points, ECE 0.060 on 8B) or the default L0 plus L1
(200 labels, accuracy essentially unchanged, ECE 0.084). The rule in
[docs/multimodal.md](multimodal.md) for when the first route is safe is stated
about task shape, not per dataset, and `pets20` tested it: the rule said
content-free would be *safe* there, and it was — but safe is not useful when
there is no label prior to remove.

## Where it did not help

We would rather you find these here than in production.

- **L1 on top of the content-free L0 made POPE worse** on 4B and 8B: ECE
  0.059 → 0.123 and 0.060 → 0.150, while test NLL fell (8B 0.653 → 0.376).
  The remaining errors are confident hallucinations, which dominate the NLL,
  so the NLL-optimal temperature (4.7 and 5.8) flattens every answer. One
  temperature cannot fix confident wrong answers; don't stack the two routes
  above.
- **Coverage at 5 percent risk dipped** on `ai2d` 8B (0.823 → 0.803), and on
  `pope` 2B under the content-free L0 (0.913 → 0.890). Both were already well
  ordered raw; averaging rotations traded a little of that for stability.
- **Accuracy gains are small under the default**, −0.7 to +3.6 points; `pope`
  2B loses 0.7. The gains are in stability and calibration, as on text.
- **One model family.** Everything here is Qwen3-VL.

## Reproducibility checks behind these numbers

- Every number comes from one run of the committed `bench/run_mm.sh` on the
  committed code, and that run reproduced every earlier run of the same
  configuration exactly — every metric of every row, L1 temperatures included.
- A late change to how images are ordered for the backend was checked
  against all 412,960 prompts the bench builds (every item, permutation,
  reversed-option probe and content-free probe): identical user turns and
  identical image order, so it cannot move a number.
- Batch parity: the answer distribution at batch size 1 and 8, on prompts
  that differ in image size and length, agrees to 5.6e-10 (2B) and 1.3e-12
  (8B); `tests/test_vlm_engine.py` asserts it on a real model.

## Full tables

`pets20` and `pope`, library default (`bench/results_mm/2026-09-22`):

| model | task | K | n | level | acc | brier | ece | flip | cov@5% |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | pets20 (L0 prior: batch) | 20 | 300 | raw | 0.797 | 0.344 | 0.152 | 0.163 | 0.540 |
|  |  |  |  | L0-cf | 0.787 | 0.370 | 0.169 | 0.180 | 0.447 |
|  |  |  |  | L0-bc | 0.823 | 0.311 | 0.126 | 0.147 | 0.580 |
|  |  |  |  | L0-perm | 0.790 | 0.344 | 0.158 | 0.087 | 0.577 |
|  |  |  |  | L0-perm+cf | 0.793 | 0.361 | 0.157 | 0.083 | 0.587 |
|  |  |  |  | L0-perm+bc | 0.830 | 0.287 | 0.117 | 0.070 | 0.670 |
|  |  |  |  | L0 | 0.830 | 0.287 | 0.117 | 0.070 | 0.670 |
|  |  |  |  | L1 | 0.833 | 0.250 | 0.037 | 0.073 | 0.673 |
| Qwen3-VL-2B-Instruct | pope (L0 prior: batch) | 2 | 300 | raw | 0.917 | 0.135 | 0.047 | 0.053 | 0.913 |
|  |  |  |  | L0-cf | 0.920 | 0.138 | 0.049 | 0.013 | 0.890 |
|  |  |  |  | L0-bc | 0.910 | 0.135 | 0.053 | 0.037 | 0.913 |
|  |  |  |  | L0-perm | 0.910 | 0.128 | 0.051 | 0.000 | 0.920 |
|  |  |  |  | L0-perm+cf | 0.923 | 0.135 | 0.047 | 0.000 | 0.890 |
|  |  |  |  | L0-perm+bc | 0.910 | 0.128 | 0.050 | 0.000 | 0.920 |
|  |  |  |  | L0 | 0.910 | 0.128 | 0.050 | 0.000 | 0.920 |
|  |  |  |  | L1 | 0.910 | 0.121 | 0.047 | 0.000 | 0.923 |
| Qwen3-VL-4B-Instruct | pets20 (L0 prior: batch) | 20 | 300 | raw | 0.813 | 0.346 | 0.168 | 0.130 | 0.233 |
|  |  |  |  | L0-cf | 0.780 | 0.416 | 0.202 | 0.163 | 0.433 |
|  |  |  |  | L0-bc | 0.823 | 0.329 | 0.162 | 0.127 | 0.613 |
|  |  |  |  | L0-perm | 0.833 | 0.310 | 0.148 | 0.073 | 0.623 |
|  |  |  |  | L0-perm+cf | 0.833 | 0.308 | 0.149 | 0.070 | 0.613 |
|  |  |  |  | L0-perm+bc | 0.843 | 0.296 | 0.140 | 0.060 | 0.637 |
|  |  |  |  | L0 | 0.843 | 0.296 | 0.140 | 0.060 | 0.637 |
|  |  |  |  | L1 | 0.847 | 0.246 | 0.110 | 0.070 | 0.680 |
| Qwen3-VL-4B-Instruct | pope (L0 prior: batch) | 2 | 300 | raw | 0.877 | 0.234 | 0.110 | 0.010 | 0.643 |
|  |  |  |  | L0-cf | 0.900 | 0.175 | 0.066 | 0.027 | 0.483 |
|  |  |  |  | L0-bc | 0.873 | 0.231 | 0.113 | 0.007 | 0.680 |
|  |  |  |  | L0-perm | 0.877 | 0.225 | 0.109 | 0.000 | 0.713 |
|  |  |  |  | L0-perm+cf | 0.907 | 0.165 | 0.059 | 0.000 | 0.673 |
|  |  |  |  | L0-perm+bc | 0.877 | 0.222 | 0.109 | 0.000 | 0.720 |
|  |  |  |  | L0 | 0.877 | 0.222 | 0.109 | 0.000 | 0.720 |
|  |  |  |  | L1 | 0.877 | 0.180 | 0.044 | 0.000 | 0.723 |
| Qwen3-VL-8B-Instruct | pets20 (L0 prior: batch) | 20 | 300 | raw | 0.883 | 0.216 | 0.099 | 0.140 | 0.743 |
|  |  |  |  | L0-cf | 0.883 | 0.216 | 0.099 | 0.110 | 0.580 |
|  |  |  |  | L0-bc | 0.887 | 0.212 | 0.095 | 0.133 | 0.740 |
|  |  |  |  | L0-perm | 0.900 | 0.177 | 0.084 | 0.060 | 0.807 |
|  |  |  |  | L0-perm+cf | 0.907 | 0.177 | 0.078 | 0.083 | 0.783 |
|  |  |  |  | L0-perm+bc | 0.907 | 0.172 | 0.076 | 0.047 | 0.813 |
|  |  |  |  | L0 | 0.907 | 0.172 | 0.076 | 0.047 | 0.813 |
|  |  |  |  | L1 | 0.903 | 0.155 | 0.053 | 0.053 | 0.797 |
| Qwen3-VL-8B-Instruct | pope (L0 prior: batch) | 2 | 300 | raw | 0.843 | 0.304 | 0.143 | 0.007 | 0.447 |
|  |  |  |  | L0-cf | 0.910 | 0.151 | 0.056 | 0.120 | 0.837 |
|  |  |  |  | L0-bc | 0.843 | 0.301 | 0.146 | 0.007 | 0.510 |
|  |  |  |  | L0-perm | 0.843 | 0.299 | 0.145 | 0.000 | 0.537 |
|  |  |  |  | L0-perm+cf | 0.903 | 0.177 | 0.060 | 0.000 | 0.487 |
|  |  |  |  | L0-perm+bc | 0.847 | 0.296 | 0.143 | 0.000 | 0.633 |
|  |  |  |  | L0 | 0.847 | 0.296 | 0.143 | 0.000 | 0.633 |
|  |  |  |  | L1 | 0.847 | 0.232 | 0.084 | 0.000 | 0.627 |

Environment: {"gpu": "NVIDIA H100 NVL", "torch": "2.10.0+cu128", "transformers": "4.57.6"}. Dates: 2026-09-22. Test items sampled with seed 0; L1 temperature fit on a disjoint calibration split.

`pope` with the content-free prior as L0 (`bench/results_mm_cf/2026-09-22`):

| model | task | K | n | level | acc | brier | ece | flip | cov@5% |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | pope (L0 prior: content_free) | 2 | 300 | raw | 0.917 | 0.135 | 0.047 | 0.053 | 0.913 |
|  |  |  |  | L0-cf | 0.920 | 0.138 | 0.049 | 0.013 | 0.890 |
|  |  |  |  | L0-bc | 0.910 | 0.135 | 0.053 | 0.037 | 0.913 |
|  |  |  |  | L0-perm | 0.910 | 0.128 | 0.051 | 0.000 | 0.920 |
|  |  |  |  | L0-perm+cf | 0.923 | 0.135 | 0.047 | 0.000 | 0.890 |
|  |  |  |  | L0-perm+bc | 0.910 | 0.128 | 0.050 | 0.000 | 0.920 |
|  |  |  |  | L0 | 0.923 | 0.135 | 0.047 | 0.000 | 0.890 |
|  |  |  |  | L1 | 0.920 | 0.132 | 0.045 | 0.000 | 0.890 |
| Qwen3-VL-4B-Instruct | pope (L0 prior: content_free) | 2 | 300 | raw | 0.877 | 0.234 | 0.110 | 0.010 | 0.643 |
|  |  |  |  | L0-cf | 0.900 | 0.175 | 0.066 | 0.027 | 0.483 |
|  |  |  |  | L0-bc | 0.873 | 0.231 | 0.113 | 0.007 | 0.680 |
|  |  |  |  | L0-perm | 0.877 | 0.225 | 0.109 | 0.000 | 0.713 |
|  |  |  |  | L0-perm+cf | 0.907 | 0.165 | 0.059 | 0.000 | 0.673 |
|  |  |  |  | L0-perm+bc | 0.877 | 0.222 | 0.109 | 0.000 | 0.720 |
|  |  |  |  | L0 | 0.907 | 0.165 | 0.059 | 0.000 | 0.673 |
|  |  |  |  | L1 | 0.910 | 0.196 | 0.123 | 0.000 | 0.677 |
| Qwen3-VL-8B-Instruct | pope (L0 prior: content_free) | 2 | 300 | raw | 0.843 | 0.304 | 0.143 | 0.007 | 0.447 |
|  |  |  |  | L0-cf | 0.910 | 0.151 | 0.056 | 0.120 | 0.837 |
|  |  |  |  | L0-bc | 0.843 | 0.301 | 0.146 | 0.007 | 0.510 |
|  |  |  |  | L0-perm | 0.843 | 0.299 | 0.145 | 0.000 | 0.537 |
|  |  |  |  | L0-perm+cf | 0.903 | 0.177 | 0.060 | 0.000 | 0.487 |
|  |  |  |  | L0-perm+bc | 0.847 | 0.296 | 0.143 | 0.000 | 0.633 |
|  |  |  |  | L0 | 0.903 | 0.177 | 0.060 | 0.000 | 0.487 |
|  |  |  |  | L1 | 0.900 | 0.231 | 0.150 | 0.000 | 0.487 |

Environment: {"gpu": "NVIDIA H100 NVL", "torch": "2.10.0+cu128", "transformers": "4.57.6"}. Dates: 2026-09-22. Test items sampled with seed 0; L1 temperature fit on a disjoint calibration split.

`ai2d` (`bench/results_mcq/2026-09-22`):

| model | task | K | n | level | acc | brier | ece | flip | cov@5% |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | ai2d (L0 prior: none) | 4 | 300 | raw | 0.833 | 0.267 | 0.092 | 0.153 | 0.693 |
|  |  |  |  | L0-cf | 0.577 | 0.623 | 0.206 | 0.353 | 0.160 |
|  |  |  |  | L0-perm | 0.837 | 0.246 | 0.076 | 0.047 | 0.770 |
|  |  |  |  | L0-perm+cf | 0.600 | 0.561 | 0.146 | 0.147 | 0.230 |
|  |  |  |  | L0 | 0.837 | 0.246 | 0.076 | 0.047 | 0.770 |
|  |  |  |  | L1 | 0.837 | 0.223 | 0.057 | 0.047 | 0.783 |
| Qwen3-VL-4B-Instruct | ai2d (L0 prior: none) | 4 | 300 | raw | 0.860 | 0.231 | 0.108 | 0.100 | 0.787 |
|  |  |  |  | L0-cf | 0.707 | 0.480 | 0.205 | 0.257 | 0.130 |
|  |  |  |  | L0-perm | 0.873 | 0.224 | 0.094 | 0.053 | 0.817 |
|  |  |  |  | L0-perm+cf | 0.753 | 0.393 | 0.154 | 0.097 | 0.427 |
|  |  |  |  | L0 | 0.873 | 0.224 | 0.094 | 0.053 | 0.817 |
|  |  |  |  | L1 | 0.873 | 0.182 | 0.053 | 0.053 | 0.823 |
| Qwen3-VL-8B-Instruct | ai2d (L0 prior: none) | 4 | 300 | raw | 0.860 | 0.249 | 0.112 | 0.097 | 0.823 |
|  |  |  |  | L0-cf | 0.747 | 0.423 | 0.166 | 0.233 | 0.093 |
|  |  |  |  | L0-perm | 0.870 | 0.227 | 0.100 | 0.037 | 0.813 |
|  |  |  |  | L0-perm+cf | 0.787 | 0.378 | 0.135 | 0.083 | 0.150 |
|  |  |  |  | L0 | 0.870 | 0.227 | 0.100 | 0.037 | 0.813 |
|  |  |  |  | L1 | 0.870 | 0.183 | 0.037 | 0.037 | 0.803 |

Environment: {"gpu": "NVIDIA H100 NVL", "torch": "2.10.0+cu128", "transformers": "4.57.6"}. Dates: 2026-09-22. Test items sampled with seed 0; L1 temperature fit on a disjoint calibration split.
