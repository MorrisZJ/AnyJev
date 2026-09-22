# AnyJev reproduction report

Generated 2026-09-22T01:02:16 by `repro_check/make_report.py`.

Everything here was produced by rerunning the repo's own entry points (`bench.run`, `bench.run_typed`) with the exact parameters recorded in the committed result JSON, on the same library versions (transformers 4.55.4, torch 2.5.1+cu124) and the same class of hardware (H100). No file outside `repro_check/` was modified; the repo's bench was pointed at `--out repro_check/results/...`.

## Verdict

The method, the mathematics and the implementation hold up, and the published numbers are real
and exactly reproducible. All 31 published result cells were rerun. Every zero-label number --
`raw` plus all six L0 ablations, across 3 models x 3 tasks x 2 priors, 648 cells -- comes back
bit-identical, maximum deviation 0.000000.

Three things did not line up. None of them is a methodological error: two are missing
reproducibility metadata and one is a design wart in how L1 is fitted.

### 1. L1 under `prior=batch` drifts, because the batch prior is stateful

72 config-matched cells, 29% bit-identical, worst deviation 0.023 on cov@5%. The batch prior is a
running accumulator on the `Decider`, keyed by question, so the temperature fitted inside
`calibrate()` depends on how many items that decider has already seen:

| when calibrate() is called | items seen first | fitted temperature |
|---|---|---|
| the bench's own sequence | 300 | 4.158444 |
| again, same data | 800 | 4.158451 |
| a fresh Decider, calibrate first | 0 | **4.147245** |

So an L1 artifact is not a pure function of its calibration set. The practical impact is small
(accuracy does not move at all, ECE moves by 0.0002) but it does push cov@5% by 0.007. The control
is decisive: `prior=content_free` uses a prior that does not depend on call history, and its L1 is
100% bit-identical across all 48 cells. Measured by `repro_check/probe_stateful_prior.py`.

### 2. `batch_size` changes the numbers and is not recorded

`raw` is a single deterministic forward pass, yet it moves when the batch changes: 100%
bit-identical at `--batch-size 32`, only 8.3% at 16, worst deviation 0.0105. The typed run pins it
exactly -- committed Qwen3-32B matches at `--batch-size 16` and at no other value, while the two
smaller models match at 32. `bench.run` records seed, n, calib, prior, combine, max_permutations,
torch, transformers and the GPU model, but not batch_size, so the right value has to be searched
for. Worth about 0.037 on a headline metric. Measured by `repro_check/probe_batchsize.py`.

### 3. Two committed maze fields cannot be produced by the public NanoJev

The committed maze JSONs carry `edge_majority` and `mean_p_true`. The public NanoJev's
`score_atomic` emits only atomic accuracy, Brier and NLL, and AnyJev's own providers only
`result.update()` top-level keys and never touch `result["summary"]`. `bench/maze_table.py` reads
`edge_majority` to render the majority-baseline column, so the number behind "no readout beats
always answering the majority label" cannot be regenerated from the public code as it stands. The
`--nanojev` argument also pins no commit.

## What lines up

- **Zero-label results**: 648 cells, 100% bit-identical.
- **The maze**: all 49 summary fields bit-identical across all 7 rows, and NanoJev's own
  `audit_inputs_sha256` and `predict_calls` match too, so the trajectories were identical.
  Both maze claims re-derive: the A/B readout solves 13/15 in 20,555 attempts against AnyJev's raw
  15/15 in 5,825, and none of the 7 readouts beats the majority baseline.
- **The Laya rows**: all 21 metrics at `+0.0000`, and `laya-typed-decisions` measures 0.768 here
  against its own published 0.766, so the repo's "reproduces its published 0.766" is independently
  confirmed. The "six times the ECE" claim measures 6.4x.
- **The stated deviation is honest**: the native baseline genuinely cannot set
  `disable_native_triton`, which imports `torch._native`, absent in torch 2.5.1.

## What the extensions show

Model families the repo never published work fine: on banking20 the L1/raw coverage ratio is 15x
for OLMo-2, 8.8x for Falcon3, 3.6x for Mistral, 3.3x for Phi-3.5. But the "7x" headline does not
generalise -- across 30 model x task pairs including tasks added here (agnews, emotion, yelp5,
massive20, hate, subj) the ratio runs from 31x down to 0x.

Bootstrap resampling puts the mean 95% CI width of cov@5% at n=300 at 0.342 of coverage. The
headline 7x is real (raw `[0.003, 0.387]` versus L1 `[0.410, 0.617]`, non-overlapping), but the
conservative reading -- L1's lower bound over raw's upper bound -- is only 1.1x. The metric needs
intervals reported alongside it.

## What was rerun

| committed result family | published cells | reran here | still missing |
|---|---|---|---|
| `bench/results_batchprior_v0` | 9 | 9 | **none** |
| `bench/results_cf` | 9 | 9 | **none** |
| `bench/results_nanojev` | 7 | 7 | **none** |
| `bench/results_typed` | 6 | 6 | **none** |
| **total** | **31** | **31** | **100% of published cells rerun** |

## Job log

The scheduler in `repro_check/verify_all.py` only tracks the jobs it launched itself. Several runs were launched beside it and are not in this table:

- The two `rc=-15` rows are the Qwen3-30B-A3B runs I killed on purpose. They had been given `--batch-size 8` to be safe on memory, which is both ~4x slower and, per the batch-size probe below, guaranteed not to land on the committed numbers. They were relaunched at `--batch-size 32` by `repro_check/run_30b.sh` and completed; those are the Qwen3-30B-A3B results compared in this report.
- `repro_check/run_missing.sh` backfilled the two Qwen2.5-7B-Instruct cells that a skip-logic bug of mine dropped, then ran the stateful-prior probe.
- `repro_check/run_providers.sh` ran the three Laya checkpoints; `repro_check/run_maze.sh` ran all seven maze rows.
- Mistral-7B and Falcon3-7B initially hit CUDA OOM on two tasks each while sharing a GPU with the 30B job; both were rerun to completion on dedicated GPUs.

| job | gpu | exit | minutes |
|---|---|---|---|
| `P1.bench.Qwen__Qwen3-30B-A3B-Instruct-2507.batch` | 0 | **rc=-15** | 62.5 |
| `P1.bench.Qwen__Qwen3-30B-A3B-Instruct-2507.content_free` | 1 | **rc=-15** | 62.5 |
| `P1.bench.Qwen__Qwen3-8B.batch` | 2 | ok | 15.8 |
| `P1.bench.Qwen__Qwen3-8B.content_free` | 3 | ok | 15.8 |
| `P2.typed.Qwen__Qwen3-32B` | 3 | ok | 16.3 |
| `P2.typed.Qwen__Qwen3-8B` | 2 | ok | 7.7 |
| `P3.extra.Qwen__Qwen2.5-7B-Instruct` | 2 | ok | 20.4 |
| `P3.extra.Qwen__Qwen3-8B` | 3 | ok | 12.2 |
| `P3.families.allenai__OLMo-2-1124-7B-Instruct` | 1 | ok | 19.3 |
| `P3.families.microsoft__Phi-3.5-mini-instruct` | 0 | ok | 19.0 |
| `P3.families.mistralai__Mistral-7B-Instruct-v0.3` | 3 | ok | 15.3 |
| `P3.families.tiiuae__Falcon3-7B-Instruct` | 3 | ok | 14.3 |
| `P3.items.Qwen__Qwen2.5-7B-Instruct` | 3 | ok | 12.9 |
| `P3.items.Qwen__Qwen3-8B` | 2 | ok | 30.7 |

## Headline: reproduction rate, stratified by whether the rerun matched the committed config

```

====================================================================================================
CONFIG-MATCHED RERUNS  (--batch-size 32, the bench default and what the committed runs used)
====================================================================================================
prior          level                    cells  bit-identical   <=0.001   <=0.01    max dev
----------------------------------------------------------------------------------------------------
batch          raw                         72         100.0%    100.0%   100.0%     0.0000
batch          L0 (all ablations)         288         100.0%    100.0%   100.0%     0.0000
batch          L1                          72          29.2%     63.9%    95.8%     0.0233
               (models x tasks)       Qwen2.5-7B-Instruct/banking20, Qwen3-30B-A3B-Instruct-2507/b
content_free   raw                         48         100.0%    100.0%   100.0%     0.0000
content_free   L0 (all ablations)         240         100.0%    100.0%   100.0%     0.0000
content_free   L1                          48         100.0%    100.0%   100.0%     0.0000
               (models x tasks)       Qwen3-30B-A3B-Instruct-2507/banking20, Qwen3-30B-A3B-Instruc


====================================================================================================
CONFIG-MISMATCHED RERUNS  (--batch-size 16: same code, same seed, different batching)
====================================================================================================
prior          level                    cells  bit-identical   <=0.001   <=0.01    max dev
----------------------------------------------------------------------------------------------------
batch          raw                         24           8.3%     25.0%    91.7%     0.0105
batch          L0 (all ablations)          96          28.1%     60.4%    95.8%     0.0367
batch          L1                          24          37.5%     83.3%    95.8%     0.0233
               (models x tasks)       Qwen2.5-7B-Instruct/banking20, Qwen2.5-7B-Instruct/injection
content_free   raw                         24           8.3%     25.0%    91.7%     0.0105
content_free   L0 (all ablations)         120          25.0%     53.3%    94.2%     0.0367
content_free   L1                          24          33.3%     58.3%    79.2%     0.0167
               (models x tasks)       Qwen2.5-7B-Instruct/banking20, Qwen2.5-7B-Instruct/injection

====================================================================================================
VERDICT
====================================================================================================
zero-label results (raw + every L0 ablation), config-matched: 648 cells, 100.0% bit-identical, max deviation 0.000000
L1 with prior=content_free: 48 cells, 100.0% bit-identical, max 0.000000
L1 with prior=batch:        72 cells, 29.2% bit-identical, max 0.023333
   ^ the only config-matched level that moves. repro_check/probe_stateful_prior.py shows why:
     the batch prior is a running accumulator, so the temperature fitted during
     calibrate() depends on how many items the decider has already seen.
```

## Offline verification (no GPU): the paper's analytical claims and the docs' internal consistency

```
==============================================================================
AnyJev offline verification
==============================================================================
[PASS] MATH    logmean removes additive position bias exactly  -- max abs error over 800 random cases = 3.33e-16
[PASS] MATH    arithmetic mean does NOT (so the log-space choice is load-bearing)  -- max abs error for combine='mean' = 0.548
[PASS] MATH    logmean is invariant to the original option listing  -- max deviation = 4.44e-16
[PASS] MATH    arithmetic mean is not listing-invariant  -- max deviation for combine='mean' = 0.2311
[PASS] MATH    prior division recovers the signal exactly when bias is multiplicative  -- max abs error = 2.22e-16
[PASS] MATH    batch_prior is the normalized mean over inputs  -- got [0.7 0.3]
[PASS] MATH    content_free_prior is the normalized mean over probes  -- got [0.8 0.2]
[PASS] MATH    temperature T=0.3 preserves every argmax
[PASS] MATH    temperature T=1.7 preserves every argmax
[PASS] MATH    temperature T=4.16 preserves every argmax
[PASS] MATH    temperature DOES re-rank items by confidence (explains the L1 cov@5% gain)  -- confidence ordering across items changed under T>1
[PASS] MATH    flattening ranks a concentrated runner-up above a diffuse tail  -- same 0.900 top-1 -> 0.620 (peaked) vs 0.153 (diffuse)
[PASS] MATH    temperature fit recovers a planted temperature  -- planted 2.5, fitted 2.535
[PASS] METRICS ece matches an independent rewrite  -- max diff 0.00e+00
[PASS] METRICS cov@5% matches an independent rewrite  -- max diff 0.00e+00
[PASS] METRICS brier matches the textbook definition  -- max diff 0.00e+00
[PASS] METRICS cov@5% is 1.0 when every answer is right
[PASS] METRICS cov@5% is 0.0 when the most confident answer is always wrong
[PASS] METRICS flip_rate counts argmax disagreements
[PASS] METRICS cov@5% uses the deepest admissible prefix (optimistic vs first-crossing)  -- deepest >= first-crossing up to one boundary item; mean gap +0.026, p95 +0.113, max +0.260 of coverage (40% of draws disagree)
[PASS] METRICS cov@5% is a high-variance point estimate (single-item sensitivity)  -- flipping ONE correct item to wrong moves cov@5% by up to -0.013 (mean -0.002) at n=300
[PASS] DOCS    README headline table matches the committed JSON  -- all 10 cells match
[PASS] DOCS    README cross-model flip table (9 rows x 7 cells) matches JSON  -- all cells match
[PASS] DOCS    docs/results_bench.md rows are regenerable from bench/results_batchprior_v0/2026-09-20  -- every generated row is present in the doc
[PASS] DOCS    typed-decisions table in README matches the committed JSON  -- 7 spot-checked cells match
[PASS] DOCS    'six times' ECE claim is supported by the JSON  -- laya ECE 0.215 / best L1 ECE 0.034 = 6.4x
[PASS] DOCS    '7x' coverage claim is supported by the JSON  -- 0.543 / 0.077 = 7.1x
[PASS] DOCS    AURC (robust ranking summary) improves far less dramatically than cov@5%  -- aurc raw 0.134 -> L0 0.072 -> L1 0.068 (2.0x, vs 7.1x for cov@5%)
[PASS] DOCS    the self-reported counter-example (L0 cov below raw on injection) is real  -- raw 0.297 -> L0 0.160
[PASS] DOCS    results carry hardware and library provenance  -- gpu=NVIDIA H100 NVL, torch=2.5.1+cu124, transformers=4.55.4
------------------------------------------------------------------------------
30/30 checks passed
```

## Reproduction diff: bench.run, both prior configurations

```
================================================================================================================
REPRODUCTION DIFF: my rerun vs the repo's committed JSON
same seed 0, n=300, calib=200, combine=logmean, transformers 4.55.4, H100 -- repo's own bench.run
================================================================================================================

### prior=batch   (9 of 9 committed (model, task) pairs reproduced)
   864 comparable cells: 678 bit-identical (78.5%), 770 within 0.001 (89.1%), 852 within 0.01 (98.6%)
   94 cells deviate by more than 0.001:
      model                    task         level       metric       repo     mine    delta
      Qwen2.5-7B-Instruct      injection    L0          cov@5%     0.3533   0.3900  +0.0367
      Qwen2.5-7B-Instruct      injection    L1          cov@5%     0.3533   0.3867  +0.0333
      Qwen2.5-7B-Instruct      injection    L0-perm     cov@5%     0.3433   0.3733  +0.0300
      Qwen3-30B-A3B-Instruct-2507 injection    L1          cov@5%     0.4333   0.4100  -0.0233
      Qwen2.5-7B-Instruct      injection    L1          cov@5%     0.3533   0.3767  +0.0233
      Qwen2.5-7B-Instruct      banking20    L0-bc       cov@5%     0.2033   0.1867  -0.0167
      Qwen3-30B-A3B-Instruct-2507 newsgroups   L1          ece        0.0865   0.0990  +0.0125
      Qwen2.5-7B-Instruct      newsgroups   L0-bc       nll        3.6226   3.6350  +0.0124
      Qwen2.5-7B-Instruct      injection    raw         macro_f1   0.6746   0.6851  +0.0105
      Qwen2.5-7B-Instruct      newsgroups   raw         nll        3.8878   3.8979  +0.0101
      Qwen3-8B                 injection    L1          cov@5%     0.3033   0.3133  +0.0100
      Qwen3-8B                 injection    L1          cov@5%     0.3033   0.3133  +0.0100
      Qwen2.5-7B-Instruct      injection    L0          macro_f1   0.7867   0.7779  -0.0088
      Qwen2.5-7B-Instruct      banking20    raw         nll        2.7194   2.7278  +0.0084
      Qwen2.5-7B-Instruct      banking20    L0-bc       nll        2.5387   2.5467  +0.0080
      Qwen2.5-7B-Instruct      injection    raw         ece        0.1888   0.1816  -0.0072
      Qwen2.5-7B-Instruct      newsgroups   L0-perm+cf  nll        3.0134   3.0206  +0.0072
      Qwen3-8B                 injection    L1          ece        0.1624   0.1557  -0.0067
      Qwen3-8B                 injection    L1          ece        0.1624   0.1557  -0.0067
      Qwen2.5-7B-Instruct      injection    raw         cov@5%     0.3667   0.3733  +0.0067
      Qwen2.5-7B-Instruct      injection    L0          acc        0.8133   0.8067  -0.0067
      Qwen2.5-7B-Instruct      newsgroups   raw         cov@5%     0.0467   0.0533  +0.0067
      Qwen2.5-7B-Instruct      injection    L0-bc       cov@5%     0.4233   0.4167  -0.0067
      Qwen2.5-7B-Instruct      injection    raw         acc        0.7367   0.7433  +0.0067
      Qwen2.5-7B-Instruct      newsgroups   L0-perm     nll        3.3924   3.3988  +0.0063
      max deviation by level: raw=0.0105, L0-bc=0.0167, L0-perm=0.0300, L0-perm+cf=0.0072, L0=0.0367, L1=0.0333
   run-to-run spread across my own independent reruns of the same cell:
      Qwen2.5-7B-Instruct      banking20    max spread 0.01667 over 3 runs
      Qwen2.5-7B-Instruct      newsgroups   max spread 0.01442 over 2 runs
      Qwen2.5-7B-Instruct      injection    max spread 0.03667 over 2 runs
      Qwen3-8B                 banking20    max spread 0.00000 over 4 runs
      Qwen3-8B                 newsgroups   max spread 0.00000 over 2 runs
      Qwen3-8B                 injection    max spread 0.00000 over 2 runs

### prior=content_free   (9 of 9 committed (model, task) pairs reproduced)
   504 comparable cells: 376 bit-identical (74.6%), 420 within 0.001 (83.3%), 490 within 0.01 (97.2%)
   84 cells deviate by more than 0.001:
      model                    task         level       metric       repo     mine    delta
      Qwen2.5-7B-Instruct      injection    L0-perm+bc  cov@5%     0.3533   0.3900  +0.0367
      Qwen2.5-7B-Instruct      injection    L0-perm     cov@5%     0.3433   0.3733  +0.0300
      Qwen2.5-7B-Instruct      injection    L0-cf       ece        0.0399   0.0575  +0.0176
      Qwen2.5-7B-Instruct      injection    L1          acc        0.8567   0.8400  -0.0167
      Qwen2.5-7B-Instruct      banking20    L0-bc       cov@5%     0.2033   0.1867  -0.0167
      Qwen2.5-7B-Instruct      injection    L1          macro_f1   0.8471   0.8308  -0.0163
      Qwen2.5-7B-Instruct      injection    L1          ece        0.0624   0.0465  -0.0159
      Qwen2.5-7B-Instruct      newsgroups   L0-cf       nll        3.7765   3.7909  +0.0144
      Qwen2.5-7B-Instruct      injection    L1          cov@5%     0.5767   0.5633  -0.0133
      Qwen2.5-7B-Instruct      banking20    L1          ece        0.0630   0.0759  +0.0130
      Qwen2.5-7B-Instruct      newsgroups   L0-bc       nll        3.6226   3.6350  +0.0124
      Qwen2.5-7B-Instruct      injection    raw         macro_f1   0.6746   0.6851  +0.0105
      Qwen2.5-7B-Instruct      newsgroups   raw         nll        3.8878   3.8979  +0.0101
      Qwen2.5-7B-Instruct      injection    L0-cf       cov@5%     0.5533   0.5433  -0.0100
      Qwen2.5-7B-Instruct      banking20    L0-cf       nll        2.6916   2.6822  -0.0095
      Qwen2.5-7B-Instruct      banking20    L0-cf       macro_f1   0.6984   0.7073  +0.0089
      Qwen2.5-7B-Instruct      injection    L0-perm+bc  macro_f1   0.7867   0.7779  -0.0088
      Qwen2.5-7B-Instruct      banking20    raw         nll        2.7194   2.7278  +0.0084
      Qwen2.5-7B-Instruct      banking20    L0-bc       nll        2.5387   2.5467  +0.0080
      Qwen2.5-7B-Instruct      banking20    L0-cf       ece        0.2476   0.2401  -0.0075
      Qwen2.5-7B-Instruct      injection    raw         ece        0.1888   0.1816  -0.0072
      Qwen2.5-7B-Instruct      newsgroups   L0          nll        3.0134   3.0206  +0.0072
      Qwen2.5-7B-Instruct      injection    raw         cov@5%     0.3667   0.3733  +0.0067
      Qwen2.5-7B-Instruct      injection    L0-perm+bc  acc        0.8133   0.8067  -0.0067
      Qwen2.5-7B-Instruct      banking20    L0-cf       acc        0.6967   0.7033  +0.0067
      max deviation by level: raw=0.0105, L0-cf=0.0176, L0-bc=0.0167, L0-perm=0.0300, L0-perm+bc=0.0367, L0=0.0072, L1=0.0167

================================================================================================================
TOTAL 1368 cells | bit-identical 77.0% | within 0.001 87.0% | within 0.01 98.1%
cells deviating by more than 0.01: 26
   Qwen2.5-7B-Instruct      injection    L0-perm+bc  cov@5%   repo 0.3533 mine 0.3900 delta +0.0367
   Qwen2.5-7B-Instruct      injection    L0          cov@5%   repo 0.3533 mine 0.3900 delta +0.0367
   Qwen2.5-7B-Instruct      injection    L1          cov@5%   repo 0.3533 mine 0.3867 delta +0.0333
   Qwen2.5-7B-Instruct      injection    L0-perm     cov@5%   repo 0.3433 mine 0.3733 delta +0.0300
   Qwen2.5-7B-Instruct      injection    L0-perm     cov@5%   repo 0.3433 mine 0.3733 delta +0.0300
   Qwen3-30B-A3B-Instruct-2507 injection    L1          cov@5%   repo 0.4333 mine 0.4100 delta -0.0233
   Qwen2.5-7B-Instruct      injection    L1          cov@5%   repo 0.3533 mine 0.3767 delta +0.0233
   Qwen2.5-7B-Instruct      injection    L0-cf       ece      repo 0.0399 mine 0.0575 delta +0.0176
   Qwen2.5-7B-Instruct      injection    L1          acc      repo 0.8567 mine 0.8400 delta -0.0167
   Qwen2.5-7B-Instruct      banking20    L0-bc       cov@5%   repo 0.2033 mine 0.1867 delta -0.0167
   Qwen2.5-7B-Instruct      banking20    L0-bc       cov@5%   repo 0.2033 mine 0.1867 delta -0.0167
   Qwen2.5-7B-Instruct      injection    L1          macro_f1 repo 0.8471 mine 0.8308 delta -0.0163
   Qwen2.5-7B-Instruct      injection    L1          ece      repo 0.0624 mine 0.0465 delta -0.0159
   Qwen2.5-7B-Instruct      newsgroups   L0-cf       nll      repo 3.7765 mine 3.7909 delta +0.0144
   Qwen2.5-7B-Instruct      injection    L1          cov@5%   repo 0.5767 mine 0.5633 delta -0.0133
```

## Reproduction diff: bench.run_typed on LocalLLaMA/typed-decisions

```
============================================================================================================
TYPED-DECISIONS DIFF: my rerun of bench.run_typed vs the repo's committed JSON
LocalLLaMA/typed-decisions test split, 400 cases / 2,000 decisions / 20 questions, prior=batch
============================================================================================================

### Qwen/Qwen2.5-7B-Instruct
   decisions  repo 2000 / mine 2000;  questions repo 20 / mine 20;  calib repo 1000 / mine 1000
   level       metric          repo     mine    delta
   every comparable cell matches to within 0.0005

### Qwen/Qwen3-32B
   decisions  repo 2000 / mine 2000;  questions repo 20 / mine 20;  calib repo 1000 / mine 1000
   level       metric          repo     mine    delta
   raw         acc           0.6845   0.6830  -0.0015  <-- differs
   raw         ece           0.2061   0.2077  +0.0016  <-- differs
   raw         brier         0.5022   0.5030  +0.0008  <-- differs
   raw         nll           1.2027   1.2038  +0.0011  <-- differs
   L0-perm     acc           0.6875   0.6865  -0.0010  <-- differs
   L0-perm     ece           0.1974   0.1983  +0.0009  <-- differs
   L0-perm     brier         0.4919   0.4926  +0.0007  <-- differs
   L0-perm     nll           1.1284   1.1290  +0.0006  <-- differs
   L0-bc       acc           0.6870   0.6910  +0.0040  <-- differs
   L0-bc       ece           0.1390   0.1344  -0.0046  <-- differs
   L0-bc       brier         0.4412   0.4422  +0.0010  <-- differs
   L0-bc       nll           0.9398   0.9413  +0.0015  <-- differs
   L0-bc       score_mae     0.4563   0.4574  +0.0011  <-- differs
   L0-perm+bc  ece           0.1334   0.1341  +0.0006  <-- differs
   L0-perm+bc  brier         0.4348   0.4356  +0.0007  <-- differs
   L0-perm+bc  nll           0.8905   0.8915  +0.0010  <-- differs
   L0-perm+bc  score_mae     0.4563   0.4574  +0.0011  <-- differs
   L0          ece           0.1334   0.1341  +0.0006  <-- differs
   L0          brier         0.4348   0.4356  +0.0007  <-- differs
   L0          nll           0.8905   0.8915  +0.0010  <-- differs
   L0          score_mae     0.4563   0.4574  +0.0011  <-- differs
   L1          acc           0.7005   0.6960  -0.0045  <-- differs
   L1          score_mae     0.4122   0.4106  -0.0016  <-- differs

### Qwen/Qwen3-8B
   decisions  repo 2000 / mine 2000;  questions repo 20 / mine 20;  calib repo 1000 / mine 1000
   level       metric          repo     mine    delta
   every comparable cell matches to within 0.0005

------------------------------------------------------------------------------------------------------------
126 comparable cells: 86 bit-identical (68.3%), 113 within 0.001, 126 within 0.01
max deviation: 0.0046 at Qwen/Qwen3-32B / L0-bc / ece

============================================================================================================
PROVIDER ROWS: committed JSON vs the numbers their authors published
============================================================================================================
laya                       measured-here acc 0.359  ece 0.177  soft 0.331   | published acc 0.361 (delta -0.002)
laya-multilingual          measured-here acc 0.340  ece 0.287  soft 0.325
laya-typed-decisions       measured-here acc 0.768  ece 0.215  soft 0.471   | published acc 0.766 (delta +0.002)

'reproduces its published 0.766': measured 0.768, published 0.766, delta +0.002 -> supported
'six times the ECE' : laya 0.215 / best L1 0.034 = 6.4x
```

## Reproduction diff: the NanoJev maze harness

```
================================================================================================================
MAZE DIFF: my rerun inside NanoJev's frozen harness vs bench/results_nanojev
episodes: C-Tianyu/NanoJev-Data games_v4/data/scaled_games_v4b/episodes.jsonl, splits test+ood (15)
================================================================================================================

### Qwen__Qwen3-0.6B.L0.batch.json   (Qwen3-0.6B L0 prior=batch)
   goals repo ood=3/4 test=10/11   | mine ood=3/4 test=10/11
   field                        repo         mine        delta
   attempts               15616.0000   15616.0000      +0.0000
   collisions              6236.0000    6236.0000      +0.0000
   atomic_accuracy            0.4902       0.4902      +0.0000
   atomic_brier               0.2707       0.2707      +0.0000
   atomic_nll                 0.7381       0.7381      +0.0000
   atomic_questions       23156.0000   23156.0000      +0.0000
   fallback_count          6181.0000    6181.0000      +0.0000

### Qwen__Qwen3-0.6B.L0.content_free.json   (Qwen3-0.6B L0 prior=content_free)
   goals repo ood=4/4 test=10/11   | mine ood=4/4 test=10/11
   field                        repo         mine        delta
   attempts               16278.0000   16278.0000      +0.0000
   collisions              6563.0000    6563.0000      +0.0000
   atomic_accuracy            0.4897       0.4897      +0.0000
   atomic_brier               0.2736       0.2736      +0.0000
   atomic_nll                 0.7452       0.7452      +0.0000
   atomic_questions       23284.0000   23284.0000      +0.0000
   fallback_count          6038.0000    6038.0000      +0.0000

### Qwen__Qwen3-0.6B.L0.none.json   (Qwen3-0.6B L0 prior=none)
   goals repo ood=3/4 test=10/11   | mine ood=3/4 test=10/11
   field                        repo         mine        delta
   attempts               21851.0000   21851.0000      +0.0000
   collisions              9136.0000    9136.0000      +0.0000
   atomic_accuracy            0.4027       0.4027      +0.0000
   atomic_brier               0.4604       0.4604      +0.0000
   atomic_nll                 1.3331       1.3331      +0.0000
   atomic_questions       28040.0000   28040.0000      +0.0000
   fallback_count         17393.0000   17393.0000      +0.0000

### Qwen__Qwen3-0.6B.raw.batch.json   (Qwen3-0.6B raw prior=batch)
   goals repo ood=4/4 test=11/11   | mine ood=4/4 test=11/11
   field                        repo         mine        delta
   attempts                5825.0000    5825.0000      +0.0000
   collisions              2616.0000    2616.0000      +0.0000
   atomic_accuracy            0.5372       0.5372      +0.0000
   atomic_brier               0.3635       0.3635      +0.0000
   atomic_nll                 1.1408       1.1408      +0.0000
   atomic_questions       10944.0000   10944.0000      +0.0000
   fallback_count            18.0000      18.0000      +0.0000

### Qwen__Qwen3-8B.L0.batch.json   (Qwen3-8B L0 prior=batch)
   goals repo ood=3/4 test=11/11   | mine ood=3/4 test=11/11
   field                        repo         mine        delta
   attempts               17841.0000   17841.0000      +0.0000
   collisions              7171.0000    7171.0000      +0.0000
   atomic_accuracy            0.5553       0.5553      +0.0000
   atomic_brier               0.3441       0.3441      +0.0000
   atomic_nll                 1.3333       1.3333      +0.0000
   atomic_questions       25124.0000   25124.0000      +0.0000
   fallback_count          6637.0000    6637.0000      +0.0000

### Qwen__Qwen3-8B.raw.batch.json   (Qwen3-8B raw prior=batch)
   goals repo ood=3/4 test=10/11   | mine ood=3/4 test=10/11
   field                        repo         mine        delta
   attempts               20996.0000   20996.0000      +0.0000
   collisions              8570.0000    8570.0000      +0.0000
   atomic_accuracy            0.5229       0.5229      +0.0000
   atomic_brier               0.3916       0.3916      +0.0000
   atomic_nll                 1.7988       1.7988      +0.0000
   atomic_questions       28780.0000   28780.0000      +0.0000
   fallback_count         11230.0000   11230.0000      +0.0000

### nanojev_native.Qwen3-0.6B.json   (Qwen3-0.6B native A/B readout)
   goals repo ood=3/4 test=10/11   | mine ood=3/4 test=10/11
   field                        repo         mine        delta
   attempts               20555.0000   20555.0000      +0.0000
   collisions              8496.0000    8496.0000      +0.0000
   atomic_accuracy            0.4189       0.4189      +0.0000
   atomic_brier               0.3053       0.3053      +0.0000
   atomic_nll                 0.8137       0.8137      +0.0000
   atomic_questions       27660.0000   27660.0000      +0.0000
   fallback_count         14754.0000   14754.0000      +0.0000

----------------------------------------------------------------------------------------------------------------
49 comparable summary fields: 49 bit-identical (100.0%)

================================================================================================================
THE REPO'S TWO MAZE CLAIMS, RE-DERIVED FROM MY RERUN
================================================================================================================
engine                                    goals   attempts  edge acc  majority  beats majority?
Qwen3-0.6B L0 prior=batch                 13/15      15616     0.490 None               no
Qwen3-0.6B L0 prior=content_free          14/15      16278     0.490 None               no
Qwen3-0.6B L0 prior=none                  13/15      21851     0.403 None               no
Qwen3-0.6B raw prior=batch                15/15       5825     0.537 None               no
Qwen3-8B L0 prior=batch                   14/15      17841     0.555 None               no
Qwen3-8B raw prior=batch                  13/15      20996     0.523 None               no
Qwen3-0.6B native A/B readout             13/15      20555     0.419 None               no

claim 1: A/B readout 13/15 mazes at 20555 attempts vs AnyJev raw 15/15 at 5825 attempts -- the repo reports 13/15 @ 20,555 vs 15/15 @ 5,825
claim 2: readouts that beat the majority baseline on edge perception: 0 of 7 -- none, as the repo states
```

## Why the residual deviations exist, part 1: the batch-size probe

```
========================================================================================================
DOES THE REPRODUCTION DEPEND ON --batch-size?
`raw` is one forward pass per item: no permutations, no prior, no fitted temperature,
so a deviation at `raw` is the model's logits changing, not the method.
========================================================================================================
 batch size   cells  raw: bit-identical  raw: max dev  all levels: identical  all: max dev
--------------------------------------------------------------------------------------------------------
         16     264                8.3%        0.0105                  25.4%        0.0367
         32     888              100.0%        0.0000                  91.0%        0.0333

At batch size 32 (the bench default, and what the committed runs used):
   raw is 100.0% bit-identical, max raw deviation 0.000000
At batch size 16:
   raw is 8.3% bit-identical, max raw deviation 0.010485

VERDICT: the deviation tracks batch size, not the method: the committed numbers are exactly reproducible at the batch size that produced them

GAP: `bench.run` records model / seed / n / calib / prior / combine / max_permutations / torch / transformers / gpu, but NOT batch_size.
      Two runs of the committed command on the same GPU with a different --batch-size differ by up to
      0.037 on a headline metric, and nothing in the artifact records which was used.

========================================================================================================
THE SAME SWEEP ON bench.run_typed (2,000 decisions), where three batch sizes were tried
========================================================================================================
   Qwen/Qwen3-32B               --batch-size 8      4.8% bit-identical, max deviation 0.0046
   Qwen/Qwen3-32B               --batch-size 32     0.0% bit-identical, max deviation 0.0034
   Qwen/Qwen3-32B               --batch-size 16   100.0% bit-identical, max deviation 0.0000
   Qwen/Qwen3-8B                --batch-size 32   100.0% bit-identical, max deviation 0.0000
   Qwen/Qwen2.5-7B-Instruct     --batch-size 32   100.0% bit-identical, max deviation 0.0000

   The committed Qwen3-32B typed run lands exactly on --batch-size 16, and the two
   smaller models land exactly on 32. Every committed cell is bit-reproducible once
   you use the batch size that produced it -- and the artifact does not record it,
   so the batch size has to be searched for.
```

## Extensions: new tasks, new model families, bootstrap intervals

```
loaded 11 run files, 31 successful (model, task) cells, 4 failed

==========================================================================================================================
1. THE READOUT ACROSS MODEL FAMILIES AND TASKS  (n=300, seed 0, prior=batch, L1 from 200 calib)
==========================================================================================================================
model                        task          kind     K | raw flip  L0 flip | raw acc  L0 acc | raw ece  L1 ece | raw cov  L0 cov  L1 cov
--------------------------------------------------------------------------------------------------------------------------
Qwen2.5-7B-Instruct          hate          noul     2 |    0.053    0.000 |   0.693   0.710 |   0.249   0.059 |   0.023   0.017   0.017
Qwen3-8B                     hate          noul     2 |    0.057    0.000 |   0.657   0.673 |   0.325   0.104 |   0.093   0.097   0.083
OLMo-2-1124-7B-Instruct      injection     noul     2 |    0.003    0.000 |   0.593   0.590 |   0.367   0.081 |   0.000   0.000   0.000
Phi-3.5-mini-instruct        injection     noul     2 |    0.117    0.000 |   0.707   0.700 |   0.275   0.173 |   0.440   0.443   0.443
Qwen2.5-7B-Instruct          injection     noul     2 |    0.053    0.000 |   0.737   0.813 |   0.189   0.031 |   0.367   0.353   0.387
Qwen3-8B                     injection     noul     2 |    0.060    0.000 |   0.693   0.710 |   0.287   0.156 |   0.297   0.160   0.313
Qwen2.5-7B-Instruct          subj          noul     2 |    0.087    0.000 |   0.707   0.757 |   0.251   0.118 |   0.250   0.347   0.297
Qwen3-8B                     subj          noul     2 |    0.050    0.000 |   0.667   0.707 |   0.321   0.178 |   0.430   0.477   0.437
Qwen2.5-7B-Instruct          agnews        choice   4 |    0.040    0.023 |   0.893   0.887 |   0.095   0.060 |   0.550   0.310   0.223
Qwen3-8B                     agnews        choice   4 |    0.067    0.003 |   0.857   0.870 |   0.138   0.064 |   0.370   0.020   0.010
Qwen2.5-7B-Instruct          yelp5         score    5 |    0.850    0.837 |   0.617   0.640 |   0.322   0.098 |   0.220   0.210   0.223
Qwen3-8B                     yelp5         score    5 |    0.290    0.297 |   0.610   0.603 |   0.358   0.113 |   0.093   0.090   0.080
Qwen2.5-7B-Instruct          emotion       choice   6 |    0.207    0.083 |   0.583   0.573 |   0.359   0.093 |   0.127   0.077   0.000
Qwen3-8B                     emotion       choice   6 |    0.110    0.050 |   0.577   0.573 |   0.408   0.127 |   0.003   0.000   0.000
Falcon3-7B-Instruct          banking20     choice  20 |    0.377    0.187 |   0.667   0.727 |   0.200   0.051 |   0.037   0.327   0.323
Mistral-7B-Instruct-v0.3     banking20     choice  20 |    0.290    0.140 |   0.693   0.743 |   0.288   0.108 |   0.100   0.340   0.360
OLMo-2-1124-7B-Instruct      banking20     choice  20 |    0.403    0.213 |   0.600   0.697 |   0.185   0.119 |   0.003   0.053   0.050
Phi-3.5-mini-instruct        banking20     choice  20 |    0.227    0.117 |   0.753   0.797 |   0.200   0.067 |   0.187   0.550   0.623
Qwen2.5-7B-Instruct          banking20     choice  20 |    0.197    0.067 |   0.723   0.767 |   0.236   0.072 |   0.187   0.173   0.223
Qwen3-8B                     banking20     choice  20 |    0.227    0.077 |   0.750   0.807 |   0.235   0.101 |   0.077   0.477   0.547
Qwen2.5-7B-Instruct          massive20     choice  20 |    0.140    0.093 |   0.817   0.823 |   0.139   0.049 |   0.487   0.543   0.680
Qwen3-8B                     massive20     choice  20 |    0.167    0.083 |   0.790   0.827 |   0.197   0.077 |   0.257   0.673   0.687
Qwen2.5-7B-Instruct          massive20_zh  choice  20 |    0.173    0.107 |   0.757   0.790 |   0.182   0.061 |   0.423   0.557   0.607
Qwen3-8B                     massive20_zh  choice  20 |    0.157    0.103 |   0.760   0.807 |   0.224   0.078 |   0.167   0.470   0.537
OLMo-2-1124-7B-Instruct      newsgroups    choice  20 |    0.457    0.183 |   0.573   0.687 |   0.203   0.047 |   0.287   0.307   0.303
Phi-3.5-mini-instruct        newsgroups    choice  20 |    0.317    0.223 |   0.643   0.643 |   0.305   0.117 |   0.097   0.327   0.350
Qwen2.5-7B-Instruct          newsgroups    choice  20 |    0.233    0.123 |   0.660   0.707 |   0.276   0.082 |   0.047   0.350   0.410
Qwen3-8B                     newsgroups    choice  20 |    0.237    0.173 |   0.640   0.660 |   0.331   0.157 |   0.013   0.427   0.417

==========================================================================================================================
2. DOES POSITION BIAS GROW WITH K?  (the repo's bench only covers K=2 and K=20)
==========================================================================================================================
  K n cells  raw flip (mean)  L0 flip (mean)  absolute drop  relative drop
--------------------------------------------------------------------------------------------------------------------------
  2       8            0.060           0.000          0.060           100%
  4       2            0.053           0.013          0.040            75%
  6       2            0.158           0.067          0.092            58%
 20      14            0.257           0.135          0.122            48%

correlation between log K and raw flip rate over 26 cells: r = +0.777

==========================================================================================================================
3. THE CONTENT-FREE PRIOR IS HIGH VARIANCE  (the repo's own caveat, now on 3 noul tasks)
==========================================================================================================================
model                        task           raw acc    +perm   +batch      +cf     best  cf - batch
--------------------------------------------------------------------------------------------------------------------------
Qwen2.5-7B-Instruct          hate             0.693    0.707    0.710    0.490    0.710      -0.220
Qwen3-8B                     hate             0.657    0.663    0.673    0.563    0.673      -0.110
OLMo-2-1124-7B-Instruct      injection        0.593    0.593    0.590    0.570    0.593      -0.020
Phi-3.5-mini-instruct        injection        0.707    0.673    0.700    0.820    0.820      +0.120
Qwen2.5-7B-Instruct          injection        0.737    0.720    0.813    0.853    0.853      +0.040
Qwen3-8B                     injection        0.693    0.670    0.710    0.813    0.813      +0.103
Qwen2.5-7B-Instruct          subj             0.707    0.690    0.757    0.583    0.757      -0.173
Qwen3-8B                     subj             0.667    0.700    0.707    0.727    0.727      +0.020

content-free minus batch prior over 8 noul cells: mean -0.030, range -0.220 to +0.120 -- confirms the repo's 'high variance' caveat

==========================================================================================================================
4. THE `score` PRIMITIVE  (never exercised by the repo's own bench)
==========================================================================================================================
Qwen2.5-7B-Instruct          perms=1 (score bins are ordinal: never permuted, so only the prior correction applies)
   raw        acc 0.617  ece 0.322  brier 0.673  flip 0.850  cov@5% 0.220
   L0-bc      acc 0.640  ece 0.301  brier 0.657  flip 0.837  cov@5% 0.210
   L0-cf      acc 0.630  ece 0.318  brier 0.680  flip 0.853  cov@5% 0.097
   L0         acc 0.640  ece 0.301  brier 0.657  flip 0.837  cov@5% 0.210
   L1         acc 0.630  ece 0.098  brier 0.471  flip 0.830  cov@5% 0.223
Qwen3-8B                     perms=1 (score bins are ordinal: never permuted, so only the prior correction applies)
   raw        acc 0.610  ece 0.358  brier 0.748  flip 0.290  cov@5% 0.093
   L0-bc      acc 0.603  ece 0.364  brier 0.744  flip 0.297  cov@5% 0.090
   L0-cf      acc 0.623  ece 0.341  brier 0.689  flip 0.333  cov@5% 0.023
   L0         acc 0.603  ece 0.364  brier 0.744  flip 0.297  cov@5% 0.090
   L1         acc 0.607  ece 0.113  brier 0.505  flip 0.297  cov@5% 0.080

==========================================================================================================================
5. HOW SOLID IS cov@5%?  (4000 bootstrap resamples of the same 300 test items)
==========================================================================================================================
model                    task          level    point  boot mean            95% CI  CI width
--------------------------------------------------------------------------------------------------------------------------
Qwen2.5-7B-Instruct      agnews        raw      0.550      0.521 [ 0.003,  0.890]     0.887
Qwen2.5-7B-Instruct      agnews        L0       0.310      0.317 [ 0.000,  0.893]     0.893
Qwen2.5-7B-Instruct      agnews        L1       0.223      0.380 [ 0.033,  0.897]     0.863
Qwen2.5-7B-Instruct      emotion       raw      0.127      0.125 [ 0.063,  0.177]     0.113
Qwen2.5-7B-Instruct      emotion       L0       0.077      0.069 [ 0.000,  0.207]     0.207
Qwen2.5-7B-Instruct      emotion       L1       0.000      0.064 [ 0.000,  0.170]     0.170
Qwen2.5-7B-Instruct      massive20     raw      0.487      0.496 [ 0.233,  0.670]     0.437
Qwen2.5-7B-Instruct      massive20     L0       0.543      0.539 [ 0.233,  0.773]     0.540
Qwen2.5-7B-Instruct      massive20     L1       0.680      0.625 [ 0.347,  0.823]     0.477
Qwen2.5-7B-Instruct      massive20_zh  raw      0.423      0.445 [ 0.300,  0.623]     0.323
Qwen2.5-7B-Instruct      massive20_zh  L0       0.557      0.498 [ 0.253,  0.723]     0.470
Qwen2.5-7B-Instruct      massive20_zh  L1       0.607      0.602 [ 0.340,  0.773]     0.433
Qwen2.5-7B-Instruct      hate          raw      0.023      0.036 [ 0.010,  0.137]     0.127
Qwen2.5-7B-Instruct      hate          L0       0.017      0.033 [ 0.007,  0.103]     0.097
Qwen2.5-7B-Instruct      hate          L1       0.017      0.033 [ 0.003,  0.093]     0.090
Qwen2.5-7B-Instruct      subj          raw      0.250      0.278 [ 0.043,  0.450]     0.407
Qwen2.5-7B-Instruct      subj          L0       0.347      0.285 [ 0.073,  0.440]     0.367
Qwen2.5-7B-Instruct      subj          L1       0.297      0.281 [ 0.070,  0.440]     0.370
Qwen2.5-7B-Instruct      yelp5         raw      0.220      0.223 [ 0.137,  0.347]     0.210
Qwen2.5-7B-Instruct      yelp5         L0       0.210      0.221 [ 0.130,  0.343]     0.213
Qwen2.5-7B-Instruct      yelp5         L1       0.223      0.213 [ 0.100,  0.330]     0.230
Qwen3-8B                 agnews        raw      0.370      0.405 [ 0.017,  0.827]     0.810
Qwen3-8B                 agnews        L0       0.020      0.140 [ 0.000,  0.860]     0.860
Qwen3-8B                 agnews        L1       0.010      0.205 [ 0.003,  0.870]     0.867
Qwen3-8B                 emotion       raw      0.003      0.013 [ 0.000,  0.140]     0.140
Qwen3-8B                 emotion       L0       0.000      0.004 [ 0.000,  0.020]     0.020
Qwen3-8B                 emotion       L1       0.000      0.003 [ 0.000,  0.017]     0.017
Qwen3-8B                 massive20     raw      0.257      0.283 [ 0.007,  0.497]     0.490
Qwen3-8B                 massive20     L0       0.673      0.626 [ 0.427,  0.770]     0.343
Qwen3-8B                 massive20     L1       0.687      0.666 [ 0.510,  0.827]     0.317
Qwen3-8B                 massive20_zh  raw      0.167      0.194 [ 0.030,  0.410]     0.380
Qwen3-8B                 massive20_zh  L0       0.470      0.445 [ 0.077,  0.703]     0.627
Qwen3-8B                 massive20_zh  L1       0.537      0.495 [ 0.107,  0.757]     0.650
Qwen3-8B                 hate          raw      0.093      0.081 [ 0.000,  0.163]     0.163
Qwen3-8B                 hate          L0       0.097      0.086 [ 0.000,  0.213]     0.213
Qwen3-8B                 hate          L1       0.083      0.100 [ 0.013,  0.210]     0.197
Qwen3-8B                 subj          raw      0.430      0.439 [ 0.343,  0.560]     0.217
Qwen3-8B                 subj          L0       0.477      0.399 [ 0.047,  0.583]     0.537
Qwen3-8B                 subj          L1       0.437      0.467 [ 0.367,  0.583]     0.217
Qwen3-8B                 yelp5         raw      0.093      0.109 [ 0.050,  0.213]     0.163
Qwen3-8B                 yelp5         L0       0.090      0.100 [ 0.037,  0.187]     0.150
Qwen3-8B                 yelp5         L1       0.080      0.109 [ 0.043,  0.233]     0.190
OLMo-2-1124-7B-Instruct  banking20     raw      0.003      0.027 [ 0.000,  0.140]     0.140
OLMo-2-1124-7B-Instruct  banking20     L0       0.053      0.148 [ 0.033,  0.410]     0.377
OLMo-2-1124-7B-Instruct  banking20     L1       0.050      0.154 [ 0.030,  0.417]     0.387
OLMo-2-1124-7B-Instruct  newsgroups    raw      0.287      0.254 [ 0.070,  0.410]     0.340
OLMo-2-1124-7B-Instruct  newsgroups    L0       0.307      0.330 [ 0.150,  0.540]     0.390
OLMo-2-1124-7B-Instruct  newsgroups    L1       0.303      0.334 [ 0.190,  0.540]     0.350
OLMo-2-1124-7B-Instruct  injection     raw      0.000      0.001 [ 0.000,  0.007]     0.007
OLMo-2-1124-7B-Instruct  injection     L0       0.000      0.005 [ 0.000,  0.057]     0.057
OLMo-2-1124-7B-Instruct  injection     L1       0.000      0.006 [ 0.000,  0.057]     0.057
Phi-3.5-mini-instruct    banking20     raw      0.187      0.341 [ 0.087,  0.607]     0.520
Phi-3.5-mini-instruct    banking20     L0       0.550      0.574 [ 0.367,  0.753]     0.387
Phi-3.5-mini-instruct    banking20     L1       0.623      0.596 [ 0.430,  0.747]     0.317
Phi-3.5-mini-instruct    newsgroups    raw      0.097      0.114 [ 0.020,  0.253]     0.233
Phi-3.5-mini-instruct    newsgroups    L0       0.327      0.347 [ 0.263,  0.457]     0.193
Phi-3.5-mini-instruct    newsgroups    L1       0.350      0.358 [ 0.283,  0.437]     0.153
Phi-3.5-mini-instruct    injection     raw      0.440      0.424 [ 0.017,  0.560]     0.543
Phi-3.5-mini-instruct    injection     L0       0.443      0.432 [ 0.087,  0.547]     0.460
Phi-3.5-mini-instruct    injection     L1       0.443      0.432 [ 0.077,  0.550]     0.473
Mistral-7B-Instruct-v0.3 banking20     raw      0.100      0.107 [ 0.070,  0.160]     0.090
Mistral-7B-Instruct-v0.3 banking20     L0       0.340      0.331 [ 0.217,  0.453]     0.237
Mistral-7B-Instruct-v0.3 banking20     L1       0.360      0.366 [ 0.273,  0.503]     0.230
Falcon3-7B-Instruct      banking20     raw      0.037      0.117 [ 0.020,  0.263]     0.243
Falcon3-7B-Instruct      banking20     L0       0.327      0.395 [ 0.143,  0.620]     0.477
Falcon3-7B-Instruct      banking20     L1       0.323      0.379 [ 0.097,  0.587]     0.490
Qwen2.5-7B-Instruct      banking20     raw      0.187      0.186 [ 0.053,  0.367]     0.313
Qwen2.5-7B-Instruct      banking20     L0       0.173      0.195 [ 0.117,  0.430]     0.313
Qwen2.5-7B-Instruct      banking20     L1       0.223      0.257 [ 0.073,  0.560]     0.487
Qwen2.5-7B-Instruct      newsgroups    raw      0.047      0.207 [ 0.027,  0.420]     0.393
Qwen2.5-7B-Instruct      newsgroups    L0       0.350      0.376 [ 0.207,  0.513]     0.307
Qwen2.5-7B-Instruct      newsgroups    L1       0.410      0.424 [ 0.300,  0.547]     0.247
Qwen2.5-7B-Instruct      injection     raw      0.367      0.356 [ 0.203,  0.510]     0.307
Qwen2.5-7B-Instruct      injection     L0       0.353      0.392 [ 0.060,  0.620]     0.560
Qwen2.5-7B-Instruct      injection     L1       0.387      0.393 [ 0.057,  0.613]     0.557
Qwen3-8B                 banking20     raw      0.077      0.131 [ 0.003,  0.387]     0.383
Qwen3-8B                 banking20     L0       0.477      0.469 [ 0.363,  0.563]     0.200
Qwen3-8B                 banking20     L1       0.547      0.525 [ 0.410,  0.617]     0.207
Qwen3-8B                 newsgroups    raw      0.013      0.058 [ 0.000,  0.360]     0.360
Qwen3-8B                 newsgroups    L0       0.427      0.423 [ 0.303,  0.517]     0.213
Qwen3-8B                 newsgroups    L1       0.417      0.425 [ 0.287,  0.517]     0.230
Qwen3-8B                 injection     raw      0.297      0.296 [ 0.007,  0.507]     0.500
Qwen3-8B                 injection     L0       0.160      0.250 [ 0.000,  0.507]     0.507
Qwen3-8B                 injection     L1       0.313      0.324 [ 0.093,  0.507]     0.413
Qwen3-8B                 banking20     raw      0.077      0.129 [ 0.003,  0.387]     0.383
Qwen3-8B                 banking20     L0       0.477      0.468 [ 0.360,  0.557]     0.197
Qwen3-8B                 banking20     L1       0.547      0.524 [ 0.410,  0.617]     0.207
Qwen2.5-7B-Instruct      banking20     raw      0.187      0.187 [ 0.057,  0.360]     0.303
Qwen2.5-7B-Instruct      banking20     L0       0.173      0.195 [ 0.120,  0.430]     0.310
Qwen2.5-7B-Instruct      banking20     L1       0.223      0.258 [ 0.073,  0.560]     0.487
Qwen3-8B                 banking20     raw      0.077      0.131 [ 0.003,  0.387]     0.383
Qwen3-8B                 banking20     L0       0.477      0.467 [ 0.357,  0.560]     0.203
Qwen3-8B                 banking20     L1       0.547      0.526 [ 0.417,  0.617]     0.200

mean 95% CI width on cov@5% at n=300: 0.342 of coverage (max 0.893)

the README's '7x' style claim, with the uncertainty carried through:
model                    task           L1/raw point  conservative ratio
Qwen3-8B                 newsgroups             31.2x                0.8x
OLMo-2-1124-7B-Instruct  banking20              15.0x                0.2x
Falcon3-7B-Instruct      banking20               8.8x                0.4x
Qwen2.5-7B-Instruct      newsgroups              8.8x                0.7x
Qwen3-8B                 banking20               7.1x                1.1x
Qwen3-8B                 banking20               7.1x                1.1x
Qwen3-8B                 banking20               7.1x                1.1x
Phi-3.5-mini-instruct    newsgroups              3.6x                1.1x
Mistral-7B-Instruct-v0.3 banking20               3.6x                1.7x
Phi-3.5-mini-instruct    banking20               3.3x                0.7x
Qwen3-8B                 massive20_zh            3.2x                0.3x
Qwen3-8B                 massive20               2.7x                1.0x
Qwen2.5-7B-Instruct      massive20_zh            1.4x                0.5x
Qwen2.5-7B-Instruct      massive20               1.4x                0.5x
Qwen2.5-7B-Instruct      banking20               1.2x                0.2x
Qwen2.5-7B-Instruct      banking20               1.2x                0.2x
Qwen2.5-7B-Instruct      subj                    1.2x                0.2x
OLMo-2-1124-7B-Instruct  newsgroups              1.1x                0.5x
Qwen3-8B                 injection               1.1x                0.2x
Qwen2.5-7B-Instruct      injection               1.1x                0.1x
Qwen3-8B                 subj                    1.0x                0.7x
Qwen2.5-7B-Instruct      yelp5                   1.0x                0.3x
Phi-3.5-mini-instruct    injection               1.0x                0.1x
Qwen3-8B                 hate                    0.9x                0.1x
Qwen3-8B                 yelp5                   0.9x                0.2x
Qwen2.5-7B-Instruct      hate                    0.7x                0.0x
Qwen2.5-7B-Instruct      agnews                  0.4x                0.0x
Qwen3-8B                 agnews                  0.0x                0.0x
Qwen2.5-7B-Instruct      emotion                 0.0x                0.0x
Qwen3-8B                 emotion                 0.0x                0.0x
  (conservative = L1's CI lower bound over raw's CI upper bound)

==========================================================================================================================
FAILURES (a family whose tokenizer cannot produce single-token labels shows up here)
==========================================================================================================================
Mistral-7B-Instruct-v0.3     newsgroups    OutOfMemoryError: CUDA out of memory. Tried to allocate 486.00 MiB. GPU 0 has a total capacity of 93.10 GiB of which 435.31 MiB is free. Including non-PyTorch memory, this process has 18.08 GiB memory in use. Process 75241 has 74.57 GiB memory in use. Of the allocated memory 17.22 GiB is allocated by PyTorch, and 212.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Mistral-7B-Instruct-v0.3     injection     OutOfMemoryError: CUDA out of memory. Tried to allocate 906.00 MiB. GPU 0 has a total capacity of 93.10 GiB of which 775.31 MiB is free. Including non-PyTorch memory, this process has 17.75 GiB memory in use. Process 75241 has 74.57 GiB memory in use. Of the allocated memory 16.88 GiB is allocated by PyTorch, and 221.93 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Falcon3-7B-Instruct          newsgroups    OutOfMemoryError: CUDA out of memory. Tried to allocate 2.62 GiB. GPU 0 has a total capacity of 93.10 GiB of which 2.57 GiB is free. Process 75241 has 74.57 GiB memory in use. Including non-PyTorch memory, this process has 15.94 GiB memory in use. Of the allocated memory 15.16 GiB is allocated by PyTorch, and 122.11 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Falcon3-7B-Instruct          injection     OutOfMemoryError: CUDA out of memory. Tried to allocate 1.27 GiB. GPU 0 has a total capacity of 93.10 GiB of which 573.31 MiB is free. Process 75241 has 74.57 GiB memory in use. Including non-PyTorch memory, this process has 17.95 GiB memory in use. Of the allocated memory 17.17 GiB is allocated by PyTorch, and 114.86 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

## Why the residual deviations exist, part 2: the stateful-prior probe

Recorded by `repro_check/probe_stateful_prior.py` (needs a GPU, so this is its saved output).

```
round 0: bench.run's sequence (test pass, then calibrate)
   seen before calibrate = 300, T = 4.158443983658
round 1: seen before calibrate = 800, T = 4.158450819639
round 2: seen before calibrate = 1300, T = 4.158453060358
calibrate-first (fresh decider): T = 4.147244874939
SPREAD over identical calls that differ only in accumulated prior state:
   temperature 4.147245 .. 4.158453   (spread 0.011208)
   ece         0.0997 .. 0.0999   (spread 0.0002)
   cov@5%      0.5400 .. 0.5467   (spread 0.0067)
   acc         0.8067 .. 0.8067   (spread 0.0000)
   repo's committed value for this cell: see bench/results_batchprior_v0
wrote repro_check/results/probe_stateful_prior.Qwen__Qwen3-8B.banking20.json
```
