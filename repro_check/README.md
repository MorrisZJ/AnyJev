# repro_check

An independent reproduction of every published result in this repo, plus a few
extensions. Nothing outside this directory was modified; the repo's own
entry points were rerun unchanged with `--out repro_check/results/...`.

**Start with [REPORT.md](REPORT.md).** It opens with the verdict and then carries
the full diff of every rerun cell against the committed JSON.

The short version: all 31 published result cells were rerun, and every
zero-label number -- `raw` plus all six L0 ablations, 648 cells across 3 models
x 3 tasks x 2 priors -- comes back bit-identical. Three things did not line up,
none of them a methodological error: L1 under `prior=batch` drifts because the
batch prior is a running accumulator, `batch_size` changes the numbers but is
not recorded in the result artifacts, and two maze summary fields in the
committed JSON cannot be produced by the public NanoJev checkout.

## Layout

| file | what it does |
|---|---|
| `REPORT.md` | the assembled report; regenerate with `make_report.py` |
| `check_offline.py` | 30 checks with no GPU: the debiasing maths, the metric implementations, and whether the committed JSON matches the README tables |
| `summary.py` | the headline reproduction rate, stratified by whether the rerun matched the committed config |
| `compare.py` | diff of `bench.run` reruns against `bench/results_batchprior_v0` and `bench/results_cf` |
| `compare_typed.py` | diff of `bench.run_typed` and the Laya provider rows against `bench/results_typed` |
| `compare_maze.py` | diff of the maze rows against `bench/results_nanojev`, and re-derivation of the two maze claims |
| `probe_batchsize.py` | why the residual deviations exist: they track `--batch-size`, not the method |
| `probe_stateful_prior.py` | why L1 drifts under `prior=batch`: the fitted temperature depends on how many items the decider has already seen |
| `analyze.py` | the extensions: added tasks, non-Qwen families, bootstrap intervals on cov@5% |
| `verify_all.py` | the scheduler that ran phases 0-4 across four GPUs |
| `extra_tasks/` | benchmark tasks the repo's own bench does not cover, including a `score` task |

## Rerunning

Offline checks need nothing but the repo:

```bash
python repro_check/check_offline.py
```

The full GPU sweep, which downloads the models it needs:

```bash
python repro_check/verify_all.py --gpus 0,1,2,3 --phases 0,1,2,3
bash   repro_check/run_providers.sh 0    # the three Laya checkpoints
bash   repro_check/run_maze.sh 0         # clones NanoJev, fetches their episodes
python repro_check/make_report.py
```

Pin `transformers==4.55.4` to match the committed `env` block. Pass the batch
size the committed run used or the numbers will not land exactly: 32 for
`bench.run` throughout and for `bench.run_typed` on the 7-8B models, 16 for
`bench.run_typed` on Qwen3-32B, and `--batch-states 2` for the native maze row.

`results/` here holds the JSON this report is built from. The maze JSONs
(~10 MB each, one per-episode observation record apiece), the per-item
probability dumps and the NanoJev checkout are gitignored; the commands above
regenerate them.
