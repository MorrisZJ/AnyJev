"""Assemble repro_check/REPORT.md from everything the pipeline produced.

    python repro_check/make_report.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SECTIONS = [
    ("Headline: reproduction rate, stratified by whether the rerun matched the committed config",
     "summary.out"),
    ("Offline verification (no GPU): the paper's analytical claims and the docs' internal consistency",
     "check_offline.out"),
    ("Reproduction diff: bench.run, both prior configurations", "compare.out"),
    ("Reproduction diff: bench.run_typed on LocalLLaMA/typed-decisions", "compare_typed.out"),
    ("Reproduction diff: the NanoJev maze harness", "compare_maze.out"),
    ("Why the residual deviations exist, part 1: the batch-size probe", "probe_batchsize.out"),
    ("Extensions: new tasks, new model families, bootstrap intervals", "analyze.out"),
]

FINDINGS = """
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
"""


def run(script: str) -> str:
    p = os.path.join(HERE, script)
    if not os.path.exists(p):
        return f"(missing {script})"
    out = subprocess.run(["python3", p], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, HF_HOME="/mnt/persist/hf-cache"))
    return out.stdout + (("\nSTDERR:\n" + out.stderr[-2000:]) if out.returncode else "")


def stateful_section() -> str:
    """The stateful-prior probe needs a GPU, so replay what it recorded rather
    than rerunning it at report time."""
    log = os.path.join(HERE, "logs", "probe_stateful_prior.log")
    if not os.path.exists(log):
        return "(probe not run)"
    keep = [ln for ln in open(log).read().splitlines()
            if ln.strip() and "it/s]" not in ln and "Loading checkpoint" not in ln]
    return "\n".join(keep)


def job_table() -> str:
    p = os.path.join(HERE, "results", "job_state.json")
    if not os.path.exists(p):
        return "(no job state recorded)"
    state = json.load(open(p))
    lines = ["| job | gpu | exit | minutes |", "|---|---|---|---|"]
    for name, s in sorted(state.items()):
        status = "ok" if s["rc"] == 0 else f"**rc={s['rc']}**"
        lines.append(f"| `{name}` | {s['gpu']} | {status} | {s['seconds'] / 60:.1f} |")
    return "\n".join(lines)


def cells_in(path: str):
    """Every (identity, unit-of-work) pair a result JSON stands for."""
    try:
        r = json.load(open(path))
    except Exception:
        return []
    if "tasks" in r:  # bench.run
        prior = r.get("prior", "batch")
        return [(r["model"], f"{t['task']} prior={prior}") for t in r["tasks"] if "error" not in t]
    if "checkpoint" in r:  # bench.providers.laya
        return [(r["checkpoint"], "typed-decisions")]
    if "levels" in r:  # bench.run_typed
        return [(r["model"], "typed-decisions")]
    if "summary" in r:  # the maze providers
        tag = "maze native_AB" if r.get("engine") == "nanojev_native" \
            else f"maze {r.get('level')} prior={r.get('prior')}"
        return [(r.get("model", "?"), tag)]
    return []


def coverage_summary() -> str:
    """Which of the repo's published cells did we manage to rerun?"""
    committed = {}
    for p in glob.glob(os.path.join(ROOT, "bench/results*/*/*.json")):
        fam = p.split("bench/")[1].split("/")[0]
        committed.setdefault(fam, set()).update(cells_in(p))

    mine = set()
    for p in glob.glob(os.path.join(HERE, "results", "**", "*.json"), recursive=True):
        if os.path.basename(p).startswith(("job_state", "compare_summary", "offline_checks", "probe_")):
            continue
        mine.update(cells_in(p))

    lines = ["| committed result family | published cells | reran here | still missing |",
             "|---|---|---|---|"]
    tot = done_tot = 0
    for fam in sorted(committed):
        cells = committed[fam]
        done = cells & mine
        tot += len(cells)
        done_tot += len(done)
        missing = sorted(f"{m.split('/')[-1]} {t}" for m, t in (cells - done))
        lines.append(f"| `bench/{fam}` | {len(cells)} | {len(done)} | "
                     f"{', '.join(missing) if missing else '**none**'} |")
    lines.append(f"| **total** | **{tot}** | **{done_tot}** | "
                 f"**{100 * done_tot / tot:.0f}% of published cells rerun** |")
    return "\n".join(lines)


def main():
    parts = [
        "# AnyJev reproduction report",
        "",
        f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by `repro_check/make_report.py`.",
        "",
        "Everything here was produced by rerunning the repo's own entry points (`bench.run`, "
        "`bench.run_typed`) with the exact parameters recorded in the committed result JSON, on "
        "the same library versions (transformers 4.55.4, torch 2.5.1+cu124) and the same class of "
        "hardware (H100). No file outside `repro_check/` was modified; the repo's bench was pointed at "
        "`--out repro_check/results/...`.",
        FINDINGS,
        "## What was rerun",
        "",
        coverage_summary(),
        "",
        "## Job log",
        "",
        "The scheduler in `repro_check/verify_all.py` only tracks the jobs it launched itself. Several "
        "runs were launched beside it and are not in this table:",
        "",
        "- The two `rc=-15` rows are the Qwen3-30B-A3B runs I killed on purpose. They had been "
        "given `--batch-size 8` to be safe on memory, which is both ~4x slower and, per the "
        "batch-size probe below, guaranteed not to land on the committed numbers. They were "
        "relaunched at `--batch-size 32` by `repro_check/run_30b.sh` and completed; those are the "
        "Qwen3-30B-A3B results compared in this report.",
        "- `repro_check/run_missing.sh` backfilled the two Qwen2.5-7B-Instruct cells that a skip-logic "
        "bug of mine dropped, then ran the stateful-prior probe.",
        "- `repro_check/run_providers.sh` ran the three Laya checkpoints; `repro_check/run_maze.sh` ran all "
        "seven maze rows.",
        "- Mistral-7B and Falcon3-7B initially hit CUDA OOM on two tasks each while sharing a GPU "
        "with the 30B job; both were rerun to completion on dedicated GPUs.",
        "",
        job_table(),
        "",
    ]
    for title, outfile in SECTIONS:
        script = outfile.replace(".out", ".py")
        body = run(script)
        parts += [f"## {title}", "", "```", body.rstrip(), "```", ""]
    parts += ["## Why the residual deviations exist, part 2: the stateful-prior probe", "",
              "Recorded by `repro_check/probe_stateful_prior.py` (needs a GPU, so this is its saved output).",
              "", "```", stateful_section().rstrip(), "```", ""]
    path = os.path.join(HERE, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
