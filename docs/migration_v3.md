# Migration note: the version-3 tree (0.1.0)

What changed between the research tree behind `docs/research_log.md` and this one. The public API only
grew (`fit_head`, `route`, `adapt`, `level="auto"`, `observe` / `observations`, `export_artifacts` /
`load_artifacts` carrying the adaptation statistics and, on request, the observations); artifacts written by
0.0.2 still load. Everything removed below was either a closed negative result or an internal draft.

## Code removed

| removed | what it was | where the record is |
|---|---|---|
| `anyjev/universal.py` | question-agnostic heads (six functional forms) | research log entries 1, 3, 9, 11; JSON under `bench/results_universal/` |
| `anyjev/cascade.py` | confidence-gated early exit over heads at several depths | research log entries 4, 10 (tables); JSON `bench/results_exit/2026-09-22/<model>.cascade.json` (see below) |
| `bench/universal_study.py`, `bench/universal_e0.py` | the LOQO / cross-domain protocols and the phase-0 kill tests | entries 0, 1, 3, 9, 11 |
| `bench/cascade_study.py` | the cascade protocol | entries 4, 10 |
| `bench/heads_study.py`, `bench/heads_table.py` | the first per-question head study (solver kinds, self-labels) | entry 0; JSON `bench/results_heads/2026-09-22/Qwen__Qwen3-8B.typed.json` |
| `bench/distill_study.py` | heads on a 32B teacher's zero-label answers | entry 7; JSON `bench/results_distill/2026-09-22/Qwen__Qwen3-8B.from.Qwen__Qwen3-32B.json` |
| `tests/test_universal.py` (13 cases), `tests/test_cascade.py` (4) | tests of the removed modules | three extraction tests carried into `tests/test_pools.py` |
| `bench/tasks/extra.py`, `bench/tasks/games.py` | seven research tasks and the game oracle pools, registered only for the universal study | entry 9 (cross-domain rows) |
| `docs/launch-kit.md`, `docs/post-draft.md`, `docs/pr-drafts.md`, `docs/plan-v0.1.md`, the empty `docs/method_v3/` | internal drafts | none needed |

The universal head, the cascade and the zero-label-teacher distillation were tried, measured, and closed
(`ROADMAP.md`, "Closed"); their JSON is kept as the record, their code is not shipped.

## Where the shared machinery went

The block-loop feature caches that the depth, Jev-mode, label-efficiency, listing-order and distillation
studies replay were written by the universal study. That part was carried over unchanged:

| symbol | from | to |
|---|---|---|
| `DecisionPool`, `listing_perms`, `build_rows`, `extract_pool` | `anyjev/universal.py` | `bench/pools.py` (the library no longer ships them) |
| `pool_path`, `save_pool`, `load_pool`, `get_pool`, `typed_questions`, `decisions_to_arrays`, `typed_records`, `task_records` | `bench/universal_study.py` | `bench/pools.py` |
| `per_question_probs` | `bench/cascade_study.py` | `bench/distill_heads.py` |

Renamed command (same npz files, same names, byte-identical caches; existing caches are reused, never
overwritten):

```
# before
python -m bench.universal_study --model Qwen/Qwen3-4B --protocol loqo --use-loop --extract-only --out bench/results_exit --layers 12,14,...,36
# now
python -m bench.extract_pools --model Qwen/Qwen3-4B --layers 12,14,16,18,20,22,24,26,28,30,32,34,36 --out bench/results_exit
```

`bench.exit_study`, `bench.jev_mode_table`, `bench.labels_study`, `bench.paraphrase_study`, `bench.distill_heads`
and `scripts/build_heads.py` import from `bench.pools`; the `--universal` / `--lams` options of `bench.exit_study`
are gone (they never produced a shipped number).

## Result directories dropped

Cited by no shipped doc, table generator or README:

- `bench/results_v01/2026-09-21/`, `bench/results_typed_v01/2026-09-21/`, `bench/results_nanojev/2026-09-21/`: the
  runs before the reproduction fixes (running L1 prior, no `batch_size` / `dtype` in `env`). The 2026-09-22 runs in
  the same directories are the ones every table uses; the four numbers the README's reproducibility paragraph
  quotes from the earlier runs (banking20 raw flip 0.227; maze L0 15,616 attempts; L1 coverage at 5% risk on
  newsgroups 0.42 on Qwen3-8B and 0.37 on Qwen3-30B-A3B, against 0.237 and 0.003 in the 2026-09-22 runs) are
  stated there as coming from these dropped files.
- `bench/results_cf/2026-09-20/`: an early content-free-prior run superseded by the `L0-perm+cf` rows of
  `bench/results_v01/2026-09-22/`.
- `bench/results_distill/labels/Qwen__Qwen3-32B/`: 23 npz files written by the removed `bench.distill_study label`.
- `handoff/games/superseded/`: the first maze design's results and a style preview that was never README material.

Kept (each with its citer): `bench/results_v01/2026-09-22`, `results_typed_v01/2026-09-22`, `results_typed`
(Laya rows), `results_exit/2026-09-22` (including `Qwen__Qwen3-8B.cascade.json` and `Qwen__Qwen3-32B.cascade.json`,
research log entries 4 and 10), `results_paraphrase`, `results_distill/2026-09-22` + `synth` + `synth_labels`,
`results_nanojev/2026-09-22`, `results_universal` and `results_heads` (kill-test records), `results_small`,
`results_typed_small`, `results_batchprior_v0`, `results_typed_diag` (`docs/when_l0_helps.md`, `bench.diag_l0`,
`bench.models_table`), `results_dump` and `results_typed_dump` (`bench.prior_study`), the four `results_adaptive_*`
(`docs/results_adaptive.md`), `results_latency*` (`docs/results_latency.md`).

Every result JSON left in the tree carries `batch_size`, dtype, backend, versions and the git commit in `env`.

## Artifact format

`anyjev-heads/<model>.json` now stores the head arrays as base64 float32 (`anyjev.heads.encode_array`; exact,
about ten times smaller than number lists): 1.8 / 2.2 / 3.5 / 1.8 / 4.4 MB for the 1.7B / 4B / 8B / 30B-A3B / 32B,
13.6 MB in total, instead of 7-18 MB each. `LinearHead.from_dict` and `Decider.load_artifacts` read both the compact
and the list format; `scripts/compact_heads.py` converts a file in place after checking the round trip is bit-exact
(`tests/test_heads.py`).

## The cascade JSON

`bench/results_exit/2026-09-22/Qwen__Qwen3-8B.cascade.json` and `Qwen__Qwen3-32B.cascade.json`, the result files of
research log entries 4 and 10, are kept in the tree (copied from the GPU host after the code was removed). The two
tables in those entries were re-derived from them cell by cell, and `docs/jev_mode.md`, `docs/method_v3.md`
section 2.5 and `ROADMAP.md` quote them. The study code (`anyjev/cascade.py`, `bench/cascade_study.py`,
`tests/test_cascade.py`) is not shipped: the conclusion (a cascade did not beat one fixed block on either model)
is a closed result, and the JSON is its record.

## Other small changes

- `.gitignore`: the `bench/results_heads/features/` and `bench/results_universal/features/` lines went with the
  studies that wrote those caches; `bench/results_exit/features/`, `bench/results_paraphrase/features/` and
  `bench/results_distill/synth_features/` stay (uncommitted caches on the GPU host, rebuilt by `bench.extract_pools`).
- `scripts/regen_docs.sh` feeds `bench.diag_l0` the directories `docs/when_l0_helps.md` names
  (`results_typed_diag`, `results_batchprior_v0`, `results_small`), and `docs/diag_l0_output.txt` was regenerated
  from them (200 typed + 30 bench points).
- The offline prior study was replayed on the committed dumps: 230 (model, question) units, not the 164 the
  0.0.2-era docs quoted; `docs/when_l0_helps.md` carries the regenerated table.
- `pyproject.toml`: version 0.1.0; the `hf` extra needs `transformers>=4.53` (the L2 block loop uses
  `transformers.masking_utils`).
- CI lints `demo/`, `scripts/`, `handoff/` and `space/` as well: `ruff check anyjev bench demo handoff scripts space tests`.

## README claims corrected in this pass

Each was checked against the JSON it now names: the flip.gif caption (0.999 / 0.82, not "1.00 confidence"); the
cost sentence (0.25 s per decision is the K=20, ~110-token case); the Laya paragraph (Brier compares our metric
across rows, 2.7-2.8 points behind Jev at L0 / L1, 0.46-0.47 soft accuracy, the 0.3175 random baseline); the
NanoJev sentence (the held-out gameplay table is not reproduced here; the raw readout says Yes 85% of the time);
`fit_head` timing (2-8 s per question on the 1.7B-8B, prefill included); the L1 ECE range (0.036-0.055); the
wording-adaptation numbers (0.74-0.75 with 30 unlabelled states); the 2048 and Minesweeper sentences (L1 score
1,518; no ECE 0.41 -> 0.08); the injection coverage numbers (0.013 vs 0.313); the content-free swing (+7 to +12
over raw on the injection `noul`); the coverage-variance bullet (no unsupported 0.013 / 0.026); the "one model
family" bullet (five other architectures are measured at raw / L0 / L1).
