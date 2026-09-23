# Working agreement for agents and humans in this repo

## Ground rules

1. **No fabricated data, ever.** Bench numbers come from committed result JSON under `bench/results_*/<date>/` (a run writes to `bench/results/<date>/`; the runs behind the shipped tables are `bench/results_v01/2026-09-22/`, `bench/results_typed_v01/2026-09-22/`, `bench/results_exit/2026-09-22/`, `bench/results_paraphrase/2026-09-22/`, `bench/results_distill/2026-09-22/` and `bench/results_nanojev/2026-09-22/`; `docs/migration_v3.md` lists the rest). If a run did not happen, the cell is empty. Every number in prose names its JSON.
2. **No hidden generation.** Any code path that samples tokens in decision mode is a bug.
3. **Level is mandatory.** Every `Decision` carries `level` (`raw` / `L0` / `L1` / `L2`). `auto` is a request to `decide()`, never a result level. Tests assert it.
4. **Backends are thin.** A backend implements `next_token_logprobs(prompts, token_ids)`; the transformers backend adds the optional methods the Decider probes for (`score_shared` for shared-prefix scoring, `hidden_states` / `hidden_states_to` for L2). Nothing else goes in `anyjev/backends/`: debiasing, calibration, heads, routing and adaptation live above the backend and are tested once.
5. **Small changes.** One issue, one change, under ~400 lines excluding tests and fixtures.
6. **Tests before features.** A calibration method lands with a unit test on synthetic logits where the correct answer is known analytically (see `anyjev/backends/fake.py`, which also plants hidden states for L2), plus one bench regression.
7. **Licenses are checked** before any dataset or third-party code is used. Record it in the task loader and in `THIRD_PARTY.md`.
8. **No names we do not own** in identifiers, package names, or API paths beyond the project name itself. Attribution lives in docs.
9. **Git is the maintainer's.** Agents do not run `git add`, `git commit`, or `git push`. Leave the working tree for the maintainer to review and commit.

## Definition of done

- Code + tests + docstring + one line in `CHANGELOG.md`.
- Bench changes: results JSON regenerated and committed, with hardware and model versions (`bench/run.py` records them, including batch size and dtype), and the docs regenerated with `bash scripts/regen_docs.sh` (plus, for `docs/results_adaptive.md` and `docs/results_latency.md`, the regeneration command each of those two names).
- A new model row for L2: `bench/extract_pools.py`, `bench.exit_study`, `bench.jev_mode_table`, `bench.exit_latency`, `scripts/build_heads.py`, and the artifact under `anyjev-heads/` (`docs/method_v3.md` section 8).
- Backends: a smoke test against a real engine, marked `@pytest.mark.engine`, skipped in CI without the engine.

## Environment

- Python 3.10+. `pip install -e ".[dev]"` for the core; `.[hf,bench]` for real models and datasets (`transformers>=4.53` for the L2 block loop).
- `ruff check anyjev bench demo handoff scripts space tests && pytest -q` is what CI runs; both must be green on CPU with numpy alone.
- Bench runs record `nvidia-smi`, torch, and transformers versions. Do not mix hardware within one results table.

## Ask a human about

- Any change to the public HTTP schema (when the server exists).
- Any new dataset.
- Any claim in docs that compares us to a named product.
- Any dependency with a non-permissive license.
