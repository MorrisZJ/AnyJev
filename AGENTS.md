# Working agreement for agents and humans in this repo

## Ground rules

1. **No fabricated data, ever.** Bench numbers come from runs logged under `bench/results/<date>/`. If a run did not happen, the cell is empty.
2. **No hidden generation.** Any code path that samples tokens in decision mode is a bug.
3. **Level is mandatory.** Every `Decision` carries `level` (`raw` / `L0` / `L1`). Tests assert it.
4. **Backends are thin.** A backend implements `next_token_logprobs(prompts, token_ids)` and nothing else. If logic is not backend-specific, it does not live in `anyjev/backends/`.
5. **Small changes.** One issue, one change, under ~400 lines excluding tests and fixtures.
6. **Tests before features.** A calibration method lands with a unit test on synthetic logits where the correct answer is known analytically (see `anyjev/backends/fake.py`), plus one bench regression.
7. **Licenses are checked** before any dataset or third-party code is used. Record it in the task loader and in `THIRD_PARTY.md`.
8. **No names we do not own** in identifiers, package names, or API paths beyond the project name itself. Attribution lives in docs.
9. **Git is the maintainer's.** Agents do not run `git add`, `git commit`, or `git push`. Leave the working tree for the maintainer to review and commit.

## Definition of done

- Code + tests + docstring + one line in `CHANGELOG.md`.
- Bench changes: results JSON regenerated and committed, with hardware and model versions (`bench/run.py` records them).
- Backends: a smoke test against a real engine, marked `@pytest.mark.engine`, skipped in CI without the engine.

## Environment

- Python 3.10+. `pip install -e ".[dev]"` for the core; `.[hf,bench]` for real models and datasets.
- Bench runs record `nvidia-smi`, torch, and transformers versions. Do not mix hardware within one results table.

## Ask a human about

- Any change to the public HTTP schema (when the server exists).
- Any new dataset.
- Any claim in docs that compares us to a named product.
- Any dependency with a non-permissive license.
