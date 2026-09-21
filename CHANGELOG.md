# Changelog

## 0.0.2

- README GIF: `scripts/find_flip_example.py` finds real items where raw readout flips under option reversal and L0 does not; `scripts/make_flip_gif.py` renders them (`assets/flip.gif`, Qwen3-8B on BANKING77). `space/` holds the Hugging Face Space entry point.

- Decider.save_artifacts / load_artifacts: L1 artifacts as one JSON file per model, refused on model mismatch.
- ROADMAP.md, CONTRIBUTING.md, CREDITS.md, GitHub Actions CI (ruff + pytest on 3.10 and 3.12).

- Core: `Question` (choice / score / noul), `Decider`, `Decision` with mandatory level.
- Readout: chat-template prompts, single-token label mapping with collision and multi-token errors.
- L0: cyclic-shift permutation marginalization (log-space combine by default) and label-free prior correction: batch calibration by default (low variance in the bench), content-free contextual calibration opt-in (high variance: large wins on one noul task, losses on ordinal scores).
- Readout: noul label tokens stay bound to their option when the phrasing order is swapped (a position-bound readout silently swapped Yes/No in the second phrasing).
- L1: temperature scaling artifacts keyed by (model, question).
- HFBackend accepts `revision`.
- vLLM backend parity check (`scripts/vllm_parity.py`): 15/15 argmax agreement with transformers on Qwen2.5-7B-Instruct; restricted log-softmax differs by up to 0.6 nats only on labels below 1e-7 probability.
- Backends: transformers (`HFBackend`), vLLM OpenAI-compatible server (`VLLMBackend`, allowed_token_ids + logprobs), synthetic biased model (`FakeBackend`) for tests.
- Question.score accepts explicit ordered `levels`; value is the expected level index.
- Bench: `bench.providers.nanojev_maze` runs AnyJev as the engine inside NanoJev's frozen maze exploration harness; `bench.providers.nanojev_native_maze` reruns their untuned-Qwen baseline for an apples-to-apples row.
- Bench: `bench.run_typed` runs LocalLLaMA/typed-decisions (the set Laya and Jev report on) with overall / per-workflow / per-type metrics; `bench.providers.laya` runs Laya checkpoints on the same decisions.
- Bench: `bench.table` aggregates result JSONs into one table; runner saves after every task and merges per task across invocations.
- Bench: metrics (accuracy, macro-F1, Brier, NLL, ECE, flip rate, coverage-risk), tasks `newsgroups`, `injection`, `banking20`, runner `python -m bench.run`.
