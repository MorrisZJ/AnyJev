# Changelog

## Unreleased

- Multimodal state: a state can carry images next to its text (`anyjev.Image` from a path, URL, `data:` URI, bytes, or PIL image). `split_state` pulls them out in traversal order and leaves `<image i>` markers; the readout splits the user turn at those markers so a picture lands where it sat in the state. Typed questions, the readout, L0 and L1 are unchanged, and text-only states render exactly as before. Contract: [docs/multimodal.md](docs/multimodal.md).
- The content-free prior blanks every modality, not just the text: each picture is replaced by a flat grey image, so the probe measures a label prior rather than the model's honest reading of a real picture. Cached per `(question, image count)` and shared across states.
- Backend contract grows an optional `images` argument and two optional attributes (`accepts_images`, `chat_renderer`); text-only backends are untouched, and a state with images against a text-only backend raises instead of silently dropping the picture.
- `VLMBackend` (transformers `AutoModelForImageTextToText`, needs transformers >= 4.57 for Qwen3-VL). It does **not** pass explicit `position_ids`: vision models derive them from the image grid, and the text backend's left-padding fix would corrupt them. Batch parity on Qwen3-VL-2B-Instruct at batch 1 vs 8 with varying image sizes and prompt lengths: answer distributions agree to 5.6e-10, argmax 9/9, log-probs differ by up to 0.25 nats only on labels below 1e-9 probability.
- The images handed to a backend are ordered by the placeholders the prompt actually rendered (`readout.prompt_images`), each placed once at its first marker, so a literal "<image 2>" in the user's own text or an MMMU-style question can never mis-pair pictures and placeholders. Path images are read eagerly and do not hold file handles.
- `FakeBackend` goes multimodal with `image_content`, so the image path is unit-tested without decoding a pixel or downloading a model.
- `bench.run --backend vlm --max-pixels N --model-path P`; results record `images_per_item`, `batch_size`, `max_pixels`, and the prior per task (runs of one model with different priors merge into one file).
- Multimodal bench tasks, built to mirror the text ones: `pets20` (Oxford-IIIT Pet, 20 breeds, one fixed label set — the image-side `banking20`) and `pope` (object-hallucination `noul`, the image-side `injection`; the probed object moves from the question into the state so one question covers the set). `ai2d` as a per-item 4-way MCQ through the new `bench.run_mcq`, for sets where every item carries its own options; there the batch prior has one state per question to pool over, so L0 is permutation-only, and the content-free prior stays off because it cost 8 to 24 accuracy points. Reported like the text bench: L0 is the library default (`bench/results_mm`, `bench/results_mcq`), the content-free prior is an ablation row, and POPE with the content-free prior as L0 lives in `bench/results_mm_cf` as `bench/results_cf` does for text. `bench/run_mm.sh` regenerates all of it. Licenses and the candidates rejected on license are in `THIRD_PARTY.md`.
- `scripts/smoke_vlm.py` and `tests/test_vlm_engine.py` (`pytest -m engine`, needs `ANYJEV_VLM_MODEL`): real-model smoke on colour swatches plus the batch-parity check.
- README quickstart for images (verified verbatim on Qwen3-VL-2B-Instruct). `pillow` joins the `dev` extra: the multimodal unit tests need it, and CI installs only `.[dev]`.

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
