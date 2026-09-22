# Roadmap

AnyJev is under active development. This file is the plan; `CHANGELOG.md` is what actually landed. Dates are targets, not promises. Anything marked **help wanted** is a good first PR; each one is a single file.

## Where we are (v0.0.2, 2026-09-21)

- Library: choice / noul / score, one prefill, no generation. L0 debiasing on by default (cyclic-shift marginalization, batch prior), L1 temperature scaling, every result carries its level.
- Backends: transformers, vLLM (OpenAI-compatible server, `allowed_token_ids`).
- Bench: three tasks with ablations; LocalLLaMA/typed-decisions with Laya's three checkpoints rerun on the same decisions; NanoJev's maze harness with their own baseline rerun.
- Measured, not claimed: [docs/results_bench.md](docs/results_bench.md), [docs/results_typed.md](docs/results_typed.md), [docs/results_maze.md](docs/results_maze.md).

## Next (v0.1, target week of 2026-09-28)

1. **`pip install anyjev`** on PyPI, with a CPU-only extra so the tests and the fake backend run anywhere.
2. **Span readout for more than 26 options.** Score each option string under the prompt (teacher-forced), which also gives exact PMI correction for multi-word labels and lifts the letter-label cap. Needed for the full 77-way BANKING77 and for Jev's 255-option Choice.
3. **Conformal abstention (L1).** Split-conformal threshold with a user-set target error rate; `decision.abstained` plus a coverage-risk curve in the bench.
4. **Latency and throughput column.** p50 / p95 per decision for the transformers and vLLM backends, with prefix caching on, on a documented GPU. Today the HF path at 20 permutations is about 0.25 s per decision at batch 32 on one H100; the vLLM path is the fast one and has no published number yet.
5. **Live demo.** The Gradio flip demo (`demo/app.py`) as a Hugging Face Space, and a GIF of it at the top of the README.
6. **Llama and Gemma rows** in every table, so the claims are not one model family.

## After that (v0.2, October)

- **Jev-compatible HTTP server.** `POST /v1/decisions` mirroring the OpenRouter decisions schema, with a recorded-fixture conformance suite. Compatibility is advertised only when the suite passes; requests with a chat-completions body get a 400 and a pointer.
- **Backends: SGLang, llama.cpp, MLX, Ollama** (**help wanted**, one file each under `anyjev/backends/`; the contract is `next_token_logprobs(prompts, token_ids)` and nothing else).
- **Bench providers for other open decision models** (**help wanted**, one file each under `bench/providers/`): kev, minojev, openJev Verdict, von, reflex, SemIf. Same 2,000 decisions, same metrics.
- **Dirichlet calibration and histogram binning** as L1 alternatives to temperature scaling; per-question artifact files that can be shipped with a model.
- **Multilingual slices** in the bench (MASSIVE intents, Chinese and Korean where the datasets allow).
- **Batch-prior strength.** The batch prior over-corrects when the label marginal is skewed (seen on the maze). A shrinkage knob, and a way to estimate it without labels, is the open research item.
- **Multimodal state** — *images landed early, see [docs/multimodal.md](docs/multimodal.md); transformers backend only so far, vLLM multimodal, video and audio still open.* A state can carry images (and later video frames or audio) next to text; the typed question and the readout do not change, only the prompt builder and the backend. Vision-language backends for transformers (`AutoModelForImageTextToText`: Qwen2.5-VL / Qwen3-VL, Gemma 3, LLaVA) and vLLM's multimodal server. Same L0 debiasing, same L1 artifacts, same level field. The open vision decision models (Laya Vision, PlayJev) become bench providers on the same inputs.

## Later (v0.3)

- **L2 recipe:** a tiny readout head or LoRA on the answer position, trained with a proper scoring rule, minutes on one GPU, with the bench showing what it buys over L1 and where it stops generalizing.
- **Quantization vs calibration.** Decision readout is prefill-only and compute-bound, the regime where FP8 / NVFP4 / W4A4 pay off; measure ECE drift per precision and whether a per-precision temperature recovers it.
- **Chained decisions.** A 95 percent judge applied 20 times is 36 percent end to end; with calibrated per-step probabilities the compounded risk is computable. A chained-task bench.
- **Multimodal bench.** Image-conditioned typed decisions with ground truth: game frames (NanoJev's ViZDoom and PlayJev-style frames, "which action / is the target in view"), document screenshots (invoice fields, UI states for agent gating), and a safety gate on images. Order-flip and calibration reported exactly as for text; the question is whether position and label-prior bias look the same when the state is pixels.

## Non-goals

No base-model training, no prompt-compilation framework, no routing or gateway logic. Those belong upstream or downstream of a decision layer.

## How to help

Open an issue with a task, a backend, or a model you want in the table. Every table in this repo is regenerated by one command from committed JSON; a PR that adds a row is a PR that adds a JSON file and the code that produced it. See `CONTRIBUTING.md`.
