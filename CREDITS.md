# Credits

AnyJev stands on other people's work. Attribution lives here, not in identifiers.

## Projects we run, adapt to, or compare against

- **TypeSafe AI, Jev** — the product that defined the interface (state, typed questions, calibrated answers in one pass). Not affiliated; their published numbers appear in our tables clearly marked as published, never as measured by us.
- **Laya** (NandhaKishorM/laya, Apache-2.0) — open trained decision models; we run their three checkpoints through their own `predict` API on LocalLLaMA/typed-decisions (`bench/providers/laya.py`).
- **NanoJev** (TianyuCodings/NanoJev, MIT) — nano replica with games; we run their frozen maze exploration code and their own untuned-Qwen baseline script (`bench/providers/nanojev_maze.py`, `bench/providers/nanojev_native_maze.py`) and their released episode data.
- **LocalLLaMA/typed-decisions** (Apache-2.0) — the 400-case, 2,000-decision workflow set with teacher soft labels.
- The open readout clones that documented the problem first: SemIf, LitJev, the OpenJev servers, poorjev, open-llm-classifier. Their READMEs said the probabilities were not calibrated and the order mattered; this repo measures it.

## Reported and contributed

People outside the project whose reports and patches changed the code. Issue numbers are in the
CHANGELOG next to what they fixed.

- **[@efronh](https://github.com/efronh)** — found that L2 was broken on transformers 5 and
  identified the renamed `create_causal_mask` argument in the report (#4).

## Methods implemented

- Contextual calibration: Zhao et al., "Calibrate Before Use", ICML 2021, arXiv:2102.09690
- Batch calibration: Zhou et al., "Batch Calibration", ICLR 2024, arXiv:2309.17249
- Permutation debiasing: Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024, arXiv:2309.03882
- Temperature scaling: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017, arXiv:1706.04599
- Surface-form competition (planned span readout): Holtzman et al., EMNLP 2021, arXiv:2104.08315
- Conformal prediction (planned): Angelopoulos and Bates, 2021, arXiv:2107.07511
- Reliability gating (planned): Chen et al., "Routing Without Training: Controllable-Ratio LLM Offloading via Reliability Gating", 2026, arXiv:2607.20481

## Datasets

See `THIRD_PARTY.md` for every dataset and its license.
