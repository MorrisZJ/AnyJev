# AnyJev

**English** | [简体中文](README.zh-CN.md)

**Turn any LLM into a Jev-style decision model. Typed decisions, real probabilities, no training.**

![Reverse the option order: raw logit readout flips its answer with 1.00 confidence, AnyJev L0 gives the same answer both ways](assets/flip.gif)

*Qwen3-8B, a real BANKING77 item, real outputs. Left: raw next-token readout. Right: AnyJev L0, zero labels. Regenerate with `scripts/find_flip_example.py` and `scripts/make_flip_gif.py`.*

Give it a state and typed questions, get back a decision and a probability per question in one prefill, no generation. Works with the model you already run.

What makes it different from reading logits with `max_tokens=1`:

- **Training-free debiasing on by default (L0).** Cyclic-shift marginalization removes option-order bias; a label-free prior estimate removes the model's label bias. Zero labels.
- **Post-hoc calibration when you have labels (L1).** Temperature scaling per (model, question), stored as a small artifact.
- **Every result says which one you got.** `decision.level` is `raw`, `L0`, or `L1`. Downstream code can refuse to act on the wrong one.
- **A benchmark that reports calibration.** Accuracy, Brier, ECE, order-flip rate, coverage at 5% risk, from one command.

> Not affiliated with, endorsed by, or derived from TypeSafe AI or Jev. All comparisons are measured and reproducible from `bench/results/`.

## Install

```bash
pip install "anyjev[hf]"        # library + transformers backend, from PyPI
pip install anyjev              # library only (numpy); bring your own backend
```

The benchmark is not in the wheel. It needs the datasets, the results directory, and the other projects' code, so it runs from a checkout:

```bash
git clone https://github.com/MorrisZJ/AnyJev && cd AnyJev
pip install -e ".[hf,bench,dev]"   # plus datasets and pytest
```

## Ten lines

```python
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

d = Decider(HFBackend("Qwen/Qwen3-8B"))

route = Question.choice("Which handler should process this request?",
                        ["billing", "technical", "sales", "other"], name="route")
safe  = Question.noul("Is the proposed tool call destructive or irreversible?", name="safe")
done  = Question.score("How complete is the task on a 0 to 1 scale?", bins=5, name="done")

state = {"conversation": [...], "proposed_tool_call": {...}}
r = d.decide(state, [route, safe, done])

r["route"].argmax          # "billing"
r["route"].distribution    # {"billing": 0.81, "technical": 0.07, ...}
r["safe"].p_true           # 0.12
r["done"].value            # 0.35
r.level                    # "L0"  (debiased, not calibrated)
```

With labels:

```python
art = d.calibrate(safe, calib_states, calib_labels)   # ~100 to 500 examples -> L1 artifact
r = d.decide(state, [safe], level="L1")
```

## Why L0 is not optional

A `noul` question, "Is this email spam?". Raw readout on the email gives P(Yes) = 0.62. The same prompt with the email replaced by `N/A` gives P(Yes) = 0.70: the model leans Yes regardless of content. Divide by that prior and renormalize, and the answer is P(Yes) = 0.41. The judgment flips. Position bias does the same thing to `choice` questions when you reorder the options. Both are fixable without training, and AnyJev fixes them by default.

## Levels

Full contract in [docs/levels.md](docs/levels.md).

| Level | Needs | Does | Does not |
|---|---|---|---|
| `raw` | nothing | restricted softmax over label tokens (what the clones do) | anything about bias or calibration |
| `L0` | nothing | removes position bias and label-prior bias | make the model's uncertainty calibrated |
| `L1` | 100 to 500 labels per question | temperature scaling on top of L0 | survive distribution shift beyond the calibration set |

## Benchmark

```bash
python -m bench.run --model Qwen/Qwen3-8B --tasks newsgroups,injection,banking20 --n 300 --calib 200
```

Results land in `bench/results/<date>/` as Markdown and JSON with hardware and library versions. `python -m bench.table bench/results/<date>` regenerates the table below; nothing is typed in by hand.

### Reorder the options and one in five answers changes

Three open models, three tasks, 300 test items each. `flip` is the fraction of items whose answer changes when the option list is reversed (`choice`) or the phrasing is swapped between "Yes or No" and "No or Yes" (`noul`). `raw` is what every logit-reading clone does. `L0` is AnyJev's default, zero labels. `L1` adds one temperature fit on 200 labels.

| model | task | K | raw flip | L0 flip | raw acc | L0 acc | raw ECE | L1 ECE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | banking20 | 20 | 0.227 | 0.077 | 0.750 | 0.807 | 0.235 | 0.100 |
| Qwen3-8B | newsgroups | 20 | 0.237 | 0.173 | 0.640 | 0.660 | 0.331 | 0.157 |
| Qwen3-8B | injection | 2 | 0.060 | 0.000 | 0.693 | 0.710 | 0.287 | 0.162 |
| Qwen2.5-7B-Instruct | banking20 | 20 | 0.197 | 0.067 | 0.723 | 0.767 | 0.236 | 0.072 |
| Qwen2.5-7B-Instruct | newsgroups | 20 | 0.233 | 0.123 | 0.660 | 0.707 | 0.276 | 0.082 |
| Qwen2.5-7B-Instruct | injection | 2 | 0.053 | 0.000 | 0.737 | 0.813 | 0.189 | 0.037 |
| Qwen3-30B-A3B-Instruct-2507 | banking20 | 20 | 0.143 | 0.097 | 0.733 | 0.767 | 0.246 | 0.079 |
| Qwen3-30B-A3B-Instruct-2507 | newsgroups | 20 | 0.133 | 0.100 | 0.730 | 0.740 | 0.249 | 0.086 |
| Qwen3-30B-A3B-Instruct-2507 | injection | 2 | 0.093 | 0.000 | 0.723 | 0.767 | 0.253 | 0.080 |

Full table with every ablation row (permutation only, each prior alone, Brier, coverage at 5% risk): [docs/results_bench.md](docs/results_bench.md). One H100, bf16, transformers 4.55.4; regenerate with `python -m bench.table bench/results_batchprior_v0/2026-09-20`.

### On Laya's own benchmark, zero-shot

| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| laya-multilingual (zero-shot), measured here | 0.340 | 0.325 | 0.287 | 0.269 | 0.688 |
| laya (zero-shot), measured here | 0.359 | 0.331 | 0.177 | 0.227 | 0.694 |
| Qwen2.5-7B-Instruct + raw logits (clone baseline) | 0.621 | 0.514 | 0.287 | 0.209 | 0.437 |
| Qwen3-8B + raw logits (clone baseline) | 0.626 | 0.520 | 0.328 | 0.210 | 0.621 |
| Qwen2.5-7B-Instruct + AnyJev L0, zero-shot | 0.628 | 0.506 | 0.200 | 0.176 | 0.451 |
| Qwen2.5-7B-Instruct + AnyJev L1, temperature from 200 train cases | 0.632 | 0.452 | 0.047 | 0.149 | 0.443 |
| Qwen3-8B + AnyJev L0, zero-shot | 0.640 | 0.523 | 0.273 | 0.196 | 0.617 |
| Qwen3-8B + AnyJev L1, temperature from 200 train cases | 0.646 | 0.457 | 0.056 | 0.143 | 0.474 |
| Qwen3-32B + raw logits (clone baseline) | 0.684 | 0.556 | 0.206 | 0.144 | 0.488 |
| Qwen3-32B + AnyJev L0, zero-shot | 0.700 | 0.548 | 0.133 | 0.128 | 0.456 |
| Qwen3-32B + AnyJev L1, temperature from 200 train cases | 0.701 | 0.502 | 0.034 | 0.120 | 0.412 |
| Jev 1.13.0 (published by TypeSafe / Laya; not rerun) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |
| laya-typed-decisions (fine-tuned on this set's train split), measured here | 0.768 | 0.471 | 0.215 | 0.118 | 0.243 |

Laya's headline is 0.766 on this set, above Jev's 0.727. All rows except Jev were measured here on the same 2,000 decisions; the Laya fine-tuned checkpoint reproduces its published number. Read it two ways. On argmax accuracy, Laya fine-tuned on the train split wins, and a 32B open model with zero training is 2.7 points behind Jev. On the probabilities, which is what a System One model is for, the picture changes. The fine-tuned Laya's ECE (0.215) is six times AnyJev L1's (0.034), and its soft accuracy (0.471) is below every zero-shot Qwen row. Two AnyJev L1 rows are the exception. Temperature scaling trades soft accuracy for calibration, so the 7B and 8B models fall to 0.452 and 0.457 soft accuracy while their Brier improves. On Brier alone the fine-tuned Laya stays narrowly ahead, 0.118 against AnyJev L1's best of 0.120. Laya's zero-shot checkpoints, the ones you would use on a question they were not trained for, score 0.34 to 0.36 against a 0.32 random baseline. Full table with per-workflow and per-type breakdown: [docs/results_typed.md](docs/results_typed.md); regenerate with `python -m bench.run_typed --model <model>` and `python -m bench.providers.laya`.

### Inside NanoJev's maze harness

| engine | goal test | goal ood | attempts | collisions | edge acc | majority | edge Brier | edge questions |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B + AnyJev L0 (batch prior) | 10/11 | 3/4 | 15616 | 6236 | 0.490 | 0.618 | 0.271 | 23156 |
| Qwen3-0.6B + AnyJev L0 (content_free prior) | 10/11 | 4/4 | 16278 | 6563 | 0.490 | 0.614 | 0.274 | 23284 |
| Qwen3-0.6B + AnyJev L0 (none prior) | 10/11 | 3/4 | 21851 | 9136 | 0.403 | 0.597 | 0.460 | 28040 |
| Qwen3-0.6B + AnyJev raw | 11/11 | 4/4 | 5825 | 2616 | 0.537 | 0.539 | 0.363 | 10944 |
| Qwen3-8B + AnyJev L0 (batch prior) | 11/11 | 3/4 | 17841 | 7171 | 0.555 | 0.600 | 0.344 | 25124 |
| Qwen3-0.6B native A/B readout (NanoJev's 'Untuned Qwen' protocol) | 10/11 | 3/4 | 20555 | 8496 | 0.419 | 0.607 | 0.305 | 27660 |

NanoJev's README compares its trained 0.6B model against "Untuned Qwen3-0.6B", which reads A/B logits for four Boolean questions per maze cell. We reimplemented that readout and ran it inside NanoJev's scaled_maze pipeline ([docs/SCALED_GAMES.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/SCALED_GAMES.md)), on its 15 test and out-of-distribution episodes at sizes 8 to 50, with their frozen exploration code and their pinned Qwen3-0.6B revision. Only the engine that answers the Boolean changes. This is not the evaluation behind the 2/10 maze figure in NanoJev's held-out gameplay table. That figure comes from a separate 274-case suite that we did not run, so the rows above neither restate nor contest it. NanoJev publishes no untuned-Qwen baseline on the scaled suite, so the last row is our own reimplementation of their protocol rather than a number of theirs. Two things are true at once. The untuned readout's result depends heavily on how you read it. The same Qwen3-0.6B goes from 13/15 mazes and 20,555 attempts under the A/B readout to 15/15 and 5,825 attempts under AnyJev's raw Yes/No readout. And no LLM readout, not even Qwen3-8B, answers "is one step north clear?" better than always saying the majority label (edge accuracy 0.40 to 0.56 against a 0.54 to 0.62 majority). The raw row comes closest, and at 0.537 against a 0.539 majority it is a tie rather than a win. The maze differences come from how each readout's average probability interacts with the controller's p >= 0.5 probe rule, not from map reading. We report it because it is the comparison NanoJev invites; we do not headline it. Full table: [docs/results_maze.md](docs/results_maze.md).

## Status and roadmap

v0.0.1. The library, both backends, and all three benches are real and measured; the tables above are regenerated from committed JSON. Actively developed: the plan with dates is in [ROADMAP.md](ROADMAP.md) (next up: PyPI release, span readout for more than 26 options, conformal abstention, latency column, a live demo, Llama and Gemma rows; then a Jev-compatible server, more backends, and multimodal state). Backends and bench providers are one file each; see [CONTRIBUTING.md](CONTRIBUTING.md). What landed: [CHANGELOG.md](CHANGELOG.md). Who we build on: [CREDITS.md](CREDITS.md).

## License

Apache-2.0. Datasets keep their own licenses, see `THIRD_PARTY.md`.
