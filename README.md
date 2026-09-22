<div align="center">

<img src="assets/banner.png" width="100%" alt="AnyJev — turn any LLM into a Jev-style decision model. Typed decisions, real probabilities, no training. Order-flip rate 0.230 to 0.073, calibration error 0.240 to 0.095, auto-decidable at 5% risk 7.7% to 52.0%.">

[![PyPI](https://img.shields.io/pypi/v/anyjev?color=3b82f6)](https://pypi.org/project/anyjev/)
[![Python](https://img.shields.io/pypi/pyversions/anyjev)](https://pypi.org/project/anyjev/)
[![CI](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml/badge.svg)](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**English** · [简体中文](README.zh-CN.md) · [Levels contract](docs/levels.md) · [Results](docs/results_bench.md) · [Roadmap](ROADMAP.md)

</div>

<p align="center">
  <b>Jiamu Zhang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Tianze Yang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Yucheng Shi</b><sup>2</sup> &nbsp;&nbsp;&nbsp; <b>Liang Wu</b><sup>1</sup>
</p>
<p align="center">
  <sub><sup>1</sup>&nbsp;Nokia, Sunnyvale, CA &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>&nbsp;Tencent Hunyuan</sub>
</p>

---

![Reverse the option order: raw logit readout flips its answer with 1.00 confidence, AnyJev L0 gives the same answer both ways](assets/flip.gif)

<div align="center">
<sub>Qwen3-8B, a real BANKING77 item, real outputs.</sub><br>
<sub><b>Left:</b> raw next-token readout — reverse the options and the answer flips, at 1.00 confidence.</sub><br>
<sub><b>Right:</b> AnyJev L0, zero labels — same answer both ways.</sub>
</div>

---

## Why not just read the logits?

Give AnyJev a **state** and a set of **typed questions**; get back a decision and a probability per question, read straight off the model's next-token distribution — no generation, no parsing, no fine-tuning, on the model you already run.

That much you can do yourself with `max_tokens=1` plus logprobs. The problem is what you get: a **ranking that moves when you reorder the options**, and a confidence number you **cannot threshold on**. Both are properties of the readout rather than of the model's knowledge, and both are fixable without a single label.

<div align="center">

| | raw logits | **AnyJev L0** | **AnyJev L1** |
|:--|:--:|:--:|:--:|
| Labels required | none | **none** | 100–500 |
| Answer flips when options are reversed | 0.230 | **0.073** | 0.077 |
| Accuracy | 0.747 | **0.803** | 0.807 |
| Calibration error (ECE) | 0.240 | 0.184 | **0.095** |
| **Auto-decidable at ≤5% error** | **7.7%** | **46.3%** | **52.0%** |

<sub>Qwen3-8B, BANKING77 20-way, 300 test items. Full table incl. every ablation: <a href="docs/results_bench.md">docs/results_bench.md</a></sub>

</div>

The last row is the point. Accuracy moves by 6 points, but the share of traffic you can safely automate goes from **7.7% to 52.0%**, a 6.8× difference on this task (a point estimate at n=300; the interval is wide, see Limitations). With raw logits a "0.9" is not trustworthy enough to act on, so everything goes to a human. Once the probability means what it says, you can set a threshold.

> [!NOTE]
> Not affiliated with, endorsed by, or derived from TypeSafe AI or Jev. Every comparison here is measured and reproducible from `bench/`, except rows explicitly marked as published by their authors.

## Quickstart

```bash
pip install "anyjev[hf]"        # library + transformers backend
pip install anyjev              # library only (numpy); bring your own backend
```

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

art = d.calibrate(safe, calib_states, calib_labels)   # 100-500 examples -> L1 artifact
r = d.decide(state, [safe], level="L1")
```

**Images in the state.** Same questions, same levels, a vision-language backend (`pip install "anyjev[vlm]"`, transformers ≥ 4.57 for Qwen3-VL):

```python
from anyjev import Decider, Image, Question
from anyjev.backends.hf_vlm import VLMBackend

d = Decider(VLMBackend("Qwen/Qwen3-VL-2B-Instruct"))

page = Question.choice("What is the user's screen showing?",
                       ["a login form", "a payment page", "an error message", "something else"], name="page")
stuck = Question.noul("Is the user blocked from continuing?", name="stuck")

state = {"screenshot": Image("screenshot.png"), "note": "user says the app is stuck"}
r = d.decide(state, [page, stuck])

r["page"].distribution     # {"an error message": 0.99..., ...}
r["stuck"].p_true
r.level                    # "L0"
```

`Image` takes a path, URL, bytes, or a PIL image, anywhere in the state. Position debiasing carries over unchanged; whether to use the content-free prior depends on the task — contract and caveats in [docs/multimodal.md](docs/multimodal.md).

The benchmark is not in the wheel — it needs the datasets, the results directory, and the other projects' code, so it runs from a checkout:

```bash
git clone https://github.com/nokia-applied-research/AnyJev && cd AnyJev
pip install -e ".[hf,bench,dev]"
```

## How it works

Three typed primitives: `choice` (up to 26 options), `noul` (Yes/No with a real `p_true`), and `score` (2–10 ordinal bins with an expected value). A backend does exactly one thing — return next-token log-probabilities — so adding one is a single file; transformers and vLLM ship today.

Everything above that backend is the two corrections that make the number usable. **Cyclic-shift marginalization** shows a K-option list in K rotations so every option sits at every position once, combined in log space. **Prior correction** estimates the model's label prior without labels and divides it out: a `noul` asking "is this email spam?" reads P(Yes) = 0.62, the same prompt with the body replaced by `N/A` reads P(Yes) = 0.70 — the model leans Yes regardless of content — and dividing by that prior gives 0.41, flipping the judgment.

Full contract in [docs/levels.md](docs/levels.md).

| Level | Needs | Does | Does **not** |
|---|---|---|---|
| `raw` | nothing | restricted softmax over label tokens (what the clones do) | anything about bias or calibration |
| `L0` | nothing | removes position bias and label-prior bias | make the model's uncertainty calibrated |
| `L1` | 100 to 500 labels per question | temperature scaling on top of L0 | survive distribution shift beyond the calibration set |

Every `Decision` carries its `level`, so downstream code can refuse to act on the wrong one.

**Cost.** L0 trades compute for stability: a `choice` with K options costs K prefills (2 for `noul`, 1 for `score`), all sharing the state prefix and all batchable, with nothing ever generated. On one H100 the transformers path at 20 permutations is about **0.25 s per decision at batch 32**. `max_permutations` caps K.

---

## Benchmark

```bash
python -m bench.run --model Qwen/Qwen3-8B --tasks newsgroups,injection,banking20 --n 300 --calib 200
```

Results land in `bench/results/<date>/` as Markdown and JSON with hardware and library versions. Every table and figure below is regenerated from that JSON; **nothing is typed in by hand.**

![Four panels across three open models and three tasks: order-flip rate, expected calibration error, accuracy, and coverage at 5% risk, comparing raw logit readout against AnyJev L0 and L1](assets/results.png)

### Reorder the options and one in five answers changes

`flip` is the fraction of items whose answer changes when the option list is reversed (`choice`) or the Yes/No phrasing order is swapped (`noul`). Three open models, three tasks, 300 test items each.

| model | task | K | raw flip | L0 flip | raw acc | L0 acc | raw ECE | L1 ECE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | banking20 | 20 | 0.230 | **0.073** | 0.747 | **0.803** | 0.240 | **0.095** |
| Qwen3-8B | newsgroups | 20 | 0.233 | **0.177** | 0.637 | **0.660** | 0.334 | **0.138** |
| Qwen3-8B | injection | 2 | 0.060 | **0.000** | 0.693 | **0.700** | 0.288 | **0.161** |
| Qwen2.5-7B-Instruct | banking20 | 20 | 0.197 | **0.080** | 0.723 | **0.757** | 0.237 | **0.070** |
| Qwen2.5-7B-Instruct | newsgroups | 20 | 0.237 | **0.127** | 0.663 | **0.710** | 0.273 | **0.096** |
| Qwen2.5-7B-Instruct | injection | 2 | 0.070 | **0.000** | 0.737 | **0.790** | 0.188 | **0.049** |
| Qwen3-30B-A3B-Instruct-2507 | banking20 | 20 | 0.143 | **0.097** | 0.730 | **0.770** | 0.249 | **0.086** |
| Qwen3-30B-A3B-Instruct-2507 | newsgroups | 20 | 0.140 | **0.093** | 0.737 | **0.740** | 0.242 | **0.096** |
| Qwen3-30B-A3B-Instruct-2507 | injection | 2 | 0.103 | **0.000** | 0.730 | **0.757** | 0.248 | **0.090** |

Every ablation row (permutation only, each prior alone, Brier, coverage at 5% risk): [docs/results_bench.md](docs/results_bench.md). One H100, bf16, transformers 4.55.4.

### On Laya's own benchmark, zero-shot

| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| laya-multilingual (zero-shot), measured here | 0.340 | 0.325 | 0.287 | 0.269 | 0.688 |
| laya (zero-shot), measured here | 0.359 | 0.331 | 0.177 | 0.227 | 0.694 |
| Qwen2.5-7B-Instruct + raw logits (clone baseline) | 0.620 | 0.514 | 0.287 | 0.209 | 0.437 |
| Qwen3-8B + raw logits (clone baseline) | 0.626 | 0.520 | 0.328 | 0.210 | 0.621 |
| Qwen2.5-7B-Instruct + AnyJev L0, zero-shot | 0.628 | 0.512 | 0.234 | 0.188 | 0.439 |
| Qwen2.5-7B-Instruct + AnyJev L1, temperature from 200 train cases | 0.628 | 0.461 | 0.038 | 0.148 | 0.425 |
| Qwen3-8B + AnyJev L0, zero-shot | 0.647 | 0.530 | 0.290 | 0.198 | 0.591 |
| Qwen3-8B + AnyJev L1, temperature from 200 train cases | 0.648 | 0.468 | 0.055 | 0.140 | 0.444 |
| Qwen3-32B + raw logits (clone baseline) | 0.684 | 0.556 | 0.206 | 0.144 | 0.488 |
| Qwen3-32B + AnyJev L1, temperature from 200 train cases | 0.699 | 0.508 | 0.036 | 0.119 | 0.416 |
| Qwen3-32B + AnyJev L0, zero-shot | 0.700 | 0.555 | 0.149 | 0.129 | 0.449 |
| Jev 1.13.0 (published by TypeSafe / Laya; not rerun) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |
| laya-typed-decisions (fine-tuned on this set's train split), measured here | 0.768 | 0.471 | 0.215 | 0.118 | 0.243 |

All rows except Jev were measured here on the same 2,000 decisions; the fine-tuned Laya checkpoint reproduces its published 0.766. **Read it two ways.** On argmax accuracy the fine-tuned Laya wins, and a zero-training 32B open model lands 2.8 points behind Jev. On the probabilities — what a System One model is for — the fine-tuned Laya's ECE (0.215) is **six times** AnyJev L1's (0.036), though it stays narrowly ahead on Brier, 0.118 against 0.119. Temperature scaling trades soft accuracy for calibration, so the 7B and 8B L1 rows drop to about 0.45 there. Laya's zero-shot checkpoints, the ones you would use on a question they were not trained for, score 0.34 to 0.36 against a 0.32 random baseline.

Per-workflow and per-type breakdown: [docs/results_typed.md](docs/results_typed.md).

### Inside NanoJev's maze harness

We reimplemented NanoJev's "Untuned Qwen3-0.6B" A/B readout and swapped only the Boolean engine inside their frozen scaled_maze pipeline. **Two things are true at once.** The untuned readout's score depends heavily on how you read it — the same Qwen3-0.6B goes from 13/15 mazes and 20,500 attempts under the A/B readout to 15/15 and 5,825 under AnyJev's raw Yes/No readout. And no readout, not even Qwen3-8B, beats always answering the majority label on "is one step north clear?" (edge accuracy 0.40 to 0.55 against a 0.54 to 0.61 majority). This is not the evaluation behind the 2/10 figure in their held-out gameplay table, which comes from a separate 274-case suite we did not run. We report it because it is the comparison NanoJev invites; we do not headline it.

Full table and protocol: [docs/results_maze.md](docs/results_maze.md).

### A closed-form head on the same hidden state (preview)

The raw readout is itself a linear head: the `lm_head` rows of the label tokens, applied to the last-position hidden state. `anyjev.heads` fits a different matrix for one question on a small labelled set, in closed form (shrunk LDA, ridge, reduced-rank regression, or the difference of class means), with the layer, the regularisation and the temperature chosen by cross-validation on that set alone. No gradients, no weight updates, seconds on a CPU after the one prefill you already pay for.

| typed-decisions, Qwen3-8B, 200 labels per question, 20 questions × 100 test decisions | acc | ECE | Brier |
|---|---|---|---|
| raw logits | 0.626 | 0.330 | 0.688 |
| AnyJev L0 (permutation) | 0.635 | 0.320 | 0.669 |
| AnyJev L1 (temperature) | 0.626 | 0.174 | 0.482 |
| **closed-form head, chosen per question by cross-validation** | **0.771** | **0.120** | **0.339** |
| laya-typed-decisions, fine-tuned on all 300 train cases per question | 0.768 | 0.215 | — |

Per kind: `choice` 0.60 → 0.75, `noul` 0.71 → 0.85, `score` 0.59 → 0.73; 19 of 20 questions improve. On BANKING77 (K = 20, 200 labels) the ridge head reaches 0.843 against 0.800 for L0 and 0.747 raw, ECE 0.046. Two caveats, both measured. A head fit on one option order is not order-invariant (0.95 of answers flip under reversal; fit on shift-averaged features that drops to 0.11–0.19, at K prefills). And the labels have to come from the task: heads fit on the model's own answers gain nothing, and heads fit on its thinking-mode answers lose accuracy. One model and one seed so far. `python -m bench.heads_study`, `python -m bench.heads_table`; JSON in `bench/results_heads/`. Brier here is the multi-class sum, not the per-option mean of the table above.

### Two games with a built-in oracle

`python -m demo.games.twenty48` and `python -m demo.games.minesweeper` play the same seeds under raw, L0 and L1 and score every decision against an oracle (a depth-2 expectimax; the exact mine posterior). Qwen3-8B on 2048, five games with the board after each legal move shown: score 1,744 raw → 1,982 L0, flip 0.20 → 0.12, ECE 0.46 → 0.33 → 0.08 at L1; random 855, oracle 10,154. On Minesweeper no readout of an 8B or 32B model beats random (board cleared 0.71 against 0.68): the model cannot read the number constraints, and L1 only makes its P(safe) honest (ECE 0.41 → 0.08). Both run on a synthetic biased model without a GPU (`--backend fake`); see [demo/games/README.md](demo/games/README.md).

---

## Reproducibility

An independent rerun of every published cell (3 models × 3 tasks × 2 priors, the maze rows, the Laya rows) came back bit-identical on every zero-label number at the recorded settings, and found one design wart: L1 artifacts used a running prior, so they depended on what the decider had scored before. Fixed: an artifact now freezes the prior it was fit with, and every result JSON records batch size and dtype (bf16 logits move by up to 0.01 with the batch shape). Every table above was regenerated under the fixed code from committed JSON, `bash scripts/regen_docs.sh`.

---

## Limitations

We would rather you find these here than in production.

- **L0 is not a free win on every task.** On Qwen3-8B's prompt-injection split, L0's coverage at 5% risk (0.160) lands *below* raw (0.297). The content-free prior swings hardest: +8 to +12 points on one `noul` task, −3 on ordinal scores, −9 on another model's `noul`. Measure on your task; the bench prints every ablation from the same forward passes, so it costs nothing extra.
- **The batch prior needs a batch.** It activates only after `min_prior_n` (default 8) items of the same question, and assumes the batch's label marginal is not extreme. On questions whose true majority label exceeds about 65 percent it costs accuracy (−0.02 at the default strength 0.75, −0.04 at full strength, over 164 (model, question) points in [docs/when_l0_helps.md](docs/when_l0_helps.md)), and no label-free rule can tell that case from a biased model.
- **Calibration cannot fix a model that cannot answer.** In the maze harness, no readout beats the majority-class baseline on edge perception. AnyJev makes uncertainty *legible*, not smaller.
- **Coverage at 5% risk is a high-variance point estimate** at n = 300: flipping one item moves it by up to 0.013, and its deepest-admissible-prefix definition sits above a first-crossing one by 0.026 on average. Read the 7× as a direction, not a constant.
- **26 options max** in the current letter readout; span readout lifts the cap.
- **L1 does not survive distribution shift** beyond its calibration set, and it reshapes confidence without changing the ranking.
- **One model family in the tables so far.** Everything above is Qwen; Llama and Gemma rows are on the roadmap.

## Status

**v0.0.2.** The library, both backends, and all three benches are real and measured. Actively developed — the plan with dates is in [ROADMAP.md](ROADMAP.md). **Next up:** closed-form heads across models and label counts, span readout beyond 26 options, conformal abstention, Llama and Gemma rows. **After that:** a Jev-compatible HTTP server, more backends. **Landed since v0.0.2:** multimodal state ([docs/multimodal.md](docs/multimodal.md)).

Backends and bench providers are one file each and several are marked **help wanted** — see [CONTRIBUTING.md](CONTRIBUTING.md). What landed: [CHANGELOG.md](CHANGELOG.md). Who we build on: [CREDITS.md](CREDITS.md).

## Citation

```bibtex
@software{anyjev2026,
  title  = {AnyJev: Turn any LLM into a Jev-style decision model},
  author = {Zhang, Jiamu and Yang, Tianze and Shi, Yucheng and Wu, Liang},
  year   = {2026},
  url    = {https://github.com/nokia-applied-research/AnyJev}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE). Datasets keep their own licenses, see [THIRD_PARTY.md](THIRD_PARTY.md).
