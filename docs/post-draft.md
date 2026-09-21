# The model you already run is a Jev. Here is what it gets right, what it gets wrong, and the zero-label fix.

*Draft. Every table is regenerated from `bench/results*/`; nothing is typed in by hand.*

A week after TypeSafe launched Jev, there are two kinds of open alternatives. Trained ones (Laya, NanoJev, kev, minojev) ship a small model with decision heads. Readout ones (SemIf, LitJev, the OpenJev servers) point at an open LLM and read the next-token logits for the option labels. The trained ones publish a headline number against Jev. The readout ones publish a README that says the probabilities are not calibrated and the answer changes when you reorder the options.

AnyJev is a readout layer that fixes the second sentence and measures the first. This post is the measurements.

## 1. Reorder the options, one in five answers changes

| model | task | K | n | level | acc | brier | ece | flip | cov@5% |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | banking20 | 20 | 300 | raw | 0.723 | 0.500 | 0.236 | 0.197 | 0.187 |
|  |  |  |  | L0-bc | 0.740 | 0.476 | 0.221 | 0.153 | 0.203 |
|  |  |  |  | L0-perm | 0.750 | 0.445 | 0.216 | 0.073 | 0.150 |
|  |  |  |  | L0-perm+cf | 0.753 | 0.441 | 0.207 | 0.083 | 0.070 |
|  |  |  |  | L0 | 0.767 | 0.420 | 0.197 | 0.067 | 0.173 |
|  |  |  |  | L1 | 0.767 | 0.341 | 0.072 | 0.067 | 0.220 |
| Qwen2.5-7B-Instruct | injection | 2 | 300 | raw | 0.737 | 0.412 | 0.189 | 0.053 | 0.367 |
|  |  |  |  | L0-bc | 0.827 | 0.278 | 0.091 | 0.053 | 0.423 |
|  |  |  |  | L0-perm | 0.720 | 0.450 | 0.208 | 0.000 | 0.343 |
|  |  |  |  | L0-perm+cf | 0.853 | 0.214 | 0.053 | 0.000 | 0.567 |
|  |  |  |  | L0 | 0.813 | 0.292 | 0.097 | 0.000 | 0.353 |
|  |  |  |  | L1 | 0.807 | 0.271 | 0.037 | 0.000 | 0.353 |
| Qwen2.5-7B-Instruct | newsgroups | 20 | 300 | raw | 0.660 | 0.621 | 0.276 | 0.233 | 0.047 |
|  |  |  |  | L0-bc | 0.667 | 0.598 | 0.268 | 0.230 | 0.060 |
|  |  |  |  | L0-perm | 0.703 | 0.556 | 0.254 | 0.133 | 0.337 |
|  |  |  |  | L0-perm+cf | 0.687 | 0.562 | 0.264 | 0.127 | 0.420 |
|  |  |  |  | L0 | 0.707 | 0.549 | 0.250 | 0.123 | 0.350 |
|  |  |  |  | L1 | 0.707 | 0.436 | 0.082 | 0.123 | 0.410 |
| Qwen3-30B-A3B-Instruct-2507 | banking20 | 20 | 300 | raw | 0.733 | 0.502 | 0.246 | 0.143 | 0.253 |
|  |  |  |  | L0-bc | 0.740 | 0.481 | 0.237 | 0.147 | 0.370 |
|  |  |  |  | L0-perm | 0.760 | 0.456 | 0.212 | 0.107 | 0.217 |
|  |  |  |  | L0-perm+cf | 0.773 | 0.424 | 0.202 | 0.087 | 0.217 |
|  |  |  |  | L0 | 0.767 | 0.435 | 0.210 | 0.097 | 0.343 |
|  |  |  |  | L1 | 0.767 | 0.348 | 0.079 | 0.097 | 0.477 |
| Qwen3-30B-A3B-Instruct-2507 | injection | 2 | 300 | raw | 0.723 | 0.528 | 0.253 | 0.093 | 0.417 |
|  |  |  |  | L0-bc | 0.743 | 0.460 | 0.232 | 0.097 | 0.420 |
|  |  |  |  | L0-perm | 0.730 | 0.490 | 0.240 | 0.000 | 0.427 |
|  |  |  |  | L0-perm+cf | 0.797 | 0.361 | 0.169 | 0.000 | 0.457 |
|  |  |  |  | L0 | 0.767 | 0.418 | 0.193 | 0.000 | 0.433 |
|  |  |  |  | L1 | 0.763 | 0.305 | 0.080 | 0.000 | 0.433 |
| Qwen3-30B-A3B-Instruct-2507 | newsgroups | 20 | 300 | raw | 0.730 | 0.507 | 0.249 | 0.133 | 0.110 |
|  |  |  |  | L0-bc | 0.727 | 0.511 | 0.252 | 0.140 | 0.247 |
|  |  |  |  | L0-perm | 0.743 | 0.491 | 0.235 | 0.097 | 0.393 |
|  |  |  |  | L0-perm+cf | 0.747 | 0.484 | 0.225 | 0.097 | 0.343 |
|  |  |  |  | L0 | 0.740 | 0.491 | 0.239 | 0.100 | 0.307 |
|  |  |  |  | L1 | 0.740 | 0.398 | 0.086 | 0.100 | 0.363 |
| Qwen3-8B | banking20 | 20 | 300 | raw | 0.750 | 0.490 | 0.235 | 0.227 | 0.077 |
|  |  |  |  | L0-bc | 0.753 | 0.485 | 0.234 | 0.220 | 0.437 |
|  |  |  |  | L0-perm | 0.800 | 0.382 | 0.189 | 0.077 | 0.337 |
|  |  |  |  | L0-perm+cf | 0.803 | 0.383 | 0.185 | 0.080 | 0.167 |
|  |  |  |  | L0 | 0.807 | 0.373 | 0.180 | 0.077 | 0.477 |
|  |  |  |  | L1 | 0.807 | 0.315 | 0.100 | 0.077 | 0.543 |
| Qwen3-8B | injection | 2 | 300 | raw | 0.693 | 0.577 | 0.287 | 0.060 | 0.297 |
|  |  |  |  | L0-bc | 0.730 | 0.521 | 0.250 | 0.057 | 0.013 |
|  |  |  |  | L0-perm | 0.670 | 0.624 | 0.311 | 0.000 | 0.303 |
|  |  |  |  | L0-perm+cf | 0.813 | 0.331 | 0.140 | 0.000 | 0.460 |
|  |  |  |  | L0 | 0.710 | 0.568 | 0.272 | 0.000 | 0.160 |
|  |  |  |  | L1 | 0.707 | 0.384 | 0.162 | 0.000 | 0.303 |
| Qwen3-8B | newsgroups | 20 | 300 | raw | 0.640 | 0.683 | 0.331 | 0.237 | 0.013 |
|  |  |  |  | L0-bc | 0.640 | 0.675 | 0.332 | 0.220 | 0.220 |
|  |  |  |  | L0-perm | 0.657 | 0.641 | 0.315 | 0.177 | 0.347 |
|  |  |  |  | L0-perm+cf | 0.653 | 0.642 | 0.317 | 0.173 | 0.347 |
|  |  |  |  | L0 | 0.660 | 0.632 | 0.308 | 0.173 | 0.427 |
|  |  |  |  | L1 | 0.660 | 0.479 | 0.157 | 0.170 | 0.417 |

Position bias is the big one on `choice`. At K=20, reversing the option list changed 13 to 24 percent of answers at `raw` on all three models. Reading all K cyclic shifts and combining in log space cut that to 7 to 17 percent and added 1 to 6 accuracy points, with no labels and K short prompts under one shared state. Label-prior bias is the big one on `noul`; a label-free prior estimate (mean over a batch of real inputs, or a content-free probe) fixes most of it.

Debiasing is not calibration. After L0 the models are still overconfident; one temperature fit on 200 labels brings ECE from 0.2 to 0.3 down to 0.04 to 0.16 without changing a single argmax. That is why every result carries a `level`.

## 2. On Laya's own benchmark

Laya's headline is 0.766 on LocalLLaMA/typed-decisions, above Jev's 0.727. We ran Laya's three checkpoints and three open Qwen models with AnyJev on the same 2,000 decisions.

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

Read it two ways. On argmax accuracy, Laya fine-tuned on the train split wins (0.768), Jev is second, and a 32B open model with zero training is 2.7 points behind Jev (0.700). On the probabilities, which is the whole point of a System One model, the picture changes. The fine-tuned Laya's ECE (0.215) is six times AnyJev L1's (0.034), and its soft accuracy (0.471) is below every zero-shot Qwen row (0.51 to 0.56). Two things keep this honest: temperature scaling trades soft accuracy for calibration, so the 7B and 8B L1 rows fall to 0.452 and 0.457 while their Brier improves; and on Brier alone the fine-tuned Laya stays narrowly ahead, 0.118 against AnyJev L1's best of 0.120. Fine-tuning taught it the answers; it did not teach it to state its confidence. And Laya's zero-shot checkpoints, the ones you would use on a question they were not trained for, score 0.34 to 0.36 against a 0.32 random baseline; any open instruct model with AnyJev scores 0.63 to 0.70 on the same questions with no training at all.

## 3. Inside NanoJev's maze

NanoJev's README compares its trained 0.6B model against "Untuned Qwen3-0.6B", which reads A/B logits for four Boolean questions per maze cell. Their held-out gameplay table puts that baseline at 2 of 10 mazes, on a 274-case suite we did not run. What follows is a different evaluation and should not be read against that figure. We reimplemented the same A/B readout and ran it inside NanoJev's scaled_maze pipeline, on its 15 test and out-of-distribution episodes at sizes 8 to 50, with their frozen exploration code and the same pinned Qwen3-0.6B commit, changing only the engine that answers the Boolean. NanoJev publishes no untuned-Qwen row on the scaled suite, so that row below is our reimplementation of their protocol rather than a number of theirs.

| engine | goal test | goal ood | attempts | collisions | edge acc | majority | edge Brier | edge questions |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B + AnyJev L0 (batch prior) | 10/11 | 3/4 | 15616 | 6236 | 0.490 | 0.618 | 0.271 | 23156 |
| Qwen3-0.6B + AnyJev L0 (content_free prior) | 10/11 | 4/4 | 16278 | 6563 | 0.490 | 0.614 | 0.274 | 23284 |
| Qwen3-0.6B + AnyJev L0 (none prior) | 10/11 | 3/4 | 21851 | 9136 | 0.403 | 0.597 | 0.460 | 28040 |
| Qwen3-0.6B + AnyJev raw | 11/11 | 4/4 | 5825 | 2616 | 0.537 | 0.539 | 0.363 | 10944 |
| Qwen3-8B + AnyJev L0 (batch prior) | 11/11 | 3/4 | 17841 | 7171 | 0.555 | 0.600 | 0.344 | 25124 |
| Qwen3-0.6B native A/B readout (NanoJev's 'Untuned Qwen' protocol) | 10/11 | 3/4 | 20555 | 8496 | 0.419 | 0.607 | 0.305 | 27660 |

Two things are true at once. The untuned model's result depends heavily on how you read it: the same Qwen3-0.6B goes from 13/15 mazes and 20,555 attempts under NanoJev's A/B readout to 15/15 and 5,825 attempts under AnyJev's raw Yes/No readout, so the "Untuned Qwen" row in a README is a property of the readout as much as the model. And no LLM readout, not even Qwen3-8B, answers "is one step north clear?" better than the majority label (edge accuracy 0.40 to 0.56 against a 0.54 to 0.62 majority, with the raw row's 0.537 against 0.539 a tie rather than a win): the maze differences come from how each readout's average probability interacts with the controller's p >= 0.5 probe rule, not from map reading. Debiasing (L0) centers the probabilities at 0.5 and, on this skewed task, that is worse for the controller than raw's optimism. The right takeaway is not "AnyJev wins the maze"; it is "this maze does not measure what it claims for untrained models".

## What this does not show

- One model family. Llama and Gemma are next.
- K=20 with letter labels. More than 26 options needs span scoring, not yet in v0.0.1.
- Teacher labels on typed-decisions come from two frontier models with 0.735 self-agreement; nobody, including Jev, can score much above that.
- No hosted-model row measured by us. Jev's numbers are the published ones.

## Try it

```bash
pip install anyjev
```

```python
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend
d = Decider(HFBackend("Qwen/Qwen3-8B"))
r = d.decide(state, [Question.choice("Which team?", ["billing", "technical", "sales"]),
                     Question.noul("Is this a prompt injection?")])
r[0].distribution, r[1].p_true, r.level   # {'billing': 0.81, ...}, 0.12, 'L0'
```

Backends: transformers, vLLM. Next: SGLang, llama.cpp, MLX, an OpenRouter-compatible `/v1/decisions` server, conformal abstention, span readout. Backend adapters and bench providers are one file each; PRs welcome.
