| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| laya-multilingual (zero-shot), measured here | 0.340 | 0.325 | 0.287 | 0.269 | 0.688 |
| laya (zero-shot), measured here | 0.359 | 0.331 | 0.177 | 0.227 | 0.694 |
| SmolLM2-1.7B-Instruct + raw logits (clone baseline) | 0.411 | 0.353 | 0.099 | 0.217 | 0.639 |
| SmolLM2-1.7B-Instruct + AnyJev L1, temperature from 200 train cases | 0.415 | 0.382 | 0.055 | 0.182 | 0.618 |
| SmolLM2-1.7B-Instruct + AnyJev L0, zero-shot | 0.420 | 0.336 | 0.073 | 0.198 | 0.673 |
| OLMo-2-1124-7B-Instruct + raw logits (clone baseline) | 0.452 | 0.412 | 0.233 | 0.218 | 0.660 |
| Qwen3-1.7B + raw logits (clone baseline) | 0.468 | 0.434 | 0.484 | 0.299 | 0.664 |
| Qwen3-1.7B + AnyJev L0, zero-shot | 0.494 | 0.441 | 0.396 | 0.261 | 0.657 |
| OLMo-2-1124-7B-Instruct + AnyJev L1, temperature from 200 train cases | 0.494 | 0.420 | 0.052 | 0.165 | 0.554 |
| OLMo-2-1124-7B-Instruct + AnyJev L0, zero-shot | 0.499 | 0.403 | 0.039 | 0.178 | 0.562 |
| Qwen3-1.7B + AnyJev L1, temperature from 200 train cases | 0.499 | 0.403 | 0.055 | 0.172 | 0.600 |
| Qwen3-4B + raw logits (clone baseline) | 0.547 | 0.484 | 0.411 | 0.262 | 0.656 |
| Mistral-7B-Instruct-v0.3 + raw logits (clone baseline) | 0.552 | 0.482 | 0.342 | 0.229 | 0.713 |
| Qwen3-4B + AnyJev L0, zero-shot | 0.564 | 0.494 | 0.373 | 0.243 | 0.622 |
| Qwen3-4B + AnyJev L1, temperature from 200 train cases | 0.567 | 0.434 | 0.043 | 0.159 | 0.500 |
| Mistral-7B-Instruct-v0.3 + AnyJev L1, temperature from 200 train cases | 0.597 | 0.448 | 0.042 | 0.149 | 0.504 |
| Qwen3-30B-A3B-Instruct-2507 + raw logits (clone baseline) | 0.599 | 0.508 | 0.343 | 0.221 | 0.755 |
| Mistral-7B-Instruct-v0.3 + AnyJev L0, zero-shot | 0.608 | 0.499 | 0.234 | 0.181 | 0.671 |
| Qwen2.5-7B-Instruct + raw logits (clone baseline) | 0.620 | 0.514 | 0.287 | 0.209 | 0.437 |
| granite-3.3-8b-instruct + raw logits (clone baseline) | 0.621 | 0.516 | 0.277 | 0.190 | 0.635 |
| Qwen3-8B + raw logits (clone baseline) | 0.626 | 0.520 | 0.328 | 0.210 | 0.621 |
| Qwen2.5-7B-Instruct + AnyJev L0, zero-shot | 0.628 | 0.512 | 0.234 | 0.188 | 0.439 |
| Qwen2.5-7B-Instruct + AnyJev L1, temperature from 200 train cases | 0.628 | 0.461 | 0.038 | 0.148 | 0.425 |
| Qwen3-30B-A3B-Instruct-2507 + AnyJev L0, zero-shot | 0.630 | 0.524 | 0.287 | 0.194 | 0.689 |
| Qwen3-30B-A3B-Instruct-2507 + AnyJev L1, temperature from 200 train cases | 0.630 | 0.459 | 0.047 | 0.144 | 0.448 |
| Phi-4-mini-instruct + AnyJev L0, zero-shot | 0.631 | 0.470 | 0.062 | 0.150 | 0.466 |
| Phi-4-mini-instruct + raw logits (clone baseline) | 0.632 | 0.496 | 0.151 | 0.165 | 0.554 |
| Phi-4-mini-instruct + AnyJev L1, temperature from 200 train cases | 0.633 | 0.456 | 0.051 | 0.145 | 0.452 |
| granite-3.3-8b-instruct + AnyJev L0, zero-shot | 0.643 | 0.513 | 0.197 | 0.164 | 0.541 |
| granite-3.3-8b-instruct + AnyJev L1, temperature from 200 train cases | 0.643 | 0.450 | 0.049 | 0.143 | 0.445 |
| Qwen3-8B + AnyJev L0, zero-shot | 0.647 | 0.530 | 0.290 | 0.198 | 0.591 |
| Qwen3-8B + AnyJev L1, temperature from 200 train cases | 0.648 | 0.468 | 0.055 | 0.140 | 0.444 |
| Qwen3-32B + raw logits (clone baseline) | 0.684 | 0.556 | 0.206 | 0.144 | 0.488 |
| Qwen3-32B + AnyJev L1, temperature from 200 train cases | 0.699 | 0.508 | 0.036 | 0.119 | 0.416 |
| Qwen3-32B + AnyJev L0, zero-shot | 0.700 | 0.555 | 0.149 | 0.129 | 0.449 |
| Jev 1.13.0 (0.727 as listed in Laya's BENCHMARKS.md; not measured here) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |
| laya-typed-decisions (fine-tuned on this set's train split), measured here | 0.768 | 0.471 | 0.215 | 0.118 | 0.243 |

LocalLLaMA/typed-decisions test split, 400 cases, 2,000 decisions, all rows except Jev measured on the same decisions. soft_acc = sum of predicted x teacher probabilities. brier_mean divides by the number of options, as Laya reports it.
