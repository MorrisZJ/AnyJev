| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| SmolLM2-1.7B-Instruct + raw logits (clone baseline) | 0.411 | 0.353 | 0.099 | 0.217 | 0.639 |
| SmolLM2-1.7B-Instruct + ablation L0-perm (zero-shot) | 0.412 | 0.350 | 0.089 | 0.214 | 0.639 |
| SmolLM2-1.7B-Instruct + ablation L0-perm+bc (zero-shot) | 0.420 | 0.336 | 0.073 | 0.198 | 0.673 |
| SmolLM2-1.7B-Instruct + ablation L0-bc (zero-shot) | 0.412 | 0.337 | 0.070 | 0.198 | 0.673 |
| SmolLM2-1.7B-Instruct + AnyJev L0 (zero-shot) | 0.420 | 0.336 | 0.073 | 0.198 | 0.673 |
| SmolLM2-1.7B-Instruct + AnyJev L1 (temperature from train labels) | 0.415 | 0.382 | 0.055 | 0.182 | 0.618 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.338 / 0.304 (n=500)
- customer_service: 0.474 / 0.329 (n=500)
- invoice_processing: 0.470 / 0.366 (n=500)
- security_incidents: 0.400 / 0.343 (n=500)
