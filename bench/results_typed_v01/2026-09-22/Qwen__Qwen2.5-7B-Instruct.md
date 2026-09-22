| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct + raw logits (clone baseline) | 0.620 | 0.514 | 0.287 | 0.209 | 0.437 |
| Qwen2.5-7B-Instruct + ablation L0-perm (zero-shot) | 0.619 | 0.516 | 0.288 | 0.208 | 0.437 |
| Qwen2.5-7B-Instruct + ablation L0-perm+bc (zero-shot) | 0.628 | 0.512 | 0.234 | 0.188 | 0.439 |
| Qwen2.5-7B-Instruct + ablation L0-bc (zero-shot) | 0.622 | 0.510 | 0.242 | 0.192 | 0.439 |
| Qwen2.5-7B-Instruct + AnyJev L0 (zero-shot) | 0.628 | 0.512 | 0.234 | 0.188 | 0.439 |
| Qwen2.5-7B-Instruct + AnyJev L1 (temperature from train labels) | 0.628 | 0.461 | 0.038 | 0.148 | 0.425 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.610 / 0.485 (n=500)
- customer_service: 0.694 / 0.567 (n=500)
- invoice_processing: 0.600 / 0.514 (n=500)
- security_incidents: 0.608 / 0.482 (n=500)
