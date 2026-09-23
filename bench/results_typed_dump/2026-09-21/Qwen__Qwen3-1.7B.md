| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-1.7B + raw logits (clone baseline) | 0.469 | 0.434 | 0.482 | 0.299 | 0.663 |
| Qwen3-1.7B + ablation L0-perm (zero-shot) | 0.488 | 0.447 | 0.443 | 0.279 | 0.663 |
| Qwen3-1.7B + ablation L0-perm+bc (zero-shot) | 0.482 | 0.434 | 0.341 | 0.242 | 0.677 |
| Qwen3-1.7B + ablation L0-bc (zero-shot) | 0.483 | 0.434 | 0.356 | 0.248 | 0.677 |
| Qwen3-1.7B + AnyJev L0 (zero-shot) | 0.482 | 0.434 | 0.341 | 0.242 | 0.677 |
| Qwen3-1.7B + AnyJev L1 (temperature from train labels) | 0.482 | 0.393 | 0.048 | 0.175 | 0.607 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.470 / 0.419 (n=500)
- customer_service: 0.560 / 0.495 (n=500)
- invoice_processing: 0.468 / 0.433 (n=500)
- security_incidents: 0.432 / 0.390 (n=500)
