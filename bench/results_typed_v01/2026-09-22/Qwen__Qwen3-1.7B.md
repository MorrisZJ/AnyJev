| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-1.7B + raw logits (clone baseline) | 0.468 | 0.434 | 0.484 | 0.299 | 0.664 |
| Qwen3-1.7B + ablation L0-perm (zero-shot) | 0.493 | 0.448 | 0.439 | 0.278 | 0.664 |
| Qwen3-1.7B + ablation L0-perm+bc (zero-shot) | 0.494 | 0.441 | 0.396 | 0.261 | 0.657 |
| Qwen3-1.7B + ablation L0-bc (zero-shot) | 0.483 | 0.437 | 0.429 | 0.275 | 0.657 |
| Qwen3-1.7B + AnyJev L0 (zero-shot) | 0.494 | 0.441 | 0.396 | 0.261 | 0.657 |
| Qwen3-1.7B + AnyJev L1 (temperature from train labels) | 0.499 | 0.403 | 0.055 | 0.172 | 0.600 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.450 / 0.403 (n=500)
- customer_service: 0.596 / 0.519 (n=500)
- invoice_processing: 0.482 / 0.442 (n=500)
- security_incidents: 0.448 / 0.401 (n=500)
