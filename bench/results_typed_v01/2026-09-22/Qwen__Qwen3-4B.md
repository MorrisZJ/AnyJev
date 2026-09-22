| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-4B + raw logits (clone baseline) | 0.547 | 0.484 | 0.411 | 0.262 | 0.656 |
| Qwen3-4B + ablation L0-perm (zero-shot) | 0.562 | 0.490 | 0.396 | 0.253 | 0.656 |
| Qwen3-4B + ablation L0-perm+bc (zero-shot) | 0.564 | 0.494 | 0.373 | 0.243 | 0.622 |
| Qwen3-4B + ablation L0-bc (zero-shot) | 0.550 | 0.491 | 0.388 | 0.247 | 0.622 |
| Qwen3-4B + AnyJev L0 (zero-shot) | 0.564 | 0.494 | 0.373 | 0.243 | 0.622 |
| Qwen3-4B + AnyJev L1 (temperature from train labels) | 0.567 | 0.434 | 0.043 | 0.159 | 0.500 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.464 / 0.423 (n=500)
- customer_service: 0.682 / 0.574 (n=500)
- invoice_processing: 0.538 / 0.504 (n=500)
- security_incidents: 0.572 / 0.473 (n=500)
