| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-4B + raw logits (clone baseline) | 0.547 | 0.484 | 0.412 | 0.262 | 0.656 |
| Qwen3-4B + ablation L0-perm (zero-shot) | 0.563 | 0.491 | 0.396 | 0.253 | 0.656 |
| Qwen3-4B + ablation L0-perm+bc (zero-shot) | 0.574 | 0.491 | 0.333 | 0.231 | 0.562 |
| Qwen3-4B + ablation L0-bc (zero-shot) | 0.564 | 0.491 | 0.346 | 0.232 | 0.562 |
| Qwen3-4B + AnyJev L0 (zero-shot) | 0.574 | 0.491 | 0.333 | 0.231 | 0.562 |
| Qwen3-4B + AnyJev L1 (temperature from train labels) | 0.571 | 0.425 | 0.056 | 0.161 | 0.501 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.484 / 0.425 (n=500)
- customer_service: 0.688 / 0.575 (n=500)
- invoice_processing: 0.542 / 0.497 (n=500)
- security_incidents: 0.582 / 0.469 (n=500)
