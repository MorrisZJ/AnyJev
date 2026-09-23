| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-4B + raw logits (clone baseline) | 0.546 | 0.484 | 0.414 | 0.262 | 0.655 |
| Qwen3-4B + ablation L0-perm (zero-shot) | 0.565 | 0.491 | 0.395 | 0.253 | 0.655 |
| Qwen3-4B + ablation L0-perm+bc (zero-shot) | 0.575 | 0.492 | 0.332 | 0.231 | 0.560 |
| Qwen3-4B + ablation L0-bc (zero-shot) | 0.564 | 0.491 | 0.346 | 0.232 | 0.560 |
| Qwen3-4B + AnyJev L0 (zero-shot) | 0.575 | 0.492 | 0.332 | 0.231 | 0.560 |
| Qwen3-4B + AnyJev L1 (temperature from train labels) | 0.572 | 0.425 | 0.059 | 0.160 | 0.499 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.486 / 0.426 (n=500)
- customer_service: 0.690 / 0.575 (n=500)
- invoice_processing: 0.544 / 0.498 (n=500)
- security_incidents: 0.578 / 0.468 (n=500)
