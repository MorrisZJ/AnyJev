| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-8B + raw logits (clone baseline) | 0.627 | 0.520 | 0.326 | 0.210 | 0.622 |
| Qwen3-8B + ablation L0-perm (zero-shot) | 0.635 | 0.525 | 0.318 | 0.209 | 0.622 |
| Qwen3-8B + ablation L0-perm+bc (zero-shot) | 0.644 | 0.524 | 0.269 | 0.195 | 0.615 |
| Qwen3-8B + ablation L0-bc (zero-shot) | 0.636 | 0.518 | 0.281 | 0.197 | 0.615 |
| Qwen3-8B + AnyJev L0 (zero-shot) | 0.644 | 0.524 | 0.269 | 0.195 | 0.615 |
| Qwen3-8B + AnyJev L1 (temperature from train labels) | 0.646 | 0.458 | 0.054 | 0.143 | 0.475 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.606 / 0.473 (n=500)
- customer_service: 0.738 / 0.589 (n=500)
- invoice_processing: 0.616 / 0.544 (n=500)
- security_incidents: 0.616 / 0.489 (n=500)
