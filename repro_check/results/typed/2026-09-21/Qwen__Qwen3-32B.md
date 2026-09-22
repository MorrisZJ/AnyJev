| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-32B + raw logits (clone baseline) | 0.683 | 0.556 | 0.208 | 0.144 | 0.488 |
| Qwen3-32B + ablation L0-perm (zero-shot) | 0.686 | 0.557 | 0.198 | 0.142 | 0.488 |
| Qwen3-32B + ablation L0-perm+bc (zero-shot) | 0.700 | 0.548 | 0.134 | 0.128 | 0.457 |
| Qwen3-32B + ablation L0-bc (zero-shot) | 0.691 | 0.543 | 0.134 | 0.129 | 0.457 |
| Qwen3-32B + AnyJev L0 (zero-shot) | 0.700 | 0.548 | 0.134 | 0.128 | 0.457 |
| Qwen3-32B + AnyJev L1 (temperature from train labels) | 0.696 | 0.502 | 0.034 | 0.120 | 0.411 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.646 / 0.483 (n=500)
- customer_service: 0.750 / 0.600 (n=500)
- invoice_processing: 0.704 / 0.602 (n=500)
- security_incidents: 0.698 / 0.508 (n=500)
