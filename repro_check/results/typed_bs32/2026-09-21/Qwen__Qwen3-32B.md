| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-32B + raw logits (clone baseline) | 0.685 | 0.556 | 0.205 | 0.144 | 0.488 |
| Qwen3-32B + ablation L0-perm (zero-shot) | 0.687 | 0.557 | 0.198 | 0.142 | 0.488 |
| Qwen3-32B + ablation L0-perm+bc (zero-shot) | 0.699 | 0.548 | 0.136 | 0.128 | 0.456 |
| Qwen3-32B + ablation L0-bc (zero-shot) | 0.690 | 0.543 | 0.136 | 0.129 | 0.456 |
| Qwen3-32B + AnyJev L0 (zero-shot) | 0.699 | 0.548 | 0.136 | 0.128 | 0.456 |
| Qwen3-32B + AnyJev L1 (temperature from train labels) | 0.699 | 0.502 | 0.032 | 0.120 | 0.410 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.640 / 0.483 (n=500)
- customer_service: 0.752 / 0.600 (n=500)
- invoice_processing: 0.708 / 0.602 (n=500)
- security_incidents: 0.694 / 0.508 (n=500)
