| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-32B + raw logits (clone baseline) | 0.684 | 0.556 | 0.206 | 0.144 | 0.488 |
| Qwen3-32B + ablation L0-perm (zero-shot) | 0.688 | 0.557 | 0.197 | 0.142 | 0.488 |
| Qwen3-32B + ablation L0-perm+bc (zero-shot) | 0.700 | 0.555 | 0.149 | 0.129 | 0.449 |
| Qwen3-32B + ablation L0-bc (zero-shot) | 0.693 | 0.553 | 0.158 | 0.130 | 0.449 |
| Qwen3-32B + AnyJev L0 (zero-shot) | 0.700 | 0.555 | 0.149 | 0.129 | 0.449 |
| Qwen3-32B + AnyJev L1 (temperature from train labels) | 0.699 | 0.508 | 0.036 | 0.119 | 0.416 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.638 / 0.488 (n=500)
- customer_service: 0.740 / 0.601 (n=500)
- invoice_processing: 0.714 / 0.614 (n=500)
- security_incidents: 0.706 / 0.517 (n=500)
