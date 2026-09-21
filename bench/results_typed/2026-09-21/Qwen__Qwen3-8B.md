| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-8B + raw logits (clone baseline) | 0.626 | 0.520 | 0.328 | 0.210 | 0.621 |
| Qwen3-8B + ablation L0-perm (zero-shot) | 0.633 | 0.525 | 0.320 | 0.210 | 0.621 |
| Qwen3-8B + ablation L0-perm+bc (zero-shot) | 0.640 | 0.523 | 0.273 | 0.196 | 0.617 |
| Qwen3-8B + ablation L0-bc (zero-shot) | 0.632 | 0.517 | 0.285 | 0.198 | 0.617 |
| Qwen3-8B + AnyJev L0 (zero-shot) | 0.640 | 0.523 | 0.273 | 0.196 | 0.617 |
| Qwen3-8B + AnyJev L1 (temperature from train labels) | 0.646 | 0.457 | 0.056 | 0.143 | 0.474 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.600 / 0.473 (n=500)
- customer_service: 0.736 / 0.588 (n=500)
- invoice_processing: 0.608 / 0.542 (n=500)
- security_incidents: 0.618 / 0.490 (n=500)
