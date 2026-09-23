| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| OLMo-2-1124-7B-Instruct + raw logits (clone baseline) | 0.449 | 0.412 | 0.236 | 0.218 | 0.660 |
| OLMo-2-1124-7B-Instruct + ablation L0-perm (zero-shot) | 0.465 | 0.416 | 0.223 | 0.219 | 0.660 |
| OLMo-2-1124-7B-Instruct + ablation L0-perm+bc (zero-shot) | 0.499 | 0.403 | 0.044 | 0.177 | 0.562 |
| OLMo-2-1124-7B-Instruct + ablation L0-bc (zero-shot) | 0.510 | 0.404 | 0.049 | 0.177 | 0.562 |
| OLMo-2-1124-7B-Instruct + AnyJev L0 (zero-shot) | 0.499 | 0.403 | 0.044 | 0.177 | 0.562 |
| OLMo-2-1124-7B-Instruct + AnyJev L1 (temperature from train labels) | 0.499 | 0.419 | 0.059 | 0.165 | 0.556 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.496 / 0.365 (n=500)
- customer_service: 0.568 / 0.471 (n=500)
- invoice_processing: 0.510 / 0.397 (n=500)
- security_incidents: 0.422 / 0.378 (n=500)
