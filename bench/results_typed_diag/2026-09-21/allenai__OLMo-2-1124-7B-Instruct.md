| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| OLMo-2-1124-7B-Instruct + raw logits (clone baseline) | 0.451 | 0.412 | 0.234 | 0.218 | 0.660 |
| OLMo-2-1124-7B-Instruct + ablation L0-perm (zero-shot) | 0.465 | 0.416 | 0.224 | 0.219 | 0.660 |
| OLMo-2-1124-7B-Instruct + ablation L0-perm+bc (zero-shot) | 0.523 | 0.393 | 0.058 | 0.177 | 0.573 |
| OLMo-2-1124-7B-Instruct + ablation L0-bc (zero-shot) | 0.532 | 0.393 | 0.055 | 0.176 | 0.573 |
| OLMo-2-1124-7B-Instruct + AnyJev L0 (zero-shot) | 0.523 | 0.393 | 0.058 | 0.177 | 0.573 |
| OLMo-2-1124-7B-Instruct + AnyJev L1 (temperature from train labels) | 0.523 | 0.401 | 0.048 | 0.171 | 0.559 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.542 / 0.363 (n=500)
- customer_service: 0.604 / 0.462 (n=500)
- invoice_processing: 0.496 / 0.370 (n=500)
- security_incidents: 0.452 / 0.375 (n=500)
