| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| OLMo-2-1124-7B-Instruct + raw logits (clone baseline) | 0.453 | 0.412 | 0.232 | 0.218 | 0.660 |
| OLMo-2-1124-7B-Instruct + ablation L0-perm (zero-shot) | 0.466 | 0.416 | 0.223 | 0.219 | 0.660 |
| OLMo-2-1124-7B-Instruct + ablation L0-perm+bc (zero-shot) | 0.527 | 0.393 | 0.060 | 0.177 | 0.574 |
| OLMo-2-1124-7B-Instruct + ablation L0-bc (zero-shot) | 0.533 | 0.393 | 0.054 | 0.177 | 0.574 |
| OLMo-2-1124-7B-Instruct + AnyJev L0 (zero-shot) | 0.527 | 0.393 | 0.060 | 0.177 | 0.574 |
| OLMo-2-1124-7B-Instruct + AnyJev L1 (temperature from train labels) | 0.524 | 0.400 | 0.045 | 0.171 | 0.560 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.546 / 0.363 (n=500)
- customer_service: 0.602 / 0.462 (n=500)
- invoice_processing: 0.506 / 0.370 (n=500)
- security_incidents: 0.454 / 0.375 (n=500)
