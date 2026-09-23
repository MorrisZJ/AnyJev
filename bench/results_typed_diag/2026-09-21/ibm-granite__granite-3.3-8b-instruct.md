| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| granite-3.3-8b-instruct + raw logits (clone baseline) | 0.622 | 0.515 | 0.256 | 0.184 | 0.636 |
| granite-3.3-8b-instruct + ablation L0-perm (zero-shot) | 0.634 | 0.518 | 0.237 | 0.177 | 0.636 |
| granite-3.3-8b-instruct + ablation L0-perm+bc (zero-shot) | 0.624 | 0.483 | 0.128 | 0.163 | 0.505 |
| granite-3.3-8b-instruct + ablation L0-bc (zero-shot) | 0.607 | 0.482 | 0.155 | 0.169 | 0.505 |
| granite-3.3-8b-instruct + AnyJev L0 (zero-shot) | 0.624 | 0.483 | 0.128 | 0.163 | 0.505 |
| granite-3.3-8b-instruct + AnyJev L1 (temperature from train labels) | 0.621 | 0.432 | 0.062 | 0.153 | 0.485 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.574 / 0.443 (n=500)
- customer_service: 0.676 / 0.543 (n=500)
- invoice_processing: 0.628 / 0.497 (n=500)
- security_incidents: 0.616 / 0.449 (n=500)
