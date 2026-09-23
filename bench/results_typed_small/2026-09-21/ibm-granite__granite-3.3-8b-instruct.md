| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| granite-3.3-8b-instruct + raw logits (clone baseline) | 0.622 | 0.515 | 0.257 | 0.184 | 0.635 |
| granite-3.3-8b-instruct + ablation L0-perm (zero-shot) | 0.634 | 0.518 | 0.238 | 0.176 | 0.635 |
| granite-3.3-8b-instruct + ablation L0-perm+bc (zero-shot) | 0.621 | 0.483 | 0.131 | 0.163 | 0.503 |
| granite-3.3-8b-instruct + ablation L0-bc (zero-shot) | 0.604 | 0.482 | 0.158 | 0.169 | 0.503 |
| granite-3.3-8b-instruct + AnyJev L0 (zero-shot) | 0.621 | 0.483 | 0.131 | 0.163 | 0.503 |
| granite-3.3-8b-instruct + AnyJev L1 (temperature from train labels) | 0.618 | 0.432 | 0.059 | 0.153 | 0.483 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.566 / 0.444 (n=500)
- customer_service: 0.676 / 0.543 (n=500)
- invoice_processing: 0.624 / 0.496 (n=500)
- security_incidents: 0.616 / 0.450 (n=500)
