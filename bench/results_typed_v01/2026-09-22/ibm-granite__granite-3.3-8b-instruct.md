| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| granite-3.3-8b-instruct + raw logits (clone baseline) | 0.621 | 0.516 | 0.277 | 0.190 | 0.635 |
| granite-3.3-8b-instruct + ablation L0-perm (zero-shot) | 0.631 | 0.520 | 0.260 | 0.181 | 0.635 |
| granite-3.3-8b-instruct + ablation L0-perm+bc (zero-shot) | 0.643 | 0.513 | 0.197 | 0.164 | 0.541 |
| granite-3.3-8b-instruct + ablation L0-bc (zero-shot) | 0.630 | 0.513 | 0.220 | 0.171 | 0.541 |
| granite-3.3-8b-instruct + AnyJev L0 (zero-shot) | 0.643 | 0.513 | 0.197 | 0.164 | 0.541 |
| granite-3.3-8b-instruct + AnyJev L1 (temperature from train labels) | 0.643 | 0.450 | 0.049 | 0.143 | 0.445 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.580 / 0.447 (n=500)
- customer_service: 0.680 / 0.557 (n=500)
- invoice_processing: 0.666 / 0.555 (n=500)
- security_incidents: 0.646 / 0.492 (n=500)
