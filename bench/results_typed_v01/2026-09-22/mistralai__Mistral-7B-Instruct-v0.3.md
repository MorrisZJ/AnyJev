| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 + raw logits (clone baseline) | 0.552 | 0.482 | 0.342 | 0.229 | 0.713 |
| Mistral-7B-Instruct-v0.3 + ablation L0-perm (zero-shot) | 0.582 | 0.499 | 0.307 | 0.204 | 0.713 |
| Mistral-7B-Instruct-v0.3 + ablation L0-perm+bc (zero-shot) | 0.608 | 0.499 | 0.234 | 0.181 | 0.671 |
| Mistral-7B-Instruct-v0.3 + ablation L0-bc (zero-shot) | 0.578 | 0.484 | 0.277 | 0.206 | 0.671 |
| Mistral-7B-Instruct-v0.3 + AnyJev L0 (zero-shot) | 0.608 | 0.499 | 0.234 | 0.181 | 0.671 |
| Mistral-7B-Instruct-v0.3 + AnyJev L1 (temperature from train labels) | 0.597 | 0.448 | 0.042 | 0.149 | 0.504 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.564 / 0.439 (n=500)
- customer_service: 0.624 / 0.537 (n=500)
- invoice_processing: 0.598 / 0.523 (n=500)
- security_incidents: 0.646 / 0.496 (n=500)
