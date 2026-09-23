| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 + raw logits (clone baseline) | 0.552 | 0.482 | 0.342 | 0.229 | 0.713 |
| Mistral-7B-Instruct-v0.3 + ablation L0-perm (zero-shot) | 0.582 | 0.499 | 0.307 | 0.204 | 0.713 |
| Mistral-7B-Instruct-v0.3 + ablation L0-perm+bc (zero-shot) | 0.597 | 0.487 | 0.213 | 0.179 | 0.647 |
| Mistral-7B-Instruct-v0.3 + ablation L0-bc (zero-shot) | 0.573 | 0.476 | 0.259 | 0.202 | 0.647 |
| Mistral-7B-Instruct-v0.3 + AnyJev L0 (zero-shot) | 0.597 | 0.487 | 0.213 | 0.179 | 0.647 |
| Mistral-7B-Instruct-v0.3 + AnyJev L1 (temperature from train labels) | 0.599 | 0.437 | 0.042 | 0.153 | 0.495 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.564 / 0.434 (n=500)
- customer_service: 0.642 / 0.541 (n=500)
- invoice_processing: 0.566 / 0.497 (n=500)
- security_incidents: 0.614 / 0.477 (n=500)
