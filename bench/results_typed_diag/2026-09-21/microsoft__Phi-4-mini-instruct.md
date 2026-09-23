| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Phi-4-mini-instruct + raw logits (clone baseline) | 0.632 | 0.496 | 0.151 | 0.165 | 0.553 |
| Phi-4-mini-instruct + ablation L0-perm (zero-shot) | 0.629 | 0.491 | 0.138 | 0.164 | 0.553 |
| Phi-4-mini-instruct + ablation L0-perm+bc (zero-shot) | 0.617 | 0.453 | 0.063 | 0.155 | 0.487 |
| Phi-4-mini-instruct + ablation L0-bc (zero-shot) | 0.614 | 0.448 | 0.063 | 0.156 | 0.487 |
| Phi-4-mini-instruct + AnyJev L0 (zero-shot) | 0.617 | 0.453 | 0.063 | 0.155 | 0.487 |
| Phi-4-mini-instruct + AnyJev L1 (temperature from train labels) | 0.623 | 0.435 | 0.059 | 0.152 | 0.463 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.588 / 0.422 (n=500)
- customer_service: 0.658 / 0.505 (n=500)
- invoice_processing: 0.620 / 0.451 (n=500)
- security_incidents: 0.600 / 0.434 (n=500)
