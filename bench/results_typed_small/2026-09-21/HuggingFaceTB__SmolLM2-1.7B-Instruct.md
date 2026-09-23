| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| SmolLM2-1.7B-Instruct + raw logits (clone baseline) | 0.410 | 0.353 | 0.101 | 0.217 | 0.639 |
| SmolLM2-1.7B-Instruct + ablation L0-perm (zero-shot) | 0.412 | 0.350 | 0.089 | 0.214 | 0.639 |
| SmolLM2-1.7B-Instruct + ablation L0-perm+bc (zero-shot) | 0.389 | 0.331 | 0.047 | 0.198 | 0.699 |
| SmolLM2-1.7B-Instruct + ablation L0-bc (zero-shot) | 0.374 | 0.331 | 0.039 | 0.198 | 0.699 |
| SmolLM2-1.7B-Instruct + AnyJev L0 (zero-shot) | 0.389 | 0.331 | 0.047 | 0.198 | 0.699 |
| SmolLM2-1.7B-Instruct + AnyJev L1 (temperature from train labels) | 0.388 | 0.351 | 0.040 | 0.194 | 0.684 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.326 / 0.301 (n=500)
- customer_service: 0.454 / 0.328 (n=500)
- invoice_processing: 0.396 / 0.352 (n=500)
- security_incidents: 0.378 / 0.342 (n=500)
