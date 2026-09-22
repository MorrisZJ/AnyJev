| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 + raw logits (clone baseline) | 0.599 | 0.508 | 0.343 | 0.221 | 0.755 |
| Qwen3-30B-A3B-Instruct-2507 + ablation L0-perm (zero-shot) | 0.620 | 0.522 | 0.314 | 0.202 | 0.755 |
| Qwen3-30B-A3B-Instruct-2507 + ablation L0-perm+bc (zero-shot) | 0.630 | 0.524 | 0.287 | 0.194 | 0.689 |
| Qwen3-30B-A3B-Instruct-2507 + ablation L0-bc (zero-shot) | 0.618 | 0.512 | 0.303 | 0.208 | 0.689 |
| Qwen3-30B-A3B-Instruct-2507 + AnyJev L0 (zero-shot) | 0.630 | 0.524 | 0.287 | 0.194 | 0.689 |
| Qwen3-30B-A3B-Instruct-2507 + AnyJev L1 (temperature from train labels) | 0.630 | 0.459 | 0.047 | 0.144 | 0.448 |
| laya-typed-decisions (fine-tuned on train; published) | 0.766 | 0.471 | 0.213 | 0.061 | 0.242 |
| laya (zero-shot; published) | 0.361 | 0.332 | 0.175 | 0.316 | 0.694 |
| Jev 1.13.0 (published by Laya) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |

Per workflow (acc / soft_acc), AnyJev L0:
- agent_trace_observability: 0.588 / 0.463 (n=500)
- customer_service: 0.642 / 0.561 (n=500)
- invoice_processing: 0.650 / 0.577 (n=500)
- security_incidents: 0.640 / 0.495 (n=500)
