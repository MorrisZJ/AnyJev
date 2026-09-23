| backend | state_tokens | k | batch_raw_ms | batch_L0_full_ms | single_raw_ms | single_L0_full_ms | L0/raw single (full) | L0/raw single (shared) |
|---|---|---|---|---|---|---|---|---|
| vllm | 121 | 4 | 9.2 | 28.2 | 33.3 | 61.8 | 1.86x |  |
| vllm | 121 | 20 | 11.9 | 155.4 | 29.6 | 168.1 | 5.68x |  |
| vllm | 1006 | 4 | 37.3 | 109.6 | 62.7 | 89.5 | 1.43x |  |
| vllm | 1006 | 20 | 40.1 | 221.2 | 68.6 | 224.9 | 3.28x |  |
