| backend | state_tokens | k | batch_raw_ms | batch_L0_full_ms | single_raw_ms | single_L0_full_ms | L0/raw single (full) | L0/raw single (shared) |
|---|---|---|---|---|---|---|---|---|
| vllm | 121 | 4 | 10.9 | 36.4 | 33.5 | 69.9 | 2.09x |  |
| vllm | 121 | 20 | 12.7 | 222.0 | 38.1 | 262.4 | 6.89x |  |
| vllm | 1006 | 4 | 37.5 | 146.9 | 58.1 | 178.7 | 3.08x |  |
| vllm | 1006 | 20 | 42.9 | 838.5 | 72.4 | 874.5 | 12.07x |  |
