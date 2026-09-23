| backend | state_tokens | k | batch_raw_ms | batch_L0_full_ms | batch_L0_shared_ms | single_raw_ms | single_L0_full_ms | single_L0_shared_ms | L0/raw single (full) | L0/raw single (shared) |
|---|---|---|---|---|---|---|---|---|---|---|
| hf | 110 | 4 | 10.7 | 42.8 | 22.1 | 48.5 | 54.6 | 108.5 | 1.13x | 2.24x |
| hf | 110 | 20 | 16.0 | 314.3 | 217.1 | 49.6 | 317.0 | 238.1 | 6.39x | 4.80x |
| hf | 994 | 4 | 62.2 | 226.6 | 79.0 | 69.5 | 231.2 | 143.9 | 3.33x | 2.07x |
| hf | 994 | 20 | 69.3 | 1232.6 | 328.8 | 74.1 | 1232.8 | 323.4 | 16.64x | 4.37x |
