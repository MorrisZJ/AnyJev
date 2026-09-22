# Latency: what L0 costs relative to raw

Milliseconds per decision, one H100 NVL, bf16, K-way `choice`, batch prior (no extra prompts, so L0's only
overhead is the K prefills). Every measurement uses states never seen before in the process, so a server-side
prefix cache can only help between the K shifts of one state, which is the deployment case. `batch` is
`decide_batch` over 32 states (throughput); `single` is `decide` on one state (latency). Regenerate with
`python -m bench.latency`.

## transformers backend, Qwen3-8B

| state tokens | K | batch raw | batch L0 full | batch L0 shared prefix | single raw | single L0 full | single L0 shared prefix |
|---|---|---|---|---|---|---|---|
| 110 | 4 | 10.7 | 42.8 (4.0x) | 22.1 (2.1x) | 48.5 | 54.6 (1.1x) | 108.5 (2.2x) |
| 110 | 20 | 16.0 | 314.3 (19.6x) | 217.1 (13.6x) | 49.6 | 317.0 (6.4x) | 238.1 (4.8x) |
| 994 | 4 | 62.2 | 226.6 (3.6x) | 79.0 (1.3x) | 69.5 | 231.2 (3.3x) | 143.9 (2.1x) |
| 994 | 20 | 69.3 | 1232.6 (17.8x) | 328.8 (4.7x) | 74.1 | 1232.8 (16.6x) | 323.4 (4.4x) |

The shared-prefix path (`HFBackend.score_shared`) computes the state once and scores the K option layouts
against the cached KV. It pays off when the prefix is long: at ~1000 tokens it turns L0 from 17x raw into 4.4x
at K=20 and from 3.3x into 2.1x at K=4. On short states a single batched forward over the full prompts is
cheaper than two forwards, which is why the Decider only shares prefixes of at least 256 tokens by default
(`shared_min_prefix_tokens`). Note the single-request raw floor of ~50 ms: eager-mode transformers overhead,
not model compute, which is why K=4 costs only 13 percent more than raw on a short state.

## vLLM backend, Qwen2.5-7B-Instruct (vLLM 0.7.0), prefix caching off and on

| state tokens | K | cache | batch raw | batch L0 | single raw | single L0 |
|---|---|---|---|---|---|---|
| 121 | 4 | off | 10.9 | 36.4 (3.3x) | 33.5 | 69.9 (2.1x) |
| 121 | 4 | **on** | 9.2 | 28.2 (3.1x) | 33.3 | 61.8 (1.9x) |
| 121 | 20 | off | 12.7 | 222.0 (17.5x) | 38.1 | 262.4 (6.9x) |
| 121 | 20 | **on** | 11.9 | 155.4 (13.1x) | 29.6 | 168.1 (5.7x) |
| 1006 | 4 | off | 37.5 | 146.9 (3.9x) | 58.1 | 178.7 (3.1x) |
| 1006 | 4 | **on** | 37.3 | 109.6 (2.9x) | 62.7 | 89.5 (1.4x) |
| 1006 | 20 | off | 42.9 | 838.5 (19.5x) | 72.4 | 874.5 (12.1x) |
| 1006 | 20 | **on** | 40.1 | 221.2 (5.5x) | 68.6 | 224.9 (3.3x) |

With caching off the numbers match the K-prefill model (K=20 at 1000 tokens: 19.5x raw). Turning on
`--enable-prefix-caching` is a server flag, not a code change, and gives the same effect as the transformers
shared-prefix path: 12x to 3.3x at K=20, 3.1x to 1.4x at K=4 on long states.

## What is left

After prefix sharing the remaining L0 overhead is the K option layouts themselves; at K=20 each layout is
about as long as a short state. Two levers on the roadmap: adaptive shifts (read 3 to 5 layouts instead of K
when the early ones already agree; opt-in today, see `bench.adaptive_table`) and shorter layouts (labels only,
option text in the shared prefix).
