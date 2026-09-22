# Adaptive cyclic shifts: what early stopping costs

Opt-in: `Decider(adaptive_shifts=True, adaptive_min_shifts=2, adaptive_margin=0.1, adaptive_order="spread")`.
The shifts are read one at a time; after each, every shift read so far is prior-corrected and the item stops
when they all agree on the winner and the running marginal's top-1 minus top-2 probability clears the
margin. The marginal is then an average over the shifts actually read. Two tasks (K=20, n=300), full L0 as
the reference, batch prior, one H100. Regenerate with `bench.run --adaptive` and `bench.adaptive_table`.

## Reading order matters more than the margin

| model | task | full: shifts / acc / ECE / flip | consecutive order: shifts / acc / ECE / flip | spread order: shifts / acc / ECE / flip |
|---|---|---|---|---|
| Qwen3-1.7B | banking20 | 20 / 0.673 / 0.299 / 0.187 | 4.8 / 0.640 / 0.341 / 0.270 | 7.8 / 0.677 / 0.298 / 0.200 |
| Qwen3-1.7B | newsgroups | 20 / 0.610 / 0.351 / 0.237 | 4.2 / 0.613 / 0.362 / 0.277 | 6.6 / 0.613 / 0.355 / 0.247 |
| SmolLM2-1.7B | banking20 | 20 / 0.737 / 0.583 / 0.147 | 13.5 / 0.760 / 0.574 / 0.263 | 14.3 / 0.733 / 0.553 / 0.183 |
| SmolLM2-1.7B | newsgroups | 20 / 0.643 / 0.481 / 0.190 | 12.7 / 0.630 / 0.432 / 0.243 | 15.8 / 0.647 / 0.482 / 0.223 |

`shifts` is the mean number of layouts read per item (full = K = 20). Margins of 0.05, 0.1 and 0.2 gave
identical results: these models are overconfident enough that the margin never binds, and stopping is decided
by winner agreement alone.

**Consecutive order** (shift 0, 1, 2, ...) stops early and pays for it: each shift moves every option by one
position, so the first few layouts are nearly the same and agree for the wrong reason. On Qwen3-1.7B it read
4.8 of 20 shifts but lost 3.3 accuracy points and gave back 8 points of the flip-rate reduction.

**Spread order** (shift 0, K/2, K/4, 3K/4, ..., the van der Corput sequence) puts each option in well separated
positions from the start. It stops less eagerly, 6.6 to 7.8 shifts on Qwen3-1.7B, and keeps accuracy within
0.4 points and flip rate within 1 to 1.3 points of the full cycle. That is a 2.6x to 3x cut in prefills at
essentially no loss, and it is the default order.

SmolLM2 barely stops early under either order (13 to 16 shifts): its answers are genuinely order-unstable, so
the shifts keep disagreeing. The stopping rule is reporting that honestly rather than saving compute on a model
that needs the full cycle.

## Where it fits

Prefix sharing (transformers `score_shared`, vLLM prefix caching) removes the repeated state; adaptive shifts
remove most of the repeated option layouts. Together on a 1000-token state at K=20 the arithmetic is roughly
one prefill plus 7 short suffix reads, against 20 full prefills for a naive L0. It stays opt-in because the
marginal over a subset of shifts is an approximation; when you need the exact cancellation of position bias
(the probabilities, not just the argmax), read the full cycle.
