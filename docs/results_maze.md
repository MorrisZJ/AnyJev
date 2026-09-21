| engine | goal test | goal ood | attempts | collisions | edge acc | majority | edge Brier | edge questions |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B + AnyJev L0 (batch prior) | 10/11 | 3/4 | 15616 | 6236 | 0.490 | 0.618 | 0.271 | 23156 |
| Qwen3-0.6B + AnyJev L0 (content_free prior) | 10/11 | 4/4 | 16278 | 6563 | 0.490 | 0.614 | 0.274 | 23284 |
| Qwen3-0.6B + AnyJev L0 (none prior) | 10/11 | 3/4 | 21851 | 9136 | 0.403 | 0.597 | 0.460 | 28040 |
| Qwen3-0.6B + AnyJev raw | 11/11 | 4/4 | 5825 | 2616 | 0.537 | 0.539 | 0.363 | 10944 |
| Qwen3-8B + AnyJev L0 (batch prior) | 11/11 | 3/4 | 17841 | 7171 | 0.555 | 0.600 | 0.344 | 25124 |
| Qwen3-8B + AnyJev raw | 10/11 | 3/4 | 20996 | 8570 | 0.523 | 0.606 | 0.392 | 28780 |
| Qwen3-0.6B native A/B readout (NanoJev's 'Untuned Qwen' protocol) | 10/11 | 3/4 | 20555 | 8496 | 0.419 | 0.607 | 0.305 | 27660 |

NanoJev scaled_maze episodes (11 test + 4 ood, sizes 8 to 50), frozen exploration code and pinned Qwen3-0.6B revision from TianyuCodings/NanoJev; only the engine that answers 'is one step <dir> clear?' changes. This is the scaled_maze pipeline from their docs/SCALED_GAMES.md, not the 274-case held-out gameplay suite where their README reports Untuned Qwen3-0.6B at 2/10; those numbers are not comparable to these. NanoJev publishes no untuned-Qwen row on the scaled suite, so the last row is our reimplementation of their readout protocol. edge acc/Brier: the model's Boolean answers against the true local geometry on the cells it visited; majority = accuracy of always answering the more common label on those same cells.
