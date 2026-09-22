| engine | goal test | goal ood | attempts | collisions | edge acc | majority | edge Brier | edge questions |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B + AnyJev L0 (batch prior) | 10/11 | 3/4 | 18880 | 7692 | 0.408 | 0.611 | 0.307 | 25876 |
| Qwen3-0.6B + AnyJev L0 (content_free prior) | 11/11 | 3/4 | 17973 | 7291 | 0.455 | 0.611 | 0.280 | 25152 |
| Qwen3-0.6B + AnyJev L0 (none prior) | 10/11 | 3/4 | 21851 | 9136 | 0.403 | 0.597 | 0.460 | 28040 |
| Qwen3-0.6B + AnyJev raw | 11/11 | 4/4 | 5825 | 2616 | 0.537 | 0.539 | 0.363 | 10944 |
| Qwen3-8B + AnyJev L0 (batch prior) | 11/11 | 2/4 | 19313 | 7771 | 0.550 | 0.600 | 0.349 | 25984 |
| Qwen3-8B + AnyJev raw | 10/11 | 3/4 | 20996 | 8570 | 0.523 | 0.606 | 0.392 | 28780 |
| Qwen3-0.6B native A/B readout (NanoJev's 'Untuned Qwen' protocol) | 10/11 | 3/4 | 20500 | 8436 | 0.421 | 0.606 | 0.305 | 27336 |

NanoJev scaled_maze episodes (test + ood), frozen exploration code from TianyuCodings/NanoJev; only the engine that answers 'is one step <dir> clear?' changes. edge acc/Brier: the model's Boolean answers against the true local geometry on the cells it visited; majority = accuracy of always answering the more common label on those same cells.
