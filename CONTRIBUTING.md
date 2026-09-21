# Contributing

Thanks for looking. The fastest way to contribute is to add one file.

## Add a backend (`anyjev/backends/<engine>.py`)

Implement one method:

```python
def next_token_logprobs(self, prompts: Sequence[str], token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]
```

For prompt `i`, return the log-probability of each id in `token_ids[i]` at the next position, from the full-vocabulary log-softmax. Expose `.tokenizer` (needs `.encode(text, add_special_tokens=False)`; a chat template is used if present) and `.name`. Nothing else goes in a backend: debiasing and calibration live above it and are tested once. Add a parity check against `HFBackend` like `scripts/vllm_parity.py`, marked `@pytest.mark.engine` if it needs the engine.

## Add a bench task (`bench/tasks/<task>.py`)

Register a loader that returns a `Task` with a `Question` and `(state, label_index)` items, the dataset license and source. Datasets are downloaded at run time, never vendored. Add the license to `THIRD_PARTY.md`.

## Add a bench provider (`bench/providers/<project>.py`)

Run another open decision model on the same decisions (`bench/tasks/typed_decisions.py`) and write a results JSON that `bench/typed_table.py` can render. See `bench/providers/laya.py`.

## Rules

- **No fabricated numbers.** A table row is a committed JSON under `bench/results*/` plus the command that produced it, with hardware and library versions (the runners record them). If a run did not happen, the cell is empty.
- **No hidden generation.** A code path that samples tokens in decision mode is a bug.
- **Level is mandatory.** Every `Decision` carries `level`; tests assert it.
- **Small PRs.** One thing per PR, under about 400 lines excluding tests and fixtures.
- **Tests before features.** Calibration methods land with a unit test on synthetic logits where the answer is known analytically (`anyjev/backends/fake.py` makes this easy).
- **Compare fairly.** When a table names another project, run it on identical inputs with its own recommended settings, say which protocol, and link the file that produced the number.

`pip install -e ".[dev]"` and `pytest` runs the whole suite on CPU in under a second.
