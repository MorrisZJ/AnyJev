> Looking for the Jev-mode (L2) demo? `python -m demo.jev_mode --backend fake` runs it on the synthetic model in seconds; `--lifecycle` plays the deployment lifecycle (day 0 at L0, `observe()` solving the head at 30 labels, a rewording recentred from traffic, export / restart / load); `python -m demo.jev_mode` runs the shipped heads on a real model (Qwen3-4B on a GPU, Qwen3-1.7B on a CPU; `demo/jev_mode.py`).

# Games as decision benchmarks

Two games, one model, one typed question each. The game engine carries an oracle, so every
decision is scored, not only the final result. Both demos print a live text frame per turn
and a summary table per readout level; both run on a synthetic biased model without a GPU.

```
python -m demo.games.twenty48    --model Qwen/Qwen3-8B --games 5 --levels raw,L0,L1 --calibrate 300
python -m demo.games.minesweeper --model Qwen/Qwen3-8B --games 5 --levels raw,L0,L1 --calibrate 400
python -m demo.games.twenty48    --backend fake --games 2      # no GPU: synthetic model with a known position bias
```

## 2048: a 4-way choice under randomness

Question: `choice("Which move should be played next?", ["up", "down", "left", "right"])`.
The state is the board, the legal moves, and (by default) the board after each legal move,
before the random tile appears; the model compares outcomes instead of simulating slides.
The oracle is a depth-2 expectimax over the classic 2048 heuristic (smoothness, monotonicity,
empty cells, largest tile). Per policy the demo reports:

- **oracle agree**: share of turns whose played move is the expectimax-best;
- **regret**: (best − played) / (best − worst) in oracle value, 0 for the oracle's move;
- **flip**: share of turns where listing the four moves in reverse changes the answer;
- **ece / brier**: the move probabilities against the oracle's move;
- **illegal mass**: probability the readout put on moves the board did not allow;
- baselines on the same seeds: random, first-listed (a fully position-biased reader), oracle.

L1 fits a temperature on boards labelled by the oracle (no human labels). The reversed list
is calibrated separately, so its flip rate is the disagreement of two calibrated deployments.

## Minesweeper: the game is a probability

Question: `noul("Is the candidate cell (marked ?) safe to reveal?")`, once per candidate cell
(the frontier plus one cell no number touches). The cell with the highest P(safe) is clicked.
The exact posterior P(mine | revealed numbers, mine count) is enumerated for every candidate
(frontier components enumerated by backtracking, far cells share the remaining mines), so the
model's probabilities are compared with the true ones cell by cell:

- **safest pick**: share of turns where the clicked cell had the lowest exact P(mine);
- **regret**: exact P(mine) of the clicked cell minus that minimum;
- **avoidable deaths**: deaths on a cell that was not the safest available;
- **ece / brier**: P(safe) of every candidate against whether it was a mine;
- **|P − exact|**: mean gap between the model's P(safe) and the exact posterior;
- **board cleared / win rate**: coverage at the risk the model chose for itself.

The five-game runs behind the top-level README's Minesweeper sentence are
`../results/minesweeper_qwen3_8b_5g.json` and `../results/minesweeper_qwen3_32b_5g.json`.
The 2048 sentence there quotes `../results/twenty48_qwen3_8b_5g.json`; `../results/twenty48_qwen3_8b.json`
and `../results/twenty48_qwen3_8b_preview.json` are the earlier two-game runs on the first two of the
same seeds, kept only so the five-game run can be compared with them.

L1 fits a temperature on self-play cells labelled by the game (the oracle plays, every
candidate of every turn is one labelled example). The first click is free and fixed at the
centre, as in most solvers.

## Files

- `common.py`: backends (transformers or the synthetic `FakeBackend`), one `Policy` per
  level with the reversed-order probe, tallies, ANSI frames.
- `twenty48.py`, `minesweeper.py`: engine, oracle, state text, turn loop, summary.
- `../results/`: JSON written with `--json`; the tables in the top-level README come from these.
- Tests: `tests/test_demo_games.py` (engines, exact posterior against brute force, both demos on the fake model).
