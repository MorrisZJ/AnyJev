"""The game demos on the synthetic biased model: engines, oracles, and the raw-vs-L0 story."""
import json

import numpy as np

from demo.games import minesweeper as ms
from demo.games import twenty48 as t48


# ---------------------------------------------------------------- 2048 engine + oracle
def test_slide_merges_once_per_pair():
    assert t48.slide_left([2, 2, 2, 2]) == ([4, 4, 0, 0], 8)
    assert t48.slide_left([2, 0, 2, 4]) == ([4, 4, 0, 0], 4)
    assert t48.slide_left([4, 2, 2, 0]) == ([4, 4, 0, 0], 4)
    assert t48.slide_left([0, 0, 0, 0]) == ([0, 0, 0, 0], 0)


def test_apply_move_is_a_rotation_of_slide_left():
    b = [[2, 0, 0, 2], [0, 0, 0, 0], [4, 4, 0, 0], [0, 0, 0, 8]]
    up, _, _ = t48.apply_move(b, "up")
    assert [row[0] for row in up] == [2, 4, 0, 0]
    down, _, _ = t48.apply_move(b, "down")
    assert [row[3] for row in down] == [0, 0, 2, 8]
    right, gain, changed = t48.apply_move(b, "right")
    assert right[0] == [0, 0, 0, 4] and gain == 12 and changed
    assert not t48.apply_move([[2, 4, 8, 16]] * 1 + [[0] * 4] * 3, "up")[2]


def test_oracle_beats_random_and_labels_legal_moves_only():
    g = t48.Game2048(3)
    vals = t48.move_values(g.board, depth=2)
    assert set(vals) == set(t48.MOVES)
    assert all((v is None) == (m not in g.legal()) for m, v in vals.items())
    oracle = t48.baseline_game("oracle", 3, depth=2, max_moves=150)
    rnd = t48.baseline_game("random", 3, depth=2, max_moves=150)
    assert oracle["score"] > rnd["score"]


def test_2048_demo_l0_removes_the_position_bias(tmp_path, capsys):
    out = tmp_path / "t48.json"
    t48.main(["--backend", "fake", "--games", "1", "--max-moves", "80", "--no-live", "--every", "1000",
              "--levels", "raw,L0", "--no-baselines", "--json", str(out)])
    s = json.loads(out.read_text())["summary"]
    raw, l0 = s["raw · fake"], s["L0 · fake"]
    # the synthetic model prefers the first-listed move: raw flips under reversal, L0 does not
    assert raw["flip"] > 0.3 and l0["flip"] == 0.0
    assert l0["agree"] > raw["agree"]
    assert "oracle agree" in capsys.readouterr().out


# ---------------------------------------------------------------- minesweeper engine + exact posterior
def test_first_click_is_safe_and_flood_fills():
    g = ms.Minesweeper(8, 8, 10, seed=1)
    assert g.reveal((4, 4)) and (4, 4) in g.revealed and (4, 4) not in g.mines
    assert len(g.mines) == 10 and not (set(g.neighbors((4, 4))) & g.mines)
    assert len(g.revealed) >= 9   # the free click and its 8 neighbours at least


def test_exact_posterior_matches_brute_force():
    """A tiny board whose configurations we can enumerate directly."""
    g = ms.Minesweeper(3, 4, 2, seed=5)
    g.reveal((0, 0))
    probs, exact = ms.mine_probabilities(g)
    assert exact
    hidden = g.hidden()
    # brute force: every placement of the mines among the hidden cells consistent with the numbers
    import itertools
    consistent = []
    for combo in itertools.combinations(hidden, g.n_mines):
        mines = set(combo)
        if all(sum(1 for n in g.neighbors(x) if n in mines) == g.number(x) for x in g.revealed):
            consistent.append(mines)
    assert consistent
    for x in hidden:
        p = sum(1 for mines in consistent if x in mines) / len(consistent)
        assert abs(probs[x] - p) < 1e-9, (x, probs[x], p)
    assert abs(sum(probs.values()) - g.n_mines) < 1e-9


def test_oracle_policy_rarely_dies():
    wins = [ms.baseline_game("oracle", _Args(), seed)["cleared"] for seed in range(6)]
    assert np.mean(wins) > 0.8


class _Args:
    rows, cols, mines, seed = 8, 8, 10, 0


def test_minesweeper_demo_runs_on_the_fake_model(tmp_path):
    out = tmp_path / "ms.json"
    ms.main(["--backend", "fake", "--games", "1", "--no-live", "--every", "1000", "--levels", "raw,L0,L1",
             "--calibrate", "60", "--no-baselines", "--json", str(out)])
    rec = json.loads(out.read_text())
    assert set(rec["summary"]) == {"raw · fake", "L0 · fake", "L1 · fake"}
    for s in rec["summary"].values():
        assert s["decisions"] > 0 and 0 <= s["ece"] <= 1
