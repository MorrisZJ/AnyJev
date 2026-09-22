"""The maze bench: generator, explorer, and the play loop on a synthetic backend."""
from collections import deque

import pytest

from anyjev.backends.fake import FakeBackend
from bench.games.maze import DIRS, Explorer, Maze
from bench.run_maze import play, prepare


def reachable(m: Maze):
    seen, q = {m.start}, deque([m.start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (r + dr, c + dc)
            if m.is_open(n) and n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def run(m: Maze, p_open, **kw):
    ex = Explorer(m.size, m.start, m.goal, **kw)
    while not (status := ex.done(m)):
        if ex.needs_prediction():
            ex.remember(p_open(ex.pos))
        ex.step(m)
    return status, ex


def test_generator_is_seeded_bordered_and_solvable():
    a, b = Maze.generate(21, seed=3), Maze.generate(21, seed=3)
    assert a.walls == b.walls and a.walls != Maze.generate(21, seed=4).walls
    assert all(a.walls[0]) and all(a.walls[-1]) and all(row[0] and row[-1] for row in a.walls)
    assert a.goal in reachable(a)
    with pytest.raises(ValueError):
        Maze.generate(20, seed=0)


def test_view_is_centred_and_walls_outside():
    m = Maze.generate(11, seed=0)
    rows = [line.split(": ", 1)[1].split() for line in m.view(m.start).splitlines()[1:]]
    assert rows[2][2] == "@" and rows[0] == ["#"] * 5          # row -2 is outside the maze
    for d, (r, c) in {"north": (1, 2), "east": (2, 3), "south": (3, 2), "west": (2, 1)}.items():
        assert (rows[r][c] == ".") == m.truth(m.start)[d]


def test_oracle_never_collides_and_bias_does():
    m = Maze.generate(21, seed=1)
    oracle = lambda c: {d: 1.0 if v else 0.0 for d, v in m.truth(c).items()}   # noqa: E731
    for policy in ("expected_cost", "threshold"):
        status, ex = run(m, oracle, policy=policy)
        assert status == "goal" and ex.collisions == 0
    status, ex = run(m, lambda c: {d: 0.9 for d in DIRS}, wall_cost=0)
    assert status == "goal" and ex.collisions > 0
    assert ex.steps == len(ex.log) + ex.wall_cost * ex.collisions == ex.log[-1]["t"]


def test_explorer_recovers_from_a_model_that_says_wall_everywhere():
    m = Maze.generate(15, seed=2)
    status, ex = run(m, lambda c: {d: 0.1 for d in DIRS}, wall_cost=0)
    assert status == "goal"


def test_explorer_refuses_to_move_blind():
    m = Maze.generate(11, seed=0)
    with pytest.raises(RuntimeError):
        Explorer(m.size, m.start, m.goal).step(m)


@pytest.mark.parametrize("level", ["raw", "L0", "L1"])
def test_play_records_level_and_every_prediction(level):
    m = Maze.generate(11, seed=0)
    views = [Maze.generate(11, s).view(c) for s in (100, 101) for c in Maze.generate(11, s).open_cells()[:10]]
    truth = [Maze.generate(11, s).truth(c) for s in (100, 101) for c in Maze.generate(11, s).open_cells()[:10]]
    be = FakeBackend(lambda state, opt: 0.0, label_prior={"Yes": 1.5})
    run_ = play(prepare(be, level, views, truth, "batch"), level, m, policy="expected_cost", wall_cost=0)
    assert run_["level"] == level and run_["status"] == "goal"
    assert run_["cells_seen"] == len(run_["predictions"]) > 0
    assert sum(e["hit"] for e in run_["moves"]) == run_["collisions"]
