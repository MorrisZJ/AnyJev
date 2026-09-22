"""A seeded grid maze, a text view of the agent's surroundings, and an explorer
that moves only on the model's answers.

The explorer never reads the maze. At every newly reached cell it gets four
probabilities, P(the cell one step north / east / south / west is open floor),
and it only learns the truth by trying a move: a wall costs a collision, an
opening moves it. Everything the model is judged on is in `view()`.

    m = Maze.generate(51, seed=0)
    ex = Explorer(m.size, m.start, m.goal)
    while not ex.done(m):
        if ex.needs_prediction():
            ex.remember(p_open_from_the_model(m.view(ex.pos)))
        ex.step(m)
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]
DIRS = ("north", "east", "south", "west")
DELTA = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}


def _add(c: Cell, d: str) -> Cell:
    return (c[0] + DELTA[d][0], c[1] + DELTA[d][1])


@dataclass
class Maze:
    """walls[r][c] is True for a wall tile. Tiles at odd (r, c) are rooms."""

    size: int
    walls: List[List[bool]]
    start: Cell
    goal: Cell
    seed: int

    @staticmethod
    def generate(size: int, seed: int, loops: float = 0.04) -> "Maze":
        """Recursive-backtracker maze on an odd `size`, then knock out a
        `loops` fraction of the interior walls that separate two rooms so
        there is more than one route."""
        if size < 5 or size % 2 == 0:
            raise ValueError("size must be odd and >= 5")
        rng = random.Random(seed)
        walls = [[True] * size for _ in range(size)]
        start = (1, 1)
        walls[1][1] = False
        stack = [start]
        while stack:
            r, c = stack[-1]
            nbrs = [(r + 2 * dr, c + 2 * dc, dr, dc) for dr, dc in DELTA.values()
                    if 0 < r + 2 * dr < size - 1 and 0 < c + 2 * dc < size - 1
                    and walls[r + 2 * dr][c + 2 * dc]]
            if not nbrs:
                stack.pop()
                continue
            nr, nc, dr, dc = rng.choice(nbrs)
            walls[r + dr][c + dc] = False
            walls[nr][nc] = False
            stack.append((nr, nc))
        inner = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)
                 if walls[r][c] and (r % 2) != (c % 2)]
        for r, c in rng.sample(inner, int(loops * len(inner))):
            walls[r][c] = False
        return Maze(size, walls, start, (size - 2, size - 2), seed)

    def is_open(self, c: Cell) -> bool:
        r, q = c
        return 0 <= r < self.size and 0 <= q < self.size and not self.walls[r][q]

    def open_cells(self) -> List[Cell]:
        return [(r, c) for r in range(self.size) for c in range(self.size) if not self.walls[r][c]]

    def truth(self, c: Cell) -> Dict[str, bool]:
        return {d: self.is_open(_add(c, d)) for d in DIRS}

    def view(self, c: Cell, radius: int = 2) -> str:
        """The (2r+1)x(2r+1) window around `c` as text. Outside the maze is wall."""
        lines = [f"A {2 * radius + 1}x{2 * radius + 1} window of a grid maze, centred on you. "
                 "'#' is wall, '.' is open floor, '@' is you. North is up, east is right."]
        for dr in range(-radius, radius + 1):
            row = []
            for dc in range(-radius, radius + 1):
                cell = (c[0] + dr, c[1] + dc)
                row.append("@" if dr == dc == 0 else ("." if self.is_open(cell) else "#"))
            lines.append(f"row {dr:+d}: " + " ".join(row))
        return "\n".join(lines)


@dataclass
class Explorer:
    """Verified-edge exploration driven by P(open) per direction.

    A frontier edge is an untried direction, at a cell the model has seen,
    that leads somewhere unvisited. Two policies pick the next one:

    - "expected_cost": the edge with the fewest expected steps per new cell,
      (walk there + 1) / P(open), taken over the whole known map; the explorer
      commits to it until it is tried. Nothing is ruled out, a low P(open)
      only makes an edge expensive, so the probabilities have to be comparable
      across cells and directions for it to choose well.
    - "threshold": edges with P(open) >= `threshold` first (nearest cell,
      then most likely), the rest only once those run out. A wrong "wall"
      hides a corridor until then.

    A wrong "open" costs a collision: the try itself plus `wall_cost` steps
    of penalty. With wall_cost = 0, trying every direction blind is hard to
    beat and perception barely matters; with a real penalty the explorer has
    to trust the probabilities, and expected_cost prices a try at
    (walk + 1 + wall_cost * (1 - P)) / P.
    """

    size: int
    pos: Cell
    goal: Cell
    threshold: float = 0.5
    policy: str = "expected_cost"
    wall_cost: int = 10
    horizon: Optional[int] = None
    target: Optional[Tuple[Cell, str]] = None
    steps: int = 0
    collisions: int = 0
    preds: Dict[Cell, Dict[str, float]] = field(default_factory=dict)
    tried: Set[Tuple[Cell, str]] = field(default_factory=set)
    open_edges: Dict[Cell, Set[str]] = field(default_factory=dict)
    visited: Set[Cell] = field(default_factory=set)
    log: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.policy not in ("expected_cost", "threshold"):
            raise ValueError("policy must be 'expected_cost' or 'threshold'")
        self.visited.add(self.pos)
        if self.horizon is None:
            self.horizon = 4 * self.size * self.size

    # ---- the model's side -------------------------------------------
    def needs_prediction(self) -> bool:
        return self.pos not in self.preds

    def remember(self, p_open: Dict[str, float]) -> None:
        self.preds[self.pos] = {d: float(p_open[d]) for d in DIRS}

    # ---- the environment's side ---------------------------------------
    def done(self, maze: Maze) -> Optional[str]:
        if self.pos == maze.goal:
            return "goal"
        if self.steps >= self.horizon:
            return "horizon"
        if self.needs_prediction():
            return None
        if self._next() is None:
            return "stuck"
        return None

    def _frontier(self, c: Cell, confident: Optional[bool] = None) -> List[str]:
        p = self.preds.get(c)
        if p is None:
            return []
        out = [d for d in DIRS if (c, d) not in self.tried and _add(c, d) not in self.visited
               and (confident is None or (p[d] >= self.threshold) == confident)]
        return sorted(out, key=lambda d: -p[d])

    def _routes(self) -> Dict[Cell, Tuple[int, Optional[str]]]:
        """BFS over verified-open edges: cell -> (distance, first move from pos)."""
        routes = {self.pos: (0, None)}
        queue = deque([self.pos])
        while queue:
            c = queue.popleft()
            dist, first = routes[c]
            for d in sorted(self.open_edges.get(c, ())):
                n = _add(c, d)
                if n not in routes:
                    routes[n] = (dist + 1, first or d)
                    queue.append(n)
        return routes

    def _next(self):
        """(first move, is_try) under the current policy, or None when nothing is left."""
        if self.policy == "threshold":
            return self._plan(confident=True) or self._plan(confident=False)
        if self.target is None or self.target in self.tried or _add(*self.target) in self.visited:
            routes, best, self.target = self._routes(), None, None
            for c, (dist, _) in routes.items():
                for d in self._frontier(c):
                    p = max(self.preds[c][d], 1e-6)
                    cost = (dist + 1 + self.wall_cost * (1 - p)) / p
                    if best is None or cost < best:
                        best, self.target = cost, (c, d)
            if self.target is None:
                return None
        cell, d = self.target
        if cell == self.pos:
            return d, True
        return self._routes()[cell][1], False

    def _plan(self, confident: bool):
        """(first move, is_try) toward the nearest cell with a frontier edge of this kind."""
        here = self._frontier(self.pos, confident)
        if here:
            return here[0], True
        seen, queue = {self.pos: None}, deque([self.pos])
        while queue:
            c = queue.popleft()
            if c != self.pos and self._frontier(c, confident):
                while seen[c][0] != self.pos:
                    c = seen[c][0]
                return seen[c][1], False
            for d in sorted(self.open_edges.get(c, ())):
                n = _add(c, d)
                if n not in seen:
                    seen[n] = (c, d)
                    queue.append(n)
        return None

    def step(self, maze: Maze) -> dict:
        if self.needs_prediction():
            raise RuntimeError("the explorer needs P(open) for this cell before it can move")
        plan = self._next()
        if plan is None:
            raise RuntimeError("nothing left to try")
        d, is_try = plan
        target = _add(self.pos, d)
        self.steps += 1
        self.tried.add((self.pos, d))
        if maze.is_open(target):
            back = DIRS[(DIRS.index(d) + 2) % 4]
            self.open_edges.setdefault(self.pos, set()).add(d)
            self.open_edges.setdefault(target, set()).add(back)
            self.tried.add((target, back))
            self.pos = target
            self.visited.add(target)
            event = {"dir": d, "kind": "try" if is_try else "walk", "pos": list(target), "hit": False}
        else:
            self.collisions += 1
            self.steps += self.wall_cost
            event = {"dir": d, "kind": "try", "pos": list(self.pos), "hit": True}
        event["t"] = self.steps
        self.log.append(event)
        return event
