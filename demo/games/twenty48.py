"""2048 as a decision benchmark: a 4-way choice under randomness, scored by expectimax.

Every turn the model sees the board and picks a direction. The raw next-token readout lists
the four moves in a fixed order and inherits the model's position bias: it plays the
first-listed move far more often than the board warrants, and its answer changes when the
list is reversed. AnyJev L0 reads the same model through every cyclic shift of the list, so
its answer cannot depend on the order. A depth-2 expectimax over the classic 2048 heuristic
labels the best move of every turn, so the demo reports decision quality, not only score:
oracle agreement, regret, flip rate under option reversal, and calibration of the move
probabilities against the oracle's choice.

    python -m demo.games.twenty48 --model Qwen/Qwen3-8B --games 3
    python -m demo.games.twenty48 --model Qwen/Qwen3-8B --levels raw,L0,L1 --calibrate 300
    python -m demo.games.twenty48 --backend fake --games 2 --no-live     # synthetic model, CPU
"""
from __future__ import annotations

import argparse
import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anyjev import Question
from demo.games.common import (
    FakeContent,
    Policy,
    Screen,
    Tally,
    add_common_args,
    bar,
    fmt_table,
    make_backend,
    make_decider,
    paint,
    parse_levels,
    running_line,
    side_by_side,
    write_json,
)

MOVES = ("up", "down", "left", "right")
QUESTION = Question.choice("Which move should be played next?", MOVES, name="move")
Board = List[List[int]]


# ---------------------------------------------------------------- engine
def slide_left(row: Sequence[int]) -> Tuple[List[int], int]:
    tiles = [x for x in row if x]
    out: List[int] = []
    gain, i = 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gain += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (len(row) - len(out)), gain


def apply_move(board: Board, move: str) -> Tuple[Board, int, bool]:
    """Returns (new board, points gained, whether anything moved)."""
    n = len(board)
    if move == "left":
        lines = [list(r) for r in board]
    elif move == "right":
        lines = [list(reversed(r)) for r in board]
    elif move == "up":
        lines = [[board[r][c] for r in range(n)] for c in range(n)]
    elif move == "down":
        lines = [[board[r][c] for r in reversed(range(n))] for c in range(n)]
    else:
        raise ValueError(move)
    slid, gain = [], 0
    for ln in lines:
        s, g = slide_left(ln)
        slid.append(s)
        gain += g
    if move == "left":
        nb = slid
    elif move == "right":
        nb = [list(reversed(r)) for r in slid]
    elif move == "up":
        nb = [[slid[c][r] for c in range(n)] for r in range(n)]
    else:
        nb = [[slid[c][n - 1 - r] for c in range(n)] for r in range(n)]
    return nb, gain, nb != [list(r) for r in board]


class Game2048:
    def __init__(self, seed: int, size: int = 4):
        self.rng = random.Random(seed)
        self.size = size
        self.board: Board = [[0] * size for _ in range(size)]
        self.score = 0
        self.moves = 0
        self.spawn()
        self.spawn()

    def spawn(self) -> None:
        empty = [(r, c) for r in range(self.size) for c in range(self.size) if not self.board[r][c]]
        if empty:
            r, c = self.rng.choice(empty)
            self.board[r][c] = 4 if self.rng.random() < 0.1 else 2

    def legal(self) -> List[str]:
        return [m for m in MOVES if apply_move(self.board, m)[2]]

    @property
    def over(self) -> bool:
        return not self.legal()

    @property
    def max_tile(self) -> int:
        return max(v for row in self.board for v in row)

    def play(self, move: str) -> int:
        nb, gain, changed = apply_move(self.board, move)
        if not changed:
            raise ValueError(f"illegal move {move}")
        self.board = nb
        self.score += gain
        self.moves += 1
        self.spawn()
        return gain


# ---------------------------------------------------------------- oracle: expectimax on the ov3y heuristic
def _lines(board: Board) -> List[List[float]]:
    n = len(board)
    lg = [[math.log2(v) if v else 0.0 for v in row] for row in board]
    return lg + [[lg[r][c] for r in range(n)] for c in range(n)]


def heuristic(board: Board) -> float:
    """Smoothness, monotonicity, empty cells and the largest tile, weighted as in the widely
    copied 2048 AI (ov3y): 0.1, 1.0, 2.7, 1.0, all on log2 tile values."""
    n = len(board)
    empty = sum(1 for row in board for v in row if v == 0)
    smooth = 0.0
    mono = 0.0
    all_lines = _lines(board)
    for lines in (all_lines[:n], all_lines[n:]):
        dec = inc = 0.0
        for ln in lines:
            # smoothness: each tile against the next non-empty tile along the line
            for i in range(n):
                if ln[i]:
                    j = i + 1
                    while j < n and not ln[j]:
                        j += 1
                    if j < n:
                        smooth -= abs(ln[i] - ln[j])
            # monotonicity: walk consecutive non-empty tiles, accumulate drops and rises
            cur, nxt = 0, 1
            while nxt < n:
                while nxt < n and not ln[nxt]:
                    nxt += 1
                if nxt >= n:
                    nxt -= 1
                a, b = ln[cur], ln[nxt]
                if a > b:
                    dec += b - a
                elif b > a:
                    inc += a - b
                cur, nxt = nxt, nxt + 1
        mono += max(dec, inc)
    biggest = max(v for row in board for v in row)
    return 0.1 * smooth + 1.0 * mono + 2.7 * empty + 1.0 * (math.log2(biggest) if biggest else 0.0)


def _best_after_spawn(board: Board, depth: int) -> float:
    best = None
    for m in MOVES:
        nb, _, changed = apply_move(board, m)
        if not changed:
            continue
        v = heuristic(nb) if depth <= 1 else _chance(nb, depth - 1)
        best = v if best is None else max(best, v)
    return best if best is not None else heuristic(board) - 100.0   # dead board


def _chance(board: Board, depth: int) -> float:
    empty = [(r, c) for r in range(len(board)) for c in range(len(board)) if not board[r][c]]
    if not empty or depth <= 0:
        return heuristic(board)
    total = 0.0
    for r, c in empty:
        for v, p in ((2, 0.9), (4, 0.1)):
            board[r][c] = v
            total += p * _best_after_spawn(board, depth)
            board[r][c] = 0
    return total / len(empty)


def move_values(board: Board, depth: int = 2) -> Dict[str, Optional[float]]:
    """Expectimax value of each move (None if illegal). depth=2: move, spawn, move, evaluate."""
    out: Dict[str, Optional[float]] = {}
    for m in MOVES:
        nb, _, changed = apply_move(board, m)
        out[m] = _chance(nb, depth - 1) if changed else None
    return out


# ---------------------------------------------------------------- the state the model sees
def state_text(g: Game2048, legal: Sequence[str], preview: bool = True) -> str:
    """The board as text. With `preview`, the board after each legal move (before the random
    spawn) is shown too, so the model compares outcomes instead of simulating slides in its
    head; the spawn stays unknown, so the choice is still a decision under randomness."""
    n = g.size
    lines = [f"2048 on a {n}x{n} board. A move slides every tile in that direction; two equal tiles "
             f"that collide merge into their sum, then a new 2 (or 4) appears in a random empty cell. "
             f"Score so far: {g.score}.",
             "Board, top row first (0 = empty):"]
    lines += [" ".join(f"{v:5d}" for v in row) for row in g.board]
    lines.append("Legal moves this turn: " + ", ".join(legal) + ".")
    if preview:
        for m in legal:
            nb, gain, _ = apply_move(g.board, m)
            empty = sum(1 for row in nb for v in row if not v)
            lines.append(f"After {m} (+{gain} points, {empty} empty cells, before the new tile appears):")
            lines += ["    " + " ".join(f"{v:5d}" for v in row) for row in nb]
    lines.append("Good play keeps the largest tiles together in one corner, keeps each row and column "
                 "ordered, and keeps empty cells.")
    return "\n".join(lines)


def fake_logits(vals: Dict[str, Optional[float]]) -> Dict[str, float]:
    """What the synthetic model 'knows': the oracle's move values, standardised; illegal moves low."""
    lv = [v for v in vals.values() if v is not None]
    mu, sd = float(np.mean(lv)), float(np.std(lv)) + 1e-6
    return {m: (max(-3.0, min(3.0, 1.5 * (v - mu) / sd)) if v is not None else -4.0) for m, v in vals.items()}


# ---------------------------------------------------------------- one turn
def turn(policy: Policy, g: Game2048, depth: int, tally: Tally, fake: Optional[FakeContent],
         preview: bool = True) -> Dict[str, Any]:
    vals = move_values(g.board, depth)
    legal = [m for m in MOVES if vals[m] is not None]
    st = state_text(g, legal, preview)
    if fake is not None:
        fake.set(st, fake_logits(vals))
    probs, _, flip = policy.decide(st)
    mask = np.array([1.0 if vals[m] is not None else 0.0 for m in MOVES])
    masked = probs * mask
    illegal_mass = float(1.0 - masked.sum())
    masked = masked / masked.sum() if masked.sum() > 0 else mask / mask.sum()
    chosen = MOVES[int(np.argmax(masked))]
    lv = [vals[m] for m in legal]
    best = legal[int(np.argmax(lv))]
    spread = max(lv) - min(lv)
    regret = (vals[best] - vals[chosen]) / spread if spread > 1e-9 else 0.0
    tally.add(masked, MOVES.index(best), chosen == best, regret, flip, illegal_mass=illegal_mass)
    g.play(chosen)
    return {"probs": probs, "masked": masked, "chosen": chosen, "best": best, "flip": flip, "legal": legal}


# ---------------------------------------------------------------- rendering
TILE_STYLE = {0: (240, 236), 2: (238, 230), 4: (238, 229), 8: (231, 215), 16: (231, 209), 32: (231, 203),
              64: (231, 196), 128: (238, 227), 256: (238, 226), 512: (238, 220), 1024: (231, 214),
              2048: (231, 208)}


def render_board(board: Board) -> List[str]:
    out = []
    for row in board:
        cells = []
        for v in row:
            fg, bg = TILE_STYLE.get(v, (231, 201))
            cells.append(paint(f"{v:^6d}" if v else "   ·  ", fg=fg, bg=bg, bold=v >= 128))
        out.append("".join(cells))
    return out


def render_policy(p: Policy, g: Game2048, info: Optional[Dict[str, Any]], tally: Tally) -> List[str]:
    lines = [paint(f" {p.name} ", bold=True, bg=24, fg=231) + ("" if not g.over else paint("  game over", fg=203)),
             f"score {g.score:,}   max {g.max_tile}   move {g.moves}"]
    lines += render_board(g.board)
    if info is None:
        return lines
    for i, m in enumerate(MOVES):
        p_ = float(info["probs"][i])
        tag = ""
        if m == info["chosen"]:
            tag += paint(" ◀ played", fg=45, bold=True)
        if m == info["best"]:
            tag += paint(" ★ oracle", fg=220)
        if m not in info["legal"]:
            tag += paint(" (illegal)", fg=240)
        lines.append(f"{m:<6s}{bar(p_)} {p_:.2f}{tag}")
    if info["flip"] is not None:
        lines.append("reversed list: " + (paint("FLIP", fg=203, bold=True) if info["flip"] else paint("same", fg=114)))
    lines.append(paint(running_line(tally), fg=245))
    return lines


def frame(policies, games, last, tallies, step, seed, elapsed) -> List[str]:
    cols = [render_policy(p, games[p.name], last[p.name], tallies[p.name]) for p in policies]
    head = paint(f"2048  seed {seed}  step {step}  {elapsed:.0f}s", bold=True)
    return [head, ""] + side_by_side(cols)


# ---------------------------------------------------------------- games
def race(policies: List[Policy], seed: int, depth: int, max_moves: int, screen: Screen,
         tallies: Dict[str, Tally], fake: Optional[FakeContent], preview: bool = True) -> Dict[str, Any]:
    games = {p.name: Game2048(seed) for p in policies}
    last: Dict[str, Optional[Dict[str, Any]]] = {p.name: None for p in policies}
    step, t0 = 0, time.time()
    while step < max_moves and any(not g.over for g in games.values()):
        step += 1
        for p in policies:
            if not games[p.name].over:
                last[p.name] = turn(p, games[p.name], depth, tallies[p.name], fake, preview)
        screen.frame(frame(policies, games, last, tallies, step, seed, time.time() - t0), step)
    screen.frame(frame(policies, games, last, tallies, step, seed, time.time() - t0), step, final=True)
    return {"seed": seed, "steps": step,
            **{p.name: {"score": games[p.name].score, "max_tile": games[p.name].max_tile,
                        "moves": games[p.name].moves} for p in policies}}


def baseline_game(kind: str, seed: int, depth: int, max_moves: int) -> Dict[str, Any]:
    g = Game2048(seed)
    rng = random.Random(seed + 7)
    while not g.over and g.moves < max_moves:
        legal = g.legal()
        if kind == "oracle":
            vals = move_values(g.board, depth)
            m = max(legal, key=lambda x: vals[x])
        elif kind == "first-listed":
            m = legal[0]
        else:
            m = rng.choice(legal)
        g.play(m)
    return {"score": g.score, "max_tile": g.max_tile, "moves": g.moves}


def calibration_set(n: int, seed: int, depth: int, fake: Optional[FakeContent], preview: bool = True,
                    eps: float = 0.3):
    """Oracle-labelled boards from epsilon-noisy oracle play: the label of a board is the
    expectimax-best move. No human labels anywhere."""
    states, labels = [], []
    rng = random.Random(seed + 10_000)
    game_seed = seed + 10_000
    while len(states) < n:
        g = Game2048(game_seed)
        game_seed += 1
        while not g.over and len(states) < n:
            vals = move_values(g.board, depth)
            legal = [m for m in MOVES if vals[m] is not None]
            st = state_text(g, legal, preview)
            if fake is not None:
                fake.set(st, fake_logits(vals))
            best = max(legal, key=lambda x: vals[x])
            states.append(st)
            labels.append(MOVES.index(best))
            g.play(rng.choice(legal) if rng.random() < eps else best)
    return states, labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_games=3)
    ap.add_argument("--oracle-depth", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=1500)
    ap.add_argument("--no-baselines", action="store_true", help="skip random / first-listed / oracle games")
    ap.add_argument("--no-preview", dest="preview", action="store_false", default=True,
                    help="show only the current board, not the board after each legal move")
    args = ap.parse_args(argv)
    levels = parse_levels(args)

    fake = FakeContent() if args.backend == "fake" else None
    backend = make_backend(args, fake)
    decider = make_decider(args, backend)
    label = "fake" if fake else args.model.split("/")[-1]
    policies = [Policy(f"{lv} · {label}", decider, lv, QUESTION) for lv in levels]
    if "L1" in levels:
        t0 = time.time()
        states, labels_ = calibration_set(args.calibrate, args.seed, args.oracle_depth, fake, args.preview)
        art = next(p for p in policies if p.level == "L1").calibrate(states, labels_)
        print(f"L1: temperature {art['temperature']:.2f} fit on {len(states)} oracle-labelled boards "
              f"({time.time() - t0:.0f}s)")

    screen = Screen(args.live, args.every)
    tallies = {p.name: Tally() for p in policies}
    records = []
    for i in range(args.games):
        records.append(race(policies, args.seed + i, args.oracle_depth, args.max_moves, screen, tallies, fake,
                            args.preview))

    rows = []
    for p in policies:
        s = tallies[p.name].summary()
        scores = [r[p.name]["score"] for r in records]
        tiles = [r[p.name]["max_tile"] for r in records]
        rows.append([p.name, len(records), f"{np.mean(scores):,.0f} ± {np.std(scores):,.0f}", max(tiles),
                     f"{np.mean([r[p.name]['moves'] for r in records]):.0f}", s["agree"], s["regret"],
                     s.get("flip", float("nan")), s["ece"], s["brier"], s["illegal_mass"]])
    baselines = {}
    if not args.no_baselines:
        for kind in ("random", "first-listed", "oracle"):
            bs = [baseline_game(kind, args.seed + i, args.oracle_depth, args.max_moves) for i in range(args.games)]
            baselines[kind] = bs
            sc = [b["score"] for b in bs]
            rows.append([kind, len(bs), f"{np.mean(sc):,.0f} ± {np.std(sc):,.0f}", max(b["max_tile"] for b in bs),
                         f"{np.mean([b['moves'] for b in bs]):.0f}", "", "", "", "", "", ""])
    header = ["policy", "games", "score", "best tile", "moves", "oracle agree", "regret", "flip", "ece", "brier",
              "illegal mass"]
    print()
    print(fmt_table(rows, header))
    print("\noracle agree: share of turns whose played move is the expectimax-best; regret: (best - played) / "
          "(best - worst) in oracle value; flip: share of turns where reversing the option list changes the "
          "answer; ece/brier: move probabilities against the oracle's move; illegal mass: probability the "
          "readout put on moves the board did not allow.")
    write_json(args.json, {"game": "2048", "model": args.model if not fake else "fake", "levels": levels,
                           "preview": args.preview,
                           "oracle_depth": args.oracle_depth, "games": records, "baselines": baselines,
                           "summary": {p.name: tallies[p.name].summary() for p in policies}})


if __name__ == "__main__":
    main()
