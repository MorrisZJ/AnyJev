"""Minesweeper as a calibration benchmark: the game is a probability, and it can be computed.

Every turn the model is asked one `noul` question per candidate cell, "is this cell safe to
reveal?", and the cell with the highest P(safe) is clicked. Click a mine and the game ends,
so survival depends on how well P(safe) ranks the cells; how much of the board gets cleared
is coverage at a risk the model itself chose. The exact posterior P(mine | revealed numbers)
is enumerated for every candidate, so the demo compares the model's probabilities with the
true ones cell by cell: raw readout, AnyJev L0 (two phrasings, label-free prior), and L1
(temperature fit on self-play boards labelled by the game itself, no human labels).

    python -m demo.games.minesweeper --model Qwen/Qwen3-8B --games 5
    python -m demo.games.minesweeper --model Qwen/Qwen3-8B --levels raw,L0,L1 --calibrate 400
    python -m demo.games.minesweeper --backend fake --games 3 --no-live       # synthetic model, CPU
"""
from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from anyjev import Question
from bench import metrics
from demo.games.common import (
    FakeContent,
    Policy,
    Screen,
    Tally,
    add_common_args,
    fmt_table,
    make_backend,
    make_decider,
    paint,
    parse_levels,
    side_by_side,
    write_json,
)

Cell = Tuple[int, int]
QUESTION = Question.noul("Is the candidate cell (marked ?) safe to reveal, i.e. not a mine?", name="safe")


# ---------------------------------------------------------------- engine
class Minesweeper:
    def __init__(self, rows: int, cols: int, mines: int, seed: int):
        self.rows, self.cols, self.n_mines = rows, cols, mines
        self.rng = random.Random(seed)
        self.mines: Set[Cell] = set()
        self.revealed: Set[Cell] = set()
        self.placed = False
        self.dead = False
        self.turns = 0

    def cells(self) -> List[Cell]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]

    def _place(self, first: Cell) -> None:
        keep = set(self.neighbors(first)) | {first}
        pool = [x for x in self.cells() if x not in keep]
        self.mines = set(self.rng.sample(pool, self.n_mines))
        self.placed = True

    def number(self, cell: Cell) -> int:
        return sum(1 for n in self.neighbors(cell) if n in self.mines)

    def reveal(self, cell: Cell) -> bool:
        """Click a cell. Returns False (and ends the game) on a mine; zeros flood-fill."""
        if not self.placed:
            self._place(cell)
        if cell in self.mines:
            self.dead = True
            return False
        stack = [cell]
        while stack:
            x = stack.pop()
            if x in self.revealed:
                continue
            self.revealed.add(x)
            if self.number(x) == 0:
                stack.extend(n for n in self.neighbors(x) if n not in self.revealed)
        return True

    @property
    def safe_total(self) -> int:
        return self.rows * self.cols - self.n_mines

    @property
    def won(self) -> bool:
        return len(self.revealed) == self.safe_total

    @property
    def over(self) -> bool:
        return self.dead or self.won

    def hidden(self) -> List[Cell]:
        return [x for x in self.cells() if x not in self.revealed]

    def frontier(self) -> List[Cell]:
        return [x for x in self.hidden() if any(n in self.revealed for n in self.neighbors(x))]


# ---------------------------------------------------------------- exact posterior
class _Budget(Exception):
    pass


def _enumerate(cells: List[Cell], cons: List[Tuple[List[Cell], int]], max_mines: int, budget: List[int]):
    """Count the mine assignments of one frontier component that satisfy its constraints,
    by total mine count k: returns (counts[k], per-cell counts[cell][k])."""
    idx = {c: i for i, c in enumerate(cells)}
    cons_idx = [([idx[c] for c in cs], need) for cs, need in cons]
    cell_cons: List[List[int]] = [[] for _ in cells]
    for j, (ids, _) in enumerate(cons_idx):
        for i in ids:
            cell_cons[i].append(j)
    assigned = [0] * len(cons)
    unassigned = [len(ids) for ids, _ in cons_idx]
    assign = [0] * len(cells)
    counts: Dict[int, int] = defaultdict(int)
    cell_counts: List[Dict[int, int]] = [defaultdict(int) for _ in cells]

    def rec(i: int, k: int) -> None:
        budget[0] -= 1
        if budget[0] < 0:
            raise _Budget()
        if i == len(cells):
            counts[k] += 1
            for t in range(len(cells)):
                if assign[t]:
                    cell_counts[t][k] += 1
            return
        for v in (0, 1):
            if v and k + 1 > max_mines:
                continue
            ok = True
            for j in cell_cons[i]:
                assigned[j] += v
                unassigned[j] -= 1
                need = cons_idx[j][1]
                if assigned[j] > need or assigned[j] + unassigned[j] < need:
                    ok = False
            if ok:
                assign[i] = v
                rec(i + 1, k + v)
                assign[i] = 0
            for j in cell_cons[i]:
                assigned[j] -= v
                unassigned[j] += 1

    rec(0, 0)
    return dict(counts), [dict(cc) for cc in cell_counts]


def _conv(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    out: Dict[int, int] = defaultdict(int)
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] += va * vb
    return dict(out)


def mine_probabilities(g: Minesweeper, node_budget: int = 3_000_000) -> Tuple[Dict[Cell, float], bool]:
    """Exact P(mine) for every hidden cell given the revealed numbers and the mine count.
    Frontier cells are enumerated per connected component; the cells no number touches
    share the remaining mines uniformly. Returns (probabilities, exact) where exact=False
    means the enumeration budget was exceeded and the global density was used instead."""
    hidden = set(g.hidden())
    M = g.n_mines
    density = {x: M / max(1, len(hidden)) for x in hidden}
    if not g.placed or not hidden:
        return density, False
    cons: List[Tuple[List[Cell], int]] = []
    for x in g.revealed:
        hn = [n for n in g.neighbors(x) if n in hidden]
        if hn:
            cons.append((hn, g.number(x)))
    frontier = sorted({c for cs, _ in cons for c in cs})
    far = [x for x in hidden if x not in set(frontier)]
    n_far = len(far)
    # connected components of frontier cells that share a constraint
    parent = {c: c for c in frontier}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for cs, _ in cons:
        for c in cs[1:]:
            parent[find(c)] = find(cs[0])
    comps: Dict[Cell, List[Cell]] = defaultdict(list)
    for c in frontier:
        comps[find(c)].append(c)
    budget = [node_budget]
    polys, per_cell = [], []
    try:
        for root, cells in comps.items():
            ccons = [(cs, need) for cs, need in cons if find(cs[0]) == root]
            # order cells by walking the constraint graph so pruning bites early
            order, seen, queue = [], set(), [cells[0]]
            adj: Dict[Cell, Set[Cell]] = defaultdict(set)
            for cs, _ in ccons:
                for a in cs:
                    adj[a].update(cs)
            while queue:
                x = queue.pop(0)
                if x in seen:
                    continue
                seen.add(x)
                order.append(x)
                queue.extend(sorted(adj[x] - seen))
            counts, cell_counts = _enumerate(order, ccons, M, budget)
            polys.append(counts)
            per_cell.append(dict(zip(order, cell_counts)))
    except _Budget:
        return density, False
    total = {0: 1}
    for p in polys:
        total = _conv(total, p)
    W = sum(v * comb(n_far, M - k) for k, v in total.items() if 0 <= M - k <= n_far)
    if W == 0:
        return density, False
    out: Dict[Cell, float] = {}
    for i, cells in enumerate(per_cell):
        others = {0: 1}
        for j, p in enumerate(polys):
            if j != i:
                others = _conv(others, p)
        for cell, cc in cells.items():
            num = 0
            for kc, vc in cc.items():
                for ko, vo in others.items():
                    if 0 <= M - kc - ko <= n_far:
                        num += vc * vo * comb(n_far, M - kc - ko)
            out[cell] = num / W
    if n_far:
        p_far = sum(v * comb(n_far, M - k) * (M - k) for k, v in total.items() if 0 <= M - k <= n_far) / (W * n_far)
        for x in far:
            out[x] = p_far
    return out, True


# ---------------------------------------------------------------- what the model sees
def board_rows(g: Minesweeper, mark: Optional[Cell] = None) -> List[str]:
    rows = []
    for r in range(g.rows):
        cells = []
        for c in range(g.cols):
            if (r, c) == mark:
                cells.append("?")
            elif (r, c) in g.revealed:
                n = g.number((r, c))
                cells.append(str(n) if n else "-")
            else:
                cells.append(".")
        rows.append(f"r{r:<2d} " + "  ".join(cells))
    return rows


def state_text(g: Minesweeper, cand: Cell) -> str:
    lines = [f"Minesweeper on a {g.rows}x{g.cols} board with {g.n_mines} mines. A revealed cell shows how many "
             f"of its 8 neighbours are mines ('-' means none). '.' is an unrevealed cell. "
             f"The cell marked '?' is the candidate.",
             "    " + "  ".join(f"c{c}" for c in range(g.cols))]
    lines += board_rows(g, cand)
    lines.append(f"Revealed {len(g.revealed)} of {g.safe_total} safe cells so far. "
                 f"Candidate: row {cand[0]}, column {cand[1]} (the '?').")
    return "\n".join(lines)


def candidates(g: Minesweeper, rng: random.Random) -> List[Cell]:
    """Every frontier cell plus one cell no number touches (they are all alike to the exact
    oracle, so one stands for all of them)."""
    front = g.frontier()
    far = [x for x in g.hidden() if x not in set(front)]
    return front + ([rng.choice(sorted(far))] if far else [])


def fake_logit(p_safe_exact: float) -> float:
    return 3.0 * (2.0 * p_safe_exact - 1.0)


# ---------------------------------------------------------------- one turn
def turn(policy: Policy, g: Minesweeper, rng: random.Random, tally: Tally, cal: Dict[str, list],
         fake: Optional[FakeContent]) -> Dict[str, Any]:
    cands = candidates(g, rng)
    exact, is_exact = mine_probabilities(g)
    states = [state_text(g, c) for c in cands]
    if fake is not None:
        for st, c in zip(states, cands):
            z = fake_logit(1.0 - exact[c])
            fake.set(st, {"Yes": z, "No": -z})
    decs = policy.decide_batch(states)
    p_safe = np.array([d.p_true for d in decs])
    p_mine_true = np.array([exact[c] for c in cands])
    i = int(np.argmax(p_safe))
    cell = cands[i]
    best = float(p_mine_true.min())
    regret = float(p_mine_true[i] - best)
    truth = [0 if c not in g.mines else 1 for c in cands]      # 0 = "Yes", safe
    tally.add([p_safe[i], 1 - p_safe[i]], truth[i], regret <= 1e-9, regret)
    cal["p_model"].extend(p_safe.tolist())
    cal["p_exact"].extend((1 - p_mine_true).tolist())
    cal["truth"].extend(truth)
    cal["exact_flag"].append(is_exact)
    g.turns += 1
    safe = g.reveal(cell)
    return {"cands": cands, "p_safe": p_safe, "p_mine_true": p_mine_true, "chosen": cell, "safe": safe,
            "best": best, "exact": is_exact}


# ---------------------------------------------------------------- rendering
NUM_FG = {1: 33, 2: 34, 3: 196, 4: 20, 5: 124, 6: 37, 7: 0, 8: 244}


def _heat(p: float) -> Tuple[int, int]:
    """(fg, bg) for a P(mine): green when safe, red when not."""
    if p < 0.05:
        return 231, 22
    if p < 0.15:
        return 231, 28
    if p < 0.3:
        return 16, 70
    if p < 0.5:
        return 16, 178
    if p < 0.75:
        return 231, 166
    return 231, 124


def grid(g: Minesweeper, probs: Optional[Dict[Cell, float]], cands: Sequence[Cell], chosen: Optional[Cell],
         safe: Optional[bool]) -> List[str]:
    cset = set(cands)
    out = ["   " + "".join(f"{c:>3d}" for c in range(g.cols))]
    for r in range(g.rows):
        cells = []
        for c in range(g.cols):
            x = (r, c)
            if x in g.revealed:
                n = g.number(x)
                cells.append(paint(f"{n:>3d}", fg=NUM_FG.get(n, 245)) if n else paint("  ·", fg=238))
            elif g.dead and x in g.mines:
                cells.append(paint("  *", fg=231, bg=196, bold=True) if x == chosen else paint("  *", fg=203))
            elif probs is not None and x in cset:
                fg, bg = _heat(probs[x])
                txt = f"{min(99, int(round(100 * probs[x]))):>3d}"
                cells.append(paint(txt, fg=fg, bg=bg, bold=(x == chosen)))
            elif x == chosen:
                cells.append(paint("  ■", fg=231, bg=(22 if safe else 196), bold=True))
            else:
                cells.append(paint("  ▒", fg=240))
        out.append(f"{r:>2d} " + "".join(cells))
    return out


def render_policy(p: Policy, g: Minesweeper, info: Optional[Dict[str, Any]], tally: Tally, cal) -> List[str]:
    status = paint(" won ", fg=231, bg=28) if g.won else (paint(" BOOM ", fg=231, bg=196) if g.dead else "")
    head = [paint(f" {p.name} ", bold=True, bg=24, fg=231)
            + f"  turn {g.turns}  revealed {len(g.revealed)}/{g.safe_total}  " + status]
    if info is None:
        return head + grid(g, None, [], None, None)
    pm = {c: 1 - float(v) for c, v in zip(info["cands"], info["p_safe"])}
    pe = {c: float(v) for c, v in zip(info["cands"], info["p_mine_true"])}
    blocks = [[paint("board", fg=245)] + grid(g, None, info["cands"], info["chosen"], info["safe"]),
              [paint("model P(mine) %", fg=245)] + grid(g, pm, info["cands"], info["chosen"], info["safe"]),
              [paint("exact P(mine) %" + ("" if info["exact"] else " (approx)"), fg=245)]
              + grid(g, pe, info["cands"], info["chosen"], info["safe"])]
    r, c = info["chosen"]
    outcome = paint("safe", fg=114) if info["safe"] else paint("mine", fg=203, bold=True)
    line = (f"clicked ({r},{c}): model P(safe) {1 - pm[(r, c)]:.2f}, exact {1 - pe[(r, c)]:.2f}, "
            f"safest available {1 - info['best']:.2f} -> " + outcome)
    s = tally.summary()
    gap = float(np.mean(np.abs(np.array(cal["p_model"]) - np.array(cal["p_exact"])))) if cal["p_model"] else 0.0
    run = paint(f"picked the safest cell {100 * s.get('agree', 0):.0f}% of turns   "
                f"mean |P(safe) - exact| {gap:.2f}", fg=245)
    return head + side_by_side(blocks, gap=3) + [line, run]


# ---------------------------------------------------------------- games
def new_game(args, seed: int) -> Minesweeper:
    g = Minesweeper(args.rows, args.cols, args.mines, seed)
    g.reveal((args.rows // 2, args.cols // 2))       # the opening click is free and fixed
    return g


def race(policies: List[Policy], args, seed: int, screen: Screen, tallies, cals, fake) -> Dict[str, Any]:
    games = {p.name: new_game(args, seed) for p in policies}
    rngs = {p.name: random.Random(seed * 7919 + 1) for p in policies}
    last: Dict[str, Optional[Dict[str, Any]]] = {p.name: None for p in policies}
    step, t0 = 0, time.time()
    death: Dict[str, Optional[Dict[str, Any]]] = {p.name: None for p in policies}
    while any(not g.over for g in games.values()):
        step += 1
        for p in policies:
            g = games[p.name]
            if g.over:
                continue
            info = turn(p, g, rngs[p.name], tallies[p.name], cals[p.name], fake)
            last[p.name] = info
            if not info["safe"]:
                p_true = float(info["p_mine_true"][info["cands"].index(info["chosen"])])
                death[p.name] = {"turn": g.turns, "p_mine_true": p_true, "safest_available": info["best"]}
        title = f"minesweeper {args.rows}x{args.cols}/{args.mines}  seed {seed}  turn {step}  {time.time() - t0:.0f}s"
        lines = [paint(title, bold=True), ""]
        for p in policies:
            lines += render_policy(p, games[p.name], last[p.name], tallies[p.name], cals[p.name]) + [""]
        screen.frame(lines, step, final=all(g.over for g in games.values()))
    out: Dict[str, Any] = {"seed": seed}
    for p in policies:
        g = games[p.name]
        out[p.name] = {"won": g.won, "cleared": len(g.revealed) / g.safe_total, "turns": g.turns,
                       "death": death[p.name]}
    return out


def baseline_game(kind: str, args, seed: int) -> Dict[str, Any]:
    g = new_game(args, seed)
    rng = random.Random(seed * 7919 + 1)
    while not g.over:
        cands = candidates(g, rng)
        if kind == "oracle":
            exact, _ = mine_probabilities(g)
            cell = min(cands, key=lambda c: (exact[c], c))
        else:
            cell = rng.choice(cands)
        g.turns += 1
        g.reveal(cell)
    return {"won": g.won, "cleared": len(g.revealed) / g.safe_total, "turns": g.turns}


def calibration_set(n: int, args, fake: Optional[FakeContent]):
    """Self-play with the exact oracle; every candidate of every turn is one labelled example
    (safe or mine, known to the game). No human labels."""
    states, labels = [], []
    seed = args.seed + 10_000
    while len(states) < n:
        g = new_game(args, seed)
        rng = random.Random(seed)
        seed += 1
        while not g.over and len(states) < n:
            cands = candidates(g, rng)
            exact, _ = mine_probabilities(g)
            for c in cands:
                st = state_text(g, c)
                if fake is not None:
                    z = fake_logit(1.0 - exact[c])
                    fake.set(st, {"Yes": z, "No": -z})
                states.append(st)
                labels.append(0 if c not in g.mines else 1)
            g.turns += 1
            g.reveal(min(cands, key=lambda c: (exact[c], c)))
    return states, labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_games=5)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--mines", type=int, default=10)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args(argv)
    levels = parse_levels(args)

    fake = FakeContent() if args.backend == "fake" else None
    backend = make_backend(args, fake)
    decider = make_decider(args, backend)
    label = "fake" if fake else args.model.split("/")[-1]
    policies = [Policy(f"{lv} · {label}", decider, lv, QUESTION) for lv in levels]
    if "L1" in levels:
        t0 = time.time()
        states, labels_ = calibration_set(args.calibrate, args, fake)
        art = next(p for p in policies if p.level == "L1").calibrate(states, labels_)
        print(f"L1: temperature {art['temperature']:.2f} fit on {len(states)} self-play cells "
              f"({sum(labels_)} mines, {time.time() - t0:.0f}s)")

    screen = Screen(args.live, args.every)
    tallies = {p.name: Tally() for p in policies}
    cals = {p.name: {"p_model": [], "p_exact": [], "truth": [], "exact_flag": []} for p in policies}
    records = [race(policies, args, args.seed + i, screen, tallies, cals, fake) for i in range(args.games)]

    rows = []
    for p in policies:
        s = tallies[p.name].summary()
        cal = cals[p.name]
        P = np.stack([np.array(cal["p_model"]), 1 - np.array(cal["p_model"])], axis=1)
        gap = float(np.mean(np.abs(np.array(cal["p_model"]) - np.array(cal["p_exact"]))))
        deaths = [r[p.name]["death"] for r in records if r[p.name]["death"]]
        avoidable = sum(1 for d in deaths if d["p_mine_true"] > d["safest_available"] + 1e-9)
        rows.append([p.name, len(records), f"{np.mean([r[p.name]['won'] for r in records]):.2f}",
                     f"{np.mean([r[p.name]['cleared'] for r in records]):.2f}",
                     f"{np.mean([r[p.name]['turns'] for r in records]):.1f}", s["agree"], s["regret"],
                     f"{avoidable}/{len(deaths)}", metrics.ece(P, cal["truth"]), metrics.brier(P, cal["truth"]), gap])
    baselines = {}
    if not args.no_baselines:
        for kind in ("random", "oracle"):
            bs = [baseline_game(kind, args, args.seed + i) for i in range(args.games)]
            baselines[kind] = bs
            rows.append([kind, len(bs), f"{np.mean([b['won'] for b in bs]):.2f}",
                         f"{np.mean([b['cleared'] for b in bs]):.2f}", f"{np.mean([b['turns'] for b in bs]):.1f}",
                         "", "", "", "", "", ""])
    header = ["policy", "games", "win rate", "board cleared", "turns", "safest pick", "regret", "avoidable deaths",
              "ece", "brier", "|P - exact|"]
    print()
    print(fmt_table(rows, header))
    print("\nsafest pick: share of turns where the clicked cell had the lowest exact P(mine) among the candidates; "
          "regret: exact P(mine) of the clicked cell minus that minimum; avoidable deaths: deaths on a cell that "
          "was not the safest available; ece/brier: P(safe) of every candidate against whether it was a mine; "
          "|P - exact|: mean gap between the model's P(safe) and the exact posterior.")
    write_json(args.json, {"game": "minesweeper", "model": args.model if not fake else "fake", "levels": levels,
                           "board": [args.rows, args.cols, args.mines], "games": records, "baselines": baselines,
                           "summary": {p.name: tallies[p.name].summary() for p in policies}})


if __name__ == "__main__":
    main()
