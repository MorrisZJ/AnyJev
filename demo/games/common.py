"""Shared pieces for the game demos: backends, one policy per readout level, ANSI rendering,
decision-quality tallies. Nothing here knows the rules of any game."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anyjev import Decider, Question  # noqa: E402
from anyjev.question import Question as Q  # noqa: E402
from bench import metrics  # noqa: E402

LEVELS = ("raw", "L0", "L1")


# ---------------------------------------------------------------- CLI + backends
def add_common_args(ap: argparse.ArgumentParser, default_games: int = 3) -> None:
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--backend", default="hf", choices=["hf", "fake"],
                    help="fake = synthetic model with a known position bias, runs on CPU")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--levels", default="raw,L0", help="comma list of raw,L0,L1 (L1 needs --calibrate)")
    ap.add_argument("--prior", default="batch", choices=["batch", "content_free", "none"])
    ap.add_argument("--prior-strength", type=float, default=None)
    ap.add_argument("--games", type=int, default=default_games, help="games per policy")
    ap.add_argument("--seed", type=int, default=0, help="first seed; game i uses seed+i")
    ap.add_argument("--calibrate", type=int, default=0,
                    help="L1: number of oracle-labelled decisions to fit the temperature on")
    ap.add_argument("--live", dest="live", action="store_true", default=None,
                    help="redraw in place every step (default when stdout is a terminal)")
    ap.add_argument("--no-live", dest="live", action="store_false")
    ap.add_argument("--every", type=int, default=10, help="non-live: print a frame every N steps")
    ap.add_argument("--json", default=None, help="write the summary and per-game records here")
    ap.add_argument("--fake-bias", default="2.0,0,0,0",
                    help="fake backend: additive logit per option position (the bug L0 removes)")
    ap.add_argument("--fake-temperature", type=float, default=0.5,
                    help="fake backend: <1 makes it overconfident (what L1 fixes)")


class FakeContent:
    """Content term for `FakeBackend`: the game writes per-option logits for each state text
    before the decision is asked, so the synthetic model 'knows' the oracle up to its biases."""

    def __init__(self):
        self.table: Dict[str, Dict[str, float]] = {}

    def set(self, state_text: str, logits: Dict[str, float]) -> None:
        self.table[state_text] = dict(logits)

    def __call__(self, state: str, option: str) -> float:
        return self.table.get(state, {}).get(option, 0.0)


def make_backend(args, fake_content: Optional[FakeContent] = None):
    if args.backend == "fake":
        from anyjev.backends.fake import FakeBackend
        bias = [float(x) for x in args.fake_bias.split(",") if x.strip()]
        return FakeBackend(fake_content or FakeContent(), position_bias=bias,
                           temperature=args.fake_temperature)
    from anyjev.backends.hf import HFBackend
    return HFBackend(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)


def make_decider(args, backend) -> Decider:
    return Decider(backend, prior=args.prior, prior_strength=args.prior_strength)


def parse_levels(args) -> List[str]:
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        raise SystemExit(f"unknown level(s) {bad}; choose from {LEVELS}")
    if "L1" in levels and not args.calibrate:
        raise SystemExit("L1 needs --calibrate N (oracle-labelled decisions to fit the temperature)")
    return levels


def reversed_question(q: Question) -> Question:
    """Same question, options listed in reverse: the order-flip probe."""
    centers = tuple(reversed(q.centers)) if q.centers is not None else None
    return Q(q.kind, q.text, tuple(reversed(q.options)), q.name, q.scale, q.ordered, centers)


# ---------------------------------------------------------------- policies
class Policy:
    """One readout level of one decider, playing through one question. `decide` returns the
    probabilities over the question's options in their canonical order, plus whether the
    argmax changes when the options are listed in reverse (the order-flip probe)."""

    def __init__(self, name: str, decider: Decider, level: str, question: Question):
        self.name, self.decider, self.level, self.q = name, decider, level, question
        self.qr = reversed_question(question)
        self.calibrated = False

    def calibrate(self, states: Sequence[str], labels: Sequence[int]) -> Dict[str, Any]:
        art = self.decider.calibrate(self.q, states, labels)
        if self.q.kind != "noul":
            k = self.q.k
            self.decider.calibrate(self.qr, states, [k - 1 - y for y in labels])
        self.calibrated = True
        return art

    def decide(self, state: str, flip_probe: bool = True) -> Tuple[np.ndarray, Any, Optional[bool]]:
        d = self.decider.decide(state, [self.q], level=self.level)[0]
        flip = None
        if flip_probe and self.q.kind != "noul":
            dr = self.decider.decide(state, [self.qr], level=self.level)[0]
            flip = bool(int(np.argmax(d.probs)) != int(np.argmax(dr.probs[::-1])))
        return np.asarray(d.probs, dtype=float), d, flip

    def decide_batch(self, states: Sequence[str]) -> List[Any]:
        return self.decider.decide_batch(list(states), self.q, level=self.level)


# ---------------------------------------------------------------- tallies
class Tally:
    """Decision-quality bookkeeping for one policy across games."""

    def __init__(self):
        self.probs: List[np.ndarray] = []      # model distribution over options (canonical order)
        self.labels: List[int] = []            # oracle's best option
        self.agree: List[bool] = []
        self.regret: List[float] = []          # (best - chosen) / (best - worst), 0 = oracle's move
        self.flips: List[bool] = []
        self.extra: Dict[str, List[float]] = {}
        self.games: List[Dict[str, Any]] = []

    def add(self, probs, label, agree, regret, flip=None, **extra):
        self.probs.append(np.asarray(probs, dtype=float))
        self.labels.append(int(label))
        self.agree.append(bool(agree))
        self.regret.append(float(regret))
        if flip is not None:
            self.flips.append(bool(flip))
        for k, v in extra.items():
            self.extra.setdefault(k, []).append(float(v))

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"decisions": len(self.labels)}
        if self.labels:
            P = np.stack(self.probs)
            out.update(agree=float(np.mean(self.agree)), regret=float(np.mean(self.regret)),
                       ece=metrics.ece(P, self.labels), brier=metrics.brier(P, self.labels))
        if self.flips:
            out["flip"] = float(np.mean(self.flips))
        for k, v in self.extra.items():
            out[k] = float(np.mean(v))
        return out


def running_line(t: Tally) -> str:
    s = t.summary()
    parts = []
    if "agree" in s:
        parts.append(f"agree {100 * s['agree']:.0f}%")
    if "flip" in s:
        parts.append(f"flip {100 * s['flip']:.0f}%")
    if "ece" in s:
        parts.append(f"ece {s['ece']:.2f}")
    return "  ".join(parts)


# ---------------------------------------------------------------- ANSI rendering
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(text: str, fg: Optional[int] = None, bg: Optional[int] = None, bold: bool = False) -> str:
    if not use_color():
        return text
    codes = []
    if bold:
        codes.append("1")
    if fg is not None:
        codes.append(f"38;5;{fg}")
    if bg is not None:
        codes.append(f"48;5;{bg}")
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m" if codes else text


def visible_len(s: str) -> int:
    return len(_ANSI.sub("", s))


def bar(p: float, width: int = 12) -> str:
    n = int(round(max(0.0, min(1.0, p)) * width))
    return "█" * n + "░" * (width - n)


def side_by_side(blocks: Sequence[Sequence[str]], gap: int = 4) -> List[str]:
    widths = [max((visible_len(ln) for ln in b), default=0) for b in blocks]
    height = max((len(b) for b in blocks), default=0)
    out = []
    for i in range(height):
        cells = []
        for b, w in zip(blocks, widths):
            ln = b[i] if i < len(b) else ""
            cells.append(ln + " " * (w - visible_len(ln)))
        out.append((" " * gap).join(cells).rstrip())
    return out


class Screen:
    """Prints frames: in place when live, every N steps otherwise."""

    def __init__(self, live: Optional[bool], every: int = 10):
        self.live = sys.stdout.isatty() if live is None else live
        self.every = max(1, every)
        self.t0 = time.time()
        self._printed = None

    def frame(self, lines: Sequence[str], step: int, final: bool = False) -> None:
        if self.live:
            sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            if final:
                print()
        elif (final or step % self.every == 0) and self._printed != (step, final or step % self.every == 0):
            if final and self._printed == (step, True):
                return
            print("\n".join(lines) + "\n")
            sys.stdout.flush()
            self._printed = (step, True)


def fmt_table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    cells = [[str(h) for h in header]] + [[("%.3f" % v if isinstance(v, float) else str(v)) for v in r]
                                          for r in rows]
    w = [max(len(c[i]) for c in cells) for i in range(len(header))]
    lines = ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells[0])) + " |",
             "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for c in cells[1:]:
        lines.append("| " + " | ".join(v.ljust(w[i]) for i, v in enumerate(c)) + " |")
    return "\n".join(lines)


def write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote {path}")
