"""Typed question specs: the three decision primitives."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Sequence

MAX_OPTIONS = 26  # letter-readout limit; span readout will lift this


class QuestionError(ValueError):
    pass


@dataclass(frozen=True)
class Question:
    """A typed question. Build with Question.choice / .score / .noul."""

    kind: str                      # "choice" | "score" | "noul"
    text: str
    options: tuple                 # option labels, shown to the model (score: bin labels)
    name: Optional[str] = None
    scale: tuple = (0.0, 1.0)      # score only: value range
    ordered: bool = False          # True for score bins: never permuted
    centers: Optional[tuple] = None  # score with explicit levels: the value of each level

    # ---- constructors -------------------------------------------------
    @staticmethod
    def choice(text: str, options: Sequence[str], name: Optional[str] = None) -> "Question":
        opts = tuple(str(o) for o in options)
        if len(opts) < 2:
            raise QuestionError("choice needs at least 2 options")
        if len(opts) > MAX_OPTIONS:
            raise QuestionError(f"choice supports at most {MAX_OPTIONS} options in letter readout")
        if len(set(opts)) != len(opts):
            raise QuestionError("choice options must be unique")
        return Question("choice", text, opts, name)

    @staticmethod
    def noul(text: str, name: Optional[str] = None) -> "Question":
        return Question("noul", text, ("Yes", "No"), name)

    @staticmethod
    def score(text: str, bins: int = 5, scale: Sequence[float] = (0.0, 1.0),
              name: Optional[str] = None, levels: Optional[Sequence[str]] = None) -> "Question":
        """Either `bins` equal-width bins over `scale`, or explicit ordered
        `levels` (descriptions); with levels the value is the level index."""
        if levels is not None:
            lv = tuple(str(x) for x in levels)
            if len(lv) < 2 or len(lv) > 10:
                raise QuestionError("score supports 2 to 10 levels")
            return Question("score", text, lv, name, (0.0, float(len(lv) - 1)), ordered=True,
                            centers=tuple(float(i) for i in range(len(lv))))
        if bins < 2 or bins > 10:
            raise QuestionError("score supports 2 to 10 bins")
        lo, hi = float(scale[0]), float(scale[1])
        if not hi > lo:
            raise QuestionError("scale must be (lo, hi) with hi > lo")
        edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
        labels = tuple(f"{edges[i]:g} to {edges[i + 1]:g}" for i in range(bins))
        return Question("score", text, labels, name, (lo, hi), ordered=True)

    # ---- derived ------------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.options)

    @property
    def key(self) -> str:
        """Stable hash of the spec, used to key calibration artifacts."""
        payload = json.dumps(
            {"kind": self.kind, "text": self.text, "options": list(self.options),
             "scale": list(self.scale), "centers": list(self.centers) if self.centers else None},
            sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def id(self) -> str:
        return self.name or f"q_{self.key[:8]}"

    def bin_centers(self) -> list:
        """score only: the value each bin stands for."""
        if self.kind != "score":
            raise QuestionError("bin_centers is only defined for score questions")
        if self.centers is not None:
            return list(self.centers)
        lo, hi = self.scale
        n = self.k
        return [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]
