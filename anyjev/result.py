"""Result objects. Every result carries its calibration level."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from anyjev.question import Question


@dataclass
class Decision:
    question: Question
    probs: np.ndarray            # [K] over question.options, sums to 1
    level: str                   # "raw" | "L0" | "L1" | "L2"  ("auto" is a request to decide(), never a result level)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ---- generic ------------------------------------------------------
    @property
    def distribution(self) -> Dict[str, float]:
        return {o: float(p) for o, p in zip(self.question.options, self.probs)}

    @property
    def argmax(self) -> str:
        return self.question.options[int(np.argmax(self.probs))]

    @property
    def answer(self):
        if self.question.kind == "noul":
            return self.p_true >= 0.5
        if self.question.kind == "score":
            return self.value
        return self.argmax

    @property
    def confidence(self) -> float:
        return float(np.max(self.probs))

    # ---- noul ---------------------------------------------------------
    @property
    def p_true(self) -> float:
        if self.question.kind != "noul":
            raise AttributeError("p_true is only defined for noul questions")
        return float(self.probs[0])

    # ---- score --------------------------------------------------------
    @property
    def value(self) -> float:
        if self.question.kind != "score":
            raise AttributeError("value is only defined for score questions")
        centers = np.asarray(self.question.bin_centers())
        return float(np.dot(self.probs, centers))

    def require(self, level: str) -> "Decision":
        """Raise LevelError unless this decision carries at least `level`. Lets downstream
        code refuse to act on an L0 probability where it needs an L1 one."""
        return require_level(self, level)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.question.kind,
            "level": self.level,
            "answer": self.answer,
            "confidence": self.confidence,
        }
        if self.question.kind == "noul":
            out["probability"] = self.p_true
        elif self.question.kind == "score":
            out["value"] = self.value
            out["distribution"] = self.distribution
        else:
            out["distribution"] = self.distribution
        return out

    def __repr__(self) -> str:
        return f"Decision({self.question.id}: {self.answer!r}, conf={self.confidence:.3f}, level={self.level})"


LEVEL_ORDER = {"raw": 0, "L0": 1, "L1": 2, "L2": 3}


class LevelError(ValueError):
    """Raised when a decision does not carry the calibration level the caller required."""


def require_level(decision: "Decision", level: str) -> "Decision":
    if level not in LEVEL_ORDER:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVEL_ORDER)}")
    if LEVEL_ORDER[decision.level] < LEVEL_ORDER[level]:
        raise LevelError(
            f"question {decision.question.id!r} is at level {decision.level}, caller requires {level}; "
            + ("calibrate() it first" if level == "L1" else "decide() at a higher level")
        )
    return decision


class DecisionSet:
    """Results for one state, addressable by question name or index."""

    def __init__(self, decisions: List[Decision], level: str):
        self._items = decisions
        self._by_id = {d.question.id: d for d in decisions}
        self.level = level

    def __getitem__(self, key) -> Decision:
        if isinstance(key, int):
            return self._items[key]
        return self._by_id[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def require(self, level: str) -> "DecisionSet":
        """Raise LevelError unless every decision carries at least `level`."""
        for d in self._items:
            require_level(d, level)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "questions": {d.question.id: d.to_dict() for d in self._items}}

    def __repr__(self) -> str:
        return f"DecisionSet(level={self.level}, {self._items!r})"
