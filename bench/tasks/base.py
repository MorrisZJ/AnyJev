from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from anyjev.question import Question

TASKS: Dict[str, Callable[[], "Task"]] = {}


@dataclass
class Task:
    name: str
    question: Question
    items: List[Tuple[Any, int]]          # (state, label index into question.options)
    license: str
    source: str
    notes: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def split(self, n_test: int, n_calib: int, seed: int = 0):
        rng = random.Random(seed)
        idx = list(range(len(self.items)))
        rng.shuffle(idx)
        test = [self.items[i] for i in idx[:n_test]]
        calib = [self.items[i] for i in idx[n_test:n_test + n_calib]]
        return test, calib


def register(name: str):
    def deco(fn):
        TASKS[name] = fn
        return fn
    return deco


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]()
