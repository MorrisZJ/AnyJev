from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from anyjev.question import Question

TASKS: Dict[str, Callable[[], "Task"]] = {}
MCQ_TASKS: Dict[str, Callable[[], "MCQTask"]] = {}


def _split(items: list, n_test: int, n_calib: int, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n_test]], [items[i] for i in idx[n_test:n_test + n_calib]]


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
        return _split(self.items, n_test, n_calib, seed)


@dataclass
class MCQTask:
    """A set whose every item carries its own question *and* its own options.

    That is the shape of almost every multimodal MCQ benchmark, and `Task`
    cannot express it: there, one `Question` is shared by every item, which is
    what lets the batch prior pool over states. Here each item is its own
    question, so the batch prior has one sample to work with and the levers
    are permutation marginalization and the content-free prior. K is constant
    within a task, so the metrics stack.
    """

    name: str
    question_kind: str                    # "choice" today
    k: int
    items: List[Tuple[Any, Question, int]]   # (state, question, label index)
    license: str
    source: str
    notes: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def split(self, n_test: int, n_calib: int, seed: int = 0):
        return _split(self.items, n_test, n_calib, seed)


def register(name: str):
    def deco(fn):
        TASKS[name] = fn
        return fn
    return deco


def register_mcq(name: str):
    def deco(fn):
        MCQ_TASKS[name] = fn
        return fn
    return deco


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]()


def get_mcq_task(name: str) -> MCQTask:
    if name not in MCQ_TASKS:
        raise KeyError(f"unknown mcq task {name!r}; known: {sorted(MCQ_TASKS)}")
    return MCQ_TASKS[name]()
