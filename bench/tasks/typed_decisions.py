"""LocalLLaMA/typed-decisions: 4 workflows x 5 typed questions per case, soft
teacher labels. The set Laya reports on (400 test cases, 2,000 decisions).
Apache-2.0. Each case carries several questions, so this loader returns
*decisions* grouped by question rather than a single-question Task."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anyjev.question import Question

DATASET = "LocalLLaMA/typed-decisions"


@dataclass
class Decision:
    case_id: str
    workflow: str
    qname: str
    question: Question
    state: Any
    gold_index: int                 # index into question.options
    gold_probs: List[float]         # soft label, same order
    gold_score: Optional[float]     # score questions: expected level


def build_question(qname: str, spec: Dict[str, Any]) -> tuple:
    """Returns (Question, option_keys) where option_keys map our option index
    back to the dataset's label strings."""
    t, instr, crit = spec["type"], spec["instructions"], spec.get("criteria")
    if t == "choice":
        keys = list(crit.keys()) if isinstance(crit, dict) else [str(c) for c in crit]
        opts = [f"{k}: {crit[k]}" for k in keys] if isinstance(crit, dict) else keys
        return Question.choice(instr, opts, name=qname), keys
    if t == "noul":
        # criteria: {"true": ..., "false": ...}; our option 0 is Yes (=true)
        text = f"Is the following statement true? {instr}"
        if isinstance(crit, dict):
            text += f"\nYes means: {crit.get('true', '')}\nNo means: {crit.get('false', '')}"
        return Question.noul(text, name=qname), ["true", "false"]
    if t == "score":
        levels = list(crit.values()) if isinstance(crit, dict) else [str(c) for c in crit]
        return Question.score(instr, levels=levels, name=qname), [str(i) for i in range(len(levels))]
    raise ValueError(f"unknown question type {t}")


def load(split: str = "test", workflow: str = "all", limit_cases: Optional[int] = None) -> List[Decision]:
    """limit_cases is applied per workflow (rows are grouped by workflow in the
    "all" config), so a limited load still covers every workflow."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, workflow, split=split)
    out: List[Decision] = []
    seen: Dict[str, int] = defaultdict(int)
    for row in ds:
        if limit_cases is not None and seen[row["workflow"]] >= limit_cases:
            continue
        seen[row["workflow"]] += 1
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qname, spec in questions.items():
            q, keys = build_question(qname, spec)
            g = gold[qname]
            probs = [float(g["probabilities"].get(k, 0.0)) for k in keys]
            gi = keys.index(str(g["label"]))
            gs = float(g["score"]) if spec["type"] == "score" and "score" in g else None
            out.append(Decision(row["id"], row["workflow"], qname, q, state, gi, probs, gs))
    return out


def group_by_question(decisions: List[Decision]) -> Dict[str, List[Decision]]:
    groups: Dict[str, List[Decision]] = defaultdict(list)
    for d in decisions:
        groups[d.question.key].append(d)
    return dict(groups)
