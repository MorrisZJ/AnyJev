"""BANKING77 restricted to its 20 most frequent intents (letter readout caps K
at 26; the full 77-way version needs span readout). Comparable in spirit to
openJev Verdict 2.0 and jev-baselines-eval, not in absolute numbers."""
from __future__ import annotations

from collections import Counter

from anyjev.question import Question
from bench.tasks.base import Task, register

K = 20


@register("banking20")
def load() -> Task:
    from datasets import load_dataset

    # mteb mirror: plain parquet (no loading script), text / label / label_text
    ds = load_dataset("mteb/banking77")
    train, test = ds["train"], ds["test"]
    names = {}
    for row in train:
        names[int(row["label"])] = row["label_text"]
    top = [c for c, _ in Counter(int(x) for x in train["label"]).most_common(K)]
    top_sorted = sorted(top)
    pretty = [names[c].replace("_", " ") for c in top_sorted]
    remap = {c: i for i, c in enumerate(top_sorted)}
    items = [(row["text"], remap[int(row["label"])]) for row in test if int(row["label"]) in remap]
    q = Question.choice("What is the customer's intent?", pretty, name="intent")
    return Task("banking20", q, items, license="CC-BY-4.0",
                source="https://huggingface.co/datasets/mteb/banking77",
                notes=f"top-{K} intents by train frequency; {len(items)} test items")
