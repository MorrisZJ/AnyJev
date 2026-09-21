"""20 Newsgroups as a 20-way topic choice. K=20 makes position bias bite.
Same dataset jev-orderby-bench used, so numbers are comparable."""
from __future__ import annotations

from anyjev.question import Question
from bench.tasks.base import Task, register

LABELS = {
    "alt.atheism": "atheism", "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "Microsoft Windows", "comp.sys.ibm.pc.hardware": "PC hardware",
    "comp.sys.mac.hardware": "Mac hardware", "comp.windows.x": "X Window System",
    "misc.forsale": "items for sale", "rec.autos": "cars", "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball", "rec.sport.hockey": "hockey", "sci.crypt": "cryptography",
    "sci.electronics": "electronics", "sci.med": "medicine", "sci.space": "space",
    "soc.religion.christian": "Christianity", "talk.politics.guns": "gun politics",
    "talk.politics.mideast": "Middle East politics", "talk.politics.misc": "politics (other)",
    "talk.religion.misc": "religion (other)",
}


@register("newsgroups")
def load() -> Task:
    from datasets import load_dataset

    ds = load_dataset("SetFit/20_newsgroups", split="test")
    names = list(LABELS.values())
    key_by_raw = {raw: i for i, raw in enumerate(LABELS)}
    items = []
    for row in ds:
        text = (row["text"] or "").strip()
        if len(text) < 40:
            continue
        items.append((text[:2000], key_by_raw[row["label_text"]]))
    q = Question.choice("Which newsgroup topic does this post belong to?", names, name="topic")
    return Task("newsgroups", q, items, license="see dataset card (SetFit/20_newsgroups)",
                source="https://huggingface.co/datasets/SetFit/20_newsgroups")
