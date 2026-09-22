"""Oxford-IIIT Pet restricted to 20 of its 37 breeds: the image-side analogue
of `banking20`.

Built to mirror the text tasks rather than the multimodal-benchmark
convention. One question and one fixed 20-option label set shared by every
item, so K=20 makes position bias bite the way it does on `banking20` and
`newsgroups`, and so the batch prior has a real batch to estimate over. Most
multimodal MCQ sets give every item its own options, which rules both of
those out (see `bench/run_mcq.py`).

Fine-grained breeds also put a small VLM in the informative band -- not at the
ceiling where calibration metrics stop meaning anything, and not at the floor
where a model guessing uniformly flips constantly for reasons that have
nothing to do with position bias.

CC BY-SA 4.0 (tagged on the mirror, and the original Oxford-IIIT Pet release
is CC BY-SA 4.0).
"""
from __future__ import annotations

from anyjev.media import Image
from anyjev.question import Question
from bench.tasks.base import Task, register

DATASET = "timm/oxford-iiit-pet"
K = 20


@register("pets20")
def load() -> Task:
    import datasets
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test")
    names = ds.features["label"].names
    # the test split is balanced at 100 per breed, so "most frequent" would be
    # a coin flip; take the first K by name instead, which is deterministic
    # and keeps both cats and dogs in the option list
    keep = sorted(range(len(names)), key=lambda i: names[i])[:K]
    pretty = [names[i].replace("_", " ") for i in keep]
    remap = {c: i for i, c in enumerate(keep)}

    ds = ds.cast_column("image", datasets.Image(decode=False))   # keep bytes, decode lazily
    items = [(Image(row["image"]["bytes"]), remap[int(row["label"])])
             for row in ds if int(row["label"]) in remap]

    q = Question.choice("Which breed is the pet in this photo?", pretty, name="breed")
    return Task("pets20", q, items, license="CC-BY-SA-4.0",
                source="https://huggingface.co/datasets/timm/oxford-iiit-pet",
                notes=f"first {K} of 37 breeds by name; {len(items)} test images")
