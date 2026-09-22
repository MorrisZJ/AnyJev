"""POPE object-hallucination probing as a `noul`.

The point of this task is the label prior. POPE fixes the ground-truth Yes
rate at exactly 0.5 by construction, and the POPE paper reports open VLMs
answering "Yes" at up to 99 percent against that 0.5 -- a readout bias with a
known target, which is exactly what the L0 prior correction divides out. The
balanced marginal also means the batch prior's "the batch's label marginal is
not extreme" assumption holds here by construction rather than by hope.

One reformulation, and it is deliberate. POPE asks a different question per
item ("Is there a snowboard in the image?"), which would give every item its
own `Question` and leave the batch prior with one sample per question. We ask
one fixed question and put the probed object in the state instead. Same
information, same task, and one question over the whole set, so the prior is
estimated over thousands of items. It also normalizes the 57 upstream
questions that misspell "image" as "imange".

QA pairs MIT (RUCAIBox/POPE); images are COCO val2014 and keep their own
terms. The three subsets (random / popular / adversarial) are pooled here;
they form a difficulty ladder that would make a ready-made L1 shift probe.
"""
from __future__ import annotations

import re

from anyjev.media import Image
from anyjev.question import Question
from bench.tasks.base import Task, register

DATASET = "lmms-lab-encoder/POPE"   # moved from lmms-lab/POPE, which now redirects here
# Enough for any (test + calib) split the bench asks for without holding all
# 9,000 JPEGs in memory. Applied after a fixed shuffle so all three subsets
# and all 79 objects stay represented.
CAP = 3000

_QUESTION = re.compile(r"^Is there (?:a |an |the )?(.+?) in the (?:image|imange)\?$", re.I)


@register("pope")
def load() -> Task:
    import datasets
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test")
    ds = ds.cast_column("image", datasets.Image(decode=False))   # keep JPEG bytes, decode lazily
    ds = ds.shuffle(seed=0).select(range(min(CAP, len(ds))))

    items, skipped = [], 0
    for row in ds:
        match = _QUESTION.match(row["question"])
        if match is None:
            skipped += 1
            continue
        state = {"image": Image(row["image"]["bytes"]), "object": match.group(1)}
        items.append((state, 0 if row["answer"].strip().lower() == "yes" else 1))

    q = Question.noul("Is the object named in the state visible in the image?", name="present")
    return Task("pope", q, items, license="MIT (QA pairs, RUCAIBox/POPE); images COCO val2014",
                source="https://huggingface.co/datasets/lmms-lab-encoder/POPE",
                notes=f"{len(items)} of {CAP} sampled items; {skipped} off-template questions dropped; "
                      "object moved from the question into the state")
