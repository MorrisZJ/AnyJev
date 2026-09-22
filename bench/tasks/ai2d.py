"""AI2D science diagrams as a per-item 4-way `choice`.

The one multimodal MCQ set on the shortlist with a *constant* K=4 across
every item, which is what makes its flip rate directly comparable to the
text `banking20` and `newsgroups` rows without a K confound. Options are
descriptive phrases with no inherent ordering, so a cyclic shift is
semantically free -- the clean case for permutation marginalization.

One filter, and it is not optional. About a fifth of AI2D questions refer to
labels printed on the diagram, so their options *are* single characters
("which of these define dairy item" with options ['c', 'D', 'b', 'a']).
Those collide head-on with the A/B/C/D letter readout: the model would be
asked to answer "B" to mean the option whose text is "D". They are dropped.

CC BY-SA (AI2, via the AWS Registry of Open Data) for the annotations; the
diagrams were collected from Google Images and keep their own terms. The HF
mirror carries no license tag, so both claims are sourced upstream.
"""
from __future__ import annotations

from anyjev.media import Image
from anyjev.question import Question
from bench.tasks.base import MCQTask, register_mcq

DATASET = "lmms-lab-encoder/ai2d"   # moved from lmms-lab/ai2d, which now redirects here
K = 4
CAP = 1500          # after a fixed shuffle; well above any (test + calib) split
MAX_LABEL_CHARS = 2  # options this short are diagram label references, not answers


def _is_diagram_label_question(options) -> bool:
    return all(len(str(o).strip()) <= MAX_LABEL_CHARS for o in options)


@register_mcq("ai2d")
def load() -> MCQTask:
    import datasets
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test")
    ds = ds.cast_column("image", datasets.Image(decode=False))
    ds = ds.shuffle(seed=0)

    items, dropped = [], 0
    for row in ds:
        options = [str(o) for o in row["options"]]
        if len(options) != K or len(set(options)) != K or _is_diagram_label_question(options):
            dropped += 1
            continue
        state = Image(row["image"]["bytes"])
        q = Question.choice(row["question"], options, name="diagram")
        items.append((state, q, int(row["answer"])))
        if len(items) >= CAP:
            break

    return MCQTask("ai2d", "choice", K, items,
                   license="CC BY-SA (AI2 annotations); diagrams from Google Images, own terms",
                   source="https://huggingface.co/datasets/lmms-lab-encoder/ai2d",
                   notes=f"{len(items)} items kept, {dropped} dropped (diagram-label options "
                         f"or not exactly {K} distinct options)")
