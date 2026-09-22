"""Extra bench tasks for the reproduction, registered into bench.tasks.base.TASKS.

The repo's own bench covers K=2 (noul) and K=20 (choice) only, and never
exercises the `score` primitive at all. These tasks fill the gaps that matter
for the paper's central claim (position bias grows with K, prior bias bites on
noul):

    agnews        choice, K=4    small-K regime
    emotion       choice, K=6    mid-K regime
    massive20     choice, K=20   K=20 outside the banking domain
    massive20_zh  choice, K=20   the same 20 intents in Chinese (roadmap item)
    hate          noul           a second noul, to test the high-variance prior claim
    subj          noul           a third noul
    yelp5         score, 5 bins  the primitive their bench never tests
"""
from __future__ import annotations

from collections import Counter

from anyjev.question import Question
from bench.tasks.base import Task, register

MASSIVE_K = 20


def _pretty(intent: str) -> str:
    return intent.replace("_", " ")


@register("agnews")
def load_agnews() -> Task:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="test")
    names = ["World", "Sports", "Business", "Science and Technology"]
    items = [(row["text"].strip()[:2000], int(row["label"])) for row in ds if row["text"].strip()]
    q = Question.choice("Which news section does this article belong to?", names, name="section")
    return Task("agnews", q, items, license="unspecified (AG News mirror)",
                source="https://huggingface.co/datasets/fancyzhx/ag_news",
                notes=f"K=4; {len(items)} test items")


@register("emotion")
def load_emotion() -> Task:
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    items = [(row["text"].strip()[:2000], int(row["label"])) for row in ds if row["text"].strip()]
    q = Question.choice("Which emotion does this message express?", names, name="emotion")
    return Task("emotion", q, items, license="unspecified (dair-ai/emotion)",
                source="https://huggingface.co/datasets/dair-ai/emotion",
                notes=f"K=6; {len(items)} test items")


def _massive(lang: str, task_name: str) -> Task:
    """Top-20 intents by English test frequency, so the en and zh tasks share
    the same option set and the same Question text ordering."""
    from datasets import load_dataset

    en = load_dataset("mteb/amazon_massive_intent", "en", split="test")
    top = [c for c, _ in Counter(en["label_text"]).most_common(MASSIVE_K)]
    keep = sorted(top)
    remap = {c: i for i, c in enumerate(keep)}

    ds = en if lang == "en" else load_dataset("mteb/amazon_massive_intent", lang, split="test")
    items = [(row["text"].strip()[:2000], remap[row["label_text"]])
             for row in ds if row["label_text"] in remap and row["text"].strip()]
    q = Question.choice("What is the user's intent?", [_pretty(c) for c in keep], name="intent")
    return Task(task_name, q, items, license="CC-BY-4.0 (MASSIVE)",
                source="https://huggingface.co/datasets/mteb/amazon_massive_intent",
                notes=f"lang={lang}; top-{MASSIVE_K} intents by en test frequency; {len(items)} items")


@register("massive20")
def load_massive_en() -> Task:
    return _massive("en", "massive20")


@register("massive20_zh")
def load_massive_zh() -> Task:
    return _massive("zh-CN", "massive20_zh")


@register("hate")
def load_hate() -> Task:
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "hate", split="test")
    # label 1 = hateful -> option 0 ("Yes"), matching the repo's injection task
    items = [(row["text"].strip()[:2000], 0 if int(row["label"]) == 1 else 1)
             for row in ds if row["text"].strip()]
    q = Question.noul("Does this message contain hate speech directed at a person or group?",
                      name="hate")
    return Task("hate", q, items, license="unspecified (tweet_eval)",
                source="https://huggingface.co/datasets/cardiffnlp/tweet_eval",
                notes=f"noul; {len(items)} test items")


@register("subj")
def load_subj() -> Task:
    from datasets import load_dataset

    ds = load_dataset("SetFit/subj", split="test")
    # label_text is "subjective" / "objective"; subjective -> option 0 ("Yes")
    items = [(row["text"].strip()[:2000], 0 if row["label_text"] == "subjective" else 1)
             for row in ds if row["text"].strip()]
    q = Question.noul("Is this sentence a subjective opinion rather than an objective statement?",
                      name="subjective")
    return Task("subj", q, items, license="unspecified (SetFit/subj)",
                source="https://huggingface.co/datasets/SetFit/subj",
                notes=f"noul; {len(items)} test items")


@register("yelp5")
def load_yelp5() -> Task:
    from datasets import load_dataset

    ds = load_dataset("Yelp/yelp_review_full", split="test")
    levels = ["1 star (terrible)", "2 stars (poor)", "3 stars (average)",
              "4 stars (good)", "5 stars (excellent)"]
    items = [(row["text"].strip()[:2000], int(row["label"])) for row in ds if row["text"].strip()]
    items = items[:4000]                      # the split is 50k; we only ever sample a few hundred
    q = Question.score("How many stars does this review give?", levels=levels, name="stars")
    return Task("yelp5", q, items, license="Yelp DSA (research use)",
                source="https://huggingface.co/datasets/Yelp/yelp_review_full",
                notes=f"score with 5 ordered levels; {len(items)} items sampled from the test split")
