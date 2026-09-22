"""AnyJev: turn any causal LLM into a Jev-style decision model.

Typed decisions (choice / score / noul) with probabilities, read from the
model's next-token logits in one prefill, no generation. Training-free
debiasing (L0) is on by default; post-hoc calibration (L1) when you have labels.

Not affiliated with, endorsed by, or derived from TypeSafe AI or Jev.
"""
from anyjev.decider import Decider
from anyjev.question import Question
from anyjev.result import Decision, DecisionSet, LevelError

__all__ = ["Question", "Decision", "DecisionSet", "Decider", "LevelError"]
try:  # single source of truth: the installed package metadata
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("anyjev")
except Exception:  # not installed (source checkout); keep in step with pyproject.toml
    __version__ = "0.0.2"
