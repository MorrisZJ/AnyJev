"""How `HFBackend` loads a model, and which attribute it thinks the backbone is.

Both were reported by @lws2004 in issue #5. Passing `device_map=` to `from_pretrained` makes
transformers require `accelerate`, which the `hf` extra does not install, so the backend could
not be constructed on a clean machine for any device at all; and assuming the backbone is called
`.model` raised `AttributeError` out of `score_shared` on architectures that call it something
else. The first two tests need no torch, so they guard the regression even where the engine
tests skip.
"""
from __future__ import annotations

import sys
import types

import pytest


class _Cfg:
    num_hidden_layers, hidden_size = 4, 8


class _Stub:
    """Just enough of a `PreTrainedModel` for the constructor, remembering how it was moved."""

    base_model_prefix = "model"

    def __init__(self):
        self.config = _Cfg()
        self.model = types.SimpleNamespace(name="llama-shaped backbone")
        self.moved_to = None

    def eval(self):
        return self

    def to(self, device):
        self.moved_to = device
        return self


def _patch(monkeypatch, stub=None):
    """Patch the two names `HFBackend.__init__` imports, and record from_pretrained's kwargs."""
    transformers = pytest.importorskip("transformers")
    seen = {}
    model = stub if stub is not None else _Stub()

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kw):
            seen.update(kw)
            seen["_name"] = name
            return model

    class _Tok:
        @staticmethod
        def from_pretrained(name, **kw):
            return types.SimpleNamespace(padding_side="right", pad_token="<pad>", eos_token="</s>")

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Auto)
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or types.SimpleNamespace(float32="float32"))
    return seen, model


def test_no_device_map_is_passed_and_the_model_is_moved_instead(monkeypatch):
    from anyjev.backends.hf import HFBackend

    seen, model = _patch(monkeypatch)
    be = HFBackend("some/model", device="cpu", dtype="float32")

    assert "device_map" not in seen, "device_map makes transformers require accelerate (issue #5)"
    assert model.moved_to == "cpu", "the model has to reach the device some other way"
    assert be.model is model


def test_device_map_is_still_reachable_for_real_sharding(monkeypatch):
    from anyjev.backends.hf import HFBackend

    seen, model = _patch(monkeypatch)
    HFBackend("some/model", device="cuda", dtype="float32", device_map="auto")

    assert seen["device_map"] == "auto", "an explicit device_map must still be honoured"
    assert model.moved_to is None, "transformers placed it; moving it again would undo the sharding"


def test_the_backbone_is_resolved_not_guessed(monkeypatch):
    from anyjev.backends.hf import HFBackend

    gpt2ish = _Stub()
    gpt2ish.base_model_prefix = "transformer"          # what GPT-2 calls it
    gpt2ish.transformer = types.SimpleNamespace(name="gpt2-shaped backbone")
    del gpt2ish.model                                  # and it has no `.model` at all
    _patch(monkeypatch, gpt2ish)

    be = HFBackend("some/gpt2", device="cpu", dtype="float32")
    assert be.backbone is gpt2ish.transformer

    llamaish = _Stub()
    _patch(monkeypatch, llamaish)
    assert HFBackend("some/llama", device="cpu", dtype="float32").backbone is llamaish.model


def test_shared_prefix_works_on_a_model_whose_backbone_is_not_called_model(tmp_path):
    """The reported failure, end to end on a real (tiny, randomly initialised) GPT-2."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # the vocabulary has to match the real tokenizer loaded below, or the embedding lookup
    # runs off the end; everything else is kept tiny so nothing is downloaded or trained
    cfg = AutoConfig.for_model("gpt2", n_embd=32, n_layer=2, n_head=2, n_positions=128,
                               vocab_size=50257)
    src = tmp_path / "gpt2"
    AutoModelForCausalLM.from_config(cfg).save_pretrained(src)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(src)

    from anyjev.backends.hf import HFBackend

    be = HFBackend(str(src), device="cpu", dtype="float32", batch_size=2)
    assert not hasattr(be.model, "model") and be.backbone is be.model.transformer

    ids = [be.tokenizer.encode(" Yes")[0], be.tokenizer.encode(" No")[0]]
    out = be.score_shared([("A prefix that both suffixes share.\n", ["Yes", "No"])], [ids])
    assert len(out) == 1 and len(out[0]) == 2 and all(len(row) == 2 for row in out[0])
