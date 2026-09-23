"""Real-engine smoke for VLMBackend. Skipped unless ANYJEV_VLM_MODEL names a
vision-language checkpoint (hub id or local path) and a GPU is available.

    ANYJEV_VLM_MODEL=/path/to/Qwen3-VL-2B-Instruct pytest -m engine
"""
import os

import numpy as np
import pytest

MODEL = os.environ.get("ANYJEV_VLM_MODEL")
pytestmark = [pytest.mark.engine,
              pytest.mark.skipif(not MODEL, reason="set ANYJEV_VLM_MODEL to run against a real VLM")]

COLOURS = {"red": (220, 30, 30), "green": (30, 180, 60), "blue": (40, 60, 210)}


@pytest.fixture(scope="module")
def backend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from anyjev.backends.hf_vlm import VLMBackend

    return VLMBackend(MODEL, batch_size=8)


def swatch(rgb, size=(224, 224)):
    from PIL import Image as PILImage

    return PILImage.new("RGB", size, rgb)


def test_reads_an_unambiguous_picture(backend):
    from anyjev import Decider, Question

    q = Question.choice("What colour fills this image?", list(COLOURS), name="colour")
    decs = Decider(backend).decide_batch([swatch(c) for c in COLOURS.values()], q)
    assert [d.argmax for d in decs] == list(COLOURS)
    assert all(d.level == "L0" and d.diagnostics["n_images"] == 1 for d in decs)


def test_batching_does_not_change_the_answer_distribution(backend):
    """Left padding plus image-grid position ids is where a VLM backend goes
    quietly wrong; prompts here differ in image size and text length."""
    from anyjev import Question
    from anyjev.readout import (
        answer_labels,
        build_prompt,
        label_ids_for_perm,
        map_label_tokens,
        prompt_images,
        render_chat,
    )
    from anyjev.state import split_state

    q = Question.choice("What colour fills this image?", list(COLOURS))
    base = map_label_tokens(backend.tokenizer, answer_labels(q))
    prompts, ids, images = [], [], []
    for n, rgb in enumerate(COLOURS.values()):
        text, imgs = split_state({"image": swatch(rgb, (224 + 112 * n, 224)), "note": "a swatch. " * (1 + 8 * n)})
        spec = build_prompt(text, q, [0, 1, 2], images=imgs)
        prompts.append(render_chat(backend.processor, spec))
        ids.append(label_ids_for_perm(q, base, [0, 1, 2]))
        images.append(prompt_images(spec))

    def probs(x):
        p = np.exp(np.asarray(x) - np.max(x))
        return p / p.sum()

    backend.batch_size = 1
    one = backend.next_token_logprobs(prompts, ids, images=images)
    backend.batch_size = 8
    many = backend.next_token_logprobs(prompts, ids, images=images)
    assert max(float(np.max(np.abs(probs(a) - probs(b)))) for a, b in zip(one, many)) < 1e-4


def test_hidden_states_is_the_same_forward_as_the_readout(backend):
    """L2 reads `hidden_states`; the label log-probs of that forward must equal the readout's."""
    from anyjev import Decider, Question
    from anyjev.readout import build_prompt, prompt_images, render_chat_parts
    from anyjev.state import split_state

    q = Question.choice("What colour fills this image?", list(COLOURS))
    d = Decider(backend)
    labels, ids = d._labels_for(q)
    prompts, images = [], []
    for rgb in COLOURS.values():
        text, imgs = split_state({"image": swatch(rgb)})
        spec = build_prompt(text, q, [0, 1, 2], d.system, labels, imgs)
        pre, suf = render_chat_parts(d.renderer, spec)
        prompts.append(pre + suf)
        images.append(prompt_images(spec))
    readout = np.stack(backend.next_token_logprobs(prompts, [ids] * 3, images=images))
    feats, lps, _ = backend.hidden_states(prompts, [backend.n_layers // 2, backend.n_layers],
                                          token_ids=[ids] * 3, images=images)
    assert feats.shape == (3, 2, backend.hidden_size)
    assert float(np.max(np.abs(readout - np.stack(lps)))) < 1e-5


def test_l2_head_on_pictures(backend):
    from anyjev import Decider, Question

    q = Question.choice("What colour fills this image?", list(COLOURS), name="colour")
    shades = [(dr, dg) for dr in (-20, 0, 20) for dg in (-20, 0, 20)]
    states, labels = [], []
    for j, rgb in enumerate(COLOURS.values()):
        for dr, dg in shades:
            states.append(swatch(tuple(min(255, max(0, c + s)) for c, s in zip(rgb, (dr, dg, 0)))))
            labels.append(j)
    d = Decider(backend)
    d.fit_head(q, states, labels)
    decs = d.decide_batch([swatch(c) for c in COLOURS.values()], q, level="L2")
    assert [x.argmax for x in decs] == list(COLOURS)
    assert all(x.level == "L2" and x.diagnostics["early_stop"] is False for x in decs)
