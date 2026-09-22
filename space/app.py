"""Hugging Face Space entry point. Reorder the options, watch raw readout flip and AnyJev hold still.

    pip install gradio
    python space/app.py --model Qwen/Qwen3-8B

One model, one state, one choice question. Left: raw next-token readout with
the options in the order you typed them and in reverse. Right: AnyJev L0 on
the same two orders. The bars that move are the bug; the bars that do not
are the fix.

The model default follows the hardware. L0 runs one prefill per cyclic shift,
so a four-option question costs four prefills per order and the page renders
two orders. On a CPU Space that is eight prefills per click, which is why the
CPU default is the 0.6B model that the maze benchmark already reports.
"""
from __future__ import annotations

import argparse
import os

from anyjev import Decider
from anyjev.question import Question

# ZeroGPU attaches a GPU only while a decorated function runs, so the decorator
# has to wrap the inference call. The environment variable is the reliable
# signal: torch sees no CUDA device at startup on ZeroGPU, so probing for one
# would pick the CPU path and load the small model on an A10G. The import is
# guarded broadly because a failure here must not take the Space down, and
# without it the decorator becomes a no-op on ordinary hardware.
ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true", "yes")

try:  # pragma: no cover - depends on the hosting environment
    import spaces

    def on_gpu(fn):
        return spaces.GPU(duration=120)(fn)
except Exception as exc:  # pragma: no cover
    if ZERO_GPU:
        print(f"[anyjev] ZeroGPU requested but 'spaces' is unusable: {exc!r}")
        ZERO_GPU = False

    def on_gpu(fn):
        return fn

GPU_MODEL = "Qwen/Qwen3-8B"
CPU_MODEL = "Qwen/Qwen3-0.6B"

DEFAULT_STATE = (
    "Customer message: Hi, I was charged twice for order #4471 last Tuesday. "
    "The app also crashes every time I open the receipts page. Can someone help?"
)
DEFAULT_OPTIONS = "billing\ntechnical\nsales\nother"
DEFAULT_QUESTION = "Which team should handle this message?"


def build(decider: Decider, model_name: str, device: str):
    import gradio as gr

    @on_gpu
    def run(state, question, options_text):
        opts = [o.strip() for o in options_text.splitlines() if o.strip()]
        if len(opts) < 2:
            return {}, {}, {}, {}, "need at least two options"
        q = Question.choice(question, opts, name="q")
        qr = Question(q.kind, q.text, tuple(reversed(opts)), "q", q.scale, q.ordered)
        a = decider.decide(state, [q])["q"]
        b = decider.decide(state, [qr])["q"]
        raw_a = dict(zip(opts, map(float, a.diagnostics["raw_probs"])))
        raw_b = dict(zip(reversed(opts), map(float, b.diagnostics["raw_probs"])))
        l0_a = a.distribution
        l0_b = b.distribution
        flip_raw = max(raw_a, key=raw_a.get) != max(raw_b, key=raw_b.get)
        flip_l0 = max(l0_a, key=l0_a.get) != max(l0_b, key=l0_b.get)
        msg = (f"raw: {'FLIPPED' if flip_raw else 'same answer'} when options reversed  |  "
               f"AnyJev L0: {'flipped' if flip_l0 else 'same answer'}  |  "
               f"label-token mass {a.diagnostics['answer_mass']:.3f}  |  level {a.level}")
        return raw_a, raw_b, l0_a, l0_b, msg

    with gr.Blocks(title="AnyJev: reorder the options") as demo:
        gr.Markdown("# Reorder the options. Watch raw logit readout flip. Watch AnyJev not.")
        gr.Markdown(
            f"Running `{model_name}` on {device}. Reverse the option list and click again. "
            "The raw readout is what every logit-reading clone does. "
            "[Library and benchmark](https://github.com/MorrisZJ/AnyJev)."
        )
        with gr.Row():
            state = gr.Textbox(DEFAULT_STATE, label="state", lines=4)
            with gr.Column():
                question = gr.Textbox(DEFAULT_QUESTION, label="question")
                options = gr.Textbox(DEFAULT_OPTIONS, label="options, one per line", lines=4)
        btn = gr.Button("decide (original order and reversed order)")
        with gr.Row():
            r1 = gr.Label(label="raw, options as typed")
            r2 = gr.Label(label="raw, options reversed")
        with gr.Row():
            l1 = gr.Label(label="AnyJev L0, options as typed")
            l2 = gr.Label(label="AnyJev L0, options reversed")
        msg = gr.Markdown()
        btn.click(run, [state, question, options], [r1, r2, l1, l2, msg])
    return demo


def main():
    import torch
    from anyjev.backends.hf import HFBackend

    # On ZeroGPU the GPU is present for decorated calls even though the process
    # starts without one, so trust the package rather than probing for CUDA.
    if ZERO_GPU:
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "MODEL", GPU_MODEL if device == "cuda" else CPU_MODEL))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = ap.parse_args()

    dtype = "bfloat16" if device == "cuda" else "float32"
    decider = Decider(HFBackend(args.model, device=device, dtype=dtype, batch_size=8))
    build(decider, args.model, device).launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
