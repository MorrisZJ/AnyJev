"""Reorder the options, watch raw readout flip and AnyJev hold still.

    pip install gradio
    python demo/app.py --model Qwen/Qwen3-8B

One model, one state, one choice question. Left: raw next-token readout with
the options in the order you typed them and in reverse. Right: AnyJev L0 on
the same two orders. The bars that move are the bug; the bars that do not
are the fix.
"""
from __future__ import annotations

import argparse

from anyjev import Decider, Question

DEFAULT_STATE = (
    "Customer message: Hi, I was charged twice for order #4471 last Tuesday. "
    "The app also crashes every time I open the receipts page. Can someone help?"
)
DEFAULT_OPTIONS = "billing\ntechnical\nsales\nother"
DEFAULT_QUESTION = "Which team should handle this message?"


def build(decider: Decider):
    import gradio as gr

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    from anyjev.backends.hf import HFBackend
    decider = Decider(HFBackend(args.model))
    build(decider).launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
