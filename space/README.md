---
title: AnyJev - reorder the options
emoji: 🔀
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Drag the option order. Raw logit readout flips. AnyJev does not.
---

# Reorder the options. Watch raw logit readout flip. Watch AnyJev not.

Every open "Jev clone" reads the next-token logits for the option labels and calls the softmax a
probability. Reverse the option list and one in five answers changes on open 7B to 30B models.
AnyJev reads all cyclic shifts of the list, combines them in log space, removes the model's label
prior, and labels every result by how much debiasing and calibration it carries. No training.

**Try it:** paste a support ticket as the state, list four teams as options, click decide. Then reverse
the option order in the textbox and click again. The left column is the raw readout, the right column is
AnyJev L0. The bars that move are the bug; the bars that do not are the fix.

Model: set by the `MODEL` variable (default `Qwen/Qwen3-1.7B` for CPU; use `Qwen/Qwen3-8B` on a GPU).
Library and benchmark: https://github.com/nokia-applied-research/AnyJev . `pip install anyjev`.
