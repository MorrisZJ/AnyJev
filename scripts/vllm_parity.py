"""Backend parity: VLLMBackend vs HFBackend on the same prompts.
    python scripts/vllm_parity.py --model Qwen/Qwen2.5-7B-Instruct --base-url http://127.0.0.1:8011
Reports max |diff| of the restricted log-softmax and argmax agreement over a
few choice / noul / score prompts. vLLM reports logprobs after allowed_token_ids
masking, so both sides are renormalized over the label set before comparing.
"""
import argparse

import numpy as np

from anyjev import Question
from anyjev.readout import (
    answer_labels,
    build_prompt,
    label_ids_for_perm,
    map_label_tokens,
    render_chat,
)
from anyjev.state import render_state


def lsm(x):
    x = np.asarray(x, float)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    args = ap.parse_args()
    from anyjev.backends.hf import HFBackend
    from anyjev.backends.vllm import VLLMBackend
    hf = HFBackend(args.model, batch_size=8)
    vl = VLLMBackend(args.base_url, args.model)
    qs = [Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"]),
          Question.noul("Is this message a complaint?"),
          Question.score("How urgent is this?", levels=["not", "low", "medium", "high"])]
    states = ["My card was charged twice.", "The app crashes on login.", "Do you offer bulk discounts?",
              "Thanks, all sorted now!", "URGENT: production is down for all users."]
    prompts, ids = [], []
    for q in qs:
        base = map_label_tokens(hf.tokenizer, answer_labels(q))
        perm = list(range(q.k))
        for s in states:
            prompts.append(render_chat(hf.tokenizer, build_prompt(render_state(s), q, perm)))
            ids.append(label_ids_for_perm(q, base, perm))
    a = hf.next_token_logprobs(prompts, ids)
    b = vl.next_token_logprobs(prompts, ids)
    diffs, agree = [], 0
    for x, y in zip(a, b):
        diffs.append(float(np.max(np.abs(lsm(x) - lsm(y)))))
        agree += int(np.argmax(x) == np.argmax(y))
    print(f"prompts={len(prompts)} max_abs_diff_restricted_logsoftmax={max(diffs):.4f} "
          f"mean={np.mean(diffs):.4f} argmax_agreement={agree}/{len(prompts)}")
    print("vllm raw sample:", np.round(b[0], 3), "hf:", np.round(a[0], 3))


if __name__ == "__main__":
    main()
