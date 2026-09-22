"""Real-model smoke for the multimodal backend, on pictures whose answer is
not in doubt.

    python scripts/smoke_vlm.py /mnt/persist/checkpoints/Qwen3-VL-2B-Instruct

Also checks batch parity. A vision model derives its position ids from the
image grid, so a backend that batches prompts of different lengths under left
padding can be quietly wrong; this compares batch size 1 against 8 on the same
prompts and prints the worst disagreement.
"""
import argparse
import json

import numpy as np
from PIL import Image as PILImage

from anyjev import Decider, Question
from anyjev.backends.hf_vlm import VLMBackend

COLOURS = {"red": (220, 30, 30), "green": (30, 180, 60), "blue": (40, 60, 210)}


def swatch(rgb, size=(224, 224)):
    return PILImage.new("RGB", size, rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=None)
    args = ap.parse_args()

    backend = VLMBackend(args.model, batch_size=args.batch_size, max_pixels=args.max_pixels)
    d = Decider(backend, record_content_free=True)

    colour = Question.choice("What colour fills this image?", list(COLOURS), name="colour")
    states = [swatch(rgb) for rgb in COLOURS.values()]
    truth = list(COLOURS)

    for level in ("raw", "L0"):
        decs = d.decide_batch(states, colour, level=level)
        got = [x.argmax for x in decs]
        print(f"{level:3} answers={got} truth={truth} "
              f"acc={np.mean([a == b for a, b in zip(got, truth)]):.2f} "
              f"conf={[round(x.confidence, 3) for x in decs]} "
              f"answer_mass={decs[0].diagnostics['answer_mass']:.3f}")

    # the three primitives on one state, the README example with a picture
    qs = [colour,
          Question.noul("Is this image mostly a warm colour?", name="warm"),
          Question.score("How saturated is this image, 0 to 1?", bins=5, name="sat")]
    r = d.decide({"screenshot": swatch(COLOURS["red"]), "note": "uploaded by the user"}, qs)
    print(json.dumps(r.to_dict(), indent=1))

    # Batch parity, on prompts that deliberately differ in length: different
    # image sizes and different caption lengths, so left padding and the image
    # grid both vary within a batch.
    from anyjev.readout import (
        answer_labels,
        build_prompt,
        label_ids_for_perm,
        map_label_tokens,
        prompt_images,
        render_chat,
    )
    from anyjev.state import split_state

    base = map_label_tokens(backend.tokenizer, answer_labels(colour))
    prompts, ids, images = [], [], []
    for n, (name, rgb) in enumerate(COLOURS.items()):
        size = (224 + 112 * n, 224)
        caption = "a swatch. " * (1 + 8 * n)
        text, imgs = split_state({"image": swatch(rgb, size), "note": caption})
        for perm in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
            spec = build_prompt(text, colour, perm, images=imgs)
            prompts.append(render_chat(backend.processor, spec))
            ids.append(label_ids_for_perm(colour, base, perm))
            images.append(prompt_images(spec))
    lengths = [len(backend.tokenizer.encode(p)) for p in prompts]
    backend.batch_size = 1
    one = backend.next_token_logprobs(prompts, ids, images=images)
    backend.batch_size = args.batch_size
    many = backend.next_token_logprobs(prompts, ids, images=images)

    def restricted(x):
        p = np.exp(np.asarray(x) - np.max(x))
        return p / p.sum()

    d_logp = max(float(np.max(np.abs(a - b))) for a, b in zip(one, many))
    d_prob = max(float(np.max(np.abs(restricted(a) - restricted(b)))) for a, b in zip(one, many))
    agree = sum(int(np.argmax(a) == np.argmax(b)) for a, b in zip(one, many))
    print(f"batch parity: prompts={len(prompts)} text_len={min(lengths)}..{max(lengths)} "
          f"max_abs_diff_logprob={d_logp:.5f} max_abs_diff_prob={d_prob:.2e} "
          f"argmax_agreement={agree}/{len(prompts)}")


if __name__ == "__main__":
    main()
