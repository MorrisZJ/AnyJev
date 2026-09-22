"""Render assets/flip.gif from one entry of assets/flip_examples.json (real model outputs).

Four bar panels: raw as typed, raw reversed, L0 as typed, L0 reversed. Frames step through
"type options" -> "raw answers" -> "reverse options" -> "raw flips" -> "AnyJev L0 holds".
    python scripts/make_flip_gif.py --index 0 --out assets/flip.gif
"""
import argparse
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def panel(ax, dist, order, title, highlight, muted=False):
    ax.clear()
    names = order
    vals = [dist[n] for n in names]
    colors = ["#c0392b" if (n == highlight and not muted) else ("#bbbbbb" if muted else "#4a6fa5") for n in names]
    ax.barh(range(len(names))[::-1], vals, color=colors)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels([n if len(n) <= 30 else n[:28] + "…" for n in names], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_title(title, fontsize=12, loc="left")
    for i, v in enumerate(vals):
        if v > 0.75:
            ax.text(v - 0.02, len(names) - 1 - i, f"{v:.2f}", va="center", ha="right", fontsize=10, color="white")
        else:
            ax.text(v + 0.02, len(names) - 1 - i, f"{v:.2f}", va="center", fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="assets/flip_examples.json")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="assets/flip.gif")
    args = ap.parse_args()
    data = json.load(open(args.examples))
    ex = data["examples"][args.index]
    all_opts = ex["options"]
    # keep the panels readable: the options that matter in any of the four readouts, in typed order
    keep = set()
    for k in ("raw_as_typed", "raw_reversed", "l0_as_typed", "l0_reversed"):
        keep.update(sorted(ex[k], key=ex[k].get, reverse=True)[:3])
    keep.update([ex["gold"]])
    opts = [o for o in all_opts if o in keep][:7]
    rev = opts[::-1]
    hidden = len(all_opts) - len(opts)
    model = data["model"].split("/")[-1]
    state = textwrap.fill(ex["state"], 70)

    frames = []
    steps = [
        ("Options as typed. Raw next-token readout answers:", ["raw_as_typed"], [opts], [ex["raw_argmax"][0]], False),
        ("Reverse the option order. Raw readout now answers:", ["raw_as_typed", "raw_reversed"], [opts, rev],
         [ex["raw_argmax"][0], ex["raw_argmax"][1]], False),
        ("Same model, same prompts, AnyJev L0 (zero labels):", ["raw_as_typed", "raw_reversed", "l0_as_typed"],
         [opts, rev, opts], [ex["raw_argmax"][0], ex["raw_argmax"][1], ex["l0_argmax"]], False),
        ("Reverse again. AnyJev holds:", ["raw_as_typed", "raw_reversed", "l0_as_typed", "l0_reversed"],
         [opts, rev, opts, rev], [ex["raw_argmax"][0], ex["raw_argmax"][1], ex["l0_argmax"], ex["l0_argmax"]], False),
    ]
    titles = {"raw_as_typed": "raw, options as typed", "raw_reversed": "raw, options reversed",
              "l0_as_typed": "AnyJev L0, as typed", "l0_reversed": "AnyJev L0, reversed"}
    for caption, keys, orders, highs, _ in steps:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
        fig.suptitle(f"{model}  |  {data['question']}", fontsize=12, y=0.99)
        fig.text(0.02, 0.915, "state: " + state, fontsize=10, family="monospace", va="top")
        fig.text(0.02, 0.06, caption, fontsize=12, weight="bold", color="#222")
        if hidden:
            fig.text(0.02, 0.02, f"{len(all_opts)} options in the prompt; showing the {len(opts)} that get probability mass. "
                    
                     f"Order shown = order in the prompt. Red = the answer.", fontsize=8.5, color="#555")
        for ax in axes.flat:
            ax.axis("off")
        for k, order, high, ax in zip(keys, orders, highs, axes.flat):
            ax.axis("on")
            panel(ax, ex[k], order, titles[k], high)
        fig.subplots_adjust(top=0.80, bottom=0.13, left=0.24, right=0.985, hspace=0.6, wspace=0.75)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = Image.frombuffer("RGBA", (w, h), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
        frames.append(img)
        plt.close(fig)
    durations = [1800, 2200, 2200, 3200]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    frames[-1].save(args.out.replace(".gif", "_last.png"))
    print("wrote", args.out, "frames", len(frames))


if __name__ == "__main__":
    main()
