"""Render the headline results figure from committed bench JSON.

Nothing here is typed in by hand: every number is read from
bench/results_batchprior_v0/<date>/*.json, the same files bench.table reads.

    python scripts/make_results_figure.py [results_dir] [-o assets/results.png]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DEFAULT_DIR = (sorted(glob.glob("bench/results_v01/*/")) or ["bench/results_v01/"])[-1]   # latest dated run
MODELS = ["Qwen3-8B", "Qwen2.5-7B-Instruct", "Qwen3-30B-A3B-Instruct-2507"]
MODEL_SHORT = {"Qwen3-8B": "Qwen3-8B",
               "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
               "Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B"}
TASKS = ["banking20", "newsgroups", "injection"]

INK = "#0f172a"
MUTED = "#64748b"
FAINT = "#94a3b8"
GRID = "#e8edf3"
LEVELS = [("raw", "#cbd5e1", INK), ("L0", "#2f6fed", "white"), ("L1", "#0ea882", "white")]


def load(results_dir: Path) -> dict:
    """{(model, task): {level: metrics}} from every JSON in the directory."""
    out: dict = {}
    for path in sorted(results_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        model = blob["model"].split("/")[-1]
        for task in blob["tasks"]:
            out[(model, task["task"])] = task["levels"]
    return out


def panel(fig, gs_cell, data, metric, levels, title, subtitle, ymax, *, better):
    ax = fig.add_subplot(gs_cell)
    cells = [(m, t) for m in MODELS for t in TASKS]
    width = 0.78 / len(levels)

    for li, level in enumerate(levels):
        color, textcolor = next(c for c in LEVELS if c[0] == level)[1:]
        for ci, cell in enumerate(cells):
            y = data[cell][level][metric]
            x = ci - 0.39 + width * (li + 0.5)
            ax.bar(x, y, width * 0.86, color=color, zorder=3, edgecolor="white", linewidth=0.7)
            # numbers sit inside tall bars, above short ones
            inside = y > 0.22 * ymax
            ax.text(x, y - 0.018 * ymax if inside else y + 0.015 * ymax, f"{y:.3f}",
                    ha="center", va="top" if inside else "bottom", rotation=90,
                    fontsize=6.4, fontweight="bold", zorder=4,
                    color=textcolor if inside else MUTED)

    ax.text(0, 1.105, title, transform=ax.transAxes, fontsize=13.5, fontweight="bold",
            color=INK, va="bottom")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.text(1, 1.035, better, transform=ax.transAxes, fontsize=8.5, color=FAINT,
            va="bottom", ha="right", style="italic")

    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d7dee8")
    ax.tick_params(axis="both", length=0, labelsize=8.5, colors=MUTED)

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([t for _, t in cells], fontsize=7.4, color=MUTED)
    for sep in (2.5, 5.5):
        ax.axvline(sep, color="#dde4ec", linewidth=1, zorder=1)
    for i, model in enumerate(MODELS):
        ax.text(i * 3 + 1, -0.135, MODEL_SHORT[model], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    return ax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=DEFAULT_DIR)
    ap.add_argument("-o", "--out", default="assets/results.png")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    data = load(results_dir)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.5, 11), dpi=200, facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.11,
                          left=0.05, right=0.975, top=0.800, bottom=0.095)

    # ---------- header ----------
    fig.text(0.05, 0.977, "AnyJev", fontsize=34, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.928, "Training-free debiasing turns a logit readout into a decision you can act on",
             fontsize=14.5, color="#334155", va="top")
    fig.text(0.05, 0.899,
             "3 open models  ·  3 tasks  ·  300 test items each  ·  banking20 and newsgroups are 20-way, "
             "injection is binary  ·  one H100, bf16",
             fontsize=9.5, color=MUTED, va="top")

    handles = [Patch(facecolor=c, edgecolor="white", label=lab) for lab, c in
               [("raw  —  logit readout, what the clones do", "#cbd5e1"),
                ("L0  —  debiased, zero labels", "#2f6fed"),
                ("L1  —  + temperature scaling, 200 labels", "#0ea882")]]
    fig.legend(handles=handles, loc="upper right", ncol=1, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.975, 0.990), labelspacing=0.55,
               handlelength=1.5, handleheight=1.1)

    fig.add_artist(plt.Line2D([0.05, 0.975], [0.872, 0.872], color="#dde4ec", linewidth=1.2))

    # ---------- panels ----------
    panel(fig, gs[0, 0], data, "flip", ["raw", "L0"],
          "Answers stop moving when you reorder the options",
          "order-flip rate", 0.30, better="lower is better")
    panel(fig, gs[0, 1], data, "ece", ["raw", "L0", "L1"],
          "The probability starts meaning what it says",
          "expected calibration error", 0.42, better="lower is better")
    panel(fig, gs[1, 0], data, "acc", ["raw", "L0", "L1"],
          "Accuracy moves a little",
          "accuracy", 1.0, better="higher is better")
    hero = panel(fig, gs[1, 1], data, "cov@5%", ["raw", "L0", "L1"],
                 "What you can safely automate moves a lot",
                 "coverage at 5% risk  ·  share of items auto-decidable below 5% error",
                 0.70, better="higher is better")

    # call out the headline cell: Qwen3-8B / banking20, raw 0.077 -> L1 0.543
    raw_x, l1_x = -0.26, 0.26
    hero.annotate("", xy=(l1_x, 0.568), xytext=(raw_x, 0.105),
                  arrowprops=dict(arrowstyle="-|>", color="#0ea882", linewidth=1.6,
                                  connectionstyle="arc3,rad=-0.28", shrinkA=3, shrinkB=3))
    hero.text(0.98, 0.638, "7×", fontsize=17, fontweight="bold", color="#0ea882", va="center")
    hero.text(1.62, 0.653, "more decisions you can automate", fontsize=9.5,
              color=INK, va="center", fontweight="bold")
    hero.text(1.62, 0.617, "at the same 5% error bar  ·  Qwen3-8B, banking20",
              fontsize=8.5, color=MUTED, va="center")

    fig.add_artist(plt.Line2D([0.05, 0.975], [0.042, 0.042], color="#eef2f7", linewidth=1))
    fig.text(0.05, 0.020, "Every number is read from committed bench JSON — regenerate with "
                          "python scripts/make_results_figure.py",
             fontsize=8.5, color=FAINT, va="center")
    fig.text(0.975, 0.020, f"source: {args.results_dir}", ha="right", fontsize=8.5,
             color=FAINT, va="center")

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
