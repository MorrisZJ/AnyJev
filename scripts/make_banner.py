"""Render the README banner from committed bench JSON.

The three stats on the banner are read from the same files bench.table reads,
so the header cannot drift away from the measured numbers.

    python scripts/make_banner.py [results_dir] [-o assets/banner.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

DEFAULT_DIR = "bench/results_batchprior_v0/2026-09-20"
HEADLINE = ("Qwen3-8B", "banking20")

BG_TOP = "#0a1020"
BG_BOTTOM = "#16263f"
WHITE = "#f8fafc"
DIM = "#8fa3bf"
BLUE = "#5b9dff"
GREEN = "#2ee6a8"


def load_headline(results_dir: Path) -> dict:
    model, task = HEADLINE
    for path in sorted(results_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        if blob["model"].split("/")[-1] != model:
            continue
        for entry in blob["tasks"]:
            if entry["task"] == task:
                return entry["levels"]
    raise SystemExit(f"no results for {model}/{task} in {results_dir}")


def stat(ax, x, label, before, after, *, pct=False):
    """One before -> after stat block, anchored at x in axes coords."""
    fmt = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.3f}")
    ax.text(x, 0.335, label.upper(), color=DIM, fontsize=9.5, va="center",
            fontweight="bold", transform=ax.transAxes)
    ax.text(x, 0.185, fmt(before), color="#5b6b82", fontsize=21, va="center",
            fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.079, 0.185, "→", color="#41536d", fontsize=17, va="center",
            transform=ax.transAxes)
    ax.text(x + 0.112, 0.185, fmt(after), color=GREEN, fontsize=25, va="center",
            fontweight="bold", transform=ax.transAxes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=DEFAULT_DIR)
    ap.add_argument("-o", "--out", default="assets/banner.png")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    lv = load_headline(results_dir)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 4.4), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # vertical gradient backdrop
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", [BG_BOTTOM, BG_TOP])
    ax.imshow(grad, extent=(0, 1, 0, 1), aspect="auto", cmap=cmap, zorder=0)

    # faint dot grid, denser to the right
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.linspace(0.02, 0.99, 60), np.linspace(0.05, 0.95, 16))
    ax.scatter(gx, gy, s=1.6, c="#ffffff", alpha=0.045 + 0.05 * gx / gx.max(),
               marker="s", zorder=1, linewidths=0)

    # accent glow behind the wordmark: many faint rings so no edge is visible
    for r in np.linspace(0.34, 0.03, 26):
        ax.add_patch(plt.Circle((0.105, 0.73), r, color=BLUE, alpha=0.012, zorder=1,
                                linewidth=0, transform=ax.transAxes))

    ax.text(0.042, 0.745, "AnyJev", color=WHITE, fontsize=58, fontweight="bold",
            va="center", transform=ax.transAxes, zorder=3)
    ax.text(0.30, 0.775, "Turn any LLM into a Jev-style decision model",
            color=WHITE, fontsize=19, va="center", transform=ax.transAxes, zorder=3)
    ax.text(0.30, 0.685, "Typed decisions  ·  real probabilities  ·  no training",
            color=BLUE, fontsize=13.5, va="center", transform=ax.transAxes, zorder=3)

    # stat strip
    strip = FancyBboxPatch((0.035, 0.10), 0.93, 0.34, transform=ax.transAxes,
                           boxstyle="round,pad=0.006,rounding_size=0.012",
                           facecolor="#ffffff", alpha=0.045, edgecolor="#ffffff",
                           linewidth=0.8, zorder=2)
    strip.set_edgecolor((1, 1, 1, 0.09))
    ax.add_patch(strip)

    stat(ax, 0.065, "order-flip rate", lv["raw"]["flip"], lv["L0"]["flip"])
    stat(ax, 0.375, "calibration error", lv["raw"]["ece"], lv["L1"]["ece"])
    stat(ax, 0.675, "auto-decidable at 5% risk", lv["raw"]["cov@5%"], lv["L1"]["cov@5%"], pct=True)
    for x in (0.335, 0.635):
        ax.plot([x, x], [0.14, 0.40], color="#ffffff", alpha=0.10, linewidth=1,
                transform=ax.transAxes, zorder=3)

    ax.text(0.965, 0.055, f"{HEADLINE[0]} · {HEADLINE[1]} · 300 items · measured, not claimed",
            color="#5b6b82", fontsize=9, ha="right", va="center", transform=ax.transAxes, zorder=3)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG_TOP)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
