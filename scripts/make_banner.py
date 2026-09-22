"""Render the README banner from committed bench JSON.

The three stats on the banner are read from the same files bench.table reads,
so the header cannot drift away from the measured numbers. Two themes are
written; it sits on the page rather than punching a dark block into it.

    python scripts/make_banner.py [results_dir]
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

THEME = dict(bg_top="#ffffff", bg_bottom="#e8eefb", ink="#0f172a", dim="#64748b",
             faint="#94a3b8", accent="#2563eb", good="#0f9d76",
             strip="#0f172a", strip_alpha=0.035, rule=0.10, dots=0.055, glow=0.016)


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


def stat(ax, t, x, label, before, after, *, pct=False):
    """One before -> after stat block, anchored at x in axes coords."""
    fmt = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.3f}")
    ax.text(x, 0.360, label.upper(), color=t["dim"], fontsize=9.5, va="center",
            fontweight="bold", transform=ax.transAxes, zorder=3)
    ax.text(x, 0.205, fmt(before), color=t["faint"], fontsize=21, va="center",
            fontweight="bold", transform=ax.transAxes, zorder=3)
    ax.text(x + 0.079, 0.205, "→", color=t["faint"], fontsize=17, va="center",
            transform=ax.transAxes, zorder=3)
    ax.text(x + 0.112, 0.205, fmt(after), color=t["good"], fontsize=25, va="center",
            fontweight="bold", transform=ax.transAxes, zorder=3)


def render(lv: dict, out: Path) -> None:
    t = THEME
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 5.0), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", [t["bg_bottom"], t["bg_top"]])
    ax.imshow(grad, extent=(0, 1, 0, 1), aspect="auto", cmap=cmap, zorder=0)

    gx, gy = np.meshgrid(np.linspace(0.02, 0.99, 60), np.linspace(0.05, 0.95, 16))
    ax.scatter(gx, gy, s=1.6, c=t["ink"], alpha=t["dots"], marker="s", zorder=1, linewidths=0)

    for r in np.linspace(0.46, 0.03, 40):
        ax.add_patch(plt.Circle((0.5, 0.72), r, color=t["accent"], alpha=t["glow"] * 0.7,
                                zorder=1, linewidth=0, transform=ax.transAxes))

    ax.text(0.5, 0.825, "AnyJev", color=t["ink"], fontsize=54, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes, zorder=3)
    ax.plot([0.463, 0.537], [0.700, 0.700], color=t["accent"], alpha=0.55,
            linewidth=2.2, solid_capstyle="round", transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.633, "Turn any LLM into a Jev-style decision model",
            color=t["ink"], fontsize=19, ha="center", va="center",
            transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.553, "Typed decisions  ·  real probabilities  ·  no training",
            color=t["accent"], fontsize=13.5, ha="center", va="center",
            transform=ax.transAxes, zorder=3)

    strip = FancyBboxPatch((0.035, 0.105), 0.93, 0.35, transform=ax.transAxes,
                           boxstyle="round,pad=0.006,rounding_size=0.012",
                           facecolor=t["strip"], alpha=t["strip_alpha"],
                           edgecolor="none", zorder=2)
    ax.add_patch(strip)

    stat(ax, t, 0.065, "order-flip rate", lv["raw"]["flip"], lv["L0"]["flip"])
    stat(ax, t, 0.375, "calibration error", lv["raw"]["ece"], lv["L1"]["ece"])
    stat(ax, t, 0.675, "auto-decidable at 5% risk", lv["raw"]["cov@5%"], lv["L1"]["cov@5%"], pct=True)
    for x in (0.335, 0.635):
        ax.plot([x, x], [0.145, 0.425], color=t["dim"], alpha=t["rule"], linewidth=1,
                transform=ax.transAxes, zorder=3)

    ax.text(0.965, 0.055, f"{HEADLINE[0]} · {HEADLINE[1]} · 300 items · measured, not claimed",
            color=t["faint"], fontsize=9, ha="right", va="center",
            transform=ax.transAxes, zorder=3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["bg_top"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=DEFAULT_DIR)
    ap.add_argument("-o", "--out", default="assets/banner.png")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir

    out = Path(args.out)
    render(load_headline(results_dir), out if out.is_absolute() else root / out)


if __name__ == "__main__":
    main()
