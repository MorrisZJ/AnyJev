"""Render the serving-time routing figure: which path one decision takes.

A portrait chart in the house style of assets/how_it_works.png. Every box is a
code path in anyjev/decider.py; the one number on it, the adaptation threshold,
is read from Decider's signature so the picture cannot drift from the code.

    python scripts/make_route_figure.py [-o assets/route_tree.png]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath

T = dict(bg_top="#ffffff", bg_bottom="#e8eefb", ink="#0f172a", dim="#64748b", faint="#94a3b8",
         accent="#2563eb", accent_soft="#dbe7ff", head_soft="#eef4ff", good="#0f9d76", good_soft="#dcfce7",
         warm="#d97706", warm_soft="#fef3c7", card="#ffffff", edge="#e2e8f0", pill="#e2e8f0",
         line="#64748b", dots=0.055, glow=0.016)

W, H = 12.0, 10.0   # inches; the axes use inch coordinates with equal aspect


def adapt_min_n(root: Path) -> int:
    src = (root / "anyjev" / "decider.py").read_text(encoding="utf-8")
    m = re.search(r"adapt_min_n:\s*int\s*=\s*(\d+)", src)
    if not m:
        raise SystemExit("could not find adapt_min_n in anyjev/decider.py")
    return int(m.group(1))


# ------------------------------------------------------------- drawing ----
def rbox(ax, x, y, w, h, *, fc, ec="none", lw=0, r=0.12, z=3, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha))


def card(ax, cx, cy, w, h, *, fc=None, ec=None):
    x, y = cx - w / 2, cy - h / 2
    rbox(ax, x + 0.025, y - 0.04, w, h, fc=T["ink"], alpha=0.05, r=0.14, z=2)
    rbox(ax, x, y, w, h, fc=fc or T["card"], ec=ec or T["edge"], lw=0.9, r=0.14, z=3)


def diamond(ax, cx, cy, hx, hy, text):
    ax.add_patch(Polygon([[cx, cy + hy], [cx + hx, cy], [cx, cy - hy], [cx - hx, cy]], closed=True,
                         facecolor=T["warm_soft"], edgecolor=T["warm"], linewidth=1.1, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9.4, color=T["ink"], zorder=6,
            linespacing=1.25)


def chip(ax, x, y, text, *, fc, tc, w=None, h=0.28, fs=8.4, z=6):
    w = w or (0.082 * len(text) + 0.28)
    rbox(ax, x, y, w, h, fc=fc, r=h / 2, z=z)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold", zorder=z + 1)
    return w


def pill(ax, cx, cy, text):
    w, h = 0.075 * len(text) + 0.26, 0.24
    rbox(ax, cx - w / 2, cy - h / 2, w, h, fc=T["pill"], r=h / 2, z=7)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.8, color=T["dim"], zorder=8)


def edge(ax, pts, *, color=None, lw=1.3):
    verts = [tuple(p) for p in pts]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes), arrowstyle="-|>", mutation_scale=12,
                                 color=color or T["line"], linewidth=lw, zorder=5, shrinkA=0, shrinkB=1))


def render(n_adapt: int, out: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")

    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", [T["bg_bottom"], T["bg_top"]])
    ax.imshow(grad, extent=(0, W, 0, H), aspect="auto", cmap=cmap, zorder=0)
    gx, gy = np.meshgrid(np.linspace(0.3, W - 0.3, 48), np.linspace(0.25, H - 0.25, 40))
    ax.scatter(gx, gy, s=1.6, c=T["ink"], alpha=T["dots"], marker="s", zorder=1, linewidths=0)
    for r in np.linspace(3.6, 0.3, 30):
        ax.add_patch(Circle((W * 0.42, H * 0.70), r, color=T["accent"], alpha=T["glow"] * 0.6,
                            zorder=1, linewidth=0))

    ax.text(0.5, 9.62, "ONE DECISION AT SERVING TIME", fontsize=9.2, fontweight="bold", color=T["dim"],
            va="center", zorder=6)
    ax.text(0.5, 9.27, "Which path a decision takes.", fontsize=18, fontweight="bold", color=T["ink"],
            va="center", zorder=6)
    ax.text(0.5, 8.96, "A stored head answers from one truncated forward. Without one, the same call falls back "
            "to L1 or L0, exactly as before.", fontsize=10, color=T["dim"], va="center", zorder=6)

    # ---- entry and the routing check ----
    card(ax, 5.0, 8.52, 2.6, 0.52)
    ax.text(5.0, 8.52, "state + question", ha="center", va="center", fontsize=11, fontweight="bold",
            color=T["ink"], zorder=6)
    edge(ax, [(5.0, 8.26), (5.0, 8.03)])
    diamond(ax, 5.0, 7.35, 1.05, 0.68, "route: a stored head\nfor this question?")

    # the bus from the check to the four branches
    bus_y, row_y, top_y = 6.42, 5.62, 6.00
    ax.plot([5.0, 5.0], [6.67, bus_y], color=T["line"], lw=1.3, zorder=4)
    ax.plot([1.75, 10.0], [bus_y, bus_y], color=T["line"], lw=1.3, zorder=4)
    for x in (1.75, 4.35, 6.95):
        edge(ax, [(x, bus_y), (x, top_y)])
    edge(ax, [(10.0, bus_y), (10.0, 6.22)])
    pill(ax, 1.75, 6.19, "exact layout")
    pill(ax, 4.35, 6.19, "same options, other wording")
    pill(ax, 6.95, 6.19, "same option set, other order")
    pill(ax, 8.72, bus_y, "none")

    # ---- the three head branches ----
    for cx, w, l1, l2 in ((1.75, 2.3, "head", "with its own μ, σ"),
                          (4.35, 2.5, "head + running μ, σ", "of this wording's requests"),
                          (6.95, 2.5, "head + running μ, σ", "probabilities remapped by option text")):
        card(ax, cx, row_y, w, 0.76, fc=T["head_soft"], ec="#c7d7fb")
        ax.text(cx, row_y + 0.14, l1, ha="center", va="center", fontsize=9.8, fontweight="bold",
                color=T["ink"], zorder=6)
        ax.text(cx, row_y - 0.15, l2, ha="center", va="center", fontsize=8.4, color=T["dim"], zorder=6)

    # adaptation from traffic
    ux, uy, uw, uh = 5.65, 4.38, 3.7, 0.76
    card(ax, ux, uy, uw, uh, fc=T["good_soft"], ec="#a7dcc8")
    ax.text(ux, uy + 0.14, "update sum, sum of squares, n for this question", ha="center", va="center",
            fontsize=9.4, fontweight="bold", color=T["ink"], zorder=6)
    ax.text(ux, uy - 0.15, f"use them once n ≥ {n_adapt}  ·  no labels involved", ha="center",
            va="center", fontsize=8.4, color=T["good"], zorder=6)
    edge(ax, [(4.35, row_y - 0.38), (4.35, 4.95), (5.0, 4.95), (5.0, uy + uh / 2)])
    edge(ax, [(6.95, row_y - 0.38), (6.95, 4.95), (6.3, 4.95), (6.3, uy + uh / 2)])

    # the truncated forward and the L2 decision
    fx, fy, fw, fh = 3.4, 3.05, 4.3, 1.0
    card(ax, fx, fy, fw, fh)
    ax.text(fx, fy + 0.22, "one prompt, forward to block b*", ha="center", va="center", fontsize=9.8,
            fontweight="bold", color=T["ink"], zorder=6)
    ax.text(fx, fy - 0.17, "p = softmax(((h − μ) / σ · W + b) / T)", ha="center", va="center",
            fontsize=9.2, color=T["accent"], family="DejaVu Sans Mono", zorder=6)
    edge(ax, [(1.75, row_y - 0.38), (1.75, fy + fh / 2)])
    edge(ax, [(ux, uy - uh / 2), (ux, 3.78), (4.6, 3.78), (4.6, fy + fh / 2)])

    dx, dy, dw, dh = 3.4, 1.55, 4.3, 1.04
    card(ax, dx, dy, dw, dh)
    edge(ax, [(fx, fy - fh / 2), (fx, dy + dh / 2)])
    ax.text(dx - dw / 2 + 0.3, dy + 0.24, "Decision", ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=T["ink"], zorder=6)
    chip(ax, dx - dw / 2 + 1.42, dy + 0.10, "level L2", fc=T["ink"], tc="white", w=0.9)
    ax.text(dx - dw / 2 + 0.3, dy - 0.17, "diagnostics: blocks_executed, routed_from,", ha="left",
            va="center", fontsize=8.4, color=T["dim"], zorder=6)
    ax.text(dx - dw / 2 + 0.3, dy - 0.38, "reordered, adapted, adapt_n", ha="left", va="center",
            fontsize=8.4, color=T["dim"], zorder=6)

    # ---- no head: the fallback column ----
    diamond(ax, 10.0, row_y, 1.0, 0.6, "temperature\nartifact?")
    fallbacks = ((4.15, T["good_soft"], "#a7dcc8", T["good"], "L1", "prior correction, temperature"),
                 (2.62, T["accent_soft"], "#c7d7fb", T["accent"], "L0", "prior correction"))
    for cy, fc, ec, tc, lvl, tail in fallbacks:
        card(ax, 10.0, cy, 2.7, 1.06, fc=fc, ec=ec)
        chip(ax, 10.0 - 1.35 + 0.28, cy + 0.20, lvl, fc=tc, tc="white", w=0.52)
        ax.text(10.0 - 1.35 + 0.92, cy + 0.34, "K shifted prompts, full forward,", ha="left", va="center",
                fontsize=8.4, color=T["ink"], zorder=6)
        ax.text(10.0 - 1.35 + 0.92, cy + 0.12, tail, ha="left", va="center", fontsize=8.4, color=T["ink"],
                zorder=6)
        ax.text(10.0 - 1.35 + 0.28, cy - 0.28, f"Decision, level {lvl}", ha="left", va="center", fontsize=9,
                fontweight="bold", color=tc, zorder=6)
    edge(ax, [(10.0, row_y - 0.6), (10.0, 4.15 + 0.53)])
    pill(ax, 10.0, 4.84, "yes")
    edge(ax, [(11.0, row_y), (11.55, row_y), (11.55, 2.62), (11.35, 2.62)])
    pill(ax, 11.55, 4.15, "no")

    ax.text(W - 0.5, 0.26, "every box is a code path in anyjev/decider.py  ·  "
            f"n ≥ {n_adapt} is Decider's adapt_min_n  ·  regenerate with python scripts/make_route_figure.py",
            fontsize=7.6, color=T["faint"], ha="right", va="center", zorder=6)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=T["bg_top"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="assets/route_tree.png")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    render(adapt_min_n(root), out if out.is_absolute() else root / out)


if __name__ == "__main__":
    main()
