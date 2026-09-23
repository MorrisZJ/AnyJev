"""Render the "a head that maintains itself" figure from committed bench JSON.

Four steps in the house style of assets/how_it_works.png: fit, head, serve, adapt.
Every number on the figure is read from the result files the docs cite, so the
picture cannot drift away from the tables:

    bench/results_paraphrase/<date>/Qwen__Qwen3-8B.paraphrase.b24.json   rewording, recentring, refit
    bench/results_exit/<date>/Qwen__Qwen3-8B.jevmode.json                fixed block and its accuracy
    bench/results_exit/<date>/Qwen__Qwen3-8B.latency.json                cost against a plain forward
    bench/results_exit/<date>/Qwen__Qwen3-8B.artifact.json               seconds per closed-form solve
    anyjev-heads/*.json                                                   shipped heads, size on disk

    python scripts/make_head_loop_figure.py [-o assets/head_loop.png]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath

MODEL = "Qwen3-8B"
WORDINGS = ("w1", "w2", "w3")

T = dict(bg_top="#ffffff", bg_bottom="#e8eefb", ink="#0f172a", dim="#64748b", faint="#94a3b8",
         accent="#2563eb", accent_soft="#dbe7ff", good="#0f9d76", good_soft="#dcfce7",
         warm="#d97706", warm_soft="#fef3c7", card="#ffffff", edge="#e2e8f0", rail="#e6ebf3",
         chip_grey="#e2e8f0", dots=0.055, glow=0.016)


# ---------------------------------------------------------------- data ----
def latest(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no file matches {pattern}")
    return Path(hits[-1])


def load(root: Path) -> dict:
    tag = f"Qwen__{MODEL}"
    jev = json.loads(latest(str(root / f"bench/results_exit/*/{tag}.jevmode.json")).read_text())
    art = json.loads(latest(str(root / f"bench/results_exit/*/{tag}.artifact.json")).read_text())
    lat = json.loads(latest(str(root / f"bench/results_exit/*/{tag}.latency.json")).read_text())
    block = int(jev["recommended_block"])
    n_blocks = int(jev["n_blocks"])
    # the rewording study at the block the heads actually ship at, not whichever file sorts last
    para = json.loads(latest(str(root / f"bench/results_paraphrase/*/{tag}.paraphrase.b{block}.json")).read_text())
    assert int(para["block"]) == block, (para["block"], block)

    rows = para["rows"]
    pb = jev["per_block"][str(block)]

    plain = [r for r in lat["rows"] if r["blocks"] == n_blocks and r["mode"].startswith("raw")]
    trunc = [r for r in lat["rows"] if r["blocks"] == block and r["mode"].startswith("truncated")]
    cost = None
    if plain and trunc:
        st = plain[0]["state_tokens"]
        p0 = next((r for r in plain if r["state_tokens"] == st), plain[0])
        t0 = next((r for r in trunc if r["state_tokens"] == st), trunc[0])
        cost = t0["batch_ms"] / p0["batch_ms"]

    fits = [v["fit_seconds"] for v in art["validation"].values() if "fit_seconds" in v]
    n_calib = sorted({v["n_calib"] for v in art["validation"].values() if "n_calib" in v})

    heads = []
    for f in sorted(glob.glob(str(root / "anyjev-heads/*.json"))):
        d = json.loads(Path(f).read_text())
        heads.append((Path(f).stat().st_size / 1e6, len(d.get("heads") or d.get("artifacts") or {})))

    return dict(
        as_is=rows["orig-head@orig"]["acc"],
        reworded=[rows[f"orig-head@{w}"]["acc"] for w in WORDINGS],
        recentred=[rows[f"ta@30-head@{w}"]["acc"] for w in WORDINGS],
        refit=[rows[f"own-head@{w}"]["acc"] for w in WORDINGS],
        n_unlabelled=30,
        block=block, n_blocks=n_blocks, depth=pb["relative_depth"], acc=pb["acc"], ece=pb["ece"],
        cost=cost, fit_min=min(fits), fit_max=max(fits), n_calib=n_calib,
        n_models=len(heads), heads_per_model=sorted({h for _, h in heads}),
        mb_min=min(s for s, _ in heads), mb_max=max(s for s, _ in heads),
    )


def rng(vals, nd=2) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}–{hi:.{nd}f}"


# ------------------------------------------------------------- drawing ----
# The axes are in inches: xlim (0, 16), ylim (0, 5), aspect equal, so circles are circles.
W, H = 16.0, 5.0


def rbox(ax, x, y, w, h, *, fc, ec="none", lw=0, r=0.12, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha))


def card(ax, x, y, w, h):
    rbox(ax, x + 0.03, y - 0.05, w, h, fc=T["ink"], alpha=0.05, r=0.16, z=2)
    rbox(ax, x, y, w, h, fc=T["card"], ec=T["edge"], lw=0.8, r=0.16, z=3)


def chip(ax, x, y, text, *, fc, tc, w=None, h=0.30, fs=8.6, bold=True, z=5, align="left"):
    w = w or (0.083 * len(text) + 0.26)
    if align == "right":
        x = x - w
    rbox(ax, x, y, w, h, fc=fc, r=h / 2, z=z)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=z + 1)
    return w


def chevrons(ax, x, y, n=3, color=None, z=4):
    color = color or T["accent"]
    for i in range(n):
        cx = x + i * 0.13
        ax.add_patch(Polygon([[cx, y - 0.16], [cx + 0.11, y], [cx, y + 0.16],
                              [cx + 0.045, y + 0.16], [cx + 0.155, y], [cx + 0.045, y - 0.16]],
                             closed=True, facecolor=color, edgecolor="none",
                             alpha=0.28 + 0.24 * i, zorder=z))


def header(ax, x, y, n, title, sub):
    ax.add_patch(Circle((x + 0.24, y), 0.165, facecolor=T["accent"], edgecolor="none", zorder=5))
    ax.text(x + 0.24, y, str(n), ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="white", zorder=6)
    ax.text(x + 0.54, y + 0.01, title, ha="left", va="center", fontsize=14.5, fontweight="bold",
            color=T["ink"], zorder=6)
    ax.text(x, y - 0.42, sub, ha="left", va="center", fontsize=9.6, color=T["dim"], zorder=6)


def bus(ax, x_from, x_to, y_top, y_bot, *, color, lw, label, z=6, r=0.16):
    """A return path under the cards: down from x_from, left along y_bot, up into x_to."""
    verts = [(x_from, y_top), (x_from, y_bot + r), (x_from, y_bot), (x_from - r, y_bot),
             (x_to + r, y_bot), (x_to, y_bot), (x_to, y_bot + r), (x_to, y_top)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO]
    ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes), arrowstyle="-|>", mutation_scale=13,
                                 color=color, linewidth=lw, zorder=z, shrinkA=0, shrinkB=1))
    ax.text((x_from + x_to) / 2, y_bot + 0.13, label, fontsize=8.4, color=color, ha="center",
            va="center", fontweight="bold", zorder=z + 1)


def render(d: dict, out: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")

    # background: the banner's gradient, dot grid and glow
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", [T["bg_bottom"], T["bg_top"]])
    ax.imshow(grad, extent=(0, W, 0, H), aspect="auto", cmap=cmap, zorder=0)
    gx, gy = np.meshgrid(np.linspace(0.3, W - 0.3, 64), np.linspace(0.25, H - 0.25, 20))
    ax.scatter(gx, gy, s=1.6, c=T["ink"], alpha=T["dots"], marker="s", zorder=1, linewidths=0)
    for r in np.linspace(3.4, 0.3, 30):
        ax.add_patch(Circle((W * 0.5, H * 0.66), r, color=T["accent"], alpha=T["glow"] * 0.6,
                            zorder=1, linewidth=0))

    # title block
    ax.text(0.5, 4.64, "A HEAD THAT MAINTAINS ITSELF", fontsize=9.2, fontweight="bold",
            color=T["dim"], va="center", zorder=6)
    ax.text(0.5, 4.27, "One closed-form solve. After that the head follows its question by itself.",
            fontsize=19.5, fontweight="bold", color=T["ink"], va="center", zorder=6)

    # four cards
    cw, ch, gap, x0, y0 = 3.48, 2.92, 0.38, 0.5, 0.98
    xs = [x0 + i * (cw + gap) for i in range(4)]
    for x in xs:
        card(ax, x, y0, cw, ch)
    for i in range(3):
        chevrons(ax, xs[i] + cw + 0.06, y0 + ch * 0.52)
    top = y0 + ch - 0.36            # header baseline
    l1, l2, l3 = top - 2.00, top - 2.22, top - 2.44   # three footer lines inside a card

    # ---- 1. fit ----
    x = xs[0]
    header(ax, x + 0.28, top, 1, "Fit", f"{rng(d['n_calib'], 0)} labelled states per question")
    for k in range(3):
        yy = top - 0.98 - k * 0.30
        rbox(ax, x + 0.30, yy, 2.86, 0.22, fc=T["rail"], r=0.06, z=5)
        ax.add_patch(Circle((x + 0.46, yy + 0.11), 0.055, facecolor=T["accent"], zorder=6))
        rbox(ax, x + 0.62, yy + 0.075, 1.35 - 0.25 * k, 0.07, fc=T["faint"], r=0.03, z=6, alpha=0.55)
        chip(ax, x + 2.52, yy + 0.02, "label", fc=T["good_soft"], tc=T["good"], w=0.58, h=0.18, fs=6.8)
    ax.text(x + 0.30, l1, "one forward, one closed-form solve", fontsize=9.6, color=T["ink"],
            fontweight="bold", va="center", zorder=6)
    ax.text(x + 0.30, l2, f"{d['fit_min']:.0f}–{d['fit_max']:.0f} s per head  ·  no gradients",
            fontsize=8.8, color=T["dim"], va="center", zorder=6)
    ax.text(x + 0.30, l3, "the model's weights are untouched", fontsize=8.8, color=T["dim"],
            va="center", zorder=6)

    # ---- 2. head ----
    x = xs[1]
    header(ax, x + 0.28, top, 2, "Head", "one per model and question")
    rbox(ax, x + 0.30, top - 1.66, 2.88, 0.92, fc="#f8fafc", ec=T["edge"], lw=0.8, r=0.10, z=5)
    ax.text(x + 0.48, top - 0.95, "artifact", fontsize=8.4, color=T["faint"], va="center", zorder=6,
            fontweight="bold")
    cx = x + 0.48
    for name, fc, tc in (("W", T["accent"], "white"), ("b", T["accent"], "white"),
                         ("μ", T["good"], "white"), ("σ", T["good"], "white"),
                         ("T", T["chip_grey"], T["ink"])):
        cx += chip(ax, cx, top - 1.47, name, fc=fc, tc=tc, w=0.42, h=0.34, fs=10) + 0.10
    ax.text(x + 0.30, l1, "solved once", fontsize=9.2, color=T["accent"], va="center",
            fontweight="bold", zorder=6)
    ax.text(x + 1.36, l1, "re-estimated from traffic", fontsize=9.2, color=T["good"],
            va="center", fontweight="bold", zorder=6)
    hp = "/".join(map(str, d["heads_per_model"]))
    ax.text(x + 0.30, l2, f"{hp} heads per model  ·  {d['mb_min']:.1f}–{d['mb_max']:.1f} MB as JSON",
            fontsize=8.8, color=T["dim"], va="center", zorder=6)
    ax.text(x + 0.30, l3, f"shipped for {d['n_models']} Qwen3 models", fontsize=8.8,
            color=T["dim"], va="center", zorder=6)

    # ---- 3. serve ----
    x = xs[2]
    header(ax, x + 0.28, top, 3, "Serve at L2", "the forward stops at the fixed block")
    n, b = d["n_blocks"], d["block"]
    bx, by, bw, bh = x + 0.30, top - 1.30, 2.88, 0.30
    seg = bw / n
    for i in range(n):
        rbox(ax, bx + i * seg + 0.012, by, seg - 0.024, bh, fc=T["accent"] if i < b else T["rail"],
             r=0.03, z=5)
    ax.plot([bx + b * seg, bx + b * seg], [by - 0.12, by + bh + 0.12], color=T["ink"], lw=1.4, zorder=6)
    ax.text(bx + b * seg, by + bh + 0.26, f"stop at block {b} of {n}", fontsize=8.8, color=T["ink"],
            ha="center", va="center", fontweight="bold", zorder=6)
    ax.text(bx, by - 0.28, "block 0", fontsize=7.6, color=T["faint"], va="center", zorder=6)
    ax.text(bx + bw, by - 0.28, f"{n}", fontsize=7.6, color=T["faint"], va="center", ha="right", zorder=6)
    ax.text(x + 0.30, l1, f"{d['depth'] * 100:.0f}% of the depth, one prompt per state",
            fontsize=9.6, color=T["ink"], fontweight="bold", va="center", zorder=6)
    ax.text(x + 0.30, l2, f"accuracy {d['acc']:.3f}  ·  ECE {d['ece']:.3f}", fontsize=8.8,
            color=T["dim"], va="center", zorder=6)
    if d["cost"]:
        ax.text(x + 0.30, l3, f"{d['cost']:.2f}× the cost of a plain forward", fontsize=8.8,
                color=T["dim"], va="center", zorder=6)

    # ---- 4. adapt ----
    x = xs[3]
    header(ax, x + 0.28, top, 4, "Adapt", "when the question arrives changed")
    right = x + cw - 0.30
    rows = [
        ("reworded", f"{d['n_unlabelled']} unlabelled requests",
         "recentre μ σ", T["good_soft"], T["good"], f"{rng(d['reworded'])} → {rng(d['recentred'])}"),
        ("options reordered", "nothing to fit",
         "remap", T["accent_soft"], T["accent"], "same head"),
        ("new option set", "labels again",
         "refit", T["warm_soft"], T["warm"], f"{rng(d['refit'])}"),
    ]
    for k, (when, note, what, fc, tc, num) in enumerate(rows):
        yy = top - 0.86 - k * 0.58
        ax.text(x + 0.30, yy, when, fontsize=9.4, color=T["ink"], fontweight="bold",
                va="center", zorder=6)
        ax.text(x + 0.30, yy - 0.27, note, fontsize=8.0, color=T["dim"], va="center", zorder=6)
        chip(ax, right, yy - 0.14, what, fc=fc, tc=tc, h=0.28, fs=8.2, align="right")
        ax.text(right, yy - 0.27, num, fontsize=8.6, color=tc, fontweight="bold",
                va="center", ha="right", zorder=6)
    ax.text(x + 0.30, l3, f"the same head on its own wording: {d['as_is']:.2f}", fontsize=8.2,
            color=T["faint"], va="center", zorder=6)

    # return paths under the cards
    bus(ax, xs[3] + cw * 0.62, xs[2] + cw * 0.50, y0 - 0.02, 0.62, color=T["good"], lw=1.7,
        label="serve again, no new labels")
    bus(ax, xs[3] + cw * 0.38, xs[0] + cw * 0.50, y0 - 0.02, 0.36, color=T["faint"], lw=1.2,
        label="a new option set goes back to labels")

    ax.text(W - 0.5, 0.13, f"{MODEL} · every number is read from committed bench JSON · "
            "regenerate with python scripts/make_head_loop_figure.py",
            fontsize=7.6, color=T["faint"], ha="right", va="center", zorder=6)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=T["bg_top"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="assets/head_loop.png")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    d = load(root)
    for k in ("as_is", "reworded", "recentred", "refit", "block", "n_blocks", "depth", "acc", "ece",
              "cost", "fit_min", "fit_max", "n_calib", "heads_per_model", "mb_min", "mb_max"):
        print(f"  {k:16} {d[k]}")
    out = Path(args.out)
    render(d, out if out.is_absolute() else root / out)


if __name__ == "__main__":
    main()
