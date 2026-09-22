"""Render assets/maze.gif from a bench.run_maze result (real runs, every step replayed).

Three mazes side by side on one step clock: raw logits, AnyJev L0, AnyJev L1.
Walls the explorer bumped into turn red and stay red. The compass under each
maze is the model's P(open) for the four directions at its latest cell: arm
length is the probability, green if it lands on the right side of 0.5.

    python scripts/make_maze_gif.py bench/results_games/<date>/maze.Qwen__Qwen3-8B.json --seed 0
"""
import argparse
import json
import math

from PIL import Image, ImageDraw, ImageFont

INK, DIM, FAINT = "#0f172a", "#64748b", "#94a3b8"
ACCENT, GOOD, BAD = "#2563eb", "#0f9d76", "#e5484d"
WALL, FLOOR, SEEN, CARD, LINE = "#d5ddea", "#ffffff", "#c7d8f6", "#ffffff", "#e2e8f0"
DELTA = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
PANELS = [("raw", "raw logits", "one prompt, restricted softmax", FAINT),
          ("L0", "AnyJev L0", "zero labels", ACCENT),
          ("L1", "AnyJev L1", "temperature from 200 labels", GOOD)]
W, H, TILE, TRAIL = 1200, 780, 6, 90


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}{'-Bold' if bold else ''}.ttf", size)


def blend(a, b, t):
    a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


class Replay:
    """One run, advanced step by step, drawing onto a persistent maze image."""

    def __init__(self, ep, run):
        self.run, self.walls = run, ep["walls"]
        self.size = len(self.walls)
        self.goal, self.path = tuple(ep["goal"]), [tuple(ep["start"])]
        self.img = Image.new("RGB", (self.size * TILE, self.size * TILE), FLOOR)
        self.draw = ImageDraw.Draw(self.img)
        for r, row in enumerate(self.walls):
            for c, ch in enumerate(row):
                if ch == "#":
                    self._tile((r, c), WALL)
        self._tile(self.path[0], SEEN)
        self.t, self.hits, self.hit_count, self.pred_i = 0, {}, 0, -1
        self.correct = self.asked = 0

    def _tile(self, cell, color):
        r, c = cell
        self.draw.rectangle([c * TILE, r * TILE, c * TILE + TILE - 1, r * TILE + TILE - 1], fill=color)

    def advance(self, t):
        preds, moves = self.run["predictions"], self.run["moves"]
        while self.pred_i + 1 < len(preds) and preds[self.pred_i + 1]["step"] <= min(t, len(moves)):
            self.pred_i += 1
            p = preds[self.pred_i]
            self.asked += 4
            self.correct += sum((p["p_open"][d] >= 0.5) == p["truth"][d] for d in DELTA)
        while self.t < min(t, len(moves)):
            m = moves[self.t]
            self.t += 1
            if m["hit"]:
                r, c = m["pos"]
                wall = (r + DELTA[m["dir"]][0], c + DELTA[m["dir"]][1])
                self.hits[wall] = self.hits.get(wall, 0) + 1
                self.hit_count += 1
                self._tile(wall, blend(WALL, BAD, min(1.0, 0.55 + 0.15 * self.hits[wall])))
            else:
                self.path.append(tuple(m["pos"]))
                self._tile(self.path[-1], SEEN)

    @property
    def finished(self):
        return self.t >= len(self.run["moves"])

    def frame(self):
        img = self.img.copy()
        d = ImageDraw.Draw(img)
        gr, gc = self.goal
        d.rectangle([gc * TILE - 2, gr * TILE - 2, gc * TILE + TILE + 1, gr * TILE + TILE + 1], fill=GOOD)
        tail = self.path[-TRAIL:]
        for i, (r, c) in enumerate(tail):
            col = blend("#c7d8f6", ACCENT, (i + 1) / len(tail))
            d.rectangle([c * TILE, r * TILE, c * TILE + TILE - 1, r * TILE + TILE - 1], fill=col)
        r, c = self.path[-1]
        cx, cy = c * TILE + TILE / 2, r * TILE + TILE / 2
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=ACCENT, width=2)
        d.ellipse([cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5], fill=INK)
        return img


def compass(d, cx, cy, pred, arm=34):
    d.ellipse([cx - arm - 6, cy - arm - 6, cx + arm + 6, cy + arm + 6], outline=LINE, width=1)
    for name, (dr, dc) in DELTA.items():
        ex, ey = cx + dc * (arm + 16), cy + dr * (arm + 16)
        d.text((ex, ey), name[0].upper(), font=font(11, bold=True), fill=FAINT, anchor="mm")
        if pred is None:
            continue
        p, truth = pred["p_open"][name], pred["truth"][name]
        col = GOOD if (p >= 0.5) == truth else BAD
        x1, y1 = cx + dc * arm * p, cy + dr * arm * p
        d.line([cx, cy, cx + dc * arm, cy + dr * arm], fill="#eef2f7", width=7)
        d.line([cx, cy, x1, y1], fill=col, width=7)
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=INK)


def render(data, ep, t, reps, total):
    img = Image.new("RGB", (W, H), "#f5f7fc")
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([0, y, W, y], fill=blend("#ffffff", "#e8eefb", y / H))
    model = data["model"].split("/")[-1]
    d.text((34, 26), "AnyJev", font=font(30, bold=True), fill=INK)
    d.line([34, 66, 88, 66], fill=ACCENT, width=3)
    d.text((170, 28), f"Find the exit on {data['size']}×{data['size']}, one probability at a time",
           font=font(20, bold=True), fill=INK)
    d.text((170, 58), f"{model} answers four Yes/No questions per new cell: is the tile north / east / south / "
           "west of me open?  Same maze, same explorer; only the readout changes.", font=font(12), fill=DIM)
    pw, px0, py = 368, 28, 96
    for i, ((level, title, sub, color), rep) in enumerate(zip(PANELS, reps)):
        x = px0 + i * (pw + 18)
        d.rounded_rectangle([x, py, x + pw, py + 600], radius=14, fill=CARD, outline=LINE)
        d.rounded_rectangle([x + 18, py + 16, x + 22, py + 44], radius=2, fill=color)
        d.text((x + 32, py + 13), title, font=font(18, bold=True), fill=INK)
        d.text((x + 32, py + 37), sub, font=font(11), fill=DIM)
        m = rep.frame()
        mx, my = x + (pw - m.width) // 2, py + 62
        img.paste(m, (mx, my))
        if rep.finished:
            status = rep.run["status"]
            label = f"exit reached · step {rep.run['steps']:,}" if status == "goal" else f"gave up · {status}"
            fw = d.textlength(label, font=font(13, bold=True)) + 28
            bx, by = mx + (m.width - fw) / 2, my + m.height - 44
            d.rounded_rectangle([bx, by, bx + fw, by + 30], radius=15, fill=GOOD if status == "goal" else BAD)
            d.text((bx + fw / 2, by + 15), label, font=font(13, bold=True), fill="white", anchor="mm")
        sy = my + m.height + 22
        d.text((x + 22, sy), "STEPS", font=font(10, bold=True), fill=FAINT)
        d.text((x + 22, sy + 14), f"{rep.t:,}", font=font(26, bold=True), fill=INK)
        d.text((x + 22, sy + 58), "WALL HITS", font=font(10, bold=True), fill=FAINT)
        d.text((x + 22, sy + 72), f"{rep.hit_count:,}", font=font(26, bold=True), fill=BAD)
        acc = rep.correct / rep.asked if rep.asked else float("nan")
        d.text((x + 22, sy + 116), "EDGES READ RIGHT", font=font(10, bold=True), fill=FAINT)
        d.text((x + 22, sy + 130), "–" if math.isnan(acc) else f"{acc:.0%}", font=font(18, bold=True), fill=DIM)
        pred = rep.run["predictions"][rep.pred_i] if rep.pred_i >= 0 else None
        compass(d, x + pw - 90, sy + 82, pred)
        d.text((x + pw - 90, sy + 10), "P(open) here", font=font(10, bold=True), fill=FAINT, anchor="mm")
    by = H - 58
    d.rounded_rectangle([34, by, W - 150, by + 8], radius=4, fill="#e2e8f0")
    d.rounded_rectangle([34, by, 34 + (W - 184) * t / total, by + 8], radius=4, fill=INK)
    d.text((W - 34, by + 4), f"step {t:,}", font=font(14, bold=True, mono=True), fill=INK, anchor="rm")
    d.text((34, by + 22), f"Real runs replayed from bench/results_games · maze seed {ep['seed']} (picked before "
           f"running) · red tile = a wall the explorer walked into · arm green = P(open) on the right side of 0.5",
           font=font(10.5), fill=FAINT)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--out", default="assets/maze.gif")
    args = ap.parse_args()
    data = json.load(open(args.result))
    ep = next(e for e in data["episodes"] if e["seed"] == args.seed)
    reps = [Replay(ep, ep["runs"][level]) for level, *_ in PANELS]
    total = max(len(r.run["moves"]) for r in reps)
    ts = sorted({round(total * (i / (args.frames - 1)) ** 1.35) for i in range(args.frames)})
    frames = []
    for t in ts:
        for rep in reps:
            rep.advance(t)
        frames.append(render(data, ep, t, reps, total).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    durations = [70] * (len(frames) - 1) + [3500]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    frames[-1].convert("RGB").save(args.out.replace(".gif", "_last.png"))
    print("wrote", args.out, len(frames), "frames")


if __name__ == "__main__":
    main()
