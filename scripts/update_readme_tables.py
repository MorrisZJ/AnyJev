"""Replace the generated tables inside README.md and README.zh-CN.md with fresh output.

    python scripts/update_readme_tables.py bench/results_v01 bench/results_typed_v01 bench/results_typed

Three blocks are replaced in both READMEs, each recognised by its header row: the headline bench
table (`| model | task | K | raw flip |`), the typed-decisions table (`| system | acc |`) and the
small-models table (`| model | label mass |`). Prose around them is untouched.
"""
import glob
import json
import re
import shutil
import subprocess
import sys
import tempfile

BENCH, TYPED, TYPED_LAYA = (sys.argv[1:] + ["bench/results_v01", "bench/results_typed_v01", "bench/results_typed"])[:3]


def latest(d):
    return sorted(glob.glob(d + "/*/"))[-1]


def gen(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return "\n".join(line for line in out.splitlines() if line.startswith("|"))


headline = gen([sys.executable, "-m", "bench.readme_table", BENCH, "--models",
                "Qwen3-8B,Qwen2.5-7B-Instruct,Qwen3-30B-A3B-Instruct-2507"])
tmp = tempfile.mkdtemp()
for f in glob.glob(latest(TYPED) + "*.json") + glob.glob(latest(TYPED_LAYA) + "laya__*.json"):
    shutil.copy(f, tmp)
typed = gen([sys.executable, "-m", "bench.typed_table", tmp])
# the README carries a curated subset of docs/results_typed.md: the two Laya zero-shot rows, the
# three models the headline table uses, the Jev row and the fine-tuned Laya row.
README_TYPED = re.compile(r"\|(-|\s(system |Qwen2\.5-7B-Instruct|Qwen3-8B|Qwen3-32B|Jev |laya))")
typed = "\n".join(line for line in typed.splitlines() if README_TYPED.match(line))
small = gen([sys.executable, "-m", "bench.models_table", BENCH, TYPED])
small = "\n".join(line for line in small.splitlines()
                  if not re.search(r"\| (Qwen3-30B|Qwen3-32B)", line))


def replace_block(text, header_prefix, new_block):
    lines = text.split("\n")
    i = next((k for k, line in enumerate(lines) if line.startswith(header_prefix)), None)
    if i is None:          # this README does not carry the table; leave it alone
        print(f"  (no block starting with {header_prefix!r}; skipped)")
        return text
    j = i
    while j < len(lines) and lines[j].startswith("|"):
        j += 1
    return "\n".join(lines[:i] + new_block.split("\n") + lines[j:])


# the hero mini-table and the sentence under it: Qwen3-8B on banking20, raw / L0 / L1
hero_json = [p for p in glob.glob(latest(BENCH) + "*.json") if p.endswith("Qwen__Qwen3-8B.json")][0]
hero = next(t for t in json.load(open(hero_json))["tasks"] if t["task"] == "banking20")["levels"]
H = {lvl: hero[lvl] for lvl in ("raw", "L0", "L1")}
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
ratio = H["L1"]["cov@5%"] / H["raw"]["cov@5%"] if H["raw"]["cov@5%"] else float("inf")
acc_pts = round(100 * (H["L0"]["acc"] - H["raw"]["acc"]))
hero_rows = {
    "en": [f"| Answer flips when options are reversed | {H['raw']['flip']:.3f} | "
           f"**{H['L0']['flip']:.3f}** | {H['L1']['flip']:.3f} |",
           f"| Accuracy | {H['raw']['acc']:.3f} | **{H['L0']['acc']:.3f}** | {H['L1']['acc']:.3f} |",
           f"| Calibration error (ECE) | {H['raw']['ece']:.3f} | {H['L0']['ece']:.3f} | **{H['L1']['ece']:.3f}** |",
           f"| **Auto-decidable at ≤5% error** | **{pct(H['raw']['cov@5%'])}** | "
           f"**{pct(H['L0']['cov@5%'])}** | **{pct(H['L1']['cov@5%'])}** |"],
    "zh": [f"| 选项倒序后答案改变的比例 | {H['raw']['flip']:.3f} | **{H['L0']['flip']:.3f}** | {H['L1']['flip']:.3f} |",
           f"| 准确率 | {H['raw']['acc']:.3f} | **{H['L0']['acc']:.3f}** | {H['L1']['acc']:.3f} |",
           f"| 校准误差（ECE） | {H['raw']['ece']:.3f} | {H['L0']['ece']:.3f} | **{H['L1']['ece']:.3f}** |",
           f"| **错误率 ≤5% 时可自动决策的比例** | **{pct(H['raw']['cov@5%'])}** | "
           f"**{pct(H['L0']['cov@5%'])}** | **{pct(H['L1']['cov@5%'])}** |"],
}



def cov(model_file, task):
    """raw and L1 coverage at 5% risk for one (model, task) cell of the same run."""
    p = [q for q in glob.glob(latest(BENCH) + "*.json") if q.endswith(model_file)][0]
    lv = next(t for t in json.load(open(p))["tasks"] if t["task"] == task)["levels"]
    return lv["raw"]["cov@5%"], lv["L1"]["cov@5%"]


# the spread behind the headline: a clear gain, a no-change cell and a loss, all from the same run
GAIN = cov("Qwen__Qwen3-8B.json", "newsgroups")
FLAT = cov("Qwen__Qwen3-30B-A3B-Instruct-2507.json", "injection")
LOSS = cov("Qwen__Qwen3-30B-A3B-Instruct-2507.json", "newsgroups")
spread = {
    "en": (f"It is not uniform: over the nine (model, task) cells of the same run the raw → L1 comparison "
           f"runs from {GAIN[0]:.3f} → {GAIN[1]:.3f} (Qwen3-8B, newsgroups) through "
           f"{FLAT[0]:.3f} → {FLAT[1]:.3f} (Qwen3-30B-A3B, injection: no change) to "
           f"{LOSS[0]:.3f} → {LOSS[1]:.3f} (Qwen3-30B-A3B, newsgroups: a loss)."),
    "zh": (f"但它并不均匀：同一次运行的九个（模型, 任务）单元里，raw → L1 的同一对比从 "
           f"{GAIN[0]:.3f} → {GAIN[1]:.3f}（Qwen3-8B，newsgroups）、"
           f"{FLAT[0]:.3f} → {FLAT[1]:.3f}（Qwen3-30B-A3B，injection，没有变化）一直到 "
           f"{LOSS[0]:.3f} → {LOSS[1]:.3f}（Qwen3-30B-A3B，newsgroups，反而变差）。"),
}
hero_sentence = {
    "en": (f"The last row is the point. Accuracy moves by {acc_pts} points, but the share of traffic "
           f"you can safely automate goes from **{pct(H['raw']['cov@5%'])} to {pct(H['L1']['cov@5%'])}**, "
           f"a {ratio:.1f}× difference on this task (a point estimate at n=300; the interval is wide, "
           f"see Limitations). With raw logits a \"0.9\" is not trustworthy enough to act on, so everything "
           f"goes to a human. Once the probability means what it says (L1, or L2 below), you can set a "
           f"threshold; at L0 the ranking is stable but the number is still not calibrated. " + spread["en"]),
    "zh": (f"最后一行才是重点。准确率只动了 {acc_pts} 个点，但可以安全自动化的流量从 **{pct(H['raw']['cov@5%'])} 涨到 "
           f"{pct(H['L1']['cov@5%'])}**，在这个任务上相差 {ratio:.1f} 倍（n=300 的点估计，区间很宽，见“局限”）。"
           f"直接读 logits 时那个“0.9”不足以支撑你去行动，于是所有请求都得转人工；一旦概率真的表示它字面的意思"
           f"（L1，或下文的 L2），你才能设阈值；在 L0 上排序是稳的，但那个数字仍然没有校准。" + spread["zh"]),
}
ALT_KEY = {"en": "Zero labels:", "zh": "零标签："}
alt = {
    "en": (f"Zero labels: order-flip rate {H['raw']['flip']:.3f} to {H['L0']['flip']:.3f}. "
           f"100–500 labels: calibration error {H['raw']['ece']:.3f} to {H['L1']['ece']:.3f}, "
           f"auto-decidable at 5% risk {pct(H['raw']['cov@5%'])} to {pct(H['L1']['cov@5%'])}."),
    "zh": (f"零标签：选项顺序翻转率 {H['raw']['flip']:.3f} 降到 {H['L0']['flip']:.3f}。"
           f"100–500 条标签：校准误差 {H['raw']['ece']:.3f} 降到 {H['L1']['ece']:.3f}，"
           f"5% 风险下可自动决策比例 {pct(H['raw']['cov@5%'])} 升到 {pct(H['L1']['cov@5%'])}。"),
}


def replace_hero(text, lang):
    lines = text.split("\n")
    # rows: the four metric rows follow the "Labels required" row
    i = next(k for k, line in enumerate(lines)
             if line.startswith("| Labels required") or line.startswith("| 需要标签"))
    lines[i + 1:i + 5] = hero_rows[lang]
    # the sentence: starts with "That last row" / "最后一行"
    j = next(k for k, line in enumerate(lines)
             if line.startswith(("That last row is the point", "The last row is the point", "最后一行才是重点")))
    lines[j] = hero_sentence[lang]
    # banner alt text: replace the numeric tail after the tagline
    for k, line in enumerate(lines):
        if "<img" in line and "banner.png" in line:
            a = line.find(ALT_KEY[lang])
            b = line.find('">', a)
            if a > 0 and b > a:
                lines[k] = line[:a] + alt[lang] + line[b:]
    return "\n".join(lines)


for path, lang in (("README.md", "en"), ("README.zh-CN.md", "zh")):
    s = open(path).read()
    s = replace_block(s, "| model | task | K | raw flip |", headline)
    s = replace_block(s, "| system | acc |", typed)
    s = replace_block(s, "| model | label mass |", small)
    s = replace_hero(s, lang)
    open(path, "w").write(s)
    print("updated", path, "| hero:", {k: round(v["cov@5%"], 3) for k, v in H.items()}, "ratio", round(ratio, 1))
