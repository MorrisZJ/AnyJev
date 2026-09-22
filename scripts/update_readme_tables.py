"""Replace the generated tables inside README.md and README.zh-CN.md with fresh output.

    python scripts/update_readme_tables.py bench/results_v01 bench/results_typed_v01 bench/results_typed

Three blocks are replaced in both READMEs, each recognised by its header row: the headline bench
table (`| model | task | K | raw flip |`), the typed-decisions table (`| system | acc |`) and the
small-models table (`| model | label mass |`). Prose around them is untouched.
"""
import re
import subprocess
import sys

BENCH, TYPED, TYPED_LAYA = (sys.argv[1:] + ["bench/results_v01", "bench/results_typed_v01", "bench/results_typed"])[:3]


def latest(d):
    import glob
    return sorted(glob.glob(d + "/*/"))[-1]


def gen(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return "\n".join(line for line in out.splitlines() if line.startswith("|"))


import glob
import os
import shutil
import tempfile

headline = gen([sys.executable, "-m", "bench.readme_table", BENCH, "--models",
                "Qwen3-8B,Qwen2.5-7B-Instruct,Qwen3-30B-A3B-Instruct-2507"])
tmp = tempfile.mkdtemp()
for f in glob.glob(latest(TYPED) + "*.json") + glob.glob(latest(TYPED_LAYA) + "laya__*.json"):
    shutil.copy(f, tmp)
typed = gen([sys.executable, "-m", "bench.typed_table", tmp])
small = gen([sys.executable, "-m", "bench.models_table", BENCH, TYPED])
small = "\n".join(l for l in small.splitlines() if not re.search(r"\| (Qwen3-30B|Qwen3-32B)", l))


def replace_block(text, header_prefix, new_block):
    lines = text.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith(header_prefix)), None)
    if i is None:          # this README does not carry the table; leave it alone
        print(f"  (no block starting with {header_prefix!r}; skipped)")
        return text
    j = i
    while j < len(lines) and lines[j].startswith("|"):
        j += 1
    return "\n".join(lines[:i] + new_block.split("\n") + lines[j:])


# the hero mini-table and the sentence under it: Qwen3-8B on banking20, raw / L0 / L1
import json as _json
hero_json = [p for p in glob.glob(latest(BENCH) + "*.json") if p.endswith("Qwen__Qwen3-8B.json")][0]
hero = next(t for t in _json.load(open(hero_json))["tasks"] if t["task"] == "banking20")["levels"]
H = {lvl: hero[lvl] for lvl in ("raw", "L0", "L1")}
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
ratio = H["L1"]["cov@5%"] / H["raw"]["cov@5%"] if H["raw"]["cov@5%"] else float("inf")
acc_pts = round(100 * (H["L0"]["acc"] - H["raw"]["acc"]))
hero_rows = {
    "en": [f"| Answer flips when options are reversed | {H['raw']['flip']:.3f} | **{H['L0']['flip']:.3f}** | {H['L1']['flip']:.3f} |",
           f"| Accuracy | {H['raw']['acc']:.3f} | **{H['L0']['acc']:.3f}** | {H['L1']['acc']:.3f} |",
           f"| Calibration error (ECE) | {H['raw']['ece']:.3f} | {H['L0']['ece']:.3f} | **{H['L1']['ece']:.3f}** |",
           f"| **Auto-decidable at ≤5% error** | **{pct(H['raw']['cov@5%'])}** | **{pct(H['L0']['cov@5%'])}** | **{pct(H['L1']['cov@5%'])}** |"],
    "zh": [f"| 选项倒序后答案改变的比例 | {H['raw']['flip']:.3f} | **{H['L0']['flip']:.3f}** | {H['L1']['flip']:.3f} |",
           f"| 准确率 | {H['raw']['acc']:.3f} | **{H['L0']['acc']:.3f}** | {H['L1']['acc']:.3f} |",
           f"| 校准误差（ECE） | {H['raw']['ece']:.3f} | {H['L0']['ece']:.3f} | **{H['L1']['ece']:.3f}** |",
           f"| **错误率 ≤5% 时可自动决策的比例** | **{pct(H['raw']['cov@5%'])}** | **{pct(H['L0']['cov@5%'])}** | **{pct(H['L1']['cov@5%'])}** |"],
}
hero_sentence = {
    "en": (f"The last row is the point. Accuracy moves by {acc_pts} points, but the share of traffic you can safely automate "
           f"goes from **{pct(H['raw']['cov@5%'])} to {pct(H['L1']['cov@5%'])}**, a {ratio:.1f}× difference on this task "
           f"(a point estimate at n=300; the interval is wide, see Limitations). With raw logits a \"0.9\" is not trustworthy "
           f"enough to act on, so everything goes to a human. Once the probability means what it says, you can set a threshold."),
    "zh": (f"最后一行才是重点。准确率只动了 {acc_pts} 个点，但可以安全自动化的流量从 **{pct(H['raw']['cov@5%'])} 涨到 "
           f"{pct(H['L1']['cov@5%'])}**，在这个任务上相差 {ratio:.1f} 倍（n=300 的点估计，区间很宽，见\"局限\"）。直接读 logits 时那个 "
           f"\"0.9\" 不足以支撑你去行动，于是所有请求都得转人工；一旦概率真的表示它字面的意思，你才能设阈值。"),
}
alt = {
    "en": (f"Order-flip rate {H['raw']['flip']:.3f} to {H['L0']['flip']:.3f}, calibration error {H['raw']['ece']:.3f} to {H['L1']['ece']:.3f}, "
           f"auto-decidable at 5% risk {pct(H['raw']['cov@5%'])} to {pct(H['L1']['cov@5%'])}."),
    "zh": (f"选项顺序翻转率 {H['raw']['flip']:.3f} 降到 {H['L0']['flip']:.3f}，校准误差 {H['raw']['ece']:.3f} 降到 {H['L1']['ece']:.3f}，"
           f"5% 风险下可自动决策比例 {pct(H['raw']['cov@5%'])} 升到 {pct(H['L1']['cov@5%'])}。"),
}


def replace_hero(text, lang):
    lines = text.split("\n")
    # rows: the four metric rows follow the "Labels required" row
    i = next(k for k, l in enumerate(lines)
             if l.startswith("| Labels required") or l.startswith("| 需要标签"))
    lines[i + 1:i + 5] = hero_rows[lang]
    # the sentence: starts with "That last row" / "最后一行"
    j = next(k for k, l in enumerate(lines) if l.startswith(("That last row is the point", "The last row is the point",
                                                              "最后一行才是重点")))
    lines[j] = hero_sentence[lang]
    # banner alt text: replace the numeric tail after the tagline
    for k, l in enumerate(lines):
        if 'src="assets/banner.png"' in l:
            key = "Order-flip rate" if lang == "en" else "选项顺序翻转率"
            a = l.find(key)
            b = l.find('">', a)
            if a > 0 and b > a:
                lines[k] = l[:a] + alt[lang] + l[b:]
    return "\n".join(lines)


for path, lang in (("README.md", "en"), ("README.zh-CN.md", "zh")):
    s = open(path).read()
    s = replace_block(s, "| model | task | K | raw flip |", headline)
    s = replace_block(s, "| system | acc |", typed)
    s = replace_block(s, "| model | label mass |", small)
    s = replace_hero(s, lang)
    open(path, "w").write(s)
    print("updated", path, "| hero:", {k: round(v["cov@5%"], 3) for k, v in H.items()}, "ratio", round(ratio, 1))
