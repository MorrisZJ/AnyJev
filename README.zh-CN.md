<div align="center">

<img src="assets/banner.png" width="100%" alt="AnyJev —— 把任意 LLM 变成 Jev 风格的决策模型。类型化的决策、真实的概率、不需要训练。选项顺序翻转率 0.227 降到 0.077，校准误差 0.235 降到 0.100，5% 风险下可自动决策比例 7.7% 升到 54.3%。">

<br>

[![PyPI](https://img.shields.io/pypi/v/anyjev?color=3b82f6)](https://pypi.org/project/anyjev/)
[![Python](https://img.shields.io/pypi/pyversions/anyjev)](https://pypi.org/project/anyjev/)
[![CI](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml/badge.svg)](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[English](README.md) · **简体中文** · [档位约定](docs/levels.md) · [完整结果](docs/results_bench.md) · [路线图](ROADMAP.md)

</div>

---

<div align="center">

### 作者

**Jiamu Zhang**<sup>1</sup> · **Tianze Yang**<sup>1</sup> · **Yucheng Shi**<sup>2</sup> · **Liang Wu**<sup>1</sup>

<sup>1</sup> Nokia, Sunnyvale, CA &nbsp;&nbsp;·&nbsp;&nbsp; <sup>2</sup> Tencent Hunyuan

</div>

---

![把选项顺序倒过来：直接读 logits 会以 1.00 的置信度翻转答案，AnyJev L0 两种顺序给出同一个答案](assets/flip.gif)

<div align="center">
<sub>Qwen3-8B，一条真实的 BANKING77 样本，真实输出。</sub><br>
<sub><b>左：</b>直接读 next-token logits —— 把选项倒过来，答案就翻了，置信度还是 1.00。</sub><br>
<sub><b>右：</b>AnyJev L0，零标签 —— 两种顺序同一个答案。</sub><br>
<sub>用 <code>scripts/find_flip_example.py</code> 和 <code>scripts/make_flip_gif.py</code> 重新生成。</sub>
</div>

---

## 这是什么

给它一个 **state** 和一组**类型化的问题**，每个问题返回一个决策和一个概率，直接从模型的 next-token 分布上读出来 —— **不生成 token，不用解析，不用微调**。用的就是你已经加载在显存里的那个模型。

```python
r = decider.decide(state, [route, is_destructive, is_done])

r["route"].distribution     # {"billing": 0.81, "technical": 0.07, ...}
r["is_destructive"].p_true  # 0.12
r["is_done"].value          # 0.35
r.level                     # "L0"
```

### 为什么不直接读 logits

因为 `max_tokens=1` 加 logprobs 给你的，是一个**换个选项顺序就会变的排序**，和一个**没法拿来设阈值的置信度**。这两件事都不需要训练就能修，AnyJev 默认修掉。

<div align="center">

| | 直接读 logits | **AnyJev L0** | **AnyJev L1** |
|:--|:--:|:--:|:--:|
| 需要标签 | 无 | **无** | 100–500 条 |
| 选项倒序后答案改变的比例 | 0.227 | **0.077** | 0.077 |
| 准确率 | 0.750 | **0.807** | 0.807 |
| 校准误差（ECE） | 0.235 | 0.180 | **0.100** |
| **错误率 ≤5% 时可自动决策的比例** | **7.7%** | **47.7%** | **54.3%** |

<sub>Qwen3-8B，BANKING77 20 分类，300 条测试样本。含全部消融行的完整表：<a href="docs/results_bench.md">docs/results_bench.md</a></sub>

</div>

最后一行才是重点。准确率只动了 6 个点，但**可以安全自动化的流量从 7.7% 涨到 54.3%**，相差 7 倍。直接读 logits 时那个 "0.9" 不足以支撑你去行动，于是所有请求都得转人工；一旦概率真的表示它字面的意思，你才能设阈值。

---

## 亮点

- **默认开启免训练去偏（L0）。** 循环移位边际化消除选项顺序偏差；无标签的先验估计消除模型的标签偏好。零标签，零微调。
- **有标签时做事后校准（L1）。** 按（模型，问题）拟合温度，存成一个很小的 JSON artifact。载入在别的模型上拟合的 artifact 会直接报错。
- **每个结果都标明自己是哪一档。** `decision.level` 是 `raw`、`L0` 或 `L1`，下游代码可以拒绝使用不够格的概率。
- **三个类型化原语。** `choice`（最多 26 个选项）、`noul`（Yes/No，给出真实的 `p_true`）、`score`（2–10 个有序 bin，给出期望值）。
- **只有一个方法的后端接口。** 后端只负责返回 next-token 的 log 概率，别的什么都不做。transformers 和 vLLM 已经可用，新增一个后端就是一个文件。
- **一个真正报告校准的 benchmark**，不只是准确率：Brier、ECE、选项顺序翻转率、5% 风险下的覆盖率 —— 一条命令出表，全部由提交的 JSON 重新生成。

> [!NOTE]
> 与 TypeSafe AI 及 Jev 无关，未获其认可，也不派生自它们。本文档中的每一项对比都是我们实测的、可从 `bench/` 复现，明确标注为"由原作者发布"的行除外。

---

## 安装

```bash
pip install "anyjev[hf]"        # 库 + transformers 后端
pip install anyjev              # 只装库（仅依赖 numpy），后端自备
```

Benchmark 不在 wheel 里 —— 它需要数据集、结果目录和其他项目的代码，所以要从仓库里跑：

```bash
git clone https://github.com/nokia-applied-research/AnyJev && cd AnyJev
pip install -e ".[hf,bench,dev]"
```

## 十行代码

```python
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

d = Decider(HFBackend("Qwen/Qwen3-8B"))

route = Question.choice("这个请求应该交给哪个处理器？",
                        ["billing", "technical", "sales", "other"], name="route")
safe  = Question.noul("这个工具调用是否具有破坏性或不可逆？", name="safe")
done  = Question.score("任务完成度在 0 到 1 之间是多少？", bins=5, name="done")

state = {"conversation": [...], "proposed_tool_call": {...}}
r = d.decide(state, [route, safe, done])

r["route"].argmax          # "billing"
r["route"].distribution    # {"billing": 0.81, "technical": 0.07, ...}
r["safe"].p_true           # 0.12
r["done"].value            # 0.35
r.level                    # "L0"（已去偏，未校准）
```

有标签时：

```python
art = d.calibrate(safe, calib_states, calib_labels)   # 约 100 到 500 条样本 -> L1 artifact
r = d.decide(state, [safe], level="L1")
```

---

## 为什么 L0 不是可选项

一个 `noul` 问题："这封邮件是垃圾邮件吗？"直接读 logits，**P(Yes) = 0.62**。把邮件正文换成 `N/A`、其余不变，模型给出 **P(Yes) = 0.70** —— 不管内容是什么它都偏向 Yes。除以这个先验再归一化，答案变成 **P(Yes) = 0.41**，判断翻转。

位置偏差对 `choice` 问题做的是同一件事：把选项顺序换一下，答案就变。这两种偏差都是**读法**的性质，不是模型知识的性质，而且一条标签都不用就能修。

## 三个档位

完整约定见 [docs/levels.md](docs/levels.md)。

| 档位 | 需要什么 | 做什么 | **不**做什么 |
|---|---|---|---|
| `raw` | 无 | 在标签 token 上做受限 softmax（各个克隆项目的做法） | 任何关于偏差或校准的事 |
| `L0` | 无 | 消除位置偏差和标签先验偏差 | 让模型自身的不确定性变得校准 |
| `L1` | 每个问题 100 到 500 条标签 | 在 L0 之上做温度缩放 | 在校准集之外的分布偏移下仍然可靠 |

## 开销

L0 是用算力换稳定性：K 个选项的 `choice` 需要 **K 次 prefill**（`noul` 2 次，`score` 1 次），它们共享 state 前缀，而且都能 batch 掉。全程不生成任何 token，所以循环里没有自回归解码。`max_permutations` 可以给 K 封顶。

在一张 H100 上，transformers 路径在 20 个排列时约为**每个决策 0.25 秒（batch 32）**。vLLM 路径更快，目前还没有公布的数字 —— 延迟列在[路线图](ROADMAP.md)上。

---

## Benchmark

```bash
python -m bench.run --model Qwen/Qwen3-8B --tasks newsgroups,injection,banking20 --n 300 --calib 200
```

结果以 Markdown 和 JSON 落在 `bench/results/<日期>/`，带硬件和库版本。`python -m bench.table bench/results/<日期>` 重新生成下面的表；**没有任何数字是手打的。**

![四个面板，覆盖三个开源模型和三个任务：选项顺序翻转率、期望校准误差、准确率、5% 风险下的覆盖率，对比直接读 logits 与 AnyJev L0 / L1](assets/results.png)

<div align="center">
<sub>用 <code>python scripts/make_results_figure.py</code> 重新生成 —— 它读的是和 <code>bench.table</code> 同一批已提交的 JSON。</sub>
</div>

### 把选项顺序倒过来，五分之一的答案会变

三个开源模型、三个任务、每个任务 300 条测试样本。`flip` 是选项列表倒序（`choice`）或措辞在 "Yes or No" 与 "No or Yes" 之间互换（`noul`）时答案发生变化的样本比例。`raw` 是所有读 logits 的克隆项目的做法。`L0` 是 AnyJev 的默认值，零标签。`L1` 在此基础上用 200 条标签拟合一个温度。

| model | task | K | raw flip | L0 flip | raw acc | L0 acc | raw ECE | L1 ECE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | banking20 | 20 | 0.227 | **0.077** | 0.750 | **0.807** | 0.235 | **0.100** |
| Qwen3-8B | newsgroups | 20 | 0.237 | **0.173** | 0.640 | **0.660** | 0.331 | **0.157** |
| Qwen3-8B | injection | 2 | 0.060 | **0.000** | 0.693 | **0.710** | 0.287 | **0.162** |
| Qwen2.5-7B-Instruct | banking20 | 20 | 0.197 | **0.067** | 0.723 | **0.767** | 0.236 | **0.072** |
| Qwen2.5-7B-Instruct | newsgroups | 20 | 0.233 | **0.123** | 0.660 | **0.707** | 0.276 | **0.082** |
| Qwen2.5-7B-Instruct | injection | 2 | 0.053 | **0.000** | 0.737 | **0.813** | 0.189 | **0.037** |
| Qwen3-30B-A3B-Instruct-2507 | banking20 | 20 | 0.143 | **0.097** | 0.733 | **0.767** | 0.246 | **0.079** |
| Qwen3-30B-A3B-Instruct-2507 | newsgroups | 20 | 0.133 | **0.100** | 0.730 | **0.740** | 0.249 | **0.086** |
| Qwen3-30B-A3B-Instruct-2507 | injection | 2 | 0.093 | **0.000** | 0.723 | **0.767** | 0.253 | **0.080** |

含全部消融行（只做排列、单独使用每种先验、Brier、5% 风险下的覆盖率）的完整表：[docs/results_bench.md](docs/results_bench.md)。一张 H100，bf16，transformers 4.55.4；用 `python -m bench.table bench/results_batchprior_v0/2026-09-20` 重新生成。

### 在 Laya 自己的 benchmark 上，零样本

| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| laya-multilingual (zero-shot), measured here | 0.340 | 0.325 | 0.287 | 0.269 | 0.688 |
| laya (zero-shot), measured here | 0.359 | 0.331 | 0.177 | 0.227 | 0.694 |
| Qwen2.5-7B-Instruct + raw logits (clone baseline) | 0.621 | 0.514 | 0.287 | 0.209 | 0.437 |
| Qwen3-8B + raw logits (clone baseline) | 0.626 | 0.520 | 0.328 | 0.210 | 0.621 |
| Qwen2.5-7B-Instruct + AnyJev L0, zero-shot | 0.628 | 0.506 | 0.200 | 0.176 | 0.451 |
| Qwen2.5-7B-Instruct + AnyJev L1, temperature from 200 train cases | 0.632 | 0.452 | 0.047 | 0.149 | 0.443 |
| Qwen3-8B + AnyJev L0, zero-shot | 0.640 | 0.523 | 0.273 | 0.196 | 0.617 |
| Qwen3-8B + AnyJev L1, temperature from 200 train cases | 0.646 | 0.457 | 0.056 | 0.143 | 0.474 |
| Qwen3-32B + raw logits (clone baseline) | 0.684 | 0.556 | 0.206 | 0.144 | 0.488 |
| Qwen3-32B + AnyJev L0, zero-shot | 0.700 | 0.548 | 0.133 | 0.128 | 0.456 |
| Qwen3-32B + AnyJev L1, temperature from 200 train cases | 0.701 | 0.502 | **0.034** | 0.120 | 0.412 |
| Jev 1.13.0 (published by TypeSafe / Laya; not rerun) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |
| laya-typed-decisions (fine-tuned on this set's train split), measured here | **0.768** | 0.471 | 0.215 | **0.118** | **0.243** |

Laya 的头条数字是这个数据集上的 0.766，高于 Jev 的 0.727。除 Jev 外的每一行都是我们在同样的 2,000 个 decision 上实测的；Laya 的微调 checkpoint 复现了它公布的数字。**这张表要从两个角度读。** 看 argmax 准确率，在 train 切分上微调过的 Laya 赢，零训练的 32B 开源模型比 Jev 低 2.7 个点。看概率质量，也就是 System One 模型存在的意义，情况就不一样了：微调后的 Laya 的 ECE（0.215）是 AnyJev L1（0.034）的**六倍**，它的 soft accuracy（0.471）低于表中每一个零样本的 Qwen 行。两个 AnyJev L1 行是例外 —— 温度缩放用 soft accuracy 换校准，7B 和 8B 的 soft accuracy 降到 0.452 和 0.457，同时 Brier 改善。单看 Brier，微调后的 Laya 仍然略微领先，0.118 对 AnyJev L1 最好的 0.120。Laya 的零样本 checkpoint，也就是你在它没训练过的问题上会用到的那个，只有 0.34 到 0.36，随机基线是 0.32。

含按 workflow、按题型拆分的完整表：[docs/results_typed.md](docs/results_typed.md)；用 `python -m bench.run_typed --model <model>` 和 `python -m bench.providers.laya` 重新生成。

### 放进 NanoJev 的迷宫 harness

| engine | goal test | goal ood | attempts | collisions | edge acc | majority | edge Brier | edge questions |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B + AnyJev L0 (batch prior) | 10/11 | 3/4 | 15616 | 6236 | 0.490 | 0.618 | 0.271 | 23156 |
| Qwen3-0.6B + AnyJev L0 (content_free prior) | 10/11 | 4/4 | 16278 | 6563 | 0.490 | 0.614 | 0.274 | 23284 |
| Qwen3-0.6B + AnyJev L0 (none prior) | 10/11 | 3/4 | 21851 | 9136 | 0.403 | 0.597 | 0.460 | 28040 |
| Qwen3-0.6B + AnyJev raw | 11/11 | 4/4 | 5825 | 2616 | 0.537 | 0.539 | 0.363 | 10944 |
| Qwen3-8B + AnyJev L0 (batch prior) | 11/11 | 3/4 | 17841 | 7171 | 0.555 | 0.600 | 0.344 | 25124 |
| Qwen3-0.6B native A/B readout (NanoJev's 'Untuned Qwen' protocol) | 10/11 | 3/4 | 20555 | 8496 | 0.419 | 0.607 | 0.305 | 27660 |

NanoJev 的 README 把它训练的 0.6B 模型和 "Untuned Qwen3-0.6B" 对比，后者对迷宫每个格子的四个布尔问题读 A/B 两个字母的 logits。我们复刻了这个读法，把它放进 NanoJev 的 scaled_maze 流水线（[docs/SCALED_GAMES.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/SCALED_GAMES.md)）里跑，用它的 15 个测试和分布外 episode、尺寸 8 到 50、它自己冻结的探索代码和它 pin 的 Qwen3-0.6B revision，只换回答布尔问题的引擎。这**不是** NanoJev held-out gameplay 表里那个 2/10 迷宫成绩所对应的评测 —— 那个数字来自另一个 274-case 套件，我们没有跑过，所以上面的行既不复述也不反驳它。NanoJev 没有在 scaled 套件上发布过 untuned Qwen 基线，所以最后一行是我们对其协议的复刻，不是他们的数字。

**两件事同时成立。** 未训练读法的结果高度依赖读法：同一个 Qwen3-0.6B，在 A/B 读法下是 13/15 个迷宫、20,555 步，在 AnyJev 的 raw Yes/No 读法下是 15/15、5,825 步。同时，没有任何一种 LLM 读法，包括 Qwen3-8B，在"向北走一步是否畅通"这个问题上比一直回答多数类更准（边判断准确率 0.40 到 0.56，多数类基线 0.54 到 0.62）。其中 raw 那行最接近，0.537 对 0.539，是打平而不是取胜。迷宫上的差异来自各种读法的平均概率与控制器 `p >= 0.5` 探测规则的相互作用，而不是读懂了地图。我们报告它，因为这是 NanoJev 发起的对比；我们不拿它做标题。完整表：[docs/results_maze.md](docs/results_maze.md)。

---

## 局限

这些我们宁可你在这里看到，而不是在生产环境里撞上。

- **L0 不是每个任务都稳赢。** 在 Qwen3-8B 的 prompt-injection 切分上，L0 的 5% 风险覆盖率（0.160）反而*低于* raw（0.297）。content-free 先验的方差最大：在某个 `noul` 任务上 +8 到 +12 个点，在有序 `score` 上 −3，在另一个模型的 `noul` 上 −9。请在你自己的任务上测 —— bench 的所有消融行都来自同一批前向，不额外花钱。
- **batch 先验需要"一批"数据。** 它要攒够 `min_prior_n`（默认 8）条同一问题的样本才启用，并且假设这批数据的标签边缘分布不极端。边缘分布偏斜时它会过度校正 —— 一个收缩系数是路线图上的开放研究项。
- **校准救不了答不出来的模型。** 在迷宫 harness 里，没有任何读法能在边判断上打过多数类基线。AnyJev 让不确定性变得*可读*，而不是变小。
- **当前字母读法最多 26 个选项。** span 读法会解除这个上限，是路线图上的下一项。
- **L1 扛不住分布偏移**，而且它只重塑置信度、不改变排序。
- **目前表里只有一个模型家族。** 上面全部是 Qwen，Llama 和 Gemma 行在路线图上。

## 状态

**v0.0.2。** 库、两个后端、三个 benchmark 都是真实可运行、实测过的；每一张表都由提交的 JSON 重新生成。持续开发中 —— 带日期的计划在 [ROADMAP.md](ROADMAP.md)。

**接下来：** 超过 26 个选项的 span 读法、conformal 弃答、延迟列、在线 demo、Llama 和 Gemma 行。**再之后：** Jev 兼容的 HTTP 服务端、更多后端、多模态 state。

后端和 benchmark provider 都是一个文件一个，其中几项标了 **help wanted** —— 见 [CONTRIBUTING.md](CONTRIBUTING.md)。已完成的改动：[CHANGELOG.md](CHANGELOG.md)。我们站在谁的肩膀上：[CREDITS.md](CREDITS.md)。

## 引用

```bibtex
@software{anyjev2026,
  title  = {AnyJev: Turn any LLM into a Jev-style decision model},
  author = {Zhang, Jiamu and Yang, Tianze and Shi, Yucheng and Wu, Liang},
  year   = {2026},
  url    = {https://github.com/nokia-applied-research/AnyJev}
}
```

## 许可证

Apache-2.0 —— 见 [LICENSE](LICENSE)。数据集保留各自的许可证，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
