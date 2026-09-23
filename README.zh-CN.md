<div align="center">

<img src="assets/banner.png" width="100%" alt="AnyJev —— 把任意 LLM 变成 Jev 风格的决策模型。类型化的决策、真实的概率、不需要训练。选项顺序翻转率 0.230 降到 0.073，校准误差 0.240 降到 0.095，5% 风险下可自动决策比例 7.7% 升到 52.0%。">

[![PyPI](https://img.shields.io/pypi/v/anyjev?color=3b82f6)](https://pypi.org/project/anyjev/)
[![Python](https://img.shields.io/pypi/pyversions/anyjev)](https://pypi.org/project/anyjev/)
[![CI](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml/badge.svg)](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[English](README.md) · **简体中文** · [档位约定](docs/levels.md) · [完整结果](docs/results_bench.md) · [路线图](ROADMAP.md)

</div>

<p align="center">
  <b>Jiamu Zhang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Tianze Yang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Yucheng Shi</b><sup>2</sup> &nbsp;&nbsp;&nbsp; <b>Liang Wu</b><sup>1</sup>
</p>
<p align="center">
  <sub><sup>1</sup>&nbsp;Nokia, Sunnyvale, CA &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>&nbsp;Tencent Hunyuan</sub>
</p>

---

![把选项顺序倒过来：直接读 logits 会以 1.00 的置信度翻转答案，AnyJev L0 两种顺序给出同一个答案](assets/flip.gif)

<div align="center">
<sub>Qwen3-8B，一条真实的 BANKING77 样本，真实输出。</sub><br>
<sub><b>左：</b>直接读 next-token logits —— 把选项倒过来，答案就翻了，置信度还是 1.00。</sub><br>
<sub><b>右：</b>AnyJev L0，零标签 —— 两种顺序同一个答案。</sub>
</div>

> [!TIP]
> **未完待续。** L0 和 L1 已经可用。下一步是 **L2**：每个问题一个闭式 head，一个 prompt 作答，配上路由、`level="auto"` 和部署指南。已完成和接下来要做的见[路线图](#路线图已完成与未完待续)。

---

## 为什么不直接读 logits

给 AnyJev 一个 **state** 和一组**类型化的问题**，每个问题返回一个决策和一个概率，直接从模型的 next-token 分布上读出来 —— 不生成 token、不用解析、不用微调，用的就是你已经在跑的那个模型。

这些你自己用 `max_tokens=1` 加 logprobs 也能做。问题在于你拿到的是什么：一个**换个选项顺序就会变的排序**，和一个**没法拿来设阈值的置信度**。这两件事都是**读法**的性质，不是模型知识的性质，而且一条标签都不用就能修。

<div align="center">

| | 直接读 logits | **AnyJev L0** | **AnyJev L1** |
|:--|:--:|:--:|:--:|
| 需要标签 | 无 | **无** | 100–500 条 |
| 选项倒序后答案改变的比例 | 0.230 | **0.073** | 0.077 |
| 准确率 | 0.747 | **0.803** | 0.807 |
| 校准误差（ECE） | 0.240 | 0.184 | **0.095** |
| **错误率 ≤5% 时可自动决策的比例** | **7.7%** | **46.3%** | **52.0%** |

<sub>Qwen3-8B，BANKING77 20 分类，300 条测试样本。含全部消融行的完整表：<a href="docs/results_bench.md">docs/results_bench.md</a></sub>

</div>

最后一行才是重点。准确率只动了 6 个点，但可以安全自动化的流量从 **7.7% 涨到 52.0%**，在这个任务上相差 6.8 倍（n=300 的点估计，区间很宽，见"局限"）。直接读 logits 时那个 "0.9" 不足以支撑你去行动，于是所有请求都得转人工；一旦概率真的表示它字面的意思，你才能设阈值。

> [!NOTE]
> 与 TypeSafe AI 及 Jev 无关，未获其认可，也不派生自它们。本文档中的每一项对比都是我们实测的、可从 `bench/` 复现，明确标注为"由原作者发布"的行除外。

## 快速开始

```bash
pip install "anyjev[hf]"        # 库 + transformers 后端
pip install anyjev              # 只装库（仅依赖 numpy），后端自备
```

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

art = d.calibrate(safe, calib_states, calib_labels)   # 100-500 条样本 -> L1 artifact
r = d.decide(state, [safe], level="L1")
```

Benchmark 不在 wheel 里 —— 它需要数据集、结果目录和其他项目的代码，所以要从仓库里跑：

```bash
git clone https://github.com/nokia-applied-research/AnyJev && cd AnyJev
pip install -e ".[hf,bench,dev]"
```

## 工作原理

三个类型化原语：`choice`（最多 26 个选项）、`noul`（Yes/No，给出真实的 `p_true`）、`score`（2–10 个有序 bin，给出期望值）。后端只做一件事 —— 返回 next-token 的 log 概率 —— 所以新增一个后端就是一个文件；transformers 和 vLLM 已经可用。

后端之上就是让数字变得可用的两项修正。**循环移位边际化**把 K 个选项的列表转 K 次，让每个选项在每个位置各坐一遍，在 log 空间合并。**先验校正**在没有标签的情况下估出模型的标签先验再除掉：一个问"这封邮件是垃圾邮件吗"的 `noul` 读出 P(Yes) = 0.62，把正文换成 `N/A` 后读出 P(Yes) = 0.70 —— 不管内容是什么它都偏向 Yes —— 除以这个先验后得到 0.41，判断翻转。

完整约定见 [docs/levels.md](docs/levels.md)。

| 档位 | 需要什么 | 做什么 | **不**做什么 |
|---|---|---|---|
| `raw` | 无 | 在标签 token 上做受限 softmax（各个克隆项目的做法） | 任何关于偏差或校准的事 |
| `L0` | 无 | 消除位置偏差和标签先验偏差 | 让模型自身的不确定性变得校准 |
| `L1` | 每个问题 100 到 500 条标签 | 在 L0 之上做温度缩放 | 在校准集之外的分布偏移下仍然可靠 |

每个 `Decision` 都带着自己的 `level`，下游代码可以拒绝使用不够格的概率。

**开销。** L0 用算力换稳定性：K 个选项的 `choice` 需要 K 次 prefill（`noul` 2 次，`score` 1 次），它们共享 state 前缀、都能 batch 掉，而且全程不生成任何 token。在一张 H100 上，transformers 路径在 20 个排列时约为**每个决策 0.25 秒（batch 32）**。`max_permutations` 可以给 K 封顶。

## 路线图：已完成与未完待续

打勾的是现在 `main` 里已有的；没打勾的是接下来要做的。带目标日期的计划见 [ROADMAP.md](ROADMAP.md)，实际落地的内容见 [CHANGELOG.md](CHANGELOG.md)。

**已完成**
- [x] 类型化问题：`choice`、`noul`、`score`，一次 prefill 读出，不生成任何 token。
- [x] **L0**，零标签：循环移位边际化 + 无标签先验校正（默认 batch prior，强度 0.75）。
- [x] **L1**：温度缩放，先验冻结进 artifact；`export_artifacts` / `load_artifacts` 每个模型一个 JSON。
- [x] 档位强制：`decide(..., require="L1")` 在达不到档位时直接报错，而不是拿更弱的概率去做决定。
- [x] 后端：transformers 和 vLLM；共享前缀打分、可选的自适应移位、延迟测量。
- [x] Benchmark：三个任务及全部消融、typed-decisions 上对比 Laya、NanoJev 的迷宫 harness、自带 oracle 的 2048 和扫雷。
- [x] 离线拟合闭式 head：`anyjev.heads.fit_head`（预览，结果见下文「同一个 hidden state 上的闭式 head（预览）」）。

**未完待续**
- [ ] **`Decider` 里的 L2**：每个问题加载一个 head，一个 prompt 作答，在 head 所在的第 b* 层提前退出（见下方「方法」图）。
- [ ] **路由**：布局、措辞、选项顺序变化时仍能路由到存好的 head；每个问题 n ≥ 30 后启用运行中的 μ/σ。
- [ ] **`level="auto"`**，以及把 head artifact 放进 `export_artifacts` / `load_artifacts`，和温度放在一起。
- [ ] **部署指南**：从闭环收集标签、何时重拟合、换基座模型时重解 head（见下方「部署」图）。
- [ ] 超过 26 个选项的 span 读法，以及 L1 之上的 conformal 弃答。
- [ ] 表格里加入 Llama 和 Gemma。
- [ ] Jev 兼容的 HTTP 服务端；SGLang、llama.cpp、MLX、Ollama 后端（**欢迎贡献**）。
- [ ] 通过视觉语言后端支持多模态 state。

*未完待续……*

### 方法：一次决策（未完待续）

有存好的 head 就一次前向作答；没有的话，和现在一样回落到 L1 或 L0。绿色框已经在 `main` 里，虚线框未完待续。

```mermaid
flowchart TD
    A["state + 问题"] --> R{"路由：这个问题<br/>有存好的 head 吗？"}
    R -- "布局完全一致" --> H1["head，用它自己的 μ/σ"]
    R -- "选项相同，<br/>措辞不同" --> H2["head + 这种措辞的<br/>请求的运行中 μ/σ"]
    R -- "选项集合相同，<br/>顺序不同" --> H3["head + 运行中 μ/σ，<br/>概率按选项文本重新对应"]
    H2 --> U["更新这个问题的 (sum, sumsq, n)；<br/>n ≥ 30 后启用"]
    H3 --> U
    H1 --> F["一个 prompt，前向到第 b* 层<br/>p = softmax(((h − μ) / σ · W + b) / T)"]
    U --> F
    F --> D2["Decision，档位 L2<br/>诊断：blocks_executed、routed_from、<br/>reordered、adapted、adapt_n"]
    R -- "没有" --> T{"有温度<br/>artifact 吗？"}
    T -- "有" --> L1["L1：K 个移位 prompt，完整前向，<br/>先验校正，温度"]
    T -- "没有" --> L0["L0：K 个移位 prompt，完整前向，<br/>先验校正"]
    L1 --> D1["Decision，档位 L1"]
    L0 --> D0["Decision，档位 L0"]

    classDef shipped fill:#dcfce7,stroke:#0f9d76,color:#0f172a
    classDef preview fill:#fef3c7,stroke:#d97706,color:#0f172a
    classDef planned fill:#f8fafc,stroke:#94a3b8,stroke-dasharray:5 4,color:#475569
    class A,T,L1,L0,D1,D0 shipped
    class R,H1,H2,H3,U,F,D2 planned
```

> [!NOTE]
> **未完待续。** 虚线部分正在开发。上线之前，每个决策都走右边的分支：有温度 artifact 的用 L1，没有的用 L0。

### 部署：生命周期（未完待续）

第 0 天零标签用 L0 上线，让业务闭环自己产生标签，几秒钟拟合 head；只有换基座模型时才需要重解。绿色已在 `main` 里，琥珀色是预览，虚线未完待续。

```mermaid
flowchart LR
    S0["第 0 天：定义问题，<br/>用 level auto 上线；<br/>全部以 L0 作答"] --> C["从闭环收集标签：<br/>人工审核、业务结果，<br/>或被替换掉的那个 LLM；<br/>每个问题 20–300 条"]
    C --> FH["每个问题 fit_head，几秒钟；<br/>export_artifacts → JSON"]
    FH --> SV["上线：有 head 路由到的用 L2，<br/>其余用 L0"]
    SV --> W{"变了什么？"}
    W -- "措辞或顺序" --> SV
    W -- "新的选项集合" --> C
    W -- "state 分布变化，<br/>抽检准确率下降" --> C
    W -- "新的基座模型" --> RS["用存下的带标签 state<br/>重解每一个 head"]
    RS --> SV

    classDef shipped fill:#dcfce7,stroke:#0f9d76,color:#0f172a
    classDef preview fill:#fef3c7,stroke:#d97706,color:#0f172a
    classDef planned fill:#f8fafc,stroke:#94a3b8,stroke-dasharray:5 4,color:#475569
    class C shipped
    class S0,FH preview
    class SV,W,RS planned
```

> [!NOTE]
> **未完待续。** 现在已经可以用 L0 上线、收集标签、拟合 L1 温度或离线 head。把这些 head 以 L2 上线、换基座模型时重解 head，是接下来的工作。

---

## Benchmark

```bash
python -m bench.run --model Qwen/Qwen3-8B --tasks newsgroups,injection,banking20 --n 300 --calib 200
```

结果以 Markdown 和 JSON 落在 `bench/results/<日期>/`，带硬件和库版本。下面每一张表和图都由这些 JSON 重新生成；**没有任何数字是手打的。**

![四个面板，覆盖三个开源模型和三个任务：选项顺序翻转率、期望校准误差、准确率、5% 风险下的覆盖率，对比直接读 logits 与 AnyJev L0 / L1](assets/results.png)

### 把选项顺序倒过来，五分之一的答案会变

`flip` 是选项列表倒序（`choice`）或 Yes/No 措辞顺序互换（`noul`）时答案发生变化的样本比例。三个开源模型、三个任务、每个任务 300 条测试样本。

| model | task | K | raw flip | L0 flip | raw acc | L0 acc | raw ECE | L1 ECE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | banking20 | 20 | 0.230 | **0.073** | 0.747 | **0.803** | 0.240 | **0.095** |
| Qwen3-8B | newsgroups | 20 | 0.233 | **0.177** | 0.637 | **0.660** | 0.334 | **0.138** |
| Qwen3-8B | injection | 2 | 0.060 | **0.000** | 0.693 | **0.700** | 0.288 | **0.161** |
| Qwen2.5-7B-Instruct | banking20 | 20 | 0.197 | **0.080** | 0.723 | **0.757** | 0.237 | **0.070** |
| Qwen2.5-7B-Instruct | newsgroups | 20 | 0.237 | **0.127** | 0.663 | **0.710** | 0.273 | **0.096** |
| Qwen2.5-7B-Instruct | injection | 2 | 0.070 | **0.000** | 0.737 | **0.790** | 0.188 | **0.049** |
| Qwen3-30B-A3B-Instruct-2507 | banking20 | 20 | 0.143 | **0.097** | 0.730 | **0.770** | 0.249 | **0.086** |
| Qwen3-30B-A3B-Instruct-2507 | newsgroups | 20 | 0.140 | **0.093** | 0.737 | **0.740** | 0.242 | **0.096** |
| Qwen3-30B-A3B-Instruct-2507 | injection | 2 | 0.103 | **0.000** | 0.730 | **0.757** | 0.248 | **0.090** |

全部消融行（只做排列、单独使用每种先验、Brier、5% 风险下的覆盖率）：[docs/results_bench.md](docs/results_bench.md)。一张 H100，bf16，transformers 4.55.4。

### 在 Laya 自己的 benchmark 上，零样本

| system | acc | soft_acc | ece | brier_mean | score_mae |
|---|---|---|---|---|---|
| laya-multilingual (zero-shot), measured here | 0.340 | 0.325 | 0.287 | 0.269 | 0.688 |
| laya (zero-shot), measured here | 0.359 | 0.331 | 0.177 | 0.227 | 0.694 |
| Qwen2.5-7B-Instruct + raw logits (clone baseline) | 0.620 | 0.514 | 0.287 | 0.209 | 0.437 |
| Qwen3-8B + raw logits (clone baseline) | 0.626 | 0.520 | 0.328 | 0.210 | 0.621 |
| Qwen2.5-7B-Instruct + AnyJev L0, zero-shot | 0.628 | 0.512 | 0.234 | 0.188 | 0.439 |
| Qwen2.5-7B-Instruct + AnyJev L1, temperature from 200 train cases | 0.628 | 0.461 | 0.038 | 0.148 | 0.425 |
| Qwen3-8B + AnyJev L0, zero-shot | 0.647 | 0.530 | 0.290 | 0.198 | 0.591 |
| Qwen3-8B + AnyJev L1, temperature from 200 train cases | 0.648 | 0.468 | 0.055 | 0.140 | 0.444 |
| Qwen3-32B + raw logits (clone baseline) | 0.684 | 0.556 | 0.206 | 0.144 | 0.488 |
| Qwen3-32B + AnyJev L1, temperature from 200 train cases | 0.699 | 0.508 | 0.036 | 0.119 | 0.416 |
| Qwen3-32B + AnyJev L0, zero-shot | 0.700 | 0.555 | 0.149 | 0.129 | 0.449 |
| Jev 1.13.0 (published by TypeSafe / Laya; not rerun) | 0.727 | 0.580 | 0.144 | 0.148 | 0.391 |
| laya-typed-decisions (fine-tuned on this set's train split), measured here | 0.768 | 0.471 | 0.215 | 0.118 | 0.243 |

除 Jev 外的每一行都是我们在同样的 2,000 个 decision 上实测的；微调后的 Laya checkpoint 复现了它公布的 0.766。**这张表要从两个角度读。** 看 argmax 准确率，微调后的 Laya 赢，零训练的 32B 开源模型比 Jev 低 2.8 个点。看概率质量，也就是 System One 模型存在的意义：微调后的 Laya 的 ECE（0.215）是 AnyJev L1（0.036）的**六倍**，但单看 Brier 它仍略微领先，0.118 对 0.119。温度缩放用 soft accuracy 换校准，所以 7B 和 8B 的 L1 行在这一项上掉到 0.45 左右。Laya 的零样本 checkpoint，也就是你在它没训练过的问题上会用到的那个，只有 0.34 到 0.36，随机基线是 0.32。

按 workflow、按题型拆分的完整表：[docs/results_typed.md](docs/results_typed.md)。

### 放进 NanoJev 的迷宫 harness

我们复刻了 NanoJev 的 "Untuned Qwen3-0.6B" A/B 读法，在它冻结的 scaled_maze 流水线里只替换回答布尔问题的引擎。**两件事同时成立。** 未训练读法的结果高度依赖读法：同一个 Qwen3-0.6B，在 A/B 读法下是 13/15 个迷宫、20,500 步，在 AnyJev 的 raw Yes/No 读法下是 15/15、5,825 步。同时，没有任何一种读法，包括 Qwen3-8B，在"向北走一步是否畅通"上比一直回答多数类更准（边判断准确率 0.40 到 0.55，多数类基线 0.54 到 0.61）。这不是 NanoJev held-out gameplay 表里那个 2/10 成绩所对应的评测 —— 那个数字来自另一个我们没有跑过的 274-case 套件。我们报告它，因为这是 NanoJev 发起的对比；我们不拿它做标题。

完整表和协议：[docs/results_maze.md](docs/results_maze.md)。

### 同一个 hidden state 上的闭式 head（预览）

raw 读法本身就是一个线性 head：标签 token 在 `lm_head` 里的那几行，作用在最后一个位置的 hidden state 上。`anyjev.heads` 用一小批带标签的样本，为一个问题闭式地拟合另一个矩阵（收缩 LDA、ridge、reduced-rank regression，或类均值之差），层数、正则和温度都只在这批样本上用交叉验证选。没有梯度、不改权重，在你本来就要付的那一次 prefill 之后，CPU 上几秒钟。

| typed-decisions，Qwen3-8B，每题 200 个标签，20 题 × 100 条测试 | acc | ECE | Brier |
|---|---|---|---|
| raw logits | 0.626 | 0.330 | 0.688 |
| AnyJev L0（排列） | 0.635 | 0.320 | 0.669 |
| AnyJev L1（温度） | 0.626 | 0.174 | 0.482 |
| **闭式 head，每题按交叉验证选** | **0.771** | **0.120** | **0.339** |
| laya-typed-decisions，用全部 300 条训练样本微调 | 0.768 | 0.215 | — |

分类型：`choice` 0.60 → 0.75，`noul` 0.71 → 0.85，`score` 0.59 → 0.73；20 题里 19 题变好。在 BANKING77（K = 20，200 个标签）上 ridge head 到 0.843，L0 是 0.800，raw 是 0.747，ECE 0.046。两个 caveat，都是测出来的。在单一选项顺序上拟合的 head 不具备顺序不变性（选项倒序后 0.95 的答案会变；改用循环移位平均的特征后降到 0.11–0.19，代价是 K 次 prefill）。标签必须来自任务本身：用模型自己的答案拟合 head 没有收益，用它 thinking 模式的答案拟合反而掉准确率。目前只有一个模型、一个 seed。`python -m bench.heads_study`、`python -m bench.heads_table`；JSON 在 `bench/results_heads/`。这里的 Brier 是多分类求和，不是上表的按选项平均。

### 两个自带 oracle 的游戏

`python -m demo.games.twenty48` 和 `python -m demo.games.minesweeper` 让 raw、L0、L1 在同样的种子上各玩一局，每一步决策都对着 oracle 打分（深度 2 的 expectimax；精确的地雷后验）。Qwen3-8B 玩 2048（五局，给模型看每个合法走法之后的棋盘）：得分 raw 1,744 → L0 1,982，flip 0.20 → 0.12，ECE 0.46 → 0.33 → L1 0.08；随机 855，oracle 10,154。扫雷上 8B 和 32B 的任何读法都打不过随机（清掉 0.71 的棋盘，随机 0.68）：模型读不懂数字约束，L1 只是让它的 P(safe) 变得诚实（ECE 0.41 → 0.08）。两者都能在一个合成的有偏模型上不用 GPU 跑（`--backend fake`）；见 [demo/games/README.md](demo/games/README.md)。

---

## 可复现性

一次独立复现重跑了每一个已发布的数字（3 个模型 × 3 个任务 × 2 种先验、迷宫的所有行、Laya 的所有行）：在记录的设置下所有零标签数字逐位一致，并发现了一个设计瑕疵：L1 artifact 用的是一个持续累积的先验，因此取决于 decider 之前打过分的样本。已修：artifact 现在冻结拟合时用的先验，每个结果 JSON 都记录 batch size 和 dtype（bf16 的 logits 会随 batch 形状变动最多 0.01）。上面每张表都在修复后的代码下由提交的 JSON 重新生成，`bash scripts/regen_docs.sh`。

---

## 局限

这些我们宁可你在这里看到，而不是在生产环境里撞上。

- **L0 不是每个任务都稳赢。** 在 Qwen3-8B 的 prompt-injection 切分上，L0 的 5% 风险覆盖率（0.160）反而*低于* raw（0.297）。content-free 先验的方差最大：在某个 `noul` 任务上 +8 到 +12 个点，在有序 `score` 上 −3，在另一个模型的 `noul` 上 −9。请在你自己的任务上测 —— bench 的所有消融行都来自同一批前向，不额外花钱。
- **batch 先验需要"一批"数据。** 它要攒够 `min_prior_n`（默认 8）条同一问题的样本才启用，并且假设这批数据的标签边缘分布不极端。在真实多数类超过约 65% 的问题上它会损失准确率（默认强度 0.75 时 −0.02，全强度时 −0.04，基于 164 个 (模型, 问题) 点，见 [docs/when_l0_helps.md](docs/when_l0_helps.md)），而且没有任何无标签规则能把这种情况和有偏的模型区分开。
- **校准救不了答不出来的模型。** 在迷宫 harness 里，没有任何读法能在边判断上打过多数类基线。AnyJev 让不确定性变得*可读*，而不是变小。
- **5% 风险下的覆盖率是高方差的点估计**：n = 300 时翻转一条样本就能让它动 0.013，而且它"最深可接受前缀"的定义平均比"首次越界"的定义高 0.026。把那个 7 倍当方向看，别当常数。
- **当前字母读法最多 26 个选项**，span 读法会解除这个上限。
- **L1 扛不住分布偏移**，而且它只重塑置信度、不改变排序。
- **目前表里只有一个模型家族。** 上面全部是 Qwen，Llama 和 Gemma 行在路线图上。

## 状态

**v0.0.2。** 库、两个后端、三个 benchmark 都是真实可运行、实测过的。持续开发中 —— 带日期的计划在 [ROADMAP.md](ROADMAP.md)。**接下来：** 在 `Decider` 里上线带路由和 `level="auto"` 的 L2 head（见上文「路线图」）、闭式 head 扩到更多模型和标签数量、超过 26 个选项的 span 读法、conformal 弃答、Llama 和 Gemma 行。**再之后：** Jev 兼容的 HTTP 服务端、更多后端、多模态 state。

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
