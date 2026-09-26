<div align="center">

<img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/banner.png" width="100%" alt="AnyJev —— 把任意 LLM 变成 Jev 风格的决策模型。类型化的决策、真实的概率、不需要微调。零标签下选项顺序翻转率 0.230 降到 0.073；100–500 条标签下校准误差 0.240 降到 0.095、5% 风险下可自动决策比例 7.7% 升到 52.0%。">

[![PyPI](https://img.shields.io/pypi/v/anyjev?color=3b82f6)](https://pypi.org/project/anyjev/)
[![Python](https://img.shields.io/pypi/pyversions/anyjev)](https://pypi.org/project/anyjev/)
[![CI](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml/badge.svg)](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[English](https://github.com/nokia-applied-research/AnyJev/blob/main/README.md) · **简体中文** · [⚡ 跑起来](#-跑起来) · [📊 结果](#-有标签之后l2) · [🧭 路线图](#-路线图) · [📖 档位约定](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md)

</div>

<p align="center">
  <b>Jiamu Zhang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Tianze Yang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Yucheng Shi</b><sup>2</sup> &nbsp;&nbsp;&nbsp; <b>Liang Wu</b><sup>1</sup>
</p>
<p align="center">
  <sub><sup>1</sup>&nbsp;Nokia, Sunnyvale, CA &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>&nbsp;Tencent Hunyuan</sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/flip.gif" width="100%" alt="把选项顺序倒过来：直接读 logits 会翻转答案，AnyJev L0 两种顺序给出同一个答案">
  <br>
  <sub>Qwen3-8B，一条真实的 BANKING77 样本。图中每个数字都是模型的真实输出。</sub>
</p>

> [!TIP]
> **🆕 vLLM 现在能服务所有档位，包括 L2。** 一个 embed 服务的 pooler 会把闭式 head 要读的
> 那个隐状态原样交出来，所以一套决策端点就是「一个 pooling 服务 + 几 KB 的 head」。
> `python -m anyjev.pipeline <model>` 一条命令完成转换、起服务和测量。[从这里开始 ↓](#-跑起来)


## ⚡ 跑起来

三条命令，把 Hub 上的一个模型变成一个带校准概率的决策端点。

```bash
pip install "anyjev[hf]"

# 1. 只保留决策用得上的那些层，通常三分之二左右
python -m anyjev.truncate Qwen/Qwen2.5-7B-Instruct 18 ./qwen-b18

# 2. 起服务。L2 读的是隐状态，所以让 pooler 原样把它交出来
vllm serve ./qwen-b18 --task embed \
  --override-pooler-config '{"pooling_type":"LAST","normalize":false,"softmax":false}'
```

```python
from anyjev import Decider, Question
from anyjev.backends.vllm import VLLMBackend

d = Decider(VLLMBackend("http://localhost:8000", "./qwen-b18"), level="L2")
route = Question.choice("这条工单该由哪个团队处理？",
                        ["账单", "技术", "销售", "其他"], name="route")

d.fit_head(route, states, labels, layers=[-1])   # 100–300 条标签，一次闭式求解
d.decide(ticket, [route])["route"].distribution  # {"账单": 0.81, "技术": 0.07, ...}
```

**一套 L2 部署就是一个 pooling 服务加几 KB 的 head。**不需要 logits、不需要解析文本、不需要打补丁的引擎，也不生成任何 token。raw / L0 / L1 用同样的方式跑在 `--task generate` 的服务上。一个在 `transformers` 上拟合、由 vLLM 服务的 head，和在任意一侧自拟合自服务给出的答案一致——BANKING77-20 上 **99.0% 答案相同，平均 |Δp| 0.0011**。

**别信我们机器上的数字，在你自己的机器上量：**

```bash
python -m anyjev.pipeline Qwen/Qwen2.5-7B-Instruct --labels-from banking20
```

它会截断模型、起服务、用你的标签拟合 head、在留出状态上测准确率 / ECE / 每次决策的毫秒数，然后关掉服务，再用全深度跑一遍作对照。计时取 `--repeats` 次的中位数，并把这几次的离散度打在旁边——因为在一台有其他负载的机器上，**单次计时能把同一个配置报成比基线又快又慢**。

> **砍深度通常是净赚，不是取舍。** Qwen2.5-7B 从 28 层砍到 18 层，准确率**略升**、校准变好，而且更快：对一个线性 head 来说，中间层是比最后一层更好的特征空间——最后那几层忙的是把答案变成 token。`--quantization fp8` 可以用，但不推荐：它买到单问题延迟，代价是准确率。

## ✨ 它是什么

给任意开源 LLM 一个**类型化的问题**——选择、是非、打分——拿回一个**带可设阈值概率的决策**，从它 next-token 分布的一次 prefill 里读出来。不生成、不解析。直接读 logits 的话，把选项顺序换一下答案就变了，置信度也不能信；AnyJev 用零标签修好第一个，用几百条标签修好第二个。

<div align="center">

| | ⚪&nbsp;直接读&nbsp;logits | 🔵&nbsp;**L0**<br><sub>零标签</sub> | 🟢&nbsp;**L1**<br><sub>+ 温度</sub> |
|:--|:--:|:--:|:--:|
| 需要的标签 | 无 | **无** | 100–500 |
| 选项顺序倒过来后答案翻转 | 0.230 | **0.073** | 0.077 |
| 准确率 | 0.747 | **0.803** | 0.807 |
| 校准误差（ECE） | 0.240 | 0.184 | **0.095** |
| **≤5% 错误下可自动决策** | **7.7%** | **46.3%** | **52.0%** |

<sub>Qwen3-8B，BANKING77 20 分类，300 条测试 · <a href="docs/results_bench.md">全部消融</a></sub>

</div>

最后一行才是重点：准确率只动了 6 个点，但**能安全自动化的流量从 7.7% 变成 52.0%**。直接读 logits 时，一个"0.9"不足以据此行动，于是所有东西都得转人工；一旦概率名副其实，你就能设阈值了。

## 📊 有标签之后：L2

每个问题一个闭式 head，100–300 条标签几秒钟解出来——没有梯度，模型权重一个字节都不动——然后用一条在中途停下的 prompt 作答。

<div align="center">

| 模型 | L0，零标签 | **L2** | 层 | 相对一次完整前向的开销 |
|:--|:--:|:--:|:--:|:--:|
| Qwen3-1.7B | 0.494 | **0.730** | 18 / 28 | 0.70× |
| Qwen3-4B | 0.564 | **0.786** | 24 / 36 | 0.69× |
| Qwen3-8B | 0.647 | **0.771** | 24 / 36 | 0.68× |
| Qwen3-30B-A3B | 0.630 | **0.799** | 40 / 48 | — |
| Qwen3-32B | 0.700 | **0.798** | 52 / 64 | 0.84× |

<sub>LocalLLaMA/typed-decisions，20 个问题 × 300 条标签，2,000 条留出决策。L2 的合并 ECE 为
0.03–0.05。同一数据集上 Jev 为 0.727、微调后的 Laya 为 0.768，均为其作者公布的数字 ·
<a href="docs/results_exit.md">逐格数据</a></sub>

</div>

一个 1.7B 用 64% 的深度就达到了 Jev 公布的数字；一个 4B 追平了微调过的 421M Laya。**100 条标签**就能把 8B 的 head 推到 0.740。五个 Qwen3 模型的 head 随包发布在 `anyjev-heads/`，每个约 100 KB。

**head 会自己维护自己。**解出来之后只有它的特征均值和方差在动，而且是从**无标签**的流量里重估的，所以问法换了、选项顺序换了它都能自己跟上——换一种问法会让 Qwen3-8B 的 head 从 0.77 掉到 0.65–0.70，而 **30 条无标签请求**就能把它拉回 0.74–0.75，对照全部重新标注重拟合的 0.77。只有遇到新问题才需要新标签。[路由是怎么走的 →](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/method_v3.md)

## 🧠 四个档位

<p align="center">
  <img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/how_it_works.png" width="100%" alt="一个决策是怎么读出来的：问一个类型化的问题，在选项的每一种循环移位上各读一次，除掉一个不用标签就能估出的标签先验，返回一个带档位的决策">
</p>

| 档位 | 需要 | 做了什么 | **不**做什么 |
|---|---|---|---|
| `raw` | 无 | 在标签 token 上做受限 softmax | 任何关于偏差和校准的事 |
| `L0` | 无 | 在 K 种循环移位上平掉位置偏差，并除掉标签先验 | 让不确定性变得可信 |
| `L1` | 每题 100–500 条标签 | 在 L0 之上做温度缩放 | 改变排序 |
| **`L2`** | **每题 100–300 条标签** | **在中途某一层的隐状态上解一个闭式 head，每个状态一条 prompt** | **迁移到别的问题或别的模型** |

每个 `Decision` 都带着自己的 `level`，`require="L1"` 能让下游代码拒绝在更弱的档位上行动。对 K 个选项的 choice，L0 要付 K 次 prefill；**L2 比一次普通前向还便宜**。

`d.observe(q, state, label)` 会在标签到达时收集它们，攒到 30 条自己解出 head，之后在 60、120 条时重解——所以第 0 天什么都没有时跑 L0，等循环喂够了 L2 自己就来了。[完整约定 →](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md) · [方法 →](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/method_v3.md)

**手边没有 GPU？** `python -m demo.jev_mode --backend fake` 在一个合成模型上一秒内跑完整套流程。

<p align="center"><sub>
<a href="docs/jev_mode.md">Jev 模式</a> ·
<a href="demo/games/README.md">2048 与扫雷</a> ·
<a href="docs/results_maze.md">NanoJev 迷宫</a> ·
<a href="docs/when_l0_helps.md">L0 什么时候有用</a> ·
<a href="docs/results_small_models.md">小模型</a> ·
<a href="docs/research_log.md">研究日志，含负面结果</a>
</sub></p>

<sub>每个数字都由仓库内的 JSON 重新生成（`bash scripts/regen_docs.sh`）；从干净 checkout 再跑一遍，
所有零标签数字逐位复现。与 TypeSafe AI 和 Jev 无从属关系；由其作者公布的行没有在这里重跑。</sub>

## 🧭 路线图

- [x] `choice`、`noul`、`score`，一次 prefill；零标签的 L0；L1 artifact
- [x] **L2**：每题一个闭式 head，路由、无标签自适应、`observe`
- [x] 五个 Qwen3 模型的随包 head；打包好的 demo
- [ ] 🚧 **速度优化**（进行中）：让每个决策更快
- [x] **在推理引擎上跑 L2**：vLLM，经由 embed 服务的 pooler 或一个截断后的 checkpoint（`anyjev.pipeline`、`anyjev.truncate`）；SGLang 尚未支持
- [ ] **真实 agent 循环里的评测**：同样的决策放进 agent 里，和它要替换掉的那个 LLM 对比
- [ ] head 上 Hugging Face Hub、一个可交互的 Space、一份技术报告
- [ ] 更多模型（Llama、Gemma、Mistral、DeepSeek）、超过 26 个选项的 span 读法、conformal 弃答

带日期的计划和"help wanted"清单：[ROADMAP.md](https://github.com/nokia-applied-research/AnyJev/blob/main/ROADMAP.md)。

## 🔍 局限

- **在 typed-decisions 上，"准确率"衡量的是与一个教师 LLM 的一致性。** gold 是同一个模型三次采样的均值；该教师的一次新采样与它只有 0.735 的一致率。
- **L2 是按问题、按模型的。** 在别的问题上拟合的 head 对新问题没有帮助，而且目前只发布了 Qwen3 的 head。它需要 hidden state——transformers 和 vLLM 的 embed 服务都能提供，其他引擎还不行。
- **校准救不了答不出来的模型。** 在迷宫边和扫雷上，没有任何读法能赢过平凡基线。
- **L0 并非处处白赚。** 当某一个标签占绝对多数时，batch prior 会损失准确率（见 [L0 什么时候有用](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/when_l0_helps.md)）。

<sub>此外：字母读法最多 26 个选项（span 读法在路线图上，代码里还没有）；5% 风险下的覆盖率在 n=300 时方差很大；头部表格都是 Qwen 模型；这里每条决策都是孤立评测的，不是在 agent 循环里。</sub>

## 🤝 参与和引用

后端和 bench provider 都是一个文件一个，其中几个标了 **help wanted**（[ROADMAP.md](https://github.com/nokia-applied-research/AnyJev/blob/main/ROADMAP.md)、[CONTRIBUTING.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CONTRIBUTING.md)）。变更记录：[CHANGELOG.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CHANGELOG.md)。致谢：[CREDITS.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CREDITS.md)。

```bibtex
@software{anyjev2026,
  title  = {AnyJev: Turn any LLM into a Jev-style decision model},
  author = {Zhang, Jiamu and Yang, Tianze and Shi, Yucheng and Wu, Liang},
  year   = {2026},
  url    = {https://github.com/nokia-applied-research/AnyJev}
}
```

Apache-2.0，见 [LICENSE](https://github.com/nokia-applied-research/AnyJev/blob/main/LICENSE)。数据集各自保留其许可证，见 [THIRD_PARTY.md](https://github.com/nokia-applied-research/AnyJev/blob/main/THIRD_PARTY.md)。
