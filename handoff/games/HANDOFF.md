# 交接：用 AnyJev 做游戏动图（迷宫 / 贪吃蛇 / ViZDoom）

目标：做一张 README 用的游戏回放动图（参考 NanoJev 的 50×50 Maze、ViZDoom Basic 那种观感），
但**决策全部由 AnyJev 的读出方式产生**，环境和探索逻辑都在本仓库里自己实现，不用 NanoJev 的模型或代码，
视觉风格也要是我们自己的（配色沿用 `scripts/make_banner.py` 的 THEME）。

对比方式已和负责人确认：**同一个模型（Qwen3-8B）、同一个环境、同一个策略，三栏只换读出方式：
raw logits / AnyJev L0 / AnyJev L1**。顺序：迷宫 → 贪吃蛇 → ViZDoom Basic（文本标签 state）。

## 一句话现状

迷宫的全套代码可以用，但**结论不适合做宣传图**：Qwen3-8B 对局部视野的感知不够准，
一个"闭眼什么方向都先试"的基线打败了所有读出方式。负责人已决定**改做贪吃蛇**，贪吃蛇还没开始写。

## 文件

| 路径 | 内容 |
|---|---|
| `bench/games/maze.py` | 带种子的迷宫生成器（递归回溯 + 4% 打通成环）、文本局部视野 `view()`、只凭 P(open) 行动的 `Explorer`（`expected_cost` / `threshold` 两种策略，`wall_cost` 撞墙罚分） |
| `bench/run_maze.py` | 正式 runner：每个 (迷宫, level) 一个新 `Decider`；L0/L1 先在校准迷宫的无标签视野上预热 batch prior，L1 再用标签拟合每个方向的温度；每一步都存进 JSON |
| `tests/test_maze.py` | 8 个测试（生成器、视野、探索器、FakeBackend 上的 play 循环），CI 会跑 |
| `scripts/make_maze_gif.py` | 三栏同步回放渲染器（PIL）：走过的路浅蓝、最近轨迹深蓝、撞过的墙变红、四向 P(open) 罗盘 |
| `handoff/games/maze_probe.py` | 感知探针：400 个开发视野上各方向的准确率 / AUC / 平均 p |
| `handoff/games/maze_dev_probs.py` | 在开发迷宫（seed 20000–20004）上预计算 raw/L0/L1 对**每个格子**的概率 |
| `handoff/games/dev_probs.Qwen3-8B.json` | 上面脚本的输出（Qwen3-8B） |
| `handoff/games/maze_sim.py` + `maze_sim_output.txt` | 用预计算概率离线模拟 4 种策略 × 读出方式 + "闭眼全试"基线，及其结果 |

## 种子约定（防止挑结果）

- 测试迷宫：seed 0–4。**动图固定展示 seed 0**，这是跑之前就定的；全部 5 个 seed 都要进文档。
- 校准（L1 标签 + L0 预热）：seed 10000 起（`CALIB_SEED0`），和测试不重叠。
- 开发（调策略 / 调 state 格式）：seed 20000–20004。**只能在开发种子上做设计选择**，不能看测试结果再调。

## 已经测到的结论（Qwen3-8B，全部是真实运行）

1. **模型有感知信号，但 raw 概率按方向严重偏置。** 400 个开发视野上的 AUC：北 0.90、东 0.99、南 0.90、西 0.78。
   但 raw 的北向平均 P(open) 只有 0.18，真实开放率是 0.51，所以准确率只有 0.62。
   batch prior 的 L0 把准确率提高到 0.67–0.75；content-free prior 在这里很差（0.52–0.61），不要用。
2. **旧设计（threshold 0.5、撞墙不罚分，已作废；其结果 JSON 不在本仓库）在测试 seed 0 上**：raw 3764 步到达出口；L0、L1 都用完 5202 步没到出口，
   尽管它们的边准确率（0.69）和 Brier（0.25 / 0.19）都比 raw（0.64 / 0.30）好。seed 1 上反过来：L0 965 步，raw 5128 步。方差很大。
3. **开发迷宫离线模拟**（`maze_sim_output.txt`，5 个迷宫的平均步数）：

   | 策略 | raw | L0 | L1 | 闭眼全试 |
   |---|---|---|---|---|
   | threshold，不罚分 | 3357 | 4866 | 4444 | **1311** |
   | expected_cost，不罚分 | 6115 | 6192 | 3610 | **1311** |
   | threshold，撞墙罚 10 步（过关数 / 5） | 2 | 0 | 0 | 4 |
   | expected_cost，撞墙罚 10 步（过关数 / 5） | 0 | 0 | 2 | 4 |

   L1 校准确实让 expected_cost 策略比 raw 少走 41% 的步数，但**所有读出方式都输给闭眼全试**。
   原因：完美迷宫里撞墙只花 1 步，而把通往出口的岔路误判成墙一次，就要先钻完另一整棵子树。
   感知准确率低于约 90% 时还不如全试。加罚分反而让所有读出方式都走不出去。
   这和 README Limitations 里"迷宫局部感知上没有读出方式超过多数类基线"一致。

   **不要只放 raw 和 L1 两栏而不提闭眼全试基线**，那样会误导读者（AGENTS.md 规则 1）。

## 已知问题 / 没做完的

- `Explorer` 现在默认 `policy="expected_cost", wall_cost=10`，`bench/run_maze.py` 也是这个默认值；
  但按上面的结论，这组默认值并不好。如果要继续做迷宫，先在开发种子上重新选。
- `scripts/make_maze_gif.py` 用"第几次移动"当时钟；有撞墙罚分时应该改用每个事件里的 `t` 字段（累计步数）。
- 渲染观感：撞墙红色太抢眼，"exit reached" 徽章会压住迷宫右下角的出口，可以调。
- 迷宫结论还没写进 `docs/`，最后应该放到 `docs/results_games.md`，作为诚实的负面结果。
- 可以试 Qwen3-32B，看看感知能不能过 90% 这条线；这个还没跑。

## 下一步：贪吃蛇（已确认的方案，未开始）

- `bench/games/snake.py`：带种子的棋盘（建议 12×12），蛇和食物，`step(action)` 返回吃到 / 死亡。
- 每一步问一个 `Question.choice`，选项**固定**为 `["up", "down", "left", "right"]`。
  选项固定，question key 就不变，batch prior 和 L1 artifact 才能共用；掉头（撞自己脖子）算死亡，让模型自己避免。
- state 文本：ASCII 棋盘 + 明确的坐标（蛇头、朝向、食物、身体从头到尾），并说明 up = 行号 −1 之类的约定。
- L1 标签：BFS 神谕给出的"沿最短安全路径吃到食物"的那一步（平局按选项顺序打破）；
  校准状态从神谕玩的其他种子的对局里采样，200 个。
- 策略：取 argmax 贪心；每局步数上限（比如 300），很久没吃到食物也要截断。
- 指标：吃到的食物数、存活步数、死因；再加 flip rate（选项反序后答案是否改变），这正是 raw 和 L0 的区别，和 README 首图的论点一致。
- **先做感知探针**：在开发状态上看 raw / L0 的走法质量（和神谕一致的比例、致死走法比例），确认模型能玩，再写 runner 和渲染器。
- 渲染器可以直接复用 `make_maze_gif.py` 的卡片、配色、进度条和底部说明；把罗盘换成 4 个选项的概率条。

## 之后：ViZDoom Basic

负责人选的是**文本标签 state**：用 ViZDoom 的 labels buffer 取出敌人的屏幕位置写成文本，
`choice` 在 left / right / shoot 里选，用 main 上的纯文本后端。`vizdoom` 是 MIT 协议。
新依赖和数据集按 AGENTS.md 要先问负责人，并记到 `THIRD_PARTY.md`。

## 环境

- 从仓库根目录跑；需要 `pip install -e ".[hf,bench,dev]"`（torch、transformers、datasets）以及渲染用的 PIL。
- 常用命令：

```bash
python -m bench.run_maze --model Qwen/Qwen3-8B --seeds 0,1,2,3,4
python scripts/make_maze_gif.py bench/results_games/<date>/maze.Qwen__Qwen3-8B.json --seed 0
python handoff/games/maze_dev_probs.py Qwen/Qwen3-8B     # 大约 15 分钟
python handoff/games/maze_sim.py handoff/games/dev_probs.Qwen3-8B.json
python handoff/games/maze_probe.py Qwen/Qwen3-8B
ruff check anyjev bench demo handoff scripts space tests && pytest -q
```

- 迷宫单局在 H100 上 30–120 秒；5 个开发迷宫全格子预计算大约 15 分钟。
