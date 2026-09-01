# E3 Prefill/Decode 路径 crossover 重跑报告（2026-08-25）

## 1. 结论摘要

本次重新生成了 Qwen3-30B-A3B-FP8 Track A 的逐层专家计划，并在同一模型、输入、KV、
逐层专家坐标和 runner 下重跑 hybrid（H）、always-copy（A）和 always-remote（R）。最终
run root 为：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/e3-formal-20260825T101855Z
```

总 verifier 返回 `status=ok`、`errors=[]`：30 个可执行 custom arm-case 全部通过，15 个
不支持的 arm-case 明确写为 N/A；10 个完整 H/A/R cell 中，三条路径的输出 token hash
逐 cell 完全一致。KT 的 5 个 workload block 全部通过，共产生 13 个合法 batch cell。

观测 E2E 最优策略热力图中，10 个可执行 cell 有 9 个为 always-remote、1 个为 hybrid，
always-copy 为 0 个。唯一的 hybrid 最优点是 `(prompt=512, output=128, B1)`，比
always-remote 快 95.55 ms（1.20%）。由于本轮只有 1 个独立冷进程 block，该差异只能作为
描述性点估计，不能称为统计显著。

更稳定的阶段结论来自 H/A 配对：hybrid 的 remote Decode 稳态中位数在所有可执行 cell
均低于 always-copy，付出的固定 cutover 为 126.68--130.13 ms。以实测 E2E 为锚点的局部
仿真给出 H 相对 A 的 output break-even 为 12--92 tokens，具体取决于 prompt 和 batch。
但 hybrid 相对 always-remote 并未普遍获胜：本计划非常冷，480 个坐标中 469 个在
calibration Decode 零命中，因此本轮不足以支撑“Prefill-copy/Decode-remote 普遍优于
always-remote”的强主张。

本报告是可审计的单 block 描述性 E3，不满足评测计划中机制 microbenchmark 每点至少
10 个独立冷进程的论文级重复要求，不报告 CI 或显著性。

## 2. 新冻结专家计划

Calibration artifact：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/e3-calibration-20260825T091247Z
```

| 项目 | 值 |
|---|---:|
| 模型 | Qwen3-30B-A3B-FP8 |
| MoE 层 / 每层专家 | 48 / 128 |
| 每层 CPU-cold / GPU-hot | 10 / 118 |
| 总 cold 坐标 | 480 |
| 主专家参数 selected bytes | 2,264,924,160 B（2.109375 GiB） |
| scale bytes | 552,960 B |
| selected bytes（含 scale） | 2,265,477,120 B |
| GPU-hot 主参数 | 26,726,105,088 B |
| alias backing | 94,371,840 B（90 MiB） |
| calibration Decode tokens / route slots | 224 / 86,016 |
| cold 坐标 calibration Decode 命中 | 11 slots（0.012788%） |
| calibration Decode 零命中坐标 | 469 / 480 |

冻结与复验 hash：

```text
routing source SHA256 : f663c22d08d7e3a4d8ef40385252a0c5e32c000fcc68a04437f67aef09ba355a
plan SHA256           : 69d625693e7bd31ce00b9847bcfa0c80ccb487e5c0da03f1c985276ebf1f99af
coordinate SHA256     : c8754744311aebcdc1a5d768c849a346b33c51027ce3d0998b225e0107dde898
exact names SHA256    : 6142b29ba13671bd2c0c1e18525a64e1bd3f676ad369add9d3d2323086ac4141
KT placement SHA256   : 4b0b4d951b1bf7f600d31d3b3898b50a48613f6d0fcbe004cc44f228c129f6e2
```

KT logical-count mask 已在生成时验证为 frozen hot plan 的精确 top-k mask；运行日志进一步
验证 48/48 层均加载 118 个 GPU experts，即每层保留同一计划的 10 个 CPU experts。

## 3. 实验配置与正确性门

内部三路径使用同一个 direct-engine eager runner、相同调度和计时边界：

- H：Prefill alias/copy，首个 Decode 前切为 remote；
- A：Prefill/Decode 都保持 sealed alias copy 路径；
- R：Prefill/Decode 都使用 remote map；
- strict batch：B1/B4/B8；
- warmup Decode steps：5；其余 Decode steps 全部计入；
- selected 参数：96 个专家大 tensor、480 个 `(layer, expert)` 坐标；
- 执行顺序按 cell 轮换 H/A/R，seed=0。

Verifier 对每个成功 custom case 检查：输出长度和同 cell hash、运行时坐标 hash、selected
bytes、H cutover、A sealed 状态、H/R remote-map 全字节覆盖、cold/hot residency、
`UNKNOWN=0`、无 scheduler preemption、shutdown 后 `session_count=0`。所有检查通过。

本次路径 runner 为 eager，不把结果写成 CUDA Graph crossover。KT 使用 SGLang eager、
localhost HTTP SSE；其 TTFT/E2E 包含框架和传输开销，而 custom 是 direct-engine 计时，因此
KT 只作为 exact-coordinate 辅助表，不与 H/A/R 混算 speedup。

## 4. 内部路径实验数据

下表为用户可见 cycle E2E（ms，越低越好）。粗体标出同 cell 最小值。

| Prompt/Output | Batch | Hybrid H | Always-copy A | Always-remote R | 最优 |
|---|---:|---:|---:|---:|---|
| 128/32 | B1 | 2,127.31 | 2,086.46 | **1,990.83** | R |
| 128/32 | B4 | 2,145.52 | 2,152.12 | **2,110.96** | R |
| 128/32 | B8 | 2,156.78 | 2,143.12 | **2,018.06** | R |
| 512/128 | B1 | **7,835.53** | 8,100.91 | 7,931.08 | H |
| 512/128 | B4 | 8,107.20 | 8,326.86 | **7,990.95** | R |
| 512/128 | B8 | 8,164.82 | 8,349.45 | **8,091.53** | R |
| 2,048/128 | B1 | 8,027.66 | 8,244.79 | **7,936.98** | R |
| 2,048/128 | B4 | N/A | N/A | N/A | N/A |
| 2,048/128 | B8 | N/A | N/A | N/A | N/A |
| 8,192/128 | B1/B4/B8 | N/A | N/A | N/A | N/A |
| 512/512 | B1 | 31,263.78 | 32,166.15 | **31,098.48** | R |
| 512/512 | B4 | 31,611.02 | 32,837.81 | **31,532.52** | R |
| 512/512 | B8 | 32,303.80 | 33,854.47 | **32,152.45** | R |

H 的 cutover 在全部 10 个成功 cell 中为 126.68--130.13 ms。稳态 Decode step 中位数
为：H 59.53--62.42 ms，A 62.48--65.71 ms，R 60.40--64.46 ms。H 与 R 最终都是
remote Decode；二者约 0--2 ms 的 Decode 差异和约百分之一的总时间差，在单 block 下应
视为运行噪声候选，不能解释为不同 Decode 机制。

## 5. 最优策略热力图

面板为 strict batch，单元格为“最优 arm / E2E ms”：

| Prompt/Output | B1 | B4 | B8 |
|---|---:|---:|---:|
| 128/32 | R / 1,990.83 | R / 2,110.96 | R / 2,018.06 |
| 512/128 | H / 7,835.53 | R / 7,990.95 | R / 8,091.53 |
| 2,048/128 | R / 7,936.98 | N/A | N/A |
| 8,192/128 | N/A | N/A | N/A |
| 512/512 | R / 31,098.48 | R / 31,532.52 | R / 32,152.45 |

该热力图的直接含义是：对这份极冷的 10-experts/layer 计划，避免 Prefill copy/cutover
通常更合适。它不等价于“remote 对任意专家集合都最好”；如果 CPU 专家更频繁参与 Prefill，
copy 的复用收益会增大，边界必须重新测量。

## 6. Break-even

### 6.1 Hybrid 与 always-copy

为了隔离 Decode remote 相对每步 copy 的摊销，使用以下局部模型：

```text
T_path(O) = measured_cycle_e2e(O0) + (O - O0) * steady_decode_median
```

它保留实测 Prefill、cutover、warmup 和固定成本，只用实测稳态 Decode 中位数外推 output。
下表给出 H 开始优于 A 的最小整数 output tokens：

| Prompt | B1 | B4 | B8 |
|---:|---:|---:|---:|
| 128 | 47 | 31 | 36 |
| 512（ordinary） | 47 | 51 | 53 |
| 2,048 | 53 | N/A | N/A |
| 512（long-generation anchor） | 92 | 12 | 41 |

因此，在当前路径实现中，约 127--130 ms 的一次性 cutover 能被 H 相对 A 每 Decode step
约 2--4 ms 的优势在几十个 output token 内摊销。这一配对支持“长 Decode 应避免每步 copy”。

### 6.2 三策略局部最优区间

同一模型在 `O=1..4096` 上形成以下描述性最优区间：

| Prompt | Batch | 局部模型最优区间 |
|---:|---:|---|
| 128 | B1 | R: 1--799；H: 800--4,096 |
| 128 | B4 | R: 1--42；H: 43--4,096 |
| 128 | B8 | R: 1--360；H: 361--4,096 |
| 512 ordinary | B1 | R: 1--79；H: 80--4,096 |
| 512 ordinary | B4 | A: 1--2；R: 3--867；H: 868--4,096 |
| 512 ordinary | B8 | A: 1--14；R: 15--544；H: 545--4,096 |
| 2,048 | B1 | A: 1--15；R: 16--826；H: 827--4,096 |
| 512 long anchor | B1 | R: 1--3,046；H: 3,047--4,096 |
| 512 long anchor | B4 | R: 1--733；H: 734--4,096 |
| 512 long anchor | B8 | R: 1--4,096 |

这些不是对每个 output 长度的独立实测。尤其 H/R 的 Decode 机制相同、斜率差很小，远离
实测 anchor 的交点对单次噪声高度敏感；它们只能用于生成下一轮密集扫描点，不能直接作为
生产策略阈值。可发布的 break-even 需要至少 10 个冷进程重复，并在交点附近增加真实
output 长度。

## 7. KTransformers exact-coordinate 辅助数据

KT 每层 118 GPU / 10 CPU experts，坐标来自同一冻结计划。TTFT 和 E2E 为 HTTP SSE client
wall time；Decode 为 batched step 中位数。

| Prompt/Output | Batch | TTFT ms | Decode step ms | E2E ms |
|---|---:|---:|---:|---:|
| 128/32 | B1 | 987.30 | 171.92 | 6,685.52 |
| 128/32 | B4 | 1,089.07 | 253.62 | 8,939.00 |
| 128/32 | B8 | 1,397.31 | 253.51 | 9,284.25 |
| 512/128 | B1 | 1,033.68 | 197.69 | 24,870.18 |
| 512/128 | B4 | 1,598.22 | 254.90 | 34,669.79 |
| 512/128 | B8 | 5,679.31 | 438.88 | 71,950.70 |
| 2,048/128 | B1 | 2,648.73 | 246.08 | 33,843.72 |
| 2,048/128 | B4 | 9,270.37 | 284.27 | 47,439.44 |
| 2,048/128 | B8 | 10,717.78 | 320.17 | 53,129.85 |
| 8,192/128 | B1 | 5,396.46 | 146.26 | 24,950.10 |
| 512/512 | B1 | 2,118.85 | 205.01 | 101,778.46 |
| 512/512 | B4 | 1,551.32 | 283.69 | 147,106.01 |
| 512/512 | B8 | 4,872.78 | 312.14 | 163,385.86 |

KT 同 batch 的 conditioning/measurement 输出 hash 一致。KT 与 custom 的生成 hash 不同，
且两者计时边界不同；结合既有跨框架 teacher-forced correctness hold，本表不得用于宣称
跨框架 speedup 或总排名。

## 8. N/A 与公平性边界

- custom `2,048/B4`、`2,048/B8` 和全部 `8,192/B*` 在单步 Prefill token 数达到
  8,192 时，平台后端会触发 TMA descriptor 700、CUBLAS/illegal-address。最终 run 根据
  已观察到的 backend preflight 失败 fail-closed 预登记 N/A，没有把失败 timing 混入结果。
- KT RAG 只运行 B1；B4/B8 所需 KV 分别为 3,277,848,576 和 6,555,697,152 B，超过
  2.8 GB 预登记 KT KV headroom。
- 原生 vLLM pinned UVA/copy 只能卸载完整 Parameter，不能表达 Qwen fused expert tensor
  内的同一 480 个 `(layer, expert)` slice。因此 Track A 的 pinned UVA 和 copy-on-forward
  都明确记为 N/A；用整 Parameter 近似会改变 selected bytes 和专家坐标，违反公平性协议。
- 早期 runner-shadow、KV 无 guard、TMA/illegal-address attempts 只作为诊断保留；本报告
  的性能值只来自 `e3-formal-20260825T101855Z`。

## 9. 实验结论

1. 新计划、坐标和 KT mask 已正确冻结，三条内部路径在同 cell 上生成完全一致，remote map、
   residency、UNKNOWN 和清理证据全部通过。
2. remote Decode 相对 always-copy 有明确的单次运行摊销形态：H/A break-even 落在
   12--92 output tokens，说明阶段切换能避免长 Decode 的重复 copy 成本。
3. 本轮没有证明 hybrid 普遍优于 always-remote。实际离散热力图为 R 9/10、H 1/10；
   该极冷计划使 remote-read 需求很低，而 hybrid 仍支付约 128 ms cutover。
4. 因此当前最诚实的阶段策略结论是：对这份计划优先 always-remote；只有在实测表明
   Prefill copy 的复用收益足以覆盖 cutover，且 output 超过 H/A 摊销点时，才选择 hybrid。
5. C1“阶段专用路径优于单一路径”目前只得到 H 对 A 的有限支持，尚未通过 H 对 R 的广泛
   验证。下一轮应增加更高 route-density 的合法专家计划、在预测交点附近密集扫描 output，
   并补齐每点 10 个独立冷进程。

## 10. 数据与复验入口

```text
analysis/verification.json
analysis/custom_metrics.csv
analysis/best_strategy_heatmap.csv
analysis/best_strategy_heatmap.svg
analysis/pairwise_break_even.csv
analysis/strategy_regions.csv
analysis/kt_exact_coordinate_metrics.csv
analysis/na_matrix.csv
```

复验命令：

```bash
python3 experiments/eurosys_moe_offload_evaluation/verify_formal_e3.py \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-formal-20260825T101855Z

python3 experiments/eurosys_moe_offload_evaluation/plot_e3_heatmap.py \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-formal-20260825T101855Z/analysis/best_strategy_heatmap.csv \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-formal-20260825T101855Z/analysis/best_strategy_heatmap.svg
```

实验退出后无 GPU compute process，`nvidia-smi` 显存为 0 MiB；运行前后 stock
`nvidia-uvm.ko` SHA256 均为
`9e77e3c50fcc40bbc64b50128ff1d05a1775fa75deb71976c1adeffbf278831f`。
