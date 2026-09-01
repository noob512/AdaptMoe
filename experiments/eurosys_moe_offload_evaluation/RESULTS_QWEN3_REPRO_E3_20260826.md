# Qwen3 旧对比复现与 E3 crossover 实验报告

日期：2026-08-26（Asia/Shanghai）

模型：`Qwen/Qwen3-30B-A3B-FP8`

## 1. 执行结论

1. Qwen3 下 ours 与 KTransformers（KT）的旧对比已成功复现。
   - ours：2/4 GiB × eager/Graph × B1/B4，共 8/8 case 成功。
   - KT：2/4 GiB × eager/Graph，共 4/4 server case 成功。
   - 输出稳定性、专家 mask/坐标、Graph 命中和退出清理均通过。
2. Qwen3 E3 已完整执行并通过 verifier。
   - custom Track A：45 个预注册记录，其中 30 个 `ok`、15 个预注册 `N/A`、0 个 `error`。
   - KT：5/5 workload block 成功，形成 13 个 exact-coordinate measurement cell。
   - H/A/R 在 10 个共同有效坐标上的输出 token hash 全部一致。
   - verifier：`status=ok`、`errors=[]`。
3. 本轮没有复现此前 Qwen 动态 post-warmup rebalance 路径中的 hot-expert prefetch/PDE/PTE 故障，因此正式重跑未触发代码修复分支。
4. 当前只有一个独立 process block；结果是描述性点估计，没有置信区间，尚不满足 publication-grade repetition 要求。

## 2. 固定专家集合与实验路径

E3 三条 custom 路径使用同一 frozen expert plan：

- 48 个 MoE 层；
- 每层固定 10 个 CPU/cold experts，共 480 个 `(layer, expert)` 坐标；
- main expert weights：2,264,924,160 B（2.109375 GiB）；
- frozen plan SHA256：`69d625693e7bd31ce00b9847bcfa0c80ccb487e5c0da03f1c985276ebf1f99af`；
- coordinate SHA256：`c8754744311aebcdc1a5d768c849a346b33c51027ce3d0998b225e0107dde898`。

路径定义：

- H：hybrid Prefill-copy / Decode-remote；
- A：always-copy；
- R：always-remote；
- KT：相同 480 个逻辑专家坐标，118 GPU experts/layer、10 CPU experts/layer。

native vLLM pinned-UVA/copy-on-forward 在 exact-coordinate Track A 中为 `N/A`：其公开配置只能按整个 Parameter tensor offload，不能精确复现固定的 480 个 expert-slice 坐标。它不是运行失败，也不能用近似坐标替代进入公平排名。

## 3. 旧 Qwen3 对比复现

Decode median（ms）：

| 配置 | B1 | B4 |
|---|---:|---:|
| ours 2 GiB eager | 59.23 | 59.30 |
| ours 2 GiB Graph | 9.26 | 14.50 |
| ours 4 GiB eager | 57.77 | 57.17 |
| ours 4 GiB Graph | 9.98 | 17.83 |
| KT 2 GiB eager | 132.96 | 294.58 |
| KT 2 GiB Graph | 31.78 | 131.97 |
| KT 4 GiB eager | 157.06 | 805.89 |
| KT 4 GiB Graph | 38.79 | 701.09 |

解释限制：ours 使用同步 EngineCore 边界；KT 使用 localhost HTTP/SSE。该表证明旧实验可复现及各自路径稳定，但不应直接把比值表述为严格的跨框架 speedup。

## 4. E3 最优策略热力图数据

Cycle E2E（ms）：

| workload | prompt/output | batch | H | A | R | 最优 |
|---|---:|---:|---:|---:|---:|---|
| short-dialogue | 128/32 | 1 | 2,652.16 | 2,368.87 | 2,207.79 | R |
| short-dialogue | 128/32 | 4 | 2,483.80 | 2,604.70 | 2,405.21 | R |
| short-dialogue | 128/32 | 8 | 2,608.86 | 2,381.59 | 2,252.43 | R |
| ordinary-chat | 512/128 | 1 | 8,668.82 | 9,024.73 | 8,518.02 | R |
| ordinary-chat | 512/128 | 4 | 8,742.66 | 9,055.85 | 8,805.51 | H |
| ordinary-chat | 512/128 | 8 | 9,117.60 | 9,494.61 | 8,940.93 | R |
| medium-context | 2048/128 | 1 | 9,005.98 | 9,017.41 | 8,482.18 | R |
| long-generation | 512/512 | 1 | 34,909.98 | 35,795.72 | 34,043.64 | R |
| long-generation | 512/512 | 4 | 35,948.00 | 37,412.18 | 34,794.22 | R |
| long-generation | 512/512 | 8 | 35,497.48 | 37,560.66 | 35,652.24 | H |

赢家计数：R=8、H=2、A=0。

直接结论：在本轮固定 2.109 GiB 冷专家集合与测量点下，always-remote 通常最优；hybrid 只在 ordinary-chat B4 和 long-generation B8 获胜；always-copy 没有成为任何完整坐标的最优策略。阶段专用路径的收益只在较高 batch/足够长 Decode 时抵消约 247--269 ms 的 cutover 成本。

## 5. H 与 R 的 break-even

verifier 使用局部描述模型：

`anchor cycle E2E + (output_tokens - anchor_output_tokens) × steady Decode median`

| workload/batch | H 对 R 的结论 |
|---|---|
| short B1 | R 对所有 output 长度占优 |
| short B4 | output ≥46 时 H 预计占优；实测 output=32 时 R 最优 |
| short B8 | R 对所有 output 长度占优 |
| ordinary B1 | R 对所有 output 长度占优 |
| ordinary B4 | output ≥97 时 H 预计占优；实测 output=128 时 H 最优 |
| ordinary B8 | R 对所有 output 长度占优 |
| medium B1 | R 对所有 output 长度占优 |
| long B1 | R 对所有 output 长度占优 |
| long B4 | R 对所有 output 长度占优 |
| long B8 | output ≥295 时 H 预计占优；实测 output=512 时 H 最优 |

这些 break-even 是单锚点外推，不是对所有 output 长度的独立实测，应在重复实验后再作为论文定量边界。

## 6. KT exact-coordinate 结果

| workload | batch | KT cycle E2E ms | KT steady Decode median ms |
|---|---:|---:|---:|
| short-dialogue | 1 | 6,612.10 | 172.94 |
| short-dialogue | 4 | 10,308.24 | 293.82 |
| short-dialogue | 8 | 10,985.44 | 299.82 |
| ordinary-chat | 1 | 27,465.09 | 210.62 |
| ordinary-chat | 4 | 40,167.23 | 301.14 |
| ordinary-chat | 8 | 51,923.16 | 317.31 |
| medium-context | 1 | 38,593.46 | 288.56 |
| medium-context | 4 | 51,908.77 | 311.27 |
| medium-context | 8 | 108,387.51 | 712.41 |
| RAG | 1 | 23,670.37 | 147.93 |
| long-generation | 1 | 116,805.85 | 227.26 |
| long-generation | 4 | 160,105.46 | 309.68 |
| long-generation | 8 | 175,956.23 | 334.31 |

KT 结果验证了同一 logical coordinate 集合可执行。custom 与 KT 的测量边界不同，尤其 KT 包含 HTTP/SSE client boundary，因此不能仅依据 E2E 比值宣称严格跨框架 speedup。

## 7. N/A 与已知覆盖边界

custom 的 15 个 `N/A` 是运行前注册的能力边界：

- medium-context B4/B8：单 step Prefill 为 8,192/16,384 tokens；
- RAG B1/B4/B8：单 step Prefill 为 8,192/32,768/65,536 tokens；
- 每个坐标有 H/A/R 三条路径，共 15 条记录。

原因是本平台 custom backend 的既有证据表明 `>=8192` token 单步 Prefill 会触发 TMA descriptor 700 / illegal-address。正式 E3 没有把这些不安全坐标伪装成成功，也没有把 N/A 计入赢家。KT 能运行其中部分坐标；这说明限制来自 custom Prefill backend，而非 fixture 本身。

## 8. 期间出现的 Qwen 诊断错误

这些错误发生在正式旧对比/E3 之前的“自动迁回完成 warmup，再切 fault mode 2，最后只做一次 post-warmup rebalance”诊断路径；它们没有出现在本次 frozen-plan E3 中。

| attempt | 直接错误 | 定位与处理 |
|---|---|---|
| 01 | `RuntimeError('Hot Expert prefetch failed')` | mode 0 warmup 成功、0→2 切换成功；唯一一次 post-warmup rebalance 的显式 hot prefetch 失败，正式请求尚未开始，也没有新 Xid。旧异常没有保留底层 prefetch range/return-code，不能进一步区分空间不足与驱动状态转换失败。 |
| 02 | 等待 pre-rebalance fault-mode cutover 超时，observed 仍为 0 | 为追踪 attempt 01 加入逐 region Phase-F 调试后，启动路径严重变慢，外部 controller 没有在超时窗内完成 0→2；这是诊断时序/日志开销错误，不是新的 UVM 数据面根因。 |
| 03 | `Stage-M expert offload requires envelope registration` | 启动配置漏设 `VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1`，被安全门拒绝；属于易修复配置错误。 |
| 04 | 初始化长期停留，未形成正式结果 | Phase-F 逐 region 日志增长到约 32 MiB，并阻塞在文件系统 journal commit；受控终止后成功恢复 stock UVM，无新 Xid，无需重启。 |

本轮正式 E3 使用 frozen static alias/envelope plan，并显式关闭动态启动后搬运：

- `VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0`；
- `VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0`；
- `VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0`；
- `VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0`。

这并不等于“不卸载专家”：2.109 GiB/480 个 frozen cold coordinates 仍通过 `uvm_original_va` static alias plan 放在 CPU。它只避免进入尚未稳定的 post-warmup 动态 rebalance/prefetch 路径。

## 9. 正确性、清理与最终状态

- 10 个完整 custom 坐标的 H/A/R output token hash 全部相等；
- custom detailed audit failures：空；
- cold expert bytes 与固定计划一致；
- GPU 最终 0 MiB，无 compute/UVM client process；
- stock UVM 已恢复：`/usr/lib/modules/6.13.12-061312-generic/kernel/drivers/video/nvidia-uvm.ko`；
- verifier `status=ok`，`errors=[]`。

## 10. 主要产物

- 旧对比 ours：`experiments/qwen_decode_six_mechanism_compare/artifacts/repro_ours_20260826_attempt_01`
- 旧对比 KT：`experiments/qwen_decode_six_mechanism_compare/artifacts/repro_kt_20260826_attempt_01`
- E3 run root：`experiments/eurosys_moe_offload_evaluation/artifacts/e3-repro-20260826-attempt-01`
- verifier：`artifacts/e3-repro-20260826-attempt-01/analysis/verification.json`
- heatmap CSV/SVG：`analysis/best_strategy_heatmap.csv`、`analysis/best_strategy_heatmap.svg`
- break-even：`analysis/pairwise_break_even.csv`
- custom metrics：`analysis/custom_metrics.csv`
- KT metrics：`analysis/kt_exact_coordinate_metrics.csv`
