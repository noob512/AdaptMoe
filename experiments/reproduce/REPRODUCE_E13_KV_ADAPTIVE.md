# 复现 E13 KV Cache 自适应档位与固定档位对照

## 实验范围

E13 在相同 Qwen3-30B-A3B-FP8 模型、fixture、请求序列和 seed 上比较四个独立冷进程 arm：

- `adaptive`：从 K4 启动，根据预测和实时 KV 压力在 K2/K4/K6 间切换；
- `fixed_k2`、`fixed_k4`、`fixed_k6`：只在初始化时选择档位，运行期间不切换。

KV 档位与专家 Plan Class 是联动的：KV GPU target 增大时同时增加专家权重
卸载量，以保持 HBM 预算。

| 档位 | GPU KV target | CPU/offline KV | Plan Class | 专家卸载 |
| --- | ---: | ---: | --- | ---: |
| K2 | 2.145 GB | 6.442 GB | cold70 | 15.854 GB |
| K4 | 4.294 GB | 4.294 GB | cold80 | 18.119 GB |
| K6 | 6.441 GB | 2.147 GB | cold89 | 20.158 GB |

表中 GB 为十进制单位。精确几何为 5,461 个 1,572,864-byte KV block，其中一个是
null block；每档保留 16 个 GPU backup block。

## 冻结 workload

`config.json` 定义四类连续 trace：

1. `low_pressure`：4 个 Short/Output-32、B1 低速请求；
2. `high_pressure`：6 个并发 RAG-8192/Output-16 请求；
3. `dynamic_mixed`：Short、Ordinary、Medium 和 RAG cohort 按不同 engine-step 注入；
4. `burst_recovery`：运行中注入 RAG/B5；Adaptive 必须先登记 pending grow，
   在安全边界扩档后准入，再通过连续低压恢复。

qualification 使用 seed `20260901`；formal 和 capacity 使用
`20260901/20260917/20260929`，即每个 arm 三个独立冷进程。capacity 额外用
RAG-8192/Output-16 扫描 B2/B5/B6/B7/B8/B10，测量同时 resident 上限；四个 arm
在同一 repeat 内的 `trace_sha256` 必须相同。

## 前置条件

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
modinfo nvidia_uvm | rg 'filename|version|srcversion'
```

开始前必须是 GPU 0 MiB、无 compute/UVM client，且已加载 stock `nvidia_uvm`。runner 会
临时切换到定制模块，每个 case 后验证 GPU clean，整个矩阵结束后恢复 stock 模块。

首先运行不触发 GPU 正式矩阵的静态检查：

```bash
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  static-preflight
```

该命令包含 shell/Python 语法检查、E13 工具测试和 autoscale/Plan Class 单元测试。

## 单重复 qualification

`RUN_ROOT` 必须指向不存在的新目录：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-kv-adaptive-REPRO-<tag>-attempt-01-qualification \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  qualification
```

## 三重复正式矩阵

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-kv-adaptive-REPRO-<tag>-attempt-01-formal \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  formal
```

默认顺序为每个 seed 内 `adaptive -> fixed_k2 -> fixed_k4 -> fixed_k6`。完整 formal 共 12 个
case，不要并发运行或在运行期间占用 GPU。失败 attempt 必须原样保留。

## 三重复容量包络

容量矩阵必须使用新的目录，不能覆盖主 formal：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-kv-adaptive-REPRO-<tag>-attempt-01-capacity \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  capacity
```

它同样运行 12 个独立冷进程。Output-16 是测量协议的一部分：Output-1 会在首次
Prefill step 结束，post-step 快照无法观测 resident 集合，不得用于 max-running 结论。

## 不计切换开销的单轮稳态阶段对照

此协议回答“经历运行期切换后，目标档的 Prefill/Decode 是否回到直接建立该档的
性能水平”。只运行一轮，共 6 个独立冷进程：
`direct_k2/switched_k2/direct_k4/switched_k4/direct_k6/switched_k6`。

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-tier-phase-once-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  phase-once
```

`RUN_ROOT` 必须不存在，不得覆盖成功或失败 attempt。协议固定为：

- Medium-2048，B1/B8，32-token 冻结 continuation，不记录 logits；
- `direct` 在服务前建立目标档；`switched` 先完整运行源档 32-token conditioning，
  再切换 K4→K2、K6→K4 或 K4→K6；
- 源档 conditioning、切换事务以及两臂共有的目标档 B8 warmup 都在计时窗外；
- 正式 Prefill 记录完整墙钟；Decode 先跑 5 个 warmup step，再取 16 个
  whole-batch 单 token step 的中位数；
- 执行路径为 eager，目标 KV target 和 Plan Class 在测量期间必须不变，两臂输出 hash
  必须一致。

验证：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<phase-root>
jq '{status,expected_case_count,observed_case_count,\
all_prefill_cells_equivalent_within_10pct,\
all_decode_cells_equivalent_within_10pct,rows,\
failed_checks:[.checks[]|select(.ok==false)]}' "$ROOT/verification.json"
cat "$ROOT/summary.md"
```

`status=ok` 表示六臂的功能、输出、档位稳定性和资源清理通过；性能等价需要单独查看
两个 `*_equivalent_within_10pct` 字段。这是单轮 qualification，不是统计等价检验。
冻结 continuation 不保存 logits、hidden state 或专家路由 hash，所以输出 hash 门只是
受控 token 序列与完整执行检查，不得解释为自然生成的数值等价。
若某点超出 ±10% 带，仍应原样报告，不得重复运行直到得到正面结果。

注意：上述固定 5-step warmup 已被后续诊断证明不足。attempt-06 的原始数据保留，
但不得再用其 Decode 超带点归因切档机制。新的 Decode 因果比较必须使用下面的
同进程 ABBA，或另行实现滚动窗口收敛门。

## Decode 差异定因

同进程 ABBA 同时记录逐层逐 step 路由 hash、remote assignment、KV block 顺序 hash、
页表/TLB 事务、MoE/attention kernel 时间和 NVML PCIe RX：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-decode-abba-diagnostic-REPRO-<tag>-attempt-01 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  decode-diagnose
```

CPU backing 4K/64K/2M 构成、冷进程离散度、时钟、P-state 和 PCIe link 诊断：

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-decode-backing-diagnostic-REPRO-<tag>-attempt-01 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  decode-backing-diagnose
```

Nsight Systems 单例 kernel/span/gap 诊断：

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-decode-nsys-diagnostic-REPRO-<tag>-attempt-01 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  decode-nsys-diagnose
```

Nsight 入口用 `cudaProfilerStart/Stop` 限定 condition window，NVTX 标注每个 step。
成功条件包括 `timeline.nsys-rep`、`timeline.sqlite` 和 `analysis.json` 均存在且
`analysis.status=ok`。Nsight 会显著放大 wall-time；只用其 kernel duration、类别和
span/gap 诊断，不把被剖析 wall-time 写入性能表。

当前权威诊断结果为：K4/K6 四个 ABBA 点 switched/direct 为
`0.983×/0.986×/1.013×/1.000×`；原 K4/B8 `1.324×` 与 K6 约 `0.78×` 的偏离消失。
错误原因是独立冷进程的固定 5-step warmup 没有保证 remote-read 路径收敛，不是切档
改变了稳态 Decode 路径。微观变化集中在 MoE/remote-read 侧，但不得在没有专用硬件
计数器时进一步声称是某一级 TLB。

## 离线验证、聚合与绘图

顶层命令成功时已自动执行以下步骤。需要重建分析产物时，先选择新的
`--output-dir`，不覆盖已发布证据：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<successful-formal-root>
CFG=experiments/eurosys_moe_offload_evaluation/e13/config.json
PY=workloads/vllm/vllm/.venv/bin/python

"$PY" experiments/eurosys_moe_offload_evaluation/e13/verify_e13_kv_adaptive.py \
  "$ROOT" --config "$CFG" --profile formal \
  --output "$ROOT/verification.recheck.json"

"$PY" experiments/eurosys_moe_offload_evaluation/e13/aggregate_e13_kv_adaptive.py \
  "$ROOT" --config "$CFG" --profile formal \
  --output-dir "$ROOT/analysis-recheck"

"$PY" experiments/eurosys_moe_offload_evaluation/e13/plot_e13_kv_adaptive.py \
  "$ROOT" --config "$CFG" --profile formal \
  --output-dir "$ROOT/analysis-recheck"
```

主要产物为：

- `verification.json`：12 个 case 的 fail-closed 验收；
- `analysis/e13_summary.{json,csv,md}`：分场景均值、样本标准差与 min/max；
- `analysis/e13_cases.csv`：每个冷进程的原子统计；
- `analysis/e13_global.csv`：完整 trace 的 arm 聚合；
- `analysis/e13_timeseries.csv`：KV 压力、档位、Plan Class、队列和吞吐时序；
- `analysis/e13_capacity_{cases,envelope}.csv`：逐进程与三重复容量包络；
- `analysis/e13_kv_pressure_tier_throughput_latency_timeseries.{png,svg}`：可视化。

## 验收条件

```bash
jq '{status,expected_case_count,observed_case_count}' "$ROOT/verification.json"
jq '{global_best_fixed,scenario_comparisons,global_comparison}' \
  "$ROOT/analysis/e13_summary.json"
```

必须满足：

- `verification.status=ok` 且 12/12 case 存在、返回码为 0；
- 每个 repeat 的 prompt/trace hash 一致，成功请求有完整 TTFT/TPOT/E2E 指标；
- Fixed 档位不变且运行期 resize 为 0；
- Adaptive 覆盖 K2/K4/K6，同时观测到 grow 和 shrink，且 resize 尝试无失败；
- Adaptive 的全部 4 个 late cohort 都有统一控制器记录；burst 必须出现 grow 及其后的
  recovery shrink，且 RAG/B5 请求必须等扩档完成后才准入；
- 三队列守恒、KV target 与 Plan Class 一致；
- 无 Xid/MMU fault、GPU 或 UVM client 泄漏，最终恢复 stock `nvidia_uvm`。

capacity profile 还必须满足：Adaptive/Fixed-K2/Fixed-K4/Fixed-K6 的
`maximum_fully_resident_offered_concurrency` 分别为 7/2/5/7，B8/B10 超额请求计入
`requests_not_simultaneously_resident`，不能标成 OOM 或失败。

完整输出 token hash 作为数值敏感性诊断，必须报告分叉；当前验收的硬门是同一
prompt/trace、请求完整生成与共同首 token 一致。不得把该门表述为逐 token 完全一致。

## 限制

E13 是单机单卡、phase-batched、eager 路径的三 seed qualification，不是开环生产服务
soak。capacity profile 给出当前 B≤10、context≤8,208 的 fully-resident 上限；请求仍可
排队后完成，所以这不是物理 OOM/拒绝上限。per-scenario best-fixed 是事后 oracle；
整条 trace 的 global best-fixed 才是长期固定策略的主要对照。

late cohort 需要增长时会先留在 runner admission gate 外，等待当前运行请求排空后切换
Plan Class。逻辑到达时间在等待前记录，所以这段安全边界等待完整计入 queueing/TTFT/E2E；
当前 formal 中 burst arrival→admission 平均为 26.519 s。

## 当前成功证据

2026-09-01 三重复目录：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/
```

该目录 12/12 case 返回 0，`verification.json` 为 `status=ok`，合计 384/384
请求成功。修复后的 burst 三轮均为 `K2→K4→K2`，且 5 个 RAG 请求在扩档后
准入。Adaptive 完整 trace 吞吐为 0.2163 req/s，只有 global best Fixed-K6 的
0.714×；因此该成功 artifact 是机制 qualification 和负性能结果，
不是“Adaptive 性能更好”的证据。

容量包络目录：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-kv-adaptive-20260901-attempt-13-capacity-formal/
```

该目录 12/12 case、456/456 请求通过。Adaptive/Fixed-K2/Fixed-K4/Fixed-K6 的
fully-resident 上限为 B7/B2/B5/B7；Adaptive capacity 吞吐为 best Fixed-K4 的
0.702×。

完整解释见：

```text
experiments/eurosys_moe_offload_evaluation/
  RESULTS_E13_KV_ADAPTIVE_20260901.md
```

不计切换开销的单轮稳态阶段证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-tier-phase-once-20260901-attempt-06/
```

该目录 6/6 arm 返回 0，`verification.status=ok`，冻结输出 hash 6/6 一致。
Prefill 6/6 cell 在 switched/direct ±10% 内。Decode 原始表只有 3/6，但该解释已被
同进程 ABBA 定因结果取代：超带来自 warmup 协议错误，不能归因切档。该 attempt 是
正式受控序列原始结果；自由 greedy 的 attempt-05 因 K4/B1、K4/B8 输出 hash 分叉而
只作辅助诊断。
