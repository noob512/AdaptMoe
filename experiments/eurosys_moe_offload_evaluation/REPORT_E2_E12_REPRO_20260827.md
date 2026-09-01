# EuroSys MoE Offload E2--E12 复现实验报告（2026-08-27）

## 1. 结论摘要

本轮从 E2 开始重新生成独立 artifact，没有把旧结果复制为本轮结果。E2、E3、E4、
E7、E10 的三个 qualification、E11 定向测试和 E12 的 1,000-replay Graph smoke
均得到可验证的通过结果；E5 的全部三组补充配对已完成，E6 已完成五档 difference
formal。E8 仍在 provenance 门阻塞；E9 已补齐冻结 reference、完成 calibration 和七臂
qualification，但因跨臂 greedy token 漂移未通过总体验收。
完整论文矩阵尚未完成，因此本报告不发布跨系统 speedup，也不把单次 qualification
写成正式论文重复。

| 实验 | 本轮状态 | 证据级别 | 核心结果 | 是否满足计划中的完整正式范围 |
|---|---|---|---|---|
| E2 | 通过 | 单次 ours/Qwen qualification | 16/16 cells，116/116 请求成功，62/116 满足 SLO | 否：缺跨系统 equal-HBM 和每点 5 个进程 |
| E3 | 通过 | 单进程描述性 qualification | 10 个有效 crossover cell：always-remote 8，hybrid 2 | 否：缺每点 10 个冷进程，5 个长 Prefill cell 为 N/A |
| E4 | 通过 | 5 个独立进程 block | 160 phase records；C1--C32 无持续退化；PCIe RX 峰值 6.027 GiB/s | 部分：CPU DRAM 带宽因 KVM 为 N/A |
| E5 | 通过 | 各配对每 batch 10 个冷进程 | DESTROY/PARK、sequential/grouped PTE、stable/non-stable VA 三组补测均通过 | 是；DESTROY/PARK 只比较 transition timing，见输入哈希边界 |
| E6 | 部分通过 | 五档 difference、每档 10 个冷进程 | 5/10/25/50/100% outbound median 480.98/637.26/943.25/1400.21/2303.34 ms | 否：缺 full rebuild、逐 wave、全层 batched transaction 和 break-even |
| E7 | 通过 | 同步控制面 microbenchmark | 热度更新 median/p95 277.392/439.168 µs | 否：缺 workload-shift E2E |
| E8 | 阻塞 | 静态 preflight | 缺 fresh build receipt；MoE-Infinity commit 也漂移 | 否 |
| E9 | 未通过 | 7-arm qualification | 7/7 case 运行成功，但 uniform 与 no-mixed 出现 greedy token 漂移 | 否：未启动 5-block formal |
| E10 | 通过 | 三项 E2E/路径 qualification | 三队列、stepped/direct、2--10 GiB 均通过 | 否：缺自动控制器 5-seed open-loop 正式矩阵 |
| E11 | 通过 | scheduler unit qualification | 3 passed，76 deselected | 否：仅支持 phase-batched scope |
| E12 | 通过 | 1,000 Graph replay smoke | 0 errors；median 45.9045 ms；49.015 GiB/s logical H2D | 否：不是 6 h、1,000 次 PARK/RESUME 或故障注入 |

尚未解决项及继续执行条件见
[UNRESOLVED_E2_E12_REPRO_20260827.md](UNRESOLVED_E2_E12_REPRO_20260827.md)，
已解决错误见
[ERROR_FIX_LOG_E2_E12_REPRO_20260827.md](ERROR_FIX_LOG_E2_E12_REPRO_20260827.md)。

## 2. 环境、依赖与数据检查

### 2.1 平台

| 项目 | 实际值 |
|---|---|
| 仓库提交 | `6fe5033aa9645d1c4eda61a0d1bc52e189c3c864`，工作区有既有未提交修改 |
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB |
| NVIDIA driver | 595.71.05 |
| Kernel | `6.13.12-061312-generic` |
| CUDA toolkit | 12.8 (`Build cuda_12.8.r12.8/compiler.35583870_0`) |
| Python | 3.12.3，`workloads/vllm/vllm/.venv/bin/python` |
| CPU | Intel Xeon Gold 6530，14 vCPU，单 NUMA node `0-13` |
| 虚拟化 | KVM |
| 文件系统可用空间 | 135 GiB |
| Qwen snapshot | `d206ba732169f29bb77fbf80fc2c4b81d4d30782` |
| DeepSeek snapshot | `85864749cd611b4353ce1decdb286193298f64c7` |

每个 GPU runner 在执行前检查模型、fixture、plan、Python、GPU 空闲和 UVM 模块；涉及
custom UVM 的 runner 保存 `module.before.json`、`module.custom.json`、
`module.after.json`，并在结束后恢复 stock 模块。本轮最终检查为 GPU 0 MiB、无 compute
process，`nvidia_uvm` 为：

```text
/lib/modules/6.13.12-061312-generic/kernel/drivers/video/nvidia-uvm.ko
```

### 2.2 冻结输入与公平性边界

- E2 使用 `eurosys-pa-v2-20260824` Qwen evaluation fixtures 和 cold19/cold25 plans；
  KV 固定为 2 GiB。
- E3/E4 使用冻结的 2-GiB exact-coordinate plan，selected bytes 为 2,264,924,160 B；
  E3 的 native vLLM expert-slice exact-coordinate 路径按能力检查写为 N/A。
- E6 使用冻结的 DeepSeek C2/C4/C6/C8 class manifest。
- E10 使用 Qwen 冻结 expert plans、KV block geometry 和 incremental wave plan。
- E8 没有绕过 provenance 门；E9 使用补充并冻结的新 reference snapshot，calibration 与
  evaluation split 已验证不相交。

## 3. 两项重点风险的验证

### 3.1 warmup 后 rebalance / 热门专家恢复

E2 在正式请求前记录了
`cpu_offload_uvm_original_va_sealed_hot_residency_reconciled`，即 warmup/Graph capture 后
重新把 survivor 热门专家迁回 GPU。serving gate 结果如下：

| block | 热专家 GPU/CPU/unknown | 冷专家 CPU/GPU/unknown | gate free / minimum |
|---|---:|---:|---:|
| C1/cold19 | 24,687,673,344 / 0 / 0 B | 4,303,355,904 / 0 / 0 B | 971,767,808 / 201,326,592 B |
| C4/C8/C16/cold25 | 23,328,718,848 / 0 / 0 B | 5,662,310,400 / 0 / 0 B | 1,185,677,312 / 201,326,592 B |

两个 block 的 hot prefetch driver result 均为 0，Parameter VA 在 reconciliation 前后不变。
正式请求期间的 hot-read gate 分别通过 103 和 370 次，没有观察到热门专家被 KV cache
持续挤到 CPU 或 unknown。

### 3.2 warmup 阶段预留临时缓冲区

E2 配置在正式 serving 前捕获并保留最大 8,192/16,384-token pure-Prefill Graph，启用
`VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=2147483648`、workspace warmup audit 和
201,326,592-B minimum-free gate。结果为：

| block | Prefill Graph | Decode Graph | allocator reserved before/after | allocator cache |
|---|---|---|---:|---|
| C1/cold19 | 128--8,192 tokens，共 6 个 | B1/B4/B8/B16，共 4 个 | 34,701,574,144 / 34,701,574,144 B | preserved |
| C4/C8/C16/cold25 | 128--16,384 tokens，共 7 个 | B1/B4/B8/B16，共 4 个 | 35,494,297,600 / 35,494,297,600 B | preserved |

reserved bytes 未在 reconciliation/cleanup 间增加，说明正式请求没有再触发同级别临时池
扩容；16/16 cells 全部通过且无 fatal CUDA/UVM pattern。E5 的独立 dual-topology 初始
配置也曾因多余 scheduler/workspace 预留触发相同安全门；第 7 节记录了根因、修复和
22 个冷进程复测，不能用 E2 的通过结论替代该独立证据。

## 4. E2：Qwen 自适应并发 qualification

### 4.1 配置与命令

```bash
bash experiments/eurosys_moe_offload_evaluation/run_qwen_e2_adaptive_concurrency.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-qwen3-adaptive-concurrency-REPRO-20260827-attempt-01

python3 experiments/eurosys_moe_offload_evaluation/summarize_qwen_e2_adaptive.py \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-qwen3-adaptive-concurrency-REPRO-20260827-attempt-01
```

关键参数：2 GiB KV、`max_model_len=8320`、`max_num_batched_tokens=16384`、
`max_num_seqs=16`、`gpu_memory_utilization=0.97`、`FULL_SEPARATE`；C1 使用 cold19/
8,192-token Graph，C4/C8/C16 使用 cold25/16,384-token Graph。实际 server 命令保存在
各 block 的 `command.txt`。

### 4.2 结果

| workload/C | TTFT mean ms | client TPOT mean ms | steady Decode TPOT ms | SLO 通过/请求 | goodput req/s |
|---|---:|---:|---:|---:|---:|
| short/C1 | 621.62 | 34.14 | 10.90 | 1/1 | 0.595 |
| short/C4 | 2,292.60 | 43.37 | 17.71 | 1/4 | 0.240 |
| short/C8 | 2,389.00 | 45.43 | 18.53 | 1/8 | 0.246 |
| short/C16 | 2,496.68 | 53.94 | 27.77 | 1/16 | 0.231 |
| ordinary/C1 | 586.13 | 17.15 | 10.56 | 1/1 | 0.362 |
| ordinary/C4 | 2,883.60 | 24.86 | 18.35 | 4/4 | 0.565 |
| ordinary/C8 | 3,210.49 | 36.08 | 29.77 | 8/8 | 0.941 |
| ordinary/C16 | 3,759.43 | 46.66 | 40.18 | 16/16 | 1.580 |
| RAG/C1 | 993.16 | 19.35 | 13.84 | 1/1 | 0.290 |
| RAG/C4 | 4,154.63 | 30.35 | 23.72 | 4/4 | 0.373 |
| RAG/C8 | 10,650.34 | 27.28 | 20.54 | 8/8 | 0.346 |
| RAG/C16 | 20,288.14 | 30.49 | 23.74 | 12/16 | 0.280 |
| long/C1 | 583.37 | 13.41 | 12.04 | 1/1 | 0.134 |
| long/C4 | 6,908.57 | 21.40 | 19.71 | 1/4 | 0.048 |
| long/C8 | 7,974.54 | 24.23 | 22.55 | 1/8 | 0.045 |
| long/C16 | 8,376.37 | 45.07 | 43.46 | 1/16 | 0.030 |

验收：`status=ok`，16/16 cells、116/116 requests、62 SLO requests，所有 block verifier
和 cleanup 通过，fatal patterns absent，stock UVM restored。peak HBM 为 31,953 MiB
(C1) 和 31,247 MiB (C4/C8/C16)。

差异/限制：文档中的 E2 正式定义要求跨系统 equal-HBM、每 cell 5 个独立进程；本轮只有
ours/Qwen 的一个 16-cell qualification，不能进入跨系统 speedup 主表。

Artifact：
[`artifacts/e2-qwen3-adaptive-concurrency-REPRO-20260827-attempt-01/`](artifacts/e2-qwen3-adaptive-concurrency-REPRO-20260827-attempt-01/)
（`summary.json`、`metrics.csv`、server/UVM/NVML 日志、模块快照）。

## 5. E3：Prefill/Decode crossover

### 5.1 配置与命令

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e3-repro-20260827-attempt-01" \
PROCESS_BLOCKS=1 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e3.sh

workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/verify_formal_e3.py \
  --profile qwen3 "$RUN_ROOT"

python3 experiments/eurosys_moe_offload_evaluation/plot_e3_heatmap.py \
  "$RUN_ROOT/analysis/best_strategy_heatmap.csv" \
  "$RUN_ROOT/analysis/best_strategy_heatmap.svg"
```

同一 480 个 `(layer, expert)` 坐标、2,264,924,160 selected bytes，扫描 H=hybrid、
A=always-copy、R=always-remote，B1/B4/B8。KT exact-coordinate 同时采集；native vLLM
因没有 expert-slice plan interface 写为 N/A。

### 5.2 结果

| workload | B1 | B4 | B8 |
|---|---|---|---|
| short, 128→32 | R, 2,134.76 ms | R, 2,319.25 ms | R, 2,340.53 ms |
| ordinary, 512→128 | R, 8,724.56 ms | R, 8,795.06 ms | R, 9,251.67 ms |
| medium, 2,048→128 | H, 8,716.87 ms | N/A | N/A |
| RAG, 8,192→128 | N/A | N/A | N/A |
| long, 512→512 | R, 34,921.33 ms | R, 34,495.47 ms | H, 35,966.05 ms |

10 个有效 cell 中 R 胜 8 个，H 胜 2 个；H/A/R 对应输出 hash 在每个有效 cell
内一致。5 个 N/A 来自预注册的平台门：单步 Prefill `>=8192` tokens 曾触发 TMA
descriptor 700 / illegal-address，runner 没有静默退回 eager。

![E3 best-strategy heatmap](artifacts/e3-repro-20260827-attempt-01/analysis/best_strategy_heatmap.svg)

验收：verifier `status=ok`，30 custom runs 成功、15 custom N/A、KT 13 measurement
cells；完整 break-even 表在 `analysis/pairwise_break_even.csv`。

差异/限制：仅 1 个独立 process block，`publication_repetition_requirement_met=false`；
这些是描述性点估计，无 CI，不满足每点至少 10 个冷进程的机制实验标准。

Artifact：
[`artifacts/e3-repro-20260827-attempt-01/`](artifacts/e3-repro-20260827-attempt-01/)。

## 6. E4：并发与 PCIe 饱和

### 6.1 配置与命令

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e4-repro-20260827-attempt-01" \
PROCESS_BLOCKS=5 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e4.sh

workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/verify_formal_e4.py "$RUN_ROOT"
```

固定 always-remote exact-coordinate plan；5 个独立进程按 asc/desc/asc/desc/asc 扫描
C1--C32；每档 128 输出 token，placement 为 480 remote experts、
2,264,924,160 B CPU-resident，H2D copy bytes=0。

### 6.2 结果

| C | request TPOT median ms | Decode token/s | PCIe RX GiB/s | GPU util % |
|---:|---:|---:|---:|---:|
| 1 | 69.23 | 14.87 | 0.210 | 11.64 |
| 4 | 70.27 | 58.01 | 0.509 | 16.24 |
| 8 | 71.18 | 114.71 | 1.911 | 19.40 |
| 16 | 71.26 | 229.52 | 3.155 | 23.93 |
| 24 | 68.43 | 359.37 | 4.622 | 26.88 |
| 32 | 67.18 | 484.27 | 6.027 | 29.03 |

预注册判据为：相对各 block 的 C1--C4 pooled median，TPOT `>=1.10x`，至少 4/5
blocks 同意，并持续 3 个并发档。本轮所有并发的 agreement blocks 均为 0，故
`first_sustained_degradation_concurrency=null`，remote-read 验证到 C32。PCIe RX 峰值
为 C32 的 6.027 GiB/s，即 Gen5 x16 理论 58.688 GiB/s 的 10.27%。160 phase
records、5/5 blocks、0 preemption，shutdown 后 driver sessions=0。

CPU DRAM 带宽为 N/A：KVM 不暴露 uncore IMC PMU/内存控制器 PCI 配置，Intel PCM
报告无 memory controller。PCIe RX 只能作为 host-memory traffic 下界。

![E4 concurrency summary](artifacts/e4-repro-20260827-attempt-01/analysis/e4_concurrency_summary.svg)

Artifact：
[`artifacts/e4-repro-20260827-attempt-01/`](artifacts/e4-repro-20260827-attempt-01/)。

## 7. E5：稳定 VA、Graph 与 PTE 消融

### 7.1 dual Graph headroom：失败、修复与复测

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e5-dual-topology-REPRO-20260827-attempt-01" \
KV_CACHE_BYTES=167772160 ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_dual_topology_graph/run_experiment.sh qualify

RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e5-dual-topology-REPRO-20260827-attempt-02" \
KV_CACHE_BYTES=134217728 ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_dual_topology_graph/run_experiment.sh qualify

RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e5-dual-topology-REPRO-20260827-attempt-03" \
KV_CACHE_BYTES=134217728 MAX_NUM_BATCHED_TOKENS=1217 ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_dual_topology_graph/run_experiment.sh qualify

RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e5-dual-topology-REPRO-20260827-attempt-04-formal" \
KV_CACHE_BYTES=134217728 MAX_NUM_BATCHED_TOKENS=1217 \
FORMAL_PERFORMANCE_REPEATS=10 ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_dual_topology_graph/run_experiment.sh formal
```

attempt-01/02 的 B1/B4 都在第二次 Graph residency reconciliation fail-closed：

```text
RuntimeError: Insufficient GPU headroom after expert residency reconciliation:
free_bytes=172752896, required_bytes=201326592
RuntimeError: Original-VA Graph residency reconciliation failed;
the process is poisoned and must be torn down
```

把 KV 从 160 MiB 降到 128 MiB 后数值不变。进一步对比 allocator 时间线发现，四个冻结
capture size 的最大实际 Prefill shape 为 1,217 tokens，但 scheduler 仍按 2,048 tokens
预留 workspace。只把 `max_num_batched_tokens` 收紧为 1,217 后，第一次和第二次
reconciliation 的 free HBM 都精确增加 50,331,648 B；第二次从 172,752,896 B 提高到
223,084,544 B，高于未改动的 201,326,592-B 门 21,757,952 B。四个 capture size、
128-MiB KV、双 576-MiB backing、PARK/RESUME 和安全门均未删减或降低。

attempt-03 qualification 2/2 通过。attempt-04 formal 为 2 个 audit + B1/B4 各 10 个
performance 冷进程，共 22/22；22 个进程的第二次 free/reserved 值都分别为
223,084,544/32,711,376,896 B。Graph capture count、Parameter VA、PTE/residency、两次
PARK/RESUME、输出、fault 和 shutdown gates 全部通过。主要 median 如下：

| batch | Prefill ms | steady Decode ms | cutover ms | RESUME ms | re-PARK ms | cycle E2E ms |
|---:|---:|---:|---:|---:|---:|---:|
| B1 | 69.612 | 10.988 | 49.145 | 39.696 | 48.708 | 383.127 |
| B4 | 84.614 | 21.978 | 49.291 | 41.652 | 48.137 | 621.569 |

### 7.2 eager/Graph remote Decode 正式配对

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e5-phase-switch-REPRO-20260827-attempt-01-qualification" \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh qualify

RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e5-phase-switch-REPRO-20260827-attempt-02-formal" \
FORMAL_PERFORMANCE_REPEATS=10 ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh formal
```

H/G/R/C 在 B1/B4 各有 1 个 audit 和 10 个 performance 冷进程，88/88 成功。同 batch
四臂输出 hash 一致；Parameter VA、2,415,919,104-B remote map、CPU physical residency、
Graph replay、mode-2 fault 和 cleanup 均通过。

| batch | H alias→remote eager | G alias→remote Graph | R remote eager | C remote Graph | G/H steady 改善 | C/R steady 改善 |
|---:|---:|---:|---:|---:|---:|---:|
| B1 | 53.917 ms | 11.269 ms | 54.094 ms | 11.301 ms | 79.10% | 79.11% |
| B4 | 54.957 ms | 22.363 ms | 54.615 ms | 22.358 ms | 59.31% | 59.06% |

H/G 的 cutover median 为 312.102--322.818 ms，因此主 first-Decode/E2E 仍包含拓扑切换
成本；表中的稳态改善只隔离 Decode Graph，不把 cutover 隐藏进稳态。

### 7.3 eager/Graph alias Prefill 正式配对

原 harness 的 formal 只允许每臂 3 个冷进程。为满足计划的 10-process 标准，新增
`FORMAL_PROCESSES_PER_ARM`，verifier 要求 T 恰为 1、Z/A/G 同数且至少 3，并验证 batch
order 平衡。本轮设置为 10，实际 Z/A/G 各 10 个进程，`1,4`/`4,1` 均为 5:5；每进程
1 次 conditioning + 3 次 measurement。

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e5-cold5-prefill-REPRO-20260827-attempt-02-formal" \
FORMAL_PROCESSES_PER_ARM=10 ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_cold5_alias_prefill/run_experiment.sh formal
```

冻结 cold-5 plan 为 48 layers × 5 experts、96 parameters、每 Prefill selected bytes
1,132,462,080 B、backing 47,185,920 B。31/31 cases 通过；同 execution class 输出 hash
一致，route trace、Graph、alias lifecycle、fault 和清理全部通过。

| batch | all-GPU Graph Z | alias eager A | alias Graph G | G 相对 A 改善 |
|---:|---:|---:|---:|---:|
| B1 | 20.537 ms | 82.475 ms | 42.809 ms | 48.09% |
| B4 | 35.335 ms | 79.474 ms | 57.974 ms | 27.05% |

alias Graph 仍比 all-GPU Graph 慢 108.45%/64.07%，这是 offload/alias 数据路径成本，不能
误写成 Graph 回归；同 placement 下 A/G 配对才隔离 Prefill Graph 收益。

### 7.4 DESTROY/rebuild vs PARK/RESUME

补充 formal 每个 batch 各有 10 个独立冷进程。B1/B4 的完整 DESTROY/rebuild median
分别为 3,661.51/3,281.96 ms，PARK/RESUME 为 19.777/19.911 ms；对应 185.14x/
164.83x。PARK 为快速恢复保留 1,207,959,552 B GPU backing。配对的模型、Graph、KV、
专家集合和 retained backing geometry 相同；但 PARK artifact 早于 DESTROY-only 的 CPU
state reconciliation 修复，source hash 和 decode-step count 不完全相同，因此该配对只
用于 transition timing，不比较请求性能。

### 7.5 sequential vs grouped PTE

两臂每个 batch 各 10 个独立冷进程，关键执行输入和 custom module hash 相同。grouped
相对 sequential 的 RESUME wall speedup 为 B1 5.499x、B4 5.473x；完整 cycle E2E
speedup 为 1.087x/1.051x。所有输出、PTE/residency、fault 和 cleanup 门通过。

### 7.6 stable VA vs non-stable VA + recapture

non-stable formal 为 10 个独立冷进程、20 generations；每代 8/8 Parameter VA 改变，
Graph 全量销毁并重捕获，输出与 stable 对照一致，fault/cleanup 均通过。

| batch | stable rebuild+recapture median | non-stable reallocate+recapture median | 额外成本 | 比率 |
|---:|---:|---:|---:|---:|
| B1 | 3,661.51 ms | 7,316.46 ms | 3,654.95 ms | 1.998x |
| B4 | 3,281.96 ms | 5,944.80 ms | 2,662.85 ms | 1.811x |

non-stable 路径额外包含 Parameter 重分配 median 690.08/836.97 ms。该结果量化的是
改变 Parameter VA 后重建资源和 Graph 的总代价，不把删除 stable-VA verifier 断言当作
通过。

Artifacts：
[`attempt-01`](artifacts/e5-dual-topology-REPRO-20260827-attempt-01/)、
[`attempt-02`](artifacts/e5-dual-topology-REPRO-20260827-attempt-02/)、
[`dual qualification`](artifacts/e5-dual-topology-REPRO-20260827-attempt-03/)、
[`dual formal`](artifacts/e5-dual-topology-REPRO-20260827-attempt-04-formal/)、
[`phase-switch formal`](artifacts/e5-phase-switch-REPRO-20260827-attempt-02-formal/)、
[`cold-5 Prefill formal`](artifacts/e5-cold5-prefill-REPRO-20260827-attempt-02-formal/)、
[`DESTROY/PARK comparison`](artifacts/e5-destroy-vs-park-comparison-REPRO-20260828-attempt-01/comparison.json)、
[`PTE comparison`](artifacts/e5-pte-paired-comparison-REPRO-20260828-attempt-01/comparison.json)、
[`stable/non-stable comparison`](artifacts/e5-stable-vs-nonstable-va-REPRO-20260828-attempt-01/comparison.json)。

## 8. E6：Plan Class qualification

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e6-plan-class-REPRO-20260827-attempt-01" \
ALLOW_GPU_RUN=1 \
bash experiments/hybrid_offload_2_4_6_8_compare/run_graph_plan_class_sequence.sh
```

结果 `status=ok`：序列 `C2→C4→C6→C8→C2`，Graph capture count 从模型加载后的 8
保持到最终 8；五次输出 SHA-256 均为
`a365db717e87663b5f807b34fc6013b44f98cd925bea9480cece8e65eac690b9`，Parameter/
upload/source VA 稳定，cleanup errors 为空。

| class | selected bytes | backing bytes | switch ms |
|---|---:|---:|---:|
| C2 | 2,249,195,520 | 173,015,040 | 14,120.16 |
| C4 | 4,498,391,040 | 346,030,080 | 7,443.96 |
| C6 | 6,297,747,456 | 484,442,112 | 6,054.95 |
| C8 | 8,546,942,976 | 657,457,152 | 7,677.95 |
| C2 (return) | 2,249,195,520 | 173,015,040 | 13,309.32 |

该结果只证明固定 class 激活、差集 backing 与 GraphExec 复用。

### 8.1 五档 difference formal

新增 5/10/25/50/100% 对称 difference，10 个独立进程 block 全部返回 0，verifier
`status=ok`。切换延迟随 changed expert slots 单调增加：

| requested diff | 实际 diff | changed slots | outbound median | return median |
|---:|---:|---:|---:|---:|
| 5% | 5.061% | 25 | 480.98 ms | 462.93 ms |
| 10% | 9.919% | 49 | 637.26 ms | 607.75 ms |
| 25% | 25.101% | 124 | 943.25 ms | 934.90 ms |
| 50% | 50.000% | 247 | 1,400.21 ms | 1,421.94 ms |
| 100% | 100.000% | 494 | 2,303.34 ms | 2,391.73 ms |

当前 source/residency 操作为真实 delta，但 REBIND 仍提交完整 alias map；尚未补齐旧 full
plan/session/Graph rebuild、串行逐 wave、真正全层 batched transaction 和
`requests_to_amortize`，故 E6 仍为部分完成。

Artifact：
[`artifacts/e6-plan-class-REPRO-20260827-attempt-01/`](artifacts/e6-plan-class-REPRO-20260827-attempt-01/)。
五档 formal：
[`artifacts/e6-difference-REPRO-20260828-attempt-01-formal/`](artifacts/e6-difference-REPRO-20260828-attempt-01-formal/)。

## 9. E7：热度控制面 microbenchmark

实际命令保存在 artifact：

```bash
cd workloads/vllm/vllm && PYTHONPATH=. .venv/bin/python \
  ../../../experiments/vllm_uvm_adaptive_expert_plan/run_gpu_control_plane.py
```

结果 `ok=true`：128 experts，500 iterations，每轮 65,536 assignments；热度更新
median 277.392 µs、p95 439.168 µs；两轮 observed assignment count 与 expected
完全一致，double-buffer epoch 2→3，5.3-GiB 预算安全向下选择 C4。

限制：这是同步 maintenance/control-plane microbenchmark，不含异步 pinned D2H、
evaluation route、Jaccard、shift 后收敛或端到端性能恢复。

Artifact：
[`artifacts/e7-heat-control-plane-REPRO-20260827-attempt-01/`](artifacts/e7-heat-control-plane-REPRO-20260827-attempt-01/)。

## 10. E8：MoE-Infinity cache/prefetch

```bash
bash experiments/broad_offload_compare/run_moe_infinity_qualification.sh preflight
```

返回码 1：

```text
error: STORE_BUILD_RECEIPT must point to a fresh extension build receipt
```

进一步只读核验：runner 冻结提交
`ba5651897a80d9c9b7a1500cef2c68adaa63db0f`，当前干净的 MoE-Infinity checkout 为
`92c2bc9266e33bbed22405356ca7ca7e71ae8974`；现有约 30-GiB store 只有
`archer_index/archer_param_*`，没有与活动 `_store.so` 对齐的 fresh receipt。未绕过
commit、patch、ELF 或 receipt hash 门，因此没有生成性能数字。

## 11. E9：KTransformers placement/CPU 消融

原缺失 snapshot 已以冻结 reference
`inputs/e9_sharegpt_selection_reference_20260828.json` 补齐。calibration/evaluation source
indices 不相交；calibration 记录 3,896,832 routing slots，并冻结 4,736 GPU experts、
1,408 CPU experts。该主机只有一个 NUMA node，跨 NUMA 对照合法记为 N/A。

实际运行命令：

```bash
ALLOW_GPU_RUN=1 MODE=qualification PROCESS_BLOCKS=1 \
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e9-formal-REPRO-20260828-attempt-01-qualification" \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e9.sh
```

baseline、threads7、threads28、eager、uniform、no-overlap、no-mixed 共 7/7 case 返回 0，
每臂 C1+C8 共 9 个请求均成功输出 32 tokens；per-case runtime verifier、kernel fault 和
cleanup 通过，peak HBM 26,385--26,667 MiB。单轮描述性结果如下，格式为
`throughput tokens/s / mean TTFT ms / mean TPOT ms`：

| arm | C1 | C8 |
|---|---:|---:|
| baseline | 9.714 / 1806.86 / 47.97 | 7.366 / 6737.10 / 902.26 |
| threads7 | 51.991 / 226.37 / 12.53 | 56.892 / 3130.64 / 43.90 |
| threads28 | 4.238 / 4766.99 / 89.80 | 3.505 / 11174.16 / 1989.42 |
| eager | 3.523 / 2146.10 / 223.73 | 7.550 / 6072.65 / 895.67 |
| uniform | 1.017 / 4839.72 / 859.03 | 5.667 / 15934.57 / 936.93 |
| no-overlap | 10.174 / 1557.54 / 51.21 | 9.953 / 6075.32 / 632.15 |
| no-mixed | 12.682 / 1350.65 / 37.82 | 7.470 / 6913.05 / 882.43 |

顶层 verifier 未通过：baseline/threads/eager/no-overlap 的 greedy token IDs 一致；uniform
在 C1 为 0/1、C8 为 2/8 请求 exact-match，no-mixed 在 C1 为 1/1、C8 为 7/8。直接原因
是 placement 或 batch formation 改变 CPU/GPU kernel 与归约形状后，greedy 边界处数值
差异扩散成 token 序列差异；现有 artifact 没有 teacher-forced logits/top-1 证据，不能把
它降级为无害舍入误差，也不能删除 exact gate。故以上性能数字只作诊断，E9 qualification
状态为“未通过”，没有启动 5-block formal。

Artifact：
[`artifacts/e9-formal-REPRO-20260828-attempt-01-qualification/`](artifacts/e9-formal-REPRO-20260828-attempt-01-qualification/)。

## 12. E10：三队列与联合 HBM qualification

### 12.1 三队列 E2E

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e10-three-queue-REPRO-20260827-attempt-01" \
ALLOW_GPU_RUN=1 bash experiments/vllm_uvm_three_queue_e2e/run_experiment.sh qualify

RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e10-three-queue-REPRO-20260827-attempt-02" \
ALLOW_GPU_RUN=1 bash experiments/vllm_uvm_three_queue_e2e/run_experiment.sh qualify
```

attempt-01 在 post-warmup rebalance 因 `unregistered_regions` 失败。最小修复后 attempt-02
`status=ok`：KV target 24→40→20 blocks，8 个 backup 中消费 4 个，迁入 16 个 offline
blocks；每层卸载/恢复 1 个、共 48 experts，226,492,416 B；3 个 Graph，capture count
不增加；Prefill/Decode replay、三队列守恒、roundtrip、shutdown 均通过，kernel faults=0。

### 12.2 dynamic/direct path parity

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e10-path-parity-REPRO-20260827-attempt-01" \
ALLOW_GPU_RUN=1 bash experiments/vllm_uvm_kv_expert_path_parity/run_experiment.sh qualify
```

dynamic 与 direct 的 high/low 最终状态、输出、资源和 VA 等价；无 recapture/fault。
相对 direct：high Prefill/Decode median 差 0.43%/4.05%，low 差 0.26%/5.08%，均在
15%/10% 预注册容差内。主 parity 指标只比较相同稳态，明确排除了 transition、RESUME、
PARK wall time。

### 12.3 2--10 GiB stepped/direct

```bash
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e10-large-kv-REPRO-20260827-attempt-01" \
ALLOW_GPU_RUN=1 bash experiments/vllm_uvm_large_kv_multilevel/run_experiment.sh qualify
```

stepped 序列 `2→4→8→10→8→4→2 GiB` 和 direct 2/4/8/10 GiB 全部通过；对应 remote
experts/layer 为 70/80/99/108。最大稳态 Decode median 差为 descending 8 GiB 的
7.32%，所有比较仍在容差内；输出一致、Parameter/source/KV VA 稳定、Prefill/Decode
Graph replay、PARK/RESUME、热点 GPU 驻留、fault scan 和清理均通过。

Artifacts：
[`three-queue attempt-01`](artifacts/e10-three-queue-REPRO-20260827-attempt-01/)、
[`three-queue attempt-02`](artifacts/e10-three-queue-REPRO-20260827-attempt-02/)、
[`path-parity`](artifacts/e10-path-parity-REPRO-20260827-attempt-01/)、
[`large-kv`](artifacts/e10-large-kv-REPRO-20260827-attempt-01/)。

差异/限制：这三项是 E10 preflight/机制 qualification。尚未运行 fixed K2/K10、oracle、
auto/no-backup、auto+backup、auto+waves、full-joint 的 5-seed open-loop 正式 trace；
不能把人工传入 target 的 stepped/direct 结果称为自动控制器收益。

## 13. E11：phase epoch scheduler

```bash
cd workloads/vllm/vllm
.venv/bin/python -m pytest -q tests/v1/core/test_scheduler.py -k uvm_phase
```

结果：3 passed、76 deselected；GPU after=0 MiB，stock UVM restored。覆盖 chunked
Prefill/Decode 分离、waiting cohort cutover 和 KV-aware FCFS cohort 切分。

限制：没有执行新请求持续 Prefill、旧请求持续 Decode 的多系统 E2E trace；当前合法结论
仅为 phase-batched scheduler qualification，不声称通用 continuous batching。

Artifact：
[`artifacts/e11-phase-epoch-tests-REPRO-20260827-attempt-01/result.json`](artifacts/e11-phase-epoch-tests-REPRO-20260827-attempt-01/result.json)。

## 14. E12：1,000-replay Graph smoke

实际命令完整保存在
`e12-graph-soak-1000-REPRO-20260827-attempt-01.json.command.txt`，关键参数为 4 layers、
576 MiB/layer、2 banks、每 bank 2 consumers、3 warmup + 1,000 measured replays。

```bash
cd workloads/vllm/vllm
env CUDA_VISIBLE_DEVICES=0 taskset -c 0-13 .venv/bin/python \
  ../../../experiments/vllm_p3_static_alias_graph_integration/smoke_static_alias_graph.py \
  --output ../../../experiments/eurosys_moe_offload_evaluation/artifacts/e12-graph-soak-1000-REPRO-20260827-attempt-01.json \
  --manifest ../../../experiments/eurosys_moe_offload_evaluation/artifacts/e12-graph-soak-1000-REPRO-20260827-attempt-01.json.manifest.jsonl \
  --layer-mib 576 --warmup-replays 3 --measured-replays 1000
```

结果 `status=ok`：full-byte compare 0 errors，source mutation 被观察且两代 source/output
hash 分别一致，consumer/upload VA 在 replay 后稳定，allocator closed。GPU time
min/median/mean/max 为 45.8621/45.9045/45.9051/45.9544 ms；host submit median
77.664 µs；每 replay 逻辑 H2D 2,415,919,104 B，对应 49.015 GiB/s。

限制：这只是 Graph copy/replay smoke，不含模型请求、6/24 小时、1,000 次 PARK/RESUME、
K2/K4/K8/K10 峰谷、barrier failure injection 或 custom→stock 生命周期 orchestrator。

Artifacts：
[`result.json`](artifacts/e12-graph-soak-1000-REPRO-20260827-attempt-01.json)、
[`command`](artifacts/e12-graph-soak-1000-REPRO-20260827-attempt-01.json.command.txt)、
[`manifest`](artifacts/e12-graph-soak-1000-REPRO-20260827-attempt-01.json.manifest.jsonl)。

## 15. 修复后的回归验证

E10 修复后执行：

```bash
workloads/vllm/vllm/.venv/bin/python -m pytest -q \
  workloads/vllm/vllm/tests/test_uvm_explicit_expert_regions.py \
  -k 'layer_compiles or layer_hot_set or dynamic_swap'
```

最终结果为 `4 passed, 16 deselected`。同一轮还修正了该测试文件中一个旧
`SimpleNamespace` 夹具缺少 `_uvm_static_alias_owned_experts()` 桩的问题；该修正只影响
测试构造，不改变运行时逻辑。E10 three-queue attempt-02 及其后的 path-parity、large-KV
均在真实 GPU 上通过，构成运行时复测。

E5 修复后另完成：dual Graph 22/22、phase-switch H/G/R/C 88/88、cold-5 Prefill
31/31；对应 formal artifact 中 96/316/99 个 JSON 均可解析，fatal log hits 均为 0。
cold-5 formal 协议修改前后运行 `test_cpu_offload_exact_static.py` 为
`77 passed, 2 warnings`，`bash -n`、`py_compile` 和 `git diff --check` 均通过。三轮结束后
GPU 为 0 MiB、无 compute/UVM client，stock UVM SHA-256 与运行前一致。

## 16. 总体结论

本轮证据确认了四点：

1. E2 的 warmup 后热点 rebalance 和提前 workspace/Graph reserve 在两个 Qwen 进程块中
   都有效，116 个请求没有出现热门专家后续访问失效。
2. 固定 exact-coordinate remote-read 在本机 C1--C32 内未出现预注册的持续 TPOT 退化，
   但 CPU DRAM 计数不可用，不能把原因单独归于 PCIe。
3. static-alias-owned 专家不能同时交给显式 region registry 管理；排除双重 ownership 后，
   三队列、动态/直接路径等价和 2--10 GiB roundtrip 均通过。
4. E5 的三组补充配对均已完成，结果显示 retained topology、grouped PTE 和 stable VA
   分别显著降低切换成本；E6 difference 延迟随改动比例单调增加。E2 跨系统 equal-HBM、
   E3/E6 其余正式机制、E7 shift、E8 provenance、E9 exact-generation、E10 自动 trace、
   E11 mixed batching 和 E12 长 soak/故障注入仍是投稿前硬缺口；在补齐前不能发布总体
   系统 speedup 或完整鲁棒性主张。
