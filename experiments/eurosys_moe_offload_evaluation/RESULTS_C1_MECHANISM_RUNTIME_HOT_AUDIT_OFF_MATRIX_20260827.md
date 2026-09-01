# C1 跨机制、关闭运行时热门专家审计矩阵结果

日期：2026-08-27

## 1. 最终状态

本轮 C1 单轮 feasibility matrix 已完成并暂停，没有后台实验。

| 验收项 | 结果 |
|---|---:|
| 机制 x workload | 6 x 4 |
| 最终成功 cell | 24/24 |
| 正式请求成功 | 24/24 |
| 满足 TTFT <= 2,000 ms 且 TPOT <= 100 ms | 20/24 |
| 最终成功日志 fatal | 0 |
| 总 attempt | 78 |
| 容量/配置边界失败 attempt | 52 |
| GPU/端口/lock cleanup | 通过 |

四个 SLO 失败均为 MoE-Infinity：Short、Ordinary、Long 的 TTFT 通过但 TPOT 超过
100 ms；RAG 的 TTFT 和 TPOT 都超标。`summary.json` 的 `status=ok` 表示请求完成
qualification 和 24-cell 完整性通过，不表示所有 cell 通过 SLO 或 exact-generation。

本轮是单个独立冷进程样本、非 equal-HBM pilot，不能作为论文最终机制排名。

主矩阵结束后的 RAG exact-generation 配对诊断也已完成：ours eager 和“Prefill
eager + Decode-C1 Graph”都与 vLLM all-GPU eager 逐 token 一致；只有同时捕获
Prefill Graph 的 `FULL_SEPARATE` 路径分叉。由此已把 ours/RAG 的分叉定位到
Prefill CUDA Graph 路径，而不是 Decode Graph、cold alias、PARK/RESUME、remote
expert 或 2K chunked-prefill 本身。该诊断不改变上表主矩阵结果。

## 2. 冻结实验条件

- DeepSeek-V2-Lite-Chat BF16，snapshot
  `85864749cd611b4353ce1decdb286193298f64c7`；
- RTX 5090 32,607 MiB，driver 595.71.05；
- C1，每 cell 3 个 calibration warmup + 1 个正式请求；
- Short 128/32、Ordinary 512/128、RAG 8192/128、Long 512/512；
- seed 20260823，prefix cache disabled；
- ours 显式关闭 runtime hot CUDA-read/checksum 和 hot physical-residency audit；
- 启动期 Graph/workspace 预留、热门专家恢复/再平衡、cold residency、stable VA、
  PARK/remote-map 返回值和 fatal 检查仍保留。

详细方法见 `REPRODUCE_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_MATRIX.md`。

## 3. 最终指标

单位均为 ms；HBM 为 NVML 采样峰值 MiB。

| 机制 | workload | 最小可行配置 | TTFT | TPOT | E2E | Peak HBM |
|---|---|---|---:|---:|---:|---:|
| vLLM all-GPU | Short | no offload | 87.40 | 57.62 | 1,873.69 | 30,943 |
| vLLM all-GPU | Ordinary | no offload | 95.96 | 58.24 | 7,492.93 | 31,033 |
| vLLM all-GPU | RAG | 2K x 4 Prefill | 390.40 | 56.43 | 7,556.56 | 32,003 |
| vLLM all-GPU | Long | no offload | 102.11 | 57.19 | 29,327.76 | 31,053 |
| vLLM pinned-UVA | Short | `cpu_offload_gb=0.01` | 96.41 | 59.18 | 1,931.13 | 30,931 |
| vLLM pinned-UVA | Ordinary | `cpu_offload_gb=0.01` | 85.74 | 58.43 | 7,506.81 | 31,025 |
| vLLM pinned-UVA | RAG | `0.01`, 2K x 4 | 465.81 | 57.21 | 7,731.65 | 31,991 |
| vLLM pinned-UVA | Long | `cpu_offload_gb=0.01` | 126.80 | 56.44 | 28,966.38 | 31,045 |
| vLLM copy | Short | `cpu_offload_gb=0.01` | 90.51 | 57.46 | 1,871.82 | 30,963 |
| vLLM copy | Ordinary | `cpu_offload_gb=0.01` | 108.90 | 59.80 | 7,703.35 | 31,015 |
| vLLM copy | RAG | `0.01`, 2K x 4 | 390.71 | 57.71 | 7,720.35 | 32,015 |
| vLLM copy | Long | `cpu_offload_gb=0.01` | 84.58 | 59.17 | 30,320.31 | 31,011 |
| ours | Short | cold1/layer | 86.15 | 8.87 | 361.06 | 30,799 |
| ours | Ordinary | cold1/layer | 93.25 | 7.68 | 1,068.49 | 30,799 |
| ours | RAG | cold1/layer，2K x 4 | 385.00 | 11.74 | 1,875.39 | 31,697 |
| ours | Long | cold1/layer | 98.95 | 7.60 | 3,981.09 | 30,799 |
| KTransformers | Short | 63 GPU/1 CPU expert/layer | 415.96 | 11.84 | 783.03 | 31,051 |
| KTransformers | Ordinary | 63 GPU/1 CPU expert/layer | 430.03 | 8.88 | 1,558.14 | 31,121 |
| KTransformers | RAG | cold1, KV=8322 | 1,335.15 | 10.79 | 2,705.07 | 32,041 |
| KTransformers | Long | cold1, KV=1056 | 503.64 | 9.18 | 5,193.82 | 31,141 |
| MoE-Infinity | Short | device ratio 0.83 | 1,300.40 | 327.16 | 11,442.36 | 31,693 |
| MoE-Infinity | Ordinary | ratio 0.85, 42 KV blocks | 1,078.83 | 315.44 | 41,139.49 | 31,727 |
| MoE-Infinity | RAG | ratio 0.35, 520 blocks | 2,175.97 | 366.22 | 48,685.60 | 32,047 |
| MoE-Infinity | Long | ratio 0.85, 66 blocks | 1,318.99 | 338.67 | 174,380.65 | 31,777 |

## 4. 热门专家审计关闭后的实测效果

Short cold1 与此前 audit-on pilot 的对照：

| 指标 | audit on | audit off | 变化 |
|---|---:|---:|---:|
| phase transition | 426.84 | 37.78 | -389.06 ms / -91.1% |
| 第一个 ITL | 446.55 | 53.77 | -392.79 ms |
| 后续 ITL mean | 7.21 | 7.37 | 基本不变 |
| 汇总 TPOT | 21.38 | 8.87 | -58.5% |
| E2E | 942.24 | 361.06 | -61.7% |

audit-off Short transition 分解为：PARK 32.99 ms、remote map 2.40 ms、其余同步和
cold-only 检查约 2.39 ms。`hot_physical_audit_enabled=false`，`hot_physical=null`。
结果支持“此前巨大切换开销主要来自三轮热门专家同步审计”的判断；稳态 Decode
没有显著变化。

ours 四类请求的 transition 和稳态 Decode：

| workload | transition | first ITL | later ITL mean |
|---|---:|---:|---:|
| Short | 37.78 | 53.77 | 7.37 |
| Ordinary | 37.41 | 54.94 | 7.30 |
| RAG cold1, 2K x 4 | 34.91 | 53.23 | 11.41 |
| Long | 47.39 | 65.75 | 7.48 |

RAG 最初把完整 8K 请求错误地当作单个 Graph/workspace shape，直到 cold4 才可运行；
改为保持请求不变的 2K x 4 后，cold1 即通过，只 PARK/映射 26 个 experts。旧 cold1–cold3
失败保留为“单步 8K shape 容量不足”的边界证据，不再代表最终精确预留配置。

## 5. 横向观察

以 vLLM all-GPU 为参照，ours 的 TPOT 分别快约 6.50x、7.58x、4.81x、7.53x；
E2E 分别快约 5.19x、7.01x、4.03x、7.37x。这些只是单轮系统端到端观察；虽然 RAG
两边都使用 2K x 4，ours RAG 输出 hash 仍不一致，不能把这些数字直接写成最终论文
speedup。

三种原生 vLLM 路径的延迟非常接近。本轮 `cpu_offload_gb=0.01` 只卸载一个完整
Parameter，约 14.25 MiB，不等价于 ours/KT 的每层 cold1，也不是 equal-HBM。

KTransformers 的 TTFT 明显高于 ours，但 TPOT 为 8.88–11.84 ms；其输出 hash 与
原生 vLLM 全部不同，当前只能比较请求完成和测得延迟，不能确认 exact-generation。

MoE-Infinity 四类请求均可完成，但 TPOT 为 315–366 ms，全部未满足 SLO。RAG 的
8K 单步 Prefill 需要把 device ratio 降至 0.35；0.36 即 OOM，边界已按 0.01 封闭。

## 6. exact-generation

- vLLM all-GPU、pinned-UVA、copy：四类请求全部一致；
- ours：Short、Ordinary、Long 一致；最终 RAG 2K x 4 在输出 token 5 首次分叉；
- KTransformers：4/4 与原生 hash 不同；
- MoE-Infinity：4/4 与原生 hash 不同。

因此请求完成 qualification 为 24/24，但 exact-generation gate 尚未全通过。

### 6.1 ours/RAG 配对定位

配对运行保持同一 8192/128 fixture、BF16、cold1、260 MiB KV、2K x 4 Prefill、
greedy/seed 和 PARK/RESUME 机制，只改变 CUDA Graph 覆盖范围：

| ours/RAG 执行模式 | Prefill | Decode-C1 | 输出 hash | 与 all-GPU eager |
|---|---|---|---|---|
| `FULL_SEPARATE`（主矩阵） | Graph | Graph | `db6f9300...` | 不一致，token 5 首次分叉 |
| `NONE` | eager | eager | `d471ac81...` | 完全一致 |
| `FULL_DECODE_ONLY` | eager | Graph | `d471ac81...` | 完全一致 |

Decode-only 诊断 1/1 请求和 128/128 输出 token 均成功，TTFT 1601.47 ms、TPOT
11.81 ms、E2E 3101.93 ms；无 CUDA/PTE/PDE/Xid/fatal，cold 429 MiB 全部 remote，
`hot_physical_audit_enabled=false`，结束后 GPU/端口和 stock UVM 均恢复。eager
诊断为 TTFT 1566.16 ms、TPOT 58.91 ms、E2E 9047.13 ms。两组延迟只用于验证
执行路径与输出，不能和主矩阵的 `FULL_SEPARATE` 性能值互换。

这组最小差分说明 Decode Graph 可以保持正确输出，分叉的必要差异在 Prefill Graph
执行/KV 写入重放路径。当前尚未定位到具体 kernel、buffer 或 KV 页，因此
`FULL_SEPARATE` RAG 仍未通过 exact-generation gate。

## 7. 本轮修复与边界

1. 关闭 ours 正式请求关键路径的 hot CUDA-read 和 hot physical audit；
2. 原生 RAG 8K/4K MLA workspace OOM 后，保持请求不变改为 2K x 4；
3. ours RAG 由单步 8K 改为相同请求的 2K x 4，cold1 首次即成功；
4. KTransformers RAG 的 SGLang 两个内部 token 槽通过 8320→8322 修复；
5. MoE-Infinity 按请求精确配置 12/42/520/66 个 KV blocks；
6. MoE-Infinity 暴露 `max_tokens_per_step`，允许 RAG 8K 请求被 scheduler 接纳；
7. 汇总器只过滤成功 KT 正常 shutdown 的 KeyboardInterrupt traceback。
8. 定向清理三个由早先人工中断遗留、经 stdout FD 追溯到失败 case 的 NVML sampler。
9. 增加 `NONE` 和 `FULL_DECODE_ONLY` 配对模式，用相同 RAG 请求把生成分叉定位到
   Prefill CUDA Graph 路径；诊断不改变主矩阵默认 `FULL_SEPARATE`。

详细记录见 `ERROR_FIX_LOG_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_20260827.md`。

## 8. 证据与清理

机器汇总：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  c1-mechanism-runtime-hot-audit-off-matrix-20260827-attempt-01/
    summary.json
    metrics.csv
    execution.tsv
```

最终检查：24/24 final cells、78 attempts、52 failed boundary attempts；成功日志中
无 fatal。GPU 0 MiB、无 compute process、30401–30406 均空闲、GPU lock 可获取，
stock `nvidia_uvm.ko` srcversion `85A79790636BBD99BA3E43B` 已恢复。

RAG 配对诊断证据：

```text
artifacts/c1-rag-exact-generation-eager-diagnostic-20260827-attempt-03/
artifacts/c1-rag-exact-generation-decode-graph-diagnostic-20260827-attempt-04/
```

Decode-only attempt-04 的 `verification.status=ok`、`failure_count=0`、server/client
return code 均为 0；`module.after.json` 记录 GPU 0 MiB、无 compute/UVM client，且
loaded srcversion 为 stock 版本。

尚未解决的问题见
`UNRESOLVED_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_20260827.md`。
