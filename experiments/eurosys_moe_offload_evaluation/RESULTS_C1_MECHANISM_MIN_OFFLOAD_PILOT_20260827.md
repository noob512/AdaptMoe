# Short-dialogue C1 跨机制最小卸载可行性 Pilot

日期：2026-08-27
Artifact：`artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01`

## 1. 结论

本轮统一 Short-dialogue C1 请求后，以下配置均能完成 3 个 calibration
warmup 和 1 个正式请求：vLLM all-GPU、vLLM pinned UVA、vLLM
copy-on-forward、ours cold1、KTransformers cold1，以及 MoE-Infinity
`device-memory-ratio=0.83`。

ours 和 KTransformers 已把固定专家卸载量降到当前非空最小粒度：26 个 MoE
层各卸载 1 个相同的 calibration-cold expert，共 26 个专家、449,839,104 B。
MoE-Infinity 使用全局 demand cache，没有“每层 cold-N”语义；本轮以 0.01 ratio
粒度确定 0.83 成功、0.84 失败。

这是一轮 feasibility/maximum-resident pilot，不是正式 equal-HBM E2 排名：各机制
的峰值 HBM 不相等，执行模式也不同，且每个 arm 只有一个正式请求。

## 2. 冻结请求与资源边界

- 模型：DeepSeek-V2-Lite-Chat BF16，snapshot
  `85864749cd611b4353ce1decdb286193298f64c7`。
- 请求：冻结 evaluation request 0，128 prompt tokens、32 output tokens、C1。
- warmup：冻结 calibration fixture 的 3 个请求，每个输出 8 tokens。
- seed：20260823；CPU：`taskset -c 0-13`，14 threads。
- 客户端：所有 arm 都使用同一个 HTTP/SSE client 和同一个冻结 workload config。
- vLLM KV：6,291,456 B，实际 192 tokens；足以容纳 160-token 请求并保留
  两个 16-token blocks。
- ours Graph：只捕获 Decode batch 1 和 pure-Prefill 128 tokens；不捕获
  C4/C8/C16 或长上下文 Graph。
- ours workspace warmup：128 tokens；fused-MoE chunk size 128。
- KTransformers：`max-total-tokens=192`、Decode Graph batch 1。

所有正式请求的 prompt-token-ID SHA256 均为
`eb9a77425f9afefa503200bd45a694762c265707a3255a0c6fcbaceb15c73d5e`。

## 3. 结果

| Arm | 最小卸载/缓存配置 | 成功 | TTFT ms | TPOT ms | E2E ms | SLO | Peak HBM MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM all-GPU | 无卸载 | 1/1 | 100.43 | 56.95 | 1865.97 | 1/1 | 30,943 |
| vLLM pinned UVA | `cpu-offload-gb=0.01`；实际 14,942,208 B | 1/1 | 100.80 | 56.00 | 1836.76 | 1/1 | 30,931 |
| vLLM copy-on-forward | `cpu-offload-gb=0.01`；实际 14,942,208 B | 1/1 | 97.43 | 59.45 | 1940.35 | 1/1 | 30,963 |
| ours | cold1；1 expert/layer，449,839,104 B | 1/1 | 279.53 | 21.38 | 942.24 | 1/1 | 30,799 |
| KTransformers | 63 GPU + 1 CPU expert/layer，449,839,104 B | 1/1 | 370.55 | 9.58 | 667.66 | 1/1 | 31,051 |
| MoE-Infinity | global demand cache，device ratio 0.83 | 1/1 | 938.07 | 289.99 | 9927.76 | 0/1 | 31,925 |

SLO 使用 TTFT <= 2000 ms 且 TPOT <= 100 ms。MoE-Infinity 请求成功，但未满足
TPOT SLO。

vLLM all-GPU、pinned UVA、copy-on-forward 和 ours 的正式输出 token hash
完全相同。KTransformers 和 MoE-Infinity 的输出 hash 分别不同，因此本轮不能宣称
跨框架 exact-generation 等价；它们只通过了请求成功、token 数量和指标完整性检查。

## 4. 最小粒度与空间预留验证

### 4.1 原生 vLLM

原生 `cpu_offload_gb` 按完整 Parameter tensor 选择，不支持每层一个 expert slice。
最小正预算 0.01 GiB 实际选择两个 layer-0 attention tensor：

- `kv_a_proj_with_mqa.weight`：2,359,296 B；
- `q_proj.weight`：12,582,912 B；
- 合计：14,942,208 B。

因此这两个 arm 只能称为“最小正整-tensor卸载”，不能称为 cold1 expert arm。

### 4.2 ours

cold1 plan 由 measurement 前冻结的 calibration routing trace 生成，26 层各取
访问频率最低的一个 expert。运行证据：

- Decode batch-1 Graph 和 pure-Prefill 128-token Graph 均捕获成功；
- 6 MiB pre-KV hole 成功分配并在 pure-Prefill capture 前释放；
- 128-token allocator reserve 在 all-remote 初始化阶段保留 12,582,912 B；
- 初始化/sampler warmup 后恢复 serving residency：cold 449,839,104 B，
  hot 28,339,863,552 B，free 1,529,610,240 B；
- 恢复前后 allocator reserved 都是 31,727,812,608 B，没有因恢复热门专家破坏
  已建立的 Graph/workspace pool；
- 正式请求的 PARK、remote map 和 RESUME 均成功；正式 Decode 时 cold bytes
  全部在 CPU、hot bytes 全部在 GPU，unknown bytes 为 0。

这实现了本轮要求的“先为确定的 C1 shape 建立 KV/Graph/workspace，再恢复热门专家”，
没有沿用完整 E2 矩阵的 2 GiB KV 和全 Graph 集，也没有依赖正式请求期间的临时大分配。

### 4.3 KTransformers

KTransformers 使用由同一个 cold1 plan 转换出的 frequency mask。日志确认 26 个
MoE 层均为 63 个 GPU experts、1 个 CPU expert，共 1,638 个 GPU experts 和 26 个
CPU experts。Prefill 为 eager，Decode batch 1 使用 CUDA Graph。

### 4.4 MoE-Infinity

MoE-Infinity 没有固定逐层 cold plan。本轮 0.83 的 serving admin budget 为：

- expert cache：23,964,172,183 B；
- KV cache：242,062,345 B，136 blocks；
- activation reserve ratio：0.10；
- 正式结束后 KV 136/136 blocks 全部归还。

边界尝试：

| device ratio | 结果 | 失败位置 |
|---:|---|---|
| 0.80 | 成功 | -- |
| 0.83 | 成功 | -- |
| 0.84 | 失败 | warmup-1 demand cache `cudaMalloc` OOM |
| 0.85 | 失败 | warmup-1 demand cache `cudaMalloc` OOM |
| 0.90 | 失败 | warmup-1 demand cache `cudaMalloc` OOM |

0.84/0.85/0.90 均先达到 healthy，warmup-0 成功，然后第二个 warmup 引入新的
专家集合时在 `device_caching_allocator.cpp:28` 触发 CUDA OOM。降低 device ratio
增加可驱逐/卸载余量后，0.83 完整通过；请求、KV ratio 和模型未改变。

## 5. MoE-Infinity 构建修复

旧 `_store` 缺少与当前 tree 对应的 fresh receipt。本轮没有绕过 provenance gate，
而是用固定重建脚本在 commit
`92c2bc9266e33bbed22405356ca7ca7e71ae8974` 上重新构建 SM120 `_store`。
receipt 状态为 complete，active extension SHA256 为
`3946666dde229528870cde08989e0681ce2f9f35becf4d8a2e5ca8dfa0a30f3b`。

## 6. 验收与限制

- 六个最终 arm 的 server readiness、请求成功、指标完整性、fatal-log 和 GPU cleanup
  检查均通过。
- 实验结束 GPU memory 为 0 MiB，无 compute process/监听端口，stock UVM srcversion
  `85A79790636BBD99BA3E43B` 已恢复。
- 当前数据只有一个正式请求，不应用于统计显著性或稳定排名。
- peak HBM 不相等，不能直接作为 equal-HBM speedup 表。
- vLLM native arms 是 eager；ours 使用 Prefill/Decode FULL_SEPARATE Graph；
  KTransformers 只在 Decode 使用 Graph；MoE-Infinity 回退到 torch SDPA。
- 下一步若进入论文正式矩阵，应先决定是否统一 Graph/eager 能力口径，然后在每个
  机制的实际峰值 HBM 上配平，并运行独立冷进程重复。

## 7. 关键证据

- 统一 workload：`configs/c1_short_mechanism_pilot.json`
- cold1 plan：`artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/cold1_plan/cold_plan_c1.json`
- KT mask：`artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/kt_cold1_placement/kt_frequency_2g.pt`
- MoE-Infinity receipt：`artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/moe_infinity_store_build/store_build_receipt.json`
- 每个 arm 的实际命令、环境、server/client log、NVML、verification 均位于对应
  artifact 子目录。

## 8. 复现与 phase-transition 根因补充

详细逐 arm 复现步骤、custom UVM 生命周期、验收命令、常见故障，以及 `ours`
首次 Decode 约 426.84 ms 切换开销的分解见：

```text
REPRODUCE_C1_MECHANISM_MIN_OFFLOAD_PILOT.md
```

需要更正一个容易误读的计时口径：`alias_detach_ns=289.43 ms` 并不是 PARK ioctl
本身。PARK 实测为 38.45 ms，remote map 为 2.20 ms；约 386.09 ms、即完整
transition 的 90.5%，来自 PARK 前后和 remote-map 后对 28.34 GiB 热专家执行的
重复 residency/CUDA-read 正确性审计。该诊断解释了首个 ITL，但不改变本表原始
端到端指标。
