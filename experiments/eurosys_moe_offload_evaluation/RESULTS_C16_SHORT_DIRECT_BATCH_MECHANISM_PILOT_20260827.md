# Short-dialogue C16 无 HTTP 单批跨机制 Pilot

日期：2026-08-27
主 artifact：`artifacts/c16-short-direct-batch-pilot-20260827-attempt-04`

## 1. 结论

本轮按一个 `1 x 16` 批次重跑 Short-dialogue C16。正式计时前，16 个
128-token prompt 已全部进入引擎；没有 HTTP/client，也没有 `1+15` 到达偏斜。
六个机制最终都完成 16/16 个请求，结构性 qualification 通过。

vLLM all-GPU、pinned-UVA、copy-on-forward 和 ours 的 16 组输出 token hash
逐请求完全相同。KTransformers 只有 5/16 与基线相同，MoE-Infinity 为 6/16，
所以后二者仍未通过跨框架 exact-generation gate。

这是单轮、非 equal-HBM pilot。它能说明 direct-batch 可行性、容量边界和单轮性能，
不能直接作为论文最终排名。

## 2. 请求与计时口径

- 模型：DeepSeek-V2-Lite-Chat BF16，snapshot
  `85864749cd611b4353ce1decdb286193298f64c7`。
- workload：冻结 Short-dialogue evaluation 的前 16 条请求。
- 每条请求：128 prompt tokens、32 output tokens；贪心生成、忽略 EOS。
- warmup：冻结 calibration 的 3 条请求，每条输出 8 tokens。
- CPU：`taskset -c 0-13`，14 threads；GPU：RTX 5090 32 GiB，TP=1。
- 计时起点：完整 16 条请求入队之后、第一次 engine/scheduler step 之前。
- SLO：TTFT <= 2000 ms 且 TPOT <= 100 ms。

vLLM/ours/MoE-Infinity 使用 16 次内存内 `add_request` 后才调用首个 step。
KTransformers 使用一次 `Engine.generate(input_ids=list[list[int]])`，SGLang 日志确认：

```text
Prefill batch, #new-seq: 16, #new-token: 2048, #running-req: 0
```

这里 `#running-req: 0` 表示该 Prefill 之前没有旧请求；16 条新请求由
`#new-seq: 16` 证明，并不表示批次为空。

## 3. 最终结果

下表为 16 个请求的均值。因为同一个 scheduler step 同时产出每条请求的 token，
vLLM direct runner 中各请求的计时相同；原始 JSON 同时保留 min/p50/p95/p99/max。

| 机制 | 最小/有效配置 | 成功 | TTFT ms | TPOT ms | E2E ms | SLO | Peak HBM MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM all-GPU | 不卸载 | 16/16 | 68.74 | 53.72 | 1733.91 | 16/16 | 31,255 |
| vLLM pinned-UVA | `cpu-offload-gb=0.01` | 16/16 | 89.45 | 53.23 | 1739.54 | 16/16 | 31,233 |
| vLLM copy-on-forward | `cpu-offload-gb=0.01` | 16/16 | 66.17 | 53.23 | 1716.33 | 16/16 | 31,225 |
| ours | 每层 cold1，共 449,839,104 B | 16/16 | 83.52 | 19.34 | 683.18 | 16/16 | 31,187 |
| KTransformers | 每层 63 GPU + 1 CPU expert | 16/16 | 1124.89 | 397.90 | 13459.89 | 0/16 | 31,473 |
| MoE-Infinity | global cache；device ratio 0.80 | 16/16 | 13037.28 | 1929.60 | 72854.78 | 0/16 | 32,109 |

相对 all-GPU，ours 的 TTFT 高 21.5%，但 TPOT 降低 64.0%（2.78x），E2E
降低 60.6%（2.54x）。这是本次单轮 direct-engine 口径，不应与此前 HTTP 数据做
绝对比较。

KTransformers 和 MoE-Infinity 的请求都完成，但高并发 TPOT 超过 SLO。
KTransformers 的 Decode 日志约为 39--43 aggregate token/s；MoE-Infinity 的
non-paged fallback 逐序列执行，C16 E2E 约 72.85 s。

## 4. 空间与 Graph 配置

### 4.1 ours

- 每个 26 个 MoE 层卸载一个 calibration-cold expert，共 26 experts、
  449,839,104 B。
- KV 固定 100,663,296 B，可容纳 3,232 tokens；需求上界按
  `16 x 192 = 3,072` tokens 预留。
- CUDA Graph：Decode B1/B4/B8/B16，加一个 pure-Prefill 2,048-token Graph。
- pre-KV hole：100,663,296 B，在 KV 初始化前建立、Prefill capture 前释放。
- allocator/workspace warmup：2,048 tokens；运行时 hot-residency 全量审计关闭；
  cold residency 检查、PARK/RESUME 和 remote mapping 保留。
- 正式批次切换记录约 10.56 ms，其中 PARK 约 6.92 ms、remote map 约
  1.89 ms；没有只读 PTE fence。
- 16/16 请求首步为同一个 Prefill cohort，正式阶段无 HTTP 到达偏斜。

### 4.2 KTransformers

- 由同一 cold1 frequency plan 转换：每层 63 GPU experts、1 CPU expert。
- `context-length=194`、`max-total-tokens=3104`、`max-prefill-tokens=2048`。
- Decode Graph batch sizes 为 1/4/8/16；Prefill eager。

### 4.3 MoE-Infinity

- `device_memory_ratio=0.80`。
- 为保持绝对 KV 容量不变，`kv_cache_ratio=0.0145625`；实测
  339,762,207 B、192 blocks，即 3,072 tokens。
- 实测 expert cache budget 为 22,991,548,181 B；activation reserve ratio 0.10。
- 结束时 192/192 KV blocks 全部归还，pending/cancelled 都为 0。
- PagedAttention/FlashAttention 在该环境不可用，使用 torch SDPA 和 non-paged
  per-sequence HF cache；不能把该结果解释为优化后的 MoE-Infinity 上限。

## 5. 错误、根因与复测

### 5.1 KTransformers NUMA 类型

首次 direct runner 把 `kt_numa_nodes` 设成整数 `0`；CLI 通常会把它规范化为
`[0]`，但直接构造 `ServerArgs` 不会。`kt_kernel` 调用 `len(numa_nodes)`，因此
启动时报 `TypeError: object of type 'int' has no len()`。

修复为 `kt_numa_nodes=[0]` 后，计算 return code 0、16/16 成功。

### 5.2 KTransformers scheduler 验证器

旧验证器错误要求同一 Prefill 日志的 `#running-req=16`。当前 SGLang 中该字段是
本步之前已运行的旧请求，单个全新 C16 cohort 应为 0。验证器改为同时要求
`new-seq=16,new-token=2048,running-req=0`，已有计算结果复验通过。

### 5.3 MoE-Infinity C16 logits 峰值

原始 `device ratio=0.83` 在 C16 Prefill 因 50 MiB FP32 logits 分配 OOM。
比例 0.829/0.827 仍因动态专家缓存和 activation 峰值失败。比例 0.80 时，16 个
独立 128-token forward 均完成，但旧 non-paged runner 保留 16 份约 50 MiB logits，
随后 `torch.cat` 再申请 800 MiB，形成约 1.6 GiB 重复峰值并 OOM。

最小代码修复在 `MoE-Infinity/moe_infinity/serving/model_runner.py`：预分配一次
最终 packed tensor，并逐序列写入切片；输出 shape/dtype/device 和数值契约不变。
相关现有单测 5/5 通过。相同 `ratio=0.80` 复测 16/16 成功，无 OOM/fatal。

失败尝试完整保留于主 artifact 的 `retries/`，没有覆盖成功结果。

## 6. 正确性与限制

- all-GPU、pinned-UVA、copy 和 ours：16/16 output hash 相同。
- KTransformers：5/16 与 all-GPU 相同。
- MoE-Infinity：6/16 与 all-GPU 相同。
- 后两者的分叉根因尚未定位；只能报告请求完成和性能指标，不可宣称 exact
  generation 等价。
- 本轮每个最终 arm 只有一个冷进程，没有方差/置信区间。
- Peak HBM 为 31,187--32,109 MiB，且各机制预算语义不同，不是 equal-HBM。
- ours、KTransformers 和 MoE-Infinity 的 Graph/attention 能力不同。

## 7. 运行后状态

- GPU used memory：0 MiB；无 compute process。
- stock `nvidia_uvm` 已加载；custom UVM 未残留。
- C1/E2 两把 GPU lock 均可重新获取。
- 没有后台实验。

机器汇总见主 artifact 的 `summary.json`。详细复现命令和故障排查见
`REPRODUCE_C16_SHORT_DIRECT_BATCH_MECHANISM_PILOT.md`。
