# Stage M UVM Hybrid Residency Scheduler: Implementation Roadmap and Research Assessment

Date: 2026-05-21

## 1. Executive Summary

本项目的目标不是继续做单点参数调优，而是形成一个系统性方案：

```text
UVM-Aware Hybrid Residency Scheduler
```

核心判断是：

```text
vLLM pure CPU offload 的主要弱点是粗粒度、同步、缺少未来访问预测；
我们的优势应来自应用层语义驱动的细粒度 residency 调度：

scratch: device_direct 绕开 UVM fault
weight: MoE router/top-k 驱动 hot expert prefetch / cold evict
KV: active KV block protection
runtime: application-layer policy runtime
backend: CUDA/UVM 只作为执行后端
```

目前项目已经完成了 scratch device_direct 的关键验证，并开始把分散的 UVM 原型能力收敛到应用层 `uvm_policy_runtime`。下一阶段的重点是验证 KV active protection，并逐步把 MoE weight hotset residency 接入统一 runtime。

## 2. Current Status

### 2.1 已被实验支持的结论

当前最强的实验证据来自 Stage M wide-phase `cuda_malloc_async device_direct`：

| 指标 | baseline trace | device_direct | 变化 |
|---|---:|---:|---:|
| gap2 faults | 496,703 | 703 | -99.86% |
| unknown faults | 550,885 | 1,401 | -99.75% |
| parsed main faults | 1,217,465 | 724,033 | -40.53% |
| output tok/s | 5.49 | 6.06 | +10.38% |
| TPOT | 175.90 ms | 160.33 ms | -8.85% |

这说明：

```text
gap2 基本可以视为 runtime scratch / workspace 热点；
cuda_malloc_async device_direct 能有效把这类 scratch 从 UVM page fault 路径中移走；
scratch 方向已经具备工程可行性和性能收益。
```

但同一组实验也暴露了问题：

| 指标 | baseline trace | device_direct | 变化 |
|---|---:|---:|---:|
| weight faults | 660,775 | 707,309 | +7.04% |
| KV faults | 5,805 | 15,323 | +163.96% |

这说明：

```text
scratch 被解决后，系统压力会转移到 weight 和 KV；
如果没有全局 residency 调度，device_direct scratch 可能间接扰动 KV/weight residency；
下一步必须做 KV active protection 和 MoE weight hotset residency。
```

## 3. 已经实现的内容

### 3.1 UVM execution backend

已有文件：

```text
vllm/vllm/device_allocator/uvm.py
```

已有能力：

```text
prefetch_range_to_device
prefetch_range_to_cpu
batch_prefetch_ranges_to_device
batch_prefetch_ranges_to_cpu
advise_range_preferred_location
batch_advise_ranges_preferred_location
driver_batch_prefetch_ranges_to_device
driver_batch_query_ranges_residency_bytes
batch_prefetch_ranges_to_device_with_executor
batch_offload_ranges_to_cpu_with_executor
```

定位：

```text
这是执行后端，不是策略层。
它负责把应用层策略转成 cudaMemPrefetch / cudaMemAdvise / driver batch 操作。
```

### 3.2 Residency ledger / region registry

已有文件：

```text
vllm/vllm/device_allocator/uvm_pool_residency.py
```

已有能力：

```text
按 pool 注册 region: weights / kv / scratch
记录 ptr / bytes / state / active / evictable
维护 estimated GPU residency
维护 global / pool budget
记录 action trace
支持 real residency sampling
记录 scratch summary
记录 KV sizing / metadata / offload / prefetch-back 事件
```

当前定位：

```text
这是 residency intent 账本。
它不主动迁移内存，也不直接做策略决策。
```

### 3.3 Cross-pool action coordinator

已有文件：

```text
vllm/vllm/device_allocator/uvm_pool_coordinator.py
```

已有能力：

```text
按 pool/action/scope 请求预算
支持 trace_only / enforce
支持 global bytes per step
支持 kv / weights / scratch bytes per step
记录 coordinator trace
记录 pressure
```

当前定位：

```text
这是动作准入层。
它适合作为未来 ActionQueue / BudgetMap 的基础，但还不是完整调度器。
```

### 3.4 Scratch device_direct

已有文件：

```text
vllm/uvm_test/uvm_allocator.cpp
```

已有能力：

```text
device_direct_enable
device_direct_min_bytes / max_bytes / max_total_bytes
device_direct_backend = cuda_malloc / cuda_malloc_async
cuda_malloc_async pool release threshold
scratch_pool_enable
scratch_pool_budget_bytes
scratch_pool_mode = trace_only / enforce
scratch_pool_target_phases
phase/class/gap-watch based eligibility
device_direct telemetry
scratch_pool telemetry
```

已验证效果：

```text
gap2 fault 基本消除；
unknown faults 大幅下降；
TPOT 和 tok/s 有实际改善。
```

当前缺口：

```text
已有 size_bucket telemetry，但尚未实现真正的 size-bucket reusable cache。
短生命周期 scratch 仍可能产生 allocator overhead。
```

### 3.5 MoE weight hot expert prefetch / cold offload prototype

已有文件：

```text
vllm/vllm/model_executor/layers/fused_moe/layer.py
```

已有能力：

```text
MoE router top-k trace
expert weight region id
expert weight ptr/bytes extraction
router_topk / plan_intersection / oracle_replay prefetch source
prefetch max bytes per step
prefetch max experts per layer
prefetch cooldown
prefetch hit trace
driver residency query
batch prefetch executor
cold expert plan-based offload
residency hot-set enforcement prototype
victim selection / recent active protection
```

当前定位：

```text
这是 weight hotset residency 的强原型。
它证明所需语义在 vLLM 应用层可获得。
但实现目前深嵌在 FusedMoE layer 内，后续应逐步迁到 uvm_policy_runtime。
```

### 3.6 KV runtime policy

已有文件：

```text
vllm/vllm/v1/core/uvm_kv_runtime_policy.py
vllm/vllm/v1/core/kv_cache_manager.py
vllm/vllm/v1/core/block_pool.py
```

已有能力：

```text
KV runtime pressure trace
KV admission check
prefix-cache free block candidate trace
safe free-block data offload prototype
prefetch-back before block reuse
Stage L5 KV metadata registration
Stage L6 KV pointer map loading
```

本轮新增能力：

```text
KV active protection MVP
```

新增开关：

```bash
VLLM_UVM_POLICY_RUNTIME_ENABLE
VLLM_UVM_KV_ACTIVE_PROTECT_ENABLE
VLLM_UVM_KV_ACTIVE_PROTECT_MODE
VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE
VLLM_UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS
```

支持模式：

```text
trace_only
advise
prefetch
advise_prefetch
```

调用链：

```text
KVCacheManager.allocate_slots()
  -> _maybe_protect_active_kv_blocks()
  -> UvmKvRuntimePolicy.protect_active_blocks()
  -> UvmPolicyRuntime.protect_kv_region()
```

意义：

```text
active KV blocks 现在可以在 vLLM 应用层被识别、登记、保护；
trace_only 可验证 active set 是否正确；
advise_prefetch 可尝试真实降低 KV faults。
```

### 3.7 Application-layer policy runtime facade

新增文件：

```text
vllm/vllm/device_allocator/uvm_policy_runtime.py
```

当前能力：

```text
UvmPolicyRuntime
protect_kv_region
```

复用：

```text
uvm_pool_residency.py
uvm_pool_coordinator.py
uvm.py
```

当前定位：

```text
这是统一 policy runtime 的第一步。
目前只接了 KV active protection。
后续 weight hotset / cold evict / scratch bucket cache 都应逐步接入这里。
```

### 3.8 Phase 5/6 KV active protection 实验脚本

新增文件：

```text
run_stage_m_kv_active_protection_p10.sh
```

实验 case：

| case | 目的 |
|---|---|
| `device_direct_baseline` | 固定当前最佳 wide-phase device_direct scratch，不启用 KV active protection |
| `kv_active_trace` | Phase 5，验证 active KV block 识别和 telemetry |
| `kv_active_advise_prefetch` | Phase 6，验证 active KV advise+prefetch 是否降低 KV faults |

输出：

```text
kv_active_protection_summary.json
kv_active_protection_summary.csv
kv_active_protection_summary.md
```

关注指标：

```text
KV faults
TPOT
output tok/s
active_protect_attempt_records
active_protect_issued_records
active_protect_fallback_records
kv_pointer_map_entries
```

## 4. 待实现内容

### 4.1 Phase 5: KV active protection trace 实验

目标：

```text
验证 active KV blocks 是否能在 decode path 中被正确识别。
```

需要运行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

RUN_ROOT=log/stage_m_kv_active_protection_p10_run \
PROMPTS=10 \
OUTPUT_LEN=64 \
REQUEST_RATE=1 \
SERVER_MAX_NUM_SEQS=1 \
SERVER_GPU_MEMORY_UTILIZATION=0.70 \
MAX_MODEL_LEN=4096 \
./run_stage_m_kv_active_protection_p10.sh
```

验收标准：

```text
kv_active_trace:
  active_protect_attempt_records > 0
  active_protect_issued_records > 0
  active_kv_protect_issued_lines > 0
  failed_requests = 0
```

注意：

```text
trace_only 不真实迁移，所以不要求 KV faults 下降。
```

### 4.2 Phase 6: KV active protection advise_prefetch 实验

目标：

```text
验证 active KV preferred GPU + prefetch 是否能压低 KV faults。
```

验收标准：

```text
kv_active_advise_prefetch vs device_direct_baseline:
  KV faults 下降，或至少不显著上升
  TPOT 不显著恶化
  output tok/s 不显著下降
  failed_requests = 0
  active_protect_issued_records > 0
  kv_pointer_map_entries > 0
```

如果没有改善，需要检查：

```text
KV pointer map 是否准确
prefetch 是否发生太晚
prefetch 是否和 compute 串行
active set 是否过大
KV faults 是否主要来自 non-active / prefix cache block
```

### 4.3 Scratch size-bucket reusable cache

目标：

```text
在已解决 gap2 fault 的基础上，降低 cuda_malloc_async 频繁分配/释放的 overhead。
```

计划：

```text
size_bucket -> free list
allocate:
  if bucket has cached buffer:
    reuse
  else:
    cudaMallocAsync

free:
  if under cached budget:
    return to bucket
  else:
    cudaFreeAsync
```

新增 telemetry：

```text
scratch_bucket_hit_records
scratch_bucket_miss_records
scratch_cached_bytes
scratch_reuse_count
scratch_cache_evict_count
```

风险：

```text
cache 过大可能挤压 KV / weight；
需要和全局 HBM budget 联动。
```

### 4.4 MoE weight hotset 接入 policy runtime

目标：

```text
将 FusedMoE 内已有 hot expert prefetch / cold expert offload 原型接入 uvm_policy_runtime。
```

第一步不改变行为，只做 facade 化：

```text
FusedMoE:
  collect router top-k / expert ptr / bytes
  call runtime.prefetch_weight_regions(...)

Policy runtime:
  request residency
  request coordinator action
  call UVM backend
  record unified telemetry
```

后续策略：

```text
current layer experts
next layer predicted experts
recently used experts
long-cold experts
budget-based cold evict
```

验收指标：

```text
weight faults 下降
prefetch useful bytes / total prefetch bytes 提升
reaccess-after-evict 降低
KV faults 不因 weight prefetch 上升
TPOT 改善或稳定
```

### 4.5 Conservative cold expert evict

目标：

```text
在 weight_hot_budget 超限时，主动释放 cold expert residency。
```

MVP：

```text
trace-only evict candidates
protect active experts
protect recently evicted/reaccessed experts
只 evict long-cold experts
优先 advise_cpu，再评估 prefetch_cpu
```

风险：

```text
错误 evict 会导致下一步 re-fault；
需要严格统计 reaccess_after_evict。
```

### 4.6 Global HBM BudgetMap

目标：

```text
把 scratch / KV / weight 的预算显式统一。
```

预算公式：

```text
total_hbm_budget
  = weight_hot_budget
  + kv_active_budget
  + scratch_reserved_budget
  + safety_margin
```

初始建议：

```text
scratch_reserved_budget = 1 GiB
kv_active_budget        = 4-6 GiB
safety_margin           = 1-2 GiB
weight_hot_budget       = remaining HBM
```

优先级：

```text
active KV > scratch reserved > hot weights > inactive prefix KV > cold weights
```

### 4.7 Async execution model

目标：

```text
让迁移尽可能 overlap compute，而不是在 fault path 同步发生。
```

计划：

```text
compute_stream:
  current layer compute

prefetch_stream:
  next layer hot weights
  active KV

evict_stream:
  cold weights
  inactive KV, if enabled
```

风险：

```text
如果 prefetch 太晚或 stream 同步点过多，性能可能不升反降。
```

### 4.8 vLLM pure CPU offload 对比实验

最终目标不是只比 pure UVM，而是比 vLLM pure CPU offload。

需要建立 baseline：

```text
same model
same prompts
same output len
same max_model_len
same request_rate
same gpu_memory_utilization
same max_num_seqs
```

核心指标：

```text
TPOT
output tok/s
TTFT
failed requests
GPU replayable faults
weight faults
KV faults
unknown faults
CPU-GPU migration bytes
prefetch useful ratio
```

## 5. 项目创新点

### 5.1 从 fault-driven UVM 转向 semantic-driven UVM

传统 UVM 路径通常是：

```text
GPU access -> page fault -> driver migrates page -> replay
```

本项目的核心创新是把策略上移：

```text
vLLM semantic event -> object-level policy -> async prefetch/evict -> avoid fault
```

也就是说，UVM page migration 仍然是执行后端，但决策单位不是 page，而是：

```text
MoE expert
KV block
scratch buffer
decode phase
request active set
```

这比单纯调 UVM 参数更系统。

### 5.2 应用层 runtime 思路

本项目采用控制面与执行后端分离的系统设计：

```text
把高频、复杂、语义丰富的系统层开销移动到应用层 runtime；
系统层保留执行能力；
应用层负责策略、预测和聚合。
```

在本项目中，对应关系是：

| 层次 | 责任 |
|---|---|
| vLLM semantic layer | router top-k、KV block table、phase、request state |
| policy runtime | hotset、active set、budget、action decision |
| CUDA/UVM backend | prefetch、advise、device allocation |
| UVM driver | page table、fault replay、actual migration |

### 5.3 scratch device_direct 和 UVM residency 调度结合

很多 offload / UVM 工作主要关注 weights 或 KV，但本项目发现 runtime scratch / workspace 也可能形成稳定热 fault gap。

创新点：

```text
将 runtime scratch 从 managed UVM 路径中剥离；
使用 cuda_malloc_async device_direct；
同时把 scratch 预算纳入全局 HBM budget。
```

这不是简单换 allocator，而是：

```text
scratch 不再参与 UVM fault；
weight/KV 通过 residency scheduler 管理；
scratch reserved budget 显式防止挤压 KV/weight。
```

### 5.4 MoE router-driven weight residency

MoE 模型天然有未来访问信号：

```text
router top-k experts
layer id
expert heat
recent active experts
```

本项目计划用这些信号决定：

```text
哪些 expert weight 应 prefetch 到 GPU；
哪些 cold expert 可以 evict / demote；
什么时候迁移可以 overlap compute。
```

这比 pure CPU offload 的“访问时才搬”更主动。

### 5.5 KV active protection

KV cache 的语义单位不是 page，而是：

```text
request -> block table -> active KV blocks
```

本项目实现的 active KV protection 将 active blocks 标记为：

```text
active=True
evictable=False
preferred GPU / prefetch GPU
```

这可以避免 scratch/weight 策略把当前 decode 需要的 KV 随机挤出。

### 5.6 统一 fault/perf 可解释性

项目已经形成较强 telemetry：

```text
fault classification
address region mapping
allocator correlation
gap watch
phase breakdown
residency trace
coordinator trace
active KV telemetry
device_direct telemetry
```

这对论文很重要，因为论文不能只报告 tok/s，还要解释：

```text
为什么 fault 下降；
哪些对象被迁移；
哪些迁移有效；
哪些迁移反而伤害性能。
```

## 6. 是否有希望发表学术会议

### 6.1 初步判断

有希望，但前提是必须从“工程原型”推进到“可复现实验 + 清晰机制贡献”。

当前状态更接近：

```text
有强工程原型；
有初步关键实验结果；
有明确系统设计方向；
但还缺最终闭环实验和对比基线。
```

如果后续能证明：

```text
UVM hybrid scheduler 在真实 MoE inference 上超过 vLLM pure CPU offload；
收益来自 semantic-driven residency，而不是偶然参数组合；
不同 prompt / output_len / request_rate / memory pressure 下趋势稳定；
overhead 可控；
fault/perf 解释闭环完整；
```

则具备投稿系统/机器学习系统方向会议的潜力。

### 6.2 可能适合的会议方向

更可能适合：

```text
MLSys
ASPLOS
OSDI / SOSP, if system contribution is very strong
EuroSys
USENIX ATC
SC
HPCA / MICRO, if memory-system angle足够底层
```

如果论文重点是 LLM serving 系统：

```text
MLSys
OSDI / SOSP
EuroSys
ATC
```

如果重点是 GPU memory / UVM / migration mechanism：

```text
ASPLOS
HPCA
MICRO
SC
```

### 6.3 当前距离论文还缺什么

#### 6.3.1 缺 pure offload 强 baseline

必须正面对比：

```text
vLLM pure CPU offload
vLLM default / no offload
pure UVM
UVM + scratch device_direct
UVM + scratch + KV active protection
UVM + scratch + KV + weight hotset
```

只和 pure UVM 比不够。

#### 6.3.2 缺完整 ablation

需要分解贡献：

```text
scratch device_direct only
KV active protection only
weight hotset only
scratch + KV
scratch + weight
scratch + KV + weight
```

每个组件都要回答：

```text
fault 降了多少？
性能变了多少？
overhead 是多少？
失败场景是什么？
```

#### 6.3.3 缺泛化实验

需要覆盖：

```text
不同 prompt 数
不同 output_len
不同 request_rate
不同 max_num_seqs
不同 gpu_memory_utilization
不同模型或至少不同 MoE workload
不同 HBM pressure
```

如果只在 Qwen3-30B-A3B-FP8、P10 小规模上有效，论文说服力不足。

#### 6.3.4 缺迁移有效性指标

需要新增或稳定统计：

```text
prefetch useful bytes
wrong prefetch bytes
prefetch latency hidden ratio
evict reaccess rate
active KV fault rate before/after
scratch allocator reuse hit rate
HBM budget violation / reject records
```

#### 6.3.5 缺 overhead 分析

需要回答：

```text
policy runtime 本身多慢？
每 step 增加多少 CPU overhead？
prefetch/advise 调用频率是多少？
是否影响 scheduler throughput？
是否增加 CUDA stream sync？
```

### 6.4 论文可能的核心论点

一个较清晰的论文主张可以是：

```text
LLM inference memory management should not treat UVM page faults as the normal control path.
By lifting residency policy to the application layer and using model/request semantics,
we can convert reactive page migration into predictive object-level migration.
```

中文表达：

```text
LLM 推理中的 UVM 不应以 page fault 作为主要控制路径。
利用 vLLM 应用层语义，可以将被动 page migration 转成主动 object-level residency scheduling。
```

可能的贡献点：

1. 提出 UVM-Aware Hybrid Residency Scheduler。
2. 证明 runtime scratch fault gap 可通过 device_direct 消除。
3. 提出 MoE router-driven expert weight residency。
4. 提出 request-level active KV protection。
5. 构建统一 telemetry，解释 fault/perf 变化。
6. 与 vLLM pure CPU offload 做端到端对比。

### 6.5 投稿风险

主要风险：

```text
如果最终只在单机单模型单配置上提升，贡献会显得像工程调参。
如果 weight hotset 无法显著降低 fault 或隐藏迁移，核心优势不够强。
如果 KV active protection 的 prefetch overhead 抵消收益，系统故事会变弱。
如果无法超过 pure CPU offload，只能定位为 UVM profiling / diagnostic work。
```

规避方式：

```text
把对比目标锁定为 vLLM pure CPU offload；
形成清晰 ablation；
展示不同 workload 的稳定趋势；
用 telemetry 证明机制有效；
把失败场景也系统分析出来。
```

## 7. Recommended Next Steps

### 7.1 近期工程顺序

建议优先级：

| 优先级 | 任务 | 原因 |
|---:|---|---|
| 1 | 跑 Phase 5 KV active trace | 验证新接入路径是否正确 |
| 2 | 跑 Phase 6 KV active advise_prefetch | 看 KV faults 是否能被压住 |
| 3 | 实现 scratch size-bucket reusable cache | device_direct 已有效，下一步降 allocator overhead |
| 4 | 将 MoE hot expert prefetch 接入 policy runtime | weight 是超过 pure offload 的核心 |
| 5 | 建立 vLLM pure CPU offload baseline | 明确最终目标 |
| 6 | 做 scratch/KV/weight ablation | 为论文和工程决策提供证据 |

### 7.2 近期实验顺序

建议实验：

```text
E1: device_direct baseline vs kv_active_trace vs kv_active_advise_prefetch
E2: scratch bucket cache on/off
E3: weight hotset trace_only
E4: weight hotset prefetch enforce
E5: conservative cold evict trace_only
E6: full scheduler vs vLLM pure CPU offload
```

### 7.3 成功标准

短期成功：

```text
Phase 5:
  active KV telemetry 正确；
  没有 failed requests；
  overhead 很低。

Phase 6:
  KV faults 不再因 device_direct scratch 反弹；
  TPOT 不恶化。
```

中期成功：

```text
weight faults 下降；
unknown faults 接近 0；
KV faults 稳定；
TPOT / tok/s 优于 current device_direct-only baseline。
```

最终成功：

```text
TPOT < vLLM pure CPU offload
output tok/s > vLLM pure CPU offload
failed requests = 0
fault/perf 改善可由 telemetry 解释
```

## 8. Bottom Line

当前项目已经从“UVM fault profiling”进入“应用层语义驱动 residency scheduler”的实现阶段。

最扎实的结果是：

```text
scratch / gap2 fault 可以通过 cuda_malloc_async device_direct 基本解决，
且已经带来 TPOT / tok/s 改善。
```

最关键的未完成部分是：

```text
KV active protection 是否能稳定压住 KV fault 反弹；
MoE hot expert residency 是否能降低 weight fault 并隐藏迁移；
全局预算是否能避免 scratch / KV / weight 互相挤压。
```

学术上，这个方向有潜力，但最终能否投稿强会议，取决于是否能完成：

```text
超过 vLLM pure CPU offload 的端到端结果
+ 清晰 ablation
+ 多 workload 泛化
+ 可解释 telemetry
+ runtime overhead 分析
```

如果这些证据补齐，该项目可以形成一个有竞争力的系统论文故事：

```text
From reactive UVM page faults to semantic-driven object-level residency scheduling for LLM inference.
```
