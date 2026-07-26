# Stage M UVM-Aware Hybrid Residency Scheduler Design

Date: 2026-05-21

## 0. Executive Summary

本方案的目标不是继续围绕单个参数做调参，而是形成一个有机会在性能上超过 vLLM 纯 CPU offload 的系统性内存管理方案。

核心判断：

```text
vLLM pure CPU offload:
  静态 / 粗粒度 / 需求触发 / 容易同步阻塞

UVM-Aware Hybrid Residency Scheduler:
  动态 / 对象级 / 预测触发 / 尽量异步 overlap
```

本方案不把 UVM page fault 当作正常数据路径，而是把 UVM 作为可迁移统一地址空间，把 CUDA/UVM 迁移接口作为执行后端。真正的策略应运行在 vLLM 应用层，利用模型语义做 object-level residency scheduling。

一句话原则：

```text
UVM page migration is an execution backend, not the scheduling policy.
```

中文表述：

```text
页迁移只是执行后端，不是调度策略。
所有可预测访问都应该在 fault 发生前，被转成对象级异步 prefetch / evict。
```

最终架构：

| 层级 | 职责 |
|---|---|
| vLLM semantic layer | 提供 MoE router top-k、layer、decode step、KV block、tensor region 等语义 |
| application-layer policy runtime | 维护 hot expert、active KV、scratch pool、预算账本和迁移动作队列 |
| CUDA/UVM execution backend | 执行 `cudaMemPrefetchAsync`、`cudaMemAdvise`、`cudaMallocAsync`、fault telemetry |
| driver/system layer | 继续负责页表、实际 page migration、fault replay、低层计数 |

这一路线借鉴 bpftime 的思想：复杂、高频、语义丰富的策略不要下沉到系统层，而应尽量上移到应用层或用户态 runtime；系统层只提供必要的执行能力和观测接口。

## 1. 背景

当前实验已经完成了多轮 UVM fault profiling、gap-watch、device_direct scratch、phase matrix 和 fault/perf A/B。

关键现象有三类：

1. pure UVM 下主要 fault 来自 weight 和 unknown runtime scratch。
2. gap2 是稳定、热、短生命周期的 unknown gap，基本可以判断为 runtime scratch / workspace。
3. `cuda_malloc_async + wide decode phases + device_direct` 可以基本消除 gap2 fault，并带来实际 TPOT 改善。

pure UVM `PROMPTS=10` fault profile：

```text
parsed address faults = 1,518,179
weight faults         =   931,079  (61.33%)
unknown faults        =   582,196  (38.35%)
kv_cache faults       =     4,904  (0.32%)
gap2 faults           =   481,363
gap2 / unknown        =    82.68%
gap2 / parsed         =    31.71%
```

`cuda_malloc_async + wide decode phases` A/B：

```text
gap2 faults          496,703 -> 703       (-99.86%)
unknown faults       550,885 -> 1,401     (-99.75%)
parsed main faults   1,217,465 -> 724,033 (-40.53%)
raw replayable       4,513,473 -> 2,414,081 (-46.51%)
dedup replayable     2,376,355 -> 1,721,267 (-27.57%)
output tok/s         5.49 -> 6.06 (+10.38%)
TPOT                 175.90 ms -> 160.33 ms (-8.85%)
```

同时也观察到副作用：

```text
weight faults 660,775 -> 707,309 (+7.04%)
KV faults       5,805 ->  15,323 (+163.96%)
```

这说明 scratch 路线已经有效，但还不够。要超过 vLLM pure offload，必须继续解决 weight residency 和 KV active protection。

## 2. 目标和非目标

### 2.1 目标

本阶段目标：

1. 形成一个能和 vLLM pure CPU offload 正面对比的 UVM hybrid memory scheduler。
2. scratch 不再走 UVM page fault。
3. weight 不再完全依赖 fault-driven 迁移，而是利用 MoE router 做 hot expert prefetch。
4. KV active blocks 不被 scratch / weight 随机挤出。
5. 所有可预测迁移尽量提前发生，且尽量与 compute overlap。
6. 保留 trace-only / enforce 双模式，使每个策略都可观测、可回退。

成功标准不是单个 fault 指标归零，而是：

```text
TPOT < vLLM pure CPU offload TPOT
output tok/s > vLLM pure CPU offload tok/s
failed requests = 0
unknown faults 接近 0
weight faults 明显低于 pure UVM / pure offload 的同步等待成本
KV faults 不显著上升
```

### 2.2 非目标

本阶段明确不做：

1. 不以继续扫描 `TARGET_PHASES`、`DEVICE_DIRECT_MAX_BYTES` 为主线。
2. 不把所有策略下沉到 UVM driver。
3. 不重写 UVM 底层 page migration 作为 MVP。
4. 不追求所有 weight 全量 GPU resident。
5. 不把 runtime scratch 全部粗暴改成普通 `cudaMalloc`。
6. 不用单次 fault 总数代替端到端性能判断。

## 3. 为什么不能只依赖 UVM page fault

UVM 的页迁移机制有价值，但它不符合 inference 的策略语义。

### 3.1 页适合作为执行单位，不适合作为决策单位

Inference 的真实访问单位不是 page，而是：

```text
MoE:
  layer -> expert -> w13 / w2 tensor

KV:
  request -> block table -> active KV blocks

scratch:
  operator temporary buffer -> size bucket -> lifetime

decode:
  current layer -> next layer -> next token
```

因此，页粒度适合作为 driver 的最终搬运粒度，但不适合作为 scheduler 的调度粒度。

| 对象 | 页级 fault 是否适合 | 应使用的调度粒度 |
|---|---|---|
| runtime scratch / gap2 | 不适合 | scratch buffer / size bucket |
| MoE expert weights | fault 粒度过晚 | expert tensor region |
| dense weights | 可批量预取 | layer / tensor region |
| KV cache | page 过细 | KV block / request active set |
| temporary activations | 通常不应 managed fault | device local buffer |

### 3.2 fault-driven 太晚

GPU page fault 的触发点已经在 compute critical path 上。对 decode inference 来说，TPOT 对每一步阻塞非常敏感。即使 UVM 能正确迁移页面，只要迁移发生在 fault 之后，就已经晚了。

因此本方案的目标不是“让 fault 更快”，而是：

```text
avoid predictable faults before they happen
```

### 3.3 是否需要改 UVM 底层

需要，但不是第一阶段的主路径。

第一阶段应优先做应用层对象级 scheduler，因为：

1. 当前最大收益来自避免 fault，而不是优化 fault handler。
2. CUDA 已经提供可用的执行接口：`cudaMemPrefetchAsync`、`cudaMemAdvise`、`cudaMallocAsync`。
3. vLLM 应用层拥有 driver 无法知道的 MoE router、KV block、decode step 语义。
4. driver 修改风险高、验证慢，而且容易把策略写死在错误层级。

后续若应用层 scheduler 已证明有效，再考虑 UVM 底层增强：

```text
fault batching / coalescing
region-level migration hint
per-allocation residency priority
active KV page protection
large-page / contiguous-range prefetch
migration stream priority
page fault fast path
```

## 4. 应用层策略运行时

这里采用控制面与执行后端分离的系统设计：

```text
将高频、复杂、可在用户态完成的系统层开销移到应用层 / 用户态 runtime。
内核或系统层保留必要的执行与隔离能力。
```

对应到本项目：

```text
不要让 UVM driver 猜测模型语义。
不要让每次内存事件都走重系统路径。
应该在 vLLM 进程内维护策略状态、预算账本和动作队列。
```

### 4.1 控制面和数据面分离

| 平面 | 位置 | 职责 |
|---|---|---|
| semantic control plane | vLLM Python/C++ runtime | 识别 expert、KV block、phase、scratch |
| policy runtime | vLLM in-process extension | 维护 maps、预算、热度、动作队列 |
| execution plane | CUDA/UVM APIs | 执行 prefetch、evict、advise、device allocation |
| kernel/driver plane | NVIDIA UVM driver | 页表、fault replay、实际 migration |

### 4.2 应用层 runtime maps

建议实现一个 in-process `uvm_policy_runtime`，维护以下 maps：

```text
ExpertRegionMap:
  (layer_id, expert_id, tensor_role) -> address range, bytes, state

ExpertHeatMap:
  (layer_id, expert_id) -> recent_count, ema_score, last_used_step

KVActiveMap:
  (request_id, block_id) -> ptr, bytes, active_state, last_access_step

ScratchPoolMap:
  bucket_size -> free_list, live_bytes, cached_bytes, peak_bytes

BudgetMap:
  pool -> budget_bytes, estimated_live_bytes, pending_bytes

ActionQueue:
  PREFETCH_WEIGHT
  EVICT_WEIGHT
  PROTECT_KV
  PREFETCH_KV
  DIRECT_SCRATCH_ALLOC
  RELEASE_SCRATCH
```

### 4.3 为什么不全部放系统层

系统层适合做：

```text
page table update
fault handling
DMA / migration
low-level counter
driver-level residency query
```

系统层不适合做：

```text
MoE router hotset prediction
request-level KV active set decision
decode phase awareness
scratch lifetime classification
model-specific budget priority
```

这些策略强依赖 vLLM 和模型语义，放在应用层更自然，也更容易快速迭代。

## 5. 总体架构

```text
+----------------------------------------------------------------+
| vLLM Semantic Layer                                            |
| - decode step / layer id / phase                               |
| - MoE router top-k and scores                                  |
| - KV block table and request state                             |
| - tensor names, address ranges, allocation roles               |
+------------------------------+---------------------------------+
                               |
                               v
+----------------------------------------------------------------+
| UVM Policy Runtime                                              |
| - region registry                                               |
| - hot expert tracker                                            |
| - KV active protector                                           |
| - scratch device_direct pool                                   |
| - global HBM budget manager                                     |
| - action coalescer and async queues                             |
+------------------------------+---------------------------------+
                               |
                               v
+----------------------------------------------------------------+
| CUDA / UVM Execution Backend                                    |
| - cudaMemPrefetchAsync                                          |
| - cudaMemAdvise                                                 |
| - cudaMallocAsync / cudaFreeAsync                               |
| - CUDA streams and events                                       |
| - fault / allocator telemetry                                   |
+------------------------------+---------------------------------+
                               |
                               v
+----------------------------------------------------------------+
| NVIDIA UVM Driver                                               |
| - page fault handling                                           |
| - migration                                                     |
| - residency                                                     |
| - replayable fault accounting                                   |
+----------------------------------------------------------------+
```

核心设计原则：

1. vLLM 应用层负责语义和策略。
2. CUDA/UVM 层负责执行迁移动作。
3. driver 不承担模型语义推理。
4. 每个策略必须支持 `trace_only` 和 `enforce`。
5. 高频事件在应用层聚合，避免每次事件都变成系统层开销。
6. 所有可预测迁移都应 batch / coalesce 后异步提交。

## 6. 内存分池

系统将内存分为三个主池：

| Pool | 对象 | 机制 | 目标 |
|---|---|---|---|
| `weight_pool` | MoE expert weights / dense weights | UVM managed + hotset prefetch / cold evict | 减少 weight fault 和同步等待 |
| `kv_pool` | active KV cache blocks / prefix cache | UVM managed + active protection | 避免 active KV 被挤出 |
| `scratch_pool` | runtime scratch / gap2 temporary buffers | `cuda_malloc_async` device_direct | 绕开 UVM page fault |

### 6.1 weight_pool

weight_pool 不应全部改成 device allocation，因为权重总量超过单卡 HBM。它应继续使用 UVM managed memory，但调度单位必须从 page 提升到 tensor region / expert region。

Region 描述：

```text
WeightRegion:
  name
  layer_id
  role: attention | dense_mlp | moe_w13 | moe_w2 | norm | lm_head
  expert_id optional
  ptr
  bytes
  managed: true
  resident_state: UNKNOWN | CPU | GPU | PREFETCH_PENDING | EVICT_PENDING
```

第一优先级是 MoE expert weights，因为当前 top matched fault regions 主要是：

```text
model.layers.*.mlp.experts.w13_weight
model.layers.*.mlp.experts.w2_weight
```

### 6.2 kv_pool

KV 应按 block 管理，而不是按 page 管理。

```text
KVBlockState:
  request_id
  block_id
  layer_id
  ptr
  bytes
  state:
    ACTIVE_CURRENT_STEP
    ACTIVE_RECENT
    PREFIX_REUSABLE
    INACTIVE_COLD
  last_access_step
  ref_count
```

active KV 的优先级高于 cold weights。当前 device_direct wide phase 中 KV faults 上升，说明如果没有保护策略，scratch direct 和 weight residency 会扰动 KV。

### 6.3 scratch_pool

scratch_pool 固定采用当前最佳方向：

```text
backend = cuda_malloc_async
target_phases =
  enabled:attention
  enabled:moe
  enabled:model_forward
  enabled:compute_logits
  enabled:sampler
budget = 1 GiB
pool_release_threshold = 1 GiB
```

scratch 的目标不是迁移，而是绕开 UVM：

```text
runtime scratch should not enter managed fault path
```

后续应从“每次 direct allocation”升级为 size bucket cache。

## 7. 全局 HBM 预算

### 7.1 预算公式

```text
total_hbm_budget
  = weight_hot_budget
  + kv_active_budget
  + scratch_reserved_budget
  + safety_margin
```

建议初始值：

```text
scratch_reserved_budget = 1 GiB
kv_active_budget        = 4-6 GiB
safety_margin           = 1-2 GiB
weight_hot_budget       = total_hbm_budget
                          - scratch_reserved_budget
                          - kv_active_budget
                          - safety_margin
```

`total_hbm_budget` 不应等于物理 HBM，可以来自：

```text
physical_hbm * 0.90
```

或：

```text
SERVER_GPU_MEMORY_UTILIZATION * physical_hbm
```

### 7.2 优先级

预算紧张时优先级：

```text
1. active KV blocks
2. scratch reserved budget
3. current layer hot experts
4. next layer predicted hot experts
5. recently used experts
6. dense weights
7. cold experts
8. inactive prefix KV
```

这个顺序的原因：

1. active KV fault 直接打断 decode。
2. scratch reserved budget 是 gap2 消除的前提。
3. MoE hot experts 是 weight fault 的主要来源。
4. cold experts 最适合主动迁出 CPU。

### 7.3 预算账本

应用层 runtime 维护估算账本：

```text
PoolBudgetState:
  total_budget_bytes
  weight_hot_budget_bytes
  kv_active_budget_bytes
  scratch_reserved_budget_bytes
  safety_margin_bytes

PoolLiveState:
  estimated_weight_gpu_bytes
  estimated_kv_gpu_bytes
  scratch_device_live_bytes
  scratch_cached_bytes
  pending_prefetch_bytes
  pending_evict_bytes
  migration_inflight_bytes
```

MVP 可以先使用 action accounting 估算 resident bytes，后续再接入 driver residency query。

## 8. Object-Level Region Registry

调度器必须有统一 region registry。所有策略都不应直接操作裸地址，而应操作带语义的 region。

```text
Region:
  region_id
  pool: weight | kv | scratch
  ptr
  bytes
  object_type
  object_name
  layer_id optional
  expert_id optional
  request_id optional
  block_id optional
  phase optional
  allocation_time
  last_access_step
  predicted_next_use_step
  state
```

Region registry 的来源：

| Region 类型 | 来源 |
|---|---|
| weight regions | model loader / tensor allocation trace |
| MoE expert mapping | model architecture + parameter names |
| KV blocks | vLLM block manager |
| scratch buffers | allocator trace / allocation wrapper |

Region registry 是 UVM scheduler 和 vLLM 语义之间的桥。

## 9. Weight Hot Expert Residency

### 9.1 目标

weight 策略是超过 pure offload 的核心。

pure offload 的问题是访问时才搬，且搬运粒度通常偏粗。UVM scheduler 应利用 MoE router 预测下一步 expert，并提前做异步 prefetch。

### 9.2 输入信号

每个 decode step 收集：

```text
DecodeStepContext:
  step_id
  batch_size
  request_ids
  current_layer_id
  next_layer_id
  router_topk[layer_id][token] -> expert_ids
  router_scores[layer_id][token] -> scores
  phase
  memory_pressure
```

### 9.3 Expert heat

```text
ExpertHeat:
  layer_id
  expert_id
  last_used_step
  recent_count
  ema_score
  current_batch_hits
  resident_state
  bytes
```

MVP hot score：

```text
hot_score = current_batch_hits + recent_count
```

后续可扩展：

```text
hot_score =
  alpha * router_score
  + beta * recent_count
  + gamma * recency_score
  + delta * batch_frequency
```

### 9.4 Hotset policy

```text
protect:
  current layer experts
  next layer predicted experts
  experts used in last N decode steps

prefetch:
  hot experts not known resident on GPU

evict:
  cold experts outside protected set when budget exceeded
```

### 9.5 Prefetch action

对 expert 的 `w13_weight` 和 `w2_weight` region 执行：

```text
cudaMemPrefetchAsync(region.ptr, region.bytes, gpu_id, prefetch_stream)
```

prefetch 提交时机：

```text
after router result of layer L:
  update expert heat
  prefetch next predicted experts if known

before compute of layer L:
  submit prefetch for current layer protected experts
```

### 9.6 Evict action

预算超限时选择 cold experts：

```text
cold_score =
  age_since_last_used
  - recent_count_weight
  - predicted_next_use_weight
```

evict 可以先保守使用：

```text
cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)
```

再进入强执行：

```text
cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, evict_stream)
```

MVP 建议从 `prefetch-only + trace evict candidates` 开始。等确认 hotset 命中率稳定，再开启 enforce evict。

## 10. KV Active Protection

### 10.1 目标

KV 不应被 scratch / weight 的迁移随机挤出。尤其 wide async 实验显示：

```text
KV faults: 5,805 -> 15,323 (+163.96%)
```

虽然整体 TPOT 仍改善，但这说明 KV 需要明确优先级。

### 10.2 Active set

每个 decode step：

```text
active_kv_set =
  all KV blocks required by active requests for current decode step
```

状态转换：

```text
INACTIVE_COLD -> ACTIVE_CURRENT_STEP
ACTIVE_CURRENT_STEP -> ACTIVE_RECENT
ACTIVE_RECENT -> PREFIX_REUSABLE
PREFIX_REUSABLE -> INACTIVE_COLD
```

### 10.3 Protection action

active KV：

```text
cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, gpu_id)
cudaMemPrefetchAsync(ptr, bytes, gpu_id, prefetch_stream)
```

inactive KV 可选：

```text
cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)
```

MVP 只做 active protection，不做 aggressive KV offload。

原因：

1. KV fault 目前不是最大项。
2. 错误 offload active KV 会严重伤害 decode。
3. 现阶段主要矛盾仍是 scratch 和 weight。

## 11. Scratch Device Direct Strategy

### 11.1 固定最佳配置

当前最佳配置应作为默认 enforce 路径：

```bash
DEVICE_DIRECT_BACKEND=cuda_malloc_async
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD=1073741824
DEVICE_DIRECT_MAX_TOTAL_BYTES=1073741824
SCRATCH_POOL_BUDGET_BYTES=1073741824
SCRATCH_POOL_MODE=enforce
TARGET_PHASES=enabled:attention,enabled:moe,enabled:model_forward,enabled:compute_logits,enabled:sampler
```

### 11.2 Size bucket cache

为降低 allocator overhead，应引入 scratch bucket：

```text
ScratchBucket:
  bucket_size
  free_list
  live_list
  live_bytes
  cached_bytes
  peak_live_bytes
  alloc_count
  reuse_count
```

建议 buckets：

```text
4 KiB
16 KiB
64 KiB
256 KiB
1 MiB
4 MiB
16 MiB
64 MiB
```

分配流程：

```text
if allocation matches scratch phase and scratch class:
  bucket = round_up(size)
  if free_list[bucket] not empty:
    return cached device buffer
  else if scratch budget allows:
    return cudaMallocAsync(bucket)
  else:
    fallback according to mode
else:
  managed allocation
```

释放流程：

```text
if buffer is scratch device_direct:
  return to bucket free_list
else:
  normal free
```

### 11.3 动态收缩

如果出现明显副作用：

```text
weight_faults rising
KV_faults rising
HBM pressure high
scratch_cached_bytes too high
```

则 scheduler 可以动态降低：

```text
scratch_cached_bytes_limit
scratch_reserved_budget
max_bucket_size
```

## 12. Async Execution Model

调度必须避免把迁移动作放到 compute critical path。

建议使用三类 stream：

```text
compute_stream:
  current layer compute

prefetch_stream:
  hot weight prefetch
  active KV prefetch

evict_stream:
  cold weight eviction
  inactive KV demotion
```

动作提交原则：

1. 对同一 region 去重。
2. 小 region 合并为 batch。
3. 迁移动作不等待完成，除非当前 compute 必须访问。
4. compute 前只对极少数 critical region 做必要同步。
5. evict 低优先级，不得抢占当前 step 所需 prefetch。

Action queue：

```text
Action:
  type:
    PREFETCH_WEIGHT
    EVICT_WEIGHT
    PROTECT_KV
    PREFETCH_KV
    DIRECT_SCRATCH_ALLOC
    RELEASE_SCRATCH
  region_id
  ptr
  bytes
  priority
  deadline_step
  stream
  mode: trace_only | enforce
```

## 13. Runtime Flow

每个 decode step 的逻辑：

```text
begin_decode_step:
  collect active requests
  update active KV set
  protect active KV blocks
  reserve scratch budget

for each layer L:
  before_layer:
    submit prefetch for current layer protected experts
    submit prefetch for active KV if not already submitted

  compute_layer:
    run attention / moe / model_forward
    scratch allocations use device_direct bucket pool

  after_router_or_moe:
    read router top-k
    update expert heat
    plan next hot expert set
    enqueue prefetch for next layer / next step

  after_layer:
    release scratch buffers to async pool
    update budget accounting
    enqueue cold expert evict candidates if over budget

end_decode_step:
  flush action queue
  emit telemetry
```

MVP 中，如果 router top-k 只在当前 layer 内部可见，则先实现：

```text
current layer observe -> next token / recent-N prediction
```

而不是强行要求完美的 `L+1` 预测。

## 14. Telemetry

必须保留可解释性，否则无法判断是否真的优于 pure offload。

每轮实验记录：

```text
performance:
  output tok/s
  TPOT
  TTFT
  failed requests

faults:
  parsed faults
  weight faults
  KV faults
  unknown faults
  gap2 faults
  replayable raw delta
  replayable dedup delta

scratch:
  device_direct_actual_records
  scratch_live_bytes
  scratch_peak_live_bytes
  scratch_cached_bytes
  scratch_reuse_count
  budget_reject_records

weight:
  hotset size
  prefetch bytes
  prefetch count
  evict candidate bytes
  evict actual bytes
  expert hit rate
  wrong prefetch bytes

KV:
  active blocks
  protected bytes
  KV prefetch bytes
  active KV faults

budget:
  estimated_weight_gpu_bytes
  estimated_kv_gpu_bytes
  scratch_device_live_bytes
  pending_migration_bytes
  HBM pressure
```

关键派生指标：

```text
expert_hotset_hit_rate =
  bytes_accessed_from_prefetched_hotset / total_expert_bytes_accessed

prefetch_usefulness =
  useful_prefetch_bytes / total_prefetch_bytes

KV_protection_success =
  active_KV_faults_after / active_KV_faults_before

scratch_success =
  gap2_faults_after / gap2_faults_before
```

## 15. MVP Plan

### Phase 0: Baseline 对齐

必须先跑出 vLLM pure CPU offload baseline：

```text
same model
same prompts
same output length
same request rate
same max model len
same server max num seqs
```

记录：

```text
TPOT
output tok/s
TTFT
failed requests
CPU-GPU transfer indicators if available
```

### Phase 1: 固定 scratch device_direct

固定当前最佳 scratch 配置：

```text
cuda_malloc_async
wide decode phases
1 GiB scratch reserved budget
pool release threshold 1 GiB
```

目标：

```text
unknown faults near zero
gap2 faults near zero
TPOT not worse than baseline UVM
```

### Phase 2: Trace-only weight hotset

接入 MoE router top-k，但只 trace：

```text
which experts would be prefetched
prefetch bytes
hotset hit rate
evict candidates
budget pressure
```

不实际迁移。

目标：

```text
证明 router hotset 有稳定预测价值
证明 prefetch bytes 在预算内
证明不会明显挤压 KV / scratch
```

### Phase 3: Weight prefetch enforce

开启：

```text
hot expert cudaMemPrefetchAsync(..., gpu)
no cold evict yet
```

目标：

```text
weight faults 下降
TPOT 下降
KV faults 不显著上升
```

### Phase 4: Conservative cold evict

开启保守 cold evict：

```text
只 evict long-cold experts
只在 weight budget 超限时 evict
evict stream 低优先级
```

先使用 `cudaMemAdvise`，再尝试 CPU prefetch enforce。

### Phase 5: KV active protection

开启 active KV：

```text
active KV preferred GPU
active KV prefetch
inactive KV 只 trace，不 enforce offload
```

目标：

```text
KV faults 不再因 scratch/weight 策略反弹
```

## 16. Implementation Hooks

建议实现位置：

```text
vLLM model loader:
  build weight region registry from named tensors

MoE layer / router:
  emit router top-k and layer id to policy runtime

KV block manager:
  emit active KV blocks per decode step

allocator wrapper:
  classify scratch allocation by phase and size
  route eligible scratch to device_direct bucket pool

scheduler runtime:
  maintain maps
  plan actions
  submit CUDA/UVM async calls

telemetry:
  write json/csv/md summaries
```

候选代码区域需要结合当前仓库进一步确认，初步包括：

```text
vllm/vllm/device_allocator/
vllm/vllm/v1/core/
vllm/vllm/model_executor/
vllm/uvm_test/
```

## 17. Comparison Against vLLM Pure Offload

本方案相对 pure offload 的理论优势：

| 维度 | vLLM pure CPU offload | UVM hybrid scheduler |
|---|---|---|
| 粒度 | 通常较粗 | expert / tensor / KV block / scratch buffer |
| 触发 | 需求触发 | 预测触发 |
| MoE router 利用 | 弱 | 强 |
| scratch | 容易混入统一内存压力 | device_direct 绕开 UVM fault |
| KV | 缺少统一优先级 | active block protection |
| overlap | 容易同步等待 | prefetch / evict stream overlap |
| 策略位置 | offload 机制内 | vLLM 应用层语义 runtime |

本方案不是要替代 UVM driver，而是要避免把 driver fault path 当作性能主路径。

## 18. Risks and Guardrails

### 18.1 过度 prefetch

风险：

```text
hotset 预测错误导致无用迁移，挤出 KV 或真正 hot weights
```

防护：

```text
max_prefetch_bytes_per_step
max_prefetch_experts_per_layer
prefetch cooldown
hotset hit-rate telemetry
disable prefetch under high pressure
```

### 18.2 cold evict 错误

风险：

```text
刚 evict 的 expert 下一步又被访问，造成 re-fault
```

防护：

```text
evict trace-only first
long-cold threshold
protected recent-N steps
low-priority evict stream
```

### 18.3 KV 被挤出

风险：

```text
scratch direct 和 weight prefetch 增加 HBM 压力，active KV faults 上升
```

防护：

```text
active KV priority higher than cold weights
KV active protection
scratch reserved budget explicit accounting
dynamic scratch cache shrink
```

### 18.4 allocator overhead

风险：

```text
device_direct 消除了 fault，但引入 allocator 同步成本
```

防护：

```text
cuda_malloc_async backend
pool release threshold
size bucket reuse
allocation telemetry
```

### 18.5 driver residency 不可精确观测

风险：

```text
应用层估算 resident bytes 和实际 UVM residency 不一致
```

防护：

```text
trace action accounting
sampled driver query if available
fault feedback correction
budget safety margin
```

## 19. Final Direction

当前实验已经证明：

```text
scratch / gap2 可以通过 cuda_malloc_async device_direct 基本解决。
```

下一步真正决定能否超过 vLLM pure CPU offload 的，是：

```text
MoE weight hotset residency
+ async prefetch / evict
+ KV active protection
+ application-layer policy runtime
```

最终系统定位应是：

```text
vLLM pure offload:
  static coarse-grained CPU-GPU hierarchy

UVM-Aware Hybrid Residency Scheduler:
  dynamic predictive object-level memory scheduler over UVM
```

最重要的路线选择：

```text
不要把 UVM page fault 当成正常路径。
不要一开始重写 UVM driver。
先在 vLLM 应用层构建 bpftime-like in-process policy runtime。
用对象级语义提前生成迁移动作。
让 CUDA/UVM 只作为异步执行后端。
```

## 20. Implementation Progress

### 20.1 Completed: policy runtime facade

已新增应用层策略运行时门面：

```text
vllm/device_allocator/uvm_policy_runtime.py
```

当前它不是重写原有 Stage K / Stage L 逻辑，而是复用：

```text
uvm_pool_residency.py      residency ledger / region registry
uvm_pool_coordinator.py    cross-pool action admission
uvm.py                    CUDA/UVM execution primitives
```

第一版实现的动作是：

```text
protect_kv_region
```

支持模式：

```text
trace_only
advise
prefetch
advise_prefetch
```

默认关闭，避免影响已有实验：

```bash
VLLM_UVM_POLICY_RUNTIME_ENABLE=0
```

### 20.2 Completed: KV active protection MVP

已在 KV runtime 中加入 active KV protection 的最小闭环：

```text
vllm/v1/core/uvm_kv_runtime_policy.py
vllm/v1/core/kv_cache_manager.py
```

调用点：

```text
KVCacheManager.allocate_slots()
  -> _maybe_protect_active_kv_blocks()
  -> UvmKvRuntimePolicy.protect_active_blocks()
  -> UvmPolicyRuntime.protect_kv_region()
```

设计原则：

```text
只在请求当前 KV block table 已经确定后提交 active protection。
不修改 vLLM block 表。
不 aggressive offload inactive KV。
trace_only 不要求 KV pointer map。
advise / prefetch 模式复用已有 Stage L6 KV pointer map。
```

新增开关：

```bash
VLLM_UVM_KV_ACTIVE_PROTECT_ENABLE=1
VLLM_UVM_KV_ACTIVE_PROTECT_MODE=trace_only
VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE=0
VLLM_UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS=0
```

如果要实际执行 GPU preferred / prefetch：

```bash
VLLM_UVM_POLICY_RUNTIME_ENABLE=1
VLLM_UVM_KV_RUNTIME_ENABLE=1
VLLM_UVM_KV_ACTIVE_PROTECT_ENABLE=1
VLLM_UVM_KV_ACTIVE_PROTECT_MODE=advise_prefetch
VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE=<kv_pointer_map_jsonl>
```

### 20.3 Completed: KV active protection Phase 5/6 experiment

已完成 Phase 5/6 实验脚本与一次 p10 运行：

```text
workloads/vllm/run_stage_m_kv_active_protection_p10.sh
log/stage_m_kv_active_protection_p10_run/kv_active_protection_summary.md
```

结果说明：

```text
device_direct_baseline:      KV faults=22928 TPOT=246.07ms tok/s=3.89
kv_active_trace:             KV faults=34600 TPOT=297.15ms tok/s=3.23
kv_active_advise_prefetch:   KV faults=41798 TPOT=328.67ms tok/s=2.93
```

结论：

```text
active KV block 识别和调用链已经打通。
trace / advise_prefetch 都能产生 issued 记录，且 fallback=0。
当前策略没有降低 KV faults，TPOT 和吞吐反而变差。
因此 KV active protection 先标记为功能闭环完成、性能不达标，继续推进后续 phase。
```

### 20.4 Completed: scratch size-bucket reuse cache MVP

已实现 Phase 7 第一版 scratch device_direct reuse cache：

```text
vllm/uvm_test/uvm_allocator.cpp
```

行为：

```text
仅在 scratch_pool + device_direct + cuda_malloc_async 路径启用。
默认关闭。
free 阶段将合格 scratch device-direct 指针按 size_bucket 保留。
alloc 阶段先按 size_bucket 分组、exact size 匹配复用。
cache entry 带 CUDA event，只有释放流上的 event ready 后才允许复用。
cache 保留的物理显存继续占用 device_direct / scratch pool budget。
```

新增环境变量：

```bash
VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE=1
VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES=67108864
VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET=8
```

已在通用 runner 中透传：

```text
workloads/vllm/run_kv_fault_ratio.sh
```

新增 CLI：

```bash
--uvm-scratch-pool-reuse-cache-enable <0|1>
--uvm-scratch-pool-reuse-cache-max-bytes <n>
--uvm-scratch-pool-reuse-cache-max-entries-per-bucket <n>
```

已扩展 metrics 汇总：

```text
workloads/vllm/summarize_gap_watch_metrics.py
```

新增实验脚本：

```text
workloads/vllm/run_stage_m_scratch_size_bucket_reuse_cache_ab_p10.sh
```

对比项：

```text
scratch_direct_no_cache
scratch_direct_bucket_cache
```

核心观测：

```text
cache_hits
cache_misses
cache_fresh_backend_allocations
cache_inserts
cache_insert_rejects
cache_peak_cached_bytes
TPOT / tok/s / TTFT
raw replayable faults / dedup faults
gap2 unknown faults
```

运行方式：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./run_stage_m_scratch_size_bucket_reuse_cache_ab_p10.sh
```

### 20.5 Verification

已完成 Python 静态编译检查：

```bash
python3 -m py_compile \
  workloads/vllm/vllm/vllm/device_allocator/uvm_policy_runtime.py \
  workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py \
  workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py
```

结果：通过。

当前新增 Phase 7 之后还需要执行：

```bash
bash -n workloads/vllm/run_kv_fault_ratio.sh
bash -n workloads/vllm/run_stage_m_scratch_size_bucket_reuse_cache_ab_p10.sh
python3 -m py_compile workloads/vllm/summarize_gap_watch_metrics.py
make -C workloads/vllm/vllm/uvm_test uvm
cp workloads/vllm/vllm/uvm_test/uvm_allocator.so \
  workloads/vllm/vllm/vllm/uvm_allocator.abi3.so
```

### 20.6 Next implementation steps

下一批修改建议：

1. 跑 Phase 7 `scratch_direct_no_cache` vs `scratch_direct_bucket_cache`，确认 allocator overhead 是否下降、TPOT 是否稳定。
2. 将 FusedMoE 中已有 hot expert prefetch 逐步接入 `uvm_policy_runtime`，先保持行为等价。
3. 增加 cold expert evict 保守策略，只 evict long-cold experts，先 trace 再 enforce。
4. 统一全局 HBM BudgetMap，显式协调 `weight_hot_budget + kv_active_budget + scratch_reserved_budget + safety_margin`。
5. 拆分 compute / prefetch / evict stream，减少同步路径干扰。
6. 与 vLLM pure CPU offload 做最终 A/B，对比 TPOT、tok/s、failed requests 和 fault 结构。
