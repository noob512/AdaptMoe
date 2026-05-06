# vLLM UVM Stage L 三池分区与 GPU Residency Manager 实现方案

## 1. 背景与目标

当前 Stage K 已经实现了 high-level action coordinator：它可以在 Stage I
expert weight prefetch/offload、Stage J prefix-cache eviction 等安全动作执行前
做 grant/deny，并统一输出 JSONL trace。

但是 Stage K 管的是“本 step 允许发起多少 bytes 的动作”，不是“当前 GPU 上
每个 pool 实际驻留了多少 bytes”。因此它还不能完成最初目标：

```text
限制 weights / KV cache / runtime scratch 的 GPU 显存驻留量，
避免它们相互挤占；
weights 超预算时只 offload/evict weights；
KV 超预算时只处理 KV；
scratch 超预算时只限制 scratch；
通过限制 cold expert weights 的 GPU resident bytes，
把显存容量让给 KV cache。
```

Stage L 建议新增一个 Pool Residency Manager。它不是替代 Stage K，而是补齐
Stage K 没有维护的 GPU residency 状态与 per-pool 驻留预算。

至少维护三个 pool：

```text
weights_pool        模型权重 / MoE expert weights
kv_pool             KV cache blocks
scratch_pool        runtime 临时缓冲区 / workspace / attention scratch
```

同时维护一个全局预算：

```text
global_gpu_resident_budget
```

Stage L 的核心目标：

```text
weights 只能占 weights GPU resident 预算
KV 只能占 KV GPU resident 预算
scratch 只能占 scratch GPU resident 预算
某个 pool 超预算时，优先只在本 pool 内 reclaim
global 超预算时，按优先级协调，但不能误驱逐 active KV 或 active scratch
```

## 2. Stage L 与已有 Stage 的关系

Stage L 不是从零开始，而是承接已有能力。

已有基础：

```text
Stage E:
  - weights allocation telemetry
  - weight tensor semantic address map
  - MoE routing trace

Stage F:
  - allocator 侧 pool registry telemetry

Stage G:
  - runtime scratch device-direct admission
  - scratch pool budget telemetry

Stage H:
  - 基于 weight map + routing trace 生成 hot/cold expert plan

Stage I:
  - MoE expert weight prefetch/offload 执行动作
  - raw range prefetch / advise 能力

Stage J:
  - KV runtime block pressure
  - KV soft admission deny
  - safe prefix-cache metadata eviction

Stage K:
  - global / weights / KV / scratch action coordinator
```

Stage L 新增的是：

```text
GPU resident bytes 账本
per-object residency state
per-pool resident budget enforce
weights hot/cold residency 闭环
KV block residency metadata
scratch live bytes 与统一预算接入
```

需要特别区分：

```text
allocation live bytes:
  这块 managed/device memory 还没有 free。

GPU resident bytes:
  这块 managed memory 的页面当前被认为驻留在 GPU，或者 device-direct memory
  当前占用 GPU 显存。
```

当前 Stage E/F/G 主要统计 allocation live bytes 或 device-direct live bytes。
Stage L 要管理的是 GPU resident bytes。

## 3. 新增文件与模块边界

建议新增 Python 控制面：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
```

该模块负责：

```text
1. 读取 residency 相关环境变量。
2. 注册 weights / KV / scratch 对象。
3. 维护对象级 residency state。
4. 维护 per-pool GPU resident bytes。
5. 对 GPU residency 请求做 grant / deny。
6. 在 enforce 模式下选择同 pool victim。
7. 输出 residency JSONL trace。
8. 提供全局单例和便捷 API。
```

Stage L 不应直接在该模块里调用复杂 vLLM 内部逻辑。它应提供决策和账本 API，
由安全调用点执行真正动作：

```text
weights:
  fused_moe/layer.py 执行 prefetch/offload。

KV:
  kv_cache_manager.py / block_pool.py 执行 safe block offload/prefetch-back。

scratch:
  uvm_allocator.cpp 或 Python bridge 在 allocation/free 时更新 scratch live。
```

## 4. 环境变量设计

建议新增：

```text
VLLM_UVM_RESIDENCY_ENABLE=0|1
VLLM_UVM_RESIDENCY_MODE=trace_only|enforce
VLLM_UVM_RESIDENCY_TRACE_FILE=/tmp/vllm_uvm_pool_residency_stage_l.jsonl

VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES=0
VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES=0
VLLM_UVM_RESIDENCY_KV_GPU_BYTES=0
VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES=0

VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES=0
VLLM_UVM_RESIDENCY_PRIORITY=kv,scratch,weights

VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE=unknown|cpu|gpu
VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET=0|1
VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD=0|1

VLLM_UVM_RESIDENCY_KV_ENABLE_SAFE_OFFLOAD=0|1
VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE=0|1

VLLM_UVM_RESIDENCY_SCRATCH_HOOK_ENABLE=0|1
```

语义：

```text
trace_only:
  只记录 would-grant / would-deny / would-evict，不改变原执行路径。

enforce:
  超预算时真实 deny，或先执行同 pool reclaim，再 grant。
```

预算为 0 的含义建议为“不启用该预算限制”，与已有 Stage D/E/G/K 风格保持一致。

## 5. 核心数据结构

### 5.1 PoolKind

```python
class PoolKind:
    WEIGHTS = "weights"
    KV = "kv"
    SCRATCH = "scratch"
```

### 5.2 ResidencyState

```python
class ResidencyState:
    GPU = "gpu"
    CPU = "cpu"
    PREFETCHING_GPU = "prefetching_gpu"
    PREFETCHING_CPU = "prefetching_cpu"
    UNKNOWN = "unknown"
```

第一版可以用 optimistic accounting：

```text
prefetch_to_gpu issued 后先进入 PREFETCHING_GPU
event/sync 确认后进入 GPU

prefetch_to_cpu issued 后先进入 PREFETCHING_CPU
event/sync 确认后进入 CPU
```

如果短期不做 event tracking，可以在 trace_only 中只记录 intent，在 enforce 中使用同步
或 conservative update，避免账本过度乐观。

### 5.3 WeightResidencyRegion

weights 以 expert slice 或 tensor region 为对象：

```text
region_id
pool = weights
layer_id
expert_id
role
tensor_name
ptr
bytes
state
last_access_step
last_prefetch_step
last_offload_step
heat_score
evictable
active
source
```

region_id 建议使用稳定 key：

```text
weights:<layer_id>:<expert_id>:<role>
```

对于 dense/shared weights，可以用：

```text
weights:<tensor_name>
```

第一版重点做 MoE expert weights，不要求 dense/shared weights 全量 offload。

### 5.4 KvResidencyBlock

KV 以 block 为对象：

```text
region_id
pool = kv
block_id
group_id
layer_id 或 kv_cache_group
request_id
ptr
bytes
state
ref_cnt
is_prefix_cache
block_hash
last_access_step
last_prefetch_step
last_offload_step
evictable
```

第一版只允许处理安全对象：

```text
ref_cnt == 0
位于 free queue
或 prefix-cache free block
```

不要在第一版处理 active request block。

### 5.5 ScratchResidencyAllocation

scratch 以 allocation 为对象：

```text
region_id
pool = scratch
ptr
bytes
phase
backend
state
alloc_count
lifetime
stream
```

对于 device-direct scratch，state 可直接认为 GPU。
对于 managed scratch，可标记 UNKNOWN 或 GPU_HINTED。

## 6. 三池预算关系

推荐采用硬分区加安全余量：

```text
global_budget =
    total_gpu_memory * gpu_memory_utilization
    - cuda_context_reserve
    - fragmentation_margin

weight_budget = explicit_weight_budget
scratch_budget = measured_peak_scratch + scratch_margin
kv_budget = global_budget - weight_budget - scratch_budget - safety_margin
```

也可以全显式配置：

```text
VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES
VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES
VLLM_UVM_RESIDENCY_KV_GPU_BYTES
VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES
```

典型目标场景：

```text
全部 expert weights: 30GB
有效 hot expert set: 8GB
weight_gpu_budget: 8GB
KV cache 使用 global - 8GB - scratch_budget 后的剩余空间
```

注意：weights allocation live bytes 仍然可能是 30GB，因为 managed memory 还存在。
Stage L 限制的是 GPU resident bytes，而不是 managed allocation 总大小。

## 7. PoolResidencyManager API 设计

建议提供全局单例：

```python
get_uvm_pool_residency_manager()
```

便捷函数：

```python
register_uvm_residency_region(...)
request_uvm_gpu_residency(...)
mark_uvm_residency_state(...)
release_uvm_residency_region(...)
record_uvm_residency_access(...)
get_uvm_residency_snapshot(...)
```

### 7.1 register_region

```python
register_region(
    pool: str,
    region_id: str,
    ptr: int,
    bytes: int,
    state: str = "unknown",
    metadata: dict | None = None,
)
```

用途：

```text
weights:
  Stage E weight map / Stage I MoE layer 注册 expert slice。

KV:
  block manager 创建 block metadata 时注册 block range。

scratch:
  allocator allocation trace 或 C++ hook 注册 scratch allocation。
```

### 7.2 request_gpu_residency

```python
decision = request_gpu_residency(
    pool: str,
    region_id: str,
    bytes: int,
    scope_key: str | None = None,
    active_region_ids: set[str] | None = None,
    metadata: dict | None = None,
)
```

返回：

```text
allowed
would_deny
reason
pool_budget_bytes
pool_gpu_resident_bytes
global_budget_bytes
global_gpu_resident_bytes
victims
```

如果需要 reclaim，则 victims 只从同 pool 选择，除非显式进入 global reclaim。

### 7.3 mark_state

```python
mark_state(
    region_id: str,
    state: str,
    bytes_delta_policy: str = "auto",
    metadata: dict | None = None,
)
```

状态变化时更新账本：

```text
CPU/UNKNOWN -> GPU:
  pool_gpu_resident_bytes += bytes

GPU -> CPU:
  pool_gpu_resident_bytes -= bytes

PREFETCHING_*:
  是否计入预算由 mode 决定。
```

建议 enforce 下 PREFETCHING_GPU 先计入预算，防止并发 prefetch 超发。
PREFETCHING_CPU 在确认完成前仍计入 GPU，防止过早释放账本。

## 8. Weights hot/cold residency enforce

weights 是最适合先做实的 pool，因为它具备：

```text
地址稳定
对象语义清晰
MoE routing 提供热度
Stage H 已有 hot/cold plan
offload 不需要修改 request block table
```

### 8.1 注册 expert slice

在 Stage I 当前 `_uvm_expert_weight_slices()` 计算出：

```text
role
ptr
size_bytes
local_expert_id
global_expert_id
layer_id
```

后注册：

```text
region_id = weights:<layer_id>:<global_expert_id>:<role>
```

metadata：

```text
layer_name
layer_id
global_expert_id
local_expert_id
role
plan_hot
plan_cold
```

### 8.2 hot prefetch 前请求 residency

Stage I 当前 prefetch 前只做：

```text
per-step bytes budget
Stage K coordinator request
```

Stage L 后增加：

```python
residency_decision = request_uvm_gpu_residency(
    pool="weights",
    region_id=region_id,
    bytes=size_bytes,
    scope_key=f"moe_step:{pid}:{step}",
    active_region_ids=current_active_regions,
)
```

如果 trace_only：

```text
记录 would_grant / would_deny / would_evict
原动作可继续按 Stage I mode 执行
```

如果 enforce：

```text
allowed=False:
  跳过 hot prefetch，记录 residency_deny。

allowed=True 且 victims 非空:
  先 offload victims，再 prefetch hot region。
```

### 8.3 victim 选择

weights victim 必须满足：

```text
pool == weights
state == GPU
evictable == True
region_id not in active_region_ids
不是当前 step active expert
```

排序建议：

```text
1. cold plan 优先
2. heat_score 低优先
3. last_access_step 旧优先
4. bytes 大者可优先，用更少 victim 腾足空间
```

第一版可以只选 Stage H offload_plan 中的 cold expert。

### 8.4 offload 与 prefetch 状态更新

victim offload：

```python
mark_state(victim, PREFETCHING_CPU)
prefetch_range_to_cpu(victim.ptr, victim.bytes)
sync/event 确认
mark_state(victim, CPU)
```

hot prefetch：

```python
mark_state(region, PREFETCHING_GPU)
prefetch_range_to_device(ptr, size, device)
sync/event 或 optimistic confirm
mark_state(region, GPU)
```

第一版建议：

```text
trace_only:
  不强制同步，只记录动作。

enforce:
  可以通过环境变量选择同步确认，保证 residency 账本可信。
```

## 9. KV cache 初始化按预算分配 blocks

这是实现“weights 让位给 KV cache”的关键。

如果 vLLM 的 KV block 数量已经在启动时确定，那么运行时把 cold weights
prefetch 到 CPU 并不会自动增加 KV blocks。必须在 KV cache 初始化或 profiling
阶段使用 Stage L 的预算。

推荐计算：

```text
effective_kv_budget_bytes =
    global_gpu_resident_budget
    - weight_gpu_resident_budget
    - scratch_gpu_resident_budget
    - safety_margin
```

然后：

```text
num_kv_blocks = effective_kv_budget_bytes // bytes_per_block
```

实现位置应在 vLLM 生成 `kv_cache_config` / 计算 `num_gpu_blocks` 的路径附近，
而不是 C++ allocator 层。

建议启动顺序：

```text
1. load model with UVM weights
2. 生成 weight map
3. 应用初始 weight residency plan
4. 确定 weight_gpu_budget
5. profile 或配置 scratch_budget
6. 计算 kv_budget
7. 初始化 KV cache blocks
```

如果短期无法改变 vLLM profiling 顺序，可以先支持显式：

```text
VLLM_UVM_RESIDENCY_KV_GPU_BYTES
```

并在 check 脚本中验证：

```text
budget_bytes -> budget_blocks -> num_blocks_to_allocate/admission
```

## 10. KV block residency metadata

Stage J 当前已经在 block-manager 边界计算 runtime pressure，但还没有 KV data
offload/prefetch。Stage L 需要补充 block 级 residency metadata。

### 10.1 block 注册

当 block pool 创建 KV blocks 或 KV cache config 可用时，注册：

```text
region_id = kv:<group_id>:<block_id>
bytes = bytes_per_block
state = GPU
```

如果能计算每个 block 对应的 raw pointer range，则记录：

```text
ptr
bytes
kv_cache_tensor_name
group_id
block_id
```

如果第一版拿不到 ptr，可以先做 metadata-only residency，后续再接 raw prefetch。

### 10.2 admission 计算

在 `UvmKvRuntimePolicy.should_allow_allocation()` 中，逐步从：

```text
used_blocks = total_blocks - free_blocks
```

升级为：

```text
kv_gpu_resident_bytes
projected_kv_gpu_resident_bytes =
    kv_gpu_resident_bytes + num_blocks_to_allocate * bytes_per_block
```

超预算时：

```text
trace_only:
  记录 would_deny / would_offload_candidate。

enforce:
  先尝试 KV pool 内 safe offload。
  如果仍不足，再 soft deny admission。
```

## 11. KV safe-block offload / prefetch-back

第一版 KV offload 只处理 safe blocks：

```text
ref_cnt == 0
在 free queue
prefix-cache block
```

不要先处理 active request block。

### 11.1 safe offload

候选选择：

```text
pool == kv
state == GPU
ref_cnt == 0
is_prefix_cache == True 或 in_free_queue == True
```

执行：

```python
mark_state(block, PREFETCHING_CPU)
prefetch_range_to_cpu(ptr, size)
sync/event 确认
mark_state(block, CPU)
```

这一步才会让：

```text
kv_gpu_resident_bytes -= size
kv_cpu_resident_bytes += size
```

不同于 Stage J prefix-cache metadata eviction，safe-block offload 需要真的迁移 KV data。

### 11.2 prefetch-back

当 CPU-resident KV block 即将被使用，例如：

```text
prefix cache hit
block 被 touch
进入 attention kernel 前
```

必须先执行：

```python
mark_state(block, PREFETCHING_GPU)
prefetch_range_to_device(ptr, size, device)
等待完成或插入 stream dependency
mark_state(block, GPU)
```

否则 attention kernel 可能直接访问 CPU-resident page，造成 page fault 风暴。

### 11.3 第一版边界

第一版不做：

```text
active request block CPU swap
active block table 修改
跨 request KV data reload 状态机
异步 prefetch 与 attention kernel 的复杂流水
```

这些应放在 Stage L2/J3。

## 12. Scratch pool 管理

scratch 和 weights/KV 不同。scratch 生命周期短，经常是 kernel workspace 或
attention 临时缓冲区，用完后会很快释放，并且相对 weights/KV 通常占用更小。
因此 scratch pool 的目标不应是运行时 eviction，而应是在启动、profile 或配置阶段
选择足够保守的预算与参数，让临时缓冲区在正常运行中不需要被迫驱逐。

换句话说，scratch pool 应作为三池中的“预留容量池”：

```text
启动/初始化时为 scratch 预留 measured_peak_scratch + margin；
运行时只做 admission control 和 telemetry；
超预算时优先回退 managed UVM 或记录 over-budget；
不对 active scratch 做 eviction；
不依赖 scratch eviction 来恢复系统容量。
```

当前 Stage G 已经实现：

```text
runtime_scratch 分类
target phase admission
scratch_pool_budget
device-direct cudaMalloc/cudaMallocAsync
free 时归还 scratch live bytes
```

Stage L 应把 Stage G 纳入统一 residency，但语义应更偏向“预留与校验”，而不是
“运行时驱逐”：

```text
scratch_gpu_live_bytes
scratch_gpu_budget_bytes
scratch_peak_bytes
scratch_reserved_bytes
scratch_margin_bytes
```

预算建议：

```text
scratch_budget =
    measured_peak_scratch
    + scratch_margin
    + optional_backend_fragmentation_margin
```

`measured_peak_scratch` 可以来自一次 profile / warmup 运行，也可以先由用户显式配置。
如果没有 profile 结果，第一版应使用保守默认值，并在 summary 中提示 observed peak，
方便下一轮调整。

运行时策略：

```text
scratch 在 target phase 内申请：
  如果 scratch_gpu_live_bytes + size <= scratch_budget:
      允许 device-direct
  否则:
      trace_only:
          记录 scratch_over_budget，仍可继续原路径；
          summary 中提示 scratch_budget 过小，需要调大预留。
      enforce:
          拒绝 device-direct，回退 managed UVM；
          不尝试 eviction；
          只有在显式 strict 模式下才 deny allocation。
```

第一版建议只做：

```text
profile / warmup 期 scratch peak 观测
startup-time scratch budget reserve
admission control
free-time budget release
trace
```

明确不做 scratch eviction。因为 scratch 通常正在被即将运行或已经排队的 kernel 使用，
运行时驱逐临时缓冲区的风险和同步成本都高于收益。正确方向是通过更合适的
`scratch_budget`、target phases、device-direct backend 参数和安全 margin，让它从一开始
就不进入需要 eviction 的状态。

## 13. 三池协作规则

### 13.1 per-pool 规则

```text
weights 超预算:
  只 offload weights cold expert。

KV 超预算:
  只 offload/evict KV safe blocks；
  如果不足，则 soft deny 新 KV block admission。

scratch 超预算:
  视为启动期 scratch 预留不足或参数不合适；
  运行时只限制 scratch device-direct admission / 回退 managed；
  不驱逐 active scratch，也不把 scratch eviction 作为恢复容量的手段。
```

### 13.2 global 规则

当 global_gpu_resident_budget 超预算时，按 priority 选择让路 pool：

```text
默认 priority = kv,scratch,weights
```

含义：

```text
KV 优先保留，因为影响请求容量与 decode 连续性。
scratch 次优先，因为影响 kernel 正常运行。
weights cold expert 最适合让路。
```

但 global reclaim 也必须遵守安全边界：

```text
不能 offload 当前 active expert。
不能 offload active KV block。
不能驱逐 active scratch。
```

如果 KV 想增长，优先要求 weights pool 释放 cold weights。
如果 scratch 想增长，优先说明 scratch 预留预算需要调大；运行时只限制 scratch
device-direct admission 或回退 managed，不应抢 active KV，也不应触发 scratch eviction。

## 14. Trace 设计

新增 trace file：

```text
VLLM_UVM_RESIDENCY_TRACE_FILE=/tmp/vllm_uvm_pool_residency_stage_l.jsonl
```

建议 action：

```text
residency_config
residency_register_region
residency_request
residency_grant
residency_deny
residency_evict_candidate
residency_offload_issued
residency_prefetch_issued
residency_state_update
residency_sync
residency_summary
```

### 14.1 residency_request

关键字段：

```text
pool
region_id
requested_bytes
scope_key
mode
pool_budget_bytes
pool_gpu_resident_bytes
pool_remaining_bytes
global_budget_bytes
global_gpu_resident_bytes
global_remaining_bytes
would_deny
allowed
reason
metadata
```

### 14.2 residency_evict_candidate

关键字段：

```text
pool
region_id
bytes
state
heat_score
last_access_step
evictable
active
reason
```

### 14.3 residency_state_update

关键字段：

```text
pool
region_id
old_state
new_state
bytes
pool_gpu_resident_bytes_before
pool_gpu_resident_bytes_after
global_gpu_resident_bytes_before
global_gpu_resident_bytes_after
reason
```

### 14.4 residency_summary

至少输出：

```text
weight_gpu_resident_bytes
weight_cpu_resident_bytes
weight_gpu_budget_bytes

kv_gpu_resident_bytes
kv_cpu_resident_bytes
kv_gpu_budget_bytes

scratch_gpu_live_bytes
scratch_gpu_budget_bytes

global_gpu_resident_bytes
global_gpu_budget_bytes

requests
grants
denies
would_denies
eviction_candidates
offload_issued
prefetch_issued
state_updates
```

## 15. 验收目标

### 15.1 trace-only 验收

```text
1. residency_config 存在。
2. weights expert slice 能注册为 region。
3. hot expert prefetch 前输出 residency_request。
4. 超预算时输出 would_deny / residency_evict_candidate。
5. trace_only 下不改变原执行路径，failed_requests=0。
6. residency_summary 中 per-pool bytes 字段存在。
```

### 15.2 weights enforce 验收

```text
1. 设置 weight_gpu_budget 小于 hot+cold expert 总大小。
2. hot prefetch 前能选择 cold weights victim。
3. cold victim offload 后 state 从 GPU -> CPU。
4. hot expert prefetch 后 state 从 CPU/UNKNOWN -> GPU。
5. weight_gpu_resident_bytes <= weight_gpu_budget。
6. 不 offload 当前 active expert。
```

### 15.3 scratch enforce 验收

```text
1. scratch allocation 能进入 residency trace。
2. scratch_gpu_live_bytes 随 allocation/free 增减。
3. profile / warmup 能记录 scratch_peak_bytes。
4. summary 能输出 scratch_reserved_bytes / scratch_margin_bytes。
5. 超 budget 时 enforce 回退 managed 或拒绝 device-direct。
6. trace 中明确提示 scratch_budget 过小，需要调大启动期预留。
7. free 后 scratch_gpu_live_bytes 归还。
8. 不出现 scratch eviction action。
```

### 15.4 KV sizing 验收

```text
1. KV cache 初始化读取 effective_kv_budget_bytes。
2. budget_bytes 能换算成 num_kv_blocks。
3. weight_budget 变小时，kv_budget 随之变化。
4. summary 中记录 kv_budget_bytes / kv_budget_blocks。
```

### 15.5 KV safe offload 验收

```text
1. 只选择 ref_cnt==0 的 free/prefix-cache block。
2. safe block offload 后 state 从 GPU -> CPU。
3. kv_gpu_resident_bytes 下降。
4. prefix/cache hit 或 touch 前能 prefetch-back。
5. unsafe KV offload records == 0。
```

## 16. 推荐开发顺序

在原始六步基础上细化为：

```text
1. 新增 Pool Residency Manager：
   三池预算、状态、trace、summary。

2. 接入 weights region 注册：
   由 weight map / MoE layer 注册 expert slice。

3. 做 weights hot/cold residency enforce：
   cold offload + hot prefetch + 状态更新。

4. 接入 scratch pool：
   把 Stage G live bytes / device-direct admission 纳入统一预算；
   增加 profile/warmup peak 观测和启动期 scratch 预留；
   运行时不做 scratch eviction。

5. 改 KV cache 初始化 sizing：
   按 global - weights - scratch 计算 KV budget。

6. 加 KV block residency metadata：
   block_id -> state / bytes / ptr range。

7. 做 KV safe-block CPU offload / GPU prefetch-back：
   只处理 ref_cnt==0 / free queue / prefix-cache safe blocks。

8. 最后再做 active KV swap / reload：
   需要 scheduler、block table、attention prefetch-back 完整状态机。
```

## 17. 第一版非目标

Stage L 第一版不应尝试一次性解决：

```text
1. 完整 active KV cache CPU swap。
2. 修改 active request block table。
3. dense/shared weights 全量 offload。
4. scratch eviction。
5. CUDA driver 级真实 resident page 精确查询。
6. 跨 pool 自动最优调度。
```

第一版应优先实现可验证闭环：

```text
三池对象注册
三池 GPU resident 账本
weights hot/cold enforce
scratch budget 接入
KV 初始化按预算 sizing
KV safe-block metadata/offload 原型
```

## 18. 一句话总结

Stage L 应把现有系统从“allocation telemetry + action budget”推进到
“三池 GPU residency 管理”。先把 weights / KV / scratch 的 GPU resident bytes
账本做准，再让 cold weights 真实让路，然后让 KV cache 初始化吃到这部分预算，
最后再实现复杂的 KV data offload/prefetch-back。
