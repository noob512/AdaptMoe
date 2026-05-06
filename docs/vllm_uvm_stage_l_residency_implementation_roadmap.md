# vLLM UVM Stage L 后续实施路线：三池显存驻留隔离与权重让位 KV Cache

本文档在 `docs/vllm_uvm_stage_l_pool_residency_manager_plan.md` 的基础上，给出后续可落地的分阶段实施方案。目标不是一次性完成所有复杂能力，而是按可验证闭环逐步实现：

```text
限制 weights / KV cache / runtime scratch 的 GPU 显存驻留占用；
weights 超预算时只处理 weights；
KV 超预算时只处理 KV；
scratch 超预算时只限制 scratch admission；
通过限制 cold expert weights 的 GPU residency，把显存容量让给 KV cache；
最终避免 weights 和 KV cache 在策略层相互干扰。
```

本文档中的“GPU residency”应理解为应用层可管理的估算驻留量，而不是 CUDA driver 级精确 page residency。UVM 页面可能被 driver 被动迁移，因此实现和 trace 中建议使用：

```text
estimated_gpu_resident_bytes
```

而不是声称掌握真实 GPU resident page bytes。

## 1. 当前基础与缺口

### 1.1 已有基础

当前项目已经具备这些能力：

```text
Stage E:
  - weights allocation telemetry
  - weight semantic map
  - MoE routing trace

Stage F:
  - pool registry
  - kv_cache / weights / runtime_scratch 分类和 live bytes telemetry

Stage G:
  - runtime scratch device-direct admission
  - scratch budget over/reject telemetry

Stage H:
  - 基于 weight map + routing trace 的 hot/cold expert plan

Stage I:
  - MoE expert hot prefetch
  - cold expert advise_cpu / prefetch_cpu
  - plan-gated expert action

Stage J:
  - KV runtime block pressure
  - soft admission deny
  - safe prefix-cache metadata eviction

Stage K:
  - high-level action coordinator
  - per-step global / per-pool action bytes budget
```

### 1.2 关键缺口

当前系统还不能保证：

```text
weights 只占用 8GB estimated GPU residency；
KV cache 可以按剩余空间扩大；
weights 超预算时只 offload cold weights；
KV 超预算时只处理 safe KV blocks 或 deny KV admission；
scratch 不抢占 weights/KV，也不被运行时驱逐；
global policy 不误碰 active expert / active KV / active scratch。
```

核心原因：

```text
Stage K 管 action bytes，不管 resident capacity；
Stage F/G 管 allocation live/device-direct live，不管 managed UVM page residency intent；
Stage I 能执行 expert prefetch/offload，但没有 per-pool residency 账本；
Stage J 当前不做 KV data offload/prefetch-back；
KV cache block 数量通常初始化时确定，运行时 offload weights 不会自动增加 KV blocks。
```

因此 Stage L 必须分成“账本、weights enforce、scratch reserve、KV sizing、KV metadata/offload”几个子阶段。

## 2. 总体目标形态

最终目标是形成三池隔离：

```text
weights_pool:
  只管理模型权重，第一优先级是 MoE expert weights。
  超预算时只从 weights_pool 中选择 cold / inactive expert victim。

kv_pool:
  只管理 KV cache blocks。
  超预算时只处理 safe KV blocks，或拒绝新 KV block admission。

scratch_pool:
  管理 runtime temporary buffers / workspace。
  不做 eviction，只做 admission、reserve、fallback managed、telemetry。
```

全局规则：

```text
global_estimated_gpu_resident_bytes
  = weights_estimated_gpu_resident_bytes
  + kv_estimated_gpu_resident_bytes
  + scratch_estimated_gpu_resident_bytes

global_budget_bytes
  >= 三池预算总和 + safety_margin
```

典型目标配置：

```text
总 expert weights allocation live bytes: 30GB
有效 hot expert set estimated GPU residency: 8GB
scratch reserve: 2GB
global managed GPU residency budget: 22GB
KV budget: 22GB - 8GB - 2GB - margin
```

重要区别：

```text
30GB weights allocation live bytes 仍然存在于 managed memory。
Stage L 限制的是 estimated GPU residency，不是 managed allocation 总大小。
```

## 3. 分阶段路线总览

建议按以下阶段推进：

```text
L0: PoolResidencyManager trace-only 账本。
L1: 接入 weights expert slice residency request。
L2: weights hot/cold residency enforce。
L3: scratch residency reserve / Stage G 指标接入。
L4: KV cache sizing，把 weights budget 让出的空间用于 KV blocks。
L5: KV block metadata-only residency。
L6: KV safe-block data offload / prefetch-back 原型。
L7: Stage K + Stage L 联合调度和 A/B 性能验证。
```

第一阶段必须保守：

```text
先证明账本可信，再做 enforce；
先做 weights enforce，再做 KV data offload；
scratch 永远不做 eviction；
global reclaim 第一版只允许推荐 weights cold victim，不真实动 KV/scratch。
```

## 4. L0：PoolResidencyManager Trace-only 账本

### 4.1 目标

L0 只做账本和 trace，不改变 vLLM 运行路径。

要回答：

```text
三池 region 能否注册？
state update 是否正确增减 estimated bytes？
budget request 是否能判断 would_deny？
summary 是否能输出三池状态？
trace-only 是否不会影响 benchmark correctness？
```

### 4.2 新增文件

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/check_stage_l_success.py
docs/vllm_uvm_stage_l_residency_implementation.md
```

### 4.3 环境变量

L0 先实现最小集合：

```text
VLLM_UVM_RESIDENCY_ENABLE=0|1
VLLM_UVM_RESIDENCY_MODE=trace_only
VLLM_UVM_RESIDENCY_TRACE_FILE=/tmp/vllm_uvm_pool_residency_stage_l.jsonl

VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES=0
VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES=0
VLLM_UVM_RESIDENCY_KV_GPU_BYTES=0
VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES=0
VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES=0
VLLM_UVM_RESIDENCY_PRIORITY=kv,scratch,weights
```

预算 `0` 表示不限额。

### 4.4 数据结构

Region：

```text
region_id
pool
ptr
bytes
state
estimated_gpu_resident
evictable
active
last_access_step
last_state_update_ts
metadata
```

State：

```text
unknown
gpu
cpu
prefetching_gpu
prefetching_cpu
released
```

Pool counters：

```text
weights_estimated_gpu_resident_bytes
kv_estimated_gpu_resident_bytes
scratch_estimated_gpu_resident_bytes
global_estimated_gpu_resident_bytes
```

### 4.5 API

```python
get_uvm_pool_residency_manager()

register_uvm_residency_region(
    pool: str,
    region_id: str,
    ptr: int | None,
    bytes: int,
    state: str = "unknown",
    evictable: bool = False,
    metadata: dict | None = None,
)

request_uvm_gpu_residency(
    pool: str,
    region_id: str,
    bytes: int,
    scope_key: str | None = None,
    active_region_ids: set[str] | None = None,
    metadata: dict | None = None,
)

mark_uvm_residency_state(
    region_id: str,
    state: str,
    metadata: dict | None = None,
)

record_uvm_residency_access(
    region_id: str,
    step: int,
    metadata: dict | None = None,
)

get_uvm_residency_snapshot()
```

### 4.6 Request 语义

`request_uvm_gpu_residency()` 判断：

```text
当前 pool estimated bytes + requested bytes 是否超过 pool budget；
当前 global estimated bytes + requested bytes 是否超过 global budget；
如果超预算，trace-only 下 allowed=True 但 would_deny=True；
如果未超预算，allowed=True 且 would_deny=False。
```

L0 不真实选择 victim，只可以输出 would-victim candidate。

### 4.7 Trace action

```text
residency_config
residency_register_region
residency_request
residency_state_update
residency_access
residency_would_evict_candidate
residency_summary
```

### 4.8 验收

`check_stage_l_success.py --self-test`：

```text
1. 构造 weights / KV / scratch regions。
2. 标记部分 region 为 gpu。
3. 设置小 budget。
4. 发起 request。
5. 要求 would_deny > 0。
6. 要求三池 summary 存在。
7. 要求 state_update 正确增减 estimated bytes。
8. 要求没有 offload/prefetch/reclaim action。
```

成功标准：

```text
trace_jsonl_present=True
residency_config_present=True
region_register_records_present=True
all_pools_visible=True
state_updates_present=True
estimated_bytes_accounting_correct=True
would_deny_records_present=True
summary_present=True
no_real_actions=True
```

## 5. L1：Weights Expert Slice Residency Request

### 5.1 目标

把 Stage I expert slices 接入 L0 账本，但仍保持 trace-only。

要证明：

```text
MoE active expert slice 可以注册为 weights region；
hot prefetch 前会 request GPU residency；
cold offload candidate 可以标记为 evictable；
active expert 不会成为 victim candidate；
trace-only 不改变 Stage I 原有行为。
```

### 5.2 修改文件

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_l_success.py
```

### 5.3 Region ID

MoE expert slice：

```text
weights:<layer_id>:<global_expert_id>:<role>
```

metadata：

```text
layer_name
layer_id
global_expert_id
local_expert_id
role
ptr
bytes
plan_hot
plan_cold
active_this_step
heat_score
source=stage_i
```

### 5.4 接入点

在 Stage I `_maybe_prefetch_uvm_expert_weights()` 中：

```text
1. 计算 active expert pairs。
2. 计算 role / ptr / size_bytes。
3. register weights region。
4. record access。
5. request GPU residency。
6. trace-only 下继续执行 Stage I 原逻辑。
```

在 `_maybe_offload_uvm_cold_expert_weights()` 中：

```text
1. 注册 cold expert region。
2. 标记 evictable=True。
3. 如果 cold offload action issued，可以 mark state intent 为 cpu 或 prefetching_cpu。
```

L1 阶段不要求 enforce，不要求真实按 residency deny。

### 5.5 Stage K 与 Stage L 顺序

从 L1 开始，weights prefetch 前建议顺序为：

```text
Stage L: request residency capacity
Stage K: request action bytes budget
Stage I: issue prefetch/offload action
```

trace-only 下两个 gate 都不改变行为。

### 5.6 验收

GPU probe：

```bash
cd workloads/vllm
./check_stage_l_success.py --gpu-run --mode trace_only --weights-only
```

成功标准：

```text
weights_region_register_records > 0
weights_residency_request_records > 0
active_expert_access_records > 0
would_deny_records_present_when_budget_small=True
active_experts_not_victim_candidates=True
failed_requests=0
```

## 6. L2：Weights Hot/Cold Residency Enforce

### 6.1 目标

L2 是第一个真正达成“限制 weights GPU residency”的阶段。

目标场景：

```text
全部 MoE expert weights allocation live bytes 约 30GB；
只允许 hot/effective expert set estimated GPU residency 约 8GB；
cold expert 如果驻留 GPU 超过 budget，则在 weights pool 内 offload；
hot expert prefetch 前，如果预算不足，只选择 cold weights victim；
不碰 KV，不碰 scratch。
```

### 6.2 新增环境变量

```text
VLLM_UVM_RESIDENCY_MODE=trace_only|enforce
VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES=<bytes>
VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE=unknown|cpu|gpu
VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET=0|1
VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD=0|1
VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP=0
VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_POLICY=cold_plan_lru|cold_plan_heat
```

### 6.3 Victim 规则

weights victim 必须满足：

```text
pool == weights
state in (gpu, prefetching_gpu)
evictable == True
region_id not in active_region_ids
不是当前 step active expert
属于 Stage H offload_plan 或 heat_score 低
```

排序：

```text
1. Stage H offload_plan 命中优先
2. heat_score 低优先
3. last_access_step 旧优先
4. bytes 大优先
```

严格禁止：

```text
offload 当前 topk active expert；
offload dense/shared weights；
offload 没有 ptr 的 region；
因为 KV 压力直接 offload KV 外的对象，除非 global policy 显式选择 weights cold victim。
```

### 6.4 Enforce 流程

hot prefetch 前：

```text
1. request weights GPU residency。
2. 如果预算够：
   - mark prefetching_gpu 或 gpu intent。
   - 继续 Stage K action budget。
   - issue prefetch_range_to_device。

3. 如果预算不够：
   - 从 weights pool 中选择 cold victims。
   - 对 victims issue prefetch_range_to_cpu 或 advise_cpu。
   - mark victims prefetching_cpu -> cpu。
   - 释放 estimated weights GPU bytes。
   - 再 request hot region。
   - 如果仍不足：
       trace residency_deny；
       skip hot prefetch；
       不影响正常 compute，最多退化成 UVM fault。
```

是否同步：

```text
trace_only:
  不同步，只记录 intent。

enforce + VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD=1:
  offload victim 后同步确认，再释放 estimated bytes。

enforce + sync=0:
  conservative 模式下 prefetching_cpu 仍计入 GPU，直到下一次确认或 summary checkpoint。
```

建议第一版 enforce 默认：

```text
sync_offload=1
max_victims_per_step 小一些
只处理 Stage H offload_plan 的 cold experts
```

### 6.5 与 KV 容量的关系

L2 本身只能限制 weights estimated residency，不能自动增加 KV blocks。

它提供的是前置条件：

```text
weights_pool_budget = 8GB
weights estimated GPU residency 被限制在 8GB 附近
因此初始化 KV cache 时可以把剩余容量划给 KV
```

真正扩大 KV cache 容量在 L4。

### 6.6 验收

```bash
./check_stage_l_success.py --gpu-run --weights-enforce \
  --weight-gpu-bytes 8589934592 \
  --require-weight-victim
```

成功标准：

```text
weight_budget_bytes ~= 8GB
weights_estimated_gpu_resident_bytes <= weight_budget_bytes
cold_weight_victim_records > 0
weight_offload_issued_records > 0
hot_prefetch_request_records > 0
active_weight_victim_records == 0
kv_action_records == 0
scratch_eviction_records == 0
failed_requests == 0
```

## 7. L3：Scratch Residency Reserve / Stage G 接入

### 7.1 目标

scratch 不做 eviction。L3 目标是把 scratch 作为预留容量池纳入统一报告和预算：

```text
scratch estimated/live GPU bytes 可见；
scratch peak 可见；
scratch budget 可配置；
超预算时限制 scratch device-direct admission 或回退 managed；
不驱逐 scratch；
不通过 scratch reclaim 给 weights/KV 让路。
```

### 7.2 实现方式

优先在 C++ allocator 侧继续维护 scratch counters：

```text
scratch_pool_device_direct_live_bytes
scratch_pool_device_direct_peak_live_bytes
scratch_pool_budget_over_allocs
scratch_pool_budget_reject_allocs
```

新增 Stage L 风格 summary 字段：

```text
scratch_estimated_gpu_resident_bytes
scratch_gpu_budget_bytes
scratch_peak_bytes
scratch_reserved_bytes
scratch_margin_bytes
scratch_eviction_records=0
```

第一版不要从 C++ allocator hot path 调 Python `PoolResidencyManager`。

### 7.3 预算公式

```text
scratch_budget =
    measured_peak_scratch
    + scratch_margin
    + backend_fragmentation_margin
```

如果无 profile 结果：

```text
使用显式 VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES；
summary 输出 observed_peak，提示下次调参。
```

### 7.4 验收

```text
scratch_summary_present=True
scratch_peak_bytes_present=True
scratch_budget_bytes_present=True
scratch_over_budget_records_present_when_budget_small=True
scratch_device_direct_reject_or_managed_fallback_in_enforce=True
scratch_eviction_records == 0
failed_requests == 0
```

## 8. L4：KV Cache Sizing，让 weights 让出的空间变成 KV blocks

### 8.1 目标

这是实现“限制 weights 后增大 KV cache 容量”的关键阶段。

核心事实：

```text
运行时 offload cold weights 不会自动增加 KV block 数；
KV cache blocks 通常在 vLLM 初始化 / profiling 阶段决定；
因此必须在 KV cache 初始化时使用 Stage L 预算。
```

### 8.2 预算公式

显式预算模式：

```text
kv_gpu_budget_bytes = VLLM_UVM_RESIDENCY_KV_GPU_BYTES
```

派生预算模式：

```text
effective_kv_budget_bytes =
    global_gpu_budget_bytes
    - weight_gpu_budget_bytes
    - scratch_gpu_budget_bytes
    - safety_margin_bytes

kv_budget_blocks =
    effective_kv_budget_bytes // bytes_per_block
```

目标例子：

```text
global_gpu_budget_bytes = 22GB
weight_gpu_budget_bytes = 8GB
scratch_gpu_budget_bytes = 2GB
safety_margin_bytes = 1GB
effective_kv_budget_bytes = 11GB
```

如果原来 weights 不限制，30GB expert weights 会挤压 KV；限制 weights residency 后，KV 初始化预算可以明确得到剩余空间。

### 8.3 实现位置

应定位 vLLM 生成 KV cache config / 计算 `num_gpu_blocks` 的路径，而不是 allocator：

```text
vLLM GPU worker profiling
cache_config.num_gpu_blocks 计算路径
KVCacheConfig / kv_cache_tensors 创建路径
KVCacheManager 初始化前
```

实现策略：

```text
1. 读取 Stage L budgets。
2. 获取 bytes_per_block。
3. 计算 residency_kv_budget_blocks。
4. 对 vLLM 原 num_gpu_blocks 做 min/max clamp 或 override。
5. 输出 trace：
   - original_num_gpu_blocks
   - residency_kv_budget_bytes
   - residency_kv_budget_blocks
   - final_num_gpu_blocks
```

第一版可以先不自动扩容，只做：

```text
budget_bytes -> budget_blocks report；
Stage J runtime admission 使用 budget_blocks；
check 脚本证明 weight_budget 变化会影响 derived kv_budget。
```

第二版再改实际 `num_gpu_blocks`。

### 8.4 和 weights enforce 的关系

正确启动顺序：

```text
1. 确定 weight_gpu_budget，例如 8GB。
2. 确定 scratch_gpu_budget，例如 profile peak + margin。
3. 计算 KV budget。
4. 初始化 KV cache blocks。
5. 运行时 Stage I/L2 保证 weights estimated residency 不超过 weight budget。
```

如果只做第 5 步，不做第 4 步，KV 容量不会真正增大。

### 8.5 验收

```text
residency_kv_sizing_config_present=True
weight_budget_bytes_present=True
scratch_budget_bytes_present=True
derived_kv_budget_bytes_present=True
derived_kv_budget_blocks_present=True
final_num_gpu_blocks_present=True
weight_budget_smaller_makes_kv_budget_larger=True
```

后续真实改 num blocks 后增加：

```text
final_num_gpu_blocks_changes_with_budget=True
benchmark_no_failed_requests=True
kv_fault_ratio_not_regressed_unexpectedly=True
```

## 9. L5：KV Block Metadata-only Residency

### 9.1 目标

先把 KV block 纳入账本，但不迁移 KV data。

要证明：

```text
每个 KV block 有 region_id；
每个 block 有 bytes_per_block；
block state 可记录；
Stage J runtime pressure 可用 residency 账本而非只用 free_blocks；
safe candidates 可 trace；
不做真实 KV offload。
```

### 9.2 Region ID

```text
kv:<group_id>:<block_id>
```

metadata：

```text
block_id
group_id
bytes_per_block
ref_cnt
is_prefix_cache
in_free_queue
block_hash
state
source=stage_j
```

### 9.3 接入点

```text
KVCacheManager 初始化后注册 blocks。
UvmKvRuntimePolicy.should_allow_allocation() 更新 projected KV residency。
BlockPool candidate snapshot 时附带 residency state。
```

第一版如果拿不到 raw pointer：

```text
ptr=None
state=gpu 或 unknown
offload_capable=False
```

### 9.4 验收

```text
kv_region_register_records > 0
kv_region_bytes_per_block_positive=True
kv_residency_request_records > 0
kv_safe_candidate_records > 0
kv_data_offload_records == 0
unsafe_kv_candidate_records == 0
failed_requests == 0
```

## 10. L6：KV Safe-block Data Offload / Prefetch-back 原型

### 10.1 目标

这不是 Stage L 第一版目标。只有在 L5 稳定后才做。

目标：

```text
只对 ref_cnt==0 / free queue / prefix-cache safe blocks 做 data offload；
使用前 prefetch-back；
不处理 active request blocks；
不修改 active block table。
```

### 10.2 前置条件

必须先具备：

```text
block_id -> raw KV pointer range 映射；
明确每个 block 的 bytes 和 group/layer layout；
prefix/cache hit 或 block touch 前的安全 prefetch-back hook；
stream synchronization 或 dependency 方案；
unsafe block 检查；
rollback / fail-open 机制。
```

### 10.3 执行规则

offload candidate：

```text
pool == kv
state == gpu
ref_cnt == 0
in_free_queue == True
ptr != None
offload_capable == True
```

prefetch-back trigger：

```text
CPU-resident block 即将被 reuse；
prefix cache hit 即将 touch block；
attention kernel 前发现 block state != gpu。
```

### 10.4 第一版限制

```text
不 offload active request blocks；
不做异步复杂流水；
不让 KV offload 失败影响请求正确性；
失败时回退不 offload，只记录。
```

## 11. L7：Stage K + Stage L 联合调度

### 11.1 分工

```text
Stage L:
  管 resident capacity。
  判断某个对象能否成为 GPU resident。
  选择同 pool victim。

Stage K:
  管 action bandwidth。
  判断本 step 还能发起多少 bytes prefetch/offload。
```

### 11.2 调用顺序

weights hot prefetch：

```text
1. Stage L request weights GPU residency。
2. Stage L 如需 victim，只选择 weights cold expert。
3. Stage K request offload action bytes。
4. issue victim offload。
5. Stage K request prefetch action bytes。
6. issue hot prefetch。
```

KV admission：

```text
1. Stage L / Stage J 计算 KV projected residency。
2. 如果 KV pool 超预算，只找 KV safe candidates。
3. Stage K request KV action bytes。
4. 如果无法 reclaim，则 Stage J soft deny admission。
```

scratch allocation：

```text
1. Stage G/L 判断 scratch budget。
2. Stage K 可选记录 scratch action/admission。
3. budget 不足时回退 managed，不 eviction。
```

### 11.3 强隔离规则

必须在代码和 trace 中强制：

```text
weights local pressure -> only weights victim
KV local pressure -> only KV victim / KV admission deny
scratch local pressure -> only scratch admission/fallback
global pressure -> first trace-only; enforce only allowed to choose configured safe pool
```

第一版 global enforce 建议：

```text
只允许 weights cold expert 作为 global reclaim victim；
KV 和 scratch 只 trace，不真实 reclaim。
```

## 12. 关键验收矩阵

### 12.1 隔离性

```text
weights_over_budget:
  weight_victim_records > 0
  kv_victim_records == 0
  scratch_eviction_records == 0

kv_over_budget:
  kv_candidate_records > 0
  weight_victim_records == 0
  scratch_eviction_records == 0

scratch_over_budget:
  scratch_fallback_or_reject_records > 0
  weight_victim_records == 0
  kv_victim_records == 0
  scratch_eviction_records == 0
```

### 12.2 weights 让位 KV

必须证明两件事：

```text
1. weights estimated GPU residency 被限制在 weight budget。
2. KV budget / KV blocks 使用 global - weights - scratch 后的剩余容量。
```

只证明第 1 点不够，因为 KV blocks 不会自动增加。

### 12.3 安全性

```text
active_weight_victim_records == 0
active_kv_victim_records == 0
scratch_eviction_records == 0
failed_requests == 0
runner_log_clean == True
```

### 12.4 性能解释

记录：

```text
TPOT
TTFT
ITL p99
output tok/s
weight_offload_bytes
weight_prefetch_bytes
kv_budget_blocks
scratch_peak_bytes
coordinator_denies
residency_would_denies
```

性能退化时必须能解释是：

```text
Python trace overhead
sync offload overhead
prefetch/offload bandwidth contention
KV block 数变化
UVM fault 变化
```

## 13. 推荐脚本

### 13.1 L0/L1 self-test

```bash
./check_stage_l_success.py --self-test
```

验证三池账本。

### 13.2 weights enforce

```bash
./check_stage_l_success.py \
  --gpu-run \
  --weights-enforce \
  --weight-gpu-bytes 8589934592 \
  --require-weight-victim
```

### 13.3 KV sizing

```bash
./check_stage_l_success.py \
  --self-test \
  --kv-sizing \
  --global-gpu-bytes 23622320128 \
  --weight-gpu-bytes 8589934592 \
  --scratch-gpu-bytes 2147483648
```

### 13.4 isolation A/B

```bash
./stage_l_residency_ab_sweep.py
```

建议 variants：

```text
baseline_no_residency
trace_only_residency
weight_budget_8gb_trace
weight_budget_8gb_enforce
weight_budget_8gb_enforce_kv_sizing
scratch_budget_enforce
kv_metadata_trace
```

输出：

```text
variant
weight_budget_gb
weights_estimated_gpu_resident_gb
kv_budget_gb
kv_budget_blocks
scratch_peak_gb
weight_victims
kv_victims
scratch_evictions
failed_requests
TPOT
TTFT
out_tok_s
```

## 14. 最推荐的近期实现目标

近期不要直接碰 KV data offload。建议先完成：

```text
1. L0 PoolResidencyManager trace-only。
2. L1 Stage I weights region request。
3. L2 weights hot/cold enforce，限制 expert weights estimated GPU residency。
4. L4 的 KV sizing report，先证明 8GB weight budget 会导出更大的 KV budget。
```

这四步完成后，就能比较可信地证明：

```text
我们可以把 30GB expert weights 的 managed allocation live bytes，
限制成约 8GB estimated GPU residency；
这部分让出的 resident capacity 可以作为 KV cache 初始化预算；
weights 压力只处理 weights；
KV 压力只处理 KV admission/candidates；
scratch 不参与 eviction。
```

## 15. 最终判断

实现“weights 和 KV cache 不相互干扰”的关键不是简单增加一个 eviction 函数，而是建立三层机制：

```text
第一层：Pool registry
  知道对象属于 weights / KV / scratch 哪个 pool。

第二层：Residency accounting
  知道每个 pool 估算占用了多少 GPU residency。

第三层：Pool-local action executor
  weights 只执行 weights cold expert offload；
  KV 只执行 KV safe-block action 或 admission deny；
  scratch 只执行 admission/fallback。
```

对于用户关心的例子：

```text
全部 expert weights 约 30GB；
有效 hot expert set 约 8GB；
希望限制 weights GPU residency 到 8GB；
剩余显存给 KV cache。
```

推荐落地方式是：

```text
先用 Stage L2 限制 weights estimated GPU residency；
再用 Stage L4 在 KV cache 初始化时按 global - 8GB - scratch 计算 KV budget；
最后才考虑 KV safe-block data offload / prefetch-back。
```

这样才能真正实现：

```text
weights 不挤占 KV；
KV 不错误驱逐 weights；
weights eviction 只发生在 weights pool 内；
KV eviction 只发生在 KV pool 内；
scratch 不被运行时驱逐。
```
