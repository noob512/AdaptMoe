# Stage N 原生 UVM Block 分池配额方案

本文档是 Stage N 分池管理方案的最新版本。早期 region 级全表扫描、全局 2MB bucket、按语义 active/hot/cold 做复杂 victim 选择的路线已经不再作为主线。当前目标是：

```text
尽量继承原生 UVM 的 range/block 管理关系；
在 Stage N 上增加 KV / weights / scratch 的容量控制；
只保证各 pool 的 resident 容量稳定；
超预算时只从同 pool 的 resident queue 中选择 victim；
第一版先做 post-fault correction，稳定后再推进 pre-fault hard quota。
```

## 核心结论

推荐方案不是把 GPU 物理显存切成固定 2MB 块，也不是把 Stage N block 定义成裸 `align_down(address, 2MB)` 的全局 bucket。推荐方案是：

```text
vLLM allocator/register_region 提供 pool-tagged range；
UVM driver 在 fault/access 命中时查 tagged range；
再查原生 UVM managed_range + native block_index；
lazy materialize 一个 Stage N phase_n_block wrapper；
phase_n_block 继承原生 UVM block/range identity；
每个 pool 维护 resident queue 和 quota counters；
超预算时只从同 pool queue 中 FIFO/LRU 驱逐 victim。
```

简化成一句话：

```text
range 负责 pool 归属；
原生 UVM block 负责 residency 管理粒度；
Stage N block wrapper 负责 per-pool accounting 和 eviction queue。
```

## 当前已实现基础

当前项目已经有一批可复用能力，不需要推倒重来。

### vLLM 应用层

已实现：

- `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`
  - 通过 `cudaMallocManaged` 提供 UVM allocation。
  - 能分类 `WeightPersistent`、`KvPersistent`、`RuntimeScratch`、`RuntimeWorkspace`、`WarmupWorkspace`。
  - 能把 allocation 映射到 `kv`、`weights`、`scratch`。
  - 能向 driver 注册 region。
  - 能输出 scratch tag/untag/reject trace。
- `workloads/vllm/vllm/vllm/device_allocator/uvm.py`
  - 已有 ctypes wrapper：`driver_set_pool_budget`、`driver_get_pool_state`、`driver_reclaim_pool`。
- `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py`
  - 已能读取 pool budget 环境变量。
  - 已能下发 driver pool budget。
  - 已能记录 driver pool state。
- `run_stage_n_*` 脚本
  - 已有 tag-only smoke、three-pool validation、P7 A/B、hard partition proof 等实验脚本。

这些能力在新方案中继续保留。vLLM 仍然负责告诉 driver 哪段 VA 属于哪个 pool。

### UVM 系统层

已实现：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
  - 已定义 pool id：`KV`、`WEIGHT`、`SCRATCH`。
  - 已定义模式：`TRACE_ONLY`、`SOFT`、`ENFORCE`。
  - 已定义 `SET_POOL_BUDGET`、`GET_POOL_STATE`、`UPDATE_REGION_STATE`、`QUERY_REGION`、`QUERY_REGION_BY_VA`、`RECLAIM_POOL`。
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h`
  - 已有 `phase_f_regions`、`phase_f_region_ranges`、`phase_n_pools`。
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
  - 已实现 region 注册、pool budget、pool state、manual reclaim、auto reclaim hook。
  - 已有 migrate-to-CPU helper 调用路径。
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`
  - 已在 replayable fault batch 外层接入 Stage N auto hook。

新方案是在这些基础上改变内部管理粒度：从 region full scan 迁移到 lazy materialized native-UVM-block wrapper。

## 为什么放弃 region full scan

当前 region 级方案的主要路径是：

```text
GET_POOL_STATE / RECLAIM_POOL:
    scan phase_f_regions
    找出目标 pool 的 region
    对每个 region 做 residency query
    统计 candidate/protected/evictable
    排 victim
    可选 migrate-to-CPU
```

问题是：

```text
O(region_count) 全表扫描；
每个 region 可能触发昂贵 residency query；
auto reclaim 接在 fault path 后会放大成本；
vLLM 大模型场景 region 数量很容易膨胀；
小规模 hard partition probe 也可能陷入长时间 fault/reclaim storm。
```

所以后续不再继续优化 region full scan。region 仍然作为 pool tag 来源存在，但 reclaim/accounting 走 block queue fast path。

## 原生 UVM Block 语义

UVM 中有：

```c
#define UVM_VA_BLOCK_BITS 21
#define UVM_VA_BLOCK_SIZE (1ULL << UVM_VA_BLOCK_BITS)
```

含义是：原生 UVM block 最大为 2MB，并且不会跨 2MB 边界。但它不是任意全局 2MB bucket。原生 UVM 的层级是：

```text
uvm_va_space
  -> uvm_va_range
      -> uvm_va_block
```

原生 `uvm_va_block_t` 属于单个 parent managed range，创建时也要求：

```text
block.start >= managed_range.start
block.end   <= managed_range.end
block.size  <= UVM_VA_BLOCK_SIZE
```

因此必须区分：

```text
原生 UVM va_block:
  identity ~= managed_range + block_index
  不跨 parent uvm_va_range
  size <= 2MB

Stage N 全局 2MB bucket:
  identity = align_down(address, 2MB)
  可能把不同 UVM range 落在同一 2MB window 的地址合并
  这是自定义 quota bucket，不是原生 UVM block
```

当前推荐方案选择前者：

```text
phase_n_block identity = managed_range + native_block_index
```

不推荐把 `phase_n_block` 简单定义成裸全局 2MB aligned bucket。

## 新方案目标与非目标

### 目标

第一阶段目标是容量稳定：

```text
kv_resident_blocks <= kv_budget_blocks + allowed_post_fault_slack
weights_resident_blocks <= weights_budget_blocks + allowed_post_fault_slack
scratch_resident_blocks <= scratch_budget_blocks + allowed_post_fault_slack
```

并保证：

```text
KV 超预算时只驱逐 KV block
weights 超预算时只驱逐 weights block
scratch 超预算时只驱逐 scratch block
cross_pool_victim_errors = 0
fault_path_region_scan_records = 0
```

中长期目标是 Pre-fault Hard Quota：

```text
在 make-resident admission 之前检查 pool quota；
如果 pool 已满，先从同 pool 驱逐 victim；
驱逐成功后再允许当前 fault block resident；
最终 max_over_blocks = 0。
```

### 非目标

第一版不做：

- 不做 GPU 物理显存 2MB chunk 预切分。
- 不接管 UVM GPU PMM chunk allocator。
- 不在最内层 va_block lock 环境里跨 block reclaim。
- 不依赖 active/hot/cold 语义。
- 不把 dense weights、MoE weights、active weights 做复杂区分。
- 不保证第一版任何时刻都绝不短暂超预算。

## 数据结构设计

### `uvm_va_space_t`

建议新增：

```c
struct radix_tree_root phase_n_blocks;
uvm_phase_n_block_pool_t phase_n_block_pools[UVM_PHASE_N_POOL_STATE_MAX];
```

职责：

- `phase_n_blocks`：按 `managed_range + native_block_index` 或等价稳定 key 索引。
- `phase_n_block_pools`：每个 pool 的 resident queue 和 counters。

不建议新增以裸 2MB global address 为唯一 key 的 block range tree。fault 地址应先走 tagged range，再走原生 UVM block identity。

### `uvm_phase_n_block_t`

建议结构：

```c
typedef struct uvm_phase_n_block
{
    uvm_va_range_managed_t *managed_range;
    NvU64 tagged_region_id;
    NvU64 native_block_index;
    NvU64 block_start;
    NvU64 block_end;
    NvU64 covered_start;
    NvU64 covered_end;
    NvU32 pool;
    NvU32 flags;
    NvBool mixed_pool;
    NvBool gpu_resident;
    NvBool queued;
    NvBool in_reclaim;
    NvU64 resident_epoch;
    NvU64 last_access_epoch;
    struct list_head queue_node;
} uvm_phase_n_block_t;
```

字段含义：

- `managed_range/native_block_index`：继承原生 UVM block identity。
- `tagged_region_id`：触发 materialize 的 Stage N tagged range。
- `block_start/block_end`：原生 UVM block 实际 VA 边界，大小不超过 2MB。
- `covered_start/covered_end`：Stage N tagged range 与原生 UVM block 的交集。
- `pool`：KV / weights / scratch。
- `mixed_pool`：同一个原生 UVM block 内出现多个 Stage N pool tag。
- `gpu_resident`：Stage N 的保守 resident boolean。
- `queued`：是否在 per-pool resident queue 中。
- `in_reclaim`：避免重复选择同一 victim。
- `last_access_epoch`：LRU 使用。

第一版可以按 block boolean 计费：

```text
只要该原生 UVM block 有过 GPU fault/access，就计为 resident 1 block。
```

这比 resident bytes 精确计费保守，但能避免 fault path 查询 page mask。

### `uvm_phase_n_block_pool_t`

建议结构：

```c
typedef struct uvm_phase_n_block_pool
{
    NvU64 budget_bytes;
    NvU64 budget_blocks;
    NvU64 resident_blocks;
    NvU64 materialized_blocks;
    NvU64 mixed_blocks;
    NvU64 reclaim_attempts;
    NvU64 reclaim_success;
    NvU64 reclaim_failures;
    NvU64 same_pool_victims;
    NvU64 cross_pool_victim_errors;
    NvU64 quota_over_events;
    NvU64 max_over_blocks;
    struct list_head resident_queue;
} uvm_phase_n_block_pool_t;
```

## Tagged Range 与 Lazy Materialization

`register_region` 只负责给 VA range 打 pool tag：

```text
uvm_api_phase_f_register_region(region):
    保存原始 region，兼容旧查询
    插入 phase_f_region_ranges
    记录 pool / flags / role / region_id
    记录 2MB 对齐状态
    不立即创建 phase_n_block
```

fault/access 命中时再 lazy materialize：

```text
phase_n_auto_hook(address):
    tagged_region = range_tree_find(phase_f_region_ranges, address)
    if tagged_region missing:
        skip=no_region
        return

    uvm_va_range = uvm_va_range_find(va_space, address)
    managed_range = uvm_va_range_to_managed_or_null(uvm_va_range)
    if managed_range missing:
        skip=no_managed_range
        return

    native_index = uvm_va_range_block_index(managed_range, address)
    native_block = uvm_va_range_block_create(managed_range, native_index)
    if native_block missing:
        skip=no_native_block
        return

    block = find_phase_n_block(managed_range, native_index)
    if block missing:
        block = create_phase_n_block(managed_range, native_index)
        block.pool = tagged_region.pool
        block.block_start/end = native_block.start/end
        block.covered_start/end = intersection(tagged_region, native_block)
        detect Stage N pool-tag overlap inside this native block
        if multiple pools overlap:
            block.mixed_pool = true
            pool.mixed_blocks++

    mark accessed/resident
    enqueue into pool resident queue
    if pool over budget:
        reclaim same-pool victims
```

这样保留了原生 UVM 的按需 block 创建思路，同时让 Stage N 获得自己的 per-pool queue/counter。

## Mixed Pool 定义与处理

`mixed_pool` 的最新定义是：

```text
同一个原生 UVM block 覆盖的 VA 范围内，
Stage N 自己注册了多个不同 pool 的 tagged subrange。
```

它不是指：

```text
一个原生 uvm_va_block_t 同时拥有多个 uvm_va_range_t。
```

原生 UVM block 属于单个 parent managed range，这一点不应被破坏。

处理策略：

```text
driver 在 block materialize 时检测 mixed_pool；
mixed_pool block 不进入强制 reclaim；
记录 phase_n_block_mixed_pool；
hard proof 要求 mixed_pool_blocks = 0。
```

如果出现 mixed pool，优先修 allocator/tagging，而不是在 driver 内做复杂 sub-block/page-mask 策略。

## Allocator 2MB 对齐与 Pool Arena

虽然 `phase_n_block` 不再是裸全局 2MB bucket，allocator 对齐仍然重要。UVM 注释也说明，用户态按 `UVM_VA_BLOCK_SIZE` 自然对齐分配能获得更好效率。

当前 `uvm_malloc` 仍是直接：

```cpp
cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
```

它不能显式保证：

```text
返回 pointer 2MB 对齐；
registered range 长度按 2MB round up；
不同 pool 不共享同一个原生 UVM block。
```

推荐后续补 allocator 层：

```text
KV / weights / scratch 分配时：
1. 判断 pool 类型；
2. 按 2MB round up 请求大小；
3. usable pointer 尽量 2MB 对齐；
4. 注册给 driver 的 range 尽量按原生 UVM block 边界覆盖；
5. free 时由 aligned_ptr 找回 raw_ptr，再 cudaFree(raw_ptr)。
```

更稳的方式是 per-pool arena/suballocator：

```text
KV arena       -> 只切 KV allocation
weights arena  -> 只切 weight allocation
scratch arena  -> 只切 scratch allocation
```

目标是让一个原生 UVM block 内只出现一个 Stage N pool tag。driver 仍保留 `mixed_pool_blocks` 检测，它是 allocator/tagging 失败的验收信号。

## Resident Queue 与 Reclaim

每个 pool 维护一个 resident queue：

```text
phase_n_block_pools[pool].resident_queue
phase_n_block_pools[pool].resident_blocks
phase_n_block_pools[pool].budget_blocks
```

fault/access 命中 block 后：

```text
if block.gpu_resident == false:
    block.gpu_resident = true
    block.queued = true
    pool.resident_blocks++
    list_add_tail(block, pool.resident_queue)

if policy == LRU:
    list_move_tail(block, pool.resident_queue)

while pool.resident_blocks > pool.budget_blocks + allowed_slack:
    victim = pop same-pool victim
    migrate victim to CPU
```

第一版 queue policy：

```text
uvm_phase_n_block_queue_policy=fifo|lru
```

推荐默认 FIFO：

- fault path 更轻。
- 验证 same-pool-only 更简单。
- 后续再开启 LRU。

victim 选择只允许从当前 pool queue 中取：

```text
skip in_reclaim block
skip mixed_pool block
skip stale/dead block
```

迁移策略：

```text
如果 block 已绑定原生 UVM identity 且没有 mixed_pool：
    migrate block_start..block_end
否则：
    migrate covered_start..covered_end
```

成功后：

```text
block.gpu_resident = false
block.queued = false
pool.resident_blocks--
pool.same_pool_victims++
```

失败后：

```text
pool.reclaim_failures++
记录 backoff
保留 over-budget violation
```

## GET_POOL_STATE 与 RECLAIM_POOL

### GET_POOL_STATE

block quota 开启时：

```text
GET_POOL_STATE(pool):
    直接读取 phase_n_block_pools[pool] counters
    不做 region full scan
```

核心字段：

```text
budget_blocks
resident_blocks
materialized_blocks
mixed_blocks
quota_over_events
max_over_blocks
same_pool_victims
cross_pool_victim_errors
reclaim_success
reclaim_failures
```

兼容旧字段时可以映射：

```text
resident_gpu_bytes ~= resident_blocks * UVM_VA_BLOCK_SIZE
candidate_bytes    ~= resident_blocks * UVM_VA_BLOCK_SIZE
victim_count        ~= queued victim count
```

但 proof 不应再依赖 region candidate bytes。

### RECLAIM_POOL

block quota 开启时：

```text
RECLAIM_POOL(pool):
    从 phase_n_block_pools[pool].resident_queue pop victim
    只允许 same-pool victim
    调 migrate-to-CPU helper
    更新 block/pool counters
```

保留旧 region reclaim 路径作为 fallback：

```text
uvm_phase_n_region_scan_reclaim_enable=1
```

默认关闭。

## Auto Hook 路径

继续复用当前安全点：

```text
uvm_gpu_replayable_faults.c
  -> phase_n_auto_reclaim_after_replayable_batch()
  -> uvm_phase_n_auto_reclaim_outer_hook()
```

hook 内部改为：

```text
address
  -> Stage N tagged range
  -> native UVM block identity
  -> phase_n_block wrapper
  -> pool resident queue
  -> same-pool eviction
```

不再：

```text
address
  -> region
  -> uvm_phase_n_reclaim_pool_execute()
  -> scan all regions
```

## Post-Fault 与 Pre-Fault 路线

### 第一阶段：Post-Fault Bounded Quota

第一阶段不打断 UVM 原生 fault/make-resident 流程：

```text
UVM 先把 fault block 拉到 GPU；
outer hook 再更新 Stage N block residency；
如果 pool 超预算，再从同 pool queue reclaim victim。
```

验收允许：

```text
max_over_blocks <= allowed_post_fault_slack
```

建议第一版：

```text
allowed_post_fault_slack = 1 或 fault batch size 上限
```

优点：

- 更接近原生 UVM。
- 不在 va_block lock 内做跨 block reclaim。
- 先验证 block queue、same-pool reclaim、counter/accounting 是否可靠。

### 第二阶段：Pre-Fault Hard Quota

Post-fault correction 稳定后，把检查点前移到 admission 之前：

```text
fault address -> tagged range -> native UVM block identity -> pool
if pool full:
    reclaim same-pool victim first
if reclaim success:
    allow make-resident
else:
    deny/block/fallback with explicit evidence
```

建议模式：

```text
UVM_PHASE_N_PRE_FAULT_QUOTA_MODE=trace|soft|enforce
```

- `trace`：只记录 would-check / would-reclaim / would-block。
- `soft`：尝试 pre-admission reclaim，失败后回退 UVM 原行为。
- `enforce`：要求 reclaim-to-budget 或产生明确 deny/block 证据。

最终验收：

```text
pre_fault_admission_records > 0
pre_fault_same_pool_reclaim_records > 0
pre_fault_cross_pool_victim_errors = 0
max_over_blocks = 0
failed_requests = 0
无 fault replay hang
TTFT/TPOT 无明显长尾退化
```

## 需要修改的文件

### 系统层

`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h`

- 新增 `phase_n_blocks`。
- 新增 `phase_n_block_pools`。
- 初始化/销毁 block wrapper 与 pool queue。

`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`

- `REGISTER_REGION` 只记录 tagged range 和对齐状态。
- 新增 `phase_n_find_or_create_block_wrapper()`。
- 使用 `managed_range + native_block_index` 作为 block identity。
- `GET_POOL_STATE` 优先返回 block counters。
- `RECLAIM_POOL` 优先从 same-pool resident queue reclaim。
- 保留 region scan fallback，但默认关闭。

`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

- 保留 outer hook 安全点。
- hook 内部走 block queue fast path。
- 记录 post-fault quota over/reclaim trace。

`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`

第一版可以复用现有 ioctl。后续可扩展：

- block quota counters
- queue policy
- block state query
- mixed block count
- same-pool victim count
- pre-fault quota mode

### 应用层

`workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

- 增加 2MB 对齐 / padding 方案。
- 优先实现 per-pool arena/suballocator。
- 输出 allocator alignment trace。
- free 时维护 aligned pointer 到 raw pointer 的映射。

`workloads/vllm/vllm/vllm/device_allocator/uvm.py`

- 兼容新增 counters。
- 暴露 block quota state。

`workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py`

- 输出 block quota config/state。
- 记录 `phase_n_block_materialize`、`phase_n_block_reclaim`、`phase_n_block_mixed_pool`。

### 脚本与验收

新增或更新：

- `workloads/vllm/run_stage_n_block_quota_smoke.sh`
- `workloads/vllm/run_stage_n_p8_block_quota_probe.sh`
- `workloads/vllm/check_stage_n_block_quota.py`
- `workloads/vllm/run_stage_n_p7_ab_sweep.sh`

## 实施阶段

### P8a：Tagged Range + Lazy Block Trace

目标：

```text
register_region 只记录 tagged range；
fault/access 命中时 lazy materialize Stage N block wrapper；
不改变 reclaim 行为。
```

验收：

```text
tagged_range_register_records > 0
block_materialize_records > 0
kv_materialized_block_records > 0
weight_materialized_block_records > 0
scratch_materialized_block_records > 0
mixed_pool_blocks = 0
fault_path_region_scan_records = 0
```

### P8b：Manual Same-Pool Block Reclaim

目标：

```text
RECLAIM_POOL 不扫 region；
从 per-pool resident queue 选择 victim；
只迁移 same-pool block。
```

验收：

```text
manual_block_reclaim_kv_success > 0
manual_block_reclaim_weights_success > 0
manual_block_reclaim_scratch_success > 0
same_pool_victim_records > 0
cross_pool_victim_errors = 0
region_scan_records = 0
```

### P8c：Auto Post-Fault Block Quota

目标：

```text
outer hook 自动更新 block residency；
pool 超预算时自动 same-pool reclaim；
不做 region full scan。
```

验收：

```text
phase_n_block_auto_hook_records > 0
phase_n_block_over_budget_records > 0
phase_n_block_reclaim_success_records > 0
same_pool_victim_records > 0
cross_pool_victim_errors = 0
fault_path_region_scan_records = 0
max_over_blocks <= allowed_post_fault_slack
failed_requests = 0
```

### P8d：Capacity Stability Proof

实验：

```text
synthetic 2MB/native-block smoke
three-pool same-pool proof
vLLM mini probe
```

验收：

```text
block_quota_pass = 1
mixed_pool_blocks = 0
same_pool_only = 1
resident_blocks <= budget_blocks + allowed_post_fault_slack
TTFT/TPOT 明显优于 region enforce probe
```

### P8e：Pre-Fault Hard Quota

目标：

```text
从 post-fault correction 推进到 pre-fault admission control；
最终 max_over_blocks = 0。
```

阶段：

```text
trace -> soft -> enforce
```

验收：

```text
pre_fault_admission_records > 0
pre_fault_same_pool_reclaim_records > 0
max_over_blocks = 0
deny_or_block_records > 0 或 reclaim_to_budget_records > 0
failed_requests = 0
无 replay hang
```

### P9：Physical Chunk Pool Partition

仅作为研究方向，不作为下一版目标。它需要进入：

- `uvm_va_block_make_resident`
- GPU chunk allocation
- `uvm_pmm_gpu`
- UVM eviction path
- PTE split/merge
- 2MB/64K/4K chunk fallback

工程风险远高于当前 Stage N block quota 目标。

## 新验收指标

旧指标不再作为核心 hard proof：

```text
candidate_bytes
selected_victim_bytes
victim_count
driver_pool_state_records
```

新指标：

```text
block_quota_enable = 1
tagged_range_register_records > 0
block_materialize_records > 0
kv_resident_blocks <= kv_budget_blocks + slack
weights_resident_blocks <= weights_budget_blocks + slack
scratch_resident_blocks <= scratch_budget_blocks + slack
same_pool_victim_records > 0
cross_pool_victim_errors = 0
mixed_pool_blocks = 0
fault_path_region_scan_records = 0
block_reclaim_success_records > 0
failed_requests = 0
```

post-fault 版本允许：

```text
max_over_blocks <= allowed_post_fault_slack
```

pre-fault 版本要求：

```text
max_over_blocks = 0
```

## 推荐执行顺序

1. 实现 tagged range + lazy block trace。
2. 增加 allocator 2MB alignment / per-pool arena trace。
3. 验证 `mixed_pool_blocks = 0`。
4. 实现 manual same-pool block reclaim。
5. 实现 auto post-fault block quota。
6. 跑 vLLM mini probe，确认不再出现 region enforce fault storm。
7. 跑 P8 capacity stability proof。
8. 再进入 P8e pre-fault trace/soft/enforce。
9. 只有 P8e 仍无法满足目标时，再研究 physical chunk partition。

最终路线：

```text
不要继续优化 region full scan；
不要把 phase_n_block 做成裸全局 2MB bucket；
继承原生 UVM managed_range/block_index；
用 Stage N block wrapper 维护 per-pool resident queue；
先 post-fault correction，后 pre-fault hard quota。
```
