# Stage N Driver Pool-Aware Residency Controller Plan

Date: 2026-05-26

## 0. 当前结论

长期方向从 Stage M 的应用层 residency scheduler 转向 Stage N 的 driver 内置
pool-aware residency controller：

```text
vLLM / 其他框架:
  只注册语义 tag，不做迁移决策

UVM driver:
  维护 region table、VA 反查、pool budget、真实 residency accounting
  对 KV cache、weight、临时缓冲区三个 pool 执行 protected/evictable
  状态管理、victim selection、migrate/evict

BPF struct_ops:
  可选实验策略插件
  不承载 correctness
```

当前项目已经不是从零开始。Phase F 已经完成了 driver-backed range action、
region register/unregister、batch region action、real residency query 和 C++
helper。缺口在于：region 还没有进入 fault/eviction 快路径，driver 还没有
pool budget/accounting/victim selection，vLLM 仍有应用层 budget/victim 估算和
主动 prefetch/offload 路径。

补充结论：Stage M 的 scratch/device_direct 实验不应被理解为“长期放弃临时缓冲区
管理”。相反，它证明了临时缓冲区只需要一个很小但稳定的 GPU-resident 空间
就能显著减少热路径 cudaMalloc/cudaFree 和 UVM fault 干扰。Stage N 因此采用
三池分区：

```text
KV cache pool:
  大容量，可驱逐 inactive KV，保护 active request KV。

Weight pool:
  dense/shared weight 默认 protected，MoE expert 可按冷/热状态驱逐。

Temp scratch pool:
  小容量，默认约 100MiB，用于 runtime scratch/workspace。
  优先复用同尺寸/size-class buffer，超 budget 时先回收 transient scratch。
```

这里保留的不是应用层 `device_direct` 决策本身，而是它给出的工程事实：
临时缓冲区应该有独立 pool 和固定预算，避免挤占 KV/weight residency。

因此当前阶段定义为：

```text
当前阶段: P7 稳定性/性能 A/B harness 已落地，等待长实验数据
下一检查点: 运行多请求 A/B 与预算 sweep，按结果收敛 accounting、限速和 enforce 前策略
```

## 1. 实时计划表

这张表作为实施看板使用。每完成一个 PR 或阶段实验，都更新 `状态`、`证据`
和 `下一步`。

| 阶段 | 状态 | 目标 | 当前证据 | 下一步 | 验收标准 |
|---|---|---|---|---|---|
| P0: 规格冻结 | In Progress | 明确 ABI、pool、role、flags、state、budget mode | 本文档定义初版三池语义；Phase F 已有 region/action ABI | 随 P1 smoke 结果微调 enum/flag | 文档明确 KV/weight/temp-scratch tag、状态转换和驱逐规则 |
| P1a: Driver region ioctl 基础 | Done | 支持 register/unregister/batch-region action | `uvm_ioctl.h` 已有 `UVM_PHASE_F_REGISTER_REGION`、`UNREGISTER_REGION`、`BATCH_REGIONS`；`uvm.c` 已注册到 `phase_f_regions` | 保持兼容，扩展 state/query ioctl | smoke test 可注册 region 并按 region id query/action |
| P1b: Python ctypes region 封装 | Done | Python 能调用 driver region ioctl | `uvm.py` 已新增 `driver_register_region`、`driver_unregister_region`、`driver_batch_query_regions_residency`、feature probe | 补 Python/GPU smoke | Python self-test 可注册、查询、注销 region |
| P1c: vLLM tag-only 接入 | Smoke Passed | vLLM 只打 tag，不主动迁移 | `uvm_pool_residency.py` 已可将 registered region 同步到 driver；新增 tag-only 安全阀；Stage N tag-only 三池 smoke 已确认应用不主动 prefetch/offload | 保持回归覆盖 | driver 能看到 region；应用不调用 prefetch/offload/safe_offload |
| P2: Driver VA->region lookup | Fault Stats Classification In Progress | fault/evict 路径能由 VA 找到 pool/role/flags | driver register/unregister 已维护 `phase_f_region_ranges`，`QUERY_REGION_BY_VA` 已可用；replayable fault stats 已开始按 VA->region 做 pool/role 分类 | 继续补齐 fault stats 观测与回归验证 | fault stats 可按 pool/role 分类地址 |
| P3: Pool budget/state ioctl | Control Plane Done | 三池预算和 region state 下沉到 driver | 已新增 `SET_POOL_BUDGET`、`GET_POOL_STATE`、`UPDATE_REGION_STATE`、`QUERY_REGION`、`QUERY_REGION_BY_VA`；Python/C++ helper 已封装；scratch 默认可配置约 100MiB temp budget | 运行 phase-n-region smoke；接入 vLLM 启动配置 | driver 内部保存并查询 KV/weight/temp-scratch budget/state |
| P4: Real residency accounting | Scan Calibration Done | driver 维护真实 resident bytes | `GET_POOL_STATE` 已对已注册 region 做 scan/query 汇总并写回 per-pool resident_gpu/resident_cpu/reclaim bytes | 接 migrate/make-resident/evict/free 增量更新 | `pool_resident_gpu_bytes` 与 query 误差可控 |
| P5: Driver victim selection | Batch Trace Done | driver 自主选择 victim | `GET_POOL_STATE` 已计算 over_budget/reclaim_target/victim_candidate/selected_victim trace 字段，并返回最多 8 个 batch victim 候选；scratch 只把 `TRANSIENT|EVICTABLE` 且非 active/protected 的 region 作为候选 | 用 vLLM 三池 smoke 验证候选质量 | 超 budget 时 driver 选出同 pool 可驱逐页面/region |
| P6: 接入 UVM eviction/migration | Auto Soft Reclaim Smoke Passed | 使用 UVM 原生路径执行 reclaim | `phase-n-reclaim` 三池 smoke 已通过；`RECLAIM_POOL` 已抽成 `uvm_phase_n_reclaim_pool_execute()` 并调用 `uvm_migrate_managed_range_to_cpu()`；replayable fault batch 外层安全点已接入执行型 `phase_n_auto_reclaim_hook`；`REQUIRE_PHASE_N_AUTO_RECLAIM=1` validation 已 PASS，最新 `would_reclaim=20`、`execute=1`、`success=1`、`reclaimed=1`、`failed_requests=0` | 收敛 P7 稳定性和性能策略；补长期增量 accounting；扩大多请求 A/B | 应用不再发起 CPU/GPU prefetch，driver 能在安全点自动 soft reclaim 并保持请求成功 |
| P7: 稳定性和性能优化 | Harness Ready | 降低抖动、带宽竞争和锁开销 | P6 smoke 已加入 `every/scan_every/max_victims/cooldown/failure_backoff`，并优先小 victim 降低权重回收扰动；新增 `run_stage_n_p7_ab_sweep.sh`，可跑 native UVM / Stage N trace / Stage N soft / soft sweep / enforce probe，并生成 TSV/Markdown 摘要 | 运行多请求 A/B、scratch/cooldown/max-victim sweep；根据结果补长期 accounting 或调整 per-pool 策略 | TPOT/TTFT 不明显恶化，failed_requests=0，auto reclaim 成功，fault 更稳定 |
| P8: BPF 可选插件 | Pending | 快速实验 victim scoring | 当前不需要 BPF 承载正确性 | 暴露只读 kfunc/struct_ops score hook | 不加载 BPF 时 C 策略完整可用 |
| P9: 通用化 | Pending | 支持非 vLLM workload | vLLM 内已有较多专用接入 | 提供 C/Python tag SDK，适配 llama.cpp/SGLang | 新应用只需少量 tag 接入 |

## 2. 当前实现基线

### 2.1 Driver / UVM 已有能力

已有文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h
```

已实现能力：

```text
UVM_PHASE_F_BATCH_EXECUTOR
UVM_PHASE_F_REGISTER_REGION
UVM_PHASE_F_UNREGISTER_REGION
UVM_PHASE_F_BATCH_REGIONS

UVM_PHASE_F_ACTION_PREFETCH_CPU
UVM_PHASE_F_ACTION_PREFETCH_GPU
UVM_PHASE_F_ACTION_ADVISE_CPU
UVM_PHASE_F_ACTION_QUERY_RESIDENCY
```

`uvm_va_space_t` 中已有：

```c
struct radix_tree_root phase_f_regions;
```

这说明 driver 已经能保存 userspace 提交的 region metadata，但目前它主要用于
region id action，不足以支持 fault/eviction 快路径。

### 2.2 User-space helper 已有能力

已有文件：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/vllm/device_allocator/uvm.py
```

C++ helper 已经导出：

```text
uvm_driver_phase_f_register_region
uvm_driver_phase_f_unregister_region
uvm_driver_batch_query_residency_regions
uvm_driver_phase_f_register_query_region
```

Python `uvm.py` 已补齐 region register/unregister/batch-region query 封装。
这些 API 仍沿用 Phase F helper，失败时返回负数，便于在 vLLM runtime 中作为
可选 driver tag path 使用。

### 2.3 vLLM 当前应用层 residency 能力

已有文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
```

当前已有：

```text
应用层 region registry
应用层 pool budget 估算
应用层 victim candidate 选择
real residency sampling
KV allocate_slots 路径中的 runtime policy hook
MoE expert weight register/prefetch/active eviction 相关调用点
scratch device_direct 和 scratch reuse cache
```

这些能力证明 tag 来源和语义采集已经存在。当前第一版改造已经让
`uvm_pool_residency.py` 能把带真实 pointer 的 region 同步注册到 driver，
并通过 `VLLM_UVM_RESIDENCY_TAG_ONLY=1` 禁用应用层主动迁移路径。
长期目标仍然要求把 budget、victim selection 和迁移动作从应用层移到 driver。

### 2.4 Stage M 实验证据

Stage M scratch size-bucket reuse cache A/B：

```text
run_root: workloads/vllm/log/stage_m_scratch_size_bucket_reuse_cache_ab_p10_run

scratch_direct_no_cache:
  output tok/s: 4.00
  TPOT: 238.54 ms
  TTFT: 71055.52 ms
  parsed faults: 938283
  unknown faults: 5646
  gap2 faults: 4961
  replayable delta: 2273706
  dedup delta: 1588755

scratch_direct_bucket_cache:
  output tok/s: 3.83
  TPOT: 249.87 ms
  TTFT: 69298.48 ms
  parsed faults: 936616
  unknown faults: 6200
  gap2 faults: 5649
  replayable delta: 2257670
  dedup delta: 1569196
```

结论：

```text
cache treatment tok/s -4.25%
TPOT +4.75%
TTFT -2.47%
parsed faults -1667
unknown faults +554
gap2 faults +688
```

应用层 scratch cache 没有带来稳定收益，且 telemetry 里 cache hit 字段导出不完整。
但同一批实验也说明：临时缓冲区分配热点集中，小规模额外 GPU 空间就能覆盖大量
runtime scratch/workspace 需求。因此 Stage N 的判断应拆成两部分：

```text
不继续扩大应用层 scratch migration/offload 决策。
保留并系统化 scratch 小池思想，把它下沉为 driver 管理的 temp scratch pool。
```

也就是说，`device_direct` 不作为最终 correctness 路径，但其“约 100MiB 小池能
减少临时缓冲区分配/fault 干扰”的经验应进入 Stage N 三池设计。

## 3. 目标和非目标

### 3.1 目标

1. vLLM 只负责注册内存语义，不再主动做 residency 决策。
2. UVM driver 维护 KV cache、weight、temp scratch 三个 pool 的预算、真实
   resident bytes、状态和候选集合。
3. 超 budget 时，driver 在同 pool 内优先选择 victim 并调用 UVM 原生迁移/驱逐路径。
4. dense/shared weight、active KV、正在使用的 scratch 能被 protected。
5. MoE expert、inactive KV、evictable scratch 能被 driver 选择为 victim。
6. BPF 只作为可选 scoring 插件，driver C 策略必须完整可用。
7. temp scratch pool 默认保留约 100MiB GPU 空间，优先服务 runtime scratch、
   cuBLAS/cuDNN/workspace、MoE/attention 临时 buffer，避免污染 KV/weight pool。

### 3.2 非目标

1. 不在第一版追求真正的硬件显存硬隔离。
2. 不让 vLLM 在 allocate path 上执行复杂 victim selection。
3. 不把 correctness 依赖到 BPF。
4. 不一次性重写 UVM 全部 fault/migration 机制。
5. 不把应用层 `device_direct` allocator 继续作为长期 correctness 路径；它保留
   为对照、fallback 和 size-class 设计参考。
6. 不让 scratch 临时缓冲区与 KV/weight 共用无限制预算；scratch 必须作为第三个
   独立 pool 进入 driver accounting 和 victim selection。

## 4. ABI 设计

### 4.1 Pool enum

建议固定为小整数，保证 ioctl ABI 稳定：

```c
typedef enum
{
    UVM_POOL_UNKNOWN = 0,
    UVM_POOL_KV = 1,
    UVM_POOL_WEIGHT = 2,
    UVM_POOL_SCRATCH = 3,
    UVM_POOL_ACTIVATION = 4,
    UVM_POOL_MAX
} UVM_PHASE_N_POOL;
```

### 4.2 Role enum

```c
typedef enum
{
    UVM_ROLE_UNKNOWN = 0,
    UVM_ROLE_KV_BLOCK = 1,
    UVM_ROLE_DENSE_WEIGHT = 2,
    UVM_ROLE_MOE_EXPERT_W13 = 3,
    UVM_ROLE_MOE_EXPERT_W2 = 4,
    UVM_ROLE_SCRATCH = 5,
    UVM_ROLE_WORKSPACE = 6,
} UVM_PHASE_N_ROLE;
```

### 4.3 Flags

```c
#define UVM_REGION_FLAG_EVICTABLE      (1u << 0)
#define UVM_REGION_FLAG_PROTECTED      (1u << 1)
#define UVM_REGION_FLAG_ACTIVE         (1u << 2)
#define UVM_REGION_FLAG_PREFETCHABLE   (1u << 3)
#define UVM_REGION_FLAG_PINNED         (1u << 4)
#define UVM_REGION_FLAG_TRANSIENT      (1u << 5)
```

语义：

| flag | 含义 | driver 行为 |
|---|---|---|
| `EVICTABLE` | 可以被选为 victim | 进入候选集合 |
| `PROTECTED` | 不允许被 budget reclaim 驱逐 | victim selection 必须跳过 |
| `ACTIVE` | 当前 step/request 正在使用 | victim selection 必须跳过，或等待 cooldown |
| `PREFETCHABLE` | 可以被主动迁回 GPU | 后续 prefetch policy 可使用 |
| `PINNED` | 近似硬保护 | 只允许 memory pressure fallback 处理 |
| `TRANSIENT` | 生命周期短 | scratch/workspace 低优先级保留 |

### 4.4 Region state

region state 与真实 residency 分离。state 是应用提供的语义提示，resident bytes
是 driver 观测结果。

```c
typedef enum
{
    UVM_REGION_STATE_UNKNOWN = 0,
    UVM_REGION_STATE_COLD = 1,
    UVM_REGION_STATE_WARM = 2,
    UVM_REGION_STATE_HOT = 3,
    UVM_REGION_STATE_ACTIVE = 4,
    UVM_REGION_STATE_INACTIVE = 5,
} UVM_PHASE_N_REGION_STATE;
```

### 4.5 新增 ioctl

在 Phase F ABI 上扩展，不破坏已有 action executor：

```text
UVM_PHASE_N_SET_POOL_BUDGET
UVM_PHASE_N_GET_POOL_STATE
UVM_PHASE_N_UPDATE_REGION_STATE
UVM_PHASE_N_QUERY_REGION
UVM_PHASE_N_QUERY_REGION_BY_VA
UVM_PHASE_N_RECLAIM_POOL
```

建议结构：

```c
typedef struct
{
    NvU32 pool;
    NvU32 gpu_id;
    NvU64 budget_bytes NV_ALIGN_BYTES(8);
    NvU32 mode;
    NvU32 flags;
    NV_STATUS rmStatus;
} UVM_PHASE_N_SET_POOL_BUDGET_PARAMS;

typedef struct
{
    NvU32 pool;
    NvU32 gpu_id;
    NvU64 budget_bytes NV_ALIGN_BYTES(8);
    NvU64 resident_gpu_bytes NV_ALIGN_BYTES(8);
    NvU64 resident_cpu_bytes NV_ALIGN_BYTES(8);
    NvU64 reclaim_bytes NV_ALIGN_BYTES(8);
    NvU64 protected_bytes NV_ALIGN_BYTES(8);
    NvU64 evictable_bytes NV_ALIGN_BYTES(8);
    NvU64 over_budget_bytes NV_ALIGN_BYTES(8);
    NvU64 reclaim_target_bytes NV_ALIGN_BYTES(8);
    NvU64 victim_candidate_bytes NV_ALIGN_BYTES(8);
    NvU64 victim_candidate_regions NV_ALIGN_BYTES(8);
    NvU64 selected_victim_region_id NV_ALIGN_BYTES(8);
    NvU64 selected_victim_bytes NV_ALIGN_BYTES(8);
    NvU64 over_budget_events NV_ALIGN_BYTES(8);
    NvU64 reclaim_success_events NV_ALIGN_BYTES(8);
    NvU64 reclaim_failed_events NV_ALIGN_BYTES(8);
    NvU64 accounting_corrections NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_PHASE_N_GET_POOL_STATE_PARAMS;

typedef struct
{
    NvU64 region_id NV_ALIGN_BYTES(8);
    NvU32 state;
    NvU32 flags_set;
    NvU32 flags_clear;
    NvU32 priority;
    NvU64 last_access_epoch NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_PHASE_N_UPDATE_REGION_STATE_PARAMS;

typedef struct
{
    NvU32 pool;
    NvU32 gpu_id;
    NvU64 target_bytes NV_ALIGN_BYTES(8);
    NvU64 current_epoch NV_ALIGN_BYTES(8);
    NvU64 cooldown_epochs NV_ALIGN_BYTES(8);
    NvU32 max_victims;
    NvBool async;
    NvU32 flags;
    NvU32 attempted_regions;
    NvU32 success_regions;
    NvU32 failure_regions;
    NvU32 victim_count;
    NvU64 resident_gpu_bytes_before NV_ALIGN_BYTES(8);
    NvU64 over_budget_bytes NV_ALIGN_BYTES(8);
    NvU64 reclaim_target_bytes NV_ALIGN_BYTES(8);
    NvU64 attempted_bytes NV_ALIGN_BYTES(8);
    NvU64 reclaimed_bytes NV_ALIGN_BYTES(8);
    NvU64 cooldown_skipped_bytes NV_ALIGN_BYTES(8);
    NvU64 victim_region_ids[8] NV_ALIGN_BYTES(8);
    NvU64 victim_bytes[8] NV_ALIGN_BYTES(8);
    NV_STATUS victim_status[8];
    NV_STATUS rmStatus;
} UVM_PHASE_N_RECLAIM_POOL_PARAMS;
```

## 5. Driver 数据结构

### 5.1 Region object

扩展当前 `uvm_phase_f_region_t`，或者新增 `uvm_pool_region_t`：

```c
typedef struct
{
    NvU64 region_id;
    NvU64 base;
    NvU64 length;
    NvU64 end;

    NvU32 pool;
    NvU32 role;
    NvU32 flags;
    NvU32 state;
    NvU32 priority;

    NvU32 layer_id;
    NvU32 expert_id;
    NvU64 last_access_epoch;
    NvU64 registered_epoch;

    struct list_head pool_node;
    struct rb_node va_node;
} uvm_pool_region_t;
```

### 5.2 VA 反查结构

当前结构：

```text
phase_f_regions: region_id -> region
```

需要新增：

```text
phase_n_regions_by_id: region_id -> region
phase_n_regions_by_va: interval tree, VA range -> region
phase_n_pool_region_lists[POOL_MAX]
```

lookup 目标：

```text
fault_address -> region -> pool/role/flags/state
va_block/chunk -> region -> pool
```

第一版可以用 UVM 已有 `uvm_range_tree_t` 或 kernel interval tree。关键要求：

1. register 时检查同一 VA space 内的 region 是否重叠。
2. unregister 时同时从 id tree、VA tree、pool list 删除。
3. va_space destroy 时释放所有 region。
4. fault path lookup 必须可在读锁下完成，不能引入长时间写锁。

### 5.3 Pool state

在 `uvm_va_space_t` 中增加：

```c
typedef struct
{
    NvU64 budget_bytes;
    NvU64 resident_gpu_bytes;
    NvU64 resident_cpu_bytes;
    NvU64 protected_bytes;
    NvU64 evictable_bytes;
    NvU64 over_budget_bytes;
    NvU64 reclaim_target_bytes;
    NvU64 victim_candidate_bytes;
    NvU64 victim_candidate_regions;
    NvU64 selected_victim_region_id;
    NvU64 selected_victim_bytes;

    NvU64 over_budget_events;
    NvU64 reclaim_attempt_events;
    NvU64 reclaim_success_events;
    NvU64 reclaim_failed_events;
    NvU64 accounting_corrections;

    NvU32 mode;
    NvU32 flags;
} uvm_pool_state_t;

uvm_pool_state_t phase_n_pools[UVM_POOL_MAX];
```

`mode` 建议：

```text
TRACE_ONLY: 只统计，不驱逐
SOFT: 超 budget 时尝试 reclaim，失败则走原 UVM 行为
ENFORCE: 超 budget 时必须先 reclaim，失败则返回压力信号或降级
```

## 6. Real Residency Accounting

### 6.1 第一版：scan/query 校准

先复用 Phase F range residency query 的逻辑：

```text
periodic scan region table
  -> query each registered region resident_gpu_bytes/cpu_bytes
  -> aggregate by pool
  -> write pool_state
```

优点：

```text
实现快
风险低
能验证 pool budget ioctl 和 observability
```

缺点：

```text
不是实时
大 region 多时有扫描开销
不能作为最终 eviction fast path 的唯一依据
```

当前实现状态：

```text
已实现于 P4 scan calibration:
  GET_POOL_STATE(pool)
    -> 快照当前 pool 已注册 region
    -> 对每个 region 调用真实 residency query
    -> 汇总 resident_gpu_bytes / resident_cpu_bytes
    -> 对无法查询的 region 累加 reclaim_bytes
    -> 若 resident_gpu_bytes > budget，计算 over_budget_bytes
    -> 在同 pool 内筛选 trace-only victim candidate
       KV/weight: EVICTABLE && !PROTECTED && !ACTIVE && !PINNED
       scratch/temp: TRANSIENT && EVICTABLE && !PROTECTED && !ACTIVE && !PINNED
    -> 计算 reclaim_target_bytes、victim_candidate_bytes、
       victim_candidate_regions、selected_victim_region_id
    -> 写回 va_space->phase_n_pools[pool]

当前限制:
  这是 ioctl 查询时同步 scan，不是 fault/migrate 热路径增量账本。
  大量 region 时有 O(region_count) 查询开销，因此不应在每个 decode step 调用。
  victim selection 目前是 trace-only，不执行 migrate/evict。
  它适合 smoke、周期性校准、策略调试和 P5/P6 reclaim 前的基线观测。
```

### 6.2 第二版：增量 accounting

最终需要在 UVM 实际改变 residency 的路径中更新：

```text
make resident
migrate CPU -> GPU
migrate GPU -> CPU
evict
discard
free
VA block split/merge
```

更新粒度建议：

```text
VA block/page chunk -> lookup region -> pool delta
```

处理原则：

1. accounting 以 driver 真实 page residency 为准。
2. region state 只影响 victim selection，不直接等价于 resident bytes。
3. 增量账本允许周期性 scan 修正 drift。
4. 所有修正要记入 `accounting_corrections`，方便定位漏记路径。

## 7. Driver Victim Selection

### 7.1 总体流程

```text
incoming allocation/page residency request
  -> lookup incoming pool
  -> projected = pool_resident_gpu_bytes + incoming_gpu_bytes
  -> if projected <= budget: allow
  -> else:
       target_reclaim = projected - budget + hysteresis_margin
       victims = select_victims(pool, target_reclaim)
       migrate/evict victims to CPU
       retry incoming operation
```

同 pool 优先：

```text
KV 超 budget -> 先驱逐 KV victim
WEIGHT 超 budget -> 先驱逐 evictable weight victim
SCRATCH 超 budget -> 先驱逐 transient scratch/workspace
```

只有在系统级 OOM 或 UVM 原生 pressure 路径中，才考虑跨 pool fallback。

### 7.2 KV pool policy

```text
候选:
  pool == KV
  EVICTABLE
  !PROTECTED
  !ACTIVE

排序:
  last_access_epoch 最老优先
  request 已结束优先
  prefix cache 可重算/低优先级 block 优先
```

初始保守规则：

```text
active request KV blocks: PROTECTED | ACTIVE
finished/request-free KV blocks: EVICTABLE
prefix-cache cold blocks: EVICTABLE, low priority
```

### 7.3 Weight pool policy

```text
dense/shared weight:
  默认 PROTECTED

MoE expert weight:
  EVICTABLE unless ACTIVE/PROTECTED
  cold expert 优先
  last router hit 更老优先
```

第一版不需要 driver 理解 MoE router，只接收 vLLM 更新的 `last_access_epoch` 和
`ACTIVE` flag。复杂 hotness 以后可由 BPF 或 userspace hint 提供。

### 7.4 Scratch pool policy

```text
scratch/workspace:
  默认预算约 100MiB，可通过 driver pool budget 配置覆盖
  优先复用同尺寸或 size-class buffer
  TRANSIENT 优先回收
  生命周期短、低 priority 优先回收
  ACTIVE scratch 不驱逐
```

scratch 当前 `device_direct` 路线可以保留为 fallback 和对照，但长期应统一为
driver-managed temp pool：

```text
应用注册 scratch/temp tag
driver 识别 temp scratch pool
driver 在 pool 内维护 small-buffer cache / size-class reuse
driver 决定是否保留 GPU resident、迁移到 CPU、释放或回收
```

建议第一版 scratch/temp size-class：

```text
4KiB, 8KiB, 16KiB, 32KiB, 64KiB,
128KiB, 256KiB, 512KiB,
1MiB, 2MiB, 4MiB, 8MiB, 16MiB, 32MiB, 64MiB
```

预算策略：

```text
scratch_budget_default = 100MiB
scratch_budget_min = 64MiB
scratch_budget_max = 256MiB unless explicitly configured
reclaim_target = over_budget + one_size_class_margin
```

victim 规则：

```text
candidate:
  pool == SCRATCH
  TRANSIENT
  EVICTABLE
  !ACTIVE
  !PROTECTED

sort:
  freed/inactive allocation first
  oldest last_access_epoch
  larger exact-size cache bucket over target pressure first
  lower priority first
```

这相当于把 Stage M 的“应用侧 scratch reuse cache / device_direct 小缓冲区”下沉为
driver 内置 temp pool。应用仍然只负责 tag，不负责判断哪个临时 buffer 应该留下。

### 7.5 防抖机制

必须从第一版就预留：

```text
hysteresis: reclaim 超过刚好 budget 的额外 margin
cooldown: 刚迁入/刚访问 region 暂不驱逐
batching: 每次 reclaim 聚合多个 pages/regions
rate limit: 限制单位时间迁移量
failure backoff: 连续失败后暂时回退到原生 UVM 行为
```

## 8. 接入 UVM Eviction/Migration 路径

### 8.0 三池分区执行模型

Stage N 最终执行模型不是一个全局 LRU，而是三个 pool 各自有预算、候选集和
reclaim 策略：

```text
KV cache pool:
  面向较大、结构化、可按 block 标记 active/inactive 的 KV block。
  reclaim 单位优先是 KV block 或 VA block page set。

Weight pool:
  面向 dense/shared weight 和 MoE expert weight。
  dense/shared 默认 protected，MoE expert 可按 expert 维度冷/热回收。

Temp scratch pool:
  面向短生命周期 runtime scratch/workspace。
  reclaim 单位优先是 allocation/size-class bucket，而不是模型语义 block。
```

预算检查：

```text
incoming_pool = VA/tag lookup result
projected = pool_resident_gpu_bytes[incoming_pool] + incoming_bytes

if incoming_pool == SCRATCH:
  if projected > scratch_budget:
    first try scratch size-class reuse/free inactive transient buffers
    then evict/migrate cold scratch pages
    never reclaim KV/weight unless global OOM fallback

if incoming_pool == KV:
  only select KV victims first

if incoming_pool == WEIGHT:
  only select evictable weight victims first
```

这样能避免临时缓冲区和 KV/weight 互相污染：scratch 热分配不会挤掉 active KV，
KV 超载也不会误伤正在使用的 workspace。

### 8.1 Phase F 保留为控制面对照

Phase F 当前 action executor 仍有价值：

```text
smoke test
debug
explicit A/B
residency query
fallback action
```

但 Stage N 的数据面不应依赖应用提交 action list。

### 8.2 Stage N 目标接入点

需要在 UVM 内部路径中找准三类 hook：

```text
1. 即将使 page GPU resident 的路径
2. UVM 原生 memory pressure / PMM eviction 路径
3. page 被 migrate/evict/discard/free 后的 accounting 路径
```

推荐顺序：

1. 先只在 fault stats 或 service replayable fault 路径做 VA->region 分类打印。
2. 再在 make-resident 前做 budget check，但只 trace，不 reclaim。
3. 再接入 reclaim selection，调用现有 UVM migrate/evict helper。
4. 最后改成 enforce mode，并保留 module param 开关。

### 8.3 锁和性能约束

1. fault path 只允许读侧快速 lookup。
2. victim selection 可以异步或在 reclaim worker 中完成。
3. 不在高频 fault handler 中扫描全量 region。
4. region table 更新由 ioctl 慢路径完成。
5. accounting 更新必须避免对全局锁造成长时间持有。

### 8.4 P6 soft reclaim 控制面第一版

当前实现先落地 control-plane soft reclaim，而不是直接修改 hot fault path：

```text
UVM_PHASE_N_RECLAIM_POOL:
  1. driver 扫描同 pool registered regions
  2. query 每个 region 的真实 GPU resident bytes
  3. 过滤 victim:
       EVICTABLE
       !PROTECTED
       !ACTIVE
       !PINNED
       scratch 额外要求 TRANSIENT
  4. 按 priority、last_access_epoch、resident bytes、region_id 选最多 8 个 victims
  5. 如果 pool mode 是 SOFT/ENFORCE，则 driver 调 UVM migrate-to-CPU 路径
  6. 失败只记录 failure，不破坏请求正确性
```

hysteresis：

```text
target = resident_gpu_bytes - budget + reclaim_margin
KV/weight margin cap: 64MiB
scratch margin cap: 16MiB
```

cooldown：

```text
current_epoch/cooldown_epochs 由调用者传入
last_access_epoch 距离当前 epoch 太近的 region 会被跳过
```

当前限制：

```text
当前已完成控制面 ioctl 手动触发 soft reclaim
迁移执行已从 `uvm_api_migrate(filp)` 改为 `uvm_migrate_managed_range_to_cpu()`
已新增 `uvm_phase_n_reclaim_pool_over_budget()`，后续 hot path 可直接传入 va_space 调用
make-resident 成功迁入 GPU 后已有默认关闭的自动 hook preflight：
  `uvm_phase_n_make_resident_reclaim_hook()`
  module params:
    uvm_phase_n_auto_reclaim_enable=0
    uvm_phase_n_auto_reclaim_every=64
    uvm_phase_n_auto_reclaim_scan_every=256
    uvm_phase_n_auto_reclaim_execute=0
    uvm_phase_n_auto_reclaim_max_victims=2
    uvm_phase_n_auto_reclaim_cooldown_epochs=0
  locked context 内仍只做 VA->region lookup、pool budget/mode 检查和
  `would_reclaim` trace，不在 va_block lock 内执行迁移。
replayable fault batch 外层已有执行型 automatic soft reclaim：
  `uvm_phase_n_auto_reclaim_outer_hook()`
  在不持 va_block lock 的安全点调用 `uvm_phase_n_reclaim_pool_execute()`。
  支持 scan throttle、under-budget cached fast skip、success cooldown、
  failure backoff、batch victim 上限和 small-victim tie-break。
PMM eviction pressure 外层 hook 仍待 P7/P6.2 评估
reclaimed_bytes 是迁移前 resident bytes 的 best-effort 估算
长期增量 accounting 仍待 P7/P4 第二版接入
```

最新 smoke 证据：

```text
run_root:
  workloads/vllm/log/phase_f_crash_diag/latest_phase-n-reclaim_run

phase_n_reclaim:
  attempted=1
  success=1
  failure=0
  victim_count=1
  budget=524288
  resident_before=1048576
  over=524288
  target=4718592
  attempted_bytes=1048576
  reclaimed=1048576
  failed_bytes=0
  candidate_bytes=1048576
  candidate_regions=1
  success_events=1
  failed_events=0
  mode=1

victim:
  region_id=1
  bytes=1048576
  status=0x0
```

这说明 `UVM_PHASE_N_RECLAIM_POOL` 已被加载中的 driver 接收，driver 能在同 pool
内选出 victim，并通过 UVM migrate-to-CPU 路径完成一次 soft reclaim。

2026-05-27 更新：

```text
代码侧:
  `uvm_phase_f_query_resident_bytes()` 已改为直接接收 `uvm_va_space_t *`。
  residency scan 也已兼容注册 mm、普通 ioctl current mm 和无 mm hot path。
  新增 `uvm_migrate_managed_range_to_cpu()`，内部复用 UVM managed migration，
  优先使用 va_space 注册 mm；在未来 kernel worker/hot path 场景无 mm 时自动
  使用 `UVM_MIGRATE_FLAG_SKIP_CPU_MAP`。
  `uvm_phase_n_reclaim_pool_execute()` 已不再依赖 userspace `filp`。
  新增 `uvm_phase_n_reclaim_pool_over_budget()` 作为自动触发入口。

验证侧:
  重新运行 `phase-n-reclaim` 时当前加载模块仍是旧版，log 显示
  `ioctl 82/83 not found`，因此这次失败是模块未重载，不是 reclaim 逻辑失败。
  `make -C kernel-module/nvidia-module/kernel-open modules NV_KERNEL_MODULES=nvidia-uvm`
  已编译 `uvm.c`、`uvm_migrate.c` 并链接到 `nvidia-uvm.o`；最终仍在单独构建
  `nvidia-uvm` 的 modpost 阶段因缺少 core `nvidia` 模块符号失败，这是既有
  构建方式限制。
```

2026-05-27 make-resident hook preflight 更新：

```text
代码侧:
  新增默认关闭的自动 hook 参数：
    uvm_phase_n_auto_reclaim_enable=0
    uvm_phase_n_auto_reclaim_every=64
    uvm_phase_n_auto_reclaim_scan_every=256
    uvm_phase_n_auto_reclaim_execute=0
    uvm_phase_n_auto_reclaim_max_victims=2
    uvm_phase_n_auto_reclaim_cooldown_epochs=0
  新增 `uvm_phase_n_make_resident_reclaim_hook()`：
    在调用方已持有 va_space lock 的上下文里做 VA->region lookup。
    读取命中 region 的 pool、budget、resident、over_budget、victim candidate。
    输出 `phase_n_auto_reclaim_hook ... epoch=... every=... max_victims=...
    cooldown_epochs=... would_reclaim=... execute=0 skip=locked_context`。
  `uvm_va_block_make_resident()` 在 GPU residency 成功后调用该 hook。

安全边界:
  这个 hook 当前只做 trace/preflight，不在 va_block lock 内执行迁移。
  执行型 reclaim 仍需放到异步 worker 或不持 block lock 的外层安全点，
  否则可能递归进入 migrate/make-resident 并触发锁顺序风险。

验证侧:
  `make -C kernel-module/nvidia-module modules` 已通过。
  当前仍有既有 warning: uvm_ioctl frame size 1168 > 1024。
```

2026-05-27 auto hook 验收脚本更新：

```text
代码侧:
  `run_stage_n_tag_only_three_pool_smoke.sh` 新增 kernel_since_start.log，
  从本次 smoke 启动时刻开始收集 kernel log，并统计：
    phase_n_auto_hook_records
    phase_n_auto_hook_would_reclaim_records
    phase_n_auto_hook_locked_context_records
  默认不要求 hook，保持既有 tag-only 三池 smoke 行为不变。
  设置 `REQUIRE_PHASE_N_AUTO_HOOK=1` 后，summary/verdict 会把
  `phase_n_auto_hook_records > 0` 作为强验收条件。

  `run_stage_n_three_pool_validation.sh` 会把 `REQUIRE_PHASE_N_AUTO_HOOK`
  透传给 tag-only smoke，并在总 summary 中显示 Auto Hook Evidence。

验证侧:
  `bash -n workloads/vllm/run_stage_n_tag_only_three_pool_smoke.sh` PASS
  `bash -n workloads/vllm/run_stage_n_three_pool_validation.sh` PASS
  `make -C kernel-module/nvidia-module modules` PASS
  built nvidia-uvm.ko srcversion=DA82B7CC5DABD4019BA8F33
  当前仍有既有 warning: uvm_ioctl frame size 1168 > 1024。

运行方式:
  重载 nvidia_uvm 时启用：
    uvm_phase_n_auto_reclaim_enable=1
    uvm_phase_n_auto_reclaim_every=1
  然后运行：
    REQUIRE_PHASE_N_AUTO_HOOK=1 RESET_COUNTERS=1 \
      UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
      ./run_stage_n_three_pool_validation.sh
```

2026-05-27 auto hook 验收闭环：

```text
问题定位:
  `RESET_COUNTERS=1` 会让 `run_kv_fault_ratio.sh` 重载 nvidia_uvm。
  之前该重载路径没有传入 `uvm_phase_n_auto_reclaim_*` 参数，因此 hook 被重载回
  enable=0。
  修复后，`run_stage_n_tag_only_three_pool_smoke.sh` 会在
  `REQUIRE_PHASE_N_AUTO_HOOK=1` 时自动传入：
    --uvm-phase-n-auto-reclaim-enable 1
    --uvm-phase-n-auto-reclaim-every 64
    --uvm-phase-n-auto-reclaim-scan-every 256

  另一个假阴性来自日志源：`phase_n_auto_reclaim_hook` 写入的是
  `/proc/driver/nvidia-uvm/phase_f_route_log`，不是 kernel journal。
  tag-only smoke 现在会保存：
    phase_f_route_log_snapshot.log
  并从该文件统计 `phase_n_auto_hook_*` counters。

代码侧:
  fault service 主要路径不调用 `uvm_va_block_make_resident()` wrapper，
  而是直接调用 `uvm_va_block_make_resident_copy()` +
  `uvm_va_block_make_resident_finish()`。
  已在 `uvm_va_block_service_finish()` 的 finish 之后补
  `uvm_phase_n_make_resident_reclaim_hook()`。

验证侧:
  run_root=workloads/vllm/log/stage_n_three_pool_validation_20260527_093758
  tag_only_status=0
  phase_n_auto_hook_records=495
  phase_n_reclaim_kv_victim_count=1
  phase_n_reclaim_weights_victim_count=1
  phase_n_reclaim_scratch_victim_count=1
  failed_requests=0
  Verdict=PASS

当前限制:
  本轮 hook records 主要是 `skip=no_region`，说明 preflight 已进入 fault path，
  但这些 fault VA 尚未命中 Stage N region range tree。
  下一步要把 hook 触发点与已注册 KV/weight/scratch region 的生命周期对齐，
  让 trace 从 `skip=no_region` 进入 pool budget / would_reclaim 分支。
```

## 9. vLLM Tag Adapter 改造

### 9.1 Python ctypes wrapper

`workloads/vllm/vllm/vllm/device_allocator/uvm.py` 已增加：

```python
driver_register_region(
    region_id: int,
    ptr: int,
    nbytes: int,
    pool: int,
    layer_id: int = 0,
    expert_id: int = 0,
    role: int = 0,
    flags: int = 0,
) -> int

driver_unregister_region(region_id: int) -> int

driver_batch_query_regions_residency(
    region_ids: Sequence[int],
    device: int = 0,
) -> int
```

同时增加 feature probe：

```text
has_driver_region_ioctl()
has_pool_budget_ioctl()
has_region_state_ioctl()
```

当前状态：

```text
has_driver_region_ioctl(): 已实现
has_pool_budget_ioctl(): 返回 False，等待 P3
has_region_state_ioctl(): 返回 False，等待 P3
```

### 9.2 Residency manager 降级为 tag adapter

当前：

```text
register_uvm_residency_region()
  -> Python registry

request_uvm_gpu_residency()
  -> 应用层 budget check
  -> 应用层 victim candidate
  -> 可能触发 prefetch/offload
```

目标：

```text
register_uvm_residency_region()
  -> Python trace
  -> driver_register_region

mark_uvm_residency_state()
  -> Python trace
  -> driver_update_region_state

request_uvm_gpu_residency()
  -> deprecated
  -> trace-only compatibility shim
  -> 不再发起迁移/驱逐
```

第一版实现：

```text
VLLM_UVM_DRIVER_TAG_ENABLE=1
  register_uvm_residency_region 会尝试 driver_register_region
  trace 中记录 driver_region_id / driver_register_status / driver_tag

VLLM_UVM_RESIDENCY_TAG_ONLY=1
  默认打开 driver tag sync
  request_uvm_gpu_residency 返回 allowed=True
  reason=tag_only_no_application_residency_decision
  共享 prefetch/offload helper 不再发起 cudaMemPrefetch/cudaMemAdvise
```

限制：

```text
只有 ptr/bytes 有效的 region 才能注册到 driver。
默认要求 metadata.driver_capable 不是 False。
KV metadata-only region 会保留 Python trace，不注册 driver。
driver state update ioctl 尚未实现，mark_state 仍是 Python trace-only。
```

### 9.3 KV tag 接入

`kv_cache_manager.py` 的 `allocate_slots()` 已经是天然接入点，但长期只做状态更新：

```text
allocate_slots:
  new KV block -> register region if needed
  request active blocks -> UPDATE_REGION_STATE ACTIVE | PROTECTED
  inactive/free blocks -> UPDATE_REGION_STATE EVICTABLE, clear ACTIVE
```

不在这里做：

```text
应用层 victim selection
应用层 prefetch_cpu/gpu
safe_offload
```

### 9.4 Weight tag 接入

MoE expert weight 当前已经在 `fused_moe/layer.py` 中有 region metadata。目标：

```text
model load:
  dense/shared weights -> WEIGHT + DENSE_WEIGHT + PROTECTED
  MoE expert w13/w2 -> WEIGHT + MOE_EXPERT_* + EVICTABLE

router top-k:
  selected expert -> UPDATE_REGION_STATE ACTIVE/HOT
  previous expert -> clear ACTIVE, update last_access_epoch
```

第一版可只注册并更新状态，不做 driver prefetch。

### 9.5 Scratch tag 接入

scratch 当前由 allocator trace/device_direct 识别。目标：

```text
allocation:
  if runtime scratch/workspace:
    register region pool=SCRATCH role=SCRATCH/WORKSPACE flags=EVICTABLE|TRANSIENT

free:
  unregister region

phase changes:
  active scratch -> ACTIVE
  released scratch -> clear ACTIVE
```

## 10. Observability

必须新增可观测项，否则 driver 策略很难调：

```text
per-pool:
  budget_bytes
  resident_gpu_bytes
  resident_cpu_bytes
  protected_bytes
  evictable_bytes
  reclaim_attempt/success/fail
  over_budget_events

per-region:
  region_id
  pool
  role
  flags
  state
  resident_gpu_bytes
  resident_cpu_bytes
  last_access_epoch

fault stats:
  fault_pool_counts
  fault_role_counts
  unknown_unregistered_faults
  protected_faults
  evictable_faults
```

输出路径建议：

```text
ioctl GET_POOL_STATE
debugfs/proc summary
trace log
existing replayable fault stats extension
gap-watch summary extension
```

第一阶段最关键指标：

```text
registered_region_count_by_pool
faults_by_pool
unknown_faults_not_in_region
pool_resident_gpu_bytes
pool_over_budget_events
driver_reclaim_success/fail
```

## 11. 测试与验收

### 11.1 Unit / smoke

| 测试 | 目标 |
|---|---|
| region register duplicate id | 返回 `NV_ERR_*`，不泄漏 |
| overlapping region register | 明确拒绝或记录父子关系，第一版建议拒绝 |
| unregister | id tree、VA tree、pool list 全删除 |
| VA lookup | 地址命中正确 region，边界地址正确 |
| pool budget ioctl | set/get 一致 |
| update state ioctl | flags/state/priority/epoch 可查询 |

### 11.2 Synthetic managed allocation

```text
cudaMallocManaged A/B/C
register A as KV
register B as WEIGHT
register C as SCRATCH
touch pages
query pool state
force budget below resident bytes
验证 trace-only/soft/enforce 行为
```

### 11.3 vLLM tag-only smoke

验收：

```text
server 启动成功
KV/weight/temp scratch region 都被注册
GET_POOL_STATE 能看到 KV/WEIGHT/SCRATCH 三个 pool
SCRATCH pool budget 默认为约 100MiB，或按 env 显式配置
fault stats 能按 pool 分类
不出现应用层 prefetch/offload issued
failed requests = 0
```

### 11.4 Driver policy A/B

最小对照组：

```text
A: native UVM
B: Stage M scratch device_direct best baseline
C: Stage N tag-only trace
D: Stage N pool budget soft
E: Stage N pool budget enforce
```

核心指标：

```text
TPOT
TTFT
output tok/s
raw replayable faults
dedup replayable faults
faults_by_pool
KV fault ratio
scratch/temp fault ratio
unknown/unregistered fault ratio
pool reclaim success/fail
scratch size-class hit/miss/reclaim
```

## 12. 风险和缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| fault path lookup 增加开销 | TPOT 变差 | 先 trace-only 测量 lookup cost；使用读锁/RCU 风格；避免扫描 |
| accounting 漏记或重复记 | budget 判断错误 | 周期 scan 校准；记录 correction counter |
| tag 生命周期不准 | 驱逐 active 数据 | active/protected 默认保守；unknown 不驱逐 |
| 过度 reclaim 抖动 | fault 增加、吞吐下降 | hysteresis、cooldown、rate limit |
| 与 UVM 原生 eviction 冲突 | 非预期迁移 | policy hook 先 trace，再 soft，再 enforce |
| 应用 tag 不通用 | 只能服务 vLLM | 提供 C/Python SDK，其他框架只接 register/update/unregister |
| BPF verifier/版本问题 | 策略不可加载 | BPF 可选；C fallback 必须完整 |

## 13. BPF 可选设计

BPF 不作为 Stage N 必需项。driver C 策略先完成后，再开放只读查询和 scoring：

```c
bpf_gpu_chunk_pool()
bpf_gpu_chunk_region_id()
bpf_gpu_chunk_flags()
bpf_gpu_region_last_access_epoch()
bpf_gpu_pool_budget()
bpf_gpu_pool_resident_bytes()
```

BPF 返回：

```text
score
skip reason
optional priority adjustment
```

driver 仍负责：

```text
预算正确性
protected/active 跳过
最终 victim selection fallback
迁移/驱逐执行
错误处理
```

这样既保留开源内核直接修改的稳定路径，又能为后续策略实验提供快速迭代入口。

## 14. 推荐近期执行顺序

### Step 1: 补 Python driver region wrapper

文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm.py
```

状态：Done

动作：

```text
声明 ctypes argtypes/restype
封装 driver_register_region
封装 driver_unregister_region
封装 driver_batch_query_regions_residency
增加 feature probe
```

验收：

```text
Python self-test 注册 managed pointer region
按 region id 查询 residency
注销 region
无 kernel error
```

### Step 2: vLLM tag-only driver registration

文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py
```

状态：In Progress

动作：

```text
register_uvm_residency_region 同步调用 driver_register_region
mark_uvm_residency_state 保持 trace，等待 driver state ioctl
request_uvm_gpu_residency 改成 trace-only 或 env gate 禁用动作
```

当前已实现：

```text
VLLM_UVM_DRIVER_TAG_ENABLE=1 可把带 ptr 的 region 注册进 Phase F driver registry
VLLM_UVM_RESIDENCY_TAG_ONLY=1 可关闭应用层迁移动作
residency trace 增加 driver_tag_register_* 计数
```

验收：

```text
driver log 能看到 KV/weight/scratch region
应用层没有 offload/prefetch issued
bench 正常完成
```

### Step 3: Driver VA->region lookup

文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
```

动作：

```text
增加 VA interval tree
register/unregister 同步维护 id tree 和 VA tree
增加 lookup helper
fault stats 中打印 pool/role 分类
```

验收：

```text
fault address 能命中 region
unknown/unregistered fault 单独统计
无明显 fault handler 开销回退
```

### Step 4: Pool budget/state ioctl

动作：

```text
新增 SET_POOL_BUDGET
新增 GET_POOL_STATE
新增 UPDATE_REGION_STATE
新增 QUERY_REGION / QUERY_REGION_BY_VA
```

验收：

```text
vLLM 可设置 KV=10GiB、WEIGHT=10GiB、SCRATCH=N
driver query 能返回预算和状态
状态更新可被 fault stats/victim candidate 看到
```

### Step 5: Trace-only budget check

动作：

```text
在 make-resident/fault service 相关路径计算 projected pool usage
只记录 would_reclaim，不执行驱逐
```

验收：

```text
能解释何时哪个 pool 超 budget
不会改变 TPOT/TTFT
```

### Step 6: Soft reclaim

动作：

```text
实现 C 版 victim selection
调用 UVM 原生 migrate/evict helper
失败则回退原 UVM 行为
```

验收：

```text
超 budget 时 reclaim_success > 0
failed requests = 0
TPOT/TTFT 不明显恶化
```

### Step 7: Enforce mode

动作：

```text
严格执行 pool budget
加入 hysteresis/cooldown/rate limit
补齐 accounting correction
```

验收：

```text
KV/weight/scratch resident bytes 被限制在预算附近
UVM 原生 fault 与 driver reclaim 不冲突
性能优于或至少不差于 Stage M 最佳 baseline
```

## 15. 成功判据

Stage N 第一里程碑成功：

```text
vLLM 只打 tag
driver 能按 VA 识别 KV/weight/scratch
driver 能查询 pool state
fault stats 按 pool 分类
应用层迁移动作全部关闭
```

Stage N 第二里程碑成功：

```text
driver 能维护真实 per-pool resident bytes
driver 能在超 budget 时选择 victim
soft reclaim 不造成请求失败
```

Stage N 第三里程碑成功：

```text
KV 10GiB / WEIGHT 10GiB / SCRATCH 100MiB budget 可配置
active KV 不被驱逐
cold MoE expert 可被驱逐
scratch/workspace 不再污染 KV/weight residency
临时缓冲区分配/fault 相比无 scratch pool 明显下降
TPOT/TTFT 稳定，fault ratio 可解释
```

## 16. 下一次更新应填写

## 17. 2026-05-26 实施更新

本次已经把 Stage N 从纯计划推进到 driver 控制面可用的第一版：

```text
已完成:
  P1b Python ctypes region wrapper
  P1c vLLM tag-only driver registration 第一版
  P2 VA->region lookup 的控制面结构
  P3 pool budget / region state / query ioctl 控制面

尚未完成:
  fault stats 按 pool/role 分类
  real residency incremental accounting
  driver victim selection
  UVM 原生 migrate/evict 路径 reclaim hook
```

补充更新：P6 soft reclaim 控制面第一版已经落地并通过 smoke。它仍不是最终
hot-path driver reclaim，但已经把“driver 选 victim 并执行迁移”从应用层下沉
到了 UVM driver 内部 helper，ioctl handler 只是入口之一。

代码改动：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h
  新增 Stage N pool/state/query ioctl ABI。

kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.c
  `uvm_phase_f_region_t` 扩展 state/priority/last_access_epoch。
  `uvm_va_space_t` 新增 region VA range tree 和 per-pool state。

kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
  register/unregister region 同步维护 id radix tree 和 VA range tree。
  新增 SET_POOL_BUDGET、GET_POOL_STATE、UPDATE_REGION_STATE、
  QUERY_REGION、QUERY_REGION_BY_VA handler。
  新增 RECLAIM_POOL handler：扫描同 pool region，按 driver victim policy
  选择最多 8 个 victim，并在 SOFT/ENFORCE mode 下调用 UVM migrate-to-CPU。
  RECLAIM_POOL 主流程已抽成 `uvm_phase_n_reclaim_pool_execute()`；该函数现在
  只依赖 `uvm_va_space_t *`，不再依赖 userspace `filp`。
  新增 `uvm_phase_n_reclaim_pool_over_budget()`，为下一步 make-resident/PMM
  eviction 自动触发复用做准备。
  新增 `uvm_phase_n_make_resident_reclaim_hook()` 和
  `uvm_phase_n_auto_reclaim_*` module params；当前只做热路径 trace/preflight，
  不在 locked context 内执行迁移。

kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
  `uvm_va_block_make_resident()` 在 GPU residency 成功后调用
  `uvm_phase_n_make_resident_reclaim_hook()`，输出
  `phase_n_auto_reclaim_hook`、`would_reclaim`、`skip=locked_context`。

kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_migrate.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_migrate.h
  新增 `uvm_migrate_managed_range_to_cpu()`，让 Stage N reclaim 直接调用
  UVM managed migration 内部 helper，而不是通过 `uvm_api_migrate()` 间接
  复用 ioctl 路径。该 helper 兼容有注册 mm、普通 ioctl current mm 和未来
  kernel worker 无 mm 三种场景。

workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/uvm_test/phase_f_smoke.cpp
  新增 Stage N userspace helper 和 `--step phase-n-region` smoke。
  新增 `uvm_driver_phase_n_reclaim_pool` helper 和 `--step phase-n-reclaim`
  smoke。
  `phase-n-reclaim` reclaim 后会再次查询 pool state，输出 post-state
  resident/candidate/event 字段。

workloads/vllm/vllm/vllm/device_allocator/uvm.py
  新增 Stage N ctypes wrapper、feature probe 和 query helper。
  `driver_get_pool_state` 返回 over_budget/reclaim_target/victim_candidate 字段。
  新增 `driver_reclaim_pool()` 和 `has_pool_reclaim_ioctl()`。

workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
  tag-only 模式下 request_gpu_residency 不做应用层决策。
  register/state/access 会尝试同步 driver tag/state。
  `VLLM_UVM_DRIVER_POOL_BUDGETS` 可把 pool budget 写入 driver。
  `get_uvm_driver_pool_state_snapshot()` 可抓取三池 driver state。
  新增 `soft_reclaim_uvm_driver_pool()` 诊断入口；默认不在 vLLM 热路径中
  自动触发。

workloads/vllm/run_stage_n_tag_only_three_pool_smoke.sh
workloads/vllm/run_stage_n_three_pool_validation.sh
  新增 auto hook 验收 plumbing：`kernel_since_start.log`、`phase_n_auto_hook_*`
  counters，以及 `REQUIRE_PHASE_N_AUTO_HOOK=1` 强检查开关。
```

新增运行配置：

```text
VLLM_UVM_RESIDENCY_TAG_ONLY=1
VLLM_UVM_DRIVER_TAG_ENABLE=1
VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE=1
VLLM_UVM_DRIVER_POOL_BUDGET_MODE=trace_only|soft|enforce
VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES=104857600
VLLM_UVM_DRIVER_POOL_BUDGETS=kv=10737418240,weights=10737418240,scratch=104857600
# temp/workspace 会被解析为 scratch alias:
VLLM_UVM_DRIVER_POOL_BUDGETS=kv=10737418240,weight=10737418240,temp=104857600
```

三池预算建议：

```text
KV cache:
  初始 10GiB，根据 max_num_seqs/max_model_len 和实际 KV block size 调整。

Weight:
  初始 10GiB，dense/shared weight protected，MoE expert 可逐步纳入 evictable。

Temp scratch:
  默认 100MiB。该值来自 Stage M 观察：少量额外 GPU 空间即可覆盖大量
  runtime scratch/workspace 热分配。第一轮建议 sweep 64/100/128/256MiB。
```

验证结果：

```text
python3 -m py_compile uvm.py uvm_pool_residency.py
  PASS

git diff --check
  PASS

make -C workloads/vllm/vllm/uvm_test -j2 phase_f_smoke
  PASS

VLLM_UVM_PHASE_F_IOCTL_MODE=policy ./run_stage_m_phase_f_smoke.sh --no-enable-tests --step phase-n-reclaim
  PASS
  attempted=1 success=1 failure=0 reclaimed=1048576 victim_status=0x0

make -C kernel-module/nvidia-module/kernel-open modules NV_KERNEL_MODULES=nvidia-uvm
  uvm.c / uvm_migrate.c / uvm_va_space.c 编译通过，nvidia-uvm.o 链接通过
  最终 modpost 因只构建 nvidia-uvm、缺少 nvidia 核心模块符号失败
  这不是 Stage N 代码编译错误
  当前仍有既有 warning: uvm_ioctl frame size 1152 > 1024

2026-05-27 本轮 smoke:
  `phase-n-reclaim` 在当前机器上失败于 register-region，kernel log:
  `ioctl 82 not found` / `ioctl 83 not found`。
  判断：当前加载的 nvidia_uvm 不是包含 Stage N ioctl 的新模块，需要完整构建
  并重载 nvidia/nvidia-uvm 后再验证 post-state resident_gpu。

2026-05-27 make-resident hook preflight:
  `make -C kernel-module/nvidia-module modules`
    PASS
  新增 `phase_n_auto_reclaim_hook` trace/preflight，默认关闭。
  该 hook 已接入 GPU make-resident 成功路径，但执行型 reclaim 仍显式为
  `execute=0 skip=locked_context`，等待异步/外层安全点。

2026-05-27 auto hook validation plumbing:
  `run_stage_n_tag_only_three_pool_smoke.sh`
    新增 kernel_since_start.log 和 phase_n_auto_hook_* counters。
    `REQUIRE_PHASE_N_AUTO_HOOK=1` 时要求 hook trace > 0。
  `run_stage_n_three_pool_validation.sh`
    透传 REQUIRE_PHASE_N_AUTO_HOOK，并在总 summary 中输出 Auto Hook Evidence。
  `make -C kernel-module/nvidia-module modules`
    PASS
  `bash -n` 两个 Stage N smoke/validation 脚本
    PASS

2026-05-27 auto hook validation PASS:
  修复 `RESET_COUNTERS=1` 重载模块时未传 auto hook 参数的问题。
  修复 tag-only smoke 只读 kernel journal、不读 procfs route log 的假阴性。
  在 `uvm_va_block_service_finish()` 补 replayable fault finish hook。
  `REQUIRE_PHASE_N_AUTO_HOOK=1 RESET_COUNTERS=1 UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 ./run_stage_n_three_pool_validation.sh`
    PASS
  run_root=log/stage_n_three_pool_validation_20260527_093758
  phase_n_auto_hook_records=495
  failed_requests=0

2026-05-27 automatic soft reclaim validation PASS:
  修复 userspace 多 `/dev/nvidia-uvm` fd 场景下 region/budget/state 只注册到单个
  VA space 的问题；driver registration 现在会应用到已有 UVM fd，auto hook 不再只
  看到 `skip=no_region`。
  在 replayable fault batch 外层加入 `uvm_phase_n_auto_reclaim_outer_hook()`，
  在不持 va_block lock 的安全点执行 `uvm_phase_n_reclaim_pool_execute()`。
  新增/接入稳定性参数：
    `uvm_phase_n_auto_reclaim_every`
    `uvm_phase_n_auto_reclaim_scan_every`
    `uvm_phase_n_auto_reclaim_max_victims`
    `uvm_phase_n_auto_reclaim_cooldown_epochs`
    `uvm_phase_n_auto_reclaim_failure_backoff_epochs`
  自动路径支持 under-budget cached fast skip、success cooldown、failure backoff、
  batch victim 上限和 post-reclaim pool-state accounting。
  victim tie-break 在同 priority/last_access_epoch 下优先较小 resident bytes，
  避免 smoke 中迁出大块 weight 后造成明显性能回退。
  `REQUIRE_PHASE_N_AUTO_RECLAIM=1 RESET_COUNTERS=1 UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 ./run_stage_n_three_pool_validation.sh`
    PASS
  run_root=log/stage_n_three_pool_validation_20260527_110849
  phase_n_auto_hook_records=79
  phase_n_auto_hook_would_reclaim_records=20
  phase_n_auto_hook_outer_records=59
  phase_n_auto_hook_pool_records=69
  phase_n_auto_hook_execute_records=1
  phase_n_auto_hook_success_records=1
  phase_n_auto_hook_reclaimed_records=1
  driver_pool_state_failures=0
  driver_pool_state_all_pools_present=1
  allocator_driver_pool_state_all_status_ok=1
  phase_n_reclaim_kv_victim_count=1
  phase_n_reclaim_weights_victim_count=1
  phase_n_reclaim_scratch_victim_count=1
  failed_requests=0
  benchmark:
    Successful requests: 1
    Failed requests: 0
    Benchmark duration: 4.56s
    Mean TTFT: 744.21ms
    Mean TPOT: 187.69ms
```

已完成的 P6 检查点：

```text
1. auto hook 已命中注册 Stage N region/pool，不再停留在 `skip=no_region`。
2. hook trace 已进入 pool/budget 分支：
   `phase_n_auto_reclaim_hook ... pool=... budget=... would_reclaim=...`
3. 执行型 reclaim 安全点采用 replayable fault batch 外层、不持 va_block lock 的路径。
4. 执行型路径已补 rate limit、scan throttle、success cooldown、failure backoff、
   batch victim 上限和 post-reclaim accounting。
```

```text
更新日期: 2026-05-27
当前阶段: P7 稳定性/性能 A/B harness 已落地，等待长实验数据
完成项:
  - driver-backed KV/weights/scratch tag-only smoke
  - GET_POOL_STATE 三池强验收
  - manual phase-n-reclaim 三池 victim/reclaim/post-state 验收
  - replayable fault batch 外层 automatic soft reclaim
  - rate limit / scan throttle / cooldown / failure backoff / small-victim preference
  - P7 A/B/sweep 脚本 `run_stage_n_p7_ab_sweep.sh`
新增证据:
  - log/stage_n_three_pool_validation_20260527_110849
  - DRY_RUN=1 的 P7 harness 语法/命令生成验证
回归指标:
  - failed_requests=0
  - Mean TTFT=744.21ms
  - Mean TPOT=187.69ms
阻塞项:
  - 暂无 P6 阻塞；P7 长实验结果尚未采集
下一检查点:
  - 多请求 A/B：native UVM / Stage M / Stage N trace / Stage N soft
  - 长期增量 accounting：migrate/evict/free/make-resident 后持续更新 pool state
  - hysteresis、per-pool cooldown、failure backoff 策略调参
  - enforce mode 前的预算贴近度和性能回归验收
```

### 14.5 P7 A/B 与调参脚本

P7 的长实验入口已经做成独立脚本：

```text
workloads/vllm/run_stage_n_p7_ab_sweep.sh
```

脚本能力：

```text
默认 variants:
  native_uvm
  stage_n_trace
  stage_n_soft

可选 variants:
  stage_n_soft_sweep
  stage_n_enforce_probe

输出:
  p7_runs.tsv
  p7_ab_summary.tsv
  p7_ab_summary.md
  每个 variant/repeat 的 command.sh、driver.log 和原始 run artifacts

验收检查:
  failed_requests=0
  Stage N soft 要求 would_reclaim / execute / success / reclaimed 都 > 0
  TTFT/TPOT 相对 baseline 的回退低于阈值
```

推荐先做 dry-run，确认命令和 run root：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
DRY_RUN=1 \
RUN_ROOT=log/stage_n_p7_ab_sweep_dryrun \
P7_VARIANTS=native_uvm,stage_n_trace,stage_n_soft \
REPEATS=1 \
PROMPTS=4 \
./run_stage_n_p7_ab_sweep.sh
```

正式多请求 A/B：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/stage_n_p7_ab_sweep_$(date -u +%Y%m%d_%H%M%S) \
P7_VARIANTS=native_uvm,stage_n_trace,stage_n_soft \
REPEATS=3 \
PROMPTS=8 \
REQUEST_RATE=2 \
OUTPUT_LEN=32 \
SERVER_MAX_NUM_SEQS=4 \
RESET_COUNTERS=1 \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_p7_ab_sweep.sh
```

scratch/cooldown/max-victim sweep：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/stage_n_p7_soft_sweep_$(date -u +%Y%m%d_%H%M%S) \
P7_VARIANTS=stage_n_soft_sweep \
REPEATS=2 \
PROMPTS=8 \
REQUEST_RATE=2 \
P7_SCRATCH_SWEEP_BYTES_CSV=16777216,67108864,104857600 \
P7_COOLDOWN_SWEEP_EPOCHS_CSV=0,1024,4096 \
P7_MAX_VICTIMS_SWEEP_CSV=1,2 \
RESET_COUNTERS=1 \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_p7_ab_sweep.sh
```

enforce 前探测：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/stage_n_p7_enforce_probe_$(date -u +%Y%m%d_%H%M%S) \
P7_VARIANTS=stage_n_enforce_probe \
REPEATS=2 \
PROMPTS=8 \
REQUEST_RATE=2 \
RESET_COUNTERS=1 \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_p7_ab_sweep.sh
```

建议下一次更新目标：

```text
运行 P7 A/B/sweep 后，把 `p7_ab_summary.md` 中的最佳参数写回默认配置；
若出现 TPOT/TTFT 回退或 reclaim 抖动，再补 per-pool cooldown/accounting 修正。
```
