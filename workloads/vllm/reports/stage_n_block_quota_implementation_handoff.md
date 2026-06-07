# Stage N 原生 UVM Block 分池配额实现交接文档

本文档用于新开对话窗口继续实现 Stage N 分池管理。它汇总了前期实验结论、最新方案、不可再走的旧路线、需要修改的文件、推荐实现顺序、验收指标和实验脚本形态。

主参考文档：

```text
workloads/vllm/reports/stage_n_va_block_pool_quota_plan.md
```

当前最新设计一句话：

```text
不要继续优化 region full scan；
不要把 phase_n_block 做成裸全局 2MB bucket；
继承原生 UVM managed_range/block_index；
用 Stage N block wrapper 维护 per-pool resident queue；
先 post-fault correction，后 pre-fault hard quota。
```

## 当前实现阶段标注

更新时间：2026-06-07

```text
当前已实现到：P8b - Manual Same-Pool Block Reclaim
```

- P8a `Tagged Range + Lazy Block Trace`：已实现并通过模块编译。
- P8b `Manual Same-Pool Block Reclaim`：已实现 driver block-quota fast path。
  - `RECLAIM_POOL` 在 `uvm_phase_n_block_quota_enable=1` 时不再扫 region。
  - 默认 `uvm_phase_n_block_quota_trace_only=1` 仍保持 P8a trace-only 行为。
  - 当 `uvm_phase_n_block_quota_trace_only=0` 且 pool mode 为 `SOFT/ENFORCE` 时，从 per-pool `resident_queue` 选择 same-pool block victim。
  - victim 跳过 `mixed_pool`、`in_reclaim`、`stale/nonresident`、wrong-pool、未标记 evictable、protected/active/pinned block。
  - 实际迁移通过已有 `uvm_migrate_managed_range_to_cpu()`，迁移范围使用 tagged range 与 native block 的 page-aligned 覆盖交集。
  - 成功后更新 `gpu_resident=false`、出队、`resident_blocks--`、`reclaim_success/same_pool_victims++`；失败则重新入队并累计 failure。
- P8c `Auto Post-Fault Block Quota`：未实现。
- P8d `Capacity Stability Proof`：未实现。
- P8e `Pre-Fault Hard Quota`：未实现。

## 1. 历史背景与当前结论

### 已经证明可行的部分

当前项目已经实现并验证过：

- vLLM allocator 能通过 `cudaMallocManaged` 提供 UVM allocation。
- vLLM 能把 allocation/region 分类为 `kv`、`weights`、`scratch`。
- driver 已有 region registry：
  - `phase_f_regions`
  - `phase_f_region_ranges`
- driver 已有 pool budget/state/reclaim ioctl：
  - `SET_POOL_BUDGET`
  - `GET_POOL_STATE`
  - `RECLAIM_POOL`
  - `QUERY_REGION`
  - `QUERY_REGION_BY_VA`
- driver 已有 manual reclaim 和 auto reclaim hook。
- replayable fault batch 外层已有 Stage N hook。
- P7/hard partition 脚本已经能跑出证据。

### 已经放弃的旧路线

不要继续沿着以下方向做主线：

```text
region full scan reclaim
region candidate sort
按 active/hot/cold 做复杂 victim selection
全局 align_down(address, 2MB) quota bucket
GPU physical chunk 硬切分
```

原因：

- region full scan 在 vLLM 大模型场景下开销极高。
- hard partition probe 已经出现长时间 fault/reclaim storm。
- 当前瓶颈不是“victim 排序不够好”，而是扫描 region 和 residency query 太重。
- 原生 UVM `uvm_va_block_t` 不等于全局 2MB bucket，裸 2MB bucket 会人为制造 UVM 本来没有的 mixed range 问题。

## 2. 关键概念澄清

### 原生 UVM block

原生 UVM 层级是：

```text
uvm_va_space
  -> uvm_va_range
      -> uvm_va_block
```

关键源码：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_range.h`
  - `uvm_va_range_managed_t` 内有 `uvm_va_block_t **blocks`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_range.c`
  - `uvm_va_range_block_index()`
  - `uvm_va_range_block_create()`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
  - `uvm_va_block_create()`

原生 UVM block 的 identity 近似是：

```text
managed_range + native_block_index
```

它不会跨 parent managed range。即使两个独立 1MB `cudaMallocManaged` allocation 在 VA 上相邻、落在同一个 2MB aligned window 内，原生 UVM 也会把它们作为两个 managed range 下的两个 block 管理。

### Stage N block wrapper

新的 `phase_n_block` 不应该是：

```text
key = align_down(address, 2MB)
```

而应该是：

```text
key = managed_range + native_block_index
```

`phase_n_block` 是原生 UVM block 的轻量 wrapper，用于 Stage N 自己的：

```text
pool ownership
resident boolean
queue membership
same-pool victim selection
quota counters
```

### Tagged range

vLLM/driver 现有 region registry 仍然保留。`register_region` 不负责创建 block，只负责记录：

```text
VA range
pool
role
flags
region_id
alignment metadata
```

fault/access 命中时，再从 address 找到 tagged range 和原生 UVM block identity。

### mixed_pool

最新定义：

```text
同一个原生 UVM block 覆盖的 VA 范围内，
Stage N 自己注册了多个不同 pool 的 tagged subrange。
```

它不是：

```text
一个原生 uvm_va_block_t 同时拥有多个 uvm_va_range_t。
```

两个独立 `cudaMallocManaged` range 不会被原生 UVM 合成一个 va_block。

## 3. 当前代码入口

### Driver 侧

主要文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_api.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu.c
```

已有结构和入口：

```text
uvm_va_space.h:
  uvm_phase_f_region_t
  phase_f_regions
  phase_f_region_ranges
  phase_n_pools

uvm.c:
  uvm_api_phase_f_register_region()
  uvm_api_phase_f_unregister_region()
  uvm_api_phase_n_get_pool_state()
  uvm_phase_n_reclaim_pool_execute()
  uvm_api_phase_n_reclaim_pool()
  uvm_phase_n_make_resident_reclaim_hook()
  uvm_phase_n_auto_reclaim_outer_hook()
  uvm_phase_n_reclaim_pool_over_budget()

uvm_api.h:
  uvm_phase_n_make_resident_reclaim_hook()
  uvm_phase_n_auto_reclaim_outer_hook()
  uvm_phase_n_reclaim_pool_over_budget()

uvm_ioctl.h:
  UVM_PHASE_F_REGISTER_REGION
  UVM_PHASE_N_GET_POOL_STATE
  UVM_PHASE_N_RECLAIM_POOL
  UVM_PHASE_N_POOL_KV
  UVM_PHASE_N_POOL_WEIGHT
  UVM_PHASE_N_POOL_SCRATCH
```

已有 hook 位置：

```text
uvm_va_block.c:
  make-resident finish 后会调用 uvm_phase_n_make_resident_reclaim_hook(..., execute_allowed=NV_FALSE)

uvm_gpu_replayable_faults.c:
  replayable fault batch 外层已有 auto hook 路径
```

已有 fault stats 可复用：

```text
uvm_gpu.c:
  update_stats_parent_gpu_phase_n_fault_instance()
  通过 phase_f_region_ranges 做 address -> region -> pool/role stats
```

### vLLM 侧

主要文件：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
```

已有入口：

```text
uvm_allocator.cpp:
  uvm_malloc()
  uvm_free()
  uvm_driver_phase_n_set_pool_budget_internal()
  uvm_driver_phase_n_get_pool_state_internal()
  uvm_driver_phase_n_reclaim_pool_internal()
  uvm_driver_phase_n_update_region_state_internal()
  uvm_driver_phase_n_query_region_internal()
  uvm_driver_phase_n_query_region_by_va_internal()

uvm.py:
  ctypes wrapper for phase_n ioctl helpers

uvm_pool_residency.py:
  pool budget config
  driver pool state logging
```

已有实验脚本：

```text
workloads/vllm/run_stage_n_tag_only_three_pool_smoke.sh
workloads/vllm/run_stage_n_three_pool_validation.sh
workloads/vllm/run_stage_n_p7_ab_sweep.sh
workloads/vllm/run_stage_n_hard_partition_proof.sh
workloads/vllm/check_stage_n_hard_partition.py
```

## 4. 新增设计

### 新增 driver 数据结构

在 `uvm_va_space.h` 中新增 Stage N block wrapper 和 per-pool queue。

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
    NvBool stale;
    NvU64 resident_epoch;
    NvU64 last_access_epoch;
    struct list_head queue_node;
} uvm_phase_n_block_t;
```

建议 pool state：

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

在 `uvm_va_space_t` 中新增：

```c
struct radix_tree_root phase_n_blocks;
uvm_phase_n_block_pool_t phase_n_block_pools[UVM_PHASE_N_POOL_STATE_MAX];
```

注意：

```text
phase_n_blocks 的 key 必须包含 managed_range/native_block_index 的信息。
不要只用 align_down(address, 2MB)。
```

如果 radix tree 不适合复合 key，第一版可以：

```text
key = hash64((uintptr_t)managed_range, native_block_index)
```

或先用小型 list/hash table 实现 trace-only，等行为正确后再优化。

### 新增 helper

建议在 `uvm.c` 中新增以下静态 helper：

```text
uvm_phase_n_find_tagged_region(va_space, address)
uvm_phase_n_native_block_identity(va_space, address, &managed_range, &native_index, &native_block)
uvm_phase_n_block_key(managed_range, native_index)
uvm_phase_n_find_block(va_space, managed_range, native_index)
uvm_phase_n_create_block(va_space, tagged_region, managed_range, native_index, native_block)
uvm_phase_n_find_or_create_block(va_space, address, &block)
uvm_phase_n_detect_mixed_pool(va_space, block)
uvm_phase_n_mark_block_resident(va_space, block, epoch)
uvm_phase_n_pop_same_pool_victim(va_space, pool, trigger_block)
uvm_phase_n_reclaim_block_to_cpu(va_space, block, gpu_id)
uvm_phase_n_reclaim_pool_from_queue(va_space, pool, gpu_id, max_victims)
```

第一版 trace-only 可以只实现到：

```text
find tagged region
find native block identity
create block wrapper
detect mixed_pool
increment counters
log trace
```

不做实际 reclaim。

## 5. 实现阶段

### P8a：Tagged Range + Lazy Block Trace

目标：

```text
register_region 只记录 tagged range；
fault/access 命中时 lazy materialize Stage N block wrapper；
不改变 reclaim 行为；
确认 managed_range/native_block_index 路径可用；
确认 mixed_pool_blocks = 0。
```

具体改动：

1. `uvm_va_space.h`

   - 增加 `uvm_phase_n_block_t`。
   - 增加 `uvm_phase_n_block_pool_t`。
   - 在 `uvm_va_space_t` 中增加 `phase_n_blocks` 和 `phase_n_block_pools`。

2. `uvm.c`

   - 初始化/销毁新结构。
   - `REGISTER_REGION` 保持 tagged range 逻辑，不创建全部 block。
   - 在 auto hook 或 make-resident hook 中调用 lazy materialize trace helper。
   - 输出 trace：

     ```text
     phase_n_tagged_range_register
     phase_n_block_materialize
     phase_n_block_mixed_pool
     phase_n_block_summary
     ```

3. `GET_POOL_STATE`

   - block quota enable 时返回 block counters。
   - 不要在 P8a 中改变旧 region path 的默认行为。

验收：

```text
tagged_range_register_records > 0
block_materialize_records > 0
kv_materialized_block_records > 0
weight_materialized_block_records > 0
scratch_materialized_block_records > 0
mixed_pool_blocks = 0
fault_path_region_scan_records = 0
failed_requests = 0
```

### P8b：Manual Same-Pool Block Reclaim

目标：

```text
RECLAIM_POOL 在 block quota enable 时不扫 region；
从 per-pool resident_queue 中取 victim；
只允许 same-pool victim；
调用 migrate-to-CPU helper。
```

具体改动：

1. block materialize/access 后：

   ```text
   block.gpu_resident = true
   block.queued = true
   pool.resident_blocks++
   list_add_tail(block, pool.resident_queue)
   ```

2. `RECLAIM_POOL(pool)`：

   ```text
   if block_quota_enable:
       victim = pop same-pool victim
       migrate victim block_start..block_end 或 covered_start..covered_end
       update counters
   else:
       fallback old region reclaim
   ```

3. victim skip 条件：

   ```text
   skip trigger block
   skip mixed_pool
   skip in_reclaim
   skip stale/dead
   skip wrong pool
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
不进入 region full scan。
```

hook 路径：

```text
address
  -> tagged range
  -> native UVM block identity
  -> phase_n_block wrapper
  -> pool resident queue
  -> same-pool reclaim
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

目标：

```text
证明各 pool 容量稳定；
证明不再出现 region enforce fault storm；
证明 same-pool-only eviction。
```

需要新增脚本：

```text
workloads/vllm/run_stage_n_block_quota_smoke.sh
workloads/vllm/check_stage_n_block_quota.py
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

只有 P8a-P8d 稳定后再做。

目标：

```text
把 quota check 前移到 make-resident admission 前；
最终 max_over_blocks = 0。
```

模式：

```text
UVM_PHASE_N_PRE_FAULT_QUOTA_MODE=trace|soft|enforce
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

## 6. vLLM allocator 对齐计划

这不是 P8a 的硬依赖，但建议并行做 trace。

原因：

```text
原生 UVM block 不会跨 UVM managed range；
但同一个 managed range 内如果 Stage N 注册不同 pool subrange，仍会 mixed_pool；
allocator 2MB 对齐/per-pool arena 可以降低 mixed_pool 风险并改善 UVM block 效率。
```

建议：

```text
VLLM_UVM_POOL_ARENA_ENABLE=0/1
VLLM_UVM_ALLOC_ALIGN_2MB=0/1
```

第一版只做日志：

```text
TRACE_UVM_ALLOC_ALIGNMENT ptr=... size=... aligned_2mb=... pool=...
TRACE_UVM_POOL_ARENA pool=... raw=... usable=... bytes=...
```

不要第一步就大改 allocator 语义。先确认：

```text
mixed_pool_blocks 是否真实出现；
哪些 allocation/range 导致 mixed；
是否需要 per-pool arena。
```

## 7. 验收脚本形态

### Block quota smoke

建议新增：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

RUN_ROOT=log/stage_n_block_quota_smoke_$(date -u +%Y%m%d_%H%M%S) \
RESET_COUNTERS=1 \
UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=1 \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_block_quota_smoke.sh
```

第一版可以复用 `phase_f_smoke` 或新增一个小 C++ smoke：

```text
cudaMallocManaged A/B/C
register as KV/weights/scratch
touch addresses to trigger native UVM block materialization
query block quota state
verify materialize counters
verify mixed_pool_blocks = 0
```

### vLLM mini probe

建议新增：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

RUN_ROOT=log/stage_n_p8_block_quota_probe_$(date -u +%Y%m%d_%H%M%S) \
P7_VARIANTS=stage_n_block_quota_trace \
REPEATS=1 \
PROMPTS=1 \
REQUEST_RATE=1 \
OUTPUT_LEN=8 \
SERVER_MAX_NUM_SEQS=1 \
RESET_COUNTERS=1 \
UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_p7_ab_sweep.sh
```

验收：

```text
failed_requests = 0
block_materialize_records > 0
mixed_pool_blocks = 0
fault_path_region_scan_records = 0
```

## 8. 构建与加载提醒

UVM module 构建时不要只编 `nvidia-uvm` 而缺少其他 module symbol。之前出现过：

```text
ERROR: modpost: "nvUvmInterface..." undefined
```

优先使用项目现有完整构建方式。如果只改 UVM 并确认符号环境完整，可尝试：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open
make -j"$(nproc)" modules
```

vLLM UVM helper：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
make -C vllm/uvm_test uvm
```

加载前先停止 vLLM：

```bash
pkill -f "vllm serve" || true
pkill -f "run_stage_n" || true
```

具体 reload 命令以当前机器已有脚本/流程为准。新窗口实施前应先确认当前加载的 `nvidia-uvm.ko` 是否来自本工作树。

## 9. 新窗口开工顺序

建议新窗口按下面顺序执行：

1. 阅读：

   ```text
   workloads/vllm/reports/stage_n_va_block_pool_quota_plan.md
   workloads/vllm/reports/stage_n_block_quota_implementation_handoff.md
   ```

2. 只实现 P8a trace-only。

3. 不改旧 region reclaim 默认行为。

4. 在 `uvm_va_space.h` 添加 block wrapper/pool queue。

5. 在 `uvm.c` 添加 lazy materialize helper。

6. 在 auto hook 或 make-resident hook 中只记录 materialize trace。

7. 编译 driver 和 `uvm_test` helper。

8. 先跑 synthetic/smoke，不直接跑长 vLLM benchmark。

9. P8a PASS 后再做 P8b manual reclaim。

10. P8b PASS 后再做 P8c auto post-fault quota。

## 10. 不要踩的坑

### 不要把不同 UVM range 合成一个 Stage N block

错误：

```text
key = align_down(address, 2MB)
```

正确：

```text
key = managed_range + native_block_index
```

### 不要在 P8a 做实际 reclaim

P8a 只做 trace/materialize/counter。否则调试面太大。

### 不要在 va_block lock 内跨 block migrate

Pre-fault hard quota 必须等 P8a-P8d 稳定后再做。

### 不要让 GET_POOL_STATE 继续触发 hot path scan

block quota enable 后，GET_POOL_STATE 应优先读 block counters。

### 不要把 mixed_pool 当作常态处理

mixed_pool 是 allocator/tagging 问题，应当导致 proof 失败。第一版不要在 driver 内做复杂 subrange/page-mask reclaim。

## 11. 最小完成定义

下一轮最小可交付成果是 P8a：

```text
新增 phase_n_block wrapper 数据结构；
fault/access 能 lazy materialize block wrapper；
identity 继承 managed_range + native_block_index；
能输出 block_materialize trace；
GET_POOL_STATE 能看到 block counters；
mixed_pool_blocks 可统计；
旧 region 路径仍可回退；
small smoke failed_requests = 0。
```

P8a 完成后，再推进 P8b/P8c。
