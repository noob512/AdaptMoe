# Stage N Native VA Range / VA Block 分池管理实现文档

更新时间：2026-06-09

本文档描述 Stage N 分池配额管理从当前 `phase_n_block_t` wrapper 实现，演进到直接管理原生 UVM `uvm_va_range_managed_t` 与 `uvm_va_block_t` 的性能优先方案。

## 当前实现状态

```text
当前已实现到：P8f - Native VA Range / VA Block Pool Quota Fast Path
```

本次已落地：

```text
uvm_phase_n_native_block_enable=1
  -> REGISTER_REGION 将 KV/WEIGHT tag attach 到 native uvm_va_range_managed_t
  -> register-time split 对齐 region base/end 边界
  -> 已存在 native va_block 继承 tag，并对 GPU-resident block 做 backfill
  -> 新创建 va_block 从 parent managed_range 继承 tag
  -> make-resident finish 后直接更新 va_block->phase_n、per-pool FIFO 和 counters
  -> GET_POOL_STATE 继续走 block fast path
  -> RECLAIM_POOL 直接从 native va_block FIFO 选择 same-pool victim
  -> destroy/unregister 时清理 native FIFO/counters
```

现有 wrapper path 仍保留为 fallback：

```text
uvm_phase_n_block_quota_enable=1
uvm_phase_n_native_block_enable=0
  -> 继续使用 phase_n_block_t wrapper 路径
```

当前阶段的语义分类先保持简单：

```text
KV     -> pool=UVM_PHASE_N_POOL_KV
WEIGHT -> pool=UVM_PHASE_N_POOL_WEIGHT
```

暂不要求细分 KV cache 的 layer、expert、head、block id，也暂不要求细分 weight 的 layer/expert/parameter name。当前目标是先把这两类内存稳定分开，并在 driver 中按原生 UVM block 做 resident accounting、FIFO victim selection 和 same-pool reclaim。

## 1. 当前已实现状态

当前项目已经实现到：

```text
P8e-a - Allocator-side Early KV Tag
```

已有能力如下。

### 1.1 Tagged range registry

driver 已有 Phase F/Stage N tagged range registry：

```text
uvm_va_space_t:
  phase_f_regions
  phase_f_region_ranges
```

相关位置：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
```

`UVM_PHASE_F_REGISTER_REGION` 当前会创建 `uvm_phase_f_region_t`，并把其中的 `va_node` 插入：

```c
va_space->phase_f_region_ranges
```

它提供的能力是：

```text
fault/access address -> registered region -> pool/role/flags
```

也就是说，当前 driver 的语义源头仍然是旁路 range registry，而不是 native `uvm_va_range_managed_t` 本身。

### 1.2 Stage N block wrapper

当前已经新增：

```text
uvm_phase_n_block_t
uvm_phase_n_block_pool_t
va_space->phase_n_blocks
va_space->phase_n_block_pools[]
```

当前 block identity 是：

```text
managed_range + native_block_index
```

`phase_n_block_t` 保存：

```text
managed_range
native_block_index
block_start / block_end
covered_start / covered_end
pool / flags
mixed_pool
gpu_resident
queued
in_reclaim
stale
resident_epoch / last_access_epoch
queue_node
```

每个 pool 有自己的 FIFO queue：

```text
va_space->phase_n_block_pools[pool].resident_queue
```

因此当前已经不是 region full scan 的设计，而是 block 粒度的 sidecar accounting：

```text
native uvm_va_block_t
  -> managed_range + native_block_index
  -> phase_n_block_t wrapper
  -> per-pool resident_queue
```

### 1.3 P8a/P8b/P8c/P8d/P8e-a 能力

已完成的阶段可以概括为：

```text
P8a: tagged range + lazy block trace
P8b: manual same-pool block reclaim
P8c: auto post-fault block quota
P8d: synthetic/vLLM block quota proof scripts
P8e-a: allocator-side early KV tag
```

其中关键点：

- `GET_POOL_STATE` 在 `uvm_phase_n_block_quota_enable=1` 时走 block-quota fast path，不再扫 region。
- `RECLAIM_POOL` 在 block quota enable 时从 per-pool resident queue 选 same-pool victim。
- FIFO 策略已经明确：第一次 resident 时入队尾，后续访问只更新 epoch，不移动队列。
- `uvm_migrate_managed_range_to_cpu()` 被复用为实际迁移路径。
- vLLM allocator 已支持 `VLLM_UVM_DRIVER_KV_TAG_ENABLE=1`，在 KV managed allocation 后立即向 driver 注册整段 KV VA。

### 1.4 当前问题

P8e-a 解决的是：

```text
KV tag 不再等 Python 后同步，而是在 allocator 侧尽早注册到 driver。
```

但它没有完全解决：

```text
已经 resident 的 native UVM block 如何进入 Stage N block accounting。
```

原因是当前 `phase_n_block_t` 仍然主要通过 hook 中的 address materialize 创建：

```text
address
  -> phase_f_region_ranges
  -> managed_range + native_block_index
  -> phase_n_block_t
  -> resident_queue / resident_blocks
```

真实 vLLM 的 KV 初始化虽然已经证明会产生大量 fault，且离线分析可以正确判断这些 fault 属于 KV，但当前 driver online accounting 仍可能漏掉部分 native block。主要原因是当前主路径仍然围绕 fault address 和 wrapper materialize，而不是直接在 native `uvm_va_block_t` residency 更新处维护状态。

## 2. 新目标：性能优先的 native 管理路线

科研项目当前目标调整为：

```text
性能优先，而不是最小侵入优先。
```

因此下一阶段不应继续长期优化 `phase_n_block_t` wrapper，而应把 Stage N 语义和队列状态直接推进 native UVM 结构：

```text
uvm_va_range_managed_t 保存 allocation/pool tag
uvm_va_block_t 保存 block resident/FIFO/reclaim 状态
per-pool resident_queue 直接挂 uvm_va_block_t
```

目标 hot path 从当前：

```text
fault/make-resident address
  -> phase_f_region_ranges range lookup
  -> uvm_va_range_find
  -> native_block_index
  -> phase_n_blocks radix lookup
  -> phase_n_block_t wrapper
  -> wrapper resident_queue
```

变为：

```text
uvm_va_block_make_resident_finish()
  -> va_block->managed_range->phase_n_tag
  -> va_block->phase_n_state
  -> list_add_tail(&va_block->phase_n_queue_node, pool_queue)
```

这样可以去掉 hot path 上的额外 range lookup、radix lookup 和 wrapper allocation/lookup。

## 3. 语义简化：只区分 KV 与 WEIGHT

当前阶段不做细粒度语义：

```text
KV:
  pool = UVM_PHASE_N_POOL_KV
  role = generic_kv 或 kv_block
  flags = EVICTABLE

WEIGHT:
  pool = UVM_PHASE_N_POOL_WEIGHT
  role = generic_weight
  flags = EVICTABLE 或 PROTECTED，按实验配置决定
```

暂不依赖：

```text
layer_id
expert_id
head_id
kv block id
weight tensor name
```

这些字段可以继续保留在 `uvm_phase_f_region_t` 或 userspace trace 中，但 native fast path 只看：

```text
valid
pool
role
flags
region_id/allocation_id
base/length
```

这样实现成本更低，也更符合当前验证目标：先证明 driver 可以把 KV 和 WEIGHT 分池、计数、FIFO 驱逐。

## 4. P8f 实现计划总览

建议把下一阶段命名为：

```text
P8f - Native VA Range / VA Block Pool Quota
```

子阶段：

```text
P8f-a: Native va_range semantic tag
P8f-b: Native va_block FIFO state
P8f-c: make-resident direct accounting
P8f-d: direct va_block reclaim
P8f-e: split/destroy/unregister lifecycle maintenance
P8f-f: wrapper hot path retirement / debug fallback
```

## 5. P8f-a：把语义 tag 放进 native va_range

### 5.1 修改目标

修改：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_range.h
```

在 `uvm_va_range_managed_t` 中增加 Stage N range tag。

建议新增结构：

```c
typedef struct
{
    NvBool valid;
    NvBool mixed;
    NvU64 region_id;
    NvU64 allocation_id;
    NvU64 base;
    NvU64 length;
    NvU32 pool;
    NvU32 role;
    NvU32 flags;
} uvm_phase_n_range_tag_t;
```

然后在 `uvm_va_range_managed_struct` 中增加：

```c
uvm_phase_n_range_tag_t phase_n_tag;
```

### 5.2 register_region 行为变化

当前 `uvm_api_phase_f_register_region()` 只做：

```text
创建 uvm_phase_f_region_t
插入 phase_f_region_ranges
插入 phase_f_regions
记录 phase_n_tagged_range_register trace
```

P8f-a 后应增加 native attach：

```text
REGISTER_REGION(base, length, pool)
  -> 保留 phase_f_region_ranges 兼容路径
  -> 找到覆盖 [base, end] 的 managed va_range
  -> 必要时 split va_range，使语义边界和 va_range 边界对齐
  -> 给覆盖范围内的 managed_range 写 phase_n_tag
```

### 5.3 split 策略

为了让 hot path 简单，注册时可以接受较高开销：

```text
如果 base 落在某个 va_range 中间：
  split at base - 1

如果 end 落在某个 va_range 中间：
  split at end
```

split 后，被 tag 的 managed range 应尽量满足：

```text
managed_range->va_range.node.start >= tag.base
managed_range->va_range.node.end   <= tag.end
```

这样 make-resident 时无需再查 `phase_f_region_ranges`，只读：

```c
va_block->managed_range->phase_n_tag
```

### 5.4 重叠 tag 处理

当前阶段只区分 KV 和 WEIGHT，所以建议采取性能优先策略：

```text
同 pool 重复注册:
  允许幂等更新，或者保持已有 tag

不同 pool 重叠:
  标记 mixed
  不进入 native fast FIFO
  fallback 到 phase_f_region_ranges/wrapper debug path
```

科研验证阶段更推荐直接输出 trace 并拒绝 fast path：

```text
phase_n_native_range_tag_mixed
phase_n_native_range_tag_reject
```

避免错误地把一个 mixed block 当成纯 KV 或纯 WEIGHT 驱逐。

## 6. P8f-b：把 FIFO 状态放进 native va_block

### 6.1 修改目标

修改：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h
```

在 `uvm_va_block_t` 中新增 Stage N block 状态。

建议新增结构：

```c
typedef struct
{
    struct list_head queue_node;

    NvU64 region_id;
    NvU64 allocation_id;
    NvU64 resident_epoch;
    NvU64 last_access_epoch;

    NvU32 pool;
    NvU32 role;
    NvU32 flags;

    NvBool tag_valid;
    NvBool gpu_resident;
    NvBool queued;
    NvBool in_reclaim;
    NvBool mixed_pool;
    NvBool stale;
} uvm_phase_n_va_block_state_t;
```

然后在 `uvm_va_block_struct` 中增加：

```c
uvm_phase_n_va_block_state_t phase_n;
```

当前先按单 GPU 实验实现：

```text
phase_n.gpu_resident
```

后续多 GPU 可升级为：

```c
uvm_processor_mask_t resident_gpus;
```

### 6.2 block 创建时继承 range tag

修改：

```text
uvm_va_range_block_create()
```

当创建 `uvm_va_block_t` 后，如果 parent managed range 有 valid tag：

```text
block->phase_n.tag_valid = true
block->phase_n.region_id = managed_range->phase_n_tag.region_id
block->phase_n.allocation_id = managed_range->phase_n_tag.allocation_id
block->phase_n.pool = managed_range->phase_n_tag.pool
block->phase_n.role = managed_range->phase_n_tag.role
block->phase_n.flags = managed_range->phase_n_tag.flags
INIT_LIST_HEAD(&block->phase_n.queue_node)
```

如果 range tag 是 mixed：

```text
block->phase_n.mixed_pool = true
block 不进入 native fast FIFO
```

### 6.3 block split 时继承状态

修改：

```text
uvm_va_block_split()
```

新 block 应继承旧 block 的语义 tag：

```text
new_block->phase_n.pool = existing_block->phase_n.pool
new_block->phase_n.role = existing_block->phase_n.role
new_block->phase_n.flags = existing_block->phase_n.flags
new_block->phase_n.tag_valid = existing_block->phase_n.tag_valid
```

resident 状态要谨慎处理：

```text
如果 split 前旧 block 已 gpu_resident:
  根据新旧 block 的 resident page mask 判断各自是否仍有 GPU resident pages
  分别维护 queued/resident_blocks
```

第一版可以保守处理：

```text
split 发生时将相关 block 标记 stale
从 FIFO 出队
等待下一次 make-resident hook 重新入队
```

这牺牲 split 时瞬间 accounting 精度，但可以降低初版复杂度。

## 7. P8f-c：make-resident 直接更新 native block accounting

### 7.1 现有接入点

当前 driver 已经在 make-resident 完成后调用：

```text
uvm_phase_n_make_resident_reclaim_hook()
```

相关位置：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
  uvm_va_block_make_resident()
  uvm_va_block_service_finish()
```

但当前 hook 参数是：

```c
va_space, address, length, gpu_id, cause, execute_allowed
```

它仍然需要通过 address 再回查 region 和 wrapper。

### 7.2 新 hook

新增 native block hook：

```c
void uvm_phase_n_va_block_make_resident_hook(uvm_va_space_t *va_space,
                                             uvm_va_block_t *va_block,
                                             uvm_va_block_region_t region,
                                             uvm_processor_id_t dest_id,
                                             NvU32 cause,
                                             NvBool execute_allowed);
```

调用位置替换或并行现有 hook：

```text
uvm_va_block_make_resident_finish() 后
且 dest_id 是 GPU 时
```

### 7.3 direct accounting 逻辑

核心逻辑：

```text
if !uvm_phase_n_block_quota_enable:
  return

if dest_id 不是 GPU:
  return

if !va_block->phase_n.tag_valid:
  return

if va_block->phase_n.mixed_pool:
  return

pool = va_block->phase_n.pool
block_pool = &va_space->phase_n_block_pools[pool]

if !va_block->phase_n.gpu_resident:
  va_block->phase_n.gpu_resident = true
  va_block->phase_n.resident_epoch = epoch
  block_pool->resident_blocks++

va_block->phase_n.last_access_epoch = epoch

if !va_block->phase_n.queued:
  list_add_tail(&va_block->phase_n.queue_node, &block_pool->resident_queue)
  va_block->phase_n.queued = true

update pool pressure
```

FIFO 语义：

```text
第一次 GPU resident 的时间决定队列顺序；
后续访问不把 block 移到队尾。
```

这符合当前科研目标：先实现简单、可解释、低开销的 FIFO reclaim。

## 8. P8f-d：RECLAIM_POOL 直接选择 native va_block victim

### 8.1 当前 P8b/P8c 路径

当前 `RECLAIM_POOL` 在 block quota enable 时走：

```text
uvm_phase_n_reclaim_pool_blocks()
  -> 从 phase_n_block_pools[pool].resident_queue 选择 phase_n_block_t
  -> 复制 victim snapshot
  -> 调 uvm_migrate_managed_range_to_cpu()
  -> finish reclaim
```

### 8.2 新路径

P8f-d 后，queue 节点直接是：

```text
uvm_va_block_t.phase_n.queue_node
```

victim selection：

```text
1. block_pool = &va_space->phase_n_block_pools[pool]
2. 从 resident_queue head 开始遍历
3. container_of(queue_node, uvm_va_block_t, phase_n.queue_node)
4. 检查：
   - tag_valid
   - pool 相同
   - gpu_resident
   - queued
   - !in_reclaim
   - !stale
   - !mixed_pool
   - flags 包含 EVICTABLE
   - flags 不包含 PROTECTED/ACTIVE/PINNED
5. uvm_va_block_retain(block)
6. list_del_init(&block->phase_n.queue_node)
7. block->phase_n.queued = false
8. block->phase_n.in_reclaim = true
9. 复制 migrate_base/migrate_length/pool/region_id 等 snapshot
10. 释放 va_space lock
11. 调迁移 helper
12. 重新加锁 finish
13. uvm_va_block_release(block)
```

必须使用原生 refcount：

```c
uvm_va_block_retain()
uvm_va_block_release()
```

不能只保存裸 `uvm_va_block_t *`。原因是选 victim 后会释放 lock 执行迁移，期间 block 生命周期必须被 pin 住。

### 8.3 迁移范围

当前阶段只区分 KV/WEIGHT，且 P8f-a 会尽量 split range 让语义边界对齐，所以迁移范围可以先用整个 native block：

```text
migrate_base = va_block->start
migrate_length = va_block->end - va_block->start + 1
```

如果后续需要兼容 subrange mixed case，再引入：

```text
covered_start / covered_end
```

但性能优先的主路径应尽量避免 subrange 计算。

## 9. P8f-e：生命周期维护

直接管理 native `va_range` / `va_block` 后，必须维护以下生命周期。

### 9.1 va_range split

位置：

```text
uvm_va_range_split()
uvm_va_range_split_blocks()
```

要求：

```text
new_managed_range->phase_n_tag 继承 existing_managed_range->phase_n_tag
```

如果 split 是为了 register_region 边界对齐：

```text
split 后对目标 range 写入新 tag
非目标 range 保留旧 tag 或清空 tag
```

### 9.2 va_block split

位置：

```text
uvm_va_block_split()
```

要求：

```text
new_block->phase_n 继承 existing_block->phase_n 的 tag 字段
queue/resident 字段根据 residency mask 重新计算
```

初版可以采用保守方案：

```text
split 时旧 block/new block 都从 Stage N FIFO 中移除并标 stale
下一次 make-resident 再重新入队
```

### 9.3 va_range destroy/free

位置：

```text
uvm_va_range_destroy_managed()
uvm_va_block_kill()
uvm_va_block_destroy()
```

要求：

```text
如果 block->phase_n.queued:
  list_del_init()
  queued=false

如果 block->phase_n.gpu_resident:
  resident_blocks--
  gpu_resident=false

block->phase_n.stale=true
```

### 9.4 unregister_region

当前 allocator-side KV tag 已经在 free 时 unregister。

P8f 后 unregister 不仅要移除：

```text
phase_f_region_ranges
phase_f_regions
```

还要清理 native tags：

```text
覆盖范围内 managed_range->phase_n_tag.valid=false
覆盖范围内 va_block->phase_n.tag_valid=false
已 queued 的 block 出队
resident counter 修正
```

### 9.5 CPU migration / native eviction

如果 block 被迁移回 CPU，必须把 native block 从 GPU resident queue 中移除。

初版可以在 Stage N 自己的 reclaim finish 中更新：

```text
gpu_resident=false
resident_blocks--
```

后续应补充原生 CPU migration/eviction hook，覆盖非 Stage N 触发的迁移。

## 10. 数据结构迁移策略

### 10.1 保留现有结构作为 fallback

P8f 初期不建议直接删除：

```text
phase_f_region_ranges
phase_n_blocks
phase_n_block_t
```

建议保留：

```text
phase_f_region_ranges:
  兼容 Python tag、debug 查询、mixed/subrange fallback

phase_n_block_t:
  debug fallback，验证 native accounting 与 wrapper accounting 差异
```

新增 module 参数：

```text
uvm_phase_n_native_block_enable=0/1
```

行为：

```text
0 -> 使用当前 P8e-a wrapper path
1 -> 使用 P8f native va_range/va_block fast path
```

### 10.2 pool state 复用

`phase_n_block_pools[]` 可以继续复用：

```text
budget_bytes
budget_blocks
resident_blocks
materialized_blocks
mixed_blocks
reclaim_attempts
reclaim_success
reclaim_failures
same_pool_victims
resident_queue
```

只是 `resident_queue` 的节点来源从：

```text
uvm_phase_n_block_t.queue_node
```

变为：

```text
uvm_va_block_t.phase_n.queue_node
```

如果需要兼容两种模式，建议不要共用同一个 queue node 类型。可以新增：

```text
phase_n_native_block_pools[]
```

或者在同一个 pool state 中明确由 module 参数保证队列只放一种节点。

性能优先主线建议：

```text
P8f native enable 后，resident_queue 只允许挂 native va_block。
```

## 11. userspace / vLLM 侧配合

### 11.1 KV

当前已经实现 allocator-side early KV tag：

```text
VLLM_UVM_DRIVER_KV_TAG_ENABLE=1
```

KV managed allocation 后直接注册：

```text
pool=KV
role=kv_block
flags=EVICTABLE
```

P8f 可以复用这个入口，不需要 Python 再逐 child pointer 注册 KV。

### 11.2 WEIGHT

当前阶段只需确保 weight allocation/register 最终能传：

```text
pool=WEIGHT
role=generic_weight
flags=EVICTABLE 或 PROTECTED
```

如果 weight 现阶段仍由 Python residency manager 注册，也可以保留这条路径。P8f 的 driver register 会把它 attach 到 native `va_range`。

### 11.3 Python 重叠注册

如果 allocator 已经注册整段 KV，则 Python pointer-map child tags 应跳过 driver register，只保留 metadata/trace。

策略：

```text
allocator-side KV tag enable:
  Python KV child pointer 不再 driver_register_region

weight:
  暂时允许 Python 或 allocator 注册，但同一 VA 不能重复注册不同 pool
```

## 12. 验收标准

P8f-a 验收：

```text
REGISTER_REGION(KV/WEIGHT)
  -> phase_n_native_range_tag_attach
  -> managed_range->phase_n_tag.valid=1
  -> pool 正确
```

P8f-b 验收：

```text
uvm_va_range_block_create()
  -> va_block->phase_n.tag_valid=1
  -> va_block->phase_n.pool=KV/WEIGHT
```

P8f-c 验收：

```text
KV/WEIGHT make-resident to GPU
  -> phase_n_native_block_resident
  -> resident_blocks++
  -> block 入对应 pool FIFO
```

P8f-d 验收：

```text
pool budget = 2 blocks
访问 3 个同 pool blocks
  -> resident_blocks=3
  -> over_blocks=1
  -> reclaim 选择 FIFO head
  -> same-pool victim
  -> migrate to CPU
  -> resident_blocks=2
```

P8f-e 验收：

```text
free/unregister 后:
  no stale queued block
  resident_blocks 不为负
  GET_POOL_STATE 不报告已释放 allocation
```

真实 vLLM 验收：

```text
KV:
  allocator_kv_driver_tag_success_records > 0
  phase_n_native_block_resident pool=KV > 0
  GET_POOL_STATE(KV).resident_blocks > 0

WEIGHT:
  phase_n_native_block_resident pool=WEIGHT > 0
  GET_POOL_STATE(WEIGHT).resident_blocks > 0

quota:
  over_budget 事件可见
  same-pool reclaim 成功事件可见
```

## 13. 推荐 trace

新增 trace 名称建议：

```text
phase_n_native_range_tag_attach
phase_n_native_range_tag_split
phase_n_native_range_tag_mixed
phase_n_native_block_init
phase_n_native_block_resident
phase_n_native_block_enqueue
phase_n_native_block_over_budget
phase_n_native_block_reclaim_select
phase_n_native_block_reclaim
phase_n_native_block_dequeue
phase_n_native_block_destroy
phase_n_native_block_unregister
```

每条 trace 至少包含：

```text
pool
region_id
allocation_id
range_start/range_end 或 block_start/block_end
resident_blocks
budget_blocks
queued
gpu_resident
status
```

## 14. 性能预期

当前 wrapper path 的 hot path 成本：

```text
range tree lookup phase_f_region_ranges
native range/block lookup
radix tree lookup phase_n_blocks
可能分配 phase_n_block_t
wrapper queue update
```

P8f native path 的 hot path 成本：

```text
读 va_block->managed_range->phase_n_tag
读写 va_block->phase_n
list_add_tail
counter++
```

因此预期改进：

```text
更少的 lookup
更少的动态分配
更少的 address-based rematerialize
KV 初始化大量 fault/make-resident 场景下 accounting 更完整
RECLAIM_POOL victim selection 更接近 O(number of queued native blocks)
```

注册时会增加开销：

```text
REGISTER_REGION 可能触发 va_range split
unregister/free 需要清理 native block state
```

但这些不是主要 hot path。对于 vLLM 长时间推理场景，用注册时开销换 fault/migration/reclaim path 简化是合理的。

## 15. 风险与约束

### 15.1 mixed allocation

如果一个 managed allocation 内部混合了 KV 和 WEIGHT，单个 `va_range->phase_n_tag.pool` 不够表达。

当前阶段处理策略：

```text
尽量通过 allocator 保证一个 allocation 只属于一个 pool
发现不同 pool 重叠则标 mixed 并退出 native fast path
```

### 15.2 split 复杂度

UVM 支持：

```text
va_range split
va_block split
range destroy
block kill/destroy
```

P8f 必须补生命周期维护，否则 FIFO 中可能残留 stale block。

### 15.3 多 GPU

当前先按单 GPU 实验实现。多 GPU 下 `gpu_resident` 需要升级为 mask 或 per-GPU state：

```text
resident_gpus
per_gpu_queued
per_gpu_pool_queue
```

### 15.4 与原生 eviction 的一致性

如果 block 因原生 UVM eviction 或非 Stage N migration 离开 GPU，Stage N 必须同步出队/修正 counter。

第一版可以先覆盖 Stage N reclaim 自己触发的 CPU migration，后续补全所有 residency loss hook。

## 16. 推荐实现顺序

第一步：

```text
增加 uvm_phase_n_range_tag_t
增加 managed_range->phase_n_tag
REGISTER_REGION attach tag 到 managed_range
只支持 KV/WEIGHT
不支持 mixed fast path
```

第二步：

```text
增加 uvm_phase_n_va_block_state_t
增加 va_block->phase_n
block_create 继承 range tag
加 trace 验证 block init
```

第三步：

```text
新增 uvm_phase_n_va_block_make_resident_hook()
在 make_resident finish 后直接入 native FIFO
GET_POOL_STATE 读取 native resident_blocks
```

第四步：

```text
RECLAIM_POOL native fast path
直接从 FIFO 取 uvm_va_block_t
retain/release block
迁移成功后更新 native state
```

第五步：

```text
补 split/destroy/unregister 清理
加 synthetic 测试
加真实 vLLM KV/WEIGHT 测试
```

第六步：

```text
对比 wrapper path 与 native path
确认 native path 稳定后，把 phase_n_block_t 降级为 debug fallback
```

## 17. 当前阶段结论

当前 P8e-a 已经证明：

```text
allocator-side early KV tag 能生效；
driver 能收到 KV tag；
wrapper block quota 能在部分路径工作；
但真实 vLLM KV accounting 仍可能漏掉已 resident native block。
```

因此下一阶段 P8f 的核心不是继续修补 fault address 采样，而是：

```text
把语义写进 native va_range；
把 resident/FIFO/reclaim 状态写进 native va_block；
让 make-resident 完成点直接维护 per-pool queue 和 counter。
```

在当前科研目标下，这是更合适的性能优先路线。
