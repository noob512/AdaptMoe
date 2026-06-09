# UVM Remote-First Background Promotion 设计与实施计划

本文档总结当前 UVM 驱动中的相关事实、研究动机、目标机制和分阶段实施方案。目标是实现一种面向只读数据的 latency-aware promotion 机制：

```text
急需页先远程访问，尽快 replay；
当前页和预测页后台搬运到 GPU；
后台 copy 完成后，短暂切换 GPU PTE；
后续访问无缝走 GPU 本地副本。
```

第一版只面向只读数据，不处理写/混合访问。只读场景允许 CPU 副本和 GPU 副本共存，不需要 dirty tracking、写保护或版本校验，风险最低，也最适合先验证研究假设。

## 1. 背景与问题

当前 UVM fault handling 的吞吐导向策略会把当前 fault 页和预测预取页放在同一轮服务中处理。这样有利于顺序访问和长期吞吐，但对延迟敏感场景可能不利。

典型问题是：

```text
一个急需页触发 GPU fault
UVM 计算 prefetch hint
prefetch pages 被合并到 new_residency_mask
同步进入 make_resident_copy
当前急需页必须等待 unmap/populate/copy/map/replay 整套流程
额外 prefetch 页也会消耗 copy engine 和 PCIe/NVLink 带宽
```

在 LLM 推理中，这类延迟放大尤其值得关注：

- 首 token 或关键请求路径对 tail latency 敏感。
- 部分只读权重、写满后的 KV cache block、metadata/索引类数据，访问模式可能是读多写少或写后只读。
- 如果 speculative migration 太激进，预测页搬运可能阻塞真正急需页。

因此，本方案把 demand servicing 与 speculative promotion 解耦：

```text
demand path: 只服务当前急需访问，优先低延迟
promotion path: 后台搬运预测页和当前热点页，优化后续访问
```

## 2. 当前代码事实

### 2.1 fault 起点是页粒度

UVM replayable fault handling 会按 `PAGE_SIZE` 对齐 fault address，并计算 `page_index`。在当前 Ubuntu 环境中 `PAGE_SIZE=4096`，所以一次 fault 的基本触发粒度是 4KB 页。

相关代码：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`
- `uvm_gpu_replayable_faults.c:612`
- `uvm_gpu_replayable_faults.c:1575`
- `uvm_gpu_replayable_faults.c:1693`

这支持如下事实：

```text
当前被 GPU 访问且需要服务的页会以 page_index 进入 service mask。
```

### 2.2 prefetch 页会被合并进同步迁移

prefetch 逻辑默认启用：

- `uvm_perf_prefetch_enable = 1`
- `UVM_PREFETCH_THRESHOLD_DEFAULT = 51`
- `UVM_PREFETCH_MIN_FAULTS_DEFAULT = 1`

相关代码：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_perf_prefetch.c:39`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_perf_prefetch.c:42`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_perf_prefetch.c:283`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_perf_prefetch.c:488`

在 VA block service 中，`uvm_va_block_get_prefetch_hint()` 会把 `prefetch_pages_mask` 标记为 `UVM_FAULT_ACCESS_TYPE_PREFETCH`，然后合并到 `new_residency_mask`：

```c
uvm_page_mask_or(new_residency_mask, new_residency_mask, prefetch_pages_mask);
```

相关代码：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11524`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11557`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11569`

这意味着：

```text
当前 demand 页和 prefetch 页会进入同一个 residency/copy 服务路径。
```

### 2.3 合并后的页会同步进入 make-resident/copy

`uvm_va_block_service_locked()` 先计算 prefetch hint，然后调用 `uvm_va_block_service_copy()`：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:12102`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:12119`

`uvm_va_block_service_copy()` 会调用：

- `uvm_va_block_make_resident_read_duplicate()`
- 或 `uvm_va_block_make_resident_copy()`

普通迁移路径中，`uvm_va_block_make_resident_copy()` 会：

```text
unmap all mapped processors
populate destination pages/chunks
copy resident pages
finish residency and mapping update
```

相关代码：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:4779`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:4820`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:4824`

这正是当前 critical path 过重的根源。

### 2.4 “固定补到 16KB”目前不能直接作为代码事实

当前代码能证明的是：

```text
UVM 可能把单页 fault 放大为多页迁移。
```

但目前没有看到固定规则：

```text
4KB fault -> 一定补到 16KB DMA/prefetch
```

相关事实：

- CPU chunk 支持 2MB、64KB、4KB 混合管理。
- 一个 64KB slot 包含 `64KB / PAGE_SIZE` 个小 chunk；4KB 页下是 16 个页。
- prefetch 是 heuristic，可能是 0 页、若干页、更大 region，甚至部分路径可能预取整个 region。

因此论文或报告中建议使用更准确表述：

```text
UVM may amplify a single 4KB demand fault into a larger synchronous migration due to prefetch heuristics and chunk/page-region handling.
```

不要直接写“固定 16KB”，除非后续通过 instrumentation 统计每次 fault 的实际 copy bytes 后确认。

## 3. 目标机制

目标机制称为 Remote-First Background Promotion。

核心路径：

```text
GPU replayable READ fault
  -> 当前 demand 页仍 resident on CPU/system memory
  -> 先为 GPU 建立 CPU/system memory remote READ mapping
  -> fault replay，kernel 尽快继续
  -> demand 页 + prefetch 页进入后台 promotion pending mask
  -> 后台 worker 分配 GPU chunk/page
  -> 后台 CPU->GPU copy
  -> copy 完成后，短暂 unmap GPU remote PTE
  -> map GPU local READ PTE
  -> 后续访问走 GPU 本地副本
```

### 3.1 只读版本的约束

第一版只处理满足以下条件的访问：

```text
operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS
faulting processor 是 GPU
new_residency 是同一个 GPU
所有 demand 页访问类型 <= READ
所有 demand 页当前 CPU resident
非 HMM
ReadMostly 或 read_duplication 未禁用
CPU/system memory 可被该 GPU 访问
单 GPU 环境优先
```

不满足条件时直接回退原始 UVM 同步迁移路径。

### 3.2 非目标

第一版不处理：

- GPU 写 fault。
- CPU 写回。
- 多 GPU 读写一致性。
- dirty tracking。
- HMM read duplication。
- 真正低优先级 copy engine/channel 调度。
- access-counter 128 次提前 promotion。

这些是第二阶段或第三阶段工作。

## 4. 实施总览

实施可以分为十个阶段：

```text
1. 增加实验开关和统计
2. 给 VA block 增加后台 promotion 状态
3. 在 service path 中拆分 demand mask 和 prefetch mask
4. demand 页先走远程 READ mapping
5. demand + prefetch 页进入后台队列
6. 新增只读后台 copy helper
7. copy 完成后短暂切换 GPU PTE
8. 正确处理后台 worker 的锁和生命周期
9. 增加后台 copy 限流，避免抢占紧急带宽
10. 增加验证脚本和统计输出
```

## 5. 具体改动计划

### 5.1 增加 module params

建议先在 `uvm_va_block.c` 顶部参数区附近增加：

```c
static int uvm_latency_remote_first_enable = 0;
module_param(uvm_latency_remote_first_enable, int, S_IRUGO);

static int uvm_latency_bg_promote_enable = 0;
module_param(uvm_latency_bg_promote_enable, int, S_IRUGO);

static unsigned uvm_latency_bg_promote_max_pages = 512;
module_param(uvm_latency_bg_promote_max_pages, uint, S_IRUGO);
```

语义：

```text
uvm_latency_remote_first_enable=1
    允许 READ fault 使用远程优先服务。

uvm_latency_bg_promote_enable=1
    允许把 demand + prefetch 页放入后台 promotion。

uvm_latency_bg_promote_max_pages=N
    每次后台 worker 最多 copy N 个页，避免过度占用 PCIe/copy engine。
```

默认值必须保持 0，确保默认 UVM 行为不变。

### 5.2 增加统计

建议至少统计：

```c
atomic64_t remote_first_pages;
atomic64_t remote_first_faults;
atomic64_t bg_promote_enqueued_pages;
atomic64_t bg_promote_copied_pages;
atomic64_t bg_promote_committed_pages;
atomic64_t bg_promote_cancelled_pages;
atomic64_t bg_promote_fallback_sync_pages;
atomic64_t bg_promote_throttled_rounds;
```

第一版可以先用 `UVM_INFO_PRINT` 或 debug procfs 暴露。后续再接入正式 telemetry。

### 5.3 扩展 `uvm_va_block_gpu_state_t`

位置：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h
```

在 `uvm_va_block_gpu_state_t` 内增加：

```c
// Pages waiting for read-only background promotion on this GPU.
uvm_page_mask_t bg_promote_pending;

// Pages currently being promoted by the background worker.
uvm_page_mask_t bg_promote_inflight;
```

含义：

```text
pending:
    已经被 demand/prefetch 选中，但 worker 还没开始处理。

inflight:
    worker 已经取走，正在 populate/copy/switch。
```

### 5.4 扩展 `uvm_va_block_t`

在 `uvm_va_block_t` 中增加：

```c
// Deferred work item for read-only background promotion.
nv_kthread_q_item_t bg_promote_q_item;

// GPUs that have pending background promotion pages in this block.
uvm_processor_mask_t bg_promote_gpus;
```

在 block 初始化处仿照 eviction mapping 初始化：

```c
nv_kthread_q_item_init(&block->bg_promote_q_item,
                       block_bg_promote_entry,
                       block);
```

现有参照：

```text
uvm_va_block.c:1304
nv_kthread_q_item_init(&block->eviction_mappings_q_item, block_add_eviction_mappings_entry, block);
```

后台队列第一版可复用：

```text
g_uvm_global.global_q
```

该队列已用于 deferred work：

```text
uvm_global.h:104
uvm_global.c:101
```

## 6. 拆分 demand 和 prefetch

### 6.1 当前同步合并点

当前 `uvm_va_block_service_locked()` 中：

```text
uvm_va_block_get_prefetch_hint()
  -> prefetch_pages_mask OR 到 new_residency_mask
uvm_va_block_service_copy()
  -> 同步迁移 new_residency_mask
```

需要在这个位置插入 remote-first 判断。

位置：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:12077
```

### 6.2 保存 demand mask

在调用 `uvm_va_block_get_prefetch_hint()` 前保存原始 `new_residency_mask`：

```c
new_residency = uvm_processor_mask_find_first_id(&service_context->resident_processors);
new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;

uvm_page_mask_copy(&service_context->block_context->scratch_page_mask,
                   new_residency_mask);

demand_mask = &service_context->block_context->scratch_page_mask;
```

然后调用原始 prefetch hint：

```c
uvm_va_block_get_prefetch_hint(va_block, policy, service_context);
prefetch_mask = &service_context->prefetch_hint.prefetch_pages_mask;
```

### 6.3 新增 helper 判断是否接管

伪代码：

```c
status = try_remote_first_bg_promote(va_block,
                                     processor_id,
                                     new_residency,
                                     demand_mask,
                                     prefetch_mask,
                                     service_context,
                                     &handled);
if (status != NV_OK)
    return status;

if (handled)
    return NV_OK;
```

`handled=true` 表示：

```text
demand 页已经被 remote mapped；
promotion 页已经入队；
无需进入同步 make_resident_copy；
fault service 可以继续返回并 replay。
```

## 7. demand 页远程 READ mapping

远程 mapping 使用已有函数：

```text
uvm_va_block_map()
```

位置：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:8385
```

对 demand 页执行：

```c
status = uvm_va_block_map(va_block,
                          service_context->block_context,
                          processor_id,
                          service_context->region,
                          demand_mask,
                          UVM_PROT_READ_ONLY,
                          UvmEventMapRemoteCausePolicy,
                          &va_block->tracker);
```

这个调用会根据当前 resident location 为 GPU 建映射。如果页仍 resident on CPU/system memory，则 GPU 会得到 remote READ mapping。

### 7.1 接管条件检查

`try_remote_first_bg_promote()` 必须检查：

```c
if (!uvm_latency_remote_first_enable)
    return handled=false;

if (service_context->operation != UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS)
    return handled=false;

if (!UVM_ID_IS_GPU(processor_id))
    return handled=false;

if (!uvm_id_equal(processor_id, new_residency))
    return handled=false;

if (uvm_va_block_is_hmm(va_block))
    return handled=false;

if (!all_pages_are_read_faults(demand_mask, service_context))
    return handled=false;

if (!all_pages_cpu_resident(va_block, demand_mask))
    return handled=false;

if (!gpu_can_access_cpu_residency(va_space, processor_id))
    return handled=false;
```

第一版宁可保守，不要扩大适用范围。

## 8. 后台 promotion 入队

### 8.1 入队页集合

后台 promotion 的页集合为：

```text
bg_mask = demand_mask | prefetch_mask
```

这样当前正在远程读的页和预测未来需要的页都会后台搬运。

需要限制页数：

```text
bg_mask = first N pages of bg_mask
N = uvm_latency_bg_promote_max_pages
```

### 8.2 enqueue helper

伪代码：

```c
static NV_STATUS block_bg_promote_enqueue(uvm_va_block_t *block,
                                          uvm_gpu_id_t gpu_id,
                                          uvm_va_block_context_t *block_context,
                                          const uvm_page_mask_t *pages)
{
    uvm_gpu_t *gpu = uvm_gpu_get(gpu_id);
    uvm_va_block_gpu_state_t *gpu_state;

    gpu_state = uvm_va_block_gpu_state_get_alloc(block, gpu);
    if (!gpu_state)
        return NV_ERR_NO_MEMORY;

    uvm_page_mask_or(&gpu_state->bg_promote_pending,
                     &gpu_state->bg_promote_pending,
                     pages);

    uvm_processor_mask_set(&block->bg_promote_gpus, gpu_id);

    uvm_va_block_retain(block);
    if (!nv_kthread_q_schedule_q_item(&g_uvm_global.global_q,
                                      &block->bg_promote_q_item)) {
        uvm_va_block_release_no_destroy(block);
    }

    return NV_OK;
}
```

注意：

- 需要持有 `va_block->lock`。
- `nv_kthread_q_schedule_q_item()` 如果发现同一个 q_item 已经 pending，会返回 false。
- pending mask 已经合并了新页，所以返回 false 不代表丢任务。

参照：

```text
uvm_va_block.c:13097
uvm_va_block.c:13102
```

## 9. 后台 worker 设计

### 9.1 worker 生命周期

参考 `block_add_eviction_mappings()`：

```text
uvm_va_block.c:12850
```

后台 worker 结构：

```c
static void block_bg_promote_entry(void *args)
{
    UVM_ENTRY_VOID(block_bg_promote(args));
}
```

`block_bg_promote()` 流程：

```text
1. lock block，获取 va_space，检查 block 是否 dead
2. retain mm
3. 分配 uvm_va_block_context_t
4. va_space_down_read()
5. 再次检查 block 是否 dead
6. 使用 UVM_VA_BLOCK_LOCK_RETRY 调用 block_bg_promote_locked()
7. va_space_up_read()
8. 释放 block_context
9. release mm
10. uvm_va_block_release(block)
```

必须使用 `UVM_VA_BLOCK_LOCK_RETRY`，因为后台 populate/copy 可能需要 GPU memory allocation retry：

```text
uvm_va_block.h:2264
```

### 9.2 locked worker 逻辑

伪代码：

```c
static NV_STATUS block_bg_promote_locked(uvm_va_block_t *block,
                                         uvm_va_block_retry_t *retry,
                                         uvm_va_block_context_t *block_context)
{
    for_each_gpu_id_in_mask(gpu_id, &block->bg_promote_gpus) {
        gpu_state = uvm_va_block_gpu_state_get(block, gpu_id);
        if (!gpu_state)
            continue;

        promote_mask = take_limited_pending_pages(gpu_state,
                                                  uvm_latency_bg_promote_max_pages);
        if (uvm_page_mask_empty(promote_mask))
            continue;

        uvm_page_mask_or(&gpu_state->bg_promote_inflight,
                         &gpu_state->bg_promote_inflight,
                         promote_mask);

        status = block_bg_promote_readonly_locked(block,
                                                  retry,
                                                  block_context,
                                                  gpu_id,
                                                  promote_mask);

        uvm_page_mask_andnot(&gpu_state->bg_promote_inflight,
                             &gpu_state->bg_promote_inflight,
                             promote_mask);

        if (status != NV_OK)
            restore_or_cancel_pending_pages(...);
    }

    if (still_has_pending_pages(block))
        reschedule block->bg_promote_q_item;

    return NV_OK;
}
```

## 10. 后台只读 copy helper

不能直接调用：

```text
uvm_va_block_make_resident_read_duplicate()
```

原因是它会在 copy 前调用 `block_prep_read_duplicate_mapping()`，撤销远程映射并 revoke write：

```text
uvm_va_block.c:5131
uvm_va_block.c:5155
```

这与目标相反。目标是：

```text
copy 完成前保留 remote READ mapping；
copy 完成后再短暂 switch PTE。
```

### 10.1 新 helper 流程

新增：

```c
static NV_STATUS block_bg_promote_readonly_locked(uvm_va_block_t *block,
                                                  uvm_va_block_retry_t *retry,
                                                  uvm_va_block_context_t *block_context,
                                                  uvm_processor_id_t gpu_id,
                                                  uvm_va_block_region_t region,
                                                  const uvm_page_mask_t *promote_mask)
```

流程：

```text
1. 过滤 promote_mask
   - 必须 CPU resident
   - 不能已经 GPU resident
   - 不能是 discarded 页
   - 不能需要 WRITE/ATOMIC

2. 设置 make_resident 上下文
   - dest_id = gpu_id
   - cause = 新增 cause 或临时使用 UVM_MAKE_RESIDENT_CAUSE_PREFETCH/ACCESS_COUNTER

3. block_populate_pages()
   - 分配 GPU page/chunk

4. block_copy_resident_pages()
   - CPU -> GPU
   - transfer mode 使用 COPY，不使用 MOVE

5. 更新 residency
   - GPU resident mask OR migrated_pages
   - CPU resident 保留
   - block_set_resident_processor(block, gpu_id)
   - read_duplicated_pages OR migrated_pages

6. PTE switch
   - 先 unmap GPU remote PTE
   - 再 map GPU local READ PTE

7. 更新统计
```

可复用函数参考：

```text
block_populate_pages()
    uvm_va_block.c:5160

block_copy_resident_pages()
    uvm_va_block.c:5164

read-dup 状态更新
    uvm_va_block.c:5209
    uvm_va_block.c:5214
```

### 10.2 PTE switch

由于 GPU 已经有 remote READ PTE，直接调用 `uvm_va_block_map()` 可能被 pte mask 判断跳过。因此 copy 完成后要先 unmap GPU 当前 mapping，再重新 map。

伪代码：

```c
status = uvm_va_block_unmap(va_block,
                            block_context,
                            gpu_id,
                            region,
                            promote_mask,
                            &local_tracker);
if (status != NV_OK)
    return status;

status = uvm_va_block_map(va_block,
                          block_context,
                          gpu_id,
                          region,
                          promote_mask,
                          UVM_PROT_READ_ONLY,
                          UvmEventMapRemoteCausePolicy,
                          &local_tracker);
```

这个阶段只做页表更新和 TLB invalidation，不应包含大规模 copy。

## 11. 正确性约束

### 11.1 只读一致性

第一版只支持 read-only/read-mostly 页面。必须保证：

```text
CPU 副本仍然是 coherent source
GPU 副本只用于 READ
没有 CPU/GPU WRITE race
没有 atomic mapping
read_duplicated_pages 正确标记
```

若发生写 fault，必须回退现有 UVM 逻辑并打破 read duplication。

### 11.2 block 死亡和释放

后台 worker 必须 retain VA block：

```c
uvm_va_block_retain(block);
```

worker 结束时 release：

```c
uvm_va_block_release(block);
```

如果 schedule 失败，需要：

```c
uvm_va_block_release_no_destroy(block);
```

参照 eviction deferred mapping：

```text
uvm_va_block.c:13098
uvm_va_block.c:13102
uvm_va_block.c:13105
```

### 11.3 锁顺序

后台 worker 不应在没有 VA space read lock 的情况下修改 mappings。推荐顺序：

```text
retain mm
allocate context
va_space_down_read()
UVM_VA_BLOCK_LOCK_RETRY()
va_space_up_read()
```

不要在 fault critical path 中长时间持有锁等待后台 copy。

### 11.4 allocation retry

涉及 GPU chunk allocation 时必须支持 retry。直接 `uvm_mutex_lock(&block->lock)` 后调用 populate/copy 不够安全。应使用：

```text
UVM_VA_BLOCK_LOCK_RETRY()
```

## 12. 限流策略

仅仅“后台 copy”不等于低延迟。copy engine 和 PCIe/NVLink 仍然共享，过度后台搬运仍可能干扰 urgent remote read。

第一版必须加限流：

```text
每个 worker round 最多处理 uvm_latency_bg_promote_max_pages
默认 512 页，即 2 MiB
如果仍有 pending 页，重新 schedule
```

后续可以加入自适应策略：

```text
如果 replayable fault backlog 高，则降低 max_pages
如果 remote-first fault 少且 future read 热，则提高 max_pages
如果检测到 over-prefetch 命中率低，则缩小 prefetch mask
```

## 13. 与 access-counter 提前 promotion 的关系

本文档的第一版不接入 access counter。原因是：

```text
replayable READ fault 场景更容易验证 remote-first 是否有效；
access-counter 路径本身是 hotness signal，适合作为第二阶段增强；
先证明机制正确，再证明策略更优。
```

第二阶段可以把硬件 access-counter threshold 从 256 改为 128 或引入软件半阈值：

```text
128 次左右：后台开始 promotion
256 次左右：如果 promotion 已完成，直接切本地
如果没完成，继续 remote read 或回退同步迁移
```

现有 access-counter 服务路径：

```text
uvm_gpu_access_counters.c:1125
uvm_gpu_access_counters.c:1218
uvm_gpu_access_counters.c:1287
```

需要注意，access-counter 现有路径会把通知页视为 `UVM_FAULT_ACCESS_TYPE_PREFETCH`，并走 `uvm_va_block_service_locked()`，仍然可能同步迁移。

## 14. 验证计划

### 14.1 驱动级统计验证

需要确认：

```text
remote-first 是否真的没有进入同步 make_resident_copy
background promotion 是否真的 copy 了 demand + prefetch 页
copy 前 GPU 是否仍保留 remote READ mapping
copy 后 GPU PTE 是否切到 local GPU resident page
```

建议统计：

```text
fault_demand_pages
fault_prefetch_pages
sync_migrated_pages
remote_first_pages
bg_promote_enqueued_pages
bg_promote_copied_pages
bg_promote_committed_pages
pte_switch_pages
fallback_sync_pages
```

### 14.2 性能实验

最小对照组：

```text
baseline_default_uvm
    原始 UVM，同步迁移。

remote_first_only
    demand 页远程读，不后台搬。

remote_first_bg_exact
    demand 页远程读，demand + exact prefetch 后台搬。

remote_first_bg_over
    demand 页远程读，故意扩大 prefetch，测试带宽污染。
```

指标：

```text
request_ready_ms
future_ms
total_ms
first_pass_ms
last_pass_ms
remote_to_local_transition_pass
copy_bytes
sync_migration_bytes
background_migration_bytes
```

### 14.3 预期结果

理想结果：

```text
request_ready_ms 接近 remote_only
future_ms 接近 local_prefetch
background exact 明显优于 first_touch_migrate
background over 变差，但受限流保护，恶化小于用户态 cudaMemPrefetchAsync over 版本
```

如果结果是：

```text
request_ready_ms 仍然明显高于 remote_only
```

则说明后台 copy 仍在争用关键资源，需要进一步降低 `uvm_latency_bg_promote_max_pages` 或引入真正低优先级 copy channel。

## 15. 风险与陷阱

### 15.1 远程访问也会占 PCIe

remote read 不是免费操作。它减少了 fault critical path 中的 populate/copy/unmap 等同步开销，但数据仍然经过 PCIe/NVLink。后台 copy 过猛时仍会抢带宽。

### 15.2 已有 read-dup 路径不是目标机制

`uvm_va_block_make_resident_read_duplicate()` 会在 copy 前撤销远程映射。它适合创建 read duplicate，但不满足：

```text
copy 期间继续 remote read
copy 完成后再切 local
```

因此需要新增 helper，而不是简单复用 read-dup 函数。

### 15.3 PTE switch 需要小心

如果 GPU 已经有 remote READ PTE，直接 map local 可能被已有 pte bit 跳过。必须先 unmap 再 map。

### 15.4 只读判定必须保守

第一版只用 ReadMostly/read-duplication policy 或明确只读实验范围。不要自动假设 KV cache 永远只读。KV cache 在 prefill/decode 写入阶段仍然由 GPU 写，只有写满后的 block 才适合只读 promotion。

### 15.5 HMM 不纳入第一版

HMM 下 CPU PTE 可由 Linux 升级，read duplication 尚有 TODO。第一版排除 HMM。

## 16. 分阶段里程碑

### Milestone 0: instrumentation only

不改行为，只统计：

```text
demand mask 页数
prefetch mask 页数
实际同步 copy 页数
每次 copy bytes
```

目的：确认单页 fault 放大为多页同步迁移的程度。

### Milestone 1: remote-first only

实现：

```text
READ demand 页远程 READ mapping
不后台 copy
```

目标：

```text
首延迟接近 remote_only
确认 fault critical path 下降
```

### Milestone 2: background exact promotion

实现：

```text
demand + prefetch exact 后台 copy
copy 完成后 PTE switch
```

目标：

```text
首延迟接近 remote_first_only
后续访问接近 local_prefetch
```

### Milestone 3: over-prefetch 限流

实现：

```text
扩大 prefetch mask
限制每轮后台 copy 页数
观察 over-prefetch 对首延迟的污染程度
```

目标：

```text
证明限流能降低 speculative migration 对 urgent path 的干扰
```

### Milestone 4: access-counter 提前 promotion

实现：

```text
低于 256 阈值时提前后台 promotion
达到正式阈值时如果已完成则本地访问
未完成则继续 remote 或回退同步迁移
```

目标：

```text
证明热页可在真正需要本地化前提前准备
减少 256 次后同步迁移开销
```

## 17. 最小实现建议

第一版应该严格限定为：

```text
GPU replayable READ fault
CPU resident
ReadMostly/只读
非 HMM
单 GPU
demand remote map
demand + prefetch background copy
copy done 后 GPU PTE switch
```

这个版本最小、可验证、风险最低。证明这个机制有效后，再扩展到 access-counter 提前 promotion 和 KV cache 写满后只读 block。

## 18. 最终判断

这个方向有研究价值，但必须准确表述创新点：

```text
不是“UVM 固定补到 16KB”；
而是“UVM 会把 demand fault 与 speculative prefetch 合并进同步迁移路径，
导致额外 copy 和页表处理污染 critical path”。
```

真正的创新点是：

```text
latency-aware split servicing:
urgent demand pages use remote READ first;
speculative/current hot pages are promoted in background;
local PTE switch occurs only after copy completion.
```

该机制如果实现正确，预期能同时获得：

```text
接近 remote-only 的首延迟；
接近 GPU local 的后续读取吞吐；
比用户态 cudaMemPrefetchAsync 更低的提交和同步干扰；
比默认 UVM prefetch 更可控的带宽占用。
```
