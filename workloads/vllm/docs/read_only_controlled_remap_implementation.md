# Read-only controlled remap 当前实现说明

日期：2026-06-29

本文档总结当前 MoE 专家权重只读 controlled remap 的实现状态、用户态到内核态调用链，以及专家权重如何被分配、搬运、映射和清理。

## 目标边界

当前目标是只读专家权重路径：

- 支持 `copy_only -> gpu_copy_only -> slot_va_block_remap` 的渐进验证。
- slot backing 使用驱动侧 GPU VIDMEM。
- 专家权重内容先搬入 slot VIDMEM，再把原 managed expert VA 的 GPU PTE 临时改成只读 slot-backed PTE。
- vLLM Stage M 初始化放置后默认 reset/clear slot PTE，避免长期把实验 PTE 留在生产执行路径里。

当前明确不做：

- 不支持可写 PTE。
- 不做 dirty tracking。
- 不做 slot backing 到原始权重的 copyback。
- 不把 slot-backed PTE 当成完整生产级 UVM residency。
- 不改变 KV cache 或其他可写数据路径语义。

## 内核 ABI

主要接口在 `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`。

`UVM_MOE_SLOT_REMAP_BATCH` 是 ioctl 91，用于提交一批 slot remap/copy 记录。每条 `UVM_MOE_SLOT_REMAP_ENTRY` 包含：

- `layer_id`：MoE layer id。
- `expert_id`：专家 id。
- `slot_id`：该专家在当前 layer 的 slot id。
- `role`：专家权重角色，例如 gate/up 或 down。
- `va_base` / `bytes`：源或目标用户 VA 范围。
- `slot_offset`：该 range 在 slot backing 里的字节偏移。
- `status`：每条 entry 的返回状态。

当前 mode：

| mode | 名称 | 作用 |
|---:|---|---|
| 0 | `dry_run` | 参数检查，不修改 slot pool |
| 1 | `validate` | 只记录 slot 元数据 |
| 2 | `remap` | 原生 migrate/map 语义探测 |
| 3 | `copy_only` | sysmem backing + CPU copy |
| 4 | `gpu_copy_only` | VIDMEM backing + GPU/CE copy |
| 5 | `slot_pte_remap` | 早期单 range 直接 PTE smoke |
| 6 | `slot_va_block_remap` | 当前受控 VA-block-aware 只读真实 PTE remap |

`UVM_MOE_SLOT_POOL_QUERY` 是 ioctl 92，用于查询和 reset slot pool。它暴露：

- backing：`backing_kind`、`slot_backing_bytes`、`backed_slots`。
- copy：`copy_count`、`copy_bytes`、`copy_failed_count`、`vidmem_evict_count`。
- PTE：`pte_remap_count`、`pte_remap_bytes`、`pte_clear_count`、`pte_failure_count`。
- VA block：`pte_block_unmap_count`、`pte_block_tracker_wait_count`。
- rollback：`pte_rollback_count`、`pte_rollback_bytes`。
- lifecycle guard：fault/migrate/unmap/split/release/evict/dangerous-op counters。
- residual scan：`pte_residual_va_block_count`、`pte_residual_page_count`、`pte_residual_gpu_mask`。

reset 后必须满足：

```text
active_slots == 0
backed_slots == 0
slot_backing_bytes == 0
pte_residual_va_block_count == 0
pte_residual_page_count == 0
```

否则认为 slot-backed PTE 或 VA block metadata 没清干净。

## vLLM 用户态调用链

Stage M 初始化放置从 vLLM worker 初始化阶段进入：

```text
API server / EngineCore process
  -> EngineCore.__init__()
  -> EngineCore._initialize_kv_caches()
  -> model_executor.initialize_from_config(kv_cache_configs)
  -> Executor.collective_rpc("initialize_from_config")
  -> GPUWorker.initialize_from_config()
  -> GPUWorker.model_runner.initialize_kv_cache()
  -> GPUModelRunner.run_uvm_init_time_weight_placement()
  -> for module in model.modules():
       module.run_uvm_init_weight_offload()
  -> FusedMoE._uvm_cold_slot_pool_driver_remap_backend()
  -> driver_moe_slot_remap_batch()
  -> ioctl(UVM_MOE_SLOT_REMAP_BATCH)
```

相关文件：

- `workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py`
- `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`
- `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`
- `workloads/vllm/vllm/vllm/device_allocator/uvm.py`

`GPUModelRunner.run_uvm_init_time_weight_placement()` 做三件事：

1. 检查 `VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE` 或 `VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE`。
2. 遍历模型模块，调用每个 FusedMoE module 的 `run_uvm_init_weight_offload()`。
3. 如果 driver mode 是 `slot_pte_remap` 或 `slot_va_block_remap`，执行 `_maybe_clear_uvm_init_time_slot_ptes_after_placement()`，通过 query reset 清理 slot PTE 和 backing。

## 专家权重发现

每个 FusedMoE layer 内部维护专家权重切片。当前只关心只读专家权重，典型角色是：

- `moe_gate_up`：gate/up fused weight，通常来自 `w13_weight`。
- `moe_down`：down projection weight，通常来自 `w2_weight`。

`run_uvm_init_weight_offload()` 会读取热/冷计划，得到当前 layer 需要处理的专家集合。对于每个专家：

```text
global_expert_id
  -> local_expert_id
  -> _uvm_expert_weight_slices(local_expert_id, target_roles)
       -> (role, ptr, size_bytes)
```

每个切片同时注册到用户态 residency metadata：

```text
register_uvm_residency_region(
  pool="weights",
  region_id=<layer/expert/role>,
  ptr=<expert weight VA>,
  bytes=<expert weight bytes>,
  metadata={
    source="stage_m",
    pool_class="cold",
    init_cold=True,
    evictable=True,
    prefetchable=True,
    plan_policy=<routing plan policy>,
  },
)
```

这部分 metadata 主要用于观测和计划，不等于内核真实 PTE 状态。

## 专家 slot 分配

用户态 slot pool 按 layer 维护：

```text
_UVM_COLD_SLOT_POOL_LAYERS[layer_id] = {
  capacity,
  free_cursor,
  slots: [
    {
      slot_id,
      owner_layer,
      owner_expert,
      state,
      epoch,
      last_access_step,
      ready_event_step,
      release_event_step,
    },
    ...
  ],
  expert_to_slot,
}
```

分配流程：

```text
active_cold_expert_ids
  -> 去重
  -> 已有 READY/IN_USE 且 owner_expert 匹配：slot hit，标记 IN_USE
  -> 没有可用 mapping：slot miss
  -> 从 free_cursor 开始找 FREE/READY slot
  -> 复用 slot 时删除 old_owner_expert -> slot 的旧映射
  -> 更新 owner_expert/state/epoch
```

关键约束：

- 同一 layer 内每个 active expert 最多绑定一个 slot。
- 当前 batch 中 protected slot 不会被复用。
- 若 capacity 不够，记录 overflow，不提交无法容纳的专家。
- 同一 expert 的多个权重 range 共享同一个 `slot_id`，通过 `slot_offset` 区分。

## entry 构造和 slot layout

对于每个 miss expert，`_uvm_cold_slot_pool_driver_remap_backend()` 生成一组 ioctl entries。

核心逻辑：

```text
slot_offset = 0
for role, ptr, size_bytes in _uvm_expert_weight_slices(...):
    rounded_base = page_align_down(ptr)
    rounded_end = page_align_up(ptr + size_bytes)
    rounded_bytes = rounded_end - rounded_base

    entry = {
      layer_id,
      expert_id,
      slot_id,
      role,
      va_base=ptr,
      bytes=size_bytes,
      slot_offset=slot_offset,
    }

    slot_offset += rounded_bytes
```

也就是说，一个 expert slot 的物理 backing 形如：

```text
slot VIDMEM backing
  offset 0                         : moe_gate_up page-rounded range
  offset gate_up_rounded_bytes     : moe_down page-rounded range
  offset ...                       : 后续 role range
```

之前发现过一个重要 bug：内核 slot 只记录一个 PTE range，会导致同一 expert 的第二个 range 覆盖第一个 range 的 metadata。现在 slot 支持记录多个 PTE range，reset/rollback/release 会逐一清理。

## 为什么需要 host mmap staging

`slot_va_block_remap` 需要先把专家权重内容搬到驱动 slot VIDMEM。最初如果直接在内核 `gpu_copy_only` 中对 managed expert VA 执行 `copy_from_user`，会触发 UVM CPU fault，并可能卡在 `uvm_va_space_cpu_fault`。

因此当前 vLLM 路径使用 host mmap staging：

```text
expert managed VA
  -> 用户态 ctypes.memmove()
  -> anonymous mmap host_buffer
  -> host_addr page aligned
  -> prefill entry va_base=host_addr
```

然后先提交一次 `gpu_copy_only`：

```text
driver_moe_slot_remap_batch(
  prefill_entries,
  mode="gpu_copy_only",
  validate_managed=False,
  round_to_pages=False,
  fd_target_entries=original_expert_entries,
  fd_target_round_to_pages=True,
)
```

这里有两个 VA 概念：

- copy 源 VA：host mmap staging，不是 managed expert VA，避免 UVM CPU fault。
- fd target VA：原 managed expert VA，用来打开/adopt 对应 UVM VA space。

## 专家权重搬运链条

当前 read-only controlled remap 的完整搬运链条是：

```text
FusedMoE expert weight tensor
  -> ptr/bytes from _uvm_expert_weight_slices()
  -> page-rounded host mmap staging
  -> driver_moe_slot_remap_batch(mode="gpu_copy_only")
  -> ioctl 91
  -> uvm_api_moe_slot_remap_batch()
  -> uvm_moe_slot_pool_apply_batch(copy_mode=GPU)
  -> uvm_moe_slot_pool_apply_entry_locked()
  -> uvm_moe_slot_pool_copy_gpu_entry_locked()
  -> uvm_moe_slot_pool_ensure_vidmem_backing_locked()
       - enforce VIDMEM budget
       - evict inactive backing if needed
       - uvm_mem_alloc_vidmem()
       - uvm_mem_map_gpu_kernel()
       - GPU memset
       - preserve old backing if resizing
  -> per chunk:
       - uvm_mem_alloc_sysmem_dma_and_map_cpu_kernel()
       - copy_from_user(host staging -> sysmem DMA staging)
       - uvm_mem_map_gpu_kernel(staging)
       - CE copy CPU_TO_GPU staging -> slot VIDMEM
  -> slot copy counters updated
```

这一阶段只证明内容迁移到 VIDMEM slot backing；它仍然不改变用户 VA 映射。

## 只读 PTE 安装链条

VIDMEM prefill 成功后，vLLM 再提交真实 remap：

```text
driver_moe_slot_remap_batch(
  entries,
  mode="slot_va_block_remap",
  validate_managed=<config>,
  round_to_pages=True,
  reset_slot_pool_after=True by default,
)
```

内核路径：

```text
uvm_api_moe_slot_remap_batch()
  -> uvm_moe_slot_pool_apply_slot_pte_batch(use_va_block_path=true)
  -> retain current/UVM mm
  -> uvm_va_space_down_write()
  -> for each entry:
       - find slot by (layer_id, slot_id)
       - validate owner_expert_id
       - ensure slot has VIDMEM backing
       - uvm_moe_slot_pool_install_va_block_slot_mapping()
           -> slice entry by VA block
           -> uvm_va_block_find_create_managed()
           -> UVM_VA_BLOCK_LOCK_RETRY()
                -> uvm_va_block_unmap() existing GPU mapping
           -> UVM_VA_BLOCK_LOCK_RETRY()
                -> uvm_va_block_map_moe_slot_readonly()
           -> copy block tracker
           -> uvm_tracker_wait_deinit()
       - record PTE range in slot multi-range vector
       - update pte_remap counters
  -> on any failure:
       - rollback installed ranges
       - mark remaining OK entries as rolled back/rejected
  -> uvm_va_space_up_write()
  -> release retained mm
```

`uvm_va_block_map_moe_slot_readonly()` 会通过 GPU page table HAL 写入只读 PTE：

```text
slot VIDMEM physical address + slot_offset + page_offset
  -> make_pte(..., UVM_PROT_READ_ONLY, cached flag for VIDMEM)
  -> write PTE
  -> set gpu_state->moe_slot_pte_bits
```

`moe_slot_pte_bits` 是关键 metadata。它告诉后续 UVM 路径：这些 PTE 不是普通 GPU-resident managed pages，而是 MoE slot-backed PTE。

## reset / clear 链条

Stage M controlled remap 当前默认在 placement 之后清理 slot PTE：

```text
GPUModelRunner._maybe_clear_uvm_init_time_slot_ptes_after_placement()
  -> driver_moe_slot_pool_query(max_entries=0)
  -> driver_moe_slot_pool_query(reset=True)
  -> driver_moe_slot_pool_query(max_entries=0)
  -> require active/backed/backing/residual all zero
```

内核 reset：

```text
uvm_api_moe_slot_pool_query(reset=True)
  -> retain mm
  -> uvm_va_space_down_write()
  -> uvm_moe_slot_pool_reset_locked()
       -> for each active slot:
            - uninstall all recorded VA-block slot PTE ranges
            - wait trackers
            - free slot backing
            - clear slot owner/range metadata
       -> scan VA blocks for residual moe_slot_pte_bits
  -> uvm_va_space_up_write()
  -> release mm
```

重要修复点：

- `uvm_va_block_unmap_moe_slot()` 现在在普通 unmap 运行期间保留 `moe_slot_pte_bits`，避免普通 UVM unmap 把 slot-backed PTE 短暂当成普通 resident GPU mapping。
- slot 支持多个 PTE range，避免同一 expert 的 gate/up 和 down range 清理不完整。
- `mm == NULL` 的 release/teardown guard 改成只 forget metadata，不进入 VA-block uninstall/tracker wait，避免在 teardown 异常上下文中做危险 wait。

## 生命周期 guard

slot-backed PTE 是只读实验映射，遇到更深 UVM 生命周期路径时采用保守策略：clear/fallback，而不是继续假装普通 residency。

已经接入的 guard 包括：

- fault path：fault 前清 slot PTE，回到正常 managed fault。
- migrate path：native migrate/remap 前清 slot PTE。
- native unmap/mmu notifier：unmap 前清 slot PTE。
- VA block split：split 前清 slot PTE。
- VA/GPU VA-space release：release 时清理或 forget slot metadata。
- eviction：active slot PTE 不作为普通 eviction page 处理；低 VIDMEM budget 下先清或拒绝。
- access counter：遇到 slot PTE 保守 clear/fallback。
- discard：discard 前清 slot PTE。

目标是避免后续 UVM 逻辑把 slot PTE 当成普通 GPU resident page。

## vLLM controlled E2E 默认隔离

为了让 E2E 聚焦只读专家权重 controlled remap，脚本在 `slot_pte_remap` 和 `slot_va_block_remap` 下默认关闭：

- `KV_DEVICE_DIRECT`
- fixed-weight device-direct

原因是 KV cache 是可写、生命周期复杂、会走 native CUDA allocation/RM 路径；它不属于当前只读专家权重 remap 目标。之前实验中过早把 KV cudaMalloc/native allocation 压力混入 controlled remap，导致问题定位变脏。

## 当前验证状态

已验证过的局部能力：

- `gpu_copy_only`：VIDMEM backing + GPU/CE copy 可用。
- readback/content verify：slot VIDMEM 内容可读回并校验。
- `slot_va_block_remap` controlled smoke：multi-entry、same-slot multi-range、cross-block、rollback、native unmap/split smoke 通过过。
- Stage M vLLM 路径中 48 层 MoE 可完成 driver backend：
  - `backend_ok=48`
  - `remap_failures=0`
  - `copy_failed=0`
  - `reset_ok=48`
  - `max_after_active_slots=0`
  - `max_after_residual_va_blocks=0`
  - `max_after_residual_pages=0`

尚未验证为稳定可用：

- tiny vLLM E2E READY/startup handshake 仍未稳定通过。
- 进程退出/mm teardown 仍可暴露 UVM VA-space/mm lock 相关卡死。
- 因此当前实现还不能称为 vLLM 生产稳定可用，只能称为只读 controlled remap 的核心数据搬运和 PTE smoke 骨架已建立。

