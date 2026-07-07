# MoE Slot 双缓冲复用机制说明

日期：2026-07-07

本文档总结当前仓库中基于 **MoE slot pool + `slot_va_block_remap`** 实现的 GPU 端固定小容量空间复用机制。重点场景是：

```text
CPU / UVM managed 数据：12MiB，划分为 6 个 2MiB VA block
GPU 固定空间：4MiB，划分为 2 个 2MiB slot

round 0: VA block 1,2 -> GPU slot 0,1
round 1: VA block 3,4 -> GPU slot 0,1，旧 block 1,2 直接丢弃
round 2: VA block 5,6 -> GPU slot 0,1，旧 block 3,4 直接丢弃
```

当前结论：这个机制已经可以支持 **只读数据** 的固定 4MiB 双 slot 复用。它不是普通 UVM migration，也不是完整生产级 UVM residency。它通过 driver-owned VIDMEM slot backing 和临时 GPU PTE remap，让目标 managed VA 在 GPU 侧读到 slot 中的新内容。

## 目标边界

支持：

- 固定 GPU VIDMEM backing，例如 2 个 2MiB slot，总计 4MiB。
- 多个 managed VA block 轮流映射到同一组 slot。
- 旧 VA block 的 slot-backed GPU PTE 被清理，不搬回 CPU。
- GPU kernel 继续使用原来的 managed VA 访问数据。
- slot backing 复用期间保持物理 backing 总量不增长。

不支持：

- 可写 slot-backed PTE。
- GPU 写入后的 dirty tracking。
- slot backing 到原 managed backing 的 copyback。
- 把 slot-backed PTE 当作完整普通 UVM GPU residency。
- 不受控的并发 fault/migrate/write 生命周期。

因此它适合只读专家权重、只读临时窗口、只读流式数据等场景；不适合 GPU 需要写回结果的 KV cache 或 activation 更新路径，除非后续补写语义。

## 为什么普通 UVM migration 不够

普通 `uvm_api_migrate()` / CUDA managed memory migration 只表达：

```text
把某个 VA range 迁移到某个 processor
```

它不表达：

```text
把 VA block X 映射到我指定的一块 GPU 物理 backing
```

也不提供固定 4MiB staging/window 的显式复用语义。UVM 会按自己的 VA block/chunk 管理 resident pages、eviction 和 backing。要实现“1-2、3-4、5-6 都复用同一块 4MiB GPU 空间”，需要绕过普通 migration 的物理 backing 选择，改为手动维护 slot backing，并改写目标 GPU PTE。

## 核心思路

机制分成两步：

1. **slot prefill**

   先把 CPU 侧数据搬到 driver-owned GPU VIDMEM slot 中。当前使用 `gpu_copy_only` mode：

   ```text
   user/staging CPU buffer
     -> UVM_MOE_SLOT_REMAP_BATCH(mode=gpu_copy_only)
     -> driver-owned VIDMEM slot backing
   ```

2. **VA block remap**

   再把目标 managed VA block 的 GPU PTE 临时改成指向 slot backing。当前使用 `slot_va_block_remap` mode：

   ```text
   managed target VA block
     -> VA block GPU unmap + tracker wait
     -> install readonly slot-backed GPU PTE
     -> GPU kernel 从原 VA 读到 slot 内容
   ```

复用 slot 时，继续使用相同 `(layer_id, slot_id)`，但换新的 `expert_id` 或 owner。driver 会清理该 slot 之前安装过的 slot-backed PTE metadata，保留已有 `slot_mem` backing，然后把新数据 copy 进去并安装新的 PTE。

## 内核 ABI

主要定义在：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`

### `UVM_MOE_SLOT_REMAP_BATCH`

ioctl 编号：`91`

每条 `UVM_MOE_SLOT_REMAP_ENTRY` 包含：

```text
layer_id
expert_id
slot_id
role
va_base
bytes
slot_offset
flags
status
```

对双 slot 复用而言，关键字段是：

- `layer_id`：实验中固定为一个逻辑 pool，例如 `42`。
- `slot_id`：物理逻辑 slot 编号，双 slot 为 `0` 和 `1`。
- `expert_id`：当前 owner。复用时换 owner，触发旧映射清理。
- `va_base`：copy 阶段表示源 CPU/staging VA；remap 阶段表示目标 managed VA。
- `bytes`：每个 VA block 大小，这里是 `2MiB`。
- `slot_offset`：slot 内偏移。双 slot 模式每个 slot 独立使用，通常为 `0`。如果一个 logical slot 内放多个 range，则用不同 offset。

相关 mode：

| mode | 名称 | 作用 |
|---:|---|---|
| 4 | `gpu_copy_only` | 分配或复用 VIDMEM slot backing，并把 CPU/staging 数据 copy 到 slot |
| 6 | `slot_va_block_remap` | VA-block-aware 地清目标 GPU 映射，再安装只读 slot-backed PTE |

### `UVM_MOE_SLOT_POOL_QUERY`

ioctl 编号：`92`

用于查询和 reset slot pool。关键观测字段：

- `active_slots`：当前有效 logical slots 数量。
- `backed_slots`：带 backing 的 slots 数量。
- `slot_backing_bytes`：所有 active slot 的 backing 总字节数。
- `copy_count` / `copy_bytes`：copy 到 slot 的累计次数和字节数。
- `pte_remap_count` / `pte_remap_bytes`：slot-backed PTE 安装累计次数和字节数。
- `pte_clear_count`：旧 slot-backed PTE 清理次数。
- `pte_block_unmap_count` / `pte_block_tracker_wait_count`：VA block unmap 和 tracker wait 次数。
- `vidmem_evict_count`：slot backing 被预算压力 evict 的次数。
- `pte_failure_count` / `copy_failed_count` / `pte_dangerous_op_reject_count`：健康检查字段。
- `pte_residual_va_block_count` / `pte_residual_page_count`：reset 后残留检查字段。

双 slot 复用成功时，典型期望是：

```text
active_slots == 2
backed_slots == 2
slot_backing_bytes == 4194304
copy_count == 6
pte_remap_count == 6
pte_clear_count >= 4
vidmem_evict_count == 0
copy_failed_count == 0
pte_failure_count == 0
pte_dangerous_op_reject_count == 0
```

## 内核实现原理

### 1. Slot pool 元数据

`uvm.c` 中维护全局 slot pool：

```text
g_uvm_moe_slot_pool[UVM_MOE_SLOT_POOL_MAX_SLOTS]
```

每个 slot 记录：

```text
layer_id / slot_id / owner_expert_id
slot_mem / slot_gpu / slot_bytes / backing_kind
pte_va_space / pte_gpu / pte_ranges
copy_count / copy_bytes
pte_remap_count / pte_clear_count
```

查找 key 是 `(layer_id, slot_id)`，不是 `(layer_id, expert_id)`。所以当新一轮继续提交相同 `layer_id + slot_id`，但 `expert_id` 改变时，driver 会认为这个 slot 被新的 owner 覆盖。

### 2. 复用时保留 backing、清 metadata

复用的关键函数是 `uvm_moe_slot_pool_clear_metadata_keep_backing_locked()`。

它做的事情是：

```text
保存 slot_mem / slot_gpu / slot_bytes / backing_kind
等待未完成 async copy
释放旧 slot-backed PTE mapping
memset 清 slot 元数据
恢复 slot_mem / slot_gpu / slot_bytes / backing_kind
```

因此旧 VA block 的 GPU PTE 会被清掉，但 2MiB slot backing 本身保留，下一轮 copy 可以直接复用这块 backing。

### 3. VIDMEM backing 分配与预算

`gpu_copy_only` 进入：

```text
uvm_api_moe_slot_remap_batch()
  -> uvm_moe_slot_pool_apply_batch(copy_mode=GPU)
  -> uvm_moe_slot_pool_apply_entry_locked()
  -> uvm_moe_slot_pool_copy_gpu_entry_locked()
  -> uvm_moe_slot_pool_ensure_vidmem_backing_locked()
```

`uvm_moe_slot_pool_ensure_vidmem_backing_locked()` 会检查：

```text
如果 slot 已经有同 GPU、VIDMEM、且 slot_bytes >= required_bytes：
    直接复用，不重新分配
否则：
    根据 uvm_moe_slot_pool_vidmem_budget_bytes 做预算 eviction
    uvm_mem_alloc_vidmem(required_bytes)
    uvm_mem_map_gpu_kernel()
    memset zero
```

双 slot 实验中每个 slot `required_bytes=2MiB`，预算设置为 `4MiB`，因此稳定保持两个 slot，总 backing 为 4MiB。

### 4. CPU/staging 到 VIDMEM slot 的 copy

`gpu_copy_only` copy path 使用 DMA-capable sysmem staging 和 GPU CE copy：

```text
copy_from_user(source VA -> staging sysmem)
uvm_mem_map_gpu_kernel(staging)
CE memcpy(staging -> slot_mem)
```

注意：如果源数据本身是 managed VA，直接在 ioctl 内 `copy_from_user()` 可能触发 UVM CPU fault，导致复杂锁路径甚至卡死。更稳妥做法是先在用户态把数据复制到 page-aligned host staging，再提交 `gpu_copy_only`。本次 probe 使用匿名 `mmap` staging 作为 source。

### 5. VA-block-aware PTE remap

`slot_va_block_remap` 进入：

```text
uvm_api_moe_slot_remap_batch()
  -> uvm_moe_slot_pool_apply_slot_pte_batch(use_va_block_path=true)
  -> uvm_moe_slot_pool_install_va_block_slot_mapping()
```

对每个目标 VA range，它按 VA block 切片：

```text
find/create target VA block
uvm_va_block_unmap(target block, gpu)
wait block tracker for UVM page-table update pushes
uvm_va_block_map_moe_slot_readonly()
record pte range metadata
```

`uvm_va_block_map_moe_slot_readonly()` 会写 GPU 4K PTE：

```text
PTE physical address = slot_mem + slot_offset
PTE protection = UVM_PROT_READ_ONLY
set gpu_state->pte_bits[GPU_READ]
clear GPU_WRITE / GPU_ATOMIC
set gpu_state->moe_slot_pte_bits
```

`moe_slot_pte_bits` 是非常关键的 VA block metadata。它告诉后续 UVM 路径：这些 GPU PTE 指向 driver-owned slot backing，不是普通 managed GPU-resident page。

#### Tracker wait 的真实语义

这里的 `tracker wait` 不是一个“GPU 用户 kernel 引用计数”，也不能自动知道某个 CUDA kernel 是否还在通过旧 PTE 访问这个 VA block。

当前代码等待的是 `uvm_va_block_t::tracker` 中记录的 UVM 内部 GPU push。这个 tracker entry 本质上是：

```text
UVM channel + tracking semaphore value
```

slot remap 路径在 unmap/map GPU PTE 时，会通过 UVM 的 MEMOPS channel 提交页表写入、TLB invalidate 等 push，并把这些 push 加到 VA block tracker 中。随后代码拷贝 `block->tracker` 到 local tracker，并调用 `uvm_tracker_wait_deinit()`，确保这些 UVM 页表修改 push 已完成。

因此它能保证：

- 旧 GPU PTE 的 unmap/清理 push 完成。
- 新 slot-backed readonly PTE 的 map/install push 完成。
- UVM 自己在这个 VA block tracker 里串起来的依赖完成。

它不能保证：

- 已经 launch 的用户 CUDA kernel 已经停止访问旧 VA。
- 其它 stream 上的用户 kernel 不会和下一轮 slot 复用并发。
- driver 能从页表本身反推出“还有多少线程正在使用旧翻译结果”。

所以双 slot 复用的正确用法是：用户态必须把每一轮 compute 和下一轮 slot reuse 串起来。最保守的方式是：

```text
round N remap
launch GPU kernel reading block N/N+1
cudaStreamSynchronize / cudaEventSynchronize
round N+1 reuse slot and remap next blocks
```

如果未来要做到完全异步流水，需要把 remap ioctl 做成 stream-ordered，或者让 ioctl 显式接收/等待用户 compute 完成 fence，再让后续 kernel 等待 remap 完成 fence。当前实验版还没有这层用户 kernel fence 集成。

### 6. 为什么旧数据不会搬回 CPU

旧窗口失效时，走的是 slot PTE 清理：

```text
uvm_moe_slot_pool_release_pte_mapping_locked()
  -> uvm_moe_slot_pool_uninstall_va_block_slot_mapping()
  -> uvm_va_block_unmap_moe_slot()
  -> uvm_va_block_unmap()
```

这条路径清的是 GPU page table 和 `moe_slot_pte_bits` metadata，不会把 slot backing 的内容 copy back 到 CPU，也不会把 slot 内容写回原 managed VA。因为当前 slot PTE 是只读的，语义上旧 slot 内容可以直接丢弃。

### 7. 普通 UVM 生命周期 guard

因为 slot-backed PTE 不是普通 residency，代码在多条 UVM 生命周期路径上做了保守处理：

- fault 前清 slot PTE，回退普通 fault。
- native migrate 前清 slot PTE，回退普通 migrate。
- unmap/mmu-notifier 前清 slot PTE。
- VA block split 前清 slot PTE。
- VA-space / GPU VA-space release 前清 slot PTE。
- PMM eviction 遇到 slot-backed root chunk 时避免普通 eviction 直接回收。

这些 guard 的目标是：遇到超出受控只读路径的操作时，先清掉实验 PTE，不把它伪装成普通 UVM managed residency。

## 双 slot 复用流程

以 6 个 VA block 为例：

```text
managed_base = 2MiB aligned
block_i_va = managed_base + (i - 1) * 2MiB

slot 0: (layer=42, slot_id=0)
slot 1: (layer=42, slot_id=1)
```

### Round 0：block 1,2

```text
gpu_copy_only:
  source(block1 payload) -> slot 0, expert_id=1
  source(block2 payload) -> slot 1, expert_id=2

slot_va_block_remap:
  block1 VA -> slot 0
  block2 VA -> slot 1

GPU read:
  read(block1 VA) == slot0 payload
  read(block2 VA) == slot1 payload
```

### Round 1：block 3,4

```text
gpu_copy_only:
  source(block3 payload) -> slot 0, expert_id=3
  source(block4 payload) -> slot 1, expert_id=4

slot owner 改变：
  slot 0 旧 owner expert_id=1 被覆盖
  slot 1 旧 owner expert_id=2 被覆盖
  旧 block1/block2 的 slot-backed PTE 被清理
  slot_mem backing 保留

slot_va_block_remap:
  block3 VA -> slot 0
  block4 VA -> slot 1
```

### Round 2：block 5,6

同理：

```text
slot 0: expert_id=5, block5 VA
slot 1: expert_id=6, block6 VA
旧 block3/block4 PTE 被清理
slot_mem backing 继续复用
```

最终 slot pool 中仍只有：

```text
slot 0 -> owner_expert_id=5 -> block5 VA
slot 1 -> owner_expert_id=6 -> block6 VA
slot_backing_bytes=4MiB
```

## 模块参数

运行前需要确保：

```bash
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable
echo 2097152 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes
echo 4194304 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes
```

含义：

- `uvm_moe_slot_pte_remap_enable=1`：打开实验 slot-backed PTE remap。
- `uvm_moe_slot_pte_remap_max_bytes=2097152`：允许单条 remap entry 覆盖 2MiB。
- `uvm_moe_slot_pool_vidmem_budget_bytes=4194304`：把 slot VIDMEM backing 预算限制为 4MiB。

实验脚本会临时设置这些参数，并在退出时恢复原值。

## 使用方式

### 直接运行双 slot probe

脚本：

```text
workloads/vllm/check_moe_slot_double_buffer_reuse.py
```

命令：

```bash
python3 workloads/vllm/check_moe_slot_double_buffer_reuse.py \
  --device 0 \
  --out-json log/moe_slot_double_buffer_reuse/result.json
```

默认行为：

- 分配约 14MiB managed memory，取 12MiB 2MiB-aligned 区间作为 6 个 VA block。
- 初始化 managed VA 为 old payload。
- 用两个 2MiB slot 分三轮覆盖 6 个 VA block。
- 每轮用 GPU kernel 从目标 managed VA readback。
- 每轮检查 slot pool backing、PTE、copy 和 health counters。
- 结束后 reset slot pool，并恢复模块参数。

### 复用逻辑的最小伪代码

```python
block_bytes = 2 * 1024 * 1024
layer_id = 42

for round_index in range(3):
    first_block = round_index * 2

    # 1. prefill slot backing
    submit_ioctl(
        mode="gpu_copy_only",
        entries=[
            {
                "layer_id": layer_id,
                "slot_id": 0,
                "expert_id": first_block + 1,
                "va_base": staging_payload_for_block[first_block + 0],
                "bytes": block_bytes,
                "slot_offset": 0,
            },
            {
                "layer_id": layer_id,
                "slot_id": 1,
                "expert_id": first_block + 2,
                "va_base": staging_payload_for_block[first_block + 1],
                "bytes": block_bytes,
                "slot_offset": 0,
            },
        ],
    )

    # 2. install slot-backed readonly GPU PTE for target managed VAs
    submit_ioctl(
        mode="slot_va_block_remap",
        validate_managed=True,
        entries=[
            {
                "layer_id": layer_id,
                "slot_id": 0,
                "expert_id": first_block + 1,
                "va_base": managed_base + (first_block + 0) * block_bytes,
                "bytes": block_bytes,
                "slot_offset": 0,
            },
            {
                "layer_id": layer_id,
                "slot_id": 1,
                "expert_id": first_block + 2,
                "va_base": managed_base + (first_block + 1) * block_bytes,
                "bytes": block_bytes,
                "slot_offset": 0,
            },
        ],
    )

    # 3. GPU kernel reads the original target VA.
    run_gpu_read(managed_base + first_block * block_bytes, 2 * block_bytes)
```

## 正确性验证结果

已运行：

```bash
python3 workloads/vllm/check_moe_slot_double_buffer_reuse.py \
  --device 0 \
  --out-json log/moe_slot_double_buffer_reuse/result.json
```

结果：

```text
passed=True
```

关键计数：

```text
final_active_slots=2
final_backed_slots=2
final_slot_backing_bytes=4194304
total_copy_count=6
total_pte_remap_count=6
total_pte_clear_count=4
vidmem_evict_count=0
```

逐轮观测：

| round | VA blocks | GPU readback | old payload | slot_backing_bytes | pte_clear_count |
|---:|---|---|---|---:|---:|
| 0 | 1,2 | 匹配当前 slot payload | 不匹配 | 4194304 | 0 |
| 1 | 3,4 | 匹配当前 slot payload | 不匹配 | 4194304 | 2 |
| 2 | 5,6 | 匹配当前 slot payload | 不匹配 | 4194304 | 4 |

解释：

- round 0 后，两个 slot 首次安装，无旧 PTE，`pte_clear_count=0`。
- round 1 复用 slot 0/1，清掉 block 1/2 的 slot-backed PTE，`pte_clear_count=2`。
- round 2 再次复用 slot 0/1，清掉 block 3/4 的 slot-backed PTE，`pte_clear_count=4`。
- backing 始终保持 `4194304`，说明没有为 6 个 block 分配 12MiB GPU backing。
- `vidmem_evict_count=0`，说明不是靠预算 eviction 腾空间，而是同两个 slot backing 原地复用。

结果文件：

```text
log/moe_slot_double_buffer_reuse/result.json
```

## 与原生 UVM prefetch 的性能对比

先给结论：当前实验版双 slot 机制的 **raw latency / bandwidth 不如原生 `cudaMemPrefetchAsync` / UVM prefetch**。它的主要收益不是搬得更快，而是：

- GPU backing 容量可被硬限制为固定窗口，例如 4MiB。
- 旧窗口可以直接丢弃，不需要 copyback。
- 上层可以精确控制“当前只有哪些 VA block 在 GPU 上可读”。

换句话说，它是一个 **capacity / residency 控制机制**，不是当前形态下的高速 prefetch 替代品。

### 当前双 slot probe 的耗时

本次 6 个 2MiB VA block、两个 2MiB slot 的 probe 中，每轮处理 4MiB 数据：

| 阶段 | 数据量 | p50 延迟 | mean 延迟 | 近似有效带宽 |
|---|---:|---:|---:|---:|
| `gpu_copy_only` prefill | 4MiB | 3.456ms | 3.423ms | 约 1.13GiB/s |
| `slot_va_block_remap` | 4MiB PTE window | 1.814ms | 1.816ms | 不主要按数据带宽衡量 |
| prefill + remap | 4MiB | 5.223ms | 5.239ms | 约 0.75GiB/s |

逐轮原始数据：

| round | prefill ms | remap ms | total ms |
|---:|---:|---:|---:|
| 0 | 3.605 | 1.867 | 5.472 |
| 1 | 3.207 | 1.814 | 5.020 |
| 2 | 3.456 | 1.767 | 5.223 |

这些数字来自：

```text
log/moe_slot_double_buffer_reuse/result.json
```

注意这里的 `gpu_copy_only` 不是纯 `cudaMemcpyAsync`。它包含：

```text
copy_from_user(user/staging VA -> driver sysmem staging)
uvm_mem_map_gpu_kernel(staging)
CE copy(staging -> driver-owned VIDMEM slot)
等待 copy tracker 完成
```

因此小窗口下固定开销很明显。

### 已有 native prefetch / copy microbench 参考

仓库里已有一些不完全同口径、但有参考价值的结果：

| 实验 | 数据量 | case | p50 complete | 有效带宽 |
|---|---:|---|---:|---:|
| `contiguous_prefetch_vs_copy_20mib.json` | 20MiB | native CUDA prefetch, 1 range | 3.273ms | 5.97GiB/s |
| `contiguous_prefetch_vs_copy_20mib.json` | 20MiB | native CUDA prefetch, 10 ranges | 3.055ms | 6.39GiB/s |
| `contiguous_prefetch_vs_copy_20mib.json` | 20MiB | ordinary H2D copy, 1 range | 0.638ms | 30.60GiB/s |
| `uvm_native_cuda_prefetch_no_expert_fastpath_noncontig_10x2mib_rerun.json` | 10 * 2MiB | native CUDA prefetch, non-contig | 3.260ms | 5.99GiB/s |
| `uvm_expert_fast_regions_vs_cuda_prefetch_contig_ids_10x2mib_rerun.json` | 10 * 2MiB | expert fast regions | 2.401ms | 8.14GiB/s |

这些结果说明：

- 原生 UVM prefetch 常见有效带宽约 6GiB/s 量级。
- 项目里的 expert fast path 可到约 8GiB/s 量级。
- 普通 pinned H2D copy 明显更快，可到 30GiB/s 以上。
- 当前 slot remap probe 的 end-to-end 小窗口有效带宽只有约 0.75GiB/s，主要被 ioctl、staging、VA block unmap、PTE install、tracker wait 等固定开销吃掉。

### 为什么 slot 机制更慢

主要原因有四个。

1. `gpu_copy_only` 多了一次 driver sysmem staging：

   用户态 source 先 `copy_from_user()` 到 driver staging，再由 CE copy 到 VIDMEM slot。native prefetch 走的是 UVM migration/prefetch 路径，数据和 residency 更新更贴近原生路径。

2. 每轮要改 GPU page table：

   `slot_va_block_remap` 需要先 unmap 目标 VA block 的旧 GPU PTE，等待 UVM tracker，再写只读 slot-backed PTE 和 TLB invalidate。这部分在当前 probe 中约 1.8ms/4MiB。

3. 现在是同步 ioctl 形态：

   每轮 prefill 和 remap 都在 ioctl 返回前完成，当前没有和用户 compute 做 stream-ordered overlap。

4. 单轮只有 4MiB，固定开销占比高：

   对 2MiB/4MiB 这种小窗口，ioctl、锁、VA block 查找、tracker wait、PTE batch 的成本很容易压过实际 copy 成本。

### 什么时候仍然值得用 slot 机制

如果目标只是“尽快把 managed 数据搬到 GPU 上”，native UVM prefetch 更合适。

如果目标是下面这种受控窗口语义，slot 机制才有价值：

```text
总 managed 数据很大，例如 12MiB / 12GiB / 更多
GPU 只允许固定窗口，例如 4MiB / 几百 MiB
每轮只读当前窗口
旧窗口可丢弃
不希望 UVM 自己决定 backing 和 eviction
```

对本文的 6-block 例子，native prefetch 如果依次 prefetch block 1..6，GPU residency/backing 可能随访问历史增长到接近 12MiB，或者依赖 UVM eviction 回收。slot 机制则强制：

```text
active GPU slot backing = 2 * 2MiB = 4MiB
```

这会牺牲 raw prefetch 性能，换来严格的 GPU memory footprint。

### 后续性能优化方向

如果要让 slot 机制接近 native prefetch，需要重点优化：

- 去掉或减少 `copy_from_user()` staging，例如让用户态提供 pinned staging / dma-buf / 已注册 host buffer。
- 支持 async `gpu_copy_only`，并让 remap 等待 copy fence，而不是整轮同步。
- 把 remap 做成 stream-ordered，让下一轮 prefill/remap 与上一轮 compute overlap。
- 批量处理更多 VA block，摊薄 ioctl 和 VA block lock/tracker 开销。
- 对固定 2MiB VA block 建 fast path，减少 4K PTE 写入和 split/unmap 成本。
- 如果应用能接受显式 pointer indirection，优先考虑显式 CUDA slot tensor / hot-cold layout，避免改 UVM PTE。

## 如何判断机制是否真的生效

建议同时看三类证据。

### 1. 功能正确性

每轮 GPU readback 必须满足：

```text
gpu_readback_matches_slot_payload == true
gpu_readback_matches_old_payload == false
```

这说明 GPU 通过原 managed VA 读到了 slot backing，而不是原 managed 内容。

### 2. 固定容量复用

最终必须满足：

```text
active_slots == 2
backed_slots == 2
slot_backing_bytes == 4194304
```

这说明 GPU 端只保留两个 2MiB backing。

### 3. 旧映射清理

复用后必须看到：

```text
pte_clear_count >= 4
```

对 3 轮双 slot 来说，round 1 清旧 block 1/2，round 2 清旧 block 3/4，总共至少 4 次旧 slot-backed PTE 清理。

同时 health counters 必须为 0：

```text
copy_failed_count == 0
pte_failure_count == 0
pte_dangerous_op_reject_count == 0
```

reset 后必须没有残留：

```text
pte_residual_va_block_count == 0
pte_residual_page_count == 0
```

## 常见问题

### 为什么不是 `slot_offset=0` 和 `slot_offset=2MiB` 放在同一个 slot？

两种表达都可以：

- 一个 logical slot，`slot_offset=0/2MiB`，总 backing 4MiB。
- 两个 logical slot，每个 slot 2MiB，`slot_offset=0`。

本文档和 probe 使用第二种，因为它更直接对应“双 slot”概念：

```text
slot_id=0 -> 2MiB
slot_id=1 -> 2MiB
```

如果要把两个 2MiB range 放到同一个 logical slot，则需要同一个 `(layer_id, slot_id)`、相同 owner，并设置不同 `slot_offset`。已有 smoke 中的 `--same-slot-ranges` 覆盖过这类路径。

### 为什么 `copy_count=6`，但每个最终 entry 的 `copy_count=1`？

全局 `copy_count=6` 表示总共 copy 了 6 个 VA block 的 payload。

最终 query 只剩两个 active slots，owner 是 round 2 的 expert 5/6。每个 slot 的 metadata 在复用时被清过，所以 per-slot `copy_count` 显示当前 owner 的一次 copy；全局 counter 保留累计值。

### 为什么 `pte_block_unmap_count` 比 6 大？

安装新映射时需要 unmap 目标 VA block；复用旧 slot 时还要清旧 VA block 的 slot-backed PTE。所以三轮结束后：

```text
6 次新目标 install unmap
4 次旧目标 clear unmap
合计约 10
```

这和实验结果中的 `pte_block_unmap_count=10`、`pte_block_tracker_wait_count=10` 一致。

### 旧 VA block 后续再被 GPU 访问会怎样？

旧 slot-backed PTE 被清掉后，它不再指向 slot backing。后续如果 GPU 再访问旧 VA block，会回到普通 UVM fault/migrate 语义，读到原 managed backing 中的内容，而不是旧 slot 内容。当前机制没有把 slot 内容 copy back 到 managed backing。

### GPU 写旧或当前 slot-backed VA 会怎样？

当前 PTE 是只读的。写访问不属于支持路径。已有 guard 目标是清掉 slot PTE 并回退普通 UVM 语义，而不是把写入落到 slot backing 并维护 dirty 状态。

## 后续扩展方向

如果要把这个机制从只读实验推进到更通用的窗口复用，需要补：

- 可写 PTE mode。
- slot dirty bitmap。
- slot -> managed backing 的 copyback 或明确 discard-write API。
- 与 UVM fault/migrate/evict 的并发状态机。
- per-VA-block 生命周期引用，避免 remap 与 fault/write/unmap 竞争。
- 更细粒度的用户态 API，直接表达 fixed window / double buffer，而不是复用 MoE 的 `layer_id/expert_id/slot_id` 命名。

在这些工作完成前，推荐继续把它当作“只读 controlled remap / fixed GPU read window”使用。
