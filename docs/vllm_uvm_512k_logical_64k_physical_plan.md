# vLLM UVM 512KiB Logical Envelope / 64KiB Physical Backing Plan

本文档记录后续执行计划：以 **512KiB logical envelope = 8 个 64KiB PMM chunks** 为目标，继续推进 vLLM MoE cold expert offload 与 UVM reserved slot 机制。

核心变化是：**不再把 512KiB 当作 UVM 原生 PMM chunk size**。512KiB 只作为 vLLM/runtime 的逻辑管理单位；driver 底层实际使用当前系统已经支持的 **64KiB user PMM chunk** 来实现。

## 总体原则

Phase S0 已确认当前系统状态：

```text
pmm user chunk sizes = [4KiB, 64KiB, 2MiB]
has_512k = false
has_64k = true
```

因此新路线是：

```text
logical envelope size = 512KiB
physical backing unit = 64KiB
one logical 512KiB slot = 8 * 64KiB PMM chunks
```

这样既避免强行扩展 PMM 支持 512KiB，又能把 2MiB envelope 的牵连范围降到 1/4。

## Phase S0: 现状确认，已完成

目标：确认当前系统不能直接用 512KiB PMM chunk，但可以用 64KiB 作为物理 backing 单元。

验收结果：

```text
pmm mask = 0x211000
supported user chunks = 4KiB, 64KiB, 2MiB
512KiB native chunk = unsupported
64KiB native chunk = supported
```

当前 reserved pool 仍然是：

```text
slot_bytes = 512KiB
physical_bytes = 2MiB
48 slots backing = 96MiB
```

目标实现后应变成：

```text
slot_bytes = 512KiB
chunk_size = 64KiB
chunk_count = 8
physical_bytes = 512KiB
48 slots backing = 24MiB
```

相关结果文档：

```text
docs/vllm_uvm_pmm_512k_phase_s0_results_20260713.md
```

## Phase S1: Reserved Slot 支持 512KiB Logical / 64KiB Physical

目标：先不碰 expert release，只改 reserved pool prealloc，让它真正分配 64KiB PMM chunks。

实现点：

```text
driver_expert_reserved_pool_prealloc()
  slot_bytes = 512KiB
  choose physical_chunk_size = 64KiB
  chunk_count = slot_bytes / 64KiB = 8
  uvm_pmm_gpu_alloc_user(..., chunk_count=8, chunk_size=64KiB, ...)
```

需要改的数据结构：

```text
uvm_expert_reserved_pool_slot_t 增加或保存:
  chunk_size = 64KiB
  chunk_count = 8
  physical_bytes = 512KiB
```

同时改掉当前假设：

```text
reserved_slot_chunk_from_offset()
```

它现在按 `UVM_CHUNK_SIZE_MAX=2MiB` 算 chunk index，必须改成按 `slot->chunk_size` 算：

```text
chunk_index  = slot_offset / slot->chunk_size
chunk_offset = slot_offset % slot->chunk_size
```

验收：

```text
prealloc 48 个 512KiB slot
success_count = 48
failure_count = 0
logical_slot_bytes = 24MiB
reserved_backing_bytes = 24MiB

每个 entry:
  slot_bytes = 512KiB
  physical_bytes = 512KiB
  chunk_count = 8
```

## Phase S2: 512KiB Logical Subrange Release Smoke

目标：从一个 2MiB managed block 里，只释放一个 512KiB logical range；底层释放 8 个 64KiB PMM chunks。

释放流程：

```text
输入:
  base = 2MiB block 内某个 512KiB-aligned VA
  length = 512KiB

driver:
  1. 找到 VA block
  2. 找到目标 512KiB 覆盖的 pages
  3. 确保目标 pages CPU resident
  4. 清目标 pages GPU PTE / mapping
  5. 将覆盖目标区间的 GPU chunk split 到 64KiB
  6. 只摘除目标 8 个 64KiB chunks
  7. 更新 gpu_state residency / chunk ownership
  8. 将这 8 个 chunks 通过 PMM 正规 free 路径释放
  9. 邻近 1.5MiB 保持 GPU resident
```

注意：不能手动把 chunk 挂 free list。必须走 PMM split/free/merge 的正规状态机，保持 VA block、PTE、residency mask、PMM ownership 一致。

验收：

```text
release 前:
  2MiB GPU resident

release 后:
  target 512KiB:
    GPU resident = 0
    CPU resident = 512KiB

  neighbor 1.5MiB:
    GPU resident 仍保持
```

## Phase S3: 512KiB Restore Smoke

目标：验证释放出的 512KiB 能通过 reserved slot 恢复到原 VA。

流程：

```text
1. 分配 2MiB managed block
2. GPU prefetch 整个 block
3. release 中间 512KiB
4. prealloc 一个 512KiB logical slot
   backing = 8 * 64KiB
5. 从 CPU resident pages copy 到 slot chunks
6. 给原 VA target 512KiB 安装 readonly slot-backed PTE
7. GPU 从原 VA 读回校验
```

验收：

```text
slot prealloc:
  reserved_backing_bytes = 512KiB

restore:
  success_count = 1
  accepted_bytes = 512KiB

读回:
  target 512KiB 正确
  neighbor 数据不被破坏
```

## Phase S4: vLLM Envelope 泛化为 512KiB Logical Envelope

目标：把当前 2MiB envelope 改成可配置逻辑粒度。

新增配置：

```bash
VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES=524288
```

代码泛化：

```text
_uvm_build_2mb_envelope_records
  -> _uvm_build_chunk_envelope_records

weights:<layer>:uvm2m:<role>:...
  -> weights:<layer>:uvmchunk:524288:<role>:...
```

对齐逻辑：

```text
envelope_start = align_down(ptr, 512KiB)
envelope_end   = align_up(ptr + size, 512KiB)
```

预期切分：

```text
moe_down   ~= 1.5MiB -> 3 envelopes
moe_gateup ~= 3.0MiB -> 6 envelopes
每个 expert 两个 role -> 9 个 512KiB envelopes
```

cold8 单层最坏：

```text
8 experts * 9 envelopes = 72 logical slots/layer
72 * 512KiB = 36MiB
```

建议 slot 配置：

```bash
RESERVED_SLOT_BYTES=524288
RESERVED_SLOT_COUNT=128 或 160
```

对应 backing：

```text
128 * 512KiB = 64MiB
160 * 512KiB = 80MiB
```

比当前 `48 * 2MiB = 96MiB` 更省。

## Phase S5: cold8 机制组实验

实验参数：

```bash
RESERVED_SLOT_BYTES=524288
RESERVED_SLOT_COUNT=160
VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=0
```

先尝试关闭 CPU-source refresh。因为 512KiB logical envelope 不再大面积牵连邻近 hot/warm expert，source 被普通 fault 破坏的概率应明显降低。

验收：

```text
Stage-M 后:
  cold logical GPU bytes -> 0
  cold logical CPU bytes -> expected

released envelope bytes:
  接近 logical cold bytes
  waste ratio 显著低于 2MiB 方案

slot prealloc:
  success_count == slot_count
  reserved_backing_bytes == slot_count * 512KiB

runtime:
  reserved_prefetch_failed = 0
  completed_requests = 32/32
  尽量不需要 PREFETCH_CPU_SOURCE=1

性能:
  tok/s 明显优于 2MiB envelope + CPU-source refresh
```

## Phase S6: 对比报告

对比三组：

```text
A. baseline / no offload
B. 2MiB envelope + CPU-source refresh
C. 512KiB logical envelope + 64KiB physical chunks
```

核心表：

```text
logical cold bytes
released envelope bytes
waste ratio
slot backing bytes
runtime prefetch bytes
CPU-source refresh 次数
reserved prefetch failures
completed requests
tok/s
```

## 预期效果

当前 2MiB 方案：

```text
logical cold ~= 1.69GiB
released envelope ~= 2.46GiB
extra ~= 788MiB
```

512KiB logical envelope 预期：

```text
moe_down 1.5MiB -> 3 * 512KiB
moe_gateup 3MiB -> 6 * 512KiB
envelope extra 接近 0
```

最终判断：

```text
512KiB 不适合作为 native PMM chunk size
512KiB 适合作为 logical envelope size
64KiB 是当前最稳的 physical PMM backing unit
```

推荐下一步先做：

```text
Phase S1: 512KiB logical slot / 64KiB physical backing prealloc smoke
```
