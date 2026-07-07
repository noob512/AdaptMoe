# UVM Expert Fast Prefetch 与 Reserved Backing 固定窗口融合实施计划

日期：2026-07-07

本文档给出一个实施计划：把已有的 **MoE expert batch fast prefetch** 机制和新的 **预分配固定 VIDMEM backing / 指定物理空间复用** 机制结合起来。

目标不是简单替换 `cudaMemPrefetchAsync`，而是获得下面这种语义：

```text
router/topk 选中多个 expert region
driver 只使用预先分配好的固定 GPU resident backing pool
UVM prefetch / make_resident 把选中 expert 搬到指定 reserved backing
expert 原 managed VA 继续使用普通 UVM GPU PTE
resident/read_duplicated_pages metadata 继续由 UVM 管理
旧 owner 被显式 invalid/discard 后，reserved backing 可被下一批 expert 复用
```

也就是说，最终希望得到：

```text
expert_fast_regions 的 region-id / VA-block batch 收集能力
  +
UVM PMM / VA block populate 路径里的固定 reserved backing 分配能力
```

本文档的主目标是 **路线 B：reserved-backed native residency**，即对 UVM 的 PMM 分配、VA block populate、resident metadata、fault policy 做较大幅度修改，让普通 UVM residency 本身使用指定 backing。必要时，可以把 UVM 降级为“元数据与同步管理层”来使用：保留它对 VA range、VA block、GPU page table、tracker、mmu notifier、lifetime cleanup 的管理能力，但关闭或拦截 native UVM 按需 migration / prefetch / GPU fault recovery 的数据路径。数据搬运、backing 选择、backing 复用和失败策略由新的 reserved-backed fast path 显式控制。

这个 fixed reserved backing 机制不是要求所有 expert 权重都走固定分配路径。实际策略由 KV cache 压力决定：

```text
KV cache 压力小 / GPU memory 充足:
  expert 权重可以全部常驻 GPU
  使用普通 UVM residency 或普通 GPU allocation
  不需要 reserved-backed 周期性搬运

KV cache 压力大 / GPU memory 需要让给 KV cache:
  只有部分 expert 权重常驻 GPU
  其余 expert 卸载到 CPU managed backing
  被卸载的 expert 在每轮被选中时，必须通过 reserved-backed path 搬入固定 GPU window
```

因此 reserved-backed path 的约束对象是 **offloaded experts**，不是全部 experts。常驻 GPU experts 可以继续走普通 fast prefetch / ordinary residency；只有被 placement policy 判定为 CPU-offloaded 的 experts，才需要强制使用固定 reserved backing，以保证 KV cache 高压场景下 GPU footprint 受控。

## 背景

### 现有 expert fast prefetch

参考：

```text
docs/uvm_expert_prefetch_batch_experiments.md
```

当前推荐路径是：

```text
Python region_id list
  -> UVM_PHASE_F_BATCH_REGIONS
  -> uvm_phase_f_expert_read_dup_fast_batch_regions()
  -> region_id -> registered expert region
  -> collect selected ranges into VA-block entries
  -> sort/dedup entries
  -> uvm_va_block_expert_promote_read_duplicate_batch()
```

它的核心优点：

- 能一次处理多个 expert。
- 能把 region_id 转换成多个 VA block entry。
- 对 4MiB/6MiB 这类跨多个 VA block 的 expert 明显快于 serial CUDA prefetch。
- `async=true` 可以跳过最后的 `uvm_tracker_wait()`，让 copy/map completion 与后续 compute 部分重叠。

但它的目标是普通 UVM residency：

```text
CPU managed pages
  -> UVM 分配/使用普通 GPU resident backing
  -> 更新 resident/read_duplicated_pages metadata
  -> 安装普通 GPU PTE
```

更准确地说：当前 native UVM prefetch / `expert_fast_regions` 的 ABI 不提供“指定某块 GPU physical backing / 预留 backing”的参数，默认实现也会从普通 UVM PMM 路径分配 GPU resident backing。因此从用户态调用者视角看，它不能要求“必须使用这块 4MiB/几百 MiB 窗口”。

但这不是理论上做不到。如果修改 UVM 内部的 GPU backing 分配路径，例如：

```text
uvm_va_block_make_resident_read_duplicate()
  -> block_populate_pages()
  -> uvm_va_block_populate_pages_gpu()
  -> block_populate_gpu_chunk()
  -> uvm_pmm_gpu_alloc_user()
```

就可以新增一种“从预留 PMM chunk / 指定 slot chunk 分配”的策略，让普通 UVM residency 本身使用指定 backing。也就是说，指定 backing 有两种路线；本文档以路线 B 为主：

```text
路线 A：slot-backed PTE（已有实验原型 / 对照路径）
  不把 slot backing 记为普通 resident page
  只临时把目标 VA 的 GPU PTE 指向 slot_mem
  旧 slot owner 可直接 discard

路线 B：reserved-backed native residency（主要目标）
  修改 UVM PMM / VA block populate 路径
  让普通 UVM resident backing 来自预留 chunk pool
  继续更新 resident/read_duplicated_pages metadata
  默认继续使用普通 UVM PTE、accounting 和 tracker 语义
  fault / eviction / on-demand migration 在 strict fixed-footprint 模式下需要被限制、改写或关闭
```

路线 B 更接近“原生 UVM prefetch + 指定物理空间”，也是本计划的目标。它的代价是侵入性更强：需要处理 PMM chunk 预留、chunk 生命周期、VA block split/merge、eviction、fault fallback、read-duplicate accounting，以及多个 expert 复用同一 backing 时如何让 UVM metadata 显式失效旧 resident copy。

### 现有 MoE slot pool

参考：

```text
workloads/vllm/docs/moe_slot_double_buffer_reuse.md
```

当前 slot 机制已有这些能力：

- `UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC = 8`
  - 为 `(layer_id, slot_id)` 预分配 driver-owned VIDMEM backing。
- `UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT = 9`
  - 提交 CPU/staging 到 VIDMEM slot 的异步 CE copy。
- `UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP = 6`
  - 对目标 managed VA block 安装只读 slot-backed GPU PTE。
- slot owner 改变时：
  - 清旧 slot-backed PTE metadata。
  - 保留 `slot_mem` backing。
  - 不 copyback。

但当前 slot copy path 主要是：

```text
copy_from_user(user/staging VA -> driver sysmem staging)
CE copy(staging -> slot_mem)
```

这没有复用 expert fast prefetch 中已经做好的 managed VA block 收集、CPU page 判定、batch CE copy 等能力，所以小窗口性能明显偏低。

## 总体结论

本文优先选择路线 B，也就是 reserved-backed native residency：修改 `uvm_va_block_expert_promote_read_duplicate_batch()` 相关的 populate/copy/metadata 路径，或者在它旁边新增 reserved-backed variant，让普通 UVM residency 直接使用预留 GPU backing。这样 GPU 侧仍使用普通 managed VA 和普通 UVM PTE，但 backing 来源被固定窗口控制。

需要保留下面这些普通 UVM residency 语义：

- 它会 populate GPU resident pages。
- 它会更新 `resident` / `read_duplicated_pages`。
- 它会安装普通 GPU PTE。
- 它会触发 UVM residency accounting / hooks。

但需要替换其中的 backing allocation 和复用策略：

```text
默认：
  block_populate_gpu_chunk()
    -> uvm_pmm_gpu_alloc_user()
    -> 任意普通 PMM user chunk

目标：
  block_populate_gpu_chunk_reserved()
    -> uvm_expert_reserved_pool_get_chunk()
    -> 指定 reserved backing pool / slot chunk
```

推荐新增 reserved-backed fast path，而不是直接把所有普通 UVM allocation 全局改掉：

```text
uvm_va_block_expert_promote_read_duplicate_reserved_batch()
```

它复用 expert fast path 的前半段：

```text
region_id -> expert region -> VA block entries -> sort/dedup
```

后半段保留普通 UVM residency metadata，但替换 backing 分配和复用控制：

```text
保留：
  populate GPU resident pages
  copy CPU managed pages -> GPU backing
  update resident/read_duplicated_pages
  install ordinary GPU PTE

替换：
  GPU backing 分配必须来自 reserved backing pool
  reuse 前显式 invalidate/discard 旧 owner 的 resident metadata
  strict 模式下禁止 native fault/migration 自动分配 pool 外 backing

改为：
  CPU managed physical pages -> reserved GPU resident backing
  target managed VA GPU PTE -> ordinary UVM PTE for that reserved backing
  reserved pool metadata records owner/range/epoch
```

这里的原则是：优先保留 UVM 的 VA block、resident mask、read-duplicate、PTE、tracker、cleanup 等元数据体系；但 backing 分配、固定窗口复用和 strict 模式下的 fault/migration 策略由 reserved-backed path 接管。必要时，可以只保留 UVM 元数据管理功能，关闭其原生按需搬运机制，避免 native fault/migration 自动分配 pool 外 GPU backing。

## 目标语义

### 输入

用户态提交本轮 **selected 且处于 CPU-offloaded 状态的 experts**：

```text
region_id[]
```

同时提交 reserved backing assignment：

```text
region_id -> reserved_slot_id / reserved_offset
```

常驻 GPU 的 selected experts 不进入这个 ioctl。它们已经有普通 GPU residency，不需要占用 reserved backing window。

region registry 已经包含：

```text
base
length
pool
layer_id
expert_id
role
flags
```

因此大多数情况下 ABI 不需要用户重复传 `base/length/layer/expert/role`，只需要传 `region_id` 和目标 reserved backing 信息。

### 输出

每个成功 entry 满足：

```text
entry 只覆盖 CPU-offloaded expert
reserved GPU backing 中的数据 == region 原 CPU managed 内容
region 原 managed VA 的 GPU PTE 是普通 UVM PTE，但其 backing 来自 reserved pool
GPU kernel 继续用原 expert pointer 读取数据
resident/read_duplicated_pages metadata 与 reserved backing owner 一致
reserved backing bytes 不超过预设预算
常驻 GPU expert 的 residency 不受 reserved pool 限制
```

### 不支持

第一阶段仍然不支持：

- GPU 写后 dirty 数据保留。
- reserved backing -> CPU managed backing copyback。
- pool 外 native on-demand migration 自动兜底。
- 同一个 reserved backing 同时作为多个 active owner 的普通 resident copy。
- driver 自动判断用户 kernel 是否还在读旧 reserved backing。

用户态仍必须保证：

```text
复用 reserved backing 前，上一个使用该 backing 的 GPU compute 已完成。
```

## ABI 设计建议

### 不推荐直接扩展 `UVM_PHASE_F_BATCH_REGIONS`

`UVM_PHASE_F_BATCH_REGIONS_PARAMS` 当前只有：

```text
action
count
gpu_uuid
async
region_ids[]
results[]
```

它没有 reserved backing assignment 字段。如果硬塞到已有 struct，容易破坏 ABI 对齐，也无法表达：

```text
region_id -> reserved_slot_id
region_id -> reserved_offset
region_id -> reuse / discard / strict fault 语义
```

### 推荐新增 reserved control plane ioctl

接口拆成两层：

```text
Phase 1:
  UVM_EXPERT_RESERVED_POOL
  只负责 reserved PMM chunk pool 的 prealloc / query / reset

Phase 2:
  UVM_EXPERT_RESERVED_PREFETCH_BATCH
  负责 region_id -> reserved slot assignment -> VA block native residency
```

当前 `uvm_ioctl.h` 里 `UVM_IOCTL_BASE(81..97)` 已经被 Phase-F、Phase-N、MoE slot query/readback 等接口占用。Phase 1 已使用下一个编号：

```c
#define UVM_EXPERT_RESERVED_POOL UVM_IOCTL_BASE(98)
```

如果后续新增 Phase 2 prefetch batch ioctl，先以 `rg "UVM_IOCTL_BASE"` 重新确认编号，避免 ABI 冲突；不要复用 `98`。

Phase 1 pool entry：

```c
typedef struct
{
    NvU32 pool_id;                         // IN/OUT
    NvU32 slot_id;                         // IN/OUT
    NvU32 flags;                           // IN, allow_evict 等
    NvU32 status;                          // OUT
    NvU32 state;                           // OUT
    NvU32 chunk_count;                     // OUT
    NvU64 slot_bytes NV_ALIGN_BYTES(8);    // IN/OUT
    NvU64 physical_bytes NV_ALIGN_BYTES(8);// OUT
    NvU64 epoch NV_ALIGN_BYTES(8);         // OUT
    NvU64 first_chunk_address NV_ALIGN_BYTES(8); // OUT debug
    NvU64 last_chunk_address NV_ALIGN_BYTES(8);  // OUT debug
} UVM_EXPERT_RESERVED_POOL_ENTRY;
```

Phase 1 pool params：

```c
typedef struct
{
    NvU32 mode;                            // query / prealloc / reset
    NvU32 count;                           // prealloc entry count
    NvProcessorUuid gpu_uuid;              // prealloc target GPU
    NvU32 max_entries;                     // query output cap
    NvU32 success_count;
    NvU32 failure_count;
    NvU32 total_slots;
    NvU32 returned_slots;
    NvU32 active_slots;
    NvU32 backed_slots;
    NvU32 chunk_size;
    NvU64 reserved_backing_bytes NV_ALIGN_BYTES(8);
    NvU64 logical_slot_bytes NV_ALIGN_BYTES(8);
    NvU64 current_epoch NV_ALIGN_BYTES(8);
    NvU64 prealloc_count NV_ALIGN_BYTES(8);
    NvU64 reset_count NV_ALIGN_BYTES(8);
    NvU64 alloc_failed_count NV_ALIGN_BYTES(8);
    UVM_EXPERT_RESERVED_POOL_ENTRY entries[MAX];
    NV_STATUS rmStatus;
} UVM_EXPERT_RESERVED_POOL_PARAMS;
```

Phase 2 prefetch batch entry 仍建议只表达 region 到 slot 的映射：

```c
typedef struct
{
    NvU64 region_id NV_ALIGN_BYTES(8);     // IN
    NvU32 pool_id;                         // IN，reserved backing pool namespace
    NvU32 slot_id;                         // IN，reserved backing slot
    NvU32 flags;                           // IN
    NvU32 status;                          // OUT
    NvU64 slot_offset NV_ALIGN_BYTES(8);   // IN
    NvU64 bytes_override NV_ALIGN_BYTES(8);// IN，0 表示使用 registry length
} UVM_EXPERT_RESERVED_PREFETCH_ENTRY;
```

`pool_id` 是 reserved backing namespace。`layer_id/expert_id/role` 可以从 region registry 推导，第一版不必重复放入 prefetch entry。

Phase 2 prefetch params：

```c
typedef struct
{
    NvU32 mode;
    NvU32 count;
    NvProcessorUuid gpu_uuid;
    NvBool async;
    NvU32 success_count;
    NvU32 failure_count;
    NvU64 requested_bytes;
    NvU64 accepted_bytes;
    UVM_EXPERT_RESERVED_PREFETCH_ENTRY entries[MAX];
    NV_STATUS rmStatus;
} UVM_EXPERT_RESERVED_PREFETCH_BATCH_PARAMS;
```

Phase 2 mode 建议：

| mode | 语义 |
|---:|---|
| 0 | dry_run / validate only |
| 1 | resolve reserved backing assignment only |
| 2 | invalidate old owner metadata only |
| 3 | reserved-backed prefetch / read-duplicate |
| 4 | install ordinary UVM PTE only，要求 reserved backing 已填好并已更新 metadata |

Phase 1 已实现的是 pool control，不执行 region prefetch。Phase 2 可以先只实现 mode 3，等验证后再拆细。

## 实施阶段

### Phase 0：明确 baseline 与接口约束

目标：明确 reserved-backed native residency 的 correctness contract，并把 fast prefetch 的 region 输入与 reserved backing assignment 对齐。

工作项：

1. 固化 region registry 到 reserved backing assignment 的用户态映射：

   ```text
   selected expert ids
     -> region ids
     -> reserved pool ids
     -> reserved slot ids
     -> reserved offsets
   ```

2. 明确 reserved backing pool sizing：

   ```text
   reserved_slot_count * reserved_slot_bytes = fixed GPU footprint
   ```

3. 定义 vLLM 侧策略：

   ```text
   KV cache pressure low:
     all experts GPU-resident
     selected experts 直接计算或普通 expert_fast_regions async

   KV cache pressure high:
     hot / high-priority experts 常驻 GPU
     cold / overflow experts CPU-offloaded
     only CPU-offloaded selected experts 使用 reserved-backed native residency path
   ```

4. 固化 placement policy 的输出：

   ```text
   expert_id -> placement:
     GPU_RESIDENT
     CPU_OFFLOADED_RESERVED
   ```

5. 加一个文档级 correctness contract：

   ```text
   reserved-backed path 只服务 CPU_OFFLOADED_RESERVED experts
   reserved backing 复用前需要 stream/event 同步
   旧 owner resident/read_duplicated metadata 必须显式失效
   strict 模式不允许 offloaded expert 触发 pool 外 native on-demand migration
   GPU_RESIDENT experts 不受 reserved pool 约束
   ```

交付物：

- 设计文档。
- 用户态 reserved backing assignment helper 的接口草案。
- Phase 0 helper：

  ```text
  workloads/vllm/vllm/vllm/device_allocator/uvm_expert_placement.py
  ```

- Phase 0 单元测试：

  ```text
  workloads/vllm/vllm/uvm_test/test_uvm_expert_placement.py
  ```

当前 helper 已覆盖：

```text
KV cache pressure low:
  selected experts -> GPU_RESIDENT
  不生成 reserved assignment

KV cache pressure high:
  仅 selected 且 CPU-offloaded experts -> CPU_OFFLOADED_RESERVED
  GPU_RESIDENT experts 不进入 reserved ioctl
  同一 expert 的多个 region/role 共用一个 reserved slot，并用 slot_offset 排布
  reserved slot 不足或单 expert 超过 slot_bytes 时进入 overflow
```

### Phase 1：reserved backing pool 与 PMM chunk 预留

目标：在 UVM driver 层建立可被普通 VA block residency 使用的固定 backing pool，但不在 PMM 内部新增 reserved flag 或 reserved free list。

已实现的 Phase 1 选择：

```text
uvm.c 外层 reserved pool cache
  -> 调用现有 uvm_pmm_gpu_alloc_user()
  -> 获得 TEMP_PINNED user chunks
  -> 不调用 uvm_pmm_gpu_unpin_allocated()
  -> 把 chunks 从普通 VA block ownership 之外保存在 reserved pool slot 中
  -> reset/release 时调用 uvm_pmm_gpu_free()
```

这满足“在现有 `uvm_pmm_gpu_alloc_user()` 外部做一层缓存管理”的要求。`TEMP_PINNED` chunk 已经不会挂在普通 PMM free/used list 上，也不会被 VA block eviction 当作普通 allocated residency chunk 扫描；因此 Phase 1 不需要 `UVM_PMM_GPU_CHUNK_FLAG_RESERVED`，也不需要在 PMM free list / eviction scan 热路径增加 reserved 判断。

已实现内核 ABI：

```text
UVM_EXPERT_RESERVED_POOL = UVM_IOCTL_BASE(98)

mode=query:
  查询 active/backed slots、reserved_backing_bytes、chunk_size、epoch、debug physical address

mode=prealloc:
  输入 pool_id / slot_id / slot_bytes
  默认使用 UVM_PMM_ALLOC_FLAGS_NONE
  entry flag ALLOW_EVICT 显式允许 UVM_PMM_ALLOC_FLAGS_EVICT

mode=reset:
  释放所有 TEMP_PINNED chunks
```

已实现代码落点：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h
  UVM_EXPERT_RESERVED_POOL_* ABI

kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
  g_uvm_expert_reserved_pool
  uvm_api_expert_reserved_pool()
  uvm_expert_reserved_pool_prealloc_entry()
  uvm_expert_reserved_pool_query_locked()
  uvm_expert_reserved_pool_reset_locked()

workloads/vllm/vllm/vllm/device_allocator/uvm.py
  driver_expert_reserved_pool_prealloc()
  driver_expert_reserved_pool_query()
  driver_expert_reserved_pool_reset()
```

当前边界：

```text
1. Phase 1 只预留 backing，不接 VA block，不安装 PTE，不更新 resident/read_duplicated metadata。
2. 重复 prealloc 同一 pool_id/slot_id 且已有 backing 足够大时幂等成功。
3. 同一 pool_id/slot_id 如果请求更大 slot 或不同 GPU，返回 SLOT_IN_USE；需要先 reset。
4. query 返回 first_chunk_address / last_chunk_address 仅用于 smoke/debug，不作为稳定用户 ABI。
5. owner region / old metadata invalidation / strict fault guard 属于 Phase 2/3。
```

最小 4MiB 双 slot smoke：

```python
from vllm.device_allocator.uvm import (
    driver_expert_reserved_pool_prealloc,
    driver_expert_reserved_pool_query,
    driver_expert_reserved_pool_reset,
)

slot_bytes = 2 * 1024 * 1024

prealloc = driver_expert_reserved_pool_prealloc([
    {"pool_id": 0, "slot_id": 0, "slot_bytes": slot_bytes},
    {"pool_id": 0, "slot_id": 1, "slot_bytes": slot_bytes},
], device=0)

query = driver_expert_reserved_pool_query()

assert prealloc["ok"]
assert query["reserved_backing_bytes"] == 4 * 1024 * 1024

driver_expert_reserved_pool_reset()
```

第一阶段仍可复用现有 `UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC` 的观测经验，但实现目标不是 slot-backed PTE，而是让这些 PMM chunks 后续作为普通 VA block GPU chunks 接入 residency。

### Phase 2：VA block populate 使用 reserved backing

目标：修改 UVM prefetch/read-duplicate 的 GPU chunk 分配路径，让 selected expert 的普通 residency backing 来自 reserved pool。

当前实现状态：

```text
Phase 2A 已实现：
  UVM_EXPERT_RESERVED_PREFETCH_BATCH ABI
  region_id -> reserved pool/slot/offset 的校验和 assignment
  reserved slot owner metadata 记录
  Python wrapper driver_expert_reserved_prefetch_regions()
  strict counters:
    ordinary_resident_bytes = 0
    pool_external_alloc_count = 0
    footprint_contract_broken = 0

Phase 2C 已实现为短期实验路径：
  mode=reserved_prefetch 会把 CPU-resident managed expert pages copy 到 reserved PMM chunks
  对原 managed VA 安装 readonly slot-backed GPU PTE
  复用 slot 前先卸载旧 owner 的 slot-backed PTE 并等待 VA block tracker
  invalidate/reset/release 会清理 slot-backed PTE 与 owner metadata
  不填 gpu_state->chunks[]
  不更新 ordinary resident/read_duplicated_pages metadata

后续 native residency 深改仍待实现：
  把 reserved chunk 安全接入 VA block gpu_state->chunks[]
  更新 ordinary resident/read_duplicated_pages metadata
  安装普通 UVM GPU PTE
  处理 reserved chunk 的 PMM lifecycle / eviction exclusion
```

注意：当前 `mode=reserved_prefetch` 不再走 unsupported guard，而是执行
“copy + readonly slot-backed PTE install”。它仍然不是普通 UVM residency：
reserved chunks 不进入 `gpu_state->chunks[]`，strict counters 中
`ordinary_resident_bytes` / `pool_external_alloc_count` /
`footprint_contract_broken` 仍应保持 0。

新增或改造函数：

```text
block_populate_gpu_chunk_reserved()
uvm_va_block_populate_pages_gpu_reserved()
uvm_va_block_expert_promote_read_duplicate_reserved_batch()
```

关键路径：

```text
region ids
  -> uvm_phase_f_expert_batch_collect_region()
  -> resolve registered expert region
  -> lock/revalidate VA blocks
  -> release old owner slot-backed PTE
  -> copy CPU managed pages into reserved PMM chunks
  -> install readonly slot-backed GPU PTE
  -> record reserved slot owner metadata
```

当前 Phase 2C 限制：

- 只读 expert 权重；GPU 写入/dirty/writeback 语义未实现。
- 源页必须已经 CPU-resident，且单个 VA block slice 内来自同一 CPU NUMA node。
- 不处理 HMM / confidential-computing / discarded page。
- 不接管 GPU fault recovery；strict 模式下不应让 offloaded expert fallback 到 pool 外 native migration。

#### 多 expert 复用同一个 reserved slot 的 smoke/benchmark

为了验证“多个 offloaded expert 轮流复用同一个固定 slot”是否成立，新增了
`workloads/vllm/vllm/uvm_test/expert_reserved_prefetch_smoke.cu`。该程序支持：

```bash
/tmp/expert_reserved_prefetch_smoke \
  --experts 4 \
  --repeats 5 \
  --bytes 2097152 \
  --native-baseline
```

实验流程：

```text
1. cudaMallocManaged 分配 N 个 expert，每个 expert 2MiB。
2. CPU 为每个 expert 写入不同 pattern。
3. 注册 N 个 Phase-F expert region。
4. 只 PREALLOC 一个 2MiB reserved slot。
5. 对每个 expert 轮流调用 mode=reserved_prefetch，全部使用同一个 pool=0/slot=0。
6. 每次 remap 后立即启动 GPU kernel，从该 expert 的原 managed VA 读数据并校验 pattern。
7. 下一次 remap 会自动清旧 owner 的 slot-backed PTE，等待 VA block tracker，然后把新 expert copy 到同一 slot 并安装新 PTE。
8. 最后调用 invalidate 清除当前 slot-backed mapping。
```

实测结果，RTX 5090，4 experts，每个 2MiB，重复 5 轮：

```text
slot_reuse_begin experts=4 repeats=5 bytes=2097152 slot_count=1 slot_bytes=2097152 gpu=NVIDIA GeForce RTX 5090
reserved_slot_reuse_prefetch samples=20 bytes=2097152 avg_ms=0.161 p50_ms=0.152 p95_ms=0.199 min_ms=0.149 max_ms=0.208 avg_GiBps=12.168
slot_reuse_correctness experts=4 repeats=5 remaps=20 kernel_errors=0
invalidate success=1 invalidated=1 entry_status=0
```

带 `cudaMemPrefetchAsync` baseline 的同条件粗略对比：

```text
native_prefetch_baseline samples=20 bytes=2097152 avg_ms=0.365 p50_ms=0.359 p95_ms=0.396 min_ms=0.343 max_ms=0.445 avg_GiBps=5.356
reserved_slot_reuse_prefetch samples=20 bytes=2097152 avg_ms=0.156 p50_ms=0.153 p95_ms=0.178 min_ms=0.143 max_ms=0.194 avg_GiBps=12.507
slot_reuse_correctness experts=4 repeats=5 remaps=20 kernel_errors=0
```

该实验说明：

- 多个 expert 可以轮流复用同一个 reserved slot。
- 正确性由每次 remap 后的 GPU kernel 读原 managed VA 校验证明，20 次 remap 均为 `kernel_errors=0`。
- 当前小规模 2MiB 实验中，reserved slot-backed path 的端到端 ioctl 时间低于 native `cudaMemPrefetchAsync` baseline。
- 该性能数字包含旧 PTE 清理、tracker wait、CPU->reserved chunk copy、新 PTE install；但它仍只是 smoke/微基准，真实 vLLM 性能还要看 expert size、batching、stream overlap、NUMA、KV cache 压力和调度策略。

关键原则：

```text
当前 Phase 2C：
  保留 managed VA 不变。
  使用 VA block PTE/tracker/lifetime 管理能力。
  不接入 ordinary resident/read_duplicated metadata。
  不填 gpu_state->chunks[]。

后续 native residency 深改：
  才保留 ordinary UVM resident metadata 并替换 allocation source。

共同要求：
  不要让普通 PMM 在 strict mode 下偷偷分配 pool 外 backing。
```

### Phase 3：strict fault / native migration guard 与 vLLM 可用性收敛

基于已经实现的 Phase 2C，原先 Phase 3 中“清旧 owner ordinary
resident/read_duplicated metadata”的设计不再适合作为下一步主线。原因是
Phase 2C 并没有把 reserved chunks 接入 `gpu_state->chunks[]`，也没有把
offloaded expert 标记成普通 GPU-resident。因此旧 owner 复用时不存在“普通
resident metadata 指向同一块 reserved backing”的问题。

Phase 2C 已经完成的旧 owner 处理是：

```text
unmap old owner slot-backed GPU PTE
clear old owner moe_slot_pte_bits / PTE bits
wait VA block tracker
remove old owner from reserved pool owner table
do not copy back to CPU
```

所以当前更合理的 Phase 3 目标是：**在 slot-backed 快速路径上补齐 strict
模式边界和 vLLM 接入所需的可观测性 / 防回退机制**，确保 CPU-offloaded
expert 不会在未调度 `reserved_prefetch` 时，被 GPU fault 或 native prefetch
悄悄迁移到 pool 外普通 GPU backing。

需要新增：

```text
CPU_OFFLOADED_RESERVED region strict 标记
reserved slot-backed owner query / debug counters
uvm_reserved_fault_policy for slot-backed mode
native prefetch / migrate guard for strict offloaded experts
fault_policy_violation_count
pool_external_alloc_count enforcement
```

当前 Phase 3S 已实现的部分：

- 新增 `UVM_PHASE_N_REGION_FLAG_CPU_OFFLOADED_RESERVED`，vLLM/用户态可以把
  被 KV cache 压力挤到 CPU 的 selected expert 显式标成
  `CPU_OFFLOADED_RESERVED`。
- GPU replayable fault dispatch 在发现该标记后，不再进入普通
  `service_fault_batch_block()`，而是把 fault 标为 fatal/cancel，并增加
  `num_phase_n_reserved_fault_policy_violations` 与
  `num_phase_n_reserved_pool_external_alloc_attempts`。
- `UVM_MIGRATE` / native prefetch-to-GPU 路径在进入普通 `uvm_migrate()` 前扫描
  请求范围；只要覆盖到 `CPU_OFFLOADED_RESERVED` expert，就返回
  `NV_ERR_INVALID_ACCESS_TYPE`，并增加
  `num_phase_n_reserved_migrate_policy_violations` 与
  `num_phase_n_reserved_pool_external_alloc_attempts`。
- 普通 `GPU_RESIDENT` expert 和没有该标记的 expert 不受影响，可以继续走 native
  residency、fast prefetch 或普通 fault recovery。

注意：该 guard 依赖 Phase-N range tagging 生效。也就是说，region 注册后必须能把
`flags` 写入 managed range 的 `phase_n_tag`，否则 fault/migrate 路径无法识别
`CPU_OFFLOADED_RESERVED`。

Phase 3S：slot-backed 快速路径的 strict guard：

```text
if region placement == CPU_OFFLOADED_RESERVED and strict_footprint:
  allowed:
    mode=reserved_prefetch installs slot-backed readonly PTE
    mode=invalidate clears slot-backed PTE
    explicit CPU residency / CPU source backing remains valid

  rejected or counted as violation:
    native GPU fault that would allocate ordinary GPU backing
    native cudaMemPrefetchAsync / UVM migrate-to-GPU for that region
    write/atomic GPU access to readonly slot-backed expert
    best-effort fallback that allocates pool outside reserved slots
```

fault / migration policy：

```text
strict fixed footprint:
  GPU fault on expert region without assigned reserved backing -> return fault/kill/error or block until user schedules prefetch
  native prefetch/migrate to GPU for expert region -> reject unless reserved assignment is provided

best effort:
  allow native UVM fallback, but mark footprint_contract_broken
```

Phase 3N：native residency 深改，暂不作为当前下一步：

```text
uvm_va_block_reserved_owner_invalidate()
uvm_reserved_pool_reassign_slot()
block_populate_gpu_chunk_reserved()
uvm_va_block_populate_pages_gpu_reserved()
uvm_va_block_expert_promote_read_duplicate_reserved_batch()

reuse reserved backing 前：
  unmap old owner ordinary GPU PTE
  clear old owner resident GPU mask
  clear old owner read_duplicated_pages
  detach reserved chunk from old gpu_state->chunks[]
  attach reserved chunk to new gpu_state->chunks[]
```

只有选择 Phase 3N 时，才需要清 ordinary resident/read_duplicated metadata。
在当前 Phase 2C/Phase 3S 路线中，这些 metadata 本来没有被设置，应继续保持
不触碰。

### Phase 4：batch 化、async 化和 vLLM 接入

目标：把 reserved-backed path 做成 `expert_fast_regions async` 的同级 executor。

新增 executor：

```text
VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async
```

调度流程：

```text
FusedMoE.select_experts()
  -> topk_ids
  -> placement policy 查询 expert state
  -> split selected experts:
       GPU_RESIDENT experts
       CPU_OFFLOADED_RESERVED experts
  -> only CPU_OFFLOADED_RESERVED expert region ids
  -> reserved allocator assigns fixed slots for offloaded experts
  -> driver_expert_reserved_prefetch_regions(offloaded_region_ids, slots, async=True)
  -> later MoE compute uses original expert pointers
```

reserved allocator 需要维护：

```text
expert_id -> placement state
slot_id -> current region_id / layer / expert / role
slot_id -> last CUDA event using this backing
free reserved slots
pending copy/map slots
pool policy: strict / best_effort
```

placement policy 由 KV cache 压力驱动：

```text
KV cache 低水位:
  尽量把 expert 全部或大部分保留在 GPU
  reserved path 可为空

KV cache 高水位:
  释放部分 expert 的 GPU residency 给 KV cache
  被释放/卸载的 expert 标记为 CPU_OFFLOADED_RESERVED
  后续每次被 router 选中时通过 reserved pool 周期性搬入
```

复用规则：

```text
如果 reserved slot 仍被上一轮 compute 使用:
  等待对应 CUDA event

否则:
  invalidate old owner UVM residency metadata
  copy new region into same reserved backing
  install ordinary UVM PTE for new owner
```

async 语义：

```text
async=false:
  ioctl 等 copy + ordinary UVM map tracker 完成。

async=true:
  ioctl 提交 copy + map push 后返回。
  map push 必须 acquire copy tracker。
  用户态在 compute 前通过 event/query/touch 或 stream 同步保证完成。
```

### Phase 5：测试与性能评估

#### Correctness smoke

最小测试：

```text
模拟 KV cache 高压：
  6 个 expert 全部标记为 CPU_OFFLOADED_RESERVED

6 个 2MiB expert region
2 个 2MiB reserved slots
round0: expert 1,2 -> reserved slot 0,1
round1: expert 3,4 -> reserved slot 0,1
round2: expert 5,6 -> reserved slot 0,1
```

检查：

```text
GPU readback matches selected expert payload
reserved_backing_bytes == 4MiB
ordinary UVM GPU PTE installed for current owners
current owners resident/read_duplicated metadata correct
old owners no longer GPU-resident on reused backing
pool外 GPU resident bytes == 0 in strict mode
```

#### Multi-expert benchmark

扩展已有：

```text
workloads/vllm/bench_uvm_expert_batch_prefetch.py
```

新增 case：

```text
expert_reserved_regions_sync
expert_reserved_regions_async
```

对比：

```text
cuda_prefetch_serial
expert_fast_regions async
expert_reserved_regions async
ordinary H2D copy
```

矩阵：

```text
10 x 2MiB
10 x 4MiB
10 x 6MiB
placement:
  all_gpu_resident
  mixed_hot_gpu_cold_offloaded
  mostly_offloaded_under_kv_pressure
reserved_slot_count = 2 / 4 / 8 / 10
reserved_budget = reserved_slot_count * expert_size
```

必须同时报告：

```text
p50 submit ms
p50 complete ms
effective GiB/s
reserved_backing_bytes
ordinary UVM GPU resident bytes
reserved_reuse_count
pool_external_alloc_count
fault_policy_violation_count
gpu_resident_expert_count
cpu_offloaded_expert_count
```

成功标准不是只看速度，还要看：

```text
reserved_backing_bytes 被固定预算限制住
CPU-offloaded experts 的 GPU residency 只指向 reserved pool 内 backing
GPU-resident experts 可以保留普通 GPU residency
strict 模式下 offloaded experts 不产生 pool 外 GPU resident bytes
```

## 推荐实现顺序

### Milestone A：reserved backing pool

实现：

- reserved backing pool metadata。
- PMM reserved chunk prealloc。
- reserved pool query/reset。

目标：

```text
固定 GPU backing 预算正确
reserved chunks 不进入普通 PMM free/evict 路径
可以按 pool_id/slot_id 查询 owner/epoch/state
```

不以性能为主要目标。

### Milestone B：reserved region ioctl

实现：

- 新 ioctl `UVM_EXPERT_RESERVED_PREFETCH_BATCH`。
- 新用户态 wrapper `driver_expert_reserved_prefetch_regions()`。
- 内核中复用 `phase_f_regions` registry。
- ABI 表达 `region_id -> pool_id/slot_id/offset`。

目标：

```text
用户态不再需要 region_id -> base/length 的重复展开
ABI 能表达 region_id -> reserved backing assignment
```

### Milestone C：Phase 2C slot-backed reserved prefetch

实现：

- 新 helper `uvm_va_block_copy_and_map_reserved_readonly()`。
- CPU managed pages 通过 CE copy 到 reserved PMM chunks。
- 原 managed VA 安装 readonly slot-backed GPU PTE。
- `gpu_state->moe_slot_pte_bits` 标记特殊 PTE。
- 不填 `gpu_state->chunks[]`。
- 不更新 ordinary resident/read_duplicated metadata。

目标：

```text
保持 expert managed VA 不变
GPU backing 固定在 reserved pool 内
多个 offloaded expert 可轮流复用固定 slot
strict counters 保持 ordinary/external/broken 为 0
```

### Milestone D：slot-backed strict guard + fault policy

实现：

- reserved slot reuse 前清旧 owner slot-backed PTE。
- release/invalidate/reset 清 owner table 和 slot-backed PTE。
- strict 模式拦截 CPU_OFFLOADED_RESERVED expert 的 pool 外 native GPU migration/fault allocation。
- 已实现 strict native migrate-to-GPU guard，命中时返回 `NV_ERR_INVALID_ACCESS_TYPE`。
- 已实现 strict GPU fault guard，命中时 fatal/cancel fault，避免进入普通 UVM populate。
- 已暴露 per-GPU proc stats 计数：
  `phase_n_reserved_fault_policy_violations`、
  `phase_n_reserved_migrate_policy_violations`、
  `phase_n_reserved_pool_external_alloc_attempts`。
- best-effort 模式允许 fallback 并标记 `footprint_contract_broken` 仍是后续扩展。

目标：

```text
旧 owner 不再引用被复用 backing
strict fixed-footprint 不被 UVM on-demand fault 打破
vLLM 调度层能够区分 GPU_RESIDENT 和 CPU_OFFLOADED_RESERVED expert
```

### Milestone D-N：native residency 深改，可选后续

实现：

- reserved chunk 安全挂入 `gpu_state->chunks[]`。
- 普通 UVM resident/read_duplicated metadata 与 reserved owner 一致。
- 拦截所有释放 / eviction / migration 路径，reserved chunk 只能回 reserved pool。

目标：

```text
普通 UVM residency 使用指定 backing
在 native UVM metadata 下保持 fixed footprint
```

### Milestone E：vLLM E2E

实现：

- `VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async`
- fixed reserved window config。
- reserved backing event/fence 管理。
- fallback 到 `expert_fast_regions async`。

目标：

```text
在真实 MoE decode 中验证：
  GPU memory footprint 受控
  throughput/TPOT 不因 reserved-backed path 大幅退化
  无 D-state / release 残留 / PTE failure
```

## 代码落点与修改清单

### 1. Kernel ABI：`uvm_ioctl.h`

文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h
```

新增内容：

```text
Phase 1 已实现：
  UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
  UVM_EXPERT_RESERVED_POOL_MODE
  UVM_EXPERT_RESERVED_POOL_ENTRY
  UVM_EXPERT_RESERVED_POOL_PARAMS
  UVM_EXPERT_RESERVED_POOL = UVM_IOCTL_BASE(98)

Phase 2 待新增：
  UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES
  UVM_EXPERT_RESERVED_PREFETCH_MODE
  UVM_EXPERT_RESERVED_PREFETCH_ENTRY
  UVM_EXPERT_RESERVED_PREFETCH_BATCH_PARAMS
  UVM_EXPERT_RESERVED_PREFETCH_BATCH
```

Phase 1 pool entry 已落地为 slot 级别：

```c
typedef struct
{
    NvU32 pool_id;
    NvU32 slot_id;
    NvU32 flags;
    NvU32 status;
    NvU32 state;
    NvU32 chunk_count;
    NvU64 slot_bytes NV_ALIGN_BYTES(8);
    NvU64 physical_bytes NV_ALIGN_BYTES(8);
    NvU64 epoch NV_ALIGN_BYTES(8);
    NvU64 first_chunk_address NV_ALIGN_BYTES(8);
    NvU64 last_chunk_address NV_ALIGN_BYTES(8);
} UVM_EXPERT_RESERVED_POOL_ENTRY;
```

Phase 2 prefetch entry 建议只表达最小必要信息：

```c
typedef struct
{
    NvU64 region_id NV_ALIGN_BYTES(8);     // IN
    NvU32 pool_id;                         // IN
    NvU32 slot_id;                         // IN，reserved backing slot
    NvU32 flags;                           // IN
    NvU32 status;                          // OUT
    NvU64 slot_offset NV_ALIGN_BYTES(8);   // IN
    NvU64 bytes_override NV_ALIGN_BYTES(8);// IN，0 表示使用 registry length
} UVM_EXPERT_RESERVED_PREFETCH_ENTRY;
```

`pool_id` 是 reserved backing namespace。`layer_id/expert_id/role` 可以从 region registry 推导，第一版不必重复放入 entry。

### 2. Kernel route：`uvm.c`

文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
```

Phase 1 已新增 API handler：

```c
static NV_STATUS uvm_api_expert_reserved_pool(
    UVM_EXPERT_RESERVED_POOL_PARAMS *params,
    struct file *filp);
```

并在 ioctl route 表里注册：

```c
UVM_ROUTE_CMD_HEAP_INIT_CHECK(UVM_EXPERT_RESERVED_POOL,
                              uvm_api_expert_reserved_pool);
```

Phase 2 待新增 API handler：

```c
static NV_STATUS uvm_api_expert_reserved_prefetch_batch(
    UVM_EXPERT_RESERVED_PREFETCH_BATCH_PARAMS *params,
    struct file *filp);
```

并在 ioctl route 表里注册：

```c
UVM_ROUTE_CMD_HEAP_INIT_CHECK(UVM_EXPERT_RESERVED_PREFETCH_BATCH,
                               uvm_api_expert_reserved_prefetch_batch);
```

Phase 1 handler 的职责：

```text
1. 校验 mode / count / gpu_uuid / pool_id / slot_id / slot_bytes。
2. PREALLOC：调用 uvm_pmm_gpu_alloc_user() 分配 UVM_CHUNK_SIZE_MAX chunks。
3. 保持 chunks 为 TEMP_PINNED，并记录到 g_uvm_expert_reserved_pool。
4. QUERY：返回 active/backed slots、reserved_backing_bytes、epoch、chunk debug address。
5. RESET：调用 uvm_pmm_gpu_free() 释放 reserved chunks。
```

Phase 2 handler 的职责：

```text
1. 校验 count / mode / gpu_uuid / pool_id / slot_id / slot_offset / bytes_override。
2. 通过 va_space->phase_f_regions 查 region_id。
3. Phase 2A：先完成 region -> reserved slot assignment，不触发普通 PMM fallback。
4. Phase 2C：mode=reserved_prefetch 时按 VA block slice 遍历 registered expert region。
5. Phase 2C：复用 reserved slot assignment，为每个 slice 计算 slot_offset。
6. Phase 2C：调用 reserved slot-backed VA block helper 完成 CPU->reserved chunk copy 和 readonly PTE install。
7. Phase 2C：mode=invalidate/reset/release 时卸载 slot-backed PTE 并清 owner metadata。
```

核心路径：

```text
region ids
  -> va_space->phase_f_regions lookup
  -> reserved pool slot lookup
  -> uvm_va_block_find_create* per VA block slice
  -> uvm_va_block_unmap(gpu)
  -> uvm_va_block_copy_and_map_reserved_readonly()
  -> readonly slot-backed GPU PTE
```

注意：这条 Phase 2C 实验路径刻意没有调用普通 read-duplicate populate，
因此不会分配 pool 外 GPU resident backing。后续要切换到 native residency
深改时，再把 `uvm_va_block_expert_promote_read_duplicate_reserved_batch()`
或等价 helper 接入普通 residency metadata。

### 3. Reserved pool / PMM：`uvm_pmm_gpu.h/c` 与 `uvm.c`

文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_pmm_gpu.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_pmm_gpu.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
```

Phase 1 当前不新增 PMM API，不实现 `uvm_pmm_gpu_alloc_user_reserved()`。实现方式是在 `uvm.c` 中外包一层 reserved pool allocator：

```c
uvm_pmm_gpu_alloc_user(&gpu->pmm,
                       chunk_count,
                       UVM_CHUNK_SIZE_MAX,
                       flags,
                       chunks,
                       NULL);
```

关键是不让 reserved chunks 被普通 allocation/free/evict 误用。这里依赖 PMM 原生语义：`uvm_pmm_gpu_alloc_user()` 返回的 user chunks 是 `TEMP_PINNED`，未 unpin 前不进入普通 allocated residency 链路。

reserved pool metadata 至少包含：

```text
pool_id
slot_id
gpu
slot_bytes
chunk list
owner_va_space
owner_region_id
owner_va_base / owner_bytes
owner_va_block refs
epoch
state: free/loading/resident/in_use/error
```

### 4. Reserved-backed VA block helper：`uvm_va_block.h/c`

文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
```

新增声明：

```c
typedef struct
{
    uvm_gpu_chunk_t **chunks;
    size_t chunk_count;
    NvU64 backing_offset;
    NvU64 bytes;
    NvU32 pool_id;
    NvU32 slot_id;
    NvU64 epoch;
} uvm_expert_reserved_assignment_t;

NV_STATUS uvm_va_block_expert_promote_read_duplicate_reserved_batch(
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_gpu_t *dest_gpu,
    uvm_va_block_expert_batch_entry_t *entries,
    const uvm_expert_reserved_assignment_t *assignments,
    size_t entry_count,
    uvm_tracker_t *out_tracker,
    uvm_processor_mask_t *involved_processors);
```

实现上参考 `uvm_va_block_expert_promote_read_duplicate_batch()` 的阶段划分：

```text
prepare/revalidate
reserved chunk attach/populate
copy resident CPU pages to reserved GPU chunks
update resident/read_duplicated_pages
ordinary UVM PTE map
tracker aggregation
```

必须替换或旁路：

```text
block_populate_gpu_chunk() 里的普通 uvm_pmm_gpu_alloc_user()
```

但不要跳过普通 metadata update。路线 B 的目标就是让 UVM 认为当前 expert 真实 GPU-resident，只是 backing 来自 reserved pool。

### 5. Fault / migration guard

需要检查并改造这些入口：

```text
GPU fault service path
native cudaMemPrefetchAsync / UVM migrate path
access counter migration path
VA block eviction path
```

strict fixed-footprint 模式下：

```text
expert managed VA 如果未绑定 reserved backing：
  不允许自动分配普通 GPU resident backing

expert managed VA 如果 fault 到旧 owner：
  清旧 metadata 后返回错误或要求用户重新 prefetch

eviction：
  reserved chunks 不参与普通 PMM eviction
  owner metadata 只能通过 reserved pool reuse/release 路径失效
```

### 6. Python UVM wrapper：`uvm.py`

文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm.py
```

新增：

```text
Phase 1 已实现：
UVM_EXPERT_RESERVED_POOL_* ctypes struct
driver_expert_reserved_pool_prealloc()
driver_expert_reserved_pool_query()
driver_expert_reserved_pool_reset()

Phase 2 待实现：
UVM_EXPERT_RESERVED_PREFETCH_* ctypes struct
_driver_expert_reserved_prefetch_regions_direct_ioctl()
driver_expert_reserved_prefetch_regions()
```

当前代码状态：

```text
Phase 2A/2C 已实现：
UVM_EXPERT_RESERVED_PREFETCH_* ctypes struct
driver_expert_reserved_prefetch_regions()
mode=reserved_prefetch -> copy CPU pages into reserved PMM chunks
mode=reserved_prefetch -> install readonly slot-backed GPU PTE
mode=invalidate -> uninstall slot-backed GPU PTE and clear owner metadata

注意：
driver_expert_reserved_prefetch_regions() 默认 prefer_existing=True，
因为 Phase 2 handler 需要复用 CUDA/vLLM 已注册 GPU 的 UVM fd / va_space。
```

建议 wrapper 输入：

```python
driver_expert_reserved_prefetch_regions(
    region_ids: Sequence[int],
    slot_ids: Sequence[int],
    *,
    pool_id: int,
    slot_offsets: Sequence[int] | None = None,
    device: int = 0,
    async_: bool = True,
    strict_footprint: bool = True,
) -> dict[str, object]
```

返回 dict 至少包含：

```text
ok
rm_status
ioctl_ret
success_count / failure_count
requested_bytes / accepted_bytes
reserved_backing_bytes
ordinary_resident_bytes
pool_external_alloc_count
footprint_contract_broken
```

`strict_footprint=True` 时，如果内核 reserved-backed fast path 失败，不自动 fallback 到 `driver_expert_read_dup_fast_regions()`。

### 7. vLLM 调度：`fused_moe/layer.py`

文件：

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
```

扩展 executor 白名单：

```text
cpp_batch
driver_batch
auto
expert_fast_regions_async
expert_reserved_regions_async
```

新增 reserved allocator 状态：

```text
slot_count
slot_bytes
slot_id -> current_region_id
slot_id -> current_layer/expert/role
slot_id -> last_use_cuda_event
slot_id -> pending_prefetch_event
strict_footprint
```

调度伪代码：

```text
selected_experts = router/topk result
region_ids = lookup registered expert regions
slot_ids = assign fixed reusable reserved slots

for each slot to reuse:
    wait slot.last_use_cuda_event

driver_expert_reserved_prefetch_regions(region_ids, slot_ids, async=True)

record pending_prefetch_event
before compute:
    wait pending_prefetch_event if ioctl async completion is exposed to CUDA
    otherwise keep conservative sync/query barrier

launch MoE compute with original expert pointers
record last_use_cuda_event for each slot
```

第一版如果没有 stream-ordered completion event，可以先保守使用同步 ioctl 或 driver query 等待 completion；等内核 tracker/用户态 event 语义稳定后再开启真正 overlap。

### 8. 测试与 benchmark 文件

新增或扩展：

```text
workloads/vllm/check_expert_reserved_prefetch.py
workloads/vllm/check_expert_reserved_reuse.py
workloads/vllm/bench_uvm_expert_batch_prefetch.py
workloads/vllm/bench_uvm_expert_batch_pipeline.py
```

测试顺序：

```text
1. 6 x 2MiB expert，2 x 2MiB reserved slot，三轮复用 correctness。
2. 10 x 2MiB/4MiB/6MiB expert，多 reserved_slot_count sweep。
3. async=false 与 async=true 对比。
4. strict_footprint 失败路径测试。
5. vLLM tiny MoE E2E。
```

每次结果必须同时记录：

```text
latency / bandwidth
reserved_backing_bytes
active_reserved_slots / backed_reserved_slots
ordinary UVM GPU resident bytes
pool_external_alloc_count
fault_policy_violation_count
metadata_invalidate_count
```

## 关键风险

### 1. Phase 2C 路线中不要误用 ordinary residency metadata

当前 slot-backed 路线没有把 reserved chunks 接入 `gpu_state->chunks[]`，因此
ordinary `resident/read_duplicated_pages` 不应该被设置。如果某条路径误把
slot-backed PTE 当成普通 GPU resident mapping，就可能触发错误的 migration、
eviction 或 writeback 判断。

当前复用 reserved backing 前必须处理旧 owner：

```text
unmap old owner slot-backed GPU PTE
clear old owner moe_slot_pte_bits / GPU PTE bits
clear or update reserved owner table
invalidate TLB
wait VA block tracker
```

### 1N. native residency 深改时，普通 UVM metadata 必须和 reserved owner 一致

如果后续选择 Phase 3N / native residency 路线，才需要恢复下面这组更重的
metadata invariant：

```text
unmap old owner ordinary GPU PTE
clear old owner resident GPU mask
clear old owner read_duplicated_pages
clear or update reserved owner table
invalidate TLB
```

否则 UVM 会认为旧 owner 仍然 GPU-resident，后续 fault/map/evict 可能把同一块 reserved backing 同时当成多个 VA block 的有效 resident copy。

### 2. UVM 只作为元数据管理层时的边界

如果完整 native residency 语义和固定 footprint 冲突，可以只使用 UVM 的元数据和同步能力：

```text
允许使用：
  VA space / VA range / VA block lookup
  region registry
  block lock / retry
  page table write / TLB invalidate
  tracker ordering
  mmu notifier / VA-space release cleanup
  reserved owner/epoch metadata

可关闭或改写：
  native on-demand migrate / prefetch 数据路径
  pool 外普通 GPU resident page populate
  GPU fault 后自动分配普通 PMM backing
  普通 eviction 对 reserved chunks 的回收
```

如果 GPU 访问了未绑定 reserved backing 的 **CPU-offloaded expert VA**，strict fixed-footprint 模式应把它视为 contract 违例，不能静默触发 native migration 到 pool 外 backing。常驻 GPU expert 不受这个限制。

### 3. copy 与 slot-backed PTE install 的顺序

如果 async=true，必须保证：

```text
copy to reserved backing 完成
  happens-before
slot-backed readonly GPU PTE 对 compute 可见
```

否则 GPU 可能通过原 expert VA 读到半写入数据。

### 4. reserved backing 复用与用户 kernel 并发

driver 不能自动知道用户 kernel 是否还在读旧 reserved backing。vLLM 侧必须有：

```text
reserved slot last-use event
reserved slot reuse waits event
```

### 5. fallback 语义

如果 CPU-offloaded expert 的 reserved-backed path 因 source 非 CPU resident、HMM、跨 NUMA、reserved slot 不足、PTE 状态复杂而失败，不能悄悄 fallback 到普通 expert_fast_regions，因为那会破坏固定 GPU footprint 目标。GPU-resident expert 可以继续使用普通 residency 或普通 fast prefetch。

建议：

```text
fixed footprint strict mode:
  reserved-backed path 失败则返回错误

best effort mode:
  可 fallback 到 expert_fast_regions，但必须在结果里标记 footprint contract broken
```

## 最小可行路径

最短路径是：

```text
Phase 1 reserved backing pool / PMM chunk 预留
  -> 固定 GPU backing 预算

Phase 2C reserved slot-backed readonly prefetch
  -> 原 managed VA 指向 fixed reserved backing

Phase 3S strict fault/native migration guard
  -> 阻止 CPU_OFFLOADED_RESERVED expert 通过 native fault/prefetch 分配 pool 外 backing

Phase 4 async/batch + vLLM executor
  -> 与 compute overlap 并做 E2E 验证
```

不建议下一步立刻转向完整 native residency。Phase 2C 已经证明 slot-backed
固定窗口复用具备正确性和性能优势，下一步优先补 strict guard、vLLM executor、
批量 remap 和 stream/event fence。等 E2E 收敛后，再决定是否需要 Phase 3N 的
`gpu_state->chunks[]` native residency 深改。
