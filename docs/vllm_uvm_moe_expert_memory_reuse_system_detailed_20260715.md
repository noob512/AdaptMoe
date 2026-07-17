# 面向 MoE 推理的 vLLM/UVM 专家权重显存复用系统详解

日期：2026-07-15

本文档总结当前仓库中面向 MoE 推理的 vLLM/UVM 专家权重显存复用系统。它不是只记录某一次实验，而是把项目要解决的问题、vLLM 接入点、NVIDIA UVM driver 改造、PMM 显存 backing 管理、KV cache 动态扩容、fault guard 验证方法和实验效果串成一份完整说明。

相关实验记录可继续参考：

```text
docs/vllm_uvm_512k_static_cold8_reserved_results_20260713.md
docs/vllm_sharegpt_cold8_512k_restore_detail_results_20260713.md
docs/vllm_zero_hit_runtime_check_ablation_20260714.md
docs/vllm_pre_service_expert_release_kv_grow_results_20260714.md
docs/vllm_pre_service_kv_grow_replicate_long_context_detailed_20260714.md
docs/vllm_pre_service_kv_grow_all_managed_fault_observe_results_20260714.md
workloads/vllm/vllm/uvm_test/LOWTRACE_FOURWAY_EXPERIMENT.md
```

## 1. 项目一句话概括

本项目在 vLLM 的 FusedMoE 执行路径和 NVIDIA UVM kernel driver 之间建立了一条可控的显存复用路径：

```text
冷门 MoE expert 权重:
  保持原 managed VA 不变
  从 GPU residency 释放到 CPU
  释放出的 GPU PMM backing 可被复用

运行时若该 expert 被 router 选中:
  将 CPU-resident expert pages 拷贝到固定 reserved VIDMEM slot
  在原 expert managed VA 上安装 readonly slot-backed GPU PTE
  Triton/FusedMoE kernel 仍然使用原来的 expert pointer

KV cache 压力变大时:
  在请求前安全阶段卸载不会被使用的 expert
  把更多 KV cache blocks 搬到 GPU
  扩大 vLLM allocator 可见 block 范围
  请求阶段验证没有 GPU-side managed page fault
```

更短地说：**expert 的虚拟地址保持稳定，但物理 GPU backing 可以从“常驻专家权重”转化为“可复用 slot 或更多 KV cache block”。**

## 2. 为什么要做这个系统

### 2.1 MoE 推理的显存结构

以本项目主要实验使用的 `Qwen/Qwen3-30B-A3B-FP8` 为例，MoE 推理中显存主要被几类对象占用：

```text
常驻模型权重:
  embedding
  attention weights
  norm
  router/gate
  lm_head
  MoE expert weights

运行期状态:
  KV cache
  attention metadata
  sampler/logits buffer
  Triton/cuBLAS/workspace
  temporary tensors
```

MoE 的关键特点是：模型拥有大量 expert，但每个 token 每层只会路由到其中一小部分 expert。也就是说，**expert 总参数量很大，但单步实际使用的 expert 子集很小**。

这带来一个很自然的问题：

```text
如果某些 expert 在当前请求集合里几乎不会被访问，
是否可以把它们从 GPU 显存中移走，
把显存留给更大的 KV cache？
```

### 2.2 KV cache 与长上下文压力

KV cache 与上下文长度近似线性相关。vLLM 使用 paged KV cache，把 KV cache 切成 block 并由 allocator 管理。长 prompt 或多请求并发会显著增加 KV block 需求。

在普通 cudaMalloc 路径下，本项目的容量扫测显示：

```text
all experts resident + 不启用 UVM/offload + 普通 cudaMalloc:
  默认自动 KV 参数:
    Available KV cache memory 为负，初始化失败

  固定 --kv-cache-memory-bytes:
    1 GiB 成功
    2 GiB / 2.5 GiB / 3 GiB 在实际分配 KV tensor 时 CUDA OOM
```

这说明如果所有专家都常驻 GPU，普通分配路径很难再容纳更大的 KV cache。

### 2.3 为什么不能只依赖 UVM 自动迁移

UVM managed memory 的默认能力是：CPU 和 GPU 共享一个虚拟地址，GPU 访问不在本地的页时触发 replayable page fault，由驱动迁移页面并重放指令。

这对实验很方便，但对高性能推理不够可控：

```text
自动 fault/migration 的发生点由 GPU 实际访问决定；
prefill/decode 热路径里触发 fault 会造成不可预测延迟；
KV cache 是 writable 数据，GPU 写 CPU-resident KV page 风险更大；
大量 fault 可能导致 GPU/UVM/CUDA 同步路径卡住；
fault 发生后再迁移，已经晚于我们希望的调度点。
```

因此本项目的目标不是“让 UVM 自动兜底”，而是：

```text
请求开始前，把本次请求会用到的 expert/KV 都准备成 GPU-resident；
请求开始后，理想情况下 expert 和 KV 都不应该产生 GPU-side UVM fault。
```

后续 all-managed observe 实验也正是用这个原则来验证：请求开始后捕捉所有 managed range 的 GPU fault，结果为 0。

## 3. 系统设计目标

本项目有几个核心设计目标。

### 3.1 Expert managed VA 必须稳定

FusedMoE/Triton kernel 已经持有 expert weight tensor 的地址。生产路径不希望在运行时改 PyTorch tensor、改 parameter 对象、或让 kernel 感知“这个 expert 在 CPU 还是 GPU”。

因此 expert 的虚拟地址必须保持不变：

```text
original expert pointer:
  vLLM parameter tensor data_ptr()
  Triton/FusedMoE kernel argument
  Phase-F registered region base

restore 后:
  kernel 仍访问 original expert pointer
  只是该 VA 的 GPU PTE 指向 reserved slot backing
```

### 3.2 Expert restore 的 GPU footprint 要固定

如果每次命中冷 expert 都让 UVM 普通迁移机制重新申请 GPU backing，那么 expert staging footprint 会随访问模式波动，容易挤压 KV cache。

所以引入 reserved VIDMEM slot pool：

```text
reserved_slot_count * reserved_slot_bytes = 固定 GPU staging footprint
```

例如当前 512KiB 路线：

```text
slot_bytes = 512 KiB
slot_count = 128
reserved_backing_bytes = 64 MiB
```

无论 runtime 有多少冷 expert 被访问，能够同时映射的 slot backing 都被限制在这个固定窗口内。

### 3.3 Expert 是只读数据，KV cache 是可写数据

这是系统里最重要的区分之一。

Expert weight：

```text
推理期间只读
可以从 CPU copy 到 slot
可以给原 VA 安装 readonly PTE
不需要把写回 CPU 当作正确性路径
```

KV cache：

```text
推理期间持续写入
不能用 readonly slot-backed PTE
不能让 GPU 在请求路径里写 CPU-resident page
必须在使用前真实 prefetch 回 GPU
并且只有确认 GPU resident 后才能暴露给 allocator
```

因此本项目对 expert 和 KV 使用两套不同机制：

```text
expert:
  reserved slot-backed readonly PTE remap

KV cache:
  block-level cudaMemPrefetchAsync CPU/GPU migration
  dynamic pool target 控制 allocator 可见 block id
  CPU-resident blocks 处于 offline/quarantine 状态
```

### 3.4 只让必要数据使用 UVM

当前稳定实验使用窄口径 UVM 分配策略：

```text
VLLM_USE_UVM=0
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_KV_CACHE_MANAGED=1
```

含义是：

```text
UVM managed:
  MoE expert main weights
  KV cache

native CUDA allocation:
  embedding
  attention weights
  router/gate
  norm
  lm_head
  FP8 scale / aux tensor
  runtime temporary buffer / workspace
```

这样可以避免把非目标对象纳入 UVM fault 和 residency 管理，降低诊断噪音。

## 4. vLLM 基础：这个项目改到了哪些路径

### 4.1 vLLM 推理流程简化图

vLLM v1 engine 的大致流程可以抽象为：

```text
load model
initialize KV cache
warmup / compile / graph / sampler warmup
EngineCore READY
receive requests
prefill
decode loop
```

对于 MoE 层，单次 forward 内部大致是：

```text
hidden states
-> router/gate logits
-> top-k expert ids
-> FusedMoE kernel 根据 top-k 访问 expert weights
-> scatter/gather outputs
```

本项目接入点集中在：

```text
模型加载和 expert 参数分配
KV cache tensor 分配
warmup 后 Stage-M expert placement
MoE router top-k 之后、FusedMoE compute 之前
KV block allocator 分配 block 之前
EngineCore READY 前的 pre-service maintenance
```

### 4.2 关键 vLLM 文件

主要修改路径：

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py
workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py
workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py
workloads/vllm/vllm/vllm/v1/core/block_pool.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/vllm/vllm/v1/engine/core.py
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/uvm_test/vllm_generate_sequence.py
```

## 5. UVM 基础：为什么这个项目需要改 driver

### 5.1 cudaMallocManaged 与 managed VA

用户态 allocator shim `uvm_allocator.cpp` 的核心分配函数是：

```text
uvm_malloc(...)
  -> cudaMallocManaged(&ptr, size, cudaMemAttachGlobal)
```

这带来两个性质：

```text
1. CPU 和 GPU 共享同一个虚拟地址 ptr。
2. 该地址背后的物理页可以在 CPU memory 和 GPU VIDMEM 之间迁移。
```

vLLM 侧通过 `torch.cuda.memory.CUDAPluggableAllocator` 把特定 allocation scope 路由到这个 allocator。

### 5.2 UVM VA block 与 PTE

NVIDIA UVM driver 内部用 VA block 管理 managed range。一个 VA block 维护：

```text
VA 范围
CPU resident page mask
GPU resident page mask
GPU PTE read/write/atomic bits
GPU PMM chunks
tracker / migration state
```

GPU 访问某个 managed VA 时，如果 GPU PTE 缺失或权限不满足，可能触发 replayable fault。普通 UVM 会在 fault handler 中迁移页面、更新 PTE、重放指令。

本项目在 driver 里做的事情是：**绕过普通“fault 后迁移”的被动路径，在请求前主动改变指定 VA 范围的 GPU PTE 指向。**

### 5.3 PMM 是什么

PMM 是 UVM driver 内部的 GPU physical memory manager。它管理 GPU VIDMEM backing chunk 的分配、释放、拆分、合并和驱逐。

可以把它粗略类比为 GPU 显存里的页/块分配器：

```text
普通 UVM residency:
  VA block gpu_state->chunks[] 持有 PMM chunk
  GPU PTE 指向这些 chunk
  residency metadata 认为这些 pages GPU resident

reserved slot:
  driver-owned pool 持有 PMM chunk
  不挂到普通 gpu_state->chunks[]
  只在 restore 时作为 readonly PTE backing
```

本项目确认当前 GPU user PMM 支持的 chunk sizes 是：

```text
4 KiB
64 KiB
2 MiB
```

没有原生 512KiB PMM chunk。因此项目选择：

```text
512KiB logical envelope = 8 * 64KiB physical PMM chunks
```

这既避免改 PMM 支持新的 chunk size，也避免 2MiB envelope 牵连过大。

## 6. PMM 与 reserved slot 机制

### 6.1 为什么从 2MiB 改成 512KiB logical envelope

早期使用 2MiB envelope 时，一个小 expert slice 可能会牵连邻近数据。对于 Qwen3 MoE expert，典型切分近似为：

```text
moe_gate_up ~= 3.0 MiB -> 6 个 512KiB envelope
moe_down    ~= 1.5 MiB -> 3 个 512KiB envelope
单 expert 两个 role -> 9 个 512KiB envelope
```

如果用 2MiB envelope，牵连范围更大，reserved slot 也更浪费。改成 512KiB logical envelope 后：

```text
slot_bytes = 512KiB
physical backing = 8 * 64KiB
一个 slot 只服务一个 512KiB logical range
```

### 6.2 reserved pool ioctl

UVM driver 新增 ioctl：

```text
UVM_EXPERT_RESERVED_POOL = UVM_IOCTL_BASE(98)
```

支持模式：

```text
QUERY
PREALLOC
RESET
```

入口结构里有：

```text
pool_id
slot_id
slot_bytes
chunk_count
physical_bytes
state
epoch
first_chunk_address
last_chunk_address
reserved_backing_bytes
logical_slot_bytes
```

driver 中对应核心结构是：

```text
uvm_expert_reserved_pool_slot_t
```

其中记录：

```text
valid
pool_id / slot_id
state
chunk_count
chunk_size
slot_bytes
physical_bytes
gpu
chunks
owner_va_space
owner_region_id
owner_va_base
owner_bytes
owner_layer_id / owner_expert_id / owner_role
epoch
```

### 6.3 reserved slot 如何分配 PMM chunk

driver 选择 chunk size 的逻辑是：

```text
if slot_bytes <= 512KiB and GPU PMM supports 64KiB:
  chunk_size = 64KiB
else:
  chunk_size = 2MiB
```

对于 512KiB slot：

```text
physical_bytes = align_up(512KiB, 64KiB) = 512KiB
chunk_count = 512KiB / 64KiB = 8

uvm_pmm_gpu_alloc_user(..., chunk_count=8, chunk_size=64KiB, ...)
```

这些 chunks 被保存在 reserved pool slot 中，不挂到普通 managed VA block 的 `gpu_state->chunks[]`。

### 6.4 reserved prefetch ioctl

UVM driver 新增 ioctl：

```text
UVM_EXPERT_RESERVED_PREFETCH_BATCH = UVM_IOCTL_BASE(99)
```

支持模式：

```text
DRY_RUN
RESOLVE
INVALIDATE
RESERVED_PREFETCH
```

关键输入：

```text
region_id       # Phase-F registered expert region id
pool_id
slot_id
slot_offset
bytes_override
STRICT_FOOTPRINT flag
```

`RESERVED_PREFETCH` 做三件事：

```text
1. 根据 region_id 找到 registered expert managed VA range。
2. 找到 reserved pool slot，检查 slot backing 足够。
3. 将 CPU-resident managed pages 拷到 slot chunks，
   并在原 managed VA 上安装 readonly slot-backed GPU PTE。
```

这个路径特别强调：

```text
不把 slot chunks 挂到普通 UVM residency metadata；
ordinary_resident_bytes 应保持 0；
pool_external_alloc_count 应保持 0；
strict footprint contract 不能破。
```

### 6.5 readonly slot-backed PTE 的含义

driver 中 `uvm_va_block_copy_and_map_reserved_readonly()` 做的是：

```text
检查目标 pages 已 CPU resident
检查不是 HMM / confidential computing
按 page 从 CPU copy 到 reserved slot chunk
拆分 2MiB/big PTE 到 4KiB PTE
写 GPU PTE 指向 reserved VIDMEM physical address
设置 GPU_READ
清除 GPU_WRITE / GPU_ATOMIC
标记 moe_slot_pte_bits
```

这样 GPU kernel 从原 expert VA 读数据时，实际读的是 reserved slot 中的物理 backing。

这就是“VA 稳定、物理 backing 可换”的核心。

### 6.6 slot owner metadata 与清理

每个 slot 记录当前 owner：

```text
owner_va_space
owner_region_id
owner_va_base
owner_bytes
owner_slot_offset
owner_layer_id
owner_expert_id
owner_role
owner_epoch
```

slot 被复用前，driver 会先卸载旧 owner 的 slot-backed PTE：

```text
uvm_expert_reserved_pool_release_slot_mapping_locked(...)
  -> uvm_moe_slot_pool_uninstall_va_block_slot_mapping(...)
  -> clear slot owner metadata
```

这避免一个 slot 同时服务多个 VA mapping。

## 7. Expert offload/release 机制

### 7.1 Phase-F/Phase-N region tag

vLLM 在用户态为 expert weight slice 注册 region。driver tag 里会记录：

```text
region_id
base
length
pool = UVM_PHASE_N_POOL_WEIGHT
layer_id
expert_id
role
flags
```

UVM pool id 当前定义：

```text
UVM_PHASE_N_POOL_UNKNOWN = 0
UVM_PHASE_N_POOL_KV     = 1
UVM_PHASE_N_POOL_WEIGHT = 2
UVM_PHASE_N_POOL_SCRATCH = 3
```

这些 tag 用于：

```text
release 指定 expert region
reserved prefetch 根据 region_id 解析 VA
fault guard 判断 fault 属于 KV 还是 weight
pool budget / residency 统计
```

### 7.2 Stage-M init-time offload

vLLM 在 warmup 后、正式服务请求前执行 Stage-M：

```text
gpu_worker.py:
  Stage M post-warmup placement begin
  model_runner.run_uvm_init_time_weight_placement()

gpu_model_runner.py:
  遍历 MoE modules
  module.run_uvm_init_weight_offload()
  synchronize
  clear old slot PTE
  post-Stage-M reserved slot prealloc
  residency probe
```

Stage-M 的关键目的：

```text
请求前卸载计划内 cold/zero-hit experts；
释放 GPU PMM backing；
之后再分配 reserved slot 或扩大 KV target；
不要在 request hot path 里做大规模释放。
```

### 7.3 expert offload release ioctl

UVM driver 新增 ioctl：

```text
UVM_EXPERT_OFFLOAD_RELEASE = UVM_IOCTL_BASE(100)
```

输入是 `region_id` 和 GPU uuid，输出包括：

```text
pool
layer_id
expert_id
role
base
length
attempted_blocks
success_blocks
failure_blocks
released_bytes
failed_bytes
first_error
```

支持 flags：

```text
REQUIRE_EXPERT
STRICT_2M_BLOCKS
SUBCHUNK_64K
```

512KiB envelope 使用：

```text
SUBCHUNK_64K
```

### 7.4 512KiB subrange release 的原理

对于一个 512KiB-aligned expert logical region，release 路径会：

```text
1. 根据 region_id 找到 Phase-F registered region。
2. 确认它是 expert weight pool。
3. 要求 region length == 512KiB 且 base 512KiB 对齐。
4. 对 region 边界执行 native split。
5. 对覆盖的 VA block 调用:
   uvm_va_block_release_gpu_subchunks_to_cpu(..., UVM_CHUNK_SIZE_64K, ...)
6. 先保证 release region CPU resident。
7. 确认目标 region 的 GPU resident/PTE bits 已清。
8. 摘除对应 64KiB GPU chunks。
9. 调用 uvm_pmm_gpu_free() 归还 PMM。
10. 更新 Phase-N block pool resident accounting。
```

因此 release 后：

```text
expert managed VA 仍有效；
CPU 仍有 expert 内容；
GPU PMM backing 被释放；
后续访问必须通过 reserved restore 或显式 prefetch。
```

## 8. vLLM FusedMoE 执行器接入

### 8.1 Expert 权重选择性 managed allocation

在 `fused_moe/layer.py` 中，MoE layer 增加：

```text
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
```

开启时，expert weight 创建进入：

```text
uvm_expert_weight_allocation_scope()
  -> uvm_local_memory_pool()
  -> CUDAPluggableAllocator
  -> uvm_malloc
  -> cudaMallocManaged
```

这样只让 MoE expert main weights 走 managed allocation，而不是把整个模型都交给 UVM。

### 8.2 Expert region 与 envelope 记录

vLLM 为 expert 生成 residency records，包含：

```text
layer_id
global_expert_id
local_expert_id
role = moe_gate_up / moe_down
ptr
bytes
driver_region_id
pool = weights
metadata
```

当启用 512KiB envelope 时，逻辑 region 会按 512KiB 对齐扩展：

```text
envelope_start = align_down(ptr, 512KiB)
envelope_end   = align_up(ptr + size, 512KiB)
```

一个 expert role 可能展开为多个 512KiB logical regions。

### 8.3 从 router top-k 到 active experts

MoE forward 中，router 计算 top-k expert ids。项目在 `select_experts` 后、FusedMoE compute 前增加 gating：

```text
topk_ids
-> _uvm_active_expert_pairs(topk_ids)
-> 根据 layer/expert plan bitmask 判断是否命中 offloaded expert
-> 命中才进入 reserved prefetch
-> 未命中直接 no-hit fast return
```

相关优化：

```text
per-layer expert bitmask
GPU-side mask tensor gate
reserved_prefetch_no_offloaded_active fast path
```

目标是避免每次 MoE forward 都展开所有 plan、查询 driver、或清理 wrapper。

### 8.4 expert_reserved_regions_async executor

用户态执行器在：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py
```

它负责：

```text
读取 slot_count / slot_bytes / pool_id
确保 reserved pool 已 prealloc
按 layer/step 将 active cold expert regions 分配到 slots
调用 driver_expert_reserved_prefetch_regions(..., mode="reserved_prefetch")
记录 success/failed region ids
维护 slot reuse / cleanup / CUDA event 生命周期
支持 strict footprint
```

FusedMoE 层调用：

```text
get_expert_reserved_regions_executor(...)
executor.prefetch_for_layer(...)
```

如果 prefetch 成功，vLLM residency state 会标成：

```text
state = prefetching_gpu
placement = CPU_OFFLOADED_RESERVED
cpu_offloaded_reserved = True
executor = expert_reserved_regions_async
```

随后 FusedMoE kernel 仍按原 expert pointer 读权重。

## 9. KV cache 动态扩容机制

### 9.1 为什么 KV 不能用 expert 的 readonly slot 机制

KV cache 是 writable。GPU attention kernel 会写 key/value。对 writable KV 使用只读 slot-backed remap 不正确；让 GPU 请求路径自动 fault 到 CPU-resident KV page 也不安全。

正确设计是：

```text
CPU-resident KV block:
  不在 allocator 可分配集合中
  不允许 request path 写
  属于 offline/quarantine

使用前:
  batch prefetch 回 GPU
  同步确认 GPU resident
  再扩大 allocator target
  再允许 vLLM 分配
```

### 9.2 KV cache 选择性 managed allocation

`gpu_worker.py` 增加：

```text
VLLM_UVM_KV_CACHE_MANAGED=1
```

开启时，KV cache 初始化进入：

```text
_maybe_get_kv_cache_uvm_memory_pool_context()
  -> uvm_local_memory_pool(enable_prefetch=True)
```

普通 pre-service allocations 默认保持 native CUDA。项目中已移除普通
pre-service allocation 的 managed 开关，非 expert/KV 对象默认继续走
native CUDA allocation。

### 9.3 KV pointer map

KV cache 在 vLLM 中不是一个简单平坦数组；不同 layer/group 会对应多个 range。项目在 KV 初始化后写 pointer map：

```text
gpu_model_runner.initialize_kv_cache_tensors()
  -> _write_stage_l6_kv_pointer_map(...)
```

KV runtime policy 使用该 map 将：

```text
block_id + group_id
```

转换成一批 UVM address ranges，然后批量执行：

```text
batch_prefetch_ranges_to_cpu(...)
batch_prefetch_ranges_to_device(...)
```

底层是：

```text
cudaMemPrefetchAsync(ptr, size, device, stream)
```

注意：该接口只表示 prefetch call accepted；为了实验确认真实 residency，本项目增加：

```text
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1
```

同步后再记录 residency probe。

### 9.4 dynamic pool target

vLLM 的 KV block allocator 在 `block_pool.py` 中被接入：

```text
BlockPool.get_new_blocks()
  -> uvm_kv_runtime_policy.select_free_blocks_for_allocation(...)
  -> 只返回当前 dynamic target 范围内可分配 block
  -> on_blocks_allocated(...)
```

KVCacheManager 分配前还会问：

```text
uvm_kv_runtime_policy.should_allow_allocation(...)
```

dynamic target 的含义是：

```text
logical KV tensor 已经有完整 block 数；
但 allocator 当前只允许使用 block_id <= target_blocks 的 blocks；
target 之外的 blocks 可以被 quarantine/offload；
grow 时再搬回 GPU 并暴露给 allocator。
```

### 9.5 READY 前 pre-service shrink/grow

`engine/core.py` 增加 READY 前 maintenance：

```text
VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY=1
VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_TARGET_BLOCKS=1236
VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS=1364
```

执行顺序：

```text
EngineCore init 尾部
  -> apply target 1236
  -> 将 1237..1364 从 allocator 可见集合隔离
  -> 对隔离 blocks 做 physical prefetch_cpu
  -> grow target 1364
  -> 对 1237..1364 physical prefetch_gpu
  -> 确认 GPU resident
  -> READY handshake
  -> request_begin
```

这里的原则是：**不在 request hot path 里做大规模 KV 物理迁移。**

之前失败实验已经证明，如果把 128 个 writable KV blocks 在 request path 内迁到 CPU，EngineCore 可能进入不可恢复的 CUDA/UVM 状态。把动作移到 READY 前安全 barrier 后，同样 128 blocks 可以正常完成。

## 10. Fault guard 与正确性验证

### 10.1 manual migration fault guard

UVM replayable fault handler 增加参数：

```text
uvm_manual_migration_fault_mode:
  0 = normal UVM service
  1 = observe/log and continue
  2 = log and cancel instead of migrating

uvm_manual_migration_fault_pool_mask:
  0 = all managed ranges
  otherwise bit N applies to pool N
```

pool id：

```text
KV     = 1
WEIGHT = 2
```

因此：

```text
pool_mask = 2
  表示 1 << UVM_PHASE_N_POOL_KV
  只观察/拦截 KV pool

pool_mask = 0
  表示所有 managed ranges
  包括 expert weights 和 KV cache
```

fault log 会包含：

```text
fault_addr
access type
pool
role
region_id
allocation_id
tag_valid
range_base
range_len
```

### 10.2 为什么要捕捉请求开始后的所有 managed fault

理想实现中：

```text
expert:
  cold expert 如果会被用到，必须在 FusedMoE compute 前 reserved restore；
  zero-hit experts 不会被访问；
  请求中不应靠 UVM fault 自动搬 expert。

KV:
  会被写入的 block 必须已经 GPU resident；
  CPU-resident block 不应被 allocator 选中；
  请求中不应靠 UVM fault 自动搬 KV。
```

因为本实验口径下只有 expert 和 KV 使用 UVM managed allocation，所以 all-managed observe 模式下：

```text
request-time manual_migration_fault records = 0
```

就是很强的及时搬运证据。

### 10.3 driver tag 的诊断意义

早期实验出现过：

```text
driver_registered = 0
driver_tag_reasons = missing_ptr_or_size
```

这会导致 fault guard 不能正确按 KV range 命中，诊断变盲。后续加入：

```text
VLLM_UVM_RESIDENCY_KV_FORCE_DRIVER_TAG=1
```

确保 KV range/tensor 真正注册 driver tag，从而 fault guard 能判断 fault 属于 KV 还是 weight。

## 11. 端到端生命周期

### 11.1 冷 expert runtime restore 生命周期

```text
load model
  expert weights 通过 cudaMallocManaged 分配
  注册 Phase-F/Phase-N weight regions

warmup
  初始化 kernel / sampler / attention backend

Stage-M
  根据 cold plan 找到要卸载的 expert regions
  调用 UVM_EXPERT_OFFLOAD_RELEASE
  expert GPU PMM backing 释放
  expert VA 保持有效，内容在 CPU resident
  post-Stage-M prealloc reserved slots

request
  router top-k 得到 active experts
  per-layer bitmask 判断是否命中 offloaded expert
  no hit: fast return
  hit:
    executor 分配 reserved slot
    UVM_EXPERT_RESERVED_PREFETCH_BATCH reserved_prefetch
    CPU pages -> reserved slot chunks
    original expert VA -> readonly slot-backed GPU PTE
    FusedMoE kernel 读取 original expert VA

cleanup/reuse
  根据 event/step 延迟清理旧 mapping
  slot 可服务下一个 expert
```

### 11.2 expert release + KV grow 生命周期

```text
load model
  expert weights managed
  KV cache managed, logical 2GiB

warmup
  所有需要服务前初始化的 kernel 完成

Stage-M
  卸载固定 fixture 中 zero-hit experts
  释放约 1.05GiB expert GPU residency

pre-service KV maintenance
  target 先限制到 1236 blocks
  tail 128 blocks quarantine/offload
  grow 到 1364 blocks
  tail 128 blocks prefetch 回 GPU
  同步并确认 GPU resident

request
  开启 all-managed observe 或 KV-only fail-fast
  allocator 可见 1364 blocks
  长 prompt 可分配到新增 block id
  请求后 fault records = 0
```

## 12. 实验结果总结

### 12.1 512KiB static cold8 reserved 机制跑通

实验：

```text
docs/vllm_uvm_512k_static_cold8_reserved_results_20260713.md
```

关键结果：

```text
Stage-M release:
  attempted bytes = 1811939328
  released GPU bytes = 1755316224
  cold GPU bytes: 1755316224 -> 0

reserved slot prealloc:
  slot_count = 128
  slot_bytes = 524288
  chunk_size = 65536
  success_count = 128
  failure_count = 0
  reserved_backing_bytes = 67108864

runtime selective restore:
  restored 512KiB entries = 2052
  failed restored entries = 0
  accepted bytes = 1075838976
```

结论：

```text
512KiB logical / 64KiB physical reserved slot 机制可在 vLLM 中完成：
  expert release
  slot prealloc
  selective restore
  generation
  no Xid/MMU fault
```

### 12.2 cold8 restore-heavy 为什么慢

实验：

```text
docs/vllm_sharegpt_cold8_512k_restore_detail_results_20260713.md
```

结果：

```text
completed_requests = 32/32
tokens/s = 11.247
touched_cold_experts = 314
untouched_cold_experts = 70
restore_entry_count = 20610
restore_bytes ~= 10.3 GiB
ioctl_failure_entries = 0
```

解释：

```text
机制正确，但 cold plan 不够冷；
很多 planned cold experts 仍被 runtime 反复访问；
重复 512KiB restore traffic 成为主要性能瓶颈。
```

典型高 restore expert：

```text
L1/E51  310.5 MiB
L1/E22  189.0 MiB
L12/E113 162.0 MiB
```

请求级别也呈现：

```text
restore MiB 越高，请求生成时间越长
```

### 12.3 zero-hit offload 的固定开销定位

实验：

```text
docs/vllm_zero_hit_runtime_check_ablation_20260714.md
```

结果：

```text
clean no-offload 2GiB:
  19.249 tok/s

zero-hit no-trace, runtime check on:
  17.065 tok/s

zero-hit offload-only, runtime check off:
  19.538 tok/s
```

结论：

```text
zero-hit 情况下，如果 runtime reserved/prefetch 检查仍开启，
每次 MoE forward 的 plan intersection / active expert extraction / no-hit fast return
会累计成明显开销。

关闭 runtime check 后，性能恢复到 clean baseline 水平。
```

因此后续优化重点是：

```text
让 no-hit fast path 更轻；
用 per-layer bitmask 和 GPU-side mask gate 避免 Python 侧无效展开；
对确定 zero-hit 的计划，可实验性关闭 runtime restore path。
```

### 12.4 pre-service expert release + KV grow 成功

实验：

```text
docs/vllm_pre_service_expert_release_kv_grow_results_20260714.md
```

结果：

```text
clean baseline:
  completed = 8/8
  tokens/s = 21.638
  generate seconds = 23.662

zero-hit expert release + pre-service KV grow:
  completed = 8/8
  tokens/s = 17.788
  generate seconds = 28.783
```

机制证据：

```text
expert release:
  released bytes = 1102577664 ~= 1051.5 MiB
  failed entries = 0
  cold_plan_gpu_bytes: 1102577664 -> 0

KV grow:
  initial_target_blocks = 1236
  grow_target_blocks = 1364
  added_blocks = 128
  added KV bytes = 192 MiB
  KV CPU offload success = 128/128
  KV GPU prefetch success = 128/128
  managed probe failures = 0

fault:
  KV-only manual fault guard enabled before request
  new Xid/MMU fault = false
  dmesg faultish lines = 0
```

注意：短请求没有真正用到 tail blocks，它验证的是：

```text
expert release 成功；
KV target 增加成功；
新增 blocks 请求前 GPU resident；
请求阶段没有 KV fault。
```

### 12.5 长上下文验证

实验：

```text
docs/vllm_pre_service_kv_grow_replicate_long_context_detailed_20260714.md
docs/vllm_pre_service_kv_grow_all_managed_fault_observe_results_20260714.md
```

结果：

```text
short clean baseline:
  8/8, 20.76 tok/s

short mechanism:
  8/8, 17.53 tok/s

long clean baseline:
  1/1, 3.36 tok/s

long mechanism:
  1/1, 4.62 tok/s
```

长 prompt：

```text
actual_prompt_tokens_no_special ~= 20512
tokens_per_block = 16
estimated required KV blocks ~= 1282
```

机制组 KV target：

```text
1236 -> 1364 blocks
```

allocator 证据：

```text
max selected KV block id = 1283
selected blocks above 1236 = 47
selected above-target range = 1237..1283
cpu_candidates_max = 0
allocation_no_free_blocks = 0
```

all-managed observe：

```text
uvm_manual_migration_fault_mode = 1
uvm_manual_migration_fault_pool_mask = 0
manual_migration_fault records = 0
Xid/MMU/fatal/faultish records = 0
```

结论：

```text
长上下文确实使用了新增 KV blocks；
请求开始后 expert weights 和 KV cache 都没有产生 GPU-side managed page fault；
pre-service 搬运足够及时。
```

## 13. 曾经失败的实验说明了什么

### 13.1 request path 内做大规模 KV 迁移会卡死

早期把 KV shrink/offload 放到第一个 request path 中，结果出现：

```text
request_begin 后卡死
EngineCore 不退出
GPU memory 被占住
kill -9 无法正常回收
```

后来通过 begin/end 物理迁移日志确认：

```text
prefetch_cpu 本身可以完成；
卡死发生在 CPU-resident writable KV blocks 暴露给请求路径之后。
```

正确修复是：

```text
不要在 request hot path 做大规模 KV migration；
CPU-resident KV blocks 必须 offline/quarantine；
只有 prefetch 回 GPU 并确认后才能放回 allocator 可分配集合。
```

### 13.2 `grow_target_blocks=0` bug

曾经新增二阶段 pre-service grow 开关时，把 `"0"` 当成有效 target，导致 baseline 准备把 KV target 从 1236 改成 0，触发危险大规模迁移。

修复：

```text
grow_target_blocks <= 0 直接忽略
```

对应代码：

```text
workloads/vllm/vllm/vllm/v1/engine/core.py
```

### 13.3 driver tag 未注册会让 fault guard 失效

早期 KV driver tag 记录里出现：

```text
driver_registered = 0
driver_tag_reasons = missing_ptr_or_size
```

导致 manual fault guard 无法把 fault 地址映射到 KV range。后续强制 KV driver tag 后，才能可靠使用 pool mask 捕捉 fault。

### 13.4 full managed guard 与 KV-only guard 的差别

KV-only guard：

```text
mode = 2
pool_mask = 2
只拦截 KV pool fault
```

all-managed observe：

```text
mode = 1
pool_mask = 0
记录所有 managed fault 但继续运行
```

由于实验口径只让 expert 和 KV managed，all-managed observe 更适合最终证明“请求后没有任何 managed GPU fault”。但 fail-fast 全 managed 可能误伤非目标实验阶段，因此性能实验常用 KV-only guard。

## 14. 性能收益与代价

### 14.1 已证明的收益

当前系统已经证明：

```text
1. 可以从 GPU residency 中释放计划内 expert 权重。
2. 可以用固定 reserved slot pool restore 被访问的 cold expert。
3. 可以在同一进程内增加 vLLM allocator 可用 KV blocks。
4. 长上下文实际分配到了新增 KV block id。
5. 请求开始后没有 expert/KV managed GPU fault。
6. 普通 cudaMalloc all-resident 情况下 2GiB KV 会 OOM，而 managed/offload 路径可以走到 2GiB logical KV。
```

### 14.2 性能代价来源

主要代价分几类：

```text
pre-service cost:
  Stage-M expert release
  reserved slot prealloc
  KV physical prefetch CPU/GPU
  residency/fault guard setup

request-time cost:
  如果 cold expert 被访问，会有 512KiB restore copy/remap
  如果 zero-hit 但 runtime path 开启，会有每 MoE forward 的 fast-path 检查开销
  trace/residency bookkeeping 也会影响吞吐
```

重要结论：

```text
“专家被卸载”本身不必然导致性能大幅下降；
大幅下降通常来自 runtime 真实 restore 流量，或者 no-hit path 仍然太重。
```

### 14.3 long mechanism 为什么可能比 long clean baseline 更快

长上下文实验中：

```text
long clean baseline = 3.36 tok/s
long mechanism      = 4.62 tok/s
```

这不应简单理解为“机制一定加速”。更合理解释是：

```text
long clean baseline 启动时所有 expert resident，KV/显存压力更高；
mechanism 释放了约 1.04GiB zero-hit experts，并在请求前只暴露 GPU-resident KV blocks；
长 prompt 的 prefill 对显存压力更敏感，释放 expert 后有可能降低 allocator/driver 压力；
但单次 long prompt 结果样本很小，仍需更多 fixture 和重复实验。
```

## 15. 如何复现实验

### 15.1 模块状态检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

cat /sys/module/nvidia_uvm/srcversion
cat /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode
cat /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_pool_mask

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

ps -eo pid,stat,comm,args | rg 'VLLM|EngineCore|vllm_generate_sequence' || true
```

期望：

```text
srcversion = 386A1FA2E18DDDFCE9EFE88
manual fault mode = 0
manual fault pool mask = 0
GPU memory.used 接近 0 MiB
无残留 EngineCore
```

如果重启后模块不对，需要在 GPU 空闲时重载本仓库 UVM 模块。

### 15.2 short mechanism 复现

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=/tmp/vllm_pre_service_kv_grow_replicate_short_$(date +%Y%m%d_%H%M%S) \
FIXTURE_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json \
PLAN_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=8 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=2048 \
KV_CACHE_MEMORY_BYTES=2147483648 \
INITIAL_TARGET_BLOCKS=1236 \
GROW_TARGET_BLOCKS=1364 \
KV_PRE_SERVICE_APPLY=1 \
KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_FORCE_DRIVER_TAG=1 \
KV_DRIVER_TAG_ENABLE=0 \
KV_POINTER_TAG_GRANULARITY=tensor \
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=0 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

这里使用 all-managed observe：

```text
mode = 1
pool_mask = 0
```

用于观察所有 managed request-time faults。

### 15.3 long mechanism 复现

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=/tmp/vllm_long_context_kv_grow_$(date +%Y%m%d_%H%M%S)/mechanism \
FIXTURE_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json \
PLAN_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=1 \
MAX_MODEL_LEN=21840 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=2048 \
KV_CACHE_MEMORY_BYTES=2147483648 \
INITIAL_TARGET_BLOCKS=1236 \
GROW_TARGET_BLOCKS=1364 \
KV_PRE_SERVICE_APPLY=1 \
KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_FORCE_DRIVER_TAG=1 \
KV_DRIVER_TAG_ENABLE=0 \
KV_POINTER_TAG_GRANULARITY=tensor \
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=0 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

验收重点：

```text
expert release failed = 0
KV CPU success = 128/128
KV GPU success = 128/128
selected block id > 1236
cpu_candidates_max = 0
allocation_no_free_blocks = 0
manual_migration_fault records = 0
Xid/MMU/fatal/faultish records = 0
```

## 16. 当前边界与未来工作

### 16.1 zero-hit plan 只对固定 fixture 成立

当前“不会被使用的 expert”是从固定请求 fixture 的 pilot routing trace 中得到的。它不代表任意未来请求都不会访问这些 expert。

生产系统需要：

```text
更稳健的热度预测；
在线统计；
按 request batch 更新 plan；
访问风险估计；
命中后快速 restore；
避免 zero-hit 假设被破坏。
```

### 16.2 还未严格证明具体 PMM chunk identity 复用

当前已经证明：

```text
expert GPU residency 被释放；
reserved slot backing 可以分配；
KV blocks 可以 grow 并 prefetch GPU；
同一进程内机制闭环成功。
```

但还没有严格证明：

```text
某个 expert 释放出的具体 PMM chunk
被某个 KV block 精确复用。
```

要证明这一点，需要 PMM chunk identity tracing：

```text
记录 expert release freed chunk physical address
记录 reserved/KV allocation acquired chunk physical address
做 identity-level 匹配
```

### 16.3 runtime restore 仍需优化

cold8 restore-heavy 实验说明，如果 plan 中 cold expert 仍被频繁命中，性能会明显下降。

可优化方向：

```text
更精确的 cold expert 选择；
prefill 阶段单独策略；
slot cleanup delay / reuse 策略优化；
多 slot bank / double buffering；
restore 与 compute overlap；
按 layer 预取下一步可能 expert；
减少 512KiB envelope 重复 restore。
```

### 16.4 no-hit path 仍需低开销化

zero-hit runtime check ablation 说明，每 MoE forward 的固定检查成本可累计到十几秒级别。

优化方向：

```text
per-layer static flag: 本层没有 offloaded expert 时完全跳过；
GPU-side mask gate 避免 CPU 侧 top-k 拷贝；
缓存 active-expert intersection；
将 Python bookkeeping 下沉到 C++/CUDA；
减少 trace/residency record 热路径写入。
```

### 16.5 KV dynamic pool 应产品化为安全状态机

当前实验已经验证：

```text
READY 前 quarantine
request 间 grow back
request 后 no fault
```

但生产化还需要：

```text
明确 block lifecycle: gpu_free / cpu_offline / gpu_prefetching / allocatable
强制 allocator 只选择 GPU-resident blocks
异步 batch migration 与超时保护
fail-fast rollback
长上下文多请求并发验证
与 prefix cache / block reuse / eviction 的完整一致性检查
```

## 17. 简历描述可以如何理解

原始简历描述：

```text
面向 MoE 推理的 vLLM/UVM 专家权重显存复用系统
```

更完整的技术含义是：

```text
在 vLLM 的 FusedMoE router 和执行器中识别 active expert；
在 warmup 后把 cold/zero-hit expert 从 GPU residency 中释放；
通过自定义 UVM driver ioctl 维护 fixed-size reserved VIDMEM slot pool；
命中 offloaded expert 时，将 CPU-resident expert pages 拷到 reserved slot；
在原 managed VA 上安装 readonly slot-backed PTE；
让 Triton/FusedMoE kernel 无需改 pointer 即可读取 restored expert；
同时用 KV dynamic pool 把释放出的显存预算转化为更多可用 KV blocks；
用 manual fault guard 验证请求阶段没有依赖自动 UVM migration。
```

对应能力点：

```text
vLLM:
  理解模型加载、KV cache、BlockPool、FusedMoE router/top-k、EngineCore READY 生命周期；
  能把系统级内存策略插入 pre-service barrier 与 MoE hot path。

UVM:
  理解 managed VA、VA block、GPU PTE、replayable fault、range tag、driver ioctl；
  能扩展 driver 实现自定义 residency/release/remap 语义。

PMM:
  理解 GPU VIDMEM chunk 分配、64KiB/2MiB chunk、split/free、driver-owned backing；
  能把模型语义上的 expert region 映射到 PMM backing 生命周期。

系统实验:
  能设计 fault guard、restore trace、residency probe、long-context fixture；
  能区分机制正确性、性能瓶颈、实验配置 bug 和 driver 状态污染。
```

## 18. 当前最重要的结论

```text
1. reserved slot-backed expert restore 机制已经跑通。
2. 512KiB logical / 64KiB physical backing 是当前可行粒度。
3. cold expert 选择质量决定 runtime restore 性能。
4. zero-hit offload 本身可以接近 baseline，但 runtime no-hit check 必须优化。
5. KV cache 必须按 block 做安全状态管理，不能把 CPU-resident writable blocks 暴露给 request path。
6. pre-service expert release + KV grow 可以让长上下文实际使用新增 KV blocks。
7. all-managed request-time fault observe 为 0，说明在当前固定 fixture 下搬运是及时的。
```
