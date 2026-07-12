# 简历项目总结：面向 MoE 推理的 vLLM/UVM 专家权重显存复用系统

> 说明：本文档用于把当前科研项目整理成简历与面试材料。效果数据尽量采用已有实验结果，不建议写无法复现的夸大指标。可以把表述写得更有冲击力，但核心数字应能在实验日志或脚本中自洽。

## 一句话版本

基于 NVIDIA UVM 驱动和 vLLM MoE 执行路径，实现了一套面向大规模 MoE 推理的专家权重按需搬运与固定显存窗口复用机制：通过 driver-owned reserved VIDMEM slot、managed VA 的 readonly slot-backed PTE 重映射、CUDA event/ring slot 生命周期管理，以及 vLLM router-aware offload policy，在保持 expert 虚拟地址稳定的前提下，让 CPU-offloaded 专家权重复用固定 GPU physical backing，降低 KV cache 压力下的专家权重显存占用。

## 简历可直接使用版本

### 中文简历项目描述

**vLLM/UVM MoE 专家权重显存复用与按需搬运系统**

- 修改 NVIDIA UVM kernel driver 与 vLLM FusedMoE 执行路径，设计并实现 `reserved slot-backed PTE` 机制：在 GPU 端预分配固定大小 VIDMEM slot pool，将 CPU-offloaded expert 权重按需拷入 reserved slot，并把原 managed VA 的 GPU PTE 临时重映射到该 slot，实现“虚拟地址不变、物理 backing 可复用”的专家权重访问路径。
- 新增 `UVM_EXPERT_RESERVED_PREFETCH_BATCH` ioctl、reserved pool prealloc/query/reset、region -> slot assignment、slot owner metadata、strict footprint counters，并在 VA block 中引入 `moe_slot_pte_bits` 区分普通 UVM residency 与 slot-backed readonly PTE，避免普通 UVM eviction/migration 误释放 reserved backing。
- 在 vLLM 中新增 `expert_reserved_regions_async` executor，接入 FusedMoE router top-k 结果、per-layer cold expert placement plan、Stage-M init-time offload、CUDA event/ring slot cleanup，支持每层按 active expert 动态判断是否搬运 CPU-offloaded expert。
- 优化 runtime no-hit 路径：用 `dict[int, list[bool]]` per-layer bitmask 和 GPU-side mask gate 过滤 active expert，只有命中 CPU-offloaded expert 时才进入通用 prefetch 框架；修复空 cleanup wrapper 带来的 per-layer 开销，使冷专家未命中场景 p50 延迟接近 baseline。
- 在 RTX 5090 + Qwen3-30B-A3B-FP8 上完成端到端验证：12 个 4MiB reserved slots 支撑每层 2 个 offloaded expert 的双层/ring 复用；真实 vLLM 单 prompt 生成成功，reserved prefetch/cleanup 无失败；微基准中 reserved slot 搬运较 native `cudaMemPrefetchAsync` 提升约 1.75x，单 slot 复用 smoke 中提升约 2.3x。

### English Resume Version

**vLLM/UVM Expert Weight Paging and Fixed GPU Slot Reuse for MoE Inference**

- Designed and implemented a reserved GPU slot reuse mechanism across NVIDIA UVM and vLLM FusedMoE, enabling CPU-offloaded MoE expert weights to be copied into preallocated VIDMEM slots and accessed through the original managed virtual addresses via readonly slot-backed GPU PTE remapping.
- Added a new UVM ioctl path for batched expert reserved prefetch, slot pool preallocation/query/reset, region-to-slot assignment, owner metadata tracking, strict fixed-footprint counters, and VA-block-level `moe_slot_pte_bits` to separate experimental slot-backed mappings from ordinary UVM residency.
- Integrated the driver path into vLLM with an `expert_reserved_regions_async` executor, router-aware active expert filtering, per-layer cold expert placement plans, Stage-M initialization-time CPU offload, CUDA event based slot lifetime management, and ring/double-slot reuse.
- Reduced no-hit runtime overhead by introducing per-layer expert bitmasks and an early GPU-side mask gate, ensuring the generic prefetch framework is only entered when an active expert is actually CPU-offloaded.
- Validated on RTX 5090 with Qwen3-30B-A3B-FP8: fixed 48MiB reserved window for dynamic expert paging, successful end-to-end vLLM generation, zero reserved prefetch failures in tested runs, and up to 1.75x faster layer-level transfer versus native UVM prefetch in microbenchmarks.

## 项目背景

### 解决的问题

MoE 模型的 expert 权重规模大，但每次 token/layer 只会激活少量 expert。以 Qwen3-30B-A3B-FP8 这类模型为例，每层有大量 expert，但 router top-k 每次只选中其中一部分。传统部署方式通常有两个极端：

- 所有 expert 常驻 GPU：延迟低，但显存压力大，容易挤压 KV cache。
- 依赖普通 UVM demand paging / prefetch：实现简单，但无法指定 GPU physical backing，也难以保证 expert 权重只占用固定显存窗口。

本项目的目标是实现第三条路径：

```text
GPU-resident hot experts:
  继续留在 GPU，走普通 vLLM/FusedMoE 计算路径

CPU-offloaded cold experts:
  仅在被 router 选中时搬入固定 reserved slot
  计算时仍使用原 expert managed VA
  计算结束后延迟 cleanup，slot 可被后续层/后续 expert 复用
```

核心收益是把 CPU-offloaded expert 的 GPU footprint 限制为：

```text
reserved_slot_count * reserved_slot_bytes
```

而不是随着 offloaded expert 总数线性增长。

## 当前实现总览

### 1. UVM driver 层：reserved VIDMEM slot pool

新增/扩展了 driver-owned reserved backing pool，用于在 GPU 上预留固定物理显存窗口。

关键能力：

- prealloc：初始化时分配固定数量 slot。
- query：查询 active/backed slots、reserved backing bytes、epoch、owner metadata。
- reset/release：清理 owner PTE 并释放 reserved backing。
- slot owner tracking：记录当前 slot 对应的 region/layer/expert/role。

关键接口：

- `UVM_EXPERT_RESERVED_POOL`
- `UVM_EXPERT_RESERVED_PREFETCH_BATCH`
- Python wrapper：
  - `driver_expert_reserved_pool_prealloc()`
  - `driver_expert_reserved_pool_query()`
  - `driver_expert_reserved_prefetch_regions()`

### 2. UVM driver 层：reserved_prefetch 快速路径

当用户态调用 `mode=reserved_prefetch` 时，driver 执行：

```text
region_id
  -> 查 registered expert region
  -> 校验 pool_id / slot_id / slot_offset / bytes
  -> 若 slot 有旧 owner，先卸载旧 owner 的 slot-backed PTE
  -> 等待 VA block tracker，避免 GPU 仍使用旧页表
  -> 将 CPU-resident managed pages copy 到 reserved PMM chunks
  -> 在原 managed VA 上安装 readonly slot-backed GPU PTE
  -> 更新 slot owner metadata
```

这个路径的关键点是：

- **虚拟地址不变**：vLLM/Triton/FusedMoE kernel 仍然使用原 expert weight pointer。
- **物理 backing 可复用**：不同 expert 可以在不同时间映射到同一 reserved slot。
- **不填 `gpu_state->chunks[]`**：reserved chunk 不进入普通 UVM residency 生命周期，避免被普通 eviction/migration/free 路径误处理。
- **readonly PTE**：当前版本只支持只读 expert 权重，GPU 写入应触发错误或被 strict guard 拦截。

### 3. VA block 元数据与 strict guard

为了让 UVM 知道这些 PTE 不是普通 resident mapping，在 VA block GPU state 里使用：

```text
moe_slot_pte_bits
```

作用：

- 标记哪些页是 slot-backed readonly mapping。
- unmap/invalidate/reset 时只清特殊 PTE。
- eviction/migration 路径识别并跳过 slot-backed 页。
- strict counters 验证没有走普通 PMM fallback：
  - `ordinary_resident_bytes = 0`
  - `pool_external_alloc_count = 0`
  - `footprint_contract_broken = 0`

这也是为什么当前实现不是完整 native UVM residency，而是一个专属、实验性、性能优先的 slot-backed fast path。

### 4. vLLM 层：reserved executor

新增 executor：

```bash
VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async
```

职责：

- 管理 fixed reserved slots。
- 将 selected offloaded experts 分配到 slot。
- 对每个 expert 的 `moe_gate_up` 和 `moe_down` 两段权重分别建立 region。
- 调用 driver `reserved_prefetch`。
- 在计算后 record CUDA event。
- slot 复用前等待旧 event，再 invalidate 旧 owner。

当前真实模型中，Qwen3 FP8 expert region 大小大致为：

```text
moe_gate_up: 3MiB
moe_down:    1.5MiB
```

因此实验中使用：

```text
slot_bytes = 4MiB
每个 expert 需要 2 个 slot
每层 offload 2 个 expert
双层/ring 复用需要 2 layers * 2 experts * 2 roles = 8 slots
实验使用 12 slots，固定 reserved footprint = 48MiB
```

### 5. vLLM 层：placement policy

实现了基于 KV cache 压力和 cold expert plan 的 placement 思路：

```text
GPU_RESIDENT:
  hot experts，继续留在 GPU

CPU_OFFLOADED_RESERVED:
  cold/offloaded experts，被选中时通过 reserved slot 搬入
```

重要点：

- 不是所有 expert 都走 reserved path。
- 只有被标记为 `CPU_OFFLOADED_RESERVED` 且被 router 选中的 expert 才触发 reserved prefetch。
- cold expert plan 在初始化时加载为 per-layer bitmask，避免每次遍历复杂 plan 结构。

### 6. runtime fast gate 优化

早期实现中 `_maybe_prefetch_uvm_expert_weights()` 每层都会进入，即使 cold expert 没被选中，也会做大量 Python/driver bookkeeping。

优化后：

```text
topk_ids
  -> per-layer offloaded expert bitmask
  -> GPU-side mask[topk_ids].any()
  -> 命中才进入 _maybe_prefetch_uvm_expert_weights()
  -> 未命中直接返回
```

进一步修复：

- 如果没有 active reserved slots，不再尝试 cleanup wrapper。
- no-hit 路径不会展开 region。
- no-hit 路径不会 probe driver。
- no-hit 路径不会 register residency metadata。

这使 cold expert 未命中场景从显著慢于 baseline，优化到 p50 基本接近 baseline。

## 已完成实验结果

### 实验 1：单 slot 多 expert 轮流复用 smoke

配置：

```text
4 experts
每个 expert 2MiB
只预留 1 个 2MiB reserved slot
重复 5 轮，共 20 次 remap
```

结果：

```text
reserved_slot_reuse_prefetch:
  avg_ms = 0.156
  p50_ms = 0.153
  avg_GiBps = 12.507

native cudaMemPrefetchAsync baseline:
  avg_ms = 0.365
  p50_ms = 0.359
  avg_GiBps = 5.356

correctness:
  remaps = 20
  kernel_errors = 0
```

可写进简历的表述：

> 在 2MiB expert 粒度微基准中，reserved slot-backed prefetch 较 native UVM prefetch 平均延迟降低约 57%，有效带宽提升约 2.3x。

### 实验 2：20 层 MoE layer bench

配置：

```text
20 layers
每层 90 experts
80 resident experts
10 offloaded experts
每个 expert 2MiB
每层搬运 20MiB 到 10 个 reserved slots
```

结果：

```text
reserved_prefetch_layers:
  avg_ms = 0.877
  avg_GiBps = 22.274

native_prefetch_gpu_layers:
  avg_ms = 1.537
  avg_GiBps = 12.705

comparison:
  reserved_vs_native_gpu_speedup = 1.753x

correctness:
  kernel_errors = 0
  checked_outputs = 921600
```

可写进简历的表述：

> 在 20 层 MoE 模拟推理中，每层 20MiB expert 搬运的 reserved path 平均达到 22.3GiB/s，相比 native UVM prefetch 提升约 1.75x，并通过 92 万余输出元素校验。

### 实验 3：event + 双 bank slot 生命周期验证

配置：

```text
20 layers
每层 3 offloaded experts
每个 expert 两段权重：3MiB gate_up + 1.5MiB down
2 banks
每个 bank 6 slots
总计 12 slots * 4MiB = 48MiB
```

结果：

```text
reserved_prefetch:
  avg_ms = 1.461

reuse_event_wait:
  avg_ms = 0.007

deferred_invalidate:
  avg_ms = 0.908

correctness:
  kernel_errors = 0
  status = pass
```

压力版长 kernel：

```text
reuse_event_wait:
  avg_ms = 2.145
  max_ms = 4.796
status = pass
```

结论：

- event 能精确保护 slot 复用边界。
- 双 bank/ring bank 可以避免本层结束时强制 device synchronize。
- 复用时旧 event 通常已完成，等待成本可接近 0。

### 实验 4：真实 vLLM end-to-end active expert offload

配置：

```text
model: Qwen/Qwen3-30B-A3B-FP8
prompt: 1
max_tokens: 8
offload plan: 48 层，每层 2 个 active expert
reserved slots: 12 * 4MiB = 48MiB
```

结果：

```text
reserved_prefetch_issued = 466
reserved_prefetch_failed = 0
reserved_layer_done_cleanup = 202
slot_event_wait = 454

reserved prefetch ioctl avg = 3.08ms
source CPU-offload avg = 0.90ms
event_wait avg = 25.45us

生成成功:
  "Unified memory is a memory management architecture that"
```

结论：

- reserved executor 能在真实 FusedMoE forward 中运行。
- 12 个 slot 可被高频复用，每个 slot 使用约 96-100 次。
- 真实 vLLM 运行结束后无新的 Xid/MMU fault/UVM assert。

### 实验 5：cold expert no-hit 性能优化

目标：

验证每层 offload 2 个 cold expert，但这些 expert 没被当前 prompt 选中时，系统不会显著影响 vLLM 性能。

关键数据：

```text
baseline clean:
  avg = 1.148s
  p50 = 1.087s

Stage-M offload only:
  avg = 1.174s
  p50 = 1.092s
  logical offload = 432MiB

早期 reserved runtime no-hit:
  avg = 1.606s
  p50 = 1.549s

优化后 reserved runtime no-hit:
  avg = 1.238s
  p50 = 1.113s

skip topk diagnostic after cleanup guard:
  avg = 1.176s
  p50 = 1.087s
```

结论：

- Stage-M offload 本身开销很低。
- 主要开销来自 runtime prefetch no-hit 路径的空 cleanup / topk 决策。
- 通过 bitmask fast gate 和 cleanup guard，p50 延迟基本回到 baseline。
- avg 仍有少量开销，主要来自 GPU-side `mask[topk_ids].any().item()` 决策同步。

简历中可写：

> 将 cold expert 未命中路径从每层通用 prefetch 框架降级为 O(1) bitmask gate，使真实 vLLM 单请求 p50 延迟从 1.55s 降至 1.11s，接近 1.09s baseline。

## 项目亮点

### 亮点 1：虚拟地址稳定，物理 backing 可复用

vLLM/Triton/FusedMoE 算子不需要替换 weight pointer。算子仍访问原 managed VA，driver 通过 GPU PTE 把该 VA 指向 reserved slot。

这解决了一个很关键的问题：

```text
用户态/框架看到的 expert pointer 不变
GPU 页表里的 physical backing 可变
reserved slot 可被不同 expert 重复使用
```

### 亮点 2：固定 GPU footprint contract

普通 UVM prefetch 无法保证使用哪块 GPU physical backing。本项目通过 reserved slot pool 明确限制：

```text
reserved footprint = slot_count * slot_bytes
```

例如真实实验中：

```text
12 slots * 4MiB = 48MiB
```

即使 CPU-offloaded expert 总数更多，reserved path 的动态 GPU 窗口仍被限制在 48MiB 量级。

### 亮点 3：不接入 `gpu_state->chunks[]`，规避 UVM 生命周期复杂性

短期实现没有把 reserved chunk 当作普通 UVM resident pages，而是专属 slot-backed readonly PTE。

优点：

- 避免普通 eviction/migration/free 路径误释放 reserved backing。
- 避免修改 PMM 内部 free list / eviction scanner 带来的全局性能风险。
- 更容易做 fixed-footprint 实验。

代价：

- 当前只支持 readonly expert 权重。
- 不支持 GPU 写回 dirty/writeback 语义。
- 不是完整 native UVM residency。

### 亮点 4：event/ring slot 生命周期管理

slot 复用不是靠猜测 “layer n+2 一定安全”，而是记录 CUDA event：

```text
compute 提交后 record event
slot 复用前 wait event
event 完成后 invalidate old owner
copy/map new owner
```

这个设计能解释为什么不会在 GPU kernel 仍读旧 expert 时复用 slot。

### 亮点 5：router-aware cold expert policy

不是盲目 offload expert，而是结合 routing trace / cold expert plan：

```text
active_experts = unique(topk_ids)
offloaded_mask[layer][expert_id] = True/False
need_prefetch = active_experts ∩ offloaded_set
```

如果交集为空，直接跳过，不展开 region，不触发 driver probe。

### 亮点 6：真实模型端到端验证

不是只做 CUDA microbenchmark，而是接到了真实 vLLM FusedMoE forward：

- Qwen3-30B-A3B-FP8。
- 真实 `moe_gate_up` / `moe_down` region。
- 真实 router top-k。
- 真实 FusedMoE/Triton kernel 访问原 managed VA。
- 真实 prompt generation 成功。

## 当前代码修改地图

### Driver / UVM

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
  - 新增 reserved pool / reserved prefetch ABI。
  - 新增 `UVM_EXPERT_RESERVED_PREFETCH_BATCH` ioctl。
  - 输出 strict counters。

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
  - reserved pool slot table。
  - prealloc/query/reset。
  - slot owner metadata。
  - region -> slot assignment。
  - old owner PTE release。
  - reserved prefetch ioctl handler。

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`
  - 声明 slot-backed readonly mapping helper。
  - 说明该路径不属于普通 UVM migration/residency。

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
  - `moe_slot_pte_bits` 管理。
  - `uvm_va_block_copy_and_map_reserved_readonly()`。
  - slot-backed PTE install/unmap。
  - eviction/migration 路径跳过特殊 PTE。

### vLLM Python / Executor

- `workloads/vllm/vllm/vllm/device_allocator/uvm.py`
  - Python ioctl wrapper。
  - reserved pool prealloc/query/reset。
  - reserved prefetch batch wrapper。
  - 多 UVM fd/VA-space 选择修复，避免 `INVALID_DEVICE`。

- `workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py`
  - `expert_reserved_regions_async` executor。
  - slot assignment。
  - event wait / cleanup。
  - prefetch/invalidate trace。

- `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py`
  - `CPU_OFFLOADED_RESERVED` driver flag。
  - region metadata 注册。

- `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`
  - FusedMoE runtime 接入。
  - Stage-M init-time offload plan。
  - top-k active expert 过滤。
  - per-layer bitmask fast gate。
  - cleanup delay / event / stream trace。

### 测试与 benchmark

- `workloads/vllm/vllm/uvm_test/expert_reserved_prefetch_smoke.cu`
  - 单/多 expert slot reuse correctness。

- `workloads/vllm/vllm/uvm_test/expert_reserved_layer_bench.cu`
  - 20 层 MoE-like layer bench。

- `workloads/vllm/vllm/uvm_test/expert_reserved_double_slot_event_probe.cu`
  - 双 bank + event 生命周期测试。

- `workloads/vllm/vllm/uvm_test/vllm_generate_once.py`
  - 真实 vLLM single prompt generate benchmark。

## 建议写在简历上的量化结果

可以选择 2-3 条，不要全堆：

1. **搬运性能**
   - “在 20MiB/layer MoE 搬运微基准中，reserved slot path 达到 22.3GiB/s，较 native UVM prefetch 提升 1.75x。”

2. **slot 复用**
   - “实现单 reserved slot 在 20 次 expert remap 中复用，GPU kernel 通过原 managed VA 校验，`kernel_errors=0`。”

3. **真实 vLLM**
   - “在 Qwen3-30B-A3B-FP8 上完成真实 FusedMoE end-to-end 验证，12 个 4MiB slots 支撑每层 2 个 expert 的双层/ring 复用。”

4. **runtime overhead**
   - “通过 per-layer bitmask gate 与 cleanup guard，将 cold expert no-hit p50 延迟从 1.55s 降至 1.11s，接近 1.09s baseline。”

5. **显存窗口**
   - “将 offloaded expert 的动态 GPU staging footprint 约束在固定 48MiB reserved window，而不是随 offloaded expert 数量线性增长。”

更稳妥的最终简历 bullet：

```text
Built a fixed-footprint expert weight paging path for vLLM MoE inference by extending NVIDIA UVM with reserved VIDMEM slots and readonly slot-backed PTE remapping; validated on Qwen3-30B-A3B-FP8 with a 48MiB reusable GPU window, achieving up to 1.75x faster expert transfer than native UVM prefetch and near-baseline p50 latency for cold-expert no-hit paths.
```

## 面试可能被问到的问题与回答

### Q1：为什么 native UVM prefetch 不够？

native UVM prefetch 的目标是把 managed pages 迁移到 GPU resident backing，并更新普通 UVM residency metadata。它不能让用户指定“必须使用哪块 GPU physical backing”，因此无法保证多个 expert 复用固定 48MiB/几百 MiB 窗口。

本项目的关键改动是把 backing source 从普通 UVM PMM 分配改成 driver-owned reserved slot，并通过 PTE remap 让原 managed VA 指向 reserved backing。

### Q2：你们为什么不直接改 PMM，让普通 UVM residency 使用 reserved chunks？

这是长期路线，但风险更高。因为把 reserved chunk 接入 `gpu_state->chunks[]` 后，必须处理：

- PMM lifecycle。
- eviction exclusion。
- VA block destroy/unmap/migration/free。
- dirty/writeback。
- read duplication metadata。

短期实验选择 slot-backed readonly PTE，不填 `chunks[]`，可以更快验证 fixed-footprint 与 vLLM 可行性，同时避免污染普通 PMM hot path。

### Q3：虚拟地址不变是怎么保证的？

vLLM 里的 expert weight tensor pointer 是 managed VA。reserved prefetch 时不改这个 pointer，而是在 UVM driver 内部修改该 VA 对应的 GPU PTE，让它指向 reserved slot 的 GPU physical backing。因此 kernel 仍然读原地址，但实际 physical backing 已经变成 reserved slot。

### Q4：正确性如何保证？

有几层保证：

1. copy 完成后才安装 PTE。
2. slot 复用前先卸载旧 owner PTE 并等待 VA block tracker。
3. vLLM 层使用 CUDA event 记录上一层/上一轮 slot 使用完成。
4. 复用前 wait old event，再 invalidate old owner。
5. 每次 remap 后用 GPU kernel 通过原 managed VA 读取并校验 pattern。
6. strict counters 保证没有偷偷 fallback 到普通 UVM allocation。

### Q5：如果 invalidate 后再次访问会怎样？

当前 strict contract 下，这应当报错。因为 invalidate 表示系统认为该 expert 当前不应再被访问。我们不希望为了容忍错误访问而自动 fallback 到普通 UVM migration，否则会破坏 fixed-footprint contract，也会导致性能不可控。

vLLM 层的解决方式是：如果 expert 被 router 选中，必须在 compute 前 reserved_prefetch；如果没有被选中，不应访问。

### Q6：为什么 PTE 是 readonly？

MoE expert weights 在推理阶段是只读的。readonly PTE 可以简化 dirty/writeback，一旦 GPU kernel 写 expert weight，也能尽早暴露错误。当前版本没有实现 GPU writeback 语义。

### Q7：如何避免 slot 被过早复用？

使用 CUDA event：

```text
本层 compute 提交后 record event
slot 复用前 wait event
event 完成后 invalidate old owner
然后 copy/map new owner
```

双 bank/ring slot 设计进一步减少同步开销：

```text
layer 0 -> bank A
layer 1 -> bank B
layer 2 -> 复用 bank A 前 wait layer 0 event
```

大多数情况下等到复用时旧 event 已经完成，因此不会每层强制 CPU synchronize。

### Q8：vLLM 是否有多个 CUDA stream？event 是否可靠？

实验中 Qwen3 routed expert 路径主要在 stream 0；shared expert 可能有独立 stream，但当前 offloaded routed expert 权重使用路径由 routed expert stream 覆盖。event 的可靠性前提是 record 在真正使用 expert 的 stream 上。

如果模型或配置启用更多 stream，需要扩展为：

- 每个可能访问 expert 的 stream 记录 event。
- slot 复用前等待所有相关 events。
- 调试阶段可用 device synchronize 验证是否有 stream 遗漏。

### Q9：为什么 nvidia-smi 看不出 432MiB 显存下降？

UVM managed allocation 的 accounting 不等于真实 resident bytes。`nvidia-smi` 往往统计进程上下文和 managed allocation 账面占用，不一定反映某些 page 是否已迁到 CPU。因此项目里更依赖：

- driver residency query。
- reserved pool query。
- strict counters。
- region trace。
- fault/migration counters。

简历中建议写 “logical offload 432MiB / fixed 48MiB reserved window”，不要写 “nvidia-smi 下降 432MiB”。

### Q10：这个机制和普通 cache 有什么区别？

普通 cache 通常缓存对象或 tensor；这个机制缓存的是 GPU physical backing window。expert 的 VA 不变，slot 中的 physical contents 可变。它更接近 “GPU physical page cache + explicit PTE remap”。

### Q11：为什么 no-hit 路径之前会慢？

早期每层都会进入通用 `_maybe_prefetch_uvm_expert_weights()`，即使 cold expert 没被选中，也会做：

- active expert CPU 解析。
- region 展开。
- metadata/probe。
- cleanup wrapper。
- trace/stat 判断。

优化后先用 per-layer bitmask 判断是否命中 CPU-offloaded expert，未命中直接返回；同时 active reserved steps 为空时不再跑 cleanup wrapper。

### Q12：这个项目最大的工程难点是什么？

最大的难点是跨越三个层次保持一致：

1. UVM driver 的 PTE/VA block/PMM 生命周期。
2. vLLM FusedMoE 的 router/top-k/expert tensor layout。
3. CUDA stream/event 的异步执行边界。

任何一层错了都会出现：

- stale PTE。
- illegal memory access。
- Xid/MMU fault。
- slot 被过早复用。
- UVM fallback 破坏 fixed-footprint。

### Q13：这个方案的限制是什么？

当前限制：

- 只支持 readonly expert inference。
- 依赖源页 CPU-resident。
- 不支持 HMM/confidential computing/discarded page 等复杂路径。
- 当前是 slot-backed fast path，不是完整 native UVM residency。
- hit 路径有同步搬运开销，需要进一步 async 化和 stream overlap。
- no-hit gate 仍有少量 GPU->CPU scalar sync 成本。

### Q14：下一步如何继续优化？

可以继续做：

- 把 `mask[topk_ids].any()` 融合进 router/top-k kernel，避免额外 `.item()` 同步。
- 做真正 async reserved prefetch，与下一层计算 overlap。
- 用多 bank/ring buffer 支持多 batch 多 request。
- 实现 native reserved residency 深改，让 reserved chunks 安全接入 `gpu_state->chunks[]`。
- 扩展 driver debug：fault 时输出 slot owner、region_id、layer/expert/role。
- 做真实 serving 多 prompt benchmark，而不仅是 single prompt。

## 简历中可以强调的关键词

- NVIDIA UVM
- GPU page table / PTE remapping
- Managed memory
- MoE inference
- vLLM FusedMoE
- Expert offloading
- GPU memory oversubscription
- Fixed-footprint memory window
- CUDA event synchronization
- Driver ioctl ABI
- VA block metadata
- Runtime scheduling / placement policy
- Performance profiling and microbenchmarking

## 不建议写的内容

不要写这些无法严格证明的句子：

- “生产环境吞吐提升 30%。”
- “显存实际下降 432MiB，可由 nvidia-smi 证明。”
- “完整替代 UVM prefetch。”
- “支持所有读写 managed memory。”
- “对 vLLM 性能完全零影响。”

更准确的写法：

- “在 microbenchmark 中 expert transfer 提升 1.75x。”
- “在 cold expert no-hit 场景 p50 接近 baseline。”
- “实现 fixed 48MiB reserved GPU window。”
- “当前支持 readonly expert inference path。”

## 推荐最终简历条目

### 版本 A：偏系统/内核

```text
vLLM/UVM MoE Expert Weight Paging | NVIDIA UVM, CUDA, vLLM, GPU Memory Systems
- Extended NVIDIA UVM with a reserved VIDMEM slot pool and batched expert prefetch ioctl, enabling readonly slot-backed PTE remapping from CPU-resident managed expert pages to reusable GPU physical slots while preserving original expert virtual addresses.
- Integrated the driver path into vLLM FusedMoE via a router-aware reserved executor, per-layer cold expert bitmask, Stage-M init-time CPU offload, and CUDA event based ring-slot lifecycle management.
- Validated on RTX 5090 + Qwen3-30B-A3B-FP8: fixed 48MiB reserved window for dynamic expert paging, zero prefetch failures in tested end-to-end runs, and up to 1.75x faster expert transfer than native UVM prefetch in layer-level benchmarks.
```

### 版本 B：偏 ML Systems

```text
MoE Expert Offloading for vLLM with Fixed GPU Memory Footprint
- Built a runtime expert placement system for vLLM MoE inference that keeps hot experts GPU-resident while paging cold experts through a reusable reserved GPU slot window selected by router top-k results.
- Implemented a UVM slot-backed readonly mapping path so Triton/FusedMoE kernels can access offloaded experts through unchanged managed tensor pointers, avoiding model graph changes.
- Reduced cold-expert no-hit overhead with per-layer bitmask gating and cleanup fast paths; on Qwen3-30B-A3B-FP8, p50 latency stayed near baseline while preserving a bounded 48MiB dynamic expert staging footprint.
```

### 版本 C：短版

```text
Implemented a fixed-footprint MoE expert paging path across NVIDIA UVM and vLLM: CPU-offloaded expert weights are copied into reusable GPU VIDMEM slots and exposed through original managed VAs via readonly PTE remapping. Validated on Qwen3-30B-A3B-FP8 with a 48MiB reserved slot window and up to 1.75x faster expert transfer than native UVM prefetch in microbenchmarks.
```
