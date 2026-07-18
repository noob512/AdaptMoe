# 面向 MoE 推理的 vLLM/UVM 显存复用系统：完整技术报告与面试项目说明

日期：2026-07-18

代码基线：当前工作区中的 vLLM、UVM 用户态扩展与 NVIDIA open UVM kernel module

文档定位：系统技术报告、代码导读、实验总结、简历项目材料与面试答辩底稿

---

## 1. 摘要

本项目面向大规模 MoE 模型推理中的显存竞争问题：模型拥有大量 expert 权重，但每个 token 在每层只访问少数 expert；与此同时，长上下文和高并发又需要持续增长的 KV cache。传统方案让全部 expert 常驻 GPU，会压缩 KV cache 容量；直接依赖 UVM replayable fault 自动迁移，又会把不可预测的缺页和页面搬运带入 prefill/decode 热路径。

项目在 vLLM FusedMoE、Paged KV cache、Python/C++ UVM wrapper 与 NVIDIA UVM kernel driver 之间建立了一套显式、可审计、可失败关闭的显存复用机制：

~~~text
初始化阶段:
  只让 expert 主权重和 KV cache 使用 managed VA
  完成模型加载、KV 分配、编译与 warmup

服务 READY 前:
  将初始 KV target 外的 tail 物理迁移到 CPU
  将空闲 KV blocks 拆分到 GPU-allocatable / CPU-offline 双队列
  解析 hot / warm / cold plan，注册 logical regions 和 envelope 配置
  保护 hot expert；增量模式下不 release cold/warm
  预分配固定大小的 reserved GPU PMM slot
  完整验证后打开 runtime gate

服务阶段:
  scheduler 只从 GPU-allocatable queue 分配 KV blocks
  preemption/swap pressure 触发 KV grow
  每个 pressure wave 全局每层选择一个 expert，无 capacity-safe cold 后才选 warm
  worker 先 release expert，再把 KV delta 搬回 GPU，最后提交 applied target
  router top-k 命中已卸载 expert 时，按 512 KiB envelope 查找 driver regions
  将 CPU expert pages 复制到 reserved slot
  把原 expert VA 的 GPU PTE 映射到 reserved backing
  FusedMoE kernel 继续使用原 tensor pointer
  token 数达到阈值时 warm expert 可跨层异步预取，否则同步恢复
  显式 sleep 且 scheduler idle 时将 KV target 缩回并迁出 tail
~~~

系统最核心的思想是：

> 保持 expert 的虚拟地址和上层 tensor 对象不变，只改变其 GPU 物理 backing；同时用独立的 KV dynamic pool 管理可写 KV pages，保证 CPU-resident KV block 永远不会被 scheduler 分配。

这不是单一 allocator 优化，而是一个跨 vLLM 调度器、模型执行器、内存策略层、用户态 ioctl wrapper 和 UVM 内核驱动的软硬件协同系统。

---

## 2. 项目背景与问题定义

### 2.1 MoE 的稀疏计算与稠密存储矛盾

MoE 模型通常包含很多 expert，但 router 对每个 token 只选择 top-k 个 expert。以一层 128 个 expert、top-k=8 为例：

~~~text
参数存储:
  128 个 expert 权重都需要有稳定地址

单步计算:
  每个 token 只访问 8 个 expert

结果:
  大量 expert 占据显存，但当前 step 并不参与计算
~~~

因此 expert 权重存在明显的冷热差异：

| 分类 | 含义 | 期望放置 |
|---|---|---|
| hot | 高频或明确需要保护的 expert | GPU resident、protected、pinned |
| warm | 有一定访问概率，但可利用跨层提前搬运隐藏延迟 | 根据预算选择性卸载 |
| cold | 低频或计划内明确卸载的 expert | CPU resident，命中时 reserved restore |

### 2.2 KV cache 的容量压力

KV cache 随上下文长度、batch 数和模型层数增长。vLLM 将 KV cache 切成固定 block，由 scheduler 和 BlockPool 分配。长 prompt 会快速消耗 blocks：

~~~text
required_blocks ≈ ceil(tokens / tokens_per_block)
~~~

若全部 expert 常驻 GPU，KV cache 可用预算会下降，表现为：

- 初始化阶段可用 KV memory 为负；
- 固定增大 KV cache tensor 时发生 CUDA OOM；
- 长上下文请求无法获得足够 block；
- 高并发下 scheduler 更容易因无空闲 block 而抢占请求。

### 2.3 为什么不能只使用普通 CPU offload

直接替换 PyTorch parameter 或把 tensor 搬到 CPU，会破坏 vLLM/Triton 已建立的地址、编译和 CUDA graph 假设。运行时重新创建 tensor 还会引入：

- 参数对象和 kernel argument 更新；
- graph capture 地址失效；
- allocator 碎片；
- 跨线程和跨 stream 生命周期复杂度；
- FusedMoE kernel 对新的存储布局产生感知。

本项目要求原 expert tensor 的 <code>data_ptr()</code> 始终稳定。

### 2.4 为什么不能只依赖 UVM 自动迁移

普通 managed memory 在 GPU PTE 缺失时触发 replayable fault，再由 UVM 驱动迁移并重放。它解决了可访问性，却无法提供推理系统需要的确定性：

- fault 发生在真实 GPU 访问点，时机不可控；
- prefill/decode 热路径会出现长尾；
- 多页 fault 和驱动同步可能放大停顿；
- KV 是可写数据，GPU 写 CPU-resident KV page 风险更高；
- 自动迁移重新申请 GPU backing，无法保证固定 staging footprint；
- 出错时容易退化为“机制看似可用，实际依赖 fault 兜底”。

因此本项目采用主动策略：

~~~text
启动期完成注册、初始 KV shrink 与固定 slot 预分配
每个模型 batch 前只在无 in-flight 冲突的边界执行有界 expert wave/KV resize
request compute 中只做有界的 expert slot restore
禁止 CPU-resident KV block 被分配
用 fault guard 验证没有偷偷回退到自动迁移
~~~

---

## 3. 项目目标、非目标与约束

### 3.1 目标

1. 保持 expert managed VA 和 PyTorch tensor pointer 稳定。
2. 启动时不释放 cold/warm；KV 压力增长时按 cold-first、warm-second 逐波释放。
3. 使用固定容量的 reserved slots 限制 runtime expert restore 显存占用。
4. 对原 expert VA 安装只读 slot-backed GPU PTE，使 FusedMoE kernel 无需修改。
5. 每个 pressure wave 全局每层最多选择一个 expert，并将预算用于扩大 KV cache。
6. 通过 GPU/offline 双队列及 desired/applied target 保证逻辑容量与物理 residency 一致。
7. 支持 hot 保护、cold 同步恢复、warm 跨层异步预取。
8. 使用 trace、residency probe、fault guard 和单元测试证明机制真正生效。
9. 所有危险路径默认 fail-fast，不能静默回退到不可控 UVM fault。

### 3.2 非目标

- 不修改 FusedMoE 数学语义和 top-k 算法。
- 不要求 Triton kernel 理解 CPU/GPU placement。
- 不把 readonly slot remap 用于可写 KV cache。
- 不声称当前策略适用于任意请求分布。
- 不声称已经证明 expert 释放出的某个具体 PMM physical chunk 被某个 KV block 原样复用。
- 不把所有模型权重和临时 buffer 都纳入 managed memory。

### 3.3 主要约束

- expert 权重必须是 managed allocation，才能由 UVM driver 管理同一 VA。
- reserved remap 仅适合推理期只读权重。
- 512 KiB envelope 必须满足 base 和 length 对齐要求。
- runtime restore 前 reserved backing 必须已预分配。
- KV shrink 只能在目标 delta 中没有 active block 时提交。
- KV grow 必须先完成所有 worker 的物理迁移，再扩大 applied target。
- CPU-resident KV tail 必须从 allocator 可见集合中隔离。
- batch queue 存在 in-flight forward 时，expert release 与 KV grow 必须延迟到队列排空。

---

## 4. 必要基础概念

### 4.1 vLLM 推理链路

vLLM v1 的简化流程是：

~~~text
EngineCore
  -> Scheduler.schedule()
  -> KVCacheManager.allocate_slots()
  -> ModelExecutor.execute_model()
  -> Worker.execute_model()
  -> GPUModelRunner.execute_model()
  -> model forward
  -> MoE router / top-k
  -> FusedMoE kernel
  -> Scheduler.update_from_output()
~~~

每个 engine step 同时涉及两个关键资源：

- scheduler 为请求分配 KV blocks；
- MoE router 决定本层本 step 需要哪些 expert。

### 4.2 Managed VA

自定义 allocator 最终调用 <code>cudaMallocManaged</code>。CPU 与 GPU 看到同一个虚拟地址，但页面可驻留在 CPU memory 或 GPU VIDMEM。

managed VA 的价值不是“自动迁移”本身，而是让系统能够：

~~~text
保持上层指针不变
在 driver 内显式改变物理 residency
在原 VA 上重写 GPU PTE
~~~

### 4.3 UVM VA block

UVM driver 以 VA block 管理 managed range。VA block 内维护：

- CPU/GPU resident page masks；
- GPU PMM chunks；
- PTE read/write/atomic 权限；
- tracker 和迁移状态；
- fault 与映射元数据。

本项目的 reserved remap 在请求前主动执行 copy 和 PTE install，绕过“GPU fault 后再迁移”的被动路径。

### 4.4 PMM 与物理 chunk

PMM 是 UVM 的 GPU physical memory manager。当前目标 GPU 的 user PMM 支持典型 chunk size：

~~~text
4 KiB
64 KiB
2 MiB
~~~

PMM 没有原生 512 KiB chunk，因此项目采用：

~~~text
1 个 512 KiB logical envelope
  = 8 个 64 KiB physical PMM chunks
~~~

### 4.5 Expert 与 KV 的本质差异

| 属性 | Expert weight | KV cache |
|---|---|---|
| 推理期访问 | 只读 | 持续写入 |
| 地址要求 | tensor VA 稳定 | block id 与 tensor offset 稳定 |
| CPU 状态恢复 | 可从 CPU copy 到 slot | 必须先真实 prefetch GPU |
| GPU 映射 | readonly reserved PTE | 普通 writable UVM residency |
| 容量控制 | 固定 slot_count × slot_bytes | dynamic target_blocks |
| 热路径保护 | top-k fast gate | allocator block-id gate |

这一区分决定了系统必须存在两条不同的数据面。

---

## 5. 整体架构

### 5.1 分层架构

~~~mermaid
flowchart TB
    A[请求与 EngineCore] --> B[Scheduler / KVCacheManager / BlockPool]
    A --> C[GPUModelRunner / FusedMoE]

    B --> D[KV Runtime Policy / Dual Queues]
    C --> E[Stage-M Registration / Incremental Policy]
    C --> F[Reserved Expert Executor]

    A --> M[Preemption/Sleep Adaptive Coordinator]
    M --> D
    M --> E

    D --> G[UVM Residency Registry]
    E --> G
    F --> G

    D --> H[Python UVM Wrapper]
    E --> H
    F --> H

    H --> I[C++ Allocator / ioctl Bridge]
    I --> J[NVIDIA UVM Kernel Driver]

    J --> K[VA Block / GPU PTE]
    J --> L[GPU PMM / Reserved Pool]
~~~

各层职责如下：

| 层 | 主要职责 |
|---|---|
| vLLM Engine 层 | 启动编排、preemption/sleep 检测、跨 worker resize 事务 |
| Scheduler/KV 层 | block 分配、prefix cache、GPU/offline 双队列、desired/applied target |
| FusedMoE 语义层 | plan 分类、全局候选、expert/role 映射、top-k 命中判断 |
| Residency 策略层 | 统一记录 region、state、metadata 和 driver identity |
| Reserved executor | slot bank 选择、cache hit、event、invalidate、ioctl batch |
| Python/C++ bridge | allocator 切换、range prefetch、driver ioctl 编码 |
| UVM kernel | region resolve、PMM release/prealloc、copy、PTE remap |

### 5.2 控制面与数据面

控制面负责“谁应该在哪”：

~~~text
plan:
  hot_plan
  offload_plan
  warm_plan

KV policy:
  desired_target_blocks
  applied_target_blocks
  gpu_free_queue / offline_free_queue
  block lifecycle

adaptive trigger:
  scheduler preemption -> grow
  explicit sleep + idle -> shrink

runtime gate:
  reserved backing 是否完整
  是否允许 serving remap
~~~

数据面负责“页如何真正移动”：

~~~text
Expert:
  pressure wave selects one global expert per logical layer
  GPU backing release
  CPU pages -> reserved PMM chunks
  original VA -> readonly slot-backed PTE

KV:
  ptr + block_id * page_size
  cudaMemPrefetchAsync CPU/GPU
  synchronize
  queue membership 更新
  applied target 最后提交
~~~

### 5.3 两条核心数据路径

~~~mermaid
flowchart LR
    subgraph ExpertPath[Expert 只读路径]
        E1[preemption pressure] --> E2[global cold/warm candidate merge]
        E2 --> E3[logical expert role]
        E3 --> E4[512 KiB envelope cache]
        E4 --> E5[driver release]
        E5 --> E6[CPU resident]
        E6 --> E7[reserved slot copy]
        E7 --> E8[readonly PTE remap]
        E8 --> E9[FusedMoE original pointer]
    end

    subgraph KVPath[KV 可写路径]
        K1[pending resize target] --> K2[worker physical block ranges]
        K2 --> K3[CPU/GPU migration]
        K3 --> K4[scheduler desired target]
        K4 --> K5[GPU/offline queue transition]
        K5 --> K6[applied target]
        K6 --> K7[attention read/write]
    end
~~~

---

## 6. 关键数据模型与状态

### 6.1 Expert 的四级身份

同一个 expert 在系统中有四种不同层次的身份。

| 层次 | 标识 | 用途 |
|---|---|---|
| 策略层 | <code>(layer_id, global_expert_id)</code> | hot/warm/cold 分类 |
| 语义 region | <code>(layer_id, expert_id, role)</code> | trace、状态和 top-k 语义 |
| 物理 envelope | <code>(role, aligned_ptr, 512KiB)</code> | driver registration/release/remap |
| runtime slot | <code>(pool_id, slot_id)</code> | 固定 GPU backing owner |

因此不是简单的 <code>expert_id -> chunk_id</code>，而是：

~~~text
global expert id
  -> global/local expert mapping
  -> role: moe_gate_up / moe_down
  -> tensor slice ptr + bytes
  -> 512 KiB alignment
  -> envelope region_id + driver_region_id
  -> reserved pool slot_id
~~~

### 6.2 Logical expert region

一个 logical region 代表一个 expert 的一个权重 role：

~~~text
weights:{layer_id}:{global_expert_id}:moe_gate_up
weights:{layer_id}:{global_expert_id}:moe_down
~~~

它保存：

- global/local expert id；
- role；
- logical pointer 和 bytes；
- expert_init_class；
- placement；
- protected/pinned/evictable；
- logical driver registration 信息；
- 当前 residency state。

启用 envelope 后，logical region 仍保留，但设置 <code>driver_tag_skip=True</code>，避免把不满足物理粒度的语义 slice 作为最终 driver 管理单位。

### 6.3 Envelope region

512 KiB envelope region id 由 layer、role、对齐后的 base 和 bytes 组成。它才是 driver release 和 reserved remap 的真实物理单位。

每条 envelope record 至少包含：

~~~text
region_id
driver_region_id
ptr
bytes
layer_id
global_expert_id
local_expert_id
role
logical_region_id
logical_ptr
logical_bytes
envelope_granularity_bytes
driver_release_subchunk_64k
~~~

### 6.4 Expert envelope cache

当前实现新增稳定缓存：

~~~text
key:
  (layer_id, global_expert_id, role)

value:
  list[ReservedRegionItem]
~~~

缓存字段位于 [layer.py:1286](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1286)，get-or-build 入口位于 [layer.py:3640](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3640)。

cache 的作用：

1. 增量模式下，expert 第一次 pressure release 时把 logical slice 展开并注册为 envelope items。
2. release 直接使用这些 items 的 driver_region_ids。
3. runtime restore 通过同一 key 取得同一组 items。
4. warm pipeline 也复用同一组物理描述。
5. serving path 不再重复做 align、region 注册和 registry snapshot 拼装。

cache 不是盲目信任旧值。命中后会校验：

- layer/global/local expert id；
- role；
- logical region id；
- logical ptr/bytes；
- 当前 envelope granularity。

任一字段变化就使 cache 失效并重新构建，避免权重重排或配置变化后使用过期 driver identity。

### 6.5 ReservedRegionItem

用户态 reserved executor 的输入对象在 [uvm_expert_reserved_executor.py:115](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:115)。

它把策略语义和 driver 数据合并为一个稳定对象：

~~~text
region_id
driver_region_id
ptr
bytes
layer_id
global_expert_id
local_expert_id
role
metadata
~~~

仅保存 chunk region id 不够，因为 executor 还需要 ptr、bytes、driver identity、layer bank 和 trace metadata。

### 6.6 Slot 状态

内核 slot 的典型状态：

~~~text
EMPTY
  -> RESERVED
  -> LOADING
  -> ASSIGNED
  -> old mapping released
  -> reused by next owner
~~~

用户态还维护：

~~~text
FREE / LOADING / IN_USE / PENDING_REUSE
last_event
owner_regions
owner_layer_id
owner_expert_id
epoch
~~~

<code>PENDING_REUSE</code> 表示当前计算已完成，但在 layer_mod_bank 模式下不立即 invalidate；下一次同 bank 使用时，匹配的 region 可直接 cache hit，不匹配的 stale mapping 才清理。

### 6.7 KV block 状态

vLLM 的 <code>KVCacheBlock</code> 在 [kv_cache_utils.py:108](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py:108) 中维护：

~~~text
block_id
ref_cnt
block_hash
free-list links
is_null
lifecycle_state
free_queue_kind
~~~

本项目使用的 lifecycle：

| 状态 | 含义 |
|---|---|
| never_used | 从未分配给请求 |
| used_free | 曾使用过，目前可复用 |
| cpu_offloaded | 已退出 GPU allocator target，物理上可在 CPU |

dynamic pool 的硬规则是：

~~~text
allocatable(block_id) =
  block_id > 0
  and block_id <= applied_target_blocks
  and ref_cnt == 0
  and not is_null
  and free_queue_kind == "gpu"
~~~

block 0 是 null block，永远不作为普通 KV block 分配。

空闲 block 被拆成两条独立双向队列：

~~~text
free_block_queue:
  GPU resident 且 scheduler 当前可分配

offline_block_queue:
  target 外、CPU resident、scheduler 禁止分配
~~~

<code>dynamic_pool_target_blocks</code> 表示 desired target，
<code>_dynamic_pool_applied_target_blocks</code> 表示已完成迁移并允许 allocator 使用的 target。grow 时先迁物理页再提高 applied；shrink 时要求目标区间无 active block，先隔离空闲 block，再迁出并提交 applied。

---

## 7. 从进程启动到服务 READY 的完整流程

### 7.1 启用 UVM allocator

[env_override.py:381](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/env_override.py:381) 在主要 CUDA allocation 前读取 <code>VLLM_USE_UVM</code> 并调用 allocator 启用入口。

[uvm.py:1179](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:1179) 加载 C++ pluggable allocator，并切换 PyTorch CUDA allocator。

[uvm_allocator.cpp:6694](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp:6694) 的分配入口最终使用 managed allocation。

工程上更推荐窄口径：

~~~text
VLLM_USE_UVM=0
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_KV_CACHE_MANAGED=1
~~~

这样只有 expert 主权重和 KV cache 进入 managed memory，attention、embedding、router、scale、workspace 等保持 native CUDA allocation。

### 7.2 Engine、Worker 与 ModelRunner 初始化

[core.py:79](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:79) 创建 executor、worker、scheduler 与 KV cache manager。

[gpu_worker.py:208](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:208) 初始化 CUDA device、分布式状态和 worker runtime。

[gpu_worker.py:640](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:640) 在延迟启用 reserved path 时先设置：

~~~text
VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY=0
~~~

该 gate 保证 slot backing 未准备好时，serving 不能提交 remap。

### 7.3 Expert 权重选择性 managed 分配

[gpu_model_runner.py:5484](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5484) 加载模型。

[layer.py:1306](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1306) 读取 expert managed 开关。

[fp8.py:730](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/quantization/fp8.py:730) 在 expert allocation scope 中创建 <code>w13_weight</code> 和 <code>w2_weight</code>。

结果是：

~~~text
w13_weight / w2_weight:
  managed VA
  可按 expert 维度计算 slice
  可在 driver 中 release 和 remap

scale / auxiliary tensors:
  默认仍走 native allocation
~~~

### 7.4 Profile 与固定 KV sizing

[core.py:271](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:271) 获取 KV spec 和可用显存。

[gpu_worker.py:337](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:337) 的普通路径执行 memory profile；UVM 路径可以读取固定 <code>VLLM_UVM_KV_CACHE_SIZE_GB</code>，减少 profile 对 managed residency 的扰动。

[gpu_model_runner.py:6377](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6377) 执行 dummy forward 和 compile 准备，但最终 Stage-M registration 被延后到 warmup 之后。

延后的原因：

- warmup 会触碰 expert 和 workspace；
- compile/CUDA graph 可能改变显存峰值；
- 若过早 release，warmup 自身可能把 expert 自动迁回 GPU；
- 最终 placement 应位于“所有初始化访问完成、正式请求尚未开始”的稳定边界。

### 7.5 KV cache 生成与分配

[gpu_model_runner.py:8494](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:8494) 生成每层 KV spec。

[kv_cache_utils.py:1335](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py:1335) 把 spec 和可用内存转换为 tensor config；[kv_cache_utils.py:1269](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py:1269) 生成 scheduler block config。

[gpu_model_runner.py:8160](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:8160) 分配 raw KV tensors；<code>torch.zeros</code> 会实际触碰页面并建立 managed VA。

初始化完成后保存所有 layer/group tensor，供后续通过：

~~~text
physical_ptr = tensor.data_ptr() + block_id * page_size
~~~

计算真实 KV block ranges。

### 7.6 地址审计与 physical range registry

[gpu_model_runner.py:5095](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5095) 对每个 KV tensor 检查：

- 必须是 CUDA tensor；
- 必须 contiguous；
- ptr 与 bytes 有效；
- page size 有效；
- allocation 中可表示的最大 block id 有效。

[uvm_kv_runtime_policy.py:65](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:65) 将结果注册为进程内 physical registry。

[uvm_kv_runtime_policy.py:162](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:162) 可把一组 block ids 合并为连续区间，再展开到所有 layer/group 的真实地址，降低 prefetch 调用数量。

### 7.7 Compile、kernel warmup 与 CUDA graph

在此阶段完成：

- dummy forward；
- Triton/kernel 编译；
- CUDA graph capture；
- sampler warmup；
- runtime workspace 稳定化。

Stage-M 不在 profile 中做最终 placement，而是在这些访问全部完成之后统一执行。

### 7.8 Post-warmup 总编排

总入口是 [gpu_worker.py:732](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:732)。

严格顺序如下：

~~~text
1. worker 计算并同步 offload 初始 target 外的 KV tail
2. Stage-M 解析 plan，注册 logical regions 并保护 hot experts
3. 增量模式下 cold/warm release candidate 暂置为空
4. 根据单层 envelope 最坏需求确定共享 slot budget
5. 预分配并 query reserved backing
6. runtime gate = 1
7. EngineCore 在 READY 前提交初始 applied target 和双队列 partition
~~~

这个顺序是一种启动期事务边界：

- worker 先完成 writable KV tail 的物理迁移；
- scheduler 随后才把同一 tail 放入 offline queue；
- Stage-M 只建立后续增量 release 所需的 plan、region 和 hot 保护；
- slot 在任何 expert release 之前完整预分配，明确固定 runtime footprint；
- 任一步失败都不能打开 runtime gate。

非增量兼容模式仍可在 Stage-M 一次性 release，但本报告后续主线均指
<code>VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1</code>。

### 7.9 KV tail 下调

[gpu_model_runner.py:5184](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5184) 对每个 KV tensor 计算：

~~~text
CPU tail block ids:
  target_blocks + 1 ... max_block_id

tail ptr:
  tensor_ptr + first_cpu_block * page_size

tail bytes:
  number_of_tail_blocks * page_size
~~~

[gpu_model_runner.py:5286](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5286) 批量同步迁移到 CPU，tensor VA 不变。

### 7.10 Stage-M plan 解析

[layer.py:1219](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1219) 解析 <code>offload_plan</code>。

[layer.py:1223](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1223) 合并 <code>hot_plan</code> 与 <code>prefetch_plan</code>。

[layer.py:1233](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1233) 解析 <code>warm_plan</code>。

[layer.py:2895](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2895) 的分类优先级：

~~~text
hot_plan / prefetch_plan -> hot
offload_plan             -> cold
其他                      -> warm
~~~

hot 优先级最高，避免同一 expert 在冲突 plan 中被错误 release。

### 7.11 候选排序与增量预算

[layer.py:2910](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2910) 构造潜在顺序：

~~~text
sorted(cold_pairs) + sorted(warm_pairs)
~~~

该列表在启动时只注册，不消费 release 预算。pressure wave 到来时，[layer.py:3020](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3020) 从中返回当前层下一个 capacity-safe candidate；EngineCore 再跨 worker 合并，保证全局 cold-first。

每个候选必须同时满足：

- 尚未进入 slot-managed 集合；
- 属于当前全局 wave 的 cold 或 warm class；
- 当前 worker 持有合法 local expert slice；
- 该层当前所有已卸载 envelope 加上新 expert 后不超过单 bank slots；
- expert 数和 release bytes 限制允许。

warm 的选择规则：

- 有显式 warm_plan 时，只选择计划中的 warm；
- 没有 warm_plan 时，必须同时启用 warm offload 和 complete expert plan；
- 默认不会因为“其余均为 warm”就把所有 warm 自动卸载。

### 7.12 Stage-M 注册与 pressure wave 重入

[gpu_model_runner.py:6476](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6476) 遍历模型中所有支持 Stage-M 的 FusedMoE module。

[layer.py:6035](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6035) 对本层执行：

1. 检查真实 release 是否配有 reserved restore。
2. 防止同一层重复执行。
3. 刷新 plan 并校验 plan policy。
4. 注册 warm-prefetch module。
5. 记录 Stage-M 前地址图。
6. 预注册所有 local experts。
7. 注册并保护 hot experts。
8. 构造 cold-first、warm-second 的潜在候选。
9. 若是初始增量 Stage-M，将本次 release 列表清空并返回。
10. 若是 pressure wave 重入，只保留 EngineCore 广播的 global expert id。
11. 应用 bank slot、expert 数和字节限制。
12. 对被选 expert/role 注册 logical region并建立 envelope cache。
13. 按 envelope driver ids 执行 release。
14. 成功 release 后更新 logical state和 slot-managed 集合。
15. warm expert 额外加入 offloaded-warm 集合。
16. 写入逐条 trace 与 summary。

真实 driver release 必须同时满足：

~~~text
VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1
~~~

否则直接拒绝 release，避免 serving 时只能依赖普通 UVM fault。

### 7.13 512 KiB envelope 构建与缓存

[layer.py:3299](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3299) 从 fused tensor 第一维按 local expert 计算 slice：

~~~text
slice_bytes = tensor_total_bytes / local_num_experts
ptr = tensor.data_ptr() + local_expert_id * slice_bytes
~~~

[layer.py:3390](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3390) 执行：

~~~text
envelope_start = align_down(slice_start, 512KiB)
envelope_end   = align_up(slice_end, 512KiB)
~~~

严格模式要求 logical ptr 和 bytes 本身满足粒度对齐。若 round 后超出 role tensor allocation，也会跳过，避免注册 driver 不可接受的部分 envelope。

[layer.py:3640](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3640) 完成 get-or-build。增量模式下，它通常在该 expert 第一次 pressure release 时发生：

~~~text
cache hit:
  校验 logical identity
  绑定当前 metadata
  直接返回稳定 items

cache miss:
  build envelope records
  register driver regions
  构造 ReservedRegionItem
  写入 cache
~~~

### 7.14 Pressure wave 中的 Driver release

[uvm.py:2625](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:2625) 把 envelope driver ids 编码为 release ioctl。

512 KiB 模式设置 <code>SUBCHUNK_64K</code>。内核 release 路径：

~~~text
验证 region 是 expert weight
验证 base/length 为 512 KiB 对齐
确保 CPU pages 可用
按 64 KiB 子块清 GPU residency/PTE
摘除并释放 GPU PMM chunks
保留原 managed VA 与 CPU 内容
~~~

### 7.15 Reserved slot 预分配

[layer.py:2971](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2971) 按真实 slice 对齐后的 envelope 数计算单个 wave 的最小 bank 容量。[gpu_model_runner.py:6725](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6725) 汇总所有 MoE 层并确定 slot_count；增量模式默认共享预算为 128 slots，且不能超过 driver ABI 上限。

[gpu_model_runner.py:6789](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6789) 读取 pool、slot_count、slot_bytes，先 query，再 prealloc，再 query 验证。

[uvm.py:2538](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:2538) 是 Python ioctl wrapper。

[uvm.c:828](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:828) 为每个 slot：

1. 校验参数；
2. 选择 PMM chunk size；
3. 计算 physical_bytes 和 chunk_count；
4. 检查同 pool/slot 是否已有兼容 backing；
5. 在锁外执行 PMM allocation；
6. 再次加锁处理并发插入；
7. 写入 slot 元数据并 retain GPU；
8. 返回 query 信息。

512 KiB slot 的典型结果：

~~~text
slot_bytes      = 512 KiB
chunk_size      = 64 KiB
chunk_count     = 8
physical_bytes  = 512 KiB
~~~

### 7.16 Runtime gate

[gpu_worker.py:811](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:811) 只在预分配结果满足完整配置后设置：

~~~text
VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY=1
~~~

失败时：

- gate 保持 0；
- 默认抛出异常终止启动；
- 不允许 serving 进入不完整 slot pool；
- 不允许 silent fallback。

### 7.17 Scheduler 同步 KV target

[core.py:321](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:321) 在 READY 前读取统一 target。

[uvm_kv_runtime_policy.py:587](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:587) 更新 desired target；[uvm_kv_runtime_policy.py:1181](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1181) 在确认物理迁移完成后，把 delta block 元数据迁入 offline queue 并提交 applied target。

至此，worker 的物理 KV residency、scheduler applied target 和 BlockPool 双队列一致，服务才进入 READY。cold/warm 仍在 GPU，等待 serving pressure wave 决定是否 release。

---

## 8. 一层 expert 的完整示例

假设：

~~~text
layer_id = 12
local/global experts = 0,1,2,3,4,5,6,7

hot_plan     = {0,1}
offload_plan = {6,7}
warm_plan    = {4}

envelope_bytes = 512 KiB
~~~

### 8.1 分类

~~~text
expert 0,1 -> hot
expert 6,7 -> cold
expert 2,3,4,5 -> warm
~~~

显式 warm_plan 只允许 warm expert 4 进入 release 候选，因此：

~~~text
潜在 class order = cold 6, cold 7, warm 4

READY 前:
  release = none

pressure wave 1:
  select expert 6

pressure wave 2:
  select expert 7

pressure wave 3:
  所有 worker 已无可 release cold 后，select warm expert 4
~~~

expert 2、3、5 保持原 residency，不会因为默认属于 warm 就自动 release。

若模型有 48 个 MoE 层，wave 1 不是只处理 layer 12，而是 EngineCore 从所有 worker 收集每层 cold candidate，并全局为每个逻辑层选择一个 id。因此一轮最多选择 48 个 logical experts，而不是每个 worker 各自选择 48 个不同 experts。

### 8.2 Hot 处理

expert 0、1 的两个 role 被注册为：

~~~text
GPU_RESIDENT
protected=True
pinned=True
evictable=False
~~~

若开启 hot prefetch，还会真实 prefetch 到 GPU，而不只是更新 metadata。

### 8.3 Expert 6 的 logical slices

假设：

~~~text
w13_weight base = 0x700000000000
每个 expert w13 slice = 1 MiB

w2_weight base = 0x710000000000
每个 expert w2 slice = 512 KiB
~~~

expert 6：

~~~text
moe_gate_up:
  ptr   = 0x700000600000
  bytes = 1 MiB

moe_down:
  ptr   = 0x710000300000
  bytes = 512 KiB
~~~

logical regions：

~~~text
weights:12:6:moe_gate_up
weights:12:6:moe_down
~~~

### 8.4 Envelope 展开

<code>moe_gate_up</code> 展开为两个 512 KiB regions：

~~~text
chunk A:
  ptr = 0x700000600000
  bytes = 0x80000

chunk B:
  ptr = 0x700000680000
  bytes = 0x80000
~~~

<code>moe_down</code> 展开为一个：

~~~text
chunk C:
  ptr = 0x710000300000
  bytes = 0x80000
~~~

cache：

~~~text
(12, 6, moe_gate_up) -> [A, B]
(12, 6, moe_down)    -> [C]
~~~

### 8.5 Pressure wave 1 的 Release 判定

初始 Stage-M 不为 expert 6 执行 release。第一次 KV preemption pressure 到来后：

~~~text
worker candidate:
  layer 12 -> cold expert 6

EngineCore merge:
  (layer 12, global expert 6)

broadcast execute:
  持有 expert 6 或其 TP shard 的 workers 进入增量 Stage-M
~~~

此时才从 logical slice 建立/复用 A、B、C，并提交它们的 driver_region_ids。release 结果被写入 slot-managed 集合：

~~~text
_uvm_slot_managed_global_expert_ids += {6}
~~~

EngineCore 还会检查 layer 12 的 worker result 中是否真的出现 <code>released_experts=[6]</code>。若没有，本轮 KV grow 失败关闭，不提交 applied target。

第二个 pressure wave 对 expert 7 执行相同流程。只要任一 worker/层仍报告 capacity-safe cold，warm 4 就不能进入 wave。

第三个 pressure wave 中，warm expert 4 成功 release 后：

~~~text
_uvm_slot_managed_global_expert_ids += {4}
_uvm_offloaded_warm_global_expert_ids += {4}
~~~

第二个集合使 expert 4 有资格参加下一层 warm pipeline。每次新增 expert 前还会重新计算本层全部 slot-managed experts 的 envelope 总数，确保单 bank 可容纳后续 restore。

### 8.6 Runtime 命中

若 router top-k 包含 expert 6：

~~~text
topk id 6
  -> 命中 slot-managed GPU mask
  -> 构造两个 logical role items
  -> 查 cache 得到 A、B、C
  -> reserved executor 选择 slots
  -> CPU pages copy 到 slot backing
  -> 原 w13/w2 VA 安装 readonly PTE
  -> FusedMoE 使用原 pointer 计算
~~~

如果 cache 校验失败，才 fallback 重新展开；正常 serving 不重复注册 envelope。

---

## 9. Serving 阶段完整流程

### 9.1 Engine step

[core.py:892](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:892) 的普通 step 执行：

~~~text
schedule
  -> inspect scheduler_output.preempted_req_ids
  -> optional expert wave + worker KV grow + applied target commit
  -> execute_model
  -> wait model result
  -> scheduler update
~~~

vLLM v1 在这条路径没有独立 swap event，因此项目把 scheduler preemption 作为 KV 不足/swap pressure 信号。被 preempt 的请求本 step 已退回等待态，grow 后在后续 step 恢复。

[core.py:928](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:928) 的 batch queue 可能已有 GPU forward 在运行。若此时发现 pressure：

~~~text
记录 pending grow target
  -> 暂停继续 schedule 新 batch
  -> 等待并消费已有 futures
  -> batch queue 为空
  -> 执行全局 expert wave 和 KV grow
  -> 恢复 schedule
~~~

这避免 current forward 正在读取 expert 权重时，另一个控制路径释放同一 backing。

### 9.2 Scheduler 计算 token 与 KV 需求

[scheduler.py:197](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py:197) 不把 prefill/decode 写成两套调度逻辑，而是统一计算：

~~~text
num_new_tokens =
  num_tokens_with_spec
  + output_placeholders
  - num_computed_tokens
~~~

先调度 RUNNING 请求，再处理 WAITING/RESUMED 请求。两条路径最终都调用 <code>KVCacheManager.allocate_slots()</code>。

### 9.3 KV slot 分配

[kv_cache_manager.py:207](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py:207)：

1. 释放 sliding-window 已跳过 blocks。
2. 合并原 computed tokens 与 prefix-cache hits。
3. 计算本轮所需 token slots。
4. 计算需要新增多少 blocks。
5. 询问 UVM policy 是否允许分配。
6. 比较所需数量与 target 内 free blocks。
7. touch prefix-hit blocks，防止被驱逐。
8. 从 BlockPool 获取新 blocks。
9. 缓存已完整填充的 blocks。

dynamic pool 不替换 vLLM 原有 block table、ref-count 和 prefix hash 语义，但把原 free list 拆成 GPU-allocatable 与 CPU-offline 两条队列。普通分配只从 GPU queue 取，free count 直接读取该队列长度；target resize 只迁移 delta 区间的 queue membership，不再为每次 allocation 扫描全部 blocks。

[block_pool.py:186](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:186) 的 prefix lookup 会先刷新 target，并同时检查 applied target 和 queue membership。offline cached block 保留 hash metadata，但按 cache miss 处理；grow 完成并 promote 回 GPU queue 后才可再次命中。

常规 allocation 不再调用 residency 分类器扫描全部 free blocks。[block_pool.py:316](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:316) 直接从 GPU queue 的队头取 K 个 block；offline queue 永远不作为 fallback。

### 9.4 Model forward 到 MoE top-k

[gpu_model_runner.py:2999](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:2999) 准备 batch、positions 和 KV metadata 后执行模型。

[layer.py:2466](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2466) 的 <code>select_experts()</code> 获得 <code>topk_ids/topk_weights</code>。

项目把 restore hook 放在 top-k 已知、expert kernel 尚未启动的位置。这是最晚仍能安全主动搬运、同时最早能知道准确 active experts 的时刻。

### 9.5 Slot-managed fast gate

top-k 先与 GPU bool mask 相交：

~~~text
mask[expert_id] = expert 是否真实 release 成功
hit = any(mask[topk_ids])
~~~

只有 hit 才进入 Python 侧 active pair 构造和 reserved batch。没有已卸载 expert 的层或 step 快速返回。

slot-managed mask 的来源不是 plan 本身，而是 release 成功集合。这样避免：

- plan 里有 expert，但 release 失败；
- runtime 错误认为它必须 restore；
- metadata 与真实物理状态不一致。

### 9.6 Logical item 到 cached envelope

[layer.py:3870](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3870) 对 logical items 查 envelope cache并去重。

一个 logical role 只有当全部 envelope 成功，才在结果归约时记为成功。部分 chunk 成功不能让上层把整个 expert 视为可计算。

### 9.7 Reserved executor

[uvm_expert_reserved_executor.py:589](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:589)：

1. 过滤非法 items。
2. 计算最大 required_slot_bytes。
3. query 并验证预分配 slots。
4. 根据 layer id 选择 bank。
5. 检查 bank 中是否已有同 driver_region_id、ptr、bytes。
6. 保留 warm cache hits。
7. invalidate 非 desired stale slots。
8. 为剩余 items 分配 slots。
9. 必要时确保 driver region identity。
10. 构造 assignments。
11. 批量提交 reserved-prefetch ioctl。
12. compute 前按策略同步。
13. 更新 slot owner、state、epoch 与 active-layer 集合。

### 9.8 Kernel copy 与 remap

[uvm.c:11530](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:11530) 是 batch ioctl 总入口。

[uvm.c:9735](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:9735) 安装单条 mapping：

~~~text
resolve region and slot
  -> release old slot owner mapping
  -> walk target VA blocks
  -> copy CPU pages to reserved chunks
  -> install readonly GPU PTE
  -> write new owner metadata
~~~

[uvm_va_block.c:11151](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11151) 完成 copy-and-map。

[uvm_va_block.c:7180](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:7180) 最终写入指向 reserved physical chunk 的 4 KiB PTE。

映射权限为 GPU read-only，符合 expert 推理权重只读语义。

### 9.9 FusedMoE compute

[layer.py:7416](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:7416) 和 [layer.py:7576](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:7576) 分别覆盖 chunked 与普通 forward。

kernel 仍接收原始：

~~~text
w13_weight
w2_weight
~~~

上层完全不知道这些 VA 当前指向 ordinary UVM backing 还是 reserved slot backing。

### 9.10 Cleanup

[uvm_expert_reserved_executor.py:1161](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:1161) 在 layer compute 后：

- 在当前 CUDA stream 记录 event；
- 等待 kernel 完成后才允许 slot reuse；
- dynamic policy 可立即 invalidate；
- layer_mod_bank policy 将 slot 标记为 PENDING_REUSE；
- backing 始终保留，只改变 mapping owner。

这避免 kernel 尚未读完时另一个 layer 覆盖同一 slot。

---

## 10. Warm expert 跨层流水线

### 10.1 设计动机

cold expert 只能在本层 top-k 确认后同步 restore，延迟位于关键路径。warm expert 具有一定访问概率，可以利用前一 MoE 层计算时间提前搬运。

### 10.2 Module registry

[layer.py:2848](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2848) 按进程、模型 layer sequence 和 layer id 保存 weak module registry。

[layer.py:2857](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2857) 根据 prefetch distance 查找后续 MoE 层。

### 10.3 Banked slots

layer_mod_bank 使用：

~~~text
bank_id = layer_id % bank_count
~~~

至少两个 bank 时：

~~~text
当前层 bank:
  保证当前 compute 的 mappings 不被覆盖

下一层 bank:
  异步装入 warm expert
~~~

若 source layer 与 target layer 落在同一 bank，本次 warm prefetch 跳过，优先保证正确性。

### 10.4 时序

~~~mermaid
sequenceDiagram
    participant R as Router Layer L
    participant E as Reserved Executor
    participant G as GPU Compute
    participant N as Layer L+1

    R->>E: restore current cold hits
    E-->>R: sync before current compute
    R->>E: if token threshold met, async prefetch warm experts of L+1
    R->>G: run FusedMoE layer L
    E-->>E: copy/remap into next-layer bank
    G-->>N: enter layer L+1
    N->>E: request actual top-k experts
    E->>E: reuse matching warm slots, invalidate stale slots
    E-->>N: synchronize outstanding copy
    N->>G: run FusedMoE layer L+1
~~~

### 10.5 Warm cache hit

executor 以三个条件判断复用：

~~~text
driver_region_id 相同
ptr 相同
bytes 相同
~~~

命中时不重复 copy/ioctl assignment，但目标层 compute 前仍同步，确保上一层异步 copy 完成。

### 10.6 未命中的 warm

若预取的 warm expert 没被下一层 top-k 选中：

- 不会影响数学结果；
- slot 在同 bank 下一次使用时变成 stale；
- executor 等待旧 event 后 invalidate；
- backing 被新 owner 复用。

### 10.7 Token-aware 同步/异步选择

[layer.py:6941](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6941) 读取当前层 <code>hidden_states.shape[0]</code>：

~~~text
num_tokens < VLLM_UVM_EXPERT_WARM_ASYNC_MIN_TOKENS:
  跳过 next-layer warm prefetch
  top-k 真实命中时 reserved restore 使用 async_=False
  compute 前完成同步搬运

num_tokens >= threshold:
  允许跨层 warm async prefetch
  目标层命中 cached slot 时只等待尚未完成的 copy
~~~

阈值表达的是 overlap 机会，而不是 expert 热度：短 decode batch 即使 expert 被分类为 warm，也可能没有足够计算时间隐藏 copy；大 prefill/batch 才值得占用下一 bank 和 copy engine。

---

## 11. KV block 从零开始：分配、复用与限制

### 11.1 两条 free queues 与其他索引

[kv_cache_utils.py:108](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py:108) 的 <code>KVCacheBlock</code> 增加 <code>free_queue_kind</code>，取值为 <code>gpu</code>、<code>offline</code> 或 <code>None</code>。

[block_pool.py:146](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:146) 初始化并维护：

| 结构 | 作用 |
|---|---|
| free_block_queue | applied target 内、GPU resident、ref_cnt=0 的可分配 blocks |
| offline_block_queue | applied target 外、CPU resident、ref_cnt=0 的不可分配 blocks |
| req_to_blocks | request 当前持有的 active blocks；不在任一 free queue |
| cached_block_hash_to_block | prefix hash 索引，可暂时指向 offline block |
| residency registry | block/group 到 CPU/GPU 物理状态和 trace |

block 0 是 null block，<code>free_queue_kind=None</code>，永远不作为普通 block 分配。

### 11.2 为什么两条队列仍使用双向链表

prefix cache hit、target resize 都可能从队列中间删除一个 block。block 自身保存 prev/next，使以下操作为 O(1)：

- prefix hit 后 touch；
- shrink 时 GPU queue -> offline queue；
- grow 时 offline queue -> GPU queue；
- cached block 被新用途驱逐。

队列内部仍维护 LRU/驱逐顺序，但不再用一条链表同时承载“可分配”和“仅保留元数据”两种含义。

### 11.3 新请求需要多少 block

[single_type_kv_cache_manager.py:66](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/single_type_kv_cache_manager.py:66)：

~~~text
num_required_blocks = ceil(num_tokens / block_size)

num_new_blocks =
  num_required_blocks
  - prefix-hit blocks
  - request already-owned blocks
  + evictable prefix-hit blocks that must leave a free queue
~~~

[kv_cache_manager.py:207](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py:207) 先计算需求，再用 GPU queue 的 O(1) free count 做 admission。

### 11.4 Desired target、applied target 与 queue membership

系统区分：

~~~text
desired_target:
  控制面希望达到的容量

applied_target:
  已经完成安全迁移、allocator 可以遵守的容量

allocatable(block):
  0 < block_id <= applied_target
  and ref_cnt == 0
  and free_queue_kind == "gpu"
~~~

[uvm_kv_runtime_policy.py:883](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:883) 提供统一 block-id gate；[block_pool.py:542](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:542) 的 free count 直接返回 GPU queue 长度。

### 11.5 常规分配不再扫描全部 blocks

[block_pool.py:316](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:316) 在 dynamic pool 模式下直接：

~~~text
free_block_queue.popleft_n(K)
  -> free_queue_kind=None
  -> ref_cnt += 1
~~~

offline queue 不参与候选分类，也不存在 CPU fallback。复杂度为 O(K)，不是 O(total_free_blocks)。

### 11.6 Prefix cache 复用

[block_pool.py:186](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:186) 在 hash lookup 前刷新 target，并检查：

~~~text
block_id <= applied_target
若 ref_cnt == 0，则 free_queue_kind 必须为 gpu
~~~

offline block 可以保留 hash，便于 grow 后重新命中，但当前按 cache miss 处理。合法 GPU prefix hit 调用 <code>touch()</code>，从 GPU queue 中间 O(1) 删除并增加 ref_cnt。

### 11.7 释放与复用

[block_pool.py:461](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:461) 在 ref_cnt 归零时根据 applied target 路由：

~~~text
block_id <= applied_target:
  append gpu_free_queue tail

block_id > applied_target:
  append offline_queue tail
  lifecycle = cpu_offloaded
~~~

因此 block 1 使用完后只会回到 GPU queue 尾部；它不会排在 offline blocks 9..16 后面。prefix hash 可以继续保留，直到该 block 被新内容覆盖。

### 11.8 Shrink

主线 shrink 由显式 sleep 触发，并要求 scheduler idle。目标 delta 为：

~~~text
new_target < block_id <= old_applied_target
~~~

[core.py:1058](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:1058) 先拒绝存在 unfinished requests 的 shrink；worker [gpu_model_runner.py:5374](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5374) 再把所有 layer/group 的真实 ranges 同步迁到 CPU。所有 worker 成功后，[uvm_kv_runtime_policy.py:1181](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1181) 才迁 queue membership、记录 CPU lifecycle 并降低 applied target。

若 policy 发现 delta 中仍有 active block，则保持 <code>desired &lt; applied</code> 并返回 <code>active_blocks_must_drain_before_shrink</code>；不会把 active block 强行迁出，也不会假装 shrink 已提交。

### 11.9 Grow

preemption pressure 下的 grow 顺序为：

~~~text
1. 计算 pending new target
2. 全局选择本 wave 每层一个 cold/warm expert
3. worker release 选定 expert
4. worker prefetch KV delta 到 GPU并同步
5. 所有 workers 返回 success
6. scheduler 更新 desired target
7. offline delta prepend 到 GPU queue
8. applied target = new target
~~~

若 expert release、任一 worker migration 或逐层 release 核对失败，第 6-8 步均不发生。allocator 始终读取旧 applied target。

### 11.10 具体示例

假设：

~~~text
block ids = 0..16
block 0 = null
applied_target = 8

gpu_free_queue:
  [1,2,3,4,5,6,7,8]

offline_queue:
  [9,10,11,12,13,14,15,16]
~~~

请求 A 分配 block 1：

~~~text
gpu_free_queue = [2,3,4,5,6,7,8]
block 1: ref_cnt=1, free_queue_kind=None
~~~

block 1 释放：

~~~text
gpu_free_queue = [2,3,4,5,6,7,8,1]
offline_queue  = [9,10,11,12,13,14,15,16]
~~~

下一次分配直接得到 block 2，不扫描 9..16。等 2..8 使用后，block 1 按 GPU queue 的 LRU 顺序自然复用。

若 target 从 8 grow 到 10：

~~~text
worker prefetch 9,10 -> GPU
offline remove 9,10
gpu prepend 9,10

gpu_free_queue = [9,10,2,3,4,5,6,7,8,1]
offline_queue  = [11,12,13,14,15,16]
applied_target = 10
~~~

新 promote 的 fresh blocks 放在 GPU queue 前部，优先于已使用后释放的 warm blocks。

---

## 12. 主要创新点

### 12.1 启动期与 pressure wave 的双事务边界

创新不是简单增加 offload API，而是找到正确生命周期边界：

~~~text
启动事务:
  warmup 完成
  -> KV tail 迁出
  -> scheduler 双队列/初始 applied target
  -> plan/region/hot 注册
  -> slot prealloc
  -> gate open
  -> READY

在线 grow 事务:
  preemption detected
  -> global expert candidate merge
  -> expert release wave
  -> worker KV prefetch
  -> queue promotion
  -> applied target commit
~~~

两者都发生在 model compute 之外。batch queue 有 in-flight work 时先排空，因此在线自适应不等于在 kernel 执行中任意迁移。

### 12.2 选择性 managed allocation

项目没有把整个模型交给 UVM，而是通过 local memory pool 只覆盖 expert 主权重和 KV cache。这减少了 fault surface、trace 噪声和非目标对象的 residency 干扰。

### 12.3 Expert 语义层与 driver 物理层解耦

上层按 expert/role 做策略，底层按 envelope 做 release/remap。二者通过 stable metadata 关联：

~~~text
策略可读性:
  layer/expert/hot/warm/cold

物理可执行性:
  aligned ptr/bytes/driver_region_id
~~~

这种双层模型既保留模型语义，又满足 UVM 对齐与 PMM 粒度。

### 12.4 Stable envelope cache

Stage-M 建立 <code>(layer, expert, role) -> ReservedRegionItem[]</code>，serving 直接复用。它减少：

- Python align 计算；
- registry lookup；
- 重复 driver registration；
- warm pipeline 的重复 item 构造；
- Stage-M 与 runtime 物理 identity 不一致风险。

同时通过 ptr/bytes/granularity 校验实现安全失效。

### 12.5 512 KiB logical / 64 KiB physical 的粒度桥接

2 MiB envelope 会牵连过多邻近权重，直接新增 512 KiB PMM chunk 又需要侵入 PMM。项目用 8 个原生 64 KiB chunks 表示一个 512 KiB logical slot，在不修改 PMM 支持矩阵的情况下减小放大效应。

### 12.6 原 VA 上的 readonly slot-backed PTE remap

恢复 expert 时不重建 tensor，而是：

~~~text
CPU source pages
  -> copy to driver-owned reserved chunks
  -> original expert VA installs readonly PTE
~~~

FusedMoE/Triton kernel、parameter 对象和 graph capture 均无需感知 backing 已更换。

### 12.7 固定显存 footprint 的 reserved pool

runtime staging 显存被严格限制为：

~~~text
slot_count * slot_bytes
~~~

slot backing 在 READY 前一次性预分配，runtime 只复用，不允许 miss path 懒分配。这比普通 UVM fault 时动态申请 GPU backing更可预测。

### 12.8 Hot/warm/cold 三级放置与全局增量 cold-first

相比简单 hot/cold：

- hot 明确保护；
- 启动时 cold/warm 都不 release；
- pressure wave 全局每层最多选择一个 cold；
- 所有 worker 均无 capacity-safe cold 后才允许 warm；
- warm 只在显式 plan 或完整策略允许时进入候选；
- warm 成功 release 后进入独立集合，支持跨层 pipeline。

它把容量收益和访问风险分成三档，而不是二元决策。

### 12.9 Layer-mod bank 与跨层 warm overlap

按 layer id 划分 slot banks，使当前层 compute 与下一层 warm copy 使用不同 backing 集合。cache hit 时保留 mapping，miss 时只 invalidate stale owner，从而实现：

~~~text
current layer compute
  overlap
next layer warm expert copy/remap
~~~

### 12.10 KV dynamic pool 的逻辑/物理/队列三重约束

只修改 scheduler block 数不够，只搬物理页也不够。项目同时维护：

- desired/applied block-id target；
- GPU-allocatable/offline 双队列；
- O(1) free count 与直接队头分配；
- lifecycle；
- physical pointer ranges；
- CPU/GPU prefetch；
- resize generation 和 trace。

这使 CPU-resident writable KV tail 不会被误分配。

### 12.11 Fail-closed runtime gate 与 wave commit

启动 gate 只在 slot prealloc 和 backing query 完整后打开，因此后续任意 incremental release 都已经拥有可恢复路径。每个 wave 还要求 candidate RPC、逐层 release 核对和所有 worker KV migration 全部成功；失败时不提交 applied target。restore 失败默认 fail-fast，不能静默回退到普通 UVM fault。

### 12.12 可验证的 residency 系统

项目不仅记录“API 调用成功”，还记录：

- logical/physical region；
- driver registration；
- released bytes；
- reserved backing bytes；
- selected block ids；
- active expert ids；
- slot owner/epoch；
- physical migration ranges；
- managed fault；
- Xid/MMU/fatal records。

因此可以区分“metadata 写了 CPU”与“页面真实在 CPU”，“调用了 prefetch”与“请求确实无 fault”。

### 12.13 KV 双队列与 O(1) free count

[block_pool.py:316](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:316) 在 dynamic pool 开启时直接从 GPU queue 的队头取 K 个 block；[block_pool.py:461](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:461) 在释放时根据 applied target 把 block 送回 GPU queue 或 offline queue；[block_pool.py:569](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:569) 只对 target delta 做跨队列迁移。

复杂度从多次全队列扫描变为：

~~~text
普通分配 K blocks: O(K)
释放 K blocks:       O(K)
free count:          O(1)
target resize:       O(abs(new_target - old_target))
~~~

grow 时 promote 的 fresh blocks 放到 GPU queue 前部，优先于刚释放的 warm blocks；shrink 后仍 active 的 block 不允许提交 applied target，必须等请求结束后重试，避免出现“逻辑 offline、物理仍被请求写入”的状态。

### 12.14 由 KV 压力驱动的增量 expert offload

启用增量模式后，初始 Stage-M 在 [layer.py:6035](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6035) 只完成 logical/envelope 注册、hot 保护和 reserved pool 准备，不 release cold/warm expert。

当 scheduler 发生 preemption 时，系统把它视为 vLLM v1 路径中的 swap pressure：[core.py:695](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:695) 先计算 KV grow target，再通过 worker RPC 触发一个 expert wave。[gpu_model_runner.py:6619](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6619) 先让各 worker 返回每层一个 capacity-safe 候选，[core.py:641](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:641) 再按逻辑层合并，并选择唯一、确定性的 global expert id：

~~~text
任一 worker 的任一 MoE 层仍有可 release cold:
  所有层本轮 allow_warm=False
  每个逻辑层全局选择 1 个 cold expert id

全部可 release cold 已耗尽:
  所有层本轮 allow_warm=True
  每个逻辑层全局选择 1 个 warm expert id
~~~

EngineCore 把同一份 <code>layer_name -&gt; global_expert_id</code> 选择广播到所有 worker。TP rank 对同一 expert 的 shard 一致执行；EP 中只有持有该 global expert 的 worker 实际 release，避免每个 EP worker 各选一个不同 expert。48 个 MoE 层时，一个 wave 全局最多选择 48 个 expert。单层入口 [layer.py:3039](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3039) 重新打开一次 Stage-M release，但跳过已经进入 slot-managed 集合的 expert，并在 release 前检查该层 bank 是否还能容纳所有已卸载 expert 的 envelope chunks。

wave 返回后，EngineCore 逐层核对所选 global expert 是否真实出现在至少一个 worker 的 <code>released_experts</code> 中。候选查询成功但 release 集合缺层时按失败处理，KV grow 不会继续。

reserved slots 不按 48 层永久各分一份。层按顺序执行，slot bank 可跨层复用；容量按“单层同时可能恢复的全部 envelope 数”计算。[layer.py:2971](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2971) 使用真实 slice VA 的 align-down/align-up 结果计算 chunk 数，[gpu_model_runner.py:6725](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6725) 在 READY 前统一扩展 slot budget，默认上限受 driver ABI 的 128 slots 约束。

### 12.15 Token-aware warm 搬运

[layer.py:6941](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6941) 使用当前层 <code>hidden_states.shape[0]</code> 作为本轮 num_tokens：

~~~text
num_tokens < warm_async_min_tokens:
  不启动 next-layer warm pipeline
  真实 top-k 命中时 async_=False，同步 restore

num_tokens >= warm_async_min_tokens:
  允许下一层 warm expert 使用异步 bank 预取
  当前层 compute 为 copy/remap 提供 overlap 窗口
~~~

这样短 decode step 不为几乎没有 overlap 空间的 warm copy 占用另一 bank；大 prefill 或大 batch 才启用异步 pipeline。

### 12.16 Preemption grow 与 sleep shrink

[core.py:478](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:478) 的 grow 顺序固定为：

~~~text
scheduler preemption detected
  -> 每层一个 expert 的 cold/warm wave
  -> worker 物理 prefetch KV delta 到 GPU
  -> 所有 worker 成功
  -> scheduler 提交 desired/applied target
  -> offline blocks promote 到 GPU queue
~~~

任一 expert worker 或 KV migration worker 失败时，不提交 applied target，默认 fail-fast。

batch queue 模式下可能已有模型 forward 在 GPU 上运行，不能立即 release 权重。[core.py:759](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:759) 记录 pending grow，停止继续填充 batch queue，排空已有 futures 后再执行 expert wave 和 KV grow。

[core.py:1058](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:1058) 在显式 EngineCore sleep 时执行反方向 resize。shrink 只允许 scheduler idle：worker 同步 offload KV delta 到 CPU，成功后 scheduler 才把 blocks 移入 offline queue。这里的 sleep 是显式 sleep API，不等同于一次普通 decode 空闲间隙。

---

## 13. 关键工程难题与解决方法

### 13.1 Allocator 启用时机

难题：allocator 切换太晚，expert/KV 已经是 native CUDA allocation，后续 driver 无法按 managed VA 管理。

解决：

- 全局路径在主要 CUDA allocation 前启用；
- 窄口径路径用 local memory pool 包裹精确 allocation scope；
- 对权重与 KV 地址做审计；
- 非目标对象保持 native allocation。

### 13.2 Profile 与 warmup 会污染最终 placement

难题：dummy forward、compile 和 graph capture 会触碰 expert/pages，过早 offload 会被重新拉回 GPU。

解决：

- profile 只测量和编译；
- 固定 KV size 减少 UVM profile 不确定性；
- 将 Stage-M 注册、hot 保护和 slot prealloc 移到 post-warmup；
- cold/warm release 延迟到 serving pressure boundary；
- 在 READY 前设置明确 barrier。

### 13.3 模型语义粒度与 driver 粒度不匹配

难题：expert slice 以 tensor shape 切分，driver release 要求对齐的完整 range。

解决：

- 保留 logical region；
- 生成 512 KiB aligned envelope；
- strict logical alignment；
- tensor bounds 检查；
- physical driver tag 只注册 envelope。

### 13.4 512 KiB 不是原生 PMM chunk

难题：PMM 支持 64 KiB 和 2 MiB，没有 512 KiB。

解决：

- logical slot 512 KiB；
- physical backing 为 8 × 64 KiB；
- release 使用 64 KiB subchunk；
- 不侵入 PMM chunk-size 核心实现。

### 13.5 Incremental release 与 runtime 重复展开和注册

难题：每个 MoE step 都从 expert id 重新计算 ptr、align、region id 并查 registry，会造成固定开销和 identity 风险。

解决：

- 新增 per-layer expert envelope cache；
- expert 第一次 pressure release 时构建；
- runtime 和 warm pipeline复用；
- cache hit 做强校验；
- cache miss 才 fallback rebuild。

### 13.6 Logical 成功与 chunk 部分成功

难题：一个 role 可对应多个 envelope；若只成功一部分，kernel 访问其余范围仍会失败。

解决：

- driver 按 envelope 返回 success/failed ids；
- 用户态将结果归约回 logical region；
- 全部 envelope 成功才标记 logical success；
- fail-fast 防止部分可用状态进入 compute。

### 13.7 Release 后如何保证一定能恢复

难题：先释放 expert，再发现 slots 未准备好，会失去可控 serving 路径。

解决：

- incremental driver release 要求 reserved executor 已配置；
- READY 前预分配全部 slots；
- prealloc 后 query backing；
- gate 默认 0；
- 完整验证后才设为 1；
- runtime 禁止懒分配。

### 13.8 Slot reuse 与 GPU 并发

难题：kernel 尚未读完，slot 被新 expert 覆盖会导致数据错误。

解决：

- 每层/step 记录 active slots；
- compute stream 上记录 CUDA event；
- reuse/invalidate 前等待 event；
- owner epoch 防止陈旧状态；
- banked policy 隔离当前层与下一层。

### 13.9 Warm prefetch 可能覆盖当前层

难题：下一层预取和当前层计算并发时可能争用同一 slot。

解决：

- layer_id modulo bank_count；
- 至少两个 bank；
- source/target 同 bank 时跳过；
- 当前层 cleanup 延迟到 bank reuse；
- target compute 前同步。

### 13.10 KV 可写页不能复用 expert readonly 机制

难题：如果把 CPU-resident KV 以 readonly slot 映射，attention 写入不正确；若让 fault 自动处理又会进入不可控热路径。

解决：

- expert 与 KV 分离数据面；
- KV 使用普通 UVM writable residency；
- target 外 blocks quarantine；
- grow 前真实 prefetch GPU；
- allocator 只选择 target 内 blocks。

### 13.11 Scheduler 容量、双队列与物理 residency 一致性

难题：把 target 外 block 留在同一 free queue 会导致每次 count/select 扫描全部 blocks；仅调整计数又可能让 prefix hit 或 free 路径误用 CPU tail。

解决：

- 独立维护 GPU-allocatable 与 CPU-offline 双向队列；
- <code>get_num_free_blocks()</code> 直接读取 GPU queue 长度；
- allocation 只从 GPU queue 取，不保留 CPU fallback；
- prefix hit 同时检查 applied target 和 queue membership；
- shrink delta 有 active block 时拒绝提交；
- block 0 始终排除；
- physical registry 将 block id 映射到所有 tensor ranges。

### 13.12 在线 resize 与 in-flight compute 冲突

难题：早期在第一个请求或已有 batch future 时迁移 writable KV、release expert，可能出现 EngineCore 卡死，甚至释放正在被 kernel 读取的权重。

解决：

- 初始 shrink 移到 READY 前；
- serving grow 只在 <code>schedule -&gt; execute_model</code> 之间执行；
- batch queue pressure 先记录 pending target并停止填充，排空 future 后再 resize；
- worker migration 同步完成后才提交 applied target；
- sleep shrink 只允许 scheduler idle；
- 用 fault guard 验证。

### 13.13 No-hit fast path 仍有成本

难题：即使 cold expert 从未命中，每层每 step 的 Python plan intersection、active extraction 和 trace 也会累积。

解决与当前优化：

- per-layer slot-managed set；
- GPU bool mask gate；
- release 成功后才刷新 mask；
- envelope cache；
- 本层无 offloaded expert 时快速跳过；
- 未来可将更多 bookkeeping 下沉 C++/CUDA。

### 13.14 Expert parallel 的 global/local id

难题：plan 使用 global expert id，当前 rank 的 fused tensor 第一维是 local expert id。

解决：

- 通过 expert_map 做 global-to-local；
- 只保留当前 rank 合法 local ids；
- cache key 使用 global id，slice 计算使用 local id；
- item 同时保存两者，便于 driver trace 与 tensor offset 对齐。

仅做 global-to-local 还不够：如果每个 EP worker 独立选择“本层第一个 cold”，一轮会为同一逻辑层卸载多个不同 experts。当前方案增加两阶段协调：worker 上报 <code>layer_name -&gt; candidate global id</code>，EngineCore 每层选择唯一 id 并广播；TP ranks 执行同一 expert shard，EP 中只有 owner 实际 release。

### 13.15 驱动级功能难以纯单测

难题：PMM、PTE、fault 与真实 GPU 状态无法完全由 Python mock 证明。

解决：

- Python 单测覆盖排序、mask、cache、gate、bank reuse；
- ioctl 返回逐 entry status；
- residency query 和 physical counters；
- fault guard；
- dmesg/Xid/MMU 检查；
- short/long workload 分层实验；
- 将机制正确性与性能结论分开。

### 13.16 Warm 异步搬运并非总有 overlap

难题：短 decode batch 的当前层计算时间很短，异步 warm copy 可能无法被隐藏，反而占用 bank、copy engine 和 bookkeeping。

解决：

- 以当前层实际 <code>hidden_states.shape[0]</code> 作为 num_tokens；
- 低于阈值不启动 next-layer warm pipeline；
- 真实 top-k 命中时显式传 <code>async_=False</code>；
- 达到阈值才允许异步 bank 预取；
- trace 记录 num_tokens、threshold 和 forced synchronous mode。

### 13.17 Desired/applied target 的提交顺序

难题：grow 若先扩大 allocator target，再迁物理页，新请求可能立即拿到仍在 CPU 的 writable KV block；shrink 若目标区间仍 active，强制迁出会破坏请求。

解决：

- adaptive grow 的 pending target 先保存在 EngineCore；
- worker 物理 prefetch 全部成功后，scheduler 才更新 desired、迁 queue、提交 applied；
- allocator 和 prefix cache 始终读取 applied target；
- shrink 有 active delta 时保持 desired/applied 分离并返回 deferred；
- partial/failed migration 不伪造 committed 状态。

---

## 14. 代码架构与关键入口

### 14.1 启动与 allocator

- [env_override.py:381](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/env_override.py:381)：进程启动时读取全局 UVM 开关。
- [uvm.py:1136](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:1136)：局部 managed memory pool。
- [uvm.py:1179](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:1179)：切换 PyTorch CUDA allocator。
- [uvm_allocator.cpp:6694](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp:6694)：managed allocation 实现。
- [uvm_allocator.cpp:8495](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp:8495)：批量 range prefetch fallback。

### 14.2 Engine、Worker 与 KV

- [core.py:271](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:271)：KV 初始化总入口。
- [core.py:321](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:321)：READY 前应用初始 KV target。
- [core.py:478](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:478)：expert wave、worker migration、scheduler commit 总事务。
- [core.py:641](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:641)：跨 worker 每层唯一 global expert 选择。
- [core.py:695](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:695)：preemption pressure 检测与 grow target 计算。
- [core.py:759](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:759)：batch queue 排空后的 pending grow。
- [core.py:1058](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:1058)：显式 sleep shrink。
- [gpu_worker.py:457](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:457)：worker KV resize RPC。
- [gpu_worker.py:469](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:469)：worker expert candidate/execution RPC。
- [gpu_worker.py:732](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:732)：post-warmup 总编排。
- [gpu_model_runner.py:5095](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5095)：注册 KV physical ranges。
- [gpu_model_runner.py:5286](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5286)：post-warmup KV tail offload。
- [gpu_model_runner.py:5374](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5374)：runtime KV delta 物理迁移。
- [block_pool.py:186](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:186)：prefix hit applied-target/queue gate。
- [block_pool.py:316](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:316)：GPU queue 直接分配。
- [block_pool.py:461](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:461)：free 后按 target 路由队列。
- [block_pool.py:569](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:569)：target delta 跨队列迁移。
- [uvm_kv_runtime_policy.py:587](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:587)：统一 desired target 更新。
- [uvm_kv_runtime_policy.py:1181](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1181)：desired/applied commit 与失败语义。

### 14.3 FusedMoE 与 Stage-M

- [layer.py:2895](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2895)：hot/warm/cold 分类。
- [layer.py:2910](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2910)：cold-first/warm-second 潜在候选。
- [layer.py:2971](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2971)：按真实 envelope 计算 slot requirement。
- [layer.py:3020](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3020)：下一 capacity-safe candidate。
- [layer.py:3039](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3039)：单层 incremental wave wrapper。
- [layer.py:3073](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3073)：成功 release 后标记 slot-managed/warm。
- [layer.py:3299](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3299)：expert role slice。
- [layer.py:3390](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3390)：envelope 构建与注册。
- [layer.py:3640](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3640)：envelope cache get-or-build。
- [layer.py:3870](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3870)：runtime logical-to-envelope 展开。
- [layer.py:6035](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6035)：Stage-M 注册与 incremental release 主体。
- [layer.py:6941](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6941)：token-aware warm prefetch gate。
- [layer.py:6994](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6994)：runtime reserved restore。

### 14.4 Reserved executor

- [uvm_expert_reserved_executor.py:212](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:212)：layer bank 选择。
- [uvm_expert_reserved_executor.py:335](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:335)：验证预分配 backing。
- [uvm_expert_reserved_executor.py:451](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:451)：stale slot invalidate。
- [uvm_expert_reserved_executor.py:589](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:589)：slot 复用和 prefetch 主入口。
- [uvm_expert_reserved_executor.py:1161](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:1161)：compute 后 cleanup。

### 14.5 Driver ABI 与内核

- [uvm_ioctl.h:1702](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h:1702)：reserved pool ioctl。
- [uvm_ioctl.h:1782](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h:1782)：reserved prefetch batch ioctl。
- [uvm_ioctl.h:1817](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h:1817)：expert release ioctl。
- [uvm.c:828](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:828)：reserved slot backing 分配。
- [uvm.c:9497](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:9497)：旧 owner mapping 清理。
- [uvm.c:9589](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:9589)：新 entry mapping 安装。
- [uvm.c:11530](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:11530)：reserved batch 总入口。
- [uvm_va_block.c:10966](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:10966)：CPU pages 到 reserved chunks 的 copy。
- [uvm_va_block.c:11042](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11042)：readonly PTE map。

---

## 15. 配置体系

### 15.1 Managed allocation

| 配置 | 含义 |
|---|---|
| VLLM_USE_UVM | 全局切换 managed allocator |
| VLLM_UVM_EXPERT_WEIGHTS_MANAGED | 仅 expert 主权重使用 managed pool |
| VLLM_UVM_KV_CACHE_MANAGED | 仅 KV cache 使用 managed pool |
| VLLM_UVM_KV_CACHE_SIZE_GB | 固定 KV tensor 预算，绕过不稳定 profile |

### 15.2 Stage-M

| 配置 | 含义 |
|---|---|
| VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE | placement 时机，主线为 post_warmup |
| VLLM_UVM_INIT_WEIGHT_PLAN_FILE | hot/warm/cold plan |
| VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE | 启用 release |
| VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE | trace_only 或 driver release 模式 |
| VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS | 预注册所有本地 expert |
| VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN | 使用完整 expert 覆盖 |
| VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE | 无显式 warm_plan 时是否允许 warm |
| VLLM_UVM_INIT_WEIGHT_MAX_BYTES | release 字节预算 |
| VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER | 每层 expert 数预算 |
| VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE | 真实预取 hot pages |

### 15.3 Envelope 与 reserved slots

| 配置 | 含义 |
|---|---|
| VLLM_UVM_EXPERT_ENVELOPE_ENABLE | 启用 chunk envelope |
| VLLM_UVM_EXPERT_ENVELOPE_BYTES | 当前推荐 524288 |
| VLLM_UVM_WEIGHT_EXECUTOR | expert_reserved_regions_async |
| VLLM_UVM_EXPERT_RESERVED_POOL_ID | driver pool id |
| VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT | 固定 slot 数 |
| VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES | 单 slot bytes |
| VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY | dynamic 或 layer_mod_bank |
| VLLM_UVM_EXPERT_RESERVED_BANK_COUNT | bank 数 |
| VLLM_UVM_EXPERT_RESERVED_FAIL_FAST | restore 失败时停止 |
| VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE | compute 前同步 |

### 15.4 Warm pipeline

| 配置 | 含义 |
|---|---|
| VLLM_UVM_EXPERT_WARM_PREFETCH_ENABLE | 启用跨层 warm prefetch |
| VLLM_UVM_EXPERT_WARM_PREFETCH_DISTANCE | 目标后续 MoE 层距离 |
| VLLM_UVM_EXPERT_WARM_PREFETCH_MAX_EXPERTS_PER_LAYER | 每层预取上限 |
| VLLM_UVM_EXPERT_WARM_ASYNC_MIN_TOKENS | 达到该 token 数才允许异步 warm pipeline；低于阈值强制同步 restore |

### 15.5 增量 expert offload

| 配置 | 含义 |
|---|---|
| VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE | 初始不 release，由 KV pressure 触发 expert wave |
| VLLM_UVM_INCREMENTAL_EXPERTS_PER_LAYER_PER_ROUND | 每个 pressure wave 每层最多 release 的 expert 数，默认 1 |
| VLLM_UVM_INCREMENTAL_RESERVED_SLOT_BUDGET | 增量模式自动预分配的共享 reserved slots，默认 128 |

### 15.6 KV dynamic pool

| 配置 | 含义 |
|---|---|
| VLLM_UVM_KV_DYNAMIC_POOL_ENABLE | 启用 desired/applied target 与双队列 |
| VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS | 初始 applied target；其余 tail 在 READY 前 offline |
| VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS | post-warmup GPU target |
| VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE | 统一 target 控制文件 |
| VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY | READY 前应用 target |
| VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS | READY 前二阶段 grow |
| VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE | target 变化时迁移物理 pages |
| VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR | auto/cpp_batch/driver_batch |
| VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC | grow 后提交 applied target 前是否同步 |
| VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE | 将 scheduler preemption 作为 grow 触发器 |
| VLLM_UVM_KV_ADAPTIVE_GROW_BLOCKS_PER_PREEMPTION | 每个被 preempt 请求增加的 target blocks |
| VLLM_UVM_KV_ADAPTIVE_FAIL_FAST | expert wave 或 worker resize 失败时停止服务 |
| VLLM_UVM_KV_ADAPTIVE_SHRINK_ON_SLEEP | 显式 sleep 且 scheduler idle 时缩小 target |
| VLLM_UVM_KV_ADAPTIVE_SLEEP_TARGET_BLOCKS | sleep 后保留的 GPU KV blocks |

adaptive pressure、incremental expert offload 和 sleep shrink 当前均为显式开关，默认关闭。主线最小组合为：

~~~text
VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1
VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1
VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=1
VLLM_UVM_KV_ADAPTIVE_SHRINK_ON_SLEEP=1
VLLM_UVM_EXPERT_WARM_ASYNC_MIN_TOKENS=<measured threshold>
~~~

这里的 pressure 指 <code>SchedulerOutput.preempted_req_ids</code>；sleep 指显式 <code>EngineCore.sleep()</code>，不是“某一步暂时没有 token”。

配置时必须一起校验：

~~~text
release bytes
reserved slot bytes × count
单次 top-k 最大 envelope 数
bank_count
每 bank slots
warm prefetch 最大 expert 数
每 expert role 数和 envelope 数
~~~

---

## 16. 正确性验证与测试

### 16.1 当前单元测试

[test_uvm_post_warmup_rebalance.py:15](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/v1/worker/test_uvm_post_warmup_rebalance.py:15) 覆盖：

- KV tail range 按 block/page 边界计算；
- post-warmup 顺序；
- target 自动启用 dynamic pool；
- prealloc 失败时 gate 保持关闭。

[test_uvm_expert_warm_management.py:15](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/test_uvm_expert_warm_management.py:15) 覆盖：

- cold-first、warm-second、hot 排除；
- slot-managed mask 只记录真实 release；
- Stage-M envelope cache 被 runtime restore 复用；
- 没有 reserved restore 时拒绝 driver release；
- layer_mod_bank 复用已预取 warm slot；
- envelope 对齐后的 slot requirement；
- cold wave 完成后才允许 warm wave；
- token 阈值以下不启动 warm async pipeline。

[test_uvm_dynamic_pool_queues.py:1](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/v1/core/test_uvm_dynamic_pool_queues.py:1) 覆盖：

- GPU/offline 双队列 target partition；
- block 释放后回到正确队列尾部；
- grow 后 fresh blocks 的优先级；
- offline prefix cache miss 与 grow 后重新命中；
- applied target 外的 active block在 ref-count 归零后进入 offline queue。

[test_uvm_adaptive_kv.py:1](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/v1/engine/test_uvm_adaptive_kv.py:1) 覆盖：

- 多个 preemption 按请求数计算 grow target；
- batch queue 排空前不执行 expert release/resize；
- expert wave、worker physical resize、scheduler apply 的严格顺序；
- 多 worker candidates 合并后每层只选择一个 global expert id。

本次工作区验证结果：上述修改文件与测试文件均通过 <code>python3 -m py_compile</code>，<code>git diff --check</code> 通过。当前执行环境没有安装 <code>torch</code>、<code>pytest</code> 和 <code>ruff</code>，因此新增 pytest/GPU 场景尚未在本机实际运行；需要在完整 vLLM CUDA 环境继续执行。

### 16.2 Driver 机制验证

需要同时检查：

~~~text
release:
  attempted/released/failed bytes

reserved pool:
  backed_slots
  slot_bytes
  chunk_size
  backing_bytes

restore:
  success/failure entries
  accepted bytes
  reused slot ids

mapping:
  ordinary resident bytes
  reserved PTE bits
  old owner invalidation
~~~

### 16.3 Fault guard

最终正确性标准不是“prefetch API 返回 0”，而是请求开始后：

~~~text
managed GPU faults = 0
new Xid/MMU/fatal records = 0
CPU-resident KV blocks selected = 0
restore failure entries = 0
~~~

KV-only guard 可精确拦截 KV pool；all-managed observe 在当前窄口径 managed 配置下可证明 expert 与 KV 都没有隐藏 fault。

### 16.4 分层验证方法

| 层级 | 验证内容 |
|---|---|
| 纯 Python | plan、排序、mask、cache、bank policy |
| Wrapper | ioctl 编码、逐 entry status、batch 归约 |
| Driver | PMM allocation/release、copy、PTE install |
| vLLM short | 启动闭环、无错误生成、固定开销 |
| vLLM long | 新 KV blocks 实际被分配 |
| 系统 | fault、Xid、MMU、dmesg、进程退出 |

---

## 17. 实验结果与结论

以下结果来自仓库中 2026-07-13 至 2026-07-15 的阶段性实验，代表特定模型、fixture 和配置，不应外推为通用性能承诺。它们验证的是 driver release/reserved restore 和 pre-service KV grow 机制；2026-07-18 新增的双队列、preemption 增量 wave、batch-queue drain、sleep shrink 和 token threshold 尚未重新完成 GPU 端到端实验，因此不能把历史数据直接写成新 adaptive policy 的性能结果。

### 17.1 512 KiB reserved 机制闭环

记录见 [512K static cold8 结果](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_512k_static_cold8_reserved_results_20260713.md)。

~~~text
Stage-M:
  attempted bytes = 1,811,939,328
  released bytes  = 1,755,316,224

reserved pool:
  slot_count = 128
  slot_bytes = 524,288
  chunk_size = 65,536
  backing_bytes = 67,108,864

runtime:
  restored entries = 2,052
  failed entries = 0
  accepted bytes = 1,075,838,976
~~~

结论：512 KiB logical envelope、64 KiB physical chunks、driver release、fixed slot restore 和 generation 全链路跑通。

### 17.2 Restore-heavy 的代价

记录见 [ShareGPT cold8 restore detail](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_sharegpt_cold8_512k_restore_detail_results_20260713.md)。

~~~text
completed = 32/32
throughput = 11.247 tok/s
touched cold experts = 314
restore entries = 20,610
restore traffic ≈ 10.3 GiB
~~~

结论：机制正确不等于 plan 合理。错误地把高频 expert 分类为 cold，会让重复 copy/remap 成为主瓶颈。

### 17.3 No-hit 固定开销

记录见 [zero-hit runtime check ablation](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_zero_hit_runtime_check_ablation_20260714.md)。

~~~text
clean baseline:
  19.249 tok/s

zero-hit, runtime check on:
  17.065 tok/s

zero-hit, runtime check off:
  19.538 tok/s
~~~

结论：即使没有 restore，热路径固定检查也会累计。GPU mask gate、per-layer static skip 和 envelope cache 都是必要优化。

### 17.4 历史 pre-service Expert release 与 KV grow

记录见 [pre-service expert release + KV grow](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_pre_service_expert_release_kv_grow_results_20260714.md)。

~~~text
expert release:
  released ≈ 1,051.5 MiB
  failed entries = 0

KV:
  target 1236 -> 1364
  added blocks = 128
  added logical KV = 192 MiB
  CPU offload success = 128/128
  GPU prefetch success = 128/128
~~~

短请求未真正使用 tail，但证明同一进程中 release、grow 和 READY 前搬运闭环可执行。这是当前 pressure-wave grow 的机制基础，不代表已经测量新在线触发策略的吞吐和稳定性。

### 17.5 长上下文实际使用新增 blocks

记录见 [long-context detailed result](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_pre_service_kv_grow_replicate_long_context_detailed_20260714.md)。

~~~text
prompt tokens ≈ 20,512
tokens_per_block = 16
required blocks ≈ 1,282

target:
  1236 -> 1364

observed:
  max selected block id = 1283
  selected above old target = 47
  CPU candidates selected = 0
  request-time managed faults = 0
~~~

结论：新增 KV 容量不只是 metadata；长请求实际分配到了旧 target 以上、已提前搬回 GPU 的 blocks。

### 17.6 性能结论应如何表述

可以确定：

- 机制增加了可控显存容量；
- release 与 reserved restore 功能正确；
- 长上下文可以使用新增 KV blocks；
- 请求阶段可做到无 managed fault；
- runtime 性能高度依赖 plan 准确率和 restore 流量；
- 当前数据不足以声称对所有 workload 提升吞吐。

### 17.7 新 adaptive 主线的当前验证状态

当前已经完成：

- KV 双队列和 prefix target gate 的单元测试代码；
- EngineCore preemption target、batch drain 和事务顺序测试代码；
- 全局每层唯一 expert candidate merge 测试代码；
- incremental slot requirement、cold/warm wave 和 token gate 测试代码；
- 全部相关 Python 文件静态编译；
- patch whitespace/integrity 检查。

仍需在完整 CUDA/vLLM 环境执行：

- pytest 全套回归；
- 48 层、多 TP/EP worker 的真实 candidate/release 一致性；
- 连续多次 preemption 下 cold waves 到 warm waves 的转换；
- batch queue drain 期间的吞吐和延迟；
- sleep shrink/wake 后重新 grow；
- token threshold 两侧的 copy/compute overlap；
- request-time managed fault、Xid 和 MMU fault guard。

---

## 18. 设计取舍

### 18.1 为什么不直接用 cudaMemPrefetchAsync 恢复 expert

普通 prefetch 会走 ordinary UVM residency 并动态占用 backing，难以限制 footprint。reserved slot 可以把 expert restore 占用固定在预分配窗口内。

代价是需要 driver 修改、PTE 管理和 slot lifecycle。

### 18.2 为什么不是每个 expert 一个永久 slot

每个 expert 永久 slot 会接近重新让 expert 常驻 GPU，失去复用价值。当前方案按 active envelope 临时占用 slots。

### 18.3 为什么使用 512 KiB

- 比 2 MiB 减少邻近权重牵连；
- 能用 8 个原生 64 KiB chunks 表达；
- ioctl batch 和 metadata 开销仍可接受；
- 更小粒度会增加 region、slot、PTE 和 ioctl 数。

512 KiB 是放大效应与控制开销之间的工程折中，不是理论最优常数。

### 18.4 为什么 warm 需要显式计划

把所有非 hot/cold expert 自动视为可卸载 warm，可能造成过量 restore。显式 warm_plan 或严格开关让容量决策保持保守。

### 18.5 为什么 cache 保存完整 ReservedRegionItem

只保存 region id 会迫使 runtime 重新查 registry 并拼装 ptr、bytes、driver id、layer、role 和 metadata。保存完整 item 更适合热路径，但必须校验 ptr/bytes 防止陈旧。

### 18.6 为什么 KV target 使用连续 block-id 前缀

连续规则 <code>1..target</code>：

- 判断 O(1)；
- tail range 容易合并；
- scheduler free count 简单；
- physical ptr 计算连续；
- trace 和实验易解释。

代价是不能按任意碎片化 residency 集合选择 blocks，未来可扩展为 bitmap/range set。

---

## 19. 当前边界与风险

### 19.1 Plan 泛化

当前 plan 多来自固定 fixture 的 routing trace。新请求分布可能命中原先认为 cold 的 expert，造成 restore 流量上升。

### 19.2 Restore 仍在 Python/driver 热路径

尽管有 GPU mask 和 cache，active pair 构造、batch metadata、ioctl 与同步仍有成本。

### 19.3 Warm 预取不是路由预测

当前 warm pipeline 预取的是计划内 warm 集合，并不代表已经预测下一层真实 top-k。预取过多会浪费 PCIe/copy engine 与 slots。

### 19.4 Slot 容量与最坏 top-k

增量 release 前会保守检查“本层全部 slot-managed experts 的 envelope 总数”不超过单 bank slots，因此新 expert 可能因容量不足而不再卸载。该检查比单 step top-k 更保守，但仍需用真实模型验证 role slice 对齐、TP/EP shard 和 warm pipeline 的最坏组合。128 slots 是跨层共享预算，不是给 48 层各自永久分配 128 slots。

### 19.5 KV resize 的 active block

主线 sleep shrink 在 EngineCore 层要求 scheduler idle；policy 层还会再次检查 delta 中是否有 <code>ref_cnt&gt;0</code> block。存在 active block 时不提交 applied target，等待后续重试。

adaptive grow 已采用 worker physical migration -> desired update -> queue promotion -> applied commit 的两阶段顺序。剩余风险是直接通过 control file 修改 target 时，scheduler 进程未必持有 worker physical registry；此路径可能停留在 <code>desired != applied</code> 并不断重试。生产入口应统一走 EngineCore worker-coordinated resize，而不是把控制文件当作完整在线迁移执行器。

### 19.6 PMM identity 级复用尚未证明

已经证明释放 GPU residency、分配 reserved backing 和扩展 KV target，但尚未通过 physical address trace 证明同一具体 chunk 的去向。

### 19.7 硬件与 driver 耦合

ioctl ABI、PTE helper、PMM chunk size 和 open driver 版本存在耦合，需要明确版本管理和兼容性测试。

### 19.8 安全性边界

readonly PTE 是正确性基础。若未来用于可训练权重、在线更新或写型 expert kernel，需要重新设计 writeback 和一致性协议。

### 19.9 Adaptive 触发语义

当前“swap pressure”具体映射为 vLLM v1 的 <code>preempted_req_ids</code>，不是独立 swap counter；“sleep shrink”只响应显式 sleep API，不会因短暂 idle 自动缩容。两条路径均由环境变量显式启用，默认不会改变普通 vLLM 行为。

---

## 20. 后续路线

### 20.1 短期

1. 在完整 torch/CUDA 环境运行新增 pytest、ruff 和已有 vLLM 回归。
2. 用真实 48 层、多 TP/EP 配置验证每层唯一 global expert wave。
3. 进一步减少 runtime trace 和 Python dict 构造。
4. 增加 envelope cache 命中率、失效原因和热路径耗时指标。
5. 为 warm token threshold 增加命中率、浪费字节和 overlap 指标。
6. 验证连续 preemption grow、显式 sleep shrink 与再次 grow。

### 20.2 中期

1. 基于在线 routing heatmap 动态更新 hot/warm/cold。
2. 将 active expert intersection 与 batch item 构造下沉 C++/CUDA。
3. 对 reserved ioctl 做更大 batch 和异步 completion。
4. 使用多 bank/double buffer 减少同步。
5. 在现有双队列和 desired/applied 基础上增加更明确的物理迁移子状态：

~~~text
GPU_FREE
CPU_OFFLINING
CPU_OFFLINE
GPU_PREFETCHING
GPU_FREE
ALLOCATED
~~~

6. 将 direct control-file update 收敛到 EngineCore worker-coordinated RPC。

### 20.3 长期

1. 统一 expert 与 KV 的显存预算协调器。
2. 用收益模型联合决定 release expert 数、slot 数和 KV target。
3. 建立 PMM physical chunk identity tracing。
4. 将已实现的跨 worker candidate merge 扩展为可版本化 plan 和独立 rank pool budget。
5. 探索 GPU copy engine 与 compute 的稳定 overlap。
6. 将实验性 env 配置收敛为结构化 vLLM config 和可观测指标。

---

## 21. 面试项目版本

### 21.1 项目名称

面向 MoE 大模型推理的 vLLM/UVM 专家权重与 KV Cache 显存复用系统

### 21.2 一句话版本

我在 vLLM FusedMoE 与 NVIDIA UVM driver 之间实现了一套压力驱动的显存复用机制：启动时预注册 expert 并预分配固定 slots，scheduler preemption 时全局每层逐个释放 cold/warm expert，再以 worker 物理迁移和 KV 双队列安全扩大 applied target；命中时仍在原 VA 上完成 reserved remap。

### 21.3 30 秒版本

MoE 模型有大量 expert，但每个 token 只访问少数几个；全部常驻 GPU 会挤压长上下文所需的 KV cache。我修改了 vLLM EngineCore、KV BlockPool、FusedMoE top-k 路径和 NVIDIA open UVM driver。启动时不急着卸载 expert，而是先建立双队列、plan/logical registry、envelope 规则和固定 reserved slots；出现 KV preemption pressure 时，全局每层只选一个 cold expert，无 capacity-safe cold 后才选 warm，release 后再把 KV delta 搬回 GPU并提交 applied target。命中 expert 时在原 VA 上做 slot-backed readonly remap，Triton pointer 不变。

### 21.4 2 分钟版本

项目解决的是 MoE expert 权重和 KV cache 争抢显存的问题。普通做法要么让所有 expert 常驻，导致 KV 容量不足；要么依赖 UVM 自动 page fault，延迟不可控。

我的设计分成两条路径。expert 是只读数据，所以保持 PyTorch tensor 的 managed VA 不变。READY 前只解析 hot/warm/cold plan、保护 hot、注册 logical regions，并预分配固定数量的 512 KiB slots；每个 slot 由 8 个 64 KiB PMM chunks 组成。scheduler preemption 表示 KV pressure 后，EngineCore 先收集所有 worker 每层候选，保证每个逻辑层只选择一个 global expert，并严格 cold-first。worker release 成功后，runtime top-k 命中已卸载 expert 时，executor 从 envelope cache 取得 driver regions，内核把 CPU pages copy 到 slot，再给原 expert VA 安装 readonly PTE。FusedMoE kernel 仍使用原 w13/w2 pointer。

KV 是可写数据，不能使用 readonly remap。我把 BlockPool 的空闲 block 拆为 GPU-allocatable 和 CPU-offline 两条双向队列，并区分 desired/applied target。常规分配只从 GPU queue 取，free count 为 O(1)；grow 时 worker 先把物理 ranges 搬回 GPU，所有 worker 成功后才 promote queue 并提交 applied target。显式 sleep 且 scheduler idle 时执行反向 shrink。

工程上最难的是保证分布式 expert 选择、物理 residency、队列 membership 和 scheduler 可见性一致。我使用 post-warmup gate、两阶段 worker resize、逐层 release 核对、batch queue drain、CUDA event 和 fault guard实现 fail-closed。历史实验跑通了约 1.75 GB expert release、64 MB fixed slot pool和长上下文新增 KV blocks；新 adaptive 主线已完成代码与单测补充，但仍需完整 CUDA 环境回归。

### 21.5 8 分钟展开顺序

建议按以下顺序讲：

1. 业务问题：MoE 稀疏访问与 KV 长上下文压力。
2. 为什么普通 offload、普通 UVM fault 都不够。
3. 总架构：vLLM 控制面、两条数据面、driver。
4. 启动事务：initial KV shrink、plan registration、slot prealloc。
5. 在线 grow：preemption、全局每层 candidate、expert wave、worker migration。
6. KV 路径：双队列、desired/applied target、prefix gate。
7. Expert runtime：envelope cache、slot、PTE、token-aware warm。
8. 正确性：batch drain、gate、event、逐层 release 核对、fault guard。
9. 性能：历史机制结果、restore-heavy 与 no-hit 开销。
10. 当前验证边界与下一步。

不要一开始陷入 ioctl 字段。先讲清楚“为什么必须保持 VA 不变”和“为什么 expert/KV 必须分两条路径”，面试官会更容易理解后面的内核细节。

### 21.6 简历描述

可使用以下三条，按实际参与范围调整措辞：

- 设计并实现 vLLM MoE/KV 自适应显存复用系统，以 scheduler preemption 驱动全局 cold-first expert waves，并通过 worker 物理迁移和 desired/applied target 两阶段提交动态扩大 KV 容量。
- 扩展 NVIDIA open UVM driver，新增 expert region release、reserved PMM slot pool 和 readonly PTE remap；以 512 KiB logical envelope 对接 8 × 64 KiB physical chunks，在保持 FusedMoE tensor VA 不变的前提下实现按需恢复。
- 重构 KV BlockPool 为 GPU-allocatable/CPU-offline 双队列，使 allocation 为 O(K)、free count 为 O(1)，并处理 prefix-cache target gate、batch queue drain、sleep shrink 与 token-aware warm prefetch。

### 21.7 STAR 项目叙事

Situation：

MoE 模型 expert 总权重很大，但每步只访问少数 expert；全常驻导致 KV cache 不足，普通 UVM 自动迁移又会在推理热路径引入不可控缺页。

Task：

在不修改 FusedMoE kernel pointer 和数学语义的前提下，释放低频 expert 的 GPU residency，限制 runtime restore footprint，并把显存预算安全转化为更多 KV blocks。

Action：

- 对 expert/KV 使用选择性 managed allocation；
- 将初始 KV shrink、expert registration 和 slot prealloc 放到 post-warmup/READY 前；
- 设计 preemption 驱动的全局 cold-first、每层唯一 expert wave；
- 建立 expert/role 到 512 KiB envelope 的稳定 cache；
- 修改 UVM driver 完成 PMM release、reserved prealloc、CPU-to-GPU copy 和 readonly PTE remap；
- 在 vLLM BlockPool 中实现 GPU/offline 双队列与 desired/applied target；
- 在 EngineCore 中实现 worker 迁移后提交、batch drain 和 sleep shrink；
- 根据 num_tokens 在同步 restore 与异步 warm pipeline 间切换；
- 用 CUDA event、banked slots 和 runtime gate 保证并发正确性；
- 用 unit tests、residency counters、fault guard 和长上下文 workload 验证。

Result：

历史机制实验：

- 跑通约 1.75 GB expert backing release；
- 建立 128 × 512 KiB，即 64 MiB 固定 reserved pool；
- 运行时 2,052 次 512 KiB restore entry 无失败；
- 长上下文实际选到旧 target 以上的 47 个 KV blocks；
- 请求阶段 managed fault、Xid/MMU fault 为 0；
- 同时定位到 cold plan 质量和 no-hit fast path 是主要性能瓶颈。

本轮 adaptive 工程结果：完成双队列、preemption/sleep resize、全局 expert wave 和 token-aware warm 代码及测试补充，静态编译通过；尚未在完整 CUDA 环境复现实验指标。

### 21.8 高频面试问题与回答要点

#### Q1：为什么不直接把 expert tensor 搬到 CPU，再用时搬回？

因为 FusedMoE/Triton 和 CUDA graph 已经持有 tensor 地址。重建 parameter 会破坏地址稳定性并增加 allocator/graph 复杂度。项目保持 managed VA 不变，只替换 GPU PTE 指向的 backing。

#### Q2：为什么需要改 driver？

用户态 prefetch 只能请求普通 UVM residency，无法表达“使用预分配固定 slot，并把原 VA 映射到该 backing”的语义。PMM chunk ownership、VA block PTE 和 replayable fault 状态都在内核中。

#### Q3：为什么 expert 可以 readonly remap，KV 不行？

推理期 expert 权重只读，而 KV cache 持续被 attention kernel 写入。readonly slot 对 expert 正确，对 KV 会违反写语义，因此 KV 必须使用普通 writable residency。

#### Q4：为什么是 512 KiB，不是 2 MiB 或 64 KiB？

2 MiB 对小 expert slice 的牵连和内部浪费太大；64 KiB 会显著增加 region、slot、PTE 和 ioctl 数。512 KiB 可以由 8 个原生 64 KiB PMM chunks 表达，是当前工程折中。

#### Q5：为什么要同时保留 logical region 和 envelope region？

logical region承载模型语义、plan、trace 和 role state；envelope region满足 driver 对齐和物理 release/remap。只保留其中一个会让上层难以表达策略，或让底层无法安全执行。

#### Q6：envelope cache 解决了什么？

它让 Stage-M release、runtime restore 和 warm prefetch 使用同一组 driver identities，避免 serving 每步重复 align、注册和查 registry。cache 还校验 logical ptr/bytes/granularity，避免使用陈旧映射。

#### Q7：如何避免 release 后无法恢复？

真实 release 要求 reserved executor 已配置；启动增量模式不会先 release，而是先完整预分配 slots并 query，backing 完整后才打开 gate。后续 pressure wave 还逐层核对 selected expert 是否真实 release。runtime miss 不允许懒分配，失败默认 fail-fast。

#### Q8：如何避免 slot 被提前覆盖？

compute stream 记录 CUDA event，slot reuse 前等待；owner 带 layer/step/epoch；layer_mod_bank 将当前层与下一层预取放到不同 bank。

#### Q9：KV dynamic pool 是多个队列吗？

是。修改后明确维护 GPU-allocatable free queue 与 CPU-offline free queue。分配和 free count 只访问 GPU queue；target resize 按 delta 在两条双向链表之间 O(1) remove/append。prefix hash 和 request block map 仍是索引，但 prefix hit 必须同时满足 applied target 与 GPU queue membership。

#### Q10：如何证明新增 KV 容量真的被使用？

历史 pre-service grow 实验中，长上下文请求的 max selected block id 达到 1283，超过旧 target 1236，并实际选择了 47 个旧 target 以上 blocks；同时 CPU candidates 为 0、request-time managed fault 为 0。新 preemption-driven grow 仍需重新执行同等级别的 selected-id 和 fault 验证。

#### Q11：最大的性能瓶颈是什么？

一是 cold plan 不准确导致高频 expert 反复 restore；二是 zero-hit 时每层固定检查和 Python bookkeeping。机制本身可用，但策略质量和热路径开销决定吞吐。

#### Q12：项目最难的 bug 是什么？

早期在 request path 或有 in-flight batch 时迁移 writable KV、release expert，EngineCore 可能卡死，甚至存在 kernel 仍在读旧 backing 的风险。现在初始 shrink 位于 READY 前；在线 grow 位于 schedule 与 execute 之间，batch queue 模式先停止填充并排空 future；物理迁移成功后才提交 applied target。

#### Q13：如何证明释放的显存真的被 KV 复用？

目前证明了 expert residency 释放、KV target 增长和新增 blocks 使用，但没有完成 physical chunk identity 的一一匹配。严谨的下一步是记录 PMM freed/acquired physical addresses并做 identity trace，不能把预算级复用夸大为同一 chunk 复用。

#### Q14：如果 top-k 同时命中很多 cold expert，slot 不够怎么办？

当前 strict footprint 应返回失败并 fail-fast，不能 silently fault。生产方案可增加 slots、限制 offload plan、分批执行 expert 或设计更细的 admission control。

#### Q15：如何支持在线动态 workload？

当前已经用 scheduler preemption 驱动 KV grow，并按 cold-first、warm-second执行增量 expert wave；显式 sleep 可在 idle 时 shrink。更完整的在线策略仍需要 routing heatmap、时间衰减、命中风险估计和预算协调器。plan 更新还必须与 cache、slot-managed mask 和 driver region 生命周期做版本化切换。

#### Q16：为什么双队列比一条队列加 target 过滤更好？

一条队列会让 free count、allocation selection 多次扫描全部 blocks，并反复创建 Python list。双队列使普通 allocation 只从 GPU queue 取 K 个、free count O(1)，offline blocks 不再参与热路径；target resize 只对 delta 做 O(1) remove/append。

#### Q17：为什么 expert candidate 要由 EngineCore 全局合并？

EP worker 的 local experts 不同。若每个 worker 各选“本层第一个 cold”，同一逻辑层一轮会卸载多个不同 global experts，也可能有 worker 进入 warm、另一个仍有 cold。两阶段 candidate RPC 让 EngineCore 每层只广播一个 global id，并统一决定 cold/warm class。

#### Q18：为什么 warm async 要看 num_tokens？

异步搬运只有在当前层计算足够长时才可能被隐藏。短 decode 的 overlap 窗口很小，异步 copy 反而占用 bank 和 copy engine，所以低于阈值跳过跨层预取并同步 restore；大 prefill/batch 才开启异步 pipeline。

### 21.9 面试中应主动说明的边界

1. 当前实验结果依赖特定模型与请求 fixture。
2. 已证明容量和正确性，不代表所有 workload 吞吐提升。
3. 未完成 PMM physical chunk identity 级复用证明。
4. warm pipeline 仍是计划驱动，不是高精度在线预测器。
5. 新 adaptive 路径已静态验证但尚未完成真实 GPU/pytest 回归。
6. 这是研究型系统实现，配置项和 driver ABI 仍需产品化收敛。

主动说明这些边界不会削弱项目，反而能体现对实验结论和系统风险的把握。

---

## 22. 最终总结

本项目完成了一个完整的跨层闭环：

~~~text
模型语义:
  router top-k
  hot/warm/cold
  expert/role

vLLM runtime:
  post-warmup barrier
  preemption/sleep adaptive coordinator
  desired/applied KV target
  GPU-allocatable / CPU-offline queues
  global per-layer expert wave
  FusedMoE restore hook
  token-aware warm pipeline

用户态内存层:
  managed allocation scope
  residency registry
  envelope cache
  reserved executor
  ioctl batch

UVM kernel:
  region release
  PMM slot preallocation
  CPU-to-GPU copy
  readonly PTE remap
  owner cleanup

验证:
  historical GPU mechanism experiments
  new adaptive unit-test code and static compile
  residency counters
  selected block ids
  restore trace
  fault guard
  long-context workload
~~~

它最有价值的工程结论不是“UVM 可以自动把内存搬来搬去”，而是：

> 要让统一内存在高性能推理中可用，必须把模型语义、调度器资源可见性、物理 residency、PTE 权限和生命周期 barrier 统一到同一套显式协议中。

历史 GPU 实验已经证明：在保持 FusedMoE 原指针不变的前提下，可以释放计划内 expert 的 GPU backing，通过固定 reserved slots 按需恢复，并让长上下文使用更大的 KV block 范围。当前代码进一步实现了 KV 双队列、desired/applied commit、preemption 增量 wave、全局每层唯一 expert、batch drain、sleep shrink 和 token-aware warm；这些新路径下一步必须在完整 CUDA/vLLM 环境做端到端回归和性能测量。

---

## 23. 相关材料

- [当前项目完整执行流程](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_current_project_execution_flow_20260717.md)
- [MoE expert 显存复用系统详解](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_moe_expert_memory_reuse_system_detailed_20260715.md)
- [当前实现总览](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_current_implementation_overview.md)
- [Stage-M init-time placement 设计](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/reports/stage_m_init_time_uvm_placement_design.md)
- [MoE slot 双缓冲复用说明](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/docs/moe_slot_double_buffer_reuse.md)
- [512 KiB static cold8 实验](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_512k_static_cold8_reserved_results_20260713.md)
- [长上下文 KV grow 实验](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_pre_service_kv_grow_replicate_long_context_detailed_20260714.md)
