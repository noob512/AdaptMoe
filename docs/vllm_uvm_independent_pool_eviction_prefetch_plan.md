# vLLM UVM 独立显存 Pool 驱逐与预取分阶段实现方案

本文档给出在 Stage C/D/E 基础上继续实现“KV cache、weights、runtime scratch 三类显存独立管理”的方案。重点回答两个问题：

1. “把三个部分的显存保存为三个列表分开管理”是否可行？
2. 如果可行，是否有更好的抽象和更稳的分阶段实现路线？

结论先行：三个列表作为最小原型是可行的，但不建议把最终实现停留在“裸列表”。更好的方案是实现一个统一的 **Pool Registry + Object Index + Policy Executor**：列表只是每个 pool 内部的一个索引视图，真正的决策还需要对象元数据、状态机、预算、热度、语义 owner 和安全点。

本文只基于当前 Stage C/D/E 已经完成的能力设计后续实现。

## 1. 当前基础能力

当前项目已经具备三类 pool 的观测边界：

1. Runtime scratch pool：Stage C/C2 已经能针对 hot runtime scratch 做 opt-in `device_direct`，并有 `device_direct_max_total_bytes` 总预算。
2. KV cache pool：Stage D/D2 已经能识别 KV allocation，D2 能在 KV cache 初始化阶段限制 KV blocks，使 `kv_peak_live_bytes_observed <= VLLM_UVM_KV_BUDGET_BYTES`。
3. Weights pool：Stage E/E3 已经能识别 weights allocation，统计 weights live/peak，生成 weight tensor semantic map，并可选记录 MoE expert routing 热度。

但是当前还没有完成：

1. KV runtime eviction/swap。
2. weights runtime eviction/offload。
3. weights predictive prefetch。
4. scratch pool 全局 admission control。
5. 三个 pool 之间统一的 pressure coordinator。
6. “哪个 pool 超载，只处理哪个 pool”的策略执行闭环。

因此下一阶段应该从“统一元数据和策略入口”开始，而不是直接写驱逐逻辑。

## 2. 三个列表分开管理是否可行

初步设想是维护三个列表：

1. `kv_list`
2. `weights_list`
3. `scratch_list`

每个 allocation 根据 `is_kv_allocation()`、`is_weight_allocation()` 或 runtime scratch 分类放进对应列表。这作为原型是可行的，因为当前 allocator 已经能区分：

1. `AllocationClass::KvPersistent`
2. `AllocationClass::WeightPersistent`
3. `AllocationClass::RuntimeScratch`
4. `AllocationClass::RuntimeWorkspace`

这个方案的优点：

1. 实现简单。
2. 很容易输出每个 pool 的对象数量和 live bytes。
3. 可以快速验证“分类是否正确”和“free 时是否能从对应列表删除”。
4. 对 Stage F0 的只读 telemetry 足够。

但裸列表不适合作为最终驱逐/预取执行器，原因如下。

## 3. 裸列表的问题

### 3.1 缺少对象状态

驱逐和预取不是“在列表里删一个元素”这么简单。每个对象至少需要状态：

1. `resident_gpu`
2. `managed_uvm`
3. `prefetch_inflight`
4. `offloaded_cpu`
5. `evicting`
6. `pinned`
7. `active_in_use`
8. `failed`

裸列表无法表达这些状态变化。

### 3.2 缺少语义 owner

KV block 的 owner 是 block manager / scheduler。

Weight tensor 的 owner 是 model runner / model loader / MoE layer。

Scratch allocation 的 owner 接近 allocator/runtime phase。

allocator 只靠列表不知道某个对象能不能安全迁移。比如 KV 必须知道 request/block table；weights 必须知道 tensor 名称和 expert；scratch 通常生命周期极短，很多时候不应该迁移，只应 admission/fallback。

### 3.3 缺少热度和优先级

驱逐/预取需要排序依据：

1. last access time
2. fault count
3. routing frequency
4. reuse interval
5. size
6. owner priority
7. active request priority
8. prefetch confidence

裸列表只能表达“有哪些对象”，不能表达“先处理谁”。

### 3.4 缺少并发安全和生命周期处理

allocator 的 malloc/free 可能来自多线程、多 CUDA stream。裸 vector/list 容易出现：

1. free 时对象还在 prefetch queue。
2. policy thread 正在扫描，allocation 已释放。
3. pointer 被复用。
4. prefetch/offload 和 kernel 使用冲突。

至少需要 generation id、mutex 或 lock-free 安全策略，以及 clear ownership。

### 3.5 缺少跨 pool 协调

三个 pool 分开管理不等于完全互不影响。预取 weights 会占 PCIe/NVLink 带宽，也会影响 KV/page migration。scratch device-direct 会占 GPU-only budget。需要统一 coordinator 控制：

1. 每个 pool 的 budget。
2. 每个 pool 的 pressure。
3. 全局 prefetch bandwidth。
4. 全局 safe mode / kill switch。

裸列表无法承担这些职责。

## 4. 推荐架构：Pool Registry + Object Index + Policy Executor

推荐引入三个层次：

```text
Pool Registry
├── PoolState[kv]
├── PoolState[weights]
└── PoolState[scratch]

Object Index
├── ptr -> MemoryObject
├── tensor_id/name -> MemoryObject
├── kv_block_id -> MemoryObject
└── expert_id/layer_id -> MemoryObject[]

Policy Executor
├── KVPolicyExecutor
├── WeightPolicyExecutor
└── ScratchPolicyExecutor
```

### 4.1 Pool Registry

Pool Registry 是全局预算和统计入口。每个 pool 有独立状态：

```cpp
enum class UvmPoolKind {
    KvCache,
    Weights,
    RuntimeScratch,
};

struct PoolBudget {
    size_t budget_bytes;
    size_t live_bytes;
    size_t peak_live_bytes;
    size_t reserved_bytes;
    size_t pressure_bytes;
    std::string mode;
};

struct PoolCounters {
    size_t allocations;
    size_t frees;
    size_t over_budget_events;
    size_t soft_rejects;
    size_t evict_candidates;
    size_t prefetch_candidates;
    size_t eviction_attempts;
    size_t eviction_success;
    size_t prefetch_attempts;
    size_t prefetch_success;
};
```

Stage C/D/E 现有的 counters 可以逐步迁移或镜像到这个统一结构。

### 4.2 MemoryObject

每个被管理对象使用统一 metadata：

```cpp
struct MemoryObject {
    uint64_t object_id;
    void* ptr;
    size_t size;
    int device;
    UvmPoolKind pool;
    AllocationClass alloc_class;
    std::string phase;
    std::string semantic_name;
    int layer_id;
    int expert_id;
    std::string role;
    uint64_t generation;
    uint64_t last_touch_ns;
    uint64_t fault_count;
    uint64_t prefetch_count;
    uint64_t eviction_count;
    bool pinned;
    bool active;
    bool managed_by_owner;
};
```

对于不同 pool，可以有扩展字段：

1. KV：`block_id`、`request_id`、`is_active_sequence`、`last_token_step`。
2. Weights：`tensor_name`、`role`、`layer_id`、`expert_id`、`routing_hotness`。
3. Scratch：`lifetime_us`、`device_direct_backend`、`gap_match`。

### 4.3 Object Index

不只维护三个列表，而是维护多种索引：

1. `ptr -> object`
2. `pool -> ordered objects`
3. `expert_id -> weight objects`
4. `layer_id -> weight objects`
5. `kv_block_id -> kv object`
6. `priority queue -> eviction candidates`
7. `priority queue -> prefetch candidates`

三个列表可以保留为 `pool -> objects` 的基础索引，但不要让它成为唯一数据结构。

### 4.4 Policy Executor

每个 pool 的策略执行器不同：

1. KVPolicyExecutor：只能通过 vLLM block manager/scheduler 做 swap/evict/reuse，allocator 不直接 free KV。
2. WeightPolicyExecutor：通过 model runner / weight manager 做 expert 权重 prefetch/offload，allocator 不直接破坏 parameter tensor。
3. ScratchPolicyExecutor：主要做 admission/fallback/device-direct，不做 eviction，因为 scratch 生命周期短。

### 4.5 Coordinator

Coordinator 负责跨 pool 的节流：

1. 不允许 weights prefetch 把 KV budget 挤爆。
2. 不允许 expert prefetch 和 KV swap 同时打满 PCIe。
3. 不允许 scratch device-direct 超过 scratch budget。
4. 在不确定时 fail-open，保守回到现有 managed/device-direct 路径。

## 5. 总体原则

后续实现应遵循：

1. allocator 负责观测、记账、轻量 admission，不负责语义复杂驱逐。
2. KV 驱逐必须在 vLLM block manager / scheduler 安全点执行。
3. weights offload/prefetch 必须在 model runner / MoE layer 可解释路径执行。
4. scratch 以 device-direct admission/fallback 为主，不做运行时 eviction。
5. 所有执行策略先 trace-only，再 opt-in，再小范围默认。
6. 所有策略必须有 kill switch。
7. 所有策略必须能输出 per-pool metrics 和 reason。

## 6. 分阶段路线总览

建议把后续实现拆成 Stage F 到 Stage J：

1. Stage F：统一 Pool Registry 和对象索引，不做真实驱逐。
2. Stage G：runtime scratch pool admission control 完整化。
3. Stage H：weights expert hot/cold 分类和 trace-only prefetch/offload plan。
4. Stage I：weights expert-only prefetch/offload 小范围执行。
5. Stage J：KV runtime eviction/swap 接入 vLLM block manager。
6. Stage K：统一 pressure coordinator 和跨 pool 带宽/预算节流。

下面逐阶段展开。

## 7. Stage F：统一 Pool Registry 与对象索引

### 7.1 目标

把 Stage C/D/E 已有的三套 telemetry 合并到统一 per-pool registry 中，先只做观测，不改变行为。

### 7.2 实现内容

1. 在 allocator 中新增 `UvmPoolKind`。
2. 新增 `MemoryObject` 基础结构。
3. 将 `AllocationClass` 映射到 pool：
   - `KvPersistent -> KvCache`
   - `WeightPersistent -> Weights`
   - `RuntimeScratch/RuntimeWorkspace -> RuntimeScratch`
   - 其他为 `Unknown`
4. 维护 `ptr -> MemoryObject`。
5. 维护 `pool -> object ids`。
6. 在 malloc 时注册对象。
7. 在 free 时注销对象。
8. 输出统一 `TRACE_POOL_ALLOC` / `TRACE_POOL_FREE`。
9. 输出统一 `pool_summary` JSON。

### 7.3 为什么先不做执行

这是为了验证：

1. 三个 pool 分类是否稳定。
2. free 是否能正确删除对象。
3. pointer 复用是否安全。
4. pool live bytes 是否和现有 C/D/E counters 一致。
5. no-bench 和 benchmark 下 registry 是否有泄漏。

### 7.4 验收标准

1. `pool_registry_enabled=True`。
2. `kv_pool.live_bytes == kv_live_bytes`。
3. `weight_pool.live_bytes == weight_live_bytes`。
4. `scratch_pool.live_bytes` 与 device-direct/runtime scratch records 能对应。
5. `pool_object_leaks=0` 或只剩预期长期对象。
6. 关闭 registry 后行为与当前版本一致。

### 7.5 建议文件

1. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`
2. `workloads/vllm/summarize_pool_registry_metrics.py`
3. `workloads/vllm/check_stage_f_pool_registry_success.py`
4. `docs/vllm_uvm_stage_f_pool_registry_implementation.md`

## 8. Stage G：Runtime Scratch Pool Admission Control

### 8.1 目标

把 Stage C 的 attention-only scratch device-direct 扩展成更明确的 scratch pool admission，而不是只靠 gap-watch 单点命中。

### 8.2 实现内容

1. 保留现有 Stage C device-direct 门控。
2. 将 `device_direct_live_bytes` 纳入 `scratch_pool.live_bytes` 或 `scratch_pool.device_direct_live_bytes`。
3. 增加 scratch pool budget：
   - `VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES`
   - `VLLM_UVM_SCRATCH_POOL_MODE=trace_only|enforce`
4. 当 scratch pool 超预算时：
   - trace-only：只记录。
   - enforce：拒绝新增 device-direct，fallback managed。
5. 不做 scratch eviction。
6. 支持 phase allowlist：
   - `enabled:attention`
   - `enabled:moe`
   - `enabled:compute_logits`
   - 默认仍只开 attention。

### 8.3 为什么 scratch 不适合 eviction

scratch 生命周期短，通常由 kernel/workspace 临时使用。驱逐一个短生命周期对象往往得不偿失，也很难保证 kernel 未使用。更稳的是 admission：

1. 能 device-direct 就进 scratch device-direct pool。
2. 超预算就 fallback managed。
3. 生命周期结束自然 free。

### 8.4 验收标准

1. `scratch_pool_trace_records > 0`。
2. `scratch_pool_device_direct_records > 0`。
3. `scratch_pool_peak_live_bytes <= scratch_pool_budget_bytes`。
4. 超预算时 fallback managed，而不是失败。
5. benchmark failed requests 为 0。
6. 相对 Stage C/C2 性能不明显退化。

## 9. Stage H：Weights Hot/Cold 分类与 Trace-only 计划

### 9.1 目标

利用 Stage E2/E3 产物，将 MoE expert weights 分成 hot/cold，并先输出 trace-only 的 offload/prefetch 计划，不执行迁移。

### 9.2 数据来源

1. weight semantic map：
   - layer_id
   - expert_id
   - role
   - size_bytes
   - address range
2. MoE routing trace：
   - layer_name
   - step
   - expert_token_counts
3. fault address log：
   - fault address
   - fault count
   - unique pages
4. allocator trace：
   - weight live/peak
   - phase
   - budget pressure

### 9.3 实现内容

1. 新增 join 脚本：
   - fault address -> weight map range
   - routing trace -> expert weights
2. 输出 expert heat score：
   - `routing_tokens`
   - `faults`
   - `unique_pages`
   - `bytes`
   - `reuse_interval`
3. 输出 trace-only 计划：
   - hot experts：prefetch candidates
   - cold experts：offload candidates
   - dense/shared weights：默认 pinned，不动
4. 不执行 `cudaMemPrefetchAsync`。
5. 不执行 offload。

### 9.4 策略建议

初始只处理 MoE expert weights：

1. 允许 `role in {moe_gate_up, moe_down}`。
2. 不处理 `attention`。
3. 不处理 `norm`。
4. 不处理 `embedding`。
5. 不处理 `moe_router`，因为 router 是预测必需路径。

### 9.5 验收标准

1. `weight_fault_join_records > 0`。
2. `expert_heat_records > 0`。
3. `prefetch_plan_records > 0` 或明确说明无 hot expert。
4. `offload_plan_records > 0` 或明确说明无 cold expert。
5. 计划总 bytes 不超过配置：
   - `VLLM_UVM_WEIGHT_PREFETCH_PLAN_MAX_BYTES`
   - `VLLM_UVM_WEIGHT_OFFLOAD_PLAN_MAX_BYTES`
6. trace-only 不改变 benchmark correctness。

### 9.6 建议文件

1. `workloads/vllm/join_weight_faults_with_routing.py`
2. `workloads/vllm/plan_stage_h_weight_expert_actions.py`
3. `workloads/vllm/check_stage_h_success.py`
4. `docs/vllm_uvm_stage_h_weight_hot_cold_plan.md`

### 9.7 当前实现状态

Stage H 已按 trace-only 方式落地，当前实现刻意不执行迁移：

1. `plan_stage_h_weight_expert_actions.py` 读取 Stage E weight map、MoE routing trace 和可选 fault address log，生成 expert heat ranking。
2. 对 fused MoE tensor 按第 0 维展开为 `logical_fused_expert_slice`，让 routing heat 可以映射到 `(layer_id, expert_id)`。
3. 输出 `prefetch_candidate` 和 `offload_candidate`，所有 action 都带 `mode=trace_only`。
4. `check_stage_h_success.py` 一键启动小 benchmark、开启 weight map/routing/fault trace、生成 plan，并验证 routing join、hot/cold plan、budget 和 failed requests。
5. fault join 默认作为附加证据，不作为硬要求；如果实验需要验证 fault address 与 expert range 的映射，可传 `--require-fault-join`。

## 10. Stage I：Weights Expert-only Prefetch / Offload 小范围执行

### 10.1 目标

在 Stage H trace-only 计划稳定后，只对 MoE expert weights 执行 opt-in prefetch/offload，不碰 dense/shared weights。

当前已先落地 expert-only prefetch；offload 继续后置。

### 10.2 Prefetch 执行方案

初期只实现 prefetch，不实现 offload：

1. 根据 routing trace 或最近窗口预测下一步 hot experts。
2. 找到对应 expert weight ranges。
3. 对 managed ranges 执行 `cudaMemPrefetchAsync(ptr, size, device, stream)`。
4. 每步限制 prefetch bytes。
5. 每层限制 prefetch experts。
6. 全局限制 in-flight prefetch。

建议参数：

```text
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0|1
VLLM_UVM_WEIGHT_PREFETCH_MODE=trace_only|prefetch
VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP=<n>
VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER=<n>
VLLM_UVM_WEIGHT_PREFETCH_MIN_HOTNESS=<float>
VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES=moe_gate_up,moe_down
```

### 10.3 Offload 执行方案

offload 比 prefetch 风险更大，建议晚一阶段做，并先使用 CUDA advise 而不直接拷贝：

1. 对 cold expert weights 设置 preferred location CPU。
2. 或降低 GPU preferred location。
3. 必须保留 hot expert prefetch。
4. 必须支持快速禁用。

候选 API：

1. `cudaMemAdviseSetPreferredLocation` 到 CPU。
2. `cudaMemPrefetchAsync` 到 CPU。
3. 对 explicit CPU copy/offload 要后置，因为会改变 tensor storage/placement，风险更大。

建议参数：

```text
VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0|1
VLLM_UVM_WEIGHT_OFFLOAD_MODE=trace_only|advise_cpu|prefetch_cpu
VLLM_UVM_WEIGHT_OFFLOAD_MAX_BYTES=<n>
VLLM_UVM_WEIGHT_OFFLOAD_MIN_COLD_STEPS=<n>
```

### 10.4 执行位置

不要在 allocator 里直接按 pointer 执行 weights offload。更合适的位置：

1. `gpu_model_runner.py` 或新的 `uvm_weight_manager.py`
2. MoE routing 后、expert execution 前的安全点
3. decode step 边界
4. server idle / low pressure 时

### 10.5 验收标准

1. prefetch action records > 0。
2. prefetch failures = 0 或有明确 fallback。
3. failed requests = 0。
4. weight fault 数下降或 TPOT 不恶化。
5. prefetch bytes 没超过 per-step/global budget。
6. 关闭开关后结果回到 baseline。

### 10.6 当前实现状态

Stage I 已按小范围真实 prefetch 方式落地：

1. `vllm/device_allocator/uvm.py` 新增 `prefetch_range_to_device(ptr, size, device)`，复用 allocator `.so` 的 `uvm_prefetch`。
2. `FusedMoE.select_experts()` 在 routing 后、expert kernel 前调用 `_maybe_prefetch_uvm_expert_weights(topk_ids)`。
3. 当前只处理 active local experts 的 `w13_weight` 和 `w2_weight` slice。
4. 配置项由 `run_kv_fault_ratio.sh` 注入：
   - `--uvm-weight-prefetch-enable`
   - `--uvm-weight-prefetch-mode trace_only|prefetch`
   - `--uvm-weight-prefetch-trace-file`
   - `--uvm-weight-prefetch-max-bytes-per-step`
   - `--uvm-weight-prefetch-max-experts-per-layer`
   - `--uvm-weight-prefetch-target-roles`
5. `check_stage_i_success.py` 会启动小 benchmark，验证 `prefetch_issued`、预算、weight map、routing trace、allocator metrics 和 failed requests。
6. 详细实现记录见 `docs/vllm_uvm_stage_i_weight_expert_prefetch.md`。

## 11. Stage J：KV Runtime Eviction / Swap

### 11.1 目标

在 KV 初始化预算 D2 之外，实现运行时 KV pressure 下的 scheduler-aware eviction/swap。

### 11.2 为什么必须在 vLLM block manager 做

KV cache 不是普通 tensor 列表。驱逐必须知道：

1. block table。
2. request state。
3. prefix cache。
4. active decode sequence。
5. recompute/swap 成本。
6. CPU swap space。

allocator 只能输出 pressure signal，不应直接驱逐。

### 11.3 实现内容

1. 新增 KV pressure telemetry API：
   - allocator 或 Python wrapper 输出 `kv_pool.pressure_bytes`。
2. block manager 接收 budget：
   - `VLLM_UVM_KV_RUNTIME_BUDGET_BYTES`
3. 当 KV runtime live 接近 budget：
   - 优先复用 free blocks。
   - 再选择 inactive/low-priority blocks。
   - 再 swap 到 CPU。
   - 必要时降低 admission。
4. 更新 block table。
5. scheduler 避免驱逐 active decode blocks。

### 11.4 策略顺序

推荐：

1. trace-only：只输出 would-evict block ids。
2. offline check：验证 block ids 都非 active。
3. soft admission：新请求进入前限制 KV allocation。
4. swap inactive blocks。
5. 最后才考虑 recompute。

### 11.5 验收标准

1. would-evict blocks 不包含 active decode blocks。
2. 开启 real swap 后 failed requests = 0。
3. 输出一致性正确。
4. KV pool 超载时只发生 KV action。
5. weights/scratch metrics 不被错误修改。

## 12. Stage K：统一 Pressure Coordinator

### 12.1 目标

当三个 pool 都具备独立策略后，加入统一 coordinator，避免多个策略同时抢带宽/预算。

### 12.2 需要协调的资源

1. GPU memory budget。
2. PCIe/NVLink migration bandwidth。
3. CUDA stream。
4. CUDA mempool release threshold。
5. server latency budget。

### 12.3 实现内容

1. 新增全局 policy config：
   - `VLLM_UVM_POOL_COORDINATOR_ENABLE`
   - `VLLM_UVM_POOL_COORDINATOR_MODE=trace_only|enforce`
2. 每个 pool 上报 pressure：
   - `pressure_bytes`
   - `pressure_ratio`
   - `action_queue_depth`
3. coordinator 分配 action budget：
   - KV swap bytes per second
   - weight prefetch bytes per step
   - scratch device-direct admission bytes
4. 避免同一时刻 KV swap 和 weights prefetch 同时打满带宽。
5. 输出统一 report。

### 12.4 验收标准

1. 三个 pool metrics 同时存在。
2. 超载 pool 只触发本 pool action。
3. global migration bytes 不超过配置。
4. benchmark correctness 通过。
5. TPOT/ITL 退化可解释。

## 13. 关于“列表”的推荐落地方式

可以这样落地：

```text
PoolRegistry
├── kv.objects: intrusive list / ordered map
├── weights.objects: intrusive list / ordered map
└── scratch.objects: intrusive list / ordered map
```

但不要只保存 pointer。每个元素应是 `MemoryObject`，至少包含：

1. object id
2. pointer
3. size
4. pool kind
5. allocation class
6. phase
7. state
8. generation
9. semantic key
10. last touch / heat score
11. action eligibility

列表用于枚举，hash map 用于快速查找，priority queue 用于驱逐/预取排序。

推荐最小实现：

```text
unordered_map<void*, MemoryObject> active_objects;
unordered_map<PoolKind, unordered_set<uint64_t>> pool_object_ids;
priority_queue<EvictCandidate> evict_queue;
priority_queue<PrefetchCandidate> prefetch_queue;
```

如果只做 Stage F telemetry，可以先不实现 priority queue。

## 14. 安全机制

每个执行型 stage 必须有：

1. enable 开关。
2. mode：`trace_only|enforce` 或 `trace_only|prefetch|offload`。
3. max bytes。
4. max actions per step。
5. allowlist role/phase。
6. fallback reason。
7. action error 记录。
8. offline summary。
9. one-click success check。

建议统一 reason 命名：

```text
pool_budget_unlimited
pool_budget_within_budget
pool_budget_exceeded_trace_only
pool_budget_exceeded_soft_enforce
pool_action_trace_only
pool_action_prefetch_success
pool_action_prefetch_failed
pool_action_evict_candidate
pool_action_evict_skipped_active
pool_action_evict_success
pool_action_evict_failed
```

## 15. 实验设计建议

每个阶段都应提供三类实验：

1. no-bench 初始化检查：验证 telemetry 和配置。
2. p1 小 benchmark：验证执行路径 correctness。
3. p5/p20 A/B：验证 fault 和性能变化。

每个实验至少输出：

1. per-pool live/peak/budget。
2. action records。
3. action failures。
4. failed requests。
5. output throughput。
6. TTFT/TPOT/ITL。
7. fault delta。
8. 是否触发其他 pool action。

## 16. 推荐下一步

最建议马上实现的是 Stage F：

1. 不做真实驱逐。
2. 不做真实 prefetch。
3. 只把 C/D/E 的对象统一进 Pool Registry。
4. 生成统一 pool summary。
5. 写 `check_stage_f_pool_registry_success.py`。

这样做风险最低，也能为后续所有真实执行策略提供统一基础。如果直接开始写 weights offload 或 KV eviction，很容易遇到“对象是谁、归谁管、是否活跃、是否安全”的问题，然后返工。

## 17. 阶段交付物清单

### Stage F 交付物

1. `PoolKind` / `MemoryObject` / `PoolRegistry`。
2. allocator trace：`TRACE_POOL_ALLOC/FREE`。
3. summary 脚本：`summarize_pool_registry_metrics.py`。
4. check 脚本：`check_stage_f_pool_registry_success.py`。
5. 文档：`vllm_uvm_stage_f_pool_registry_implementation.md`。

### Stage G 交付物

1. scratch pool budget 参数。
2. scratch device-direct admission summary。
3. Stage C2 compatibility check。
4. scratch pool budget sweep。

### Stage H 交付物

1. weight fault map join 脚本。
2. expert heat score summary。
3. trace-only prefetch/offload plan。
4. no behavior-change validation。

### Stage I 交付物

1. expert-only prefetch executor。
2. optional cold expert advise/offload executor。
3. action budget and bandwidth throttle。
4. p1/p5 A/B validation。

### Stage J 交付物

1. KV would-evict trace-only plan。
2. block-manager-safe eviction/swap prototype。
3. request correctness validation。
4. KV pressure benchmark.

### Stage K 交付物

1. global pressure coordinator。
2. per-pool action budget allocator。
3. cross-pool interference report。
4. full C/D/E/F/G/H/I/J integrated success check。

## 18. 最终判断

“三个列表分开管理”可以作为 Stage F 的最小起点，但它只是索引，不是完整管理方案。

更好的最终形态是：

```text
三类 pool 独立预算
+ 每个对象统一 metadata
+ 多索引查找
+ 每个 pool 自己的策略执行器
+ 全局 coordinator 限制跨 pool 干扰
+ trace-only -> opt-in enforce 的渐进开关
```

这样才能真正做到：

1. KV 超载时，只触发 KV block manager 的动作。
2. weights 超载时，只触发 weights/expert 策略。
3. scratch 超载时，只影响 scratch admission/fallback。
4. 三者的日志、预算、失败原因和性能变化都能被独立解释。
