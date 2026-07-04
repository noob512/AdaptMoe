# UVM-Based KV Cache and Expert Dynamic Residency Plan

本文档描述如何使用 NVIDIA UVM 的 fault 捕捉、VA block 元数据、迁移和 prefetch 机制，复刻 FluxMoE 论文 `2604.02715v2.pdf` 中的 KV cache 优先、expert 动态驻留和按层搬运管理机制。

本方案明确不实现论文中的压缩 GPU backend，也不实现解压缩 kernel。我们只保留：

- UVM managed virtual address 的稳定地址语义。
- UVM fault / prefetch / migrate 的数据搬运能力。
- UVM VA block / range 元数据作为 KV 和 expert 的状态管理层。
- 应用层 planner 发出策略信号，内核层负责容量约束和迁移执行。

## 1. 背景和目标

FluxMoE 的关键观察是：MoE expert weights 在多数时间处于 idle 状态，但传统 vLLM / PyTorch 会让 expert weights 长期占用 GPU HBM；与此同时，decode 阶段的 KV cache 容量直接影响并发请求数、上下文长度和吞吐。因此应当优先把 HBM 留给 KV cache，把 expert weights 视为按需流式加载的资源。

我们的目标是用 UVM 内核机制实现类似能力：

```text
vLLM / app:
  决定 KV 和 expert 的逻辑角色、访问窗口、预算变化。

UVM kernel:
  记录 managed VA block 的池子和状态。
  在 GPU 物理 chunk 分配前做 admission。
  在预算变化时迁移低优先级 block 到 CPU。
  在 attention / expert kernel 前配合主动 prefetch。
```

最终期望达到：

```text
KV logical capacity:
  由 vLLM 设置逻辑上限，例如最多 N 个 KV blocks。

KV / expert physical residency:
  由 UVM manager 动态控制当前有多少 block 真实驻留在 GPU。

Hard invariant:
  tagged KV + tagged expert 的 GPU resident bytes 不超过动态预算。
```

## 2. 非目标

第一阶段不做以下内容：

- 不实现 expert weight 压缩。
- 不实现 expert weight 解压 kernel。
- 不改变 attention / MoE compute kernel 的地址访问逻辑。
- 不要求所有 miss 都透明无损恢复；早期 enforce 模式允许返回错误，用于验证 admission。
- 不在 fault critical path 中执行复杂同步 reclaim。

长期目标可以支持 remote fallback 或更平滑的 fault 恢复，但第一阶段的重点是先证明内核层容量控制和元数据状态机正确。

## 3. FluxMoE 机制到 UVM 方案的映射

| FluxMoE 机制 | 论文语义 | UVM 方案 |
| --- | --- | --- |
| PagedTensor stable VA | expert tensor 地址稳定，物理 block 可动态映射 | `cudaMallocManaged` / managed VA range 保持稳定虚拟地址 |
| Physical tensor buffer pool | 少量 GPU 物理 buffer 被不同 expert 复用 | UVM GPU chunk / VA block 的真实 GPU residency |
| Map / Unmap | kernel 前映射 expert，使用后释放 | UVM migrate-to-GPU / migrate-to-CPU / prefetch / eviction |
| Storage hierarchy | compressed GPU + host DRAM | 只保留 GPU resident + CPU resident |
| Residency planner | KV 优先，动态调 expert residency level | vLLM planner 通过 IOCTL 调整 KV / expert budget |
| Sliding window | 当前层 + 下一层 expert 驻留 | `EXPERT_CURRENT_LAYER` + `EXPERT_NEXT_LAYER` 状态 |

核心原则：

```text
应用层决定策略。
内核层保证容量。
fault 只做观测和兜底。
active KV / current expert 不依赖 fault 临时搬运。
```

## 4. 当前代码基础

仓库已有 Phase-N 基础能力，可以直接复用：

- Pool tag：
  - `KV = 1`
  - `WEIGHT = 2`
  - `SCRATCH = 3`
  - `ACTIVATION = 4`
- 现有 IOCTL：
  - `UVM_PHASE_N_SET_POOL_BUDGET`
  - `UVM_PHASE_N_GET_POOL_STATE`
  - `UVM_PHASE_N_UPDATE_REGION_STATE`
  - `UVM_PHASE_N_QUERY_REGION`
  - `UVM_PHASE_N_QUERY_REGION_BY_VA`
  - `UVM_PHASE_N_RECLAIM_POOL`
- 现有 admission hook：

```text
block_populate_gpu_chunk()
  -> uvm_phase_n_admit_gpu_chunk()
  -> block_alloc_gpu_chunk()
```

- 现有重要文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_api.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_range.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_migrate.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/vllm/device_allocator/uvm.py
```

第一版 admission 已经可以 trace / deny tagged KV 和 WEIGHT pool 的 GPU chunk 分配。后续工作应基于这条路径推进。

## 5. 总体架构

架构分为控制面和数据面。

### 5.1 控制面

控制面在 vLLM / 应用层：

```text
1. 注册 managed range:
   KV cache range -> KV pool
   MoE expert range -> WEIGHT pool

2. 标记运行时状态:
   active KV
   near-future KV
   cold KV
   current-layer expert
   next-layer expert
   hot expert
   cold expert

3. 动态调整预算:
   KV_budget ↑ 时，expert_hot_budget ↓
   KV_budget ↓ 时，expert_hot_budget ↑

4. 发起主动 prefetch / reclaim:
   attention 前 prefetch active KV
   MoE layer 前 prefetch current expert
   layer 执行时 prefetch next expert
   budget 收缩时 reclaim expert
```

### 5.2 数据面

数据面在 UVM 内核：

```text
1. 在 GPU chunk 分配前执行 admission。
2. 维护 resident / pending / evictable / protected 统计。
3. 执行 migrate-to-GPU / migrate-to-CPU。
4. 根据状态和优先级选择 reclaim victim。
5. 记录 fault / prefetch / migrate / admission 日志。
```

### 5.3 高层执行流程

```text
decode step begins
  app computes active_kv_blocks
  app marks active KV as PROTECTED
  app requests prefetch(active KV)

attention phase
  active KV must already be GPU resident
  UVM fault should be zero or very low for active KV
  background may prefetch near-future KV or next expert

expert phase for layer i
  app marks layer i expert as EXPERT_CURRENT_LAYER | PROTECTED
  app prefetches current expert if not resident
  app prefetches layer i+1 expert asynchronously
  previous layer expert becomes evictable

budget controller tick
  if KV pressure high:
    KV_budget increases
    expert_hot_budget decreases
    reclaim WEIGHT pool
  if KV pressure low:
    KV_budget decreases or stays
    expert_hot_budget increases
```

## 6. Residency State Model

### 6.1 Pool model

```text
POOL_KV:
  KV cache blocks.

POOL_WEIGHT:
  MoE expert weights.

POOL_SCRATCH:
  temporary workspaces.

POOL_ACTIVATION:
  activation buffers.
```

第一阶段只对 `POOL_KV` 和 `POOL_WEIGHT` 强制管理。

### 6.2 KV tiers

```text
KV_ACTIVE:
  当前 batch attention 马上要读。
  必须 GPU resident。
  最高优先级。
  不建议依赖 UVM fault 临时迁移。

KV_NEAR_FUTURE:
  接下来几个 decode step 很可能访问。
  尽量 GPU resident。
  可在 expert 压力低时提前 prefetch。

KV_PREFIX_REUSABLE:
  有复用价值，例如 shared prefix / reusable prefix cache。
  非当前必须。
  可根据复用概率决定保留或迁移。

KV_COLD:
  当前窗口不需要。
  可迁移到 CPU。
  必要时由应用层提前 prefetch 回 GPU。
```

### 6.3 Expert tiers

```text
EXPERT_CURRENT_LAYER:
  当前即将执行的 MoE layer expert。
  kernel 前必须 GPU resident。
  PROTECTED。

EXPERT_NEXT_LAYER:
  下一层或未来短窗口将用到的 expert。
  异步 prefetch 目标。

EXPERT_HOT:
  高频 expert。
  KV 压力低时可以保留在 GPU。

EXPERT_COLD:
  非当前层、非下一层、低频或低置信 expert。
  优先迁移到 CPU。
```

### 6.4 Flags

建议扩展每个 `uvm_va_block_t.phase_n`：

```text
tag_valid
pool
role
state
priority
epoch
last_access_epoch
region_id
allocation_id

flags:
  GPU_RESIDENT
  CPU_RESIDENT
  PREFETCH_PENDING
  RECLAIM_PENDING
  PROTECTED
  EVICTABLE
```

`PROTECTED` 表示当前 kernel 或即将启动的 kernel 会访问该 block。reclaim worker 必须跳过。

## 7. Dynamic Budget Model

### 7.1 总预算公式

```text
HBM_total
  - safety_margin
  - scratch_workspace
  = KV_budget + expert_hot_budget + expert_streaming_budget
```

其中：

```text
KV_budget:
  KV cache 可真实驻留在 GPU 的预算。

expert_hot_budget:
  可长期保留在 GPU 的 expert 预算。

expert_streaming_budget:
  当前层和下一层 expert streaming window 的预算。

safety_margin:
  防止 UVM / CUDA runtime / NCCL / allocator 临时开销导致 OOM。

scratch_workspace:
  attention、MoE、通信和临时 kernel workspace。
```

### 7.2 预算调整规则

```text
if active_kv_demand high:
    KV_budget += delta
    expert_hot_budget -= delta
    reclaim WEIGHT pool
    expert 更多依赖 next-layer prefetch + remote/fault fallback

if active_kv_demand low:
    KV_budget -= delta or keep
    expert_hot_budget += delta
    保留更多 hot expert 在 GPU
```

### 7.3 Hard invariant

第一版至少保证单 pool：

```text
pool_resident_bytes + pool_pending_bytes <= pool_budget_bytes
```

进一步保证共享预算：

```text
KV_resident + KV_pending
+ WEIGHT_resident + WEIGHT_pending
+ expert_streaming_reserved
<= managed_hbm_budget
```

## 8. Kernel Design

### 8.1 Strict admission with pending reservation

现有 admission 只看 resident blocks，可能存在并发 race：

```text
fault A checks resident=1GB, budget=2GB -> pass
fault B checks resident=1GB, budget=2GB -> pass
both allocate 1GB
final resident=3GB
```

因此 admission 必须引入 pending reservation：

```text
admit_gpu_chunk(pool, chunk_size):
  lock budget
  projected = resident_bytes + pending_bytes + chunk_size
  if projected > budget:
      unlock
      return NV_ERR_NO_MEMORY
  pending_bytes += chunk_size
  unlock
  return NV_OK
```

成功路径：

```text
block_alloc_gpu_chunk() succeeds
make_resident succeeds
  pending_bytes -= chunk_size
  resident_bytes += chunk_size
```

失败路径：

```text
block_alloc_gpu_chunk() fails
or migrate fails
or mapping aborts
  pending_bytes -= chunk_size
```

需要新增 helper：

```text
uvm_phase_n_admit_gpu_chunk()
uvm_phase_n_commit_gpu_chunk()
uvm_phase_n_abort_gpu_chunk()
```

### 8.2 Atomic dynamic budget IOCTL

当前 `UVM_PHASE_N_SET_POOL_BUDGET` 是单 pool 更新。动态 KV/expert 互相借预算时，应新增 batch / atomic 接口：

```text
UVM_PHASE_N_SET_DYNAMIC_BUDGETS
```

建议参数：

```c
typedef struct
{
    NvU32 gpu_id;
    NvU32 flags;
    NvU64 epoch;

    NvU64 managed_hbm_budget_bytes;
    NvU64 safety_margin_bytes;
    NvU64 scratch_workspace_bytes;

    NvU64 kv_budget_bytes;
    NvU64 expert_hot_budget_bytes;
    NvU64 expert_streaming_budget_bytes;

    NvU64 reclaim_target_pool;
    NvU64 reclaim_target_bytes;
    NvU64 rmStatus;
} UVM_PHASE_N_SET_DYNAMIC_BUDGETS_PARAMS;
```

内核检查：

```text
kv_budget
+ expert_hot_budget
+ expert_streaming_budget
+ scratch_workspace
+ safety_margin
<= managed_hbm_budget
```

如果更新会导致某个 pool 立刻 over budget：

```text
trace-only:
  只记录 projected_over_bytes。

soft:
  接受预算更新，唤醒 async reclaim worker。

enforce:
  接受预算更新并阻止后续 admission。
  可选同步 reclaim 到 target。
```

### 8.3 Reclaim worker

不要在 fault path 里同步做复杂 reclaim。推荐使用 workqueue：

```text
budget update / app reclaim ioctl
  -> enqueue reclaim request
  -> worker scans victim candidates
  -> migrate selected blocks to CPU
  -> update resident counters
```

victim 顺序：

```text
1. EXPERT_COLD
2. EXPERT_HOT but not PROTECTED
3. KV_COLD
4. KV_PREFIX_REUSABLE
5. KV_NEAR_FUTURE
```

必须跳过：

```text
KV_ACTIVE
EXPERT_CURRENT_LAYER
PROTECTED
PREFETCH_PENDING
RECLAIM_PENDING
blocks with in-flight GPU access
```

### 8.4 Fault handling policy

Fault path 用于观测和兜底，不作为 active 数据的主路径：

```text
active KV fault:
  记录严重事件。
  说明 attention 前 prefetch / metadata 标记失败。

current expert fault:
  记录 miss。
  可等待 prefetch 或 fallback。

cold block fault:
  正常 admission。
  如果 budget 不足，deny 或触发 reclaim。
```

对于性能关键路径：

```text
attention kernel launch 前:
  active KV 必须 GPU resident。

expert kernel launch 前:
  current expert 必须 GPU resident。
```

### 8.5 Locking and concurrency

必须注意：

- admission 和 pending reservation 需要在同一把 pool/accounting lock 下完成。
- reclaim worker 不应在持有 `va_block` 锁时等待长时间 DMA。
- victim block 被选中后需要 refcount 或状态保护，防止并发 prefetch / fault 又把它改回 active。
- budget update 的 epoch 要写入 metadata，用于处理旧请求和新请求交错。
- pending reservation 在所有错误路径都必须 rollback。

建议 invariant：

```text
resident_bytes >= 0
pending_bytes >= 0
resident_bytes + pending_bytes <= budget_bytes, unless trace-only
protected_bytes <= resident_bytes + pending_bytes
```

## 9. Application / vLLM Design

### 9.1 Range tagging

KV allocation 时：

```text
cudaMallocManaged(kv_cache)
uvm_register_range(kv_cache, pool=KV, role=KV_CACHE)
```

Expert allocation 时：

```text
cudaMallocManaged(expert_weight)
uvm_register_range(expert_weight, pool=WEIGHT, role=EXPERT)
```

第一版可以按大 range 注册，后续按 layer / expert / tensor 粒度细分。

### 9.2 Runtime state update

应用层在关键时间点发状态更新：

```text
before attention:
  UPDATE_REGION_STATE(KV_ACTIVE, PROTECTED)
  PREFETCH(active KV)

after attention:
  UPDATE_REGION_STATE(KV_NEAR_FUTURE or KV_PREFIX_REUSABLE)

before MoE layer i:
  UPDATE_REGION_STATE(EXPERT_CURRENT_LAYER, PROTECTED)
  PREFETCH(layer i expert)

during MoE layer i:
  UPDATE_REGION_STATE(EXPERT_NEXT_LAYER)
  PREFETCH(layer i+1 expert)

after layer i:
  clear PROTECTED
  mark previous expert EVICTABLE
```

### 9.3 Planner inputs

vLLM planner 需要收集：

```text
active_kv_blocks
total_kv_blocks
context length
batch size
running requests
free HBM estimate
KV fault count
expert fault count
expert prefetch hit rate
expert load latency
decode step latency
```

### 9.4 Planner output

planner 通过 IOCTL 输出：

```text
kv_budget_bytes
expert_hot_budget_bytes
expert_streaming_budget_bytes
reclaim_target_bytes
state transitions
prefetch requests
```

### 9.5 Coarse-to-fine integration

第一版粗粒度：

```text
KV:
  按 KV block 或连续 KV cache range 注册。

Expert:
  按 MoE layer 注册。

Policy:
  current layer + next layer sliding window。
```

第二版细粒度：

```text
KV:
  active / near-future / prefix / cold 分层。

Expert:
  根据 router 输出，只 prefetch 被选中的 experts。
  hot expert 常驻。
  cold expert CPU resident。
```

## 10. Expected Execution Timeline

### 10.1 Decode step

```text
t0 app scheduler builds batch
t1 app computes active KV set
t2 app marks active KV protected
t3 app prefetches active KV to GPU
t4 attention kernel launches
t5 app computes selected experts or layer window
t6 app prefetches current expert / next expert
t7 MoE kernel launches
t8 app clears protection and updates next state
t9 planner tick adjusts budgets if needed
```

### 10.2 Budget pressure example

KV pressure increases:

```text
active_kv_demand high
  -> app sends SET_DYNAMIC_BUDGETS(KV +1GB, expert_hot -1GB)
  -> kernel marks WEIGHT over budget
  -> reclaim worker migrates EXPERT_COLD / EXPERT_HOT to CPU
  -> later expert access depends on next-layer prefetch
```

KV pressure decreases:

```text
active_kv_demand low
  -> app sends SET_DYNAMIC_BUDGETS(KV -1GB, expert_hot +1GB)
  -> selected hot experts remain GPU resident
  -> expert prefetch/fault rate decreases
```

## 11. Experiments and Scripts

### 11.1 Synthetic admission test

目标：验证 `resident + pending <= budget`。

脚本建议：

```text
workloads/vllm/run_phase_n_strict_admission_smoke.sh
```

测试内容：

```text
1. 设置 KV budget = 1GB。
2. 设置 WEIGHT budget = 1GB。
3. 分配两个 cudaMallocManaged pool。
4. 触碰超过预算的 pages。
5. 验证 admission deny 或 trace 中出现 over budget。
6. 验证 resident_bytes 不超过预算。
```

### 11.2 Dynamic budget test

目标：验证 KV 增加会触发 expert 收缩。

脚本建议：

```text
workloads/vllm/run_phase_n_dynamic_budget_smoke.sh
```

测试内容：

```text
1. 初始预算:
   KV = 1GB
   EXPERT = 1GB

2. 让 KV 和 EXPERT 都 resident。

3. 动态调整:
   KV = 1.5GB
   EXPERT = 0.5GB

4. 验证:
   EXPERT pool 被标记 over budget。
   reclaim worker 迁移 expert 到 CPU。
   KV 后续 admission 成功。
   total resident 不超过 shared budget。
```

### 11.3 Active KV prefetch test

目标：验证 active KV 不靠 fault 临时搬。

测试内容：

```text
1. 将一批 KV block 标记为 KV_ACTIVE。
2. attention 前调用 prefetch。
3. 启动读取 KV 的 kernel。
4. 检查 active KV fault count 接近 0。
5. 检查 latency 相比 fault-on-demand 更稳定。
```

### 11.4 Expert sliding window test

目标：验证 current layer + next layer expert 窗口。

测试内容：

```text
for layer i in model:
  mark layer i expert as CURRENT
  mark layer i+1 expert as NEXT
  prefetch NEXT while CURRENT computes
  clear layer i-1 protection
  reclaim old expert if over budget
```

指标：

```text
expert prefetch hit rate
expert fault count
expert resident bytes
layer latency
decode step latency
```

### 11.5 vLLM end-to-end test

目标：验证真实 decode workload。

测试维度：

```text
batch size: 32 / 64 / 128 / 256
context length: 1024 / 2048 / 4096
KV budget: small / medium / large
expert hot budget: small / medium / large
```

关键指标：

```text
tokens/s
P50/P95 decode latency
active KV fault count
expert fault count
expert prefetch hit rate
KV resident bytes
expert resident bytes
reclaim count
reclaim latency
admission deny count
```

## 12. Logging and Observability

建议统一日志关键字：

```text
phase_n_budget_update
phase_n_admission
phase_n_admission_deny
phase_n_reservation_commit
phase_n_reservation_abort
phase_n_reclaim_begin
phase_n_reclaim_victim
phase_n_reclaim_done
phase_n_prefetch_submit
phase_n_prefetch_done
phase_n_active_kv_fault
phase_n_current_expert_fault
```

`GET_POOL_STATE` 建议返回：

```text
budget_bytes
resident_bytes
pending_bytes
protected_bytes
evictable_bytes
over_budget_bytes
cpu_resident_bytes
prefetch_pending_bytes
reclaim_pending_bytes
admission_denied
fault_count
prefetch_count
reclaim_success
reclaim_failure
```

## 13. Milestones

### M1: Trace and metadata completion

目标：

- 完整记录 KV / expert block 的 pool、state、priority、epoch。
- fault / prefetch / migrate 都能打到 Phase-N 观测点。
- 不改变行为。

完成标准：

```text
synthetic managed allocation 可以看到 KV / WEIGHT fault 分池统计。
state update 后 query_region / get_pool_state 能返回正确状态。
```

### M2: Strict admission

目标：

- 加入 pending reservation。
- 保证单 pool 不越过 budget。

完成标准：

```text
resident_bytes + pending_bytes <= budget_bytes
并发 fault / prefetch 压测下也成立。
```

### M3: Shared dynamic budget

目标：

- 新增 atomic dynamic budget IOCTL。
- 支持 KV 和 expert 之间动态借预算。

完成标准：

```text
KV_budget 上调时 expert_hot_budget 同步下降。
预算更新无中间不一致状态。
```

### M4: Async reclaim worker

目标：

- budget 收缩后自动迁移低优先级 block 到 CPU。
- 不在 fault critical path 中长时间阻塞。

完成标准：

```text
EXPERT_COLD 优先被迁移。
PROTECTED block 不被 reclaim。
reclaim 后 resident_bytes 收敛到 budget。
```

### M5: Active prefetch path

目标：

- attention 前 prefetch active KV。
- expert kernel 前 prefetch current expert。

完成标准：

```text
active KV fault count 接近 0。
current expert fault count 明显下降。
```

### M6: vLLM planner

目标：

- vLLM 根据 batch、context、KV block 数、expert latency 动态调预算。

完成标准：

```text
KV pressure high 时 expert resident bytes 自动下降。
KV pressure low 时 expert hot residency 自动上升。
```

### M7: End-to-end evaluation

目标：

- 跑真实 MoE decode workload。
- 对比固定 expert resident、固定 KV budget、动态 budget 三种策略。

完成标准：

```text
动态策略在 memory-intensive workload 下提高吞吐或降低 OOM / swapping。
active KV fault 受控。
expert prefetch / reclaim 行为可解释。
```

## 14. Recommended First Implementation Slice

建议下一步先做 M2 + M3 的最小闭环：

```text
1. 在 pool accounting 中加入 pending_bytes。
2. admission 成功时 reserve pending。
3. GPU chunk 分配成功后 commit。
4. 所有失败路径 abort reservation。
5. 新增 UVM_PHASE_N_SET_DYNAMIC_BUDGETS。
6. synthetic script 验证 KV 增加时 expert budget 下降。
```

这一步不需要 vLLM planner 完整接入，也不需要真正做高性能 prefetch。只要证明：

```text
内核能严格限制真实 GPU resident capacity。
KV / expert budget 可以动态、原子地调整。
预算变化后能够观测到 over-budget pool。
```

之后再接 async reclaim 和 vLLM runtime signal，风险更小。

## 15. Open Questions

后续实现前需要确认：

1. `resident_bytes` 统计粒度使用 UVM block size、GPU chunk size，还是实际 page mask bytes。
2. 多 GPU / tensor parallel 下 budget 是 per GPU、per VA space，还是全局共享。
3. expert range 是否按 layer 注册，还是按 expert tensor 注册。
4. vLLM KV block 到 managed VA block 的映射关系是否足够稳定。
5. reclaim worker 是否允许同步等待 migrate 完成，还是只提交异步迁移并由 callback 更新状态。
6. enforce 模式下 admission deny 对 CUDA runtime 的错误传播是否可接受，还是需要更软的 fallback。

这些问题不会阻塞 M2 + M3，但会影响 M4 以后策略的准确性和性能。
