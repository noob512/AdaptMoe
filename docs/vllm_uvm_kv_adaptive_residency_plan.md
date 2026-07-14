# vLLM KV 压力驱动双向驻留与 CPU 后备池实施方案

结论先放前面：这个机制**可以实现**，但要按“固定最大池 + 压力驱动 resident window + CPU 后备池”来做，不能把它实现成运行时改 `num_gpu_blocks`。

更准确地说，目标是：

1. GPU 上保留一个固定上限的 KV logical pool。
2. 当 KV 压力过大时，优先把一部分 CPU-backed block 拉回 GPU，扩大 GPU resident set。
3. 当 KV 压力过小时，主动把一部分冷 block 搬运到 CPU，收缩 GPU resident set，把空间让给 weights / scratch / 其他池。
4. GPU pool 里仍然保持 fresh / warm / CPU-backed 的分层复用顺序。
5. 活跃 request 的 block table 不做后台迁移，只保护，不偷改。

这条路和现有代码是对得上的。当前仓库里已经有：

- `BlockPool.get_new_blocks()` 的选块入口。
- `KVCacheManager.allocate_slots()` 的 admission 入口。
- `UvmKvRuntimePolicy` 的 pressure trace、free-block 选择、Stage L5/L6 元数据和 active protect。
- `UvmPoolResidencyManager` 的 residency 账本、budget、state update、trace。
- `kv_transfer/offloading_connector` 的 request-level load/store 管线。
- `gpu_model_runner` 的 KV pointer map 导出和 `kv_transfer_group` 注册。

相关背景文档可继续沿用：

- [docs/vllm_uvm_stage_j_kv_runtime_eviction.md](./vllm_uvm_stage_j_kv_runtime_eviction.md)
- [docs/vllm_uvm_stage_l0_pool_residency_implementation.md](./vllm_uvm_stage_l0_pool_residency_implementation.md)
- [docs/vllm_uvm_stage_l_pool_residency_manager_plan.md](./vllm_uvm_stage_l_pool_residency_manager_plan.md)
- [docs/vllm_uvm_stage_l_residency_implementation_roadmap.md](./vllm_uvm_stage_l_residency_implementation_roadmap.md)
- [docs/vllm_uvm_memory_pool_evolution_plan.md](./vllm_uvm_memory_pool_evolution_plan.md)

## 1. 目标形态

我们要的不是“见块就复用”，而是三层策略。

```text
Tier 1: GPU fresh blocks
  未触达、未历史使用过的 block。

Tier 2: GPU warm free blocks
  已经用过，但当前 ref_cnt == 0 的 GPU block。

Tier 3: CPU-backed blocks
  当前不驻留在 GPU 上的 cold block。
  在高压时会被拉回 GPU，在低压时会被主动搬回 CPU。
```

这里有一个很重要的概念：

- `num_gpu_blocks` 仍然是逻辑 pool 的上限，不在运行期动态改。
- 真正动态变化的是 `target_resident_blocks` / `resident_window`，它由高/低水位控制。
- 所谓“增大 GPU 上的 KV 驻留量”，指的是在高压阶段把更多 block 保持 GPU resident；“缩小”则是主动释放一部分冷 block 到 CPU。

## 2. 现有链路怎么接

当前最关键的接入点是这几处。

| 文件 | 作用 | 现状 |
| --- | --- | --- |
| `workloads/vllm/vllm/vllm/v1/core/block_pool.py` | block 选择和 free queue 管理 | 已经能插选块策略 |
| `workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py` | admission 判定 | 已经在 `allocate_slots()` 做 pressure gate |
| `workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py` | KV runtime policy | 已有 pressure trace、free block 选择、L5/L6 元数据、active protect |
| `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py` | residency 账本 | 已有 pool budget、state、victim 选择、trace |
| `workloads/vllm/vllm/vllm/device_allocator/uvm_policy_runtime.py` | 具体 UVM 动作 facade | 已有 `advise/prefetch` 类接口 |
| `workloads/vllm/vllm/vllm/v1/kv_offload/cpu.py` | CPU backend | 已有 CPU 后备池抽象 |
| `workloads/vllm/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | load/store 管线 | 已有 scheduler/worker 双端传输流程 |
| `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py` | KV tensor 初始化与 pointer map | 已经能导出 KV pointer map |
| `workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py` | 初始化 wiring | 已有 KV / sleep / pre-service pool 入口 |

一句话总结现状：

```text
vLLM 已经有“固定逻辑池 + 选块 + CPU offload/load + residency trace”的骨架，
缺的是“按压力做双向调节的历史状态分层选块”和“resident window 的闭环控制”。
```

## 3. 需要补的状态模型

只看 `ref_cnt` 不够。建议把 KV block 的状态拆成两层。

### 3.1 Residency state

描述 block 当前驻留在哪个 medium。

```text
gpu
cpu
prefetching_gpu
prefetching_cpu
unknown
```

### 3.2 Lifecycle state

描述 block 的历史身份。

```text
never_used
used_free
cpu_offloaded
```

它们的关系是：

- `residency_state` 说的是“现在在哪”。
- `lifecycle_state` 说的是“这块以前被怎么用过”。
- 一个 block 可以是 `used_free + gpu`，也可以是 `cpu_offloaded + cpu`。

### 3.3 推荐转移

```text
never_used/gpu -> used_free/gpu -> cpu_offloaded/cpu -> prefetching_gpu -> used_free/gpu
```

约束只有一个：

- 只有 `ref_cnt == 0` 的 block 才允许 spill / prefetch / 迁移。
- active request 的 block 只允许 protect，不允许后台改 table。

## 4. 运行时策略

### 4.1 选块顺序

`BlockPool.get_new_blocks()` 的本地选块顺序仍然是：

1. 先从 free queue 里挑 `never_used` 的 GPU block。
2. 不够再挑 `used_free` 的 GPU block。
3. 再不够时，才让 policy 触发 CPU-backed block 回填。

resident controller 不等到“GPU 不够”才动作，而是根据压力高/低主动调节 GPU resident set。

### 4.2 压力驱动的双向迁移语义

CPU-backed blocks 不是只在缺块时才被拉回 GPU，也不是只在回收时才下放到 CPU，而是 resident window 的两侧：

1. 压力偏高时，优先从 CPU-backed pool 里挑冷 block prefetch/load 回 GPU，扩大 resident set。
2. 压力偏低时，优先挑 `ref_cnt == 0` 的冷 GPU block store 到 CPU，释放 GPU 空间给其它池。
3. 迁移只作用于 `ref_cnt == 0` block。
4. 目标是让 KV resident set 随压力波动，而不是等到“GPU pool 不够”才被动补空位。

### 4.3 resident window / 双阈值控制器

建议增加一个 runtime 控制器，维护：

- `pressure_high_watermark`
- `pressure_low_watermark`
- `target_resident_blocks`
- `max_resident_blocks`
- `min_resident_blocks`
- `budget_step_up`
- `budget_step_down`
- `cooldown_steps`

原则是：

- 压力高于高水位时，上调 `target_resident_blocks`，把 CPU-backed block 拉回 GPU。
- 压力低于低水位时，下调 `target_resident_blocks`，把一部分冷 GPU block 搬回 CPU。
- 让 resident set 保持滞回，避免在临界点来回抖动。

## 5. 建议的实现顺序

### Phase 0: 只补状态和 trace

目标是先把“fresh / warm / cpu-backed”这三类 block 讲清楚。

要做的事：

- 给 KV region / block metadata 增加 `lifecycle_state`。
- 在 `register_region` / `mark_state` / `record_access` 里把历史状态记全。
- 在 trace 里区分 `never_used`、`used_free`、`cpu_offloaded`，以及 resident set 的扩张/收缩方向。
- 不改变现有行为。

验收标准：

- 只有 trace，不影响 benchmark。
- 能从 JSONL 里看出 block 的生命周期。

### Phase 1: 三段式选块

目标是让 GPU free queue 的选择顺序先按 fresh，再按 warm。

要做的事：

- 修改 `UvmKvRuntimePolicy.select_free_blocks_for_allocation()`。
- 修改 `BlockPool.get_new_blocks()`，让它优先消费 policy 返回的 fresh block。
- 维持当前 `ref_cnt == 0` 安全约束。

验收标准：

- 有 fresh block 时，不去抢 warm block。
- 没有 fresh block 时，允许复用 warm block。
- 不引入 active block 迁移。

### Phase 2: CPU 后备池双向 load/store

目标是让 KV GPU resident set 可以双向伸缩，而不是只会“缺块时回填”。

当前这一阶段已经落地“运行时手动调整 KV pool size + Stage L6 物理 prefetch”的版本：

- 通过 control file 或等价 runtime 配置更新 `target_resident_blocks`。
- target 变小时，只对 `ref_cnt == 0` 的冷块触发 CPU 方向迁移；active request 不受影响。
- target 变大时，只把 CPU-backed 的安全块 prefetch 回 GPU，重新进入可分配窗口。
- 当 `VLLM_USE_UVM=1`、Stage L6 pointer map 可用，且 `VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE=prefetch_cpu` / `VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE=prefetch` 时，动态 resize 会对 KV block 的 raw ranges 调用 `prefetch_range_to_cpu()` / `prefetch_range_to_device()`。
- 当没有 UVM managed pointer、没有 pointer range，或 prefetch 调用不可发出时，保持 metadata-only 降级路径，trace 会写出 `physical_migration_issued=False` 和 fallback reason。
- 验收脚本是 `workloads/vllm/check_kv_dynamic_pool_phase2_prefetch_real.py`，实验序列为 `target_blocks=2 -> 4 -> 1`。

要做的事：

- 复用 `kv_transfer/offloading_connector` 或 `CPUOffloadingSpec` 的 load/store 机制。
- 高压时，从 CPU-backed pool 中挑 block 触发 `prepare_load` / `prefetch`，把它们拉回 GPU。
- 低压时，对 `ref_cnt == 0` 的冷 GPU block 触发 `prepare_store` / `offload`，把它们搬到 CPU。
- 迁移只针对安全 block，不碰 active request 的 block table。

验收标准：

- 高压时，能看到 `residency_kv_prefetch_back_*` trace，且 GPU resident set 扩大。
- 低压时，能看到 `residency_kv_data_offload_*` trace，且 GPU 端可用空间释放出来给其它池。
- active request 不被打断。
- 物理路径启用时，trace 里应出现 `physical_migration_issued=True`、`range_count > 0`；真实 UVM residency 快照应能看到 KV 的 GPU/CPU resident bytes 随 target 扩大/收缩。

注意：KV tensor 的最大逻辑 allocation 仍然存在，因此 `torch.cuda.memory_allocated()` / `memory_reserved()` 不一定下降；判断物理驻留变化应优先看 residency trace、real residency snapshot 和 driver/page-level 统计。

### Phase 3: 双阈值 resident window 控制

目标是让 resident window 能随压力增减。

要做的事：

- 增加 hysteresis。
- 用 `allocation_pressure` / `would_deny_allocation` / 失败重试来调节 resident window。
- 高压时上调 resident window，低压时下调 resident window。
- 收缩只影响 `ref_cnt == 0` 的冷块。

验收标准：

- 高压时 GPU resident 数量变大。
- 低压时 GPU resident 数量变小，并释放空间给其它池。
- 不会在临界点来回抖。

### Phase 4: 集成和回归

目标是把策略接到初始化和运行主链路。

要做的事：

- 在 worker 初始化时接入 CPU backing 配置。
- 在 model runner 初始化时注册 KV 生命周期元数据。
- 补全 benchmark / self-test。

验收标准：

- 低压、高压两种场景都能跑通。
- failed requests = 0。
- trace 能解释每一次 block 选择。

## 6. 文件修改计划

### 必改

| 文件 | 建议修改 |
| --- | --- |
| `workloads/vllm/vllm/vllm/v1/core/block_pool.py` | 给 `get_new_blocks()` 加 fresh/warm 两段式选择；补充选块 trace；必要时把“选块计划”从纯 list 升级成 plan 对象。 |
| `workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py` | 增加 block lifecycle 分类、resident window controller、双向 rebalance 入口；扩展现有 Stage J/L5/L6 trace。 |
| `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py` | 在 region metadata 里加入 `lifecycle_state`、`ever_used`、`cpu_backed`、`last_cpu_store_ts`、`last_gpu_prefetch_ts`；增加 budget hysteresis 账本。 |
| `workloads/vllm/vllm/vllm/device_allocator/uvm_policy_runtime.py` | 增加 KV block prefetch/store facade，统一包住 `prefetch_range_to_cpu`、`prefetch_range_to_device`、`advise_range_preferred_location`。 |
| `workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py` | 当 `get_num_free_blocks()` 不够时，先问 policy 能否 rebalance，再决定是否返回 `None`。 |
| `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py` | 初始化时写更完整的 KV pointer map 和 block 元数据，标记初始 lifecycle 为 `never_used`。 |
| `workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py` | 接入 adaptive KV 配置、CPU 后备池大小、resident window 参数。 |

### 视需要修改

| 文件 | 建议修改 |
| --- | --- |
| `workloads/vllm/vllm/vllm/v1/kv_offload/cpu.py` | 如果需要更明确的 CPU block reserve 语义，可把 `num_cpu_blocks`、`eviction_policy`、`resident_budget` 显式暴露出来。 |
| `workloads/vllm/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | 如果要把 adaptive KV 跟现有 KV transfer 串得更紧，可以在 scheduler/worker 之间补一层“rebalance plan” metadata。 |
| `workloads/vllm/vllm/vllm/config/vllm.py` | 如果要把这套能力做成正式配置项，再把 env 或 CLI 参数映射到 `kv_transfer_config`。 |

## 7. 配置项建议

### 复用现有开关

```text
VLLM_UVM_KV_RUNTIME_ENABLE
VLLM_UVM_KV_RUNTIME_MODE
VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS
VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS
VLLM_UVM_RESIDENCY_ENABLE
VLLM_UVM_RESIDENCY_MODE
VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE
VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE
VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE
```

### 建议新增

```text
VLLM_UVM_KV_ADAPTIVE_ENABLE
VLLM_UVM_KV_PRESSURE_HIGH_WATERMARK
VLLM_UVM_KV_PRESSURE_LOW_WATERMARK
VLLM_UVM_KV_MIN_HOTSET_BLOCKS
VLLM_UVM_KV_TARGET_RESIDENT_BLOCKS
VLLM_UVM_KV_MAX_RESIDENT_BLOCKS
VLLM_UVM_KV_RESIDENT_BUDGET_STEP_UP
VLLM_UVM_KV_RESIDENT_BUDGET_STEP_DOWN
VLLM_UVM_KV_RESIDENT_BUDGET_COOLDOWN_STEPS
VLLM_UVM_KV_CPU_BACKING_BLOCKS
VLLM_UVM_KV_LIFECYCLE_TRACE_FILE
```

建议默认值全部保守，且默认不开启 adaptive mode。

## 8. Trace 设计

建议继续沿用 `residency_*` 风格，不另起一套命名。

### 建议新增或扩展的 action

```text
residency_kv_lifecycle_register
residency_kv_lifecycle_update
residency_kv_allocation_plan
residency_kv_rebalance_plan
residency_kv_cpu_store_attempt
residency_kv_cpu_store_issued
residency_kv_cpu_load_attempt
residency_kv_cpu_load_issued
residency_kv_resident_budget_update
residency_kv_resident_budget_hysteresis
```

### 关键字段

```text
block_id
group_id
region_id
ref_cnt
lifecycle_state
residency_state
ever_used
cpu_backed
resident_budget_blocks
target_resident_blocks
fresh_candidates
warm_candidates
cpu_backed_candidates
spill_victims
prefetch_targets
```

## 9. 测试与验收

### 单元测试

- fresh block 优先于 warm block。
- 高压时 CPU-backed block 优先于低热度 GPU block 被拉回。
- 低压时 warm / cold GPU block 会优先被下放到 CPU。
- active block 不会被纳入 victim。
- resident window 具备 hysteresis，不会抖动。

### 集成测试

- 小池子 + 低压场景，确认主要复用 fresh block。
- 中等压力场景，确认 resident window 不抖动，且 warm block 会被优先复用。
- 高压场景，确认 CPU-backed block 被拉回 GPU，resident set 扩大。
- 低压场景，确认部分 KV block 被搬到 CPU，GPU 空间被释放给其它池。

### 验收标准

- `failed requests = 0`
- 没有活跃 request 的 block table 被后台改写
- trace 能解释每次 block 选择
- CPU 后备池动作只出现在 `ref_cnt == 0` 的 block 上

## 10. 风险与回退

### 风险

- load/store 会增加尾延迟。
- resident window 太激进会导致频繁抖动。
- 如果 pointer map 不完整，CPU 回填只能降级成 trace-only。
- 如果 offloading backend 没接好，调度可能出现“看起来能回填，实际上不能”的假阳性。

### 回退

最简单的回退方式是关掉下面这些开关：

```text
VLLM_UVM_KV_ADAPTIVE_ENABLE=0
VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE=0
VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS=0
VLLM_UVM_KV_RUNTIME_MODE=trace_only
```

回退后应退回到当前 Stage J 的语义：

- 只做 pressure trace。
- 只做安全的 free block 选择。
- 不做真正的 CPU load/store 闭环。

## 11. 最终建议

第一版别追求“真正自动扩缩容 GPU HBM”，那会把问题一下子拉到 driver / remap 层。

更稳的做法是先把这件事做成：

```text
固定最大 KV pool
+ 生命周期分类
+ resident window
+ CPU 后备池
+ 三段式选块
+ 安全的 load/store 闭环
```

这样既能保住 vLLM 现有的高速路径，也能把单机场景里 KV 驻留空间按压力做双向调节，低压时让位给别的池，高压时把 KV 拉回 GPU。
