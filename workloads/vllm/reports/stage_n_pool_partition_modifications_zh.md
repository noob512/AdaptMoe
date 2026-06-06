# Stage N 分池管理实现修改说明

本文档说明为了实现 GPU 显存分池管理，本项目已经修改了哪些文件、每个文件承担什么职责，以及这些修改如何串成一条完整链路。这里的“分池管理”主要指把 vLLM 推理过程中的 UVM 托管内存按语义拆成多个池：

- `kv`：KV cache，主要来自 decode/prefill 期间的 KV block。
- `weights`：模型权重，生命周期长，通常在模型加载后长期驻留。
- `scratch` / `temp`：运行时 scratch、workspace、warmup workspace，生命周期短、分配频繁。

整体设计是：应用层负责识别内存语义并向 driver 打 tag，系统层负责保存 region 元数据、维护每个池的预算和状态、查询 resident/candidate/victim 字段，并在 soft/enforce 模式下执行 reclaim。

## 总体链路

分池管理不是单靠 kernel 或单靠 vLLM 完成的。核心链路如下：

1. vLLM UVM allocator 拦截/记录分配。
2. allocator 根据运行 phase、size、tensor 名称、KV pointer map 等信息推断 allocation class。
3. vLLM 把 allocation class 映射到 driver pool：`kv`、`weights`、`scratch`。
4. vLLM 通过 Phase F/Stage N ioctl 向 UVM driver 注册 region，并带上 pool、role、flags、state。
5. vLLM 通过 `SET_POOL_BUDGET` 给 driver 设置三池预算。
6. driver 在 `va_space` 内保存 region tree 和 pool state。
7. `GET_POOL_STATE` 汇总每个池的 `resident_gpu_bytes`、`evictable_bytes`、`candidate_bytes`、`victim_count` 等字段。
8. `RECLAIM_POOL` 或自动 reclaim hook 根据 victim candidate 把超预算池的 region 迁移回 CPU。
9. vLLM smoke / P7 sweep / hard partition proof 脚本解析日志，判断三池 tag、pool state、auto reclaim、hard partition 证据是否满足验收。

pool id 和模式约定来自 driver ABI：

| 名称 | ID | 说明 |
|---|---:|---|
| `unknown` | 0 | 未分类或兜底 |
| `kv` | 1 | KV cache |
| `weights` | 2 | 模型权重 |
| `scratch` | 3 | 临时 scratch / workspace |
| `activation` | 4 | 预留给 activation 类内存 |

| 模式 | 值 | 语义 |
|---|---:|---|
| `trace_only` | 0 | 只观测预算、候选和 victim，不执行迁移 |
| `soft` | 1 | 超预算时允许 best-effort reclaim |
| `enforce` | 2 | 超预算时尝试 reclaim-to-budget，并输出 deny/block 证据 |

## 应用层修改（vLLM）

应用层的职责是“告诉 driver 每块内存是什么”。UVM driver 本身只能看到 VA range 和 fault，无法天然知道某个地址属于 KV、权重还是 scratch。因此 vLLM 层新增了分类、注册、预算下发、状态采样和实验验收逻辑。

### `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

这是应用层最核心的 C++ allocator shim。它负责拦截 UVM 分配、分类内存、向 driver 发送 ioctl、记录 scratch/tag/pool-state 证据。

主要修改如下：

1. 新增 Stage N ioctl wrapper。

   文件中定义了这些 ioctl 编号和参数结构：

   - `UVM_PHASE_N_SET_POOL_BUDGET_IOCTL`
   - `UVM_PHASE_N_GET_POOL_STATE_IOCTL`
   - `UVM_PHASE_N_UPDATE_REGION_STATE_IOCTL`
   - `UVM_PHASE_N_QUERY_REGION_IOCTL`
   - `UVM_PHASE_N_QUERY_REGION_BY_VA_IOCTL`
   - `UVM_PHASE_N_RECLAIM_POOL_IOCTL`

   并导出了 C ABI 函数：

   - `uvm_driver_phase_n_set_pool_budget`
   - `uvm_driver_phase_n_get_pool_state`
   - `uvm_driver_phase_n_metadata_state_smoke`
   - `uvm_driver_phase_n_reclaim_pool`

   这些函数让 Python/vLLM 层可以通过 ctypes 调 driver 控制面。

2. 新增 allocation class 和 pool kind。

   allocator 现在会把分配分为：

   - `WeightPersistent`
   - `KvPersistent`
   - `WarmupWorkspace`
   - `RuntimeScratch`
   - `RuntimeWorkspace`
   - `UnknownManaged`

   然后再映射为：

   - `WeightPersistent` -> `PoolKind::Weights`
   - `KvPersistent` -> `PoolKind::KvCache`
   - `RuntimeScratch` / `RuntimeWorkspace` / `WarmupWorkspace` -> `PoolKind::RuntimeScratch`
   - `UnknownManaged` -> `PoolKind::OtherManaged`

   这一步是分池管理的语义入口。如果这里分类错误，driver 后续的预算、candidate、victim 都会落到错误池。

3. 新增 driver scratch tag 和 scratch budget accounting。

   allocator 增加了 runtime scratch 的 live bytes 统计和准入函数：

   - `reserve_driver_scratch_budget(...)`
   - `release_driver_scratch_budget(...)`
   - `driver_scratch_live_bytes`
   - `driver_scratch_peak_live_bytes`
   - `driver_scratch_budget_over_allocs`
   - `driver_scratch_budget_reject_allocs`

   当 `VLLM_UVM_DRIVER_SCRATCH_TAG_ENABLE=1` 时，allocator 会把 runtime scratch/workspace 注册为 driver pool `scratch`。当 `VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE=1` 时，scratch 分配会先做 live-bytes 预算检查，超过预算则输出 reject 证据。

   关键日志包括：

   - `TRACE_DRIVER_SCRATCH_TAG`
   - `TRACE_DRIVER_SCRATCH_UNTAG`
   - `TRACE_DRIVER_SCRATCH_REJECT`

   这些日志用于证明 scratch 池是否真的被 tag、释放、或者被预算拒绝。

4. 新增 allocator 侧 driver pool state snapshot。

   `trace_driver_pool_state_snapshot(...)` 会主动调用 `GET_POOL_STATE` 查询 `kv`、`weights`、`scratch` 三池，并在 allocator trace 中输出：

   - `TRACE_DRIVER_POOL_STATE_SNAPSHOT`
   - `budget_bytes`
   - `resident_gpu_bytes`
   - `resident_cpu_bytes`
   - `evictable_bytes`
   - `protected_bytes`
   - `candidate_bytes`
   - `candidate_regions`
   - `selected_victim_region_id`
   - `selected_victim_bytes`
   - `victim_count`
   - `metadata_only`

   这样可以在 benchmark 进行中或 scratch free 附近抓到 transient pool 的状态，而不是只依赖 post-run 状态。

5. 新增环境变量控制。

   allocator 层新增或消费的关键环境变量包括：

   - `VLLM_UVM_KV_BUDGET_BYTES`
   - `VLLM_UVM_KV_BUDGET_MODE`
   - `VLLM_UVM_WEIGHT_BUDGET_BYTES`
   - `VLLM_UVM_WEIGHT_BUDGET_MODE`
   - `VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE`
   - `VLLM_UVM_POOL_REGISTRY_ENABLE`
   - `VLLM_UVM_DRIVER_SCRATCH_TAG_ENABLE`
   - `VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES`
   - `VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE`
   - `VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP`
   - `VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE`
   - `VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_EVERY`
   - `VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_MAX`

### `workloads/vllm/vllm/vllm/device_allocator/uvm.py`

这是 Python 侧 ctypes bridge。它把 `uvm_allocator.so` 里的 C ABI 函数包装成 Python 函数，供 vLLM residency manager 调用。

主要修改如下：

1. 注册 Stage N 函数签名。

   `_configure_library(...)` 现在会检测并配置：

   - `uvm_driver_phase_n_set_pool_budget`
   - `uvm_driver_phase_n_get_pool_state`
   - `uvm_driver_phase_n_reclaim_pool`

2. 新增预算下发接口。

   `driver_set_pool_budget(...)` 把 pool id、budget bytes、gpu id、mode、flags 传给 C++ wrapper，再进入 driver ioctl。

3. 新增 pool state 查询接口。

   `driver_get_pool_state(...)` 调用 `GET_POOL_STATE`，并把 36 个返回槽位解析成 Python dict，例如：

   - `budget_bytes`
   - `resident_gpu_bytes`
   - `resident_cpu_bytes`
   - `reclaim_bytes`
   - `protected_bytes`
   - `evictable_bytes`
   - `over_budget_bytes`
   - `reclaim_target_bytes`
   - `victim_candidate_bytes`
   - `victim_candidate_regions`
   - `selected_victim_region_id`
   - `selected_victim_bytes`
   - `victim_count`
   - `metadata_only`

4. 新增 reclaim 调用接口。

   `driver_reclaim_pool(...)` 用于 manual phase-n-reclaim 和实验脚本触发 driver reclaim，返回 attempted/success/failure/reclaimed/victim 等字段。

### `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py`

这是 vLLM 语义 residency manager。它负责把 vLLM 已经识别出的 region、pool、budget 同步给 driver，并输出结构化 JSONL 证据。

主要修改如下：

1. 新增 driver pool budget 配置读取。

   manager 读取：

   - `VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE`
   - `VLLM_UVM_DRIVER_POOL_BUDGET_MODE`
   - `VLLM_UVM_DRIVER_POOL_BUDGETS`
   - `VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES`
   - `VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY`
   - `VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_DEVICE`

2. 新增 `_sync_driver_pool_budgets_locked(...)`。

   该函数遍历 driver pool：

   - `kv`
   - `weights`
   - `scratch`

   对每个池调用 `driver_set_pool_budget(...)`，并在 residency trace 中输出：

   - `action="residency_driver_pool_budget_config"`
   - `pool`
   - `pool_id`
   - `budget_bytes`
   - `mode`
   - `status`

3. 新增 `driver_pool_state_snapshot(...)`。

   该函数调用 `driver_get_pool_state(...)` 查询三池状态，并输出：

   - `action="residency_driver_pool_state"`
   - `states.kv`
   - `states.weights`
   - `states.scratch`
   - `failures`

   验收脚本后续会要求 `driver_pool_state_failures=0`，并要求三池都出现有效 budget/resident/candidate/victim 字段。

4. 新增 manual reclaim 管理入口。

   `driver_soft_reclaim_pool(...)` 调用 driver reclaim，用于 `phase-n-reclaim` 三池验证，证明 driver 不只是能观测，还能返回 victim 并执行迁移。

### `workloads/vllm/run_kv_fault_ratio.sh`

这是主 vLLM benchmark runner。它现在不只是启动 server 和 benchmark，还负责加载 patched UVM 模块、传递 Stage N 参数、在 ready 之后打开自动 reclaim。

主要修改如下：

1. 新增 UVM 模块参数环境变量。

   - `UVM_PHASE_N_AUTO_RECLAIM_ENABLE`
   - `UVM_PHASE_N_AUTO_RECLAIM_EXECUTE`
   - `UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY`
   - `UVM_PHASE_N_AUTO_RECLAIM_EVERY`
   - `UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY`
   - `UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS`
   - `UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS`
   - `UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS`

2. 新增 ready-gated execute。

   `UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY=1` 时，脚本先以 `execute=0` 加载模块，等 vLLM server ready、KV range 建好后，再把 `/sys/module/nvidia_uvm/parameters/uvm_phase_n_auto_reclaim_execute` 写成目标值。

   这样避免模型加载期或 warmup 期过早触发 reclaim，降低 fault/reclaim storm 风险。

3. 新增参数校验和 cleanup。

   脚本会检查 module param 实际加载值是否符合预期，并在 cleanup 时把 auto reclaim execute 关回 `0`。

### `workloads/vllm/run_stage_n_tag_only_three_pool_smoke.sh`

这是三池 tag-only smoke 的核心验收脚本。它负责把一次 vLLM smoke 转换成可读 summary。

主要修改如下：

1. 设置三池预算和模式。

   脚本导出：

   - `VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE=1`
   - `VLLM_UVM_DRIVER_POOL_BUDGET_MODE`
   - `VLLM_UVM_DRIVER_POOL_BUDGETS="kv=...,weight=...,temp=..."`
   - `VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES`

2. 开启 scratch tag 和 pool-state snapshot。

   脚本导出：

   - `VLLM_UVM_DRIVER_SCRATCH_TAG_ENABLE=1`
   - `VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE=1`
   - `VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY`

3. 解析 tag-only 证据。

   summary 中会统计：

   - `scratch_driver_tag_records`
   - `scratch_driver_untag_records`
   - `scratch_driver_live_peak_bytes`
   - `scratch_driver_max_alloc_bytes`
   - `scratch_driver_over_budget_records`
   - `scratch_driver_reject_records`

4. 解析 driver pool state 证据。

   summary 中会要求：

   - `driver_pool_state_records > 0`
   - `driver_pool_state_failures = 0`
   - `driver_pool_state_all_pools_present = 1`
   - `driver_pool_state_all_budgets_positive = 1`
   - `driver_pool_state_all_required_fields_present = 1`
   - `allocator_driver_pool_state_all_status_ok = 1`

5. 解析 auto hook 和 hard partition 证据。

   当要求自动 reclaim 或 hard partition 时，summary 会统计：

   - `phase_n_auto_hook_records`
   - `phase_n_auto_hook_pool_records`
   - `phase_n_auto_hook_would_reclaim_records`
   - `phase_n_auto_hook_execute_records`
   - `phase_n_auto_hook_success_records`
   - `phase_n_auto_hook_reclaimed_records`
   - `hard_partition_pass`
   - `hard_partition_failure_reasons`

### `workloads/vllm/run_stage_n_three_pool_validation.sh`

这是三池完整验证脚本，组合了 tag-only smoke 和 manual phase-n-reclaim。

主要修改如下：

1. 先跑 tag-only 三池 smoke，确认：

   - KV / weights / scratch 都能注册 tag。
   - driver pool budget 已下发。
   - `GET_POOL_STATE` 三池状态可读且无失败。
   - scratch tag/untag/live peak/max alloc 能被解析。

2. 再跑 manual phase-n-reclaim，分别对：

   - `kv`
   - `weights`
   - `scratch`

   触发 `RECLAIM_POOL`，确认每个池都有 victim trace、candidate bytes 和 reclaim 结果。

3. 汇总成一个 summary。

   关键字段包括：

   - `phase_n_reclaim_kv_victim_count`
   - `phase_n_reclaim_weights_victim_count`
   - `phase_n_reclaim_scratch_victim_count`
   - `phase_n_reclaim_*_candidate_bytes`
   - `driver_pool_state_failures`
   - `allocator_driver_pool_state_all_status_ok`

### `workloads/vllm/run_stage_n_p7_ab_sweep.sh`

这是 P7 A/B 和参数 sweep 脚本。它用于比较 native UVM、Stage N trace、Stage N soft、Stage N enforce probe 的性能和行为。

主要修改如下：

1. 新增 variant。

   典型 variant 包括：

   - `native_uvm`
   - `stage_n_trace`
   - `stage_n_soft`
   - `stage_n_soft_sweep`
   - `stage_n_enforce_probe`

2. 新增 soft/enforce 参数 sweep。

   脚本可以扫：

   - scratch budget
   - auto reclaim cooldown
   - max victims
   - execute-on-ready

3. 新增 hard partition summary 字段。

   P7 summary 现在会汇总：

   - `Auto exec`
   - `Hard pass`
   - `Max over bytes`
   - `Hard reclaimed bytes`
   - `Hard block`
   - `Scratch reject`
   - `Failed req`

   这使得性能指标和分池控制证据可以放在同一张表中比较。

### `workloads/vllm/run_stage_n_hard_partition_proof.sh`

这是硬性分区证明实验的 wrapper。

主要修改如下：

1. 固定使用 `stage_n_enforce_probe`。
2. 默认设置严格预算：

   - KV 默认 `10 GiB`
   - weights 默认 `20 GiB`
   - scratch 默认 `100 MiB`

3. 默认打开严格验收：

   - `REQUIRE_STAGE_N_HARD_PARTITION=1`
   - `FAIL_ON_WARN=1`

4. 默认使用更激进的 reclaim 参数：

   - `UVM_PHASE_N_AUTO_RECLAIM_EVERY=1`
   - `UVM_PHASE_N_ENFORCE_RECLAIM_SCAN_EVERY=0`
   - `UVM_PHASE_N_ENFORCE_RECLAIM_MAX_VICTIMS=128`
   - `UVM_PHASE_N_ENFORCE_RECLAIM_COOLDOWN_EPOCHS=0`

   这些参数用于证明“超预算后必须 reclaim-to-budget 或产生 deny/block 证据”。由于它很激进，完整实验可能更容易触发 fault/reclaim storm，需要先用小规模 probe 验证。

### `workloads/vllm/check_stage_n_hard_partition.py`

这是硬分区验收解析器。它把“是否实现硬性限制”从普通 PASS/WARN 中单独抽出来。

主要修改如下：

1. 解析 residency trace 中的 `residency_driver_pool_state`。
2. 解析 route/kernel log 中的 `phase_n_auto_reclaim_hook`。
3. 解析 allocator trace 中的 scratch tag/untag/reject。
4. 对每个池判断：

   - 如果 `resident_gpu_bytes <= budget_bytes`，该池满足预算。
   - 如果超预算，但存在 `auto_reclaim_to_budget` 证据，也可接受。
   - 如果超预算，但存在 pool-specific deny/block/reject 证据，也可接受。
   - 如果超预算且没有 reclaim-to-budget 或 deny/block 证据，则 hard partition 不通过。

5. 输出关键字段：

   - `hard_partition_pass`
   - `hard_partition_all_driver_pools_under_budget`
   - `hard_partition_all_transient_pools_under_budget`
   - `hard_partition_sufficient_reclaim_to_budget`
   - `hard_partition_sufficient_deny_or_block`
   - `hard_partition_pool_kv_over_budget_bytes`
   - `hard_partition_pool_weights_over_budget_bytes`
   - `hard_partition_pool_scratch_over_budget_bytes`
   - `hard_partition_failure_reasons`

### `workloads/vllm/vllm/uvm_test/phase_f_smoke.cpp`

这是低层 smoke 工具的一部分。它补充了 Phase N ioctl 的直接调用测试，用来验证：

- `SET_POOL_BUDGET` 是否可调用。
- `GET_POOL_STATE` 是否返回预算和 resident/candidate/victim 字段。
- `RECLAIM_POOL` 是否能返回 victim 和 reclaim 结果。

这类 smoke 不依赖完整 vLLM server，适合快速定位 driver ioctl 或 C++ wrapper 问题。

### `workloads/vllm/build_stage_n_modules.sh`

这是新增的构建辅助脚本。它解决了一个很实际的问题：只编译 `nvidia-uvm` 时，`nvidia-uvm.ko` 依赖的 `nvUvmInterface*` 符号来自 `nvidia.ko`，单独构建容易在 `modpost` 阶段报 unresolved symbol。

脚本默认执行：

```bash
make -C kernel-module/nvidia-module/kernel-open \
  -j"$JOBS" modules NV_KERNEL_MODULES="nvidia nvidia-uvm"
make -C workloads/vllm/vllm/uvm_test uvm
```

并支持：

```bash
FORCE_REBUILD=1 ./workloads/vllm/build_stage_n_modules.sh
```

用于清理后重建 kernel module 和 allocator shim。

## 系统层修改

系统层的职责是“保存分池状态并执行控制动作”。它不理解 vLLM 的 tensor 语义，只信任应用层传入的 region metadata。

### `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`

这是 UVM ioctl ABI 定义文件。Stage N 在这里新增了分池管理需要的枚举、flags 和 ioctl 参数结构。

主要修改如下：

1. 新增 pool 枚举。

   - `UVM_PHASE_N_POOL_UNKNOWN`
   - `UVM_PHASE_N_POOL_KV`
   - `UVM_PHASE_N_POOL_WEIGHT`
   - `UVM_PHASE_N_POOL_SCRATCH`
   - `UVM_PHASE_N_POOL_ACTIVATION`

2. 新增预算模式枚举。

   - `UVM_PHASE_N_MODE_TRACE_ONLY`
   - `UVM_PHASE_N_MODE_SOFT`
   - `UVM_PHASE_N_MODE_ENFORCE`

3. 新增 region state 和 flags。

   state 用于描述冷热/活跃程度：

   - `UNKNOWN`
   - `COLD`
   - `WARM`
   - `HOT`
   - `ACTIVE`
   - `INACTIVE`

   flags 用于描述是否可回收、是否保护、是否 transient：

   - `UVM_PHASE_N_REGION_FLAG_EVICTABLE`
   - `UVM_PHASE_N_REGION_FLAG_PROTECTED`
   - `UVM_PHASE_N_REGION_FLAG_ACTIVE`
   - `UVM_PHASE_N_REGION_FLAG_PREFETCHABLE`
   - `UVM_PHASE_N_REGION_FLAG_PINNED`
   - `UVM_PHASE_N_REGION_FLAG_TRANSIENT`

4. 新增 Stage N ioctl。

   - `UVM_PHASE_N_SET_POOL_BUDGET`
   - `UVM_PHASE_N_GET_POOL_STATE`
   - `UVM_PHASE_N_UPDATE_REGION_STATE`
   - `UVM_PHASE_N_QUERY_REGION`
   - `UVM_PHASE_N_QUERY_REGION_BY_VA`
   - `UVM_PHASE_N_RECLAIM_POOL`

5. 扩展返回字段。

   `GET_POOL_STATE` 和 `RECLAIM_POOL` 都返回足够的验收字段，例如 budget、resident、over-budget、target、candidate、selected victim、victim batch、reclaim success/failure event。

### `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h`

这里把 Stage N 状态挂到了每个 `uvm_va_space_t` 上。

新增字段包括：

- `phase_f_regions`
- `phase_f_region_ranges`
- `phase_n_pools[UVM_PHASE_N_POOL_STATE_MAX]`

作用如下：

1. `phase_f_regions` 用 region id 索引，方便 `QUERY_REGION`、`UPDATE_REGION_STATE` 和 reclaim 扫描。
2. `phase_f_region_ranges` 用 VA range 索引，方便 fault hook 根据 fault address 找到对应 region。
3. `phase_n_pools` 保存每个 pool 的预算、resident 汇总、candidate 汇总、victim 信息和事件计数。

这些状态放在 `va_space` 级别，意味着不同 vLLM 进程或不同 UVM VA space 的分池状态相互隔离。

### `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_api.h`

这里暴露了 Stage N reclaim/hook 函数，使其他 UVM 源文件可以调用。

新增声明包括：

- `uvm_phase_n_reclaim_pool_over_budget(...)`
- `uvm_phase_n_make_resident_reclaim_hook(...)`
- `uvm_phase_n_auto_reclaim_outer_hook(...)`

这些声明让 replayable fault 服务路径能够在合适的外层安全点调用 Stage N 自动 reclaim。

### `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`

这是系统层修改最多的文件，包含 region 注册、预算设置、pool state 查询、victim selection、manual reclaim、auto reclaim 和 ioctl 路由。

主要修改如下：

1. 新增 Stage N module parameters。

   - `uvm_phase_n_auto_reclaim_enable`
   - `uvm_phase_n_auto_reclaim_execute`
   - `uvm_phase_n_auto_reclaim_every`
   - `uvm_phase_n_auto_reclaim_scan_every`
   - `uvm_phase_n_auto_reclaim_max_victims`
   - `uvm_phase_n_auto_reclaim_cooldown_epochs`
   - `uvm_phase_n_auto_reclaim_failure_backoff_epochs`

   它们控制自动 reclaim 是否启用、是否执行、采样频率、扫描频率、每轮 victim 数、成功 cooldown 和失败 backoff。

2. 新增 region 注册与查询。

   Phase F region 注册会把应用层传入的：

   - `region_id`
   - `base`
   - `length`
   - `pool`
   - `layer_id`
   - `expert_id`
   - `role`
   - `flags`
   - `state`

   写入 `phase_f_regions` 和 `phase_f_region_ranges`。

   Stage N 依赖这些 region metadata 做后续 pool state 汇总和 victim selection。

3. 实现 `SET_POOL_BUDGET`。

   `uvm_api_phase_n_set_pool_budget(...)` 校验 pool 和 mode 后，把预算写入：

   ```c
   va_space->phase_n_pools[pool].budget_bytes
   va_space->phase_n_pools[pool].mode
   va_space->phase_n_pools[pool].flags
   ```

   同时输出：

   ```text
   phase_n_set_pool_budget pool=... budget=... mode=...
   ```

4. 实现 `GET_POOL_STATE`。

   `uvm_api_phase_n_get_pool_state(...)` 遍历该 pool 下所有 region，汇总：

   - `resident_gpu_bytes`
   - `resident_cpu_bytes`
   - `protected_bytes`
   - `evictable_bytes`
   - `over_budget_bytes`
   - `reclaim_target_bytes`
   - `victim_candidate_bytes`
   - `victim_candidate_regions`
   - `selected_victim_region_id`
   - `selected_victim_bytes`
   - `victim_count`

   当正常 GPU residency 查询不可用时，增加了 `metadata_only` fallback。这个 fallback 解决了早期 `GET_POOL_STATE status=-38` 的问题：即使当时不能拿到完整 GPU residency，也能基于已注册 metadata 返回三池预算和候选字段，避免验收中三池 state 缺失。

5. 实现 region state update。

   `UPDATE_REGION_STATE` 可以更新：

   - `state`
   - `flags_set`
   - `flags_clear`
   - `priority`
   - `last_access_epoch`

   这为后续 cooldown、LRU-ish victim selection、active/protected 判断提供基础。

6. 实现 `QUERY_REGION` 和 `QUERY_REGION_BY_VA`。

   `QUERY_REGION` 用 region id 查 metadata；`QUERY_REGION_BY_VA` 用 fault address 或任意 VA 查 region。自动 reclaim hook 依赖 by-VA 查询判断当前 fault 属于哪个 pool。

7. 实现 `RECLAIM_POOL`。

   `uvm_phase_n_reclaim_pool_execute(...)` 的流程是：

   - 扫描目标 pool 的全部 region。
   - 查询每个 region 的 resident bytes。
   - 计算 pool 是否超预算。
   - 计算 reclaim target。
   - 根据 flags/state/priority/last access 选择 victim。
   - 可选执行 `uvm_migrate_managed_range_to_cpu(...)`。
   - 更新 pool state 中的 resident/candidate/over-budget 字段。
   - 输出 `phase_n_reclaim_pool` trace。

   它同时服务 manual reclaim 和自动 reclaim。

8. 实现自动 reclaim hook。

   当前有两类 hook 语义：

   - `uvm_phase_n_make_resident_reclaim_hook(...)`：用于记录 make-resident 相关路径，但避免在锁内直接迁移。
   - `uvm_phase_n_auto_reclaim_outer_hook(...)`：在 replayable fault batch 外层安全点触发，允许执行 soft/enforce reclaim。

   `outer_hook` 会根据地址找到 region/pool，读取 pool budget 和 mode，然后决定：

   - trace-only：只输出 hook 记录。
   - soft：超预算时 best-effort reclaim。
   - enforce：执行全池 sweep，尝试 reclaim-to-budget；如果仍超预算，则输出 `block=1 deny=1 deny_or_block=1` 证据。

   关键日志格式是：

   ```text
   phase_n_auto_reclaim_hook phase=outer...
   phase_n_auto_reclaim_hook phase=outer_enforce...
   ```

   重要字段包括：

   - `pool`
   - `budget`
   - `resident_before`
   - `resident_after`
   - `over`
   - `target`
   - `candidate_bytes`
   - `victim_count`
   - `would_reclaim`
   - `execute_requested`
   - `execute`
   - `attempted`
   - `success`
   - `failure`
   - `reclaimed`
   - `reclaim_to_budget`
   - `block`
   - `deny`
   - `deny_or_block`

9. 新增稳定性控制。

   自动 reclaim 逻辑包含：

   - `every`：降低 hook 触发频率。
   - `scan_every`：控制 full scan 频率。
   - `max_victims`：限制每次 reclaim batch。
   - `cooldown_epochs`：成功后冷却。
   - `failure_backoff_epochs`：失败后退避。

   这些机制是 P7 调参的基础，用来避免 fault path 抖动或 reclaim storm。

10. 修改 ioctl 路由。

   Stage N ioctl 接入 `uvm_ioctl(...)`：

   - `SET_POOL_BUDGET`
   - `GET_POOL_STATE`
   - `UPDATE_REGION_STATE`
   - `QUERY_REGION`
   - `QUERY_REGION_BY_VA`
   - `RECLAIM_POOL`

   其中 `GET_POOL_STATE` 使用 heap route，避免大参数结构导致 ioctl 栈帧过大。

### `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

这里接入了自动 reclaim 的安全触发点。

主要修改如下：

1. 新增 `phase_n_auto_reclaim_after_replayable_batch(...)`。

   它遍历 replayable fault batch 中的 fault entry，最多处理 `UVM_PHASE_N_AUTO_RECLAIM_OUTER_MAX_PER_BATCH` 个 entry，并调用：

   ```c
   uvm_phase_n_auto_reclaim_outer_hook(
       entry->va_space,
       entry->fault_address,
       PAGE_SIZE,
       uvm_id_value(entry->gpu->id),
       0);
   ```

2. 在 `uvm_parent_gpu_service_replayable_faults(...)` 中接入。

   在 `service_fault_batch(...)` 成功之后调用 `phase_n_auto_reclaim_after_replayable_batch(...)`。

   这个位置的意义是：它处于 replayable fault batch 外层，不在 `va_block` lock 内直接迁移 victim，比在 make-resident 或 block-locked 路径内执行 reclaim 更安全。

## 验收证据如何对应文件

| 证据字段或日志 | 来源 | 说明 |
|---|---|---|
| `residency_driver_pool_budget_config` | `uvm_pool_residency.py` | vLLM 已把三池预算下发给 driver |
| `residency_driver_pool_state` | `uvm_pool_residency.py` + `uvm.py` + driver `GET_POOL_STATE` | driver 能返回三池 state |
| `TRACE_DRIVER_POOL_STATE_SNAPSHOT` | `uvm_allocator.cpp` | allocator 运行中主动抓三池状态 |
| `TRACE_DRIVER_SCRATCH_TAG` | `uvm_allocator.cpp` | scratch 分配被注册为 driver region |
| `TRACE_DRIVER_SCRATCH_UNTAG` | `uvm_allocator.cpp` | scratch 释放时解除/更新记录 |
| `TRACE_DRIVER_SCRATCH_REJECT` | `uvm_allocator.cpp` | scratch hard budget gate 拒绝分配 |
| `phase_n_set_pool_budget` | `uvm.c` | driver 收到预算设置 |
| `phase_n_get_pool_state` | `uvm.c` | driver 汇总并返回 pool state |
| `phase_n_reclaim_pool` | `uvm.c` | manual/auto reclaim 的 victim 和迁移结果 |
| `phase_n_auto_reclaim_hook` | `uvm.c` + `uvm_gpu_replayable_faults.c` | 自动 hook 命中 pool、是否执行、是否 reclaim-to-budget |
| `hard_partition_*` | `check_stage_n_hard_partition.py` | 严格 hard partition 验收结果 |

## 当前实现状态

已经完成的能力：

- 应用层能区分 KV、weights、scratch 三类主要内存。
- 应用层能把三类内存注册成 driver region。
- 应用层能下发三池预算。
- driver 能保存每个 VA space 的 region metadata 和 pool state。
- driver `GET_POOL_STATE` 能返回三池 budget/resident/candidate/victim 字段。
- driver 支持 manual `RECLAIM_POOL`。
- replayable fault batch 外层已经接入自动 reclaim hook。
- P7 脚本能跑 native/trace/soft/enforce probe 对比。
- hard partition checker 已收紧：任一池超预算且没有 reclaim-to-budget 或 deny/block/reject 证据，就不能判定为 hard partition PASS。

仍需谨慎对待的部分：

- `enforce` 已经有 reclaim-to-budget 和 deny/block 证据路径，但完整大模型 hard partition proof 在激进参数下可能触发长时间 fault/reclaim storm。
- weights 池如果超预算很多，需要足够大的 victim batch 和多轮 reclaim，否则会出现 candidate 很多但 reclaimed bytes 不足的问题。
- scratch 是 transient pool，post-run 可能接近 0，所以必须结合 allocator 中的 `TRACE_DRIVER_SCRATCH_*` peak/live/reject 证据判断。
- 当前 hard partition 验收更严格，不应只凭 `Auto exec > 0` 或 `reclaimed_bytes > 0` 宣称硬性分区成功。

## 推荐实验入口

构建：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
./workloads/vllm/build_stage_n_modules.sh
```

三池控制面和 manual reclaim 验证：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RESET_COUNTERS=1 UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_three_pool_validation.sh
```

P7 A/B：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/stage_n_p7_ab_sweep_$(date -u +%Y%m%d_%H%M%S) \
P7_VARIANTS=native_uvm,stage_n_trace,stage_n_soft \
REPEATS=3 PROMPTS=8 REQUEST_RATE=2 OUTPUT_LEN=32 \
SERVER_MAX_NUM_SEQS=4 RESET_COUNTERS=1 \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_p7_ab_sweep.sh
```

严格 hard partition proof：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/stage_n_hard_partition_proof_$(date -u +%Y%m%d_%H%M%S) \
RESET_COUNTERS=1 UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_hard_partition_proof.sh
```

如果要先做小规模探针，降低运行压力：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/stage_n_hard_partition_probe_$(date -u +%Y%m%d_%H%M%S) \
REPEATS=1 PROMPTS=1 REQUEST_RATE=1 OUTPUT_LEN=8 \
UVM_PHASE_N_ENFORCE_RECLAIM_MAX_VICTIMS=16 \
UVM_PHASE_N_ENFORCE_RECLAIM_COOLDOWN_EPOCHS=1024 \
RESET_COUNTERS=1 UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
./run_stage_n_hard_partition_proof.sh
```
