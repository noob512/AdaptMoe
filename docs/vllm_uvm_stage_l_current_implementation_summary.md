# vLLM UVM Stage L 当前实现总结

本文档总结当前仓库中围绕 vLLM UVM 三池驻留管理、KV cache 安全迁移、以及 Stage K + Stage L 联合调度已经实现的功能、实现方式、修改文件和验收方法。

当前实现覆盖 Stage L0 到 Stage L7.3 的实验性控制面。其中 L6/L7 已经完成真实 KV pointer map、KV safe block offload、prefetch-back、以及 Stage K coordinator 统一 action budget 验证。L7.2/L7.3 已补齐 cross-pool 验收：weights expert prefetch/offload 与 KV offload/prefetch-back 可以进入同一个 coordinator 预算实验，并显式验证 priority 只作为全局 action 调度元数据，不破坏 per-pool 安全规则。

## 1. 总体目标

Stage L 的目标是在 vLLM 中建立应用层的 UVM GPU residency control plane，用于把 GPU 显存驻留意图拆成三个 pool：

```text
weights pool:
  MoE expert weights / model weights 的热冷驻留估计与 cold expert 让位。

kv pool:
  KV cache block 的 metadata 注册、安全候选筛选、safe offload、prefetch-back。

scratch pool:
  scratch/device-direct 临时内存预算报告与 fallback 语义。
```

Stage L 负责 resident capacity，即“哪些对象可以估计为 GPU resident，以及如果超预算应该从同 pool 里找什么 victim”。Stage K 负责 action bandwidth，即“本 step 还能发起多少 bytes 的 prefetch/offload action”。L7 把二者接起来，让 weights/KV/scratch 的 action 通过同一个 coordinator trace 统一计账。

重要语义：

```text
Stage L residency:
  application-managed intent, not CUDA driver page residency。

Stage K coordinator:
  action budget gate, not a raw memory migration executor。

priority:
  当前只作为 global action scheduling trace metadata，不会绕过 per-pool safety。
```

## 2. 当前阶段能力矩阵

### 2.1 Stage L0：三池 residency ledger

已实现能力：

```text
1. 新增 UvmPoolResidencyManager。
2. 支持 weights / kv / scratch 三个 pool。
3. 支持 region register、residency request、state update、access record、action record。
4. 维护 estimated_gpu_resident_bytes，而不是查询 driver page residency。
5. 输出统一 JSONL trace 和 residency_summary。
6. 支持 trace_only 与 enforce 两种模式。
```

核心文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_l_success.py
```

实现方式：

```text
UvmResidencyRegion:
  region_id / pool / ptr / bytes / state / estimated_gpu_resident / evictable / active / metadata。

UvmResidencyDecision:
  request 是否 allowed、是否 would_deny、pool/global remaining bytes、victims。

request_uvm_gpu_residency:
  统一处理三池预算判断。

record_uvm_residency_action:
  记录真实或模拟迁移动作，例如 weight victim、KV offload、prefetch-back。

mark_uvm_residency_state:
  记录 gpu / cpu / prefetching_cpu / prefetching_gpu 状态变化，并更新 estimated bytes。
```

当前边界：

```text
1. L0 是应用层估计，不等价于 UVM driver 的真实 page residency。
2. trace_only 模式不会阻断请求。
3. enforce 只适合在具体 stage 已有安全迁移路径后逐步打开。
```

### 2.2 Stage L1：weights expert residency 接入

已实现能力：

```text
1. MoE expert weight slice 被注册为 weights pool region。
2. hot expert prefetch 前会向 Stage L 请求 residency。
3. active expert 会记录 access 与 active_region_ids。
4. trace 中可看到 expert region_id、role、global_expert_id、local_expert_id、ptr、bytes。
```

核心文件：

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/check_stage_l1_success.py
```

实现方式：

```text
_maybe_prefetch_uvm_expert_weights:
  从 topk_ids 解析 active expert。
  对 moe_gate_up / moe_down 等 role 生成 weight slice。
  调用 register_uvm_residency_region(pool="weights")。
  调用 request_uvm_gpu_residency(pool="weights")。
  写入 Stage I weight prefetch trace 和 Stage L residency trace。
```

### 2.3 Stage L2：weights hot/cold enforce 与 victim 让位

已实现能力：

```text
1. 支持 weight_initial_state，默认 cold weights 可按 cpu 状态估计。
2. 支持 hot set enforce。
3. 当 hot expert residency 被拒绝时，可从 weights pool 选择 cold expert victim。
4. victim action 支持 advise_cpu / prefetch_cpu 语义。
5. active expert 不会作为 cold victim。
```

核心文件：

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/check_stage_l2_success.py
```

实现方式：

```text
_register_uvm_residency_cold_weight_candidates:
  将非当前 active expert 标为 cold candidate。

_offload_uvm_residency_weight_victims:
  对 Stage L 返回的 victims 发起 configured victim action。

request_uvm_gpu_residency(active_region_ids=...):
  residency manager 在选择 victim 时避开 active_region_ids。
```

当前边界：

```text
1. weights enforce 主要面向 MoE expert slices。
2. 全模型 dense weights 的完整热冷管理未纳入当前 L7.3 验收。
```

### 2.4 Stage L3：scratch reserve reporting

已实现能力：

```text
1. residency summary 中包含 scratch reserve、margin、live、peak、fallback/reject 统计字段。
2. scratch pool 的设计语义是预留容量池，不作为运行时 eviction victim 池。
3. scratch 超预算时应走 device-direct admission 限制或 managed fallback，而不是驱逐 active scratch。
```

核心文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_l3_success.py
```

实现方式：

```text
record_scratch_summary:
  把 Stage G/scratch 侧统计写入 Stage L summary。

summary fields:
  scratch_peak_bytes
  scratch_live_bytes
  scratch_reserved_bytes
  scratch_margin_bytes
  scratch_budget_over_records
  scratch_budget_reject_records
  scratch_eviction_records
```

当前边界：

```text
scratch 当前以报告和 fallback 语义为主，未实现 active scratch eviction。
```

### 2.5 Stage L4：KV cache sizing report

已实现能力：

```text
1. 支持根据三池预算计算 KV sizing budget。
2. 支持 report_only / clamp / override 策略字段。
3. KVCacheConfig 中保存原始 block 数、最终 block 数、原始 KV bytes、最终 KV bytes、budget blocks 等字段。
4. residency summary 中包含 kv_sizing_* 字段。
```

核心文件：

```text
workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py
workloads/vllm/vllm/vllm/v1/kv_cache_interface.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/check_stage_l4_success.py
```

实现方式：

```text
_apply_stage_l4_residency_kv_sizing:
  在 vLLM 构造 KVCacheConfig 后，根据 VLLM_UVM_RESIDENCY_* budget 计算目标 KV blocks。

_report_kv_cache_config:
  把 sizing 结果写入 residency manager。

KVCacheConfig 新增字段:
  uvm_residency_kv_sizing_enabled
  uvm_residency_kv_sizing_policy
  uvm_residency_kv_sizing_applied
  uvm_residency_kv_sizing_original_num_blocks
  uvm_residency_kv_sizing_final_num_blocks
  uvm_residency_kv_sizing_budget_bytes
  uvm_residency_kv_sizing_budget_blocks
```

当前边界：

```text
默认推荐 report_only，避免过早改变 vLLM 的 KV allocation 行为。
```

### 2.6 Stage L5：KV block metadata-only residency

已实现能力：

```text
1. BlockPool 初始化时注册 KV block metadata。
2. region_id 使用 kv:<group_id>:<block_id>。
3. 每个 KV block region 包含 bytes_per_block、ref_cnt、in_free_queue、is_prefix_cache、block_hash 等 metadata。
4. 可记录 safe candidate / unsafe candidate。
5. L5 不需要 raw pointer，ptr=None，metadata_only=True，offload_capable=False。
```

核心文件：

```text
workloads/vllm/vllm/vllm/v1/core/block_pool.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/check_stage_l5_success.py
```

实现方式：

```text
BlockPool.__init__:
  如果 UvmKvRuntimePolicy 存在，调用 register_stage_l5_kv_blocks(self)。

UvmKvRuntimePolicy.register_stage_l5_kv_blocks:
  遍历 KV block id 和 kv_cache_group id，注册 metadata-only KV regions。

record_kv_metadata_candidate:
  把 safe candidate 写入 residency trace。
```

已验证样例结果：

```text
region_records=4095
kv_safe_candidate_records>0
unsafe_candidate_records=0
kv_data_offload_records=0
failed_requests=0
```

### 2.7 Stage L6：KV raw pointer map + safe offload + prefetch-back

已实现能力：

```text
1. 在 KV tensor 分配完成后生成 raw KV pointer map sidecar。
2. 将 kv:<group_id>:<block_id> 从 metadata-only region 升级为 offload_capable=True。
3. 只对 safe KV blocks 发起 offload。
4. 支持 data offload mode:
   trace_only
   prefetch_cpu
5. 支持 prefetch-back mode:
   trace_only
   prefetch
6. 支持 state transition:
   gpu -> prefetching_cpu -> cpu -> prefetching_gpu -> gpu
7. 验收脚本支持 strict-issued，即要求真实 issued records > 0。
8. check_stage_l6_success.py 已修正 runtime trace / pointer-map sidecar 解析，避免实际成功但误报 FAIL。
```

核心文件：

```text
workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/check_stage_l6_success.py
```

实现方式：

```text
gpu_model_runner._write_stage_l6_kv_pointer_map:
  在 initialize_kv_cache_tensors 后拿到真实 kv_caches[layer].data_ptr()。
  只处理简单 contiguous CUDA tensor。
  按 KV block page size 切分出 block ranges。
  写出 stage_l6_kv_pointer_map sidecar。

UvmKvRuntimePolicy._maybe_load_stage_l6_pointer_map:
  从 pointer-map sidecar 读取 raw ptr/ranges。
  调用 register_stage_l6_kv_pointer_map。

UvmKvRuntimePolicy.register_stage_l6_kv_pointer_map:
  将 region 重新注册为 ptr>0、metadata_only=False、offload_capable=True。

UvmKvRuntimePolicy._maybe_execute_stage_l6_safe_offload:
  从 free_block_queue 中选择 ref_cnt=0 且非 null 的 safe block。
  对每个 group/block 调用 _maybe_offload_region。

UvmKvRuntimePolicy._maybe_offload_region:
  检查 safe、ptr_present、offload_capable、ranges。
  如果 mode=prefetch_cpu，则调用 prefetch_range_to_cpu。
  成功后写 residency_kv_data_offload_issued。

UvmKvRuntimePolicy._maybe_prefetch_back_region:
  当 block 再次被复用前，如果 region state 为 cpu/prefetching_cpu，则预取回 GPU。
  如果 mode=prefetch，则调用 prefetch_range_to_device。
  成功后写 residency_kv_prefetch_back_issued。
```

已验证样例结果：

```text
offload_issued=33
prefetch_issued=30
unsafe_offloads=0
pointer_map_entries=4095
kv_pointer_map_loaded=True
kv_pointer_map_registered=True
failed_requests=0
```

推荐验收命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l6_success.py --gpu-run --mode trace_only \
  --data-offload-mode prefetch_cpu \
  --prefetch-back-mode prefetch \
  --strict-issued
```

当前边界：

```text
1. pointer map 第一版只支持简单 contiguous/uniform KV cache tensor。
2. hybrid、cross-layer shared、复杂 stride layout 会 fallback，不会冒险 offload。
3. L6 的 --mode trace_only 表示 residency admission 仍不强制拒绝请求，但 data-offload-mode/prefetch-back-mode 可以发起真实 cudaMemPrefetchAsync。
```

### 2.8 Stage L7：KV actions 接入 Stage K coordinator

已实现能力：

```text
1. KV safe block offload 不再只由 Stage L6 自己决定 action budget，而是先请求 Stage K coordinator。
2. KV prefetch-back 同样通过 coordinator。
3. coordinator trace 中可看到 KV actions:
   kv_safe_block_prefetch_cpu
   kv_block_prefetch_gpu
4. coordinator mode 支持 trace_only / enforce。
5. L7 check 要求 coordinator grants、KV offload issued、KV prefetch-back issued、unsafe offloads zero。
```

核心文件：

```text
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py
workloads/vllm/check_stage_l7_success.py
```

实现方式：

```text
_maybe_offload_region:
  在真正 offload 前调用 request_uvm_pool_action(pool="kv", action="kv_safe_block_prefetch_cpu")。

_maybe_prefetch_back_region:
  在 prefetch-back 前调用 request_uvm_pool_action(pool="kv", action="kv_block_prefetch_gpu")。

metadata.stage:
  写为 stage_l7，便于 checker 区分这是 Stage K + Stage L 联合路径。
```

已验证样例结果：

```text
coordinator_grants=63
coordinator_denies=0
actions={'kv_safe_block_prefetch_cpu': 33, 'kv_block_prefetch_gpu': 30}
offload_issued=33
prefetch_issued=30
unsafe_offloads=0
failed_requests=0
```

推荐验收命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l7_success.py --gpu-run \
  --mode trace_only \
  --coordinator-mode trace_only
```

### 2.9 Stage L7.2：weights + KV cross-pool coordinator budget experiment

已实现能力：

```text
1. 新增 cross-pool 验收脚本。
2. 脚本先跑 planning phase，生成 Stage H weight expert action plan。
3. main phase 同时启用:
   Stage I weights expert prefetch
   Stage I weights cold expert offload
   Stage L6 KV safe offload / prefetch-back
   Stage K pool coordinator
4. 同一个 coordinator trace 中必须同时出现 weights pool 与 kv pool。
5. 同一个 coordinator trace 中必须同时出现:
   expert_prefetch_gpu
   expert_<offload_mode>
   kv_safe_block_prefetch_cpu
   kv_block_prefetch_gpu
```

核心文件：

```text
workloads/vllm/check_stage_l7_cross_pool_success.py
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py
```

实现方式：

```text
check_stage_l7_cross_pool_success.py planning phase:
  调用 run_kv_fault_ratio.sh。
  打开 weight map 和 MoE routing trace。
  调用 plan_stage_h_weight_expert_actions.py 生成 plan_json。

check_stage_l7_cross_pool_success.py main phase:
  打开 uvm-weight-prefetch-enable。
  打开 uvm-weight-offload-enable。
  打开 uvm-residency-kv-data-offload-enable。
  打开 uvm-residency-kv-prefetch-back-enable。
  打开 uvm-pool-coordinator-enable。
  统一传入 coordinator global / kv / weight / scratch bytes-per-step。

fused_moe.layer:
  expert hot prefetch 通过 request_uvm_pool_action(pool="weights", action="expert_prefetch_gpu")。
  cold expert offload 通过 request_uvm_pool_action(pool="weights", action=f"expert_{mode}")。

uvm_kv_runtime_policy:
  KV offload/prefetch-back 通过 request_uvm_pool_action(pool="kv", ...)。
```

推荐验收命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l7_cross_pool_success.py --gpu-run \
  --mode trace_only \
  --coordinator-mode trace_only \
  --priority kv,scratch,weights
```

关键 PASS 条件：

```text
weights_pool_visible=True
kv_pool_visible=True
weight_prefetch_actions_visible=True
weight_offload_actions_visible=True
kv_actions_visible=True
kv_offload_issued_present=True
kv_prefetch_issued_present=True
unsafe_kv_offloads_zero=True
weight_prefetch_trace_present=True
weight_offload_trace_present=True
benchmark_no_failed_requests=True
```

### 2.10 Stage L7.3：priority 语义检查

已实现能力：

```text
1. coordinator config 和 request trace 中记录 priority。
2. request trace 中记录 priority_index 和 priority_rank。
3. request trace 中记录 priority_semantics="global_action_order_only"。
4. request trace 中记录 per_pool_safety_semantics="budget_and_callsite_rules_are_independent"。
5. cross-pool checker 要求 KV 和 weights 的 priority rank 与配置一致。
6. checker 同时要求 per-pool safety 仍成立，例如 unsafe_kv_offloads_zero=True。
```

核心文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py
workloads/vllm/check_stage_l7_cross_pool_success.py
```

实现方式：

```text
UvmPoolCoordinator.__init__:
  解析 VLLM_UVM_POOL_COORDINATOR_PRIORITY 为 priority_order。

UvmPoolCoordinator._record_config_locked:
  写出 priority_semantics 和 per_pool_safety_semantics。

UvmPoolCoordinator._record_decision_locked:
  对每个 request 写出 priority_index / priority_rank。

check_stage_l7_cross_pool_success.py:
  校验 priority_config_matches。
  校验 priority_rank_matches_kv。
  校验 priority_rank_matches_weights。
  校验 priority_safety_semantics_visible。
```

当前边界：

```text
1. L7.3 当前验证的是 priority metadata 与 per-pool safety 不被破坏。
2. coordinator 当前没有实现跨 pool action queue 的真实重排序器。
3. global enforce 下一步仍建议只允许 weights cold expert 作为 global reclaim victim，KV/scratch 继续以安全 trace/fallback 为主。
```

## 3. 修改文件清单

### 3.1 核心运行时代码

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
```

新增 Stage L residency manager，负责三池 region ledger、budget decision、summary、state/action trace。

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py
```

扩展 Stage K coordinator，支持 global/pool action budget、request trace、priority rank、priority semantics、per-pool safety semantics。

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
```

接入 MoE expert weights 的 residency register、hot prefetch、cold offload、Stage K coordinator request、weight prefetch/offload trace。

```text
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
```

实现 Stage J/L5/L6/L7 KV runtime policy：metadata-only region、pointer map 加载、safe offload、prefetch-back、coordinator request、runtime trace。

```text
workloads/vllm/vllm/vllm/v1/core/block_pool.py
```

在 BlockPool 初始化和 block allocate/free 路径接入 UvmKvRuntimePolicy，用于注册 KV blocks、记录 allocation pressure、触发复用前 prefetch-back。

```text
workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py
```

实现 KV budget 和 Stage L4 residency KV sizing，写入 KVCacheConfig 并上报 residency summary。

```text
workloads/vllm/vllm/vllm/v1/kv_cache_interface.py
```

扩展 KVCacheConfig 字段，保存 UVM KV budget 与 Stage L4 sizing 结果。

```text
workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py
```

生成 Stage L6 KV pointer-map sidecar。只映射简单 contiguous CUDA tensor，复杂 layout 安全 fallback。

```text
workloads/vllm/run_kv_fault_ratio.sh
```

新增大量 CLI/env 参数，统一向 vLLM server 注入 Stage I/J/K/L 控制变量，并保存 allocator/runtime/residency/coordinator/weight trace。

### 3.2 验收脚本

```text
workloads/vllm/check_stage_l_success.py
```

Stage L0 residency manager self-test / trace check。

```text
workloads/vllm/check_stage_l1_success.py
```

Stage L1 expert weight residency 接入验收。

```text
workloads/vllm/check_stage_l2_success.py
```

Stage L2 weights hot/cold enforce 与 victim trace 验收。

```text
workloads/vllm/check_stage_l3_success.py
```

Stage L3 scratch reserve reporting 验收。

```text
workloads/vllm/check_stage_l4_success.py
```

Stage L4 KV sizing report 验收。

```text
workloads/vllm/check_stage_l5_success.py
```

Stage L5 KV metadata-only residency 验收。

```text
workloads/vllm/check_stage_l6_success.py
```

Stage L6 KV pointer map、safe offload、prefetch-back 验收。支持 --strict-issued，且会解析 runtime trace 和 pointer-map sidecar。

```text
workloads/vllm/check_stage_l7_success.py
```

Stage L7 KV actions 经 Stage K coordinator 的验收。

```text
workloads/vllm/check_stage_l7_cross_pool_success.py
```

Stage L7.2/L7.3 cross-pool 验收。要求 weights prefetch/offload 与 KV offload/prefetch-back 进入同一个 coordinator trace，并校验 priority rank 与 per-pool safety semantics。

### 3.3 设计文档

```text
docs/vllm_uvm_stage_l_residency_implementation_roadmap.md
docs/vllm_uvm_stage_l_pool_residency_manager_plan.md
docs/vllm_uvm_stage_l0_pool_residency_implementation.md
docs/vllm_uvm_stage_l1_weight_residency_integration.md
docs/vllm_uvm_stage_l2_weight_residency_enforce.md
docs/vllm_uvm_stage_l_current_implementation_summary.md
```

其中本文档是当前实现状态的汇总索引，roadmap/plan 仍作为后续阶段设计依据。

## 4. 关键环境变量与 runner 参数

### 4.1 Stage L residency

```text
VLLM_UVM_RESIDENCY_ENABLE
VLLM_UVM_RESIDENCY_STAGE
VLLM_UVM_RESIDENCY_MODE
VLLM_UVM_RESIDENCY_TRACE_FILE
VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES
VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES
VLLM_UVM_RESIDENCY_KV_GPU_BYTES
VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES
VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES
VLLM_UVM_RESIDENCY_PRIORITY
```

### 4.2 Stage L4/L5/L6 KV

```text
VLLM_UVM_RESIDENCY_KV_SIZING_ENABLE
VLLM_UVM_RESIDENCY_KV_SIZING_POLICY
VLLM_UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS
VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE
VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE
VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE
VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE
VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE
VLLM_UVM_RESIDENCY_KV_PREFETCH_DEVICE
VLLM_UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP
VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE
```

### 4.3 Stage J KV runtime

```text
VLLM_UVM_KV_RUNTIME_ENABLE
VLLM_UVM_KV_RUNTIME_MODE
VLLM_UVM_KV_RUNTIME_BUDGET_BYTES
VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS
VLLM_UVM_KV_RUNTIME_TRACE_FILE
VLLM_UVM_KV_RUNTIME_EVICTION_POLICY
VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT
VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE
VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS
```

### 4.4 Stage I weights

```text
VLLM_UVM_WEIGHT_PREFETCH_ENABLE
VLLM_UVM_WEIGHT_PREFETCH_MODE
VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE
VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP
VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER
VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES
VLLM_UVM_WEIGHT_PREFETCH_DEVICE
VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE
VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN
VLLM_UVM_WEIGHT_OFFLOAD_ENABLE
VLLM_UVM_WEIGHT_OFFLOAD_MODE
VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE
VLLM_UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP
VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER
VLLM_UVM_WEIGHT_OFFLOAD_TARGET_ROLES
```

### 4.5 Stage K coordinator

```text
VLLM_UVM_POOL_COORDINATOR_ENABLE
VLLM_UVM_POOL_COORDINATOR_MODE
VLLM_UVM_POOL_COORDINATOR_TRACE_FILE
VLLM_UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP
VLLM_UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP
VLLM_UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP
VLLM_UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP
VLLM_UVM_POOL_COORDINATOR_PRIORITY
```

## 5. 推荐验收流程

### 5.1 快速静态检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu
python3 -m py_compile \
  workloads/vllm/check_stage_l6_success.py \
  workloads/vllm/check_stage_l7_success.py \
  workloads/vllm/check_stage_l7_cross_pool_success.py \
  workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py \
  workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py \
  workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py

git diff --check -- \
  workloads/vllm/check_stage_l7_cross_pool_success.py \
  workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py
```

### 5.2 L6 严格 GPU 验收

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l6_success.py --gpu-run --mode trace_only \
  --data-offload-mode prefetch_cpu \
  --prefetch-back-mode prefetch \
  --strict-issued
```

期望：

```text
Stage L6 Success Check: PASS
offload_issued > 0
prefetch_issued > 0
pointer_map_entries > 0
kv_pointer_map_loaded=True
kv_pointer_map_registered=True
unsafe_offloads=0
failed_requests=0
```

### 5.3 L7 KV coordinator 验收

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l7_success.py --gpu-run \
  --mode trace_only \
  --coordinator-mode trace_only
```

期望：

```text
Stage L7 Success Check: PASS
coordinator_grants > 0
coordinator_denies = 0
kv_safe_block_prefetch_cpu > 0
kv_block_prefetch_gpu > 0
offload_issued > 0
prefetch_issued > 0
unsafe_offloads = 0
failed_requests = 0
```

### 5.4 L7.2/L7.3 cross-pool 验收

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l7_cross_pool_success.py --gpu-run \
  --mode trace_only \
  --coordinator-mode trace_only \
  --priority kv,scratch,weights
```

期望：

```text
Stage L7.2/L7.3 Cross-Pool Check: PASS
weights_pool_visible=True
kv_pool_visible=True
weight_prefetch_actions_visible=True
weight_offload_actions_visible=True
kv_actions_visible=True
priority_config_matches=True
priority_rank_matches_kv=True
priority_rank_matches_weights=True
priority_safety_semantics_visible=True
kv_offload_issued_present=True
kv_prefetch_issued_present=True
unsafe_kv_offloads_zero=True
weight_prefetch_trace_present=True
weight_offload_trace_present=True
benchmark_no_failed_requests=True
```

## 6. 当前已知边界

### 6.1 residency 不是 driver page truth

Stage L 的 `estimated_gpu_resident_bytes` 是应用层 intent 账本，不是 CUDA driver 的真实 resident page 查询结果。真实 UVM fault stats 仍需结合 kernel/UVM trace 判断。

### 6.2 trace_only 与真实 migration 是两层开关

`--mode trace_only` 表示 residency admission 不强制拒绝请求；但 L6 的 `--data-offload-mode prefetch_cpu` 和 `--prefetch-back-mode prefetch` 可以发起真实 cudaMemPrefetchAsync。因此 L6/L7 中可能同时出现：

```text
residency mode = trace_only
KV data migration mode = prefetch_cpu / prefetch
```

这不是矛盾，而是控制面与迁移动作面的分离。

### 6.3 L6 pointer map 仍是保守第一版

当前只映射简单 contiguous CUDA tensors。复杂 KV layout、hybrid groups、cross-layer shared tensor、non-contiguous stride 会 fallback，不会强行 offload。

### 6.4 L7.3 priority 当前是 trace semantics

当前实现会记录 priority rank，并验证 priority 不破坏 per-pool safety；但还没有实现真正的跨 pool action queue 重排序器。

### 6.5 enforce 模式仍需谨慎

L6/L7/L7.2/L7.3 的推荐验收模式仍是：

```text
residency mode = trace_only
coordinator mode = trace_only
KV data-offload-mode = prefetch_cpu
KV prefetch-back-mode = prefetch
```

在 pointer map、prefetch-back、cross-pool budget 都稳定后，才建议逐步尝试 coordinator enforce。residency admission 的强 enforce 不应早于安全迁移路径成熟，否则容易把控制面验证变成请求被预算拒绝。

## 7. 下一步建议

推荐后续分三步推进：

```text
1. L7.4:
   增加 coordinator enforce 的 A/B 小流量实验。
   先只 enforce action budget，不 enforce residency admission。

2. L8:
   实现 global pressure 下的 safe reclaim policy。
   第一版只允许 weights cold expert 作为 global reclaim victim。
   KV 和 scratch 继续保持安全边界。

3. L9:
   扩展 KV pointer map layout 覆盖面。
   支持 hybrid / shared / non-contiguous layout 的可证明安全映射。
```

推荐新增验收点：

```text
1. coordinator enforce 下 denied action 不导致 failed_requests。
2. global pressure 下 KV active block 不被 offload。
3. scratch fallback 不触发 active scratch eviction。
4. weights cold expert offload 不影响当前 active expert。
5. KV fault ratio 在 real prefetch_cpu/prefetch 模式下保持 0 或显著下降。
```
