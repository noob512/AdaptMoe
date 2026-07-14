# vLLM UVM Unified HBM Residency Phase U1 Results

日期：2026-07-12

## 目标

Phase U1 只统一观测，不改变行为：

1. 新增统一 trace-only controller。
2. 把 KV block 和 expert region 从 residency ledger 汇总为统一 object snapshot。
3. 在 scheduler step / KV pressure / KV target 变化处输出 budget 和 pressure trace。
4. 不发起真实 prefetch、offload、driver reclaim 或 state mutation。

## 实现

改动文件：

- `workloads/vllm/vllm/vllm/device_allocator/uvm_unified_residency_controller.py`
  - 新增 `UnifiedResidencyController`。
  - 默认由 `VLLM_UVM_UNIFIED_RESIDENCY_ENABLE=1` 打开。
  - 输出 `unified_config`、`unified_budget_update`、`unified_object_state`、`unified_scheduler_step`。
  - KV/expert 请求只记录：
    - `unified_kv_grow_request`
    - `unified_kv_shrink_request`
    - `unified_expert_reclaim_request`
    - `unified_expert_prefetch_request`
  - 所有 unified action 均带 `trace_only=true`。

- `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py`
  - 新增全量只读 region snapshot API：
    - `UvmPoolResidencyManager.regions_snapshot()`
    - `get_uvm_residency_regions_snapshot()`

- `workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py`
  - 每次 `schedule()` 输出一次 `scheduler_step` 级统一 snapshot。
  - pressure 字段包含 running/waiting/scheduled/preempted request 数、token budget、scheduled tokens。

- `workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py`
  - KV runtime config / allocation pressure / dynamic target grow-shrink / resize 后 snapshot 接入 unified trace。
  - 接入只调用 U1 记录函数，不改变原 KV runtime 行为。

- `workloads/vllm/check_unified_hbm_residency_phase1_trace_only.py`
  - 新增 Phase U1 验收脚本。

## Trace 可重建字段

每条 `unified_object_state` / `unified_budget_update` 都包含：

- `kv_resident_bytes`
- `expert_resident_bytes`
- `reserved_slot_budget`
- `runtime_safety_margin`
- `protected_objects`
- `evictable_objects`

对象字段包括：

- KV：`object_type=kv`、`group_id`、`block_id`、`bytes`、`residency_state`、`protection_state`
- Expert：`object_type=expert`、`layer_id`、`expert_id`、`role`、`bytes`、`residency_state`、`protection_state`

## 验收命令

```bash
cd /home/ubuntu/nvidia-uvm-gpu
workloads/vllm/check_unified_hbm_residency_phase1_trace_only.py
```

结果目录：

```text
/tmp/vllm_unified_hbm_phase1_trace_only_20260712_125151_552003
```

关键文件：

```text
/tmp/vllm_unified_hbm_phase1_trace_only_20260712_125151_552003/vllm_uvm_unified_residency_phase1.jsonl
/tmp/vllm_unified_hbm_phase1_trace_only_20260712_125151_552003/vllm_uvm_residency_phase1.jsonl
/tmp/vllm_unified_hbm_phase1_trace_only_20260712_125151_552003/phase1_trace_only_summary.json
```

## 验收结果

总体：PASS

Trace action 计数：

```text
unified_config: 1
unified_budget_update: 4
unified_object_state: 4
unified_scheduler_step: 1
unified_kv_grow_request: 1
unified_kv_shrink_request: 1
unified_expert_reclaim_request: 1
unified_expert_prefetch_request: 1
unified_layer_boundary: 2
```

合成 registry 快照：

```text
kv_resident_bytes = 131072
expert_resident_bytes = 12582912
reserved_slot_budget = 8388608
runtime_safety_margin = 1048576
protected_object_count = 3
evictable_object_count = 2
object_counts = {"expert": 3, "kv": 2}
budget_ok = true
```

验收项：

```text
unified_config_present: PASS
budget_update_present: PASS
object_state_present: PASS
scheduler_step_present: PASS
kv_object_present: PASS
expert_object_present: PASS
reconstructable_step_fields: PASS
protected_objects_present: PASS
evictable_objects_present: PASS
kv_grow_request_present: PASS
kv_shrink_request_present: PASS
expert_reclaim_request_trace_only: PASS
expert_prefetch_request_trace_only: PASS
all_unified_actions_trace_only: PASS
no_forbidden_real_migration_actions: PASS
perf_allocation_sequence_unchanged: PASS
perf_baseline_close: PASS
```

微基准：

```text
block-manager iterations = 20000
U1 disabled elapsed = 0.054358s
U1 enabled elapsed = 0.051811s
enabled / disabled ratio = 0.953
allocation sequence head unchanged = [1, 2, 3, ..., 16]
sequence_sum unchanged = 639507
```

说明：本次性能验收是无 GPU 的 block-manager 微基准，验证 U1 trace-only 接入不改变 allocation 行为，且基础开销与 baseline 基本一致。真实 vLLM 端到端长跑可后续用相同环境变量打开 U1 trace 复测。

## 复现环境变量

核心 U1 开关：

```bash
export VLLM_UVM_UNIFIED_RESIDENCY_ENABLE=1
export VLLM_UVM_UNIFIED_RESIDENCY_TRACE_FILE=/tmp/u1/unified.jsonl
export VLLM_UVM_UNIFIED_RESIDENCY_TRACE_OBJECTS=1
```

可选预算覆盖：

```bash
export VLLM_UVM_UNIFIED_MANAGED_HBM_BUDGET_BYTES=$((64 * 1024 * 1024))
export VLLM_UVM_UNIFIED_RUNTIME_SAFETY_MARGIN_BYTES=$((1024 * 1024))
export VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT=2
export VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES=$((4 * 1024 * 1024))
```
