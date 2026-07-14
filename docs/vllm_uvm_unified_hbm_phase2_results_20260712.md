# vLLM UVM Unified HBM Residency Phase U2 Results

日期：2026-07-12

## 目标

Phase U2 实现手动 budget rebalance 闭环：

1. 初始 KV 小 resident budget、expert 大 resident budget。
2. 高 KV 压力时增大 KV target、reclaim cold expert、prefetch KV tail。
3. 低 KV 压力时 shrink KV target、释放 KV HBM、增大 weights/expert budget。

## 实现

改动文件：

- `workloads/vllm/vllm/vllm/device_allocator/uvm_unified_residency_controller.py`
  - 新增 unified control file。
  - 支持多行 `key=value` 或 JSON commands。
  - 支持：
    - `kv_target_blocks=N`
    - `weights_budget_bytes=N`
    - `reclaim_pool=weights required_bytes=N`
    - `prefetch_kv_blocks=0:3,0:4`，也支持 block range。

- `workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py`
  - 新增 `apply_unified_kv_target_blocks()`。
  - 新增 `prefetch_kv_blocks_from_unified()`。
  - KV dynamic pool refresh 时轮询 unified control file。

- `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py`
  - 新增 `set_uvm_driver_pool_budget()`。
  - 支持运行时单独更新 `weights` pool budget。

- `workloads/vllm/vllm/uvm_test/phase_n_3gb_rebalance_smoke.cu`
  - 新增可选 `--low-rebalance 1`。
  - 高压后增加低压尾段：降低 KV budget，并 reclaim KV tail region。

- `workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh`
  - 新增 `LOW_REBALANCE=1` 透传。

- `workloads/vllm/check_unified_hbm_residency_phase2_manual_rebalance.py`
  - 新增 Phase U2 总验收脚本。

## Control File 示例

高 KV 压力：

```text
kv_target_blocks=4
weights_budget_bytes=33554432
reclaim_pool=weights required_bytes=4194304
prefetch_kv_blocks=0:3,0:4
```

低 KV 压力：

```text
kv_target_blocks=1
weights_budget_bytes=67108864
```

## 验收命令

```bash
cd /home/ubuntu/nvidia-uvm-gpu
workloads/vllm/check_unified_hbm_residency_phase2_manual_rebalance.py
```

结果目录：

```text
/tmp/vllm_unified_hbm_phase2_manual_rebalance_20260712_134022_393895
```

关键文件：

```text
/tmp/vllm_unified_hbm_phase2_manual_rebalance_20260712_134022_393895/phase2_manual_rebalance_summary.json
/tmp/vllm_unified_hbm_phase2_manual_rebalance_20260712_134022_393895/phase2_unified_control_trace.jsonl
/tmp/vllm_unified_hbm_phase2_manual_rebalance_20260712_134022_393895/stage_n_low_rebalance/smoke.log
```

## 验收结果

总体：PASS

### Control File 自测

Unified action 计数：

```text
unified_control_file_applied: 3
unified_kv_target_applied: 2
unified_weights_budget_update: 2
unified_driver_reclaim_pool: 1
unified_prefetch_kv_blocks: 1
unified_kv_grow_request: 1
unified_kv_shrink_request: 1
unified_budget_update: 10
unified_object_state: 10
```

Ledger 方向：

```text
initial kv_resident_bytes = 2048
high    kv_resident_bytes = 4096
low     kv_resident_bytes = 1024

initial weights_budget_bytes = 0
high    weights_budget_bytes = 33554432
low     weights_budget_bytes = 67108864
```

验收项：

```text
control_runner_completed: PASS
control_file_applied: PASS
kv_target_command_applied: PASS
weights_budget_command_applied: PASS
reclaim_command_seen: PASS
prefetch_command_seen: PASS
ledger_kv_rises_on_high_pressure: PASS
ledger_kv_drops_on_low_pressure: PASS
```

### Driver Pool State 验收

Stage-N U2 low rebalance：

```text
initial:
  kv_budget       = 536870912
  kv_resident     = 536870912
  expert_budget   = 2684354560
  expert_resident = 2684354560

after_rebalance:
  kv_budget       = 1073741824
  kv_resident     = 1073741824
  expert_budget   = 2147483648
  expert_resident = 2147483648

low_rebalance:
  kv_budget       = 536870912
  kv_resident     = 536870912
  expert_budget   = 2684354560
  expert_resident = 2147483648
```

关键结论：

```text
KV resident bytes 随 target 上升:
  536870912 -> 1073741824

weights resident bytes 在 reclaim 后下降:
  2684354560 -> 2147483648

target 降低后 KV resident bytes 下降:
  1073741824 -> 536870912
```

Driver final：

```text
phase_n_3gb_u2_final
  low_budget_ok=1
  kv_low_after_ok=1
  total_ok=1
  kv_resident_initial=536870912
  kv_resident_high=1073741824
  kv_resident_low=536870912
  expert_resident_initial=2684354560
  expert_resident_high=2147483648
  expert_resident_low=2147483648
```

验收项：

```text
driver_smoke_completed: PASS
driver_u2_total_ok: PASS
driver_kv_resident_rises: PASS
driver_weights_resident_drops_after_reclaim: PASS
driver_kv_resident_drops_on_low_target: PASS
```

## 说明

低压释放 KV HBM 的 driver 验收使用 `reclaim-kv-tail-low`，而不是单独依赖 `cudaMemPrefetchAsync(..., cudaCpuDeviceId)`。实测普通 CPU prefetch 后 driver pool state 仍可能保持 KV resident 计数不变；受控 shrink/reclaim 才能让 Phase-N pool state 明确反映 resident bytes 下降。
