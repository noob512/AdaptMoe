# vLLM UVM 统一 HBM Residency Phase U0 实现与实验结果

对应计划文档：

```text
docs/vllm_uvm_unified_hbm_residency_management_plan.md
```

Phase U0 目标是冻结当前基线，并修通进入统一 controller 前必须可复现的 smoke：

1. 记录 built/loaded `nvidia_uvm` srcversion。
2. 复跑 KV dynamic target `2 -> 4 -> 1`。
3. 修复并复跑 Phase-N 3GiB rebalance smoke。
4. 复跑 manual migration fault control smoke。
5. LOWTRACE 四组基线保留可复跑入口。

## 实现改动

### Stage-N 3GiB rebalance smoke

脚本：

```text
workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh
```

修复点：

1. 进入实验前记录 `nvidia_uvm` built/loaded srcversion。
2. 自动打开 Phase-N region reclaim 所需参数：

```text
uvm_phase_n_block_quota_enable=1
uvm_phase_n_native_block_enable=1
```

3. 实验结束后恢复原始参数。
4. 保存 UVM 参数快照、用户态 ioctl debug log、`phase_f_route_log` 增量。

根因：

```text
reclaim-expert0 result=-86
```

对应 kernel `rmStatus=86`，即 native block reclaim gate 未打开时 `uvm_api_phase_n_reclaim_region()` 直接返回 not supported。打开上述两个参数后，`reclaim_region` 可以进入实际 block reclaim 路径。

### Phase U0 总脚本

新增脚本：

```text
workloads/vllm/run_unified_hbm_residency_phase0.sh
```

默认执行：

```bash
RUN_LOWTRACE=0 workloads/vllm/run_unified_hbm_residency_phase0.sh
```

默认覆盖：

```text
stage_n_3gb_rebalance
kv_dynamic_pool_phase2_prefetch_real
manual_migration_fault_control
```

如需复跑 LOWTRACE 四组 Qwen3 基线：

```bash
RUN_LOWTRACE=1 workloads/vllm/run_unified_hbm_residency_phase0.sh
```

LOWTRACE 模式会尝试打开并在结束后恢复 reserved-slot PTE remap 参数：

```text
uvm_moe_slot_pte_remap_enable=1
uvm_moe_slot_pte_remap_max_bytes=4194304
```

## 本次实验

命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=/tmp/vllm_unified_hbm_phase0_codex_20260712_122647 \
RUN_LOWTRACE=0 \
KV_DYNAMIC_TIMEOUT=900 \
workloads/vllm/run_unified_hbm_residency_phase0.sh
```

结果目录：

```text
/tmp/vllm_unified_hbm_phase0_codex_20260712_122647
```

汇总文件：

```text
/tmp/vllm_unified_hbm_phase0_codex_20260712_122647/phase0_summary.md
/tmp/vllm_unified_hbm_phase0_codex_20260712_122647/phase0_summary.json
```

环境确认：

```text
loaded nvidia_uvm srcversion = ECCE67EF07899605EBEA40C
built  nvidia_uvm srcversion = ECCE67EF07899605EBEA40C
GPU = NVIDIA GeForce RTX 5090, 32607 MiB
```

### 结果表

| 实验 | 状态 | 关键结果 |
| --- | ---: | --- |
| Phase-N 3GiB rebalance | PASS | `total_ok=1` |
| KV dynamic `2 -> 4 -> 1` | PASS | physical offload/prefetch records = `7 / 2` |
| manual migration fault control | PASS | normal/observe/fail-fast 均符合预期 |
| LOWTRACE four-way | skipped in this quick run | 复跑入口已接入 `RUN_LOWTRACE=1` |

### Phase-N 3GiB rebalance

关键输出：

```text
phase_n_3gb_final initial_budget_ok=1 final_budget_ok=1 expert_before_ok=1 kv_tail_before_ok=1 expert_reclaim_ok=1 kv_tail_after_ok=1 total_budget_ok=1 total_ok=1 kv_resident_after=1073741824 expert_resident_after=2147483648 total_resident_after=3221225472
```

方向正确：

```text
expert0 reclaim:
  resident_before = 256 blocks
  resident_after  = 0 blocks
  reclaimed_bytes = 536870912

KV tail promote:
  resident_after  = 256 blocks
  resident_bytes  = 536870912

pool after rebalance:
  kv_resident_after      = 1073741824
  expert_resident_after  = 2147483648
  total_resident_after   = 3221225472
```

单独验证脚本可在默认参数为 0 时自配置并恢复：

```bash
RUN_ROOT=/tmp/stage_n_autoconfig_verify_20260712_122813 \
workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh
```

验证后参数恢复为：

```text
uvm_phase_n_block_quota_enable=0
uvm_phase_n_native_block_enable=0
uvm_manual_migration_fault_mode=0
```

### KV dynamic target

关键检查全部通过：

```text
resize sequence:
  shrink 6 -> 2
  grow   2 -> 4
  shrink 4 -> 1

physical_offload_records  = 7
physical_prefetch_records = 2
pointer_map_mapped_blocks = 6
```

Ledger 估算 resident window 随 target 变化：

```text
target_2 = 1179648 bytes
target_4 = 2359296 bytes
target_1 = 589824 bytes
```

真实 residency 观察也符合方向：

```text
target_2_gpu_bytes = 1523712
target_4_gpu_bytes = 2457600
target_1_gpu_bytes = 1130496

gpu_increased_on_grow   = true
gpu_decreased_on_shrink = true
```

### manual migration fault control

脚本：

```text
workloads/vllm/run_manual_migration_fault_control_smoke.sh
```

结果：

```text
manual_migration_fault_control: PASS
normal:    mode=0 kernel succeeds and no manual fault log is emitted
observe:   mode=1 logs action=observe and kernel still succeeds
fail_fast: mode=2 logs action=block and expected CUDA failure is observed
```

## LOWTRACE 基线说明

本次 quick Phase U0 没有重跑 Qwen3 LOWTRACE 四组，原因是它们是重型服务实验，并且已有冻结结果：

```text
workloads/vllm/vllm/uvm_test/LOWTRACE_FOURWAY_EXPERIMENT.md
```

已有 LOWTRACE 结果摘要：

| 方案 | 状态 | avg/request | tokens/s |
| --- | ---: | ---: | ---: |
| baseline expert+KV managed, no offload | ok, 8/8 | 1.210s | 13.22 |
| dynamic real-load + offload + slot | ok, 8/8 | 2.208s | 7.25 |
| static tail5 0% | ok, 8/8 | 1.511s | 10.59 |
| static tail5 2% | ok, 8/8 | 2.296s | 6.97 |

复跑命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_LOWTRACE=1 \
LOWTRACE_TIMEOUT_SEC=1200 \
workloads/vllm/run_unified_hbm_residency_phase0.sh
```

## 结论

Phase U0 的关键阻塞项已经修通：

```text
Phase-N 3GiB rebalance smoke PASS
KV target 2 -> 4 -> 1 PASS
manual migration fault smoke PASS
built/loaded nvidia_uvm srcversion match
```

这满足进入后续 `U1/U2` 的前置条件：driver reclaim 原型路径可用，KV 动态 resident window 可复现，fault observer/fail-fast 可用于后续 controller 漏搬诊断。
