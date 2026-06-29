# Controlled Remap V1/V2 当前状态与后续路线

日期：2026-06-27

本文档总结当前 MoE slot remap 路径已经实现的能力、验证结论、仍然存在的边界，以及接下来进入完整 remap 前需要完成的工程项。

## 结论概览

当前实现已经从最早的 `copy_only` 推进到受控真实 PTE remap，并开始进入 v2/v3 的只读一致性收口：

- `copy_only`：driver 侧 sysmem backing + CPU copy，不改变用户 VA 映射。
- `gpu_copy_only`：driver 侧 GPU VIDMEM backing + GPU/CE copy，不改变用户 VA 映射。
- `slot_pte_remap`：smoke-only 真实 GPU PTE 写入，直接让目标 PTE 指向预填好的 VIDMEM slot，但没有进入完整 VA block 软件状态一致性。
- `slot_va_block_remap`：controlled remap v1。先走 VA block unmap + tracker wait，再安装 slot-backed 只读 PTE，并支持 multi-entry、失败回滚、跨 VA block 切片。
- `controlled_remap_v2`：在只读 slot-backed PTE 被原生 UVM fault/migrate/unmap/split/eviction/access-counter 等路径覆盖前，保守清理 slot PTE 并回退到普通 managed 语义；同时记录 clear reason counters。
- `controlled_remap_v3`：补齐 vLLM 重复启动/退出更敏感的 GPU VA space unregister 与 VA-space release 清理，在拆 GPU page table/VA-space 前清 slot-backed PTE，并通过 `pte_release_clear_count` 单独观测。

当前阶段已经验证：GPU VIDMEM backing、CE copy、只读 slot-backed PTE、VA block 切片、批量 remap、回滚清理和 reset/readback smoke 路径都可用。

但它仍然不是生产级完整 remap。当前实现仍是受控只读路径：依赖预先 `gpu_copy_only` 填好的 VIDMEM slot。v2/v3 已开始覆盖 fault、native migrate、native unmap/mmu-notifier、VA block split、access-counter、VIDMEM backing eviction、GPU VA-space unregister 和进程/VA-space release 的保守 clear/fallback；后续仍需要更长 vLLM ramp、低 VIDMEM budget 压力和并发生命周期验证。

本项目当前目标限定为只读专家权重类数据。当前不实现写 PTE、copyback、dirty tracking 或可写 slot backing 语义。

## 2026-06-29 vLLM 稳定性收口更新

本轮继续面向“只读 slot-backed PTE 在 vLLM 中稳定可用”做保守生命周期收口：

- `UvmDiscard` guard：
  - `uvm_api_discard()` 入口如果发现目标范围覆盖 active slot-backed PTE，会先清理 slot PTE metadata/PTE，再重新进入普通 discard 流程。
  - 该路径计入普通 unmap 类 clear 语义，目标是避免 discard/residency 更新把 slot-backed PTE 当普通 managed GPU residency 处理。
- vLLM controlled stress 前置检查：
  - `run_kv_fault_ratio.sh` 在 `slot_pte_remap` / `slot_va_block_remap` 模式下会拒绝在 D-state 用户线程存在时继续运行。
  - 同一路径会强制检查 `uvm_moe_slot_pte_remap_enable=1`、`uvm_moe_slot_pte_remap_max_bytes` 满足请求、`uvm_moe_slot_pool_vidmem_budget_bytes` 与本轮配置一致。
  - `run_stage_m_controlled_remap_vllm_stress.sh` 会在 preflight 前检查 controlled remap 相关模块参数，并在每个 vLLM case 的 UVM 重载参数中带上 `uvm_enable_builtin_tests=1`，方便 native split preflight 稳定执行。
- 构建验证：
  - `FORCE_REBUILD=0 NV_KERNEL_MODULES='nvidia nvidia-uvm' workloads/vllm/build_stage_n_modules.sh` 已通过。
  - `bash -n` / `py_compile` / stress dry-run 已通过。

当前节点仍有 `VLLM::EngineCore` 和 `nvidia-smi` 处于 D 状态，并且 vLLM 仍持有 `/dev/nvidia-uvm`、`/dev/nvidia0`、`/dev/nvidiactl`，因此本轮没有强制 `rmmod/insmod` 或运行真实 vLLM E2E。后续 runtime 验证必须在重启后的 clean node 上执行。

## 2026-06-28 read-only controlled remap 收口更新

本轮继续沿着只读 controlled remap 收口，完成并验证了三类保守 fallback：

- GPU fault path guard：replayable 和 non-replayable fault 遇到 active slot-backed PTE 时，先按 VA block 粒度清掉 slot PTE，再回到普通 managed fault 语义。
- native unmap/mmu-notifier guard：HMM/mmu-notifier invalidate 覆盖 slot-backed PTE 时，先清理 slot pool 中的 PTE metadata，并计入 `pte_unmap_clear_count`。
- vLLM slot prefill 源路径修正：`slot_pte_remap` / `slot_va_block_remap` 在 vLLM 路径中先把专家权重内容复制到 page-aligned host mmap staging，再用 `gpu_copy_only` 填 VIDMEM slot，避免在 ioctl 内直接对 managed expert VA 做 `copy_from_user`，从而避开 `uvm_va_space_cpu_fault` 卡死风险。

本轮 smoke 与 tiny E2E 均通过：

| 验证项 | 结果 | 关键观察 |
|---|---|---|
| GPU write fault fallback | PASS | `pte_fault_clear_count=2`，slot PTE 清理后普通 managed 写入和 GPU readback 正常 |
| native migrate clear/fallback | PASS | `pte_migrate_clear_count=2`，migrate 后 GPU readback 回到原始 managed 内容 |
| HMM/native unmap guard | PASS | `pte_unmap_clear_count=2`，未复现旧模块上的 NULL deref / D 状态卡死 |
| mode 6 multi-entry 串行回归 | PASS | 2 个 64KiB entry，`remap_count=2`，reset 后恢复原始内容 |
| mode 6 cross VA block 串行回归 | PASS | 128KiB entry 跨 2 个 VA block，`pte_block_unmap_count=2`，`pte_block_tracker_wait_count=2` |
| mode 6 rollback 串行回归 | PASS | `entry_statuses=[ROLLED_BACK, REMAP_FAILED]`，rollback 后 GPU readback 匹配原始内容 |
| tiny vLLM `slot_va_block_remap` E2E | PASS | 1 prompt / 2 output tokens；driver prefill 使用 `host_mmap`；slot checker 和 after-vLLM query 均通过 |

tiny vLLM E2E 结果目录：

- `log/controlled_remap_v2_tiny_vllm_e2e_after_fault_guard/cold_streaming_vs_fault_only_summary.md`
- `log/controlled_remap_v2_tiny_vllm_e2e_after_fault_guard/cold_streaming_hot1_stream536870912_kv7516192768_s8/cold_slot_pool_report.md`
- `log/controlled_remap_v2_tiny_vllm_e2e_after_fault_guard/moe_slot_pool_state_smoke_after_vllm.json`

after-vLLM query 关键计数：

- `active_slots=192`
- `backed_slots=192`
- `copy_count=1098`
- `copy_bytes=2590507008`
- `copy_failed_count=0`
- `pte_remap_count=1098`
- `pte_remap_bytes=2590507008`
- `pte_clear_count=1098`
- `pte_fault_clear_count=117`
- `pte_unmap_clear_count=164`
- `pte_failure_count=0`
- `pte_dangerous_op_reject_count=0`
- `slot_backing_bytes=905969664`
- `vidmem_evict_count=0`

注意：这些结果证明的是只读 slot-backed PTE 在受控 vLLM 路径中可安装、可清理、可被 fault/unmap fallback 接住。它仍然不是可写 remap，也不是完整生产级 VA block residency 状态。

## 2026-06-28 生命周期 guard 扩展

本轮继续补齐更深 UVM 生命周期 guard，仍然严格保持只读语义：

- VA block split guard：
  - managed `uvm_va_range_split_blocks()` 在实际 split populated block 前清理覆盖范围内的 slot-backed PTE。
  - HMM `hmm_split_block()` 在 split 前清理覆盖范围内的 slot-backed PTE。
  - 新增 per-slot/global `pte_split_clear_count`。
- smoke 脚本新增 `--clear-via-native-split`：
  - 通过 `UVM_TEST_VA_RANGE_SPLIT` 触发 managed VA range split。
  - 校验 `pte_mapped=0`、`pte_split_clear_count` 增加、GPU readback 回到原始 managed 内容。
  - 该 smoke 需要模块开启 `uvm_enable_builtin_tests=1`。
- vLLM stress 脚本新增 controlled smoke preflight：
  - 默认在 vLLM ramp 前跑 mode 6 multi-entry 和 native migrate clear smoke。
  - 如果 `uvm_enable_builtin_tests=1`，自动追加 native split clear smoke。
  - 如果当前 `uvm_moe_slot_pool_vidmem_budget_bytes` 足够低，自动追加 active VIDMEM eviction clear smoke。
  - 每轮 vLLM 后继续检查 `copy_failed_count=0`、`pte_failure_count=0`、`pte_dangerous_op_reject_count=0`。

本轮已完成代码接线与静态验证项，runtime vLLM stress 仍应在干净节点上执行，避免旧 oops/D 状态进程污染结果。

## 2026-06-28 controlled_remap_v3 收口更新

本轮继续面向“只读数据在 vLLM 中稳定可用”做生命周期收口，仍然不加入可写 PTE、dirty tracking 或 copyback：

- GPU VA space unregister guard：
  - 在 `remove_gpu_va_space()` 拆 GPU page table/VA ranges 前，先清理该 `va_space + gpu` 上所有 active slot-backed PTE。
  - 目标是覆盖 vLLM 进程重复启动/退出、CUDA context/VA-space 注销等路径，避免 slot pool metadata 指向即将被销毁的 GPU VA space/page table。
- VA-space release 计数独立化：
  - 进程退出/VA-space release 不再计入普通 native unmap 的 `pte_unmap_clear_count`，改为 `pte_release_clear_count`。
  - query 同时暴露 per-slot/global `pte_release_clear_count`，stress 脚本会在每轮汇总打印。
- 用户态 ABI 对齐：
  - `check_moe_slot_pool_state_smoke.py`
  - `check_moe_slot_remap_ioctl_smoke.py`
  - `vllm/device_allocator/uvm.py`
  - 三处 ctypes query struct 均加入 `pte_release_clear_count`，避免 ioctl 结构偏移错位。
- release smoke：
  - `check_moe_slot_remap_ioctl_smoke.py --clear-via-release` 会启动子进程安装只读 slot PTE 后直接退出。
  - 父进程随后查询全局 slot pool，确认 `pte_release_clear_count` 增加、`pte_mapped=0`、health counters 为 0。
  - `run_stage_m_controlled_remap_vllm_stress.sh` 默认把该 smoke 放入 vLLM ramp 前的 preflight。

该更新的预期验证点：

- vLLM 多轮启动/退出后无 D 状态进程。
- `pte_failure_count=0`、`copy_failed_count=0`、`pte_dangerous_op_reject_count=0`。
- 如果退出/注销路径清过 slot PTE，`pte_release_clear_count` 应增加。
- 模块在 clean node 上可卸载。

## 当前实现内容

### 1. Kernel ABI 与 ioctl

当前新增和扩展了以下 UVM ioctl 能力。

#### `UVM_MOE_SLOT_REMAP_BATCH`

用于提交 MoE slot remap batch。当前支持的 mode：

| mode | 名称 | 语义 |
|---:|---|---|
| 0 | `dry_run` | 只校验 batch，slot pool 不变 |
| 1 | `validate` | 记录 slot metadata，不分配 backing，不改 PTE |
| 2 | `remap` | 走 UVM 原生 migrate/map 语义，用于探路原生路径 |
| 3 | `copy_only` | 分配 driver-owned sysmem backing，并从用户 VA copy 内容 |
| 4 | `gpu_copy_only` | 分配 driver-owned VIDMEM backing，通过 GPU/CE copy 内容 |
| 5 | `slot_pte_remap` | smoke-only 直接写 GPU PTE 指向 VIDMEM slot |
| 6 | `slot_va_block_remap` | controlled v1：VA block unmap/tracker wait 后写 slot-backed PTE |

新增 entry status：

| status | 名称 | 语义 |
|---:|---|---|
| 8 | `UVM_MOE_SLOT_REMAP_ENTRY_ROLLED_BACK` | 同 batch 后续 entry 失败后，此 entry 已安装的 PTE 被回滚清理 |

相关 ABI 定义位置：

- `UVM_MOE_SLOT_REMAP_ENTRY_ROLLED_BACK`：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
- `UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP`：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`

#### `UVM_MOE_SLOT_POOL_QUERY`

用于查询 driver-side MoE slot pool 状态。当前 query 能暴露：

- slot owner：`layer_id`、`slot_id`、`owner_expert_id`
- slot state：`state`、`range_count`、`epoch`、`batch_id`
- backing：`backing_kind`、`slot_bytes`、`slot_physical_bytes`
- copy counters：`copy_count`、`copy_bytes`、`copy_failed_count`
- PTE counters：`pte_remap_count`、`pte_remap_bytes`、`pte_clear_count`
- VA block counters：`pte_block_unmap_count`、`pte_block_tracker_wait_count`
- rollback/failure counters：`pte_rollback_count`、`pte_rollback_bytes`、`pte_failure_count`
- lifecycle clear counters：`pte_fault_clear_count`、`pte_migrate_clear_count`、`pte_unmap_clear_count`、`pte_split_clear_count`、`pte_release_clear_count`、`pte_evict_clear_count`
- PTE state：`pte_mapped`、`pte_status`、`pte_mode`、`pte_va_block_tracked`

#### `UVM_MOE_SLOT_POOL_READBACK`

用于 debug 验证 slot backing 内容。当前支持：

- sysmem backing readback
- VIDMEM backing 通过 GPU to CPU CE copy readback
- 验证 `gpu_copy_only` 和 slot remap 后内容迁移不是只有 counters 正确

### 2. Slot pool 与 backing 管理

当前 driver 内维护一个 MoE slot pool，记录 slot ownership、range metadata、backing 指针和 counters。

已支持：

- 按 `(layer_id, slot_id)` 查找 slot。
- 同 slot owner 替换时更新 overwrite counter。
- batch 内检查同 slot 冲突：
  - 不同 expert 抢同 slot 视为冲突。
  - 同 slot target range overlap 视为冲突。
- reset 清空 slot metadata、释放 backing、清理 PTE。
- process/VA-space release 时清理该 VA space 上仍挂着的 slot PTE。
- module reload/reset smoke 下 slot pool 能回到空状态。

### 3. `copy_only`：sysmem backing + CPU copy

`copy_only` 已实现 driver-owned sysmem slot backing：

- 为 slot 分配 sysmem backing。
- 从用户 managed VA copy 内容到 backing。
- 记录 `copy_count`、`copy_bytes`、`backing_kind=SYSMEM`。
- 支持 readback 校验内容。
- 不修改用户 VA 映射语义。

该阶段验证了 slot pool 元数据、range 聚合、copy counters、readback、reset 生命周期。

### 4. `gpu_copy_only`：VIDMEM backing + GPU/CE copy

`gpu_copy_only` 已实现 driver-owned VIDMEM slot backing：

- 根据目标 GPU 分配 VIDMEM slot backing。
- 通过 DMA-capable sysmem staging 和 GPU CE copy 将用户 VA 内容迁移到 VIDMEM slot。
- query 暴露 `backing_kind=VIDMEM`、copy counters、slot backing bytes。
- readback 通过 GPU to CPU CE copy 验证 VIDMEM 内容。
- 仍然不修改用户 VA 映射语义。

这一阶段证明：

- driver 侧 VIDMEM slot backing 可用。
- GPU/CE copy-only 路径可用。
- 内容迁移和 counters 都可验证。
- VIDMEM 资源压力可以通过预算和 eviction counter 跟踪。

### 5. `slot_pte_remap`：早期受控 PTE smoke

`slot_pte_remap` 是第一版真实 PTE smoke：

- 依赖先用 `gpu_copy_only` 预填 VIDMEM slot。
- 将目标 GPU PTE 改为指向 slot backing。
- PTE 为只读受控路径。
- 支持 GPU readback 验证 remap 后读到的是 slot fill 内容。
- reset 后清理 slot PTE，GPU readback 恢复原始 managed 内容。

限制：

- 主要是单 64KiB smoke。
- 不经过完整 VA block 软件状态一致性。
- 不适合作为生产 remap 路径。

### 6. `slot_va_block_remap`：controlled remap v1

当前重点实现是 `slot_va_block_remap`，目标是把 smoke 推进到“可批量、可回滚、可跨 VA block”的受控真实 remap。

#### VA block aware install

安装 slot-backed PTE 前会先：

1. 找到目标 managed VA 对应的 VA block。
2. 对目标 GPU range 执行 VA block owned unmap。
3. 等待 block tracker。
4. 安装指向 driver-owned VIDMEM slot 的只读 PTE。
5. 记录 block unmap 和 tracker wait counters。

#### Cross-block slicing

当一个 entry 跨越多个 UVM VA block 时，当前实现会按 VA block 切片：

- 每个 slice 单独找到 VA block。
- 每个 slice 单独 unmap GPU range。
- 每个 slice 单独等待 tracker。
- 每个 slice 安装对应 slot offset 的 PTE。
- 任意 slice 失败时，已完成 slice 会走回滚清理。

这解决了早期单 64KiB、单 VA block smoke 的限制。

#### Multi-entry batch

`slot_va_block_remap` 当前支持一个 ioctl 内多个 entry：

- 每个 entry 可对应不同 slot。
- batch 级 counters 记录 `success_count`、`failure_count`、`accepted_bytes`、`remap_count`、`tlb_invalidate_count`。
- query 可看到每个 slot 的 PTE 状态和 counters。

#### Transactional rollback

当前实现了受控事务语义：

- 同一个 `slot_va_block_remap` ioctl 中，若某个 entry apply 失败：
  - 已安装成功的前序 slot-backed PTE 会按反向顺序 rollback。
  - 已 rollback 的 entry status 变为 `ROLLED_BACK`。
  - batch 返回时 `success_count`、`accepted_bytes`、`remap_count`、`tlb_invalidate_count` 会扣回。
  - query 中暴露 `pte_rollback_count` 和 `pte_rollback_bytes`。
- 如果跨 VA block slice 部分成功后失败，也会回滚已完成 slice。

当前 smoke 中通过故障注入验证：

- entry statuses 为 `[ROLLED_BACK, REMAP_FAILED]`。
- `success_count=0`。
- `accepted_bytes=0`。
- `remap_count=0`。
- `tlb_invalidate_count=0`。
- 用户 VA GPU readback 恢复原始内容。

#### 生命周期清理

当前已覆盖：

- explicit reset 清理 slot-backed PTE。
- VA space release 清理对应 slot-backed PTE。
- rollback 清理已安装 PTE。
- slot backing 释放前先清理 PTE。

### 7. vLLM Python 侧集成

`workloads/vllm/vllm/vllm/device_allocator/uvm.py` 已同步：

- mode 名称解析：
  - `slot_pte_remap`
  - `slot_va_block_remap`
- ioctl struct ABI。
- query struct ABI。
- rollback/failure counters 输出。
- slot PTE query 字段输出。

当前脚本侧 smoke 支持：

- `--mode slot_va_block_remap`
- `--count N`
- `--remap-bytes`
- `--cross-va-block`
- `--induce-rollback-failure`
- `--verify-gpu-read`
- `--clear-via-native-migrate`
- `--clear-via-native-unmap`
- `--clear-via-native-split`
- `--evict-active-via-gpu-copy`

### 8. `controlled_remap_v2`：read-only clear/fallback guards

v2 的目标不是做写路径，也不是完整生产 remap，而是让原生 UVM 生命周期路径覆盖只读 slot-backed PTE 时有受控行为。

当前实现：

1. `uvm_api_migrate()` 入口会先检查目标 VA range 是否覆盖 active slot-backed PTE。
2. replayable/non-replayable fault service 遇到覆盖 active slot-backed PTE 的 fault range 时，先清 slot PTE，再回到普通 managed fault。
3. native unmap、mmu-notifier invalidate、VA space release、managed VMA close 会先清 slot PTE。
4. VA block split 前会先清覆盖范围内的 slot PTE。
5. access-counter service 遇到 slot-backed PTE 覆盖范围时先 clear/fallback，避免按普通 resident page 处理。
6. VIDMEM backing eviction 遇到 active slot PTE 时，只在同一 locked VA space 中先 clear + wait tracker；无法安全 clear 的 active PTE 会拒绝/计数，避免 PTE 指向已释放 backing。
7. 对 `slot_va_block_remap`，清理路径走 VA block owned unmap + tracker wait。
8. 清理后保留 slot owner 和 VIDMEM backing metadata，但 `pte_mapped=0`、`pte_bytes=0`、`pte_mode=0`。
9. query 暴露 per-slot 和 global clear reason counters：
   - `pte_fault_clear_count`
   - `pte_migrate_clear_count`
   - `pte_unmap_clear_count`
   - `pte_split_clear_count`
   - `pte_evict_clear_count`
   - `pte_dangerous_op_reject_count`
10. 当前已实际接入并验证：
   - native migrate guard：`pte_migrate_clear_count`
   - VA-space/process teardown cleanup：`pte_unmap_clear_count`
   - GPU fault fallback：`pte_fault_clear_count`
   - native unmap/mmu-notifier guard：`pte_unmap_clear_count`

验证语义：

- slot PTE 安装后，GPU readback 读到 VIDMEM slot fill 内容。
- 对同一 managed range 执行原生 `UVM_MIGRATE`。
- migrate guard 先清掉 slot-backed PTE。
- 原生 migrate 完成后，GPU readback 回到原始 managed 内容。
- query 中不再有 active `pte_mapped` entry，且 `pte_migrate_clear_count` 增加。

仍需继续压力验证：

- split clear 的 runtime smoke 和 vLLM 路径覆盖。
- access-counter clear/fallback 在 access-counter enabled 配置下的压力覆盖。
- 低 VIDMEM budget 下 active PTE eviction clear 与 vLLM 多轮启动退出。
- 进程退出、reset、module unload 与 active slot PTE 交错。

## 已验证内容

验证环境中已执行：

```bash
FORCE_REBUILD=0 NV_KERNEL_MODULES='nvidia nvidia-uvm' workloads/vllm/build_stage_n_modules.sh
```

模块重载参数：

```bash
sudo rmmod nvidia_uvm
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  uvm_moe_slot_pte_remap_enable=1 \
  uvm_moe_slot_pte_remap_max_bytes=4194304 \
  uvm_moe_slot_pool_vidmem_budget_bytes=1073741824
```

若要运行 `--clear-via-native-split` smoke，需要额外开启 built-in test ioctl：

```bash
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  uvm_moe_slot_pte_remap_enable=1 \
  uvm_moe_slot_pte_remap_max_bytes=4194304 \
  uvm_moe_slot_pool_vidmem_budget_bytes=1073741824 \
  uvm_enable_builtin_tests=1
```

通过的 smoke：

| 验证项 | 命令摘要 | 结果 |
|---|---|---|
| slot pool + sysmem copy + gpu copy | `check_moe_slot_pool_state_smoke.py --include-gpu-copy-only` | PASS |
| mode 5 回归 | `--mode slot_pte_remap --count 1 --verify-gpu-read` | PASS |
| mode 6 multi-entry | `--mode slot_va_block_remap --count 2 --verify-gpu-read` | PASS |
| mode 6 cross VA block | `--mode slot_va_block_remap --remap-bytes 131072 --cross-va-block` | PASS |
| mode 6 rollback | `--mode slot_va_block_remap --count 2 --induce-rollback-failure` | PASS |
| mode 6 native migrate clear | `--mode slot_va_block_remap --clear-via-native-migrate` | PASS |
| mode 6 native split clear | `--mode slot_va_block_remap --clear-via-native-split` | 待干净节点 runtime 验证 |
| dmesg 检查 | `dmesg` grep UVM/NVRM/Xid/BUG/WARNING | 未发现相关告警 |

关键验证观察：

- multi-entry case：
  - `success_count=2`
  - `accepted_bytes=131072`
  - `pte_mode=6`
  - `pte_block_unmap_count=2`
  - `pte_block_tracker_wait_count=2`
- cross-block case：
  - 一个 128KiB entry 横跨两个 2MiB VA block。
  - query 显示 `pte_block_unmap_count=2`。
  - query 显示 `pte_block_tracker_wait_count=2`。
- rollback case：
  - `entry_statuses=[8, 7]`
  - `success_count=0`
  - `accepted_bytes=0`
  - `remap_count=0`
  - `tlb_invalidate_count=0`
  - `pte_rollback_count=1`
  - `pte_rollback_bytes=65536`
  - rollback 后 GPU readback 匹配原始内容。
- native migrate clear case：
  - slot PTE 安装后 GPU readback 匹配 slot fill。
  - 原生 `UVM_MIGRATE` 返回成功。
  - query 显示 `pte_mapped=0`。
  - query 显示 `pte_clear_count=1`。
  - query 显示 `pte_migrate_clear_count=1`。
  - migrate 后 GPU readback 匹配原始 managed 内容。

## 当前边界与风险

### 1. 仍然不是完整生产 remap

当前 `slot_va_block_remap` 已经触碰真实 GPU PTE，但仍是受控 smoke：

- PTE 只读。
- 依赖预先 `gpu_copy_only` 填好的 VIDMEM slot。
- slot-backed PTE 状态仍主要由独立 slot pool metadata 跟踪。
- 尚未把 slot-backed PTE 作为 VA block 的一等 residency 状态。
- v2 已对 fault、migrate、unmap、split、eviction、access-counter 等路径接入保守 clear/fallback 或拒绝策略，但还需要更强 runtime 压力证明。

### 2. 生命周期路径仍需压力验证

当前已经给以下路径接入保守 clear/fallback 或拒绝策略：

- replayable fault
- non-replayable fault
- migrate
- unmap
- split
- eviction
- access counter
- mmu notifier invalidation
- VA range teardown

后续重点不是改成可写或一等 residency，而是验证这些路径在 vLLM 长时间运行、低 VIDMEM budget、进程退出和模块卸载时不会把 slot-backed PTE 当成普通 GPU resident page，也不会留下 PTE 指向已释放 backing。

### 3. Write path 明确不在当前目标内

当前目标是只读热专家权重页，不建议也不需要启用写 PTE：

- 没有 copyback 语义。
- 没有 dirty tracking。
- 没有 owner consistency。
- 没有和 UVM migration/fault write permission path 对齐。

因此当前只适合只读热专家页 smoke。任何需要写入的数据都应该继续走普通 UVM managed 路径。

### 4. Rollback 是 batch 内受控事务

当前 rollback 能覆盖：

- 同一 ioctl 内前序 entry 已安装，后续 entry apply 失败。
- 单 entry 跨 VA block 部分 slice 已安装，后续 slice 失败。

但还没有覆盖完整生产环境中的所有异步情况，例如：

- 其他 UVM path 并发修改同一 VA block。
- 进程退出和 remap apply 并发。
- GPU fault service 与 slot PTE install 并发。
- vLLM 长时间运行下 slot reuse 和 eviction 交错。

### 5. Counters 语义仍偏 debug

当前 counters 主要用于 smoke 和调试：

- batch 返回 counters 表示本次 ioctl 最终结果。
- query 中 `pte_remap_count` 是历史安装次数。
- rollback 后 query 会同时看到历史 `pte_remap_count` 和 `pte_rollback_count`。

后续如果要做产品化指标，需要区分：

- historical installs
- currently active slot PTEs
- effective successful committed remaps
- rollback/clear/failure reasons

## 接下来要实现的内容

建议下一阶段目标是 **controlled_remap_v2 stress：只读生命周期压力验证 + vLLM ramp**。

### 阶段 A：fault path 一致性收口

目标：所有 UVM 后续路径遇到只读 slot-backed PTE 时都有明确、可验证的行为。

建议策略先保守：

1. 识别 slot-backed PTE。
2. 清掉 slot-backed PTE。
3. 恢复普通 managed fault/migrate 路径。
4. 对危险路径先拒绝或强制 clear，而不是让 UVM 误以为它是普通 resident page。

已完成：

- migrate：
  - migrate range 覆盖 slot PTE 时先 clear slot mapping。
  - 避免迁移逻辑读取错误 residency state。
  - smoke 验证 migrate 后 readback 回到普通 managed 内容。
- fault：
  - replayable/non-replayable fault 遇到 slot PTE 时先 clear。
  - smoke 验证 GPU write fault 后 readback 走普通 managed 内容。
- process/VA-space teardown：
  - 当前已有 VA space release 清理。
  - cleanup reason 会计入 `pte_unmap_clear_count`。
- unmap：
  - VA block unmap 覆盖 slot PTE 时清理 slot pool metadata。
  - 与现有 `pte_clear_count` 对齐。
- split：
  - VA block split 前清理或迁移 slot PTE metadata。
- eviction：
  - VIDMEM backing 被回收前必须 clear 对应 slot-backed PTE。
  - 防止 PTE 指向已释放 backing。
- mmu notifier：
  - CPU VA invalidation 时清理 slot PTE。
- access-counter：
  - access-counter service 遇到 slot PTE 覆盖范围时 clear/fallback。

仍需验证：

- fault buffer replay 和 TLB invalidate 顺序在多轮 vLLM 下稳定。
- split/access-counter/eviction guard 的 runtime smoke 与低 budget 压力。
- process teardown、reset、module unload 与 active slot PTE 并发。

已新增 counters：

- `pte_fault_clear_count`
- `pte_migrate_clear_count`
- `pte_unmap_clear_count`
- `pte_split_clear_count`
- `pte_evict_clear_count`
- `pte_dangerous_op_reject_count`

### 阶段 B：vLLM controlled stress ramp

目标：在实际 vLLM 路径中启用 `slot_va_block_remap`，但仍保持只读、可回滚，并逐步增加压力。

建议配置：

- 小 prompt 数。
- 小 output len。
- 小 hotK。
- 只对热专家只读权重页尝试 slot VA block remap。
- 使用 `COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap` 或等价环境变量。

需要验证：

- vLLM 启动稳定。
- routing/slot metadata 能触发 mode 6。
- query 能看到 VIDMEM slot。
- query 能看到 `pte_mode=6`。
- query 能看到 VA block counters。
- GPU readback 内容正确。
- reset 后恢复普通 managed 内容。
- 进程退出后 slot PTE 清理。
- 模块卸载不报错。
- dmesg 无 UVM/NVRM/Xid/BUG/WARNING。

建议先跑 tiny，然后再扩：

1. 1 prompt、短 output。
2. 2 prompts、短 output。
3. 小批量并发。
4. 增加 hotK 和 remap bytes。
5. 低 VIDMEM budget，多轮启动退出。

### 阶段 C：更强 rollback/stress

当前 rollback smoke 已覆盖基本事务，但还需要更强组合：

- 多 entry，其中多个 entry 横跨 VA block。
- 第一个 entry 跨 block 第二个 slice 失败。
- 中间 entry 失败，前后 entry 状态正确。
- rollback 过程中某个 clear 失败的错误注入。
- reset 与已安装 slot PTE 混合。
- process exit 与 slot PTE active 混合。
- 模块卸载前 active slot PTE 清理。
- VIDMEM budget eviction 与 active PTE 的关系。

### 阶段 D：生产 remap 设计

等 fault path 和 vLLM controlled E2E 稳定后，再进入真正生产 remap 设计。

生产 remap 至少需要：

- VA block 内明确记录 slot-backed residency。
- slot backing 生命周期与 VA block 生命周期绑定。
- PTE 权限、access counter、fault replay、TLB invalidate 完整同步。
- 写路径语义：
  - read-only hot expert 页可以长期保持只读。
  - 如果需要 write，需要 dirty tracking/copyback。
- slot reuse 安全：
  - active PTE 引用的 slot 不可被复用或释放。
  - eviction 必须先 clear 所有关联 PTE 并 wait tracker。
- scheduler/policy：
  - 哪些专家进入 slot。
  - 什么时候回收。
  - 如何根据 miss、fault、copy cost 调整 hotset。
- telemetry：
  - active PTE bytes。
  - effective committed remaps。
  - rollback reasons。
  - clear reasons。
  - fault-path fallback counts。

## 推荐下一步执行顺序

建议按以下顺序推进：

1. 在干净节点运行 controlled smoke preflight：multi-entry、native migrate、native split、active eviction。
2. 跑 `run_stage_m_controlled_remap_vllm_stress.sh`，从 p2/p8/s8/s16 和低 VIDMEM budget 开始。
3. 每轮检查无 D 状态进程、`copy_failed_count=0`、`pte_failure_count=0`、`pte_dangerous_op_reject_count=0`。
4. 增加 repeats、hotK、`MAX_NUM_SEQS`、低 VIDMEM budget 压力。
5. 补 access-counter enabled 配置下的专项 smoke。
6. 扩展 rollback/stress 覆盖。
7. 再设计生产级只读 slot-backed residency。

当前不进入写 PTE。下一步最合理的工程目标是：**继续让 UVM 相关路径知道只读 slot-backed PTE 的存在，并能保守清理/回退到普通 managed fault 语义**。

## 复现实验命令

构建：

```bash
FORCE_REBUILD=0 NV_KERNEL_MODULES='nvidia nvidia-uvm' workloads/vllm/build_stage_n_modules.sh
```

加载模块：

```bash
sudo rmmod nvidia_uvm
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  uvm_moe_slot_pte_remap_enable=1 \
  uvm_moe_slot_pte_remap_max_bytes=4194304 \
  uvm_moe_slot_pool_vidmem_budget_bytes=1073741824
```

基础 slot pool 和 gpu copy：

```bash
python3 workloads/vllm/check_moe_slot_pool_state_smoke.py \
  --include-gpu-copy-only \
  --out-json /tmp/moe_slot_pool_state_smoke.json
```

mode 5 回归：

```bash
python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py \
  --mode slot_pte_remap \
  --count 1 \
  --validate-managed \
  --verify-gpu-read \
  --out-json /tmp/moe_slot_pte_remap_smoke.json
```

mode 6 multi-entry：

```bash
python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py \
  --mode slot_va_block_remap \
  --count 2 \
  --validate-managed \
  --verify-gpu-read \
  --out-json /tmp/moe_slot_va_block_multi_smoke.json
```

mode 6 cross VA block：

```bash
python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py \
  --mode slot_va_block_remap \
  --count 1 \
  --remap-bytes 131072 \
  --cross-va-block \
  --validate-managed \
  --verify-gpu-read \
  --out-json /tmp/moe_slot_va_block_cross_smoke.json
```

mode 6 rollback：

```bash
python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py \
  --mode slot_va_block_remap \
  --count 2 \
  --validate-managed \
  --verify-gpu-read \
  --induce-rollback-failure \
  --out-json /tmp/moe_slot_va_block_rollback_smoke.json
```

mode 6 native migrate clear/fallback：

```bash
python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py \
  --mode slot_va_block_remap \
  --count 1 \
  --validate-managed \
  --verify-gpu-read \
  --clear-via-native-migrate \
  --out-json /tmp/moe_slot_va_block_native_migrate_clear_smoke.json
```

dmesg 检查：

```bash
sudo dmesg --since '15 minutes ago' | rg -i 'MoE slot|UVM_ERR|NVRM|Xid|failed|BUG|WARNING|Oops|panic'
```
