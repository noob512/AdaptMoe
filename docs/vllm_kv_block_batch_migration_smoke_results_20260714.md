# KV Block Batch Migration Smoke Results - 2026-07-14

## Goal

在重新进入 vLLM 动态 KV cache 实验前，先用一个 synthetic KV cache smoke 验证：

- KV cache 使用可写数据路径，不使用 expert reserved slot、remote read 或 readonly copy/remap。
- 迁移单位按应用层 KV block 管理。
- 每个 logical KV block 由 48 个 layer ranges 组成。
- 可以批量把 block 从 GPU 搬到 CPU，再搬回 GPU。
- GPU 写入后的 KV 数据经过 CPU/GPU 往返后仍然正确。

## Implementation

新增脚本：

```text
workloads/vllm/check_kv_block_batch_migration_smoke.py
workloads/vllm/run_kv_block_batch_migration_smoke.sh
```

默认 synthetic 配置：

```text
num_blocks = 128
layers = 48
range_bytes = 32768
bytes_per_block = 48 * 32768 = 1572864 bytes ~= 1.5 MiB
migration_mode = block
```

`migration_mode=block` 表示每个 logical KV block 调一次 batch prefetch：

```text
block_id -> 48 layer ranges -> one batch_prefetch_ranges_to_cpu/device call
```

验收逻辑：

```text
1. 分配 128 个 synthetic KV blocks
2. GPU 上写入初始 payload
3. prefetch selected KV blocks 到 CPU
4. prefetch selected KV blocks 回 GPU
5. 从 GPU 校验初始 payload
6. 在 GPU 上覆盖写 selected KV blocks
7. 再次 prefetch 到 CPU
8. 再次 prefetch 回 GPU
9. 从 GPU 校验写入后的 payload
```

## Commands

48 MiB:

```bash
RUN_ROOT=/tmp/uvm_kv_block_batch_migration_smoke_codex_20260714_010417 \
workloads/vllm/run_kv_block_batch_migration_smoke.sh
```

96 MiB:

```bash
RUN_ROOT=/tmp/uvm_kv_block_batch_migration_96m_codex_20260714_010430 \
MIGRATE_COUNT=64 \
workloads/vllm/run_kv_block_batch_migration_smoke.sh
```

144 MiB:

```bash
RUN_ROOT=/tmp/uvm_kv_block_batch_migration_144m_codex_20260714_010430 \
MIGRATE_START=16 \
MIGRATE_COUNT=96 \
workloads/vllm/run_kv_block_batch_migration_smoke.sh
```

## Results

All runs passed.

| Migrated Blocks | Migrated Bytes | CPU Offload 1 | GPU Restore 1 | CPU Offload 2 After Write | GPU Restore 2 After Write |
|---:|---:|---:|---:|---:|---:|
| 32 | 48 MiB | 1536/1536 ranges, 1269.2 MiB/s | 1536/1536 ranges, 978.6 MiB/s | 1536/1536 ranges, 2021.6 MiB/s | 1536/1536 ranges, 1032.2 MiB/s |
| 64 | 96 MiB | 3072/3072 ranges, 1473.2 MiB/s | 3072/3072 ranges, 1375.7 MiB/s | 3072/3072 ranges, 2069.1 MiB/s | 3072/3072 ranges, 1250.2 MiB/s |
| 96 | 144 MiB | 4608/4608 ranges, 861.8 MiB/s | 4608/4608 ranges, 1205.8 MiB/s | 4608/4608 ranges, 2030.8 MiB/s | 4608/4608 ranges, 1040.4 MiB/s |

Functional checks:

```text
whole_gpu_prefetch_accepted = true
selected_cpu_1_accepted = true
selected_gpu_1_accepted = true
initial_payload_ok = true
gpu_write_ok = true
selected_cpu_2_accepted = true
selected_gpu_2_accepted = true
written_roundtrip_ok = true
```

## Residency Query Note

`driver_batch_query_ranges_residency_bytes()` returned:

```text
status = -6
queried_bytes = 0
```

for these synthetic KV subranges. The smoke now treats this query as diagnostic only. This does not invalidate the migration check, because:

- all CUDA/UVM batch prefetch calls accepted the expected ranges;
- GPU synchronization completed;
- GPU readback verified the original payload after CPU/GPU roundtrip;
- GPU writeback verified writable KV data survives CPU/GPU roundtrip.

## Conclusion

当前已经有一个简单 smoke 证明：KV cache 可以按 application-level block 做批量 CPU/GPU 搬运，并且适合 writable KV 数据路径。下一步可以把 vLLM dynamic KV grow/shrink 的物理迁移路径改成这种 block-level batch prefetch，避免之前 `1236 blocks * 48 ranges` 逐 range/逐 block 过重调度造成的问题。

