# KV Block Batch Large Sweep Results - 2026-07-14

## Goal

验证之前 vLLM dynamic KV 实验卡死是否由 `~1.8GiB` KV cache 物理迁移本身导致。

本实验不启动 vLLM，只使用 synthetic writable KV cache：

```text
logical KV blocks = 1364
layers = 48
range_bytes = 32768
bytes_per_block = 48 * 32768 = 1572864 bytes ~= 1.5 MiB
migrate_start = 128
```

也就是模拟：

```text
保留前 128 个 KV blocks
把后面的 KV blocks 按 block 粒度批量迁到 CPU
再按 block 粒度批量迁回 GPU
```

每个 KV block 一次 batch prefetch，内部包含 48 个 layer ranges。

## Implementation Changes

脚本：

```text
workloads/vllm/check_kv_block_batch_migration_smoke.py
workloads/vllm/run_kv_block_batch_migration_smoke.sh
```

本轮为大规模 sweep 增加了：

```text
PROGRESS_EVERY_BLOCKS
TIMEOUT_SEC
```

并修复了一个 synthetic payload bug：

```text
torch.arange(..., dtype=int16) -> dtype=int32
```

原因是当 block id 超过约 1057 时，`block_id * 31` 会在 int16 下溢出，导致 verifier 的 Python int expected 值和 GPU 初始 payload 不一致。该问题只影响 synthetic 初始数据校验，不影响 batch migration 本身；GPU 写入后的 roundtrip 在修复前也已经通过。

## Commands

512 / 768 blocks：

```bash
ROOT_BASE=/tmp/uvm_kv_block_batch_large_sweep_codex_20260714_113326
RUN_ROOT=$ROOT_BASE/count_512 NUM_BLOCKS=1364 MIGRATE_START=128 MIGRATE_COUNT=512 \
  PROGRESS_EVERY_BLOCKS=128 TIMEOUT_SEC=900 \
  workloads/vllm/run_kv_block_batch_migration_smoke.sh

RUN_ROOT=$ROOT_BASE/count_768 NUM_BLOCKS=1364 MIGRATE_START=128 MIGRATE_COUNT=768 \
  PROGRESS_EVERY_BLOCKS=128 TIMEOUT_SEC=900 \
  workloads/vllm/run_kv_block_batch_migration_smoke.sh
```

1024 / 1236 blocks after payload fix：

```bash
ROOT_BASE=/tmp/uvm_kv_block_batch_large_sweep_fixed_codex_20260714_113540
RUN_ROOT=$ROOT_BASE/count_1024 NUM_BLOCKS=1364 MIGRATE_START=128 MIGRATE_COUNT=1024 \
  PROGRESS_EVERY_BLOCKS=128 TIMEOUT_SEC=900 \
  workloads/vllm/run_kv_block_batch_migration_smoke.sh

RUN_ROOT=$ROOT_BASE/count_1236 NUM_BLOCKS=1364 MIGRATE_START=128 MIGRATE_COUNT=1236 \
  PROGRESS_EVERY_BLOCKS=128 TIMEOUT_SEC=900 \
  workloads/vllm/run_kv_block_batch_migration_smoke.sh
```

## Results

All functional checks passed. No timeout, no process hang, no Xid/MMU fault observed.

| Migrated Blocks | Migrated Bytes | Ranges | CPU Offload 1 | GPU Restore 1 | CPU Offload 2 After Write | GPU Restore 2 After Write |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 768 MiB | 24576 | 0.580s, 1324.7 MiB/s | 1.074s, 715.2 MiB/s | 0.448s, 1715.2 MiB/s | 1.080s, 710.8 MiB/s |
| 768 | 1152 MiB | 36864 | 0.782s, 1472.7 MiB/s | 1.557s, 740.0 MiB/s | 0.550s, 2093.5 MiB/s | 1.588s, 725.3 MiB/s |
| 1024 | 1536 MiB | 49152 | 1.055s, 1455.6 MiB/s | 2.585s, 594.3 MiB/s | 0.745s, 2062.5 MiB/s | 2.015s, 762.3 MiB/s |
| 1236 | 1854 MiB | 59328 | 1.277s, 1452.2 MiB/s | 2.726s, 680.2 MiB/s | 0.915s, 2026.8 MiB/s | 2.284s, 811.9 MiB/s |

For every run:

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

The full `1236` run accepted all expected ranges:

```text
selected_prefetch_cpu_1: 59328 / 59328
selected_prefetch_gpu_1: 59328 / 59328
selected_prefetch_cpu_2_after_write: 59328 / 59328
selected_prefetch_gpu_2_after_write: 59328 / 59328
```

The progress log for `1236` reached:

```text
blocks_done = 1236
blocks_total = 1236
last_block_id = 1363
accepted_ranges = 59328
```

Final GPU state after the sweep:

```text
memory.used = 0 MiB
```

## Interpretation

This experiment shows that, outside vLLM, moving the same scale as the previous initial shrink:

```text
1236 blocks * 1.5 MiB ~= 1.81 GiB
```

is not inherently enough to hang the driver when submitted as block-level batch prefetch.

Therefore the previous vLLM strong-compare hang is more likely caused by the old integration path:

```text
1236 blocks * 48 ranges/block ~= 59k per-range operations
```

combined with vLLM startup/request-0 synchronization, high HBM pressure after loading all experts, and dynamic pool shrink being applied inside the request path before `kv_dynamic_pool_resize` could be logged.

The next vLLM experiment should use the block-level batch path and avoid a huge initial physical shrink in the request path. Better options are:

```text
1. initial shrink is metadata-only, physical prefetch only for grow;
2. perform initial physical shrink before request admission, with progress logs;
3. split the shrink into bounded chunks with explicit yield/sync/report points.
```

