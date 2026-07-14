# vLLM Pre-Service Expert Release + KV Grow Attempt

Date: 2026-07-14

Run root:

```text
/tmp/vllm_pre_service_expert_release_kv_grow_20260714_174152
```

## Goal

Rerun the experiment with this intended sequence:

```text
baseline:
  no expert offload
  smaller allocator-visible KV target

experiment:
  warmup/load
  disable/guard automatic KV migration with KV-only manual fault fail-fast
  release static zero-hit tail experts
  use freed space to prefetch more KV blocks back to GPU
  increase allocator-visible KV target before serving requests
```

## Setup

Generated:

```text
/tmp/vllm_pre_service_expert_release_kv_grow_20260714_174152/kv_grow_8x64_fixture.json
/tmp/vllm_pre_service_expert_release_kv_grow_20260714_174152/tail5_offload_plan.json
```

Fixture:

```text
requests = 8
max_tokens per request = 64
max_num_seqs = 1
max_model_len = 2048
kv_cache_memory_bytes = 2147483648
```

UVM:

```text
loaded nvidia_uvm srcversion = 386A1FA2E18DDDFCE9EFE88
uvm_manual_migration_fault_pool_mask = 2
uvm_manual_migration_fault_mode = 2 at run start
```

## Code Fix Before Rerun

The previous bug where `"0"` was treated as a valid pre-service grow target was fixed:

```text
grow_target_blocks <= 0 is now ignored
```

Files touched:

```text
workloads/vllm/vllm/vllm/v1/engine/core.py
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

## What Actually Ran

Only the baseline case started:

```text
RUN_CASES = no_offload
INITIAL_TARGET_BLOCKS = 1236
KV_PRE_SERVICE_APPLY = 1
KV_DATA_OFFLOAD_MODE = prefetch_cpu
KV_POINTER_TAG_GRANULARITY = tensor
```

Important: this baseline physically moved the tail 128 KV blocks to CPU. That made it a stricter fail-fast test, but it was not the right performance baseline for the requested comparison. The next attempt should use metadata-only KV target limiting for baseline.

## Baseline Evidence Before Hang

Model loaded and KV initialized:

```text
load_end was recorded
GPU KV cache size = 21840 tokens
logical KV blocks = 1364
bytes_per_block = 1572864
```

Pre-service KV shrink completed:

```text
old_target_blocks = 1364
new_target_blocks = 1236
offloaded_blocks = 128
reason = pre_service_before_ready
```

Physical migration completed for the 128 tail blocks:

```text
kv_physical_trace records = 256
CPU migration success records = 128
last migrated block = 1364
block 1364 after migration:
  gpu_bytes = 0
  cpu_bytes = 1572864
  accepted_ranges = 48
```

Allocator selected only GPU-resident low block IDs:

```text
selected_block_ids = [1]
skipped_dynamic_pool = 128
cpu_candidates = 0
gpu_candidates = 1236
```

The run then reached:

```text
request_begin request=0
```

No `request_end` was recorded.

## Failure

The first 64-token request did not complete.

Observed after the hang:

```text
GPU utilization = 100%
GPU memory.used = 32109 MiB
uvm_manual_migration_fault_mode was changed back to 0, but the request did not recover
```

Killing the process chain changed the EngineCore process to zombie:

```text
PID = 5706
comm = VLLM::EngineCore
state = Z
```

However, the NVIDIA driver still reported GPU memory in use:

```text
GPU memory.used = 32109 MiB
nvidia_uvm refcnt = 8
rmmod nvidia_uvm failed: module in use
gpu-reset failed: GPU in use
```

`fuser` no longer showed ordinary `/dev/nvidia*` file descriptor owners after stopping exporters, so this appears to be a stale internal NVIDIA/UVM client state requiring reboot or a deeper driver reset.

## Interpretation

This attempt did not complete the requested comparison.

The failure does not invalidate the earlier successful 128-block grow-back smoke, because this run changed two conditions:

```text
previous successful smoke:
  16-token requests
  request path after grow had all target KV blocks GPU-resident

this failed baseline:
  64-token request
  128 tail KV blocks remained CPU-resident for request execution
  manual fault fail-fast was enabled
```

The most useful conclusion is:

```text
For performance baseline, do not physically offload KV blocks.
Use metadata-only dynamic target limiting so baseline measures "fewer allocatable blocks" without exposing CPU-resident writable KV to request execution.
```

The experiment case should still physically grow KV before request:

```text
expert release
-> pre-service KV target grow
-> prefetch grown KV blocks to GPU
-> expose grown blocks to allocator
-> then serve requests
```

## Corrected Next Attempt

After reboot / driver reset:

```text
baseline:
  RUN_CASES=no_offload
  INITIAL_TARGET_BLOCKS=1236
  KV_DATA_OFFLOAD_MODE=logical
  KV_PRE_SERVICE_GROW_TARGET_BLOCKS=0
  no expert offload

experiment:
  RUN_CASES=zero_hit_offload
  INITIAL_TARGET_BLOCKS=1236
  KV_DATA_OFFLOAD_MODE=prefetch_cpu
  KV_PREFETCH_BACK_MODE=prefetch
  KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364
  static tail5 expert offload
```

Pass criteria:

```text
baseline completes with dynamic_pool_target_blocks = 1236
experiment completes with final dynamic_pool_target_blocks = 1364
experiment expert release failed entries = 0
experiment physical GPU prefetch success records >= 128
manual_migration_fault records = 0 during served requests
```

This machine needs reboot or complete NVIDIA/UVM reset before the corrected attempt can run.
