# vLLM pre-service KV grow all-managed request fault observe

Date: 2026-07-14

## Goal

Rerun the pre-service expert release + KV grow experiment from:

```text
docs/vllm_pre_service_kv_grow_replicate_long_context_detailed_20260714.md
```

This run changes the request-time fault guard from KV-only to all managed
ranges. Since this experiment config only intends expert weights and KV cache to
use UVM managed allocation, an ideal run should produce no request-time GPU
managed page faults for either data class.

## Key Difference From Previous Run

Previous safety run:

```text
uvm_manual_migration_fault_mode = 2
uvm_manual_migration_fault_pool_mask = 2
```

That was KV-only fail-fast. `pool_mask=2` means bit `1 << UVM_PHASE_N_POOL_KV`,
and KV pool id is `1`.

This run:

```text
uvm_manual_migration_fault_mode = 1
uvm_manual_migration_fault_pool_mask = 0
uvm_manual_migration_fault_log_enable = 1
```

`mode=1` means observe/log and continue. `pool_mask=0` means all managed ranges,
including tagged and untagged managed ranges. This was chosen to capture all
request-time faults instead of aborting at the first one.

Relevant pool ids:

```text
UVM_PHASE_N_POOL_KV     = 1
UVM_PHASE_N_POOL_WEIGHT = 2
```

If a fault had occurred, the kernel log line would include:

```text
manual_migration_fault ... fault_addr=... access=... pool=... role=...
tag_valid=... range_base=... range_len=...
```

## Run Roots

```text
root:
  /tmp/vllm_pre_service_kv_grow_all_managed_fault_observe_20260714_194558

short mechanism:
  /tmp/vllm_pre_service_kv_grow_all_managed_fault_observe_20260714_194558/short

long mechanism:
  /tmp/vllm_pre_service_kv_grow_all_managed_fault_observe_20260714_194558/long
```

Machine state:

```text
loaded nvidia_uvm srcversion = 386A1FA2E18DDDFCE9EFE88
GPU memory.used before runs = 0 MiB
GPU memory.used after runs = 0 MiB
final uvm_manual_migration_fault_mode = 0
final uvm_manual_migration_fault_pool_mask = 0
```

## Experiment Settings

Common settings:

```text
VLLM_USE_UVM=0
VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_KV_CACHE_MANAGED=1

KV_CACHE_MEMORY_BYTES=2147483648
INITIAL_TARGET_BLOCKS=1236
GROW_TARGET_BLOCKS=1364
KV_PRE_SERVICE_APPLY=1
KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364
KV_DATA_OFFLOAD_MODE=prefetch_cpu
KV_PREFETCH_BACK_MODE=prefetch
KV_FORCE_DRIVER_TAG=1
KV_DRIVER_TAG_ENABLE=0
KV_POINTER_TAG_GRANULARITY=tensor
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1

VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=1
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=0
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1
```

Short fixture:

```text
fixture = /tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json
plan    = /tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json
requests = 8
max_model_len = 2048
```

Long fixture:

```text
fixture = /tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json
plan    = /tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json
requests = 1
max_model_len = 21840
```

## Results

| case | completed | tokens/s | expert release failed | KV CPU success | KV GPU success | all-managed manual fault records |
|---|---:|---:|---:|---:|---:|---:|
| short | 8/8 | 18.10 | 0 | 128/128 | 128/128 | 0 |
| long | 1/1 | 4.65 | 0 | 128/128 | 128/128 | 0 |

Short mechanism:

```text
expert released bytes = 1102577664 ~= 1051.5 MiB
cold_plan_gpu_bytes: 1102577664 -> 0
initial_target_blocks = 1236
grow_target_blocks = 1364
logical_prefetched_blocks = 128
KV physical CPU success = 128/128
KV physical GPU success = 128/128
prefetch fallback = 0
managed probe failures = 0
manual_migration_fault records = 0
```

Short allocator evidence:

```text
selected_total = 40
max_selected_block_id = 40
cpu_candidates_max = 0
skipped_dynamic_pool_max = 0
allocation_no_free_blocks = 0
```

Long mechanism:

```text
expert released bytes = 1093140480 ~= 1042.5 MiB
cold_plan_gpu_bytes: 1093140480 -> 0
initial_target_blocks = 1236
grow_target_blocks = 1364
logical_prefetched_blocks = 128
KV physical CPU success = 128/128
KV physical GPU success = 128/128
prefetch fallback = 0
managed probe failures = 0
manual_migration_fault records = 0
```

Long allocator evidence:

```text
selected_total = 1283
max_selected_block_id = 1283
selected blocks above 1236 = 47
selected above-target range = 1237..1283
cpu_candidates_max = 0
skipped_dynamic_pool_max = 0
allocation_no_free_blocks = 0
```

## Fault Interpretation

The guard was applied before formal request serving. `progress.jsonl` confirms:

```text
pre_request_manual_fault_guard_applied
observed:
  uvm_manual_migration_fault_mode = 1
  uvm_manual_migration_fault_pool_mask = 0
  uvm_manual_migration_fault_log_enable = 1
request_begin
...
request_end
```

Observed dmesg deltas:

```text
short:
  manual_migration_fault records = 0
  Xid/MMU/fatal/faultish records = 0

long:
  manual_migration_fault records = 0
  Xid/MMU/fatal/faultish records = 0
```

Since `pool_mask=0` applies to all managed ranges, this covers both expected
managed classes in this experiment:

```text
expert weights
KV cache
```

No request-time GPU replayable managed fault was observed for either class.
Therefore, for these fixed fixtures and plans, the pre-service migration was
timely enough: the request path did not need automatic UVM migration for managed
expert or KV data.

## Reproduction Command

Short:

```bash
RUN_ROOT=/tmp/vllm_pre_service_kv_grow_all_managed_fault_observe_20260714_194558/short \
FIXTURE_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json \
PLAN_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=8 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=2048 \
KV_CACHE_MEMORY_BYTES=2147483648 \
INITIAL_TARGET_BLOCKS=1236 \
GROW_TARGET_BLOCKS=1364 \
KV_PRE_SERVICE_APPLY=1 \
KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_FORCE_DRIVER_TAG=1 \
KV_DRIVER_TAG_ENABLE=0 \
KV_POINTER_TAG_GRANULARITY=tensor \
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=0 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

Long:

```bash
RUN_ROOT=/tmp/vllm_pre_service_kv_grow_all_managed_fault_observe_20260714_194558/long \
FIXTURE_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json \
PLAN_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=1 \
MAX_MODEL_LEN=21840 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=2048 \
KV_CACHE_MEMORY_BYTES=2147483648 \
INITIAL_TARGET_BLOCKS=1236 \
GROW_TARGET_BLOCKS=1364 \
KV_PRE_SERVICE_APPLY=1 \
KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_FORCE_DRIVER_TAG=1 \
KV_DRIVER_TAG_ENABLE=0 \
KV_POINTER_TAG_GRANULARITY=tensor \
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=0 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

## Conclusion

This all-managed fault observe rerun succeeded.

It strengthens the previous KV-only result:

```text
previous:
  no request-time KV managed GPU fault

this run:
  no request-time managed GPU fault for any managed range
```

Given this experiment config, that means no observed request-time GPU page fault
for either UVM-managed data class:

```text
expert weights
KV cache
```

Remaining boundary: this proves the fixed zero-hit expert plan and the fixed
long prompt fixture. It does not prove arbitrary future routing will avoid
faults after different experts are offloaded.
