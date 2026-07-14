# vLLM pre-service expert release + KV grow replication and long-context check

Date: 2026-07-14

Detailed Chinese reproduction doc:

```text
docs/vllm_pre_service_kv_grow_replicate_long_context_detailed_20260714.md
```

## Goal

Replicate the pre-service mechanism:

```text
warmup / load
-> release fixed zero-hit expert plan
-> shrink KV visible target to 1236 blocks
-> pre-service grow target to 1364 blocks
-> physically prefetch the added 128 KV blocks back to GPU
-> enable KV-only manual fault guard before request serving
-> serve requests
```

Additional checks:

1. After formal request serving begins, no GPU-side KV page fault should occur. If a fault occurs, record it and explain the address/cause.
2. Use a real long-prompt fixture to prove that expanded KV blocks can be allocated by vLLM, not only prefetched.

Loaded UVM module:

```text
srcversion = 386A1FA2E18DDDFCE9EFE88
```

Final machine state after all runs:

```text
GPU memory.used = 0 MiB
uvm_manual_migration_fault_mode = 0
uvm_manual_migration_fault_pool_mask = 0
no residual VLLM::EngineCore process
```

## Run Roots

```text
short clean baseline:
  /tmp/vllm_pre_service_kv_grow_replicate_short_clean_baseline_20260714_191053

short mechanism replication:
  /tmp/vllm_pre_service_kv_grow_replicate_short_20260714_185952

long clean baseline:
  /tmp/vllm_long_context_clean_baseline_20260714_191321

long mechanism:
  /tmp/vllm_long_context_kv_grow_20260714/mechanism_190548
```

## Short Fixture Result

The clean baseline used the same 8-request fixture, but did not enable expert release, KV dynamic pool, KV prefetch, runtime tracing, or manual fault guard. It only used UVM managed allocations for expert weights and KV cache.

| case | completed | output tokens | generate seconds | tokens/s | faultish dmesg/manual records |
|---|---:|---:|---:|---:|---:|
| clean baseline | 8/8 | 512 | 24.665 | 20.76 | 0 |
| zero-hit expert release + pre-service KV grow | 8/8 | 512 | 29.202 | 17.53 | 0 |

Short mechanism details:

```text
expert plan records = 240
expert released bytes = 1102577664 ~= 1051.5 MiB
expert release failed entries = 0
cold_plan_gpu_bytes: 1102577664 -> 0

initial_target_blocks = 1236
grow_target_blocks = 1364
logical_prefetched_blocks = 128
KV physical CPU success = 128/128
KV physical GPU success = 128/128
prefetch fallback = 0
managed probe failures = 0
allocation_no_free_blocks = 0
```

The short fixture did not actually need the added tail blocks:

```text
max selected KV block id = 40
selected blocks > 1236 = 0
```

So this run verifies the mechanism and performance replication, but not long-context consumption.

## Long-Context Fixture

Long fixture:

```text
/tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json

actual_prompt_tokens_no_special = 20512
max_tokens = 16
tokens_per_block = 16
estimated required KV blocks ~= 1282
initial_target_blocks = 1236
grow_target_blocks = 1364
```

An initial `max_model_len=22000` pilot failed at vLLM KV sizing:

```text
2.01 GiB KV cache is needed, available KV cache memory is 2.00 GiB
estimated maximum model length = 21840
```

This was not a UVM/offload failure. The long experiment was rerun with:

```text
max_model_len = 21840
kv_cache_memory_bytes = 2147483648
```

The long zero-hit expert plan was generated from the long prompt's own routing trace:

```text
/tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json

routing_records = 1392
layers = 48
cold experts per layer = 5
cold_require_zero_hit = true
offload_plan_records = 240
roles = moe_gate_up, moe_down
```

## Long-Context Result

Important naming note: `clean long baseline` means an all-resident,
no-extra-mechanism vLLM run with the full configured 2 GiB KV budget available
from startup. It is not the constrained "1236-block target without expert
offload" case. Therefore its success only shows that this particular long
fixture fits when vLLM is allowed to use all 1364 KV blocks from the beginning.
It does not prove that the no-offload system can survive a tighter KV physical
window or a longer context that exceeds the all-resident budget.

| case | completed | output tokens | generate seconds | tokens/s | faultish dmesg/manual records |
|---|---:|---:|---:|---:|---:|
| clean long baseline | 1/1 | 16 | 4.757 | 3.36 | 0 |
| long mechanism | 1/1 | 16 | 3.466 | 4.62 | 0 |

The long baseline is a clean no-mechanism run with the full KV cache visible.
The long mechanism run is not meant as a strict speed comparison because it uses
a different control path and pre-service maintenance, but it proves the expanded
KV target can serve a real long prompt after the target grows from 1236 to 1364.

Long mechanism details:

```text
expert plan records = 240
expert released bytes = 1093140480 ~= 1042.5 MiB
expert release failed entries = 0
cold_plan_gpu_bytes: 1093140480 -> 0

initial_target_blocks = 1236
grow_target_blocks = 1364
logical_prefetched_blocks = 128
KV physical CPU success = 128/128
KV physical GPU success = 128/128
prefetch fallback = 0
managed probe failures = 0
allocation_no_free_blocks = 0
```

Allocator evidence that the long request used grown blocks:

```text
selected KV blocks total = 1283
max selected KV block id = 1283
selected blocks above initial target 1236 = 47
selected above-target range = 1237..1283
cpu_candidates_max = 0
skipped_dynamic_pool_max = 0
dynamic_pool_target_blocks during allocation = 1364
```

This is the important difference from the short fixture: the long prompt crossed the initial 1236-block target and allocated from the newly grown range.

## Fault Guard Result

For the mechanism runs, the request script applied the KV-only guard before request serving:

```text
uvm_manual_migration_fault_mode = 2
uvm_manual_migration_fault_pool_mask = 2
uvm_manual_migration_fault_log_enable = 1
```

Observed result:

```text
short mechanism:
  new Xid/MMU/fatal/manual_migration fault records = 0

long mechanism:
  new Xid/MMU/fatal/manual_migration fault records = 0
```

No GPU-side KV fault was captured after formal request serving began. Therefore there is no fault address/cause to report for these successful runs.

## Reproduction Commands

Short clean baseline:

```bash
RUN_ROOT=/tmp/vllm_pre_service_kv_grow_replicate_short_clean_baseline_20260714_191053
mkdir -p "$RUN_ROOT"

cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
VLLM_USE_UVM=0 \
VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0 \
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
VLLM_UVM_KV_CACHE_MANAGED=1 \
VLLM_DISABLE_OPENAI_METRICS=1 \
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
VLLM_UVM_KV_RUNTIME_ENABLE=0 \
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
uv run python uvm_test/vllm_generate_sequence.py \
  --model /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782 \
  --request-fixture-json /tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 2147483648 \
  --gpu-memory-utilization 0.90 \
  --output-json "$RUN_ROOT/result.json" \
  --progress-jsonl "$RUN_ROOT/progress.jsonl"
```

Short mechanism replication:

```bash
RUN_ROOT=/tmp/vllm_pre_service_kv_grow_replicate_short_20260714_185952 \
FIXTURE_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json \
PLAN_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=8 \
MAX_MODEL_LEN=2048 \
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
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

Long fixture plan generation:

```bash
python3 workloads/vllm/build_stage_m_hot_cold_weight_plan.py \
  --routing-trace /tmp/vllm_long_context_kv_grow_20260714/pilot_routing_21840/routing_trace.jsonl \
  --output-plan /tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json \
  --report-md /tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan_report.md \
  --hot-experts-per-layer 0 \
  --cold-experts-per-layer 5 \
  --cold-require-zero-hit \
  --fallback-layers 48 \
  --fallback-experts 128 \
  --target-roles moe_gate_up,moe_down
```

Long mechanism:

```bash
RUN_ROOT=/tmp/vllm_long_context_kv_grow_20260714/mechanism_190548 \
FIXTURE_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json \
PLAN_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=1 \
MAX_MODEL_LEN=21840 \
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
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

## Conclusion

This replication succeeded.

The short run reproduced the original mechanism: planned zero-hit experts were released, 128 KV blocks were offloaded and prefetched back to GPU before request serving, requests completed, and no GPU-side KV fault was observed.

The long prompt run gives the stronger evidence that was missing before: the request actually allocated beyond the initial 1236-block target, selecting blocks `1237..1283`, while the KV-only fault guard reported no GPU-side fault. This supports the intended safe path:

```text
CPU-resident writable KV blocks are not exposed to request allocation.
Blocks are prefetched back to GPU and confirmed resident before allocator visibility grows.
Formal request serving then uses only GPU-resident KV blocks.
```

Remaining limitation: this still does not prove exact PMM chunk identity reuse between a released expert chunk and a later KV block. It proves same-process expert release, KV physical grow-back, allocator visibility growth, and no KV fault under a real long-context request.
