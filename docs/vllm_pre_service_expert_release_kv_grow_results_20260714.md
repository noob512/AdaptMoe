# vLLM pre-service expert release + KV grow experiment

Date: 2026-07-14

Run root:

```text
/tmp/vllm_pre_service_expert_release_kv_grow_kvguard_sync_20260714_184853
```

## Goal

Validate this sequence inside vLLM:

```text
1. Clean baseline:
   run the same fixed requests without expert offload, dynamic KV, driver tag,
   or residency tracing.

2. Mechanism run:
   after warmup and before serving real requests:
     a. stop unpredictable KV automatic migration by enabling KV-only
        manual fault fail-fast before requests;
     b. release previously collected zero-hit cold experts;
     c. shrink the usable KV pool to 1236 blocks;
     d. grow it back to 1364 blocks before READY;
     e. physically prefetch the grown KV blocks back to GPU;
     f. serve the same requests and compare performance.
```

The allocation policy was kept narrow:

```text
VLLM_USE_UVM=0
VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_KV_CACHE_MANAGED=1
```

That means only MoE expert main weights and KV cache use managed UVM allocation. Other model weights and temporary buffers stay on the normal CUDA allocation path.

Loaded UVM module:

```text
nvidia_uvm srcversion = 386A1FA2E18DDDFCE9EFE88
```

## Code adjustment

The KV physical prefetch helper uses `cudaMemPrefetchAsync`; the underlying allocator shim explicitly leaves stream synchronization to the caller. I added a narrow experimental switch:

```text
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1
```

When enabled, KV physical CPU/GPU prefetch does a CUDA synchronize before the physical residency query is recorded. Without this, the prefetch calls can be accepted but the immediate residency query may still see a few blocks in transit.

Changed file:

```text
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
```

Syntax check:

```bash
python3 -m py_compile \
  workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py \
  workloads/vllm/vllm/uvm_test/vllm_generate_sequence.py \
  workloads/vllm/vllm/vllm/v1/engine/core.py
```

## Request fixture

Both runs used the same 8 fixed requests:

```text
/tmp/vllm_pre_service_expert_release_kv_grow_kvguard_sync_20260714_184853/sharegpt_fixture.json
```

Each request generated 64 tokens:

```text
completed_requests = 8
output_tokens = 512
max_model_len = 2048
max_num_seqs = 1
kv_cache_memory_bytes = 2147483648
```

The zero-hit cold expert plan came from the previous pilot trace for this same fixture:

```text
/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json
```

Plan shape:

```text
48 layers
5 zero-hit experts per layer
240 expert records
roles = moe_gate_up, moe_down
```

## Results

| Case | completed | output tokens | generate seconds | tokens/s | load seconds |
|---|---:|---:|---:|---:|---:|
| clean baseline | 8/8 | 512 | 23.662 | 21.638 | 36.590 |
| zero-hit expert release + pre-service KV grow | 8/8 | 512 | 28.783 | 17.788 | 69.519 |

Compared to the clean baseline, the mechanism run is:

```text
throughput drop = 17.79%
```

This request-time gap is not caused by runtime expert restore, because this plan is zero-hit for the fixed fixture. The likely remaining cost is the enabled experimental control path: dynamic KV policy bookkeeping, allocator selection tracing, residency/physical trace, and KV-only fault guard. The pre-service work itself is counted in `load_seconds`, not in `generate seconds`.

Pre-service overhead is visible in load time:

```text
baseline load_seconds   = 36.590
mechanism load_seconds  = 69.519
delta                   = 32.929 seconds
```

The mechanism run includes Stage-M expert release plus KV shrink/grow and physical prefetch before READY.

## Expert release evidence

Stage-M released the planned zero-hit experts successfully:

```text
driver release entries = 2160
failed entries         = 0
released bytes         = 1102577664
released MiB           = 1051.5
```

Residency query for the cold plan:

```text
before Stage-M:
  cold_plan_gpu_bytes = 1102577664
  cold_plan_cpu_bytes = 29884416

after Stage-M:
  cold_plan_gpu_bytes = 0
  cold_plan_cpu_bytes = 1132462080
```

So the planned experts were really removed from GPU residency before requests started.

## KV grow evidence

The dynamic KV pool was applied before READY:

```text
initial_target_blocks = 1236
grow_target_blocks    = 1364
added_blocks          = 128
bytes_per_block       = 1572864
added KV bytes        = 201326592
added KV MiB          = 192
```

At 16 tokens per block, this increases the usable KV target by approximately:

```text
128 blocks * 16 tokens/block = 2048 tokens

1236 blocks ~= 19776 tokens
1364 blocks ~= 21824 tokens
```

Physical migration evidence:

```text
KV CPU offload success records = 128
KV GPU prefetch success records = 128
physical begin records          = 256
physical end records            = 256
unmatched begin records         = 0
managed probe failures          = 0
prefetch fallback records       = 0
physical grow pass              = true
```

The synchronized physical trace confirms all 128 grown blocks were resident on GPU before serving requests.

Allocator evidence during requests:

```text
allocation_no_free_blocks = 0
dynamic_pool_target_blocks = 1364
max selected block id in this short run = 40
cpu_candidates = 0
skipped_dynamic_pool = 0
```

This proves the usable KV target was increased and exposed to the allocator. The short 8-request fixture did not consume enough context to reach the tail blocks; it validates the capacity change and the absence of CPU-resident candidates, not a true 20k-token prompt.

## Fault guard evidence

Before the first request, the run enabled KV-only manual fault fail-fast:

```text
uvm_manual_migration_fault_mode = 2
uvm_manual_migration_fault_pool_mask = 2
uvm_manual_migration_fault_log_enable = 1
```

There were no new fault records:

```text
new experiment dmesg faultish lines = 0
new baseline dmesg faultish lines   = 0
new Xid/MMU fault                   = false
```

This is the important safety signal: with CPU-resident KV blocks quarantined and grown blocks prefetched back before requests, the request path did not touch KV pages that required forbidden automatic migration.

After the run:

```text
uvm_manual_migration_fault_mode = 0
uvm_manual_migration_fault_pool_mask = 0
GPU memory.used = 0 MiB
```

## Reproduction

Clean baseline:

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=/tmp/vllm_pre_service_expert_release_kv_grow_kvguard_sync_20260714_184853
BASE_DIR="${RUN_ROOT}/baseline_clean_rerun"
mkdir -p "${BASE_DIR}"

(
  cd workloads/vllm/vllm
  VLLM_USE_UVM=0 \
  VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0 \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
  VLLM_UVM_KV_CACHE_MANAGED=1 \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_KV_RUNTIME_ENABLE=0 \
  VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
  VLLM_UVM_RESIDENCY_TRACE_ENABLE=0 \
  VLLM_UVM_DRIVER_TAG_ENABLE=0 \
  VLLM_UVM_DRIVER_KV_TAG_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
  VLLM_UVM_WEIGHT_MAP_ENABLE=0 \
  VLLM_UVM_MOE_ADDRESS_MAP_ENABLE=0 \
  timeout 1200s uv run python uvm_test/vllm_generate_sequence.py \
    --model /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782 \
    --request-fixture-json "${RUN_ROOT}/sharegpt_fixture.json" \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --kv-cache-memory-bytes 2147483648 \
    --gpu-memory-utilization 0.90 \
    --output-json "${BASE_DIR}/result.json" \
    --progress-jsonl "${BASE_DIR}/progress.jsonl"
)
```

Mechanism run:

```bash
cd /home/ubuntu/nvidia-uvm-gpu

FIXTURE_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json \
PLAN_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json \
RUN_ROOT=/tmp/vllm_pre_service_expert_release_kv_grow_kvguard_sync_20260714_184853 \
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
KV_IN_PROCESS_GROW_AFTER_REQUEST=-1 \
KV_IN_PROCESS_GROW_TARGET_BLOCKS=0 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_FORCE_DRIVER_TAG=1 \
KV_DRIVER_TAG_ENABLE=0 \
KV_POINTER_TAG_GRANULARITY=tensor \
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 \
TIMEOUT_SEC=1200 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
VLLM_UVM_MANUAL_FAULT_USE_SUDO=1 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

## Interpretation

This run proves the intended pre-service control sequence works:

```text
warmup
-> release zero-hit experts
-> quarantine/shrink KV target
-> grow KV target before READY
-> physically prefetch grown KV blocks to GPU
-> enable KV-only fail-fast
-> serve requests without KV GPU faults
```

It also proves the expert release is large enough for the requested KV grow in this run:

```text
released expert GPU residency ~= 1051.5 MiB
new KV GPU residency           ~= 192 MiB
```

What is not yet proven:

```text
1. Exact PMM chunk identity reuse from a released expert chunk to a specific KV block.
2. A true >19776-token prompt consuming the newly exposed tail KV blocks.
```

The second point should be tested with a separate long-context fixture whose routing plan is generated from that same long-context workload. Reusing this zero-hit plan for an unrelated long prompt would not be safe, because the long prompt could select an expert that this short-fixture plan offloaded.
