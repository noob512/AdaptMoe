# vLLM KV Pre-Service Quarantine + Between-Request Grow Back

Date: 2026-07-14

Run root:

```text
/tmp/vllm_kv_between_request_grow_back_20260714_170613/run
```

This document is the consolidated record for the final 128-block KV grow-back smoke. It supersedes the earlier intermediate KV diagnosis result files for this investigation.

## What This Experiment Tested

The goal was to verify the safe runtime sequence for writable KV cache blocks:

```text
1. Finish model load / warmup.
2. Before EngineCore READY, shrink the allocator-visible KV pool.
3. Quarantine 128 tail KV blocks so vLLM cannot allocate or write them.
4. Physically prefetch those 128 KV blocks from GPU to CPU.
5. Serve request 0 using only GPU-resident allocator-visible KV blocks.
6. Pause between requests.
7. Physically prefetch the quarantined 128 KV blocks back to GPU.
8. Grow the allocator-visible KV target so those blocks become usable again.
9. Serve request 1.
10. Confirm no KV GPU fault was observed.
```

The important design invariant is:

```text
CPU-resident writable KV blocks must not be allocator-visible.
They may become allocator-visible only after GPU prefetch succeeds and residency query confirms gpu_bytes == block_bytes.
```

This directly tests the fix for the previous failure mode, where KV shrink/offload happened inside the request path after `request_begin`.

## Configuration

```text
model = Qwen/Qwen3-30B-A3B-FP8 local snapshot
request fixture = workloads/vllm/manual_kv_fault_two_request_fixture.json
requests = 2
max output tokens per request = 16
max_model_len = 2048
max_num_seqs = 1
max_num_batched_tokens = 2048
kv_cache_memory_bytes = 2147483648

logical KV blocks = 1364
bytes_per_block = 1572864
logical KV bytes = 2146959360

initial_target_blocks = 1236
quarantined blocks = 1237..1364
quarantined block count = 128
quarantined bytes = 201326592

grow_after_request = 0
grow_target_blocks = 1364
```

KV migration and diagnosis settings:

```text
KV_DATA_OFFLOAD_MODE = prefetch_cpu
KV_PREFETCH_BACK_MODE = prefetch
KV_DRIVER_TAG_ENABLE = 1
KV_FORCE_DRIVER_TAG = 1
KV_POINTER_TAG_GRANULARITY = tensor
KV_PRE_SERVICE_APPLY = 1
KV_IN_PROCESS_GROW_AFTER_REQUEST = 0
KV_IN_PROCESS_GROW_TARGET_BLOCKS = 1364

VLLM_UVM_KV_CACHE_MANAGED = 1
VLLM_UVM_EXPERT_WEIGHTS_MANAGED = 1
```

Expert offload settings:

```text
policy = static_tail5_zero_hit_v1
offloaded experts = 123..127 in every layer
layers = 48
target roles = moe_gate_up, moe_down
expert envelope bytes = 524288
```

The static tail5 plan was chosen to avoid runtime expert restore noise. The two test requests are expected not to route into these tail experts.

## Code Paths Exercised

The experiment uses these integration points:

```text
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
workloads/vllm/vllm/uvm_test/vllm_generate_sequence.py
workloads/vllm/vllm/vllm/v1/engine/core.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
```

Key mechanisms:

```text
VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY=1
  Applies the initial KV shrink before EngineCore READY.

apply_uvm_kv_dynamic_pool_target(...)
  EngineCore utility used by vllm_generate_sequence.py to grow the KV pool between requests.

VLLM_UVM_RESIDENCY_KV_FORCE_DRIVER_TAG=1
  Forces KV driver tag registration even when allocator-side KV tagging is enabled.

KV_POINTER_TAG_GRANULARITY=tensor
  Registers 48 tensor-span KV tags instead of 1364 * 48 range tags. This is fast enough for the smoke and still catches KV-pool faults at tensor-span granularity.
```

## Execution Timeline

Observed progress:

```text
load_begin
load_end
request0 begin
request0 end
between_requests_kv_dynamic_pool_grow_begin
between_requests_kv_dynamic_pool_grow_end
request1 begin
request1 end
sequence_end
```

Timestamps:

```text
request0 begin: 1784020038.0466595
request0 end:   1784020039.087133
grow begin:     1784020039.1366005
grow end:       1784020046.1049209
request1 begin: 1784020046.1634772
request1 end:   1784020047.1505227
```

The 128-block grow happened strictly between requests. It did not run inside an active request.

The load and READY-side ordering was:

```text
Stage-M expert release
-> pre-service KV dynamic pool apply
-> EngineCore READY
-> request0
-> between-request grow
-> request1
```

## Functional Result

PASS.

```text
completed_requests = 2
status = ok
output_tokens = 32
new Xid/MMU fault = false
manual_migration_fault records = 0
fault_address records = 0
```

Expert release succeeded:

```text
expert release entries = 2160
expert release failed entries = 0
expert released bytes = 1108869120
expert released MiB = 1057.5

cold_plan_gpu_bytes before = 1108869120
cold_plan_gpu_bytes after = 0
cold_plan_cpu_bytes before = 23592960
cold_plan_cpu_bytes after = 1132462080
```

KV pre-service shrink/quarantine succeeded:

```text
old_target_blocks = 1364
new_target_blocks = 1236
offloaded_blocks = 128
reason = pre_service_before_ready
skipped_unsafe_blocks = 0
```

KV between-request grow-back succeeded:

```text
old_target_blocks = 1236
new_target_blocks = 1364
prefetched_blocks = 128
reason = between_requests_manual_grow
skipped_unsafe_blocks = 0
```

## Physical Migration Evidence

The KV physical trace proves that both directions completed:

```text
physical begin records = 256
physical end records = 256
CPU success records = 128
GPU success records = 128
unmatched begin records = 0
managed_probe_failures = 0
```

Each KV block spans 48 ranges in this model run:

```text
range_count per block = 48
bytes per block = 1572864
```

First quarantined block before CPU migration:

```text
block_id = 1237
direction = cpu
gpu_bytes = 1572864
cpu_bytes = 0
managed_probe_success_count = 48
```

First quarantined block after CPU migration:

```text
block_id = 1237
accepted_ranges = 48
gpu_bytes = 0
cpu_bytes = 1572864
```

Last quarantined block before GPU prefetch back:

```text
block_id = 1364
direction = gpu
gpu_bytes = 0
cpu_bytes = 1572864
managed_probe_success_count = 48
```

Last quarantined block after GPU prefetch back:

```text
block_id = 1364
accepted_ranges = 48
gpu_bytes = 1572864
cpu_bytes = 0
```

## Allocator Evidence

Before grow, the allocator skipped the quarantined tail range:

```text
dynamic_pool_target_blocks = 1236
selected_block_ids = [1], [2]
skipped_dynamic_pool = 128
cpu_candidates = 0
gpu_candidates = 1236, then 1235
unknown_candidates = 0
```

After grow, the full KV block range became allocator-visible again:

```text
dynamic_pool_target_blocks = 1364
selected_block_ids = [3], [4]
skipped_dynamic_pool = 0
cpu_candidates = 0
gpu_candidates = 1363, then 1362
unknown_candidates = 0
```

This is the main correctness result:

```text
The 128 CPU-resident KV blocks were not visible to allocator before grow.
They became visible only after GPU prefetch back completed.
The request path selected no CPU-resident KV blocks.
```

## Fault Guard Result

KV driver tag registration was enabled:

```text
KV_FORCE_DRIVER_TAG = 1
KV_POINTER_TAG_GRANULARITY = tensor
registered KV tensor spans = 48
```

No KV manual fault records appeared in the run traces:

```text
manual_migration_fault records = 0
fault_address records = 0
```

This means the run did not observe GPU access to a KV driver-tagged CPU-resident region.

## Performance

This was a correctness smoke, not a clean throughput benchmark. Runtime tracing, physical residency probes, fsync physical trace, UVM ioctl logging, and expert offload instrumentation were enabled.

Observed timings:

```text
load_seconds = 57.076
request0 generate_seconds = 1.040
between-request grow elapsed_seconds = 6.968
request1 generate_seconds = 0.987
total_generate_seconds = 2.028
output_tokens = 32
tokens/s = 15.783
p50_request_seconds = 1.014
```

Interpretation:

```text
Request latency did not get worse after grow:
  request0 = 1.040s
  request1 = 0.987s

The visible cost is the explicit 128-block physical prefetch back:
  128 blocks = 201326592 bytes = 192 MiB
  grow elapsed = 6.968s
  average = 54.4 ms/block
```

The grow time is high because this smoke intentionally used conservative, fully synchronous diagnostics:

```text
48 ranges per KV block
128 blocks
6144 total ranges moved back
begin/end trace for every block
residency query / managed probe
fsync-enabled physical trace
```

So the performance conclusion is:

```text
The safe sequencing works and does not create steady request-path fault overhead.
The current physical grow implementation is diagnostic-grade and too slow for production.
The next performance step is to batch more ranges per ioctl, reduce trace/fsync, and move grow into a controlled maintenance window.
```

## Reproduction

Before running, confirm the correct UVM module is loaded and the GPU is clean:

```bash
cd /home/ubuntu/nvidia-uvm-gpu
cat /sys/module/nvidia_uvm/srcversion
cat /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

Expected:

```text
srcversion = 386A1FA2E18DDDFCE9EFE88
manual fault mode = 0 before/after normal runs
GPU memory.used near 0 MiB before starting
```

If UVM must be reloaded:

```bash
sudo systemctl stop node-exporter.service gpu-exporter.service 2>/dev/null || true
sudo rmmod nvidia_uvm
sudo insmod /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  uvm_enable_debug_procfs=1 \
  uvm_enable_builtin_tests=1 \
  uvm_manual_migration_fault_mode=0 \
  uvm_manual_migration_fault_pool_mask=2 \
  uvm_manual_migration_fault_log_enable=1 \
  uvm_perf_fault_log_destination=0 \
  uvm_perf_fault_log_counters=0 \
  uvm_perf_fault_log_addresses=0
cat /sys/module/nvidia_uvm/srcversion
sudo systemctl start node-exporter.service 2>/dev/null || true
```

Create a static tail5 zero-hit plan:

```bash
python3 - <<'PY'
import json
from pathlib import Path

plan = {
    "metadata": {
        "description": "Static tail5 plan for between-request KV grow-back smoke",
        "layers": 48,
        "policy": "static_tail5_zero_hit_v1",
        "tail_experts": [123, 124, 125, 126, 127],
    },
    "offload_plan": [
        {"layer_id": layer, "expert_id": expert}
        for layer in range(48)
        for expert in [123, 124, 125, 126, 127]
    ],
}

path = Path("/tmp/tail5_offload_plan.json")
path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
```

Run:

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=/tmp/vllm_kv_between_request_grow_back_repro_$(date +%Y%m%d_%H%M%S)/run \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=2 \
FIXTURE_SOURCE=/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/manual_kv_fault_two_request_fixture.json \
PLAN_SOURCE=/tmp/tail5_offload_plan.json \
INITIAL_TARGET_BLOCKS=1236 \
GROW_TARGET_BLOCKS=1364 \
GROW_AFTER_COMPLETED_REQUESTS=99 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_DRIVER_TAG_ENABLE=1 \
KV_FORCE_DRIVER_TAG=1 \
KV_POINTER_TAG_GRANULARITY=tensor \
KV_PRE_SERVICE_APPLY=1 \
KV_IN_PROCESS_GROW_AFTER_REQUEST=0 \
KV_IN_PROCESS_GROW_TARGET_BLOCKS=1364 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

Notes:

```text
GROW_AFTER_COMPLETED_REQUESTS=99
  Keeps the external control-file watcher from triggering during this two-request run.

KV_IN_PROCESS_GROW_AFTER_REQUEST=0
  Triggers the EngineCore utility grow after request 0 completes.

KV_PRE_SERVICE_APPLY=1
  Moves the initial shrink/quarantine before EngineCore READY.
```

Useful validation commands:

```bash
RUN_ROOT=/tmp/vllm_kv_between_request_grow_back_20260714_170613/run

cat "$RUN_ROOT/summary.json"
cat "$RUN_ROOT/zero_hit_offload/progress.jsonl"
jq 'select(.action=="kv_dynamic_pool_resize" or .action=="kv_dynamic_pool_target_update")' \
  "$RUN_ROOT/zero_hit_offload/kv_runtime_trace.jsonl"
jq 'select(.action=="gpu_resident_free_block_select")' \
  "$RUN_ROOT/zero_hit_offload/kv_runtime_trace.jsonl"
head -3 "$RUN_ROOT/zero_hit_offload/kv_physical_trace.jsonl"
tail -3 "$RUN_ROOT/zero_hit_offload/kv_physical_trace.jsonl"
rg -n "manual_migration_fault|fault_address|gpu_fault" "$RUN_ROOT"
```

Expected pass criteria:

```text
completed_requests = 2
status = ok
expert release failed entries = 0
physical CPU success records = 128
physical GPU success records = 128
physical unmatched begin records = 0
managed_probe_failures = 0
manual_migration_fault records = 0
allocator cpu_candidates = 0
```

## What This Proves

This run proves:

```text
1. KV cache is UVM-managed in this path.
2. 128 KV blocks can be quarantined before service and moved GPU -> CPU.
3. vLLM can serve a request while those CPU-resident blocks are hidden from allocator.
4. The same 128 blocks can be moved CPU -> GPU between requests.
5. The allocator-visible KV target can grow from 1236 to 1364 after prefetch back.
6. Subsequent request execution does not trigger observed KV GPU faults.
```

This run also explains the earlier hang more precisely:

```text
previous failing path:
  shrink/offload happened after request_begin, inside or too close to the active request path

successful path:
  shrink happens before READY
  CPU-resident blocks are hidden from allocator
  grow happens between requests
  blocks are prefetched back before allocator exposure
```

## What This Does Not Yet Prove

This run does not prove exact PMM chunk identity reuse:

```text
expert PMM chunk X -> KV block Y
```

It proves successful expert release and successful KV grow-back in the same process. To prove exact reuse, we still need PMM chunk identity tracing in the driver.

## Superseded Intermediate Result Files

The following intermediate result files were consolidated into this document and can be removed from the working tree:

```text
docs/vllm_kv_manual_fault_failfast_diagnosis_20260714.md
docs/vllm_kv_physical_migration_diagnosis_20260714.md
docs/vllm_kv_pre_service_fault_guard_results_20260714.md
docs/vllm_zero_hit_dynamic_kv_batch_grow_224to512_20260714.md
docs/vllm_zero_hit_dynamic_kv_batch_grow_after_uvm_reload_20260714.md
docs/vllm_zero_hit_dynamic_kv_strong_compare_attempt_20260714.md
docs/vllm_zero_hit_expert_offload_dynamic_kv_grow_results_20260714.md
```
