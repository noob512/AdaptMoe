# Stage M / Phase F Runtime Flow in vLLM

Date: 2026-05-14

This document explains how the current Phase F changes are used from vLLM
startup through inference. It focuses on the real execution path, not only the
smoke tests.

## 1. High-Level Flow

The current run path is:

```text
shell experiment script
  -> build uvm_allocator.so
  -> generate routing trace with pure UVM
  -> build hot/cold expert plan
  -> launch vLLM full_impl with UVM env vars
  -> vLLM loads Qwen3 MoE model with UVM allocator
  -> MoE layer registers expert weight regions
  -> init-time cold weights are offloaded through Phase F driver_batch
  -> runtime router_topk selects active experts
  -> active experts can be prefetched to GPU
  -> non-active experts can be evicted/offloaded to CPU
  -> real UVM residency snapshots query the kernel
  -> summary reports real GPU/CPU bytes by pool
```

## 2. Module and Shared Library Setup

The kernel module should be loaded with Phase F enabled and route delay disabled:

```bash
sudo rmmod nvidia_uvm || true

sudo insmod /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  uvm_enable_debug_procfs=1 \
  uvm_enable_builtin_tests=1 \
  uvm_phase_f_route_log_delay_ms=0 \
  uvm_perf_fault_log_destination=0 \
  uvm_perf_fault_log_counters=0 \
  uvm_perf_fault_log_addresses=0 \
  uvm_perf_fault_kv_range_enable=0
```

The experiment scripts rebuild:

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.so
```

This shared library is loaded by `vllm.device_allocator.uvm` through ctypes and
provides the C++ bridge to UVM driver ioctls.

## 3. UVM Allocator Initialization

When vLLM starts with:

```bash
VLLM_USE_UVM=1
```

the UVM allocator module loads `uvm_allocator.so`, initializes CUDA/UVM helper
state, and exposes Python wrappers for:

- allocation;
- managed-pointer probing;
- driver batch prefetch/offload/advise;
- real residency query.

The important Phase F mode is:

```bash
VLLM_UVM_PHASE_F_IOCTL_MODE=policy
```

This selects the production-style policy ioctl path rather than builtin test
ioctls.

## 4. Why Adopt the Existing CUDA UVM fd

CUDA runtime creates and owns a UVM VA space for managed allocations. If Phase F
opens a new `/dev/nvidia-uvm` fd, it creates a different UVM VA space and cannot
see the managed pointers registered by CUDA.

The C++ wrapper therefore scans:

```text
/proc/self/fd
```

for existing `/dev/nvidia-uvm` descriptors and duplicates one. The driver_batch
ioctl is then issued on the adopted fd, so UVM APIs operate on the same VA space
that owns the model weights.

Without this, the kernel can return invalid address errors for CUDA managed
weights even when the pointer is valid in the process.

## 5. Routing Trace and Hot/Cold Plan

Before the full Phase F run, `run_stage_m_phase_f_check.sh` performs a pure-UVM
profile pass:

1. launch vLLM without Phase F weight actions;
2. collect MoE router_topk records;
3. write `vllm_uvm_moe_routing_trace.jsonl`;
4. build a routing-aware hot/cold plan.

The plan builder outputs:

```text
routing_phase_f_weight_plan.json
routing_phase_f_weight_plan_report.md
```

The plan decides which experts are hot and which are cold per layer. For example:

```text
hot=48 cold=80
```

means each layer keeps 48 likely useful experts in the hot set and treats 80 as
cold/offload candidates.

## 6. Model Load and Weight Region Registration

During vLLM model load, the MoE layer code records each expert weight tensor.
For every registered expert role, it captures:

- pointer;
- byte size;
- layer ID;
- expert ID;
- role;
- driver-capable status;
- driver-capable reason.

The main roles are:

```text
moe_gate_up
moe_down
```

For Qwen/Qwen3-30B-A3B-FP8, full MoE coverage currently registers:

```text
48 layers * 128 experts * 2 roles = 12288 regions
total registered weight bytes ~= 27GiB
```

Registration feeds two systems:

1. Python pool residency manager for estimated residency and traces;
2. kernel Phase F region registry for driver-backed actions and query.

## 7. Init-Time Weight Offload

After model weights are loaded, Stage M init-time placement runs over MoE
modules. The cold plan is applied:

```text
cold experts -> driver_batch offload_cpu
```

In the Phase F path, offload typically maps to:

1. `ADVISE_CPU`;
2. `PREFETCH_CPU`;
3. update Python-side estimated state.

The latest active-expert runs show init offload issuing thousands of records,
for example:

```text
init_weight_offload_cpu_issued_total ~= 12192
```

This is expected when nearly all MoE expert regions are offloaded at startup.

## 8. Runtime Router-TopK Prefetch

During inference, each MoE layer obtains router-topk selected experts for the
current tokens. The runtime prefetch path can issue:

```text
active experts -> driver_batch prefetch_gpu
```

The controls are:

```bash
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1
VLLM_UVM_WEIGHT_PREFETCH_SOURCE=router_topk
VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER=<n>
VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP=<bytes>
VLLM_UVM_WEIGHT_EXECUTOR=driver_batch
```

The user-facing wrapper exposes this as:

```bash
ACTIVE_EXPERTS_PER_LAYER=<n>
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP=<bytes>
```

## 9. Runtime Active-Expert Eviction

The active eviction path is designed to avoid long-term accumulation of old
experts on GPU.

At a high level:

```text
active set = router_topk experts for this step/layer
victim set = registered resident experts not in active set
action = offload_cpu / prefetch_cpu
```

Controls:

```bash
VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE=1
VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER=<n>
VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP=<n>
VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_ACTION=offload_cpu
```

This path is useful for strict memory control, but too aggressive a setting
hurts performance. The `hot=1/cold=127` run reached only `0.255GiB` GPU resident
weight bytes, far below a 10GiB budget, and can be very slow. For performance,
use a larger hot set and a smaller eviction cap.

## 10. Pool Residency Budgets

The Stage L residency manager receives separate budgets:

```bash
VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES
VLLM_UVM_RESIDENCY_KV_GPU_BYTES
VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES
VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES
```

In the active-experts wrapper:

```bash
WEIGHT_GPU_BYTES=10737418240  # 10GiB
KV_GPU_BYTES=4294967296       # 4GiB
```

The weight budget controls the target resident MoE weight pool. The KV budget is
passed to both residency accounting and init KV sizing, so increasing it allows a
larger GPU KV cache than the previous 2GiB runs.

## 11. Real UVM Residency Measurement

The real residency path is enabled by:

```bash
VLLM_UVM_REAL_RESIDENCY_ENABLE=1
VLLM_UVM_REAL_RESIDENCY_TRACE_FILE=<run_root>/real_residency.jsonl
VLLM_UVM_REAL_RESIDENCY_DEVICE=0
```

Sampling frequency is controlled by:

```bash
VLLM_UVM_REAL_RESIDENCY_ON_REGISTER_EVERY
VLLM_UVM_REAL_RESIDENCY_ON_ACTION_EVERY
VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS
VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL
```

The Python manager groups registered ranges by pool and calls:

```text
driver_batch_query_ranges_residency_bytes()
```

The query is split into 64-range batches because the ioctl limit is 64 ranges.
For the full MoE weight pool:

```text
12288 regions / 64 = 192 query batches
```

The summary reports:

- registered bytes;
- queried bytes;
- GPU bytes;
- CPU bytes;
- unknown bytes;
- failed batches.

## 12. What the Latest Successful Run Means

The latest successful run produced:

```text
phase_f_exit_code: 0
snapshots: 16
ok_snapshots: 16
weights registered: 27.000GiB
weights queried: 27.000GiB
weights gpu: 0.255GiB
weights cpu: 26.745GiB
unknown: 0.000GiB
failed_batches: 0
```

Interpretation:

- vLLM completed the Phase F check;
- Phase F driver_batch was used;
- the complete MoE weight pool was measured;
- kernel residency query worked for all 192 batches;
- the configured 10GiB weight budget was satisfied;
- the policy was more aggressive than necessary for performance.

## 13. Recommended Performance/Memory Experiment

To move from extreme offload toward a useful performance/memory tradeoff:

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

pkill -f "vllm serve Qwen/Qwen3-30B-A3B-FP8" || true

VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
HOT_EXPERTS_PER_LAYER=48 \
COLD_EXPERTS_PER_LAYER=80 \
ACTIVE_EXPERTS_PER_LAYER=32 \
WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER=16 \
WEIGHT_MAX_VICTIMS_PER_STEP=16 \
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP=268435456 \
WEIGHT_GPU_BYTES=10737418240 \
KV_GPU_BYTES=4294967296 \
REAL_ON_REGISTER_EVERY=8192 \
REAL_ON_ACTION_EVERY=512 \
REAL_MAX_SNAPSHOTS=16 \
PROMPTS=1 \
OUTPUT_LEN=16 \
./run_stage_m_active_experts_residency.sh \
  --run-root log/stage_m_active_experts_residency_hot48_kv4g_run \
  --log-file log/stage_m_active_experts_residency_hot48_kv4g.log
```

Expected direction:

- GPU MoE weight residency should rise from `0.255GiB` toward `8-10GiB`;
- performance should improve relative to `hot=1/cold=127`;
- KV cache budget increases from 2GiB to 4GiB;
- query should still report `failed_batches=0` and `unknown=0`.

## 14. Known Limitations

- Current real residency snapshots cover registered pools. Weight coverage is
  strong for MoE expert weights; KV and scratch require explicit registration to
  appear in the same summary.
- Real residency query is intentionally expensive for 12288 regions. Use sparse
  sampling for performance experiments.
- A 10GiB budget is an upper bound, not a target. The policy must retain enough
  hot experts if the goal is to use 8-10GiB for better throughput.
- Runtime active eviction is functional but can be too aggressive. Tune hot set,
  prefetch cap, and eviction cap together.

