# Stage M / Phase F Current Modification Summary

Date: 2026-05-14

This document summarizes the current project changes for Phase F driver-backed
UVM weight placement, active-expert residency control, and real GPU residency
measurement.

## 1. Goal

The current implementation is no longer only a plumbing smoke test. Its goal is
to let vLLM manage MoE expert weights through UVM driver operations:

- register MoE expert weight ranges in the UVM VA space used by CUDA/vLLM;
- issue driver-backed batch actions for CPU offload, GPU prefetch, CPU advise,
  and residency query;
- keep most cold expert weights on CPU while allowing a configurable hot/active
  working set on GPU;
- measure real page residency in the kernel, not only a Python-side allocation
  ledger.

The latest successful memory-management check showed:

```text
phase_f_exit_code: 0
weights queried: 27.000 GiB
weights gpu: 0.255 GiB
weights cpu: 26.745 GiB
failed_batches: 0
unknown: 0
```

This proves that Phase F can control the complete 27GiB MoE expert weight pool
below the configured 10GiB GPU budget. That run used an aggressive
`hot=1/cold=127` policy, so it used far less than 10GiB; later experiments should
use larger hot sets for performance.

## 2. Kernel / UVM Module Changes

Files:

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_test_ioctl.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_test.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.h`

### 2.1 Policy ioctls

Phase F now has policy-level UVM ioctls in addition to builtin-test smoke ioctls.
The policy path is the intended route for vLLM integration.

Implemented actions:

- `PREFETCH_CPU`
- `PREFETCH_GPU`
- `ADVISE_CPU`
- `QUERY_RESIDENCY`

The batch executor accepts up to 64 ranges per ioctl. User-space wrappers split
larger requests into 64-range chunks.

### 2.2 Region registry in UVM VA space

The UVM VA space now has a Phase F region registry. vLLM can register a weight
range once with metadata, then later issue actions by region ID instead of
re-sending only raw addresses.

Recorded metadata includes:

- region ID;
- base address and length;
- pool kind, such as weights;
- layer ID;
- expert ID;
- role, such as `moe_gate_up` or `moe_down`;
- flags.

This registry is important because it ties Phase F actions to the same UVM VA
space that owns the managed allocations.

### 2.3 Real resident byte accounting

Kernel query now walks UVM VA blocks and computes page-overlap residency bytes
for a queried range. Returned counters include:

- queried bytes;
- GPU resident bytes for a target GPU UUID;
- CPU resident bytes;
- any resident bytes;
- unknown bytes.

This is the key difference from earlier accounting: it reports physical UVM page
residency, not only a Python-side estimate.

### 2.4 Diagnostics

The kernel logs additional Phase F diagnostics:

- ioctl begin/end;
- action, count, async flag, first range;
- per-range VA-space and VA-range diagnostics;
- status aggregation;
- query residency bytes.

`uvm_phase_f_route_log_delay_ms` was added as a module parameter and is passed by
the test scripts. It should be `0` for real experiments.

## 3. User-Space C++ Wrapper Changes

File:

- `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

### 3.1 Driver batch APIs

The shared library exports Phase F driver-backed functions used by Python:

- `uvm_driver_batch_prefetch_cpu`
- `uvm_driver_batch_prefetch_gpu`
- `uvm_driver_batch_advise_cpu`
- `uvm_driver_batch_query_residency`
- `uvm_driver_batch_query_residency_bytes`
- Phase F region register/unregister/batch-region helpers
- `uvm_is_managed_pointer`

### 3.2 Policy vs test ioctl mode

The wrapper supports both:

- builtin test ioctl mode for smoke tests;
- policy ioctl mode for vLLM integration.

For real vLLM runs, use:

```bash
VLLM_UVM_PHASE_F_IOCTL_MODE=policy
```

### 3.3 Adopt CUDA's UVM fd

The wrapper now scans `/proc/self/fd` for an existing `/dev/nvidia-uvm` file
descriptor and duplicates it. This is necessary because CUDA managed allocations
are registered in the CUDA runtime's existing UVM VA space.

The old behavior of opening a fresh `/dev/nvidia-uvm` fd created a different VA
space. That caused driver UVM APIs to reject CUDA managed pointers as invalid
addresses. The adopted-fd path fixes that class of failure.

### 3.4 Managed-pointer probing

`uvm_is_managed_pointer()` lets Python check whether a weight pointer is actually
CUDA managed and driver-capable before forcing driver_batch execution.

## 4. Python vLLM Integration Changes

Files:

- `workloads/vllm/vllm/vllm/device_allocator/uvm.py`
- `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py`
- `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`

### 4.1 ctypes wrappers

`uvm.py` now exposes:

- driver_batch prefetch/offload/advise/query wrappers;
- 64-range batching for driver ioctls;
- `driver_batch_query_ranges_residency_bytes()`;
- `probe_driver_capable_range()`;
- fallback behavior for `auto` executor mode.

Executor modes:

- `driver_batch`: require Phase F driver ioctl path;
- `cpp_batch`: use previous user-space CUDA helper path;
- `auto`: try driver_batch first, fallback when needed.

### 4.2 Pool residency manager

`uvm_pool_residency.py` tracks registered regions by pool:

- weights;
- KV;
- scratch.

It now also supports real residency sampling:

- enabled by `VLLM_UVM_REAL_RESIDENCY_ENABLE=1`;
- writes JSONL snapshots;
- batches queries by 64 ranges;
- reports per-pool real GPU/CPU/unknown bytes.

### 4.3 MoE weight registration and tracing

`fused_moe/layer.py` now records MoE expert regions with richer metadata:

- layer;
- expert;
- role;
- pointer;
- size;
- driver-capable flag;
- driver-capable reason.

This lets traces distinguish managed/driver-capable weights from weights that
must fallback.

### 4.4 Init-time offload

During model initialization, cold expert weights from the routing plan can be
offloaded to CPU through Phase F driver_batch.

The latest full MoE experiments registered:

```text
regions: 12288
queried weight bytes: 27.000 GiB
```

### 4.5 Runtime prefetch and active eviction

Runtime router-topk prefetch can move active experts to GPU before use.

Active expert eviction was added:

- env gate: `VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE`;
- scan cap: `VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER`;
- victim action: usually `offload_cpu`;
- protects currently active experts and evicts non-active resident candidates.

This is the start of the target policy:

```text
init: mostly offload experts to CPU
runtime: prefetch active router-topk experts
runtime: evict non-active resident experts
measure: query real UVM residency
```

## 5. Script and Experiment Changes

Files:

- `workloads/vllm/run_kv_fault_ratio.sh`
- `workloads/vllm/run_stage_l_full_impl_uvm_ab.sh`
- `workloads/vllm/run_stage_m_phase_f_check.sh`
- `workloads/vllm/run_stage_m_phase_f_smoke.sh`
- `workloads/vllm/run_stage_m_real_residency_check.sh`
- `workloads/vllm/run_stage_m_real_residency_full_pools.sh`
- `workloads/vllm/run_stage_m_active_experts_residency.sh`

### 5.1 Module reload parameter propagation

`run_kv_fault_ratio.sh` now passes:

```bash
uvm_phase_f_route_log_delay_ms="$UVM_PHASE_F_ROUTE_LOG_DELAY_MS"
```

during UVM reloads. This fixed the earlier issue where manual `insmod
uvm_phase_f_route_log_delay_ms=0` was overwritten by an internal test-script
reload that silently restored the default delay.

### 5.2 Phase F smoke tests

`run_stage_m_phase_f_smoke.sh` validates:

- UVM module parameter state;
- ping ioctl;
- batch mini ioctl;
- query/prefetch/advise paths;
- policy ioctl mode;
- adopted fd path.

### 5.3 Phase F check

`run_stage_m_phase_f_check.sh` builds the allocator library, generates a routing
trace, builds a hot/cold expert plan, runs the full implementation, and validates
that driver_batch was used.

It now exposes:

- `--hot-experts-per-layer`;
- `--cold-experts-per-layer`;
- `--weight-gpu-bytes`;
- `--kv-gpu-bytes`;
- `--weight-active-evict-enable`;
- `--weight-active-evict-max-experts-per-layer`.

### 5.4 Real residency check

`run_stage_m_real_residency_check.sh` wraps Phase F check and summarizes real UVM
residency.

Important behavior:

- generates `real_residency.jsonl`;
- generates `real_residency_summary.json`;
- generates `real_residency_summary.md`;
- still generates the summary even if the inner Phase F benchmark exits nonzero;
- includes `phase_f_exit_code` in the summary.

This allows memory-management validation even when vLLM benchmark readiness or
generation is slow.

### 5.5 Active-experts residency script

`run_stage_m_active_experts_residency.sh` is the main current experiment wrapper.
It now accepts:

- `HOT_EXPERTS_PER_LAYER`;
- `COLD_EXPERTS_PER_LAYER`;
- `ACTIVE_EXPERTS_PER_LAYER`;
- `WEIGHT_GPU_BYTES`;
- `KV_GPU_BYTES`;
- active eviction limits;
- real residency sampling controls.

This makes it possible to test memory/performance tradeoffs such as:

```text
hot=48 cold=80 weight_budget=10GiB kv_budget=4GiB
```

## 6. Current Experimental Findings

### 6.1 Driver and residency correctness

The latest `memcheck10g` run:

```text
phase_f_exit_code: 0
snapshots: 16
ok_snapshots: 16
weights registered: 27.000 GiB
weights queried: 27.000 GiB
weights gpu: 0.255 GiB
weights cpu: 26.745 GiB
unknown: 0.000 GiB
failed_batches: 0
```

Conclusion:

- complete MoE expert pool is registered;
- real UVM residency query works;
- Phase F driver_batch is functional;
- 27GiB MoE weights can be kept below a 10GiB budget.

### 6.2 Performance warning

`hot=1/cold=127` is too aggressive. It minimizes GPU weight residency, but can
make generation very slow because most expert accesses fault or require
prefetching from CPU.

For performance/memory tradeoff experiments, prefer:

```text
hot=40 cold=88
hot=48 cold=80
hot=56 cold=72
```

with a 10GiB weight budget and a larger KV budget such as 4GiB.

## 7. Current Recommended Experiment

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

Success criteria:

```text
weights queried ~= 27GiB
weights gpu <= 10GiB
weights gpu ideally ~= 8-10GiB
unknown == 0
failed_batches == 0
phase_f_exit_code == 0 if the benchmark completes
```

