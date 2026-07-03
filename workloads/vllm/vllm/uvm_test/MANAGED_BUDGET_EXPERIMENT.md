# Managed Budget Experiment

This is a small user-space experiment for separating two ideas:

```text
vLLM / caller logical capacity:
  Fixed cudaMallocManaged address space visible to the application.

GPU physical residency pressure:
  Controlled by reserving HBM before managed allocations and by later UVM
  prefetch / eviction policy.
```

## Implementation

`libuvm_managed_budget.so` is an `LD_PRELOAD` helper with two knobs.

### 1. HBM reserve

```bash
UVM_HBM_RESERVE_MB=8192 \
LD_PRELOAD=./libuvm_managed_budget.so \
python your_vllm_or_probe.py
```

Before the first `cudaMallocManaged`, the shim calls `cudaMalloc` for the
requested number of bytes. This consumes raw GPU HBM and reduces the physical
space available for UVM residency. Subsequent `cudaMallocManaged` calls are
still real managed allocations.

This models:

```text
HBM_total - reserved_HBM - safety_margin = effective UVM resident headroom
```

### 2. Managed arena cap

```bash
UVM_MANAGED_POOL_MB=16384 \
LD_PRELOAD=./libuvm_managed_budget.so \
python your_vllm_or_probe.py
```

The shim creates one fixed `cudaMallocManaged` arena and serves later
`cudaMallocManaged` / `cuMemAllocManaged` calls from that arena with a simple
first-fit suballocator. If live allocations exceed the arena capacity, the shim
returns `cudaErrorMemoryAllocation`.

This models a process-level logical managed-memory cap. A raw `cudaMalloc`
device pointer cannot be returned as a managed pointer, so the logical pool has
to be a managed arena.

Optional:

```bash
UVM_MANAGED_POOL_PREFETCH=1
```

This prefetches the entire managed arena to the selected GPU at initialization.
Keep it off for large oversubscription experiments.

## Smoke Experiment

```bash
cd workloads/vllm/vllm/uvm_test
./run_managed_budget_experiment.sh
```

Defaults:

```text
POOL_MB=128
HBM_RESERVE_MB=128
ALLOC_A_MB=64
ALLOC_B_MB=80
```

Expected behavior:

```text
baseline:
  64 MiB + 80 MiB managed allocations succeed.

hbm_reserve:
  The same allocations succeed, but cudaMemGetInfo free memory is lower by
  the raw HBM reserve.

managed_pool_cap:
  The first 64 MiB allocation succeeds.
  The second 80 MiB allocation fails because live managed bytes would exceed
  the 128 MiB arena.
```

Tune:

```bash
DEVICE=0 POOL_MB=4096 HBM_RESERVE_MB=8192 ALLOC_A_MB=2048 ALLOC_B_MB=3072 \
  ./run_managed_budget_experiment.sh
```

## Files

```text
cuda_managed_budget_preload.cpp   LD_PRELOAD implementation
managed_pool_probe.cu             CUDA smoke probe
run_managed_budget_experiment.sh  experiment runner
```
