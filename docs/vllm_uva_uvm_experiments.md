# vLLM UVA vs UVM remote access experiments

## Experiment 1

Goal: test whether replacing vLLM native CPU offload's UVA/pinned-host remote access with UVM pure remote access keeps the same performance, and compare it with default UVM migration controls.

Plan:

1. Run native `--cpu-offload-gb` once and capture the exact offloaded parameter manifest.
2. Reuse the same manifest and run the UVM pure remote matched mode:
   - `uvm_pure_remote`
3. The UVM pure remote case matches the readonly microbenchmark definition:
   - prefetch the matched managed ranges to CPU for placement
   - set `cudaMemAdviseSetPreferredLocation(CPU)`
   - set `cudaMemAdviseSetAccessedBy(GPU)` for native UVM remote mappings
4. Also run default UVM migration controls:
   - `uvm_default_migration`
   - only enables `VLLM_USE_UVM=1`
   - makes PyTorch CUDA allocator allocations use `cudaMallocManaged`
   - applies no matched manifest placement or advise policy
5. Run an all-allocation default UVM migration control:
   - `uvm_all_alloc_default_migration`
   - enables `VLLM_USE_UVM=1`
   - LD_PRELOADs `libcudamalloc_managed.so`
   - redirects intercepted `cudaMalloc`, `cudaMallocAsync`, and `cuMemAlloc` calls to managed memory
   - applies no matched manifest placement or advise policy
6. The UVM pure remote case deliberately does not use:
   - `cudaMemAdviseSetReadMostly`
   - modified-driver remote-first
   - background promotion
   - UVM CPU-offload controls
   - KV fault logging sysfs parameters from `run_kv_fault_ratio.sh`
7. Compare throughput, TTFT, TPOT, matched parameter count, allocator scope, and the prefetch/PreferredLocation/AccessedBy preparation result.

Script:

- `workloads/vllm/run_stage_m_uva_offload_vs_uvm_remote_p10.sh`

Interpretation:

- If TPOT and throughput stay within about 5% of native offload, the UVM pure remote path is performance-equivalent.
- `remote_first_*` counters are expected to stay near zero in this experiment, because this is not the modified remote-first path.
- A valid `uvm_pure_remote` run should show the manifest ranges matched and `native_remote_prefetch_cpu_successes` / `native_remote_preferred_cpu_result` / `native_remote_accessed_by_result` close to the matched range count.
- `uvm_default_migration` is not expected to match the native offload manifest; it is the no-extra-policy PyTorch allocator UVM demand-migration control.
- `uvm_all_alloc_default_migration` is the broader no-extra-policy UVM demand-migration control. It is useful for checking whether non-PyTorch CUDA allocations materially change the result.

## Experiment 2

Goal: prove that pages not touched by CPU have no CPU mapping and therefore no unmap cost, including the first transition after `PREFETCH_CPU_NO_MAP`.

Plan:

1. Allocate managed memory.
2. Do not dereference it on CPU.
3. Let GPU first-touch/write the pages.
4. Run `GPU_ONLY_RW_SOFT` once and check that the first GPU-only apply still has `gpu_only_cpu_unmap_pages == 0`.
5. Migrate them back to CPU with `PREFETCH_CPU_NO_MAP`.
6. Run `GPU_ONLY_RW_SOFT` again and check `gpu_only_cpu_unmap_pages == 0`.
7. Repeat with an explicit CPU touch after `PREFETCH_CPU_NO_MAP` to show the opposite outcome.

Script:

- `scripts/test_stage_d_gpu_first_no_cpu_map_probe.sh`

Interpretation:

- `none` case: no CPU touch, unmap bytes/pages should stay at 0.
- `touch_after_no_map` case: CPU read after `PREFETCH_CPU_NO_MAP` should recreate CPU mappings, so the next GPU-only apply should unmap pages.
