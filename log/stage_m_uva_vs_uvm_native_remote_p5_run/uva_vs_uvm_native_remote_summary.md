# vLLM UVA Offload vs UVM Remote Access

- run_root: `/home/ubuntu/nvidia-uvm-gpu/log/stage_m_uva_vs_uvm_native_remote_p5_run`
- native_manifest: `/home/ubuntu/nvidia-uvm-gpu/log/stage_m_uva_vs_uvm_native_remote_p5_run/native_cpu_offload/offload_manifest.jsonl`
- manifest_parameters: `233`
- manifest_bytes: `8728868976`

## Performance

| case | mode | allocator scope | tok/s | TPOT ms | TTFT ms | tok/s/native | TPOT/native | matched | CPU prefetch | CPU preferred | AccessedBy | remote-first pages |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_cpu_offload | native_uva | none | 8.310 | 98.610 | 1196.920 | 1.000 | 1.000 |  |  |  |  | 0 |
| uvm_pure_remote | native_remote_access | pytorch_cuda_allocator | 3.980 | 235.650 | 5159.800 | 0.479 | 2.390 | 149/233 | 149 | 149 | 149 | 0 |
| uvm_default_migration | default_uvm_migration | pytorch_cuda_allocator | 3.440 | 225.560 | 6649.830 | 0.414 | 2.287 |  |  |  |  | 0 |
| uvm_all_alloc_default_migration | all_alloc_default_uvm_migration | pytorch_cuda_allocator_plus_ldpreload_cuda_alloc | 3.330 | 231.230 | 6977.760 | 0.401 | 2.345 |  |  |  |  | 0 |

## Reading The Result

- `native_cpu_offload` is the vLLM pinned-host UVA remote-access baseline.
- `uvm_pure_remote` uses the same parameter manifest and applies native UVM `CPU prefetch + PreferredLocation(CPU) + AccessedBy(GPU)`, matching the readonly microbenchmark's pure-remote mechanism.
- `uvm_default_migration` enables the PyTorch CUDA UVM allocator and applies no matched manifest placement or advise policy; it is the default UVM demand-migration control.
- `uvm_all_alloc_default_migration` also LD_PRELOADs a CUDA allocator interceptor so cudaMalloc/cudaMallocAsync/cuMemAlloc are redirected to managed memory where interposition works.
- No UVM case uses ReadMostly, modified-driver remote-first, background promotion, or UVM CPU-offload controls.
- `remote_first_*` counters are expected to stay near zero here; these cases are not testing the modified remote-first path.
- Treat performance as matched only if TPOT and output tok/s are within about 5% of native with zero failed requests.
