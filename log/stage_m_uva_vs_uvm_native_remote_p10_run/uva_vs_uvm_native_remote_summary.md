# vLLM UVA Offload vs UVM Remote Access

- run_root: `/home/ubuntu/nvidia-uvm-gpu/log/stage_m_uva_vs_uvm_native_remote_p10_run`
- native_manifest: `/home/ubuntu/nvidia-uvm-gpu/log/stage_m_uva_vs_uvm_native_remote_p10_run/native_cpu_offload/offload_manifest.jsonl`
- manifest_parameters: `233`
- manifest_bytes: `8728868976`

## Performance

| case | mode | tok/s | TPOT ms | TTFT ms | tok/s/native | TPOT/native | matched | CPU prefetch | CPU preferred | AccessedBy | remote-first pages |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_cpu_offload | native_uva | 6.630 | 133.300 | 600.370 | 1.000 | 1.000 |  |  |  |  | 0 |
| uvm_pure_remote | native_remote_access | 3.950 | 240.830 | 1417.770 | 0.596 | 1.807 | 149/233 | 149 | 149 | 149 | 0 |
| uvm_default | default_uvm | 3.920 | 223.550 | 1556.290 | 0.591 | 1.677 |  |  |  |  | 0 |

## Reading The Result

- `native_cpu_offload` is the vLLM pinned-host UVA remote-access baseline.
- `uvm_pure_remote` uses the same parameter manifest and applies native UVM `CPU prefetch + PreferredLocation(CPU) + AccessedBy(GPU)`, matching the readonly microbenchmark's pure-remote mechanism.
- `uvm_default` only enables the UVM allocator and applies no matched manifest placement or advise policy.
- Neither UVM case uses ReadMostly, modified-driver remote-first, background promotion, or UVM CPU-offload controls.
- `remote_first_*` counters are expected to stay near zero here; these cases are not testing the modified remote-first path.
- Treat performance as matched only if TPOT and output tok/s are within about 5% of native with zero failed requests.
