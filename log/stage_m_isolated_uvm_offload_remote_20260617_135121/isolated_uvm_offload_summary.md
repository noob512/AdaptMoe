# Isolated vLLM CPU Offload UVM Remote Experiment

- run_root: `/home/ubuntu/nvidia-uvm-gpu/log/stage_m_isolated_uvm_offload_remote_20260617_135121`
- global UVM allocator: not enabled (`VLLM_USE_UVM` is unset)
- LD_PRELOAD CUDA allocator interception: not used

| case | ok/fail | tok/s | TPOT ms | TTFT ms | tok/s/native | TPOT/native | params | bytes | backend | managed | prefetch CPU | preferred CPU | AccessedBy | ReadMostly | remote-bg prepare | gate events | gate start | same names |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| native_cpu_offload | 5/0 | 12.260 | 46.640 | 257.250 | 1.000 | 1.000 | 233 | 8728868976 | {"uva": 233} | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| isolated_uvm_migration_offload | 5/0 | 12.660 | 45.650 | 170.600 | 1.033 | 0.979 | 233 | 8728868976 | {"isolated_uvm_migration": 233} | 233 | 233 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| isolated_uvm_remote_offload | 5/0 | 12.360 | 45.540 | 245.560 | 1.008 | 0.976 | 233 | 8728868976 | {"isolated_uvm_remote": 233} | 233 | 233 | 233 | 233 | 0 | 0 | 0 | 0 | True |
| isolated_uvm_hybrid_remote_offload | 5/0 | 10.810 | 70.210 | 399.600 | 0.882 | 1.505 | 233 | 8728868976 | {"isolated_uvm_hybrid_remote": 233} | 233 | 233 | 233 | 233 | 233 | 0 | 1386 | 0 | True |

## Interpretation

- `native_cpu_offload` is the vLLM pinned CPU tensor + UVA CUDA view baseline.
- `isolated_uvm_migration_offload` uses the same `--cpu-offload-gb` selection, but those selected weights are allocated from a local UVM MemPool and only prefetched to CPU, leaving later GPU reads to the default UVM demand migration/fault path.
- `isolated_uvm_remote_offload` uses the same selected weights, additionally sets PreferredLocation(CPU) and tags AccessedBy(GPU), so it exercises the native UVM remote-read path.
- `isolated_uvm_hybrid_remote_offload` uses the same selected weights, additionally sets ReadMostly, calls the modified UVM remote-read/background-promotion prepare path, and gates background promotion around module forward calls.
- The key validity checks are: same offload names, same manifest bytes, UVM managed count equals parameter count, remote-bg prepare count equals parameter count for the hybrid case, and no failed requests.
