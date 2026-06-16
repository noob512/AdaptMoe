# Isolated vLLM CPU Offload UVM Remote Experiment

- run_root: `/home/ubuntu/nvidia-uvm-gpu/log/stage_m_isolated_uvm_offload_remote_20260616_145756`
- global UVM allocator: not enabled (`VLLM_USE_UVM` is unset)
- LD_PRELOAD CUDA allocator interception: not used

| case | ok/fail | tok/s | TPOT ms | TTFT ms | tok/s/native | TPOT/native | params | bytes | backend | managed | prefetch CPU | preferred CPU | AccessedBy | same names |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| native_cpu_offload | 30/0 | 103.960 | 111.310 | 3931.070 | 1.000 | 1.000 | 233 | 8728868976 | {"uva": 233} | 0 | 0 | 0 | 0 | True |
| isolated_uvm_remote_offload | 30/0 | 106.440 | 107.750 | 3550.780 | 1.024 | 0.968 | 233 | 8728868976 | {"isolated_uvm_remote": 233} | 233 | 233 | 233 | 233 | True |

## Interpretation

- `native_cpu_offload` is the vLLM pinned CPU tensor + UVA CUDA view baseline.
- `isolated_uvm_remote_offload` uses the same `--cpu-offload-gb` selection, but those selected weights are allocated from a local UVM MemPool and prepared with CPU prefetch, PreferredLocation(CPU), and AccessedBy(GPU).
- The key validity checks are: same offload names, same manifest bytes, UVM managed count equals parameter count, and no failed requests.
