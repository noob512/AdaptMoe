# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_hybrid_smoke_20260610_113555/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_hybrid_smoke_20260610_113555/metadata.txt`

## Mechanisms

| Mechanism | Meaning |
|---|---|
| `uvm_pure_remote` | Managed memory, CPU preferred + GPU AccessedBy. The script tries to disable access-counter migration for this case; if the module parameter is read-only, keep `OPS_PER_PAGE * PASSES` below the threshold for a stricter remote-only run. |
| `uva_pinned_remote` | vLLM-style pinned CPU tensor exposed to GPU with UVA / `cudaHostGetDevicePointer`. No UVM managed page migration. |
| `hybrid_remote_first_bg` | Managed memory initially CPU-resident, `ReadMostly`; script enables the modified driver remote-first READ mapping plus background read-dup promotion for this case only. |
| `readonly_dup` | Managed memory initially CPU-resident, `cudaMemAdviseSetReadMostly`; first GPU read may build a GPU read duplicate. |
| `direct_migrate` | Managed memory initially CPU-resident with no advice; normal on-demand migration on first GPU read. |
| `counter_migrate` | Managed memory CPU preferred + GPU AccessedBy with high ops/page to trigger access-counter behavior. |

## Performance

| Mechanism | Runs | Ops/page | Data MiB | Setup ms | First pass ms | Second/last pass ms | First GiB/s | Last GiB/s | First vs pure remote | Last vs pure remote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uvm_pure_remote` | 1 | 1 | 0.1 | 1.145 | 0.046 | 0.017 | 0.005 | 0.014 | 1.00x | 1.00x |
| `uva_pinned_remote` | 1 | 1 | 0.1 | 1.764 | 0.052 | 0.018 | 0.005 | 0.013 | 0.89x | 0.93x |
| `hybrid_remote_first_bg` | 1 | 1 | 0.1 | 1.119 | 0.342 | 0.017 | 0.001 | 0.014 | 0.14x | 0.97x |
| `readonly_dup` | 1 | 1 | 0.1 | 1.095 | 0.465 | 0.016 | 0.001 | 0.015 | 0.10x | 1.06x |
| `direct_migrate` | 1 | 1 | 0.1 | 1.024 | 0.432 | 0.020 | 0.001 | 0.012 | 0.11x | 0.85x |
| `counter_migrate` | 1 | 2 | 0.1 | 1.204 | 0.048 | 0.016 | 0.010 | 0.030 | 0.96x | 1.07x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare `hybrid_remote_first_bg` with `uvm_pure_remote` for first-pass latency and with `readonly_dup` for second-pass local-read convergence.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

