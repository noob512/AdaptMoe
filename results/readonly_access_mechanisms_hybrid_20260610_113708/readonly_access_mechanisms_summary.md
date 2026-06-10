# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_hybrid_20260610_113708/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_hybrid_20260610_113708/metadata.txt`

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
| `uvm_pure_remote` | 3 | 64 | 64.0 | 35.538 | 2.972 | 2.927 | 2.629 | 2.669 | 1.00x | 1.00x |
| `uva_pinned_remote` | 3 | 64 | 64.0 | 40.712 | 2.946 | 2.907 | 2.652 | 2.688 | 1.01x | 1.01x |
| `hybrid_remote_first_bg` | 3 | 64 | 64.0 | 43.528 | 12.195 | 0.033 | 0.641 | 233.334 | 0.24x | 87.42x |
| `readonly_dup` | 3 | 64 | 64.0 | 38.614 | 14.093 | 0.034 | 0.554 | 231.612 | 0.21x | 86.77x |
| `direct_migrate` | 3 | 64 | 64.0 | 34.115 | 28.080 | 0.048 | 0.278 | 163.002 | 0.11x | 61.07x |
| `counter_migrate` | 3 | 512 | 64.0 | 35.910 | 22.714 | 14.372 | 2.752 | 4.349 | 0.13x | 0.20x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare `hybrid_remote_first_bg` with `uvm_pure_remote` for first-pass latency and with `readonly_dup` for second-pass local-read convergence.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

