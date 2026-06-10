# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_remote_first_only_check_20260610_114328/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_remote_first_only_check_20260610_114328/metadata.txt`

## Mechanisms

| Mechanism | Meaning |
|---|---|
| `uvm_pure_remote` | Managed memory, CPU preferred + GPU AccessedBy. The script tries to disable access-counter migration for this case; if the module parameter is read-only, keep `OPS_PER_PAGE * PASSES` below the threshold for a stricter remote-only run. |
| `uva_pinned_remote` | vLLM-style pinned CPU tensor exposed to GPU with UVA / `cudaHostGetDevicePointer`. No UVM managed page migration. |
| `hybrid_remote_first_only` | Managed memory initially CPU-resident, `ReadMostly`; script enables modified driver remote-first READ mapping and disables background promotion. |
| `hybrid_remote_first_bg` | Managed memory initially CPU-resident, `ReadMostly`; script enables the modified driver remote-first READ mapping plus background read-dup promotion for this case only. |
| `readonly_dup` | Managed memory initially CPU-resident, `cudaMemAdviseSetReadMostly`; first GPU read may build a GPU read duplicate. |
| `direct_migrate` | Managed memory initially CPU-resident with no advice; normal on-demand migration on first GPU read. |
| `counter_migrate` | Managed memory CPU preferred + GPU AccessedBy with high ops/page to trigger access-counter behavior. |

## Performance

| Mechanism | Runs | Ops/page | Data MiB | Setup ms | First pass ms | Second/last pass ms | First GiB/s | Last GiB/s | First vs pure remote | Last vs pure remote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uvm_pure_remote` | 3 | 64 | 64.0 | 34.938 | 2.975 | 2.933 | 2.626 | 2.663 | 1.00x | 1.00x |
| `uva_pinned_remote` | 3 | 64 | 64.0 | 42.875 | 2.945 | 2.919 | 2.653 | 2.677 | 1.01x | 1.01x |
| `hybrid_remote_first_only` | 3 | 64 | 64.0 | 38.338 | 56.122 | 2.948 | 0.139 | 2.650 | 0.05x | 1.00x |
| `hybrid_remote_first_bg` | 3 | 64 | 64.0 | 38.134 | 12.575 | 0.034 | 0.621 | 228.202 | 0.24x | 85.68x |
| `readonly_dup` | 3 | 64 | 64.0 | 42.040 | 15.101 | 0.057 | 0.517 | 137.864 | 0.20x | 51.76x |
| `direct_migrate` | 3 | 64 | 64.0 | 34.940 | 26.939 | 0.034 | 0.290 | 231.283 | 0.11x | 86.84x |
| `counter_migrate` | 3 | 512 | 64.0 | 36.224 | 22.674 | 14.441 | 2.756 | 4.328 | 0.13x | 0.20x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare `hybrid_remote_first_only` and `hybrid_remote_first_bg` to isolate the cost of background promotion.
- Compare `hybrid_remote_first_bg` with `uvm_pure_remote` for first-pass latency and with `readonly_dup` for second-pass local-read convergence.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

