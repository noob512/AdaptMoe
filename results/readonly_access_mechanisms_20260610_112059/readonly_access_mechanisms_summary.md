# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_20260610_112059/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_20260610_112059/metadata.txt`

## Mechanisms

| Mechanism | Meaning |
|---|---|
| `uvm_pure_remote` | Managed memory, CPU preferred + GPU AccessedBy. The script tries to disable access-counter migration for this case; if the module parameter is read-only, keep `OPS_PER_PAGE * PASSES` below the threshold for a stricter remote-only run. |
| `uva_pinned_remote` | vLLM-style pinned CPU tensor exposed to GPU with UVA / `cudaHostGetDevicePointer`. No UVM managed page migration. |
| `readonly_dup` | Managed memory initially CPU-resident, `cudaMemAdviseSetReadMostly`; first GPU read may build a GPU read duplicate. |
| `direct_migrate` | Managed memory initially CPU-resident with no advice; normal on-demand migration on first GPU read. |
| `counter_migrate` | Managed memory CPU preferred + GPU AccessedBy with high ops/page to trigger access-counter behavior. |

## Performance

| Mechanism | Runs | Ops/page | Data MiB | Setup ms | First pass ms | Second/last pass ms | First GiB/s | Last GiB/s | First vs pure remote | Last vs pure remote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uvm_pure_remote` | 3 | 64 | 64.0 | 35.073 | 2.954 | 2.923 | 2.644 | 2.672 | 1.00x | 1.00x |
| `uva_pinned_remote` | 3 | 64 | 64.0 | 40.397 | 2.953 | 2.923 | 2.645 | 2.673 | 1.00x | 1.00x |
| `readonly_dup` | 3 | 64 | 64.0 | 38.058 | 14.242 | 0.032 | 0.549 | 240.881 | 0.21x | 90.14x |
| `direct_migrate` | 3 | 64 | 64.0 | 33.805 | 27.646 | 0.035 | 0.283 | 221.858 | 0.11x | 83.02x |
| `counter_migrate` | 3 | 512 | 64.0 | 36.880 | 22.530 | 15.078 | 2.774 | 4.145 | 0.13x | 0.19x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

