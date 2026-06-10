# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_smoke_20260610_111819/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_smoke_20260610_111819/metadata.txt`

## Mechanisms

| Mechanism | Meaning |
|---|---|
| `uvm_pure_remote` | Managed memory, CPU preferred + GPU AccessedBy, access-counter migration disabled for this case. This approximates pure UVM remote access. |
| `uva_pinned_remote` | vLLM-style pinned CPU tensor exposed to GPU with UVA / `cudaHostGetDevicePointer`. No UVM managed page migration. |
| `readonly_dup` | Managed memory initially CPU-resident, `cudaMemAdviseSetReadMostly`; first GPU read may build a GPU read duplicate. |
| `direct_migrate` | Managed memory initially CPU-resident with no advice; normal on-demand migration on first GPU read. |
| `counter_migrate` | Managed memory CPU preferred + GPU AccessedBy with high ops/page to trigger access-counter behavior. |

## Performance

| Mechanism | Runs | Ops/page | Data MiB | Setup ms | First pass ms | Second/last pass ms | First GiB/s | Last GiB/s | First vs pure remote | Last vs pure remote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uvm_pure_remote` | 1 | 2 | 0.2 | 1.251 | 0.046 | 0.017 | 0.021 | 0.057 | 1.00x | 1.00x |
| `uva_pinned_remote` | 1 | 2 | 0.2 | 1.541 | 0.084 | 0.016 | 0.011 | 0.059 | 0.54x | 1.03x |
| `readonly_dup` | 1 | 2 | 0.2 | 1.128 | 0.428 | 0.017 | 0.002 | 0.057 | 0.11x | 1.00x |
| `direct_migrate` | 1 | 2 | 0.2 | 1.360 | 0.626 | 0.019 | 0.002 | 0.050 | 0.07x | 0.88x |
| `counter_migrate` | 1 | 4 | 0.2 | 1.295 | 0.047 | 0.018 | 0.040 | 0.104 | 0.97x | 0.90x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

