# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_smoke_20260610_111744/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_mechanisms_smoke_20260610_111744/metadata.txt`

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
| `uvm_pure_remote` | 1 | 4 | 1.0 | 1.449 | 0.049 | 0.019 | 0.155 | 0.412 | 1.00x | 1.00x |
| `uva_pinned_remote` | 1 | 4 | 1.0 | 1.760 | 0.054 | 0.019 | 0.141 | 0.393 | 0.91x | 0.95x |
| `readonly_dup` | 1 | 4 | 1.0 | 1.556 | 0.657 | 0.017 | 0.012 | 0.447 | 0.08x | 1.08x |
| `direct_migrate` | 1 | 4 | 1.0 | 1.341 | 0.735 | 0.019 | 0.010 | 0.394 | 0.07x | 0.96x |
| `counter_migrate` | 1 | 16 | 1.0 | 1.531 | 0.062 | 0.029 | 0.494 | 1.038 | 0.80x | 0.63x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

