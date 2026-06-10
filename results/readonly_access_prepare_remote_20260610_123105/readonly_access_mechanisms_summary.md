# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_prepare_remote_20260610_123105/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_access_prepare_remote_20260610_123105/metadata.txt`

## Mechanisms

| Mechanism | Meaning |
|---|---|
| `uvm_pure_remote` | Managed memory, CPU preferred + GPU AccessedBy. The script tries to disable access-counter migration for this case; if the module parameter is read-only, keep `OPS_PER_PAGE * PASSES` below the threshold for a stricter remote-only run. |
| `uva_pinned_remote` | vLLM-style pinned CPU tensor exposed to GPU with UVA / `cudaHostGetDevicePointer`. No UVM managed page migration. |
| `readmostly_prepare_remote` | Managed memory initially CPU-resident, `ReadMostly`; before the first read the benchmark manually calls Phase F `PREPARE_REMOTE_READ` to pre-build GPU->CPU READ mappings outside the fault path. |
| `hybrid_remote_first_only` | Managed memory initially CPU-resident, `ReadMostly`; script enables modified driver remote-first READ mapping and disables background promotion. |
| `hybrid_remote_first_bg` | Managed memory initially CPU-resident, `ReadMostly`; script enables the modified driver remote-first READ mapping plus background read-dup promotion for this case only. |
| `readonly_dup` | Managed memory initially CPU-resident, `cudaMemAdviseSetReadMostly`; first GPU read may build a GPU read duplicate. |
| `direct_migrate` | Managed memory initially CPU-resident with no advice; normal on-demand migration on first GPU read. |
| `counter_migrate` | Managed memory CPU preferred + GPU AccessedBy with high ops/page to trigger access-counter behavior. |

## Performance

| Mechanism | Runs | Ops/page | Data MiB | Setup ms | Prepare ms | Prepared MiB | First pass ms | Second/last pass ms | First GiB/s | Last GiB/s | First vs pure remote | Last vs pure remote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uvm_pure_remote` | 3 | 64 | 64.0 | 35.354 | 0.000 | 0.0 | 2.956 | 2.914 | 2.643 | 2.681 | 1.00x | 1.00x |
| `uva_pinned_remote` | 3 | 64 | 64.0 | 40.605 | 0.000 | 0.0 | 2.944 | 2.911 | 2.654 | 2.684 | 1.00x | 1.00x |
| `readmostly_prepare_remote` | 3 | 64 | 64.0 | 39.973 | 0.304 | 64.0 | 2.947 | 2.908 | 2.651 | 2.686 | 1.00x | 1.00x |
| `hybrid_remote_first_only` | 3 | 64 | 64.0 | 37.504 | 0.000 | 0.0 | 55.425 | 2.932 | 0.141 | 2.664 | 0.05x | 0.99x |
| `hybrid_remote_first_bg` | 3 | 64 | 64.0 | 38.577 | 0.000 | 0.0 | 12.385 | 0.033 | 0.631 | 234.666 | 0.24x | 87.52x |
| `readonly_dup` | 3 | 64 | 64.0 | 38.267 | 0.000 | 0.0 | 13.248 | 0.033 | 0.590 | 238.201 | 0.22x | 88.84x |
| `direct_migrate` | 3 | 64 | 64.0 | 33.738 | 0.000 | 0.0 | 28.054 | 0.038 | 0.278 | 207.525 | 0.11x | 77.40x |
| `counter_migrate` | 3 | 512 | 64.0 | 36.074 | 0.000 | 0.0 | 22.471 | 12.540 | 2.781 | 4.984 | 0.13x | 0.23x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare `readmostly_prepare_remote` with `uvm_pure_remote` and `hybrid_remote_first_only` to test whether pre-building ReadMostly remote READ mappings outside the fault path removes the expensive first-touch mapping cost.
- Compare `hybrid_remote_first_only` and `hybrid_remote_first_bg` to isolate the cost of background promotion.
- Compare `hybrid_remote_first_bg` with `uvm_pure_remote` for first-pass latency and with `readonly_dup` for second-pass local-read convergence.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

