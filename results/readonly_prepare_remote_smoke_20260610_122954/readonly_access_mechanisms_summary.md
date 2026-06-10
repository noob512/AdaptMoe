# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_prepare_remote_smoke_20260610_122954/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_prepare_remote_smoke_20260610_122954/metadata.txt`

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
| `uvm_pure_remote` | 1 | 64 | 16.0 | 9.554 | 0.000 | 0.0 | 0.768 | 0.740 | 2.542 | 2.639 | 1.00x | 1.00x |
| `readmostly_prepare_remote` | 1 | 64 | 16.0 | 13.719 | 0.154 | 16.0 | 0.772 | 0.736 | 2.531 | 2.652 | 1.00x | 1.00x |
| `hybrid_remote_first_only` | 1 | 64 | 16.0 | 10.086 | 0.000 | 0.0 | 14.175 | 0.744 | 0.138 | 2.625 | 0.05x | 0.99x |
| `readonly_dup` | 1 | 64 | 16.0 | 10.698 | 0.000 | 0.0 | 3.420 | 0.030 | 0.571 | 64.398 | 0.22x | 24.40x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare `readmostly_prepare_remote` with `uvm_pure_remote` and `hybrid_remote_first_only` to test whether pre-building ReadMostly remote READ mappings outside the fault path removes the expensive first-touch mapping cost.
- Compare `hybrid_remote_first_only` and `hybrid_remote_first_bg` to isolate the cost of background promotion.
- Compare `hybrid_remote_first_bg` with `uvm_pure_remote` for first-pass latency and with `readonly_dup` for second-pass local-read convergence.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

