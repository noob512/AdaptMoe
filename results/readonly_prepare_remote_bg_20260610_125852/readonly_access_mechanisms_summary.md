# Read-Only Access Mechanism Comparison

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_prepare_remote_bg_20260610_125852/readonly_access_mechanisms_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_prepare_remote_bg_20260610_125852/metadata.txt`

## Mechanisms

| Mechanism | Meaning |
|---|---|
| `uvm_pure_remote` | Managed memory, CPU preferred + GPU AccessedBy. The script tries to disable access-counter migration for this case; if the module parameter is read-only, keep `OPS_PER_PAGE * PASSES` below the threshold for a stricter remote-only run. |
| `uva_pinned_remote` | vLLM-style pinned CPU tensor exposed to GPU with UVA / `cudaHostGetDevicePointer`. No UVM managed page migration. |
| `readmostly_prepare_remote` | Managed memory initially CPU-resident, `ReadMostly`; before the first read the benchmark manually calls Phase F `PREPARE_REMOTE_READ` to pre-build GPU->CPU READ mappings outside the fault path. |
| `readmostly_prepare_remote_bg` | Managed memory initially CPU-resident, `ReadMostly`; before the first read the benchmark manually calls Phase F `PREPARE_REMOTE_READ_BG`, which pre-builds GPU->CPU READ mappings and enqueues background read-dup promotion. |
| `hybrid_remote_first_only` | Managed memory initially CPU-resident, `ReadMostly`; script enables modified driver remote-first READ mapping and disables background promotion. |
| `hybrid_remote_first_bg` | Managed memory initially CPU-resident, `ReadMostly`; script enables the modified driver remote-first READ mapping plus background read-dup promotion for this case only. |
| `readonly_dup` | Managed memory initially CPU-resident, `cudaMemAdviseSetReadMostly`; first GPU read may build a GPU read duplicate. |
| `direct_migrate` | Managed memory initially CPU-resident with no advice; normal on-demand migration on first GPU read. |
| `counter_migrate` | Managed memory CPU preferred + GPU AccessedBy with high ops/page to trigger access-counter behavior. |

## Performance

| Mechanism | Runs | Ops/page | Data MiB | Setup ms | Prepare ms | Prepared MiB | First pass ms | Second/last pass ms | First GiB/s | Last GiB/s | First vs pure remote | Last vs pure remote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uvm_pure_remote` | 3 | 64 | 64.0 | 36.995 | 0.000 | 0.0 | 2.957 | 2.916 | 2.642 | 2.679 | 1.00x | 1.00x |
| `readmostly_prepare_remote` | 3 | 64 | 64.0 | 40.007 | 0.302 | 64.0 | 2.963 | 2.915 | 2.636 | 2.680 | 1.00x | 1.00x |
| `readmostly_prepare_remote_bg` | 3 | 64 | 64.0 | 41.563 | 1.446 | 64.0 | 2.623 | 0.032 | 2.979 | 245.993 | 1.13x | 91.81x |
| `hybrid_remote_first_bg` | 3 | 64 | 64.0 | 38.101 | 0.000 | 0.0 | 12.869 | 0.034 | 0.607 | 231.139 | 0.23x | 86.27x |
| `readonly_dup` | 3 | 64 | 64.0 | 38.022 | 0.000 | 0.0 | 14.285 | 0.045 | 0.547 | 172.104 | 0.21x | 64.24x |

## How To Read This

- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.
- Compare `readmostly_prepare_remote` with `uvm_pure_remote` and `hybrid_remote_first_only` to test whether pre-building ReadMostly remote READ mappings outside the fault path removes the expensive first-touch mapping cost.
- Compare `readmostly_prepare_remote_bg` with `readmostly_prepare_remote` to isolate the extra prepare-time enqueue and background read-dup promotion effect.
- The target result for `readmostly_prepare_remote_bg` is first-pass latency close to `uvm_pure_remote`, then second/last-pass latency close to `readonly_dup` after the background promotion commits.
- Compare `hybrid_remote_first_only` and `hybrid_remote_first_bg` to isolate the cost of background promotion.
- Compare `hybrid_remote_first_bg` with `uvm_pure_remote` for first-pass latency and with `readonly_dup` for second-pass local-read convergence.
- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.
- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.
- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.
- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.

