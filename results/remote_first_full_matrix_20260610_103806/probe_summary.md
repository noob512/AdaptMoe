# Remote-First Background Promotion Counter Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_103806/probe_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_103806/metadata.txt`
- Counter before: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_103806/latency_stats_before.txt`
- Counter after: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_103806/latency_stats_after.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_103806/latency_stats_delta.txt`
- Per-case counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_103806/counter_delta_by_case.csv`

## Total Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 7018 |
| `remote_first_pages` | 38259 |
| `bg_promote_enqueued_pages` | 32768 |
| `bg_promote_copied_pages` | 32768 |
| `bg_promote_committed_pages` | 32768 |
| `bg_promote_cancelled_pages` | 0 |
| `remote_map_pages` | 38259 |
| `bg_promote_demand_pages_enqueued` | 4863 |
| `bg_promote_prefetch_pages_enqueued` | 27905 |
| `bg_promote_demand_pages_promoted` | 4863 |
| `bg_promote_prefetch_pages_promoted` | 27905 |
| `pte_switch_pages` | 32768 |
| `bg_promote_worker_rounds` | 632 |
| `bg_promote_retry_count` | 0 |
| `bg_promote_failed_not_cpu_resident` | 0 |
| `bg_promote_failed_already_gpu_resident` | 0 |
| `bg_promote_failed_allocation` | 0 |
| `bg_promote_failed_copy` | 0 |
| `bg_promote_failed_unmap` | 0 |
| `bg_promote_failed_map` | 0 |
| `bg_promote_failed_block_dead` | 0 |
| `sync_fallback_pages` | 768 |

## Per-Case Counter Delta

| Case | Profile | RF | BG | Ops/page | RF faults | RF pages | Remote map pages | BG enqueued | Demand enq | Prefetch enq | BG copied | BG committed | PTE switch | BG cancelled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | `remote_first_bg` | 1 | 1 | 128 | 785 | 5491 | 5491 | 32768 | 4863 | 27905 | 32768 | 32768 | 32768 | 0 |
| `remote_first_only` | `remote_first_only` | 1 | 0 | 128 | 6233 | 32768 | 32768 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `readonly_dup_sync` | `readonly_dup_sync` | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `counter_migrate` | `counter_migrate` | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `local_prefetch` | `local_prefetch` | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `default_uvm` | `default_uvm` | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Per-Case Worker And Failure Detail

| Case | Demand promoted | Prefetch promoted | Worker rounds | Retries | Not CPU resident | Already GPU resident | Alloc fail | Copy fail | Unmap fail | Map fail | Dead block | Sync fallback pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | 4863 | 27905 | 632 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 384 |
| `remote_first_only` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 384 |
| `readonly_dup_sync` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `counter_migrate` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `local_prefetch` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `default_uvm` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Performance

| Profile | Runs | Ops/page | Pass0 ms | Last-pass ms | Median ms | Last-pass GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | 2 | 128 | 11.412 | 0.044 | 0.045 | 356.979 |
| `remote_first_only` | 2 | 128 | 57.404 | 4.576 | 5.565 | 3.436 |
| `readonly_dup_sync` | 2 | 128 | 14.081 | 0.042 | 0.044 | 372.370 |
| `counter_migrate` | 2 | 512 | 22.632 | 10.533 | 10.541 | 5.934 |
| `local_prefetch` | 2 | 128 | 0.107 | 0.043 | 0.044 | 363.962 |
| `default_uvm` | 2 | 128 | 27.251 | 0.043 | 0.044 | 365.864 |

## Interpretation

- `remote_first_bg` should exercise the new full path: remote-first fault service plus background COPY/read-dup promotion.
  Observed: both remote-first and committed background promotion counters increased.
- `remote_first_only` disables background promotion to isolate foreground remote-first latency.
  Observed: remote-first counters increased while background counters stayed zero.
- Native read-dup first pass vs remote-first+BG first pass: `14.081 ms` vs `11.412 ms`.
- Remote-first only first pass vs remote-first+BG first pass: `57.404 ms` vs `11.412 ms`.
- `counter_migrate` uses CPU preferred + AccessedBy with `512` ops/page to test access-counter driven promotion/migration behavior.
- Default UVM first pass vs remote-first+BG first pass: `27.251 ms` vs `11.412 ms`.
- Local prefetch last pass vs remote-first+BG last pass: `0.043 ms` vs `0.044 ms`.
