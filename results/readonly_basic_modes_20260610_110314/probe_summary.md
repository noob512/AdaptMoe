# Remote-First Background Promotion Counter Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_basic_modes_20260610_110314/probe_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_basic_modes_20260610_110314/metadata.txt`
- Counter before: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_basic_modes_20260610_110314/latency_stats_before.txt`
- Counter after: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_basic_modes_20260610_110314/latency_stats_after.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_basic_modes_20260610_110314/latency_stats_delta.txt`
- Per-case counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/readonly_basic_modes_20260610_110314/counter_delta_by_case.csv`

## Total Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 4025 |
| `remote_first_pages` | 24576 |
| `bg_promote_enqueued_pages` | 0 |
| `bg_promote_copied_pages` | 0 |
| `bg_promote_committed_pages` | 0 |
| `bg_promote_cancelled_pages` | 0 |
| `remote_map_pages` | 24576 |
| `bg_promote_demand_pages_enqueued` | 0 |
| `bg_promote_prefetch_pages_enqueued` | 0 |
| `bg_promote_demand_pages_promoted` | 0 |
| `bg_promote_prefetch_pages_promoted` | 0 |
| `pte_switch_pages` | 0 |
| `bg_promote_worker_rounds` | 0 |
| `bg_promote_retry_count` | 0 |
| `bg_promote_failed_not_cpu_resident` | 0 |
| `bg_promote_failed_already_gpu_resident` | 0 |
| `bg_promote_failed_allocation` | 0 |
| `bg_promote_failed_copy` | 0 |
| `bg_promote_failed_unmap` | 0 |
| `bg_promote_failed_map` | 0 |
| `bg_promote_failed_block_dead` | 0 |
| `sync_fallback_pages` | 288 |

## Per-Case Counter Delta

| Case | Profile | RF | BG | Ops/page | RF faults | RF pages | Remote map pages | BG enqueued | Demand enq | Prefetch enq | BG copied | BG committed | PTE switch | BG cancelled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default_uvm` | `default_uvm` | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `counter_migrate` | `counter_migrate` | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `remote_first_only` | `remote_first_only` | 1 | 0 | 512 | 4025 | 24576 | 24576 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `readonly_dup_sync` | `readonly_dup_sync` | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Per-Case Worker And Failure Detail

| Case | Demand promoted | Prefetch promoted | Worker rounds | Retries | Not CPU resident | Already GPU resident | Alloc fail | Copy fail | Unmap fail | Map fail | Dead block | Sync fallback pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default_uvm` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `counter_migrate` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `remote_first_only` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 288 |
| `readonly_dup_sync` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Performance

| Profile | Runs | Ops/page | Pass0 ms | Last-pass ms | Median ms | Last-pass GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| `default_uvm` | 3 | 512 | 13.377 | 0.120 | 6.686 | 261.043 |
| `counter_migrate` | 3 | 512 | 11.727 | 10.868 | 11.273 | 2.875 |
| `remote_first_only` | 3 | 512 | 34.523 | 10.862 | 19.955 | 2.877 |
| `readonly_dup_sync` | 3 | 512 | 6.588 | 0.117 | 3.349 | 267.416 |

## Interpretation

- `remote_first_only` disables background promotion to isolate foreground remote-first latency.
  Observed: remote-first counters increased while background counters stayed zero.
- `counter_migrate` uses CPU preferred + AccessedBy with `512` ops/page to test access-counter driven promotion/migration behavior.
