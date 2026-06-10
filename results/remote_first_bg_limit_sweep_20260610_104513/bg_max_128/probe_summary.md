# Remote-First Background Promotion Counter Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_128/probe_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_128/metadata.txt`
- Counter before: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_128/latency_stats_before.txt`
- Counter after: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_128/latency_stats_after.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_128/latency_stats_delta.txt`
- Per-case counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_128/counter_delta_by_case.csv`

## Total Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 651 |
| `remote_first_pages` | 5592 |
| `bg_promote_enqueued_pages` | 32768 |
| `bg_promote_copied_pages` | 32768 |
| `bg_promote_committed_pages` | 32768 |
| `bg_promote_cancelled_pages` | 0 |
| `remote_map_pages` | 5592 |
| `bg_promote_demand_pages_enqueued` | 5560 |
| `bg_promote_prefetch_pages_enqueued` | 27208 |
| `bg_promote_demand_pages_promoted` | 5560 |
| `bg_promote_prefetch_pages_promoted` | 27208 |
| `pte_switch_pages` | 32768 |
| `bg_promote_worker_rounds` | 713 |
| `bg_promote_retry_count` | 0 |
| `bg_promote_failed_not_cpu_resident` | 0 |
| `bg_promote_failed_already_gpu_resident` | 0 |
| `bg_promote_failed_allocation` | 0 |
| `bg_promote_failed_copy` | 0 |
| `bg_promote_failed_unmap` | 0 |
| `bg_promote_failed_map` | 0 |
| `bg_promote_failed_block_dead` | 0 |
| `sync_fallback_pages` | 384 |

## Per-Case Counter Delta

| Case | Profile | RF | BG | Ops/page | RF faults | RF pages | Remote map pages | BG enqueued | Demand enq | Prefetch enq | BG copied | BG committed | PTE switch | BG cancelled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | `remote_first_bg` | 1 | 1 | 128 | 651 | 5592 | 5592 | 32768 | 5560 | 27208 | 32768 | 32768 | 32768 | 0 |

## Per-Case Worker And Failure Detail

| Case | Demand promoted | Prefetch promoted | Worker rounds | Retries | Not CPU resident | Already GPU resident | Alloc fail | Copy fail | Unmap fail | Map fail | Dead block | Sync fallback pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | 5560 | 27208 | 713 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 384 |

## Performance

| Profile | Runs | Ops/page | Pass0 ms | Last-pass ms | Median ms | Last-pass GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | 2 | 128 | 13.239 | 0.043 | 0.044 | 359.771 |

## Interpretation

- `remote_first_bg` should exercise the new full path: remote-first fault service plus background COPY/read-dup promotion.
  Observed: both remote-first and committed background promotion counters increased.
