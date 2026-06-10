# Remote-First Background Promotion Counter Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_256/probe_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_256/metadata.txt`
- Counter before: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_256/latency_stats_before.txt`
- Counter after: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_256/latency_stats_after.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_256/latency_stats_delta.txt`
- Per-case counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/bg_max_256/counter_delta_by_case.csv`

## Total Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 666 |
| `remote_first_pages` | 5171 |
| `bg_promote_enqueued_pages` | 32768 |
| `bg_promote_copied_pages` | 32768 |
| `bg_promote_committed_pages` | 32768 |
| `bg_promote_cancelled_pages` | 0 |
| `remote_map_pages` | 5171 |
| `bg_promote_demand_pages_enqueued` | 5095 |
| `bg_promote_prefetch_pages_enqueued` | 27673 |
| `bg_promote_demand_pages_promoted` | 5095 |
| `bg_promote_prefetch_pages_promoted` | 27673 |
| `pte_switch_pages` | 32768 |
| `bg_promote_worker_rounds` | 645 |
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
| `remote_first_bg` | `remote_first_bg` | 1 | 1 | 128 | 666 | 5171 | 5171 | 32768 | 5095 | 27673 | 32768 | 32768 | 32768 | 0 |

## Per-Case Worker And Failure Detail

| Case | Demand promoted | Prefetch promoted | Worker rounds | Retries | Not CPU resident | Already GPU resident | Alloc fail | Copy fail | Unmap fail | Map fail | Dead block | Sync fallback pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | 5095 | 27673 | 645 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 384 |

## Performance

| Profile | Runs | Ops/page | Pass0 ms | Last-pass ms | Median ms | Last-pass GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | 2 | 128 | 11.036 | 0.044 | 0.045 | 357.197 |

## Interpretation

- `remote_first_bg` should exercise the new full path: remote-first fault service plus background COPY/read-dup promotion.
  Observed: both remote-first and committed background promotion counters increased.
