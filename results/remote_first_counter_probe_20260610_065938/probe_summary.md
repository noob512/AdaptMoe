# Remote-First Background Promotion Counter Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_counter_probe_20260610_065938/probe_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_counter_probe_20260610_065938/metadata.txt`
- Counter before: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_counter_probe_20260610_065938/latency_stats_before.txt`
- Counter after: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_counter_probe_20260610_065938/latency_stats_after.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_counter_probe_20260610_065938/latency_stats_delta.txt`

## Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 396 |
| `remote_first_pages` | 2942 |
| `bg_promote_enqueued_pages` | 32768 |
| `bg_promote_copied_pages` | 32768 |
| `bg_promote_committed_pages` | 32768 |
| `bg_promote_cancelled_pages` | 0 |

## Performance

| Profile | Runs | Pass0 ms | Last-pass ms | Median ms | Last-pass GiB/s |
|---|---:|---:|---:|---:|---:|
| `readmostly_fault` | 2 | 7.806 | 0.043 | 0.044 | 364.530 |
| `local_prefetch` | 2 | 0.108 | 0.042 | 0.043 | 374.339 |
| `default_uvm` | 2 | 28.024 | 0.043 | 0.045 | 359.982 |

## Interpretation

- `remote_first_*` increased, so the new remote-first fault path was exercised.
- `bg_promote_enqueued_pages` increased, so pages were queued for background promotion.
- `bg_promote_committed_pages` increased, so background COPY/read-dup promotion reached the local-PTE switch stage.
