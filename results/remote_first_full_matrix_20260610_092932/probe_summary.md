# Remote-First Background Promotion Counter Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_092932/probe_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_092932/metadata.txt`
- Counter before: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_092932/latency_stats_before.txt`
- Counter after: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_092932/latency_stats_after.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_092932/latency_stats_delta.txt`
- Per-case counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_full_matrix_20260610_092932/counter_delta_by_case.csv`

## Total Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 6711 |
| `remote_first_pages` | 35656 |
| `bg_promote_enqueued_pages` | 32768 |
| `bg_promote_copied_pages` | 32768 |
| `bg_promote_committed_pages` | 32768 |
| `bg_promote_cancelled_pages` | 0 |

## Per-Case Counter Delta

| Case | Profile | RF | BG | Ops/page | RF faults | RF pages | BG enqueued | BG copied | BG committed | BG cancelled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | `remote_first_bg` | 1 | 1 | 128 | 341 | 2888 | 32768 | 32768 | 32768 | 0 |
| `remote_first_only` | `remote_first_only` | 1 | 0 | 128 | 6370 | 32768 | 0 | 0 | 0 | 0 |
| `readonly_dup_sync` | `readonly_dup_sync` | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |
| `counter_migrate` | `counter_migrate` | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 0 | 0 |
| `local_prefetch` | `local_prefetch` | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |
| `default_uvm` | `default_uvm` | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |

## Performance

| Profile | Runs | Ops/page | Pass0 ms | Last-pass ms | Median ms | Last-pass GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| `remote_first_bg` | 2 | 128 | 7.253 | 0.044 | 0.046 | 357.927 |
| `remote_first_only` | 2 | 128 | 56.731 | 3.790 | 5.522 | 4.535 |
| `readonly_dup_sync` | 2 | 128 | 14.479 | 0.042 | 0.043 | 369.209 |
| `counter_migrate` | 2 | 512 | 22.550 | 10.531 | 10.537 | 5.935 |
| `local_prefetch` | 2 | 128 | 0.122 | 0.044 | 0.044 | 357.231 |
| `default_uvm` | 2 | 128 | 28.539 | 0.043 | 0.044 | 364.094 |

## Interpretation

- `remote_first_bg` should exercise the new full path: remote-first fault service plus background COPY/read-dup promotion.
  Observed: both remote-first and committed background promotion counters increased.
- `remote_first_only` disables background promotion to isolate foreground remote-first latency.
  Observed: remote-first counters increased while background counters stayed zero.
- Native read-dup first pass vs remote-first+BG first pass: `14.479 ms` vs `7.253 ms`.
- Remote-first only first pass vs remote-first+BG first pass: `56.731 ms` vs `7.253 ms`.
- `counter_migrate` uses CPU preferred + AccessedBy with `512` ops/page to test access-counter driven promotion/migration behavior.
- Default UVM first pass vs remote-first+BG first pass: `28.539 ms` vs `7.253 ms`.
- Local prefetch last pass vs remote-first+BG last pass: `0.044 ms` vs `0.044 ms`.
