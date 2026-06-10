# Remote-First Correctness Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_correctness_smoke_20260610_101043/correctness_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_correctness_smoke_20260610_101043/metadata.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_correctness_smoke_20260610_101043/latency_stats_delta.txt`

## Correctness

| Test | Status | Observed | Expected | Wall ms |
|---|---|---:|---:|---:|
| `gpu_read_pass_0` | PASS | 7871258923749971456 | 7871258923749971456 | 2.742 |
| `gpu_read_pass_1` | PASS | 7871258923749971456 | 7871258923749971456 | 0.832 |
| `cpu_read_after_promotion` | PASS | 7871258923749971456 | 7871258923749971456 | 0.005 |
| `cpu_write_then_gpu_read` | PASS | 3772391028521199360 | 3772391028521199360 | 1.859 |
| `gpu_write_then_cpu_read` | PASS | 8233725405351542016 | 8233725405351542016 | 0.959 |
| `multi_stream_reads` | PASS | 8233725405351542016 | 8233725405351542016 | 1.691 |

## Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 8 |
| `remote_first_pages` | 238 |
| `bg_promote_enqueued_pages` | 320 |
| `bg_promote_copied_pages` | 320 |
| `bg_promote_committed_pages` | 320 |
| `bg_promote_cancelled_pages` | 0 |
| `remote_map_pages` | 0 |
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
| `sync_fallback_pages` | 0 |

## Interpretation

- All correctness checks passed for GPU read, CPU read, CPU-write break, GPU-write fallback, and multi-stream read.
- Remote-first fault handling was exercised.
- Background read-dup promotion committed pages.
