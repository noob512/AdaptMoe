# Remote-First Correctness Probe

- Raw CSV: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_correctness_20260610_103347/correctness_raw.csv`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_correctness_20260610_103347/metadata.txt`
- Counter delta: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_correctness_20260610_103347/latency_stats_delta.txt`

## Correctness

| Test | Status | Observed | Expected | Wall ms |
|---|---|---:|---:|---:|
| `gpu_read_pass_0` | PASS | 16776431105097695232 | 16776431105097695232 | 5.311 |
| `gpu_read_pass_1` | PASS | 16776431105097695232 | 16776431105097695232 | 0.921 |
| `gpu_read_pass_2` | PASS | 16776431105097695232 | 16776431105097695232 | 0.897 |
| `gpu_read_pass_3` | PASS | 16776431105097695232 | 16776431105097695232 | 0.923 |
| `cpu_read_after_promotion` | PASS | 16776431105097695232 | 16776431105097695232 | 0.062 |
| `cpu_write_then_gpu_read` | PASS | 9324542017086169088 | 9324542017086169088 | 7.382 |
| `gpu_write_then_cpu_read` | PASS | 66730090759905280 | 66730090759905280 | 8.860 |
| `multi_stream_reads` | PASS | 66730090759905280 | 66730090759905280 | 3.875 |

## Counter Delta

| Counter | Delta |
|---|---:|
| `remote_first_faults` | 88 |
| `remote_first_pages` | 834 |
| `bg_promote_enqueued_pages` | 4608 |
| `bg_promote_copied_pages` | 4608 |
| `bg_promote_committed_pages` | 4608 |
| `bg_promote_cancelled_pages` | 0 |
| `remote_map_pages` | 834 |
| `bg_promote_demand_pages_enqueued` | 811 |
| `bg_promote_prefetch_pages_enqueued` | 3797 |
| `bg_promote_demand_pages_promoted` | 811 |
| `bg_promote_prefetch_pages_promoted` | 3797 |
| `pte_switch_pages` | 4608 |
| `bg_promote_worker_rounds` | 83 |
| `bg_promote_retry_count` | 0 |
| `bg_promote_failed_not_cpu_resident` | 0 |
| `bg_promote_failed_already_gpu_resident` | 0 |
| `bg_promote_failed_allocation` | 0 |
| `bg_promote_failed_copy` | 0 |
| `bg_promote_failed_unmap` | 0 |
| `bg_promote_failed_map` | 0 |
| `bg_promote_failed_block_dead` | 0 |
| `sync_fallback_pages` | 1078 |

## Interpretation

- All correctness checks passed for GPU read, CPU read, CPU-write break, GPU-write fallback, and multi-stream read.
- Remote-first fault handling was exercised.
- Background read-dup promotion committed pages.
- PTE switch to local GPU read mapping was observed.
