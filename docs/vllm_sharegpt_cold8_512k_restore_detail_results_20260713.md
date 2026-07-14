# vLLM ShareGPT Cold8 512KiB Restore Detail Results

Date: 2026-07-13

## Goal

重新运行 512KiB static cold8 vLLM 机制实验，并详细记录哪些 offloaded cold experts 在 runtime 被访问，从而触发 reserved-slot restore。

## Command

```bash
RUN_ROOT=/tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902 \
NUM_REQUESTS=32 \
OUTPUT_TOKENS=64 \
TIMEOUT_SEC=3000 \
RESERVED_SLOT_COUNT=128 \
workloads/vllm/run_cold8_512k_static_reserved_experiment.sh
```

分析命令：

```bash
python3 workloads/vllm/analyze_cold8_reserved_restores.py \
  --run-root /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902
```

## Run Result

```text
run_root = /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902
mechanism_returncode = 0
failure_reason = None
completed_requests = 32/32
output_tokens = 2048
total_generate_seconds = 182.093
tokens/s = 11.247
avg_request_seconds = 5.690
new_xid_or_mmu_fault = false
```

## Offload And Slot State

Stage-M release:

```text
offload_plan_records = 384
issued_records = 768
issued_bytes = 1811939328
driver_release_entries = 3456
driver_release_failed_records = 0
driver_release_released_bytes = 1755316224
cold_gpu_bytes_before_stage_m = 1755316224
cold_gpu_bytes_after_stage_m_clear = 0
cold_cpu_bytes_after_stage_m_clear = 1811939328
```

Reserved slot prealloc:

```text
slot_count = 128
slot_bytes = 524288
chunk_size = 65536
success_count = 128
failure_count = 0
reserved_backing_bytes = 67108864
rm_status = 0
```

## Restore Summary

Runtime restore:

```text
cold_plan_experts = 384
touched_cold_experts = 314
untouched_cold_experts = 70
restore_entry_count = 20610
restore_event_rows = 4580
restore_bytes = 10805575680
restore_mib = 10305.0
ioctl_records = 1636
ioctl_success_entries = 20610
ioctl_failure_entries = 0
unique_restore_steps = 521
unique_restore_requests = 32
```

Interpretation:

- `restore_entry_count` counts 512KiB restore entries.
- One full `moe_gate_up` restore is 6 entries.
- One full `moe_down` restore is 3 entries.
- One full expert role pair therefore costs 9 entries, or 4.5MiB, each time it is restored.
- The same expert can be restored multiple times because slots are cleaned/reused across steps.

## Top Restored Experts

| rank | layer | expert | pilot hits | entries | MiB | steps | requests | gate_up entries | down entries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 51 | 158 | 621 | 310.5 | 69 | 15 | 414 | 207 |
| 2 | 1 | 22 | 65 | 378 | 189.0 | 42 | 15 | 252 | 126 |
| 3 | 12 | 113 | 58 | 324 | 162.0 | 36 | 11 | 216 | 108 |
| 4 | 9 | 108 | 38 | 315 | 157.5 | 35 | 5 | 210 | 105 |
| 5 | 13 | 111 | 45 | 315 | 157.5 | 35 | 7 | 210 | 105 |
| 6 | 1 | 55 | 41 | 297 | 148.5 | 33 | 32 | 198 | 99 |
| 7 | 6 | 15 | 34 | 297 | 148.5 | 33 | 32 | 198 | 99 |
| 8 | 10 | 38 | 75 | 297 | 148.5 | 33 | 32 | 198 | 99 |
| 9 | 1 | 0 | 40 | 288 | 144.0 | 32 | 31 | 192 | 96 |
| 10 | 9 | 46 | 49 | 288 | 144.0 | 32 | 31 | 192 | 96 |

These are the main cold experts responsible for repeated restore traffic. In particular, layer 1 dominates the restore cost.

## Top Requests By Restore Volume

| request | restore MiB | generate seconds |
|---:|---:|---:|
| 28 | 1147.5 | 7.825 |
| 15 | 1044.0 | 7.665 |
| 6 | 864.0 | 7.145 |
| 9 | 778.5 | 6.984 |
| 21 | 711.0 | 6.685 |
| 17 | 684.0 | 6.702 |
| 18 | 607.5 | 6.503 |
| 1 | 499.5 | 6.469 |
| 31 | 472.5 | 6.002 |
| 19 | 292.5 | 5.680 |

The slow requests are also the requests with the most restore traffic, which supports the conclusion that runtime restore volume is the dominant source of slowdown.

## Top Layers By Restore Volume

| layer | restore MiB |
|---:|---:|
| 1 | 1107.0 |
| 9 | 477.0 |
| 10 | 472.5 |
| 12 | 378.0 |
| 6 | 355.5 |
| 13 | 346.5 |
| 3 | 328.5 |
| 15 | 315.0 |
| 22 | 315.0 |
| 27 | 310.5 |

## Output Files

```text
analysis report:
  /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902/cold8_restore_analysis/cold8_restore_report.md

by expert:
  /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902/cold8_restore_analysis/cold8_restore_by_expert.csv

by restore event:
  /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902/cold8_restore_analysis/cold8_restore_events.csv

by request:
  /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902/cold8_restore_analysis/cold8_restore_by_request.csv

by layer:
  /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902/cold8_restore_analysis/cold8_restore_by_layer.csv

summary:
  /tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902/cold8_restore_analysis/cold8_restore_summary.json
```

## Conclusion

This rerun confirms that the mechanism is correct but explains why performance drops:

- 70 planned cold experts were never accessed and did not trigger restore.
- 314 planned cold experts were accessed at least once.
- The run performed 20610 successful 512KiB restores, totaling about 10.3GiB of restore traffic.
- Restore failures were zero.
- Request-level latency tracks restore volume: the heaviest restore requests are also the slowest requests.

So the performance issue is not caused by offloading itself. It is caused by a subset of "cold" experts that are still accessed often enough to create repeated 512KiB restore traffic under the current slot cleanup/reuse policy.
