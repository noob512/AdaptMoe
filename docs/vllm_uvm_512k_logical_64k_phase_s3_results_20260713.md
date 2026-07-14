# vLLM UVM 512KiB Logical / 64KiB Physical Phase S3 Results

Date: 2026-07-13

## Goal

Phase S3 validates the full 512KiB restore path:

```text
2MiB managed allocation
-> release one 512KiB target range to CPU and free its 8x64KiB GPU chunks
-> prealloc one 512KiB reserved slot backed by 8x64KiB PMM chunks
-> copy CPU-resident target pages into that reserved slot
-> install readonly slot-backed PTEs at the original target VA
-> read original VA on GPU and verify payload
```

## Implementation

Changed files:

- `workloads/vllm/check_512k_reserved_restore_phase_s3.py`
- `workloads/vllm/run_512k_reserved_restore_phase_s3.sh`

No new kernel changes were needed for Phase S3. It uses:

- Phase S1 reserved slot support for `512KiB logical / 64KiB physical`
- Phase S2 `UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K`
- existing reserved prefetch path, now able to map slots using `slot->chunk_size`

## Validation

Command:

```bash
RUN_ROOT=/tmp/uvm_512k_reserved_restore_phase_s3_codex_20260713_194500 \
workloads/vllm/run_512k_reserved_restore_phase_s3.sh
```

Result file:

```text
/tmp/uvm_512k_reserved_restore_phase_s3_codex_20260713_194500/phase_s3_512k_reserved_restore_result.json
```

Loaded module:

```text
nvidia_uvm srcversion = 386A1FA2E18DDDFCE9EFE88
```

Overall:

```text
passed = true
```

## Key Results

Release step:

```text
attempted_bytes = 524288
released_bytes = 524288
success_blocks = 1
failure_blocks = 0
rmStatus = 0
```

After release:

```text
target 512KiB:
  gpu_bytes = 0
  cpu_bytes = 524288

whole 2MiB:
  gpu_bytes = 1572864
  cpu_bytes = 524288
```

Reserved slot prealloc:

```text
success_count = 1
failure_count = 0
slot_bytes = 524288
physical_bytes = 524288
chunk_size = 65536
chunk_count = 8
logical_slot_bytes = 524288
reserved_backing_bytes = 524288
rmStatus = 0
```

Reserved prefetch restore:

```text
success_count = 1
failure_count = 0
requested_bytes = 524288
accepted_bytes = 524288
assigned_slots = 1
resolved_bytes = 524288
physical_bytes = 524288
entry_status = 0
rmStatus = 0
```

Readback checks:

```text
target_payload_restored = true
left_neighbor_payload_unchanged = true
right_neighbor_payload_unchanged = true
```

Cleanup:

```text
invalidate_ok = true
reset_after_ok = true
```

## Important Note

After restore, residency query still reports:

```text
target gpu_bytes = 0
target cpu_bytes = 524288
```

This is expected for the current reserved-slot path. The target VA is mapped
with readonly slot-backed PTEs, but those slot-backed PTEs are not counted as
ordinary GPU residency in the existing residency query. The correctness signal
is therefore:

```text
reserved prefetch accepted_bytes = 524288
assigned_slots = 1
GPU read from original target VA returns the expected payload
```

## Conclusion

Phase S3 confirms the 512KiB closed loop:

```text
release 512KiB target from ordinary GPU residency
-> free 8x64KiB PMM chunks
-> allocate a 512KiB reserved slot backed by 8x64KiB chunks
-> restore CPU-resident target data into the slot
-> GPU reads the original VA correctly
```

Neighboring data remains intact, so the 512KiB route avoids the 2MiB envelope
collateral restore cost tested in the earlier path.
