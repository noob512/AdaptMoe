# 1.5MiB 128-Expert Cold8 Selective Restore Smoke

Date: 2026-07-13

This smoke verifies whether the 512KiB logical / 64KiB physical mechanism can
handle a logical expert whose size is 1.5MiB.

Important constraint: the current `UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K`
ioctl accepts exactly one 512KiB registered region per release. Therefore a
1.5MiB logical expert is represented as 3 independent 512KiB chunks:

```text
logical expert = 3 * 512KiB regions
release        = 3 release ioctls
restore        = 3 reserved-slot restores
slot backing   = 3 * (8 * 64KiB chunks)
```

The experiment does not start vLLM.

## Files

- `workloads/vllm/check_128expert_cold8_selective_restore.py`
- `workloads/vllm/run_128expert_cold8_selective_restore.sh`

The synthetic smoke now supports:

```text
--chunks-per-expert 3
```

## Command

```bash
RUN_ROOT=/tmp/uvm_128expert_cold8_1p5mb_selective_restore_codex_20260713_200210 \
workloads/vllm/run_128expert_cold8_selective_restore.sh \
  --chunks-per-expert 3
```

Result JSON:

```text
/tmp/uvm_128expert_cold8_1p5mb_selective_restore_codex_20260713_200210/128expert_cold8_selective_restore_result.json
```

Loaded `nvidia_uvm`:

```text
srcversion = 386A1FA2E18DDDFCE9EFE88
```

GPU state after the run:

```text
memory.used = 0 MiB
```

## Setup

```text
expert_count = 128
chunks_per_expert = 3
chunk_bytes = 524288
expert_bytes = 1572864
total_bytes = 201326592
```

Cold experts:

```text
3, 17, 31, 47, 63, 79, 95, 111
```

Selected restore/access experts:

```text
17, 63, 111
```

Cold-but-not-selected experts:

```text
3, 31, 47, 79, 95
```

## Result

Overall:

```text
passed = true
```

Initial residency after migrating all experts to GPU:

```text
whole gpu_bytes = 201326592
cold experts gpu_bytes = 12582912
non-cold experts gpu_bytes = 188743680
```

After releasing 8 cold experts:

```text
whole gpu_bytes = 188743680
whole cpu_bytes = 12582912
cold experts gpu_bytes = 0
cold experts cpu_bytes = 12582912
non-cold experts gpu_bytes = 188743680
```

Release granularity:

```text
cold experts = 8
chunks per expert = 3
release chunks = 24
attempted_bytes per chunk = 524288
released_bytes per chunk = 524288
failure_blocks = 0
rmStatus = 0
```

Reserved slot prealloc:

```text
slot_count = 24
slot_bytes = 524288
chunk_size = 65536
chunk_count per slot = 8
success_count = 24
failure_count = 0
reserved_backing_bytes = 12582912
logical_slot_bytes = 12582912
alloc_failed_count = 0
rmStatus = 0
```

Selective restore:

```text
restored logical experts = 17, 63, 111
restore chunks = 9
accepted_bytes per restore = 524288
total accepted_bytes = 4718592
assigned_slots per restore = 1
failed restores = 0
```

Payload checks:

```text
cold CPU payload after release = true for all 8 logical experts
selected GPU read after restore = true for 17, 63, 111
non-cold GPU read payload = true for all 120 logical experts
```

Non-interference checks after selected reads:

```text
cold-not-selected gpu_bytes = 0
cold-not-selected cpu_bytes = 7864320
non-cold gpu_bytes = 188743680
```

## Conclusion

Yes, a 1.5MiB expert works with the current mechanism when it is managed as
three 512KiB chunks. The current driver path does not support releasing one
registered 1.5MiB region directly with `SUBCHUNK_64K`; it must be split into
512KiB release/restore units.

This is still compatible with a logical expert abstraction: the runtime can
treat an expert as one object while the UVM controller stores three backing
regions and restores all three before the expert is accessed.
