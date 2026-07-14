# 512KiB 128-Expert Cold8 Selective Restore Smoke

Date: 2026-07-13

This smoke does not start vLLM. It validates the current 512KiB logical /
64KiB physical mechanism with a synthetic MoE layer:

- 128 managed expert regions.
- Each expert is 512KiB.
- 8 cold experts are released to CPU.
- 8 reserved slots are preallocated.
- Only 3 selected cold experts are restored before GPU access.
- Non-cold experts and cold-but-not-selected experts are checked for
  non-interference.

## Files

- `workloads/vllm/check_128expert_cold8_selective_restore.py`
- `workloads/vllm/run_128expert_cold8_selective_restore.sh`

## Command

```bash
RUN_ROOT=/tmp/uvm_128expert_cold8_selective_restore_codex_20260713_193800 \
workloads/vllm/run_128expert_cold8_selective_restore.sh
```

Result JSON:

```text
/tmp/uvm_128expert_cold8_selective_restore_codex_20260713_193800/128expert_cold8_selective_restore_result.json
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

Synthetic experts:

```text
expert_count = 128
expert_bytes = 524288
total_bytes  = 67108864
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
whole gpu_bytes = 67108864
cold experts gpu_bytes = 4194304
non-cold experts gpu_bytes = 62914560
```

After releasing 8 cold experts:

```text
whole gpu_bytes = 62914560
whole cpu_bytes = 4194304
cold experts gpu_bytes = 0
cold experts cpu_bytes = 4194304
non-cold experts gpu_bytes = 62914560
```

Each cold expert release:

```text
attempted_bytes = 524288
released_bytes  = 524288
success_blocks  = 1
failure_blocks  = 0
rmStatus        = 0
```

Reserved slot prealloc:

```text
slot_count = 8
slot_bytes = 524288
chunk_size = 65536
chunk_count per slot = 8
success_count = 8
failure_count = 0
reserved_backing_bytes = 4194304
logical_slot_bytes = 4194304
alloc_failed_count = 0
rmStatus = 0
```

Selective restore:

```text
restored experts = 17, 63, 111
restore calls = 3
success_count per restore = 1
accepted_bytes per restore = 524288
assigned_slots per restore = 1
total accepted_bytes = 1572864
```

Payload checks:

```text
cold CPU payload after release = true for all 8
selected GPU read after restore = true for 17, 63, 111
non-cold GPU read payload = true for all 120
```

Non-interference checks after selected reads:

```text
cold-not-selected gpu_bytes = 0
cold-not-selected cpu_bytes = 2621440
non-cold gpu_bytes = 62914560
```

## Interpretation

The current mechanism works in this expert-level synthetic setting:

1. A 512KiB expert can be released without releasing the surrounding 1.5MiB in
   the same 2MiB group.
2. The released 512KiB regions are backed by CPU-resident payloads.
3. Reserved slots use 8 * 64KiB PMM chunks per 512KiB logical slot.
4. Only selected cold experts need to be restored before GPU access.
5. Cold experts that are not selected remain CPU-resident and do not fault back
   during selected expert access.
6. Non-cold experts remain GPU-resident and their payloads are unchanged.

Note: restored experts may still report `gpu_bytes = 0` in ordinary residency
queries because the restore path installs readonly slot-backed PTEs rather than
ordinary UVM residency. Correctness is verified by the reserved prefetch
counters and GPU readback from the original expert VA.
