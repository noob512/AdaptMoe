# vLLM UVM 512KiB Logical / 64KiB Physical Phase S1 Results

Date: 2026-07-13

## Goal

Phase S1 only changes the expert reserved slot pool prealloc path. It does not
change expert offload/release behavior yet.

Target behavior:

```text
logical slot size: 512KiB
physical PMM chunk size: 64KiB
chunks per slot: 8
48 slots total logical/backing bytes: 24MiB
```

## Implementation

Changed files:

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`
- `workloads/vllm/check_uvm_reserved_512k_64k_phase_s1.py`
- `workloads/vllm/run_uvm_reserved_512k_64k_phase_s1.sh`

Kernel-side changes:

- `uvm_expert_reserved_pool_slot_t` now stores `chunk_size`.
- `driver_expert_reserved_pool_prealloc()` now chooses 64KiB PMM chunks for
  logical slots up to 512KiB when the GPU user PMM supports 64KiB chunks.
- For a 512KiB slot, prealloc now calls:

```text
uvm_pmm_gpu_alloc_user(..., chunk_count=8, chunk_size=64KiB, ...)
```

- Reserved slot summary reports the common active backing chunk size in
  `params->chunk_size`.
- Reserved restore/copy mapping now passes `slot->chunk_size` into
  `uvm_va_block_copy_and_map_reserved_readonly()`.
- `reserved_slot_chunk_from_offset()` no longer assumes 2MiB chunks. It computes:

```text
chunk_index  = slot_offset / slot->chunk_size
chunk_offset = slot_offset % slot->chunk_size
```

User-space validation changes:

- Added a Phase S1 checker that preallocates 48 slots of 512KiB each.
- The checker validates summary bytes, per-entry logical/physical bytes, and
  per-entry `chunk_count`.
- `UVM_TEST_PMM_QUERY` is treated as diagnostic. If that query ioctl is
  unavailable, the test still passes when reserved prealloc itself returns
  `chunk_size=65536`.

## Build

Command:

```bash
make -C kernel-module/nvidia-module/kernel-open NV_KERNEL_MODULES='nvidia nvidia-uvm' modules
```

Result:

```text
build: PASS
built nvidia-uvm.ko srcversion: 10A91A1592569767291AE2F
loaded nvidia_uvm srcversion: 10A91A1592569767291AE2F
```

Because the loaded module already had the same `srcversion`, no reload was
needed for the final S1 validation run.

## Validation

Command:

```bash
RUN_ROOT=/tmp/uvm_reserved_512k_64k_phase_s1_codex_20260713_182900 \
workloads/vllm/run_uvm_reserved_512k_64k_phase_s1.sh
```

Result file:

```text
/tmp/uvm_reserved_512k_64k_phase_s1_codex_20260713_182900/phase_s1_512k_64k_result.json
```

Overall:

```text
passed = true
```

Checks:

```text
prealloc_success = true
summary_chunk_size_is_64k = true
logical_bytes_is_24m = true
reserved_backing_is_24m = true
entry_count_matches = true
entries_are_512k_slots_8x64k = true
pmm_has_or_prealloc_uses_64k = true
```

Prealloc summary:

```text
slot_count = 48
slot_bytes = 524288
success_count = 48
failure_count = 0
chunk_size = 65536
logical_slot_bytes = 25165824
reserved_backing_bytes = 25165824
active_slots = 48
backed_slots = 48
alloc_failed_count = 0
rmStatus = 0
```

Per-entry validation:

```text
slot_bytes = 524288
physical_bytes = 524288
chunk_count = 8
status = 0
```

After reset:

```text
active_slots = 0
backed_slots = 0
reserved_backing_bytes = 0
rmStatus = 0
```

## Notes

`UVM_TEST_PMM_QUERY` returned `errno=38` in this loaded-module state, so it did
not report the PMM mask directly. This does not block S1, because the reserved
pool prealloc ioctl itself successfully allocated 48 slots with
`chunk_size=65536`, and every slot returned `chunk_count=8`.

Phase S1 therefore proves the reserved pool can allocate 512KiB logical slots
using 8 physical 64KiB PMM chunks per slot. Expert release/restore at this
granularity remains the next phase.
