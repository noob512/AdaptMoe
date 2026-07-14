# vLLM UVM 512KiB Logical / 64KiB Physical Phase S2 Results

Date: 2026-07-13

## Goal

Phase S2 validates targeted release of one 512KiB logical range from a 2MiB
managed allocation. The released range is backed by 8 physical 64KiB PMM
chunks.

Expected residency transition:

```text
before release:
  full 2MiB block GPU resident

after release:
  target 512KiB: GPU resident = 0, CPU resident = 512KiB
  neighbor 1.5MiB: still GPU resident
```

## Implementation

Changed files:

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`
- `workloads/vllm/vllm/vllm/device_allocator/uvm.py`
- `workloads/vllm/check_512k_subrange_release_phase_s2.py`
- `workloads/vllm/run_512k_subrange_release_phase_s2.sh`

New release mode:

```text
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K
```

The new mode requires:

```text
region length = 512KiB
region base   = 512KiB aligned
physical chunk size = 64KiB
```

Important implementation detail:

The existing UVM full-size 2MiB VA block indexing path treats a full 2MiB block
as one 2MiB GPU chunk. To keep the VA block metadata consistent, the release
ioctl first isolates the registered 512KiB region with the existing
`uvm_va_range_split()` path. This preserves the user VA address while allowing
the isolated 512KiB VA block to be represented as 8 64KiB GPU chunks.

Release helper behavior:

```text
1. Make the target VA block region CPU resident.
2. Verify the target region is no longer GPU resident.
3. Verify target GPU PTE bits and slot-backed PTE bits are clear.
4. Require every target chunk to be exactly 64KiB and fully covered.
5. Unmap/free only those target chunks via normal PMM free path.
6. Leave neighboring VA blocks/chunks untouched.
```

This avoids manually attaching chunks to a free list. The chunks are returned
through `uvm_pmm_gpu_free()`.

## Build

Command:

```bash
make -C kernel-module/nvidia-module/kernel-open NV_KERNEL_MODULES='nvidia nvidia-uvm' modules
```

Result:

```text
build: PASS
built nvidia-uvm.ko srcversion: 386A1FA2E18DDDFCE9EFE88
loaded nvidia_uvm srcversion: 386A1FA2E18DDDFCE9EFE88
```

The module was reloaded before validation.

## Validation

Command:

```bash
RUN_ROOT=/tmp/uvm_512k_subrange_release_phase_s2_codex_20260713_193000 \
workloads/vllm/run_512k_subrange_release_phase_s2.sh
```

Result file:

```text
/tmp/uvm_512k_subrange_release_phase_s2_codex_20260713_193000/phase_s2_512k_subrange_release_result.json
```

Overall:

```text
passed = true
```

Release ioctl:

```text
ok = true
block_count = 1
attempted_blocks = 1
success_blocks = 1
failure_blocks = 0
attempted_bytes = 524288
released_bytes = 524288
rmStatus = 0
```

Before release:

```text
whole 2MiB gpu_bytes = 2097152
target 512KiB gpu_bytes = 524288
left neighbor gpu_bytes = 524288
right neighbor gpu_bytes = 1048576
```

After release:

```text
whole range gpu_bytes = 1572864
whole range cpu_bytes = 524288

target 512KiB:
  gpu_bytes = 0
  cpu_bytes = 524288

left neighbor:
  gpu_bytes = 524288
  cpu_bytes = 0

right neighbor:
  gpu_bytes = 1048576
  cpu_bytes = 0
```

Payload checks:

```text
gpu_read_before_matches_payload = true
cpu_read_after_release_matches_target_payload = true
gpu_read_after_release_left_neighbor_matches = true
gpu_read_after_release_right_neighbor_matches = true
target_still_gpu_zero_after_neighbor_reads = true
```

## Conclusion

Phase S2 confirms that the driver can release one 512KiB logical region as
8 physical 64KiB PMM chunks while preserving neighboring GPU residency.

This is different from the previous 2MiB envelope path:

```text
old path:
  release unit = 2MiB root chunk
  collateral neighbor data must be restored later

S2 path:
  release unit = 512KiB logical range
  physical free unit = 8 * 64KiB chunks
  neighboring 1.5MiB remains GPU resident
```

The next step is Phase S3: prealloc a 512KiB reserved slot and restore the
released 512KiB from CPU-resident pages back to the original VA.
