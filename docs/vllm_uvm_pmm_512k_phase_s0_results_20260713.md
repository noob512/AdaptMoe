# Phase S0: 512KiB PMM Chunk Viability

## Goal

Confirm whether 512KiB is a legal user PMM chunk size in the currently loaded
UVM module, and whether the expert reserved slot pool can preallocate:

```text
48 slots * 512KiB = 24MiB
```

with exactly 24MiB of physical backing.

## Script

```bash
RUN_ROOT=/tmp/uvm_pmm_512k_phase_s0_codex_20260713_171500 \
workloads/vllm/run_uvm_pmm_512k_phase_s0.sh --allow-fail
```

Result JSON:

```text
/tmp/uvm_pmm_512k_phase_s0_codex_20260713_171500/phase_s0_512k_result.json
```

The script performs:

1. CUDA/UVM initialization and a 2MiB managed allocation.
2. GPU migration of that allocation so the current UVM VA space owns the GPU.
3. `UVM_TEST_PMM_QUERY` for `gpu->pmm.chunk_sizes[USER]`.
4. Expert reserved pool reset.
5. Prealloc of 48 logical 512KiB slots.
6. Reserved pool reset.

## Result

Overall S0 status:

```text
passed = false
```

PMM user chunk sizes:

```text
mask      = 0x211000
sizes     = [4096, 65536, 2097152]
has_512k = false
has_2m   = true
```

So the current user PMM chunk mask contains:

```text
4KiB, 64KiB, 2MiB
```

It does not contain:

```text
256KiB, 512KiB
```

Reserved slot prealloc:

```text
slot_count = 48
slot_bytes = 524288
prealloc_success = true
logical_slot_bytes = 25165824
reserved_backing_bytes = 100663296
chunk_size = 2097152
first entry:
  slot_bytes     = 524288
  physical_bytes = 2097152
  chunk_count    = 1
```

This means the current reserved pool accepts `slot_bytes=512KiB` as a logical
slot size, but still allocates one 2MiB root chunk per slot:

```text
48 * 2MiB = 96MiB
```

instead of the expected:

```text
48 * 512KiB = 24MiB
```

## Failed Checks

```text
pmm_has_512k = false
split_2m_to_512k_legal = false
reserved_backing_is_24m = false
entries_are_single_512k_chunks = false
```

The prealloc itself succeeds, but it is not a true 512KiB backing allocation.

## Code Pointers

PMM supports split/merge internally, but split targets must exist in
`pmm->chunk_sizes[type]`:

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_pmm_gpu.c
  uvm_pmm_gpu_split_chunk()
```

On this GPU, user PMM chunk sizes are initialized from MMU page sizes plus
`PAGE_SIZE`, masked by `UVM_CHUNK_SIZES_MASK`:

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_mmu.c
  uvm_mmu_init_gpu_chunk_sizes()
```

The current expert reserved pool prealloc path rounds physical backing up to
2MiB:

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
  uvm_expert_reserved_pool_prealloc_entry()
  physical_bytes = UVM_ALIGN_UP(entry->slot_bytes, UVM_CHUNK_SIZE_MAX)
  uvm_pmm_gpu_alloc_user(..., UVM_CHUNK_SIZE_MAX, ...)
```

## Conclusion

512KiB is a good logical size for Qwen3 MoE expert-role packing arithmetic:

```text
moe_down   ~= 1.5MiB = 3 * 512KiB
moe_gateup ~= 3.0MiB = 6 * 512KiB
```

But it is not a good current PMM chunk size on this system, because the loaded
UVM user PMM mask does not include 512KiB. Using 512KiB directly would require
additional driver work:

```text
1. enable or emulate 512KiB backing chunks,
2. update reserved pool allocation away from fixed 2MiB root chunks,
3. update slot offset-to-chunk mapping, which currently assumes 2MiB chunks.
```

The more natural next design is to keep 512KiB as the logical envelope size, but
implement it as groups of supported 64KiB PMM chunks:

```text
512KiB logical envelope = 8 * 64KiB physical PMM chunks
```

That avoids adding a non-native 512KiB PMM chunk size while still reducing the
collateral damage of the current 2MiB envelope.
