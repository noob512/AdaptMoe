# vLLM cold8 2MiB envelope reserved-slot restore experiment

Date: 2026-07-13

## Goal

Validate the cold8 expert offload experiment with the new 2MiB envelope flow:

1. Register/offload/release cold expert weight regions as 2MiB-aligned envelopes.
2. Return released PMM chunks to reusable backing.
3. Preallocate reserved slots after Stage-M.
4. Restore selected cold experts by copying CPU-resident envelope pages into reserved slots and remapping the original VA.

## Scripts

Main script:

```bash
workloads/vllm/run_cold8_reserved_slot_failfast_experiment.sh
```

The script now defaults to:

```bash
RESERVED_SLOT_BYTES=2097152
RESERVED_PREFETCH_CPU_SOURCE=1
VLLM_UVM_EXPERT_2MB_ENVELOPE_ENABLE=1
VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release
VLLM_UVM_EXPERT_RESERVED_FAIL_FAST_ON_PREALLOC_FAILURE=1
```

Minimal 2MiB smoke:

```bash
workloads/vllm/run_2mb_offload_release_slot_restore.sh
```

Successful mechanism-only run:

```text
/tmp/vllm_cold8_2mb_envelope_mechonly_cpu_source_20260713_162238
```

Failed control run without CPU-source refresh:

```text
/tmp/vllm_cold8_2mb_envelope_mechonly_fix_20260713_161702
```

## Key implementation changes

- Runtime reserved prefetch now expands logical expert slices to 2MiB envelope regions.
- Logical cold expert slices use `driver_tag_skip` so they do not overlap with envelope driver ranges.
- Existing 2MiB envelope driver registrations are reused across Stage-M release and runtime restore.
- The reserved executor now returns per-entry driver results for debugging.
- The cold8 experiment script enables CPU-source refresh before reserved restore.

## Results

Minimal 2MiB smoke passed:

```text
release before: gpu_bytes=2097152 cpu_bytes=0
release after:  gpu_bytes=0       cpu_bytes=2097152
slot prealloc:  success_count=1 reserved_backing_bytes=2097152
restore:        success_count=1 accepted_bytes=2097152
payload check:  full/target/neighbor restored = true
```

vLLM cold8 mechanism run:

```text
status: ok
completed_requests: 32 / 32
output_tokens: 2048
total_generate_seconds: 219.0808
throughput: 9.35 tok/s
gpu_memory_after_load: 29721 / 32607 MiB
gpu_memory_after_generate: 31903 / 32607 MiB
```

Stage-M residency:

```text
before_stage_m:
  cold logical queried bytes: 1811939328
  gpu bytes: 1755316224
  cpu bytes: 56623104

after_stage_m_clear_before_prealloc:
  cold logical queried bytes: 1811939328
  gpu bytes: 0
  cpu bytes: 1811939328
```

2MiB envelope release:

```text
unique envelope ids released: 1301
unique released bytes: 2638217216
```

Reserved slot prealloc:

```text
slot_count: 48
slot_bytes: 2097152
success_count: 48
failure_count: 0
reserved_backing_bytes: 100663296
```

Runtime reserved prefetch:

```text
reserved prefetch ioctls: 1636
all ioctls ok: true
successful envelope entries: 7915
failed envelope entries: 0
accepted bytes: 16598958080
logical reserved_prefetch_issued records: 4580
logical reserved_prefetch_failed records: 0
```

Cold8 routing share from the reused ShareGPT fixture:

```text
cold assignments: 5464
total assignments: 4982400
cold share: 0.1097%
```

## Failure learned from the control run

Without `VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=1`, the run completed request 0, then failed on request 1:

```text
layer: model.layers.0.mlp.experts
step: 67
entry_count: 13
success_count: 6
failure_count: 7
rm_status: 65541
requested_bytes: 27262976
accepted_bytes: 12582912
reserved_backing_bytes: 100663296
```

There were no driver registration failures in that run. The failure happened inside the reserved prefetch ioctl. The likely cause is that a 2MiB envelope can include neighboring hot expert data. Those neighboring pages can fault back to normal GPU residency before a later cold expert restore. The driver restore path requires the full envelope source pages to be CPU-resident before copying into the reserved slot, so the ioctl rejected entries whose envelope source was not fully CPU-resident.

Enabling CPU-source refresh before each reserved restore fixed this issue, at the cost of extra CPU migration and lower throughput.

