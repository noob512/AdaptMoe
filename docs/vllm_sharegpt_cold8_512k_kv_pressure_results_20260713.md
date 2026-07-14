# vLLM ShareGPT Cold8 512KiB KV Pressure Experiment Results

Date: 2026-07-13

## Goal

验证两种场景：

1. KV budget 较小、专家全部常驻 GPU 的 baseline。
2. KV budget 较大时，将每层最冷的 8 个 expert 卸载到 CPU，并通过 512KiB logical / 64KiB physical reserved slot 机制按需搬回。

希望确认：

- cold experts 是否真的从 GPU residency 中移除；
- 释放出来的显存是否能被 reserved slot 复用；
- 访问前是否能按需把候选 expert 搬回；
- 卸载前后性能是否接近。

## Setup

```text
model = Qwen/Qwen3-30B-A3B-FP8
dataset = ShareGPT 32 requests
output_tokens = 64
max_model_len = 2048
max_num_seqs = 1
kv_cache_memory_bytes = 2147483648
cold_experts_per_layer = 8
reserved_slot_count = 128
reserved_slot_bytes = 524288
reserved_slot_policy = dynamic
reserved_bank_count = 1
expert_envelope_bytes = 524288
reserved_prefetch_cpu_source = 0
init_weight_offload_mode = driver_release
```

This run reused the current pilot-derived cold8 plan from the fresh ShareGPT fixture under:

```text
/tmp/vllm_sharegpt_cold8_512k_compare_codex_20260713_220335
```

The older 7/12 three-tier CSV artifact was not present anymore, so the cold8 plan was regenerated from the current pilot trace.

## Baselines

### Low KV pressure baseline

```text
result dir = /tmp/vllm_sharegpt_cold8_512k_compare_codex_20260713_220335/baseline_no_offload_1g
completed_requests = 32
output_tokens = 2048
total_generate_seconds = 98.904
tokens/s = 20.707
gpu_memory_after_generate = 31895 MiB
```

### Higher KV budget baseline

```text
result dir = /tmp/vllm_sharegpt_cold8_512k_compare_codex_20260713_220335/baseline_no_offload_2g
completed_requests = 32
output_tokens = 2048
total_generate_seconds = 106.396
tokens/s = 19.249
gpu_memory_after_generate = 32109 MiB
```

Important caveat: `--kv-cache-memory-bytes=2147483648` is the configured KV
budget. In this run, `nvidia-smi` residency only increased from 31895 MiB to
32109 MiB between the 1GiB and 2GiB no-offload baselines. With
`max_num_seqs=1`, the experiment does not prove that 2GiB of KV pages were
simultaneously resident in HBM. It proves that vLLM could initialize and run
with that configured KV budget while all experts stayed resident.

## Mechanism Run

```text
result dir = /tmp/vllm_sharegpt_cold8_512k_compare_codex_20260713_220335/cold8_offload_mechanism
mechanism returncode = 0
failure_reason = None
completed_requests = 32
output_tokens = 2048
total_generate_seconds = 179.916
tokens/s = 11.383
gpu_memory_after_load = 31317 MiB
gpu_memory_after_generate = 32091 MiB
new Xid/MMU fault = false
```

Cold share from the pilot trace:

```text
cold_share = 0.001097
```

## Expert Offload

Stage-M offload succeeded:

```text
issued_records = 768
issued_bytes = 1811939328
driver_release_failed_records = 0
driver_release_released_bytes = 1755316224
cold_gpu_bytes_before_stage_m = 1755316224
cold_gpu_bytes_after_stage_m_clear = 0
cold_cpu_bytes_after_stage_m_clear = 1811939328
```

This means the cold8 expert residency really moved off GPU.

## Reserved Slot Reuse

512KiB logical slots were successfully preallocated on 64KiB physical chunks:

```text
slot_count = 128
slot_bytes = 524288
success_count = 128
failure_count = 0
reserved_backing_bytes = 67108864
rm_status = 0
```

## Runtime Restore

The runtime selective restore path also completed cleanly:

```text
reserved_debug_prefetch_ioctl_exit records = 1636
successful restore entries = 20610
failed restore entries = 0
accepted_bytes = 10805575680
max_item_count = 72
reserved_prefetch_no_offloaded_active = 96812
```

This shows the mechanism really did restore offloaded experts on demand, and did not report runtime failures.

## Interpretation

What worked:

- cold experts were removed from GPU residency;
- the released PMM space was reused by reserved slots;
- restore happened only when the runtime touched offloaded experts;
- the run completed end-to-end with no Xid/MMU fault.

What did not meet the expectation:

- performance was not close to the all-GPU baseline;
- even with a very cold plan, the runtime restore path still introduced noticeable cost.

The most likely reasons are:

- many fine-grained restore IOCTLs and envelope-level copies during generation;
- prefill can still touch a burst of cold experts;
- trace collection itself adds some overhead in this run.

If an offloaded expert is never selected, the direct restore cost for that
expert should be zero. The remaining overhead would come from fixed mechanism
costs such as plan checks, tracing, slot bookkeeping, and cleanup. In the
measured run, however, cold experts were selected often enough to trigger 1636
restore ioctls and 20610 successful 512KiB restore entries, so the measured
slowdown is not a pure "never accessed" case.

## Conclusion

The mechanism is functionally successful:

- expert offload succeeded,
- GPU space was reclaimed,
- reserved slot backing was reused,
- selective restore worked.

But the current implementation still has a clear performance gap versus the all-GPU baseline, so the next step is to reduce restore frequency/cost or relax the runtime restore granularity.
