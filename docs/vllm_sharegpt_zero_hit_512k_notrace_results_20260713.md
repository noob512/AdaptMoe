# vLLM ShareGPT Zero-Hit 512KiB No-Trace Results

Date: 2026-07-13

## Questions

1. 当前实验中卸载掉的显存是否能被重新用作 KV cache？
2. 能否删去 trace 等额外开销，重新实验一次最佳性能，只卸载不会被用到的 experts？

## Short Answer

当前实验已经证明：offloaded expert 释放出的 PMM backing 可以被后续 reserved slot pool 复用。

但当前 vLLM run 还没有证明：这部分释放出的空间已经被同一次 run 的 KV cache 扩容使用。原因是当前时序是：

```text
load model
initialize / allocate KV cache
warmup
Stage-M expert offload/release
reserved slot prealloc
generation
```

日志中 KV cache 在 Stage-M 之前已经固定：

```text
reserved 2.00 GiB memory for KV Cache
GPU KV cache size: 21,840 tokens
Stage M post-warmup placement begin
```

所以 Stage-M 释放出来的 expert backing 可以服务于后续分配，例如 reserved slot backing；但不会自动回填到已经建好的 KV cache。要让它直接变成更多 KV cache，需要改成以下之一：

```text
A. 在 KV cache allocation 之前完成 expert release
B. 支持运行时动态扩 KV cache，并让动态 KV allocator 在 Stage-M 后重新申请 backing
```

## Experiment Setup

复用上一轮 zero-hit 实验的同一批 ShareGPT requests 和同一个 zero-hit expert plan：

```text
source_run = /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900
fixture_sha256 = f676bc2c140b38b95e53401be8516d3457a7ffd560f36ff3c8d6adb80804259f
selected_zero_hit_experts = 70
kv_cache_memory_bytes = 2147483648
slot_count = 128
slot_bytes = 524288
```

No-trace run:

```text
run_root = /tmp/vllm_sharegpt_zero_hit_512k_reserved_notrace_tag_codex_20260713_234907
```

Disabled:

```text
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
VLLM_UVM_RESIDENCY_TRACE_ENABLE=0
VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0
VLLM_UVM_EXPERT_RESERVED_DEBUG_TRACE=0
VLLM_UVM_EXPERT_RESERVED_STREAM_TRACE=0
VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE=0
```

Kept enabled because it is functional, not just tracing:

```text
VLLM_UVM_DRIVER_TAG_ENABLE=1
```

## Important Failed Attempt

I first tried disabling `VLLM_UVM_DRIVER_TAG_ENABLE` as if it were trace-only. That failed at post-Stage-M slot prealloc:

```text
run_root = /tmp/vllm_sharegpt_zero_hit_512k_reserved_notrace_codex_20260713_234739
success_count = 0
failure_count = 128
reserved_backing_bytes = 0
rm_status = 81
```

Conclusion: driver tag / driver region registration is required by the current release-to-reserved-slot path. It should not be removed when optimizing trace overhead.

## Successful No-Trace Result

Post-Stage-M reserved slot prealloc succeeded:

```text
slot_count = 128
slot_bytes = 524288
chunk_size = 65536
success_count = 128
failure_count = 0
backed_slots = 128
reserved_backing_bytes = 67108864
rm_status = 0
```

The run completed:

```text
completed_requests = 32/32
output_tokens = 2048
total_generate_seconds = 120.014
tokens/s = 17.065
new_xid_or_mmu_fault = false
gpu_memory_after_load = 31963 MiB
gpu_memory_after_generate = 32109 MiB
```

Trace files were removed from the hot runtime path. In the previous traced zero-hit run, `prefetch_trace.jsonl` alone was about 158 MiB. In this no-trace run there is no `prefetch_trace.jsonl` or `residency_trace.jsonl`; only small control artifacts remain:

```text
result.json = 48 KiB
post_stage_m_reserved_prealloc.jsonl = 60 KiB
progress.jsonl = 92 KiB
run.log = 96 KiB
```

## Performance Comparison

| run | offloaded experts | runtime restore | total generate seconds | tokens/s |
|---|---:|---:|---:|---:|
| clean no-offload, 1GiB KV budget | 0 | 0 | 98.904 | 20.707 |
| clean no-offload, 2GiB KV budget | 0 | 0 | 106.396 | 19.249 |
| no-offload pilot with routing trace | 0 | 0 | 170.918 | 11.982 |
| cold8 restore-heavy, traced | 384 | 10.3 GiB | 182.093 | 11.247 |
| zero-hit offload, traced | 70 | 0 | 152.742 | 13.408 |
| zero-hit offload, no runtime trace | 70 | 0 expected | 120.014 | 17.065 |

## Interpretation

Removing runtime trace makes a large difference:

```text
zero-hit traced:    13.41 tok/s
zero-hit no-trace:  17.06 tok/s
clean 2GiB base:    19.25 tok/s
```

This supports the earlier diagnosis:

```text
If the offloaded experts are not selected, restore traffic disappears.
The previous large slowdown was mostly trace + restore overhead, not offload itself.
```

The remaining gap to clean no-offload is likely from:

- runtime plan-intersection checks on every MoE invocation;
- reserved executor fast-path checks even when no offloaded expert is active;
- driver tag / region bookkeeping kept for correctness;
- remaining verbose stdout / warning logs in the experimental branch;
- eager-mode test harness differences.

## KV Cache Reuse Conclusion

This run should not be described as "expert memory was reused by KV cache" because KV cache allocation happens before Stage-M release. What it demonstrates is:

```text
expert release -> PMM backing becomes reusable -> reserved slot prealloc succeeds
```

To demonstrate KV reuse directly, the next experiment should change the allocation order:

```text
1. load model
2. release selected experts
3. allocate or grow KV cache
4. confirm a larger KV cache budget succeeds only with expert release
```

or implement dynamic KV growth after Stage-M and show that additional KV blocks are allocated from the freed PMM backing.
