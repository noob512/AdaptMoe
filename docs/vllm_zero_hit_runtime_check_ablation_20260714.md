# vLLM Zero-Hit Runtime Check Ablation

Date: 2026-07-14

## Question

能否证明 zero-hit no-trace 相比 clean baseline 的剩余性能下降，主要来自每层每步的 offload-plan / reserved runtime 检查？

## Code Evidence

The runtime prefetch/check path is gated by:

```python
if not self.uvm_weight_prefetch_enabled:
    return
```

in:

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
```

When `VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1`, each MoE forward enters `_maybe_prefetch_uvm_expert_weights()`. For reserved expert mode, even a no-hit case still computes active experts, intersects against the offload mask, and returns through the no-offloaded-active fast path.

When `VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0`, the function returns before that plan/intersection logic.

## Matched Ablation

All runs use the same ShareGPT fixture:

```text
fixture_sha256 = f676bc2c140b38b95e53401be8516d3457a7ffd560f36ff3c8d6adb80804259f
num_requests = 32
output_tokens = 2048 total
kv_cache_memory_bytes = 2147483648
```

New ablation run:

```text
run_root = /tmp/vllm_sharegpt_zero_hit_512k_offload_only_runtime_off_codex_20260713_235951
```

Configuration:

```text
Stage-M zero-hit expert release: enabled
VLLM_UVM_WEIGHT_PREFETCH_ENABLE = 0
VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP = 0
VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M = 0
runtime reserved restore/check path = disabled
```

This is only safe because the plan contains zero-hit experts for this exact request fixture. It should not be used for a plan that may be touched at runtime.

The log confirms Stage-M still ran:

```text
Stage M post-warmup placement begin
model_runner Stage M module loop end issued_modules=48 elapsed=1.670s
Stage M init-time weight placement visited 48 MoE modules.
Stage M post-warmup placement end
```

The log does not contain:

```text
expert reserved runtime enabled after warmup
post-Stage-M reserved prealloc
reserved_prefetch...
```

## Results

| run | runtime check path | restore | seconds | tokens/s |
|---|---|---:|---:|---:|
| clean no-offload 2GiB | none | 0 | 106.396 | 19.249 |
| zero-hit no-trace, runtime on | enabled | 0 | 120.014 | 17.065 |
| zero-hit offload-only, runtime off | disabled | 0 | 104.819 | 19.538 |

The runtime-on vs runtime-off delta is:

```text
120.014 - 104.819 = 15.195 s
```

Using the previous trace count:

```text
MoE calls = 98448
15.195 s / 98448 = 154 us per MoE call
```

This matches the earlier estimate that the residual slowdown is a small per-MoE-call overhead accumulating over many layer invocations.

## Conclusion

Yes. This ablation strongly supports the claim:

```text
The remaining slowdown in zero-hit no-trace is mainly caused by the runtime reserved/prefetch fast-path check, not by expert offload itself.
```

Evidence:

- Same fixture and same zero-hit plan.
- Stage-M offload still ran.
- Runtime prefetch/reserved path disabled.
- Throughput recovered from `17.06 tok/s` to `19.54 tok/s`, matching or slightly exceeding the clean 2GiB baseline `19.25 tok/s`.
- No new Xid/MMU fault was observed.

The next optimization target is therefore the per-MoE runtime path:

```text
_maybe_prefetch_uvm_expert_weights()
  -> active expert extraction
  -> reserved offload mask check
  -> no-offloaded-active fast return
```

For production, we cannot simply disable this path unless the offloaded set is guaranteed unreachable. The practical fix is to make the no-hit path cheaper, or statically skip it for layers/steps where no offloaded expert can be active.
