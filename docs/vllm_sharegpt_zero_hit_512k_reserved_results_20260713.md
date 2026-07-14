# vLLM ShareGPT Zero-Hit 512KiB Reserved Experiment Results

Date: 2026-07-13

## Goal

针对上轮 cold8 512KiB 实验中出现的大量 restore 流量，重新使用同一批 ShareGPT requests 做一次定向实验：

1. 从同一个 pilot routing trace 里找出真正没有被访问的 experts。
2. 尝试每层最多卸载 10 个 zero-hit experts。
3. 运行 vLLM 机制组，确认这些 offloaded experts 是否完全不触发 restore。
4. 对比 clean no-offload baseline、restore-heavy cold8 和 zero-hit offload 的性能。

## Inputs

```text
model = Qwen/Qwen3-30B-A3B-FP8
fixture = /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900/sharegpt_32x64_fixture.json
fixture_sha256 = f676bc2c140b38b95e53401be8516d3457a7ffd560f36ff3c8d6adb80804259f
num_requests = 32
output_tokens_per_request = 64
max_model_len = 2048
max_num_seqs = 1
kv_cache_memory_bytes = 2147483648
```

The fixture SHA256 matches the previous restore-detail run:

```text
/tmp/vllm_sharegpt_cold8_512k_restore_detail_codex_20260713_223902
```

So the routing decisions are comparable for the targeted zero-hit selection.

## Plan Selection

The requested idea was "unload 10 unvisited experts per layer". The actual trace does not contain 10 zero-hit experts in any layer.

```text
requested_experts_per_layer = 10
actual_zero_hit_selected_total = 70
max_selected_in_one_layer = 8
layers_with_0_zero_hit_experts = 15
layers_with_lt_10_zero_hit_experts = all 48 layers
```

Therefore this run used the strict valid policy:

```text
selection = all zero-hit experts, capped at 10 per layer
```

Layer counts:

```text
histogram selected_count_per_layer:
  0 experts: 15 layers
  1 expert : 19 layers
  2 experts: 6 layers
  3 experts: 1 layer
  4 experts: 4 layers
  6 experts: 2 layers
  8 experts: 1 layer
```

Selected experts:

```text
L0:[10]; L1:-; L2:[19,105]; L3:[83]; L4:[41,42]; L5:-; L6:-; L7:[121]
L8:-; L9:-; L10:-; L11:[38]; L12:-; L13:[116]; L14:[103]; L15:-
L16:[49,86,104,118]; L17:-; L18:[95]; L19:[121]; L20:-; L21:[18]; L22:[11]; L23:[123]
L24:[48]; L25:[69]; L26:[127]; L27:-; L28:[118]; L29:-; L30:[2]; L31:-
L32:[12]; L33:[18]; L34:-; L35:[83,123]; L36:[13,48]; L37:[17,69]; L38:-; L39:[68]
L40:[86,96,104]; L41:[4,27]; L42:[11,26,51,122]; L43:[95,103,109,116]
L44:[74,75,85,87,115,124]; L45:[25,57,59,61,68,96,97,122]
L46:[58,65,73,87,108,121]; L47:[29,40,43,75]
```

## Run Artifacts

```text
run_root = /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900
plan = /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900/zero_hit_offload_plan.json
plan_stats = /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900/zero_hit_plan_stats.json
mechanism_dir = /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900/cold8_offload_mechanism
restore_analysis = /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900/cold8_restore_analysis
```

Analysis command:

```bash
python3 workloads/vllm/analyze_cold8_reserved_restores.py \
  --run-root /tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900
```

## Offload Result

The 70 selected experts correspond to 140 logical role ranges:

```text
logical bytes = 330301440 bytes = 315 MiB
  per expert = moe_gate_up 3 MiB + moe_down 1.5 MiB = 4.5 MiB
```

Residency snapshots:

```text
after_load_model:
  cold_gpu_bytes = 330301440
  cold_cpu_bytes = 0

before_stage_m:
  cold_gpu_bytes = 320864256
  cold_cpu_bytes = 9437184

after_stage_m_clear:
  cold_gpu_bytes = 0
  cold_cpu_bytes = 330301440
```

Driver release:

```text
driver_release_records = 140
driver_release_entries = 630
attempted_bytes = 330301440
released_bytes = 320864256
success_blocks = 630
failure_blocks = 0
failed_records = 0
```

Interpretation:

The selected zero-hit expert weights were successfully moved out of GPU residency. The release path only had to release 306 MiB because about 9 MiB was already CPU resident immediately before Stage-M.

## Slot Prealloc

Reserved slot allocation succeeded:

```text
slot_count = 128
slot_bytes = 524288
chunk_size = 65536
chunk_count_per_slot = 8
success_count = 128
failure_count = 0
backed_slots = 128
reserved_backing_bytes = 67108864
rm_status = 0
```

This means the 512KiB logical / 64KiB physical reserved pool path remained healthy in this run.

## Runtime Restore

The important result is that no zero-hit expert was touched at runtime:

```text
cold_plan_experts = 70
touched_cold_experts = 0
untouched_cold_experts = 70
restore_entry_count = 0
restore_bytes = 0
restore_event_rows = 0
ioctl_records = 0
ioctl_success_entries = 0
ioctl_failure_entries = 0
```

The prefetch trace confirms the same thing:

```text
reserved_prefetch_no_offloaded_active = 98448
reserved restore ioctl records = 0
active_reserved_hit_records = 0
```

So this run validates the expected behavior: if the offloaded experts are truly not selected, the selective restore path does not copy them back.

## Performance

```text
mechanism_returncode = 0
completed_requests = 32/32
output_tokens = 2048
total_generate_seconds = 152.742
tokens/s = 13.408
new_xid_or_mmu_fault = false
```

Comparison:

| run | offloaded experts | runtime restore | total generate seconds | tokens/s |
|---|---:|---:|---:|---:|
| clean no-offload, 1GiB KV budget | 0 | 0 | 98.904 | 20.707 |
| clean no-offload, 2GiB KV budget | 0 | 0 | 106.396 | 19.249 |
| no-offload pilot with routing trace | 0 | 0 | 170.918 | 11.982 |
| cold8 restore-heavy | 384 | 10.3 GiB | 182.093 | 11.247 |
| zero-hit offload | 70 | 0 | 152.742 | 13.408 |

## Interpretation

This run answers two different questions:

1. Does unloading truly unvisited experts avoid restore traffic?

Yes. The runtime restore count was exactly zero. None of the 70 offloaded experts was accessed by the same request fixture.

2. Does performance fully return to the clean no-offload baseline?

No. It improves over restore-heavy cold8, but it does not match the clean no-offload baseline. The remaining gap is not from CPU-to-slot restore traffic. It comes from fixed experimental overhead, including:

- reserved/offload runtime checks on every MoE layer invocation;
- weight prefetch trace logging;
- residency trace logging;
- plan/intersection lookup and slot bookkeeping;
- generally running through the instrumented mechanism path rather than the clean vLLM path.

The evidence is that the no-offload pilot with routing trace was also much slower than clean no-offload, even without any expert offload. In this experiment family, tracing/instrumentation overhead is large enough that raw tokens/s should not be interpreted as pure mechanism cost unless a matched instrumentation control is used.

## Conclusion

The targeted zero-hit experiment supports the hypothesis:

```text
If an offloaded expert is not accessed, the expensive restore traffic disappears.
```

But the original "10 zero-hit experts per layer" assumption is false for this ShareGPT fixture. The strict zero-hit set contains only 70 experts total. If we force 10 experts per layer, the extra selected experts will have nonzero hits and will trigger restore again.

The next clean performance test should use a matched control:

```text
A. same fixture, clean no-offload
B. same fixture, reserved runtime + trace enabled, empty offload plan
C. same fixture, zero-hit offload plan
D. same fixture, cold8/cold10 lowest-hit plan
```

Only B vs C isolates the cost of "offload but no restore".
