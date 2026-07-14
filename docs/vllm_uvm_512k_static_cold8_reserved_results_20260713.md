# vLLM 512KiB Static Cold8 Reserved Slot Experiment

Date: 2026-07-13

## Goal

把已经在 smoke 中验证过的 512KiB release/restore 机制接入 vLLM，先使用静态 expert 方案：

- 每层从 pilot trace 里选择 8 个 cold experts。
- Stage-M 后把这些 cold expert 权重按 512KiB subrange release 到 CPU。
- 预分配 512KiB logical reserved slots，底层使用 8 个 64KiB PMM chunks。
- runtime 只在命中 offloaded expert 前选择性 restore 对应 512KiB chunks。
- 本轮不做正式 baseline，pilot 只用于生成静态 cold8 plan。

## Implementation

关键代码改动：

- `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`
  - 增加通用 envelope 配置：
    - `VLLM_UVM_EXPERT_ENVELOPE_ENABLE`
    - `VLLM_UVM_EXPERT_ENVELOPE_BYTES`
    - `VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT`
  - 将原 2MiB envelope 泛化为 configurable chunk envelope。
  - 512KiB envelope 使用 region id 形如 `uvmchunk:80000`。
  - Stage-M driver release 对 512KiB envelope 传入 `subchunk_64k=True`。
  - runtime reserved restore 展开为 512KiB envelope entries。

- `workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py`
  - driver region cache 兼容 `uvm_chunk_envelope`。
  - executor cache key 增加 `slot_bytes`，避免不同 slot size 复用错误 executor。

- `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`
  - weight address registration 识别通用 envelope 配置。

- `workloads/vllm/run_cold8_reserved_slot_failfast_experiment.sh`
  - 接入 `VLLM_UVM_EXPERT_ENVELOPE_*`。
  - summary 聚合器兼容 nested 512KiB `driver_release_entry.entries[]`。

- `workloads/vllm/run_cold8_512k_static_reserved_experiment.sh`
  - 新增静态 512KiB cold8 wrapper。
  - 默认 `RESERVED_SLOT_BYTES=524288`。
  - 默认 `RESERVED_SLOT_COUNT=128`，避免超过 ioctl max entries 128。

## Command

```bash
cd /home/ubuntu/nvidia-uvm-gpu

printf 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable >/dev/null
printf 524288 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes >/dev/null

RUN_ROOT=/tmp/vllm_cold8_512k_static_reserved_codex_20260713_202310 \
NUM_REQUESTS=8 \
OUTPUT_TOKENS=32 \
TIMEOUT_SEC=1800 \
RESERVED_SLOT_COUNT=128 \
workloads/vllm/run_cold8_512k_static_reserved_experiment.sh
```

## Environment

```text
model = Qwen/Qwen3-30B-A3B-FP8 local snapshot
dataset = ShareGPT_V3_unfiltered_cleaned_split
num_requests = 8
output_tokens = 32
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
nvidia_uvm srcversion = 386A1FA2E18DDDFCE9EFE88
```

## Results

Result directory:

```text
/tmp/vllm_cold8_512k_static_reserved_codex_20260713_202310
```

Overall:

```text
mechanism returncode = 0
failure_reason = None
pilot completed = 8/8
mechanism completed = 8/8
output_tokens = 256
mechanism total_generate_seconds = 29.150
mechanism tokens_per_second = 8.782
new Xid/MMU fault = false
```

Static cold8 plan:

```text
offload_plan_records = 384
prefetch_plan_records = 384
cold_assignments = 486
total_assignments = 2649600
cold_share = 0.018342%
```

Stage-M 512KiB release:

```text
logical release records = 768
512KiB envelope entries = 3456
ok entries = 3456
failed entries = 0
attempted bytes = 1811939328
released GPU bytes = 1755316224
```

Residency snapshot:

```text
before Stage-M:
  cold GPU bytes = 1755316224
  cold CPU bytes = 56623104

after Stage-M:
  cold GPU bytes = 0
  cold CPU bytes = 1811939328
```

The difference between attempted bytes and released GPU bytes is expected: before Stage-M, 56623104 bytes were already CPU resident.

Reserved slot prealloc:

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

Runtime selective restore:

```text
reserved prefetch ioctl records = 130
restored 512KiB entries = 2052
successful restored entries = 2052
failed restored entries = 0
accepted bytes = 1075838976
reserved_prefetch_cpu_source = 0
```

## Conclusion

The 512KiB logical / 64KiB physical mechanism works inside vLLM for a static cold8 expert plan:

- cold expert weight residency is removed from GPU after Stage-M;
- released PMM backing is reusable by 512KiB reserved slots;
- runtime selectively restores only the cold expert chunks that are actually needed;
- no CPU-source refresh was required in this run;
- generation completed successfully with no Xid/MMU fault.

This verifies the core mechanism before moving to a dynamic hot/cold expert policy.
