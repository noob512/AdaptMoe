# Explicit hot/cold expert layout microbenchmark

## Feasibility

The proposal is feasible as a standalone experiment and is a cleaner direction
than post-load `slot_va_block_remap` for vLLM stability.

The key change is to stop pretending that an already-created UVM expert tensor
can be cheaply replaced after `load_model()`. Instead, the experiment makes the
layout explicit:

- hot experts: expert IDs `[0, 63]` for every layer are materialized as GPU
  tensors at load time;
- cold experts: expert IDs `[64, 127]` remain canonical CPU/checkpoint data and
  are copied into a fixed GPU slot pool;
- cold slot pool capacity: `2 layers * 64 cold experts`;
- cold slot mapping: `slot = expert_id - 64`, `bank = layer_id % 2`.

For Qwen3-30B-A3B FP8, the observed per-expert byte size is:

```text
w13: 1536 * 2048 * 1 byte = 3,145,728 bytes
w2:  2048 *  768 * 1 byte = 1,572,864 bytes
total per expert             = 4,718,592 bytes
```

Therefore:

```text
hot GPU memory, 48 layers * 64 experts ~= 13.50 GiB
cold slot pool, 2 layers * 64 experts ~= 0.56 GiB
streaming CPU staging per cold layer    ~= 0.28 GiB
```

This is much more controlled than post-load PTE remap: no VA-block remap, no
slot-backed PTE, no UVM lifecycle guard, and no PTE fallback storm.

## Simulated call chain

The microbenchmark intentionally mimics only the weight-loading part of vLLM:

```text
CPU checkpoint tensor
  -> GPU destination tensor
  -> copy_()
```

Hot path:

```text
for each layer:
  allocate hot_w13[layer][64, ...] on GPU
  allocate hot_w2[layer][64, ...] on GPU
  fill CPU checkpoint-like tensors for expert 0..63
  hot_w13[layer].copy_(cpu_w13)
  hot_w2[layer].copy_(cpu_w2)
```

Cold path:

```text
allocate slot_w13[2, 64, ...] on GPU
allocate slot_w2[2, 64, ...] on GPU

for each layer:
  bank = layer % 2
  fill CPU checkpoint-like tensors for expert 64..127
  slot_w13[bank].copy_(cpu_w13)
  slot_w2[bank].copy_(cpu_w2)
  generate random expert IDs
  read slot[expert_id - 64] to mimic inference access
```

## Hot backing choice

The script supports:

```text
--hot-backing auto
--hot-backing uvm
--hot-backing cuda
```

`auto` tries vLLM's local UVM memory pool first and falls back to native
`cudaMalloc` if the UVM allocator shim is unavailable. For the cold slot pool,
the recommended default is native CUDA memory:

```text
--slot-backing cuda
```

That keeps the slot pool out of UVM and avoids reintroducing the same managed
PTE lifecycle problem.

## Scripts

Dry-run memory estimate:

```bash
DRY_RUN=1 workloads/vllm/run_hot_cold_expert_layout_microbench.sh
```

Small functional smoke:

```bash
workloads/vllm/run_hot_cold_expert_layout_microbench.sh
```

Qwen3 FP8-sized experiment:

```bash
FULL_QWEN3_FP8=1 \
HOT_BACKING=auto \
SLOT_BACKING=cuda \
TOPK=8 \
TOUCH_ELEMENTS=4096 \
workloads/vllm/run_hot_cold_expert_layout_microbench.sh
```

The output files are:

```text
log/current_hot_cold_expert_layout_microbench/run.log
log/current_hot_cold_expert_layout_microbench/result.json
```

## Real Checkpoint Path

The checkpoint prototype reads Qwen3-30B-A3B FP8 weights directly from
`safetensors`, without starting a vLLM server:

```text
model.safetensors.index.json
  -> safe_open(shard)
  -> model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight
  -> model.layers.{layer}.mlp.experts.{expert}.up_proj.weight
  -> model.layers.{layer}.mlp.experts.{expert}.down_proj.weight
```

For each expert, the prototype builds the same fused layout used by the
synthetic bench:

```text
w13 = concat(gate_proj.weight, up_proj.weight, dim=0)
w2  = down_proj.weight
```

Hot checkpoint path:

```text
for each selected layer:
  read experts 0..63 from safetensors
  fuse gate/up into w13 CPU staging
  allocate hot_w13[layer] and hot_w2[layer] on GPU
  copy hot CPU staging to GPU
```

Cold checkpoint path:

```text
allocate slot_w13[2, 64, ...] and slot_w2[2, 64, ...] on GPU

for each selected layer:
  bank = layer % 2
  read experts 64..127 from safetensors
  fuse gate/up into cold CPU staging
  copy cold CPU staging to slot_w13[bank], slot_w2[bank]
  randomly sample cold expert IDs
  access slot[expert_id - 64]
```

The runner is:

```bash
workloads/vllm/run_hot_cold_expert_checkpoint_microbench.sh
```

Useful modes:

```bash
# Two-layer real checkpoint smoke.
LAYER_COUNT=2 \
workloads/vllm/run_hot_cold_expert_checkpoint_microbench.sh

# Three-layer reuse smoke. This proves bank 0 is reused by layer 0 then layer 2.
LAYER_COUNT=3 \
workloads/vllm/run_hot_cold_expert_checkpoint_microbench.sh

# Full 48-layer Qwen3 FP8 data-path run.
LAYER_COUNT=48 \
POINTER_CHECKS_PER_LAYER=2 \
workloads/vllm/run_hot_cold_expert_checkpoint_microbench.sh
```

## Isolated Pointer / MoE Correctness

`bench_hot_cold_expert_checkpoint.py` includes a tiny pointer-dispatch
correctness check. For sampled hot and cold experts, it computes the same small
MoE-style forward from two sources:

```text
reference:
  original checkpoint gate/up/down tensors

split path:
  if expert < 64:
    hot_w13[layer][expert], hot_w2[layer][expert]
  else:
    slot_w13[layer % 2][expert - 64], slot_w2[layer % 2][expert - 64]
```

The tiny computation is intentionally small but uses the real FP8 values:

```text
gate = w13[:intermediate]
up   = w13[intermediate:2 * intermediate]
out  = w2 @ ((gate @ x) * (up @ x))
```

The check passes only if the split hot/cold pointer path matches the checkpoint
reference within tolerance.

## Results From 2026-06-30

All stable `cuda/cuda` runs below passed.

```text
synthetic toy:
  log/current_hot_cold_expert_layout_microbench_toy_v2/result.json
  passed=true
  per-layer random cold access verify_ok=true

synthetic Qwen3 FP8 shape, cuda hot + cuda slot:
  log/current_hot_cold_expert_layout_microbench_qwen3_fp8_cuda_cuda/result.json
  passed=true
  hot copy bytes=14,495,514,624
  hot copy time=0.799 s
  hot copy throughput=18.14 GB/s
  cold streamed bytes=14,495,514,624
  cold wall time=0.724 s
  cold effective throughput=20.03 GB/s
  peak torch CUDA allocation=15,137,376,256 bytes

real checkpoint, layers 0..2, cuda hot + cuda slot:
  log/current_hot_cold_expert_checkpoint_l0_l2_reuse_cuda_cuda/result.json
  passed=true
  latest_layer_by_bank={0: 2, 1: 1}
  pointer records=18
  pointer max_abs_diff=0.0
  peak torch CUDA allocation=1,556,350,976 bytes

real checkpoint, layers 0..47, cuda hot + cuda slot:
  log/current_hot_cold_expert_checkpoint_l0_l47_full_cuda_cuda/result.json
  passed=true
  streamed checkpoint bytes=28,991,029,248
  hot copy bytes=14,495,514,624
  hot read time=93.78 s
  cold read time=91.60 s
  latest_layer_by_bank={0: 46, 1: 47}
  pointer records=96
  pointer max_abs_diff=0.0
  peak torch CUDA allocation=15,145,895,936 bytes
```

These results prove the independent layout mechanism is implemented:

- Qwen3 FP8-sized hot/cold allocation fits in the expected memory budget.
- The fixed two-layer cold slot pool is reused correctly.
- Real checkpoint expert weights can be read without starting vLLM.
- Hot and cold split pointers produce exactly the same tiny forward result as
  the original checkpoint tensors for sampled experts.

## Backing Comparison

Native CUDA backing is the stable default today:

```text
HOT_BACKING=cuda SLOT_BACKING=cuda
```

UVM allocator-backed tensors were also tested in the synthetic Qwen3 FP8 shape:

```text
HOT_BACKING=auto SLOT_BACKING=cuda:
  effective hot backing=uvm
  passed=true
  hot copy throughput=1.69 GB/s
  process segfaulted during teardown after writing result.json

HOT_BACKING=cuda SLOT_BACKING=uvm:
  effective slot backing=uvm
  passed=true
  cold access/verify passed
  process segfaulted during teardown after writing result.json
```

So UVM backing is functionally reachable in this isolated prototype, but it is
not a stable default yet. The recommended next integration path is to keep hot
and cold slot tensors native CUDA first, then revisit UVM allocation separately
after allocator teardown is understood.

## Three-Tier Hot / Stream / UVM Prototype

`bench_hot_cold_expert_checkpoint.py` also supports a three-tier expert layout:

```text
expert 0..63:
  hot tier
  per-layer resident GPU tensors

expert 64..95:
  stream tier
  copied layer-by-layer into a 2-layer GPU slot pool

expert 96..127:
  UVM tier
  per-layer managed tensors, accessed by demand fault or explicit prefetch
```

The key arguments are:

```bash
--hot-experts 64
--cold-experts 64
--stream-experts 32
--uvm-experts 32
--uvm-backing uvm
--uvm-mode fault      # or prefetch
```

The dispatch rule is:

```text
if expert_id < 64:
  use hot_w13[layer][expert_id], hot_w2[layer][expert_id]

elif expert_id < 96:
  bank = layer % slot_layers
  slot = expert_id - 64
  use stream_slot_w13[bank][slot], stream_slot_w2[bank][slot]

else:
  local = expert_id - 96
  use uvm_w13[layer][local], uvm_w2[layer][local]
```

For `--uvm-mode prefetch`, the prototype calls
`prefetch_range_to_device(ptr, size, device)` for the selected UVM expert's
`w13` and `w2` before touching them.

Results from 2026-06-30:

```text
three-tier dry run, layers 0..1:
  log/current_three_tier_checkpoint_dry/result.json
  hot_gpu_gib=0.5625
  stream slot_gpu_gib=0.28125
  uvm_managed_gib=0.28125
  tiers={hot: [0,64], stream: [64,96], uvm: [96,128]}

three-tier layer 0, UVM fault mode:
  log/current_three_tier_checkpoint_l0_fault/result.json
  passed=true
  pointer records=6
  pointer max_abs_diff=0.0
  random access split: stream experts=[64,66,69,71], uvm experts=[101,115,120,122]
  effective UVM backing=uvm
  process segfaulted during teardown after writing result.json

three-tier layer 0, UVM prefetch mode:
  log/current_three_tier_checkpoint_l0_prefetch/result.json
  passed=true
  pointer records=6
  pointer max_abs_diff=0.0
  prefetch_success_count=8
  effective UVM backing=uvm
  process segfaulted during teardown after writing result.json

three-tier layer 0, CUDA control for the UVM tier:
  log/current_three_tier_checkpoint_l0_uvm_tier_cuda_control/result.json
  passed=true
  pointer records=6
  pointer max_abs_diff=0.0
  process exited cleanly
```

Conclusion: the three-tier dispatch and data-path prototype is feasible and
correct in the isolated benchmark. The remaining blocker is not the tiering
logic; it is the current UVM allocator teardown behavior when real UVM backing
is used.

## What This Proves

This experiment can prove:

- hot expert allocation can be kept GPU-resident without post-load remap;
- cold experts can use a fixed two-layer GPU slot pool;
- slot reuse by `bank = layer_id % 2` is deterministic;
- cold expert ID to slot mapping is stable;
- layer-by-layer copy plus random expert access works without vLLM startup,
  server readiness, KV cache, NCCL, or UVM PTE remap noise.

It does not yet prove:

- fused MoE kernels can consume the split hot/cold layout directly;
- the scheduler can predict cold experts early enough for real inference;
- full vLLM E2E stability.

Those should come after this isolated layout test passes.
