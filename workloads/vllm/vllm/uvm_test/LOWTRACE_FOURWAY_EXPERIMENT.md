# Low-Trace Four-Way vLLM UVM Expert/KV Managed Experiment

本文档记录 2026-07-12 的四组低 trace vLLM 性能实验。实验目标是验证在新的选择性 UVM 分配策略下，静态 tail expert 卸载、动态真实负载卸载和无卸载 baseline 的性能关系，并排除早前诊断 trace 对性能结论的干扰。

本轮原始结果目录：

```text
/tmp/vllm_lowtrace_fourway_20260712_123124
```

核心结论：关闭诊断 trace 后，`static tail5 0%` 不再比 `dynamic real-load` 慢；早前静态 0% 显著慢于动态方案，主要是诊断 trace 口径导致的。当前真正明显的性能损失仍来自请求阶段命中已卸载 expert 后的 reserved slot remap/prefetch 路径。

## 1. 实验目的

本轮实验回答三个问题：

1. 在只让 expert 主权重和 KV cache 使用 UVM 的机制下，无卸载 baseline 性能是多少。
2. 加上动态真实负载 expert offload + reserved slot 之后，性能下降多少。
3. 静态卸载每层最后 5 个 expert 时，0% 命中和 2% 命中的性能分别如何，以及静态方案是否天然有损性能。

早前的 static tail5 0% 诊断实验打开了 routing trace、prefetch trace、weight map 和 MoE address map。虽然没有真实搬运，但记录路径本身带来大量开销。因此本轮统一关闭这些 trace，仅保留必要的机制执行和结果汇总。

## 2. 通用实验设置

### 2.1 模型和请求

```text
model = /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782
gpu = NVIDIA GeForce RTX 5090, 32607 MiB
requests = 8
max_tokens_sequence = 16,16,16,16,16,16,16,16
total_output_tokens = 128
prompt = Compare CUDA streams and events in one concise paragraph. [request N]
kv_cache_memory_bytes = 1073741824
gpu_memory_utilization = 0.90
vLLM mode = eager
max_model_len = 512
max_num_batched_tokens = 512
max_num_seqs = 1
```

请求 fixture 保存在：

```text
/tmp/vllm_lowtrace_fourway_20260712_123124/request_fixture.json
```

### 2.2 UVM 模块要求

实验要求加载当前构建出的 custom `nvidia_uvm`，并开启 reserved slot PTE remap：

```text
loaded nvidia_uvm srcversion = 29DB0483A6BFFAFF7A20568
uvm_moe_slot_pte_remap_enable = 1
uvm_moe_slot_pte_remap_max_bytes = 4194304
```

运行前可用以下命令检查：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
modinfo -F srcversion kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
cat /sys/module/nvidia_uvm/srcversion
cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable
cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes
```

脚本会检查这些条件；如果 loaded module 与 built module 不一致，或者 PTE remap 未开启，脚本会直接阻止运行。

### 2.3 选择性 UVM 分配策略

本轮不再使用宽泛的 pre-service managed allocator。统一分配策略如下：

```text
VLLM_USE_UVM=0
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_KV_CACHE_MANAGED=1
```

含义：

```text
UVM managed:
  MoE expert 主权重:
    moe_gate_up / w13_weight
    moe_down / w2_weight
  KV cache

native cudaMalloc:
  embedding
  attention 权重
  norm
  router/gate
  lm_head
  FP8 scale / block scale / aux tensor
  runtime temporary buffer / workspace
```

这样可以避免 embedding、router、scale、metadata 等非 expert 常驻权重进入 UVM 管理范围，也避免运行期临时分配被全局 UVM allocator 污染。

### 2.4 Slot 预分配顺序

本轮 reserved slot pool 不在运行前额外抢占显存，而是在 Stage-M expert offload 完成后再预分配：

```text
PREALLOC_RESERVED_BEFORE_RUN=0
POST_STAGE_M_RESERVED_PREALLOC=1
VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=1
```

执行顺序：

```text
1. reset reserved pool
2. load model
3. expert 主权重走 cudaMallocManaged
4. KV cache 走 UVM managed
5. warmup / kernel init
6. Stage-M 将计划内 expert 主权重卸载到 CPU
7. Stage-M offload sync
8. prealloc reserved slot backing
9. 开启 runtime reserved remap/prefetch
10. 开始处理正式请求
```

本轮使用：

```text
reserved_slot_count = 96
reserved_slot_bytes = 4194304
reserved_backing_bytes = 402653184
reserved_slot_policy = dynamic
reserved_bank_count = 1
reserved_cleanup_delay_steps = 4
```

当前 UVM/Python binding 仍有 128 slot ABI 上限，因此本轮未使用 `slot_count >= 480`。

### 2.5 Trace 设置

性能组关闭主要诊断 trace：

```text
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0
VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE=0
VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY=0
VLLM_UVM_WEIGHT_MAP_ENABLE=0             # static 组
VLLM_UVM_MOE_ADDRESS_MAP_ENABLE=0        # static 组
```

注意：`dynamic real-load` 会先跑一轮 pilot routing trace 来构造真实负载 offload plan。该 pilot 只用于生成 plan，不计入最终性能结果。真正的 dynamic mechanism run 是低 trace 口径。

## 3. 四组实验定义

### 3.1 Baseline: expert+KV managed, no offload

目的：给新机制分配策略提供无卸载性能基线。

```text
expert 主权重 = UVM managed
KV cache = UVM managed
Stage-M offload = disabled
reserved slot = disabled
runtime prefetch/remap = disabled
trace = disabled
```

本轮使用 warm baseline 作为公平对比。冷 baseline 的 load 阶段出现明显 checkpoint I/O 异常，load 时间约 245s，不用于性能结论。

### 3.2 Dynamic real-load + offload + slot

目的：使用真实 routing trace 生成 active expert offload plan，测试真实命中下 runtime reserved slot 路径的性能。

步骤：

```text
1. pilot run: 关闭 offload/slot，只打开 routing trace。
2. build_offload_plan_from_routing_trace.py 从真实 routing trace 中构造 plan。
3. 每层最多选择 2 个 active experts。
4. mechanism run: 关闭 trace，执行 Stage-M offload + post-Stage-M slot prealloc + runtime reserved remap/prefetch。
```

plan 规模：

```text
offload_plan_records = 96
prefetch_plan_records = 96
Stage-M init offload = 192 role records
Stage-M init offload bytes = 452,984,832
post-Stage-M prealloc = 96/96 success
```

### 3.3 Static tail5 0%

目的：静态卸载每层最后 5 个 expert，但 synthetic routing 保证请求阶段不命中 tail expert，测量“无真实搬运”时的静态 offload 基础开销。

静态卸载计划：

```text
tail_expert_ids = 123,124,125,126,127
layers = 48
offload_plan_records = 240
roles = moe_gate_up,moe_down
Stage-M init offload = 480 role records
Stage-M init offload bytes = 1,132,462,080
tail_hit_prob = 0
post-Stage-M prealloc = 96/96 success
```

本组使用 routing simulator：

```text
VLLM_MOE_ROUTING_SIMULATION_STRATEGY=static_tail_hit
VLLM_MOE_ROUTING_STATIC_TAIL_HIT_PROB=0
```

### 3.4 Static tail5 2%

目的：静态卸载每层最后 5 个 expert，并通过 synthetic routing 注入约 2% tail expert 命中，测试少量命中时 reserved slot remap/prefetch 的性能开销。

除命中概率外，本组与 static tail5 0% 相同：

```text
tail_hit_prob = 0.02
offload_plan_records = 240
Stage-M init offload = 480 role records
Stage-M init offload bytes = 1,132,462,080
post-Stage-M prealloc = 96/96 success
```

## 4. 如何复现实验

以下命令假设工作目录为：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
```

### 4.1 准备公共变量和请求 fixture

```bash
export MODEL=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782
export RUN_ROOT=/tmp/vllm_lowtrace_fourway_$(date +%Y%m%d_%H%M%S)
export FIXTURE=${RUN_ROOT}/request_fixture.json
mkdir -p "${RUN_ROOT}"

python3 - "${FIXTURE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
prompt = "Compare CUDA streams and events in one concise paragraph."
payload = {
    "metadata": {"source": "lowtrace_fourway"},
    "requests": [
        {"request": i, "prompt": f"{prompt} [request {i}]", "max_tokens": 16}
        for i in range(8)
    ],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
```

建议运行前确认 GPU 空闲：

```bash
nvidia-smi
```

### 4.2 Baseline: expert+KV managed, no offload

```bash
BASELINE_DIR=${RUN_ROOT}/baseline_expertkv_no_offload_lowtrace_warm
mkdir -p "${BASELINE_DIR}"

cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
VLLM_USE_UVM=0 \
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
VLLM_UVM_KV_CACHE_MANAGED=1 \
VLLM_DISABLE_OPENAI_METRICS=1 \
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
uv run python uvm_test/vllm_generate_sequence.py \
  --model "${MODEL}" \
  --request-fixture-json "${FIXTURE}" \
  --kv-cache-memory-bytes 1073741824 \
  --gpu-memory-utilization 0.90 \
  --output-json "${BASELINE_DIR}/result.json" \
  --progress-jsonl "${BASELINE_DIR}/progress.jsonl" \
  2>&1 | tee "${BASELINE_DIR}/run.log"
```

生成简要 summary 可使用：

```bash
python3 - "${BASELINE_DIR}/result.json" "${BASELINE_DIR}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
req = [float(item["generate_seconds"]) for item in result.get("request_results", [])]
tokens = sum(int(item.get("output_tokens", 0)) for item in result.get("request_results", []))
summary = {
    "status": result.get("status"),
    "returncode": 0,
    "completed_requests": result.get("completed_requests"),
    "load_seconds": result.get("load_seconds"),
    "request_seconds": req,
    "total_generate_seconds": result.get("total_generate_seconds", sum(req)),
    "avg_request_seconds": sum(req) / len(req) if req else None,
    "avg_request_seconds_excluding_first": sum(req[1:]) / len(req[1:]) if len(req) > 1 else None,
    "output_tokens": tokens,
    "tokens_per_second": tokens / result.get("total_generate_seconds", sum(req)) if req else None,
    "gpu_memory_after_load": result.get("gpu_memory_after_load"),
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
```

### 4.3 Dynamic real-load + offload + slot

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=${RUN_ROOT}/dynamic_real_load_lowtrace_96slots \
MODEL="${MODEL}" \
PROMPT_EXACT="Compare CUDA streams and events in one concise paragraph." \
MAX_TOKENS_SEQUENCE=16,16,16,16,16,16,16,16 \
PLAN_MAX_EXPERTS_PER_LAYER=2 \
RESERVED_SLOT_COUNT=96 \
RESERVED_BANK_COUNT=1 \
RESERVED_SLOT_BYTES=4194304 \
RESERVED_EXPERTS_PER_LAYER=2 \
RESERVED_CLEANUP_DELAY_STEPS=4 \
PREALLOC_RESERVED_BEFORE_RUN=0 \
POST_STAGE_M_RESERVED_PREALLOC=1 \
EXPERT_WEIGHTS_MANAGED=1 \
KV_CACHE_MANAGED=1 \
KV_CACHE_MEMORY_BYTES=1073741824 \
GPU_MEMORY_UTILIZATION=0.90 \
TIMEOUT_SEC=1200 \
bash workloads/vllm/vllm/uvm_test/run_vllm_pre_service_real_load_offload_perf.sh
```

该脚本会在自己的 `RUN_ROOT` 下生成：

```text
pilot_trace/routing_trace.jsonl
real_load_offload_plan_raw.json
real_load_offload_plan.json
mechanism_perf/result.json
mechanism_perf/init_trace.jsonl
mechanism_perf/post_stage_m_reserved_prealloc.jsonl
summary.json
```

注意：`pilot_trace` 只用于生成真实负载 plan；性能结论使用 `mechanism_perf/result.json` 和顶层 `summary.json`。

### 4.4 Static tail5 0%, low trace

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=${RUN_ROOT}/static_tail5_0pct_lowtrace_96slots \
MODEL="${MODEL}" \
PROMPT_EXACT="Compare CUDA streams and events in one concise paragraph." \
MAX_TOKENS_SEQUENCE=16,16,16,16,16,16,16,16 \
TAIL_EXPERT_IDS=123,124,125,126,127 \
BASE_EXPERT_IDS=0,1,2,3,4,5,6,7 \
TAIL_HIT_PROB=0 \
RESERVED_SLOT_COUNT=96 \
RESERVED_BANK_COUNT=1 \
RESERVED_SLOT_BYTES=4194304 \
RESERVED_EXPERTS_PER_LAYER=5 \
RESERVED_CLEANUP_DELAY_STEPS=4 \
PREALLOC_RESERVED_BEFORE_RUN=0 \
POST_STAGE_M_RESERVED_PREALLOC=1 \
EXPERT_WEIGHTS_MANAGED=1 \
KV_CACHE_MANAGED=1 \
WEIGHT_MAP_ENABLE=0 \
MOE_ADDRESS_MAP_ENABLE=0 \
ROUTING_TRACE_ENABLE=0 \
PREFETCH_TRACE_ENABLE=0 \
RESERVED_COVERAGE_TRACE_ENABLE=0 \
RESERVED_COVERAGE_RESIDENCY_QUERY=0 \
KV_CACHE_MEMORY_BYTES=1073741824 \
GPU_MEMORY_UTILIZATION=0.90 \
TIMEOUT_SEC=1200 \
bash workloads/vllm/vllm/uvm_test/run_vllm_tail5_static_0pct_diag.sh
```

虽然脚本名包含 `diag`，但上述环境变量已经关闭诊断 trace，使其成为 low-trace 性能口径。

### 4.5 Static tail5 2%, low trace

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=${RUN_ROOT}/static_tail5_2pct_lowtrace_96slots \
MODEL="${MODEL}" \
PROMPT_EXACT="Compare CUDA streams and events in one concise paragraph." \
MAX_TOKENS_SEQUENCE=16,16,16,16,16,16,16,16 \
TAIL_EXPERT_IDS=123,124,125,126,127 \
BASE_EXPERT_IDS=0,1,2,3,4,5,6,7 \
TAIL_HIT_PROB=0.02 \
RESERVED_SLOT_COUNT=96 \
RESERVED_BANK_COUNT=1 \
RESERVED_SLOT_BYTES=4194304 \
RESERVED_EXPERTS_PER_LAYER=5 \
RESERVED_CLEANUP_DELAY_STEPS=4 \
PREALLOC_RESERVED_BEFORE_RUN=0 \
POST_STAGE_M_RESERVED_PREALLOC=1 \
EXPERT_WEIGHTS_MANAGED=1 \
KV_CACHE_MANAGED=1 \
WEIGHT_MAP_ENABLE=0 \
MOE_ADDRESS_MAP_ENABLE=0 \
ROUTING_TRACE_ENABLE=0 \
PREFETCH_TRACE_ENABLE=0 \
RESERVED_COVERAGE_TRACE_ENABLE=0 \
RESERVED_COVERAGE_RESIDENCY_QUERY=0 \
KV_CACHE_MEMORY_BYTES=1073741824 \
GPU_MEMORY_UTILIZATION=0.90 \
TIMEOUT_SEC=1200 \
bash workloads/vllm/vllm/uvm_test/run_vllm_tail5_static_0pct_diag.sh
```

### 4.6 汇总结果

复现后可以用下面的思路汇总各组 `summary.json`：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/tmp/vllm_lowtrace_fourway_20260712_123124")
cases = {
    "baseline": root / "baseline_expertkv_no_offload_lowtrace_warm" / "summary.json",
    "dynamic_real_load": root / "dynamic_real_load_lowtrace_96slots" / "summary.json",
    "static_tail5_0pct": root / "static_tail5_0pct_lowtrace_96slots" / "summary.json",
    "static_tail5_2pct": root / "static_tail5_2pct_lowtrace_96slots" / "summary.json",
}

for name, path in cases.items():
    data = json.loads(path.read_text())
    if name == "dynamic_real_load":
        result = data["mechanism"]
        avg = result["avg_request_seconds"]
        p50 = result["p50_request_seconds"]
        tps = result["tokens_per_second"]
        load = result["load_seconds"]
        completed = result["completed_requests"]
    elif name.startswith("static"):
        result = data["result"]
        req = result["request_seconds"]
        avg = sum(req) / len(req)
        p50 = sorted(req)[len(req) // 2 - 1]
        tps = result["output_tokens"] / result["total_generate_seconds"]
        load = result["load_seconds"]
        completed = result["completed_requests"]
    else:
        avg = data["avg_request_seconds"]
        p50 = data.get("p50_request_seconds") or sorted(data["request_seconds"])[len(data["request_seconds"]) // 2 - 1]
        tps = data["tokens_per_second"]
        load = data["load_seconds"]
        completed = data["completed_requests"]
    print(f"{name}: completed={completed}, avg={avg:.3f}s, p50={p50:.3f}s, tokens/s={tps:.2f}, load={load:.2f}s")
PY
```

## 5. 本轮实验结果

公平对比使用 warm baseline。冷 baseline 目录为：

```text
/tmp/vllm_lowtrace_fourway_20260712_123124/baseline_expertkv_no_offload_lowtrace
```

该轮 load 时间约 245s，判断为 checkpoint I/O 异常，不参与结论。

### 5.1 性能表

| 方案 | 状态 | avg/request | p50/request | tokens/s | load_seconds | GPU after load | 关键机制 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline, expert+KV managed, no offload | ok, 8/8 | 1.210s | 1.171s | 13.22 | 38.51s | 31745 MiB | 不卸载 |
| dynamic real-load + offload + slot | ok, 8/8 | 2.208s | 2.146s | 7.25 | 39.21s | 32109 MiB | 96 layer-expert plan, 453MB offload |
| static tail5 0% | ok, 8/8 | 1.511s | 1.477s | 10.59 | 52.97s | 32109 MiB | 每层卸载 123-127, 0% tail hit |
| static tail5 2% | ok, 8/8 | 2.296s | 2.259s | 6.97 | 41.26s | 32109 MiB | 每层卸载 123-127, 2% synthetic tail hit |

### 5.2 单请求时间

```text
baseline warm:
[1.483, 1.188, 1.220, 1.171, 1.160, 1.151, 1.146, 1.159]

dynamic real-load:
[2.606, 2.125, 2.176, 2.146, 2.145, 2.202, 2.140, 2.120]

static tail5 0%:
[1.792, 1.464, 1.463, 1.464, 1.456, 1.481, 1.477, 1.492]

static tail5 2%:
[2.583, 2.264, 2.254, 2.259, 2.273, 2.246, 2.252, 2.237]
```

### 5.3 Offload 和 slot 状态

```text
dynamic real-load:
  offload_plan_records = 96
  prefetch_plan_records = 96
  init_offload_records = 192
  init_offload_bytes = 452,984,832
  post_stage_m_prealloc = 96/96 success
  reserved_backing_bytes = 402,653,184

static tail5 0%:
  offload_plan_records = 240
  prefetch_plan_records = 240
  init_offload_records = 480
  init_offload_bytes = 1,132,462,080
  post_stage_m_prealloc = 96/96 success
  reserved_backing_bytes = 402,653,184
  runtime_prefetch_trace_records = 0
  routing_records = 0
  weight_map_records = 0
  address_map_records = 0

static tail5 2%:
  offload_plan_records = 240
  prefetch_plan_records = 240
  init_offload_records = 480
  init_offload_bytes = 1,132,462,080
  post_stage_m_prealloc = 96/96 success
  reserved_backing_bytes = 402,653,184
  runtime_prefetch_trace_records = 0
  routing_records = 0
  weight_map_records = 0
  address_map_records = 0
```

### 5.4 Fault / Xid 检查

本轮低 trace 性能组没有观察到新增 Xid 或 GPU MMU fault。`xid_tail` 中仍可能包含早前实验留下的历史 Xid 31 / MMU Fault 行，但本轮静态组 `new_xid_or_mmu_fault=false`，动态组的 Xid excerpt 也是历史日志残留。

## 6. 结果解释

### 6.1 静态方案不是天然有严重性能损失

早前诊断口径下：

```text
static tail5 0% ~= 2.94s/request
dynamic real-load ~= 1.81s/request
```

这看起来像静态方案本身更慢。但诊断口径打开了大量记录：routing trace、prefetch trace、weight map、address map 等。关闭 trace 后：

```text
static tail5 0% = 1.511s/request
dynamic real-load = 2.208s/request
```

因此不能把早前慢速归因于“静态方案天然有损性能”。更合理的解释是：早前 static 0% 的主要开销来自诊断 trace 和地址/权重图谱记录。

### 6.2 0% 命中仍慢于 baseline

即使命中概率为 0，static tail5 0% 仍比 baseline 慢：

```text
1.511 / 1.210 = 1.25x
```

这部分不能归因于真实 expert 搬运，因为 tail hit 为 0，且 runtime prefetch trace 记录为 0。可能来源包括：

```text
1. 更大的 Stage-M offload 状态和 plan 元数据。
2. reserved slot pool 存在后的 runtime 判定路径。
3. static routing simulator 改变了 MoE top-k 形态和 kernel 工作分布。
4. 卸载后 managed expert residency 状态带来的轻量级判定成本。
```

这部分需要进一步拆分，但它已经不是早前 2x 以上的异常慢。

### 6.3 命中 offloaded expert 后性能明显下降

static tail5 2% 与 dynamic real-load 接近：

```text
static tail5 2% = 2.296s/request
dynamic real-load = 2.208s/request
```

这符合当前机制的性能模型：只要请求阶段命中已卸载 expert，就会触发 reserved slot remap/prefetch/sync/cleanup/reuse 路径，性能明显低于 no-offload baseline。

### 6.4 当前瓶颈判断

结合之前的 expert-only managed 和 pre-service/runtime-native 实验，本轮进一步支持以下判断：

```text
1. expert 主权重和 KV cache 使用 managed 分配，本身不必然导致请求阶段严重变慢。
2. 非 expert 常驻权重和 runtime 临时 buffer 不进入 UVM 后，之前的 FAULT_PDE 邻近 VA 风险被移除。
3. 当前主要性能瓶颈在 runtime reserved slot remap/prefetch 路径。
4. 当 offloaded expert 不被命中时，静态方案性能可以明显好于真实命中动态方案，但仍有约 25% 的基础开销。
```

## 7. 注意事项和后续建议

1. `dynamic real-load` 的 pilot trace 不应计入性能，只用于构造真实 active expert plan。
2. `static tail5 0%/2%` 使用 synthetic routing simulator，输出文本和真实 routing 语义不等价，适合作为机制压测和性能分解，不应直接代表真实用户负载。
3. 本轮 slot count 为 96，受当前 ABI 上限影响，未做到覆盖 static tail5 的 480 role records。虽然本轮未卡死，但后续若要降低 slot reuse 压力，仍应考虑扩展 ABI 或改造 slot 管理。
4. 若要进一步定位 static tail5 0% 相对 baseline 的 25% 开销，建议新增一个 no-hit real routing plan：从真实 routing trace 选择未命中的 cold experts，而不是使用 routing simulator。
5. 若要优化 dynamic/static 2% 的请求性能，下一步重点应放在 runtime reserved prefetch/remap 的批量化、异步化、按层复用策略和 cleanup/reuse 减少同步上。
