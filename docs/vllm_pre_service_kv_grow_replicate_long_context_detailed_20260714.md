# vLLM pre-service expert release + KV grow 复刻与长上下文验证详细记录

日期：2026-07-14

对应摘要文档：

```text
docs/vllm_pre_service_kv_grow_replicate_long_context_results_20260714.md
```

请求期 all-managed fault observe 复验：

```text
docs/vllm_pre_service_kv_grow_all_managed_fault_observe_results_20260714.md
```

## 1. 实验目的

本实验验证一个安全的 KV cache 扩容路径：

```text
1. vLLM 加载模型并完成 warmup / pre-service 阶段
2. 正式服务请求前，不在 request path 内做 KV 物理迁移
3. 卸载通过固定 routing fixture 收集到的 zero-hit cold experts
4. 将 KV allocator 可见 target 先限制为 1236 blocks
5. 请求开始前，把新增 128 个 KV blocks 物理 prefetch 回 GPU
6. 将可见 KV target 扩大到 1364 blocks
7. 开启 KV-only manual fault guard
8. 正式服务请求
9. 验证请求阶段没有 GPU-side KV page fault
```

这次额外要求验证真实长上下文：

```text
短 fixture:
  用于复刻机制和性能对比

长 prompt fixture:
  prompt 约 20512 tokens
  初始 1236 blocks 不够
  扩容到 1364 blocks 后应能分配到 1237..1283
```

## 2. 关键设计约束

本实验严格遵守下面几个约束：

```text
只让 expert weights 和 KV cache 使用 UVM managed allocation
不让普通 pre-service allocations 使用 UVM managed
不在 request hot path 中做大规模 KV prefetch_cpu / prefetch_gpu
CPU-resident writable KV blocks 不暴露给 allocator
只有 KV block 物理搬回 GPU 并确认 GPU resident 后，才扩大 allocator target
请求开始前开启 KV-only fault guard，而不是 full managed fault guard
```

对应核心环境变量：

```text
VLLM_USE_UVM=0
VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_KV_CACHE_MANAGED=1

VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY=1
VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_TARGET_BLOCKS=1236
VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS=1364

VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE=prefetch_cpu
VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE=prefetch
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1

VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2
```

`POOL_MASK=2` 表示只对 KV driver tag 命中的 managed fault 做 fail-fast / logging。之前 full managed guard 会误伤非 KV managed 地址，所以本实验不使用 full managed guard。

## 3. 机器与模块状态

实验前确认 UVM 模块版本：

```bash
cat /sys/module/nvidia_uvm/srcversion
```

本轮使用：

```text
386A1FA2E18DDDFCE9EFE88
```

重启后若模块版本不对，需在 GPU 空闲时重新加载本仓库构建出的 UVM 模块：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,stat,comm,args | rg 'VLLM|EngineCore|vllm_generate_sequence' || true

sudo rmmod nvidia_uvm || true
sudo insmod /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
cat /sys/module/nvidia_uvm/srcversion
```

实验完成后的状态：

```text
GPU memory.used = 0 MiB
uvm_manual_migration_fault_mode = 0
uvm_manual_migration_fault_pool_mask = 0
no residual VLLM::EngineCore process
```

## 4. 代码路径

主要脚本和代码：

```text
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
workloads/vllm/build_stage_m_hot_cold_weight_plan.py
workloads/vllm/vllm/uvm_test/vllm_generate_sequence.py
workloads/vllm/vllm/vllm/v1/engine/core.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
```

关键功能点：

```text
core.py:
  READY handshake 前执行 KV dynamic pool shrink / grow
  grow_target_blocks <= 0 时忽略，避免误 shrink 到 0

uvm_kv_runtime_policy.py:
  block-level KV physical prefetch_cpu / prefetch_gpu
  VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 后同步等待再查询 residency

vllm_generate_sequence.py:
  请求开始前设置 manual fault guard

run_zero_hit_dynamic_kv_strong_compare.sh:
  统一执行 short / long mechanism case
```

## 5. 实验输入

模型：

```text
/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782
```

短请求 fixture：

```text
/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json

num_requests = 8
max_model_len = 2048
max_tokens total = 512
```

短请求 zero-hit plan：

```text
/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json
```

长上下文 fixture：

```text
/tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json

actual_prompt_tokens_no_special = 20512
max_tokens = 16
tokens_per_block = 16
estimated required KV blocks ~= 1282
```

长上下文 zero-hit plan：

```text
/tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json

routing_records = 1392
layers = 48
cold experts per layer = 5
cold_require_zero_hit = true
offload_plan_records = 240
roles = moe_gate_up, moe_down
```

注意：长上下文 plan 是从长 prompt 自己的 routing trace 生成的，不复用短请求 plan。

## 6. 通用 vLLM 参数

短请求：

```text
max_model_len = 2048
max_num_seqs = 1
max_num_batched_tokens = 2048
kv_cache_memory_bytes = 2147483648
gpu_memory_utilization = 0.90
```

长上下文：

```text
max_model_len = 21840
max_num_seqs = 1
max_num_batched_tokens = 2048
kv_cache_memory_bytes = 2147483648
gpu_memory_utilization = 0.90
```

曾尝试 `max_model_len=22000`，但 vLLM 初始化阶段失败：

```text
2.01 GiB KV cache is needed
available KV cache memory is 2.00 GiB
estimated maximum model length is 21840
```

这不是 UVM/offload 失败，而是 vLLM 自身 KV sizing 检查失败。

## 7. 实验分组

本轮实际记录了 4 组：

```text
short clean baseline:
  all experts resident
  full 2 GiB KV budget from startup
  no expert release
  no KV dynamic pool
  no manual fault guard

short mechanism:
  release short fixture zero-hit tail5 experts
  KV target 1236 -> 1364
  added 128 KV blocks physical GPU prefetch before request
  KV-only manual fault guard before request

long clean baseline:
  all experts resident
  full 2 GiB KV budget from startup
  no expert release
  no KV dynamic pool
  no manual fault guard

long mechanism:
  release long fixture zero-hit tail5 experts
  KV target 1236 -> 1364
  added 128 KV blocks physical GPU prefetch before request
  KV-only manual fault guard before request
```

重要边界：`long clean baseline` 不是“1236-block constrained no-offload”实验。它启动时直接允许 vLLM 使用完整 1364 blocks，所以它能跑长 prompt 不等于在内存压力下不卸载 expert 也能跑。

更强对比应另跑：

```text
constrained no-offload:
  all experts resident
  allocator target 固定为 1236
  no expert release
  no grow

预期:
  20512-token prompt 需要约 1282 blocks，应该出现 KV allocation 不足或无法服务
```

## 8. 运行目录

```text
short clean baseline:
  /tmp/vllm_pre_service_kv_grow_replicate_short_clean_baseline_20260714_191053

short mechanism:
  /tmp/vllm_pre_service_kv_grow_replicate_short_20260714_185952

long clean baseline:
  /tmp/vllm_long_context_clean_baseline_20260714_191321

long mechanism:
  /tmp/vllm_long_context_kv_grow_20260714/mechanism_190548
```

机制组关键文件：

```text
summary.json
zero_hit_offload/result.json
zero_hit_offload/progress.jsonl
zero_hit_offload/init_trace.jsonl
zero_hit_offload/kv_runtime_trace.jsonl
zero_hit_offload/kv_physical_trace.jsonl
zero_hit_offload/residency_trace.jsonl
zero_hit_offload/kv_pointer_map.jsonl
zero_hit_offload/run.log
```

## 9. 实验步骤

### 9.1 预检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

cat /sys/module/nvidia_uvm/srcversion
cat /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode
cat /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_pool_mask

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

ps -eo pid,stat,comm,args | rg 'VLLM|EngineCore|vllm_generate_sequence' || true
```

预期：

```text
srcversion = 386A1FA2E18DDDFCE9EFE88
manual fault mode = 0
manual fault pool mask = 0
GPU memory.used 接近 0 MiB
无残留 EngineCore
```

### 9.2 运行 short clean baseline

```bash
RUN_ROOT=/tmp/vllm_pre_service_kv_grow_replicate_short_clean_baseline_20260714_191053
mkdir -p "$RUN_ROOT"

printf '0\n' | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode >/dev/null
printf '0\n' | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_pool_mask >/dev/null
sudo dmesg > "$RUN_ROOT/dmesg_before.txt"

cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
VLLM_USE_UVM=0 \
VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0 \
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
VLLM_UVM_KV_CACHE_MANAGED=1 \
VLLM_DISABLE_OPENAI_METRICS=1 \
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
VLLM_UVM_KV_RUNTIME_ENABLE=0 \
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
timeout 1800s uv run python uvm_test/vllm_generate_sequence.py \
  --model /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782 \
  --request-fixture-json /tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 2147483648 \
  --gpu-memory-utilization 0.90 \
  --output-json "$RUN_ROOT/result.json" \
  --progress-jsonl "$RUN_ROOT/progress.jsonl" \
  > "$RUN_ROOT/run.log" 2>&1

echo "$?" > "$RUN_ROOT/returncode.txt"
sudo dmesg > "$RUN_ROOT/dmesg_after.txt"
```

### 9.3 运行 short mechanism

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=/tmp/vllm_pre_service_kv_grow_replicate_short_20260714_185952 \
FIXTURE_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/request_fixture.json \
PLAN_SOURCE=/tmp/vllm_pre_service_expert_release_kv_grow_fixed_20260714_182324/zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=8 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=2048 \
KV_CACHE_MEMORY_BYTES=2147483648 \
INITIAL_TARGET_BLOCKS=1236 \
GROW_TARGET_BLOCKS=1364 \
KV_PRE_SERVICE_APPLY=1 \
KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_FORCE_DRIVER_TAG=1 \
KV_DRIVER_TAG_ENABLE=0 \
KV_POINTER_TAG_GRANULARITY=tensor \
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

### 9.4 生成 long zero-hit expert plan

如果已有 long routing trace，可直接执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

python3 workloads/vllm/build_stage_m_hot_cold_weight_plan.py \
  --routing-trace /tmp/vllm_long_context_kv_grow_20260714/pilot_routing_21840/routing_trace.jsonl \
  --output-plan /tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json \
  --report-md /tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan_report.md \
  --hot-experts-per-layer 0 \
  --cold-experts-per-layer 5 \
  --cold-require-zero-hit \
  --fallback-layers 48 \
  --fallback-experts 128 \
  --target-roles moe_gate_up,moe_down
```

输出：

```text
offload_plan_records = 240
```

如果从头生成 routing trace，需要先用 long prompt fixture 跑一次 pilot，打开：

```text
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1
VLLM_UVM_MOE_ROUTING_TRACE_FILE=/tmp/vllm_long_context_kv_grow_20260714/pilot_routing_21840/routing_trace.jsonl
```

pilot 本身不做 expert release，也不做 KV dynamic grow，只用于收集 routing。

### 9.5 运行 long clean baseline

```bash
RUN_ROOT=/tmp/vllm_long_context_clean_baseline_20260714_191321
mkdir -p "$RUN_ROOT"

printf '0\n' | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode >/dev/null
printf '0\n' | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_pool_mask >/dev/null
sudo dmesg > "$RUN_ROOT/dmesg_before.txt"

cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
VLLM_USE_UVM=0 \
VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0 \
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
VLLM_UVM_KV_CACHE_MANAGED=1 \
VLLM_DISABLE_OPENAI_METRICS=1 \
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
VLLM_UVM_KV_RUNTIME_ENABLE=0 \
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
timeout 1800s uv run python uvm_test/vllm_generate_sequence.py \
  --model /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782 \
  --request-fixture-json /tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json \
  --max-model-len 21840 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 2147483648 \
  --gpu-memory-utilization 0.90 \
  --output-json "$RUN_ROOT/result.json" \
  --progress-jsonl "$RUN_ROOT/progress.jsonl" \
  > "$RUN_ROOT/run.log" 2>&1

echo "$?" > "$RUN_ROOT/returncode.txt"
sudo dmesg > "$RUN_ROOT/dmesg_after.txt"
```

### 9.6 运行 long mechanism

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=/tmp/vllm_long_context_kv_grow_20260714/mechanism_190548 \
FIXTURE_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_prompt_fixture.json \
PLAN_SOURCE=/tmp/vllm_long_context_kv_grow_20260714/long_zero_hit_tail5_plan.json \
RUN_CASES=zero_hit_offload \
NUM_REQUESTS=1 \
MAX_MODEL_LEN=21840 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=2048 \
KV_CACHE_MEMORY_BYTES=2147483648 \
INITIAL_TARGET_BLOCKS=1236 \
GROW_TARGET_BLOCKS=1364 \
KV_PRE_SERVICE_APPLY=1 \
KV_PRE_SERVICE_GROW_TARGET_BLOCKS=1364 \
KV_DATA_OFFLOAD_MODE=prefetch_cpu \
KV_PREFETCH_BACK_MODE=prefetch \
KV_FORCE_DRIVER_TAG=1 \
KV_DRIVER_TAG_ENABLE=0 \
KV_POINTER_TAG_GRANULARITY=tensor \
VLLM_UVM_KV_PHYSICAL_PREFETCH_SYNC=1 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2 \
VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2 \
workloads/vllm/run_zero_hit_dynamic_kv_strong_compare.sh
```

## 10. 结果汇总

| case | completed | output tokens | generate seconds | tokens/s | faultish records |
|---|---:|---:|---:|---:|---:|
| short clean baseline | 8/8 | 512 | 24.665 | 20.76 | 0 |
| short mechanism | 8/8 | 512 | 29.202 | 17.53 | 0 |
| long clean baseline | 1/1 | 16 | 4.757 | 3.36 | 0 |
| long mechanism | 1/1 | 16 | 3.466 | 4.62 | 0 |

短请求机制组：

```text
expert plan records = 240
expert released bytes = 1102577664 ~= 1051.5 MiB
expert release failed entries = 0
cold_plan_gpu_bytes: 1102577664 -> 0

initial_target_blocks = 1236
grow_target_blocks = 1364
logical_prefetched_blocks = 128
KV physical CPU success = 128/128
KV physical GPU success = 128/128
prefetch fallback = 0
managed probe failures = 0
allocation_no_free_blocks = 0
max selected KV block id = 40
```

短请求只验证机制闭环，不验证新增 blocks 被真实使用，因为它只分配到 block 40。

长上下文机制组：

```text
expert plan records = 240
expert released bytes = 1093140480 ~= 1042.5 MiB
expert release failed entries = 0
cold_plan_gpu_bytes: 1093140480 -> 0

initial_target_blocks = 1236
grow_target_blocks = 1364
logical_prefetched_blocks = 128
KV physical CPU success = 128/128
KV physical GPU success = 128/128
prefetch fallback = 0
managed probe failures = 0
allocation_no_free_blocks = 0
```

长上下文 allocator 证据：

```text
selected KV blocks total = 1283
max selected KV block id = 1283
selected blocks above initial target 1236 = 47
selected above-target range = 1237..1283
cpu_candidates_max = 0
skipped_dynamic_pool_max = 0
dynamic_pool_target_blocks during allocation = 1364
```

这证明 long prompt 实际用到了 grow 后新增的 KV blocks。

## 11. Fault guard 验证方法

机制组在 `progress.jsonl` 中应看到：

```json
{"action": "pre_request_manual_fault_guard_applied", ...}
```

期望参数：

```text
uvm_manual_migration_fault_mode = 2
uvm_manual_migration_fault_pool_mask = 2
uvm_manual_migration_fault_log_enable = 1
```

检查 dmesg 中是否有新增 GPU fault：

```bash
rg -n "Xid|MMU|FAULT|fault|illegal|manual_migration|GPU page fault|fatal" \
  /tmp/vllm_pre_service_kv_grow_replicate_short_20260714_185952_dmesg_after.txt \
  /tmp/vllm_long_context_kv_grow_20260714/mechanism_190548_dmesg_after.txt
```

本轮结果：

```text
short mechanism:
  new Xid/MMU/fatal/manual_migration fault records = 0

long mechanism:
  new Xid/MMU/fatal/manual_migration fault records = 0
```

因此本轮没有 fault address 或 fault cause 需要解释。

## 12. 结果解析命令

查看机制组 summary：

```bash
jq '.cases.zero_hit_offload | {
  result,
  expert_release: {
    released_bytes: .expert_release.released_bytes,
    failed_entries: .expert_release.failed_entries,
    before_gpu: .expert_release.before.cold_plan_gpu_bytes,
    after_gpu: .expert_release.after.cold_plan_gpu_bytes
  },
  dynamic_pool,
  kv_physical: {
    cpu_success: .kv_physical.physical_trace_cpu_success_records,
    gpu_success: .kv_physical.physical_trace_gpu_success_records,
    fallback: .kv_physical.prefetch_back_fallback,
    managed_probe_failures: .kv_physical.managed_probe_failure_records
  },
  passed_physical_grow
}' /tmp/vllm_long_context_kv_grow_20260714/mechanism_190548/summary.json
```

解析 long allocator 是否使用新增 blocks：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/vllm_long_context_kv_grow_20260714/mechanism_190548/zero_hit_offload/kv_runtime_trace.jsonl")
selected = []
cpu_candidates = []
skipped_dynamic = []
no_free = 0
for line in path.read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    if record.get("action") == "gpu_resident_free_block_select":
        selected.extend(record.get("selected_block_ids") or [])
        cpu_candidates.append(record.get("cpu_candidates") or 0)
        skipped_dynamic.append(record.get("skipped_dynamic_pool") or 0)
    if record.get("action") == "allocation_no_free_blocks":
        no_free += 1

over = [block for block in selected if block > 1236]
print("selected_total", len(selected))
print("max_selected", max(selected))
print("over_1236_count", len(over))
print("over_1236_range", (min(over), max(over)))
print("cpu_candidates_max", max(cpu_candidates))
print("skipped_dynamic_pool_max", max(skipped_dynamic))
print("allocation_no_free_records", no_free)
PY
```

期望输出：

```text
selected_total 1283
max_selected 1283
over_1236_count 47
over_1236_range (1237, 1283)
cpu_candidates_max 0
skipped_dynamic_pool_max 0
allocation_no_free_records 0
```

## 13. 性能解释

短请求：

```text
clean baseline = 20.76 tok/s
mechanism = 17.53 tok/s
下降约 15.6%
```

短请求机制组下降主要来自实验控制路径开销：

```text
KV dynamic pool bookkeeping
pre-service physical migration tracing
fault guard setup
runtime policy checks
额外 trace / summary 聚合
```

因为该计划使用 zero-hit experts，正式请求阶段没有 cold expert restore，所以这不是 runtime expert 搬运导致的下降。

长上下文：

```text
clean baseline = 3.36 tok/s
mechanism = 4.62 tok/s
```

这两个数不应解读为机制组一定更快。原因是 long clean baseline 是 full-KV no-mechanism 对照，而 long mechanism 有不同的 pre-service control path、不同 load/trace 状态和一次性 long prompt 行为。长上下文实验的主要结论是功能正确性：

```text
expanded KV target 被实际使用
没有 GPU-side KV fault
请求完成
```

## 14. 结论

本轮实验成功复刻了 pre-service expert release + KV grow 路径。

已经证明：

```text
1. zero-hit experts 按计划卸载，failed entries = 0
2. KV target 从 1236 grow 到 1364
3. 新增 128 个 KV blocks 完成 CPU/GPU 两段物理迁移
4. VLLM allocator 在 long prompt 中实际分配到 1237..1283
5. 请求开始后没有捕捉到 GPU-side KV fault
6. GPU 和 manual fault guard 在实验结束后恢复干净
```

仍未证明：

```text
某一个 expert 释放出来的具体 PMM chunk
被某一个 KV block 精确复用
```

这需要 PMM chunk identity tracing。

另外，`long clean baseline` 不是 constrained no-offload 失败实验。若要证明“没有 expert offload 时长上下文无法进行，而 expert offload 后可以进行”，下一步需要补跑：

```text
all experts resident
KV visible target 固定 1236
不做 grow
运行同一个 20512-token long fixture
```

该实验预期会因为需要约 1282 KV blocks 而触发 KV allocation 不足或无法服务。
