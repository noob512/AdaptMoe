# vLLM UVM Memory Pool 修改总览与学习指南

本文档汇总当前对 vLLM UVM memory pool 演进链路的主要修改，覆盖 Stage C2、Stage D/D2、Stage E0/E1/E2/E3。它面向后续接手该项目的开发者，目标是帮助读者理解“为什么要改、改了哪里、怎么验证、下一步还能怎么做”。

本文档只关注本轮围绕 vLLM、UVM allocator、runner、summary/check 脚本和文档的修改。

## 1. 总目标

本轮修改的核心目标是把原本混在同一个 UVM managed memory 空间中的对象逐步拆成可观测、可预算、可验证的逻辑 pool：

```text
GPU / UVM memory logical pools
├── Runtime scratch pool
│   └── attention / moe / model_forward 中的短生命周期临时分配
├── KV cache pool
│   └── initialize_kv_cache 阶段创建的 KV tensors
└── Weights pool
    ├── load_model 阶段创建的长期常驻权重
    └── MoE expert weights 的后续 hot/cold 分析基础
```

为什么要这样拆：

1. runtime scratch 通常生命周期短、CPU 访问风险低，适合优先尝试 `device_direct`。
2. KV cache 有 request/block table 语义，不能由 allocator 私自驱逐。
3. weights 有 layer/expert/tensor 语义，不能只靠 allocation size 做 offload。
4. 最终目标是做到“哪个 pool 超载，只处理哪个 pool”，而不是让 KV、weights、scratch 在同一个 UVM 空间里互相挤压且无法解释。

当前完成的是逻辑分区和观测/预算基础，不等于已经完成全部 runtime eviction/offload/prefetch。

## 2. 修改阶段总览

| 阶段 | 目标 | 当前状态 | 关键结果 |
| --- | --- | --- | --- |
| Stage C1 | runtime scratch 真实 `device_direct`，并受总预算约束 | 已实现 | attention gap-hot scratch 可走 `cudaMalloc` |
| Stage C2 | `device_direct` 支持 `cuda_malloc_async` backend 和 CUDA mempool threshold | 已实现 | async backend 可用，可验证 pool config |
| Stage D0/D1 | KV cache 独立预算遥测和 soft signal | 已实现 | allocator 能统计 KV live/peak/over/reject |
| Stage D2 | vLLM 语义层 KV 初始化预算硬约束 | 已实现 | `enforce` 下 KV blocks 会在分配前被 cap |
| Stage E0/E1 | weights 独立预算遥测和 soft signal | 已实现 | allocator 能统计 weights live/peak/over/reject |
| Stage E2 | weight tensor semantic address map | 已实现 | 权重地址可关联 tensor/layer/expert/role |
| Stage E3 | MoE expert routing aggregate trace | 已实现，可选开启 | 可记录每层每步 expert token counts |

## 3. Stage C2：`cuda_malloc_async` device-direct backend

### 3.1 背景

Stage C1 已经证明，在严格 gating 下，部分 attention runtime scratch 可以从 managed allocation 改为真实 GPU-only allocation，也就是 `placement_backend=device_direct`。

C2 的目标是让 device-direct backend 从固定 `cudaMalloc/cudaFree` 扩展为可选：

```text
cuda_malloc
cuda_malloc_async
```

并支持配置 CUDA default mempool 的 release threshold。

### 3.2 关键参数

runner 参数：

```bash
--uvm-device-direct-backend cuda_malloc|cuda_malloc_async
--uvm-device-direct-pool-release-threshold <bytes>
```

环境变量：

```text
VLLM_UVM_DEVICE_DIRECT_BACKEND
VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD
```

### 3.3 allocator 关键逻辑

修改位置：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

核心状态：

```text
device_direct_backend
device_direct_pool_release_threshold_set
device_direct_pool_release_threshold
device_direct_pool_config_attempted
device_direct_pool_config_success
device_direct_pool_config_error
```

当 backend 是 `cuda_malloc_async` 且用户设置了 release threshold 时，allocator 会：

1. 通过 `cudaDeviceGetDefaultMemPool()` 获取默认 mempool。
2. 通过 `cudaMemPoolSetAttribute(... cudaMemPoolAttrReleaseThreshold ...)` 配置 threshold。
3. 在 trace/session summary 中记录 attempted/success/error。

### 3.4 验证要点

核心成功信号：

```text
device_direct_actual_records > 0
device_direct_backend_counts 中出现 cuda_malloc_async
device_direct_pool_release_threshold_set=True
device_direct_pool_config_attempted=1
device_direct_pool_config_success=1
device_direct_pool_config_error=None
device_direct_peak_live_bytes_observed <= device_direct_max_total_bytes
```

一键检查脚本：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_c2_success.py
```

## 4. Stage D0/D1：KV budget telemetry

### 4.1 背景

KV cache 不是普通 allocation。allocator 只知道地址和大小，不知道：

1. 属于哪个 request。
2. 哪些 block 正在被使用。
3. 哪些 block 可以 swap/evict。
4. block table 如何更新。

因此 D0/D1 只在 allocator 侧做遥测和 soft signal，不在 allocator 层执行 KV eviction。

### 4.2 参数

runner 参数：

```bash
--uvm-kv-budget-bytes <bytes>
--uvm-kv-budget-mode trace_only|enforce
```

环境变量：

```text
VLLM_UVM_KV_BUDGET_BYTES
VLLM_UVM_KV_BUDGET_MODE
```

### 4.3 allocator 识别逻辑

修改位置：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

phase 分类：

```cpp
if (phase == "initialize_kv_cache") {
    return AllocationClass::KvPersistent;
}
```

vLLM 侧 phase wrapper：

```text
workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py
```

对应逻辑：

```python
with uvm_allocation_phase("initialize_kv_cache"):
    ...
```

### 4.4 新增指标

allocator 新增：

```text
kv_trace_allocs
kv_requested_bytes
kv_live_bytes
kv_peak_live_bytes
kv_budget_over_allocs
kv_budget_reject_allocs
kv_free_success_allocs
```

trace 字段：

```text
kv_budget_tracked
kv_budget_over_budget
kv_budget_reason
kv_live_bytes
kv_budget_bytes
kv_budget_remaining
kv_budget_mode
```

reason：

```text
not_kv
kv_budget_unlimited
kv_budget_within_budget
kv_budget_exceeded_trace_only
kv_budget_exceeded_soft_enforce
```

### 4.5 验证脚本

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d_success.py
```

或：

```bash
KV_BUDGET_BYTES=1048576 KV_BUDGET_MODE=trace_only ./run_stage_d_kv_budget_check.sh
```

成功标准：

```text
kv_trace_allocations > 0
kv_peak_live_bytes_observed > 0
kv_budget_bytes 与配置一致
kv_budget_mode 与配置一致
小 budget 下 kv_budget_over_records > 0
trace_only 下 kv_budget_reject_records == 0
runner log clean
```

## 5. Stage D2：KV 初始化预算硬约束

### 5.1 背景

D0/D1 的 allocator soft signal 能告诉我们 KV 超预算，但不能阻止 KV 初始化占用过大。D2 把 enforcement 上移到 vLLM KV config 生成路径。

这样做的原因：

1. vLLM KV config 层知道 KV block size。
2. vLLM KV config 层能减少 `num_blocks`。
3. 分配前 cap 比分配后失败更安全。
4. 不会产生半初始化 block table。

### 5.2 修改位置

```text
workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py
workloads/vllm/vllm/vllm/v1/kv_cache_interface.py
```

### 5.3 核心行为

当：

```text
VLLM_UVM_KV_BUDGET_BYTES > 0
VLLM_UVM_KV_BUDGET_MODE=enforce
```

D2 会：

1. 在 `get_kv_cache_configs()` 中 cap available memory。
2. 生成 KV config 后再次检查实际 tensor bytes。
3. 如超预算，按每 block 字节数缩小 `num_blocks`。
4. 如果预算不足以容纳一个 block，初始化前明确失败。
5. 最终由 allocator telemetry 验证 `kv_peak_live_bytes_observed <= budget`。

### 5.4 验证脚本

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d2_success.py
```

或：

```bash
KV_BUDGET_BYTES=2147483648 ./run_stage_d2_kv_budget_check.sh
```

典型成功结果：

```text
kv_budget_mode=enforce
kv_trace_allocations=48
kv_peak_live_bytes_observed <= kv_budget_bytes
kv_budget_over_records=0
kv_budget_reason_counts={'kv_budget_within_budget': 48}
```

## 6. Stage E0/E1：weights budget telemetry

### 6.1 背景

D2 解决了 KV 初始化预算，但 weights 仍然只是在 managed memory 大池中。Stage E0/E1 的目标是先把模型权重从 KV 和 runtime scratch 中独立识别出来。

当前 E0/E1 不做：

1. weights eviction。
2. weights offload。
3. weights prefetch。
4. 模型加载超预算硬失败。

原因是 allocator 不知道权重 tensor 名称、layer、expert，也不知道迁移安全点。

### 6.2 参数

runner 参数：

```bash
--uvm-weight-budget-bytes <bytes>
--uvm-weight-budget-mode trace_only|enforce
```

环境变量：

```text
VLLM_UVM_WEIGHT_BUDGET_BYTES
VLLM_UVM_WEIGHT_BUDGET_MODE
```

### 6.3 allocator 识别逻辑

修改位置：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

phase 分类：

```cpp
if (phase == "load_model") {
    return AllocationClass::WeightPersistent;
}
```

vLLM 侧已经用：

```python
with uvm_allocation_phase("load_model"):
    self.model = model_loader.load_model(...)
```

### 6.4 新增指标

allocator 新增：

```text
weight_trace_allocs
weight_requested_bytes
weight_live_bytes
weight_peak_live_bytes
weight_budget_over_allocs
weight_budget_reject_allocs
weight_free_success_allocs
```

trace 字段：

```text
weight_budget_tracked
weight_budget_over_budget
weight_budget_reason
weight_live_bytes
weight_budget_bytes
weight_budget_remaining
weight_budget_mode
```

reason：

```text
not_weight
weight_budget_unlimited
weight_budget_within_budget
weight_budget_exceeded_trace_only
weight_budget_exceeded_soft_enforce
```

### 6.5 summarizer

修改位置：

```text
workloads/vllm/summarize_gap_watch_metrics.py
```

新增 JSON 字段：

```text
weight_budget_bytes
weight_budget_mode
weight_trace_allocations
weight_requested_bytes
weight_live_bytes
weight_peak_live_bytes_observed
weight_min_budget_remaining_observed
weight_budget_over_records
weight_budget_reject_records
weight_budget_reason_counts
```

### 6.6 验证脚本

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_e_success.py
```

或：

```bash
WEIGHT_BUDGET_BYTES=1048576 WEIGHT_BUDGET_MODE=trace_only ./run_stage_e_weights_budget_check.sh
```

典型成功结果：

```text
weight_budget_bytes=1048576
weight_budget_mode=trace_only
weight_trace_allocations=1314
weight_live_bytes≈31GB
weight_peak_live_bytes_observed≈31GB
weight_budget_over_records=1314
weight_budget_reject_records=0
weight_budget_reason_counts={'weight_budget_exceeded_trace_only': 1314}
```

这证明 weights pool 能被独立识别和度量。

## 7. Stage E2：weight tensor semantic address map

### 7.1 背景

E0/E1 只能回答“weights 总共用了多少内存”。但后续要做 hot/cold weights、MoE expert prefetch/offload，就必须知道：

1. 某段地址属于哪个 tensor。
2. tensor 属于哪一层。
3. tensor 是否是 MoE expert weight。
4. tensor 是 `w1/w2/w3/w13` 还是 attention/norm/embedding。
5. fault address 如何 join 回具体权重。

### 7.2 修改位置

```text
workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py
workloads/vllm/summarize_stage_e_weight_map.py
```

### 7.3 输出文件

新增环境变量：

```text
VLLM_UVM_WEIGHT_MAP_ENABLE=0|1
VLLM_UVM_WEIGHT_MAP_FILE=<jsonl path>
```

runner 参数：

```bash
--uvm-weight-map-enable <0|1>
--uvm-weight-map-file <path>
```

默认由 Stage E checker 写到：

```text
vllm_uvm_weight_regions_stage_e.jsonl
```

### 7.4 JSONL 字段

每条记录表示一个 CUDA tensor 的权重地址范围：

```json
{
  "kind": "weight:param",
  "name": "model.layers.0.mlp.experts.3.w2.weight",
  "start": "0x...",
  "end": "0x...",
  "start_int": 123,
  "end_int": 456,
  "size_bytes": 4096,
  "dtype": "torch.float16",
  "shape": [8, 16],
  "device": "cuda:0",
  "layer_id": 0,
  "expert_id": 3,
  "role": "moe_down",
  "shard_id": "w2",
  "is_moe_expert": true,
  "phase": "load_model",
  "pid": 12345,
  "model": "Qwen/Qwen3-30B-A3B-FP8"
}
```

### 7.5 兼容性设计

原有 address log 仍保持旧格式：

```text
kind,name,start,end,size_bytes,size_mb
```

新增语义信息写入独立 JSONL sidecar，不破坏：

1. `analyze_uvm_fault_addresses.py`
2. `discover_gap_watch.py`
3. 旧的 KV/weight address log 解析逻辑

### 7.6 summary 脚本

新增：

```text
workloads/vllm/summarize_stage_e_weight_map.py
```

输出字段：

```text
weight_map_records
weight_map_total_bytes
weight_map_moe_expert_records
weight_map_moe_expert_bytes
weight_map_layer_count
weight_map_expert_count
weight_map_kind_counts
weight_map_role_counts
weight_map_dtype_counts
weight_map_top_layers
weight_map_top_experts
```

## 8. Stage E3：MoE expert routing trace

### 8.1 背景

有了 weight map 后，还需要知道运行时哪些 expert 被访问。否则只能知道“某个 expert weight 存在”，不能判断它冷热。

E3 在 MoE routing 后记录聚合信息：

1. 哪一层。
2. 第几次 routing step。
3. 本 step 有多少 token。
4. 哪些 expert 被选中。
5. 每个 expert 被分配了多少 token。

### 8.2 修改位置

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
```

插入点：

```text
FusedMoE.select_experts()
```

在 `topk_ids/topk_weights` 生成后，调用 `_log_uvm_moe_routing_trace()`。

### 8.3 参数

环境变量：

```text
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0|1
VLLM_UVM_MOE_ROUTING_TRACE_FILE=<jsonl path>
```

runner 参数：

```bash
--uvm-moe-routing-trace-enable <0|1>
--uvm-moe-routing-trace-file <path>
```

Stage E checker 参数：

```bash
--enable-moe-routing-trace
--require-moe-routing-trace
```

### 8.4 JSONL 字段

示例：

```json
{
  "timestamp": 1.0,
  "pid": 12345,
  "layer_name": "model.layers.0.mlp.experts",
  "step": 0,
  "num_tokens": 4,
  "top_k": 2,
  "global_num_experts": 8,
  "logical_num_experts": 8,
  "local_num_experts": 8,
  "topk_shape": [4, 2],
  "router_logits_shape": [4, 8],
  "expert_token_counts": {"3": 5, "4": 3},
  "active_experts": [3, 4],
  "topk_weight_sum": 4.0,
  "enable_eplb": false
}
```

### 8.5 性能注意

MoE routing trace 默认关闭，因为它会：

1. 对 `topk_ids` 做聚合。
2. 把聚合结果同步到 CPU。
3. 写 JSONL 文件。

它适合实验分析，不适合默认生产路径。

## 9. runner 改动

主 runner：

```text
workloads/vllm/run_kv_fault_ratio.sh
```

新增参数按类别分为三组。

KV：

```bash
--uvm-kv-budget-bytes <n>
--uvm-kv-budget-mode trace_only|enforce
```

Weights budget：

```bash
--uvm-weight-budget-bytes <n>
--uvm-weight-budget-mode trace_only|enforce
```

Weights semantic map：

```bash
--uvm-weight-map-enable <0|1>
--uvm-weight-map-file <path>
```

MoE routing trace：

```bash
--uvm-moe-routing-trace-enable <0|1>
--uvm-moe-routing-trace-file <path>
```

Device-direct backend：

```bash
--uvm-device-direct-backend cuda_malloc|cuda_malloc_async
--uvm-device-direct-pool-release-threshold <n>
```

## 10. checker 和 wrapper

### 10.1 Stage C

```text
workloads/vllm/check_stage_c1_success.py
workloads/vllm/check_stage_c2_success.py
```

用途：

1. 验证 device-direct 真实命中。
2. 验证预算不超。
3. 验证 C2 async backend 和 mempool config。

### 10.2 Stage D

```text
workloads/vllm/check_stage_d_success.py
workloads/vllm/check_stage_d2_success.py
workloads/vllm/run_stage_d_kv_budget_check.sh
workloads/vllm/run_stage_d2_kv_budget_check.sh
```

用途：

1. D0/D1 检查 KV telemetry。
2. D2 检查 KV 初始化预算硬约束。

### 10.3 Stage E

```text
workloads/vllm/check_stage_e_success.py
workloads/vllm/run_stage_e_weights_budget_check.sh
```

默认检查：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_e_success.py
```

检查内容：

```text
weight_trace_allocations > 0
weight_peak_live_bytes_observed > 0
weight_budget_bytes / mode 与配置一致
小 budget 下 weight_budget_over_records > 0
trace_only 下 weight_budget_reject_records == 0
weight_map_records > 0
runner log clean
```

验证 MoE routing trace：

```bash
./check_stage_e_success.py \
  --run-bench \
  --enable-moe-routing-trace \
  --require-moe-routing-trace
```

或：

```bash
RUN_BENCH=1 \
ENABLE_MOE_ROUTING_TRACE=1 \
REQUIRE_MOE_ROUTING_TRACE=1 \
./run_stage_e_weights_budget_check.sh
```

## 11. 如何读一次 Stage E 实验结果

一次典型 Stage E no-bench 输出：

```text
weight_budget_bytes=1048576
weight_budget_mode=trace_only
weight_trace_allocations=1314
weight_live_bytes=31185032525
weight_peak_live_bytes_observed=31185032901
weight_budget_over_records=1314
weight_budget_reject_records=0
weight_budget_reason_counts={'weight_budget_exceeded_trace_only': 1314}
```

解释：

1. `weight_trace_allocations=1314` 说明 `load_model` 阶段的权重 allocation 被捕获。
2. `weight_live_bytes≈31GB` 说明模型权重常驻规模约 31GB。
3. `weight_budget_bytes=1MiB` 是故意设置的小预算，用来触发 over-budget signal。
4. `weight_budget_over_records=1314` 说明所有权重分配都被判定为超预算。
5. `weight_budget_reject_records=0` 是正确的，因为 `trace_only` 不拒绝、不驱逐。
6. `weight_budget_reason_counts` 证明 reason 链路闭环。

如果同时看到：

```text
weight_map_records > 0
weight_map_moe_expert_records >= 0
```

说明 Stage E2 semantic map 已生成。

如果 benchmark + routing trace 下看到：

```text
moe_routing_records > 0
moe_routing_top_experts=[...]
```

说明 Stage E3 expert routing trace 已产生。

## 12. 当前没有实现的内容

当前仍未实现：

1. KV runtime eviction。
2. KV swap/recompute。
3. weights runtime eviction。
4. weights offload。
5. weights predictive prefetch。
6. 三个 pool 的统一 admission control。
7. fault address + weight map + routing trace 的自动 hot/cold join。

尤其要注意：

```text
Stage D2 已经能限制 KV 初始化大小。
Stage E 已经能识别、度量并语义标注 weights。
但“超载时只驱逐对应 pool”还没有完整实现。
```

## 13. 推荐学习顺序

建议后来者按这个顺序读代码：

1. `docs/vllm_uvm_memory_pool_evolution_plan.md`
2. `docs/vllm_uvm_device_direct_stage_c2_async_backend_implementation.md`
3. `docs/vllm_uvm_stage_d_kv_budget_telemetry_implementation.md`
4. `docs/vllm_uvm_stage_e_weights_budget_telemetry_implementation.md`
5. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`
6. `workloads/vllm/run_kv_fault_ratio.sh`
7. `workloads/vllm/summarize_gap_watch_metrics.py`
8. `workloads/vllm/check_stage_d_success.py`
9. `workloads/vllm/check_stage_d2_success.py`
10. `workloads/vllm/check_stage_e_success.py`
11. `workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py`
12. `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`
13. `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`

## 14. 推荐验证顺序

先做轻量验证：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d_success.py
./check_stage_d2_success.py
./check_stage_e_success.py
```

再做 C2 backend 验证：

```bash
./check_stage_c2_success.py
```

最后做需要 benchmark 的 MoE routing trace：

```bash
./check_stage_e_success.py \
  --run-bench \
  --enable-moe-routing-trace \
  --require-moe-routing-trace
```

## 15. 后续开发建议

下一步建议不要直接做全量 weights offload，而是按以下顺序推进：

1. 做 fault address + weight semantic map 的 join。
2. 再把 join 结果与 MoE routing trace 关联。
3. 生成 expert hot/cold ranking。
4. 只对 cold / low-risk expert weights 做 trace-only prefetch/offload 实验。
5. 引入 weights pool admission control，确保 weights 策略不挤占 KV pool。
6. 最后再考虑 runtime eviction/offload 执行器。

这样能保持每个阶段都有明确实验信号，也能避免 allocator 在不了解 vLLM 语义时做危险决策。
