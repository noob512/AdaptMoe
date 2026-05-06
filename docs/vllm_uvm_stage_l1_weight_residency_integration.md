# vLLM UVM Stage L1 Weights Expert Residency Integration 实现说明

Stage L1 在 Stage L0 的 PoolResidencyManager 基础上，把 Stage I 的 MoE expert weight slice 接入 residency 账本。该阶段仍然是保守 trace-only 集成：记录和判断 estimated GPU residency，但不因为 residency deny 改变 Stage I 的 prefetch/offload 行为。

## 1. 已实现内容

修改文件：

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/check_stage_l1_success.py
```

### 1.1 Expert Slice Region

每个 MoE expert weight slice 使用统一 region id：

```text
weights:<layer_id>:<global_expert_id>:<role>
```

示例：

```text
weights:12:37:moe_gate_up
weights:12:37:moe_down
```

metadata 包含：

```text
source=stage_i
stage=stage_l1
layer_name
layer_id
global_expert_id
local_expert_id
role
ptr
bytes
plan_hot
plan_cold
active_this_step
source_action
```

## 2. Hot Prefetch 接入

在 Stage I `_maybe_prefetch_uvm_expert_weights()` 中，针对每个 active expert slice：

```text
1. register weights region。
2. active=True，evictable=False。
3. record access。
4. request weights GPU residency。
5. trace 中写入 residency_region_id / residency_would_deny / victims。
6. 如果 Stage I 真实发出 prefetch，则 mark state=prefetching_gpu。
```

L1 不会因为 `residency_decision.allowed=False` 跳过 Stage I prefetch。这个行为留给 L2 enforce。

## 3. Cold Offload Candidate 接入

在 Stage I `_maybe_offload_uvm_cold_expert_weights()` 中，针对 cold expert offload candidate：

```text
1. register weights region。
2. active=False，evictable=True。
3. 如果 Stage I 真实发出 advise_cpu/prefetch_cpu，则 mark state=prefetching_cpu。
```

这使 L1 可以证明 same-pool victim candidate 只来自 weights pool，且 active expert 不会被选为 victim。

## 4. 检验脚本

新增：

```text
workloads/vllm/check_stage_l1_success.py
```

快速自测：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
./workloads/vllm/check_stage_l1_success.py
```

真实 GPU probe：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l1_success.py --gpu-run --mode trace_only --weight-gpu-bytes 1
```

成功标准：

```text
stage_i_weight_region_records > 0
stage_i_weight_request_records > 0
active_expert_access_records > 0
would_deny_records_present_when_budget_small=True
active_experts_not_victim_candidates=True
no_real_residency_actions=True
```

## 5. 当前边界

Stage L1 不做：

```text
1. weights residency enforce。
2. residency victim 的真实 offload。
3. KV cache sizing。
4. KV block data offload/prefetch-back。
5. scratch allocator hot path 接入。
```

这些属于后续 L2/L3/L4/L5/L6。
