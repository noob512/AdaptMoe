# vLLM UVM Stage L2 Weights Residency Enforce 实现说明

Stage L2 是 Stage L 中第一个真正让 weights estimated GPU residency budget 影响 Stage I 行为的阶段。目标是：hot expert prefetch 前先请求 weights residency；如果超预算，只从 weights pool 中选择 cold/evictable expert victim；victim offload 后重试 hot request；仍失败则跳过显式 prefetch，让正常 compute 通过 UVM fault 兜底。

## 1. 已实现内容

修改文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_l2_success.py
```

### 1.1 Residency Manager

新增 L2 配置 trace 字段：

```text
VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE
VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET
VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD
VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP
VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_POLICY
VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_ACTION
```

新增 `record_uvm_residency_action()`，用于记录 L2 的真实 policy action：

```text
residency_weight_victim_offload_issued
residency_weight_victim_offload_failed
residency_weight_victim_skip_active
residency_weight_victim_skip_no_ptr
residency_weight_hot_prefetch_skipped
```

### 1.2 Stage I Hot Prefetch 路径

在 `_maybe_prefetch_uvm_expert_weights()` 中：

```text
1. 计算当前 step 全量 active expert region ids。
2. 注册 Stage H/offload plan 中的 cold expert 为 evictable weights regions。
3. 对 hot expert slice request weights GPU residency。
4. enforce 且超预算时，只使用 request 返回的 weights victims。
5. 对 victims 执行 advise_cpu 或 prefetch_cpu。
6. victim 成功后根据 sync_offload 配置释放 estimated bytes。
7. 重试 hot request。
8. 仍失败则 trace residency_weight_hot_prefetch_skipped，并跳过显式 prefetch。
```

安全边界：

```text
不会选择 active expert victim。
不会操作 KV pool。
不会操作 scratch eviction。
不会驱逐 dense/shared weights。
```

## 2. 验收脚本

新增：

```text
workloads/vllm/check_stage_l2_success.py
```

快速自测：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
./workloads/vllm/check_stage_l2_success.py
```

真实 GPU probe：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_l2_success.py --gpu-run --mode enforce --weight-gpu-bytes 1
```

成功标准：

```text
weight_enforce_enabled=True
weights_estimated_gpu_resident_bytes <= weight_gpu_budget_bytes
cold_weight_victim_records > 0
weight_offload_issued_records > 0
hot_prefetch_request_records > 0
active_weight_victim_records == 0
kv_action_records == 0
scratch_eviction_records == 0
failed_requests == 0
```

## 3. 当前边界

L2 只限制 weights estimated residency，不会自动增加 KV cache block 数。KV cache sizing 属于 L4。

L2 的 residency bytes 是应用层 managed intent，不是 CUDA driver 精确 resident page bytes。UVM driver 仍可能因 page fault 被动迁移页面。
