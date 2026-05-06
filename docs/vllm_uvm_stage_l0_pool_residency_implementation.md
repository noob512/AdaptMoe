# vLLM UVM Stage L0 Pool Residency Manager 实现说明

Stage L0 是 Stage L 的第一步：只实现 estimated GPU residency 账本和 trace，不执行真实 reclaim、offload、prefetch 或 KV block 迁移。

## 1. 已实现内容

新增模块：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
```

能力：

```text
1. 读取 Stage L0 residency 环境变量。
2. 注册 weights / KV / scratch region。
3. 维护 region state。
4. 维护 per-pool estimated_gpu_resident_bytes。
5. 维护 global estimated_gpu_resident_bytes。
6. 对 GPU residency request 输出 grant / deny / would_deny。
7. trace-only 下不改变任何 vLLM 行为。
8. 输出 JSONL trace 和 summary。
```

重要语义：

```text
estimated_gpu_resident_bytes 是应用层估算/意图账本，
不是 CUDA driver 级真实 page residency。
```

## 2. 环境变量

```text
VLLM_UVM_RESIDENCY_ENABLE=0|1
VLLM_UVM_RESIDENCY_MODE=trace_only|enforce
VLLM_UVM_RESIDENCY_TRACE_FILE=/tmp/vllm_uvm_pool_residency_stage_l.jsonl

VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES=0
VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES=0
VLLM_UVM_RESIDENCY_KV_GPU_BYTES=0
VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES=0
VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES=0
VLLM_UVM_RESIDENCY_PRIORITY=kv,scratch,weights
```

预算为 `0` 表示不限额。

## 3. Trace action

```text
residency_config
residency_register_region
residency_request
residency_state_update
residency_access
residency_would_evict_candidate
residency_summary
```

L0 不应出现：

```text
residency_offload_issued
residency_prefetch_issued
residency_sync
```

## 4. 验收脚本

新增：

```text
workloads/vllm/check_stage_l_success.py
```

默认 self-test：

```bash
cd workloads/vllm
./check_stage_l_success.py
```

enforce 模式 self-test：

```bash
cd workloads/vllm
./check_stage_l_success.py --mode enforce
```

self-test 会构造：

```text
weights regions
KV region
scratch region
```

并验证：

```text
1. residency_config 存在。
2. 三池 region 都能注册。
3. state update 能正确增减 estimated bytes。
4. 小预算会产生 would_deny。
5. candidate 只来自同 pool。
6. summary 中三池 estimated bytes 正确。
7. L0 没有真实 offload / prefetch / sync action。
```

## 5. 当前边界

Stage L0 不做：

```text
1. Stage I weights expert prefetch/offload 接入。
2. weights hot/cold enforce。
3. scratch allocator C++ hot path 接入 Python manager。
4. KV cache sizing。
5. KV block data offload。
6. KV prefetch-back。
7. global reclaim。
```

这些属于后续 L1/L2/L3/L4/L5/L6。
