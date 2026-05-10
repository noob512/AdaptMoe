# Stage M/G Hot-Cold Weight Policy and Driver Executor Plan

Date: 2026-05-10

## 1. 背景

最近几轮 Stage L / Stage C / Stage M 实验已经把问题边界缩小到 weight 策略本身。

在 `stage_m_weight10g_kv2g_ab_hot_weight_policy` 实验中，新的 hot weight policy 已经证明：

```text
Stage M init weight offload records: 1536
Stage G allocator weight CPU offload allocations: 0
Stage G allocator weight CPU offload bytes: 0
OOM runs: 0
```

这说明之前最危险的问题已经被控制住：

```text
Stage M 和 Stage G 不再无限叠加 offload
allocator 不再把超过 weight budget 之后的所有 weight allocation 盲目迁到 CPU
```

同时，scratch / hot gap fault 得到了明显改善：

```text
pure UVM post_gap_faults: 1,387,570
full_impl post_gap_faults: 170,020
gap fault 下降约 87.75%
```

但端到端性能仍然明显差于 pure UVM：

```text
pure UVM output_tok_s: 5.72
full_impl output_tok_s: 1.09
pure UVM mean_tpot_ms: 339.87
full_impl mean_tpot_ms: 1821.40
```

因此当前核心问题已经不是“是否能 offload”，而是：

```text
offload 是否准确
prefetch 是否及时
迁移执行开销是否足够低
是否能保留真正 hot 的 8-10GiB weight 工作集
```

## 2. 当前问题判断

### 2.1 Broad cold plan 不够准确

当前 full implementation 使用的是 broad cold plan。即使已经把每层 offload expert 数量从 96 降到 16，它仍然只是静态地选中一批 expert：

```text
48 layers * 16 experts/layer * 2 roles = 1536 weight records
```

这个上限控制了 offload 规模，但并不能证明这些 expert 真的 cold。

如果被 offload 的 expert 在当前数据集 / prompt / decode 阶段频繁被 router 选中，就会出现：

```text
fault 数下降
但每 token 需要频繁 prefetch / migrate weight
TPOT 明显变差
```

### 2.2 allocator 不能做 hot/cold 语义判断

Stage G allocator 只能看到：

```text
allocation size
allocation class
当前 weight live bytes 是否超过预算
```

它看不到：

```text
这个 expert 是否 hot
当前 token 是否即将使用该 expert
这是 gate_up 还是 down_proj
当前处于 warmup / prefill / decode / moe 哪个阶段
```

因此 allocator 级“超过 10GiB 就 offload 后续 weight allocation”的策略天然不准确，只能作为显式 opt-in 的故障实验路径，不应作为默认策略。

### 2.3 驱动层也不能独立决定 hot/cold

UVM 驱动知道页、地址、fault、迁移和驻留，但不知道 vLLM / MoE 语义。把完整策略搬到驱动层会再次变成基于地址和预算的猜测。

所以新的设计原则是：

```text
vLLM 用户态负责语义决策
C++ / UVM driver 负责低开销批量执行和真实观测
```

## 3. 总体目标

新阶段目标可以称为：

```text
Stage M/G: routing-aware hot/cold weight placement
```

目标不是尽可能多 offload weight，而是稳定保留真正 hot 的 weight 工作集：

```text
保留约 8-10GiB hot / active / unknown weight 在 GPU
只 offload 明确 cold 的 expert
runtime 只 prefetch 即将访问的 expert
Stage M 和 Stage G 不允许无限叠加 offload
驱动层只执行用户态提交的 region action
```

验收目标：

```text
full_impl 不 OOM
allocator_weight_budget_cpu_offload_bytes = 0，除非显式打开实验开关
init_weight_offload_cpu_issued_total 由 routing heatmap 决定，而不是固定 broad plan
residency_weight_estimated_gpu_bytes 接近 8-10GiB
post_gap_faults 维持显著下降
mean_tpot_ms 不再比 pure UVM 恶化数倍
output_tok_s 接近或超过 pure UVM baseline
```

## 4. 推荐架构

```text
vLLM / MoE router
  - 收集 routing trace
  - 统计每层 expert 热度
  - 生成 hot/cold expert plan
  - 决定 init-time offload 和 runtime prefetch

vLLM C++ extension / allocator fast path
  - 注册 weight region
  - 批量执行 advise / prefetch
  - 提供低开销 action API

UVM driver executor
  - 接收已注册 region 的 action
  - 执行批量 prefetch CPU / GPU
  - 查询真实 residency
  - 输出 region 级 fault / residency 统计
```

关键边界：

```text
驱动层不选择 cold expert
驱动层不按 allocation 顺序驱逐 weight
驱动层不猜测 KV / weight / scratch 语义
```

## 5. 实施阶段

## 5.1 Phase A: Routing-aware profile run

先新增一个 profile 模式，用 pure UVM 或 trace_control 运行小负载，收集 MoE routing trace。

输入：

```text
model = Qwen/Qwen3-30B-A3B-FP8
dataset = ShareGPT
prompts = 2/4/8
output_len = 128/256
request_rate = 2
```

输出：

```text
vllm_uvm_moe_routing_trace.jsonl
```

每条记录至少包含：

```json
{
  "ts": 1770000000.0,
  "phase": "decode",
  "layer_id": 12,
  "token_count": 384,
  "expert_ids": [3, 17, 42],
  "topk_weights": [0.71, 0.18, 0.11],
  "request_id": "..."
}
```

如果当前已有 routing trace 能力，优先复用现有字段；不要新建平行格式。

## 5.2 Phase B: Hot/cold expert plan generator

新增 plan 生成脚本，例如：

```text
workloads/vllm/build_stage_m_hot_cold_weight_plan.py
```

输入：

```text
--routing-trace <jsonl>
--weight-map <jsonl>
--output-plan <json>
--hot-weight-bytes 8589934592
--min-hot-experts-per-layer 2
--max-cold-experts-per-layer 16
--target-roles moe_gate_up,moe_down
```

统计口径：

```text
per_layer_expert_hits
per_layer_expert_weighted_score
first_token_hits
decode_hits
prefill_hits
```

排序策略：

```text
score = decode_hits * 1.0 + prefill_hits * 0.5 + first_token_hits * 1.5
```

初始版本可以简单使用 hit count，后续再引入加权。

输出 plan：

```json
{
  "metadata": {
    "policy": "routing_heatmap_v1",
    "hot_weight_bytes": 8589934592,
    "target_roles": ["moe_gate_up", "moe_down"]
  },
  "hot_plan": [
    {
      "layer_id": 0,
      "expert_id": 7,
      "score": 128,
      "reason": "top_hot_by_routing"
    }
  ],
  "offload_plan": [
    {
      "layer_id": 0,
      "expert_id": 89,
      "score": 0,
      "reason": "cold_by_routing"
    }
  ],
  "prefetch_plan": [
    {
      "layer_id": 0,
      "expert_id": 7,
      "reason": "hot_prefetch_candidate"
    }
  ]
}
```

注意：

```text
offload_plan 只允许包含明确 cold expert
没有出现在 routing trace 中但不确定的 expert，可以先标 unknown，不直接 offload
```

## 5.3 Phase C: Stage M init-time placement using hot/cold plan

更新 Stage M：

```text
读取 hot/cold plan
hot_plan 注册为 gpu / hot / protected
offload_plan 注册为 cpu / cold / evictable
unknown expert 默认 gpu 或 unknown，不默认 cpu
```

推荐默认参数：

```text
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=offload_cpu
VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=0
VLLM_UVM_INIT_WEIGHT_MAX_BYTES=0
VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1
VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE=0
```

解释：

```text
max_experts_per_layer 不再作为主要策略
主要策略来自 routing-aware plan
Stage G allocator-level CPU offload 默认关闭
```

需要新增保护：

```text
如果 plan 不是 routing_heatmap_v1，默认禁止大规模 offload
如果 offload_plan 覆盖比例超过阈值，例如每层 > 25%，打印 warning 或拒绝
```

## 5.4 Phase D: Runtime hot prefetch only for upcoming experts

runtime prefetch 不应再使用 broad plan，而应使用 router 当前 step 的 top-k expert。

执行时机：

```text
MoE router 得到 topk_ids 后
expert kernel 运行前
对 topk_ids 对应 weight region 执行 prefetch GPU
```

限制：

```text
每 step 最大 prefetch bytes
每 layer 最大 prefetch experts
只 prefetch 当前 batch 即将访问的 expert
不 prefetch 已经 active/resident 的 expert
```

建议参数：

```text
VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch
VLLM_UVM_WEIGHT_PREFETCH_SOURCE=router_topk
VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP=33554432
VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER=2
```

## 5.5 Phase E: C++ fast path executor

在不改驱动前，先把 Python 层循环执行的 CUDA action 下沉到 C++ extension。

接口建议：

```python
register_weight_region(region_id, ptr, bytes, layer_id, expert_id, role)
batch_prefetch_cpu(region_ids, sync=False)
batch_prefetch_gpu(region_ids, sync=False)
batch_advise_cpu(region_ids)
batch_query_residency(region_ids)
```

第一版可以内部仍调用：

```text
cudaMemAdvise
cudaMemPrefetchAsync
cudaStreamSynchronize
```

目标是降低 Python 循环、JSONL 查询和逐 region API 调用成本。

## 5.6 Phase F: UVM driver executor

当用户态策略准确性验证通过后，再把执行层迁到驱动。

接口形式可以是 procfs 或 ioctl。推荐 ioctl，procfs 只用于 debug。

region 注册：

```c
struct uvm_policy_region {
    uint64_t app_region_id;
    uint64_t start;
    uint64_t length;
    uint32_t pool;      // kv / weight / scratch
    uint32_t layer_id;
    uint32_t expert_id;
    uint32_t role;
    uint32_t flags;     // hot / cold / evictable / protected
};
```

批量 action：

```c
struct uvm_policy_action_batch {
    uint32_t action;    // prefetch_cpu / prefetch_gpu / advise_cpu / query
    uint32_t count;
    uint64_t region_ids[];
};
```

驱动职责：

```text
维护 region table
执行 batch migration / advise
统计 region 级 fault
统计真实 residency
返回失败原因
```

驱动禁止事项：

```text
不根据超预算自动 offload weight
不自行选择 cold expert
不自行覆盖用户态 hot/protected 标记
```

## 6. 实验设计

## 6.1 Routing plan quality experiment

目的：

```text
验证 routing-aware plan 是否比 broad cold plan 更准确
```

流程：

```text
1. pure UVM profile run，收集 routing trace
2. 生成 hot/cold plan
3. 跑 full_impl + routing plan
4. 与 broad cold plan full_impl 对比
```

指标：

```text
init_weight_offload_cpu_issued_total
runtime_weight_prefetch_issued
runtime_weight_offload_issued
residency_weight_estimated_gpu_bytes
delta_after_dedup
delta_kv_after_dedup
post_gap_faults
output_tok_s
mean_tpot_ms
```

期望：

```text
routing plan 的 output_tok_s 高于 broad cold plan
mean_tpot_ms 低于 broad cold plan
post_gap_faults 维持低位
weight runtime offload/prefetch 次数下降或更集中
```

## 6.2 Hot weight budget sweep

目的：

```text
找到 8-10GiB hot weight 保留区间是否合理
```

参数：

```text
hot_weight_bytes = 6GiB, 8GiB, 10GiB, 12GiB
kv_gpu_bytes = 1GiB, 2GiB
scratch profile = 16m
```

预期：

```text
6GiB 可能 hot weight 不足，TPOT 差
8-10GiB 应该是主要候选
12GiB 可能挤压 KV / scratch 空间
```

## 6.3 Executor overhead experiment

目的：

```text
区分策略准确性问题和执行开销问题
```

对比：

```text
A. Python/CUDA per-region action
B. C++ batch action
C. UVM driver batch action
```

指标：

```text
per-token policy overhead
prefetch latency
cuda API call count
TPOT
ITL p99
```

只有当 routing-aware plan 已经优于 broad plan 后，才值得推进 C。

## 7. 脚本规划

新增脚本一：

```text
run_stage_m_routing_profile.sh
```

功能：

```text
运行 pure UVM 或 trace_control
开启 MoE routing trace
输出 routing trace 和 weight map
```

新增脚本二：

```text
build_stage_m_hot_cold_weight_plan.py
```

功能：

```text
读取 routing trace
读取 weight map
生成 routing_heatmap_v1 plan
输出 hot/cold expert 统计报告
```

新增脚本三：

```text
run_stage_m_routing_plan_ab.sh
```

功能：

```text
自动执行 profile -> build plan -> full_impl A/B
对比 pure UVM、broad cold plan、routing-aware plan
```

推荐第一版命令：

```bash
./run_stage_m_routing_plan_ab.sh \
  --log-file log/stage_m_routing_plan_ab.log \
  --hot-weight-bytes 8589934592 \
  --kv-gpu-bytes 2147483648 \
  --scratch-profile 16m \
  --prompts 2 \
  --output-len 128
```

## 8. 验收标准

第一阶段验收：

```text
routing profile 能生成每层 expert heatmap
plan 中 hot/cold expert 可解释
Stage M init offload 不再依赖 broad cold plan
allocator_weight_budget_cpu_offload_bytes = 0
```

第二阶段验收：

```text
residency_weight_estimated_gpu_bytes 接近配置目标，例如 8-10GiB
post_gap_faults 仍比 pure UVM 低 80% 以上
delta_after_dedup 不高于 pure UVM
full_impl mean_tpot_ms 不超过 pure UVM 150%
```

第三阶段验收：

```text
C++ / driver batch executor 相比 Python per-region action 降低 TPOT
runtime prefetch 不产生明显长尾
output_tok_s 接近或超过 pure UVM baseline
```

## 9. 风险和规避

### 风险一：profile 数据集不代表真实请求

规避：

```text
profile 支持多数据集、多 seed、多 prompt 长度
hot/cold plan 记录来源和统计覆盖率
unknown expert 默认不 offload
```

### 风险二：cold expert 在某些请求中突然变 hot

规避：

```text
runtime router_topk prefetch 优先级高于 cold 标记
连续命中 cold expert 时提升为 hot
维护 small hot override cache
```

### 风险三：driver 层执行过于激进

规避：

```text
driver 只执行显式 action
hot/protected region 禁止被 driver 自动迁走
所有 batch action 返回成功/失败计数
保留 trace_only 模式
```

### 风险四：过度追求 fault 降低导致性能下降

规避：

```text
以 output_tok_s / TPOT 作为主指标
fault 只作为解释指标
不接受 fault 下降但 TPOT 大幅恶化的配置作为默认策略
```

## 10. 推荐下一步

优先实现顺序：

```text
1. 新增 routing profile run 脚本
2. 新增 hot/cold plan generator
3. 修改 Stage M 只接受 routing-aware cold plan 做 init offload
4. 修改 runtime weight prefetch 来源为 router_topk
5. 跑 routing_plan_ab 对比 broad cold plan
6. 若策略有效，再实现 C++ batch executor
7. 最后再实现 UVM driver executor
```

当前不建议立即大改 UVM 驱动。原因是：

```text
当前最不确定的是 hot/cold 策略准确性
驱动只能降低执行开销，不能修正错误的 expert 选择
如果把 broad cold plan 下沉到驱动，只会更快地执行不准确策略
```

因此新的工程主线应该是：

```text
routing-aware semantic policy first
driver-level executor second
```

