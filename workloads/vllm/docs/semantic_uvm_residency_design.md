# vLLM Semantic UVM Residency Design

本文档整理 Stage M resident hotset 方向的后续研究方案和代码修改方案。目标不是继续盲目扩大 benchmark 矩阵，而是把 vLLM 的应用层语义转化为 UVM 可执行的 region、budget 和 action，使 KV cache、MoE expert weight、runtime scratch 的 GPU residency 能被解释、控制和验证。

## 1. 结论摘要

当前数据说明这个方向值得继续，但还不能证明机制已经可行。

已经成立或有强信号的部分：

- UVM fault 主要来自 weight，尤其是 MoE expert weight；KV fault 占比很低。
- train/test hotset coverage 显示 expert hotset 有预测性，尤其在 decode-like 小 token event 上很强。
- 当前脚本和配置管线可以跑完 12 个 dry-run case，未出现 OOM。

尚未证明的部分：

- `KV=7/9GiB` 在上一轮 dry-run 中没有生效，最终仍是约 `6GiB`。
- cold expert 没有真正 offload，`init_weight_offload_cpu_issued_total=0`。
- hot expert 没有被真实查询和证明 resident。
- live prefetch、hit-before-access、miss penalty、migration overlap 还没有被验证。

因此下一阶段的核心是：

```text
先让 KV 固定、cold offload、hot residency 真实生效；
再做 live prefetch K sweep；
最后实现在线 hotset / EWMA / bandwidth budget 策略。
```

## 2. 现有实验事实

### 2.1 Fault 归因

已有 fault attribution 实验显示：

- KV cache 不是主要 UVM fault 来源。
- weight / MoE expert weight 是主要 fault 来源。
- `w13_weight` 和 `w2_weight` 是后续策略需要单独建模的 role。

这说明优化方向应优先放在 expert weight residency / prefetch / offload，而不是只优化 KV fault。

### 2.2 Train/Test Hotset Coverage

train/test split 避免了同一段 trace 生成 hotset 再验证的过乐观问题。关键结果：

```text
num_tokens=1:
  hot64 weighted coverage ≈ 96.9%
  hot80 weighted coverage ≈ 99.2%

num_tokens=9-32:
  hot64 weighted coverage ≈ 80.4%
  hot80 weighted coverage ≈ 88.95%
  hot96 weighted coverage ≈ 93.5%

num_tokens>512:
  hot80 weighted coverage ≈ 79%
```

含义：

- decode-like event 可以使用 expert-granular hotset。
- 主压力路径 `num_tokens=9-32` 下，`hot64` 偏小，`hot80/hot96` 更合理。
- 大 prefill/chunked prefill 接近 dense，不适合简单 expert-granular hotset。

### 2.3 Dry-run 暴露的问题

上一轮矩阵：

```text
KV = 5 / 7 / 9 GiB
hotK = 48 / 64 / 72 / 80
streaming buffer = 2 GiB
max_num_seqs = 32
```

全部 case 能跑完，但关键机制没有真正生效。

KV sizing 问题：

```text
KV=5GiB  -> final KV ≈ 5GiB，生效，但 KV fault share 暴涨到约 41%
KV=7GiB  -> final KV ≈ 6GiB，未扩容
KV=9GiB  -> final KV ≈ 6GiB，未扩容
```

cold expert offload 问题：

```text
init_weight_offload_cpu_issued_total = 0
residency_weight_victim_offload_issued = 0
residency_weight_estimated_gpu_bytes ≈ 31.18GiB
```

直接原因：

```text
plan policy = train_test_static_hotset_v1
init offload accepted policies = routing_heatmap_v1, exact_used_oracle_v1
action = init_weight_offload_reject_non_routing_plan
```

因此上一轮性能变化不能解释为 hotset/residency 策略带来的收益。

## 3. 目标机制

最终机制定义为：

```text
固定 KV 池 + expert hot resident + cold expert streaming + 语义反馈闭环
```

具体含义：

- KV cache 按目标大小固定或 override，而不是权重加载后剩多少算多少。
- 每层保留 topK hot expert 的 `w13_weight` / `w2_weight` 在 GPU。
- 非 hot expert 初始 offload 到 CPU 或设置 CPU preferred。
- router 产生 top-k 后，为后续层或下一 step 的 cold expert 做提前 prefetch。
- GPU residency budget 紧张时，保护 active KV、当前层 active expert 和 hot expert。
- 用真实 fault、真实 residency、hit-before-access、TPOT 反馈修正策略。

## 4. 三层系统架构

该机制必须把上层 vLLM 语义和底层 UVM 机制连接起来。

```text
vLLM 应用层语义:
  layer_id / expert_id / role / num_tokens / scheduler step
  active expert union / top-k weight / KV block liveness

UVM 控制层:
  region registry / hot-cold plan / residency state
  budget coordinator / policy engine / action executor

底层 UVM 机制:
  cudaMemAdvise / cudaMemPrefetchAsync / CPU offload
  batch prefetch / real residency query / fault attribution
```

只在 vLLM 层记录 hotset 不够，因为无法控制实际 page residency。只在 UVM 层看 fault 也不够，因为不知道哪些 page 未来会被访问。

## 5. Semantic Region 设计

### 5.1 MoE Expert Weight Region

每个 expert weight role 注册为稳定 region：

```text
region_id = weight:{layer_id}:{expert_id}:{role}
role      = moe_gate_up | moe_down
```

metadata：

```text
layer_id
expert_id
role
ptr
bytes
dtype
is_hot
evictable
active_this_step
last_access_step
last_prefetch_step
estimated_state
real_residency_state
```

`role` 必须进入 key，因为 `w13_weight` 和 `w2_weight` 的体积、fault 频率、迁移代价不同。

### 5.2 KV Cache Region

KV cache 不能只作为一个大地址范围处理。至少需要 tensor + block range 粒度：

```text
region_id = kv:{layer_id}:{kv_tensor_id}:{block_range}
```

metadata：

```text
layer_id
kv_tensor_id
block_start
block_end
request_id / seq_id
is_active
is_prefix_cache
last_access_step
evictable
```

这样 coordinator 才能知道哪些 KV block 当前不能被挤掉。

### 5.3 Runtime Scratch Region

scratch/workspace 需要按 phase 注册：

```text
region_id = scratch:{phase}:{allocation_id}
phase     = attention | moe | model_forward | prefill_step | decode_step
```

metadata：

```text
phase
ptr
bytes
lifespan
reuse_bucket
evictable
```

这能解释早期实验中临时 buffer fault 占比高的问题，也避免 scratch 污染 hot expert pool。

## 6. Plan 和 Policy

### 6.1 Hot/Cold Plan 格式

train/test hotset plan 应包含：

```text
metadata.policy = train_test_static_hotset_v1
metadata.hot_experts_per_layer
metadata.cold_experts_per_layer
metadata.target_roles = [moe_gate_up, moe_down]

prefetch_plan = hot experts
offload_plan  = non-hot / cold experts
```

init offload 需要接受该 policy，因为它来自 routing trace 的 train/test split，而不是手工 plan。

### 6.2 Policy 输入

runtime policy 输入：

```text
current_step
layer_id
num_tokens
num_tokens_bucket
active_expert_union
topk_weight_sum_by_expert
running_requests
waiting_requests
active_kv_blocks
current_residency_budget
recent_faults
```

不要强行使用 prefill/decode 标签作为唯一决策依据。当前更可靠的是 `num_tokens` bucket：

```text
1
2-8
9-32
33-128
129-512
>512
```

### 6.3 Budget 优先级

推荐优先级：

```text
active KV
current layer active expert
next layer predicted active expert
static hot expert
runtime scratch
inactive cold expert
```

当 cold prefetch 会超过 budget：

```text
先 evict cold inactive
不 evict active KV
不 evict current/next active expert
不 evict protected hot expert
```

## 7. 代码修改方案

### 7.1 KV Sizing

现有 `clamp` 只能缩小 KV：

```text
target = min(original_blocks, budget_blocks)
```

需要新增或明确暴露策略：

```text
report_only
clamp_down
override_exact
override_at_least
```

修改点：

- `workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py`
- `workloads/vllm/run_stage_l_full_impl_uvm_ab.sh`
- `workloads/vllm/run_stage_m_resident_hotset_feasibility.sh`

脚本需要支持：

```bash
INIT_KV_SIZING_POLICY=override_exact
```

或直接透传 vLLM 原生参数：

```bash
--kv-cache-memory-bytes <target_bytes>
```

summary 必须增加硬性校验：

```text
target_kv_bytes != init_kv_final_bytes_worker0 => invalid case
```

### 7.2 接受 train/test hotset plan

修改 `run_uvm_init_weight_offload` 的 accepted policy：

```python
accepted_init_plan_policies = {
    "routing_heatmap_v1",
    "exact_used_oracle_v1",
    "train_test_static_hotset_v1",
}
```

同时保留 require plan 机制，避免误用不可信 plan。

成功后必须看到：

```text
init_weight_offload_offload_cpu_issued > 0
init_weight_offload_cpu_issued_total > 0
```

### 7.3 Batch Executor Chunk

driver batch 对过多 ranges 有上限风险。统一引入：

```text
max_ranges_per_driver_batch = 64
```

所有 batch action 都要经过 chunk：

```text
init cold offload
init hot prefetch
runtime cold prefetch
runtime victim offload
real residency query
```

### 7.4 Real Residency Query

summary 需要同时输出估计值和真实查询值：

```text
estimated_gpu_bytes
real_gpu_bytes
real_cpu_bytes
real_unknown_bytes
hot_real_gpu_ratio
cold_real_cpu_ratio
```

不要只用 `estimated_state` 证明机制生效。

### 7.5 Router Semantic Trace

router trace 从离线分析点升级为 runtime policy 输入。

每条 event 需要包含：

```text
step
layer_id
num_tokens
num_tokens_bucket
active_expert_union
expert_token_counts
expert_weight_sums
```

这些字段用于：

```text
hotset EWMA update
runtime prefetch candidates
hit-before-access attribution
num_tokens bucket policy
```

### 7.6 Scheduler / KV Semantic Trace

补充 scheduler 侧信息：

```text
running_requests
waiting_requests
scheduled_prefill_tokens
scheduled_decode_tokens
allocated_kv_blocks
preemptions
```

补充 KV manager 信息：

```text
block_id
request_id
seq_id
active/inactive
prefix_cache
last_access_step
```

这样才能把 KV residency 与实际请求生命周期关联起来。

### 7.7 Layer Timing

增加 CUDA event 或轻量 timer trace：

```text
layer_time_ms
attention_time_ms
moe_time_ms
prefetch_time_ms
num_tokens_bucket
```

H3 的核心判断：

```text
如果 prefetch_latency_ms > layer_time_ms，
则单层计算无法隐藏迁移，需要 lookahead 或降低 K/bytes。
```

## 8. 策略方案

### 8.1 Static Hotset

初始策略：

```text
num_tokens=1:
  hot64/hot80

num_tokens=9-32:
  hot80/hot96

num_tokens>512:
  关闭 expert-granular prefetch，使用 layer-level streaming 或 prefill 特化
```

### 8.2 Online EWMA

最终策略 key：

```text
(layer_id, expert_id, role)
```

分数：

```text
score = EWMA(token_count) + EWMA(topk_weight_sum) + recent_bonus - miss_penalty
```

状态：

```text
last_access_step
last_prefetch_step
resident_state
miss_count
prefetch_hit_count
load_cost_bytes
```

### 8.3 Runtime Prefetch

live prefetch 必须使用：

```text
source = router_topk
```

不要用 `plan_intersection` 做 live cold miss 覆盖，因为它只会在静态 plan 交集里动作。

控制项：

```text
max_experts_per_layer
max_bytes_per_step
lookahead_layers
target_roles
skip_resident
region_cooldown_steps
```

### 8.4 Eviction

eviction policy：

```text
candidate = cold inactive experts
protect = active KV + current active expert + protected hot expert
policy = cold_plan_lru + recent protection
```

指标：

```text
hot_eviction_count
cold_eviction_count
reaccess_after_evict
eviction_bytes/token
```

## 9. 实验计划

### Phase 0: Mechanism Sanity

目的：证明机制真的生效。

矩阵：

```text
KV = 7GiB
hotK = 80
streaming = 2GiB
runtime_prefetch = off
```

必须满足：

```text
final_kv ≈ 7GiB
cold offload issued > 0
hot real GPU ratio > 90%
weight estimated/real GPU bytes 明显下降
OOM = false
```

### Phase 1: KV + Hotset Dry-run

矩阵：

```text
KV = 6 / 7 / 8GiB
hotK = 64 / 80 / 96
streaming = 2GiB
max_num_seqs = 32
runtime_prefetch = off
```

观察：

```text
startup stability
KV blocks
weight resident bytes
fault share
TTFT / TPOT / throughput
```

### Phase 2: Hot Residency Stability

采样点：

```text
after_init
after_prefill
mid_decode
end_decode
```

指标：

```text
hot_real_gpu_ratio
hot_eviction_count
cold_pollution_bytes
reaccess_after_evict
```

### Phase 3: Timing vs Migration

按 num_tokens bucket 统计：

```text
layer_time_ms
moe_time_ms
attention_time_ms
prefetch_latency_ms
```

判断：

```text
能否单层隐藏迁移
是否需要 lookahead_layers
是否需要降低 K 或 max_bytes_per_step
```

### Phase 4: Live Prefetch K Sweep

矩阵：

```text
hotK = 64 / 80 / 96
max_bytes_per_step = 64 / 128 / 256MiB
lookahead_layers = 1 / 2 / 4
max_num_seqs = 16 / 32 / 64
```

指标：

```text
hit-before-access
miss-before-access
same-step-access
prefetch bytes/token
migration bytes/token
TPOT p50/p90/p99
throughput
faults/token
```

### Phase 5: Online Policy

比较：

```text
static hotset
last-step
EWMA
EWMA + recent protection
EWMA + bandwidth budget
EWMA + num_tokens bucket
```

## 10. 成功标准

最终机制可行需要同时满足：

```text
1. KV final bytes 可控，KV blocks 或 max_num_seqs 真实提升。
2. cold expert offload 后 GPU weight residency 明显下降。
3. hot expert real GPU residency > 90%。
4. live prefetch hit-before-access > 80%。
5. UVM weight faults/token 明显下降。
6. throughput 提升 >= 15%-20%。
7. TPOT <= baseline 1.1x，或吞吐收益能抵消 TPOT 上升。
8. 无 workspace OOM / KV allocation failure / cudaMalloc failure。
```

## 11. 风险和回退策略

### H2 风险：hotset 不稳定

如果 `num_tokens=9-32` 下 hot80/hot96 仍不足：

```text
扩大 hotK
按 workload 类型维护多个 hotset
使用在线 EWMA 替代静态 hotset
```

### H3 风险：迁移无法隐藏

如果迁移延迟超过 layer compute：

```text
降低 max_bytes_per_step
只 prefetch w13 或 w2 中收益更高的 role
增加 lookahead_layers
prefill 改 layer-level streaming
```

### UVM 行为风险

如果 real residency query 显示 prefetch 后仍不 resident：

```text
减少 GPU oversubscription
提高 hot protection 优先级
限制 scratch pool
增加 cold eviction
```

### KV 风险

如果扩大 KV 导致 workspace OOM：

```text
降低 KV target
固定 scratch budget
限制 max_num_batched_tokens
减少 hotK 或 streaming buffer
```

## 12. 推荐实施顺序

第一批代码修改：

```text
1. KV override/exact sizing。
2. 接受 train_test_static_hotset_v1。
3. batch executor chunk <=64。
4. summary invalid-case hard gate。
```

第二批代码修改：

```text
1. hot/cold real residency query。
2. weight estimated state 修正。
3. cold offload 成功/失败细粒度统计。
4. hot eviction / reaccess tracing。
```

第三批代码修改：

```text
1. scheduler semantic trace。
2. KV block semantic registry。
3. layer timing trace。
4. num_tokens bucket policy。
```

第四批代码修改：

```text
1. live router_topk prefetch。
2. lookahead prefetch。
3. hot protection eviction。
4. EWMA online hotset。
```

推荐下一步只跑一个机制 sanity case：

```bash
RUN_ROOT=/tmp/hotset_mechanism_sanity \
RUN_TRACE=0 RUN_TRAIN_TEST=0 \
RUN_KV_DRYRUN=1 RUN_HOT_RESIDENCY=1 RUN_LIVE_ABLATION=0 \
KV_BYTES_LIST="7516192768" \
HOT_K_LIST="80" \
STREAMING_BYTES_LIST="2147483648" \
MAX_NUM_SEQS_LIST="32" \
PROMPTS=32 OUTPUT_LEN=64 REQUEST_RATE=32 \
INIT_KV_SIZING_POLICY=override_exact \
workloads/vllm/run_stage_m_resident_hotset_feasibility.sh
```

这个 case 的目标不是性能，而是确认：

```text
KV 真到 7GiB
cold expert 真 offload
hot expert 真 resident
summary 没有 invalid
```

完成后再进入矩阵实验。
