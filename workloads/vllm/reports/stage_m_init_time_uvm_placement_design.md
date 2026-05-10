# Stage M Init-Time UVM Placement Design

Date: 2026-05-09

## 1. 背景和问题

当前 Stage L / Stage C 实验已经证明两点：

1. 运行期可以识别出主要的 UVM fault 热点，尤其是 gap2 这类 MoE / attention 相关 runtime scratch 区域。
2. 在运行期打开 KV、weight、scratch 策略后，仍然可能在 vLLM server ready 之前 OOM。

最新 `full_impl_enforce` A/B 实验中，配置上已经给出了限制：

```text
weight_gpu_bytes = 6 GiB
kv_gpu_bytes     = 1 GiB
scratch budget   = 16 MiB
```

但 OOM 前实际 allocator summary 仍然显示：

```text
Weight live bytes = ~31.18 GB
KV live bytes     = 6.0 GiB
Current allocated = ~37.64 GB
Peak allocated    = ~37.77 GB
```

这说明目前的限制主要是运行期 residency / policy 账本，不能在 vLLM 初始化峰值之前真正降低显存压力。OOM 发生在 MoE Triton kernel 初始化 / warmup 阶段，此时：

```text
KV data offload issued      = 0
KV prefetch back issued     = 0
device_direct actual allocs = 0
scratch eligible allocs     = 0
```

因此下一阶段目标是引入 `Stage M: init-time UVM placement`，把关键限制和迁移动作前移到 warmup 之前。

## 2. 总体结论

不建议把完整策略全部迁入 UVM 驱动层。推荐架构是：

```text
vLLM 用户态：语义决策
UVM / CUDA 执行层：早期迁移、prefetch、advise、统计
```

驱动层知道页、地址、fault 和迁移，但不知道：

```text
这块内存是 KV cache 还是 weight
某个 expert 是 hot 还是 cold
当前阶段是 model load / warmup / prefill / decode / moe
哪些 scratch allocation 可以走 cudaMalloc/cudaMallocAsync
```

这些语义必须由 vLLM、allocator、MoE 层提供。驱动层适合做低开销执行通道和真实 residency / fault 观测。

## 3. Stage M 目标

Stage M 的核心目标是让限制在初始化期真实生效：

```text
加载模型权重
生成 weight region map
应用 KV cache sizing
创建受限规模的 KV cache
执行 cold weight pre-offload
扩展 warmup/prefill scratch direct phase
执行 Triton / MoE warmup
开始服务
```

验收目标：

```text
KV live bytes 从约 6.0 GiB 降到 <= 配置预算，例如 1.0 GiB
weight cold pre-offload 在 warmup 前产生 issued records
scratch direct 在 warmup / prefill 早期出现 actual allocations
full_impl_enforce 可以完成 server ready 和 benchmark
full_impl_enforce 相对 pure_uvm_baseline 至少不 OOM，并能比较吞吐、TPOT、fault
```

## 4. 设计原则

1. 语义留在 vLLM。

   KV block 数、expert hot/cold、scratch phase classification 都由 vLLM 决定。

2. 执行尽量下沉。

   大批量 prefetch / advise 可以先用现有 CUDA runtime helper，后续再下沉到 C++ extension 或 UVM driver ioctl。

3. 初始化期优先。

   所有会影响启动峰值的策略必须发生在 Triton / MoE warmup 前。

4. 保留 trace-only fallback。

   每个机制都需要 `report_only / trace_only` 模式，便于做 A/B 和故障定位。

5. 不在驱动层猜语义。

   驱动层只接收用户态注册的 region 和 action，不根据地址模式自行判断 KV / weight / scratch。

## 5. 新增配置

### 5.1 KV init sizing

```text
VLLM_UVM_INIT_KV_SIZING_ENABLE=0|1
VLLM_UVM_INIT_KV_SIZING_POLICY=report_only|clamp|override
VLLM_UVM_INIT_KV_MAX_BYTES=<int>
VLLM_UVM_INIT_KV_MAX_BLOCKS=<int>
VLLM_UVM_INIT_KV_MIN_BLOCKS=<int>
VLLM_UVM_INIT_KV_TRACE_FILE=<path>
```

语义：

```text
report_only: 只记录原始 blocks / bytes 和建议值
clamp:       final_blocks = min(original_blocks, budget_blocks)
override:    final_blocks = budget_blocks
```

预算换算：

```text
bytes_per_block = scheduler_kv_cache_config.total_size / num_gpu_blocks
budget_blocks_by_bytes = floor(max_bytes / bytes_per_block)
budget_blocks = min_nonzero(max_blocks, budget_blocks_by_bytes)
```

推荐初始参数：

```text
VLLM_UVM_INIT_KV_SIZING_ENABLE=1
VLLM_UVM_INIT_KV_SIZING_POLICY=clamp
VLLM_UVM_INIT_KV_MAX_BYTES=1073741824
VLLM_UVM_INIT_KV_MIN_BLOCKS=128
```

### 5.2 Weight init pre-offload

```text
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0|1
VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=advise_cpu|prefetch_cpu
VLLM_UVM_INIT_WEIGHT_PLAN_FILE=<path>
VLLM_UVM_INIT_WEIGHT_TARGET_ROLES=moe_gate_up,moe_down
VLLM_UVM_INIT_WEIGHT_MAX_BYTES=<int>
VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=<int>
VLLM_UVM_INIT_WEIGHT_SYNC=0|1
VLLM_UVM_INIT_WEIGHT_TRACE_FILE=<path>
```

计划文件沿用现有 Stage H / Stage I 格式：

```json
{
  "prefetch_plan": [],
  "offload_plan": [
    {
      "layer_id": 0,
      "expert_id": 5,
      "reason": "init_time_cold_expert"
    }
  ]
}
```

推荐初始参数：

```text
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=prefetch_cpu
VLLM_UVM_INIT_WEIGHT_TARGET_ROLES=moe_gate_up,moe_down
VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=8
VLLM_UVM_INIT_WEIGHT_MAX_BYTES=8589934592
```

### 5.3 Scratch direct init phases

现有 Stage C 参数保留，扩展 phase：

```text
VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES=enabled:init,enabled:warmup,enabled:dummy_run,enabled:prefill_step,enabled:model_forward,enabled:attention,enabled:moe
```

现有失败日志出现：

```text
scratch_pool_reason=scratch_pool_phase_not_allowed
scratch_pool_eligible_allocations=0
device_direct_actual_allocations=0
```

Stage M 的目标是让 warmup / dummy / prefill 早期 scratch 能够被识别并进入 device-direct 选择逻辑。

### 5.4 Server pressure guard

脚本默认值建议保守：

```text
--server-gpu-memory-utilization 0.70
--server-max-num-seqs 2 或 4
```

并在文档和脚本中明确：这个是启动期峰值保护，不是最终性能配置。

## 6. 代码落点

### 6.1 KV cache sizing

主要落点：

```text
vllm/v1/engine/core.py
```

当前关键流程：

```text
EngineCore.__init__
  _initialize_kv_caches()
    kv_cache_specs = model_executor.get_kv_cache_specs()
    available_gpu_memory = ...
    kv_cache_configs = get_kv_cache_configs(...)
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    model_executor.initialize_from_config(kv_cache_configs)
```

应在 `get_kv_cache_configs(...)` 之后、`model_executor.initialize_from_config(...)` 之前插入：

```text
apply_uvm_init_kv_sizing(vllm_config, kv_cache_configs, scheduler_kv_cache_config)
```

需要更新：

```text
scheduler_kv_cache_config.num_blocks
kv_cache_config.num_blocks
kv_cache_tensor.size
```

如果当前 `KVCacheConfig` 结构不适合原地改，应在 `vllm/v1/core/kv_cache_utils.py` 中新增返回受限 config 的 helper：

```python
def apply_uvm_init_kv_sizing(
    kv_cache_configs: list[KVCacheConfig],
    max_bytes: int,
    max_blocks: int,
    policy: str,
) -> tuple[list[KVCacheConfig], UvmKvSizingRecord]:
    ...
```

记录到现有 residency trace：

```text
record_uvm_residency_kv_sizing(...)
```

验收日志字段：

```text
kv_sizing_applied=True
kv_sizing_original_num_gpu_blocks=<old>
kv_sizing_final_num_gpu_blocks=<new>
kv_sizing_original_kv_bytes=<old>
kv_sizing_final_kv_bytes=<new>
```

### 6.2 KV cache allocation observation

主要落点：

```text
vllm/v1/worker/gpu_worker.py
```

当前实际分配发生在：

```text
GPUWorker.initialize_from_config(...)
  with allocator.use_memory_pool(tag="kv_cache"):
      self.model_runner.initialize_kv_cache(kv_cache_config)
```

Stage M 应确保传入这里的 `kv_cache_config` 已经是 clamp 后的配置。初始化完成后记录：

```text
actual_kv_cache_bytes
actual_num_gpu_blocks
```

验收条件：

```text
Pool kv live bytes <= VLLM_UVM_INIT_KV_MAX_BYTES * tolerance
```

### 6.3 Weight cold pre-offload before warmup

主要落点：

```text
vllm/model_executor/layers/fused_moe/layer.py
vllm/v1/worker/gpu_model_runner.py
```

现有 `fused_moe/layer.py` 已有：

```text
_uvm_expert_weight_slices(...)
_maybe_prefetch_uvm_expert_weights(...)
_maybe_offload_uvm_cold_expert_weights(...)
prefetch_range_to_cpu(...)
prefetch_range_to_device(...)
advise_range_preferred_location(...)
```

需要新增一个 init-time API：

```python
def run_uvm_init_weight_offload(self) -> dict[str, int]:
    """Register and offload cold expert weights before warmup."""
```

行为：

```text
读取 VLLM_UVM_INIT_WEIGHT_PLAN_FILE
遍历当前 layer 的 offload_plan
对 target roles 计算 ptr/bytes
register_uvm_residency_region(pool="weights", ...)
request_uvm_pool_action(pool="weights", action="init_weight_prefetch_cpu", ...)
执行 prefetch_range_to_cpu 或 advise_range_preferred_location
mark_uvm_residency_state(..., state="cpu" 或 "prefetching_cpu")
写 init weight trace
```

在 `gpu_model_runner.py` 的 warmup / dummy run 前调用一个模型遍历 helper：

```python
def run_uvm_init_time_weight_placement(self) -> None:
    for module in self.model.modules():
        fn = getattr(module, "run_uvm_init_weight_offload", None)
        if callable(fn):
            fn()
```

调用时机应早于：

```text
_dummy_run(...)
Triton MoE kernel warmup
```

建议加 phase：

```python
with uvm_enabled_allocation_phase("enabled:init_weight_offload"):
    self.run_uvm_init_time_weight_placement()
```

验收日志：

```text
init_weight_offload_candidates > 0
init_weight_offload_issued > 0
weight_live_bytes 或真实 GPU resident bytes 下降
full_impl server 能通过 ready
```

注意：`cudaMemPrefetchAsync(..., cudaCpuDeviceId)` 只是迁移请求，不保证立即释放所有物理显存。若需要强同步，`VLLM_UVM_INIT_WEIGHT_SYNC=1` 时应调用 `torch.cuda.synchronize()`。

### 6.4 Scratch direct warmup coverage

主要落点：

```text
vllm/device_allocator/uvm.py
vllm/v1/worker/gpu_model_runner.py
vllm/model_executor/layers/fused_moe/layer.py
```

现有 phase：

```text
enabled:model_forward
enabled:prefill_step
enabled:attention
enabled:moe
enabled:dummy_sampler
enabled:sampler
enabled:compute_logits
```

需要补充：

```text
enabled:init
enabled:warmup
enabled:dummy_run
enabled:init_weight_offload
```

建议在 `gpu_model_runner.py::_dummy_run` 外层加：

```python
with uvm_enabled_allocation_phase("enabled:dummy_run"):
    ...
```

如果内部已有更细 phase，外层 phase 只作为 fallback，不应覆盖内层 `enabled:moe` / `enabled:attention`。

Scratch direct eligibility 应接受这些 phase 前缀：

```text
enabled:warmup
enabled:dummy_run
enabled:prefill_step
enabled:model_forward
enabled:attention
enabled:moe
```

验收日志：

```text
Scratch pool eligible allocations > 0
Device-direct actual allocations > 0
scratch_pool_phase_not_allowed 明显下降
```

## 7. 驱动层增强方案

第一阶段不依赖驱动改动。第二阶段可以增加 UVM driver policy executor，用于降低 Python / JSONL / ctypes 高频开销。

### 7.1 用户态到驱动的数据模型

Region registration:

```c
enum uvm_policy_pool {
    UVM_POLICY_POOL_KV,
    UVM_POLICY_POOL_WEIGHT,
    UVM_POLICY_POOL_SCRATCH,
};

enum uvm_policy_action {
    UVM_POLICY_ADVISE_CPU,
    UVM_POLICY_PREFETCH_CPU,
    UVM_POLICY_PREFETCH_GPU,
    UVM_POLICY_QUERY_RESIDENCY,
};

struct uvm_policy_region {
    __u64 start;
    __u64 length;
    __u32 gpu_id;
    __u32 pool;
    __u64 region_id_hash;
    __u64 flags;
};

struct uvm_policy_batch_action {
    __u32 action;
    __u32 count;
    __u64 regions_user_ptr;
};
```

接口可以先走 procfs/debugfs，后续再 ioctl：

```text
/proc/driver/nvidia-uvm/uvm_policy/register_region
/proc/driver/nvidia-uvm/uvm_policy/apply
/proc/driver/nvidia-uvm/uvm_policy/query
```

或：

```text
UVM_POLICY_REGISTER_REGION
UVM_POLICY_APPLY_BATCH
UVM_POLICY_QUERY_REGION
```

### 7.2 驱动层职责

驱动层只做：

```text
接收 region
维护 region -> pool/action metadata
执行 batch prefetch / advise
统计 region fault / migration / residency
返回真实 residency 估计
```

驱动层不做：

```text
判断 expert hot/cold
决定 KV block 数量
判断 scratch allocation 是否可 device-direct
解析 vLLM layer/expert 语义
```

### 7.3 C++ extension 中间层

为避免 Python 高频调用，可在 vLLM C++/CUDA extension 中提供：

```python
torch.ops.vllm_uvm.batch_prefetch_cpu(ptrs, sizes)
torch.ops.vllm_uvm.batch_prefetch_gpu(ptrs, sizes, device)
torch.ops.vllm_uvm.batch_advise_cpu(ptrs, sizes)
torch.ops.vllm_uvm.register_regions(region_records)
```

第一版可以直接包装 CUDA runtime API；第二版再切换到底层 UVM ioctl。

## 8. 实施阶段

### Phase M0: 文档和实验基线

产物：

```text
本设计文档
当前 failed full_impl_enforce 日志归档
small run baseline 命令
```

验收：

```text
明确 OOM 发生在 warmup / Triton load_binary
明确 runtime 策略没有来得及执行
```

### Phase M1: KV sizing 前移

改动：

```text
v1/engine/core.py
v1/core/kv_cache_utils.py
device_allocator/uvm_pool_residency.py trace record 可复用或扩展
```

验收：

```bash
VLLM_UVM_INIT_KV_SIZING_ENABLE=1 \
VLLM_UVM_INIT_KV_SIZING_POLICY=clamp \
VLLM_UVM_INIT_KV_MAX_BYTES=1073741824 \
./run_stage_l_full_impl_uvm_ab.sh \
  --log-file log/stage_m_kv_sizing_smoke.log \
  --skip-trace-control \
  --prompts 2 \
  --output-len 128 \
  --server-gpu-memory-utilization 0.70 \
  --server-max-num-seqs 2
```

成功标准：

```text
server ready
KV live bytes 接近 1 GiB
kv_sizing_applied=True
无启动期 OOM
```

### Phase M2: Weight init pre-offload

改动：

```text
fused_moe/layer.py 新增 run_uvm_init_weight_offload
gpu_model_runner.py warmup 前遍历调用
```

验收：

```text
init_weight_offload_issued > 0
offload bytes 达到配置预算的一部分
Triton warmup 前已有 offload trace
```

### Phase M3: Scratch warmup direct

改动：

```text
gpu_model_runner.py 给 dummy/warmup 加 phase
uvm.py / allocator policy 接受 warmup phase
run scripts 默认 target phases 增加 warmup/dummy_run/prefill_step
```

验收：

```text
scratch_pool_eligible_allocations > 0
device_direct_actual_allocations > 0
scratch_pool_phase_not_allowed 下降
```

### Phase M4: Driver / C++ executor

改动：

```text
新增 batch advise/prefetch C++ op
可选新增 UVM driver ioctl/procfs
替换 Python ctypes 高频路径
```

验收：

```text
相同行为下启动开销下降
trace control 开销下降
fault / residency 统计可按 region 汇总
```

## 9. 实验矩阵

最小 smoke：

```text
pure_uvm_baseline
full_impl_enforce_small
```

命令：

```bash
./run_stage_l_full_impl_uvm_ab.sh \
  --log-file log/stage_m_full_impl_small.log \
  --skip-trace-control \
  --prompts 2 \
  --output-len 128 \
  --server-gpu-memory-utilization 0.70 \
  --server-max-num-seqs 2
```

标准 A/B：

```text
pure_uvm_baseline
uvm_trace_control
full_impl_enforce
```

命令：

```bash
./run_stage_l_full_impl_uvm_ab.sh \
  --log-file log/stage_m_full_impl_ab.log \
  --prompts 4 \
  --output-len 256 \
  --request-rate 2 \
  --server-gpu-memory-utilization 0.70 \
  --server-max-num-seqs 4
```

严谨性能结论：

```bash
./run_stage_l_full_impl_uvm_ab.sh \
  --log-file log/stage_m_full_impl_ab_repeat3.log \
  --repeat 3 \
  --prompts 4 \
  --output-len 256 \
  --request-rate 2 \
  --server-gpu-memory-utilization 0.70 \
  --server-max-num-seqs 4
```

## 10. 关键指标

启动稳定性：

```text
server ready
oom_runs=0
full_impl_enforce successful_runs > 0
```

KV 初始化限制：

```text
kv_sizing_applied=True
kv_sizing_final_kv_bytes <= VLLM_UVM_INIT_KV_MAX_BYTES
Pool kv live bytes <= budget * tolerance
```

Weight 初始化迁移：

```text
init_weight_offload_issued > 0
weight offload bytes > 0
warmup 前出现 offload records
```

Scratch direct：

```text
device_direct_actual_records > 0
scratch_pool_eligible_records > 0
scratch_pool_phase_not_allowed 下降
```

性能：

```text
output_tok_s 上升
mean_tpot_ms 下降
mean_ttft_ms 不显著恶化
```

Fault：

```text
delta_after_dedup 下降
post_unknown_faults 下降
post_gap_faults 下降
delta_kv_after_dedup 不恶化
```

## 11. 风险和处理

### 11.1 KV cache 太小导致吞吐下降

风险：

```text
KV blocks 太少，调度受限，吞吐下降或请求失败
```

处理：

```text
设置 VLLM_UVM_INIT_KV_MIN_BLOCKS
先 clamp 到 1 GiB，再 sweep 1/2/4 GiB
记录 failed_requests
```

### 11.2 Weight pre-offload 影响 MoE 首 token 延迟

风险：

```text
冷 expert 被访问时需要重新迁回 GPU，TTFT 或 TPOT 变差
```

处理：

```text
保留 hot expert prefetch
只 offload broad cold plan 中低概率 experts
限制每层 offload experts
对比 TTFT / TPOT / output_tok_s
```

### 11.3 cudaMemPrefetchAsync 不保证立即释放物理显存

风险：

```text
prefetch_cpu 已发出，但 warmup 前显存仍未下降
```

处理：

```text
VLLM_UVM_INIT_WEIGHT_SYNC=1 时 torch.cuda.synchronize()
必要时先 advise_cpu，再 prefetch_cpu
记录实际 NVML used memory
```

### 11.4 Scratch direct 误用于 CPU 访问 buffer

风险：

```text
device allocation 被 CPU 访问导致错误
```

处理：

```text
target_class 仍使用 gap_hot_runtime_scratch
保留 phase allowlist
失败自动 fallback managed
记录 device_direct_fallback_records
```

### 11.5 Driver 接口维护成本高

风险：

```text
UVM driver 修改复杂，版本耦合强
```

处理：

```text
第一版只做 vLLM + CUDA runtime
第二版 C++ extension
第三版再引入 driver ioctl/procfs
```

## 12. 推荐最小实现顺序

优先级最高：

```text
1. KV sizing before KV allocation
2. Weight cold offload before warmup
3. Scratch direct phase coverage for warmup/dummy/prefill
4. full_impl A/B small smoke
```

不建议一开始就改 UVM driver。只有当用户态 Stage M 已经证明：

```text
能避免 OOM
能降低 fault
有明确性能收益或至少不明显退化
```

再把高频执行路径下沉到 C++ / driver。

## 13. Done Definition

Stage M 第一版完成的标准：

```text
full_impl_enforce 在 prompts=2/output_len=128 下不 OOM
KV live bytes 从约 6 GiB 降到配置预算附近
weight init offload 在 warmup 前有 issued records
device_direct 在 warmup/prefill 至少有 actual records
summary.md 能给出 full_impl vs pure_uvm 的有效百分比
```

Stage M 可用于性能结论的标准：

```text
prompts=4/output_len=256/repeat=3 全部成功
failed_requests=0
full_impl_enforce output_tok_s >= pure_uvm_baseline
full_impl_enforce mean_tpot_ms <= pure_uvm_baseline
post_gap_faults / post_unknown_faults 显著下降
无 OOM / server exit / traceback
```
