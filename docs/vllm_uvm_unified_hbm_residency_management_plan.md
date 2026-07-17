# vLLM UVM KV/Expert 统一 HBM Residency 管理实施计划

本文档定义下一阶段目标：把 KV cache 和 MoE expert weight 做成统一的 HBM residency 管理对象。这里的“统一”不是把 KV tensor 和 expert weight 合并成一个 malloc pool，而是让二者都使用 UVM managed 分配，并由同一个 controller 管理当前驻留在 GPU HBM 的 region/page/block 数量。

核心结论：

```text
逻辑容量:
  KV cache + expert weight 可以大于 GPU HBM。

物理目标:
  控制当前 resident 在 HBM 的 KV block 和 expert region。
  reserved slot 只作为 expert 回填的固定执行资源扣预算。

低 KV 压力:
  KV 只保留小 resident window，HBM 让给 hot experts。

高 KV 压力:
  扩大 KV resident window，回收 cold experts，把 HBM 让给 KV。

expert 再命中:
  走 LOWTRACE 已验证的 reserved slot remap/prefetch 路径提前搬回。
```

这条路线成立，但当前仓库还不是完整的统一调度器。现有代码已经具备 KV 动态窗口、expert init offload、reserved slot prefetch、driver pool budget、fault observer 等地基；缺口是把这些能力连成一个明确的控制闭环，并先修通 Phase-N reclaim 原型路径。

## 1. 目标与边界

### 1.1 目标

1. 统一注册 KV block 和 expert weight region。
2. 统一维护 HBM budget、resident bytes、protected state、evictable state。
3. 低 KV 压力时主动缩小 KV resident set。
4. 高 KV 压力时主动扩大 KV resident set，并从 expert pool 回收 HBM。
5. expert 被再次命中时，优先在上一层计算窗口内用 reserved slot remap/prefetch 拉回。
6. 服务阶段关闭 vLLM 层自动按需搬运，只保留 fault observer 作为漏搬诊断。
7. 正常受控运行时，GPU fault 数应接近 0；fault 只作为 controller 漏搬的证据。

### 1.2 非目标

1. 不把 KV tensor 和 expert weight 真的合并成一个 `cudaMallocManaged`。
2. 不运行时修改 vLLM 的 KV tensor shape 或 `num_gpu_blocks`。
3. 不迁移 active request 的 KV block table。
4. 不驱逐当前层正在使用或下一层已经保护的 expert。
5. 不依赖 GPU page fault 来完成正常服务路径的数据搬运。
6. 不用 `torch.cuda.memory_allocated()` 或 `nvidia-smi` 作为唯一判断标准。
7. 不管理 runtime scratch / temporary buffer；这类内存仍按 vLLM/PyTorch/CUDA 原路径处理，只作为安全余量预留。

## 2. 当前已有地基

### 2.1 LOWTRACE 结论

`workloads/vllm/vllm/uvm_test/LOWTRACE_FOURWAY_EXPERIMENT.md` 已经证明：

```text
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_KV_CACHE_MANAGED=1
VLLM_USE_UVM=0
```

可以让 expert 主权重和 KV cache 选择性走 UVM managed，而 embedding、attention weight、router、scale、runtime temporary buffer 等仍走 native allocation。

LOWTRACE 还验证了：

1. Stage-M 可以把计划内 expert 主权重 offload 到 CPU。
2. post-Stage-M 可以预分配 reserved slot backing。
3. 服务阶段命中 offloaded expert 时，reserved slot remap/prefetch 是主要性能损失来源。
4. 关闭重 trace 后，静态卸载但不命中 offloaded expert 的性能损失大幅降低。

这说明统一管理的核心不是“是否能 managed allocation”，而是：

```text
能否在访问前把需要的 KV/expert 放回 HBM；
能否把暂时不用的 KV/expert 从 HBM 中移走；
能否把这两个动作藏进上一层或调度空隙中。
```

### 2.2 KV 侧能力

当前 KV 侧已有：

| 文件 | 已有能力 |
| --- | --- |
| `workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock.lifecycle_state`，可区分 `never_used`、`used_free`、`cpu_offloaded` |
| `workloads/vllm/vllm/vllm/v1/core/block_pool.py` | free block 分配入口，可插入 fresh/warm/CPU-backed 选块策略 |
| `workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py` | dynamic target blocks、control file、KV pointer map、Stage L6 prefetch/offload、trace |
| `workloads/vllm/check_kv_dynamic_pool_phase2_prefetch_real.py` | 真实 vLLM `target_blocks=2 -> 4 -> 1` 物理 prefetch/offload 验收脚本 |

当前 KV 侧已经能做：

```text
target 变小:
  只处理 ref_cnt == 0 的 free block；
  调用 prefetch_range_to_cpu；
  metadata 标为 cpu_offloaded。

target 变大:
  选择 CPU-backed block；
  调用 prefetch_range_to_device；
  重新进入可分配窗口。
```

限制：

1. target 现在主要是手动调整。
2. 还没有和 expert pool budget 联动。
3. 还没有自动 pressure controller。
4. 没有把 KV pressure 的增长转化为 expert reclaim 请求。

### 2.3 Expert 侧能力

当前 expert 侧已有：

| 文件 | 已有能力 |
| --- | --- |
| `workloads/vllm/vllm/vllm/model_executor/models/qwen3_moe.py` | Qwen3 MoE expert weight 注册、offload、routing trace 相关入口 |
| `workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py` | reserved slot prefetch/remap executor |
| `workloads/vllm/vllm/vllm/device_allocator/uvm.py` | `driver_expert_reserved_prefetch_regions()` ctypes wrapper |
| `workloads/vllm/vllm/uvm_test/run_vllm_pre_service_real_load_offload_perf.sh` | pre-service load/offload 性能实验 |

当前 expert 侧已经能做：

```text
init/service 前:
  注册 expert 主权重 region；
  根据 plan 做 Stage-M CPU offload。

service 阶段:
  命中 offloaded expert 时；
  使用 reserved slot remap/prefetch 搬回；
  可记录 active expert 和 prefetch 结果。
```

限制：

1. reserved slot 现在主要服务 expert，不理解 KV pressure。
2. expert victim 选择还没有和 KV controller 统一。
3. 当前层/下一层 expert 保护边界需要和统一 controller 对齐。

### 2.4 Driver/Pool 能力

当前 driver/pool 侧已有：

| 文件 | 已有能力 |
| --- | --- |
| `workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py` | pool registry、driver pool budget、state snapshot、driver reclaim wrapper |
| `workloads/vllm/vllm/vllm/device_allocator/uvm.py` | `driver_set_pool_budget`、`driver_get_pool_state`、`driver_reclaim_pool` |
| `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c` | Phase-N pool budget、region state、query、reclaim 原型 |
| `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c` | replayable fault hook、manual migration fault observer/fail-fast 参数 |

最近新增的内核参数可用于服务阶段 fault 诊断：

```text
uvm_manual_migration_fault_mode=0
  正常 UVM fault service。

uvm_manual_migration_fault_mode=1
  只记录 fault 地址，仍允许 UVM 自动迁移。

uvm_manual_migration_fault_mode=2
  记录 fault 地址，然后 cancel fault，不允许自动迁移。
```

它的用途不是正常服务时搬运数据，而是验证 controller 是否漏搬：

```text
正常受控运行:
  mode=1，fault 应接近 0。

测试漏搬:
  mode=2，漏搬对象会产生明确 failure 和 fault VA。
```

## 3. 统一抽象

### 3.1 对象模型

统一 controller 只管理两类对象：

```text
KV block:
  粒度是 vLLM logical KV block。
  物理动作映射到一个或多个 KV raw pointer ranges。

Expert region:
  粒度是 layer + expert_id + role。
  role 通常是 w13 / w2。
  物理动作走 Stage-M offload 或 reserved slot remap/prefetch。
```

不把对象抽象成“裸 GPU byte range”作为唯一身份。裸 VA 只用于最后一步执行迁移或 fault 反查。

reserved slot 是 expert region 的搬运执行资源，不作为独立受控对象。runtime scratch / temporary buffer 不进入统一 controller，不做 hot/cold、reclaim、prefetch 或 pool victim 选择。

### 3.2 Pool 模型

统一 pool 第一版只保留两池：

```text
pool=kv
  KV cache block。

pool=weights
  MoE expert 主权重，后续可细分 dense/shared/protected/expert。
```

第一版不建议把 expert hot/cold 做成 driver pool id。hot/cold 更适合作为 region state 或 priority，避免 pool 数量膨胀。

reserved slot 不作为独立 pool 管理，而是作为 expert 回填路径的固定 HBM 预算开销。runtime scratch 不建 pool、不做动态 residency 管理，只通过 `runtime_safety_margin` 预留空间，避免它和 KV/expert 预算相互误算。

### 3.3 State 模型

每个对象至少需要以下状态：

```text
residency_state:
  gpu
  cpu
  prefetching_gpu
  prefetching_cpu
  unknown

lifecycle_state:
  never_used
  used_free
  cpu_offloaded

protection_state:
  active
  protected_next
  evictable
  pinned

priority:
  hot
  warm
  cold
```

KV 特殊约束：

```text
ref_cnt > 0:
  active，不能 offload。

ref_cnt == 0:
  可以被选为 shrink victim。
```

Expert 特殊约束：

```text
current layer active expert:
  active，不能 reclaim。

next layer predicted expert:
  protected_next，不能 reclaim，应该 prefetch。

cold unpredicted expert:
  evictable，可以被 KV 高压阶段回收。
```

### 3.4 Budget 模型

controller 维护全局 budget：

```text
managed_hbm_budget
runtime_safety_margin
reserved_slot_budget
kv_min_resident_budget
kv_target_resident_budget
kv_max_resident_budget
expert_hot_budget
expert_streaming_budget
```

核心不变量：

```text
kv_resident_bytes
+ expert_resident_bytes
+ reserved_slot_bytes
+ runtime_safety_margin
<= managed_hbm_budget
```

注意：`managed_hbm_budget` 不是 `cudaMallocManaged` 的总逻辑大小，而是希望在 HBM resident 的物理页面总量。

`reserved_slot_bytes` 是 expert 回填所需的 GPU backing 预算；`runtime_safety_margin` 是给 scratch、临时 workspace、CUDA runtime 等非受控内存留下的保守余量。controller 不会对 scratch 做迁移或驱逐决策。

## 4. 控制闭环

### 4.1 输入信号

统一 controller 每个调度步或每层收集：

```text
KV pressure:
  running request 数
  waiting request 数
  projected new KV blocks
  free GPU resident KV blocks
  CPU-backed KV blocks
  active KV blocks

Expert pressure:
  current layer active experts
  next layer predicted experts
  routing history
  per expert hit count
  offloaded expert hit/miss

Residency state:
  driver pool state snapshot
  UVM real residency snapshot
  KV runtime trace
  reserved slot occupancy

Fault observer:
  manual_migration_fault logs
  fault VA -> region mapping
```

### 4.2 低 KV 压力策略

触发条件示例：

```text
kv_projected_blocks < kv_low_watermark
free_gpu_resident_kv_blocks > kv_free_high_watermark
waiting_queue_empty_for >= cooldown_steps
```

动作：

1. 下调 `kv_target_resident_blocks`。
2. 从 `ref_cnt == 0` 的 GPU KV block 中选择冷块。
3. 对这些 block 发起 `prefetch_range_to_cpu()` 或 KV offload/store。
4. 更新 driver pool budget，把释放出的 HBM 分配给 `weights`。
5. 预取 hot experts 或保留更多 expert resident。

验收现象：

```text
residency_kv_data_offload_issued 增加
kv_gpu_resident_bytes 下降
weights_budget_bytes 或 weights_available_budget 上升
hot expert prefetch 成功率上升
服务阶段 fault 不增加
```

### 4.3 高 KV 压力策略

触发条件示例：

```text
kv_projected_blocks > kv_high_watermark
free_gpu_resident_kv_blocks < kv_free_low_watermark
waiting requests 因 KV 不足被阻塞
```

动作：

1. 保护 active KV block。
2. 上调 `kv_target_resident_blocks`。
3. 计算新增 KV resident bytes 需求。
4. 从 expert pool 中选择 cold/low probability victim。
5. 对 expert victim 触发 driver reclaim 或 Stage-M offload。
6. 对 CPU-backed KV block 发起 `prefetch_range_to_device()`。
7. 更新 KV 可分配窗口。

验收现象：

```text
residency_kv_prefetch_back_issued 增加
kv_gpu_resident_bytes 上升
weights_gpu_resident_bytes 下降
被保护 expert 没有被 reclaim
KV allocation 不因 resident window 太小而阻塞
```

### 4.4 Expert 命中恢复策略

如果 routing 预测下一层需要某个 offloaded expert：

1. 在上一层 kernel 运行期间提交 reserved slot remap/prefetch。
2. 标记 expert 为 `protected_next`。
3. 等待 prefetch event 或在 layer boundary 做轻量确认。
4. 当前层进入该 expert 前，应确保其 slot resident。
5. 如果未 ready，进入同步 fallback，并记录 miss。

目标是让 expert 回填被上一层计算时间覆盖。已有实验中单个 4.5 MiB expert reserved prefetch p50 约 0.55 ms，上一层 layer window 约 2.2 到 2.5 ms，理论上有隐藏空间。

### 4.5 Fault 策略

正常服务不依赖 GPU fault 自动搬运。

推荐模式：

```text
开发观测:
  uvm_manual_migration_fault_mode=1
  记录 fault VA，UVM 仍自动 resolve，避免服务中断。

CI 漏搬测试:
  uvm_manual_migration_fault_mode=2
  漏搬即失败，用于证明 controller 覆盖性。

生产默认:
  uvm_manual_migration_fault_mode=0 或 1
  取决于 route log 开销。
```

## 5. 修改计划

### 5.1 vLLM object registry

目标：让 KV block 和 expert region 进入同一个 registry。

修改点：

| 文件 | 修改 |
| --- | --- |
| `uvm_pool_residency.py` | 增加 unified object registry 视图，统一暴露 KV/expert region |
| `uvm_kv_runtime_policy.py` | 输出 KV block -> pointer ranges -> region id 的稳定映射 |
| `gpu_model_runner.py` | 保持 Stage L6 KV pointer map，补充 group/layer/block 元数据 |
| `qwen3_moe.py` | expert region 注册时补齐 layer/expert/role/state/priority |
| `uvm.py` | 补齐 query by VA、set state、pool budget、reclaim result 的结构化 wrapper |

验收：

```text
一个 JSONL/trace 中能同时看到:
  object_type=kv, block_id, group_id, bytes, residency_state
  object_type=expert, layer, expert_id, role, bytes, residency_state
```

### 5.2 Unified budget controller

目标：新增一个统一 controller，不再让 KV 和 expert 各自独立决策。

建议文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_unified_residency_controller.py
```

主要接口：

```python
class UnifiedResidencyController:
    def on_service_start(self) -> None: ...
    def on_scheduler_step(self, kv_pressure, routing_plan) -> None: ...
    def before_layer(self, layer_id, predicted_experts) -> None: ...
    def after_layer(self, layer_id, actual_experts) -> None: ...
    def resize_kv_target(self, target_blocks) -> None: ...
    def reclaim_expert_for_kv(self, required_bytes) -> None: ...
    def prefetch_experts(self, layer_id, experts) -> None: ...
    def snapshot(self) -> dict: ...
```

第一版可以是单进程 Python controller，后续再把 hard quota 和 victim 选择下沉到 driver。

验收：

```text
trace 中出现统一 action:
  unified_budget_update
  unified_kv_grow_request
  unified_kv_shrink_request
  unified_expert_reclaim_request
  unified_expert_prefetch_request
  unified_fault_observed
```

### 5.3 KV 接入 controller

目标：KV target 不再只由 control file 手动调，而是允许 unified controller 调整。

修改点：

| 文件 | 修改 |
| --- | --- |
| `uvm_kv_runtime_policy.py` | 增加 programmatic target update API |
| `block_pool.py` | 保持 fresh -> warm -> CPU-backed 选择顺序 |
| `kv_cache_manager.py` | 把 projected KV demand 作为 pressure signal 上报 controller |

验收：

```text
低压:
  controller 自动 target 下降。

高压:
  controller 自动 target 上升。

安全:
  active block 不 offload。
```

### 5.4 Expert 接入 controller

目标：expert offload/prefetch 由统一 controller 调度。

修改点：

| 文件 | 修改 |
| --- | --- |
| `qwen3_moe.py` | 在 MoE forward/layer boundary 上报 current/next active experts |
| `uvm_expert_reserved_executor.py` | 暴露 nonblocking prefetch request 和 completion query |
| `uvm_pool_residency.py` | expert region state 更新为 active/protected_next/evictable |
| `uvm.py` | 暴露 driver update region state / reclaim result |

验收：

```text
当前层 expert:
  active/protected，不会被 reclaim。

下一层 predicted expert:
  提前 reserved prefetch。

cold expert:
  KV 高压时可被 reclaim。
```

### 5.5 Driver pool reclaim 修复

当前 `run_stage_n_3gb_rebalance_smoke.sh` 曾出现：

```text
reclaim-expert0 result=-86
total_ok=0
```

统一 controller 前必须先修通这条原型路径。

排查顺序：

1. 确认 region tag 是否进入 driver。
2. 确认 expert region state 是否 `EVICTABLE` 且非 `PROTECTED/ACTIVE/PINNED`。
3. 确认 pool budget 是否已下发到 driver。
4. 确认 `RECLAIM_POOL` 或 `RECLAIM_REGION` 使用的 VA/range 与 managed allocation 匹配。
5. 确认当前 loaded `nvidia_uvm` 与 built module srcversion 一致。
6. 确认 reserved slot 参数与 smoke 期望一致。
7. 确认 reclaim path 返回 `-86` 的具体 NV_STATUS 映射和 fallback reason。

修复验收：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh
```

期望：

```text
expert_reclaim_ok=1
kv_tail_after_ok=1
total_budget_ok=1
total_ok=1
```

### 5.6 服务阶段 manual mode

目标：关闭 vLLM 自动按需搬运，只让 controller 主动搬运，fault observer 用来发现漏搬。

修改点：

| 层 | 设置 |
| --- | --- |
| Expert prefetch | 默认 `trace_only` 或由 controller 显式调用 |
| KV dynamic resize | 只接受 controller target，不因 fault 自动扩大 |
| UVM fault | `uvm_manual_migration_fault_mode=1` 观察，CI 用 `mode=2` fail-fast |
| Reserved runtime ready | 由 controller 在 service ready 后显式打开 |

验收：

```text
不开 controller 显式 prefetch:
  mode=2 下访问 offloaded expert/KV 会失败，并记录 fault VA。

开 controller 显式 prefetch:
  mode=2 下请求能完成，manual_migration_fault 接近 0。
```

## 6. 分阶段实施计划

### Phase U0: 基线冻结与失败 smoke 修复

目标：确保当前实验基线可复现。

任务：

1. 记录 built/loaded `nvidia_uvm` srcversion。
2. 复跑 LOWTRACE 四组核心脚本，保存 baseline。
3. 复跑 KV dynamic target `2 -> 4 -> 1`。
4. 修复 `run_stage_n_3gb_rebalance_smoke.sh` 的 `reclaim-expert0 result=-86`。
5. 复跑 manual migration fault control smoke。

验收：

```text
LOWTRACE baseline 可复现。
KV target 2 -> 4 -> 1 可复现。
Phase-N 3GB rebalance smoke PASS。
manual migration fault smoke PASS。
```

### Phase U1: 统一 registry 和 trace-only controller

目标：只统一观测，不改变行为。

任务：

1. 新增 `uvm_unified_residency_controller.py`。
2. 把 KV/expert 对象汇总到统一 snapshot。
3. 每个调度步输出 budget 和 pressure trace。
4. 不做任何真实迁移。

验收：

```text
性能与 baseline 基本一致。
trace 可重建每步:
  kv_resident
  expert_resident
  reserved_slot_budget
  runtime_safety_margin
  protected objects
  evictable objects
```

### Phase U2: 手动 budget rebalance 闭环

目标：复现你们想要的手动场景。

实验序列：

```text
初始:
  KV 小 resident budget
  expert 大 resident budget

高 KV 压力:
  增大 KV target
  reclaim cold expert
  prefetch KV tail

低 KV 压力:
  shrink KV target
  释放 KV HBM
  增大 expert hot budget
```

任务：

1. 增加统一 control file。
2. 支持 `kv_target_blocks=N`。
3. 支持 `weights_budget_bytes=N`。
4. 支持 `reclaim_pool=weights required_bytes=N`。
5. 支持 `prefetch_kv_blocks=...`。

验收：

```text
driver pool state 中:
  KV resident bytes 随 target 上升。
  weights resident bytes 在 reclaim 后下降。
  target 降低后 KV resident bytes 下降。
```

### Phase U3: 自动 KV pressure controller

目标：不用手动 target，根据 KV 压力自动调整。

任务：

1. 在 scheduler/KV manager 上报 projected KV block demand。
2. 增加 high/low watermark 和 hysteresis。
3. 加入 cooldown，避免 target 抖动。
4. 只在安全点触发 resize/reclaim。

验收：

```text
压力升高:
  unified_kv_grow_request 出现。
  expert reclaim 在 KV grow 前完成或有明确 fallback。

压力降低:
  unified_kv_shrink_request 出现。
  free KV block 被 offload。
```

### Phase U4: Expert victim selection 与 protected state

目标：KV 高压时能安全选 expert victim。

任务：

1. 从 routing trace 或在线 routing 统计维护 expert heat。
2. 当前层 active expert 标记为 protected。
3. 下一层 predicted expert 标记为 protected_next。
4. victim 只从 cold/evictable expert 中选。
5. 加入 per-layer fairness，避免总是回收同一层 expert。

验收：

```text
cross_pool_reclaim_errors = 0
active_expert_reclaim_attempts = 0
protected_next_reclaim_attempts = 0
cold expert reclaim success > 0
```

### Phase U5: Expert prefetch overlap

目标：把 expert 回填尽量藏进上一层 layer time。

任务：

1. 在 layer N 结束前预测 layer N+1 experts。
2. 发起 nonblocking reserved prefetch。
3. 用 CUDA event 或 driver completion 查询 ready 状态。
4. layer N+1 使用前只做短等待。
5. 记录 prefetch latency、overlap ratio、sync fallback 次数。

验收：

```text
reserved_prefetch_issued > 0
prefetch_overlap_ms > 0
sync_fallback_rate 下降
offloaded expert hit 的 TPOT 损失下降
```

### Phase U6: Service fault close-loop

目标：证明正常受控运行不依赖自动 fault migration。

任务：

1. 开启 `uvm_manual_migration_fault_mode=1`，观察正常服务 fault 数。
2. 开启 controller 显式 prefetch/offload。
3. CI 中使用 `mode=2` 跑小规模请求。
4. 漏搬时把 fault VA 映射回 KV block 或 expert region。

验收：

```text
mode=1:
  请求完成，manual_migration_fault 接近 0。

mode=2:
  controller 完整时请求完成。
  故意漏搬时请求失败，并能定位到具体 KV/expert object。
```

### Phase U7: 真实 vLLM E2E 评估

目标：证明统一管理优于纯 UVM faulting 和静态 offload。

实验组：

```text
A: baseline expert+KV managed, no offload
B: current dynamic real-load expert offload + reserved slot
C: KV dynamic only
D: unified HBM residency controller
E: unified controller + manual fault fail-fast CI mode
```

指标：

```text
TTFT
TPOT
output tok/s
failed requests
manual fault count
KV GPU/CPU resident bytes
weights GPU/CPU resident bytes
expert prefetch p50/p95
KV prefetch/offload p50/p95
reclaim success/failure
sync fallback count
```

验收：

```text
相对 B:
  offloaded expert hit 的同步等待下降。
  高 KV 压力下请求失败或阻塞下降。
  fault count 下降。

相对纯 UVM demand paging:
  TPOT/throughput 明显更稳定。
```

## 7. 实验脚本计划

### 7.1 修复与保留现有脚本

必须保留并修复：

```text
workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh
workloads/vllm/check_kv_dynamic_pool_phase2_prefetch_real.py
workloads/vllm/run_manual_migration_fault_control_smoke.sh
```

### 7.2 新增 unified smoke

建议新增：

```text
workloads/vllm/run_unified_hbm_residency_manual_rebalance_smoke.sh
```

场景：

```text
1. 启动 selective managed:
   expert weights managed
   KV cache managed

2. 初始 budget:
   kv_target_blocks = small
   weights_budget = large

3. 模拟高 KV pressure:
   kv_target_blocks increase
   reclaim weights
   prefetch KV blocks

4. 模拟低 KV pressure:
   kv_target_blocks decrease
   offload KV free blocks
   raise weights budget

5. 检查 pool state:
   KV/weights resident bytes 方向正确。
```

### 7.3 新增 vLLM E2E

建议新增：

```text
workloads/vllm/run_unified_hbm_residency_vllm_e2e.sh
```

场景：

```text
Qwen3 MoE
requests = 8 or 16
max_model_len = 512
max_num_batched_tokens = 512
manual fault observe mode
low trace
```

输出：

```text
summary.json
unified_trace.jsonl
kv_runtime_trace.jsonl
reserved_prefetch_trace.jsonl
pool_state_trace.jsonl
manual_fault_trace.log
```

### 7.4 新增 fail-fast 漏搬测试

建议新增：

```text
workloads/vllm/run_unified_hbm_residency_fault_failfast_probe.sh
```

场景：

```text
1. controller 正常:
   mode=2，请求成功，fault=0。

2. 故意禁用 expert prefetch:
   mode=2，请求失败，fault VA 映射到 expert。

3. 故意禁用 KV prefetch:
   mode=2，请求失败，fault VA 映射到 KV block。
```

## 8. Trace 规范

统一 trace 每条记录建议包含：

```json
{
  "ts": 0.0,
  "step": 0,
  "action": "unified_budget_update",
  "managed_hbm_budget": 0,
  "kv_target_blocks": 0,
  "kv_resident_bytes": 0,
  "weights_budget_bytes": 0,
  "weights_resident_bytes": 0,
  "reserved_slot_bytes": 0,
  "runtime_safety_margin": 0,
  "decision_reason": "kv_pressure_high",
  "protected_kv_blocks": 0,
  "protected_experts": [],
  "victims": [],
  "issued_operations": []
}
```

对象级 trace：

```json
{
  "action": "unified_object_state",
  "object_type": "expert",
  "region_id": 0,
  "layer": 0,
  "expert_id": 0,
  "role": "w13",
  "bytes": 0,
  "residency_state": "cpu",
  "protection_state": "evictable",
  "priority": "cold"
}
```

fault trace：

```json
{
  "action": "unified_fault_observed",
  "fault_addr": "0x0",
  "access": "read",
  "mapped_object_type": "kv",
  "mapped_region_id": 0,
  "mapped_block_id": 0,
  "controller_expected_resident": false,
  "mode": "observe"
}
```

## 9. 安全约束

必须满足：

1. KV `ref_cnt > 0` 不 offload。
2. active request block table 不后台修改。
3. 当前层 active expert 不 reclaim。
4. 下一层 protected expert 不 reclaim。
5. reserved slot remap 完成前不允许对应 expert kernel 访问。
6. driver reclaim 失败时必须有 fallback，不允许静默继续。
7. manual fault mode 只用于观测或 CI，正常服务不能依赖 fault 搬运。
8. 所有 budget 更新必须带 hysteresis 和 cooldown。
9. 多 rank 场景中，同一 TP group 的 budget 需要一致或显式记录不一致原因。

## 10. 风险与缓解

| 风险 | 现象 | 缓解 |
| --- | --- | --- |
| expert 预测不准 | reserved prefetch 浪费带宽，挤出 KV | 限制 prefetch top-k，加入 confidence threshold |
| KV target 抖动 | KV block 频繁 CPU/GPU 往返 | high/low watermark + cooldown |
| driver reclaim 不稳定 | `RECLAIM_POOL` 返回错误 | 先修 Stage-N 3GB smoke，再接 E2E |
| page fault 抖动 | TPOT 长尾严重 | service 阶段 mode=1 观测，mode=2 CI fail-fast |
| reserved slot 不够 | expert 命中同步等待 | 增加 slot count 或分层动态 slot 复用 |
| PCIe 带宽竞争 | KV prefetch 和 expert prefetch 相互拖慢 | controller 限制每 step migration bytes |
| runtime scratch 峰值挤占 HBM | KV/expert 预算看似满足但仍 OOM 或触发 fault | 提高 `runtime_safety_margin`，但不把 scratch 纳入 controller |
| PyTorch 显存统计误导 | memory_allocated 不下降 | 使用 UVM residency snapshot 和 driver pool state |

## 11. 推荐环境开关

选择性 managed：

```bash
export VLLM_USE_UVM=0
export VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
export VLLM_UVM_KV_CACHE_MANAGED=1
```

KV dynamic residency：

```bash
export VLLM_UVM_KV_RUNTIME_ENABLE=1
export VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS=1
export VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1
export VLLM_UVM_RESIDENCY_STAGE=stage_l6
export VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE=prefetch_cpu
export VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE=prefetch
```

Expert reserved slot：

```bash
export VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY=0
export VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=1
export VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY=dynamic
```

服务阶段 fault observer：

```bash
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_log_enable
```

CI fail-fast：

```bash
echo 2 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode
```

恢复默认：

```bash
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode
```

## 12. 最小可交付版本

第一版 MVP 不追求最优性能，只要求证明统一管理闭环成立。

MVP 包含：

1. 统一 object registry。
2. trace-only unified controller。
3. 手动 KV target grow/shrink。
4. KV grow 前能 reclaim cold expert。
5. KV shrink 后能把空间还给 expert。
6. expert 命中时能 reserved prefetch。
7. manual fault observe 能定位漏搬。
8. fail-fast 能证明自动搬运被关闭。

MVP 验收命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh
workloads/vllm/check_kv_dynamic_pool_phase2_prefetch_real.py --timeout-seconds 900
workloads/vllm/run_manual_migration_fault_control_smoke.sh
workloads/vllm/run_unified_hbm_residency_manual_rebalance_smoke.sh
```

最终判断标准：

```text
低 KV 压力:
  KV resident bytes 下降，expert resident budget 上升。

高 KV 压力:
  expert resident bytes 下降，KV resident bytes 上升。

服务阶段:
  controller 显式搬运后 fault 接近 0。
  故意漏搬时 fail-fast 能定位对象。
```

## 13. 结论

统一 HBM residency 管理可以实现，且更符合当前仓库已经铺好的能力边界：

```text
KV:
  固定最大逻辑池 + 动态 resident window。

Expert:
  managed 主权重 + CPU offload + reserved slot remap/prefetch。

Driver:
  pool budget + region state + reclaim + fault observer。

Controller:
  在安全点统一决定 KV 和 expert 谁应该 resident。
```

下一步不应直接做复杂自动调度，而应先按 `U0 -> U2` 修通手动统一 rebalance 闭环。只要手动闭环能稳定证明：

```text
KV grow 会让 expert 降 resident；
KV shrink 会把 HBM 还给 expert；
fault observer 能发现漏搬；
```

再推进自动 pressure controller 和 prefetch overlap，风险会小得多。
