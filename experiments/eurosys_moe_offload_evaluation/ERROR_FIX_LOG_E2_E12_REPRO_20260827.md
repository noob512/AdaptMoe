# E2--E12 错误与修复记录（2026-08-27）

本文件只记录本轮已经解决并完成复测的错误。尚未解决的 E6/E8/E9 及正式矩阵缺口见
[UNRESOLVED_E2_E12_REPRO_20260827.md](UNRESOLVED_E2_E12_REPRO_20260827.md)。

## F-01：E10 post-warmup 热点 prefetch 报 `unregistered_regions`

### 对应实验与失败证据

- 实验：E10 三队列 E2E qualification。
- 失败 artifact：
  [`artifacts/e10-three-queue-REPRO-20260827-attempt-01/`](artifacts/e10-three-queue-REPRO-20260827-attempt-01/)。
- 失败 case：`three_queue/result.json`，baseline 已成功，three_queue 在模型 warmup 后失败。
- 调用链：
  `gpu_worker._run_uvm_post_warmup_rebalance()` →
  `gpu_model_runner.run_uvm_init_time_weight_placement()` → hot expert prefetch。

核心错误：

```text
RuntimeError: Hot Expert prefetch failed:
result={'ok': False, 'reason': 'unregistered_regions'}
```

完整 traceback 保存在：

```text
artifacts/e10-three-queue-REPRO-20260827-attempt-01/
  three_queue/result.json
  three_queue/run.log
```

### 根因

static-alias finalize 已在 runtime controller 初始化前注册并拥有一部分专家的完整物理
region。这些专家仍保留在 `_uvm_expert_records` 中用于逻辑 inventory，但为防止同一 VA
region 被两个 owner 重复注册，它们的 envelope 没有进入显式 UVM region registry。

旧 `_uvm_init_hot_expert_records()` 只排除了 cold/offloaded 和 remote-read experts，仍把
static-alias-owned experts 当成普通 survivor 热点。post-warmup rebalance 随后把这些没有
显式 registry ownership 的 region IDs 交给 hot prefetch，registry 正确返回
`unregistered_regions`。这不是 KV block 本身访问失效，而是两种物理 region ownership
边界不一致。

### 最小修复

文件：
`workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`

在 `_uvm_init_hot_expert_records()` 中读取已经由 compile 阶段保存的
`_uvm_static_alias_owned_global_expert_ids`，并从显式热点记录中排除：

```diff
 cold_expert_ids = offloaded_ids | remote_read_ids
+static_alias_owned = set(
+    getattr(self, "_uvm_static_alias_owned_global_expert_ids", set())
+)
 return [
     record
     for _, record in sorted(self._uvm_expert_records.items())
     if record.global_expert_id not in cold_expert_ids
+    and record.global_expert_id not in static_alias_owned
 ]
```

static-alias-owned experts 的 residency 仍由 alias session/PARK/RESUME 路径管理；修复没有
把它们改为 cold，也没有减少每层计划专家数、降低 HBM 安全门或禁用 post-warmup
rebalance。

### 测试变更

文件：
`workloads/vllm/vllm/tests/test_uvm_explicit_expert_regions.py`

新增 `test_layer_hot_set_excludes_static_alias_owned_experts()`，验证：

- alias-owned expert 不出现在 hot region ID 列表；
-普通 survivor 仍被设置为 GPU；
- alias-owned record 保持 UNKNOWN，由其 alias owner 后续提交真实状态。

### 复测结果

1. 定向测试：

   ```bash
   workloads/vllm/vllm/.venv/bin/python -m pytest -q \
     workloads/vllm/vllm/tests/test_uvm_explicit_expert_regions.py \
     -k 'layer_compiles or layer_hot_set or dynamic_swap'
   ```

   最终结果：`4 passed, 16 deselected`。

2. 原失败实验重跑：

   ```bash
   RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e10-three-queue-REPRO-20260827-attempt-02" \
   ALLOW_GPU_RUN=1 \
   bash experiments/vllm_uvm_three_queue_e2e/run_experiment.sh qualify
   ```

   验证器 `status=ok`：24→40→20 KV blocks；4/8 backup blocks 被消费；16 offline
   blocks 迁入 GPU；48 experts/226,492,416 B 卸载并恢复；3 个 Graph 无 recapture；
   Prefill/Decode replay 和 roundtrip 通过；kernel faults=0；shutdown clean。

3. 扩展回归：

   - E10 path parity：dynamic/direct 输出和状态等价，性能差在预注册容差内；
   - E10 large-KV：2→4→8→10→8→4→2 GiB 与 direct 2/4/8/10 GiB 全通过；
   - 两项均为无 Graph recapture、无 fault、资源完整释放。

成功 artifacts：

- [`e10-three-queue attempt-02`](artifacts/e10-three-queue-REPRO-20260827-attempt-02/)
- [`e10-path-parity`](artifacts/e10-path-parity-REPRO-20260827-attempt-01/)
- [`e10-large-kv`](artifacts/e10-large-kv-REPRO-20260827-attempt-01/)

### 潜在影响与边界

- 正向影响：消除 alias session 与 explicit registry 的双重物理 ownership，允许
  post-warmup rebalance 只处理自己合法注册的 survivor experts。
- 兼容性：使用 `getattr(..., set())`，没有 static-alias ownership 集合的旧路径行为不变。
- 风险：如果上游错误地漏标 static-alias ownership，仍会 fail-closed，而不是静默跳过；
  真实 GPU 回归已覆盖当前 Qwen three-queue、path-parity 和 large-KV 路径。
- 不能由该修复推导 E10 自动控制器正式 5-seed trace 已完成。

## F-02：定向复验中的旧测试夹具缺少 ownership 查询桩

### 现象

在 F-01 修复后扩大定向选择范围，
`test_layer_compiles_expert_records_with_unknown_initial_state` 在进入原断言前失败：

```text
AttributeError: 'types.SimpleNamespace' object has no attribute
'_uvm_static_alias_owned_experts'
```

第一次扩展复验结果为 `1 failed, 3 passed, 16 deselected`。这是测试构造错误，不是 GPU
运行时错误：生产 `FusedMoE` 已提供 `_uvm_static_alias_owned_experts()`，旧
`SimpleNamespace` 没有模拟该必需接口。

### 根因与最小修复

`_uvm_compile_expert_records()` 在编译 envelope 时必须先查询本层 static-alias ownership，
而该单元测试希望模拟“无 static alias”场景。给该夹具补一个返回空集合的桩：

```diff
 layer = SimpleNamespace(
     uvm_expert_envelope_bytes=0x80000,
     _uvm_all_local_expert_pairs=lambda: [(3, 0)],
+    _uvm_static_alias_owned_experts=lambda _layer_id: set(),
     ...
 )
```

文件：
`workloads/vllm/vllm/tests/test_uvm_explicit_expert_regions.py`。

### 验证与影响

使用完全相同的 pytest 命令重跑后为 `4 passed, 16 deselected`。该改动只补齐测试 double
接口，不修改生产代码、实验配置或结果；空集合与测试原先假设的“所有 expert region 都由
显式 registry 注册”一致。

## F-03：E5 dual Graph capture 后 scheduler workspace 挤占安全 headroom

### 对应实验与失败证据

- 失败 artifacts：
  [`attempt-01`](artifacts/e5-dual-topology-REPRO-20260827-attempt-01/)、
  [`attempt-02`](artifacts/e5-dual-topology-REPRO-20260827-attempt-02/)。
- 两次的 B1/B4 都在 `cpu_offload_graph_static_alias_capture_succeeded` 后的第二次
  `cpu_offload_uvm_original_va_graph_residency_reconciled` 失败。

```text
RuntimeError: Insufficient GPU headroom after expert residency reconciliation:
free_bytes=172752896, required_bytes=201326592
RuntimeError: Original-VA Graph residency reconciliation failed;
the process is poisoned and must be torn down
```

第一次 reconciliation 尚有 227,278,848 B free；Graph capture 后 allocator reserved 增加
48,234,496 B，第二次只剩 172,752,896 B。把 KV 从 160 MiB 调为 128 MiB 后四个数值完全
相同，因此该尝试不是修复，只排除了 nominal KV 大小是直接控制量。

### 根因

arm D 的冻结 Graph shape 为 `[1,4,314,1217]`，最大真实 pure-Prefill shape 是 1,217，
但 runner 默认把 `max_num_batched_tokens` 设为 2,048。scheduler/attention workspace 按
2,048 预留了实验不使用的空间；这部分预留在 Graph capture 后仍由 caching allocator
持有，挤占了 192-MiB minimum-free gate，而不是由 PARK 的双 backing 或 KV 参数直接
决定。

### 最小修复

不改代码路径、不降低安全门，使用 runner 已有的可配置参数：

```text
MAX_NUM_BATCHED_TOKENS=1217
```

1,217 等于冻结 capture 集合中的最大正式 shape，因此没有改变请求、Graph、模型或实验
目标。128-MiB KV、四个 capture size、双 576-MiB backing、PARK/RESUME 和
`VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592` 均保持不变。

### 验证结果

- qualification：
  [`attempt-03`](artifacts/e5-dual-topology-REPRO-20260827-attempt-03/)，2/2 cases，
  verifier `status=ok`。
- formal：
  [`attempt-04-formal`](artifacts/e5-dual-topology-REPRO-20260827-attempt-04-formal/)，
  2 audit + B1/B4 各 10 performance，22/22 cases，verifier `status=ok`。

收紧 token budget 后，两次 reconciliation 的 free bytes 都增加 50,331,648 B；第二次
free 为 223,084,544 B，超过安全门 21,757,952 B。22 个独立冷进程的第二次 free 和
allocator reserved 都稳定为 223,084,544/32,711,376,896 B。96 个 formal JSON 均可解析，
fatal logs=0，Graph/VA/PTE/residency/roundtrip/output/cleanup 证书全部通过，最终 stock UVM
与 GPU 0 MiB 恢复。

### 潜在影响与边界

- 修复移除的是不参与冻结 workload 的 workspace，不减少 E5 placement 或 Graph coverage。
- 只有最大请求 shape 不超过 1,217 的该冻结协议可使用此值；更大 fixture 必须重新计算
  token budget，不能照抄。
- 该修复解决 dual Graph 运行阻塞，但不自动补齐 DESTROY/rebuild、sequential PTE 或
  recapture 对照。

## F-04：cold-5 formal verifier 硬编码每臂 3 个冷进程

### 现象与根因

E5 eager/Graph alias Prefill harness 的 formal order 固定为 `TZAGGZAAZG`，verifier 也硬
编码 `T=1,Z/A/G=3`。这能做机制 qualification，但不满足计划规定的每 arm/cell 至少 10
个独立冷进程；直接传入更长 `ORDER` 会被 verifier 以 `formal arm counts differ` 拒绝。

### 最小修复

涉及文件：

- `experiments/vllm_uvm_cold5_alias_prefill/run_experiment.sh`
- `experiments/vllm_uvm_cold5_alias_prefill/verify_experiment.py`

runner 新增 `FORMAL_PROCESSES_PER_ARM`：默认仍为历史值 3；高于 3 时要求偶数，并以相邻
pair 为每个 arm 生成相同数量的 `1,4`/`4,1` 进程，同时轮换 pair 顺序。verifier 改为：

- T 必须恰为 1；
- Z/A/G 计数必须相同且至少为 3；
- 若 run config 记录 formal process count，必须与实际一致；
- 每臂两种 batch order 都存在，数量差不超过 1。

没有放宽 return code、输出 hash、Graph、plan、alias、fault、service/module 或 cleanup gate。

### 验证结果

1. 静态回归：`77 passed, 2 warnings`，`bash -n`、`py_compile`、冻结 plan 和 module
   capability preflight 通过。
2. qualification：
   [`attempt-01-qualification`](artifacts/e5-cold5-prefill-REPRO-20260827-attempt-01-qualification/)，
   T/Z/A/G 4/4，verifier `status=ok`。
3. formal：
   [`attempt-02-formal`](artifacts/e5-cold5-prefill-REPRO-20260827-attempt-02-formal/)，
   实际 T=1、Z/A/G=10/10/10；每性能臂 batch order 5:5；31/31 cases，99 JSON 可解析，
   fatal logs=0，verifier `status=ok`，最终 stock UVM/0 MiB。

formal 中 alias Graph 相对 alias eager 的 Prefill median 改善为 B1 48.09%、B4 27.05%。
默认 3-process 协议保持兼容；新增规模只在显式设置环境变量时启用。

## F-05：KTransformers MLA KV pool 容量估算少算 3 倍

### 对应实验与失败证据

- 实验：E2 DeepSeek/KTransformers equal-HBM，2-GiB tier，repeat 1。
- 失败 attempts：`attempt_1`--`attempt_3`；均在 server ready 前分配 MLA KV pool 时 OOM，
  verifier 为 `status=error`，退出清理成功。
- Artifact 根目录：
  [`budget_2g/repeat_01`](artifacts/eurosys-pa-v2-20260827T005911Z-e2-nonblocked-formal-attempt-02/cases/E2/track_b_equal_system_hbm/ktransformers/deepseek_v2_lite_bf16/budget_2g/repeat_01/)。

典型错误为：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB.
GPU 0 ... 180.19 MiB is free
```

把 `mem_fraction_static` 从 0.97 依次降到 0.96/0.95 只改变最后一次失败分配大小，未消除
启动 OOM，说明根因不是单纯缺少一个固定 headroom。

### 根因与最小修复

`MLATokenToKVPool` 实际为每 token 存储压缩 KV latent 与 RoPE，即：

```text
kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576 elements
```

旧 profiler 却使用：

```text
qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192 elements
```

因此每 token KV cell 被低估为实际值的三分之一，profiled token 数约放大三倍，分配时才
耗尽 HBM。最小代码修复位于
`KTransformers/third_party/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`：
MLA 分支以 `kv_lora_rank` 替换 `qk_nope_head_dim`；非 MLA 分支不变。

### 验证与影响

修复后 `attempt_4` 已能完成 pool 分配并启动，日志记录 75,429 tokens / 2.19 GB；这把故障
从“启动期错误容量估算”推进到真实 workload headroom 校准。最终 `attempt_9` 的完整
16-cell 复测为 `status=ok`，说明修复后的容量估算覆盖启动、Prefill、Decode 与退出。

该改动只修正 MLA pool 的容量 accounting，不改变模型权重、专家 placement、请求或输出；
潜在影响是同一 `mem_fraction_static` 下的可用 KV token 数会降低到诚实容量，依赖旧错误
容量的配置必须重新校准。

## F-06：KTransformers E2 正式请求的 workspace、上下文边界与 allocator headroom

### 现象与排查轨迹

修复 F-05 后，2-GiB tier 仍暴露三个独立的运行期边界：

| attempt | 参数/现象 | 结论 |
|---:|---|---|
| 4 | static 0.97；ordinary C8/C16 OOM | 2.19-GB KV pool 留给最大 Prefill workspace 的动态 HBM 不足 |
| 5 | static 0.96；ordinary C16 后、RAG C1 OOM | 仍无足够高并发/长 Prefill headroom |
| 6 | static 0.94；RAG 四档被 8,320-token envelope 拒绝 | SGLang 内部保留两个 token slot |
| 7 | context 8,320；RAG C1 只返回 126/128，C8 342-MiB OOM | 请求 envelope 与 workspace 均未完全满足 |
| 8 | context 8,322、static 0.94；RAG C4/C8 342-MiB OOM | token 正确性恢复，动态 headroom 仍差约 50 MiB |
| 9 | context 8,322、static 0.93、expandable segments | 16/16 cells、116/116 请求通过 |

所有失败 attempt 均保留 client/server/NVML/verifier/cleanup 证据，没有删除性能 outlier 或
把错误请求计为成功。

### 最小修复

1. `run_formal_e2.sh` 冻结 `FORMAL_KT_CONTEXT_LENGTH=8322`，给 SGLang 两个内部 token slot，
   对外仍与另外两个系统的 8,320-token请求上限一致。
2. 冻结 `FORMAL_KT_MEM_FRACTION_STATIC=0.93`；修复后的日志为 37,483 KV tokens / 1.09 GB，
   Graph capture 后仍有 1.56 GB 可用动态 HBM，足以覆盖 16,384-token Prefill workspace。
3. `configs/e2_ktransformers_env.json` 增加
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减少相邻 workload 的 allocator
   fragmentation；没有降低请求长度、并发、输出 token 或 cell 数。

### 正式复测

成功证据为同一 repeat 的 `attempt_9`：顶层和 16 个 cell verifier 均 `status=ok`，
116/116 请求成功、24/116 满足冻结 SLO，peak HBM 32,109 MiB；fatal patterns 为空，
server return code 0，GPU cleanup 后 0 MiB/无 compute process。正式命令、实际参数和环境
分别保存在 `command.txt`、`env.json`、`server.log` 与 run-root provenance 中。

边界：0.93 是当前 RTX 5090/driver/DeepSeek BF16/冻结 16,384-token workload 的 calibration
结果；换 GPU、backend 或请求 envelope 必须重新测量，不能直接推广。

## F-07：E5 non-stable VA 被通用 verifier 误判为 stable-VA 失败

### 现象与根因

non-stable qualification attempt-03 已完成模型执行、Parameter 重分配和 Graph recapture，
但 verifier 无条件要求 generation 前后的 Parameter VA 相同：

```text
VerificationError: g1: low-level Parameter VAs
```

这是 verifier 将 stable arm 的不变量错误套到 non-stable 对照，不是模型或驱动失败。

### 最小修复与复测

修改：

- `experiments/vllm_uvm_dynamic_kv_plan_rebuild/run_experiment.sh`：把
  `parameter_va_mode` 写入 `run_config.json`；
- `experiments/vllm_uvm_dynamic_kv_plan_rebuild/verify_experiment.py`：stable 模式继续要求
  VA 不变；non-stable 模式要求 generation-2 before 等于 generation-1 after、每代所有 VA
  逐项改变，且 diagnostics active original VAs 指向新地址。Graph recapture、输出、fault、
  cleanup 门保持不变。

attempt-04 qualification 2/2 generations 通过；attempt-05 formal 为 10 个独立冷进程、
20 generations，全部 8/8 Parameter VAs 改变，Graph 全量重捕获，输出/故障扫描/资源释放
均通过。stable/non-stable 配对显示 B1/B4 的 non-stable 总成本为 stable 的 1.998x/1.811x。

Artifacts：

- [`non-stable formal`](artifacts/e5-nonstable-va-recapture-REPRO-20260828-attempt-05-formal/)
- [`stable/non-stable comparison`](artifacts/e5-stable-vs-nonstable-va-REPRO-20260828-attempt-01/comparison.json)

## F-08：E9 冻结 reference snapshot 缺失

### 现象与补充内容

旧 preflight 因 `snapshot_20260812.json` 不存在而停止。按任务允许补充缺失内容，新建冻结
reference：

```text
experiments/eurosys_moe_offload_evaluation/inputs/e9_sharegpt_selection_reference_20260828.json
```

同时扩展 `run_kt_qualification.sh`/`verify_kt_qualification_case.py`，记录并校验 7/12/14/28
threads、frequency/uniform、Graph/eager、overlap 和 mixed-chunk 开关；新增
`run_formal_e9.sh` 与 `verify_formal_e9.py`。calibration/evaluation source indices 不相交，
calibration 覆盖 3,896,832 routing slots，冻结 4,736 GPU 和 1,408 CPU experts。

缺失文件问题已解决，七个 qualification case 均能启动并返回 0。随后暴露的跨臂 greedy
token 漂移是新的未解决正确性问题，已单独登记为 U-08；这里不把 E9 写成通过，也没有
启动 formal。

## 未作为“已解决修复”登记的事项

- E8 的缺失 provenance 没有绕过，因此没有代码修复；E9 缺失 snapshot 已按 F-08 补齐，
  新的 exact-generation 阻塞见 U-08。
- E2 的 warmup 后 rebalance、Graph/workspace 预留和 safety gate 是本轮验证的既有防护，
  不是为制造本轮通过而临时降低的标准。
