# E2/E3 兼容性故障根因、修复与复验

日期：2026-08-31（Asia/Shanghai）

## 1. 最终结论

E2 与 E3 的旧故障不能合并为同一个“cold39 误伤 hot40 页表”问题。

- **E3 根因已明确。** `FULL_DECODE_ONLY` 模式错误地把 817,889,280 B
  （约 780 MiB）的 pre-KV native-hole sentinel 保留到 kernel warmup/Decode
  Graph 捕获阶段。额外显存压力驱逐了部分热专家页；GPU 随后访问这些页时，
  `manual_migration_fault_mode=2` 按设计禁止 CPU→GPU 自动迁移，于是失败。
- **E2 旧故障不是 hot expert 被挤占，也不是已证明的专家页表破坏。** 旧故障
  地址 `0x825500dc3000` 不属于专家 Parameter VA；当前模块下，完整 E2 在开启
  和关闭“全 Parameter 映射重建”时均成功。因此旧 E2 失败没有足够证据唯一
  归因到某条专家页表代码，不能伪造一个确定根因。
- **参数没有配错。** E2 的 2 GiB KV、cold11 计划以及 C16 均能在同一机器、
  同一模型、同一当前模块下完成。E3 的 KV 大小和 cold plan 也未改变；错误是
  临时占位内存的生命周期与 Graph 模式不匹配。

## 2. E3 的直接根因证据

### 2.1 故障处理器看到的是“数据已被迁到 CPU”

在 `manual_migration_fault_mode=1` 的观察实验中，内核对故障页记录：

```text
action=observe_migration
gpu_resident=0
cpu_resident=1
gpu_read_mapped=0
selected_residency=GPU
```

这四项合起来表示：GPU 想读热专家，但该页当时只在 CPU，正常 UVM 服务将
执行 CPU→GPU 迁移。它与“数据仍在 GPU、只是页表入口丢失”不同；后者应为
`gpu_resident=1`。

同时，Graph capture gate 的全量驻留检查实际发现 22,020,096 B 热专家已在
CPU：

```text
queried_bytes = 26,726,105,088
gpu_bytes     = 26,704,084,992
cpu_bytes     =     22,020,096
```

因此原报告根据故障地址恰好位于 cold39/hot40 边界而推断“相邻页表被删除”是
错误的。该地址只是第一个被计算访问到的已驱逐热页，不代表 cold39 操作删除了
它的目录。

### 2.2 为什么只在 long/B8/C 暴露

该 cell 使用：

- 8 个并发请求；
- 512-token prompt + 512-token output；
- 817,889,280 B KV；
- `remote_only` + `FULL_DECODE_ONLY`。

KV 初始化前，runner 为固定本地分配预留同样大小的 sentinel。KV 已经分配完成
后，这块 780 MiB 内存仍被保留。随后 kernel warmup、autotune、Graph capture 和
PyTorch allocator reserve 继续申请显存，UVM 只能驱逐 managed 热专家页腾空间。
较小 workload/并发未达到这一压力点，所以先前 15 个 run 可以成功。

## 3. E3 修复

修改位置：

- `workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py`
- `workloads/vllm/vllm/tests/v1/worker/test_uvm_post_warmup_rebalance.py`

规则改为：

1. `FULL_DECODE_ONLY` 和 eager 模式在 KV 布局确定后立即释放 sentinel；
2. 只有确实还要捕获独立 Prefill Graph 的 `FULL_SEPARATE` 才把 sentinel 保留到
   pure-Prefill capture 前；
3. `remote_only` 在所有启动 warmup 完成后再做一次最终驻留收敛，防止 allocator
   reserve 在早期收敛之后再次驱逐热专家；
4. `manual_migration_fault_mode=2` 继续 fail-fast，不靠静默自动迁移掩盖容量错误。

修复后的关键内存变化：

```text
sentinel 分配后空闲                    2,001,469,440 B
KV 完成后、释放 sentinel 前            1,181,483,008 B
立即释放 817,889,280 B 后              1,999,372,288 B
热/冷专家收敛后、kernel warmup 前       1,865,154,560 B
```

## 4. E3 隔离复验：不依赖全 Parameter 页表重建

Artifact：

`artifacts/e3-long-b8-c-sentinel-fix-no-parameter-rebuild-20260831-attempt-01/`

本轮显式设置：

```text
VLLM_UVM_REBUILD_PARAMETER_GPU_READ_MAPPINGS=0
```

唯一一个独立冷进程 `long_generation/B8/C` 完整成功：

| 指标 | 结果 |
|---|---:|
| 成功请求 | 8 / 8 |
| 每请求输出 | 512 token |
| Prefill step | 133.49 ms |
| first Decode step | 16.43 ms |
| steady Decode median | 16.08 ms |
| 请求组 E2E | 8,438.85 ms |

无 CUDA error、Xid、FAULT_PDE/FAULT_PTE；输出长度、driver session 清理均通过。
这些是单次定位实验数据，不替代正式 E3 多 arm 性能结果。

机器检查：

`artifacts/e3-long-b8-c-sentinel-fix-no-parameter-rebuild-20260831-attempt-01/verification.json`

此前保留全 Parameter 重建时也有一次成功：

`artifacts/e3-long-b8-c-sentinel-lifetime-fix-20260831-attempt-01/`

其 8/8 请求成功、Prefill 182.60 ms、steady Decode median 20.50 ms、E2E
10,579.66 ms。两个成功实验共同说明：决定正确性的变量是 sentinel 生命周期，
不是全 Parameter 页表重建。两轮性能不可用于统计比较。

## 5. E2 重新归因

旧失败 artifact：

`artifacts/e2-direct-batch-pure-phase-fence-off-20260830-current-e10-compat-attempt-01/`

旧模块 `f09ba38d...` 在 C16 首次计算报告：

```text
Xid 31, FAULT_PDE, address=0x825500dc3000
```

该地址不在本轮记录的 52 个专家 Parameter VA 中。旧运行没有保存
`torch.cuda.memory_snapshot()`，因此现在无法再判断它属于 Graph private pool、
KV、workspace 还是另一项临时分配。把它称为 hot40 地址没有证据。

后续做了三项隔离：

1. 独立冷进程仅运行 C16：16/16 请求成功；
2. 按 C1→C4→C8→C16 完整运行，开启全 Parameter 映射重建：29/29 成功；
3. 同样完整运行，关闭全 Parameter 映射重建：29/29 成功。

第 3 项 artifact：

`artifacts/e2-four-cell-no-parameter-rebuild-20260831-attempt-01/`

| Cell | TTFT p50 | TPOT p50 | E2E p50 | 成功 |
|---|---:|---:|---:|---:|
| C1 | 811.58 ms | 42.89 ms | 2,141.03 ms | 1/1 |
| C4 | 889.37 ms | 65.45 ms | 2,918.30 ms | 4/4 |
| C8 | 799.63 ms | 78.47 ms | 3,232.28 ms | 8/8 |
| C16 | 811.51 ms | 89.94 ms | 3,599.74 ms | 16/16 |

`verification.json` 为 `status=ok`，且 `parameter_mapping_rebuild=null`。上述性能含
地址快照诊断，不作为正式 E2 性能结论。

因此，关于 E2 当前能严格成立的结论只有：

- C16/2 GiB KV/cold11 参数不是必然错误；
- E10 full-dual-view 没有在 E2 中误开启；
- 全 Parameter 映射重建不是当前 E2 成功的必要条件；
- 旧单次非专家地址故障在当前版本未复现。

若要求对旧 E2 故障继续做唯一代码级归因，必须保留或重建旧 `f09ba38d...`
模块，并在每个 cell 前后保存 CUDA allocator/Graph 地址快照；仅凭现有旧 artifact
无法科学地区分多个非专家分配。

## 6. 防护与诊断策略

- `VLLM_UVM_REBUILD_PARAMETER_GPU_READ_MAPPINGS` 默认改为关闭，只作为显式页表
  诊断/修复工具；E2 和修复后的 E3 均证明正常路径不依赖它。
- E2 runner 可通过 `E2_CUDA_ADDRESS_DIAGNOSTIC=1` 保存 allocator segments、
  Graph entries 和命名 tensor 范围；关键开关现在写入 `environment.json`。
- 逐页 `manual_migration_fault` 内核日志默认关闭，避免观察模式写出海量日志；
  需要定位时通过模块参数显式开启。
- 驻留错误继续 fail-fast；不通过开启自动迁移把错误转化为不可解释的性能波动。

## 7. 验证与最终状态

- sentinel 生命周期及 post-warmup 收敛相关测试：5 passed；
- E2/E3 shell runner 语法检查：通过；
- custom UVM 模块重新编译：通过；
- E2 no-rebuild：verifier `status=ok`；
- E3 no-rebuild：result `status=ok`，8/8 请求输出完整；
- 最终已加载 stock `nvidia_uvm.ko`，无 compute process，GPU 0 MiB。

结论：需要修复的是 E3 的临时显存生命周期，而不是调整冷专家/KV 参数或继续
修改共享 hot 页表。E2 的旧单次故障应保留为未能在当前版本复现的非专家 CUDA
地址故障，不能再作为 E10 页表冲突的证据。
