# E10 CPU副本预填 + 预编译档位事务 pilot（单轮）

## 范围

本次只执行了一次 GPU pilot。固定 40-block baseline 和 three_queue 各使用一个独立
冷进程、各运行一轮，没有统计重复。最终 verifier 为 `status=ok`。

实验继续使用冻结的 cold-frequency `cold29/cold30`：每层增加 1 个冷专家，共 48 个
专家、96 个参数范围、226,492,416 B（216 MiB）。不使用请求预测，backup 8 blocks
消耗到 4 blocks 时触发释放。

## 实现

1. CPU副本预填：初始化时利用已经按 `cold30` 最大宽度分配的 pinned CPU source，填入
   `cold29` 之外的 96 个参数 slot。没有另建第二个 6.8 GiB buffer。
2. 预编译档位事务：初始化 Graph class capture 第一次执行双向切换时，缓存稳定 slot
   几何、完整 map entries、backing entries 和驻留差量；正式请求只按
   `cold29->cold30` 或 `cold30->cold29` 取缓存提交。

启动证据：

- CPU source 预填 `96` slots / `226,492,416 B`。
- 注册 `cold29->cold30`：96 个新增 CPU range，缓存 13,500 entries。
- 注册 `cold30->cold29`：96 个恢复 GPU range，缓存 13,050 entries。
- 正式双向切换的 `instruction_cache_hit` 均为 `true`。
- 释放时需要的 96 slots / 216 MiB 全部命中预填副本，运行时
  `source_refill.changed_bytes=0`。

## 单轮结果

| 指标 | 旧 grouped qualification | 本次 pilot | 变化 |
| --- | ---: | ---: | ---: |
| `cold29 -> cold30` RPC | 1964.577 ms | 1892.124 ms | -72.453 ms（-3.69%） |
| `cold30 -> cold29` RPC | 670.610 ms | 621.763 ms | -48.847 ms（-7.28%） |
| 高压 Prefill | 2218.246 ms | 2148.325 ms | -69.920 ms（-3.15%） |
| 高压 cycle E2E | 3064.595 ms | 3010.974 ms | -53.621 ms（-1.75%） |
| three_queue load | 83.885 s | 84.490 s | +0.605 s（+0.72%） |

以上是两个不同单轮 artifact 的诊断性对比，不是统计加速结论。

本次释放的 `plan_switch.wall` 为 1888.947 ms，其中：

- PARK 527.010 ms；
- 全拓扑 REBIND 261.067 ms；
- RESUME 457.601 ms；
- backing preallocate 2.448 ms；
- 其余迁移、同步和审计约 640.821 ms。

旧方案对应的其余部分约 703.617 ms。本次减少约 62.8 ms，与消除运行时216 MiB
source refill 的方向一致。但 PARK/REBIND/RESUME 仍约 1.246 s，且 REBIND 仍提交
完整的13,500 entries，所以总体降幅有限。

恢复方向不执行 PARK/RESUME；本次 REBIND 为254.212 ms，剩余约364.765 ms仍包括
216 MiB CPU-to-GPU prefetch、同步和驻留审计。

## 正确性和清理

- baseline 与 three_queue 输出 token hash 一致；
- CUDA Graph capture count 为6，无运行期重捕获；
- kernel faults 为0；
- verifier `status=ok`、`shutdown_clean=true`；
- 最终恢复 stock `nvidia-uvm`，GPU memory used 为0 MiB。

## 结论

两个方案均已在真实 GPU 路径生效：运行时216 MiB GPU-to-CPU source refill被完全
消除，正式双向切换均命中初始化期缓存。不过性能稳定目标仍未达到；本次 three_queue
相对同轮 baseline 的高压 Prefill仍高1293.50%，cycle E2E仍高217.96%。后续主要瓶颈
已经不是CPU现场生成计划，而是全拓扑暂停/恢复、全量REBIND以及同步驻留迁移。

## 产物

- Artifact：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-prefill-precompiled-pilot-20260830-attempt-01/`
- 最终验证：artifact 下的 `verification.json`
- 原始结果：artifact 下的 `baseline/result.json` 和 `three_queue/result.json`
- 冻结协议：artifact 下的 `protocol.json`
