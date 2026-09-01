# E10 backup-threshold + tiered batch qualification（单轮）

## 结论

本次唯一一次 GPU qualification 已完整执行；baseline 和 three_queue 分别使用独立冷进程，
各运行一轮，未做统计重复。两个 case 的 runner return code 均为 0，离线 verifier 最终为
`status=ok`。

机制目标已经跑通：不使用请求预测，仅在 8-block backup 队列消耗达到 4 blocks 时，
将启动时冻结的 cold-frequency 档位从 `cold29` 批量切到 `cold30`，再执行 KV
`24 -> 40`；低压请求连续三个有效调度轮后，在请求完成边界先执行 KV `40 -> 20`，
再批量恢复到 `cold29`。

批量档位方案只带来有限降时，尚未隐藏同步维护等待，也不能证明突发请求性能稳定：

- 批量释放为 1964.577 ms，旧逐层方案为 2061.117 ms，减少 4.68%。
- 批量恢复为 670.610 ms，旧逐层方案为 830.706 ms，减少 19.27%。
- 高压 Prefill 为 2218.246 ms，旧逐层方案为 2323.473 ms，减少 4.53%。
- 高压 cycle E2E 为 3064.595 ms，旧逐层方案为 3157.611 ms，减少 2.95%。

以上新旧比较是两次单轮 qualification 的诊断性对比，不是统计性能结论。

## 冻结参数

| 参数 | 值 |
| --- | ---: |
| 专家策略 | cold-frequency |
| `cold29` selected bytes | 6,568,280,064 B |
| `cold30` selected bytes | 6,794,772,480 B |
| 档位差 | 每层 1 个专家，共 48 个专家 |
| 批量驻留差 | 96 个参数范围，226,492,416 B（216 MiB） |
| 初始 / 最大 / 最小 KV target | 24 / 40 / 20 blocks |
| backup reserve | 8 blocks |
| 触发阈值 | 已消耗 4 blocks（50%） |
| 预测 | 关闭；决策记录 `predicted_blocks=0` |
| 低压门 | 连续 3 个有效低压调度轮 |

## 单轮结果

| 指标 | baseline | three_queue | three_queue 相对 baseline |
| --- | ---: | ---: | ---: |
| 高压 Prefill | 153.310 ms | 2218.246 ms | +1346.90% |
| first Decode | 744.193 ms | 749.627 ms | +0.73% |
| steady Decode median | 10.049 ms | 11.320 ms | +12.64% |
| 高压 cycle E2E | 974.134 ms | 3064.595 ms | +214.60% |
| 低压 trace step sum | 1343.642 ms | 2224.528 ms | +65.56% |

three_queue 维护 RPC：

| RPC | 时间 |
| --- | ---: |
| `cold29 -> cold30` grouped Plan Class | 1964.577 ms |
| KV grow `24 -> 40` | 94.013 ms |
| KV shrink `40 -> 20` | 159.755 ms |
| `cold30 -> cold29` grouped Plan Class | 670.610 ms |

## 机制证据

- grow 决策为 `growth_trigger_policy=backup_threshold_only`，压力 level/threshold 均为
  4 blocks，`predicted_blocks=0`、`prediction_error_blocks=0`。
- grow RPC 顺序严格为
  `activate_uvm_expert_graph_class(cold30) -> resize_uvm_kv_cache_target(40)`。
- restore RPC 顺序严格为
  `resize_uvm_kv_cache_target(20) -> activate_uvm_expert_graph_class(cold29)`。
- 两个方向的 `plan_switch` 均证明 `session_reused=true`、`graphs_retained=true`、
  `incremental_topology=true`；驻留变化分别为 96 个范围 / 216 MiB。
- 两个档位各预捕获 3 个 Graph，总 capture count 为 6；运行期保持 `6 -> 6`，无重捕获。
- baseline 与 three_queue 的两轮输出逐 token 一致，KV/Parameter VA、三队列守恒、
  GPU/CPU 驻留、内核日志和清理均通过。
- 最终恢复 stock `nvidia_uvm`，GPU memory used 为 0 MiB。

## 为什么改进有限

本次已经把 96 个变化参数范围作为一个 Plan Class 差量处理，但当前稳定 VA/Graph 保留协议
仍需切换整套 alias 拓扑。grow 的 1953.537 ms `plan_switch.wall_ns` 中，PARK 约
524.019 ms、全拓扑 REBIND 约 270.973 ms、RESUME 约 452.510 ms；此外还有 216 MiB
驻留迁移、source refill、同步和驻留审计。因此它不是只提交少数差量 range 后立即返回，
请求仍同步承担约 1.96 s。

restore 从 `remote_parked` 状态执行，无额外 PARK/RESUME，所以比旧恢复明显下降，但仍包含
约 250.526 ms 的整拓扑 REBIND 和 216 MiB CPU-to-GPU prefetch。

## 产物

- 完整 artifact：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-three-queue-tiered-batch-REPRO-20260829-attempt-01-qualification/`
- 最终验证：artifact 下的 `verification.json`
- 两个原始结果：artifact 下的 `baseline/result.json` 与 `three_queue/result.json`
- 冻结协议：artifact 下的 `protocol.json`

GPU 运行结束后的第一次离线验证曾因 verifier 错把 activation 的目标字段写成
`class_name`（实际为 `new_class`）而失败。该字段检查已离线修正并对同一产物重验；没有再次
启动模型或重复 GPU 实验。
