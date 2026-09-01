# E10 三队列 cold-frequency 单轮 qualification（2026-08-29）

## 1. 结论

E10 已按要求完整成功运行一次，没有统计重复，也没有重跑 GPU case。baseline 和
three_queue 分别使用独立冷模型进程，PID 为 `17535` 和 `17855`。最终 verifier
`status=ok`。

本轮证明了以下功能链路可以在真实 Qwen3-30B-A3B-FP8 上完整往返：GPU backup 吸收
突发、cold-frequency 专家 wave 释放 HBM、KV pool 自动 `24 -> 40` grow、连续三个
低压轮后安全 `40 -> 20` shrink，以及专家恢复。输出、VA、物理驻留、队列守恒、
CUDA Graph 和清理门均通过。

但本轮不能证明性能提升。three_queue 的高压 cycle E2E 比固定 40-block baseline 高
`235.27%`，主要原因是同步专家释放和恢复成本。该结果是单轮功能 qualification，
不是 E10 正式的七策略、5-seed、重复峰谷 open-loop 性能矩阵。

## 2. 冻结协议

- Artifact：`artifacts/e10-three-queue-cold-frequency-REPRO-20260829-attempt-01-qualification/`
- 模型：Qwen3-30B-A3B-FP8
- case 顺序：baseline，然后 three_queue
- 每个 case：独立冷进程，仅一轮
- baseline：固定 KV target `40`，关闭 backup、autoscale 和 expert wave
- three_queue：初始 target `24`，backup `8` blocks，target 范围 `20..40`，启用
  autoscale 和 expert wave
- 专家策略：`cold-frequency`
- 选择规则：每层按冻结 calibration-decode 路由计数从低到高选择，计数相同时按
  `expert_id`；每层专家数量相同

冻结计划如下：

| 计划 | 含义 | SHA256 |
|---|---|---|
| 6 GiB static plan | 每层 29 个最低频专家 | `83415546ff8d9c339e23abb465a6021d35a08cb0242e17893ac9c24bd3ee1cbb` |
| 7 GiB superset | 每层 33 个最低频专家 | `1a80ec18871d5fde392c1bd9d9e377778c528a0274d85eeaf32adea171992726` |
| incremental wave | 每层释放/恢复 1 个 cold expert | `f3fcc1f47f4e18ae4467a43f2f675d88a6e3ebd7b6572e9d110bfc61d033a0e5` |

## 3. 功能结果

| 检查项 | 单轮结果 |
|---|---:|
| KV target | `24 -> 40 -> 20` |
| grow 时迁入 GPU 的 KV blocks | 16 |
| backup reserve / 实际消费 | 8 / 4 blocks |
| 低压前三轮决定 | `observe_low_pressure -> observe_low_pressure -> shrink_pending` |
| 释放 / 恢复专家 | 48 / 48（每层 1 个） |
| 专家迁移区域 | 226,492,416 B（216 MiB） |
| CUDA Graph capture count | `3 -> 3`，无 recapture |
| baseline 与 three_queue 输出 | 完全一致 |
| kernel faults | 0 |
| shutdown | session/backing/GPU/UVM client 清零 |

实验结束后 stock `nvidia_uvm` 已恢复；artifact 的 `module.after.json` 记录 GPU 使用量
`0 MiB`、无 compute PID、无 UVM client。

## 4. 单轮性能结果

正百分比表示 three_queue 延迟高于 baseline；由于每个 case 仅一轮，这些数值不包含
置信区间，不能作为正式性能结论。

| 指标 | 固定 40-block baseline | three_queue | 变化 |
|---|---:|---:|---:|
| 高压 Prefill | 153.01 ms | 2323.47 ms | +1418.55% |
| 首次 Decode | 717.02 ms | 742.33 ms | +3.53% |
| 稳态 Decode 中位数 | 10.48 ms | 11.31 ms | +7.93% |
| 高压 cycle E2E | 941.82 ms | 3157.61 ms | +235.27% |
| 低压 trace step 总和 | 1335.42 ms | 2369.48 ms | +77.43% |

three_queue 的迁移 RPC wall time：

| 转换 | 时间 |
|---|---:|
| 48 个专家释放 | 2061.12 ms |
| KV `24 -> 40` grow | 93.76 ms |
| KV `40 -> 20` shrink | 159.82 ms |
| 48 个专家恢复 | 830.71 ms |
| 合计 | 3145.40 ms |

专家释放与恢复占上述转换时间的 `91.94%`，KV grow/shrink 占 `8.06%`。因此本轮性能
不佳的直接原因是专家 wave 仍位于同步关键路径，而不是 cold-frequency 选择没有执行，
也不是 KV resize 本身占主导。

## 5. 证据与边界

增强版离线 verifier 仅修正了 RPC 计时字段对当前 artifact 中 `wall_ns` 的兼容；没有
重新执行 baseline 或 three_queue。关键文件 SHA256：

- `verification.json`：`de68dd9da363ede8b7427b94d20394fd3225bbc3721f8e6a7cb44d67741f2a96`
- `protocol.json`：`3b959d21dd2544a6efd70835ceb9dc74c85ee31b5c5c9be3b382d4c9f515d7c7`
- `baseline/result.json`：`57e727ce13eca1309f1cd4728bbff8fb9fcdd33fe45661c410dbc8cdf9229c51`
- `three_queue/result.json`：`7ae2d1f69fad9fb4d310056576b3220712f8403006ae1fed60aa71d1cc8a2d68`

本轮能够声称“三队列联合控制链路完整可用且正确”，不能声称“提高 burst goodput”或
“保持/提升端到端性能”。正式 E10 仍需完成 fixed K2、fixed K10、oracle、auto no-backup、
auto + fixed backup、auto + backup + expert waves、full joint 七种策略，并使用至少
5-seed 的重复峰谷 open-loop trace 报告 goodput、TTFT/SLO、OOM、HBM timeline 和置信区间。
