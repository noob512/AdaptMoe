# E6 Plan Class、serial wave 与切换摊销 qualification 结果（2026-08-28）

## 结论

E6 的 full rebuild、serial wave 和全层 batched difference 三臂均已按同一冻结几何完整运行
一次。每臂包含初始 base 请求以及 5/10/25/50/100% 的 outbound/return 共 10 次切换；三个
verifier 均为 `status=ok`，所有请求输出 hash 一致：

```text
a365db717e87663b5f807b34fc6013b44f98cd925bea9480cece8e65eac690b9
```

在本轮点估计中，batched difference 的切换成本为 0.462--2.344 s，相对 full rebuild
快 6.71--26.38 倍。相对 serial wave，5% 的单 wave 基本相同；从 10% 到 100% 快
1.51--3.79 倍。full rebuild 每次销毁并重捕获 2 个 Graph，capture counter 从 2 增至
22；serial wave 和 batched difference 均保持为 2，没有请求期 recapture。

这是一组单进程、单样本 qualification，不报告 median、p95 或置信区间。verifier 中
batched arm 的 `median_ms` 仅因为 `count=1`，不作为统计结论。

## 冻结参数

| 参数 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| 模型 | DeepSeek-V2-Lite-Chat，BF16 |
| 请求 | B1，Prompt 512，Output 8 |
| KV cache | 268,435,456 B（256 MiB） |
| slot budget | 10,737,418,240 B（10 GiB） |
| 计划 | C8，26 层 × 每层 19 experts = 494 slots |
| selected bytes | 8,546,942,976 B |
| retained backing | 657,457,152 B |
| CUDA Graph | `FULL_SEPARATE`，capture sizes `[1,512]` |
| seed | 20260828 |
| requested difference | 5/10/25/50/100% |

实际变化分别为 25/49/124/247/494 slots，即
5.0607/9.9190/25.1012/50/100%。三臂使用相同 selected bytes、backing、请求几何和
`base -> difference -> base` 顺序。

## 切换成本

单位为 ms，格式为 `outbound / return`。full rebuild 使用 `rebuild.wall_ns`；serial wave
使用完整 activation wrapper `switch_ns`；batched difference 使用 verifier 认证的
`plan_switch.wall_ns`。

| Difference | Full rebuild | Serial wave | Batched difference |
| --- | ---: | ---: | ---: |
| 5% | 11666.252 / 12192.664 | 487.181 / 464.712 | 486.747 / 462.128 |
| 10% | 12814.950 / 12662.104 | 968.773 / 928.585 | 641.390 / 606.238 |
| 25% | 13090.678 / 12922.536 | 2421.190 / 2328.019 | 946.077 / 942.079 |
| 50% | 13794.415 / 13687.190 | 4549.663 / 4536.805 | 1400.371 / 1433.851 |
| 100% | 15715.351 / 15728.258 | 8760.805 / 8730.215 | 2313.373 / 2344.490 |

serial wave 的 wave 数为 1/2/5/10/19。5% 只有一个 wave，因此和单次 batched transaction
几乎相同；变化增大后，serial 路径的重复 wrapper/transaction 成本随 wave 数上升。
full rebuild 的 11.7--15.7 s 包含 plan/session rebuild、Graph reset 和 recapture，不能只
解释为数据移动。

## Break-even 点估计

每个方向严格使用该臂相邻两个请求：

```text
denominator_ms = previous_request_ms - current_request_ms
requests_to_amortize = transition_ms / denominator_ms
```

分母不大于 0 时记为“无 break-even”。三臂全部 outbound 的新计划请求均比前一个 base
请求慢，所以本轮所有 outbound 都无 break-even。return 结果如下；括号内为第一个覆盖
切换成本的整数请求数。

| Difference | Full rebuild return | Serial wave return | Batched difference return |
| --- | ---: | ---: | ---: |
| 5% | 1627.4（1628） | 98.2（99） | 51.9（52） |
| 10% | 1347.9（1348） | 65.7（66） | 无（分母 -4.423 ms） |
| 25% | 668.3（669） | 237.2（238） | 35.1（36） |
| 50% | 406.1（407） | 195.7（196） | 38.2（39） |
| 100% | 208.6（209） | 158.5（159） | 29.9（30） |

不同机制位于独立冷进程，break-even 的请求分母只在各自进程内计算，没有跨进程混用请求
耗时。单样本下数毫秒的请求差可能改变分母符号，例如 batched 10% return；这些值只用于
确认公式、记录当前点估计和冻结后续正式重复参数，不表示稳定阈值。

## 正确性与清理

- 三个 runner return code 均为 0，runner/verifier 均为 `status=ok`；
- selected bytes、retained backing、输出 token/hash 和全部 10 个切换方向一致；
- full rebuild 的每代 parameter VA 保持稳定，10 次重建均销毁并重捕获 2 个 Graph；
- serial/batched 均复用 session、保留 Graph、执行 incremental topology，capture count 恒为 2；
- cold/hot residency、reference release、distributed cleanup 和 full rebuild controlled
  hard-exit contract 通过；
- 三个成功 artifact 的 kernel window 无 Xid、PTE/PDE fault、CUDA illegal access 或 kernel
  fatal signature；artifact 结束均恢复 stock UVM，GPU 0 MiB 且无 compute/UVM client。

## 失败 attempt 与修复边界

失败目录全部保留，未将失败或中断记录补写成成功：

- batched difference attempt-01 被旧同宽 class 契约拒绝；attempt-02 因同 shape GraphExec
  重复保留，清理后只释放约 7 MiB；attempt-03 为本轮成功记录；attempt-04 被中断，不纳入；
- full rebuild attempt-01 被公共 harness 覆盖局部 runner；attempt-02 在 parked session 上直接
  destroy，首次 cold release 返回 `NV_ERR_INVALID_STATE`；attempt-03 修复 destroy 顺序后又
  暴露共享 cold range 被混合作为整组审计的问题；attempt-04/05 完成 10 次重建，但分别在
  Python/C++ 全局析构阶段以 rc134 退出；attempt-06 为成功记录；
- serial wave attempt-01 的父/子 `RUN_ROOT` 令 plan 路径错误，GPU 工作前停止；attempt-02
  是该臂第一次实际 GPU 运行并成功。

修复后的 full rebuild 在 destroy 前先 RESUME，将旧/新 cold 交集作为已知 CPU ranges 单独
证明，只实际 release 新增 cold ranges；runner 显式释放 Graph/UVM/distributed 引用，并在
cleanup 已闭合后使用仓库已有的 controlled hard-exit 合同，规避 PyTorch 2.9 多 Graph pool
的全局析构 abort。该 hard exit 不是绕过 cleanup，verifier 强制要求 cleanup 和合同同时成立。

## 结果边界

- 本轮目标是“先完整跑通并冻结参数”，没有再启动三臂统计重复；正式论文数字仍需对所有臂
  使用独立冷进程、平衡顺序和相同重复数后报告 median/p95/CI。
- 设计文档中的 95.9 s Qwen 串行 wave 是另一模型/几何的历史优化前基线，继续保留，但不与
  本轮 DeepSeek C8 点估计混算 speedup。
- 当前完成的是固定 C8、变化比例扫描；C2/C4/C6 的容量轴仍不是本轮统计覆盖范围。

## Artifacts

- 离线汇总：`artifacts/e6-qualification-summary-REPRO-20260828-attempt-01/summary.json`
- batched difference：`artifacts/e6-difference-REPRO-20260828-attempt-03-qualification/`
- full rebuild：`artifacts/e6-full-rebuild-REPRO-20260828-attempt-06-qualification/`
- serial wave：`artifacts/e6-serial-wave-REPRO-20260828-attempt-02-qualification/`
