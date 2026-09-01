# E7 专家热度与 workload shift qualification 结果（2026-08-29）

## 结论

E7 六种方法已在同一 Qwen3-30B-A3B-FP8 进程中各完整运行一次，独立 verifier 为
`status=ok`。32 个 evaluation 请求用于冻结 pre-shift frequency-cold 计划，随后每个 arm
运行相同的 32 个 code-shift 请求；每层始终有 4 个动态 remote slots，共 192 slots。

本轮最主要的功能性结果是：GPU heat + CPU EWMA 将与 full-shift oracle 一致的 slot 比例
从 36.98% 提升到 88.54%，累计动态 remote assignment 从 frequency-cold 的 27,889 降为
13,910，减少 50.12%。但 32 个 shift 请求结束时仍未达到预设 95% 收敛门，因此本轮
`EWMA convergence time` 合法结果为“超过 32 requests”，不是 32 requests。

同步 EWMA 维护累计 5.794 s，折合 181.06 ms/request；adaptive 请求中位延迟为
1,084.64 ms，反而比 frequency-cold 的 947.28 ms 高 14.50%。因此这次 qualification
证明了 heat、EWMA、真实专家交换和远端命中下降链路，但不能声称端到端性能已经恢复。

## 冻结参数

| 参数 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB |
| 模型 | Qwen3-30B-A3B-FP8，snapshot `d206ba7...` |
| pre-shift | `evaluation_short_dialogue.json`，32×128 prompt tokens |
| shift | `shift_short_dialogue.json`，32×128 prompt tokens，code heuristic |
| 输出 | 每请求固定 32 tokens，greedy，`ignore_eos=true` |
| batch/window | B1；每 4 请求一次 heat snapshot/rebalance，共 8 windows |
| base static plan | cold-6G，29 experts/layer，6,568,280,064 B |
| 动态 slots | cold-7G − cold-6G，4 experts/layer，905,969,664 B |
| 总远端计划容量 | 7,474,249,728 B（6.9609 GiB） |
| EWMA | alpha=0.25，最多 1 swap/layer/window，min delta=1 |
| KV cache | 268,435,456 B |
| Graph | `FULL_SEPARATE`，capture sizes `[1,128]` |
| seed | 20260829 |

六个 arm 的定义如下：

- `frequency_cold`：由全部 pre-shift evaluation route 冻结每层最冷 4 个可交换专家；
- `random`：在非 static-alias 专家内按冻结 seed 分层抽取 4 个；
- `frozen_static`：直接使用 cold-7G 与 cold-6G 的冻结差集；
- `gpu_heat_cpu_ewma`：从 frequency-cold 出发，每 4 请求读取 GPU heat、做 CPU EWMA 并
  执行最多一个真实 swap/layer；
- `oracle`：使用 frequency-cold arm 已观察到的完整 32-request shift route 离线选最冷
  4 个，仅作带未来信息的上界；
- `hot_adversarial`：用同一完整 shift route 选择每层最热 4 个，作为下界压力计划。

static-alias-owned 专家被显式排除在动态候选集合之外，六臂的逐层动态专家数完全相同。

## 六臂结果

下表都是本轮单次点估计。请求 p95 使用 nearest-rank；plan hit 是动态 remote slots 与
oracle slots 的重合率，Jaccard 是 48 层逐层 Jaccard 的均值。

| Arm | 中位 / p95 latency (ms) | Output tok/s | Dynamic remote rate | Total remote rate | Plan hit | Mean Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frequency-cold | 947.283 / 1277.593 | 30.950 | 1.4274% | 14.7229% | 36.98% | 0.2591 |
| random | 1047.451 / 1313.916 | 29.639 | 3.1855% | 16.4810% | 4.17% | 0.0238 |
| frozen static | 1050.033 / 1309.979 | 29.471 | 1.6877% | 14.9831% | 18.23% | 0.1147 |
| GPU heat + CPU EWMA | 1084.641 / 1363.375 | 28.204 | 0.7119% | 14.0074% | 88.54% | 0.8222 |
| oracle | 1061.692 / 1354.773 | 29.115 | 0.1784% | 13.4738% | 100% | 1.0000 |
| hot-adversarial | 1271.753 / 1511.702 | 24.961 | 12.4842% | 25.7797% | 0% | 0.0000 |

Oracle 的动态 remote assignments 为 3,485，是五个固定 arm 中的严格最小值；
hot-adversarial 为 243,916，是严格最大值。hot-adversarial 相对 oracle 的请求中位延迟高
19.79%，output tok/s 低 14.27%，说明本轮 route/placement 指标确实对应可观测的性能压力。
但各 arm 只有一次且顺序未平衡，其他相邻 arm 的小幅 latency 差异不作统计解释。

## EWMA 轨迹与维护开销

| Window | 累计请求 | Dynamic remote rate | Oracle plan hit after maintenance | Swaps | Maintenance (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 1.9273% | 43.75% | 45 | 1056.446 |
| 1 | 8 | 1.1186% | 50.52% | 43 | 1147.269 |
| 2 | 12 | 1.1203% | 60.94% | 38 | 1019.728 |
| 3 | 16 | 0.4598% | 67.19% | 27 | 711.267 |
| 4 | 20 | 0.4443% | 75.00% | 24 | 651.128 |
| 5 | 24 | 0.2838% | 82.29% | 21 | 567.827 |
| 6 | 28 | 0.1793% | 84.38% | 11 | 314.507 |
| 7 | 32 | 0.1621% | 88.54% | 12 | 325.780 |

EWMA 共执行 221 个在线 swap；按显式 ExpertRecord region geometry，promotion H2D 与 demotion
D2H requested bytes 均为 1,042,808,832 B。同步维护的 window median/p95 为
681.198/1147.269 ms。对照固定 arm 仍每 window 同步读取 heat 以获得实验指标，其 snapshot
维护中位数约 5.6--6.9 ms；这些观测开销不计入请求 latency。

最后一窗请求中位延迟为 1,142.885 ms，第一窗为 1,154.298 ms，仅改善 1.00%，且最后一窗
仍是同窗 frequency-cold 的 1.118 倍。由于未到 95% 收敛门，本结果将该值标为
`uses_last_window_when_not_converged=true`，不称为“恢复时间”。

## 正确性、bytes 与清理

- calibration 加六臂共完成 224 个请求、7,168 个输出 tokens；六臂输出 hash 和完整 route
  count hash 一致；
- dynamic slot 始终为 48×4，static alias 与动态候选无重叠；
- CUDA Graph capture counter 从加载后的 2 到最终仍为 2，无 request-time recapture；
- heat counter 为 `graph_stable_synchronous`，避免 Graph replay 固定地址与双缓冲指针切换
  不一致；
- static remote plan 为 6,568,280,064 B，动态 remote plan 为 905,969,664 B；
- 当前 UVM 计数器不能把 demand-read bytes 逐 arm 归因，因此该项明确为 unavailable；没有
  用 remote assignment × expert size 伪造 bytes；
- 成功窗口 kernel log 无 Xid、BUG/Oops、PTE/PDE fault 或 CUDA illegal-access 签名；
- runner/verifier 均为 `status=ok`，distributed cleanup 完成，最终恢复 stock UVM，GPU
  0 MiB 且无 compute/UVM client。

## 失败 attempt 与修复

- attempt-01 在 Graph capture 前启用旧 heat 路径，布尔索引和 `bincount` 使 capture
  invalidated；未进入任何 E7 arm。修复后用稳定地址的 `scatter_add_`，Graph 模式同步
  snapshot/reset，非 Graph 模式继续保留双缓冲；独立 int32/int64 CUDA Graph probe 通过。
- attempt-02 完成 calibration 和 frequency-cold arm，随后输出 hash 表因字符串/整数请求键
  不一致触发 `KeyError(0)`；修复后 synthetic parity 测试通过。
- attempt-03 为唯一完整成功的六臂 qualification。所有失败目录原样保留，未补写为成功，
  partial arm 不进入本结果表。

## 结果边界

这是“先跑通并冻结参数”的单进程、固定 arm order、每臂单样本 qualification，不报告置信
区间，也不支持 adaptive 优于任一 baseline 的统计结论。U-06 的 workload-shift E2E
功能缺口已闭合，但正式论文结果仍需至少 5 seeds、独立冷进程或平衡顺序，以及相同重复数的
median/p95/CI。异步 pinned D2H 尚未实现，当前维护必须继续标注 synchronous。

## Artifacts

- 成功：`artifacts/e7-workload-shift-REPRO-20260829-attempt-03-qualification/`
- Graph capture 失败：`artifacts/e7-workload-shift-REPRO-20260829-attempt-01-qualification/`
- runner key 失败：`artifacts/e7-workload-shift-REPRO-20260829-attempt-02-qualification/`
- 成功 result SHA256：`3190d36d37a0ca9eaccb5bbeb35d4dcfc3b9f553815e0ea7240df1ade917187c`
- 成功 verification SHA256：`77e1d258e7f236ab35db00acd88d00ba8b08f98f19286003526b8b2d929ffb29`
