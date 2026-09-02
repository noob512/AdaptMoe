# E13：KV Cache 自适应档位与初始化固定档位

日期：2026-09-01；最后更新：2026-09-02（Asia/Shanghai）

性质：单机单卡；最新在线机制为 1 个独立冷进程 seed 的 qualification，历史性能主表为
3 个独立冷进程 seed 的 formal-profile qualification

最新在线同步档 artifact：
`artifacts/e13-kv-online-synchronized-2gb-20260901-attempt-08-qualification`

历史 drain-at-idle 正式 artifact：
`artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete`

容量 artifact：`artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal`

不计切换开销的稳态阶段 artifact：
`artifacts/e13-tier-phase-once-20260901-attempt-06`

K2/K6 固定档与 Adaptive 回环 artifact：
`artifacts/e13-k2-k6-roundtrip-20260901-attempt-01`

Short/B6 固定档补充 artifact：
`artifacts/e13-fixed-k2-k6-short-b6-20260901-attempt-01`

Short/B64 固定档补充 artifact：
`artifacts/e13-fixed-k2-k6-short-b64-20260901-attempt-01`

Short B1/B16/B64 同进程档内矩阵 artifact：
`artifacts/e13-fixed-k2-k6-short-b1-b16-b64-20260901-attempt-01`

Short/B128 固定档补充 artifact：
`artifacts/e13-fixed-k2-k6-short-b128-20260901-attempt-01`

Short/B128 Decode ABBA定因 artifact：
`artifacts/e13-short-b128-decode-abba-20260901-attempt-01`

双阶段 CUDA Graph 的 Fixed/Adaptive B64/B128 回环 artifact：
`artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02`

Decode 定因 artifact：

- `artifacts/e13-decode-abba-diagnostic-20260901-attempt-01`
- `artifacts/e13-decode-backing-diagnostic-20260901-attempt-01`
- `artifacts/e13-decode-nsys-diagnostic-20260901-attempt-03`

## 1. 结论摘要

E13 已实现 scheduler quiescent point 在线扩缩、连续空闲尾部缩档，以及 KV Cache 与
专家 Plan Class 同步的约 2 GB 相邻档事务。最新单轮 qualification 的机制验收闭环，
但包含同步切换开销后仍不支持“Adaptive 端到端性能优于最佳固定档”的性能主张。

- 最新在线矩阵 4/4 冷进程返回 0、128/128 请求成功，verifier 为 `status=ok`；
  0 拒绝、0 失败、0 preemption、0 OOM，无 Xid/MMU fault，最终恢复 stock UVM，
  GPU 为 0 MiB。
- Adaptive 完成 7 次在线 resize，顺序为
  `K4→K6→K4→K2→K4→K2→K4→K2`；3 次增长和 4 次缩减全部成功，所有实际边均为
  K2↔K4 或 K4↔K6，没有直接跨 4 GB。
- K2↔K4 迁移 1,366 个 KV block，即 2.149 GB；K4↔K6 迁移 1,365 个 block，
  即 2.147 GB。对应专家卸载量同步变化 2.265 GB 和 2.038 GB。
- 4 次缩档提交前的 `max_active_block_id` 均不超过新 target；其中动态混合时仍有
  66 个 active block，burst 时仍有 9 个 active block，证明不再要求请求全部完成。
- burst 的 K2→K4 在已有请求运行时提交，`committed_with_running_requests=true`；
  5 个 late RAG 请求没有进入 deferred admission，逻辑到达至准入为 743.345 ms，
  相比历史 26.519 s 缩短 97.20%，即约 35.7×。该场景 12/12 请求（包括切换时仍在
  生成的 256-token 请求）与三个 Fixed 臂的完整输出 hash 均一致。
- 7 次在线 resize 合计 2.772 s，单次范围 130.204–733.546 ms；所有恢复调度后的
  时序点都满足 K2/cold70、K4/cold80、K6/cold89 一致，三队列错误为 0。
- 最新单轮 Adaptive 完整 trace 为 0.3090 req/s，global best Fixed-K4 为
  0.3854 req/s，Adaptive 为 0.802×。这是单 seed qualification，不能替代三重复
  formal，也不能据此声称统计显著的性能改善。

以下三重复结果来自历史 drain-at-idle 协议，用于保留原正式性能基线：

- 动态矩阵和容量包络各 12/12 case 返回 0、verifier 为 `status=ok`；分别
  384/384 和 456/456 请求成功，0 拒绝、0 失败、0 preemption、0 显式 OOM，
  无 Xid/MMU fault。
- Adaptive 三轮轨迹均为
  `K4→K2→K6→K4→K2→K4→K2`，每轮 2 次增长、4 次回缩、0 次失败尝试。
- Adaptive 完整 trace 吞吐为 `0.2163 req/s`，global best Fixed-K6 为
  `0.3030 req/s`；Adaptive 为 `0.714×`，即低 28.62%。输出 token 吞吐为
  `10.705` 对 `14.997 tok/s`。
- Adaptive 每轮 6 次切换的同步迁移墙钟平均为 `17.169 s`，范围
  `17.061–17.240 s`；当前约 148.5 s trace 无法摊销这一开销。
- 高压 RAG/B6 时 Adaptive 从 K2 增长到 K6，同时运行数从 Fixed-K2 的 2 提高到 6；
  burst RAG/B5 也会在准入前从 K2 增长到 K4，同时运行 5 个请求。Fixed-K4/K6
  仍能达到相同或更高并发且更快。
- 容量扫描 B2/B5/B6/B7/B8/B10 得到 fully-resident 上限：Adaptive/Fixed-K2/
  Fixed-K4/Fixed-K6 = `7/2/5/7`，最大 live context 为
  `57,456/16,416/41,040/57,456 tokens`。Adaptive 超过低固定档，但不超过 Fixed-K6。
- 四臂都完成相同的单请求 8,192-token Prompt 和 8,208-token 上下文；B8/B10 的
  超额请求排队后完成，没有 Adaptive 独有的 OOM/拒绝避免或更长单请求上下文能力。
- 低压、高压、动态混合、突增恢复四个场景中，Adaptive 吞吐都低于对应
  best-fixed；不能声称长期变化负载下优于任一固定档。
- 补充的单轮稳态阶段对照把切换、源档 conditioning 和目标档 warmup 排除在计时窗外：
  Prefill 6/6 cell 均落在 switched/direct `0.984×–1.006×`。Decode 原始冷进程表
  只有 3/6 落在 ±10% 带内，但后续同进程 ABBA 中 K4/K6 四点为
  `0.983×–1.013×`，原先 K4/B8 的 `1.324×` 和 K6 的约 `0.78×` 均消失。
  定因为固定 5-step warmup 未消除 remote-read 历史状态，而不是切档路径回退。
- 新增的 K2/K6 单轮回环对照中，低压 Fixed-K2 吞吐比 Fixed-K6 高 6.50%；高压
  RAG/B6 的 Fixed-K6 吞吐比 Fixed-K2 高 25.68%，并把同时运行数从 2 提升到 6。
  排除切换与目标档 warmup 后，Adaptive 初始 K2、增长后 K6、回切后 K2 的
  Prefill/Decode/batch-wall 共 18/18 个稳态 cell 均落在对应 Fixed 档的 ±10% 内。
- 补充的 Short-128/B6 单轮点中，两档都能同时驻留全部 6 个请求；Fixed-K2 吞吐
  比 Fixed-K6 高 4.91%，TTFT 和 E2E 分别低 19.04% 和 4.69%。
- 孤立 Short-128/B64 点曾出现 K6 吞吐高 12.34%的表观 crossover；最新档内
  B1→B16→B64 矩阵没有复现。新矩阵的 B16/B64 均为 K2 更快 6.67%/7.57%，并直接
  观察到 K6/B1 Decode 从约 79 ms 收敛到 63 ms，确认固定 warmup 的历史状态混杂。
- Short-128/B128 中两档均完整并发128请求；K6吞吐仅高1.12%、E2E低1.11%，而
  K2 Prefill/TTFT低14.57%/14.51%。Decode样本仍有历史快慢态，整体约1%的差距
  不构成稳定档位优势。后续同进程ABBA已确定原表K2 Decode较慢是warmup不足导致
  的历史状态采样错误，不是K2固有回退。
- 最新 `FULL_SEPARATE` 双阶段 Graph 回环重新测量消除了上述 eager/冷进程快慢态
  对正式结论的影响。Fixed-K2 在 B64/B128 的吞吐分别比 Fixed-K6 高
  15.09%/21.86%，E2E 分别低 13.12%/17.99%。排除 K2↔K6 档位事务和每档 warmup
  后，Adaptive 初始 K2、增长后 K6、回切后 K2 的吞吐、Prefill、steady Decode、
  E2E 共 24/24 个比较均位于对应 Fixed 的 ±10% 内。全部 54 个性能比较中有
  49/54 在带内；5 个 P→D cutover step 高 10.92%–24.03%，因此
  不能声称快速 P→D 切换延迟本身完全等价。

### 1.1 最新在线同步档 qualification

四臂使用相同 seed、fixture、请求序列和 `trace_sha256`。下表均为单个冷进程点估计，
不报告伪造的波动范围。

| Arm | 成功请求 | trace | req/s | output tok/s | TTFT P99 | E2E P99 | runtime resize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive | 32/32 | 104.058 s | 0.3090 | 15.294 | 28079.11 ms | 31150.33 ms | 7/7 |
| Fixed-K2 | 32/32 | 84.862 s | 0.3784 | 18.733 | 23529.13 ms | 24802.68 ms | 0 |
| Fixed-K4 | 32/32 | 83.340 s | 0.3854 | 19.076 | 20552.59 ms | 25089.61 ms | 0 |
| Fixed-K6 | 32/32 | 85.521 s | 0.3755 | 18.588 | 21031.68 ms | 26289.11 ms | 0 |

Fixed-K6 初始化也严格按 K2→K4→K6 两个相邻事务建立，不执行 K2→K6 直跳。

| 场景 | Adaptive req/s | Best fixed | Best fixed req/s | Adaptive/best-fixed |
| --- | ---: | --- | ---: | ---: |
| low_pressure | 0.349 | Fixed-K2 | 0.371 | 0.940× |
| high_pressure | 0.850 | Fixed-K6 | 0.957 | 0.888× |
| dynamic_mixed | 0.297 | Fixed-K4 | 0.388 | 0.766× |
| burst_recovery | 0.236 | Fixed-K4 | 0.308 | 0.766× |

Adaptive 的 7 个物理 KV 切换点如下。`active/max-active` 来自提交前队列快照；所有
shrink 都满足 `max-active <= new target`。

| 场景 | 切换 | active/max-active | KV 迁移 | 墙钟 |
| --- | --- | ---: | ---: | ---: |
| high | K4→K6 | 0/0 | H2D 2.147 GB | 733.546 ms |
| mixed | K6→K4 | 66/453 | D2H 2.147 GB | 130.204 ms |
| mixed | K4→K2 | 66/453 | D2H 2.149 GB | 170.171 ms |
| mixed | K2→K4 | 66/453 | H2D 2.149 GB | 714.880 ms |
| burst | K4→K2 | 9/278 | D2H 2.149 GB | 132.853 ms |
| burst | K2→K4 | 9/258 | H2D 2.149 GB | 720.882 ms |
| burst | K4→K2 | 9/258 | D2H 2.149 GB | 169.073 ms |

`resize_events.plan_after` 对 shrink 记录的是“KV 物理迁移刚完成”的事务中间态；随后同一
scheduler boundary 内同步提交目标 Plan Class，期间没有调用下一次 `schedule()`。
恢复调度后的 971 个时序点中，KV/Plan Class 不一致为 0。

完整矩阵输出 hash 为 89/96 一致，7 个分叉均来自非 burst 场景，且共同首 token 全部
一致；burst 为 36/36 配对 hash 一致。这仍是数值诊断，不能把全矩阵的 `status=ok`
解读为逐 token 完全等价。

## 2. 修复内容与最终协议

attempt-06 的 burst late cohort 直接进入 scheduler，未向控制器提交压力，因此错误地
停留在 K2。最终实现修复了以下问题：

1. 每个 late cohort 在逻辑到达时都提交 next-request 压力预测；
2. 若已有请求运行，增长成为 pending，late cohort 暂不进入 scheduler；
3. 既有运行请求清空后，在安全边界先切 Plan Class 和物理 KV target，再准入 cohort；
4. pending grow 优先取消 pending shrink，普通 `EngineCore.step()` 也会提交安全的
   pending grow；
5. 当前档能容纳 active cohort 时，不再把非紧急 high-watermark 增长拖到请求结束后
   执行，避免无效的迟到扩档；
6. verifier 现在要求每个 Adaptive run 的 burst 内都有 grow 和其后的 recovery shrink，
   并要求全部 4 个 late cohort 都留下统一控制器记录；
7. 修正完整输出 hash 的比较分母：只比较 Adaptive 与三个 Fixed，不再把
   Adaptive 自身的 32 个请求计入匹配数。

GPU 定向 qualification 依次保留为 attempt-07、08、09。attempt-08 证明直接 K2→K6
虽少一次切换，但大跨档迁移更慢；最终选择最小可容纳档位，burst 使用 K4。
capacity attempt-11 的 Output-1 请求在单步内结束，post-step 快照无法观测 resident
集合，因此保留为失败的测量设计；attempt-12 改为 Output-16 后通过 pilot，attempt-13
完成三重复正式容量包络。

实现入口：

- `e13/config.json`
- `e13/run_e13_kv_adaptive_matrix.sh`
- `e13/e13_kv_adaptive_runner.py`
- `e13/verify_e13_kv_adaptive.py`
- `e13/aggregate_e13_kv_adaptive.py`
- `e13/plot_e13_kv_adaptive.py`
- `workloads/vllm/vllm/vllm/v1/engine/core.py`

## 3. 冻结环境与档位

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB |
| Driver / kernel | 595.71.05 / 6.13.12-061312-generic |
| CPU | Intel Xeon Gold 6530，14 CPUs，单 NUMA node |
| Model | Qwen3-30B-A3B-FP8 |
| Attention / execution | TRITON_ATTN / eager |
| max model len / max seqs | 8,448 / 10 |
| 重复 seed | 20260901、20260917、20260929 |
| NVML 采样间隔 | 200 ms |

KV 总池为 8,589,410,304 bytes，每 block 1,572,864 bytes。GB 为十进制单位：

| 档位 | GPU KV target | CPU/offline KV | Plan Class | 专家卸载 |
| --- | ---: | ---: | --- | ---: |
| K2 | 2.145 GB | 6.442 GB | cold70 | 15.854 GB |
| K4 | 4.294 GB | 4.294 GB | cold80 | 18.119 GB |
| K6 | 6.441 GB | 2.147 GB | cold89 | 20.158 GB |

Adaptive 从 K4 启动；Fixed-K2/K4/K6 初始化后不再切换。每个 repeat 内四臂的 fixture、
seed、请求轨迹和 `trace_sha256` 完全相同。

## 4. 工作负载

| 场景 | 请求 |
| --- | --- |
| low_pressure | 4 个 Short-128/Output-32，B1，低速到达 |
| high_pressure | 6 个 RAG-8192/Output-16 并发请求 |
| dynamic_mixed | Short、Ordinary、Medium、RAG cohort 按 engine step 注入，共 10 请求 |
| burst_recovery | 低压→运行中注入 RAG/B5 burst→连续低压恢复，共 12 请求 |

## 5. Adaptive 与各 Fixed 档位

req/s 为 3 个独立冷进程的均值 `[min,max]`；延迟为 3 个进程合并请求后的 P99；
GPU util 为三重复均值。每臂每场景的成功数均等于 offered。

| 场景 | Arm | req/s mean [min,max] | TTFT P99 | E2E P99 | GPU util |
| --- | --- | ---: | ---: | ---: | ---: |
| low | Adaptive | 0.186 [0.184,0.188] | 1817.34 ms | 8321.19 ms | 15.36% |
| low | Fixed-K2 | 0.270 [0.268,0.272] | 397.44 ms | 3574.43 ms | 19.40% |
| low | Fixed-K4 | 0.261 [0.260,0.262] | 451.55 ms | 3668.22 ms | 21.55% |
| low | Fixed-K6 | 0.277 [0.250,0.326] | 509.39 ms | 3864.61 ms | 26.83% |
| high | Adaptive | 0.506 [0.501,0.512] | 10144.62 ms | 11987.21 ms | 66.88% |
| high | Fixed-K2 | 0.658 [0.653,0.661] | 7618.35 ms | 9191.28 ms | 60.86% |
| high | Fixed-K4 | 0.759 [0.754,0.762] | 6311.74 ms | 7945.47 ms | 72.95% |
| high | Fixed-K6 | 0.905 [0.880,0.921] | 4922.41 ms | 6816.05 ms | 92.27% |
| mixed | Adaptive | 0.229 [0.224,0.231] | 29367.60 ms | 43599.37 ms | 46.86% |
| mixed | Fixed-K2 | 0.272 [0.270,0.274] | 33862.75 ms | 35473.17 ms | 40.97% |
| mixed | Fixed-K4 | 0.276 [0.274,0.279] | 28109.66 ms | 35585.97 ms | 48.13% |
| mixed | Fixed-K6 | 0.297 [0.269,0.350] | 28636.84 ms | 36218.72 ms | 58.58% |
| burst | Adaptive | 0.171 [0.170,0.171] | 30446.49 ms | 32214.35 ms | 21.41% |
| burst | Fixed-K2 | 0.214 [0.213,0.214] | 31767.54 ms | 33309.19 ms | 23.97% |
| burst | Fixed-K4 | 0.220 [0.218,0.222] | 29438.04 ms | 31195.12 ms | 27.67% |
| burst | Fixed-K6 | 0.238 [0.216,0.277] | 29272.07 ms | 31161.69 ms | 33.69% |

完整 trace：

| Arm | req/s mean [min,max] | sample stddev | output tok/s | pooled TTFT P99 | pooled E2E P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive | 0.2163 [0.2145,0.2179] | 0.0017 | 10.705 | 30370.54 ms | 43599.37 ms |
| Fixed-K2 | 0.2728 [0.2713,0.2741] | 0.0014 | 13.501 | 33751.80 ms | 35473.17 ms |
| Fixed-K4 | 0.2799 [0.2783,0.2820] | 0.0019 | 13.857 | 29218.06 ms | 35585.97 ms |
| Fixed-K6 | 0.3030 [0.2767,0.3518] | 0.0423 | 14.997 | 28888.29 ms | 36218.72 ms |

Fixed-K6 按冻结的正确性优先字典序和吞吐均值成为 global best fixed。它的三重复方差
明显较大，因此 `0.3030 req/s` 只是三点均值，不能当作稳定总体估计。

| 场景 | Best fixed | Adaptive/best-fixed req/s | Adaptive 吞吐下降 | Adaptive E2E P99 / best-fixed |
| --- | --- | ---: | ---: | ---: |
| low | Fixed-K6 | 0.670× | 32.98% | 2.153× |
| high | Fixed-K6 | 0.559× | 44.12% | 1.759× |
| mixed | Fixed-K6 | 0.771× | 22.90% | 1.204× |
| burst | Fixed-K6 | 0.717× | 28.31% | 1.034× |
| 整条 trace | Fixed-K6 | 0.714× | 28.62% | 1.204× |

## 6. 容量包络、切换与迁移

本节的切换延迟来自历史 drain-at-idle 三重复正式矩阵，不代表最新在线同步档实现。

观测最大同时运行请求：

| 场景 | Adaptive | Fixed-K2 | Fixed-K4 | Fixed-K6 |
| --- | ---: | ---: | ---: | ---: |
| high | 6 | 2 | 5 | 6 |
| mixed | 8 | 6 | 8 | 8 |
| burst | 5 | 2 | 5 | 5 |

为补全“最大并发/上下文”指标，capacity formal 用 8,192-token Prompt、Output-16
扫描 B2/B5/B6/B7/B8/B10，三个 seed 的结果完全一致：

| Offered | Adaptive running | Fixed-K2 | Fixed-K4 | Fixed-K6 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 2 | 2 | 2 |
| 5 | 5 | 2 | 5 | 5 |
| 6 | 6 | 2 | 5 | 6 |
| 7 | 7 | 2 | 5 | 7 |
| 8 | 7 | 2 | 5 | 7 |
| 10 | 7 | 2 | 5 | 7 |

| Arm | 最大 fully-resident B | 最大 live context | capacity req/s mean [min,max] |
| --- | ---: | ---: | ---: |
| Adaptive | 7 | 57,456 tokens | 0.606 [0.591,0.613] |
| Fixed-K2 | 2 | 16,416 tokens | 0.661 [0.625,0.717] |
| Fixed-K4 | 5 | 41,040 tokens | 0.863 [0.824,0.884] |
| Fixed-K6 | 7 | 57,456 tokens | 0.837 [0.834,0.839] |

Adaptive 把 fully-resident 长上下文并发相对 Fixed-K2 提高 3.5×、相对 Fixed-K4
提高 1.4×，但与 Fixed-K6 相同。B8/B10 仍全部完成，只是分别有 1/3 个请求不能
同时 resident；它们不是 OOM。capacity trace 上 Adaptive 吞吐为 best Fixed-K4 的
0.702×，仍无端到端性能优势。

Adaptive 每轮固定发生 6 次切换，H2D 请求迁移 6.444 GB、D2H 8.593 GB：

| Seed | grow/shrink | 迁移墙钟 | trace 耗时 | burst arrival→admission |
| --- | ---: | ---: | ---: | ---: |
| 20260901 | 2 / 4 | 17.240 s | 147.383 s | 26.483 s |
| 20260917 | 2 / 4 | 17.061 s | 149.712 s | 26.608 s |
| 20260929 | 2 / 4 | 17.205 s | 148.307 s | 26.466 s |
| 均值 | 2 / 4 | 17.169 s | 148.467 s | 26.519 s |

单次切换：

| 切换 | 场景 | 墙钟 mean [min,max] | 触发到 resize 完成 mean [min,max] |
| --- | --- | ---: | ---: |
| K4→K2 | low | 2.837 [2.725,2.920] s | 5.597 [5.535,5.691] s |
| K2→K6 | high | 5.125 [5.027,5.264] s | 5.125 [5.027,5.264] s |
| K6→K4 | mixed | 2.845 [2.757,2.932] s | 39.854 [39.405,40.675] s |
| K4→K2 | burst 前低压 | 2.891 [2.854,2.922] s | 5.671 [5.644,5.687] s |
| K2→K4 | burst | 0.593 [0.565,0.616] s | 0.593 [0.565,0.616] s |
| K4→K2 | burst 恢复 | 2.878 [2.823,2.964] s | 5.968 [5.863,6.047] s |

burst 的 resize 本身只需约 0.593 s，但 late cohort 从逻辑到达到实际准入平均等待
26.519 s：控制器必须先等待正在生成 256 tokens 的既有请求结束，才能安全切换专家
Plan Class。这个等待已计入请求 TTFT/E2E。

## 7. 时序产物

- [PNG](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_kv_pressure_tier_throughput_latency_timeseries.png)
- [SVG](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_kv_pressure_tier_throughput_latency_timeseries.svg)
- [时序 CSV](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_timeseries.csv)

时序图同时展示 active/predicted/target blocks、档位、rolling output tok/s 和请求延迟。
修复后的 burst 清楚显示 K2→K4、5 个 RAG 请求准入和 K4→K2 恢复。

## 8. 正确性与数值分叉

硬门包括：prompt/trace hash 相同、完整生成长度、共同首 token 一致、Fixed target 不变、
Adaptive 全档覆盖、late cohort 控制器覆盖、burst grow→shrink、三队列守恒、KV/Plan
Class 一致、无运行期 Graph recapture、无 kernel fault、GPU clean。

完整输出 hash 仍不是完全一致：

| Fixed arm（相对 Adaptive） | 完整输出 hash | 首 token | token 位置一致 |
| --- | ---: | ---: | ---: |
| Fixed-K2 | 79/96 | 96/96 | 4332/4752 |
| Fixed-K4 | 95/96 | 96/96 | 4741/4752 |
| Fixed-K6 | 96/96 | 96/96 | 4752/4752 |
| 合计 | 270/288 | 288/288 | 13825/14256 |

K2 的较小 KV 档改变了 phase-batched 调度几何，greedy decode 在后续 token 出现数值敏感
分叉。verifier 将完整 hash 作为 diagnostic；`status=ok` 不等于逐 token 完全等价。
在补充 teacher-forced logits/top-1 门之前，本结果只能作为机制 qualification。

capacity 矩阵另有 338/342 完整输出 hash 一致、342/342 首 token 一致；分臂为
Fixed-K2 111/114、Fixed-K4 113/114、Fixed-K6 114/114。

## 9. 分场景结论

本节保留历史 drain-at-idle formal 的分场景结论。最新在线单轮结果见 1.1；它消除了
等待运行请求完成的限制，但尚未把 Adaptive 提升到 best-fixed 的端到端性能水平。

### 低压力

Adaptive 从 K4 回缩 K2，使专家卸载从 18.119 GB 降到 15.854 GB，但新增一次 2.837 s
同步迁移。包含开销后吞吐只有 best fixed 的 0.670×，P99 E2E 为 2.153×，GPU util
也更低。低压力性能优势未得到验证。

### 高压力

Adaptive 在 RAG/B6 准入前 K2→K6，同时运行数从 Fixed-K2 的 2 提升到 6。但 5.125 s
切换计入时延后，吞吐只有 Fixed-K6 的 0.559×。所有固定档均完成请求，未观察到
Adaptive 独有的 OOM/拒绝避免。独立容量扫描进一步确认 Adaptive 可 fully-resident
B7，超过 Fixed-K2 的 B2 和 Fixed-K4 的 B5，但与 Fixed-K6 的 B7 相同。

### 动态混合

Adaptive 能 K6→K4，但因全局空闲门，触发到完成平均为 39.854 s。吞吐为 best fixed
的 0.771×，P99 E2E 高 20.4%；长期变化负载优势未得到验证。

### 压力突增与恢复

原先缺失的机制已补全：三轮均检测 late RAG/B5，取消待缩档，等待安全边界，
K2→K4 后一次准入 5 个请求，随后 K4→K2；无切换失败或额外迟到 K6 扩档。
但 26.519 s 准入等待使吞吐只有 best fixed 的 0.717×。这证明机制正确响应并稳定恢复，
同时暴露“必须排空正在运行请求”的主要性能限制。

## 10. 混杂因素与局限

1. K2/K4/K6 同时改变 KV target 与 cold70/80/89 专家卸载，是联合 HBM 策略比较，
   不能拆成纯 KV 控制器贡献。
2. 最新实现已把全局 request-idle 门改为相邻 token step 之间的 scheduler quiescent
   point，但切换事务本身仍为同步；异步复制、后台 planner 和迁移/推理重叠尚未实现。
3. trace 约 44–150 s，是合成 phase-batched 离线调度，不是 open-loop HTTP soak；
   3 seeds 低于计划中的 5-seed 论文门。
4. Fixed arm 顺序固定，未做 Latin-square；仍可能有运行顺序偏差。
5. 本轮是 eager，Graph capture count 始终为 0；“无 recapture”不是 Graph 性能证据。
6. capacity profile 测得 B≤10、context≤8,208 下的 fully-resident 上限；调度器允许
   超额请求排队且没有 deadline/admission SLO，因此没有测出物理 OOM/拒绝边界。
7. NVML 与 requested migration bytes 不是独立 PCIe/逐页硬件 fault 计数。
8. Fixed-K6 的吞吐样本标准差为 0.0423 req/s，明显高于其他 arm；只有 3 个点。
9. 完整输出仍有 18/288 request-hash 分叉，需要 teacher-forced 数值门。

## 11. 不计切换开销的稳态 Prefill/Decode 对照

按用户要求只运行一轮、不做重复。六个独立冷进程覆盖 K2/K4/K6：

- `direct` 在接收正式请求前直接建立目标 KV target 和目标 Plan Class；
- `switched` 先在源档完整服务一个 32-token conditioning 请求，然后执行
  K4→K2、K6→K4 或 K4→K6 的运行期切换；
- 源档 conditioning、切换事务和共同的目标档 warmup 全部在计时窗外；
- 正式点均为 Medium-2048、B1/B8、32-token 冻结 continuation，不记录 logits。
  Prefill 记录完整 Prefill 墙钟；Decode 先 warmup 5 step，再取 16 个 whole-batch
  单 token 稳态 step 的中位数。

单位为 ms；S/D 为 `switched/direct` 延迟，越接近 1 越相似。

| 目标档 | B | Direct Prefill | Switched Prefill | S/D | Direct Decode | Switched Decode | S/D | 冻结输出 hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K2 | 1 | 411.19 | 410.27 | 0.998× | 92.48 | 89.69 | 0.970× | 一致 |
| K2 | 8 | 1117.57 | 1117.27 | 1.000× | 95.93 | 91.50 | 0.954× | 一致 |
| K4 | 1 | 460.64 | 463.58 | 1.006× | 98.46 | 96.92 | 0.984× | 一致 |
| K4 | 8 | 1208.42 | 1213.75 | 1.004× | 74.57 | 98.73 | 1.324× | 一致 |
| K6 | 1 | 501.91 | 493.87 | 0.984× | 92.14 | 72.14 | 0.783× | 一致 |
| K6 | 8 | 1290.71 | 1283.32 | 0.994× | 94.05 | 73.83 | 0.785× | 一致 |

六个 Prefill cell 全部通过预注册的 ±10% 数值带，范围为
`0.984×–1.006×`。Decode 只有 K2/B1、K2/B8 和 K4/B1 通过；K4/B8 的
switched 慢 32.4%，K6/B1 和 K6/B8 的 switched 反而快 21.7% 和 21.5%。

6/6 进程返回 0，12/12 Prefill/Decode 性能点功能完成，6/6 logical cell 的
冻结输出 hash 一致；KV target、Plan Class 在测量前后稳定，Graph count 为 0，
无 Xid/MMU fault，最终 GPU 0 MiB 并恢复 stock UVM。verifier 为 `status=ok`，
但 `all_decode_cells_equivalent_within_10pct=false`。

这张表的 6/6 功能结论仍成立，但 Decode 性能因果解释已被下述定因实验取代。
它不能证明切档造成 Decode 回退，也不能证明统计等价。切换开销被故意排除，表中
数据同样不能用于推断在线切换的 TTFT/E2E 代价。冻结 continuation 只控制后续 token，
没有保存 logits 或 hidden state；输出 hash 一致不是自然生成的数值等价证明。

### 11.1 Decode 差异定因

同进程 ABBA 固定模型实例、物理页、请求和 token 序列，在 direct/switched 之间交错
测量。下表的 center 是两个 direct 或两个 switched 位置的中值中心，S/D 越接近 1
越相似。

| 档位 | B | Direct center | Switched center | S/D |
| --- | ---: | ---: | ---: | ---: |
| K4 | 1 | 91.61 ms | 90.05 ms | 0.983× |
| K4 | 8 | 92.70 ms | 91.40 ms | 0.986× |
| K6 | 1 | 89.04 ms | 90.23 ms | 1.013× |
| K6 | 8 | 92.20 ms | 92.22 ms | 1.000× |

原单轮表中的 K4/B8 `1.324×` 回退和 K6 约 `0.78×` 的反向加速完全消失。
四点均在 ±1.8% 内。K4 每层路由 hash、remote assignment 和 Plan 专家集合一致；
K6 路由/remote assignment 有轻微差异，但 direct/switched 的 MoE kernel 时间差小于
1%。KV block ID/顺序 hash 在条件间并不都相同，而性能仍相同，因此 KV block 顺序
不是充分原因。

页表与 backing 诊断得到：

- GPU alias 全部为 4K leaf，经 2M PDE 成批切换；64K leaf 为 0；
- steady Decode 的 PTE/TLB 事务计数没有增量，排除测量窗内页表重建或 TLB
  invalidate；
- 六个冷进程的 CPU backing 构成完全相同：4K/64K/2M chunk 分别为
  `1,152/1,968/18,258`，按字节计 2M 占 `99.652%`；
- 映射和显存占用不变时，K4/B1 trial-1 在同一进程内由前半 `142.34 ms`
  降到后半 `90.27 ms`；K6/B1 trial-3 由 `134.14 ms` 降到 `94.95 ms`；
- K4/B1 三个冷进程中位数为 `97.04/142.56/146.37 ms`，最大/最小达
  `1.508×`。过程中无迁移，SM/显存时钟、P-state 和 PCIe Gen5 ×16 均稳定；
  慢态 PCIe RX 较低，进入快态后提高。

Nsight Systems 使用 CUDA profiler API 精确包住 condition window，共解析 90 个
Decode step。它把每步 wall-time 扰动到约 0.5 s，因此该 wall-time 不进入性能表；
kernel 级数据仍显示 common B8 的 MoE 中位时间从 warmup 的 `27.20 ms` 降到正式窗
`13.85 ms`、drain 的 `10.79 ms`，而 attention 保持
`1.225/1.221/1.222 ms`。这把历史相关变化定位到 MoE/CPU-remote-read 路径，而非
attention；但现有 GB202 可用计数器不足以在 GMMU 页表缓存、PCIe read-request 状态
等硬件子机制之间唯一归因。

因此最终定因为：原协议仅做固定 5-step Decode warmup，且 direct/switched 位于不同
冷进程，测量窗会随机落入 remote-read 慢态或快态，产生虚假的 `+32.4%` 与约
`-21%` 差异。正确结论是“没有观察到切档后的稳态 Decode 回退”；这仍是单次同进程
qualification，不是统计等价证明。后续正式协议必须改用收敛门（滚动窗口稳定）或
同进程随机化 ABBA，不能继续以固定 5-step warmup 作因果比较。

失败 attempt 保留了两个本轮发现并修复的真实机制错误：attempt-01 中 KV 回缩后
延迟的 Plan Class 没有在 idle scheduler 边界提交；attempt-02–04 在减少卸载量时
先重建了双页表、后恢复热专家 GPU residency，恢复后的普通页因而仍无 PTE，
导致 Decode PTE fault/Xid 31。修复后，Plan Class 激活传递
`RESUME_PREFILL_BEFORE_PLAN_CLASS`，并在热页恢复后执行一次非嵌套的完整双页表
刷新。attempt-05 已能完整运行，且 6/6 Prefill 和 6/6 Decode 落在 ±10%，
但自由 greedy 的 K4/B1、K4/B8 输出 hash 分叉，所以只作辅助诊断；改用冻结
continuation 的 attempt-06 才是正式单轮结果。

### 11.2 Fixed-K2/Fixed-K6 与 Adaptive K2→K6→K2 回环

该补充实验只启动一轮，共三个独立冷进程：`fixed_k2`、`fixed_k6` 和
`roundtrip`。每个进程内先执行不计时的目标档 warmup，再采集多次稳态样本；这些
进程内样本不等价于独立重复。三个进程合计 245/245 请求完成，verifier 为
`status=ok`。

低压为 Short-128/B1→32，高压为 RAG-8192/B6→16。下表为进程内样本中位数；
吞吐越高越好，其余延迟越低越好。

| 压力 | 档位 | req/s | TTFT | TPOT | E2E | max running |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 低压 | Fixed-K2 | 0.4279 | 351.79 ms | 64.03 ms | 2336.70 ms | 1 |
| 低压 | Fixed-K6 | 0.4018 | 444.30 ms | 65.92 ms | 2488.57 ms | 1 |
| 高压 | Fixed-K2 | 0.8068 | 3585.29 ms | 91.53 ms | 4962.39 ms | 2 |
| 高压 | Fixed-K6 | 1.0140 | 2854.75 ms | 204.10 ms | 5916.21 ms | 6 |

低压下 K2 相对 K6 的吞吐高 6.50%，TTFT、TPOT、E2E 分别低 20.82%、2.87%、
6.10%，因此该点支持“小 KV 压力时保留更多专家在 GPU 上更有利”。高压下 K6
吞吐高 25.68%，TTFT 低 20.37%，且 B6 全部同时运行；K2 最多同时运行 2 个请求，
必须拆成三个 cohort 排队。K2 并没有 OOM，也不是完全不能完成 B6；它以串行化换取
较低的单请求 Decode 竞争，所以 TPOT 和 E2E 反而优于 K6。由此可声称的是 K6 的
高压吞吐与并发容量优势，不能声称其每项请求延迟都更好。

为确认低压结论在 batch 增大后是否仍成立，另用两个冷进程补测
Short-128/B6→32。每臂先做一次排除 warmup，再取三个进程内样本；下表给出中位数
和 `[min,max]`。

| 档位 | req/s | TTFT | TPOT | E2E | Prefill | steady Decode step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed-K2 | 2.5194 [2.5094,2.5200] | 365.56 [365.37,365.66] ms | 65.01 [65.00,65.31] ms | 2380.83 [2380.32,2390.37] ms | 365.27 [365.06,365.29] ms | 63.31 [63.08,63.38] ms |
| Fixed-K6 | 2.4015 [2.3837,2.4068] | 451.50 [449.25,453.81] ms | 65.94 [65.83,66.69] ms | 2497.87 [2492.28,2516.49] ms | 451.16 [448.93,453.51] ms | 63.12 [62.94,64.07] ms |

Short/B6 需要的请求 KV 仅约 `6×ceil((128+32)/16)=60` blocks，远低于 K2 的
1,364-block target，因此两档 `max-running=6`，没有容量差异。K2 相对 K6 的吞吐
高 4.91%，TTFT/TPOT/E2E 分别低 19.04%/1.41%/4.69%；Prefill 低 19.04%，而
steady Decode step 只慢 0.29%。因此 Short/B6 的优势主要来自 K2 保留更多 GPU
专家后的 Prefill，不是 KV 容量或 Decode 差异。

随后按相同协议单独测量 Short-128/B64→32。当前正式配置的 `max_num_seqs=10`，
因此该专用 runner 明确提升为 64；`64×128=8,192` 个 Prefill tokens，仍保持冻结的
`max_num_batched_tokens=8192`。原 Short fixture 有 32 条请求，协议将相同冻结顺序
复制一轮组成 B64，prefix cache 关闭。每臂仍为一次排除 warmup和三个进程内样本。

| 档位 | req/s | TTFT | TPOT | E2E | Prefill | steady Decode step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed-K2 | 20.5385 [20.4558,20.6166] | 520.57 [520.55,520.93] ms | 83.54 [83.16,83.95] ms | 3110.57 [3098.46,3123.08] ms | 517.10 [516.96,517.40] ms | 80.17 [79.99,80.42] ms |
| Fixed-K6 | 23.0724 [23.0343,23.2033] | 602.81 [602.74,603.68] ms | 69.87 [69.36,70.01] ms | 2768.88 [2753.03,2773.93] ms | 600.08 [599.88,600.98] ms | 62.39 [61.89,62.61] ms |

两档都在一个完整 Prefill step 后让 64 个请求同时运行，`max-waiting=0`，没有拆成
B10/B32。B64 约需 640 个请求 KV blocks，仍小于 K2 的 1,364 blocks；但这是低 KV
占用与高执行并发的组合。K2 的 Prefill 和 TTFT 分别低 13.83% 和 13.64%，K6 的
steady Decode step 和 TPOT 分别低 22.19% 和 16.36%，Decode 优势最终使 K6 吞吐
高 12.34%、E2E 低 10.98%。因此 K2 的低压优势只在本轮 B1/B6 成立，不能外推到
Short/B64；档位策略除 KV 压力外还需要纳入执行 batch/Decode 并行度。

B64 两臂各三个正式样本的完整 schedule 都是 1 个 Prefill step 加 31 个 Decode
step，输出 hash 完全一致。由于仍是两个独立冷进程的单轮 qualification，该表观
crossover 不是统计显著性或纯专家卸载量因果证明。后续下述同进程档内 batch sweep
已经证明它不能作为稳态机制结论。

最新补充矩阵严格按用户指定顺序执行：一个 Fixed-K2 冷进程内依次完成
B1→B16→B64，再启动一个 Fixed-K6 冷进程执行相同顺序。两臂统一
`max_num_seqs=64`、Output-32、prefix cache关闭；每个 batch 独立做一次排除 warmup，
再取三个进程内样本。下表为中位数。

| B | 档位 | req/s | TTFT | TPOT | E2E | Prefill | Decode step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Fixed-K2 | 0.4209 | 355.99 ms | 65.15 ms | 2375.49 ms | 355.81 ms | 62.85 ms |
| 1 | Fixed-K6 | 0.3339 | 445.03 ms | 82.24 ms | 2994.33 ms | 444.83 ms | 79.13 ms |
| 16 | Fixed-K2 | 6.5401 | 387.19 ms | 66.39 ms | 2445.20 ms | 386.51 ms | 63.97 ms |
| 16 | Fixed-K6 | 6.1309 | 473.56 ms | 68.84 ms | 2608.48 ms | 472.86 ms | 64.14 ms |
| 64 | Fixed-K2 | 24.8022 | 514.87 ms | 66.48 ms | 2575.70 ms | 512.20 ms | 62.63 ms |
| 64 | Fixed-K6 | 23.0573 | 599.11 ms | 70.07 ms | 2771.39 ms | 596.28 ms | 63.25 ms |

B16 中 K2 的吞吐高 6.67%，TTFT/TPOT/E2E 分别低 18.24%/3.56%/6.26%；B64 中
K2 的吞吐高 7.57%，TTFT/TPOT/E2E 分别低 14.06%/5.13%/7.06%。两个稳定 batch
的 Decode step 基本相同，K2 分别低 0.26% 和 0.98%；整体优势主要来自少卸载专家后
更快的 Prefill。

B1 的 K6 三个样本不稳定：Decode step 为 `79.94/79.13/63.04 ms`，req/s 为
`0.3317/0.3339/0.4019`。因此表中 B1 的 K2 `+26.05%` 中位数优势不能当作稳定倍率；
它反而直接证明 remote-read 路径会在同一进程中从慢态进入快态。进入 B16 后，两臂
Decode 都稳定在约 64 ms，B64 也保持 62.6–63.2 ms。孤立 B64 中 K2 的约 80 ms
正是尚未消除的慢态，而非 K6“卸载更多所以更快”。

六个逻辑 cell 的正式执行均为 1 个完整 Prefill step 加 31 个 Decode step，
`max-running=B`、`max-waiting=0`，跨档输出 hash 一致。2/2 进程、648/648 请求完成，
无 Xid/MMU fault，最终 GPU 0 MiB 并恢复 stock UVM。该矩阵仍是单轮 qualification；
B1 应改用滚动收敛门或同进程 ABBA 后才能报告稳定倍率。

为避免 B256/B512 的 KV 容量分cohort混杂，最终只补测两档都能完整驻留的
Short-128/B128→32。两臂各一个冷进程，每臂先做一次排除warmup，再取三个进程内
样本。冻结 `max_num_batched_tokens=8192`，所以每个batch包含2个Prefill step和
31个Decode step；第一步暂有64条waiting，但随后两档都达到`max-running=128`。
下表为中位数和`[min,max]`。

| 档位 | req/s | TTFT | TPOT | E2E | Prefill | steady Decode step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed-K2 | 34.8089 [34.5241,40.0317] | 770.60 [762.82,770.68] ms | 93.45 [78.27,94.41] ms | 3667.27 [3188.51,3697.24] ms | 1008.32 [1000.82,1008.60] ms | 82.39 [65.06,82.78] ms |
| Fixed-K6 | 35.1977 [31.1865,35.3267] | 901.41 [893.97,902.66] ms | 87.92 [87.79,102.93] ms | 3626.66 [3615.15,4092.52] ms | 1180.33 [1174.41,1181.57] ms | 67.39 [67.12,85.51] ms |

中位数上K6吞吐高1.12%、E2E低1.11%；K2的Prefill和TTFT分别低14.57%和
14.51%，K6的TPOT和Decode step分别低5.92%和18.20%。但K2 Decode三个样本包含
约65 ms快态和两个约82 ms慢态，K6则包含两个约67 ms快态和一个约86 ms慢态；
吞吐/E2E范围也明显重叠。因此本轮最严谨结论是两档B128总体性能近似，K2有稳定
Prefill优势，而当前协议仍不足以给Decode及约1%的整体差距稳定排序。

2/2冷进程、1,024/1,024请求完成，跨档输出hash一致，无preemption、Xid或MMU
fault，最终GPU 0 MiB并恢复stock UVM。这是单轮qualification，三个样本是进程内
样本，不是独立冷进程重复。

为定因该表的Decode反转，新增一个冷进程内`K2(A1)→K6(B1)→K6(B2)→K2(A2)`
ABBA。每段同样执行一个排除warmup和三个B128测量batch；四段都保持
`max-running=128`，切换事务不计时。结果如下：

| 段 | 档位 | 排除warmup Decode | 三个正式Decode中位数 | req/s中位数 | Prefill中位数 |
| --- | --- | ---: | --- | ---: | ---: |
| A1 | K2 | 66.50 ms | 65.22/65.59/65.73 ms | 39.8830 | 1001.33 ms |
| B1 | K6 | 67.85 ms | 67.10/68.01/68.24 ms | 35.1966 | 1175.33 ms |
| B2 | K6 | 68.27 ms | 67.76/68.40/68.12 ms | 35.1702 | 1175.79 ms |
| A2 | K2 | 85.21 ms | 84.92/85.43/84.44 ms | 33.5540 | 1010.91 ms |

同一K2逻辑档位的A2比A1慢29.48%，但Prefill只相差0.96%；初始K2快态反而比
两个K6段的Decode低约3.6%。Plan Class均为cold70、KV target均为1,364 blocks、
remote范围数均为6,720、双页表覆盖与输出hash一致。因此可以排除“K2容量小”或
“K2保留更多专家反而使Decode变慢”的解释；性能状态由执行历史决定。

原始两个冷进程表也正好符合这一采样错误：K2三个样本为
`82.39/82.78/65.06 ms`，中位数落在慢态；K6为`85.51/67.12/67.39 ms`，中位数
落在快态。一个排除warmup没有提供收敛保证。另一次相邻档隔离的故障前数据中，初始
K2在档位不变时从`84.37/83.66`自行降到`66.63 ms`，K4→K2后也稳定在约
`67.11 ms`，进一步排除K2固有慢态。

NVML显示ABBA中A2慢态的平均PCIe RX比A1快态低约18.29%，而SM clock、memory
clock和PCIe Gen5×16链路没有降级；结合既有逐kernel诊断，差异位于MoE
remote-read路径而非attention。现有硬件计数器仍不能在GMMU cache与PCIe request
shaping之间唯一拆分。因此最终定因分两层：直接原因已确定为固定warmup不足、错误
采到历史相关remote-read慢态；更底层只能定位到GMMU/PCIe请求状态，不能声称已唯一
归因某一个硬件缓存。

完整ABBA为1/1进程、2,048/2,048请求成功，verifier为`status=ok`，无Xid/MMU
fault且最终恢复stock UVM。辅助相邻档隔离在完成四段后，于K6→K4激活时发生
Xid 31 `FAULT_PTE ACCESS_TYPE_VIRT_READ`，其目录为`status=error`；只使用故障前
数据作辅助定因，不把该轮计为完整通过。该错误是另一个需要修复的Plan Class降档
可靠性问题，不改变上述B128原始排名失真的结论。

回环臂严格执行 `K2→K4→K6→K4→K2`，没有 4 GB 直跳。切换事务和每次切换后的
共同 warmup 均在计时窗外。正式点为 Medium-2048、B1/B8、32-token 冻结
continuation；每项取三个进程内样本的中位数。A/F 为 Adaptive 稳态延迟除以对应
Fixed 延迟。

| Adaptive 状态 | B | Fixed/Adaptive Prefill | A/F | Fixed/Adaptive Decode | A/F | Fixed/Adaptive batch wall | A/F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 初始 K2 | 1 | 400.18/399.41 ms | 0.998× | 62.05/62.06 ms | 1.000× | 2381.86/2382.55 ms | 1.000× |
| 初始 K2 | 8 | 1102.03/1103.81 ms | 1.002× | 63.42/63.09 ms | 0.995× | 3137.55/3133.27 ms | 0.999× |
| 增长后 K6 | 1 | 489.65/489.22 ms | 0.999× | 62.10/63.21 ms | 1.018× | 2501.88/2538.79 ms | 1.015× |
| 增长后 K6 | 8 | 1277.73/1276.62 ms | 0.999× | 63.44/64.11 ms | 1.011× | 3358.17/3382.20 ms | 1.007× |
| 回切后 K2 | 1 | 400.18/398.13 ms | 0.995× | 62.05/62.44 ms | 1.006× | 2381.86/2410.83 ms | 1.012× |
| 回切后 K2 | 8 | 1102.03/1106.25 ms | 1.004× | 63.42/63.74 ms | 1.005× | 3137.55/3163.98 ms | 1.008× |

18/18 个 Prefill、Decode、batch-wall cell 通过预注册的 ±10% 等价带：Prefill 为
`0.995×–1.004×`，Decode 为 `0.995×–1.018×`，batch wall 为
`0.999×–1.015×`。其中用户要求的 Prefill/Decode 是 12/12 个 cell；batch wall
是补充的整批端到端执行时间，不等同于 B8 的单请求 E2E。
三种状态的受控输出 hash 均与对应 Fixed 档一致。因此，本轮支持“忽略同步切换和
warmup 后，Adaptive 到达某档的稳态性能恢复到直接固定在该档的水平；K6→K2
回切没有留下可见性能回退”。

被排除的回环切换墙钟为：K2→K4 `817.45 ms`、K4→K6 `846.71 ms`、K6→K4
`1549.23 ms`、K4→K2 `1528.31 ms`。所以这张表不改变“同步切换开销仍会伤害在线
端到端性能”的主结论。它是单轮 qualification，不是置信区间或统计等价证明。
3/3 进程返回 0，所有实际边都是相邻约 2 GB 档，档位/Plan Class 在测量窗稳定，
无 Xid/MMU fault，最终 GPU 0 MiB 并恢复 stock UVM。

### 11.3 B64/B128 双阶段 CUDA Graph 与快速 P→D 切换

为避免前述 eager 数据被误用为完整机制性能，新增 Fixed-K2/Fixed-K6 两个独立冷进程，
每个进程按 B64→B128 运行。两档分别只注册和捕获自身 Plan Class，统一使用
`TRITON_ATTN`、`FULL_SEPARATE`、`prefill_graph_remote_decode`及 original-VA
park/resume。捕获集合为 Decode B64/B128 和 Prefill 8,192-token shape；每个 cell
先运行一次排除 warmup，再取三个进程内样本。下表均为中位数，单位为 ms（吞吐除外）。

| B | 档位 | req/s | TTFT | TPOT | E2E | Prefill | P→D step | steady Decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Fixed-K2 | 12.7435 | 3117.76 | 61.26 | 5016.67 | 3114.18 | 483.75 | 36.60 |
| 64 | Fixed-K6 | 11.4690 | 3302.29 | 73.10 | 5574.95 | 3298.59 | 482.19 | 46.06 |
| 128 | Fixed-K2 | 14.8471 | 4533.80 | 131.47 | 8612.79 | 5982.57 | 432.97 | 63.45 |
| 128 | Fixed-K6 | 13.3505 | 4848.30 | 152.63 | 9579.71 | 6412.61 | 447.96 | 79.81 |

完整机制下K2在两个batch均获胜：B64吞吐高11.11%、E2E低10.01%、Prefill低
5.59%、steady Decode低20.54%；B128吞吐高11.21%、E2E低10.09%、Prefill低
6.71%、steady Decode低20.51%。P→D列是包含本轮park和remote-map提交的首个
Decode step，不是纯Decode kernel；后16个Decode step的中位数用于steady Decode。

verifier逐step检查后，两个B64 cell的每个正式batch均为32/32次FULL replay，两个
B128 cell均为33/33次FULL replay，合计390/390个正式执行step使用FULL Graph，且
捕获后无Graph重捕获。每个请求周期都记录alias Prefill恢复和remote Decode cutover，
结束状态为`remote_parked`；两档输出hash一致。2/2冷进程、1,536/1,536请求完成，
无Xid、PTE/PDE或MMU fault，最终GPU为0 MiB并恢复stock UVM。

因此，在本次单轮完整机制qualification中，Short-128的B64和B128都没有达到K2的
KV容量瓶颈；保留更多GPU专家的K2在Prefill、稳态Decode和整体吞吐上都优于K6。
三个进程内样本不是三个独立冷进程重复，不能据此给出置信区间。

### 11.4 双阶段 CUDA Graph 下的 Adaptive K2→K6→K2 回环

本节是根据`experiments/reproduce`当前文档在2026-09-02重新复现后的权威完整机制数据；
11.3及2026-09-01的`attempt-03-source-validated`保留为历史记录，不再用于当前倍率。
正式artifact为
`artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02`。

在11.3的完整机制基础上重新执行一轮三臂实验：`fixed_k2`、`fixed_k6` 和
`roundtrip`各使用一个独立冷进程。固定臂只捕获自身档位的3个Graph shape；Adaptive
预捕获cold70/cold80/cold89三档共9个Graph entry，并严格执行
`K2→K4→K6→K4→K2`。初始化、四个约2 GB同步切档事务和每个目标档的一次warmup
均排除；每个B64/B128 cell保留三个进程内稳态样本。下表为中位数。

| 状态 | B | req/s | TTFT | TPOT | E2E | Prefill | P→D step | steady Decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed-K2 | 64 | 29.1489 | 726.68 ms | 46.59 ms | 2189.23 ms | 722.16 ms | 489.74 ms | 24.89 ms |
| Fixed-K2 | 128 | 39.6909 | 972.30 ms | 72.17 ms | 3212.92 ms | 1208.64 ms | 492.87 ms | 41.12 ms |
| Fixed-K6 | 64 | 25.3277 | 768.05 ms | 56.65 ms | 2519.78 ms | 763.26 ms | 470.04 ms | 32.67 ms |
| Fixed-K6 | 128 | 32.5717 | 1066.51 ms | 91.25 ms | 3917.92 ms | 1344.89 ms | 501.92 ms | 59.54 ms |
| Adaptive 初始 K2 | 64 | 28.8046 | 731.93 ms | 47.71 ms | 2215.43 ms | 726.11 ms | 525.33 ms | 24.43 ms |
| Adaptive 初始 K2 | 128 | 38.8669 | 976.26 ms | 74.35 ms | 3280.80 ms | 1213.96 ms | 550.39 ms | 41.17 ms |
| Adaptive 增长后 K6 | 64 | 24.4374 | 775.70 ms | 59.34 ms | 2612.40 ms | 770.56 ms | 537.02 ms | 33.29 ms |
| Adaptive 增长后 K6 | 128 | 31.6304 | 1071.97 ms | 95.42 ms | 4033.71 ms | 1350.85 ms | 622.52 ms | 60.75 ms |
| Adaptive 回切后 K2 | 64 | 28.7880 | 727.52 ms | 47.97 ms | 2216.38 ms | 722.85 ms | 543.23 ms | 24.96 ms |
| Adaptive 回切后 K2 | 128 | 38.4072 | 980.15 ms | 74.90 ms | 3320.69 ms | 1216.84 ms | 560.66 ms | 41.16 ms |

固定档比较中，K2/K6吞吐比为B64 `1.1509×`、B128 `1.2186×`；K2的E2E为K6的
`0.8688×/0.8201×`，Prefill为`0.9462×/0.8987×`，steady Decode为
`0.7618×/0.6906×`。两个batch都只需要远少于K2 target的KV blocks，因此没有
KV容量收益；K6多卸载专家只增加了计算/remote-read开销。

Adaptive稳态相对对应Fixed的核心结果如下：

| Adaptive状态 | B | req/s A/F | Prefill A/F | Decode A/F | E2E A/F |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始K2 | 64 | 0.988× | 1.005× | 0.982× | 1.012× |
| 增长后K6 | 64 | 0.965× | 1.010× | 1.019× | 1.037× |
| 回切后K2 | 64 | 0.988× | 1.001× | 1.003× | 1.012× |
| 初始K2 | 128 | 0.979× | 1.004× | 1.001× | 1.021× |
| 增长后K6 | 128 | 0.971× | 1.004× | 1.020× | 1.030× |
| 回切后K2 | 128 | 0.968× | 1.007× | 1.001× | 1.034× |

因此，用户指定的吞吐、Prefill、Decode和E2E共24/24个A/F比较都在预注册的±10%
等价带内；回切K2没有留下稳态性能回退。若把TTFT、TPOT、batch wall和P→D也纳入，
则为49/54在带内。超带的5点全部是P→D step：初始K2/B128为`1.117×`，增长后
K6的B64/B128为`1.142×/1.240×`，回切后K2的B64/B128为
`1.109×/1.138×`。P→D包含每个batch的park/remote-map提交，不是已排除的
K2↔K6档位事务；初始K2/B128在切档前也超带，说明额外停顿不是只有切档历史才能触发。
当前单轮数据没有唯一解释这10.92%–24.03%的额外停顿，所以不能声称P→D切换
本身与Fixed完全等价。

四个排除的档位事务墙钟依次为`7.272/6.992/7.671/6.800 s`。正式测量共
975/975个serving step为FULL Graph replay，capture count固定为Fixed每臂3、Adaptive
9，运行期无重捕获或eager fallback。3/3冷进程返回0，计入与排除warmup合计
3,840/3,840请求完成，输出hash一致；84/84协议检查通过，无Xid、PTE/PDE或MMU
fault，最终GPU 0 MiB并恢复stock UVM。正式轮加载的当前源码模块srcversion为
`D158DB985980F4A580DB740`，SHA-256为
`4168020f14fc3217f8d217e4fa8898299e4d81c0c18653c98cce9d73346afdbe`。
这仍是单轮qualification，三个进程内样本不能替代独立冷进程重复或统计等价检验。

## 12. 证据

最新 scheduler-boundary、同步约 2 GB 档 qualification：

- [online verification](artifacts/e13-kv-online-synchronized-2gb-20260901-attempt-08-qualification/verification.json)
- [online 汇总](artifacts/e13-kv-online-synchronized-2gb-20260901-attempt-08-qualification/analysis/e13_summary.md)
- [online JSON](artifacts/e13-kv-online-synchronized-2gb-20260901-attempt-08-qualification/analysis/e13_summary.json)
- [online 执行状态](artifacts/e13-kv-online-synchronized-2gb-20260901-attempt-08-qualification/execution_status.tsv)
- [online 时序图](artifacts/e13-kv-online-synchronized-2gb-20260901-attempt-08-qualification/analysis/e13_kv_pressure_tier_throughput_latency_timeseries.png)

双阶段CUDA Graph B64/B128固定档单轮矩阵：

- [verification](artifacts/e13-fixed-k2-k6-short-b64-b128-full-separate-20260901-attempt-01/verification.json)
- [汇总](artifacts/e13-fixed-k2-k6-short-b64-b128-full-separate-20260901-attempt-01/summary.md)
- [执行状态](artifacts/e13-fixed-k2-k6-short-b64-b128-full-separate-20260901-attempt-01/execution_status.tsv)

双阶段CUDA Graph B64/B128 Fixed/Adaptive回环单轮矩阵：

- [verification](artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/verification.json)
- [汇总](artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/summary.md)
- [执行状态](artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/execution_status.tsv)

历史 drain-at-idle 三重复正式矩阵：

- [verification.json](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/verification.json)
- [聚合 Markdown](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_summary.md)
- [聚合 JSON](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_summary.json)
- [每进程 CSV](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_cases.csv)
- [全局 CSV](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_global.csv)
- [执行状态](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/execution_status.tsv)
- [冻结配置](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/config.snapshot.json)
- [输入 SHA-256](artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/input_sha256.txt)

容量包络：

- [capacity verification](artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/verification.json)
- [capacity 汇总](artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/analysis/e13_summary.md)
- [capacity JSON](artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/analysis/e13_summary.json)
- [capacity CSV](artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/analysis/e13_capacity_envelope.csv)
- [capacity 时序图](artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/analysis/e13_kv_pressure_tier_throughput_latency_timeseries.png)

不计切换开销的单轮稳态阶段对照：

- [phase verification](artifacts/e13-tier-phase-once-20260901-attempt-06/verification.json)
- [phase summary](artifacts/e13-tier-phase-once-20260901-attempt-06/summary.md)
- [phase 执行状态](artifacts/e13-tier-phase-once-20260901-attempt-06/execution_status.tsv)

K2/K6 固定档与 Adaptive 回环：

- [roundtrip verification](artifacts/e13-k2-k6-roundtrip-20260901-attempt-01/verification.json)
- [roundtrip summary](artifacts/e13-k2-k6-roundtrip-20260901-attempt-01/summary.md)
- [roundtrip 执行状态](artifacts/e13-k2-k6-roundtrip-20260901-attempt-01/execution_status.tsv)

Short/B6 固定档补充：

- [Short/B6 verification](artifacts/e13-fixed-k2-k6-short-b6-20260901-attempt-01/verification.json)
- [Short/B6 summary](artifacts/e13-fixed-k2-k6-short-b6-20260901-attempt-01/summary.md)
- [Short/B6 执行状态](artifacts/e13-fixed-k2-k6-short-b6-20260901-attempt-01/execution_status.tsv)

Short/B64 固定档补充：

- [Short/B64 verification](artifacts/e13-fixed-k2-k6-short-b64-20260901-attempt-01/verification.json)
- [Short/B64 summary](artifacts/e13-fixed-k2-k6-short-b64-20260901-attempt-01/summary.md)
- [Short/B64 执行状态](artifacts/e13-fixed-k2-k6-short-b64-20260901-attempt-01/execution_status.tsv)

Short B1/B16/B64 同进程档内矩阵：

- [batch-matrix verification](artifacts/e13-fixed-k2-k6-short-b1-b16-b64-20260901-attempt-01/verification.json)
- [batch-matrix summary](artifacts/e13-fixed-k2-k6-short-b1-b16-b64-20260901-attempt-01/summary.md)
- [batch-matrix 执行状态](artifacts/e13-fixed-k2-k6-short-b1-b16-b64-20260901-attempt-01/execution_status.tsv)

Short/B128 固定档补充：

- [Short/B128 verification](artifacts/e13-fixed-k2-k6-short-b128-20260901-attempt-01/verification.json)
- [Short/B128 summary](artifacts/e13-fixed-k2-k6-short-b128-20260901-attempt-01/summary.md)
- [Short/B128 执行状态](artifacts/e13-fixed-k2-k6-short-b128-20260901-attempt-01/execution_status.tsv)

Short/B128 Decode ABBA定因：

- [B128 ABBA verification](artifacts/e13-short-b128-decode-abba-20260901-attempt-01/verification.json)
- [B128 ABBA summary](artifacts/e13-short-b128-decode-abba-20260901-attempt-01/summary.md)
- [B128 ABBA执行状态](artifacts/e13-short-b128-decode-abba-20260901-attempt-01/execution_status.tsv)
- [相邻档隔离失败记录](artifacts/e13-short-b128-decode-shrink-isolation-20260901-attempt-01/roundtrip/result.json)

Decode 定因：

- [ABBA analysis](artifacts/e13-decode-abba-diagnostic-20260901-attempt-01/analysis.json)
- [backing analysis](artifacts/e13-decode-backing-diagnostic-20260901-attempt-01/analysis.json)
- [Nsight analysis](artifacts/e13-decode-nsys-diagnostic-20260901-attempt-03/analysis.json)
- [Nsight report](artifacts/e13-decode-nsys-diagnostic-20260901-attempt-03/k4_nsys/timeline.nsys-rep)

复现文档位于 `experiments/reproduce/REPRODUCE_E13_KV_ADAPTIVE.md`。
