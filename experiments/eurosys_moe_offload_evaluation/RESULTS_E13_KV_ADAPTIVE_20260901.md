# E13：KV Cache 自适应档位与初始化固定档位

日期：2026-09-01

性质：单机单卡、3 个独立冷进程 seed 的 formal-profile qualification

权威 artifact：`artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete`

容量 artifact：`artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal`

不计切换开销的稳态阶段 artifact：
`artifacts/e13-tier-phase-once-20260901-attempt-06`

Decode 定因 artifact：

- `artifacts/e13-decode-abba-diagnostic-20260901-attempt-01`
- `artifacts/e13-decode-backing-diagnostic-20260901-attempt-01`
- `artifacts/e13-decode-nsys-diagnostic-20260901-attempt-03`

## 1. 结论摘要

E13 的 late-cohort 压力预测、扩档、恢复缩档和 fail-closed 验收已经闭环，但当前数据
不支持预设的三个性能主张。

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
2. 切换采用同步迁移和全局安全边界；异步复制、后台 planner、内核间 quiescence
   和迁移/推理重叠尚未实现。
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

## 12. 证据

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

Decode 定因：

- [ABBA analysis](artifacts/e13-decode-abba-diagnostic-20260901-attempt-01/analysis.json)
- [backing analysis](artifacts/e13-decode-backing-diagnostic-20260901-attempt-01/analysis.json)
- [Nsight analysis](artifacts/e13-decode-nsys-diagnostic-20260901-attempt-03/analysis.json)
- [Nsight report](artifacts/e13-decode-nsys-diagnostic-20260901-attempt-03/k4_nsys/timeline.nsys-rep)

复现文档位于 `experiments/reproduce/REPRODUCE_E13_KV_ADAPTIVE.md`。
