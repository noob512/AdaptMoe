# E3 B1/B8 单轮无 HTTP 实验结果（2026-08-28）

## 结论

本轮 qualification 通过。最终 artifact 为：

`artifacts/e3-repro-20260828-attempt-03-single-round-no-http`

- verifier：`status=ok`，`errors=[]`；
- custom H/A/R：21/21 个有效 run 成功，84/84 个请求完整生成；
- KTransformers direct：9/9 个 measurement cell 成功，37/37 个请求完整生成；
- 总计：121/121 个请求成功；
- batch 仅使用 B1、B8，没有改成其他 batch；
- custom 和 KTransformers 均不经过 HTTP；
- B8 在第一次执行前完整提交，不存在 `1+7`；
- Prefill 与 Decode 分步执行，无 mixed Prefill/Decode step；
- 没有 CUDA illegal access、Xid、PTE/PDE fault、OOM 或 traceback；
- 结束后 stock UVM 已恢复，GPU 为 0 MiB。

本轮只有一个独立进程块，只能给出单轮点估计，不能替代正式重复实验和置信区间。
此外，本轮 H/A/R 是 eager 路径，不能直接与开启 CUDA Graph 的 E2 custom 数据比较。

## 冻结配置

- 模型：Qwen3-30B-A3B-FP8；
- snapshot：`d206ba732169f29bb77fbf80fc2c4b81d4d30782`；
- GPU：RTX 5090 32 GiB；
- CPU：`taskset -c 0-13`，OMP/MKL 14 threads；
- 固定专家计划：480 个 `(layer, expert)`，48 层、每层 10 个 CPU 专家；
- selected bytes：2,264,924,160 B；
- custom launcher 传入候选 Graph sizes 1、4、8，但 H/A/R arm contract 均为
  `graph=false`；实际运行是 `enforce_eager=true`、`cudagraph_mode=NONE`，没有
  CUDA Graph capture/replay；
- KTransformers：118 GPU experts/layer、10 CPU experts/layer、eager、Graph disabled；
- 随机种子：0；
- custom UVM 用于 H/A/R，KTransformers 和实验结束后使用 stock UVM；
- KV：`max(round_up_16(prompt+output+16) * batch * 98,304 B, 160 MiB)`。

三个 custom arm：

- H：Prefill copy/GPU local，Decode remote-read；
- A：Prefill 与 Decode 均走 copy/GPU local；
- R：Prefill 与 Decode 均走 remote-read。

## custom 单轮结果

时间单位均为 ms。`E2E` 是 runner 的完整 cycle 时间；`Decode` 是稳态 Decode step 中位数。

| Workload | Batch | H E2E | A E2E | R E2E | H cutover | H/A/R Decode | 单轮最优 |
|---|---:|---:|---:|---:|---:|---:|---|
| Short 128→32 | 1 | 2125.08 | 2079.51 | 1995.22 | 129.19 | 60.72 / 63.82 / 61.04 | R |
| Short 128→32 | 8 | 2215.80 | 2142.97 | 2050.41 | 129.41 | 63.05 / 65.10 / 62.49 | R |
| Ordinary 512→128 | 1 | 7840.14 | 8200.99 | 7838.10 | 128.13 | 59.61 / 63.58 / 60.86 | R |
| Ordinary 512→128 | 8 | 8086.51 | 8345.48 | 8215.07 | 128.43 | 61.06 / 64.15 / 63.20 | H |
| Medium 2048→128 | 1 | 7916.03 | 8283.43 | 7972.97 | 131.83 | 59.96 / 63.95 / 61.44 | H |
| Long 512→512 | 1 | 31797.83 | 32782.23 | 31523.08 | 129.41 | 61.62 / 63.77 / 61.22 | R |
| Long 512→512 | 8 | 31998.38 | 33838.82 | 31943.89 | 128.41 | 61.82 / 65.76 / 62.08 | R |

单轮 winner 为 R 5 个、H 2 个、A 0 个。H 的阶段切换固定约 128–132 ms；只有当 Prefill/Decode 的累计收益能够覆盖该成本时，H 才胜出。Ordinary/B1 的 H 与 R 只差约 2.04 ms，远小于正式统计结论所需的重复尺度，不能据此声称 R 稳定更快。

所有 7 个有效逻辑 cell 中，H/A/R 输出 token hash 均一致。

## KTransformers direct 结果

| Workload | Batch | TTFT batch ms | Decode step median ms | E2E ms | Peak HBM MiB |
|---|---:|---:|---:|---:|---:|
| Short 128→32 | 1 | 1030.79 | 172.77 | 6468.21 | 29207 |
| Short 128→32 | 8 | 2536.29 | 254.74 | 10551.46 | 29207 |
| Ordinary 512→128 | 1 | 965.93 | 193.31 | 24463.84 | 30017 |
| Ordinary 512→128 | 8 | 4283.08 | 604.64 | 88783.01 | 30017 |
| Medium 2048→128 | 1 | 2700.23 | 241.10 | 33193.62 | 32037 |
| Medium 2048→128 | 8 | 8211.39 | 532.02 | 84761.63 | 32037 |
| RAG 8192→128 | 1 | 4065.75 | 147.27 | 23690.46 | 30647 |
| Long 512→512 | 1 | 995.44 | 200.05 | 98733.60 | 30359 |
| Long 512→512 | 8 | 3498.62 | 316.78 | 177893.57 | 30359 |

KTransformers 的 B1/B8 每个 cell 都只调用一次 `Engine.generate`；配置记录 `transport=none`、`client_used=false`、`mixed_chunk_enabled=false`。日志中的每层专家审计为 48/48 层均 118 GPU experts。

KTransformers 与 custom 在所有可配对 cell 上的输出 hash 均不同。因此这些 KTransformers 时间只能作为结构性 qualification 数据，尚不能与 H/A/R 做严格的 exact-generation 性能排名。

## N/A 与范围

custom 预注册 N/A：

- Medium/B8：`2048*8=16384`，超过 fail-closed 的 8192-token 单步 Prefill 边界；
- RAG/B1：`8192*1=8192`，达到该边界；
- RAG/B8：同时超过 Prefill 边界，计算 KV 为 6,555,697,152 B。

KTransformers 的 RAG/B8 计算 KV 为 6,555,697,152 B，超过冻结的 2.8 GB KT KV headroom，因此未运行；没有把 B8 改成较小 batch。原生 vLLM UVA/copy 因只能卸载整个 Parameter、无法精确复现 480 个 expert slices，记录为 N/A。

## 无 HTTP 与 pure-phase 证据

以 `Short/B8/H` 为例：

- 首个 scheduler step 同时包含 request 0–7；
- 每个请求调度 128 个 Prefill token，总计 1024；
- 下一 step 才调度 8 个已有请求、每个 1 个 Decode token；
- `preempted_request_ids=[]`；
- 全部日志未出现 Uvicorn、`POST /generate` 或本地 HTTP URL。

## 主要证据

- `analysis/verification.json`：总体验收；
- `analysis/custom_metrics.csv`：H/A/R 指标；
- `analysis/best_strategy_heatmap.csv`：单轮 winner；
- `analysis/kt_exact_coordinate_metrics.csv`：KT direct 指标；
- `custom/execution.tsv`、`kt/execution.tsv`：实际执行状态；
- 每个 case 的 `run.log`、`result.json`、`nvml.csv`：日志、结果与显存采样；
- `provenance/`：commit、模块、模型、fixture 与冻结计划哈希。
