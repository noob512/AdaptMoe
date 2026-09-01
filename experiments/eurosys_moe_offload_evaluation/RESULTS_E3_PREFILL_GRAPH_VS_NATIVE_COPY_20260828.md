# E3 Prefill-only：Graph-copy 与 vLLM copy-on-forward（2026-08-28）

## 结论

本轮双机制单轮 qualification 已完成。最终 artifact：

`artifacts/e3-prefill-only-graph-vs-native-copy-20260828-attempt-01`

- Graph-copy：10/10 cell 成功；
- vLLM copy-on-forward：7/7 个可运行 cell 成功，3 个大 Prefill cell 按冻结规则记为 N/A；
- 实际执行 17 个 cell，共 73/73 条请求成功；
- 7 个共同 cell 中 Graph-copy 的 Prefill 延迟均较低，单轮加速为 1.023–1.383 倍，中位数 1.053 倍；
- 7 个共同 cell 的 sampled-token hash 全部一致；
- B8 始终是 8 条请求在第一次 EngineCore step 前一次性入队，不存在 `1+7` 或分批处理；
- 无 HTTP、无客户端、无 Decode、无 mixed Prefill/Decode step；
- 无 CUDA illegal access、Xid、PTE/PDE fault、OOM 或 traceback；结束后 GPU 0 MiB，stock UVM 已恢复。

这只是单轮点估计，且原生 vLLM 无法按 expert slice 卸载，因此不能作为论文级严格等坐标排名。

## 实验范围

模型为 Qwen3-30B-A3B-FP8，snapshot `d206ba732169f29bb77fbf80fc2c4b81d4d30782`，GPU 为 RTX 5090 32 GiB。使用 B1/B8、五类冻结 fixture。`max_tokens=1`，计时只覆盖一次完整 Prompt Prefill 与首 token sampling；没有 Decode。

Graph-copy 使用 frozen exact expert coordinates：

| 几何 | 适用 cell | 每层 cold | selected bytes |
|---|---|---:|---:|
| cold10 | 普通有效 cell | 10 | 2,264,924,160 B（2.109 GiB） |
| cold38 | Medium/B8、RAG/B1 | 38 | 8,606,711,808 B（8.016 GiB） |
| cold63 | RAG/B8 | 63 | 14,269,022,208 B（13.289 GiB） |

原生 vLLM copy-on-forward 不支持 expert-slice，只能选择最接近字节数的完整 packed Parameter：4/14/24 层，对应 2.25/7.875/13.5 GiB。它是 footprint-matched 辅助基线，不是 exact-coordinate 基线。

## 两机制共同 cell

`Graph 加速` 定义为 `native Prefill ms / Graph Prefill ms`，大于 1 表示 Graph-copy 更快。

| Workload | Batch | Graph-copy ms | Native copy ms | Graph 加速 | Graph tok/s | Native tok/s | Peak HBM G/N MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short 128 | 1 | 66.34 | 76.67 | 1.156x | 1,929 | 1,669 | 30,541 / 29,025 |
| Short 128 | 8 | 85.57 | 90.12 | 1.053x | 11,967 | 11,363 | 30,549 / 29,083 |
| Ordinary 512 | 1 | 73.03 | 77.42 | 1.060x | 7,011 | 6,613 | 30,549 / 29,053 |
| Ordinary 512 | 8 | 149.28 | 152.66 | 1.023x | 27,439 | 26,830 | 30,835 / 29,771 |
| Medium 2048 | 1 | 111.67 | 115.35 | 1.033x | 18,339 | 17,755 | 30,549 / 29,205 |
| Long 512 | 1 | 73.93 | 102.25 | 1.383x | 6,925 | 5,007 | 30,549 / 29,085 |
| Long 512 | 8 | 148.69 | 153.29 | 1.031x | 27,548 | 26,721 | 30,953 / 30,045 |

Graph-copy 在 7/7 共同 cell 上单轮胜出，但除 Long/B1 外多数差距较小；没有独立进程重复，不能把 2%–6% 差距解释成稳定优势。Long/B1 的 38% 单点差距也需要重复确认。

Graph-copy 的峰值 HBM 比 native 高 908–1,516 MiB。原因之一是 Graph pool 与 custom alias/backing；另一个重要差异是 native 的完整 Parameter 卸载几何与 exact expert-slice 不相同。

## Graph-copy 独有的大 Prefill 结果

| Workload | Batch | 单步 tokens | cold | Prefill ms | tok/s | Peak HBM MiB |
|---|---:|---:|---:|---:|---:|---:|
| Medium 2048 | 8 | 16,384 | 38 | 648.31 | 25,272 | 30,617 |
| RAG 8192 | 1 | 8,192 | 38 | 584.99 | 14,004 | 30,587 |
| RAG 8192 | 8 | 65,536 | 63 | 3,711.05 | 17,660 | 30,829 |

这三项证明 FULL_SEPARATE Graph-copy 在保持 B8 为单一批次时可完成 65,536-token 单步 Prefill。它们没有可用的 native 数值，不能计算机制加速比。

## 与此前 eager H/A/R 的直接对比

旧数据来自正确参数的 `e3-repro-20260828-attempt-03-single-round-no-http`。只比较两轮都存在、且均使用 cold10 exact coordinates 的 7 个 cell。A 是 eager copy/alias，R 是 eager remote-read；H 的 Prefill 路径与 A 同属 copy/local，本表不重复列出 H。

| Workload | Batch | Graph-copy G ms | eager-copy A ms | remote R ms | 单轮最快 |
|---|---:|---:|---:|---:|---|
| Short 128 | 1 | 66.34 | 75.73 | 77.91 | G |
| Short 128 | 8 | 85.57 | 93.73 | 78.50 | R |
| Ordinary 512 | 1 | 73.03 | 81.57 | 72.13 | R |
| Ordinary 512 | 8 | 149.28 | 156.06 | 150.99 | G |
| Medium 2048 | 1 | 111.67 | 119.79 | 141.13 | G |
| Long 512 | 1 | 73.93 | 103.35 | 95.26 | G |
| Long 512 | 8 | 148.69 | 156.88 | 132.89 | R |

Graph-copy 比 eager-copy A 在 7/7 cell 更快；与 remote R 相比，Graph-copy 赢 4 个，remote 赢 3 个。所有旧 H/A/R 的首 token 与本轮 Graph/native 输出一致。由于这是跨两轮单点比较，结论仅用于选择后续正式复测范围。

## 未拆分批次的证据

每个结果都满足：

- `all_requests_admitted_before_first_step=true`；
- `client_used=false`、`http_server_started=false`；
- runner 在一次 `_validate_and_add_requests()` 中提交完整 B1/B8；
- 只调用一次 `llm_engine.step()`，且 scheduler 只被调用一次；
- B8 的 `new_request_ids` 长度为 8；
- `total_num_scheduled_tokens = prompt_tokens * 8`；
- `cached_request_ids=[]`、`resumed_request_ids=[]`、`preempted_request_ids=[]`；
- Graph-copy 的 `graph_wrapper_delta_prefill.full_replay=1`，测量期没有 recapture。

## N/A

Native copy-on-forward 的 Medium/B8、RAG/B1、RAG/B8 为 N/A。当前 RTX 5090/TRITON_ATTN eager 路径已复现单步 Prefill `>=8192` tokens 的 TMA descriptor 700 / CUDA illegal-address。为保持实验目标，没有拆 batch、启用 chunked Prefill 或更换 attention backend。

## 证据

- `summary.json`：机器可读指标与加速比；
- `verification.json`：批次、Graph、输出、错误和 cleanup 验收；
- `execution.tsv`：20 个计划 arm-cell 的状态与 HBM；
- `cases/*/result.json`：scheduler、计时、Graph replay、输出 hash 与 cleanup；
- `cases/*/command.txt`、`run.log`、`nvml.csv`：实际命令、日志和显存；
- `provenance/`：输入哈希、模块、硬件与仓库状态。
