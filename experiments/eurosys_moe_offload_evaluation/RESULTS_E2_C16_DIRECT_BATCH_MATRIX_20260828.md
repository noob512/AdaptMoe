# E2-C16 direct-batch 单轮矩阵实验报告

日期：2026-08-28（Asia/Shanghai）

## 1. 结论与边界

本轮完成的是 DeepSeek-V2-Lite-Chat BF16 的 **E2-C16 direct-batch 变体**：16 条请求在第一次 `EngineCore.step()` 前一次性入队，无 HTTP/client，不存在客户端侧 `1+15`，并强制全部 Prefill 完成后再进入 Decode。它是单进程、单轮 qualification，不是论文计划中的 equal-HBM、closed-loop、3/5 次独立进程正式矩阵。

- 计划的 24 个“机制 × workload”组合中，20 个找到可运行配置并通过结构性验收，共 320/320 条正式请求成功。
- 192/320 条请求同时满足对应 TTFT/TPOT SLO。
- all-GPU 的 Ordinary/RAG/Long 在固定完整请求和 KV 容量下 OOM，记为容量 N/A；MoE-Infinity RAG 仍阻塞。
- ours 的 RAG 结构、完整输出长度和 SLO 均通过，但与原生 vLLM 的 exact-generation 不一致，因此 **完整正确性门禁尚未通过**。
- KTransformers 与 MoE-Infinity 的跨框架 exact-generation 也尚未通过。

因此，本轮可以证明 direct-batch 执行路径和 20 个 cell 的可运行性，不能直接作为论文中的最终机制排名。

## 2. 固定配置

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB |
| 模型 | `deepseek-ai/DeepSeek-V2-Lite-Chat`，snapshot `85864749cd611b4353ce1decdb286193298f64c7` |
| 精度/并行 | BF16，TP=1，单进程 |
| CPU | `taskset -c 0-13`，OMP/MKL 14 threads |
| 随机种子 | 20260823，greedy (`temperature=0`) |
| 请求提交 | 16 次 `add_request` 均发生在首次 `step()` 前；无 HTTP/socket/client |
| 调度 | pure-phase；Prefill cohort 完成后统一 Decode；无 mixed step |
| warmup | 每个候选 3 条请求，输出 8 tokens |
| 正式重复 | 1 次 |

| Workload | Prompt/Output | KV cache | `max_num_batched_tokens` | TTFT/TPOT SLO |
|---|---:|---:|---:|---:|
| Short dialogue | 128/32 | 96 MiB | 2,048 | 2,000/100 ms |
| Ordinary chat | 512/128 | 336 MiB | 8,192 | 5,000/150 ms |
| RAG summary | 8,192/128 | 4,160 MiB | 16,384 | 30,000/200 ms |
| Long generation | 512/512 | 528 MiB | 8,192 | 5,000/150 ms |

RAG 的 16 条请求仍为一次 admission；由于 `16,384 / 8,192 = 2`，每个 Prefill engine step 最多处理 2 条请求。调度器在 16 条请求全部 Prefill 后才切换 Decode，因此这不是客户端把请求拆成两拨。

## 3. 有效结果

下表均为 16 条请求的均值；SLO 表示同时满足 TTFT 与 TPOT 的请求数。

### Short dialogue，C16

| 机制 | 最小成功配置 | TTFT (ms) | TPOT (ms) | E2E (ms) | SLO | Peak HBM (MiB) |
|---|---|---:|---:|---:|---:|---:|
| all-GPU | 不卸载 | 66.01 | 53.87 | 1,735.94 | 16/16 | 31,245 |
| pinned-UVA | 0.01 GiB budget | 87.50 | 53.79 | 1,754.95 | 16/16 | 31,233 |
| copy-on-forward | 0.01 GiB budget | 66.24 | 54.56 | 1,757.75 | 16/16 | 31,225 |
| ours | cold1/层 | 83.77 | 19.45 | 686.77 | 16/16 | 31,187 |
| KTransformers | cold1/层 | 652.72 | 338.65 | 11,150.95 | 0/16 | 31,473 |
| MoE-Infinity | device ratio 0.80 | 11,067.54 | 1,708.45 | 64,029.49 | 0/16 | 31,023 |

### Ordinary chat，C16

| 机制 | 最小成功配置 | TTFT (ms) | TPOT (ms) | E2E (ms) | SLO | Peak HBM (MiB) |
|---|---|---:|---:|---:|---:|---:|
| all-GPU | 固定配置容量 N/A | — | — | — | — | — |
| pinned-UVA | 0.5 GiB budget | 1,369.48 | 54.51 | 8,292.01 | 16/16 | 31,347 |
| copy-on-forward | 2 层完整 `w2_weight`，704 MiB | 236.30 | 56.01 | 7,349.62 | 16/16 | 31,697 |
| ours | cold5/层 | 310.28 | 35.04 | 4,759.97 | 16/16 | 31,563 |
| KTransformers | cold2/层 | 1,898.29 | 379.97 | 50,154.35 | 0/16 | 31,849 |
| MoE-Infinity | device ratio 0.75 | 12,096.48 | 1,796.71 | 240,278.63 | 0/16 | 31,953 |

### RAG summary，C16

| 机制 | 最小成功配置 | TTFT (ms) | TPOT (ms) | E2E (ms) | SLO | Peak HBM (MiB) |
|---|---|---:|---:|---:|---:|---:|
| all-GPU | 固定配置容量 N/A | — | — | — | — | — |
| pinned-UVA | 5.0 GiB budget | 37,881.14 | 340.49 | 81,123.95 | 0/16 | 31,711 |
| copy-on-forward | 15 层完整 `w2_weight`，5.15625 GiB | 2,849.41 | 139.77 | 20,599.67 | 16/16 | 32,059 |
| ours | cold19/层 | 2,901.15 | 144.85 | 21,296.80 | 16/16 | 31,705 |
| KTransformers | cold12/层 | 65,190.02 | 855.86 | 173,884.61 | 0/16 | 32,055 |
| MoE-Infinity | 0.80–0.55 均 OOM | — | — | — | — | — |

### Long generation，C16

| 机制 | 最小成功配置 | TTFT (ms) | TPOT (ms) | E2E (ms) | SLO | Peak HBM (MiB) |
|---|---|---:|---:|---:|---:|---:|
| all-GPU | 固定配置容量 N/A | — | — | — | — | — |
| pinned-UVA | 0.5 GiB budget | 1,375.41 | 54.31 | 29,128.29 | 16/16 | 31,469 |
| copy-on-forward | 3 层完整 `w2_weight`，1.03125 GiB | 243.41 | 56.02 | 28,871.18 | 16/16 | 31,759 |
| ours | cold5/层 | 309.88 | 36.93 | 19,179.38 | 16/16 | 31,749 |
| KTransformers | cold2/层 | 2,128.33 | 402.03 | 207,568.18 | 0/16 | 32,029 |
| MoE-Infinity | device ratio 0.70 | 10,647.90 | 2,133.75 | 1,100,993.53 | 0/16 | 32,097 |

## 4. 结构与正确性验收

所有 20 个有效 cell 均满足：无 HTTP/client、16 条请求在首次 engine step 前完成 admission、三次 warmup 成功、16/16 完整生成、全部 Prefill 在 Decode 前完成、无 mixed step、无 CUDA illegal access/Xid/PTE/PDE fault、进程退出后 GPU 0 MiB 且 stock UVM 恢复。

逐请求输出 hash：

- Short：all-GPU、pinned-UVA、copy-on-forward、ours 全部一致。
- Ordinary：pinned-UVA、copy-on-forward、ours 全部一致。
- Long：pinned-UVA、copy-on-forward、ours 全部一致。
- RAG：pinned-UVA 与 copy-on-forward 一致；ours 的 16/16 均发生生成分叉。
- KTransformers、MoE-Infinity 尚未通过跨框架 exact-generation。

RAG 的 ours cell 因此只能标记为“结构/SLO 通过、exact-generation 未通过”。详细分析见 `UNRESOLVED_E2_C16_DIRECT_BATCH_MATRIX_20260828.md`。

## 5. 容量扫描结果

- ours RAG：cold16 尚有 351,797,248 bytes 热点页在 CPU；cold17 尚有 170,393,600 bytes；cold18 已恢复全部热点页，但仅余 2,883,584 bytes，低于 201,326,592-byte 安全余量；cold19 通过。
- copy RAG：13 层在 512 MiB attention 临时分配处 OOM；14 层在 264 MiB MoE 临时分配处 OOM；15 层通过。
- all-GPU：Ordinary、RAG、Long 分别在 172、684、342 MiB 临时分配处 OOM。固定模型、KV 和完整请求不变时属于真实容量 N/A。
- KTransformers：较小 cold 数 OOM，增加卸载后四类 workload 均找到可运行点。

## 6. 性能观察

- ours 相对 pinned-UVA 的 TPOT 加速为 Short 2.76×、Ordinary 1.56×、RAG 2.35×、Long 1.47×。
- ours 相对 copy-on-forward 在 Short、Ordinary、Long 的 TPOT 分别快 2.80×、1.60×、1.52×；RAG 中 ours TPOT 比 copy 慢约 3.6%。
- KTransformers 和 MoE-Infinity 的结构性 direct-batch 均成立，但均未满足任何有效 cell 的 TPOT SLO。
- MoE-Infinity 当前 non-paged backend 在一个 scheduler batch 内逐序列执行 HF forward，性能差不是因为请求被拆成两批。

这些数字未做 equal-HBM，也只有单轮，不能解释为最终机制优劣。

## 7. 证据

- 主扫描：`artifacts/e2-c16-direct-batch-20260828-attempt-01/`
- MoE-Infinity 修复复测：`artifacts/e2-c16-direct-batch-20260828-attempt-03-moe-logical-kv/`
- ours RAG 容量复测：`artifacts/e2-c16-direct-batch-20260828-attempt-04-ours-rag-retry/`
- copy 精确卸载复测：`artifacts/e2-c16-direct-batch-20260828-attempt-05-copy-exact/`
- RAG Graph 诊断：`artifacts/e2-c16-direct-batch-20260828-attempt-06-rag-graph-diagnostic/`

每个有效 case 均含 `command.txt`、`environment.json`、`run.log`、`result.json`、`verification.json`、`nvml.csv`、`resource_time.txt`、kernel log 和 GPU before/after 状态。
