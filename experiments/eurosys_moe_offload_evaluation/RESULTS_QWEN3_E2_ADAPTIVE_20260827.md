# Qwen3 E2 自适应并发实验报告（2026-08-27）

## 1. 结论摘要

本轮 Qwen3 E2 的 ours 路径成功完成全部 16 个 workload × concurrency cell：116/116 个测量请求成功，两个独立进程块的 verifier 均为 `ok`，没有出现 Xid 31、FAULT_PTE、FAULT_PDE、illegal memory access、CUDA error 或 eager fallback。实验结束后 alias/backing 全部释放，GPU 显存归零并恢复 stock UVM。

本轮验证了两个关键修复：

1. 在恢复热门专家前，捕获并永久保留正式 pure-Prefill CUDA Graph。正式服务复用 Graph 私有 allocator pool，不再为大 Prefill 临时向驱动申请不匹配的新 native segment，因此没有驱逐热门 managed 专家页。
2. phase-batched scheduler 使用 KV manager 的实际 block 计费，在 2 GiB KV 无法同时容纳全部长请求时按 FCFS 拆分子 cohort，消除了 RAG C4/C8/C16 的 `Running/Waiting` 死锁。

并发自适应策略也按预期生效：C1 使用每层 cold19；C4/C8/C16 使用每层 cold25。计划只在进程块之间随并发档位选择，测量期间不在线改变专家集合。

> 范围说明：这是 ours/Qwen3 的完整 16-cell qualification pass，不是设计文档要求的五次独立进程重复，也不包含 KTransformers 等系统的 equal-HBM 横向排名。因此它证明当前 ours 路径可以稳定完成 E2 矩阵，但不能单独作为论文级 E2 跨系统主表。

## 2. 实验问题与修复

### 2.1 先前失败机制

先前流程在恢复热门专家后才捕获大 pure-Prefill Graph。16×1024 Prefill 的 Attention、dense MLP、MoE workspace 和 Graph 私有池需要新的 native allocations；碎片化 HBM 下，PMA 会通过驱逐 managed 热门权重来满足这些分配。由于 `uvm_manual_migration_fault_mode=2` 禁止访问故障时自动迁回，下一次 fused-MoE 读取被驱逐的热门专家便产生 PDE/PTE fault。

单独 `cudaMalloc`/`cudaFree` 一个 2 GiB sentinel 不能保证正式 Prefill 复用这组物理页：释放后它只是普通 driver-visible free HBM，且不是 PyTorch caching allocator 或 CUDA Graph 私有池的一部分。

### 2.2 本轮启动顺序

本轮使用以下顺序：

```text
加载模型并建立 stable-envelope
→ 全部专家保持 audited remote READ topology
→ 完成 KV、kernel、sampler warmup
→ 捕获并保留 remote uniform-Decode Graph
→ RESUME 固定 cold alias banks，热门专家仍在 CPU
→ 捕获并永久保留最终 pure-Prefill Graph
→ preserve_allocator_cache=True 恢复全部热门专家到 GPU
→ 驻留、VA、hot-read、安全余量审计
→ pure Prefill epoch / pure Decode epoch 正式服务
```

这里保留的是正式服务实际 replay 的 GraphExec，而不是捕获后销毁的 seed Graph。恢复热门专家前后 allocator reserved 字节完全不变，证明恢复过程没有清空已经建立的 Graph/allocator pool。

### 2.3 pure phase 约束

CUDA Graph 没有禁用。配置为：

```text
CompilationMode.NONE
CUDAGraphMode.FULL_SEPARATE
```

同一 scheduler step 只允许 pure Prefill 或 uniform Decode；运行时新到达的 Prefill 不会插入当前 Decode epoch。Prefill 与 Decode 分别 replay 独立 Graph，因此没有使用未经 qualification 的 mixed Prefill/Decode Graph。

## 3. 实验设置

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32 GiB |
| 模型 | `Qwen/Qwen3-30B-A3B-FP8`，snapshot `d206ba7…` |
| MoE 结构 | 48 层，每层 128 个专家 |
| KV cache | 2,147,483,648 B（2 GiB） |
| max model length | 8,320 tokens |
| max batched tokens | 16,384 |
| max sequences | 16 |
| Attention | `TRITON_ATTN` |
| phase mode | `prefill_graph_remote_decode` |
| Graph mode | `FULL_SEPARATE` |
| UVM fault mode | 2，禁止 fault-driven CPU→GPU 自动迁回 |
| 调度 | phase-batched、KV-aware FCFS cohort |
| Prefix cache | 关闭 |
| 测量 cell | 4 workloads × C1/C4/C8/C16 = 16 |
| 测量请求 | 116 |

工作负载为：

| workload | prompt tokens | output tokens | TTFT SLO | TPOT SLO |
|---|---:|---:|---:|---:|
| short_dialogue | 128 | 32 | 2,000 ms | 100 ms |
| ordinary_chat | 512 | 128 | 5,000 ms | 150 ms |
| rag_summary | 8,192 | 128 | 30,000 ms | 200 ms |
| long_generation | 512 | 512 | 5,000 ms | 150 ms |

## 4. 并发自适应专家计划

| 并发档位 | 冷专家/层 | 冷专家实例 | 冷主权重 | 最大 Prefill Graph | calibration Decode route fraction |
|---|---:|---:|---:|---:|---:|
| C1 | 19 | 912 | 4,303,355,904 B（4.008 GiB） | 8,192 tokens | 0.0058% |
| C4/C8/C16 | 25 | 1,200 | 5,662,310,400 B（5.273 GiB） | 16,384 tokens | 0.0256% |

选择规则由 launcher 在进程块启动前确定：C1 不需要超过 8K 的 pure-Prefill Graph，使用 cold19；C4/C8/C16 需要覆盖 16K Graph，使用 cold25 为 Graph pool、热权重和安全余量同时提供空间。同一进程块内的计划冻结，避免测量期间因在线换计划引入额外变量。

## 5. 完整性能数据

`客户端 TPOT` 是客户端原始定义，包含 Prefill→Decode cutover 后的首个 ITL。`稳态 Decode TPOT` 对每个请求删除首个 ITL 后再求均值，更接近设计文档要求的稳定 Decode 指标。SLO 列表示同时满足该 workload 的 TTFT 与客户端 TPOT SLO 的请求数；SLO 未满足不表示推理请求失败。

| 工作负载 | C | cold/层 | TTFT ms | 客户端 TPOT ms | 稳态 Decode TPOT ms | E2E ms | goodput req/s | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_dialogue | 1 | 19 | 567.93 | 33.08 | 10.87 | 1,593.47 | 0.627 | 1/1 |
| short_dialogue | 4 | 25 | 2,127.16 | 44.66 | 17.51 | 3,511.67 | 0.247 | 1/4 |
| short_dialogue | 8 | 25 | 2,387.99 | 44.60 | 18.17 | 3,770.59 | 0.248 | 1/8 |
| short_dialogue | 16 | 25 | 2,616.12 | 55.73 | 28.06 | 4,343.73 | 0.222 | 1/16 |
| ordinary_chat | 1 | 19 | 619.72 | 16.20 | 10.54 | 2,676.50 | 0.373 | 1/1 |
| ordinary_chat | 4 | 25 | 2,909.41 | 24.76 | 18.30 | 6,053.95 | 0.563 | 4/4 |
| ordinary_chat | 8 | 25 | 3,385.44 | 36.17 | 29.66 | 7,979.31 | 0.920 | 8/8 |
| ordinary_chat | 16 | 25 | 3,606.55 | 46.45 | 40.02 | 9,505.21 | 1.608 | 16/16 |
| rag_summary | 1 | 19 | 1,049.23 | 19.75 | 14.11 | 3,557.67 | 0.281 | 1/1 |
| rag_summary | 4 | 25 | 5,445.39 | 24.27 | 17.51 | 8,528.26 | 0.322 | 4/4 |
| rag_summary | 8 | 25 | 10,685.82 | 27.00 | 20.21 | 14,115.09 | 0.348 | 8/8 |
| rag_summary | 16 | 25 | 20,262.94 | 30.30 | 23.48 | 24,110.53 | 0.281 | 12/16 |
| long_generation | 1 | 19 | 593.02 | 13.41 | 11.87 | 7,446.61 | 0.134 | 1/1 |
| long_generation | 4 | 25 | 6,904.31 | 21.46 | 19.78 | 17,872.33 | 0.047 | 1/4 |
| long_generation | 8 | 25 | 8,162.09 | 24.78 | 22.81 | 20,825.73 | 0.044 | 1/8 |
| long_generation | 16 | 25 | 8,539.94 | 45.35 | 43.64 | 31,714.89 | 0.030 | 1/16 |

原始及完整派生数据位于：

- `artifacts/e2-qwen3-adaptive-concurrency-20260827-attempt-01/metrics.csv`
- `artifacts/e2-qwen3-adaptive-concurrency-20260827-attempt-01/summary.json`
- 每个 cell 的 `client.json`，包含逐请求 TTFT、E2E、输出 token hash 和完整 ITL 数组。

## 6. Graph、HBM 与驻留证据

| 指标 | C1 / cold19 | C4–C16 / cold25 |
|---|---:|---:|
| Prefill Graph sizes | 128–8,192，共 6 个 | 128–16,384，共 7 个 |
| Decode Graph batch sizes | B1/B4/B8/B16 | B1/B4/B8/B16 |
| serving-gate hot GPU bytes | 24,687,673,344 | 23,328,718,848 |
| serving-gate hot CPU/unknown bytes | 0 / 0 | 0 / 0 |
| serving-gate cold CPU bytes | 4,303,355,904 | 5,662,310,400 |
| serving-gate cold GPU/unknown bytes | 0 / 0 | 0 / 0 |
| allocator reserved，最终 reconciliation 前/后 | 34,701,574,144 / 34,701,574,144 B | 35,494,297,600 / 35,494,297,600 B |
| serving-gate free | 971,767,808 B（926.75 MiB） | 1,185,677,312 B（1,130.75 MiB） |
| minimum free gate | 201,326,592 B（192 MiB） | 201,326,592 B（192 MiB） |
| NVML peak HBM | 31,953 MiB | 31,245 MiB |
| hot-read gate OK | 103 | 375 |

关键证据是：正式 Graph 已存在后，热门专家全部显式迁回 GPU；最终 reconciliation 的 allocator reserved 前后完全相同，同时 free 高于 192 MiB gate。正式 16K Prefill 未再导致热门权重迁到 CPU。

## 7. RAG KV 子 cohort

2 GiB KV pool 提供 1,364 个 free blocks；单个 8K RAG 请求的 lifetime 预算为 520 blocks，保留 1 block 后一次最多安全纳入两个请求。KV-aware scheduler 因此按 FCFS 将长请求拆批。

RAG C16 的 measurement cohort 依次记录：

```text
candidates: 16 → 14 → 12 → 10 → 8 → 6 → 4 → 2
admitted:    2    2    2    2    2   2   2   2
```

最后两个请求可全部容纳，不产生 partition 日志。整个 cold25 进程块共记录 11 次 partition（还包括 RAG warmup/C4/C8）。RAG C16 的 16/16 请求均成功；12/16 满足 TTFT≤30 s、TPOT≤200 ms。并发增加后的 TTFT 上升主要来自 KV 容量约束下的排队，而不是临时缓冲区驱逐或 GPU fault。

## 8. 正确性、故障与清理

两个进程块均满足：

- 顶层 verifier：`status=ok`；
- client success rate：100%；
- Xid、PTE/PDE fault、illegal access、CUDA error：0；
- eager fallback：0；
- static alias destroy：`ok=true`；
- remaining alias count/bytes：0/0；
- remaining backing bytes：0；
- server return code：0；
- GPU cleanup：true；
- 最终 stock UVM 已恢复，GPU memory used 为 0 MiB。

## 9. 实验结论

1. **正式 Graph 先占位的方法有效。** 需要在热门专家恢复前捕获并保留最终 pure-Prefill Graph；销毁 seed Graph 或独立 sentinel `cudaFree` 都不能提供等价的 allocator-pool 所有权与确定性复用。
2. **大 Prefill 的历史故障不是 KV 本身挤占。** 本轮 2 GiB KV 不变，正确预建 Graph pool 后 16K pure Prefill 可稳定运行，热门专家保持 GPU 驻留。
3. **专家卸载量必须与最大并发 shape 联合选择。** cold19 足以覆盖 C1/8K Graph；16K Graph 需要 cold25 才能同时容纳 Graph native allocations、2 GiB KV、热专家和安全余量。
4. **2 GiB KV 对 8K RAG 的限制表现为 admission 排队。** KV-aware 子 cohort 保证完成性，但 C8/C16 的 TTFT 会随批次数增加；这是容量/调度代价，不是 UVM 页表故障。
5. **steady Decode 应排除首次 cutover ITL。** 客户端原 TPOT 含一次 PARK/remote-map/cutover 延迟，会高估稳态 Decode；本报告同时给出两种口径。

## 10. 限制与下一步

- 当前只有一个完整进程重复；论文主结果仍需至少五个独立进程重复并报告置信区间。
- 当前只有 ours/Qwen3；尚不能得出 ours 与 KT、MoE-Infinity 或 pinned vLLM 的 equal-HBM 排名。
- 并发计划目前是档位映射（C1→cold19，C4/C8/C16→cold25），不是请求运行期间在线改变 residency 的动态控制器。
- RAG C16 的高 TTFT 说明 2 GiB KV 是 admission 瓶颈；后续 equal-HBM 比较必须让所有系统使用同样的 KV 容量与 cohort admission 规则。
- 后续正式重复应保留本轮的 `summary.json`/`metrics.csv` 生成方法，并继续同时报告 client TPOT 与 steady Decode TPOT。
