# E3 Prefill：Graph-copy、vLLM copy-on-forward 与 pinned-UVA 对照

日期：2026-08-28

## 1. 执行状态

| 矩阵 | 状态 | 请求 | 有效 artifact |
|---|---:|---:|---|
| 自定义 Graph-copy | 10/10 cells | 45/45 | `e3-prefill-only-graph-copy-triton-20260828-attempt-02` |
| vLLM copy-on-forward | 10/10 cells | 45/45 | `e3-native-copy-vs-uva-phase-20260828-attempt-01` |
| vLLM pinned-UVA | 10/10 cells | 45/45 | `e3-native-copy-vs-uva-phase-20260828-attempt-01` |

所有 B8 均为 8 条请求在第一次 EngineCore step 前一次性入队。Prefill scheduler step 中没有 cached Decode request。Graph-copy 每个 cell 恰好 replay 一次 full Graph，测量期没有 recapture。最终 GPU 0 MiB，stock UVM 已恢复。

## 2. Prefill 性能

单位为 ms，越低越好。`Graph 对 copy/UVA` 使用对方延迟除以 Graph 延迟；大于 1 表示 Graph 更快。

| 请求 | B | Graph-copy | vLLM copy | UVA remote | Graph 对 copy | Graph 对 UVA | 数值最快 |
|---|---:|---:|---:|---:|---:|---:|---|
| Short 128 | 1 | 66.91 | 64.63 | **61.31** | 0.966× | 0.916× | UVA |
| Short 128 | 8 | 86.23 | **76.03** | 203.90 | 0.882× | 2.365× | copy |
| Ordinary 512 | 1 | 73.28 | **70.08** | 122.76 | 0.956× | 1.675× | copy |
| Ordinary 512 | 8 | 148.91 | **136.85** | 824.96 | 0.919× | 5.540× | copy |
| Medium 2048 | 1 | 111.61 | **102.59** | 421.06 | 0.919× | 3.773× | copy |
| Medium 2048 | 8 | 648.94 | **563.89** | 7,474.99 | 0.869× | 11.519× | copy |
| RAG 8192 | 1 | 584.44 | **458.68** | 3,784.03 | 0.785× | 6.475× | copy |
| RAG 8192 | 8 | 3,708.29 | **2,729.38** | 48,136.85 | 0.736× | 12.981× | copy |
| Long 512 | 1 | 73.77 | **70.93** | 120.26 | 0.962× | 1.630× | copy |
| Long 512 | 8 | 149.41 | **137.43** | 816.70 | 0.920× | 5.466× | copy |

## 3. 数值观察

- Graph-copy 相对 UVA 在 9/10 cells 更快，倍率为 1.630–12.981×。随着 Prompt 和 batch 增大，UVA 需要经 PCIe 反复远程读取更多数据，差距明显扩大。
- Short/B1 是唯一例外：Graph 66.91 ms、UVA 61.31 ms，Graph 慢约 9.1%。该请求的远程读取量小，Graph/alias 路径没有形成收益。
- Graph-copy 相对本轮 FLASH vLLM copy 在 10/10 cells 数值更慢，差距从 3.5% 增加到 35.9%。差距随 Prompt 增大，符合 FlashAttention 与 TRITON attention kernel 差异会在长 Prefill 中放大的特征。
- 从这张跨配置表不能得出“CUDA Graph 搬运弱于 native copy”。此前 Graph 和 native copy 都使用 TRITON 的七个共同 cell 中，Graph-copy 是 7/7 更快，单轮加速为 1.023–1.383×。

## 4. 输出一致性

- native copy 与 UVA 的 10/10 完整输出 token hash 相同。
- Graph 只生成首 token，不能直接与 native 的完整输出 hash 比较。
- 对齐每条请求的首 token 后，9/10 logical cells 完全一致。
- Medium/B8 中 8 条请求有 1 条不同：Graph token 为 `264`，native copy/UVA 为 `279`。该 cell 同时改变了 attention backend 和卸载几何，因此不满足跨三 arm exact-generation gate；其性能只能作为数值参考。

## 5. 配置差异

| 项目 | Graph-copy | vLLM copy/UVA |
|---|---|---|
| Attention | TRITON_ATTN | FLASH_ATTN |
| CUDA Graph | FULL_SEPARATE | 关闭/eager |
| 卸载粒度 | exact expert slices | whole Parameters |
| 常规卸载 | cold10，2.109 GiB selected | 63 Parameters，2.322 GiB |
| 大 Prefill 卸载 | cold38/cold63 | 232/402 Parameters |
| 请求输出 | Prefill + 1 sampled token | 完整 32/128/512 output |
| conditioning | Graph capture/warmup 后一次测量 | 1 conditioning + 1 measurement cycle |

这些差异不会破坏各自矩阵内部的结果，但会混入三机制之间的绝对延迟差异。

## 6. Flash Graph 失败尝试

为了与 native 矩阵使用相同的 `FLASH_ATTN`，首先运行了 `e3-prefill-only-graph-copy-flashattn-20260828-attempt-01`。模型初始化时 vLLM 将请求的 `FULL_SEPARATE` 降级为 `FULL_DECODE_ONLY`，而自定义 `prefill_graph_remote_decode` 路径要求 `FULL_SEPARATE`，因此契约在测量前拒绝启动。

该失败不是显存不足，没有生成性能结果。随后只把 attention backend 恢复为该机制已验证的 TRITON；请求、B1/B8、KV、expert plan 和计时边界均未改变。

## 7. 可使用的结论

当前证据可靠支持：

1. 对长 Prompt 或 B8，直接 UVA remote-read 的 Prefill 代价很高；Graph-copy 数值上可快 2.37–12.98×。
2. 极短 B1 的 UVA 可以更好，说明 copy/Graph 有一个小请求启动成本区间。
3. CUDA Graph 对 native copy 的独立贡献必须在相同 attention backend、相同卸载坐标下复测；本表不适合作为最终论文排名。

## 8. 证据文件

- Graph 机器汇总：`artifacts/e3-prefill-only-graph-copy-triton-20260828-attempt-02/summary.json`
- Graph 验收：`artifacts/e3-prefill-only-graph-copy-triton-20260828-attempt-02/verification.json`
- Native 机器汇总：`artifacts/e3-native-copy-vs-uva-phase-20260828-attempt-01/summary.json`
- Native 验收：`artifacts/e3-native-copy-vs-uva-phase-20260828-attempt-01/verification.json`
- Flash Graph 错误记录：`ERROR_FIX_LOG_E3_PREFILL_GRAPH_COPY_B1_B8_20260828.md`
