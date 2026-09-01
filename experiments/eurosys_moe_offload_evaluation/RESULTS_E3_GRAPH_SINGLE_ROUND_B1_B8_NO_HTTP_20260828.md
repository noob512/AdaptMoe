# E3 CUDA Graph 单轮 B1/B8 direct-batch 结果（2026-08-28）

## 结论

本轮为单独冷进程块、无 HTTP、整批请求一次性入队、Prefill/Decode
pure-phase 的 qualification。聚合 verifier 为 `status=ok`：10/10 个逻辑
cell、24/24 个有效 run、101/101 条请求成功并生成完整长度输出。

常规规模的 7 个 cell 使用同一 cold10 计划成对比较 G/B/C。C 胜 6 个，G
胜 1 个，B 胜 0 个。同 cell 的 G/B/C 输出 token hash 全部一致。大 Prefill
的 Medium/B8、RAG/B1、RAG/B8 只运行当前 runtime 已认证的 D 双 Graph
contract；这 3 个结果是结构性 qualification，不是 G/B/C 排名。

## Arm 定义

| Arm | Prefill | Decode | Graph mode |
|---|---|---|---|
| G | sealed alias/copy，eager | remote-read | `FULL_DECODE_ONLY` |
| B | sealed alias/copy，eager | sealed alias/copy | `FULL_DECODE_ONLY` |
| C | remote-read，eager | remote-read | `FULL_DECODE_ONLY` |
| D | sealed alias/copy Graph | remote-read Graph | `FULL_SEPARATE` |

G/B/C 的 Prefill 没有 Graph；它们只比较 Decode Graph 数据路径。D 的 Prefill
和 Decode 都是 Graph，并在两阶段间执行 PARK/RESUME 切换。

## G/B/C 成对结果

单位均为 ms；Decode 为 measured steady-step median，E2E 包含 G 的切换成本。

| Workload | Batch | Winner | G E2E | B E2E | C E2E | G/B/C Decode median |
|---|---:|---|---:|---:|---:|---:|
| Short 128→32 | 1 | C | 444.52 | 1711.12 | 358.44 | 8.76 / 52.36 / 8.75 |
| Short 128→32 | 8 | C | 629.66 | 1897.72 | 534.23 | 14.25 / 57.83 / 14.22 |
| Ordinary 512→128 | 1 | C | 1308.44 | 6767.66 | 1223.15 | 8.85 / 52.52 / 8.86 |
| Ordinary 512→128 | 8 | C | 2162.43 | 7586.32 | 2069.42 | 14.98 / 58.35 / 14.95 |
| Medium 2048→128 | 1 | G | 1872.07 | 6831.41 | 1982.83 | 13.12 / 52.71 / 13.09 |
| Long 512→512 | 1 | C | 5119.49 | 26979.07 | 5032.75 | 9.42 / 52.54 / 9.42 |
| Long 512→512 | 8 | C | 8056.82 | 29830.87 | 7937.00 | 15.23 / 58.01 / 15.20 |

G 每个 cell 的切换成本为 67.32–69.76 ms。Medium/B1 中 G 的 Prefill 为
118.74 ms，而 C 为 317.01 ms；Prefill 收益覆盖切换成本，因此 G 成为唯一
胜者。其他 6 个 cell 中，G 和 C 的稳态 Decode 基本相同，C 因没有切换而
胜出。B 的 Decode median 为 52.36–58.35 ms，显著慢于 remote Graph 路径；
本轮不能把该差异外推成跨系统结论。

## 大 Prefill D qualification

| Workload | Batch | cold/层 | KV GiB | Prefill | Cutover | Decode median | E2E | Peak HBM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medium 2048→128 | 8 | 38 | 1.605 | 649.09 | 235.63 | 41.08 | 6234.21 | 30617 |
| RAG 8192→128 | 1 | 38 | 0.763 | 584.27 | 232.31 | 15.00 | 2842.91 | 30587 |
| RAG 8192→128 | 8 | 63 | 6.105 | 3752.32 | 397.12 | 62.91 | 12177.97 | 30829 |

三格均满足：Prefill replay=1、首个 Decode replay=1、全部 measured Decode
replay=1；首次 Prefill step 同时接收完整 B1 或 B8 cohort，不存在 1+7，也没有
mixed Prefill/Decode step。

RAG/B8 最终使用 cold63 和 20,480-token reserve warmup。实际 allocator
`reserved_delta=1,094,713,344 B`（1.020 GiB），hot 恢复后空闲
2,160,852,992 B（2.012 GiB）。8/8 条请求均生成 128 token，无 CUDA
illegal access、Xid、PTE/PDE fault；shutdown session_count=0，结束后 GPU=0
MiB 且恢复 stock UVM。

## 结果边界

- 本轮只有 1 个独立进程块，只提供点估计，不提供 median/p95、bootstrap CI
  或 winner 稳定性。
- 大 Prefill 的 E/F 对称双 Graph arms 尚未获得 runtime contract，因此 D-only
  数据不能和 G/B/C 热力图合并。
- 不同大 Prefill cell 使用不同 cold tier，跨 cell 时间同时包含 workload 和
  offload 几何变化，不能用于单变量推断。
- runtime 热专家全量审计按既定要求关闭；启动、Graph capture、PARK/RESUME、
  cold/hot residency gate 和 shutdown contract 仍启用。

## Artifacts

- 聚合：`artifacts/e3-graph-single-round-b1-b8-summary-20260828-attempt-01/`
- 常规 G/B/C：`artifacts/e3-graph-single-round-b1-b8-tier2-main-20260828-attempt-01/`
- Medium/B1：`artifacts/e3-graph-single-round-b1-b8-no-http-20260828-attempt-01/groups/tier2_medium_b1/`
- Medium/B8 D：`artifacts/e3-graph-medium-b8-D-tier8-protected-20260828-attempt-12/`
- RAG/B1 D：`artifacts/e3-graph-rag-b1-D-tier8-protected-20260828-attempt-02/`
- RAG/B8 D：`artifacts/e3-graph-rag-b8-D-cold63-reserve1g-20260828-attempt-05/`
