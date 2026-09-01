# E3 D/C 配对实验：attempt 01

日期：2026-08-31

## 目标

沿用原 E3 三个大 Prefill 场景，配对比较：

- D：GPU 缓存 Prefill Graph，切换后使用远程 Decode Graph；
- C：始终远程读取，Prefill eager、Decode Graph。

每个 arm 使用独立冷进程，只运行一次，不重试。

## 结果

本轮 Medium/B8-D 成功，C 在正式请求前的 4096-token TRITON warmup 中发生
CUDA illegal-address。harness 按 fail-fast 停止，RAG/B1 和 RAG/B8 未运行。因此
成功配对为 0/3，不能计算 D/C 加速比。

| 场景 | arm | Prefill | P→D | steady Decode 中位数 | 请求 E2E | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Medium/B8 | D | 653.62 ms | 23.41 ms | 42.09 ms | 6229.11 ms | ok |
| Medium/B8 | C | — | — | — | — | warmup失败，rc=134 |
| RAG/B1 | D/C | — | — | — | — | fail-fast未运行 |
| RAG/B8 | D/C | — | — | — | — | fail-fast未运行 |

D 输出哈希为
`eff72948a0cd4ac5cfddda9143c36ac3ed0234861f939cd0955cd366fffb7a71`，
双页表覆盖率 100%，CPU range-map ioctl 为 0，Graph 无重捕获。

## 配置审计与修复

C 的执行模式本身正确：`FULL_DECODE_ONLY`，Prefill 不使用 Graph，Decode 使用
Graph。但专用 paired launcher 的组级环境仍让 C 继承了 D 的两项启动设置：

- `VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1`；
- `VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1`。

原始 E3 的 C 没有这两个设置。失败形态虽然与历史上 TRITON 大 Prefill 的 TMA/
illegal-address 限制一致，但由于启动路径不完全相同，本轮不能证明原 C 必然失败。

已修复 `run_formal_e3.sh`：C 进程现在显式关闭上述两项，并继续显式关闭 D 专用的
PARK/RESUME 和 Prefill allocator-reserve warmup。修复后尚未重新运行 GPU。

本次没有 Xid、FAULT_PDE、FAULT_PTE、内核 Oops 或 panic。退出后 stock UVM 已
恢复，GPU 0 MiB，无计算进程、UVM client 或驱动会话。

## 产物

- `artifacts/e3-d-vs-c-paired-20260831-attempt-01/`
- `artifacts/e3-d-vs-c-paired-20260831-attempt-01/summary/verification.json`
- `artifacts/e3-d-vs-c-paired-20260831-attempt-01/summary/metrics.csv`
