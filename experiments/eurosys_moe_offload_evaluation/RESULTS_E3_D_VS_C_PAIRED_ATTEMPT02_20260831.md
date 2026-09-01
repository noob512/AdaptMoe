# E3 D 与原始 C 配对实验：attempt 02

日期：2026-08-31

## 目标与协议

本轮按原 E3 请求协议比较：

- D：GPU 缓存 Prefill Graph，阶段切换后使用远程 Decode Graph；
- C：始终远程读取，Prefill eager，Decode Graph；
- 场景：Medium/B8、RAG/B1、RAG/B8；
- 每个 arm 使用独立冷进程；
- 每个 arm 只运行一次，不重试；
- 任一运行失败后立即停止。

attention backend、冷专家计划、KV 大小、请求输入和输出长度均沿用原 E3。

## 结论

attempt 02 没有完成有效的 D/C 配对。Medium/B8-D 成功，随后
Medium/B8-C 在正式请求前的 4096-token 初始化预热中触发 CUDA
illegal-address，进程返回 134。harness 按 fail-fast 停止，因此 RAG/B1 和
RAG/B8 没有运行。

本轮可证明 D 能够正确执行 Medium/B8 的大 Prefill 坐标，但没有得到同协议 C
的性能数据，所以不能计算 D/C 加速比，也不能把 C 的失败解释成“无限加速”。

## 已完成结果

| 场景 | arm | Prefill | P→D | steady Decode 中位数 | 请求 E2E | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Medium/B8 | D | 649.57 ms | 23.30 ms | 42.12 ms | 6232.72 ms | ok |
| Medium/B8 | C | — | — | — | — | warmup 失败，rc=134 |
| RAG/B1 | D/C | — | — | — | — | fail-fast 未运行 |
| RAG/B8 | D/C | — | — | — | — | fail-fast 未运行 |

D 的输出 token 哈希为
`eff72948a0cd4ac5cfddda9143c36ac3ed0234861f939cd0955cd366fffb7a71`。
双页表覆盖率为 100%，捕获前后均为两张 Graph，没有重捕获。

D 与 attempt 01 的同一坐标基本一致：Prefill、P→D、steady Decode 和 E2E
分别变化约 -0.62%、-0.47%、+0.06% 和 +0.06%。

## C 配置审计

attempt 01 中发现 C 继承了 D 的部分启动设置；attempt 02 已完成按-arm隔离。
本轮 C 的实际配置为：

- `execution_mode=full_graph_decode`；
- `graph_prefill=false`；
- `graph_decode=true`；
- `phase_mode=remote_only`；
- Graph capture size 仅为 `[8]`；
- `VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=0`；
- `VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0`；
- `VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0`；
- `VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=0`。

因此本轮运行的确是用户要求的原始 C：远程 eager Prefill 加远程 Decode Graph，
不是 C-full-Graph，也没有启用 D 的阶段切换。

## C 的失败位置

C 已成功完成：

1. 7/7 权重分片加载；
2. cold38 远程读取布局建立；
3. 约 1.61 GiB KV 分配；
4. KV 分配后的专家位置整理；
5. 确认约 8.02 GiB 冷专家远程、约 18.98 GiB 热专家位于 GPU。

随后，初始化代码把 FlashInfer/TRITON dummy warmup 限制为 4096 tokens。在这次
预热期间或其后的同步点，CUDA context 报告 illegal memory access，进程返回
134，尚未进入正式请求计时。

该形态与此前记录的 RTX 5090、TRITON、大单步 remote Prefill 限制一致。不过
CUDA 错误可能异步上报，本轮日志不足以把故障严格归到某一个具体 kernel。

可以排除的是：

- C 误用了 D 的双页表或阶段切换；
- C 误启用了 Prefill Graph；
- 显存不足；
- 驱动 Xid、FAULT_PDE、FAULT_PTE、Oops 或 panic。

## 验收与清理

verifier 状态为 `error`：

- 预期运行：6；
- 成功运行：1；
- 成功配对：0/3；
- 重试：0。

失败后 harness 已恢复 stock UVM。当前模块 SHA256 为
`9e77e3c50fcc40bbc64b50128ff1d05a1775fa75deb71976c1adeffbf278831f`，
GPU 占用 0 MiB，无计算进程或 UVM client。

## 产物

- 运行入口：`run_e3_d_vs_c_paired.sh`
- 运行目录：`artifacts/e3-d-vs-c-paired-20260831-attempt-02/`
- 验收结果：`artifacts/e3-d-vs-c-paired-20260831-attempt-02/summary/verification.json`
- D 指标：`artifacts/e3-d-vs-c-paired-20260831-attempt-02/summary/metrics.csv`
- C 配置：`artifacts/e3-d-vs-c-paired-20260831-attempt-02/groups/medium_b8/custom/b0_medium_context_b8_C/result.json`
- C 日志：`artifacts/e3-d-vs-c-paired-20260831-attempt-02/groups/medium_b8/custom/b0_medium_context_b8_C/run.log`
