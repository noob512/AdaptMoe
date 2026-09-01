# E3 D 与 C-full-Graph 配对实验：attempt 02

日期：2026-08-31

## 结论

attempt 02 按授权只运行一轮、独立冷进程、fail-fast、不重试。Medium/B8 的 D
成功；C-full-Graph 对照 F 通过 attempt 01 暴露的配置检查后，在 GPU 初始化预热
阶段触发 TMA descriptor 700 和 CUDA illegal-address。脚本立即停止，成功配对仍为
0/3，因此不能计算 D/C 性能比。

这次失败确认了：同样使用 TRITON、相同 cold38、KV、请求以及 Prefill/Decode
Graph 时，当前平台的 always-remote C 路径不能安全运行该大 Prefill 坐标；D 路径
可以运行。它证明的是当前实现的能力差异，不能把 C 的失败换算成无限加速比。

## 已完成结果

| 场景 | arm | Prefill | P→D | steady Decode 中位数 | 请求 E2E | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Medium/B8 | D | 651.50 ms | 23.56 ms | 41.79 ms | 6218.81 ms | ok |
| Medium/B8 | F（C-full-Graph） | — | — | — | — | GPU 初始化失败 |
| RAG/B1 | D/F | — | — | — | — | fail-fast 未运行 |
| RAG/B8 | D/F | — | — | — | — | fail-fast 未运行 |

D 输出哈希为
`eff72948a0cd4ac5cfddda9143c36ac3ed0234861f939cd0955cd366fffb7a71`。
它捕获两张 Graph，测量结束仍为两张，无重捕获；双页表覆盖率 100%，CPU
range-map ioctl 为 0。

本轮 D 与此前同协议 Medium/B8-D 的 Prefill 650.94 ms、P→D 24.43 ms、Decode
42.07 ms、E2E 6232.72 ms 基本一致，差异分别约 +0.09%、-3.53%、-0.65%、
-0.22%。

## F 的明确失败点

F 已经成功完成：

1. 运行配置检查；
2. 7/7 权重分片加载；
3. cold38 远程读取关系建立；
4. 约 1.61 GiB KV 分配；
5. KV 后专家位置整理。

随后在正式请求和 Graph 捕获之前，FlashInfer/TRITON 的 4096-token dummy warmup
进入 MoE 计算，三个 TMA 描述创建均返回错误 700，CUDA context 随后报告 illegal
memory access，进程返回 134。

内核日志没有 Xid、FAULT_PDE、FAULT_PTE、Oops 或 panic。由此可排除：

- D/F 配置泄漏；
- 双页表 PARK/RESUME 失败；
- 专家页表边界被清除；
- 单纯显存不足或热门专家被挤占。

该现象与仓库已记录的 RTX 5090 + TRITON 大单步 remote Prefill 限制一致。换成
FLASH_ATTN 可以绕过部分 TMA 问题，但它不支持所需的 `FULL_SEPARATE`，会降级为
Decode-only Graph，因此不能生成本实验要求的公平 D/C 双 Graph 对照。

## 安全处理

由于 `remote_only + FULL_SEPARATE` 已真实触发 CUDA illegal-address，运行时默认保护
已恢复：该组合继续 fail closed，不能作为已认证功能暴露。attempt 01 的按-arm环境
隔离修复保留，以避免 D 专用 PARK/RESUME 再次泄漏到 F。

失败后 harness 恢复 stock UVM；GPU 0 MiB，无计算进程、UVM client或驱动会话。

## 产物

- 运行目录：`artifacts/e3-d-vs-c-full-graph-paired-20260831-attempt-02/`
- 验收结果：`artifacts/e3-d-vs-c-full-graph-paired-20260831-attempt-02/summary/verification.json`
- D 指标：`artifacts/e3-d-vs-c-full-graph-paired-20260831-attempt-02/summary/metrics.csv`
- F 日志：`artifacts/e3-d-vs-c-full-graph-paired-20260831-attempt-02/groups/medium_b8/custom/b0_medium_context_b8_F/run.log`
