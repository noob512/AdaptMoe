# E3 CUDA Graph 单轮未解决问题（2026-08-28）

## 1. 大 Prefill 缺少对称三臂双 Graph 比较

Medium/B8 和 RAG/B1/B8 目前只通过 D：alias Prefill Graph → remote Decode
Graph。E/F 分别用于 always-alias 与 always-remote 的双 Graph 对照，但
`FULL_SEPARATE` runtime 目前只认证 D。F 定向尝试 fail-closed：

```text
FULL_SEPARATE is reserved for prefill_graph_remote_decode
```

Artifact：`artifacts/e3-graph-medium-b8-F-tier8-protected-20260828-attempt-13/`。
不能简单删除断言，因为 Graph topology certificate、alias Graph selective
teardown 和剩余 remote Graph 的生命周期只对 D 验证过。后续应分别实现并测试
E/F capture certificate，再补大 Prefill 三臂矩阵。

## 2. G/C 的单步大 Prefill 路径仍会 illegal access

Medium/B8 的 `2048×8=16384` 单步 Prefill 在 G/C 路径中，即使增加 cold
experts、限制 MoE chunk/FlashInfer autotune、预留 KV hole 并执行
reconciliation，仍出现 TMA descriptor 700/CUDA illegal-address。代表性 artifacts：

- `artifacts/e3-graph-medium-b8-tier8-protected-20260828-attempt-11/`
- `artifacts/e3-graph-medium-b8-C-tier4-chunk4096-hole-20260828-attempt-05/`

D 的双 Graph 保护链可运行，但没有证明 G/C 的根因已修复。建议用最小 8K/16K
shape 二分、捕获前后 allocator snapshot 和 compute-sanitizer 定位首个非法 kernel。

## 3. 只有单轮 qualification

用户要求本轮只运行一次，因此尚缺正式 E3 的独立冷进程重复、轮换 arm 顺序、
median/p95、paired bootstrap CI 和 break-even 置信范围。当前 winner 只表示本次
点估计。

## 4. RAG/B8 的 1 GiB reserve 尚缺重复稳定性

20,480-token warmup 实际保留 1.020 GiB，并在单轮中完整通过。尚未证明不同
allocator 初始状态下都不会在首次 65K Prefill replay 发生新分配或 hot eviction。
后续至少做 3 个独立冷进程，并记录 Prefill 前后 cudaMalloc、reserved/allocated
和 hot residency。

## 5. Graph pool 的精确物理分项未知

现有 NVML 给出总峰值，但 Graph pool 内已经包含捕获期 attention/MoE
workspace；不能将二者机械相加。需要捕获前后 PyTorch memory snapshot、Graph
pool allocation tag 和 UVM residency snapshot 才能精确拆分。
