# E3 D 与 C-full-Graph 配对实验：attempt 01

日期：2026-08-31

## 结论

本轮按单轮、独立冷进程、fail-fast、不重试执行。Medium/B8 的 D 成功，随后
C-full-Graph 对照 F 在模型加载前被配置检查拒绝，脚本立即停止。成功配对为
0/3，因此本轮不能给出 D/C 性能结论。

F 是原 C 的严格对称版本：冷专家在 Prefill 和 Decode 都始终由 CPU 远程读取，
但执行模式与 D 相同，均为 `FULL_SEPARATE`，Prefill 和 Decode 均使用 CUDA
Graph。这样比较不会把原 C 的 eager Prefill 差异混入结果。

## 已完成结果

| 场景 | arm | Prefill | P→D | steady Decode 中位数 | 请求 E2E | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Medium/B8 | D | 852.27 ms | 31.55 ms | 41.96 ms | 6452.07 ms | ok |
| Medium/B8 | F（C-full-Graph） | — | — | — | — | 启动前配置失败 |
| RAG/B1 | D/F | — | — | — | — | fail-fast 未运行 |
| RAG/B8 | D/F | — | — | — | — | fail-fast 未运行 |

D 的输出哈希为
`eff72948a0cd4ac5cfddda9143c36ac3ed0234861f939cd0955cd366fffb7a71`，
捕获两张 Graph，测量结束仍为两张，无重捕获；双页表覆盖率为 100%，远程
range-map ioctl 为 0。

## 失败原因

launcher 把只属于 D 的
`VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1` 作为组级环境传给了 F。
F 从加载开始就始终远程读取，没有需要暂停或恢复的 Prefill GPU 缓存目录，
因此运行时在加载权重前按设计拒绝了矛盾配置：

```text
VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 requires an
alias-prefill/remote-decode phase mode
```

这不是显存不足、Graph 失败、GPU 页表故障或模型输出错误。没有出现 Xid、
FAULT_PDE、FAULT_PTE 或 CUDA illegal address；进程退出后 stock UVM 已恢复，
GPU 占用为 0 MiB。

## 已完成修复

- 允许安全的 `remote_only + FULL_SEPARATE` 组合；远程页表在两个阶段保持不变。
- D 继续启用 PARK/RESUME；F 在每个独立进程中强制关闭该开关。
- F 强制关闭仅服务于 D 的额外 Prefill allocator-reserve warmup；F 的 Prefill
  Graph 捕获本身已经建立对应内存池。
- 新增专用 launcher 和 verifier。verifier 要求 6/6 run、3/3 配对、双方
  Prefill/Decode Graph 重放、输出哈希一致、无重捕获、D 双页表覆盖 100%、
  F 无阶段切换、无故障且最终资源清零。

修复后的静态检查和两个针对性单元测试均通过，但遵守本轮不重试约束，没有
再次启动 GPU。取得正式性能对比需要新授权运行 attempt 02。

## 产物

- 运行目录：`artifacts/e3-d-vs-c-full-graph-paired-20260831-attempt-01/`
- launcher：`run_e3_d_vs_c_full_graph_paired.sh`
- verifier：`verify_e3_d_vs_c_full_graph_paired.py`
- 验收摘要：`artifacts/e3-d-vs-c-full-graph-paired-20260831-attempt-01/summary/verification.json`
