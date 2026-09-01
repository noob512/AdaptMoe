# C1 跨机制实验未解决问题

本文只列出 2026-08-27 C1 runtime-hot-audit-off matrix 结束后仍未解决、或阻止将
本轮直接用作论文最终排名的问题。

## U1：ours RAG 的 Prefill CUDA Graph 未通过 exact-generation

- 复现：ours cold1 和 vLLM all-GPU 均使用同一个 8192/128 请求及 2K x 4 Prefill；
  ours 为 FULL_SEPARATE Graph，all-GPU 为 eager。
- 现象：两者均完成 128 tokens，但 output-token hash 不同；最终配对复测首次分叉为
  token index 5。
- 原生 hash：`d471ac81b4a0631724f49fd8bb60be5f0f19930960ba2aee1ec795adb9e9ef6b`。
- ours hash：`db6f930012b7b396cc7484b5c06706fa4134cc8826c7f5ff045440be390e968c`。
- 已完成的最小差分：ours eager 的 hash 为 `d471ac81...`，与 all-GPU eager 一致；
  ours `FULL_DECODE_ONLY`（Prefill eager、Decode-C1 Graph）的 hash 也为
  `d471ac81...`。两组仍使用 cold1、alias、PARK/RESUME、remote expert 和 2K x 4。
- 当前判断：Decode Graph、alias、PARK/RESUME、remote expert 和 chunked-prefill
  本身已被排除；分叉收敛到 Prefill CUDA Graph 执行/KV 写入重放路径。尚未定位到
  具体 kernel、静态 buffer、slot mapping 或 KV 页。
- 影响：ours RAG 延迟不能作为已通过语义等价 gate 的正式 speedup。
- 建议：下一步只比较 all-GPU Prefill Graph 与 ours Prefill Graph，并在第 1–4 个
  Prefill chunk 后校验 KV cache；随后记录输出 token 0–5 的 logits top-k/ULP，区分
  普通 Graph 数值差异与 ours 的 Prefill 静态输入/slot replay 错误。
- 证据：`artifacts/c1-rag-exact-generation-eager-diagnostic-20260827-attempt-03/` 和
  `artifacts/c1-rag-exact-generation-decode-graph-diagnostic-20260827-attempt-04/`。

## U2：KTransformers 和 MoE-Infinity 跨框架输出均不一致

- KTransformers：4/4 hash 与 vLLM baseline 不同。
- MoE-Infinity：4/4 hash 与 vLLM baseline 不同。
- 已确认：请求 token 数完整，输入 fixture、seed 和贪心生成设置冻结。
- 未确认：跨框架 attention/MoE kernel、EOS、数值精度和 generation compatibility
  的哪一项导致分叉。
- 影响：当前只能宣称请求成功和性能指标可采集，不能宣称生成完全等价。
- 建议：以 Hugging Face BF16 reference 为第三方基线，逐 token 对齐 logits 和路由。

## U3：MoE-Infinity 四类请求均未通过 SLO

- Short/Ordinary/RAG/Long TPOT：327.16/315.44/366.22/338.67 ms，均高于 100 ms。
- RAG TTFT 2175.97 ms，也高于 2000 ms。
- 这是实测性能结果，不是未处理的运行错误。
- 影响：MoE-Infinity 请求完成 qualification 通过，但 goodput/SLO qualification 失败。
- 建议：在保持相同请求和 HBM gate 下分析 expert fetch hit rate、PCIe traffic 和 CPU
  I/O；不得通过缩短输出或 warmup 回避。

## U4：本轮不是 equal-HBM，也没有重复

- 单次冷进程，无方差和置信区间；
- peak HBM 为 30,799–32,047 MiB，且不同机制的静态预算语义不同；
- 原生 vLLM 只卸载约 14.25 MiB，ours/KT 为逐层 cold experts，MoE-Infinity 为全局
  demand cache；
- RAG 原生与 ours 使用 2K x 4；KTransformers/MoE-Infinity 的内部调度仍不同。

影响：本轮适合报告可行性、容量边界和诊断开销消除，不适合论文最终机制排名。
建议下一阶段先做 equal-HBM/同执行 shape 配平，再做至少 3 个独立冷进程重复；根据
用户指示，当前已暂停，不启动这些实验。

## U5：Decode-only 诊断出现一次不可复现的 C++ 启动 abort

- 复现 attempt：`c1-rag-exact-generation-decode-graph-diagnostic-20260827-attempt-02`。
- 现象：EngineCore 在权重入口、UVM allocator 动态库加载后立即输出
  `pure virtual method called` 和 `terminate called without an active exception`；
  尚未进入权重 shard 加载、Graph capture 或正式请求。
- 排除项：不是正式请求期 CUDA/PTE 故障；在确认 GPU 干净、恢复 stock UVM 后，
  相同配置的 attempt-04 完整成功，未再次出现。
- 根因：未解决。当前缺少 native backtrace/core dump，不能把一次成功重试解释成修复。
- 影响：不影响 attempt-04 的输出定位结论，但说明该诊断启动仍存在低频稳定性风险。
- 建议：若再次出现，设置 `ulimit -c unlimited`，在同一动态库和模块指纹下收集 core、
  `gdb thread apply all bt full` 及 loader 日志；不要把它合并为 Graph 正确性故障。
