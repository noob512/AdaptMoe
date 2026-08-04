# DeepSeek-V2-Lite-Chat 等显存并行性能对比

实验日期：2026-08-04

## 实验配置

- GPU：NVIDIA GeForce RTX 5090，32607 MiB
- 模型：`deepseek-ai/DeepSeek-V2-Lite-Chat`，BF16
- 数据集：当前 ShareGPT fixture；四份 fixture 的 SHA-256 均为 `33958542f9f732b8a66d595ef78923d53051f3d7efcf8ab1ece1adaa1e03328e`
- 请求：64 个请求一次性并行提交
- Prompt：原 fixture 为 320–360 tokens；MoE-Infinity 加特殊 token 后为 323–358 tokens
- 输出：每请求固定 32 tokens，总计 2048 output tokens
- 测量状态：每次均为独立冷进程加载后的第一个 64 请求静态批次
- 当前机制：vLLM/UVM adaptive，`gpu_memory_utilization=0.97`
- MoE-Infinity：batched `MoE.generate()`，`batch_size=64`、`device_memory_ratio=0.55`、启用 prefetch
- 重复次数：双方各 3 次，生成时间不包含模型加载、tokenization 和显存快照
- 原始结果目录：`/tmp/moe_equal_vram_20260804_212047`

## 实验结果

| 机制 | 三次吞吐 (tok/s) | 平均吞吐 (tok/s) | 样本标准差 | 平均生成时间 (s) | 平均峰值显存 (MiB) |
|---|---|---:|---:|---:|---:|
| 当前 vLLM/UVM | 235.0069 / 264.1184 / 225.4172 | **241.5142** | 20.1545 | **8.5180** | **32110.0** |
| MoE-Infinity | 53.9312 / 50.1763 / 52.2807 | **52.1294** | 1.8820 | **39.3212** | **31910.3** |

## 性能差异

- 当前 vLLM/UVM 比 MoE-Infinity 多 `189.3848 tok/s`。
- 当前 vLLM/UVM 的平均吞吐是 MoE-Infinity 的 `4.6330x`，即相对 MoE-Infinity 高 `363.30%`。
- MoE-Infinity 的平均生成时间是当前 vLLM/UVM 的 `4.6162x`。
- 两者平均峰值显存仅差 `199.7 MiB`，占 vLLM 峰值的 `0.622%`；满足等显存对比要求。
- MoE-Infinity 三次吞吐变异系数为 `3.61%`；vLLM 为 `8.35%`。

## MoE-Infinity 并行能力

MoE-Infinity 可以并行。本实验没有再按请求顺序执行，而是将 64 个请求组成一个张量批次，一次调用 `MoE.generate()`，实际完成 `64/64` 请求和 `2048/2048` output tokens。MoE-Infinity 也提供 `MoE.serve()` continuous batching 接口，但当前 DeepSeek-V2-Lite-Chat 环境缺少对应的 paged-attention CUDA 路径；为避免非 paged continuous engine 的跨 decode-step KV 状态影响正确性，本轮采用 HuggingFace 兼容静态批处理路径。

## 可信度判断

- **结论可信度较高**：同一 GPU、同一模型、同一 prompt 文本、同一请求数、同一输出长度、同为 64 路静态批处理，且平均峰值显存差低于 1%。
- **吞吐差距稳定**：3 次 MoE-Infinity 均为 50.18–53.93 tok/s；3 次 vLLM/UVM 均为 225.42–264.12 tok/s，区间完全分离。
- **绝对数值可信度中等**：每侧只有 3 次重复，并且测量的是独立冷进程的第一个批次，仍包含首次内核/缓存状态的波动。
- **不能外推为所有服务负载**：本结果只覆盖 64 请求、约 340-token prompt、32-token 输出和静态批处理；连续到达、不同并发度和长输出需要另做 steady-state online benchmark。
- vLLM/UVM 之前经过低/中并发预热后的 64 请求 high phase 为 `297.6415 tok/s`；该值不是本次冷批次主对比值，但表明稳态吞吐还可能更高。

## 校准记录

- MoE-Infinity 的 `device_memory_ratio=0.78`、`0.70` 和 `0.65` 均在 64 路批处理中逐步占满约 32103–32107 MiB 后 OOM。
- `device_memory_ratio=0.55` 可稳定完成，三次峰值为 `31929 / 31929 / 31873 MiB`。
- 源码中的专家缓存预算不覆盖全部批量激活开销，且自定义 CUDA caching allocator 会保留释放的块，因此不能直接把 `device_memory_ratio` 当作最终进程峰值比例。

## 兼容性修复

MoE-Infinity 的专家 CUDA 路径原先将单专家输入硬限制为 256 tokens。实验版本将超过 256 tokens 的单专家输入分块执行，并在同一 CUDA stream 写回输出，使 DeepSeek-V2-Lite-Chat 的 64 路批处理可以完成。

## 结果文件

- 仓库摘要：`experiments/moe_infinity_compare/comparison_equal_vram_20260804.json`
- 原始统一摘要：`/tmp/moe_equal_vram_20260804_212047/comparison_summary.json`
- MoE-Infinity 三次结果：`/tmp/moe_equal_vram_20260804_212047/moe_calibration_ratio055/result.json`、`moe_trial2_ratio055/result.json`、`moe_trial3_ratio055/result.json`
- vLLM/UVM 三次结果：`/tmp/moe_equal_vram_20260804_212047/vllm_trial1_cold64/adaptive/result.json`、`vllm_trial2_cold64/adaptive/result.json`、`vllm_trial3_cold64/adaptive/result.json`
