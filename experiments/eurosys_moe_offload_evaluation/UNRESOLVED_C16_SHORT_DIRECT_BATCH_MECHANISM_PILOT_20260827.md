# C16 无 HTTP 单批跨机制实验未解决问题

## U1：KTransformers/MoE-Infinity 未通过 exact-generation gate

- KTransformers 只有 5/16 output hashes 与 vLLM all-GPU 相同。
- MoE-Infinity 只有 6/16 相同。
- 输入 token、输出长度、贪心参数和请求顺序均已冻结。
- 影响：两者只能报告完成率和性能，不能宣称语义完全等价。
- 建议：以 HF BF16 为第三方 reference，逐 token 对齐 logits top-k、MoE routing、
  attention backend 和 EOS/generation compatibility。

## U2：本轮不是 equal-HBM，也没有独立重复

- 每个最终 arm 只有一个冷进程。
- Peak HBM 为 31,187--32,109 MiB。
- native 正预算只卸载约 14.25 MiB，ours/KT 固定 cold1，MoE-Infinity 是 global
  demand cache，预算语义不同。
- 影响：不能给出置信区间，也不能作为正式 equal-HBM 排名。
- 建议：先确定统一 HBM accounting，再运行至少 3 个独立冷进程。

## U3：KTransformers 与 MoE-Infinity C16 未满足 SLO

- KTransformers：TTFT 1124.89 ms 通过，但 TPOT 397.90 ms，0/16 SLO。
- MoE-Infinity：TTFT 13037.28 ms、TPOT 1929.60 ms，0/16 SLO。
- 这属于实测性能，不是运行错误。
- MoE-Infinity 当前缺少 PagedAttention/FlashAttention，使用 torch SDPA 和逐序列
  non-paged execution；KTransformers 的 CPU expert 在 C16 Decode 上成为瓶颈。

## U4：MoE-Infinity ratio 0.80 尚无稳定性余量证明

- 单轮通过，NVML peak 为 32,109 MiB，接近 torch 可用显存上界。
- ratio 0.827/0.829 已失败，但尚未二分出修复后最大稳定 ratio。
- 影响：0.80 可作为本轮 qualification 配置，不能称为多进程重复下的稳定上界。
- 建议：正式实验使用独立冷进程对 0.79/0.80 做重复，记录最小瞬时 free HBM；
  若强调最大驻留，再在 0.80 附近做细粒度边界搜索。
