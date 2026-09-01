# E3 Prefill-only Graph vs native：未解决问题

## U-1：native 大 Prefill为 N/A

- 影响 cell：Medium/B8、RAG/B1、RAG/B8；
- 边界：单步 scheduled Prefill tokens 分别为 16,384、8,192、65,536；
- 已知错误：RTX 5090/TRITON_ATTN eager 路径在 `>=8192` tokens 已复现 TMA descriptor 700 / CUDA illegal-address；
- 当前处理：fail closed，未再次触发破坏性错误；
- 未采用方案：拆 batch、chunked Prefill、更换 attention backend，因为都会改变用户要求的请求几何或机制；
- 后续建议：修复/升级该 attention kernel 后，在相同 batch、同一个 scheduler step、相同 backend contract 下补测。

## U-2：native 无 exact expert-slice卸载接口

- Graph-copy 使用精确 `(layer, expert)` 坐标；
- native copy-on-forward 只能卸载完整 packed Parameter，本轮使用最接近字节量的 4/14/24 层；
- 影响：两者卸载坐标、总字节和峰值 HBM 不完全相同，不能作为严格 equal-footprint/exact-coordinate 主排名；
- 后续建议：为 native vLLM 实现不依赖 custom PTE 的 expert-slice copy-on-forward，或将结果明确保留为辅助系统基线。

## U-3：只有单轮点估计

- 每个 cell 只有一个冷进程；
- 影响：2%–6% 的小差距和 Long/B1 的 38% 单点差距均没有置信区间；
- 后续建议：在配置冻结后至少运行 5 个独立冷进程，交替 arm 顺序，并报告 paired median、p95 与 bootstrap CI。
