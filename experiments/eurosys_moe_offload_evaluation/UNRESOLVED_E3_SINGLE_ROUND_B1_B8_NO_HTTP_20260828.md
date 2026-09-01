# E3 B1/B8 无 HTTP：未解决问题（2026-08-28）

## U-E3-001：大单步 Prefill 的 custom cell 未运行

- 影响：Medium/B8、RAG/B1、RAG/B8 的 H/A/R，共 9 个 run；
- 条件：`prompt_tokens * batch >= 8192`；
- 已知历史故障：TMA descriptor 700 / CUDA illegal-address；
- 本轮处理：按预注册规则 fail-closed，未静默切换 eager；
- 建议：单独修复/验证该 attention backend 边界后，再补测这 3 个逻辑 cell。

## U-E3-002：KTransformers RAG/B8 超出冻结 KV headroom

- 计算 KV：6,555,697,152 B；
- 冻结 KT 上限：2,800,000,000 B；
- 本轮没有改变 B8，仅将该 cell 视为容量 N/A；
- 建议：若要运行，需要预先定义新的 equal-HBM/expert-placement tier，不能在本轮临时减少 batch。

## U-E3-003：KTransformers 与 custom exact-generation 不一致

- 9 个 KT cell 均完整返回请求，但所有可配对 cell 的输出 token hash 与 H/A/R 不同；
- H/A/R 内部 hash 全部一致；
- 影响：KT 指标不能用于严格的跨框架性能排名；
- 建议：逐项对齐 FP8 loader、logits、greedy sampling、tokenizer/input ID 和专家 placement 后重跑 exact-generation gate。

## U-E3-004：统计重复不足

- 当前 `PROCESS_BLOCKS=1`；
- verifier 明确记录 `publication_repetition_requirement_met=false`；
- 影响：winner 只是点估计，无 median/p95、paired bootstrap CI 或稳定性结论；
- 建议：路径问题解决后再执行文档要求的独立冷进程重复。

## U-E3-005：原生 vLLM exact-coordinate 对照缺失

原生 vLLM UVA/copy 只能按完整 Parameter tensor 卸载，没有 expert-slice 计划接口，无法复现冻结的 480 个坐标，因此本轮为 N/A。

## U-E3-006：本轮 H/A/R 未覆盖 CUDA Graph 模式

- launcher 传入了候选 capture sizes 1、4、8；
- 但 H/A/R 的 runner contract 均为 `graph=false`，实际为 `enforce_eager=true`、
  `cudagraph_mode=NONE`；
- 影响：本轮内部 H/A/R 对比仍保持同为 eager 的公平性，但不能与 Graph-enabled E2
  结果直接比较；
- 建议：补充 graph-enabled hybrid、always-copy、always-remote 三个对称 arm 后重跑。
