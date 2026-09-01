# 历史兼容性结果：当前权威结果索引

日期：2026-08-31（Asia/Shanghai）

本文件原先记录的是不同模块和旧实验边界，现已被以下同模块、全覆盖双页表单轮结果取代：

```text
RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```

当前可引用的结论只有：

- E2/C16 4/4通过，E2/C1 3/4严格通过；Long/C1不是GPU故障，而是流式协议检查失败；
- E3 10/10场景功能完成；RAG/C16因容量约束使用cold89；
- E10完成cold70→cold80→cold89和KV 2→4→6 GiB；
- 全部 ours 单元命中100%双页表覆盖，无range-map回退、Xid或页表故障；
- 7个有效 E2 配对点中，ours+Graph 相对 eager pinned-UVA remote 快2.08×–8.47×；
- 每个点仅一轮，不能声称统计显著性或拆分单项机制贡献。

详细复现入口：

```text
REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```
