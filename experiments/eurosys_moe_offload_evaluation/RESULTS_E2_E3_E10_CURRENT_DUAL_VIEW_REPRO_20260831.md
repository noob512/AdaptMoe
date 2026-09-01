# 历史结果入口：已由 C1/C16 全覆盖双页表结果取代

日期：2026-08-31（Asia/Shanghai）

本文件原有的29请求、24-run矩阵和早期 E10 数值属于旧协议，不再代表当前实现。当前权威
结果见：

```text
RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```

当前状态摘要：

| 实验 | 当前单轮结果 |
|---|---|
| E2/C1 | 3/4严格通过；Long/C1完整输出但流事件协议失败 |
| E2/C16 | 4/4通过 |
| E3/B1、B16 | 10/10场景功能完成，当前档双页表100%覆盖 |
| E10 | 2→4→6 GiB完整完成，verifier `status=ok` |

在7个严格验收的 E2 配对点，ours+Graph 相对 eager pinned-UVA remote 的请求 E2E 加速为
`2.08×–8.47×`。这是完整组合的单轮 qualification 结果，不能拆分为 Graph 或双页表的
单项贡献。

复现命令与验收规则见：

```text
REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```
