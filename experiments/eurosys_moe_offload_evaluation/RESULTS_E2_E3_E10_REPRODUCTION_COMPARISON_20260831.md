# 历史比较入口：当前 C1/C16 全覆盖双页表结论

日期：2026-08-31（Asia/Shanghai）

本文档原有的 E3 失败和 E10 性能回退来自修复前工作树，已不适用于当前实验。当前比较、
失败边界和 artifact 统一记录在：

```text
RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```

当前功能结果为 E2/C16 4/4、E2/C1 3/4严格通过、E3 10/10场景完成、E10 verifier
`status=ok`。Long/C1 的唯一失败是流事件分组违反严格协议，不能伪装为通过，也不是GPU或
双页表错误。

当前性能结论是：在7个严格验收的 E2 workload/concurrency 配对点，完整 ours+Graph 组合
相对无Graph、始终 pinned-UVA remote 的原生路径获得2.08×–8.47×请求E2E加速。Long/C1
的9.98×只作为观察值，不计入正式范围；E3没有同轮remote arm。该结果为单轮
qualification，不证明各机制的独立贡献或跨模型普适性。

复现方法：

```text
REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```
