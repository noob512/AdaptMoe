# EuroSys MoE实验复现文档统一入口

本目录自2026-09-01起是复现文档的唯一查看和修改入口。后续新增、修订或引用复现步骤时，
只修改本目录文件；`experiments/eurosys_moe_offload_evaluation/`及其他实验目录中的同名
文档视为来源快照或历史材料，不再作为编辑入口。

结果、runner、冻结输入和artifact仍保留在原目录。本目录不复制大型artifact。

## E0--E13当前入口

| 实验 | 当前复现文档 | 证据状态 |
| --- | --- | --- |
| E0 | [REPRODUCE_E0_E1_CURRENT.md](REPRODUCE_E0_E1_CURRENT.md) | compatibility qualification通过；缺三系统单命令重跑器 |
| E1 | [REPRODUCE_E0_E1_CURRENT.md](REPRODUCE_E0_E1_CURRENT.md) | 15 blocks执行完成；publication gate held |
| E2 | [REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md) | 当前C1/C16权威单轮协议 |
| E3主路径 | [REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md) | 当前B1/B16 D/ours协议 |
| E3三机制 | [REPRODUCE_E3_PREFILL_DECODE_THREE_MECHANISM_B1_B8.md](REPRODUCE_E3_PREFILL_DECODE_THREE_MECHANISM_B1_B8.md) | 2026-08-31 B1/B8完整矩阵；Decode 10/10补齐 |
| E4 | [REPRODUCE_E4_CURRENT.md](REPRODUCE_E4_CURRENT.md) | C1--C32、5进程qualification通过 |
| E5 | [REPRODUCE_E5_CURRENT.md](REPRODUCE_E5_CURRENT.md) | 当前模块stable/non-stable两臂通过；其他消融待同模块复验 |
| E6 | [REPRODUCE_E6_PLAN_SWITCH_QUALIFICATION.md](REPRODUCE_E6_PLAN_SWITCH_QUALIFICATION.md) | 三臂单样本qualification通过 |
| E7 | [REPRODUCE_E7_INDEPENDENT_COLD_QUALIFICATION.md](REPRODUCE_E7_INDEPENDENT_COLD_QUALIFICATION.md) | 最新六臂独立冷进程qualification |
| E8 | [REPRODUCE_E8_CURRENT.md](REPRODUCE_E8_CURRENT.md) | provenance门阻塞；无性能运行 |
| E9 | [REPRODUCE_E9_CURRENT.md](REPRODUCE_E9_CURRENT.md) | 诊断运行完成；correctness verifier未通过 |
| E10 | [REPRODUCE_E10_CURRENT_TIER_DUAL_VIEW_FULL_RUN.md](REPRODUCE_E10_CURRENT_TIER_DUAL_VIEW_FULL_RUN.md) | 当前2→4→6 GiB压力切档qualification |
| E11 | [REPRODUCE_E11_CONTINUOUS_ARRIVAL_QUALIFICATION.md](REPRODUCE_E11_CONTINUOUS_ARRIVAL_QUALIFICATION.md) | 调度行为完成；最终verifier为error |
| E12 | [REPRODUCE_E12_CURRENT.md](REPRODUCE_E12_CURRENT.md) | 1,000 replay与1,000 PARK/RESUME通过；非长时soak |
| E13 | [REPRODUCE_E13_KV_ADAPTIVE.md](REPRODUCE_E13_KV_ADAPTIVE.md) | 动态/容量各12/12；resident上限7/2/5/7，Adaptive吞吐仍回退 |

## 补充复现文档

- [Qwen3 E2自适应并发](REPRODUCE_QWEN3_E2_ADAPTIVE.md)
- [Qwen3 ours/KTransformers原结果复现](REPRODUCTION_QWEN3.md)
- [E2/E3/E10联合协议中的E10入口](REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md)

## 使用规则

1. 所有GPU输出目录必须使用不存在的新attempt目录，禁止覆盖成功或失败artifact。
2. 开始前确认GPU 0 MiB、无compute/UVM client，并满足各文档规定的stock/custom UVM状态。
3. 单轮qualification只能报告点估计，不能写成median、p95或置信区间。
4. verifier、输出hash、Graph/VA/PTE/residency、fault和cleanup门必须同时满足。
5. 文档与旧结果冲突时，以本目录文档和其明确标注的当前结果文件为准。
