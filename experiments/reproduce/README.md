# EuroSys MoE实验复现文档统一入口

最后核验：2026-09-02（Asia/Shanghai）

本目录是复现文档的唯一查看和修改入口。结果、runner、冻结输入和artifact仍保留在原实验
目录；这里不复制大型artifact。下表中的“当前”指最新可合法引用的证据，不代表所有实验都
已达到论文级完成状态。

## E0--E13当前入口

| 实验 | 当前复现文档 | 最新可用证据与状态 |
| --- | --- | --- |
| E0 | [REPRODUCE_E0_E1_CURRENT.md](REPRODUCE_E0_E1_CURRENT.md) | `eurosys-pa-v2-20260824` compatibility qualification通过；仍无三系统单命令重跑器 |
| E1 | [REPRODUCE_E0_E1_CURRENT.md](REPRODUCE_E0_E1_CURRENT.md) | 15/15 process blocks完成；跨系统输出门仍为held，只能作描述性anchor |
| E2 | [REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md) | 2026-09-01最新ours：C1 4/4、C16 4/4运行通过；跨臂仍复用8月31日pinned-UVA基线，非同轮统计 |
| E3主路径 | [REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md) | `e3-c1-c16-full-dual-20260901-reproduce-all-attempt-02`，B1/B16共10/10通过 |
| E3三机制 | [REPRODUCE_E3_PREFILL_DECODE_THREE_MECHANISM_B1_B8.md](REPRODUCE_E3_PREFILL_DECODE_THREE_MECHANISM_B1_B8.md) | 2026-09-01/02分臂复验；当前档Decode汇总10/10 cell、24/24 run通过 |
| E4 | [REPRODUCE_E4_CURRENT.md](REPRODUCE_E4_CURRENT.md) | `e4-repro-20260902-attempt-01`，5进程C1--C32通过；未观察到预注册TPOT退化点 |
| E5 | [REPRODUCE_E5_CURRENT.md](REPRODUCE_E5_CURRENT.md) | 2026-09-02 stable/non-stable各10冷进程及派生比较通过；其他消融仍为历史模块证据 |
| E6 | [REPRODUCE_E6_PLAN_SWITCH_QUALIFICATION.md](REPRODUCE_E6_PLAN_SWITCH_QUALIFICATION.md) | 2026-09-02三臂`status=ok`；最新双页表结果已取代8月28日点估计 |
| E7 | [REPRODUCE_E7_INDEPENDENT_COLD_QUALIFICATION.md](REPRODUCE_E7_INDEPENDENT_COLD_QUALIFICATION.md) | `attempt-25-targeted-canonical-pte`六臂通过；维护计入后Adaptive cycle throughput下降40.35% |
| E8 | [REPRODUCE_E8_CURRENT.md](REPRODUCE_E8_CURRENT.md) | provenance门阻塞；当前preflight仍因缺fresh build receipt返回1，无性能运行 |
| E9 | [REPRODUCE_E9_CURRENT.md](REPRODUCE_E9_CURRENT.md) | 7/7诊断case完成；correctness verifier未通过，无正式性能结论 |
| E10 | [REPRODUCE_E10_CURRENT_TIER_DUAL_VIEW_FULL_RUN.md](REPRODUCE_E10_CURRENT_TIER_DUAL_VIEW_FULL_RUN.md) | `e10-kv-2-4-6-pressure-20260902-attempt-01-scheduler-quiescent-sync-tier`通过 |
| E11 | [REPRODUCE_E11_CONTINUOUS_ARRIVAL_QUALIFICATION.md](REPRODUCE_E11_CONTINUOUS_ARRIVAL_QUALIFICATION.md) | 已按用户要求终止；9月2日完整attempt仍为`status=error`，当前只允许离线审计 |
| E12 | [REPRODUCE_E12_CURRENT.md](REPRODUCE_E12_CURRENT.md) | 1,000 Graph replay与1,000 PARK/RESUME通过；不是6/24小时整模型soak |
| E13总体 | [REPRODUCE_E13_KV_ADAPTIVE.md](REPRODUCE_E13_KV_ADAPTIVE.md) | 在线同步约2 GB切档、动态/容量和阶段诊断总入口 |
| E13双Graph回环 | [REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md](REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md) | 2026-09-02 attempt-02：3/3冷进程、84/84门通过，核心稳态24/24在±10%内 |

## 补充复现文档

- [Qwen3 E2自适应并发](REPRODUCE_QWEN3_E2_ADAPTIVE.md)：2026-08-27成功参考，尚未在9月2日模块重新运行。
- [Qwen3 ours/KTransformers原结果复现](REPRODUCTION_QWEN3.md)：历史机制复现，不替代当前E3。
- E10也可从E2/E3/E10联合协议进入，但最新结果和推荐命令以E10专用文档为准。

## 模块代际边界

最新成功artifact横跨多个custom UVM二进制：E2/E3/E4使用`68A3.../6965...`，E7/E10
使用`18E6.../90fb...`，E13最终复现使用`D158.../4168...`。必须以各文档记录的artifact
指纹解释旧结果；用当前源码新跑时必须保存新的指纹并重新验收，不能把不同模块的臂拼成
同一统计实验。

## 使用规则

1. 所有GPU输出目录必须使用不存在的新attempt目录，禁止覆盖成功或失败artifact。
2. 开始前确认GPU 0 MiB、无compute/UVM client，并满足各文档规定的stock/custom UVM状态。
3. 单轮qualification只能报告点估计，不能写成median、p95或置信区间。
4. verifier、输出hash、Graph/VA/PTE/residency、fault和cleanup门必须同时满足。
5. 阻塞或失败实验只能复核已有证据；恢复GPU运行必须另行明确授权。
6. 文档与旧结果冲突时，以本目录的“最新可用证据”段和统一汇总为准。
