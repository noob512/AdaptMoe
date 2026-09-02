# E0--E13 最新实验结果统一汇总

日期：2026-08-31--2026-09-02（Asia/Shanghai）
最后更新：2026-09-02（Asia/Shanghai）

## 1. 汇总规则

本文按 E0、E1、E2……E13 的编号顺序汇总截至 2026-09-02 已取得的最新可用结果。

- E2、E3、E10 使用最新 C1/C16、全覆盖双页表单轮结果；E2 的较早 C1/C16 六机制以及
  E3 的较早同坐标 H/A/R、G/B/C 和 native copy/UVA qualification 另表保留，只作辅助
  上下文，不与最新主表混成同一轮；
- E4、E5、E6、E7和E10使用2026-09-02最近一次成功复验；
- E5 使用当前模块的 stable/non-stable VA 两臂十重复，以及较早模块的其余机制消融；
- E12 同时保留较早的 1,000-replay smoke 和当前模块的 1,000-cycle PARK/RESUME；
- E13 使用按当前复现文档重跑且verifier通过的双阶段CUDA Graph K2/K6/Adaptive回环
  `e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02`；
- E6、E7、E11 使用各自后续专项 qualification，而不是 2026-08-27 的早期状态；
- E8和E9没有新的通过结果，保留最近已有证据；E11已按用户要求终止，只保留离线审计；
- 失败、阻塞和仅 smoke 的实验明确保留原状态，不写成通过；
- 不把不同日期、模块或协议的 artifact 拼成同一轮统计实验。

当前成功artifact横跨三个主要custom UVM代际，不能直接混算跨实验性能变化：

```text
E2/E3/E4  srcversion 68A3CA3F7E2D57082767C59
           SHA256 6965f8ee1a22029c5f83d602093318b3cca5241fa6c2ecb3839552d08e974547
E7/E10     srcversion 18E6563D21D9D2937D99EDF
           SHA256 90fb5a7d9852a754ecf0358700d36dbcd04abd19886753ee5486401a937e5e15
E13        srcversion D158DB985980F4A580DB740
           SHA256 4168020f14fc3217f8d217e4fa8898299e4d81c0c18653c98cce9d73346afdbe
```

E5、E6、E11、E12及历史消融仍以各自artifact中的模块和环境为准。相同日期也不代表
相同进程、协议或统计重复。

## 2. 总体状态

| 实验 | 最新状态 | 当前可支持的核心结论 | 证据边界 |
|---|---|---|---|
| E0 | 基础 qualification 通过 | 三系统可执行同一冻结请求并得到相同32-token hash | 不是性能排名 |
| E1 | 执行完成，publication gate 暂停 | 15/15 process blocks 完成 | 跨系统完整输出不一致，不能发布speedup |
| E2 | ours C1/C16共8/8运行通过；跨臂exact-token为6点 | 6个同时通过流式与exact-token门的点上，ours+Graph比旧pinned-UVA基线快2.09×–11.56× | 跨轮单样本、非equal-HBM；RAG两点跨臂hash不同 |
| E3 | 最新D臂B1/B16共10/10通过；阶段三机制10/10 cell、24/24 run通过 | 全覆盖双页表与双Graph正确；phase-only Decode custom在10点均快于native copy/UVA | 阶段表混合FLASH/TRITON且非equal-HBM；不是纯Graph消融 |
| E4 | C1--C32各5个process block复验通过 | 未观察到预注册TPOT持续退化；峰值PCIe RX为4.698 GiB/s | 只覆盖固定remote计划 |
| E5 | stable/non-stable各10冷进程通过 | stable-VA rebuild比non-stable reallocate/recapture快1.369×--1.522× | 其余Graph/PTE消融仍来自较早模块 |
| E6 | 三臂单轮qualification通过 | batched difference明显快于full rebuild和多wave串行切换 | 无统计重复 |
| E7 | 六臂独立冷进程qualification通过 | 自适应降低remote assignment，但同步维护使cycle吞吐下降40.35% | 每arm仅一次 |
| E8 | 阻塞 | provenance门正确阻止无fresh receipt运行 | 无性能数据 |
| E9 | 未通过总体验收 | 7/7 case可运行 | greedy token漂移，性能只能诊断使用 |
| E10 | scheduler-boundary压力序列单轮qualification通过 | 完成2→4→6 GiB KV/专家同步切档，三档双页表100%覆盖 | 不是自动联合控制器formal；切档仍需543--565 ms |
| E11 | verifier `status=error`，已终止 | phase epoch能避免mixed step，hybrid fail-closed有效 | phase-specific与remote-only输出不一致；只允许离线审计 |
| E12 | 1,000-replay smoke和1,000-cycle PARK/RESUME通过 | Graph回放、完整双页表往返和生命周期清理通过 | 不是6小时整模型soak或故障注入 |
| E13 | 双阶段Graph三臂单轮qualification通过 | Fixed-K2在Short B64/B128优于K6；Adaptive切至K6并回到K2后核心稳态24/24项与对应Fixed在±10%内 | 档位事务被排除；5个P→D点超带；不是独立重复统计 |

按实验编号计，当前有10项至少取得一条通过验收的功能或机制证据：E0、E2、E3、E4、E5、
E6、E7、E10、E12、E13。E1完成了执行但publication gate保持暂停；E8在provenance门前阻塞；
E9和E11完成了诊断运行但正确性验收未通过。这里的“10项有证据”不等于10项正式完成：
E2/E3/E6/E7/E10/E13仍主要是单轮qualification，E5只在当前模块上复验了两个臂，E12
也只满足机制次数门。完整论文主矩阵、动态长trace和鲁棒性实验仍未完成。

## 3. E0：兼容性与基础正确性

DeepSeek-V2-Lite-Chat BF16 的同一冻结 validation 请求在 ours、KTransformers 和
MoE-Infinity 上得到相同的32-token输出：

```text
69cd56b10352d7b1b4478134f1455d6cfbd6509a7e08d9fc100b4ac433bf9aa6
```

三方基础执行、接口和输出门通过。ours 的 DeepSeek maximum-resident Graph 因32-GiB卡
缺少 headroom 合法记为 N/A；这不是失败，也不能以 eager 冒充 Graph。

状态：基础 compatibility qualification 通过，但 E0 不承担性能排名；更广输入上的
teacher-forced logits、top-1 agreement和perplexity论文级数值门仍未完成。

来源：[RESULTS_E0_E1_20260824.md](RESULTS_E0_E1_20260824.md)

## 4. E1：maximum-resident anchor

三系统各完成5个独立 server process blocks，共15 blocks、150条 measurement 请求。

| 系统 | TTFT | TPOT | E2E | output tok/s | peak HBM |
|---|---:|---:|---:|---:|---:|
| ours | 129.47 ms | 54.22 ms | 1810.24 ms | 17.66 | 30,997 MiB |
| KTransformers | 408.22 ms | 9.96 ms | 717.09 ms | 44.50 | 31,559 MiB |
| MoE-Infinity | 1348.69 ms | 306.98 ms | 10865.13 ms | 2.95 | 29,379 MiB |

这些数值只是描述性 anchor。10个 evaluation prompts 中，没有一个在全部15 blocks间得到
相同完整输出；此外三系统的 HBM、执行模式和 placement 也未配平。因此 E1 的执行状态为
completed，但跨系统 correctness/publication gate 为 held，不能计算正式speedup。

来源：[RESULTS_E0_E1_20260824.md](RESULTS_E0_E1_20260824.md)

## 5. E2：C1/C16 全覆盖双页表与多机制对照

### 5.1 最新全覆盖双页表主结果

最新ours使用DeepSeek-V2-Lite-Chat BF16，只运行C1、C16，每个场景一次独立冷进程。
C1和C16均为4/4运行通过；Long/C1的`grouped_token_events=0`，旧轮的流式协议问题已消失。
pinned-UVA仍复用2026-08-31独立冷进程基线，因此倍率是跨轮描述性比较。

| 场景 | ours E2E | eager pinned-UVA remote E2E | 加速 | ours单臂验收 |
|---|---:|---:|---:|---|
| Short/C1 | 309.88 ms | 3012.17 ms | 9.72× | 通过 |
| Ordinary/C1 | 1052.93 ms | 9089.28 ms | 8.63× | 通过 |
| RAG/C1 | 1851.91 ms | 9401.91 ms | 5.08× | 通过；跨臂hash不同 |
| Long/C1 | 4116.30 ms | 47566.72 ms | 11.56× | 通过 |
| Short/C16 | 724.68 ms | 2352.90 ms | 3.25× | 通过 |
| Ordinary/C16 | 4841.16 ms | 11387.91 ms | 2.35× | 通过 |
| RAG/C16 | 21198.94 ms | 86402.18 ms | 4.08× | 通过；跨臂hash不同 |
| Long/C16 | 19719.94 ms | 41269.70 ms | 2.09× | 通过 |

这里的“严格通过”是每个 arm 自身的运行、请求长度、流式事件、故障和清理验收，不自动
等于跨 arm 输出一致。逐 token hash 审计结果是：all-GPU、pinned-UVA、copy-on-forward
三种原生 vLLM 路径在四类 workload 上相互一致；最新 ours 的 Short、Ordinary、Long 与
原生路径一致，但 RAG/C1 和 RAG/C16 不一致。因此应分两层报告：

- 按单臂运行、流式、故障和清理门，8/8点通过，跨轮ours/pinned-UVA E2E比为
  `2.09×–11.56×`；
- 再要求跨臂exact-token一致后排除两个RAG点，剩余6个有效点；这6点的范围仍为
  `2.09×–11.56×`。

### 5.2 同一冻结请求口径下的其他机制数据

下表汇总 2026-08-31 attempt 01 的参考臂和 attempt 02 的最新 ours；单位为 ms，均为单个
冷进程样本。`—` 表示本次批次没有完成该点，不代表机制普遍不可运行。

- all-GPU：原生 vLLM eager，全部权重常驻GPU，不卸载；
- pinned-UVA：原生 vLLM eager，卸载参数始终由GPU远程读取；
- copy-on-forward：原生 vLLM eager，每次forward把所需CPU参数复制到GPU；
- KTransformers：按冻结冷专家计划进行GPU/CPU混合放置；
- MoE-Infinity：使用全局按需专家缓存与offload store。

| 场景 | ours，全覆盖双表+Graph | all-GPU eager | pinned-UVA eager | copy-on-forward eager | KTransformers | MoE-Infinity |
|---|---:|---:|---:|---:|---:|---:|
| Short/C1 | 309.88 | 2796.42 | 3012.17 | 2263.21 | 744.47 | 11736.57 |
| Ordinary/C1 | 1052.93 | 10670.23 | 9089.28 | 9287.99 | 1700.49 | — |
| RAG/C1 | 1851.91† | 9241.32 | 9401.91 | 9307.59 | 3022.12 | — |
| Long/C1 | 4116.30 | 35879.16 | 47566.72 | 36868.29 | 5510.82 | — |
| Short/C16 | 724.68 | 3185.41 | 2352.90 | 2423.91 | 13312.23 | — |
| Ordinary/C16 | 4841.16 | OOM | 11387.91 | 10039.89 | 53875.96 | — |
| RAG/C16 | 21198.94† | OOM | 86402.18 | 21757.48 | 183044.19§ | — |
| Long/C16 | 19719.94 | OOM | 41269.70 | 39709.86 | — | — |

说明：

- `†`：ours 的 RAG 输出长度和单臂 verifier 通过，但 token hash 与三种原生 vLLM
  路径不一致，不能进入 exact-token 性能排名；
- `§`：KTransformers RAG/C16 留有完整 runner 性能结果，但本批次没有生成 case-level
  verification receipt，因此只列作 raw diagnostic；KTransformers 其他已完成点也尚未
  通过完整跨框架 generation gate；
- all-GPU 的 Ordinary/RAG/Long C16 分别在正式固定请求下遇到172/684/342 MiB临时分配
  OOM，属于该固定容量下的 N/A；
- MoE-Infinity 本批次只完成 Short/C1，其他点未完成，不用早期结果填入本轮空格。

在同时满足 ours 单臂严格验收、原生路径完成且跨臂 exact-token 一致的点上，描述性
E2E 比值为：

| 对照机制 | 有效配对点 | ours相对对照的E2E加速 | 缺口 |
|---|---:|---:|---|
| pinned-UVA eager remote | 6 | 2.09×–11.56× | 跨轮、非equal-HBM，卸载集合不同 |
| copy-on-forward eager | 6 | 2.01×–8.96× | 跨轮、非equal-HBM，按完整Parameter复制 |
| all-GPU eager | 4 | 4.40×–10.13× | 跨轮；C16除Short外均容量N/A |

这些比值是相同冻结请求上的单样本描述，不是单项机制消融。KTransformers 和
MoE-Infinity 因跨框架 generation gate 或本批次覆盖不完整，不计算正式加速比。


完整E2尚未完成：当前没有按完整measured peak HBM配平的DeepSeek三系统主表，也没有
Qwen ours/KT/vLLM所有tier、workload、C1/C4/C8/C16和每点5个独立进程的正式矩阵。

来源：

- D/ours B1/B16：`artifacts/e3-c1-c16-full-dual-20260901-reproduce-all-attempt-02/`
- Native FLASH copy/UVA：`artifacts/e3-native-copy-uva-reproduce-all-20260901-attempt-01/`
- Native TRITON大B8：`artifacts/e3-native-triton-large-b8-reproduce-all-20260901-attempt-01/`
- Prefill Graph-copy：`artifacts/e3-prefill-graph-copy-reproduce-all-20260901-attempt-01/`
- Decode完整汇总：`artifacts/e3-decode-complete-reproduce-all-20260901-attempt-01/`
- [RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md)
- [RESULTS_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_MATRIX_20260827.md](RESULTS_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_MATRIX_20260827.md)
- [RESULTS_E2_C16_DIRECT_BATCH_MATRIX_20260828.md](RESULTS_E2_C16_DIRECT_BATCH_MATRIX_20260828.md)
- reference arms：`artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-01/e2/`
- ours C1：`artifacts/e2-e3-e10-full-c1-c16-dual-20260901-reproduce-all-attempt-01/e2/c1/`
- ours C16：`artifacts/e2-c16-frozen-20260901-reproduce-all-attempt-02/e2/c16/`

## 6. E3：Prefill/Decode当前档双页表与完整路径对照

### 6.1 最新 D/ours 全覆盖双页表

最新 D 定义为“GPU-local/copy Prefill Graph → 当前档双页表 P→D 切换 → remote Decode
Graph”。本轮只运行 B1/B16：Short/B1、Short/B16、Ordinary/B1、Medium/B1、Long/B1
使用 cold10（2.109 GiB）；Ordinary/B16、Medium/B16、RAG/B1、Long/B16 使用 cold38
（8.016 GiB）；RAG/B16 因容量约束使用 cold89（18.773 GiB）。十个场景均达到双页表
100%覆盖，没有CPU range-map回退或Graph重捕获。

| 场景 | Batch | Prefill | P→D | steady Decode | E2E |
|---|---:|---:|---:|---:|---:|
| Short | 1 | 68.11 ms | 6.60 ms | 9.97 ms | 455.45 ms |
| Short | 16 | 105.95 ms | 6.47 ms | 18.30 ms | 754.39 ms |
| Ordinary | 1 | 74.13 ms | 6.41 ms | 9.55 ms | 1379.46 ms |
| Ordinary | 16 | 368.50 ms | 23.54 ms | 54.98 ms | 7503.39 ms |
| Medium | 1 | 112.85 ms | 6.44 ms | 13.90 ms | 1940.10 ms |
| Medium | 16 | 1123.14 ms | 24.20 ms | 57.05 ms | 8528.73 ms |
| RAG | 1 | 587.35 ms | 27.40 ms | 15.97 ms | 2894.29 ms |
| RAG | 16 | 7272.86 ms | 63.68 ms | 161.56 ms | 27996.95 ms |
| Long | 1 | 75.09 ms | 6.50 ms | 10.48 ms | 5618.18 ms |
| Long | 16 | 369.47 ms | 23.20 ms | 61.37 ms | 31558.98 ms |

RAG/C16的早期cold63配置需要14.72 GiB热专家，但工作区建立后仅剩10.17 GiB可用，因此
合法改为cold89；这是容量约束，不是页表故障。

最新结果来自`e3-c1-c16-full-dual-20260901-reproduce-all-attempt-02`，证明当前档双页表能
正确覆盖和切换，但这一轮没有成功配对纯remote或纯copy arm，
因此不能单独由上表计算 D/remote 或 D/copy 加速比。下面把已经完成的较早机制矩阵分协议
列出；这些数据补足机制行为，但不回填成2026-08-31最新D的同轮对照。

### 6.5 自定义 Graph 与原生 copy/UVA 的阶段对照（完整数据复跑）

2026-09-01/02重新采集native FLASH copy/UVA 20/20、Custom Prefill Graph-copy 10/10和
原协议`FLASH_ATTN` Custom Decode 8/8。旧表的Medium/B8、RAG/B8由当前档全覆盖双页表
D/TRITON路径补齐，并对称重跑native TRITON copy/UVA 4/4。最终Decode表10/10逐cell
同backend且完整输出hash一致；矩阵仍分为FLASH和TRITON两个协议层。

Prefill custom为`TRITON_ATTN`+exact expert slices，native为`FLASH_ATTN`+whole
Parameters。倍率定义为`native/custom`，只作数值诊断。单位均为ms。

| 请求 | B | Custom Graph-copy | Native copy | Native UVA | Graph/copy | Graph/UVA |
|---|---:|---:|---:|---:|---:|---:|
| Short 128 | 1 | 67.93 | 68.19 | 70.18 | 1.004× | 1.033× |
| Short 128 | 8 | 92.14 | 76.82 | 196.81 | 0.834× | 2.136× |
| Ordinary 512 | 1 | 73.96 | 70.62 | 121.89 | 0.955× | 1.648× |
| Ordinary 512 | 8 | 151.54 | 137.93 | 826.48 | 0.910× | 5.454× |
| Medium 2048 | 1 | 113.44 | 103.48 | 418.44 | 0.912× | 3.689× |
| Medium 2048 | 8 | 651.61 | 563.05 | 7508.83 | 0.864× | 11.524× |
| RAG 8192 | 1 | 588.59 | 461.11 | 3754.00 | 0.783× | 6.378× |
| RAG 8192 | 8 | 3696.11 | 2728.73 | 48291.27 | 0.738× | 13.065× |
| Long 512 | 1 | 75.03 | 71.55 | 122.66 | 0.954× | 1.635× |
| Long 512 | 8 | 151.16 | 138.14 | 834.24 | 0.914× | 5.519× |

Custom Graph-copy相对UVA为10/10更快1.033×–13.065×；相对native copy仅Short/B1
快1.004×，其余9/10更慢。首token仍为9/10 cell一致，Medium/B8有1/8请求不同。

Decode延迟是whole-batch单token稳态step中位数。8个C点三臂统一`FLASH_ATTN`；两个旧
缺失点三臂统一`TRITON_ATTN`，并单列协议和完整hash验收。

| 请求 | B | 三臂协议 | Custom Graph remote | Native copy | Native UVA | Custom/copy | Custom/UVA | 完整hash |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Short 128→32 | 1 | C/FLASH | 10.20 | 70.56 | 68.80 | 6.92× | 6.75× | 一致 |
| Short 128→32 | 8 | C/FLASH | 15.37 | 70.64 | 69.48 | 4.60× | 4.52× | 一致 |
| Ordinary 512→128 | 1 | C/FLASH | 10.13 | 69.90 | 67.36 | 6.90× | 6.65× | 一致 |
| Ordinary 512→128 | 8 | C/FLASH | 17.95 | 71.87 | 69.41 | 4.00× | 3.87× | 一致 |
| Medium 2048→128 | 1 | C/FLASH | 13.98 | 69.32 | 68.97 | 4.96× | 4.93× | 一致 |
| Medium 2048→128 | 8 | D/TRITON | 42.01 | 167.77 | 82.78 | 3.99× | 1.97× | 一致 |
| RAG 8192→128 | 1 | C/FLASH | 24.01 | 161.39 | 68.01 | 6.72× | 2.83× | 一致 |
| RAG 8192→128 | 8 | D/TRITON | 64.21 | 280.88 | 125.06 | 4.37× | 1.95× | 一致 |
| Long 512→512 | 1 | C/FLASH | 11.06 | 69.65 | 67.34 | 6.30× | 6.09× | 一致 |
| Long 512→512 | 8 | C/FLASH | 31.51 | 72.38 | 70.49 | 2.30× | 2.24× | 一致 |

逐cell同backend的10点中，Custom全部最快：对native copy快2.30×–6.92×、对native
UVA快1.95×–6.75×，完整输出hash 10/10一致。这些范围合并了FLASH和TRITON两个协议层，
不能作为纯remote访问、纯Graph或attention backend的单项排名。

D补充点保持原B8和完整16K/65K Prefill，未拆batch或启用chunked Prefill；两点均完整生成，
Prefill/Decode Graph replay、双页表100%覆盖、CPU range-map回退为0，并与对称native
TRITON arms完整hash一致。另做的三次
Medium/B8-G/FLASH定向尝试在cold38/cold63下仍触发原original-VA大Prefill非法访问，
证明旧N/A不能通过简单扩容合法消除。完整记录见
`RESULTS_E3_SECTION65_COMPLETE_RERUN_20260831.md`。


### 6.7 E3综合结论与边界

当前证据可以支持：

1. 最新D在B1/B16十点实现当前档双页表100%覆盖并正确完成请求；
2. 较早同坐标Decode-Graph矩阵中，hybrid G相对always-copy B为7/7更快3.014×–5.270×，
   但always-remote C在6/7点更快，说明极冷计划下不切换通常更好；
3. 原生native矩阵中copy降低Prefill成本、UVA降低Decode成本，RAG/B8已经实测出现完整
   请求crossover；
4. 自定义remote Decode Graph在10个逐cell同backend点都快于两个native eager路径，但
   这不是equal-HBM或单机制隔离结论。

仍不能支持：把较早G/B/C、H/A/R或native数值填入最新D/C/F空格；声称当前大坐标已经有
有效crossover；发布跨backend、跨placement或跨模块的统一系统排名。所有补充矩阵均只有
一个冷进程样本，没有方差或置信区间。

来源：

- [RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md)
- [RESULTS_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md](RESULTS_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md)
- [RESULTS_E3_RERUN_20260825.md](RESULTS_E3_RERUN_20260825.md)
- [RESULTS_E3_NATIVE_COPY_VS_UVA_PHASE_20260828.md](RESULTS_E3_NATIVE_COPY_VS_UVA_PHASE_20260828.md)
- [RESULTS_E3_PREFILL_GRAPH_COPY_VS_NATIVE_COPY_UVA_B1_B8_20260828.md](RESULTS_E3_PREFILL_GRAPH_COPY_VS_NATIVE_COPY_UVA_B1_B8_20260828.md)
- [RESULTS_E3_DECODE_THREE_MECHANISM_B1_B8_20260828.md](RESULTS_E3_DECODE_THREE_MECHANISM_B1_B8_20260828.md)
- [RESULTS_E3_D_VS_C_PAIRED_ATTEMPT02_20260831.md](RESULTS_E3_D_VS_C_PAIRED_ATTEMPT02_20260831.md)
- [RESULTS_E3_D_VS_C_FULL_GRAPH_PAIRED_ATTEMPT02_20260831.md](RESULTS_E3_D_VS_C_FULL_GRAPH_PAIRED_ATTEMPT02_20260831.md)

## 7. E4：并发与PCIe饱和

最近一次5个独立进程复验使用固定always-remote计划，扫描C1--C32。预注册的持续退化判据
在所有并发档位均未触发。

| C | request TPOT | Decode tok/s | PCIe RX |
|---:|---:|---:|---:|
| 1 | 87.41 ms | 11.60 | 0.166 GiB/s |
| 4 | 87.96 ms | 45.89 | 0.397 GiB/s |
| 8 | 89.80 ms | 89.87 | 1.387 GiB/s |
| 16 | 89.75 ms | 180.33 | 2.603 GiB/s |
| 24 | 85.02 ms | 284.77 | 3.841 GiB/s |
| 32 | 86.56 ms | 372.94 | 4.698 GiB/s |

截至C32，没有观察到TPOT持续恶化；PCIe RX峰值约为Gen5 x16理论单向带宽的8.00%。
CPU DRAM带宽因KVM不暴露IMC而为N/A。结论只覆盖该模型、请求和2.2649-GB remote计划。

来源：`artifacts/e4-repro-20260902-attempt-01/analysis/`

## 8. E5：稳定VA、Graph与PTE消融

E5是当前结果集中机制隔离最完整的一组formal实验，主要结果如下：

| 消融 | B1 | B4 |
|---|---:|---:|
| remote Decode：Graph相对eager改善 | 79.11% | 59.06% |
| alias Prefill：Graph相对eager改善 | 48.09% | 27.05% |
| PARK/RESUME相对DESTROY/rebuild加速 | 185.14× | 164.83× |
| grouped PTE相对sequential的RESUME加速 | 5.499× | 5.473× |
| non-stable VA相对stable的重建/重捕获比率 | 1.998× | 1.811× |

双拓扑formal共22/22冷进程通过；其B1/B4的Prefill为69.612/84.614 ms，steady Decode为
10.988/21.978 ms，cutover为49.145/49.291 ms。Graph、VA、PTE、驻留、输出和清理门通过。

这些数据来自全覆盖双页表优化之前的E5协议，仍能支持对应消融方向。2026-09-02又完成
stable-VA DESTROY/rebuild与non-stable-VA/recapture两臂，
每臂10个独立冷进程、20个plan generation，两个verifier均为`status=ok`：

| 当前模块配对 | B4 rebuild median | B1 rebuild median |
|---|---:|---:|
| stable Parameter VA + DESTROY/rebuild | 8,399.831 ms | 9,008.189 ms |
| non-stable Parameter VA + reallocate/recapture | 11,499.471 ms | 13,712.162 ms |
| non-stable / stable | 1.369× | 1.522× |

两个臂输出hash一致，fault scan和资源释放全部通过。该配对只更新stable/non-stable
rebuild证据；Graph、PARK/RESUME和grouped/sequential PTE的其余数值仍来自较早模块，尚未
完成当前模块整组复验。

来源：

- `artifacts/e5-stable-vs-nonstable-current-module-20260902-attempt-01/comparison.json`
- [RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md](RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md)
- [REPORT_E2_E12_REPRO_20260827.md](REPORT_E2_E12_REPRO_20260827.md)

## 9. E6：Plan Class与切换摊销

full rebuild、serial wave和batched difference三臂按相同冻结几何各完整运行一次，三个
verifier均为`status=ok`，输出hash一致。

| Difference | Full rebuild | Serial wave | Batched difference |
|---|---:|---:|---:|
| 5% | 17.761/18.437 s | 2.005/2.156 s | 2.010/2.101 s |
| 10% | 18.968/19.024 s | 4.018/4.130 s | 2.677/2.758 s |
| 25% | 20.068/19.238 s | 10.278/10.409 s | 4.795/4.793 s |
| 50% | 21.299/21.187 s | 21.070/20.360 s | 8.506/8.129 s |
| 100% | 24.312/23.957 s | 39.554/39.698 s | 14.969/14.987 s |

表中为outbound/return。batched difference相对full rebuild快1.60×--8.84×；10%--100%
相对serial wave快1.50×–2.65×，5%两者近似相同。full rebuild把Graph capture count从
2增加到22，serial和batched均保持2。

这是单进程、单样本qualification，不能把点估计当作median或置信区间。

来源：`artifacts/e6-summary-20260902-reproduce-all-attempt-01-latest-dual/summary.json`

## 10. E7：专家热度与workload shift

最新实验让六个arm分别使用独立冷进程，每arm执行一次32-request shift trace，六臂输出和
路由计数hash一致，Graph count保持2。

| Arm | 请求中位延迟 | cycle tok/s | 动态remote命中率 | Oracle plan hit | 同步维护 |
|---|---:|---:|---:|---:|---:|
| frequency-cold | 838.62 ms | 35.15 | 1.427% | 36.98% | 90.75 ms |
| random | 868.77 ms | 34.14 | 3.186% | 4.17% | 74.37 ms |
| frozen static | 804.58 ms | 36.30 | 1.688% | 18.23% | 72.10 ms |
| GPU heat + CPU EWMA | 851.22 ms | 20.97 | 0.712% | 88.54% | 19,765.65 ms |
| oracle | 811.88 ms | 35.98 | 0.178% | 100% | 73.36 ms |
| hot-adversarial | 1002.28 ms | 31.54 | 12.484% | 0% | 72.12 ms |

Adaptive将plan hit从36.98%提高到88.54%，动态remote assignments减少50.12%。但请求
中位延迟仍比frequency-cold高1.50%，同步维护累计19.766 s，使cycle throughput从35.15
降至20.97 tok/s，即下降40.35%，且32请求内未达到95%收敛门。

因此E7证明放置方向和机制链路正确，尚不能证明包含维护成本后的端到端性能无回退。

来源：`artifacts/e7-independent-cold-20260902-attempt-25-targeted-canonical-pte-qualification/verification.json`

## 11. E8：MoE-Infinity cache/prefetch

E8仍阻塞在provenance门：缺少与当前活动扩展、commit和约30-GiB store匹配的fresh build
receipt。preflight返回码1，实验没有绕过commit、patch、ELF或receipt hash检查。

状态：无性能数字，不能声称E8通过或失败于算法性能。

来源：[REPORT_E2_E12_REPRO_20260827.md](REPORT_E2_E12_REPRO_20260827.md)

## 12. E9：KTransformers placement/CPU消融

baseline、threads7、threads28、eager、uniform、no-overlap、no-mixed共7/7 case运行完成，
每臂C1+C8的请求均生成32 tokens。诊断点中threads7吞吐为C1 51.991 tok/s、C8
56.892 tok/s。

但是顶层correctness verifier未通过：uniform与no-mixed出现greedy token漂移，而当前没有
teacher-forced logits/top-1证据证明它只是无害舍入误差。因此所有性能值只能用于诊断，
没有启动5-block formal。

状态：运行完成但总体验收未通过。

来源：[REPORT_E2_E12_REPRO_20260827.md](REPORT_E2_E12_REPRO_20260827.md)

## 13. E10：2→4→6 GiB联合HBM压力切档

最新E10完整执行一条cold70/2 GiB→cold80/4 GiB→cold89/6 GiB压力触发序列，verifier
`status=ok`。三档全覆盖双页表均为100%，没有CPU range-map回退、Graph重捕获或内核错误。

| KV | 纯Prefill | 切档 | P→D | steady Decode | 请求E2E |
|---:|---:|---:|---:|---:|---:|
| 2 GiB | 340.74 ms | 0 | 108.40 ms | 20.11 ms | 1007.96 ms |
| 4 GiB | 399.37 ms | 542.80 ms | 117.82 ms | 24.25 ms | 1672.99 ms |
| 6 GiB | 447.60 ms | 565.27 ms | 138.35 ms | 26.78 ms | 1831.87 ms |

结果证明当前档双页表能在scheduler quiescent point按相邻约2 GiB同步切换KV容量与专家
Plan Class，并保持三档100%覆盖。完整档位事务仍需约543--565 ms，P→D为108--138 ms，
不能称为已经消除切档成本。

本轮由冻结的压力序列触发三档转换，用于验证当前档双页表和切档路径。它尚未比较fixed、
oracle、auto无backup、auto+backup和full-joint等正式E10策略，也没有K2/K4/K8/K10重复
峰谷、5 seeds、open-loop到达或goodput/SLO统计。因此只能写作E10 preflight/qualification，
不能写成自动联合HBM控制器已经完成。

来源：`artifacts/e10-kv-2-4-6-pressure-20260902-attempt-01-scheduler-quiescent-sync-tier/verification.json`

## 14. E11：continuous arrival与phase epoch

最近一次完整attempt为
`e11-continuous-arrival-20260902-attempt-09-remote-park-full-dual-strict-qualification`，
证明了调度行为，但顶层verifier保持`status=error`：

| 指标 | phase epoch + phase-specific | phase epoch + remote-only | continuous + remote-only |
|---|---:|---:|---:|
| 成功请求 | 8/8 | 8/8 | 8/8 |
| mixed steps | 0 | 0 | 1 |
| TTFT p50 | 406.31 ms | 613.60 ms | 153.10 ms |
| TPOT p50 | 54.58 ms | 77.55 ms | 81.88 ms |
| E2E p50 | 1225.00 ms | 1776.83 ms | 1381.30 ms |
| generated-token throughput | 61.02 tok/s | 48.75 tok/s | 72.97 tok/s |

phase epoch把新Prefill和旧Decode完全分开；另一个hybrid fail-closed arm在mixed request
上按预期拒绝执行。但phase-specific Full Prefill Graph与两条remote-only路径输出token IDs
不一致，不能进入性能排名。

状态：E11已按用户要求终止并撤销为完成它而做的后续修改。attempt 10--16只保留失败诊断，
不得晋升；当前只允许离线审计，不再启动GPU复现。

来源：`artifacts/e11-continuous-arrival-20260902-attempt-09-remote-park-full-dual-strict-qualification/verification.json`

## 15. E12：Graph replay与PARK/RESUME qualification

最近可用结果为3次warmup加1,000次measured Graph replays：

- full-byte compare错误数：0；
- GPU时间min/median/mean/max：45.8621/45.9045/45.9051/45.9544 ms；
- host submit median：77.664 µs；
- 每次逻辑H2D：2,415,919,104 B，对应49.015 GiB/s；
- consumer/upload VA保持稳定，allocator正常关闭。

当前模块还完成了1,000/1,000次完整双页表PARK/RESUME，逐轮验证Decode CPU-shadow和
Prefill GPU-backing视图。PARK p50/p95为0.769/0.838 ms，RESUME为0.567/0.611 ms；
共记录2,000次dual-view switch、8,000次PDE写入和2,000个TLB batches。无Xid、PDE/PTE
fault或资源泄漏，并成功恢复stock UVM和GPU 0 MiB。

状态：Graph replay smoke与1,000-cycle机制qualification通过，但尚未完成6/24小时整模型
soak、重复KV峰谷、barrier故障注入或长期性能/HBM退化检查。1,000轮循环仅3.700 s，不能
把次数门写成时间soak通过。

来源：

- [RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md](RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md)
- [REPORT_E2_E12_REPRO_20260827.md](REPORT_E2_E12_REPRO_20260827.md)

## 16. E13：双阶段 CUDA Graph 下的 KV 档位与 Adaptive 回环

### 16.1 协议与验收

当前权威轮为
`artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/`。
它在Qwen3-30B-A3B-FP8、Short-128/Output-32、B64/B128上依次运行Fixed-K2、Fixed-K6和
Adaptive三个独立冷进程。每个cell先运行一次排除warmup，再取三个进程内样本的中位数。

该轮启用`TRITON_ATTN`、`CUDAGraphMode.FULL_SEPARATE`、
`prefill_graph_remote_decode`、original-VA双页表以及PARK/RESUME快速P→D。Fixed每臂只
捕获当前Plan Class的3个Graph shape；Adaptive预捕获cold70/cold80/cold89三档共9个
Graph entry，并严格按`K2→K4→K6→K4→K2`执行四次相邻约2 GB的KV/专家同步切档。
初始化、四个档位事务和每个目标档的一次warmup均不进入性能计时。

三臂3/3返回0，正式测量2,880个请求，加排除warmup共3,840/3,840请求完成。正式服务的
975/975个step均为FULL Graph replay，捕获后没有Graph重捕获或eager fallback；84/84项
协议、输出、档位、fault和清理检查通过。输出hash一致，无Xid、PTE/PDE或MMU fault，
最终GPU为0 MiB并恢复stock UVM。

### 16.2 Fixed-K2与Fixed-K6

| B | 档位 | req/s | TTFT | TPOT | E2E | Prefill | P→D | steady Decode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | Fixed-K2 | 29.1489 | 726.68 ms | 46.59 ms | 2189.23 ms | 722.16 ms | 489.74 ms | 24.89 ms |
| 64 | Fixed-K6 | 25.3277 | 768.05 ms | 56.65 ms | 2519.78 ms | 763.26 ms | 470.04 ms | 32.67 ms |
| 128 | Fixed-K2 | 39.6909 | 972.30 ms | 72.17 ms | 3212.92 ms | 1208.64 ms | 492.87 ms | 41.12 ms |
| 128 | Fixed-K6 | 32.5717 | 1066.51 ms | 91.25 ms | 3917.92 ms | 1344.89 ms | 501.92 ms | 59.54 ms |

Fixed-K2在B64/B128的吞吐分别比K6高15.09%/21.86%，E2E分别低13.12%/17.99%。两个
Short workload都远未达到K2的KV容量限制，因此K6的额外KV容量没有收益；更多专家权重
卸载反而增加Prefill和remote Decode开销。该结论只覆盖当前低KV压力坐标，不能外推为
高压力时K2仍优于K6。

### 16.3 Adaptive切档后的稳态等价

下表均为`Adaptive/对应Fixed`；档位事务和目标档warmup已经排除。

| Adaptive状态 | B | req/s | Prefill | steady Decode | E2E |
|---|---:|---:|---:|---:|---:|
| 初始K2 | 64 | 0.988× | 1.005× | 0.982× | 1.012× |
| 增长后K6 | 64 | 0.965× | 1.010× | 1.019× | 1.037× |
| 回缩后K2 | 64 | 0.988× | 1.001× | 1.003× | 1.012× |
| 初始K2 | 128 | 0.979× | 1.004× | 1.001× | 1.021× |
| 增长后K6 | 128 | 0.971× | 1.004× | 1.020× | 1.030× |
| 回缩后K2 | 128 | 0.968× | 1.007× | 1.001× | 1.034× |

用户指定的吞吐、Prefill、steady Decode和E2E共24/24个A/F倍率均位于预注册的±10%
带内。因此可以声称：忽略档位事务及warmup后，Adaptive在初始K2、增长后K6和回缩后K2
均恢复到对应Fixed档的稳态水平，回缩没有留下稳态性能回退。

若把output tok/s、batch wall、P→D、TTFT、TPOT也纳入，则只有49/54在带内。5个超带点
全部是P→D：初始K2/B128为`1.117×`，增长后K6的B64/B128为`1.142×/1.240×`，回缩后
K2的B64/B128为`1.109×/1.138×`。P→D是每个batch首个Decode step中的PARK和
remote-map提交，不是已排除的K2↔K6档位事务；初始K2/B128在切档前也超带，所以现有数据
不能把差异唯一归因于切档历史，也不能声称P→D瞬时停顿完全等价。

四个排除的档位事务依次耗时7.272/6.992/7.671/6.800 s。本轮是每臂一个冷进程的单轮
qualification，三个cell样本是进程内重复；不能把±10%门写成统计等价检验，也不能声称
包含同步档位迁移成本后的Adaptive端到端性能已经与Fixed等价。

来源：

- [RESULTS_E13_KV_ADAPTIVE_20260901.md](RESULTS_E13_KV_ADAPTIVE_20260901.md)
- `artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/verification.json`
- [详细复现文档](../reproduce/REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md)

## 17. 当前可以与不可以得出的结论

当前可以支持：

1. 全覆盖双页表在最新E2/E3/E10中功能正确，覆盖率100%，没有range-map回退或页表故障。
2. 最新E2的ours有8/8个运行通过点；其中6个还满足跨臂exact-token一致，在这6点上
   ours+Graph组合相对旧eager pinned-UVA remote基线快2.09×–11.56×，相对旧
   copy-on-forward基线快2.01×–8.96×。
3. E5的同placement消融表明CUDA Graph能显著降低Prefill和Decode稳态开销；当前模块的
   两臂复验还表明non-stable VA重分配/重捕获比stable-VA DESTROY/rebuild慢1.369×--1.522×。
4. 较早模块的消融中，batched difference、PARK/RESUME和grouped PTE均明显优于对应旧路径。
5. E3较早同坐标Decode-Graph矩阵中hybrid相对always-copy为7/7更快，但always-remote在
   6/7点胜出；native whole-Parameter矩阵则实测到RAG/B8由copy胜出的phase crossover。
6. 固定remote计划在E4的C1--C32范围内没有出现持续TPOT退化或PCIe饱和。
7. E13双阶段Graph单轮中，低KV压力的Fixed-K2在B64/B128均优于Fixed-K6；排除档位事务
   和warmup后，Adaptive初始K2、增长后K6、回缩后K2的24/24项核心稳态指标均在对应
   Fixed的±10%内。

当前不能支持：

1. 不能声称ours在所有模型、请求和系统上全局最优；当前E2只有单轮且placement不完全相同。
2. 不能把E2的RAG/C1和RAG/C16写入跨系统exact-token性能排名；两点虽完成请求，但ours与
   原生vLLM输出hash不同。
3. 不能从E2组合对比中单独归因CUDA Graph、GPU缓存或双页表的贡献。
4. 不能发布E1跨系统排名、E9性能排名或E11正式性能结果，因为正确性门未通过。
5. 不能声称E8已运行，也不能把E12的1,000-replay或1,000-cycle机制测试称为完整鲁棒性soak。
6. 不能声称E7在线自适应已经无开销；当前同步维护使cycle吞吐下降40.35%。
7. 不能声称E3已得到完整请求路径crossover；phase-only三机制虽已补齐，但当前D与remote
   Prefill全路径配对仍没有形成有效结果。
8. 不能把E10人工压力切档写成automatic/full-joint控制器结果。
9. 不能由E13声称档位切换无开销或快速P→D完全等价；四个7秒级档位事务未计时，且5个
   P→D点比对应Fixed高10.92%--24.03%。也不能把单冷进程臂的±10%门写成统计等价证明。

## 18. 剩余工作与执行优先级

当前最关键的未完成项按依赖关系排序如下：

1. 正确性门：隔离E2的RAG Prefill Graph输出分叉，并补齐E1/E9/E11的teacher-forced
   logits、top-1 agreement和perplexity；在对应门通过前不发布相关跨系统或scheduler排名。
2. E2主矩阵：按完整measured peak HBM配平DeepSeek三系统与Qwen ours/KT/vLLM，每个主点
   5个独立process blocks并报告95% CI。
3. E3 crossover：解决或明确N/A当前平台的remote Prefill大坐标，在同模型、同expert
   coordinates、同Graph能力下完成hybrid/always-copy/always-remote配对。
4. E5/E6统计：用当前模块补齐E5其余Graph/PTE消融，并把E6三臂从单样本扩展为重复统计。
5. E7/E10动态实验：异步化heat maintenance或明确同步开销；运行5-seed重复峰谷、backup
   sizing、预测误差、open-loop trace和full-joint goodput/SLO对照。
6. E8：取得与当前MoE-Infinity扩展、commit和store一致的fresh build receipt后再运行，
   不绕过provenance门。
7. E11范围：先隔离pure-Prefill Full Graph与eager输出差异，再决定继续phase epoch还是
   冻结phase-batched claim。
8. E12：完成至少6小时整模型soak、重复KV峰谷、barrier故障注入、长期HBM/性能退化检查
   和完整custom→stock生命周期验证。
9. E13：把双Graph回环扩展为多个独立冷进程和动态长trace，纳入档位事务成本，并在K2
   容量不足的长上下文/高并发坐标验证K6的容量收益；继续隔离5个P→D超带点。

## 19. 权威复现入口

所有实验的最新可用复现状态和入口统一见：

[experiments/reproduce/README.md](../reproduce/README.md)

其中E2/E3/E10联合协议见：

[REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](../reproduce/REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md)

E13完整协议总入口和本轮双Graph回环专用入口见：

- [REPRODUCE_E13_KV_ADAPTIVE.md](../reproduce/REPRODUCE_E13_KV_ADAPTIVE.md)
- [REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md](../reproduce/REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md)

当前模块E5/E12补充qualification见：

[RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md](RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md)

本文只统一结果与证据级别；它不替代各实验原始artifact、verifier和冻结协议。
