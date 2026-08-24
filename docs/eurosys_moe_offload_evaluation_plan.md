# 面向 EuroSys 的阶段感知 MoE 卸载评测方案（v2）

日期：2026-08-24  
状态：v2 协议已冻结；E0 已通过基础兼容性 qualification，E1 已完成五重复但处于
跨框架数值正确性 hold  
协议 ID：`eurosys-pa-v2-20260824`

## 1. 评测目标与论文范围

本项目要回答的问题不是“把一部分专家放到 CPU 后某个 kernel 快了多少”，而是：

> 在 GPU HBM 同时被 MoE 专家权重和 KV cache 竞争时，系统能否保持 Parameter VA、
> KV VA 和 CUDA Graph 稳定，根据 Prefill/Decode 阶段选择不同的数据路径，并在安全
> 调度边界动态调整 KV 物理驻留与专家驻留，从而提高可服务容量和 SLO goodput？

当前可诚实支撑的系统范围是：

- 单机、单 GPU；
- phase-batched serving，即一个 scheduler step 不混合新的 Prefill 和旧请求 Decode；
- Qwen3-30B-A3B-FP8 使用 `TRITON_ATTN` 和 Prefill/Decode `FULL_SEPARATE` CUDA Graph；
- 专家计划按层对称：每个 MoE 层卸载相同数量的冷专家；
- 拓扑变化只在 scheduler idle 或等价安全 epoch 执行；
- 未支持状态 fail-closed，不依赖隐式 UVM fault 修复。

除非 E11 完成 phase-epoch continuous batching，否则论文不得声称支持通用 vLLM
continuous batching。除非新增多 GPU 结果，否则不得外推到 TP/EP 多卡部署。

## 2. 当前受测系统

### 2.1 阶段专用权重路径

系统为同一组专家维护稳定的原始 Parameter VA：

```text
Prefill
  pinned CPU source -> upload VA -> GPU backing
                                  -> Parameter VA 读取

Decode
  Parameter VA -> CPU-backed expert pages remote-read
```

Prefill 的 upload VA 和模型读取的 Parameter VA 虚拟地址不同，但在 `sealed` 状态映射
到相同 GPU backing。进入 Decode 时，PARK 将 Parameter VA 的 consumer alias 切换到
CPU-backed 权重页；下一轮 Prefill 再 RESUME。Graph 始终持有稳定 VA。

### 2.2 双 Graph 与可逆拓扑

当前实现分别捕获 Prefill alias Graph 和 Decode remote Graph，并维护：

```text
sealed <-> remote_parked
```

切换使用定制 UVM 的 MAP/SEAL/PARK/RESUME/REBIND/RESIZE ABI。Graph replay 期间不执行
页表修改；Graph、PTE、session、backing 和 residency generation 必须一致，否则拒绝
发布新状态。

### 2.3 固定 Plan Class 与差集切换

系统支持少量固定容量档位，例如 C2/C4/C6/C8。启动时建立全生命周期稳定的 upload、
pinned source、Parameter VA 和逻辑 slot envelope，并为各 class 保留独立 GraphExec
表和地址证书。正常切换只处理：

```text
retained = old ∩ new
added    = new - old
removed  = old - new
```

retained backing/session 原位复用；只 refill 发生 `slot -> expert` 变化的 source；切换
不执行 DESTROY、全量 finalize 或 CUDA Graph recapture。`STATIC_ALIAS_REBIND` 与
`RESIZE` 具有 BEGIN/COMMIT/ABORT 和反向回滚。

C2/C4/C6/C8 是 manifest 中的 Graph/slot 容量 class，不应仅凭名称推断其 GPU HBM
含义。KV target 到 class 的映射必须由实测预算生成并冻结。

### 2.4 三队列 KV pool 与统一控制器

KV tensor 和最大 VA 一次建立，运行时只改变物理驻留：

- GPU/free：GPU resident，可立即分配；
- GPU/backup：GPU resident，普通 free 不足时才开放；
- CPU/offline：CPU resident，grow 成功后进入 GPU 队列。

高压路径：

```text
backup 被使用 / preemption / 高水位
  -> 取消 pending shrink
  -> 每层对称地释放一轮或多轮冷专家
  -> KV target 按倍增或预测需求增长
  -> CPU/offline KV 迁入 GPU
  -> 补回 backup reserve
```

低压路径：

```text
R = max(实际峰值 A, 下一轮预测 P) + backup + 预测误差保护 E

连续三轮满足 R <= T/2
且无 backup 使用、preemption、UNKNOWN block
  -> 生成 shrink_pending
  -> 等待安全边界
  -> quarantine 空闲 tail
  -> GPU -> CPU migration
  -> target 减半但不低于下界
  -> 每层对称恢复此前由运行时卸载的专家
```

控制器统一处理预测、backup、preemption、实际峰值、上下界、最短驻留时间、迟滞和
prediction error，避免旧 low/high 控制器与 backup 控制器相互覆盖。

### 2.5 GPU 热度统计与 CPU 计划更新

GPU 复用 router assignment，按层累计专家 token 命中数；两个 heat buffer 轮换，CPU
维护 EWMA，并周期性生成 class 内候选集合。当前已实现同步 maintenance RPC 读取已轮换
buffer，以及同容量 class 内的冷热专家交换。

尚未完成的是将 pinned D2H 热度摘要完全异步接入请求结果通道。因此论文可以测量现有
同步维护开销，但在该路径完成前不能宣称热度返回与推理完全重叠。

## 3. 当前证据台账

### 3.1 已完成的 qualification

| 能力 | 当前证据 | 已证明 | 未证明 |
|---|---|---|---|
| 三系统兼容性 E0 | DeepSeek-V2-Lite BF16，ours/KT/MoE-Infinity 输出 hash 一致 | 三方能通过统一 OpenAI 接口执行冻结请求 | 三方正式性能排名 |
| 双 Graph 与 PARK/RESUME | Qwen B1/B4、2/4/6 GiB remote Decode | Prefill/Decode Graph、VA 与输出正确 | mixed batch、长期在线稳定性 |
| 固定 C2/C4/C6/C8 class | DeepSeek 单进程 `C2 -> C4 -> C6 -> C8 -> C2` | session/backing 差集复用、Graph capture `8 -> 8`、输出一致 | Qwen 四 class 正式 E2E、在线自动选 class |
| 物理 KV pool | Qwen 105 <-> 1,023 blocks | 固定 VA、真实 GPU/CPU residency、无 recapture、HBM 可复用 | 长时间控制器稳定性 |
| 三队列双向闭环 | Qwen `24 -> 40 -> 20` | backup 触发 grow、三轮低压 shrink、每层卸载/恢复 1 个专家 | 大并发和重复峰谷 |
| 小规模路径等价 | Qwen dynamic 与 direct high/low | 到达同状态后无显著持续损失 | 论文级重复和置信区间 |
| 真实大容量路径等价 | Qwen `2 -> 4 -> 8 -> 10 -> 8 -> 4 -> 2 GiB` | 物理 KV、多轮逐层专家差集、双 Graph、输出/VA/清理 | 自动预测、跨 class、低切换延迟 |
| 热度控制面 | 128 experts、双 buffer、CPU EWMA | 计数正确和 class 安全选择 | 异步 pinned D2H、长期 workload 收益 |

主要证据入口见第 18 节。所有这些结果都是功能或单次路径 qualification；除非其 run
root 属于新冻结协议、重复数和统计方法满足本文要求，否则不能直接进入论文主表。

### 3.2 Qwen 2--10 GiB qualification 的准确含义

该实验使用 Qwen3-30B-A3B-FP8、`TRITON_ATTN` 和 FULL Prefill/Decode Graph：

| KV 档位 | GPU KV target | 每层 remote experts | 相对 base70 的增量 waves |
|---|---:|---:|---:|
| K2 | 2,146,959,360 B | 70 | 0 |
| K4 | 4,295,491,584 B | 80 | 10 |
| K8 | 8,589,410,304 B | 99 | 29 |
| K10 | 10,737,942,528 B | 108 | 38 |

stepped 与 direct 到达相同状态后的 Prefill、Decode median 和 Decode p95 差异绝对值
均小于 0.5%，Graph capture 在每个进程内保持 `2 -> 2`。这证明没有检测到路径历史导致
的持续性能损失。

它不证明切换已经足够快。增长阶段 38 个串行 expert waves 累计约 95.9 s，而三次纯
KV grow 合计约 2.21 s；下降阶段 expert restore 合计约 25.7 s，KV shrink 合计约
9.93 s。正式动态性能实验前，应把多轮 wave 合并为单次全层差集事务。

### 3.3 正式评测基础设施现状

`experiments/eurosys_moe_offload_evaluation/` 已具备：

- 不可覆盖 run root 和 provenance 收集；
- 预 tokenized streaming client；
- 统一 localhost OpenAI 请求边界；
- ours、KTransformers、MoE-Infinity server launcher；
- E0、E1、E2 独立 verifier；
- DeepSeek 与 Qwen 的 calibration/validation/evaluation/shift fixtures；
- E2 的 workload/concurrency cell 描述和 equal-HBM 检查框架。

当前正式状态是：

- E0 已在 v2 run root 完成 provenance 补登记；三系统共同 validation 请求的 32-token
  hash 一致，KT eager/Graph 和 placement、MoE-Infinity counters、ours 的
  VA/PTE/residency/fault/cleanup 证书均已登记；
- E1 已完成三系统各 5 个独立 process blocks，共 15 blocks 和 150 条 measurement
  请求，balanced cyclic order、C1 和每分钟最多 10 次请求启动均通过；
- E1 的 10-prompt broad-generation gate 未通过：ours 和 KT 各自五次稳定，
  MoE-Infinity 仅 2/10 prompts 在五次间稳定，且 0/10 prompts 在全部 15 blocks 上
  一致；因此现有数字只能作为描述性 anchor，不能进入跨系统 speedup 表；
- E2 只有脚本、配置和 verifier，尚无完整正式矩阵；
- `eurosys-pa-v1-20260823` 已冻结，不得在文档更新后悄悄复用；v2 正式 run root 为
  `eurosys-pa-v2-20260824T142031Z-e0-e1-formal`。

### 3.4 当前 P0 缺口

1. 将多轮 expert release/restore 合并为单次全层差集事务；
2. 完成长期 workload 中 KV target、backup、plan class 和热度计划的同控制器闭环；
3. 完成或明确放弃 mixed Prefill/Decode phase-epoch scheduler；
4. 为 Qwen 固定 Graph class 做正式四档整模型 qualification；
5. 完成 E1 teacher-forced 数值门和 E2 equal-HBM 主矩阵；
6. 增加 6--24 小时 soak、故障注入和重复峰谷；
7. 若要声称热度采集与请求完全重叠，完成 pinned asynchronous D2H；
8. 多 GPU 仍未实现，保持 limitation。

## 4. 论文主张与 Research Questions

### 4.1 建议主张

| 主张 | 必需证据 | 当前状态 |
|---|---|---|
| C1：阶段专用 copy/remote 路径优于单一路径 | 同模型、同专家坐标的 hybrid/always-copy/always-remote crossover | 待正式实验 |
| C2：稳定 VA 支持可逆拓扑和无 recapture Graph | VA fingerprint、Graph counter、PTE/residency、输出 | qualification 已有，需重复统计 |
| C3：固定 class 差集切换显著低于全量重建 | full rebuild、差集 class switch、batched wave 对比 | 已有 pilot，需 Qwen/重复实验 |
| C4：三队列与统一控制器提高 KV burst goodput | fixed/oracle/automatic trace、OOM、SLO、HBM timeline | 单次双向已过，长期待做 |
| C5：热度驱动计划能适应 workload shift | static、EWMA、oracle、adversarial shift | 控制面已实现，E2E 待做 |
| C6：同等完整系统 HBM 下优于外部基线 | 统一 client、equal measured HBM、5 process blocks、CI | 尚无正式数据 |

C6 是条件性主张。如果完整 E2 不能满足预算和计时公平性，应删除“优于外部系统”的
总排名，只保留机制和系统容量主张。

### 4.2 Research Questions

- RQ1：Prefill-copy/Decode-remote 在什么 prompt、output、batch 下优于单一路径？
- RQ2：稳定 VA、双 Graph、grouped PTE 和 PARK/RESUME 分别贡献多少？
- RQ3：固定 Plan Class 差集切换相对全量重建节省多少，剩余开销在哪里？
- RQ4：KV/专家共享 HBM 能否提高可服务并发、降低 OOM 并提升 SLO goodput？
- RQ5：backup reserve 能吸收多大的突发，保留多少 HBM 才最合适？
- RQ6：GPU heat + CPU EWMA 在 workload shift 后多久恢复有效专家计划？
- RQ7：remote-read 在什么并发下先受 PCIe、CPU 内存或 GPU stall 限制？
- RQ8：系统在长时间峰谷、失败注入和路由漂移下是否保持正确并释放资源？
- RQ9：在相同 measured peak HBM 下，本方案相对 KTransformers、MoE-Infinity 和
  vLLM 基线的 TTFT、TPOT、throughput 与 goodput 如何？

## 5. 模型、平台与 claim 分工

### 5.1 核心机制主模型：Qwen

核心机制、动态 KV/专家协同和扩展性主模型固定为：

```text
Qwen3-30B-A3B-FP8
48 MoE layers
128 experts per layer
TRITON_ATTN
Prefill/Decode FULL_SEPARATE CUDA Graph
```

正式专家计划必须在每层选择相同数量 `k` 的冷专家：

```text
|cold_experts(layer_i)| = k, for every MoE layer i
```

专家 ID 可因每层热度不同而不同，但不得用“只卸载前几层”作为主实验。层不均衡方案只
可作为明确标注的诊断消融，不进入主结论。

### 5.2 三系统共同模型：DeepSeek

MoE-Infinity 当前不支持 Qwen block-FP8，因此三系统外部比较使用
DeepSeek-V2-Lite-Chat BF16：

| 模型 | ours | KTransformers | MoE-Infinity | 用途 |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B-FP8 | 支持 | 支持 | 不支持 | 核心机制、动态策略、ours/KT 扩展 |
| DeepSeek-V2-Lite BF16 | 支持 | 支持 | public-demand 支持 | 三系统 equal-HBM 表 |

DeepSeek 外部表不能替代 Qwen 核心机制实验；Qwen 表也不能给 MoE-Infinity 填入推测值。
unsupported 必须写 N/A 和原因。

### 5.3 硬件平台

Platform A 为当前必需平台：

- NVIDIA GeForce RTX 5090 32 GiB；
- Intel Xeon Gold 6530，当前可见 14 physical cores；
- 单 NUMA node；
- 固定 CPU affinity、driver、kernel、CUDA、PyTorch 和功耗采样。

Platform B 数据中心 PCIe GPU 为加分项，不是当前硬门槛。若可用，选择 A100 40/80 GiB、
L40S 48 GiB 或等价设备，只重复关键 crossover、E2 子集和动态 trace。

## 6. 基线与公平性协议

### 6.1 本方案内部变体

- `ours-full`：阶段专用双 Graph + 稳定 VA + 固定 class + 三队列控制器；
- `ours-static`：固定 KV target 和固定专家计划；
- `ours-no-backup`：保留自动预测但关闭 GPU backup；
- `ours-no-heat`：容量动态，但 class 内专家集合冻结；
- `ours-eager`：相同 topology，关闭 CUDA Graph；
- `ours-always-remote`：Prefill/Decode 都 remote-read；
- `ours-always-copy`：每阶段按相同专家集合搬运；
- `ours-full-rebuild`：旧的全量 plan/session/Graph 重建路径，仅用于切换消融；
- `ours-diff-class`：固定 Graph class 的 added/removed/retained 差集路径。

### 6.2 外部与辅助基线

- all-GPU 或 maximum-resident：性能上界和正确性 reference；
- vLLM native pinned UVA；
- vLLM copy-on-forward；
- KTransformers eager/Graph + frequency placement；
- KTransformers exact-coordinate placement，用于 Track A；
- MoE-Infinity public-demand；
- MoE-Infinity public-speculative 只有 prediction/correction/enqueue 计数均非零时才可用；
- oracle KV/expert controller，只作策略上界。

不得把公开 MoE-Infinity demand-only 结果称为 paper EAM/EAMC。compatibility patch、
extension hash、cache/store 配置和 `>256` token fallback 必须公开。

### 6.3 三条评测 Track

#### Track A：相同专家坐标

目的：隔离数据路径和执行机制。

- 同模型、精度、token IDs 和逐层冷专家坐标；
- 相同 KV bytes、batch、output length；
- 报告精确 selected bytes、GPU hot bytes、CPU bytes 和 alias backing；
- Qwen 比较 ours、vLLM 和 KTransformers；DeepSeek 可增加 MoE-Infinity cache manifest。

#### Track B：相同完整系统 HBM

目的：比较系统在相同资源约束下的最佳合法配置。

- 对齐 measured peak HBM，而不是名义 offload GiB；
- 各系统只在 calibration split 调参；
- evaluation 时 placement、threads、cache ratio、Graph sizes 全部冻结；
- 主指标为 TTFT/TPOT/goodput Pareto frontier。

#### Track C：动态策略

目的：评价 KV/专家控制器，而不是外部系统总排名。

- fixed-low、fixed-high、oracle、auto、auto+backup、full joint controller；
- 相同 arrival trace 和总 HBM 上界；
- 重点报告 OOM、admission、queueing、迁移、抖动与 SLO goodput。

Track A/B/C 必须分表，不能把 Track B 的系统差距解释为单一 PTE 机制收益。

## 7. HBM 预算与档位命名

### 7.1 避免三类“GiB”混淆

正式 artifact 必须区分：

- `Kx`：GPU-resident KV prefix，例如 K2/K4/K8/K10；
- `Cx`：固定 Graph/slot Plan Class，由 class manifest 定义；
- `Hx`：完整系统 measured peak HBM cap；
- `selected_expert_bytes`：计划选择的逻辑专家参数字节；
- `physical_backing_bytes`：去重后真实 GPU backing；
- `remote_expert_bytes`：CPU resident 且被 GPU remote-read 的专家字节。

不得把 C8、8 GiB remote experts 和 8 GiB KV 当作同一个量。

### 7.2 完整预算账本

每个 case 必须保存：

```text
usable HBM
  = device/Torch usable bytes
  - dense and always-hot weights
  - current GPU expert bytes
  - retained alias backing
  - active KV prefix
  - CUDA Graph pool/workspace
  - allocator fragmentation reserve
  - safety headroom
```

同时记录 NVML peak、PyTorch allocator、driver residency 和 native CUDA allocation。
只报告“卸载 4 GiB”而不报告 retained backing 和 Graph pool，不具备 equal-HBM 资格。

### 7.3 预算容差

沿用预注册规则：

```text
tolerance = max(3% * target_peak_HBM, 256 MiB)
```

超过容差的 case 保留 artifact，但从 equal-HBM speedup 中排除。不能看到性能后修改
容差或重新选择 comparison target。

## 8. Workload 设计

### 8.1 冻结 split

使用 calibration、validation、evaluation 和 shift 四个互斥 split。每个 fixture 保存
完整 token IDs、来源索引和 SHA-256：

- calibration：cold expert、class mapping、baseline threads/cache 的唯一调参来源；
- validation：冻结前检查；
- evaluation：只产生最终数据；
- shift：领域和长度分布变化，用于热度自适应。

现有 v1 DeepSeek/Qwen fixtures 可以作为生成输入或在新 manifest 中重新登记，但不得
因为已经看过 qualification 结果而在 evaluation split 上重选专家。

### 8.2 受控长度矩阵

| Workload | Prompt | Output | 主要问题 |
|---|---:|---:|---|
| short dialogue | 128 | 32 | TTFT 与切换固定成本 |
| ordinary chat | 512 | 128 | 常见 serving |
| medium context | 2,048 | 128 | Prefill-copy/remote crossover |
| RAG/summary | 8,192 | 128 | 长 Prefill 与 Graph/headroom |
| long generation | 512 | 512 | Decode remote 摊销与 PCIe |

每个长度点先做 backend/Graph support preflight。RTX 5090 上若 8K FULL Graph 不满足 HBM
预算，应明确标为 Graph N/A，并报告 eager 结果；不能静默切换后仍称 FULL Graph。

### 8.3 并发和到达过程

- strict batch：B1/B4/B8；
- closed loop：C1/C4/C8/C16，资源允许时 C32；
- open loop：Poisson 与冻结 trace replay；
- load：各系统独立饱和吞吐的 20/40/60/80/90%；
- open-loop 每点 20--30 分钟，至少 5 个 seed。

同时报告相同绝对 arrival rate 和相同相对饱和度，避免因系统容量不同而只展示 OOM。

### 8.4 动态压力 trace

至少包含：

```text
low steady
-> short burst consuming backup
-> sustained high pressure
-> three valid low-pressure rounds
-> idle shrink
-> repeated high/low cycles
-> workload route shift
```

Qwen 大容量 trace 使用 K2/K4/K8/K10 上下界；每个 tier 的逐层专家数量由 calibration
生成，不机械复用 qualification 的 70/80/99/108。另保留该映射作为固定复现实验。

## 9. 统一计时边界与指标

### 9.1 用户可见边界

所有系统由同一 localhost、预 tokenized streaming client 测量：

- TTFT：request body 发送完成到首个生成 token 到达；
- ITL：相邻生成 token 到达间隔；
- TPOT：排除首 token 后的平均时间；
- E2E：request body 完成到最终 token；
- throughput：成功输出 tokens / wall time；
- goodput：满足 workload SLO 的请求吞吐。

主表不计 tokenizer 和连接建立；真实文本 API 开销作为附录。不能用 custom Engine step
与 KT HTTP token interval 直接相除。

### 9.2 内部 breakdown

至少分解：

```text
TTFT
  = queue
  + RESUME
  + Prefill H2D
  + Prefill compute
  + PARK / remote map / residency audit
  + first Decode / sampling / serialization
```

动态切换还要单独记录：

- expert candidate selection；
- source refill；
- driver REBIND/RESIZE；
- expert release/restore residency；
- KV CPU->GPU 或 GPU->CPU migration；
- Graph class activation；
- rollback/reconciliation。

稳态结果可以扣除 RESUME/PARK 做机制分析，但论文主 TTFT/E2E 必须包含全部用户可见开销。

### 9.3 必报指标

用户指标：TTFT、TPOT/ITL、E2E、throughput、goodput、timeout/OOM/failure。  
资源指标：peak/steady HBM、CPU DRAM、pinned bytes、PCIe RX/TX、GPU/CPU utilization、
power 和 energy/token。  
机制指标：H2D/remote bytes、Graph capture/replay、PTE groups、PARK/RESUME、KV migrated
bytes、backup hit、prediction error、plan hit rate、wave count、class switches、rollback、
UNKNOWN residency 和 UVM fault。

## 10. 正式实验矩阵

### E0：兼容性与正确性

状态：v2 基础兼容性 qualification 已完成并通过；论文级数值正确性门仍待完成。

检查模型/精度、冻结输入、输出、Graph/eager 路径、MoE-Infinity counters、KT placement、
ours VA/PTE/residency/fault/cleanup。Unsupported 写 N/A。

2026-08-24 结果：三系统同一 validation 请求得到相同 32-token hash；KT eager/Graph
各 10/10 成功且逐条一致，26 层均加载 54 个 frequency experts；MoE-Infinity request、
token 和 KV cleanup counters 一致；ours 的 UVM 机制证据按内容 hash 重新登记。DeepSeek
maximum-resident 下 ours CUDA Graph 因 32-GiB HBM headroom 不足写为 N/A，
MoE-Infinity 公开 server 未暴露 Graph 模式也写为 N/A。

### E1：框架开销与 maximum-resident anchor

使用统一 client，在不发生动态切换的 maximum-resident 配置下测量 ours、KT 和
MoE-Infinity。每系统 5 个独立 server blocks，balanced order。该实验给出框架、HTTP、
scheduler 和 sampling 的差异下界。

状态：执行完成。三系统各 5 个独立 blocks，15/15 通过执行、限速、清理和 balanced
order 检查。均值为：ours TTFT/TPOT 129.47/54.22 ms，KT 408.22/9.96 ms，
MoE-Infinity 1348.69/306.98 ms；peak HBM 分别为 30,997/31,559/29,379 MiB。

这些数字暂不进入论文性能表。虽然所有 15 blocks 使用相同冻结输入和请求体，E1 的
exact generation gate 未通过，且 MoE-Infinity 存在进程间输出漂移。必须先完成协议中
预登记的 teacher-forced logits relative-L1、top-1 agreement 和 perplexity gate；此外
E1 本身不是 equal-HBM 比较，主排名仍由 E2 承担。

### E2：equal-HBM 主性能矩阵

Track B 固定策略，不启用在线自适应，避免控制器行为混入系统基线：

- DeepSeek：ours、KTransformers、MoE-Infinity public-demand；
- Qwen：ours、KTransformers、vLLM UVA/copy；
- HBM tiers：根据实际 peak 对齐，名义候选 0/2/4/6/8 GiB offload；
- workloads：short、ordinary、RAG、long generation；
- concurrency：C1/C4/C8/C16；
- 每个 cell 5 个独立 process blocks。

输出 equal-HBM TTFT/TPOT/goodput Pareto frontier。完整预算不匹配的点只作为附录。

### E3：Prefill/Decode 路径 crossover

Qwen Track A 固定同一逐层专家集合，比较：

- hybrid Prefill-copy/Decode-remote；
- always-copy；
- always-remote；
- vLLM pinned UVA/copy；
- KT exact-coordinate CPU experts。

扫描 prompt、output 和 B1/B4/B8，输出最优策略热力图和 break-even。该实验直接支撑
阶段专用路径，而不是依赖整系统总排名。

### E4：并发与 PCIe 饱和

扫描 C1--C32，记录 remote bytes、PCIe bandwidth、GPU stall、CPU bandwidth 和 TPOT。
定位 ours remote-read、KT CPU compute 和 MoE-Infinity fetch/cache 的首个持续退化点。

### E5：稳定 VA、Graph 与 PTE 消融

| 变体 | 隔离因素 |
|---|---|
| eager remote | Decode Graph 收益 |
| remote Graph | stable Parameter VA |
| eager alias Prefill | Prefill Graph 收益 |
| dual Graph | 完整阶段路径 |
| DESTROY/rebuild | retained session/backing 价值 |
| PARK/RESUME | 可逆拓扑 |
| sequential PTE | grouped replace 收益 |
| grouped PTE | 当前驱动路径 |

报告 output、VA、capture counter、launch、PTE、remote map、audit 和 retained backing。

### E6：Plan Class、expert wave 与切换摊销

比较：

1. 旧全量 plan/session/Graph rebuild；
2. 固定 C2/C4/C6/C8 差集切换；
3. 当前串行逐 wave release/restore；
4. 合并后的全层 batched difference transaction。

分别测量 retained/added/removed、source refill、backing bytes、REBIND、residency、Graph
activation 和 rollback。扫描 plan difference 5/10/25/50/100%。

定义真实摊销点：

```text
requests_to_amortize
  = transition_cost
    / (per-request old-plan cost - per-request new-plan cost)
```

分母非正时报告“无 break-even”。95.9 s 的现有 Qwen 串行 wave 必须作为优化前基线保留。

### E7：专家热度、选择与 workload shift

所有方法保持逐层专家数量相同：

- per-layer frequency-cold；
- per-layer random；
- frozen static plan；
- GPU heat + CPU EWMA；
- oracle evaluation route，仅作上界；
- hot-adversarial shift。

报告 plan hit rate、每层 Jaccard overlap、remote/H2D bytes、切换数、EWMA 收敛时间、
性能恢复和维护 RPC/D2H 开销。异步 D2H 未完成时必须标注 synchronous maintenance。

### E8：MoE-Infinity cache/prefetch

仅在 DeepSeek 共同模型上运行：cold cache、standard warm cache、public-demand、合法的
public-speculative、可选 SSD tier。记录 cache hit、fetch bytes、prediction/correction、
wasted prefetch 和 store I/O。paper EAM 数据只引用原论文，不与本机结果混算 speedup。

### E9：KTransformers placement 与 CPU 消融

比较 per-layer frequency/exact coordinates、eager/Graph、7/14/28 threads、NUMA、overlap
和 mixed chunk。所有调参在 calibration split 完成并冻结。长 Prefill Graph 不支持或
崩溃时保留错误 artifact，报告 eager，不删除失败点。

### E10：三队列与联合 HBM 控制器

这是当前实现更新后的动态核心实验。比较：

- fixed K2；
- fixed K10；
- oracle tier；
- auto grow/shrink，无 backup；
- auto + fixed backup；
- auto + backup + expert waves；
- full joint：KV target + class + heat-based expert set。

使用 K2/K4/K8/K10 重复峰谷和 open-loop trace，验证：

- backup 是否降低 burst admission/TTFT；
- 三个有效低压轮后才 shrink；
- 任意高压事件取消 pending shrink；
- target 上下界和最短驻留时间有效；
- 每层专家数量始终相同；
- Graph 不 recapture；
- stepped 与 direct 到达同状态后的稳态一致；
- 迁移失败时 quarantine/UNKNOWN/reconciliation 正确；
- 最终 goodput、OOM 和 HBM timeline。

显式 2--10 GiB stepped/direct qualification 作为 E10 preflight；正式 E10 必须由自动
控制器根据 trace 决策，不能直接传入人工 target 后称为自动策略。

### E11：Continuous batching / phase epoch

构造新请求持续 Prefill、旧请求持续 Decode 的 trace，比较：

- vLLM 默认 continuous batching；
- ours phase-epoch batching（若实现）；
- ours remote-only fallback；
- KT/MoE-Infinity 原生 scheduler。

报告 queueing、batch formation、GPU idle、starvation 和 goodput。若系统仍 fail-closed，
E11 的合法结论是“scope 限定为 phase-batched serving”，而不是伪造支持结果。

### E12：鲁棒性、soak 与故障注入

- 6 小时为最低，投稿前目标 24 小时；
- 至少 1,000 次 PARK/RESUME；
- 重复 K2/K4/K8/K10 峰谷；
- 合法 barrier 注入 MAP/PARK/RESUME/REBIND/KV migration 失败；
- 路由 shift 和 Graph headroom 压力；
- 检查输出、Xid、PDE fault、UNKNOWN、poison、session/backing 和 GPU 泄漏；
- 验证 custom -> stock module 生命周期。

## 11. 正确性与 fail-closed 门

每个正式 case 至少检查：

- greedy token IDs 与同框架 reference 一致；
- 跨框架使用 teacher-forced logits/top-1/perplexity 门，不能只看文本相似；
- Parameter、upload、source 和 KV VA fingerprint 稳定；
- Graph certificate、capture/replay counter 与 active class 一致；
- selected、remote、GPU backing、KV GPU/CPU bytes 精确；
- 三队列互斥且总 block 守恒；
- 每层 remote expert 数量相同，ID 属于冻结计划；
- removed expert 回收前不再被旧 epoch 使用；
- unknown residency 和 manual migration fault 为 0；
- 回滚后 driver、residency、lifecycle 和 generation 完整恢复；
- shutdown 后 session/backing/Graph/GPU/UVM client 全部释放；
- kernel log 无 Xid、`FAULT_PDE`、Oops、BUG 或 panic。

任何 correctness、fault、dirty GPU 或 budget gate 失败的 case 均不得进入 speedup，
但 artifact 必须保留。

## 12. 统计方法与环境控制

### 12.1 重复

- 机制 microbenchmark：每点至少 10 个独立冷进程；
- 系统主矩阵：每个 model/system/HBM tier 至少 5 个独立 server process；
- open-loop/dynamic trace：至少 5 个 seed；
- 固定 warmup 不计入性能，但计入正确性。

同一 server 内 48 个 Decode step 是时间样本，不是 48 个独立系统重复。

### 12.2 顺序与报告

- 使用 balanced block order 或 Latin square；
- tier 在奇偶 repeat 中正序/逆序；
- 系统启动顺序轮换；
- 报告 median、p95、p99 和 bootstrap 95% CI；
- speedup 使用 paired process-block bootstrap；
- 跨 workload 几何平均，同时保留原值；
- 不删除性能 outlier；只按预注册 invalid reason 排除；
- 同时报告成功率。

### 12.3 环境

每 100--200 ms 记录 clock、power、temperature、throttle reason、PCIe、NVML HBM 和
GPU utilization。记录 CPU affinity、NUMA、page-locked bytes、driver/kernel/CUDA、
所有 commits、dirty patches 和二进制 SHA-256。

## 13. SLO 与 goodput

沿用 v1 的初始 SLO，允许在 v2 冻结前只根据 calibration 调整一次：

| Workload | TTFT SLO | TPOT SLO |
|---|---:|---:|
| short dialogue | 2,000 ms | 100 ms |
| ordinary chat | 5,000 ms | 150 ms |
| medium context | 10,000 ms | 175 ms |
| RAG/summary | 30,000 ms | 200 ms |
| long generation | 5,000 ms | 150 ms |

SLO 冻结后不得根据 evaluation 结果修改。goodput 同时要求请求成功、输出正确并满足
TTFT/TPOT 两个门。

## 14. Artifact 与新协议

### 14.1 新鲜度规则

`eurosys-pa-v1-20260823` 的 protocol 和 provenance 已冻结，保存其历史意义。由于本文、
实现和实验矩阵已经更新，正式新结果必须：

1. 生成新的 protocol ID；
2. 重新计算本文和技术设计 SHA-256；
3. 重新登记或生成 Qwen/DeepSeek fixtures；
4. 冻结 plan-class manifests、逐层 expert plans 和 KV/class mapping；
5. 使用新的不可覆盖 run root；
6. 只从顶层 `verification.json: status=ok` 生成论文表图。

现有 qualification artifact 可以作为 preflight input，但不能复制进新 run root 冒充
正式重复。

### 14.2 目录

```text
experiments/eurosys_moe_offload_evaluation/artifacts/<paper-run-id>/
  protocol.json
  run_status.json
  provenance/
    git_commits.json
    dirty_patches/
    binaries.sha256
    environment.txt
    model_manifest.json
    fixture_manifest.json
    plan_manifests/
  cases/<experiment>/<track>/<system>/<model>/<budget>/<repeat>/<attempt>/
    command.txt
    env.json
    client.json
    server.log
    nvml.csv
    pcie.csv
    residency.json
    graph_certificate.json
    controller_trace.jsonl
    kernel.log
    returncode.txt
    verification.json
  verification.json
  summary.json
```

失败重试必须创建新 attempt。顶层 verifier 检查完整 cell 数、budget eligibility、输出、
fault、provenance 和重复结构后才发布 `status=ok`。

## 15. 执行优先级

### P0：让动态路径可测

1. 合并串行 expert waves；
2. 为 Qwen 完成固定 class 四档 qualification；
3. 跑重复峰谷和最短 1 小时 joint-controller smoke；
4. 决定 E11：实现 phase epoch，或冻结 phase-batched claim；
5. 保持已冻结的 v2 protocol/fixtures 不变，所有后续 reporting amendment 单独登记 hash。

### P1：核心机制

1. E3 phase crossover；
2. E5 Graph/VA/PTE 消融；
3. E6 full rebuild/difference/batched wave；
4. E7 heat/shift；
5. 每点 10 个冷进程。

### P2：系统主矩阵

1. E1 五重复已完成；收尾 teacher-forced 数值门；
2. E2 DeepSeek 三系统 equal-HBM；
3. E2 Qwen ours/KT/vLLM；
4. E4 concurrency/PCIe；
5. 生成 Pareto 和 CI，不提前选择赢家。

### P3：动态与鲁棒性

1. E10 自动 K2/K4/K8/K10 长 trace；
2. backup sizing 与预测误差；
3. E11 scheduler；
4. E12 soak/fault injection；
5. 可选 Platform B。

## 16. 最小可投稿门槛

至少满足：

1. Qwen 核心机制 E3/E5/E6，逐层对称计划；
2. 一个完整三系统共同模型 equal-HBM 主表；
3. Qwen ours/KT/vLLM 扩展表；
4. E1/E2 每主点 5 个 process blocks；
5. TTFT/TPOT/goodput、HBM、PCIe、CPU/GPU utilization 和 95% CI；
6. all-GPU、UVA、copy、always-remote、always-copy 和 eager/Graph 消融；
7. 固定 class 差集与 batched wave 切换结果；
8. 自动三队列至少 5-seed 重复峰谷 trace；
9. heat/shift 结果，或从主 claim 删除自适应专家；
10. mixed batching 实现，或在标题、摘要和结论中明确 phase-batched 范围；
11. 6 小时以上 soak、故障注入、fault scan 和清理验证；
12. 完整、可独立复验的 artifact。

若 E2 三系统公平性或正确性不满足，论文仍可作为机制论文投稿，但必须删除跨系统总
speedup 主张，改为稳定 VA、阶段专用路径和动态 HBM 协同的机制评测。

## 17. 结果解释规则

- 不把 qualification 的单次数字写成论文 speedup；
- 不把 stepped/direct 稳态等价解释为切换免费；
- 不把 K2/K4/K8/K10 与 C2/C4/C6/C8 混淆；
- 不把逻辑 selected bytes 当作真实 HBM freed；
- 不把不同计时边界相除；
- 不把 unsupported 当作 0 ms；
- 不把 MoE-Infinity public-demand 称为 paper EAM；
- 不在 evaluation split 上选择 cold experts 或 baseline 参数；
- 不隐藏 OOM、timeout、Xid、rollback 和 dirty-GPU attempt；
- 不把同 server 内 token steps 当作独立重复；
- 主 TTFT/E2E 必须包含 RESUME/PARK 和在线控制开销；
- 稳态 compute 和一次性 transition 必须分列；
- mixed batching、多 GPU、异步 heat D2H 未完成时保持对应 limitation。

## 18. 当前入口与证据

技术设计：

- [`moe_phase_aware_offload_technical_design.md`](moe_phase_aware_offload_technical_design.md)
- [`cudagraph_static_alias_transport.md`](cudagraph_static_alias_transport.md)

正式评测框架：

- [`../experiments/eurosys_moe_offload_evaluation/README.md`](../experiments/eurosys_moe_offload_evaluation/README.md)
- [`../experiments/eurosys_moe_offload_evaluation/protocol_template.json`](../experiments/eurosys_moe_offload_evaluation/protocol_template.json)
- [`../experiments/eurosys_moe_offload_evaluation/run_formal_e1.sh`](../experiments/eurosys_moe_offload_evaluation/run_formal_e1.sh)
- [`../experiments/eurosys_moe_offload_evaluation/run_formal_e2.sh`](../experiments/eurosys_moe_offload_evaluation/run_formal_e2.sh)

v2 E0/E1 正式结果：

- [`../experiments/eurosys_moe_offload_evaluation/RESULTS_E0_E1_20260824.md`](../experiments/eurosys_moe_offload_evaluation/RESULTS_E0_E1_20260824.md)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/summary.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/summary.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E0/verification.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E0/verification.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E1/output_divergence_diagnostic.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E1/output_divergence_diagnostic.json)

历史 qualification 的独立 artifact 已按用户要求清理。当前 E0 引用的四份机制证书已按
原 SHA256 收编进当前 v2 run；它们只用于保持当前结果的 provenance，不恢复为独立
历史实验：

- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/retained_evidence/manifest.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/retained_evidence/manifest.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/historical_results_cleanup.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/historical_results_cleanup.json)

## 19. 最终论文叙事

最合适的 EuroSys 主线是：

> 稳定虚拟地址使同一 MoE 模型能够在不替换 Parameter/KV tensor、也不重新捕获 CUDA
> Graph 的前提下切换阶段专用权重路径；固定 Plan Class 和差集事务降低专家拓扑更新
> 成本，三队列 KV pool 则把 GPU 上的短时 reserve、长期 KV 容量和专家 cache 纳入同一
> HBM 控制循环。

论文需要用 Qwen 证明机制和真实 2--10 GiB 动态范围，用 DeepSeek 共同模型完成合法的
三系统公平比较，并用长期 trace 证明控制器价值。现有结果已经证明关键机制可行，但在
batched wave、长期联合闭环、正式 equal-HBM 矩阵和 mixed batching 完成前，不应宣称
已经得到最终在线 serving speedup。
