# E2--E12 未解决问题报告（2026-08-27）

本文件只记录尚未满足、暂时无法运行或仍有阻塞条件的事项。已解决的 E5 headroom、
formal 进程数协议、E10 `unregistered_regions` 和测试夹具问题不在此处，见
[ERROR_FIX_LOG_E2_E12_REPRO_20260827.md](ERROR_FIX_LOG_E2_E12_REPRO_20260827.md)。

## 总览

| ID | 实验 | 阻塞/缺口 | 影响 |
|---|---|---|---|
| U-01 | E2 | 只有 ours/Qwen 单次 qualification | 不能形成 equal-HBM 跨系统主表或 CI |
| U-02 | E3 | 1 个冷进程；5 个长 Prefill cell N/A | 不能形成论文级 crossover/CI |
| U-03 | E4 | KVM 不暴露 CPU IMC bandwidth | 无法区分 CPU DRAM 与 PCIe/GPU stall 瓶颈 |
| U-05 | E6 | 已完成 difference 百分比；缺其余三类对照 | 尚无 full rebuild、逐 wave、全层 batched 与 break-even |
| U-06 | E7 | 只完成同步控制面 microbenchmark | 自适应专家/workload-shift claim 无 E2E 证据 |
| U-07 | E8 | 缺 fresh receipt，MoE-Infinity commit 漂移 | 无合法 cache/prefetch 数字 |
| U-08 | E9 | qualification greedy token 跨臂漂移 | 7 arms 可运行，但未通过正确性门，formal 未启动 |
| U-09 | E10 | 只完成资格测试，未跑 5-seed 自动 trace | 不能声称联合控制器提高 goodput |
| U-10 | E11 | 无 continuous-arrival 多系统 E2E | claim 必须限定为 phase-batched |
| U-11 | E12 | 无 6 h orchestrator / fault injection | 不能声称长时鲁棒性或故障恢复 |

## U-01：E2 equal-HBM 正式矩阵未完成

### 现状

本轮 artifact 只有 Qwen/ours、2-GiB KV、16 cells、一个进程重复。它通过自身
qualification，但计划要求：

- DeepSeek：ours、KTransformers、MoE-Infinity public-demand；
- Qwen：ours、KTransformers、vLLM UVA/copy；
- 实测完整 HBM 对齐的多个 tier；
- 每个 model/system/tier/workload/concurrency cell 5 个独立 server process。

### 已尝试

- 完成 Qwen/ours C1/C4/C8/C16 的 16-cell 运行和机器汇总；
- 检查实际 peak HBM、热点/冷点驻留、Graph、SLO 和 cleanup；
- E8 仍被 provenance 阻塞；E9 已补齐 fixture，但 qualification 被跨臂 greedy token
  漂移阻塞。

### 阻塞条件与影响

外部系统合法环境未同时就绪，且完整矩阵的重复尚未调度。当前结果不能进入跨系统
speedup 主表，也不能用 31,953/31,247 MiB peak HBM 代替逐系统 budget eligibility。

### 后续建议

先解决 U-07/U-08，冻结新的 protocol/run root，按 balanced order 完成每点 5 repeats；
顶层 verifier 必须检查 measured peak HBM 容差、teacher-forced correctness 和完整 cell
数量后再生成主表。

## U-02：E3 重复与长 Prefill coverage 不足

### 错误/跳过信息

5 个 custom cell 在运行前按已知平台错误 fail-closed：

```text
backend preflight: >=8192-token single-step Prefill triggers
TMA descriptor 700 / illegal-address on this platform
```

涉及 medium-context B4/B8 和 RAG B1/B4/B8。对应 N/A 文件位于：

```text
artifacts/e3-repro-20260827-attempt-01/custom/*/na.json
```

### 已尝试

- H/A/R 的 30 个可运行 case 全部成功；同 cell 输出 hash 一致；
- KT exact-coordinate 得到 13 个 measurement cells；
- verifier 明确输出 `process_blocks=1`、
  `publication_repetition_requirement_met=false`；
- 没有把不支持点静默切到 eager。

### 根因与阻塞条件

长单步 Prefill 的底层 TMA descriptor/illegal-address 问题尚未形成在当前平台安全可用的
Graph 路径。统计上也只完成一个独立进程，而计划要求每点至少 10 个冷进程。

### 影响与建议

当前 8×R、2×H 只能作为描述性结果。应先把 `>=8192` token 路径变为可验证的 Graph
实现或保留正式 N/A，再用 10 个冷进程 balanced arm order 重跑所有可支持点并计算 paired
bootstrap CI。

## U-03：E4 CPU DRAM 带宽不可用

### 完整错误信息

```text
KVM does not expose uncore IMC PMUs or server memory-controller PCI
configuration; Intel PCM reports no memory controllers
```

机器可读证据：
`artifacts/e4-repro-20260827-attempt-01/capability/cpu_dram_bandwidth.json`。

### 已尝试与根因

runner 检查了 Intel PCM/uncore capability；虚拟机只暴露 KVM guest 视图，没有 IMC PMU
或可枚举的 server memory controller。`sar`、NVML PCIe、GPM 和 GPU counters 已采集，
但它们不能替代完整 CPU DRAM read/write bandwidth。

### 影响与建议

C1--C32 无 TPOT 持续退化的结果仍有效，但不能判定瓶颈首先来自 CPU DRAM、PCIe 还是
细粒度 remote latency/GPU stall。建议在 bare-metal Platform B 上重跑相同 5-block
矩阵，冻结 PCM 版本并记录 IMC channels；本机 PCIe RX 仅作为 host traffic 下界。

## U-05：E6 其余 transaction 对照与 break-even 未运行

### 现状与已尝试

固定 `C2→C4→C6→C8→C2` qualification 通过。5/10/25/50/100% difference formal
也已完成 10 个独立进程，五档 outbound median 为 480.98/637.26/943.25/1400.21/
2303.34 ms。当前 source/residency 是 delta，但 REBIND 仍提交完整 alias map。尚未运行：

1. full plan/session/Graph rebuild；
2. 旧 95.9-s 串行逐 wave 的同机正式复测；
3. 真正全层 batched difference transaction；
4. 四类路径配对后的 break-even。

### 阻塞条件与影响

已有 difference plan generator，但没有冻结其余三条路径的同几何 runner，因此无法计算
`requests_to_amortize` 或与 95.9-s 旧串行 wave 基线做正式比较。五档结果只能归因于当前
“delta source/residency + full alias REBIND”实现。

### 后续建议

复用已冻结五档 plan，在相同 block order 中补 full rebuild、逐 wave 和全层 batched
transaction；每点 10 个冷进程，并加入稳态请求收益后计算 break-even。

## U-06：E7 workload-shift qualification 已运行，正式统计仍未完成

### 现状与阻塞

2026-08-29 已冻结并完整运行 frequency/random/static/EWMA/oracle/hot-adversarial 六种方法的
相同逐层专家数 E2E qualification：Qwen3-30B-A3B-FP8、48 层×4 dynamic slots、32 个
evaluation calibration 请求和每臂 32 个 code-shift 请求。成功 artifact 为
`e7-workload-shift-REPRO-20260829-attempt-03-qualification`，runner/verifier 均为 `status=ok`。

GPU heat + CPU EWMA 的 oracle-slot hit 从 36.98% 增至 88.54%，动态 remote assignments
相对 frequency-cold 减少 50.12%；但 32 requests 内未达到 95% 收敛门，同步维护累计
5.794 s，且请求中位延迟没有恢复。逐 arm demand-read bytes 仍无法由当前 counters 归因，
异步 pinned D2H 数据路径也尚未实现。

### 影响与建议

workload-shift 的功能性 E2E 缺口已经闭合，但本轮仅是单进程、固定 arm order、每臂单样本
qualification，不能替代正式统计。若投稿保留“自适应专家”性能主张，仍应完成至少 5 seeds、
平衡 arm order/独立冷进程和等重复数的 shift trace；在异步路径完成前必须明确 maintenance
为 synchronous。详见 `RESULTS_E7_WORKLOAD_SHIFT_QUALIFICATION_20260829.md`。

## U-07：E8 MoE-Infinity provenance 阻塞

### 复现命令与错误

```bash
bash experiments/broad_offload_compare/run_moe_infinity_qualification.sh preflight
```

返回码 1：

```text
error: STORE_BUILD_RECEIPT must point to a fresh extension build receipt
```

### 进一步只读核验

```text
expected commit: ba5651897a80d9c9b7a1500cef2c68adaa63db0f
current commit:  92c2bc9266e33bbed22405356ca7ca7e71ae8974
current checkout dirty lines: 0
```

runner 还冻结 compatibility patch SHA-256、runtime ELF SHA-256、extension SHA-256 和
receipt/build-log hashes。现有约 30-GiB DeepSeek store 有 `archer_index`、
`archer_param_0..2`，但没有 fresh `store_build_receipt.json`，因此无法证明 store 与当前
活动 extension/provenance 对齐。

### 已尝试、影响与建议

已检查 checkout、store 和 receipt 路径；没有绕过哈希门，也没有删除/重建 30-GiB
store。E8 cold/warm/public-demand/speculative cache 数据全部缺失。后续应在冻结 commit
和兼容补丁上重新构建 `_store.so`，生成 receipt/runtime fingerprint，再用
`rebuild_moe_infinity_store.sh` 明确验证或重建 store；这会改变外部 checkout/大型 store，
需单独授权和磁盘时间。

## U-08：E9 KTransformers 跨臂 greedy token 漂移

### 复现命令与完整错误

冻结 reference 已补为 `inputs/e9_sharegpt_selection_reference_20260828.json`，并完成
calibration/evaluation disjoint split。七个 qualification case 全部返回 0，但最终命令：

```bash
KTransformers/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/verify_formal_e9.py \
  experiments/eurosys_moe_offload_evaluation/artifacts/e9-formal-REPRO-20260828-attempt-01-qualification \
  --output .../verification.json
```

失败为：

```text
RuntimeError: exact output drift across E9 arms
```

### 已尝试与影响

baseline、threads7、threads28、eager、no-overlap 的 C1/C8 与 baseline exact；uniform
C1 0/1、C8 2/8 请求 exact；no-mixed C1 1/1、C8 7/8 exact。所有请求都成功输出固定 32
tokens，per-case runtime verifier、kernel fault scan 和 cleanup 均通过。token 对比显示
uniform C1 仅最后 3/32 tokens 漂移，但 C8 有 6/8 请求分叉；no-mixed C8 只有 request 0
分叉。placement/batch formation 会改变 CPU/GPU kernel 和归约形状，现有 artifact 没有
teacher-forced logits/top-1 证据，无法判定为可接受的舍入差异，也不能删除 exact gate。

### 后续建议

增加同 fixture 的 teacher-forced logits/top-1/perplexity 对照，定位首个分叉 token 前的
logit margin；若同框架规则仍要求 greedy exact，则修复 KT uniform/no-mixed 的数值或调度
非确定性后重跑 qualification。资格门通过前不得启动 5-block formal，现有性能数据仅用于
诊断。

## U-09：E10 自动联合控制器正式 trace 未运行

### 现状

已通过三队列单周期、dynamic/direct parity 和人工 stepped/direct 2--10 GiB
qualification。尚缺计划中的：

- fixed K2、fixed K10、oracle tier；
- auto grow/shrink 无 backup、fixed backup、backup+expert waves、full joint；
- K2/K4/K8/K10 重复峰谷和 open-loop trace；
- 至少 5 seeds、goodput/OOM/HBM timeline、quarantine/rollback failure case。

### 影响与建议

人工 target 到达同一状态的稳态等价不等于自动策略有效，也不包含在线 transition cost。
应冻结 arrival trace 和 seeds，由控制器自行决定 target/class/expert set；主 TTFT/E2E 必须
包括 RESUME/PARK、KV migration 和 expert wave，稳态 parity 另表报告。

## U-10：E11 continuous batching E2E 缺失

### 现状

只有 3 个 scheduler 单元测试；没有持续新 Prefill 与旧 Decode 混合到达的请求 trace，
也没有 default vLLM、remote-only、KT、MoE-Infinity 对照的 queueing/batch formation/
starvation/goodput 指标。

### 影响与建议

当前实现只能声明 phase-batched serving。若不实现 mixed continuous batching，标题、摘要、
实验和结论均需保留 scope 限定；若实现，应先补 epoch ownership/removed-expert safety 的
fail-closed tests，再运行至少 5 seeds E2E trace。

## U-11：E12 长 soak 与故障注入缺失

### 现状

完成的是 3 warmup + 1,000 CUDA Graph replay，持续约 46 秒 GPU 时间；它不是：

- 最低 6 小时（目标 24 小时）模型服务 soak；
- 1,000 次 PARK/RESUME；
- K2/K4/K8/K10 重复峰谷；
- MAP/PARK/RESUME/REBIND/KV migration barrier failure injection；
- route shift、headroom pressure、UNKNOWN/poison/reconciliation 验证；
- custom→stock 模块生命周期 orchestrator。

### 阻塞条件、影响与建议

仓库中没有把上述项目合并为本轮 protocol 的正式 orchestrator，直接循环 Graph smoke
不会覆盖目标。因而不能声称 6-h safety、故障恢复或无长期泄漏。建议先实现一个 fail-closed
orchestrator：每轮保存 controller/driver/GPU/kernel state，并在所有 fault point 验证回滚；
先做 1-h smoke，通过后再运行 6 h 和 24 h，不得把同一进程内 replay 当独立重复。
