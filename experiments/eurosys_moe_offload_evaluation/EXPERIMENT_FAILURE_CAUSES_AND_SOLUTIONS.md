# 实验错误原因与解决方案总览

最后更新：2026-08-28

本文汇总目前 E0--E12、C1/C16 direct-batch pilot，以及 E3 Prefill/Decode
phase-only 实验中已经遇到的主要错误。用途是后续复现或扩展实验时快速判断：

1. 现象属于实验配置、容量、分配时序、地址映射、框架接口，还是正确性问题；
2. 哪些问题已经修复并复测；
3. 哪些只能合法记为 N/A；
4. 哪些仍未解决，不能通过关闭断言或改变请求几何绕过。

本文是索引和操作手册。原始错误、代码差异和 artifact 证据仍以文末链接的专项记录为准。

## 1. 状态定义

| 状态 | 含义 | 后续处理 |
| --- | --- | --- |
| 已修复 | 根因已定位，最小修复已实施并通过定向或完整复测 | 可以按冻结配置复现 |
| 已规避 | 根因边界明确，但使用了不改变目标的受支持路径 | 报告中必须说明路径差异 |
| N/A | 当前平台或机制没有合法可运行配置 | 不产生性能数字 |
| 未解决 | 尚无通过正确性和实验契约的修复 | 保留失败证据，不能进入正式排名 |
| 警告 | 不影响本轮有效数据，但重复实验前应处理 | 记录并安排维护 |

以下情况不叫“修复”：

- 将 B8 拆成 `1+7`、两个 B4，或让部分请求先运行；
- 缩短 Prompt、输出长度或 KV，使原失败请求变简单；
- 在 Graph 实验中静默切到 eager；
- 删除 exact-generation、PTE/PDE、residency 或 cleanup 验收门；
- 只重试一次并把偶然成功解释为根因消失；
- 删除失败 artifact 或把未完成 attempt 纳入正式均值。

## 2. 首次失败时的固定排查顺序

### 2.1 先保存第一现场

优先保留：

- `command.txt`、环境变量和实际配置；
- `run.log`、`result.json`、`verification.json`、`returncode.txt`；
- `nvml.csv`、UVM log、kernel log；
- expert plan、exact-name 列表、fixture SHA256；
- module before/custom/after snapshot；
- 第一个 CUDA/Xid/PTE/PDE 错误，而不是进程退出后的连锁报错。

失败 attempt 使用新的不可覆盖目录重跑。不得修改原始 result 后伪装为成功。

### 2.2 检查实验契约

先确认问题不是测量方式错误：

- B1/B8/C16 是否在第一次 EngineCore step 前完整入队；
- Prefill step 是否只有新请求，Decode step 是否只有 cached requests；
- 是否出现 `1+(N-1)` 或 mixed Prefill/Decode；
- Prompt、Output、KV、`max_model_len` 和 batch 是否仍为冻结值；
- Graph 是否真的捕获和 replay，而不是参数存在但运行时为 `NONE`；
- 输出长度和 hash 是否通过。

如果契约错误，先修 runner；此时延迟数字没有解释意义。

### 2.3 再检查 GPU、模块和后台进程

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
modinfo -F srcversion nvidia-uvm
cat /sys/module/nvidia_uvm/srcversion
```

必须确认 GPU 0 MiB、无 compute process、无来源不明的 UVM client，并且实验开始时为
stock UVM。GPU 0 MiB 并不等于 UVM 无 client；孤立的 `nvidia-smi --loop-ms` sampler
也会阻止模块卸载。

### 2.4 区分四类显存问题

1. **容量估算错误**：程序认为能分配，实际创建 KV/MLA pool 时 OOM。
2. **总物理容量不相容**：hot experts、KV、Graph pool、非专家权重和必要 workspace
   之和超过 HBM；反复 rebalance 无法解决。
3. **分配顺序错误**：总容量可能足够，但 KV、Graph 或 warmup 临时分配先占用了应恢复
   hot experts 的空间。
4. **过度预留**：`max_num_batched_tokens`、allocator reserve 或不使用的 Graph shape
   大于正式请求需要。

只有第 2 类需要增加 cold experts 或合法减少某个冻结 budget；第 3 类需要 sentinel、
warmup 和 final reconcile；第 4 类应收紧到真实最大 shape。不要把四者混为“显存不足”。

### 2.5 判断 fault 发生在哪个阶段

记录故障是在：

1. 权重加载；
2. KV 建立；
3. kernel/autotune warmup；
4. CUDA Graph capture；
5. post-warmup reconcile；
6. 正式 Prefill；
7. 第一个 Decode；
8. steady Decode；
9. shutdown。

`Connection refused`、流中断、return code 134/137 常常只是更早 CUDA fault/OOM 的结果。

## 3. 快速索引

| ID | 现象 | 根因摘要 | 状态 |
| --- | --- | --- | --- |
| MEM-01 | KV/Graph 后 hot experts 留在 CPU | startup 分配晚于旧 rebalance | 已修复 |
| MEM-02 | 反复 rebalance 仍差固定 HBM | 冻结 KV、Graph、hot placement 总容量不相容 | 已修复或 N/A |
| MEM-03 | E5 capture 后只剩 172.8 MB | token envelope 2048 大于真实最大 1217 | 已修复 |
| MEM-04 | RAG/B8 cold62 仍不过 192 MiB 门 | 65K allocator reserve 过度保留 3.252 GiB | 已修复，待重复 |
| MEM-05 | KT MLA KV pool 启动 OOM | 每 token KV 被低估 3 倍 | 已修复 |
| MEM-06 | KT 正式长 Prefill OOM/少 token | static pool、动态 workspace和内部 token slot | 已修复 |
| MEM-07 | vLLM RAG 512 MiB workspace OOM | 8K 单步 Prefill瞬时容量不兼容 | 已规避 |
| MEM-08 | copy-on-forward 卸载 attention/OOM | 按参数遍历，未限制安全目标张量 | 已修复 |
| MEM-09 | MoE-Infinity KV blocks 少 1 或不足 | ratio 是相对 device ratio，且有 block floor | 已修复 |
| MEM-10 | MoE-Infinity logits 峰值约 0.8--1.6 GiB | 保存全序列 logits 并再次拼接 | 已修复 |
| MEM-11 | MoE-Infinity 重复 PagedKV | non-paged 已用 HF KV，仍物化未使用 storage | 已修复 |
| MEM-12 | MoE-Infinity RAG expert-cache OOM | C++ expert cache 无可靠容量/逐出上限 | 未解决 |
| MEM-13 | Graph/workspace 账目相加后超过物理容量 | Graph pool、workspace、allocator reserve 存在包含关系 | 统计边界 |
| UVM-01 | Long/B8 hot expert `FAULT_PDE` | 最终 warmup 后缺少 residency reconciliation | 已修复 |
| UVM-02 | flush 后仍 fault | 旧 TLB 不是唯一根因 | 结论已验证 |
| UVM-03 | Medium/RAG B8 大 Prefill fault | original-VA 大单步 conditioning 路径故障 | N/A/未解决 |
| UVM-04 | PARK/RESUME contract 被拒绝 | 启动参数缺失，仍有 live alias Graph | 已修复 |
| UVM-05 | E10 hot prefetch `unregistered_regions` | static-alias 与 explicit registry ownership 混用 | 已修复 |
| UVM-07 | E6 parked session full rebuild 返回 `NV_ERR_INVALID_STATE` | destroy 顺序及共享 cold residency 分组错误 | 已修复 |
| GRAPH-01 | 传了 capture sizes 但没有 Graph | H/A/R arm contract 明确 `graph=false` | 已修复 |
| GRAPH-02 | FLASH `FULL_SEPARATE` 被降级 | backend 只支持 UNIFORM_BATCH | 已规避 |
| GRAPH-03 | Prefill-only Long envelope 被缩小 | 把“只输出 1 token”误当作 `prompt+1` | 已修复 |
| GRAPH-04 | E6 同 shape class 退出只释放约 7 MiB | 每个 class 重复保留相同 GraphExec table | 已修复 |
| SCHED-01 | 高并发形成 `1+15`/`1+7` | HTTP/逐条 admission 提前触发 step | 已修复 |
| SCHED-02 | 新 Prefill 和旧 Decode 同一步 | default continuous batching | 已修复 |
| SCHED-03 | MoE RAG 永久 waiting | 8192 Prompt 超过 2048 step cap，且无 continuation | 已修复 |
| API-01 | KT NUMA `int has no len()` | direct API 需要 `[0]` 而不是 `0` | 已修复 |
| API-02 | KT FP8 keyword 不接受 | CLI 名与 dataclass 字段名不同 | 已修复 |
| API-03 | KT RAG 少 2 tokens | SGLang 内部需要两个保留 slot | 已修复 |
| VERIFY-01 | 正确结果被旧 verifier 拒绝 | harness 变量覆盖或日志语义误读 | 已修复 |
| VERIFY-02 | non-stable VA 被要求 VA 不变 | stable invariant 错用于对照 arm | 已修复 |
| VERIFY-07 | E6 runner/plan path 或同宽 class 被旧契约拒绝 | source 后变量覆盖、父子 root 混用和旧宽度假设 | 已修复 |
| PERF-01 | ours 首个 Decode 多约 427 ms | 28.34 GiB hot set 三轮同步审计 | 已修复为实验开关 |
| CORR-01 | ours RAG 与 baseline hash 分叉 | 收敛到 Prefill Graph/KV replay 路径 | 未解决 |
| CORR-02 | KT/MoE 跨框架 hash 不一致 | 尚无 matched-logits 根因证据 | 未解决 |
| CORR-03 | E9 uniform/no-mixed token 漂移 | placement/batch 影响数值路径 | 未解决 |
| INFRA-01 | GPU 0 MiB 但模块不能切换 | 遗留 NVML sampler 持有 UVM fd | 已修复 |
| INFRA-02 | 模块切换时一次 NVML 失败 | 查询命中 UVM reload 窗口 | 警告 |
| INFRA-04 | E8 preflight 缺 fresh receipt | extension/store provenance 不闭合 | 未解决 |
| INFRA-07 | Decode-only 偶发 `pure virtual method called` | 尚无 core/backtrace | 未解决、低频 |
| INFRA-08 | E6 10 次 rebuild 后全局析构 rc134 | PyTorch 2.9 多 Graph pool 析构顺序 | 已规避 |

## 4. 显存、KV、Graph pool 与 workspace

### MEM-01：KV/临时缓冲区挤占热门专家

**现象**

- 模型刚加载时 hot experts 在 GPU；
- KV、Graph、sampler、attention/MoE warmup 完成后，部分名义 hot 页落在 CPU；
- 正式请求在 FusedMoE 处 illegal access、`FAULT_PTE/FAULT_PDE`，或 post-capture
  safety gate 报 hot bytes/free bytes 不足。

**根因**

旧流程在 KV 或一部分 warmup 之前恢复 hot experts。后续永久 Graph pool、allocator cache
和临时 workspace 再申请 HBM，会驱逐 managed hot pages。manual fault mode 又禁止正式
请求自动迁回，所以“加载后驻留正确”并不能证明“正式请求前驻留正确”。

**解决方案**

1. KV 初始化前分配并触碰与该 cell KV 等大的 native-hole sentinel；
2. 在 warmup 阶段建立正式 shape 所需的 kernel、workspace、sampler 和 Graph pool；
3. 在适当 capture 阶段释放 sentinel；
4. 所有 startup 分配完成后执行 final cold-remote/hot-GPU reconciliation；
5. 验证 hot/cold/unknown bytes 和 minimum-free gate，再开始正式请求；
6. remote-only original-VA 路径在 final reconcile 后同步执行全 GPU TLB invalidate。

hole 是初始化期的空间占位工具，不是业务 KV，也不是 PTE fence。它解决分配顺序和连续
空间问题，不能减少最终物理占用。

**验证证据**

E2 direct-batch Short/Ordinary/RAG/Long 的最终有效 custom cells 无 fault；E3 Long/B8
在 post-warmup final reconcile 后完成 8×512 输出。详见
[E2-C16 修复记录](ERROR_FIX_LOG_E2_C16_DIRECT_BATCH_MATRIX_20260828.md) 和
[E3 Decode reconcile 记录](ERROR_FIX_LOG_E3_DECODE_TLB_RECONCILE_20260828.md)。

### MEM-02：固定配置的真实总容量不相容

**现象**

hole 能正常分配/释放，所有 Graph 也能捕获，但恢复 hot experts 时仍固定缺少数百 MiB
至约 1 GiB；重复 rebalance 得到相同缺口。

**根因**

最终必须同时驻留的集合——非专家权重、hot experts、alias backing、KV、Graph pool、
allocator cache、CUDA context 和安全余量——总和超过实际 HBM。sentinel 只能改变分配
顺序，rebalance 只能重申 placement，二者都不会减少总物理字节。

**解决方案**

- 如果实验定义允许自适应 placement：按实测缺口增加每层 cold experts，使用
  `ceil((缺口+安全余量)/每层增加一个 cold 释放的字节)` 计算候选，并从最小候选复测；
- 如果 placement、KV 和 Graph 集均已冻结：将该组合登记为容量 N/A；
- 不得通过反复调整 sentinel 顺序或关闭 safety gate 声称成功。

2-GiB KV/cold11 pure-phase 已证明可容纳其冻结 Graph 集；更小 cold 的失败不代表保护链
失效，只表示该配置容量不相容。详见
[pure-phase fence-off 复现文档](REPRODUCE_E2_DIRECT_BATCH_PURE_PHASE_FENCE_OFF.md)。

### MEM-03：E5 scheduler workspace 过度预留

**现象**

dual Graph capture 后第二次 reconcile 只有 `172,752,896 B` free，低于 192 MiB 门；把
KV 从 160 MiB 降到 128 MiB 后数值几乎不变。

**根因**

冻结 Graph 最大真实 shape 是 1,217 tokens，但 runner 以
`max_num_batched_tokens=2048` 建立 scheduler/attention workspace。多余 workspace 被
caching allocator 保留到 capture 后。

**解决方案**

设置 `MAX_NUM_BATCHED_TOKENS=1217`，保持请求、Graph shape、KV、PARK/RESUME 和 192 MiB
安全门不变。

**验证**

第二次 free 增至 `223,084,544 B`；qualification 2/2、formal 22/22 通过。这个值只适用
于最大 shape 为 1,217 的冻结协议，不能复制到更大 workload。

### MEM-04：RAG/B8 65K allocator reserve 过大

**现象**

- cold57：313 MiB hot experts 留在 CPU；
- cold60：75 MiB hot experts 留在 CPU；
- cold62：hot 全恢复，但 free 160.75 MiB，仍低于 192 MiB。

**根因**

按 65,536 tokens 做 reserve warmup，产生 `3,491,758,080 B`（3.252 GiB）永久
reserved delta。继续增加 cold 会牺牲 GPU-local experts，却不消除过度 reserve。

**解决方案**

保留能覆盖大请求的 cold63 和 65,536-token正式 Prefill Graph，但把 allocator reserve
warmup 收紧到 20,480 tokens，实际 delta 为 `1,094,713,344 B`（1.020 GiB）。正式
Prefill仍由已经捕获的 65K Graph replay，不缩短请求或 batch。

**验证与边界**

RAG/B8 8/8 完整输出，Prefill/Decode 均 replay，hot restore 后 free 2.012 GiB。当前只有
单轮，仍需跨冷进程稳定性验证。详见
[E3 Graph 错误记录](ERROR_FIX_LOG_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md)。

### MEM-05：KTransformers MLA KV 容量低估三倍

**现象**

server ready 前创建 MLA KV pool 时 OOM；调低 `mem_fraction_static` 只改变最后一次失败
分配大小。

**根因**

实际 MLA 每 token 保存 `kv_lora_rank + qk_rope_head_dim = 512+64=576` elements，旧
profiler 使用 `qk_nope_head_dim + qk_rope_head_dim = 128+64=192`，低估正好三倍。

**解决方案**

在 `model_runner_kv_cache_mixin.py` 的 MLA 分支用 `kv_lora_rank` 计算；非 MLA 分支不变。

**验证**

修复后能诚实建立 75,429-token/2.19-GB pool，后续结合 MEM-06 完成全部 16 cells。

### MEM-06：KTransformers 动态 workspace 和上下文边界

**现象**

修复 MLA accounting 后仍出现 Ordinary C16/RAG OOM，或 RAG 只输出 126/128 tokens。

**根因**

- static KV pool 留给 16,384-token Prefill workspace 的动态 HBM 不足；
- SGLang 内部在请求 envelope 外还需要两个 token slots；
- 相邻 workload 会产生 allocator fragmentation。

**解决方案**

- `context_length=8322`，对外请求仍为 8192+128；
- `mem_fraction_static=0.93`；
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**验证**

16/16 cells、116/116 请求通过。0.93 是当前 RTX 5090/driver/模型/workload 的 calibration，
换环境必须重测。

### MEM-07：原生 vLLM RAG Prefill workspace OOM

**现象**

8K 单步 Prefill 在约 512 MiB MLA 临时分配处 OOM；expandable allocator 和 4K chunk
仍不足。

**根因**

32-GiB 卡上权重、KV 与单步 8K Prefill workspace 的瞬时总量不兼容。

**解决方案**

在允许 chunked pure-Prefill 的 C1 实验中，保持 8192/128 请求不变，设置
`max_num_batched_tokens=2048`，执行四个同步 2K Prefill chunks；custom arm 使用相同
2K×4 几何配平。若实验合同要求单步 8K，则应写为 N/A，不能套用此方案。

### MEM-08：copy-on-forward 误卸载 attention 参数

**现象**

通用遍历把 attention 参数迁到 CPU，缓存 `W_UK_T` 仍引用 CPU storage；选择整个
`w13_weight` 又会产生约 704 MiB 单次复制并 OOM。

**根因**

按模型参数顺序凑 `cpu_offload_gb`，没有限制为可安全包装和复制的张量。

**解决方案**

使用 `VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE`，只选择各层完整
`model.layers.N.mlp.experts.w2_weight`，并按层数扫描最小成功配置。

### MEM-09：MoE-Infinity KV ratio 与 block floor

**现象**

- device ratio 下降后 KV 从 192 变为 191 blocks；
- Ordinary/RAG/Long 永久 waiting，`num_steps=0`。

**根因**

绝对 KV 比例为 `device_ratio * kv_ratio`；只调整 device ratio 会跨过 block 向下取整
边界。每 block 是 16 tokens/1,769,472 B，旧 ratio 也不足以覆盖实际请求。

**解决方案**

冻结目标 block 数，然后按
`new_kv_ratio = target_absolute_kv_ratio / new_device_ratio` 反算并向上取整。C1 矩阵分别
冻结 12/42/520/66 blocks；C16 保持 192 blocks，不以缩 KV 换成功。

### MEM-10：MoE-Infinity 全序列 logits 重复峰值

**现象**

C16 Short 的 16 个 `[128,102400]` FP32 logits 各约 50 MiB，旧实现先保存 list，再
`torch.cat` 申请约 800 MiB 新 tensor，峰值可接近 1.6 GiB并 OOM。

**解决方案演进**

1. 先以预分配最终 tensor、逐 slice `copy_` 消除第二份拼接；
2. E2 serving 最终只对每条序列最后 hidden state 执行 `lm_head`，因为采样不消费全序列
   logits；逐序列释放临时结果。

返回语义、dtype/device 和最终采样 token 不变。Short/Ordinary/Long 完成回归。

### MEM-11：未使用的 PagedKV 重复物化

**现象与根因**

DeepSeek non-paged backend 已使用 HF `past_key_values`，但 serving 仍建立第二份从不读取
的 PagedKV GPU tensor。

**解决方案**

增加 `materialize_paged_kv_cache=False`：逻辑 block 容量和 ownership 记账保留，物理
PagedKV storage 为 0。

### MEM-12：MoE-Infinity RAG expert cache OOM

**状态：未解决。**

消除全序列 logits 和重复 PagedKV 后，RAG C16 在 device ratio 0.80--0.55 仍于首个 8K
warmup 的 C++ expert caching allocator `cudaMalloc` OOM。这说明剩余峰值位于专家取回
cache，不是上述两份重复分配。

**安全处理**

当前记为 N/A。后续应为 device cache 增加硬容量上限、可审计逐出策略和每次 allocation/
occupancy 记录；保持 8192/128、C16 和 KV 不变重测。继续降低 ratio 只能诊断，不能代替
cache 修复。

### MEM-13：Graph pool、workspace 与 allocator reserve 不能机械相加

CUDA Graph pool 通常包含 capture 期间创建、并且 replay 仍要引用的 attention/MoE
workspace；PyTorch `reserved` 又可能包含这些 allocation 和可复用 cache。因而
“Graph pool + attention workspace + allocator reserved”直接求和会重复计算。

同理，capture 后不能为了降低 NVML 数字随意 `empty_cache()` 或释放 replay 仍引用的
buffer；这可能破坏 Graph 地址稳定性或让正式请求重新分配。

当前可靠指标是 NVML 总峰值、capture 前后 allocated/reserved delta、UVM hot/cold
residency 和 free gate。若要精确拆账，需要同一进程中的 PyTorch memory snapshot、Graph
pool allocation tag 与 UVM physical residency snapshot；现有单轮 NVML 不能给出精确物理
分项。

## 5. UVM、PTE/PDE、TLB 与 CUDA Graph

### UVM-01：Long/B8 nominal-hot expert `FAULT_PDE`

**现象**

original-VA remote 路径在进入 Decode 前出现 Xid 31、`FAULT_PDE ACCESS_TYPE_VIRT_READ`；
故障地址对应第 0 层 w13 的 expert 7，它按计划是 hot/GPU resident，不是错误卸载的 cold
expert。扩大 hole、从 cold10 增加到 cold38、关闭 CUDA Graph 都不能稳定消除。

**定位结论**

原流程在 KV 后做过 reconcile，但 kernel autotune、Graph capture、sampler warmup 和
allocator cache population 随后仍可能驱逐 hot managed page；正式请求前缺少最后一次
placement 收敛。

**最小修复**

所有 startup warmup 后：

1. 再次建立 cold remote mapping；
2. 将 hot complement 设为 GPU preferred location 并完整 prefetch；
3. 不清空已经建立的 allocator cache；
4. synchronize 后做全 GPU TLB invalidate；
5. residency 或 invalidate 证据不完整则 fail closed。

临时逐层 4-KiB device-read 断言在定位后删除，避免进入正式 timing。

**验证**

Long/B8 eager 完成 8×512，随后 CUDA Graph case 捕获 1 次、replay 511 次，均无 fault。

### UVM-02：全 GPU TLB flush 不是单独修复

两个 flush-only reproduction 仍以 return code 134 失败，排除了“旧 TLB 是唯一根因”。
`torch.cuda.empty_cache()` 更不能替代 TLB flush：它不建立 GPU MMU completion boundary，
还可能破坏已经预留的 Graph/allocator pool。

正确顺序是 **final residency reconcile → synchronize → audited all-GPC TLB invalidate**。
flush 是完成边界，reconcile 才恢复缺失的物理 placement/PTE。

### UVM-03：Medium/B8、RAG/B8 大 conditioning Prefill

**状态：未解决，Decode 性能为 N/A。**

- Medium/B8：单步 16,384 Prefill；cold38/cold63、chunk1024、autotune cap 2048 均在
  正式 Prefill 触发 Xid 31 `FAULT_PDE`；cold63 尚有约 8.2 GiB free，排除简单 HBM 不足；
- RAG/B8：单步 65,536 Prefill；cold63 下在进入 Decode 前失败。

native copy/UVA 能完成，所以问题收敛到 custom original-VA 大单步 conditioning 路径。
Graph 关闭后仍可失败，说明 Graph replay 不是必要根因。

**后续方向**

实现同步、equal-progress 的 chunked Prefill：八条请求始终属于同一 cohort，且每个
Prefill step 都没有 Decode。实现前不得用 stock scheduler 的部分 admission 或拆 B8
填补数据。详见
[E3 Decode 未解决记录](UNRESOLVED_E3_DECODE_B1_B8_20260828.md)。

### UVM-04：漏传 PARK/RESUME 导致 live Graph contract 拒绝

**现象**

RAG/B1 D arm 报 `alias_prefill_graphs_are_still_live`。

**根因**

启动时销毁 alias session，Prefill capture 又创建新 session；post-capture transition
试图在 live Graph 存在时销毁 backing，安全检查正确拒绝。

**解决方案**

补齐：

- `VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1`；
- `VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1`；
- all-remote capture profile；
- 192 MiB minimum-free contract。

复测完成双 Graph replay，退出 `session_count=0`。

### UVM-05：E10 static-alias ownership 导致 `unregistered_regions`

**现象**

E10 three-queue 在模型 warmup 后执行 hot expert prefetch，返回：

```text
Hot Expert prefetch failed: result={'ok': False, 'reason': 'unregistered_regions'}
```

**根因**

static-alias finalize 已经注册并拥有部分专家的完整物理 region；为了避免双 owner，这些
region 不再进入 explicit UVM registry，但仍保留在逻辑 `_uvm_expert_records`。旧 hot-set
生成逻辑只排除 cold/remote experts，错误地把 static-alias-owned experts 交给 explicit
registry prefetch，于是 registry 正确 fail closed。

**解决方案**

`_uvm_init_hot_expert_records()` 读取
`_uvm_static_alias_owned_global_expert_ids`，从 explicit hot prefetch 集合排除这些 ID；其
真实 residency 继续由 alias session/PARK/RESUME 管理。没有把它们改成 cold，也没有关闭
post-warmup rebalance。

**验证**

ownership 定向测试通过；E10 three-queue attempt-02、path parity 和 2--10 GiB large-KV
均通过，无 recapture/fault，退出资源完整释放。

### UVM-06：只读 PTE fence 的结论

只读 PTE fence 是 RESUME 后用 device read 触达 consumer PTE 的诊断性屏障，不是模型
计算或空间预留。逐层重读约 5 GiB 权重虽然能规避 fault，但显著增加 TTFT，不应作为最终
方案。

在 2-GiB KV/cold11、direct-batch pure-phase 四 cells 中关闭 post-RESUME fence 后仍全部
通过，说明该冻结配置不依赖 fence。必须保留初始化期 hot-read、pre-kernel alias 和
residency gate；不要把所有名称含 `READ_GATE` 的检查一起关闭。

### UVM-07：E6 parked session full rebuild 与共享 cold residency

**现象**

E6 full rebuild attempt-02 在第一次重建的 cold release 返回 `NV_ERR_INVALID_STATE` 并以
rc139 退出。attempt-03 已能 RESUME/DESTROY，但将旧、新 cold 交集与新增 cold 合并审计，
一组同时出现 CPU/GPU residency，仍 fail closed。

**根因与修复**

full rebuild 入口收到的是 `remote_parked` session，不能直接按 sealed session 的顺序
destroy。重建现在先 RESUME，再销毁旧 session；旧、新 cold 交集已被上一个计划证明在
CPU，只作为 `known_cpu_ranges` 单独复核，实际 release 仅作用于新增 cold ranges。finalize
记录 `pre_destroy_resume`、known-range reconciliation 和新 session 的完整 cold/hot
residency，任何一步失败仍中止重建。

attempt-06 完成 base 与五个 difference class 的双向 10 次 rebuild，parameter VA 全程
稳定，输出一致，cleanup 后 stock UVM 和 GPU clean。

### GRAPH-01：H/A/R 实际没有开启 CUDA Graph

**现象**

传入 capture sizes，但日志为 `enforce_eager=True`、`cudagraph_mode=NONE`、capture list
为空。

**根因**

H/A/R arm contract 本身定义 `graph=false`。参数存在不能覆盖 arm 的运行语义。

**解决方案**

Graph 对照使用 G/B/C；需要 Prefill/Decode 双 Graph 的大请求使用认证的 D arm。verifier
必须检查实际 runtime mode、capture count 和每个 measured step 的 `full_replay=1`。

### GRAPH-02：FlashAttention 不支持所需 `FULL_SEPARATE`

**现象**

FlashAttention 把 `FULL_SEPARATE` 降为 `FULL_DECODE_ONLY`，随后 static-alias contract
拒绝 Prefill Graph remote-decode 配置。

**根因**

该 backend 声明 `AttentionCGSupport.UNIFORM_BATCH`；自定义机制需要 Prefill 和 Decode
独立 full graphs，才能在 phase boundary 改变 topology。

**处理**

Prefill Graph-copy 使用已认证的 `TRITON_ATTN + FULL_SEPARATE`，并在降级时 fail closed。
FLASH native copy/UVA 与 TRITON Graph 数据可做数值参考，但不能把差异全归因于机制。

### GRAPH-03：错误缩小 Prefill-only 的 `max_model_len`

**现象**

只采样一个 token 的 runner 把 Long 512→512 的 context envelope 从 1024 改成 513。

**根因**

将 measurement output=1 与原 workload 最大 context 混为一谈。

**解决方案**

保持 `max_model_len=prompt+original_output` 和原 KV，只令 measured output 为 1。旧 attempt
整体不计入结果。

### GRAPH-04：E6 同 shape plan class 重复保留 GraphExec

**现象**

batched difference attempt-02 的所有 class 都能切换，但 C8 class 形状相同仍各自保留一份
GraphExec table，最终 teardown 只释放约 7 MiB，GPU clean gate 正确拒绝该 attempt。

**根因与修复**

plan class registry 过去把“不同 class 名”直接等价为“需要新 Graph capture”。E6 只改变
每个稳定 slot 对应的 expert identity，Graph-visible tensor shape、VA 和 dispatch descriptor
没有改变。registry 现在为等宽 class 注册同一 fully-captured table 的 alias；最终 reset 按
对象身份去重，只销毁每个 GraphExec 一次。attempt-03 的 capture count 从加载到退出均为
2，reference/session cleanup 和 GPU 0 MiB 通过。

## 6. 请求 admission、scheduler 与测量边界

### SCHED-01：HTTP 导致 `1+(N-1)`

**现象**

C16 或 B8 的首请求先 Prefill/Decode，其余请求等待；只有首请求 TTFT 约 0.56 s，其余升
至 2.2--2.5 s。

**根因**

客户端、HTTP、序列化和逐请求预处理使第一次 EngineCore step 早于其余 admission。

**解决方案**

无 HTTP，使用 in-process Processor/LLMEngine，一次构造完整请求 list；全部
`add_request`/`_validate_and_add_requests` 完成后才允许第一次 `step()`。强制记录：

- `all_requests_admitted_before_first_step=true`；
- 首 step 的 `new_request_ids == batch`；
- 无 cached/resumed/preempted；
- scheduled tokens 等于 `prompt*batch`，或有明确 token-budget 分块证明。

不得用 sleep 或放宽 scheduler window 模拟同时到达。

### SCHED-02：Prefill 与 Decode mixed step

**现象与根因**

default continuous batching 会把新到 Prefill 与已有 Decode 放进同一步，测到的是 E11
continuous-arrival 行为，不是 E2/E3 pure phase。

**解决方案**

启用 phase-batched/KV-aware cohort；先完成整个 cohort Prefill，再统一 Decode。
KTransformers 禁用 mixed chunk，MoE-Infinity 在 Decode 前清空 waiting Prefill。验收每个
Decode step：new=0、cached=batch、scheduled tokens=batch。

### SCHED-03：MoE-Infinity RAG 永久 waiting

**现象**

KV 520/520 blocks 空闲，仍 `pending=1,num_steps=0`。

**根因**

engine `max_tokens_per_step=2048`，scheduler 对 8192 Prompt 执行 admission `break`，实现
没有 chunked-Prefill continuation。

**解决方案**

把 `max_tokens_per_step` 从 API 传到 engine config；RAG 实验显式设 8192，默认仍保持
2048。这样请求真正进入 Prefill，后续 OOM 才是有效容量边界。

## 7. 框架 API、配置和 verifier

### API-01：KTransformers NUMA 节点类型

错误为 `TypeError: object of type 'int' has no len()`。CLI 会把 `--kt-numa-nodes 0`
规范化为 list，但 direct runner 直接构造 `ServerArgs` 时传入整数。修复为
`kt_numa_nodes=[0]`。

### API-02：KTransformers FP8 参数名

错误为：

```text
ServerArgs.__init__() got an unexpected keyword argument 'fp8_gemm_backend'
```

CLI 名为 `--fp8-gemm-backend`，当前 dataclass 字段是
`fp8_gemm_runner_backend`。direct Python API 改用后者，并用 `inspect.signature`/AST
校验全部 keyword。

### API-03：KTransformers RAG 少两个 token

`max-total-tokens=8320` 时只生成 126/128；增加 cold experts 无效。根因是 SGLang 内部
预留两个 token slots。设置 8322；外部 Prompt/Output 仍为 8192/128。

### API-04：Compilation mode 未显式冻结

早期 E2 runner 未显式设置 `CompilationMode.NONE`，运行时契约在 GPU 只使用约 621 MiB
时即拒绝启动。修复是让 runner 明确传入该 arm 要求的 compilation mode，并由 verifier
检查实际 mode。修复后暴露的首次 MoE illegal access 是独立的底层错误，不能把两者合并。

### VERIFY-01：公共 harness 覆盖专用 verifier

**现象**

请求已成功，但通用 verifier 不接受 `native_all_gpu`，或错误检查了别的实验字段。

**根因与修复**

被 `source` 的 harness 覆盖同名 `VERIFY`。使用独立变量 `C16_VERIFY`，固定调用
`verify_direct_batch_c16_case.py`。原始 return code/stderr 保留，用未修改 result 重新
执行正确 verifier。

### VERIFY-02：KTransformers running-req 语义误读

日志 `new-seq=16,new-token=2048,running-req=0` 中，`running-req` 表示本 step 之前已有
请求，不是加入新请求后的总数。验证器改为要求 new-seq=16、new-token=2048、旧
running=0。

### VERIFY-03：non-stable VA 被 stable invariant 拒绝

non-stable arm 已完成重分配和 recapture，但通用 verifier 无条件要求前后 VA 相同。
runner 写入 `parameter_va_mode`；stable arm 继续要求 VA 不变，non-stable arm 则要求每代
VA 改变、generation 边界一致且 diagnostics 指向新地址。formal 20 generations 通过。

### VERIFY-04：formal 重复数硬编码

E5 verifier 固定 Z/A/G 各 3 个进程，阻止计划要求的 10 个独立冷进程。新增
`FORMAL_PROCESSES_PER_ARM`，检查 arm 计数相等、两种 batch order 均存在且平衡；默认
仍兼容历史 3-process 协议。

### VERIFY-05：正常 KTransformers shutdown 被误报 fatal

成功请求后 SIGINT 关闭 scheduler/detokenizer 时，交错 traceback 以
`KeyboardInterrupt` 结束。汇总器只对已经成功、且位于 `INFO: Shutting down` 后的这种
traceback过滤；失败 case 和其他 traceback 仍为 fatal。

### VERIFY-06：旧单测夹具缺少 ownership 查询桩

生产 FusedMoE 已有 `_uvm_static_alias_owned_experts()`，旧 `SimpleNamespace` test double
没有，产生 AttributeError。给“无 static alias”夹具补返回空集合的桩；只修改测试接口，
不修改生产语义。

### VERIFY-07：E6 专用 runner、同宽 class 与父子路径契约

E6 暴露了三个只发生在 harness/validator 层的问题：full rebuild attempt-01 在 source 公共
harness 后局部 `RUNNER` 被覆盖；batched attempt-01 被旧 validator 以 class 宽度相同为由
拒绝；serial attempt-01 把父 `RUN_ROOT` 改为 `block_01` 后，plan path 错指到
`block_01/inputs`。这些 attempt 均保留，不能补写成功。

修复是使用独立 `E6_RUNNER`，validator 对等宽 class 检查 expert identity/difference 而不是
要求 selected bytes 不同，并在进入子 block 前将 plan manifest/path 解析为父 artifact 下的
绝对路径。serial attempt-01 在 GPU workload 前停止；attempt-02 是第一次实际 GPU 执行并
通过，因此没有通过重复运行筛选快样本。

## 8. 性能诊断与不应计时的审计

### PERF-01：首个 Decode 的约 427 ms 切换开销

**现象**

ours Short/C1：TTFT 更快、后续 Decode 约 7.21 ms/token，但第一个 ITL 约 446.55 ms，
汇总 TPOT 被拉到 21.38 ms。

**根因**

不是专家复制或 CUDA Graph replay慢，而是 phase transition 中对约 28.34 GiB hot experts
做三轮同步正确性审计和尾部全量 physical-residency 查询：PARK 前、PARK 后、remote map
后重复扫描。

**解决方案**

增加 `VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT`，默认开启；性能实验显式设置为 0，同时关闭
`VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE`。cold residency、stable VA、PARK/RESUME、
PTE/PDE/fault 和输出合同保留。

Short transition 降至 37.78 ms（下降 91.1%）。报告应同时给原始 E2E、steady TPOT 和
transition，不能只从 E2E 中后处理减掉成本后声称是实测结果。

## 9. 正确性分叉：尚未解决

### CORR-01：ours RAG Prefill Graph exact-generation

**现象**

ours RAG 与 vLLM baseline 均完整生成，但 hash 不一致。C1 诊断中首个分叉在 token 5；
E2-C16 中 16/16 请求均有分叉。

**已排除**

- 请求长度不足、HTTP 分批、mixed phase；
- CUDA/Xid/PTE/PDE fault；
- hot residency/headroom 不足；
- Decode Graph、alias、PARK/RESUME、remote experts 和 2K×4 chunk 本身：ours eager 与
  `FULL_DECODE_ONLY` 均能匹配 baseline。

**当前根因边界**

分叉收敛到 Prefill CUDA Graph 执行/KV 写入 replay 路径，但尚未定位具体 kernel、静态
buffer、slot mapping 或 KV page。

**处理**

该 cell 可用于结构和故障诊断，不能进入通过 exact-generation 的正式性能排名。下一步
应在各 Prefill chunk 后校验 KV，并记录 token 0--5 的 logits top-k、margin 和 ULP；建立
同 Triton MLA/同 Graph dispatcher但无 cold alias 的 matched baseline。

### CORR-02：KTransformers/MoE-Infinity 跨框架输出不一致

输入 fixture、seed、贪心设置和输出长度已冻结，但 KT/MoE 的 token hash 与 vLLM/custom
不一致；E1 中 MoE 还表现出跨 block signature 不稳定。

当前没有证据把根因唯一归到 kernel 舍入、tokenizer/generation config、KV 路径或调度。
这些结果只能证明请求完成和可测性能，不能进入要求 exact-generation 的最终跨系统排名。

后续使用共同 teacher-forced logits、首个分叉 token 的 top-1/top-2 margin、perplexity、
路由和 tokenizer config 逐项对齐。

### CORR-03：E9 uniform/no-mixed greedy token 漂移

七个 KT qualification arms 都能运行，但 uniform C1/C8 和 no-mixed C8 存在 token 漂移。
placement/batch formation 会改变 CPU/GPU kernel 和归约形状，目前没有 teacher-forced
logits 证明它只是可接受舍入。因此 formal 未启动，不能删除 exact gate。

## 10. 基础设施、清理和 provenance

### INFRA-01：孤立 NVML sampler 阻止 UVM 模块切换

**现象**

GPU 0 MiB、无 compute process，但 `wait_gpu_clean` 一直等待，`fuser` 显示 UVM client；
异常退出后 custom UVM 仍加载。

**根因**

人工中断留下父 PID 为 1 的 `nvidia-smi --loop-ms=200`，stdout FD 指向旧 case 的
`nvml.csv`，它仍打开 `/dev/nvidia-uvm`。

**解决方案**

逐 PID 检查 cmdline、父 PID 和 `/proc/<pid>/fd/1`，只终止能证明属于失败 attempt 的
sampler；随后恢复 stock UVM。禁止宽泛 `pkill nvidia-smi` 或强制卸载有未知 client 的
模块。

### INFRA-02：UVM reload 窗口的 NVML 短暂失败

custom→stock 切换瞬间，一次 `nvidia-smi` 可能报告无法连接 driver。这不是 cell 内 CUDA
fault。launcher 应在 `wait_gpu_clean` 和 module snapshot 恢复后再启动下一框架；不要把
reload 窗口的一次监控失败计为 workload 错误。

### INFRA-03：运行中在线修改 shell

Bash 可能在执行后半段才继续读取脚本；实验运行中编辑同一个文件会形成混合版本
artifact。处理是终止该 attempt，冻结脚本后用新目录完整重跑。未完成原始文件保留。

### INFRA-04：MoE-Infinity E8 provenance 不闭合

preflight 报：

```text
STORE_BUILD_RECEIPT must point to a fresh extension build receipt
```

当前 checkout commit、extension/runtime ELF、compatibility patch 和约 30-GiB store 缺少
可闭合的新 build receipt。没有绕过哈希门，也没有擅自重建大型 store。

状态仍为未解决。后续须冻结 commit/patch，重建 `_store.so`，生成 receipt 和 runtime
fingerprint，再显式验证或重建 store；完成前 E8 不产生合法 cache/prefetch 性能数字。

### INFRA-05：进程组退出 warning

native E3 每次可能记录 `destroy_process_group() was not called before program exit`。当前
return code、结果、下一进程 clean start 和最终 0 MiB 均正常，所以是非阻塞警告。正式
多重复前应在 teardown 显式 destroy process group。

### INFRA-06：E9 reference snapshot 缺失

旧 preflight 因 `snapshot_20260812.json` 缺失停止。已从冻结 calibration/evaluation split
补充 `inputs/e9_sharegpt_selection_reference_20260828.json`，并记录 source indices 和
hash。文件缺失问题已解决；随后暴露的 token 漂移属于 CORR-03，不能把 E9写成通过。

### INFRA-07：Decode-only 偶发 C++ 启动 abort

一次诊断在权重入口和 UVM allocator 动态库加载后立即出现：

```text
pure virtual method called
terminate called without an active exception
```

它发生在权重 shard、Graph capture 和正式请求之前。恢复 stock UVM、确认 GPU clean 后，
相同配置后续能完整成功，但一次成功重试不能解释成修复。

状态仍为低频未解决。若再次出现，应启用 core dump，保存相同 module/ELF fingerprint，
执行 `gdb thread apply all bt full` 并记录 dynamic-loader 日志；不要把它合并为 Prefill
Graph correctness 或请求期 PTE fault。

### INFRA-08：E6 多 Graph pool 在全局析构阶段 rc134

**现象**

full rebuild attempt-04 已完成全部 10 次重建和请求，但解释器退出时出现
`pure virtual method called`；attempt-05 又证明 runner/core/Graph/UVM/distributed 显式
cleanup 已全部成功，随后仍在 PyTorch 2.9 C++ 全局析构阶段以 rc134 退出。

**处理与边界**

该错误发生在业务结果和资源 cleanup 闭合之后，边界收敛到多 Graph pool 的全局析构顺序。
runner 显式释放所有可控引用、销毁 process group，并在验证 cleanup 状态后使用仓库已有的
controlled hard-exit 合同。专用 verifier 强制要求 `cleanup_errors=[]`、reference release、
distributed cleanup 和 hard-exit 标志同时成立，所以不能用 hard exit 跳过资源清理。

attempt-06 return code 0、verifier `status=ok`、GPU 0 MiB、无 UVM client且恢复 stock UVM。
这是受控规避，不代表 PyTorch 全局析构器的上游根因已经修复；若升级 PyTorch/CUDA，应另做
无 hard-exit 的 teardown 回归。

## 11. 合法 N/A 与平台能力限制

### 11.1 E0 DeepSeek maximum-resident Graph

32-GiB 卡没有足够 Graph headroom，按协议记 N/A。不能以 eager 数据冒充 Graph。

### 11.2 E3 native exact expert-slice

原生 vLLM copy/UVA 只能卸载完整 packed Parameter，不能复现固定 `(layer, expert)`
坐标。whole-Parameter 数值可以作为辅助对照，不能称为 exact-coordinate/equal-footprint。

### 11.3 TRITON 大单步 Prefill

当前 RTX 5090/TRITON_ATTN eager 在 `>=8192` scheduled tokens 曾触发 TMA descriptor 700/
illegal access。需要相同 backend/scheduler-step 的对称比较时，相关 cell fail closed 为
N/A；若换 FLASH_ATTN，必须明确这是另一组 backend 配置。

### 11.4 E4 CPU DRAM bandwidth

KVM guest 不暴露 uncore IMC PMU 或 memory-controller PCI topology，Intel PCM 无法采集
DRAM read/write bandwidth。NVML PCIe、GPM 和 GPU counters 不能替代它。E4 latency/PCIe
结果仍有效，但 CPU DRAM 项为 N/A；应在 bare-metal 平台补测。

### 11.5 NUMA 对照

当前主机只有一个 NUMA node，跨 NUMA实验合法记 N/A，不应通过伪造 node 配置得到数字。

## 12. 尚未完成但不属于运行错误的项目

以下是 coverage/统计缺口，不要误写为代码故障：

- E2 尚缺完整 equal-HBM 跨系统、多 tier、多独立进程正式矩阵；
- E3 多数结果为单轮点估计，缺 balanced arm order、median/p95 和 bootstrap CI；
- E3 大 Prefill 缺 E/F 对称双 Graph certificate；
- E6 full rebuild、逐 wave、全层 batched transaction 和 break-even 已完成单进程单样本
  qualification；尚缺三臂等重复数的正式 median/p95/CI 及 C2/C4/C6 容量轴；
- E7 六臂 workload-shift E2E 已完成单进程单样本 qualification；EWMA 在 32-request shift
  内由 36.98% 提升到 88.54% oracle-slot hit，但未达到 95% 收敛门，且同步维护尚未带来
  性能恢复。仍缺至少 5 seeds、平衡顺序/独立冷进程和异步 pinned D2H 的正式结果；
- E10 尚缺 5-seed 自动联合控制 trace；
- E11 尚缺 continuous-arrival 多系统 E2E；
- E12 的 1,000 Graph replay 不是 6/24-hour soak、1,000 次 PARK/RESUME 或 fault injection。

## 13. 复测完成的最低验收标准

一个“修复后成功”的 custom case 至少应同时满足：

- 非 N/A 进程 return code 0，result/verifier `status=ok`；
- 完整请求和输出 token 数；
- B8/C16 全部请求在首次 step 前入队；
- Prefill/Decode 无 mixed step；
- Graph arm 有正确 capture count 和 measured replay，无 request-time recapture；
- selected bytes、expert coordinates、KV、Graph shape 与冻结配置一致；
- hot/cold/unknown residency 和 minimum-free gate 通过；
- 无 CUDA illegal access、Xid、PTE/PDE fault；
- exact-generation gate通过，或明确标为 correctness held；
- alias/backing/session 全部释放；
- stock UVM 恢复、GPU 0 MiB、无 compute/UVM client；
- 失败 attempts 和所有配置变更可追溯。

## 14. 主要代码与配置追踪

| 问题类别 | 主要文件 |
| --- | --- |
| final reconcile / TLB | `gpu_worker.py`、`model_executor/models/utils.py`、`kernel-open/nvidia-uvm/uvm.c` |
| static alias ownership | `model_executor/layers/fused_moe/layer.py` |
| direct-batch admission | `direct_batch_phase_runner.py`、`direct_batch_c16_vllm_runner.py` |
| E3 Decode wrapper | `run_e3_decode_graph_remote_b1_b8.sh`、`run_formal_e3.sh` |
| Prefill-only | `e3_prefill_only_runner.py`、`run_e3_prefill_only_four_mechanism.sh` |
| KT MLA accounting | `KTransformers/.../model_runner_kv_cache_mixin.py` |
| MoE logits/KV | `MoE-Infinity/moe_infinity/serving/model_runner.py`、`serving/kv_cache.py`、`serving/engine.py` |
| E5 verifier | `vllm_uvm_dynamic_kv_plan_rebuild/verify_experiment.py`、`vllm_uvm_cold5_alias_prefill/verify_experiment.py` |
| E6 plan switch | `aggregate_e6_qualification.py`、`run_e6_full_rebuild_sequence.py`、`cuda_graph.py`、`model_executor/models/utils.py` |

具体行号会随代码继续演进，后续应以对应 artifact 的 `repo.commit`、`repo.status.txt` 和
provenance hash 为准，不要只引用当前工作树行号。

## 15. 原始专项记录

已解决错误：

- [E2--E12 错误与修复记录](ERROR_FIX_LOG_E2_E12_REPRO_20260827.md)
- [E2-C16 direct-batch 错误与修复](ERROR_FIX_LOG_E2_C16_DIRECT_BATCH_MATRIX_20260828.md)
- [C1 runtime-hot-audit-off 错误记录](ERROR_FIX_LOG_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_20260827.md)
- [C16 跨机制 pilot 错误记录](ERROR_FIX_LOG_C16_SHORT_DIRECT_BATCH_MECHANISM_PILOT_20260827.md)
- [E3 single-round 错误记录](ERROR_FIX_LOG_E3_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md)
- [E3 Graph 错误记录](ERROR_FIX_LOG_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md)
- [E3 native copy/UVA 记录](ERROR_FIX_LOG_E3_NATIVE_COPY_VS_UVA_PHASE_20260828.md)
- [E3 Prefill Graph-copy 记录](ERROR_FIX_LOG_E3_PREFILL_GRAPH_COPY_B1_B8_20260828.md)
- [E3 Prefill Graph/native 记录](ERROR_FIX_LOG_E3_PREFILL_GRAPH_VS_NATIVE_COPY_20260828.md)
- [E3 Decode TLB/reconcile 记录](ERROR_FIX_LOG_E3_DECODE_TLB_RECONCILE_20260828.md)
- [E6 qualification 结果与错误边界](RESULTS_E6_PLAN_SWITCH_QUALIFICATION_20260828.md)

仍未解决：

- [E2--E12 未解决问题](UNRESOLVED_E2_E12_REPRO_20260827.md)
- [E2-C16 未解决问题](UNRESOLVED_E2_C16_DIRECT_BATCH_MATRIX_20260828.md)
- [C1 跨机制未解决问题](UNRESOLVED_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_20260827.md)
- [E3 Decode B1/B8 未解决问题](UNRESOLVED_E3_DECODE_B1_B8_20260828.md)
- [E3 Graph 未解决问题](UNRESOLVED_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md)
- [E3 Prefill Graph/native 未解决问题](UNRESOLVED_E3_PREFILL_GRAPH_VS_NATIVE_COPY_20260828.md)

统一复现入口：

- [E2 C16 direct-batch](REPRODUCE_E2_C16_DIRECT_BATCH_MATRIX.md)
- [E2 pure-phase fence-off](REPRODUCE_E2_DIRECT_BATCH_PURE_PHASE_FENCE_OFF.md)
- [E3 Prefill/Decode 三机制 B1/B8](REPRODUCE_E3_PREFILL_DECODE_THREE_MECHANISM_B1_B8.md)
- [E6 Plan Class、serial wave 与切换摊销](REPRODUCE_E6_PLAN_SWITCH_QUALIFICATION.md)
