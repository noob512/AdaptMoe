# E2-C16 direct-batch 未解决问题

日期：2026-08-28

本文只记录尚未解决或暂时无法运行的问题。已解决项见 `ERROR_FIX_LOG_E2_C16_DIRECT_BATCH_MATRIX_20260828.md`。

## U1：ours RAG 未通过 vLLM exact-generation

状态：未解决；结构性执行和 SLO 已通过，但完整正确性门禁未通过。

复现条件：DeepSeek-V2-Lite BF16、RAG 8192/128、C16 单次 admission、4,160 MiB KV、cold19/层、`FULL_SEPARATE` Graph、pure-phase。

证据：

- 原生 pinned-UVA 与 copy-on-forward 的 16-request hash 集合摘要均为 `dc3b5c45f1dcc0b5534fb57fa21667bf2d9aad39eb344b49b585c6ace8516e75`。
- ours 摘要为 `793f9039b978653b25dc14a5321593e3baaa6138e5f5d93b198481186a6668a5`。
- 16/16 请求均有输出分叉；部分请求首 token 相同后分叉，部分在首 token 即分叉。
- ours case：`artifacts/e2-c16-direct-batch-20260828-attempt-04-ours-rag-retry/cases/ours/rag_summary/cold_19/`。

已排除：请求失败、长度不足、客户端分批、mixed Prefill/Decode、CUDA fault、热点驻留不足。该 cell 为 16/16 完整生成，且容量门禁和 192 MiB headroom 均通过。

已尝试的 Graph 对照：对 stock pinned-UVA 强制相同 `FULL_SEPARATE`，在引擎初始化阶段失败，完整错误为：

```text
ValueError: CUDAGraphMode.FULL_SEPARATE is not supported with
TritonMLABackend backend (support: AttentionCGSupport.NEVER); please try
cudagraph_mode=PIECEWISE, and make sure compilation mode is VLLM_COMPILE
```

证据：`artifacts/e2-c16-direct-batch-20260828-attempt-06-rag-graph-diagnostic/cases/native_pinned_uva/rag_summary/offload_4p0g/run.log`。

当前根因判断：尚不能区分长上下文 FULL Graph 的数值路径差异与 cold alias/恢复内容差异；删除断言或忽略 hash 都不能视为修复。

影响：不能把 ours RAG 结果作为通过完整 correctness gate 的论文数据，也不能用该 cell 做最终机制排名。

后续建议：

1. 在第一个分叉 token 前记录 eager/Graph top-1、top-2 logits 与 margin，判断是否为接近 tie 的 BF16 数值漂移。
2. 构造相同 Triton MLA、相同 Graph dispatcher、但不建立 cold alias 的 matched baseline；若 stock backend 不支持，则需专门的只诊断开关。
3. 对 cold backing 做分层抽样 checksum，并在 RESUME 后验证原 VA 内容；无需恢复运行时 28 GiB 全量热点审计。
4. 只有在 matched baseline 证明 Graph 自身造成分叉，或页面内容校验全部通过后，才考虑将 exact-token gate改为 logits/语义容差 gate。

## U2：MoE-Infinity RAG 在首个 8K warmup OOM

状态：未解决；device ratio 0.80、0.75、0.70、0.65、0.60、0.55 全部失败，return code 134。

完整核心错误：

```text
DIRECT_C16_WARMUP_START index=0
past_key_values should not be None in from_legacy_cache()
exception caught in Thread GPUFetchFunc0
reason: CUDA error 2 at
/home/ubuntu/nvidia-uvm-gpu/MoE-Infinity/core/memory/device_caching_allocator.cpp:28
in function allocate_and_cache: out of memory
Call: cudaMalloc(&ptr, bytes)
timeout: the monitored command dumped core
```

证据根目录：`artifacts/e2-c16-direct-batch-20260828-attempt-03-moe-logical-kv/cases/moe_infinity/rag_summary/`。

已尝试：

- 将 lm-head 投影限制到每条序列最后 hidden state，消除约 800 MiB 的全序列 FP32 logits 峰值。
- 使用 HF `past_key_values` 时不再物化未消费的第二份 PagedKV GPU tensor；逻辑 block 记账保持不变。
- 扫描六档 device ratio；错误位置始终为专家缓存 `cudaMalloc`。

根因：剩余阻塞位于 MoE-Infinity C++ expert device caching allocator，已与重复 logits 和重复 PagedKV 分配解耦。8K warmup 的专家取回峰值仍超过可用显存。

影响：MoE-Infinity RAG cell 缺失；其他三个 workload 不受影响。

后续建议：为 device caching allocator 增加可验证的硬容量上限/逐出策略，记录每次请求 bytes 与 cache occupancy；随后保持 8192/128 和 C16 不变重测。继续降低 ratio 只能作为诊断，不能替代缓存上限修复。

## U3：KTransformers 与 MoE-Infinity 跨框架 exact-generation 未通过

状态：未解决。

现象：两框架均完成相应 16-request full generation，但输出 token hash 与原生 vLLM/ours 基准不一致。KTransformers 四类 workload 均存在；MoE-Infinity 三个可运行 workload 均存在。

可能因素包括框架内模型实现、算子数值顺序、tokenizer/generation config 解释和 KV cache 路径差异；目前没有 matched-logits 证据，不能指定单一根因。

影响：KTransformers/MoE-Infinity 数据只能证明结构和性能现象，不能进入要求 exact-generation 的最终跨系统排名。

建议：冻结 tokenizer、generation config、首步 logits 和逐 token logits 对照；从首个分叉 token 反向定位，而不是只比较最终文本。
