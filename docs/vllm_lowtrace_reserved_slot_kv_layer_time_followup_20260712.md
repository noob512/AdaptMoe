# vLLM LOWTRACE Reserved Slot 搬运与 KV Layer Time 补充实验

日期：2026-07-12

本文补充验证两个问题：

1. 如果采用 `LOWTRACE_FOURWAY_EXPERIMENT.md` 中的 reserved slot remap/prefetch 搬运策略，专家权重搬运本身需要多少时间。
2. KV cache 大小、`num_tokens`、`kv_len` 与 Qwen3-MoE 每层计算耗时之间的关系，以及为什么早前表格里出现“KV 变大但 layer p50 反而下降”的现象。

## 1. 实验环境

- 工作目录：`/home/ubuntu/nvidia-uvm-gpu`
- GPU：NVIDIA GeForce RTX 5090，32607 MiB
- 模型：`/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782`
- vLLM：eager mode，Qwen3-MoE FP8
- KV 参数：
  - layers：48
  - KV per token per layer：2048 bytes
  - KV per token all layers：98304 bytes
  - block size：16 tokens
- 单 expert 大小：
  - `w13=[1536, 2048]`
  - `w2=[2048, 768]`
  - dtype：`torch.float8_e4m3fn`
  - one expert：4.5 MiB

## 2. LOWTRACE reserved slot 搬运时间

参考文档：`workloads/vllm/vllm/uvm_test/LOWTRACE_FOURWAY_EXPERIMENT.md`

LOWTRACE 文档里的关键机制不是普通 `cudaMemPrefetchAsync` 直接迁移整个 managed expert，而是：

1. Stage-M 先把计划内 expert 主权重卸载到 CPU。
2. Stage-M 后预分配 reserved slot backing。
3. 请求阶段命中 offloaded expert 时，通过 reserved slot remap/prefetch 路径把 expert 映射/拉回可计算窗口。

本次用 `workloads/vllm/vllm/uvm_test/expert_reserved_layer_bench.cu` 的微基准测试该策略。

原始日志目录：

```text
/tmp/kv_layer_reserved_strategy_20260712/
```

### 2.1 搬运时间结果

| case | bytes | reserved prefetch p50 | reserved prefetch p95 | invalidate p50 | e2e layer p50 | native UVM prefetch p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w2-like, 1 role | 1.5 MiB | 0.190 ms | 0.225 ms | 0.135 ms | 1.565 ms | 3.927 ms |
| w13-like, 1 role | 3.0 MiB | 0.321 ms | 0.414 ms | 0.243 ms | 1.971 ms | 3.188 ms |
| approx full expert, 2 roles | 4.5 MiB | 0.547 ms | 0.708 ms | 0.371 ms | 2.170 ms | 2.734 ms |

说明：

- `reserved prefetch` 是 reserved slot remap/prefetch 路径中把 offloaded expert 拉到 reserved 可计算窗口的核心时间。
- `invalidate` 是 slot 使用后的失效/清理开销。
- `e2e layer` 包含 reserved prefetch、MoE matmul compute、invalidate、validate 等微基准里的整层路径，不等于真实 vLLM 端到端请求 latency。
- `native UVM prefetch` 是普通 UVM managed pointer prefetch 到 GPU 的对照。

结论：

- 如果只看搬运动作本身，LOWTRACE reserved slot 策略明显快于普通 UVM prefetch。
- 4.5 MiB 近似完整 expert 的 reserved prefetch p50 约 0.55 ms，p95 约 0.71 ms。
- 加上 invalidate，4.5 MiB 的搬运相关 p50 约 0.92 ms。
- 这解释了为什么该策略理论上有机会藏进上一层计算窗口里。

但需要注意：LOWTRACE 四组真实 vLLM 实验中，dynamic real-load 和 static tail5 2% 仍有明显性能损失。原因不是 raw reserved prefetch 单步太慢，而是请求阶段还有 runtime decision、slot reuse、同步、cleanup、真实 MoE kernel 交互、命中 offloaded expert 后的调度扰动等额外开销。

## 3. KV / num_tokens / layer time 实验

使用脚本：

```bash
workloads/vllm/.venv/bin/python workloads/vllm/bench_kv_layer_time_prefetch_window.py \
  --skip-prefetch \
  --cases 32:1,64:1,128:1,256:1,512:1,768:1,32:1,64:1,128:1,256:1,512:1,768:1 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --run-dir workloads/vllm/results/kv_layer_time_kvlen_sweep_20260712
```

为排除顺序效应，又补跑交错顺序：

```bash
workloads/vllm/.venv/bin/python workloads/vllm/bench_kv_layer_time_prefetch_window.py \
  --skip-prefetch \
  --cases 768:1,32:1,512:1,64:1,256:1,128:1,768:1,32:1,512:1,64:1,256:1,128:1 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --run-dir workloads/vllm/results/kv_layer_time_kvlen_interleaved_20260712
```

原始结果：

```text
workloads/vllm/results/kv_layer_time_kvlen_sweep_20260712/kv_layer_time_prefetch_window.json
workloads/vllm/results/kv_layer_time_kvlen_interleaved_20260712/kv_layer_time_prefetch_window.json
```

### 3.1 timing hook 覆盖范围

当前 `qwen3_moe_layer_timing` 的 CUDA event 从 `Qwen3MoeDecoderLayer.forward()` 入口开始，到 `self.mlp(hidden_states)` 后结束。

因此它覆盖：

1. input layernorm
2. self attention
3. post-attention layernorm
4. sparse MoE MLP

它不是 attention-only timing，也不是单独 KV scan timing，而是整层 forward 的 CUDA event 时间。

## 4. 顺序 sweep 聚合结果

| case | phase | num_tokens | kv_len | KV block MiB | p50 | p95 | mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p32_n1 | decode | 1 | 34 | 4.5 | 2.234 ms | 2.713 ms | 2.314 ms |
| p64_n1 | decode | 1 | 66 | 7.5 | 2.244 ms | 2.721 ms | 2.356 ms |
| p128_n1 | decode | 1 | 130 | 13.5 | 2.231 ms | 2.710 ms | 2.337 ms |
| p256_n1 | decode | 1 | 258 | 25.5 | 2.230 ms | 2.665 ms | 2.319 ms |
| p512_n1 | decode | 1 | 514 | 49.5 | 2.238 ms | 2.696 ms | 2.344 ms |
| p768_n1 | decode | 1 | 770 | 73.5 | 2.322 ms | 2.756 ms | 2.372 ms |
| p32_n1 | prefill/chunk | 33 | 33 | 4.5 | 2.373 ms | 5.917 ms | 2.935 ms |
| p64_n1 | prefill/chunk | 65 | 65 | 7.5 | 2.290 ms | 7.411 ms | 3.331 ms |
| p128_n1 | prefill/chunk | 129 | 129 | 13.5 | 2.224 ms | 7.768 ms | 3.387 ms |
| p256_n1 | prefill/chunk | 257 | 257 | 25.5 | 2.202 ms | 13.163 ms | 4.849 ms |
| p512_n1 | prefill/chunk | 513 | 513 | 49.5 | 2.279 ms | 25.718 ms | 7.356 ms |
| p768_n1 | prefill/chunk | 769 | 769 | 73.5 | 2.319 ms | 35.245 ms | 9.476 ms |

## 5. 交错 sweep 聚合结果

| case | phase | num_tokens | kv_len | KV block MiB | p50 | p95 | mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p32_n1 | decode | 1 | 34 | 4.5 | 2.462 ms | 3.100 ms | 2.541 ms |
| p64_n1 | decode | 1 | 66 | 7.5 | 2.488 ms | 2.981 ms | 2.548 ms |
| p128_n1 | decode | 1 | 130 | 13.5 | 2.473 ms | 2.897 ms | 2.492 ms |
| p256_n1 | decode | 1 | 258 | 25.5 | 2.509 ms | 2.932 ms | 2.585 ms |
| p512_n1 | decode | 1 | 514 | 49.5 | 2.327 ms | 2.768 ms | 2.420 ms |
| p768_n1 | decode | 1 | 770 | 73.5 | 2.375 ms | 2.834 ms | 2.455 ms |
| p32_n1 | prefill/chunk | 33 | 33 | 4.5 | 2.468 ms | 6.997 ms | 3.571 ms |
| p64_n1 | prefill/chunk | 65 | 65 | 7.5 | 2.449 ms | 6.293 ms | 3.189 ms |
| p128_n1 | prefill/chunk | 129 | 129 | 13.5 | 2.440 ms | 8.042 ms | 3.518 ms |
| p256_n1 | prefill/chunk | 257 | 257 | 25.5 | 2.443 ms | 13.573 ms | 4.799 ms |
| p512_n1 | prefill/chunk | 513 | 513 | 49.5 | 2.346 ms | 25.850 ms | 7.579 ms |
| p768_n1 | prefill/chunk | 769 | 769 | 73.5 | 2.438 ms | 35.185 ms | 9.629 ms |

## 6. 为什么 KV 变多但 layer p50 可能下降

早前表里出现 “KV block size 从 4.5 MiB 到 49.5 MiB，decode layer p50 反而从 2.488 ms 变到 2.379 ms” 这类现象，不能解释为 KV 越大越快。更合理的解释如下：

1. 该 timing 是整层时间，不是 attention-only 时间。decode 阶段 `num_tokens=1` 时，整层时间包含 layernorm、attention、MoE routing、MoE MLP、kernel launch 和同步噪声；KV length 对 attention 的影响被固定开销和 MoE 计算掩盖。
2. decode p50 的变化幅度很小。顺序 sweep 中 34 到 514 KV length 的 decode p50 只有 2.230 到 2.244 ms 量级；交错 sweep 中也主要落在 2.327 到 2.509 ms。这种差距接近运行噪声，不能按单调关系解读。
3. 首个 case 有明显冷启动异常。比如顺序 sweep 的第一组 p32 decode mean 被 217 ms 的单层异常拉高；交错 sweep 第一组 p768 也出现类似冷启动 outlier。这会污染简单对比。
4. case 顺序会影响 GPU 热态、kernel autotune、cache 状态和 Python/vLLM runtime 状态。补跑交错顺序后，decode p50 仍不严格单调，说明 p50 不适合单独证明 KV-length 效应。
5. KV/`num_tokens` 的影响在 prefill/chunk 的 p95 和 mean 上非常明显：KV block 从 4.5 MiB 增加到 73.5 MiB，prefill/chunk p95 从约 6-7 ms 增到约 35 ms，mean 从约 3 ms 增到约 9.5 ms。这更符合 attention 工作量随上下文/批量增长的预期。

## 7. 对“上一层计算时间内搬回 expert”的判断

基于当前数据：

- decode 阶段整层 p50 大约 2.2-2.5 ms，p95 大约 2.7-3.1 ms。
- LOWTRACE reserved slot 搬回 4.5 MiB expert 的 prefetch p50 约 0.55 ms，p95 约 0.71 ms。
- 即使加上 invalidate，4.5 MiB 的搬运相关 p50 约 0.92 ms。

因此，在理想异步调度下，“上一层计算窗口”理论上足以隐藏一个 4.5 MiB expert 的 reserved-slot 搬回动作。

但这只能说明 raw 搬运时间可隐藏，不能说明真实 vLLM 请求已经完全隐藏。真实系统还需要继续验证：

1. prefetch/remap 是否真的与上一层计算并行，而不是被同步点串行化。
2. reserved slot cleanup / reuse 是否阻塞下一层。
3. expert 命中发生在同一层多个 token / 多 expert 时，是否出现 slot 数不足或 stream contention。
4. KV UVM 迁移与 expert reserved slot 搬运同时发生时，是否争抢 copy engine、page fault handling 或 UVM lock。

当前最稳的结论是：该策略在搬运时间上是可行的，但端到端收益取决于调度是否能真正 overlap。

## 8. 下一步建议

1. 增加 attention-only timing，把 self-attention 从整层 timing 中拆出来，直接测 KV length 对 attention 的贡献。
2. 增加 pre-expert 到 post-expert 的分段 timing，区分 attention、MoE routing、MoE matmul、reserved-slot remap/prefetch。
3. 在真实 vLLM dynamic real-load run 中记录每次 offloaded expert hit 的：
   - hit layer
   - expert id
   - reserved prefetch start/end
   - previous layer compute window
   - whether overlap succeeded
4. 做 KV migration + expert remap 同时发生的压力实验，判断二者是否互相拖慢。

