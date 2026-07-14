# vLLM KV Cache Size vs Inference Time Sweep

Date: 2026-07-12

## Purpose

测量单请求 active KV cache 大小从最小值到接近 `max_model_len=1024` 时，对 vLLM 推理时长的影响。

这里的 KV size 指当前请求已经构建出的历史 KV token 数及其 block-rounded footprint，不是整个 vLLM 启动时预分配的 KV pool 总容量。decoder-only 模型没有严格的 `decode kv_len=0`；本实验用 `prompt_words=1` 得到的 `decode kv_len=3` 作为近似最小点。

## Setup

- GPU: NVIDIA GeForce RTX 5090
- Model: `Qwen3-30B-A3B-FP8`
- vLLM: V1 engine, eager mode
- `max_model_len=1024`
- `max_num_seqs=1`
- `max_num_batched_tokens=1024`
- `max_tokens=2`
- `cpu_offload_gb=6`
- KV per token per layer: 2,048 bytes
- KV per token across 48 layers: 98,304 bytes
- Timing hook: `Qwen3MoeDecoderLayer.forward()` CUDA event, covering layernorm + attention + MoE MLP. It is not attention-only timing.

Command:

```bash
cd /home/ubuntu/nvidia-uvm-gpu
workloads/vllm/.venv/bin/python workloads/vllm/bench_kv_layer_time_prefetch_window.py \
  --skip-prefetch \
  --cases 1:1,16:1,32:1,64:1,128:1,256:1,512:1,768:1,960:1,1008:1,1008:1,960:1,768:1,512:1,256:1,128:1,64:1,32:1,16:1,1:1 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --max-tokens 2 \
  --run-dir workloads/vllm/results/kv_size_latency_sweep_20260712
```

Artifacts:

```text
workloads/vllm/results/kv_size_latency_sweep_20260712/kv_layer_time_prefetch_window.json
workloads/vllm/results/kv_size_latency_sweep_20260712/kv_layer_time_prefetch_window.md
workloads/vllm/results/kv_size_latency_sweep_20260712/qwen3_moe_layer_timing.jsonl
workloads/vllm/results/kv_size_latency_sweep_20260712/qwen3_moe_pre_expert_timing.jsonl
```

## Results

`KV block MiB` 是按 vLLM block 粒度向上取整后的全 48 层 KV footprint。每个 case 正序和逆序各跑一次，layer timing 每行聚合 96 条记录。

| prompt words | prefill tokens | decode kv_len | KV block MiB | wall mean ms | wall min/max ms | prefill p50/p95/mean ms | decode p50/p95/mean ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 3 | 1.500 | 582.085 | 306.6/857.5 | 2.500/4.036/4.082 | 2.509/2.943/4.906 |
| 16 | 17 | 18 | 3.000 | 337.639 | 324.8/350.4 | 2.450/6.320/3.541 | 2.427/2.879/2.477 |
| 32 | 33 | 34 | 4.500 | 338.534 | 334.3/342.8 | 2.484/7.591/3.510 | 2.449/2.894/2.480 |
| 64 | 65 | 66 | 7.500 | 335.871 | 334.5/337.2 | 2.426/6.212/3.136 | 2.884/3.255/2.815 |
| 128 | 129 | 130 | 13.500 | 335.570 | 322.1/349.0 | 2.428/7.934/3.495 | 2.469/2.927/2.491 |
| 256 | 257 | 258 | 25.500 | 407.812 | 407.1/408.5 | 2.451/13.226/4.966 | 2.476/2.893/2.492 |
| 512 | 513 | 514 | 49.500 | 524.637 | 509.0/540.2 | 2.514/25.344/7.402 | 2.470/2.878/2.464 |
| 768 | 769 | 770 | 73.500 | 633.148 | 616.5/649.8 | 2.521/35.027/9.618 | 2.466/2.884/2.464 |
| 960 | 961 | 962 | 91.500 | 724.405 | 721.5/727.3 | 2.512/42.826/11.418 | 2.468/2.891/2.488 |
| 1008 | 1009 | 1010 | 96.000 | 758.017 | 748.7/767.3 | 2.510/46.087/12.072 | 2.471/2.886/2.499 |

## Interpretation

1. 端到端 `generate()` wall time 随 prompt/KV 增大明显上升：warm 后大致从 300 ms 量级增长到 750 ms 量级。
2. 构建 KV 的 prefill/chunk p95 和 mean 随 KV size 增大非常明显：从约 4-8 ms p95 增到 46 ms p95。
3. decode 阶段单 token 的整层 p50 基本在 2.43-2.51 ms 附近，`p64` 有一次偏高 outlier；整体没有随 KV size 呈现强单调增长。
4. 这不代表 KV length 对 attention 没影响，而是当前 timing 覆盖整层 forward，MoE MLP、layernorm、kernel launch、runtime 噪声会掩盖单 token decode attention 的 KV scan 增量。
5. 如果要精确回答 attention-only 的 KV length 代价，需要继续加 self-attention-only CUDA event，而不是只看整层 forward。

## Practical Takeaway

对当前 Qwen3-MoE/vLLM 配置：

- KV 越大，构建 KV 的 prefill/chunk 代价明显增大。
- 单 token decode 的整层时间对 KV size 不敏感，至少在 `kv_len=3 -> 1010` 这个范围内 p50 变化很小。
- 若要用这个结果指导 KV 动态驻留，压力判断更应关注 active KV footprint 和 prefill/chunk 阶段；decode 单 token 的整层时间不能直接作为 KV scan 成本曲线。
