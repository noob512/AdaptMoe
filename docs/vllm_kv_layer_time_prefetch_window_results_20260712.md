# vLLM KV Size / Num Tokens / Layer Time Experiment

Date: 2026-07-12

## Purpose

Measure the relationship between active KV cache size, scheduled `num_tokens`,
and Qwen3-MoE per-layer compute time. Use the result to judge whether an expert
evicted by KV pressure can be prefetched back to GPU during the previous layer's
compute window.

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32 GiB
- Model: Qwen3-30B-A3B-FP8 local checkpoint
- vLLM mode: eager, V1 engine, `cpu_offload_gb=6`, `max_model_len=1024`
- KV dtype assumption: 2 bytes
- Layers: 48 sparse MoE layers
- KV per token per layer: 2,048 bytes
- KV per token across all layers: 98,304 bytes
- One expert weight size: 4.5 MiB

Experiment script:

```bash
cd /home/ubuntu/nvidia-uvm-gpu
workloads/vllm/.venv/bin/python workloads/vllm/bench_kv_layer_time_prefetch_window.py \
  --prefetch-repeats 5 \
  --expert-multipliers 1,2,4,8 \
  --cases 32:1,128:1,512:1,128:4,128:8 \
  --run-dir workloads/vllm/results/kv_layer_time_prefetch_window_real_20260712_final
```

Full result:

- `workloads/vllm/results/kv_layer_time_prefetch_window_real_20260712_final/kv_layer_time_prefetch_window.md`
- `workloads/vllm/results/kv_layer_time_prefetch_window_real_20260712_final/kv_layer_time_prefetch_window.json`
- `workloads/vllm/results/kv_layer_time_prefetch_window_real_20260712_final/qwen3_moe_layer_timing.jsonl`

## Expert Prefetch Time

| Experts | Size | UVM prefetch p50 | UVM prefetch p95 |
| --- | ---: | ---: | ---: |
| 1 | 4.5 MiB | 1.229 ms | 1.245 ms |
| 2 | 9.0 MiB | 1.859 ms | 1.871 ms |
| 4 | 18.0 MiB | 3.090 ms | 3.100 ms |
| 8 | 36.0 MiB | 5.930 ms | 5.947 ms |

## Layer Time vs KV Size

KV size below is block-rounded and counted across all 48 layers.

| Case | Phase | num_tokens | kv_len | KV size | Layer p50 | Layer p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `p32_n1` | decode full batch | 1 | 34 | 4.5 MiB | 2.488 ms | 2.947 ms |
| `p128_n1` | decode full batch | 1 | 130 | 13.5 MiB | 2.402 ms | 3.292 ms |
| `p512_n1` | decode full batch | 1 | 514 | 49.5 MiB | 2.379 ms | 2.988 ms |
| `p128_n4` | decode partial/mixed | 3 | 130 | 54.0 MiB | 2.362 ms | 4.078 ms |
| `p128_n8` | decode partial/mixed | 7 | 130 | 108.0 MiB | 2.384 ms | 4.462 ms |
| `p128_n4` | prefill/chunk | 388 | 130 | 54.0 MiB | 2.406 ms | 19.700 ms |
| `p128_n8` | prefill/chunk | 904 | 130 | 108.0 MiB | 2.388 ms | 40.191 ms |

## Conclusion

For decode-like steps, the measured per-layer p50 window is about 2.36-2.49 ms.
This is enough to hide UVM prefetch of one or two experts, but not four experts.

At p95, larger scheduled-token cases provide wider windows:

- `num_tokens=1` hides 2 experts reliably, and sometimes reaches the 4-expert
  threshold only for `p128_n1`.
- `num_tokens=3/7` decode partial/mixed rows have p95 windows of 4.08/4.46 ms,
  enough to cover 4 experts.
- Large prefill/chunk rows have much larger p95 windows, but those are not the
  steady decode case and should not be used as the sole scheduling budget.

Therefore the idea is feasible, but the safe default should be conservative:

- During decode, plan around 1-2 experts per previous-layer window.
- Allow larger batches or prefill/chunk phases to opportunistically prefetch more.
- Do not assume 4+ experts can always be hidden in single-token decode.
