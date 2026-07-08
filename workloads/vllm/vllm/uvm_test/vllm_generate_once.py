#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one tiny vLLM generate() request for UVM/MoE probes."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompt", default="Explain unified memory in one sentence.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--no-enforce-eager", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "enforce_eager": not args.no_enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "seed": args.seed,
        "disable_log_stats": True,
    }
    if args.kv_cache_memory_bytes > 0:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes

    started = time.time()
    llm = LLM(**kwargs)
    loaded = time.time()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    outputs = llm.generate([args.prompt], sampling_params)
    finished = time.time()

    generated = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    token_ids = outputs[0].outputs[0].token_ids if outputs and outputs[0].outputs else []
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "generated": generated,
        "output_tokens": len(token_ids),
        "load_seconds": loaded - started,
        "generate_seconds": finished - loaded,
        "total_seconds": finished - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
