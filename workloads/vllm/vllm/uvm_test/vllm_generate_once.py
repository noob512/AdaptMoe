#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def gpu_memory_snapshot() -> dict[str, int | str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        return {
            "name": parts[0],
            "memory_used_mib": int(parts[1]),
            "memory_total_mib": int(parts[2]),
        }
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one tiny vLLM generate() request for UVM/MoE probes."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompt", default="Explain unified memory in one sentence.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
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
    repeats = max(1, args.repeats)

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
    gpu_memory_after_load = gpu_memory_snapshot()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    repeat_results = []
    outputs = []
    for repeat_index in range(repeats):
        prompt = args.prompt
        if repeats > 1:
            prompt = f"{args.prompt} [{repeat_index}]"
        repeat_started = time.time()
        outputs = llm.generate([prompt], sampling_params)
        repeat_finished = time.time()
        token_ids = (
            outputs[0].outputs[0].token_ids
            if outputs and outputs[0].outputs
            else []
        )
        repeat_results.append(
            {
                "repeat": repeat_index,
                "prompt": prompt,
                "generate_seconds": repeat_finished - repeat_started,
                "output_tokens": len(token_ids),
            }
        )
    finished = time.time()
    gpu_memory_after_generate = gpu_memory_snapshot()

    generated = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    token_ids = outputs[0].outputs[0].token_ids if outputs and outputs[0].outputs else []
    repeat_generate_seconds = [
        float(item["generate_seconds"]) for item in repeat_results
    ]
    sorted_repeat_seconds = sorted(repeat_generate_seconds)
    repeat_avg = sum(repeat_generate_seconds) / len(repeat_generate_seconds)
    repeat_p50 = sorted_repeat_seconds[len(sorted_repeat_seconds) // 2]
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "generated": generated,
        "output_tokens": len(token_ids),
        "load_seconds": loaded - started,
        "generate_seconds": finished - loaded,
        "repeat_avg_generate_seconds": repeat_avg,
        "repeat_p50_generate_seconds": repeat_p50,
        "repeat_results": repeat_results,
        "repeats": repeats,
        "total_seconds": finished - started,
        "gpu_memory_after_load": gpu_memory_after_load,
        "gpu_memory_after_generate": gpu_memory_after_generate,
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
