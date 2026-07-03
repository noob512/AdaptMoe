#!/usr/bin/env python3
"""Measure the real vLLM time window before MoE expert ids are produced.

The probe records CUDA event time from Qwen3-MoE decoder-layer entry to the
point where FusedMoE.select_experts() has produced topk_ids. It prints a
stdout summary and removes the temporary JSONL trace at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR / "vllm"
if str(VLLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLLM_DIR))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.mean(values)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": float(max(values)),
    }


def load_records(path: Path, start_line: int) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records, start_line

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            if line_no < start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records, start_line + len(records)


def make_prompt(prompt_words: int, idx: int) -> str:
    # Same length across requests; prefix caching is disabled in the LLM config.
    words = ["system", "memory", "attention", "window"] * max(1, prompt_words // 4)
    words = words[:prompt_words]
    if words:
        words[-1] = f"case{idx % 97}"
    return " ".join(words)


def summarize_case(case_name: str, records: list[dict[str, Any]]) -> None:
    by_shape: dict[tuple[int, int], list[float]] = {}
    by_layer: dict[int, list[float]] = {}
    for rec in records:
        try:
            num_tokens = int(rec["num_tokens"])
            kv_len = int(rec["kv_len_approx"])
            elapsed = float(rec["elapsed_ms"])
            layer_id = int(rec["layer_id"])
        except (KeyError, TypeError, ValueError):
            continue
        by_shape.setdefault((num_tokens, kv_len), []).append(elapsed)
        by_layer.setdefault(layer_id, []).append(elapsed)

    print(f"case={case_name}")
    print(f"  records={len(records)} groups={len(by_shape)}")
    for (num_tokens, kv_len), values in sorted(by_shape.items()):
        stats = summarize(values)
        print(
            "  group "
            f"num_tokens={num_tokens} kv_len={kv_len} "
            f"count={int(stats['count'])} "
            f"mean={stats['mean']:.4f}ms "
            f"p50={stats['p50']:.4f}ms "
            f"p95={stats['p95']:.4f}ms "
            f"max={stats['max']:.4f}ms"
        )

    layer_medians = [percentile(values, 0.50) for values in by_layer.values()]
    layer_stats = summarize(layer_medians)
    print(
        "  per_layer_p50_distribution "
        f"layers={int(layer_stats['count'])} "
        f"mean={layer_stats['mean']:.4f}ms "
        f"p50={layer_stats['p50']:.4f}ms "
        f"p95={layer_stats['p95']:.4f}ms "
        f"max={layer_stats['max']:.4f}ms"
    )


def parse_args() -> argparse.Namespace:
    default_model = (
        "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/"
        "snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(tempfile.gettempdir()) / (
        f"vllm_pre_expert_window_{os.getpid()}.jsonl"
    )
    try:
        trace_path.unlink()
    except FileNotFoundError:
        pass

    os.environ["VLLM_UVM_PRE_EXPERT_TIMING_ENABLE"] = "1"
    os.environ["VLLM_UVM_PRE_EXPERT_TIMING_FILE"] = str(trace_path)
    os.environ["VLLM_CPU_OFFLOAD_BACKEND"] = "uva"

    import torch
    from vllm import LLM, SamplingParams

    print(
        "loading vLLM "
        f"model={args.model} cpu_offload_gb={args.cpu_offload_gb}",
        flush=True,
    )
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        distributed_executor_backend="uni",
        seed=args.seed,
    )

    cases = [
        ("kv32_n1", 32, 1),
        ("kv32_n8", 32, 8),
        ("kv32_n32", 32, 32),
        ("kv256_n1", 256, 1),
        ("kv768_n1", 768, 1),
    ]
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    line_cursor = 0
    for case_name, prompt_words, num_prompts in cases:
        prompts = [make_prompt(prompt_words, idx) for idx in range(num_prompts)]
        print(
            "running "
            f"case={case_name} prompt_words={prompt_words} "
            f"num_prompts={num_prompts}",
            flush=True,
        )
        _ = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        records, line_cursor = load_records(trace_path, line_cursor)
        summarize_case(case_name, records)

    try:
        trace_path.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
