#!/usr/bin/env python3
"""Probe W13-then-W2 copy/compute pipelining for vLLM fused MoE.

The timed pipeline is:
  copy active W13 to GPU on the compute stream
  launch W13 grouped GEMM
  concurrently copy active W2 to GPU on a copy stream
  wait for W2 copy before the W2 grouped GEMM

This is a stdout-only synthetic FP8 benchmark using Qwen3-A3B-like shapes.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR / "vllm"
if str(VLLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLLM_DIR))

import torch

import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def median(values: list[float]) -> float:
    return statistics.median(values)


def make_topk_ids(args: argparse.Namespace) -> torch.Tensor:
    rows: list[list[int]] = []
    for token in range(args.num_tokens):
        row: list[int] = []
        for k in range(args.topk):
            row.append((token * args.topk + k) % args.active_experts)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def make_quant_config(args: argparse.Namespace) -> FusedMoEQuantConfig:
    block_shape = [args.fp8_block, args.fp8_block]
    w13_scale = torch.ones(
        (
            args.num_experts,
            ceil_div(args.intermediate_size * 2, args.fp8_block),
            ceil_div(args.hidden_size, args.fp8_block),
        ),
        dtype=torch.float32,
        device="cuda",
    )
    w2_scale = torch.ones(
        (
            args.num_experts,
            ceil_div(args.hidden_size, args.fp8_block),
            ceil_div(args.intermediate_size, args.fp8_block),
        ),
        dtype=torch.float32,
        device="cuda",
    )
    return FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        a2_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        block_shape=block_shape,
    )


def fused_forward(
    hidden: torch.Tensor,
    w13_gpu: torch.Tensor,
    w2_gpu: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    return fused_moe_mod.fused_experts(
        hidden,
        w13_gpu,
        w2_gpu,
        topk_weights,
        topk_ids,
        inplace=False,
        quant_config=quant_config,
        allow_deep_gemm=False,
        allow_cutlass_block_scaled_grouped_gemm=False,
    )


def time_active_copy(
    dst: torch.Tensor,
    src: torch.Tensor,
    active_experts: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    dst[:active_experts].copy_(src, non_blocking=True)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.cuda.empty_cache()
    gc.collect()

    hidden = torch.randn(
        args.num_tokens, args.hidden_size, dtype=torch.float16, device="cuda"
    )
    topk_ids = make_topk_ids(args)
    topk_weights = torch.ones(
        args.num_tokens, args.topk, dtype=torch.float32, device="cuda"
    ) / float(args.topk)
    quant_config = make_quant_config(args)

    w13_shape = (args.num_experts, args.intermediate_size * 2, args.hidden_size)
    w2_shape = (args.num_experts, args.hidden_size, args.intermediate_size)
    active_w13_shape = (
        args.active_experts,
        args.intermediate_size * 2,
        args.hidden_size,
    )
    active_w2_shape = (args.active_experts, args.hidden_size, args.intermediate_size)

    w13_gpu = torch.empty(w13_shape, dtype=torch.float8_e4m3fn, device="cuda")
    w2_gpu = torch.empty(w2_shape, dtype=torch.float8_e4m3fn, device="cuda")
    w13_gpu.zero_()
    w2_gpu.zero_()

    w13_src = torch.empty(active_w13_shape, dtype=torch.float8_e4m3fn, pin_memory=True)
    w2_src = torch.empty(active_w2_shape, dtype=torch.float8_e4m3fn, pin_memory=True)
    w13_src.zero_()
    w2_src.zero_()

    # Compile/warm resident GPU path first.
    w13_gpu[: args.active_experts].copy_(w13_src, non_blocking=True)
    w2_gpu[: args.active_experts].copy_(w2_src, non_blocking=True)
    for _ in range(args.warmup):
        fused_forward(hidden, w13_gpu, w2_gpu, topk_weights, topk_ids, quant_config)
    torch.cuda.synchronize()

    w13_copy_ms = [
        time_active_copy(w13_gpu, w13_src, args.active_experts)
        for _ in range(args.copy_repeats)
    ]
    w2_copy_ms = [
        time_active_copy(w2_gpu, w2_src, args.active_experts)
        for _ in range(args.copy_repeats)
    ]

    compute_only_ms: list[float] = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fused_forward(hidden, w13_gpu, w2_gpu, topk_weights, topk_ids, quant_config)
        end.record()
        end.synchronize()
        compute_only_ms.append(float(start.elapsed_time(end)))

    original_invoke = fused_moe_mod.invoke_fused_moe_kernel
    w13_to_w2_ms: list[float] = []
    w13_kernel_ms: list[float] = []
    activation_gap_ms: list[float] = []

    def phase_timing_invoke(*call_args: Any, **call_kwargs: Any) -> Any:
        B = call_args[1]
        phase = "w13" if B.data_ptr() == w13_gpu.data_ptr() else "w2"
        if phase == "w13":
            state["w13_start"] = torch.cuda.Event(enable_timing=True)
            state["w13_after"] = torch.cuda.Event(enable_timing=True)
            state["w13_start"].record()
            out = original_invoke(*call_args, **call_kwargs)
            state["w13_after"].record()
            return out

        state["w2_before"] = torch.cuda.Event(enable_timing=True)
        state["w2_before"].record()
        return original_invoke(*call_args, **call_kwargs)

    fused_moe_mod.invoke_fused_moe_kernel = phase_timing_invoke
    try:
        for _ in range(args.repeats):
            state: dict[str, torch.cuda.Event] = {}
            fused_forward(hidden, w13_gpu, w2_gpu, topk_weights, topk_ids, quant_config)
            torch.cuda.synchronize()
            w13_to_w2_ms.append(float(state["w13_start"].elapsed_time(state["w2_before"])))
            w13_kernel_ms.append(float(state["w13_start"].elapsed_time(state["w13_after"])))
            activation_gap_ms.append(float(state["w13_after"].elapsed_time(state["w2_before"])))
    finally:
        fused_moe_mod.invoke_fused_moe_kernel = original_invoke

    serial_ms: list[float] = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        w13_gpu[: args.active_experts].copy_(w13_src, non_blocking=True)
        w2_gpu[: args.active_experts].copy_(w2_src, non_blocking=True)
        fused_forward(hidden, w13_gpu, w2_gpu, topk_weights, topk_ids, quant_config)
        end.record()
        end.synchronize()
        serial_ms.append(float(start.elapsed_time(end)))

    copy_stream = torch.cuda.Stream()
    w13_ready = torch.cuda.Event()
    w2_ready = torch.cuda.Event()
    phase_records: list[dict[str, Any]] = []

    def traced_invoke(*call_args: Any, **call_kwargs: Any) -> Any:
        B = call_args[1]
        expert_ids = call_args[8]
        num_tokens_post_padded = int(call_args[9].item())
        config = dict(call_args[12])
        valid_m_blocks = ceil_div(num_tokens_post_padded, int(config["BLOCK_SIZE_M"]))
        active_m_blocks = int((expert_ids[:valid_m_blocks] >= 0).sum().item())
        phase = "w13" if B.data_ptr() == w13_gpu.data_ptr() else "w2"
        if phase == "w13":
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(w13_ready)
                w2_gpu[: args.active_experts].copy_(w2_src, non_blocking=True)
                w2_ready.record()
        else:
            torch.cuda.current_stream().wait_event(w2_ready)
        phase_records.append(
            {
                "phase": phase,
                "block_m": int(config["BLOCK_SIZE_M"]),
                "active_m_blocks": active_m_blocks,
                "n_tiles": ceil_div(int(B.size(1)), int(config["BLOCK_SIZE_N"])),
                "k_tiles": ceil_div(int(B.size(2)), int(config["BLOCK_SIZE_K"])),
            }
        )
        return original_invoke(*call_args, **call_kwargs)

    pipeline_ms: list[float] = []
    fused_moe_mod.invoke_fused_moe_kernel = traced_invoke
    try:
        for _ in range(args.warmup):
            w13_gpu[: args.active_experts].copy_(w13_src, non_blocking=True)
            w13_ready.record()
            fused_forward(hidden, w13_gpu, w2_gpu, topk_weights, topk_ids, quant_config)
        torch.cuda.synchronize()
        phase_records.clear()
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            w13_gpu[: args.active_experts].copy_(w13_src, non_blocking=True)
            w13_ready.record()
            fused_forward(hidden, w13_gpu, w2_gpu, topk_weights, topk_ids, quant_config)
            end.record()
            end.synchronize()
            pipeline_ms.append(float(start.elapsed_time(end)))
    finally:
        fused_moe_mod.invoke_fused_moe_kernel = original_invoke

    w13_bytes = w13_src.numel() * w13_src.element_size()
    w2_bytes = w2_src.numel() * w2_src.element_size()
    return {
        "w13_mib": mib(w13_bytes),
        "w2_mib": mib(w2_bytes),
        "compute_only_p50_ms": median(compute_only_ms),
        "w13_copy_p50_ms": median(w13_copy_ms),
        "w2_copy_p50_ms": median(w2_copy_ms),
        "w13_to_w2_window_p50_ms": median(w13_to_w2_ms),
        "w13_kernel_p50_ms": median(w13_kernel_ms),
        "activation_gap_p50_ms": median(activation_gap_ms),
        "serial_p50_ms": median(serial_ms),
        "pipeline_p50_ms": median(pipeline_ms),
        "serial_mean_ms": statistics.mean(serial_ms),
        "pipeline_mean_ms": statistics.mean(pipeline_ms),
        "phase_records": phase_records[:2],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--active-experts", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--fp8-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--copy-repeats", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    result = run(args)
    print(
        "config "
        f"tokens={args.num_tokens} topk={args.topk} active_experts={args.active_experts} "
        f"hidden={args.hidden_size} intermediate={args.intermediate_size}"
    )
    print(f"active_w13_mib={result['w13_mib']:.3f} active_w2_mib={result['w2_mib']:.3f}")
    print(
        "p50_ms "
        f"compute_only={result['compute_only_p50_ms']:.3f} "
        f"w13_copy={result['w13_copy_p50_ms']:.3f} "
        f"w2_copy={result['w2_copy_p50_ms']:.3f} "
        f"w13_to_w2_window={result['w13_to_w2_window_p50_ms']:.3f} "
        f"w13_kernel={result['w13_kernel_p50_ms']:.3f} "
        f"activation_gap={result['activation_gap_p50_ms']:.3f} "
        f"serial_copy_both_then_compute={result['serial_p50_ms']:.3f} "
        f"pipeline_w13_then_w2_during_compute={result['pipeline_p50_ms']:.3f}"
    )
    speedup = result["serial_p50_ms"] / result["pipeline_p50_ms"]
    hidden = result["serial_p50_ms"] - result["pipeline_p50_ms"]
    print(f"pipeline_vs_serial speedup={speedup:.3f}x hidden_ms={hidden:.3f}")
    print("phase detail:")
    for record in result["phase_records"]:
        tile_loads = record["active_m_blocks"] * record["n_tiles"] * record["k_tiles"]
        print(
            f"  {record['phase']} block_m={record['block_m']} "
            f"active_m_blocks={record['active_m_blocks']} tile_loads={tile_loads}"
        )


if __name__ == "__main__":
    main()
