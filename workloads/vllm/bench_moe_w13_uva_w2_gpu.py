#!/usr/bin/env python3
"""Compare mixed UVA/GPU backing for W13 and W2 in vLLM fused MoE.

This is a stdout-only structural/perf probe with synthetic Qwen3-A3B-like FP8
expert weights. It answers whether keeping W13 remote while W2 is GPU-resident
helps compared with all-remote or all-GPU backing.
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

import vllm._C  # noqa: F401  # register torch.ops._C custom ops
import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.utils.torch_utils import get_cuda_view_from_cpu_tensor


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def median(values: list[float]) -> float:
    return statistics.median(values)


def make_topk_ids(args: argparse.Namespace) -> torch.Tensor:
    rows: list[list[int]] = []
    for token in range(args.num_tokens):
        row: list[int] = []
        for k in range(args.topk):
            if args.route_mode == "round_robin":
                expert = (token * args.topk + k) % args.active_experts
            elif args.route_mode == "same_expert":
                expert = 0
            else:
                raise ValueError(f"unknown route mode {args.route_mode!r}")
            row.append(expert)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def make_backing(shape: tuple[int, ...], dtype: torch.dtype, mode: str) -> torch.Tensor:
    if mode == "gpu":
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        # Touch once to commit GPU pages and avoid timing first-use allocation.
        tensor.zero_()
        return tensor

    if mode == "uva":
        cpu = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        if not cpu.is_pinned():
            raise RuntimeError("failed to allocate pinned CPU tensor")
        # Zero-fill on CPU so mapped pages are committed before the timed region.
        cpu.zero_()
        cuda_view = get_cuda_view_from_cpu_tensor(cpu)
        # Keep CPU backing alive through a private attribute on the returned view.
        cuda_view._uvm_cpu_backing = cpu  # type: ignore[attr-defined]
        return cuda_view

    raise ValueError(f"unknown backing mode {mode!r}")


def run_mode(args: argparse.Namespace, w13_mode: str, w2_mode: str) -> dict[str, Any]:
    torch.cuda.empty_cache()
    gc.collect()

    hidden = torch.randn(
        args.num_tokens, args.hidden_size, dtype=torch.float16, device="cuda"
    )
    topk_ids = make_topk_ids(args)
    topk_weights = torch.ones(
        args.num_tokens, args.topk, dtype=torch.float32, device="cuda"
    ) / float(args.topk)

    w13_shape = (args.num_experts, args.intermediate_size * 2, args.hidden_size)
    w2_shape = (args.num_experts, args.hidden_size, args.intermediate_size)
    w13 = make_backing(w13_shape, torch.float8_e4m3fn, w13_mode)
    w2 = make_backing(w2_shape, torch.float8_e4m3fn, w2_mode)

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
    a1_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    a2_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )

    original_invoke = fused_moe_mod.invoke_fused_moe_kernel
    phase_records: list[dict[str, Any]] = []

    def traced_invoke(*call_args: Any, **call_kwargs: Any) -> Any:
        B = call_args[1]
        expert_ids = call_args[8]
        num_tokens_post_padded = int(call_args[9].item())
        config = dict(call_args[12])
        valid_m_blocks = ceil_div(num_tokens_post_padded, int(config["BLOCK_SIZE_M"]))
        active_m_blocks = int((expert_ids[:valid_m_blocks] >= 0).sum().item())
        phase = "w13" if B.data_ptr() == w13.data_ptr() else "w2"
        n_tiles = ceil_div(int(B.size(1)), int(config["BLOCK_SIZE_N"]))
        k_tiles = ceil_div(int(B.size(2)), int(config["BLOCK_SIZE_K"]))
        phase_records.append(
            {
                "phase": phase,
                "backing": w13_mode if phase == "w13" else w2_mode,
                "block_m": int(config["BLOCK_SIZE_M"]),
                "active_m_blocks": active_m_blocks,
                "n_tiles": n_tiles,
                "k_tiles": k_tiles,
                "tile_loads": active_m_blocks * n_tiles * k_tiles,
                "matrix_mib_per_expert": B[0].numel() * B.element_size() / (1024**2),
            }
        )
        return original_invoke(*call_args, **call_kwargs)

    fused_moe_mod.invoke_fused_moe_kernel = traced_invoke
    try:
        for _ in range(args.warmup):
            fused_moe_mod.fused_experts(
                hidden,
                w13,
                w2,
                topk_weights,
                topk_ids,
                inplace=False,
                quant_config=quant_config,
                allow_deep_gemm=False,
                allow_cutlass_block_scaled_grouped_gemm=False,
            )
        torch.cuda.synchronize()

        times: list[float] = []
        phase_records.clear()
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fused_moe_mod.fused_experts(
                hidden,
                w13,
                w2,
                topk_weights,
                topk_ids,
                inplace=False,
                quant_config=quant_config,
                allow_deep_gemm=False,
                allow_cutlass_block_scaled_grouped_gemm=False,
            )
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
    finally:
        fused_moe_mod.invoke_fused_moe_kernel = original_invoke

    # Keep one representative pair of phase records. The trace repeats per run.
    representative = phase_records[:2]
    return {
        "mode": f"w13_{w13_mode}_w2_{w2_mode}",
        "p50_ms": median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "phase_records": representative,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--active-experts", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--fp8-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--route-mode",
        choices=("round_robin", "same_expert"),
        default="round_robin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    modes = [
        ("gpu", "gpu"),
        ("uva", "uva"),
        ("uva", "gpu"),
        ("gpu", "uva"),
    ]

    print(
        "config "
        f"tokens={args.num_tokens} topk={args.topk} experts={args.num_experts} "
        f"active_experts={args.active_experts} route={args.route_mode} "
        f"hidden={args.hidden_size} intermediate={args.intermediate_size}"
    )
    print("mode p50_ms mean_ms min_ms max_ms")
    results = []
    for w13_mode, w2_mode in modes:
        result = run_mode(args, w13_mode, w2_mode)
        results.append(result)
        print(
            f"{result['mode']:<18s} "
            f"{result['p50_ms']:.3f} "
            f"{result['mean_ms']:.3f} "
            f"{result['min_ms']:.3f} "
            f"{result['max_ms']:.3f}"
        )

    print("phase detail from last runs:")
    for result in results:
        print(f"{result['mode']}:")
        for record in result["phase_records"]:
            print(
                "  "
                f"{record['phase']} backing={record['backing']} "
                f"block_m={record['block_m']} "
                f"active_m_blocks={record['active_m_blocks']} "
                f"tiles={record['tile_loads']} "
                f"matrix_mib_per_expert={record['matrix_mib_per_expert']:.3f}"
            )


if __name__ == "__main__":
    main()
