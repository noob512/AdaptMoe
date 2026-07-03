#!/usr/bin/env python3
"""Measure per-layer expert-weight use in vLLM fused MoE.

This is a stdout-only probe. It runs one standalone fused_experts call with
fixed routing and records how many grouped-GEMM blocks touch a target expert.
"""

from __future__ import annotations

import argparse
import math
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


def build_topk_ids(
    *,
    num_tokens: int,
    topk: int,
    num_experts: int,
    target_expert: int,
    target_hits: int,
    device: str,
) -> torch.Tensor:
    if target_hits > num_tokens:
        raise ValueError("target_hits cannot exceed num_tokens in this probe")

    rows: list[list[int]] = []
    next_expert = 1 if target_expert == 0 else 0
    for token in range(num_tokens):
        row: list[int] = []
        if token < target_hits:
            row.append(target_expert)
        while len(row) < topk:
            candidate = next_expert % num_experts
            next_expert += 1
            if candidate == target_expert or candidate in row:
                continue
            row.append(candidate)
        rows.append(row)

    return torch.tensor(rows, dtype=torch.int32, device=device)


def make_fp8_weight(shape: tuple[int, ...], device: str) -> torch.Tensor:
    # The values are irrelevant for this structural probe; empty avoids a slow
    # temporary fp16 allocation for hundreds of MiB of Qwen-shaped weights.
    return torch.empty(shape, device=device, dtype=torch.float8_e4m3fn)


def run_case(args: argparse.Namespace, target_hits: int) -> list[dict[str, Any]]:
    device = "cuda"
    torch.cuda.empty_cache()
    torch.manual_seed(1234 + target_hits)

    hidden_states = torch.randn(
        args.num_tokens, args.hidden_size, device=device, dtype=torch.float16
    )
    topk_ids = build_topk_ids(
        num_tokens=args.num_tokens,
        topk=args.topk,
        num_experts=args.num_experts,
        target_expert=args.target_expert,
        target_hits=target_hits,
        device=device,
    )
    topk_weights = torch.ones(
        args.num_tokens, args.topk, device=device, dtype=torch.float32
    ) / float(args.topk)

    w1 = make_fp8_weight(
        (args.num_experts, args.intermediate_size * 2, args.hidden_size), device
    )
    w2 = make_fp8_weight(
        (args.num_experts, args.hidden_size, args.intermediate_size), device
    )

    block_shape = [args.fp8_block, args.fp8_block]
    w1_scale = torch.ones(
        (
            args.num_experts,
            ceil_div(args.intermediate_size * 2, args.fp8_block),
            ceil_div(args.hidden_size, args.fp8_block),
        ),
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.ones(
        (
            args.num_experts,
            ceil_div(args.hidden_size, args.fp8_block),
            ceil_div(args.intermediate_size, args.fp8_block),
        ),
        device=device,
        dtype=torch.float32,
    )
    a1_scale = torch.ones(1, device=device, dtype=torch.float32)
    a2_scale = torch.ones(1, device=device, dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )

    original_invoke = fused_moe_mod.invoke_fused_moe_kernel
    records: list[dict[str, Any]] = []

    def traced_invoke(*call_args: Any, **call_kwargs: Any) -> Any:
        # invoke_fused_moe_kernel(A, B, C, A_scale, B_scale, B_zp,
        #                         topk_weights, sorted_token_ids, expert_ids,
        #                         num_tokens_post_padded, mul_routed_weight,
        #                         top_k, config, ...)
        B = call_args[1]
        expert_ids = call_args[8]
        top_k = int(call_args[11])
        num_tokens_post_padded = int(call_args[9].item())
        config = dict(call_args[12])
        phase = "w13" if B.data_ptr() == w1.data_ptr() else "w2"
        valid_m_blocks = ceil_div(num_tokens_post_padded, int(config["BLOCK_SIZE_M"]))
        valid_expert_ids = expert_ids[:valid_m_blocks]
        target_m_blocks = int((valid_expert_ids == args.target_expert).sum().item())
        n_tiles = ceil_div(int(B.size(1)), int(config["BLOCK_SIZE_N"]))
        k_tiles = ceil_div(int(B.size(2)), int(config["BLOCK_SIZE_K"]))
        b_tile_loads = target_m_blocks * n_tiles * k_tiles
        logical_matrix_bytes = int(B[args.target_expert].numel() * B.element_size())
        records.append(
            {
                "phase": phase,
                "top_k_arg": top_k,
                "B_shape": tuple(int(x) for x in B.shape),
                "BLOCK_SIZE_M": int(config["BLOCK_SIZE_M"]),
                "BLOCK_SIZE_N": int(config["BLOCK_SIZE_N"]),
                "BLOCK_SIZE_K": int(config["BLOCK_SIZE_K"]),
                "GROUP_SIZE_M": int(config["GROUP_SIZE_M"]),
                "valid_m_blocks": valid_m_blocks,
                "target_m_blocks": target_m_blocks,
                "n_tiles": n_tiles,
                "k_tiles": k_tiles,
                "b_tile_loads": b_tile_loads,
                "logical_matrix_bytes": logical_matrix_bytes,
                "logical_matrix_mib": logical_matrix_bytes / (1024.0 * 1024.0),
                "nominal_matrix_reads": target_m_blocks,
                "nominal_bytes_mib": (
                    logical_matrix_bytes * target_m_blocks / (1024.0 * 1024.0)
                ),
            }
        )
        return original_invoke(*call_args, **call_kwargs)

    fused_moe_mod.invoke_fused_moe_kernel = traced_invoke
    try:
        for _ in range(args.warmup):
            fused_moe_mod.fused_experts(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=False,
                quant_config=quant_config,
                allow_deep_gemm=False,
                allow_cutlass_block_scaled_grouped_gemm=False,
            )
        torch.cuda.synchronize()
        records.clear()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fused_moe_mod.fused_experts(
            hidden_states,
            w1,
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
        elapsed_ms = float(start.elapsed_time(end))
    finally:
        fused_moe_mod.invoke_fused_moe_kernel = original_invoke

    target_assignments = int((topk_ids == args.target_expert).sum().item())
    for record in records:
        record["target_assignments"] = target_assignments
        record["elapsed_ms"] = elapsed_ms
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--target-expert", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--fp8-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--target-hits",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated token assignments to the target expert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print(
        "config "
        f"tokens={args.num_tokens} topk={args.topk} experts={args.num_experts} "
        f"target={args.target_expert} hidden={args.hidden_size} "
        f"intermediate={args.intermediate_size} fp8_block={args.fp8_block}"
    )
    print(
        "case phase assignments block_m m_blocks n_tiles k_tiles tile_loads "
        "matrix_mib nominal_reads nominal_mib elapsed_ms"
    )
    for raw in args.target_hits.split(","):
        target_hits = int(raw.strip())
        if target_hits <= 0:
            continue
        records = run_case(args, target_hits)
        for record in records:
            print(
                f"hits={target_hits:<2d} "
                f"{record['phase']:<3s} "
                f"{record['target_assignments']:<11d} "
                f"{record['BLOCK_SIZE_M']:<7d} "
                f"{record['target_m_blocks']:<8d} "
                f"{record['n_tiles']:<7d} "
                f"{record['k_tiles']:<7d} "
                f"{record['b_tile_loads']:<10d} "
                f"{record['logical_matrix_mib']:<10.3f} "
                f"{record['nominal_matrix_reads']:<13d} "
                f"{record['nominal_bytes_mib']:<11.3f} "
                f"{record['elapsed_ms']:.3f}"
            )


if __name__ == "__main__":
    main()
