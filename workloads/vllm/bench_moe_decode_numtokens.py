#!/usr/bin/env python3
"""Decode-like MoE expert layout benchmark with fixed num_tokens.

This standalone benchmark does not start the vLLM server. It uses the real
Qwen3 FP8 checkpoint tensors and vLLM's pinned-CPU UVA view helper to compare:

  three_tier: hot experts on GPU, stream experts copied to a 2-layer GPU slot,
              tail experts read through pinned CPU UVA
  all_uva:    all experts read through pinned CPU UVA, matching vLLM CPU
              offload at the tensor backing level
  partial_uva:
              a configurable tail of experts read through pinned CPU UVA,
              while the prefix stays GPU resident

The runtime loop fixes num_tokens and topk, builds the same routing plan for
both modes, and measures the per-layer latency of reading the active experts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_hot_cold_expert_checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    CheckpointReader,
    _alloc_gpu_tensor,
    _copy_expert_to_batch,
    _cuda_view_from_pinned_cpu,
    _dtype_bytes,
    _empty_cpu,
    _gbps,
    _gib,
    _load_expert_batch,
    _parse_layers,
)

UVA_PRIORITY_POLICIES = (
    "uva_priority",
    "uva_priority_promote",
    "uva_priority_promote_prefetch",
)
PROMOTION_POLICIES = (
    "uva_priority_promote",
    "uva_priority_promote_prefetch",
)
PROMOTION_GATED_PREFETCH_POLICIES = ("uva_priority_promote_prefetch",)
SYNC_MISS_COPY_POLICIES = ("sync_miss_copy",)


def _now() -> float:
    return time.perf_counter()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[idx]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    ms = [v * 1000.0 for v in values]
    return {
        "count": len(ms),
        "mean_ms": statistics.mean(ms),
        "p50_ms": _percentile(ms, 0.50),
        "p90_ms": _percentile(ms, 0.90),
        "p95_ms": _percentile(ms, 0.95),
        "p99_ms": _percentile(ms, 0.99),
        "max_ms": max(ms),
    }


def _event_elapsed_seconds(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end)) / 1000.0


def _count_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    xs = [float(v) for v in values]
    return {
        "count": len(xs),
        "mean": statistics.mean(xs),
        "p50": _percentile(xs, 0.50),
        "p90": _percentile(xs, 0.90),
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
        "max": max(xs),
    }


def _float_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _make_weights(args: argparse.Namespace) -> list[float]:
    if args.hot_experts + args.stream_experts + args.uva_experts != args.num_experts:
        raise ValueError(
            "hot_experts + stream_experts + uva_experts must equal num_experts"
        )
    total_prob = args.hot_prob + args.stream_prob + args.uva_prob
    if total_prob <= 0:
        raise ValueError("routing probabilities must be positive")
    hot_prob = args.hot_prob / total_prob
    stream_prob = args.stream_prob / total_prob
    uva_prob = args.uva_prob / total_prob
    weights = [0.0] * args.num_experts
    for expert in range(args.hot_experts):
        weights[expert] = hot_prob / max(1, args.hot_experts)
    stream_start = args.hot_experts
    for expert in range(stream_start, stream_start + args.stream_experts):
        weights[expert] = stream_prob / max(1, args.stream_experts)
    uva_start = stream_start + args.stream_experts
    for expert in range(uva_start, args.num_experts):
        weights[expert] = uva_prob / max(1, args.uva_experts)
    return weights


def _weighted_sample_without_replacement(
    rng: random.Random,
    weights: list[float],
    k: int,
) -> list[int]:
    available = list(range(len(weights)))
    available_weights = list(weights)
    selected: list[int] = []
    for _ in range(min(k, len(available))):
        total = sum(available_weights)
        if total <= 0:
            break
        target = rng.random() * total
        acc = 0.0
        pos = 0
        for pos, weight in enumerate(available_weights):
            acc += weight
            if acc >= target:
                break
        selected.append(available.pop(pos))
        available_weights.pop(pos)
    return selected


def _build_route_plan(
    args: argparse.Namespace,
    layers: list[int],
) -> dict[int, dict[int, list[int]]]:
    rng = random.Random(args.seed)
    if args.route_mode == "fixed_counts":
        if args.fixed_hot + args.fixed_stream + args.fixed_uva <= 0:
            raise ValueError("fixed_counts route mode requires at least one fixed expert")
        if args.fixed_hot > args.hot_experts:
            raise ValueError("fixed_hot cannot exceed hot_experts")
        if args.fixed_stream > args.stream_experts:
            raise ValueError("fixed_stream cannot exceed stream_experts")
        if args.fixed_uva > args.uva_experts:
            raise ValueError("fixed_uva cannot exceed uva_experts")
        hot_ids = list(range(args.fixed_hot))
        stream_start = args.hot_experts
        stream_ids = list(range(stream_start, stream_start + args.fixed_stream))
        uva_start = args.hot_experts + args.stream_experts
        uva_ids = list(range(uva_start, uva_start + args.fixed_uva))
        active = sorted(hot_ids + stream_ids + uva_ids)
        return {
            step: {layer: active for layer in layers}
            for step in range(args.decode_steps)
        }

    weights = _make_weights(args)
    plan: dict[int, dict[int, list[int]]] = {}
    for step in range(args.decode_steps):
        layer_plan: dict[int, list[int]] = {}
        for layer in layers:
            active: set[int] = set()
            for _ in range(args.num_tokens):
                active.update(
                    _weighted_sample_without_replacement(rng, weights, args.topk)
                )
            layer_plan[layer] = sorted(active)
        plan[step] = layer_plan
    return plan


def _count_tiers(args: argparse.Namespace, expert_ids: list[int]) -> dict[str, int]:
    stream_start = args.hot_experts
    uva_start = args.hot_experts + args.stream_experts
    return {
        "hot": sum(1 for x in expert_ids if x < stream_start),
        "stream": sum(1 for x in expert_ids if stream_start <= x < uva_start),
        "uva": sum(1 for x in expert_ids if x >= uva_start),
        "total": len(expert_ids),
    }


def _copy_single_expert(
    reader: CheckpointReader,
    *,
    layer: int,
    expert: int,
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    w13 = _empty_cpu((1, w13_shape[0], w13_shape[1]), dtype, pin_memory=pin_memory)
    w2 = _empty_cpu((1, w2_shape[0], w2_shape[1]), dtype, pin_memory=pin_memory)
    _copy_expert_to_batch(
        reader,
        layer=layer,
        expert=expert,
        local=0,
        batch_w13=w13,
        batch_w2=w2,
    )
    return w13[0], w2[0]


def _select_and_sum(
    w13: torch.Tensor,
    w2: torch.Tensor,
    ids: list[int],
    *,
    device: torch.device,
    touch_elements: int,
) -> torch.Tensor | None:
    if not ids:
        return None
    ids_gpu = torch.tensor(ids, dtype=torch.long, device=device)
    flat_w13 = w13.reshape(w13.shape[0], -1)
    flat_w2 = w2.reshape(w2.shape[0], -1)
    selected_w13 = flat_w13.index_select(0, ids_gpu)
    selected_w2 = flat_w2.index_select(0, ids_gpu)
    if touch_elements > 0:
        selected_w13 = selected_w13[:, :touch_elements]
        selected_w2 = selected_w2[:, :touch_elements]
    return selected_w13.float().sum() + selected_w2.float().sum()


def _build_sim_compute_state(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if args.sim_compute_dim <= 0 or args.sim_compute_iters <= 0:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    shape = (args.num_tokens, args.sim_compute_dim)
    weight_shape = (args.sim_compute_dim, args.sim_compute_dim)
    return {
        "x": torch.randn(shape, device=device, dtype=torch.float16, generator=gen),
        "w": torch.randn(weight_shape, device=device, dtype=torch.float16, generator=gen),
    }


def _simulate_token_compute(
    state: dict[str, torch.Tensor] | None,
    iters: int,
) -> torch.Tensor | None:
    if state is None:
        return None
    y = state["x"]
    w = state["w"]
    for _ in range(iters):
        y = torch.matmul(y, w)
    return y.float().sum()


def _empty_cuda_event() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=False)
    event.record(torch.cuda.current_stream())
    return event


def _summarize_route_plan(
    args: argparse.Namespace,
    plan: dict[int, dict[int, list[int]]],
) -> dict[str, Any]:
    tier_counts = {"hot": [], "stream": [], "uva": [], "total": []}
    for layer_plan in plan.values():
        for expert_ids in layer_plan.values():
            counts = _count_tiers(args, expert_ids)
            for key, value in counts.items():
                tier_counts[key].append(value)
    return {
        "num_tokens": args.num_tokens,
        "topk": args.topk,
        "assignments_per_layer": args.num_tokens * args.topk,
        "routing_probabilities": {
            "hot": args.hot_prob,
            "stream": args.stream_prob,
            "uva": args.uva_prob,
        },
        "active_unique_experts": {
            key: _count_stats(values) for key, values in tier_counts.items()
        },
    }


def _build_three_tier_layout(
    args: argparse.Namespace,
    reader: CheckpointReader,
    layers: list[int],
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hot_w13: dict[int, torch.Tensor] = {}
    hot_w2: dict[int, torch.Tensor] = {}
    stream_cpu: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    cold_cpu: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    uva_cpu: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    cold_uva_w13: dict[int, torch.Tensor] = {}
    cold_uva_w2: dict[int, torch.Tensor] = {}
    uva_w13: dict[int, torch.Tensor] = {}
    uva_w2: dict[int, torch.Tensor] = {}
    uva_cpu_refs: list[tuple[torch.Tensor, torch.Tensor]] = []
    cold_uva_cpu_refs: list[tuple[torch.Tensor, torch.Tensor]] = []

    hot_copy_bytes = 0
    stream_host_bytes = 0
    uva_host_bytes = 0
    t0 = _now()
    for layer in layers:
        stream_src: tuple[torch.Tensor, torch.Tensor] | None = None
        uva_src: tuple[torch.Tensor, torch.Tensor] | None = None
        if args.hot_experts:
            src_w13, src_w2, _ = _load_expert_batch(
                reader,
                layer=layer,
                expert_start=0,
                expert_count=args.hot_experts,
                w13_shape=w13_shape,
                w2_shape=w2_shape,
                dtype=dtype,
                pin_memory=args.pin_memory,
            )
            dst_w13, _ = _alloc_gpu_tensor(
                (args.hot_experts, w13_shape[0], w13_shape[1]),
                dtype,
                device,
                "cuda",
            )
            dst_w2, _ = _alloc_gpu_tensor(
                (args.hot_experts, w2_shape[0], w2_shape[1]),
                dtype,
                device,
                "cuda",
            )
            dst_w13.copy_(src_w13, non_blocking=args.pin_memory)
            dst_w2.copy_(src_w2, non_blocking=args.pin_memory)
            hot_w13[layer] = dst_w13
            hot_w2[layer] = dst_w2
            hot_copy_bytes += src_w13.nbytes + src_w2.nbytes

        if args.stream_experts:
            src_w13, src_w2, _ = _load_expert_batch(
                reader,
                layer=layer,
                expert_start=args.hot_experts,
                expert_count=args.stream_experts,
                w13_shape=w13_shape,
                w2_shape=w2_shape,
                dtype=dtype,
                pin_memory=args.pin_memory
                or args.tier_scheduler_policy in UVA_PRIORITY_POLICIES,
            )
            stream_cpu[layer] = (src_w13, src_w2)
            stream_src = (src_w13, src_w2)
            stream_host_bytes += src_w13.nbytes + src_w2.nbytes

        if args.uva_experts:
            src_w13, src_w2, _ = _load_expert_batch(
                reader,
                layer=layer,
                expert_start=args.hot_experts + args.stream_experts,
                expert_count=args.uva_experts,
                w13_shape=w13_shape,
                w2_shape=w2_shape,
                dtype=dtype,
                pin_memory=True,
            )
            uva_w13[layer] = _cuda_view_from_pinned_cpu(src_w13)
            uva_w2[layer] = _cuda_view_from_pinned_cpu(src_w2)
            uva_cpu[layer] = (src_w13, src_w2)
            uva_cpu_refs.append((src_w13, src_w2))
            uva_src = (src_w13, src_w2)
            uva_host_bytes += src_w13.nbytes + src_w2.nbytes

        if args.tier_scheduler_policy in UVA_PRIORITY_POLICIES:
            cold_count = args.stream_experts + args.uva_experts
            if cold_count:
                cold_w13 = _empty_cpu(
                    (cold_count, w13_shape[0], w13_shape[1]),
                    dtype,
                    pin_memory=True,
                )
                cold_w2 = _empty_cpu(
                    (cold_count, w2_shape[0], w2_shape[1]),
                    dtype,
                    pin_memory=True,
                )
                offset = 0
                if stream_src is not None:
                    stream_w13, stream_w2 = stream_src
                    cold_w13[offset : offset + args.stream_experts].copy_(stream_w13)
                    cold_w2[offset : offset + args.stream_experts].copy_(stream_w2)
                    offset += args.stream_experts
                if uva_src is not None:
                    uva_src_w13, uva_src_w2 = uva_src
                    cold_w13[offset : offset + args.uva_experts].copy_(uva_src_w13)
                    cold_w2[offset : offset + args.uva_experts].copy_(uva_src_w2)
                cold_cpu[layer] = (cold_w13, cold_w2)
                cold_uva_w13[layer] = _cuda_view_from_pinned_cpu(cold_w13)
                cold_uva_w2[layer] = _cuda_view_from_pinned_cpu(cold_w2)
                cold_uva_cpu_refs.append((cold_w13, cold_w2))

    torch.cuda.synchronize(device)
    layout = {
        "hot_w13": hot_w13,
        "hot_w2": hot_w2,
        "stream_cpu": stream_cpu,
        "cold_cpu": cold_cpu,
        "uva_cpu": uva_cpu,
        "cold_uva_w13": cold_uva_w13,
        "cold_uva_w2": cold_uva_w2,
        "cold_uva_cpu_refs": cold_uva_cpu_refs,
        "uva_w13": uva_w13,
        "uva_w2": uva_w2,
        "uva_cpu_refs": uva_cpu_refs,
    }
    stats = {
        "layout_seconds": _now() - t0,
        "hot_gpu_bytes": hot_copy_bytes,
        "stream_host_bytes": stream_host_bytes,
        "uva_host_bytes": uva_host_bytes,
        "cold_uva_fallback_enabled": args.tier_scheduler_policy
        in UVA_PRIORITY_POLICIES,
        "hot_copy_gbps": _gbps(hot_copy_bytes, max(_now() - t0, 1e-9)),
    }
    return layout, stats


def _build_all_uva_layout(
    args: argparse.Namespace,
    reader: CheckpointReader,
    layers: list[int],
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    uva_w13: dict[int, torch.Tensor] = {}
    uva_w2: dict[int, torch.Tensor] = {}
    uva_cpu_refs: list[tuple[torch.Tensor, torch.Tensor]] = []
    host_bytes = 0
    t0 = _now()
    for layer in layers:
        src_w13, src_w2, _ = _load_expert_batch(
            reader,
            layer=layer,
            expert_start=0,
            expert_count=args.num_experts,
            w13_shape=w13_shape,
            w2_shape=w2_shape,
            dtype=dtype,
            pin_memory=True,
        )
        uva_w13[layer] = _cuda_view_from_pinned_cpu(src_w13)
        uva_w2[layer] = _cuda_view_from_pinned_cpu(src_w2)
        uva_cpu_refs.append((src_w13, src_w2))
        host_bytes += src_w13.nbytes + src_w2.nbytes
    layout = {
        "uva_w13": uva_w13,
        "uva_w2": uva_w2,
        "uva_cpu_refs": uva_cpu_refs,
    }
    stats = {
        "layout_seconds": _now() - t0,
        "uva_host_bytes": host_bytes,
    }
    return layout, stats


def _build_partial_uva_layout(
    args: argparse.Namespace,
    reader: CheckpointReader,
    layers: list[int],
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    uva_experts = args.partial_uva_experts
    gpu_experts = args.num_experts - uva_experts
    if uva_experts <= 0 or gpu_experts <= 0:
        raise ValueError("partial_uva requires 0 < partial_uva_experts < num_experts")

    gpu_w13: dict[int, torch.Tensor] = {}
    gpu_w2: dict[int, torch.Tensor] = {}
    uva_w13: dict[int, torch.Tensor] = {}
    uva_w2: dict[int, torch.Tensor] = {}
    uva_cpu_refs: list[tuple[torch.Tensor, torch.Tensor]] = []
    gpu_bytes = 0
    uva_host_bytes = 0
    t0 = _now()
    for layer in layers:
        src_w13, src_w2, _ = _load_expert_batch(
            reader,
            layer=layer,
            expert_start=0,
            expert_count=gpu_experts,
            w13_shape=w13_shape,
            w2_shape=w2_shape,
            dtype=dtype,
            pin_memory=args.pin_memory,
        )
        dst_w13, _ = _alloc_gpu_tensor(
            (gpu_experts, w13_shape[0], w13_shape[1]),
            dtype,
            device,
            "cuda",
        )
        dst_w2, _ = _alloc_gpu_tensor(
            (gpu_experts, w2_shape[0], w2_shape[1]),
            dtype,
            device,
            "cuda",
        )
        dst_w13.copy_(src_w13, non_blocking=args.pin_memory)
        dst_w2.copy_(src_w2, non_blocking=args.pin_memory)
        gpu_w13[layer] = dst_w13
        gpu_w2[layer] = dst_w2
        gpu_bytes += src_w13.nbytes + src_w2.nbytes

        src_w13, src_w2, _ = _load_expert_batch(
            reader,
            layer=layer,
            expert_start=gpu_experts,
            expert_count=uva_experts,
            w13_shape=w13_shape,
            w2_shape=w2_shape,
            dtype=dtype,
            pin_memory=True,
        )
        uva_w13[layer] = _cuda_view_from_pinned_cpu(src_w13)
        uva_w2[layer] = _cuda_view_from_pinned_cpu(src_w2)
        uva_cpu_refs.append((src_w13, src_w2))
        uva_host_bytes += src_w13.nbytes + src_w2.nbytes

    torch.cuda.synchronize(device)
    elapsed = _now() - t0
    layout = {
        "gpu_w13": gpu_w13,
        "gpu_w2": gpu_w2,
        "uva_w13": uva_w13,
        "uva_w2": uva_w2,
        "uva_cpu_refs": uva_cpu_refs,
        "gpu_experts": gpu_experts,
        "uva_experts": uva_experts,
    }
    stats = {
        "layout_seconds": elapsed,
        "gpu_experts_per_layer": gpu_experts,
        "uva_experts_per_layer": uva_experts,
        "gpu_resident_bytes": gpu_bytes,
        "gpu_resident_gib": _gib(gpu_bytes),
        "uva_host_bytes": uva_host_bytes,
        "uva_host_gib": _gib(uva_host_bytes),
        "gpu_copy_gbps": _gbps(gpu_bytes, elapsed),
    }
    return layout, stats


def _run_three_tier(
    args: argparse.Namespace,
    layers: list[int],
    layout: dict[str, Any],
    route_plan: dict[int, dict[int, list[int]]],
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    slot_w13, _ = _alloc_gpu_tensor(
        (args.slot_layers, args.stream_experts, w13_shape[0], w13_shape[1]),
        dtype,
        device,
        "cuda",
    )
    slot_w2, _ = _alloc_gpu_tensor(
        (args.slot_layers, args.stream_experts, w2_shape[0], w2_shape[1]),
        dtype,
        device,
        "cuda",
    )
    cold_experts = args.stream_experts + args.uva_experts
    promote_w13: torch.Tensor | None = None
    promote_w2: torch.Tensor | None = None
    sync_miss_w13: torch.Tensor | None = None
    sync_miss_w2: torch.Tensor | None = None
    layer_to_cache_idx: dict[int, int] = {}
    promoted: dict[int, set[int]] = {}
    promotion_pending: dict[int, set[int]] = {}
    promotion_events: dict[tuple[int, int], torch.cuda.Event] = {}
    if args.tier_scheduler_policy in PROMOTION_POLICIES and cold_experts:
        promote_w13, _ = _alloc_gpu_tensor(
            (len(layers), cold_experts, w13_shape[0], w13_shape[1]),
            dtype,
            device,
            "cuda",
        )
        promote_w2, _ = _alloc_gpu_tensor(
            (len(layers), cold_experts, w2_shape[0], w2_shape[1]),
            dtype,
            device,
            "cuda",
        )
        layer_to_cache_idx = {layer: idx for idx, layer in enumerate(layers)}
        promoted = {layer: set() for layer in layers}
        promotion_pending = {layer: set() for layer in layers}
    if args.tier_scheduler_policy in SYNC_MISS_COPY_POLICIES and args.uva_experts:
        sync_miss_w13, _ = _alloc_gpu_tensor(
            (args.uva_experts, w13_shape[0], w13_shape[1]),
            dtype,
            device,
            "cuda",
        )
        sync_miss_w2, _ = _alloc_gpu_tensor(
            (args.uva_experts, w2_shape[0], w2_shape[1]),
            dtype,
            device,
            "cuda",
        )
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)
    checksum = torch.zeros((), dtype=torch.float32, device=device)
    layer_times: list[float] = []
    layer_after_ready_times: list[float] = []
    layer_before_ready_times: list[float] = []
    step_times: list[float] = []
    wait_times: list[float] = []
    copy_submit_times: list[float] = []
    copy_elapsed_times: list[float] = []
    uva_done_offset_times: list[float] = []
    layer_remaining_after_uva_times: list[float] = []
    promotion_tail_offset_times: list[float] = []
    layer_remaining_after_promotion_times: list[float] = []
    promotion_tail_after_layer_end_times: list[float] = []
    promotion_tail_fraction_times: list[float] = []
    sync_miss_copy_elapsed_times: list[float] = []
    records: list[dict[str, Any]] = []
    sim_compute_state = _build_sim_compute_state(args, device)
    pending_layer_events: list[
        tuple[
            torch.cuda.Event,
            torch.cuda.Event,
            torch.cuda.Event,
            int,
            int,
            dict[str, int],
        ]
    ] = []
    pending_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event, int]] = []
    pending_copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    pending_promotion_timing_events: list[
        tuple[
            torch.cuda.Event,
            torch.cuda.Event,
            torch.cuda.Event,
            torch.cuda.Event,
            int,
            int,
            int,
        ]
    ] = []
    pending_sync_miss_copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    stream_start = args.hot_experts
    uva_start = args.hot_experts + args.stream_experts
    layer_to_pos = {layer: idx for idx, layer in enumerate(layers)}

    bank_release_events = [_empty_cuda_event() for _ in range(args.slot_layers)]
    p2_stream = (
        torch.cuda.Stream(device=device)
        if args.tier_scheduler_policy in PROMOTION_GATED_PREFETCH_POLICIES
        else copy_stream
    )
    scheduled_layers: set[int] = set()
    copied_layers: set[int] = set()
    skipped_prefetch_layers: set[int] = set()
    promotion_gated_prefetch_layers: set[int] = set()
    promotion_gated_prefetch_late_layers: set[int] = set()
    promotion_gated_prefetch_events: dict[int, torch.cuda.Event] = {}
    scheduler_counters: dict[str, int] = {
        "prefetch_submitted": 0,
        "prefetch_skipped_due_to_uva": 0,
        "prefetch_after_promotion_submitted": 0,
        "prefetch_after_promotion_ready": 0,
        "prefetch_after_promotion_late": 0,
        "prefetch_after_promotion_used_layers": 0,
        "prefetch_after_promotion_used_stream_experts": 0,
        "prefetch_after_promotion_fallback_layers": 0,
        "prefetch_after_promotion_fallback_stream_experts": 0,
        "uva_p2_overlap_risk_layers": 0,
        "uva_p2_overlap_risk_experts": 0,
        "cold_fallback_layers": 0,
        "cold_fallback_experts": 0,
        "uva_active_layers": 0,
        "uva_active_experts": 0,
        "promotion_enqueued": 0,
        "promotion_submitted": 0,
        "promotion_submitted_after_uva": 0,
        "promotion_completed": 0,
        "promotion_tail_events": 0,
        "promotion_hit_experts": 0,
        "promotion_miss_experts": 0,
        "promotion_active_layers": 0,
        "sync_miss_copy_layers": 0,
        "sync_miss_copy_experts": 0,
    }

    def _empty_ready_tuple() -> tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]:
        ready_event = _empty_cuda_event()
        return ready_event, ready_event, ready_event

    def has_uva_access(step: int, layer: int) -> bool:
        if not args.uva_experts or args.disable_uva_access:
            return False
        return any(expert_id >= uva_start for expert_id in route_plan[step][layer])

    def schedule_layer(
        layer: int,
        events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]],
        *,
        reason: str,
    ) -> bool:
        if not args.stream_experts:
            return False
        if layer in scheduled_layers:
            return False
        if args.disable_runtime_copy:
            events[layer] = _empty_ready_tuple()
            scheduled_layers.add(layer)
            return True
        bank = layer % args.slot_layers
        src_w13, src_w2 = layout["stream_cpu"][layer]
        t = _now()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(bank_release_events[bank])
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_start.record(copy_stream)
            slot_w13[bank].copy_(src_w13, non_blocking=args.pin_memory)
            slot_w2[bank].copy_(src_w2, non_blocking=args.pin_memory)
            copy_end.record(copy_stream)
        copy_submit_times.append(_now() - t)
        events[layer] = (copy_start, copy_end, copy_end)
        scheduled_layers.add(layer)
        copied_layers.add(layer)
        scheduler_counters["prefetch_submitted"] += int(reason == "prefetch")
        return True

    def maybe_schedule_prefetch(
        step: int,
        layer: int,
        events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]],
    ) -> bool:
        if (
            args.tier_scheduler_policy in UVA_PRIORITY_POLICIES
            and has_uva_access(step, layer)
        ):
            events[layer] = _empty_ready_tuple()
            skipped_prefetch_layers.add(layer)
            scheduler_counters["prefetch_skipped_due_to_uva"] += 1
            return False
        return schedule_layer(layer, events, reason="prefetch")

    def maybe_mark_promotion_gated_prefetch_ready(layer: int) -> bool:
        if args.tier_scheduler_policy not in PROMOTION_GATED_PREFETCH_POLICIES:
            return False
        if layer not in promotion_gated_prefetch_layers:
            return False
        if layer in promotion_gated_prefetch_late_layers:
            return False
        ready_event = promotion_gated_prefetch_events[layer]
        if ready_event.query():
            scheduler_counters["prefetch_after_promotion_ready"] += 1
            return True
        promotion_gated_prefetch_late_layers.add(layer)
        scheduler_counters["prefetch_after_promotion_late"] += 1
        scheduler_counters["uva_p2_overlap_risk_layers"] += 1
        return False

    def schedule_prefetch_after_promotion(
        step: int,
        layer: int,
        after_event: torch.cuda.Event,
    ) -> bool:
        if args.tier_scheduler_policy not in PROMOTION_GATED_PREFETCH_POLICIES:
            return False
        if args.copy_policy != "lookahead":
            return False
        if not args.stream_experts or args.disable_runtime_copy:
            return False
        pos = layer_to_pos[layer] + 1
        if pos >= len(layers):
            return False
        target_layer = layers[pos]
        if target_layer in scheduled_layers:
            return False
        if target_layer in promotion_gated_prefetch_layers:
            return False
        if not has_uva_access(step, target_layer):
            return False

        bank = target_layer % args.slot_layers
        src_w13, src_w2 = layout["stream_cpu"][target_layer]
        t = _now()
        with torch.cuda.stream(p2_stream):
            p2_stream.wait_event(after_event)
            p2_stream.wait_event(bank_release_events[bank])
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_start.record(p2_stream)
            slot_w13[bank].copy_(src_w13, non_blocking=args.pin_memory)
            slot_w2[bank].copy_(src_w2, non_blocking=args.pin_memory)
            copy_end.record(p2_stream)
        copy_submit_times.append(_now() - t)
        events[target_layer] = (copy_start, copy_end, copy_end)
        scheduled_layers.add(target_layer)
        copied_layers.add(target_layer)
        promotion_gated_prefetch_layers.add(target_layer)
        promotion_gated_prefetch_events[target_layer] = copy_end
        scheduler_counters["prefetch_submitted"] += 1
        scheduler_counters["prefetch_after_promotion_submitted"] += 1
        return True

    def _new_promotion_ids(layer: int, cold_ids: list[int]) -> list[int]:
        if args.tier_scheduler_policy not in PROMOTION_POLICIES:
            return []
        pending = promotion_pending[layer]
        resident = promoted[layer]
        new_ids: list[int] = []
        for cold_id in cold_ids:
            if cold_id in resident or cold_id in pending:
                continue
            pending.add(cold_id)
            new_ids.append(cold_id)
            scheduler_counters["promotion_enqueued"] += 1
        return new_ids

    def submit_promotions_after_uva(
        layer: int,
        cold_ids: list[int],
        uva_done_event: torch.cuda.Event,
    ) -> torch.cuda.Event | None:
        if args.tier_scheduler_policy not in PROMOTION_POLICIES:
            return None
        new_ids = _new_promotion_ids(layer, cold_ids)
        if not new_ids:
            return None
        assert promote_w13 is not None
        assert promote_w2 is not None
        cache_idx = layer_to_cache_idx[layer]
        src_w13, src_w2 = layout["cold_cpu"][layer]
        tail_event: torch.cuda.Event | None = None
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(uva_done_event)
            for cold_id in new_ids:
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)
                copy_start.record(copy_stream)
                promote_w13[cache_idx, cold_id].copy_(
                    src_w13[cold_id], non_blocking=args.pin_memory
                )
                promote_w2[cache_idx, cold_id].copy_(
                    src_w2[cold_id], non_blocking=args.pin_memory
                )
                copy_end.record(copy_stream)
                pending_copy_events.append((copy_start, copy_end))
                promotion_events[(layer, cold_id)] = copy_end
                scheduler_counters["promotion_submitted"] += 1
                scheduler_counters["promotion_submitted_after_uva"] += 1
                tail_event = copy_end
        promoted[layer].update(new_ids)
        promotion_pending[layer].difference_update(new_ids)
        scheduler_counters["promotion_completed"] += len(new_ids)
        if tail_event is not None:
            scheduler_counters["promotion_tail_events"] += 1
        return tail_event

    start = _now()
    for step in range(args.decode_steps):
        step_wall_start = _now()
        step_start_event = torch.cuda.Event(enable_timing=True)
        step_end_event = torch.cuda.Event(enable_timing=True)
        step_start_event.record(compute_stream)
        events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = {}
        scheduled_layers = set()
        copied_layers = set()
        skipped_prefetch_layers = set()
        promotion_gated_prefetch_layers = set()
        promotion_gated_prefetch_late_layers = set()
        promotion_gated_prefetch_events = {}
        next_idx = 0
        if args.copy_policy == "lookahead":
            for _ in range(min(args.slot_layers, len(layers))):
                maybe_schedule_prefetch(step, layers[next_idx], events)
                next_idx += 1

        for layer in layers:
            layer_wall_start = _now()
            layer_start_event = torch.cuda.Event(enable_timing=True)
            layer_ready_event = torch.cuda.Event(enable_timing=True)
            layer_end_event = torch.cuda.Event(enable_timing=True)
            layer_start_event.record(compute_stream)
            if args.copy_policy == "sync":
                events = {}
                schedule_layer(layer, events, reason="sync")

            if args.tier_scheduler_policy in PROMOTION_GATED_PREFETCH_POLICIES:
                layer_start_event.synchronize()
            promotion_gated_prefetch_ready = (
                maybe_mark_promotion_gated_prefetch_ready(layer)
            )
            promotion_gated_prefetch_late = (
                args.tier_scheduler_policy in PROMOTION_GATED_PREFETCH_POLICIES
                and layer in promotion_gated_prefetch_late_layers
            )
            if args.stream_experts:
                copy_start, copy_end, ready_event = events[layer]
                wait_for_ready_event = not promotion_gated_prefetch_late
                if args.timing_mode == "sync":
                    if wait_for_ready_event:
                        wait_start = _now()
                        compute_stream.wait_event(ready_event)
                        torch.cuda.synchronize(device)
                        wait_times.append(_now() - wait_start)
                    copy_elapsed_times.append(_event_elapsed_seconds(copy_start, copy_end))
                else:
                    if layer in copied_layers:
                        pending_copy_events.append((copy_start, copy_end))
                    if wait_for_ready_event:
                        compute_stream.wait_event(ready_event)

            layer_ready_event.record(compute_stream)
            layer_ready_wall_start = _now()
            expert_ids = route_plan[step][layer]
            hot_ids = [x for x in expert_ids if x < stream_start]
            stream_ids = [x - stream_start for x in expert_ids if stream_start <= x < uva_start]
            uva_ids = [x - uva_start for x in expert_ids if x >= uva_start]
            stream_slot_ids = stream_ids
            cold_fallback_ids: list[int] = []
            promoted_ids: list[int] = []
            sync_miss_ids: list[int] = []
            uva_read_ids = uva_ids
            if args.tier_scheduler_policy in SYNC_MISS_COPY_POLICIES:
                sync_miss_ids = uva_ids
                uva_read_ids = []
            if (
                args.tier_scheduler_policy in UVA_PRIORITY_POLICIES
                and layer in skipped_prefetch_layers
            ):
                uva_read_ids = []
                if promotion_gated_prefetch_ready:
                    stream_slot_ids = stream_ids
                    active_cold_ids = [
                        args.stream_experts + uva_id for uva_id in uva_ids
                    ]
                    scheduler_counters[
                        "prefetch_after_promotion_used_layers"
                    ] += 1
                    scheduler_counters[
                        "prefetch_after_promotion_used_stream_experts"
                    ] += len(stream_ids)
                else:
                    stream_slot_ids = []
                    active_cold_ids = [
                        expert_id - stream_start
                        for expert_id in expert_ids
                        if expert_id >= stream_start
                    ]
                    if promotion_gated_prefetch_late and stream_ids:
                        scheduler_counters[
                            "prefetch_after_promotion_fallback_layers"
                        ] += 1
                        scheduler_counters[
                            "prefetch_after_promotion_fallback_stream_experts"
                        ] += len(stream_ids)
                        scheduler_counters[
                            "uva_p2_overlap_risk_experts"
                        ] += len(active_cold_ids)
                if args.tier_scheduler_policy in PROMOTION_POLICIES:
                    resident = promoted[layer]
                    promoted_ids = [
                        cold_id for cold_id in active_cold_ids if cold_id in resident
                    ]
                    cold_fallback_ids = [
                        cold_id for cold_id in active_cold_ids if cold_id not in resident
                    ]
                    scheduler_counters["promotion_hit_experts"] += len(promoted_ids)
                    scheduler_counters["promotion_miss_experts"] += len(
                        cold_fallback_ids
                    )
                    if promoted_ids:
                        scheduler_counters["promotion_active_layers"] += 1
                else:
                    cold_fallback_ids = active_cold_ids
                if cold_fallback_ids:
                    scheduler_counters["cold_fallback_layers"] += 1
                    scheduler_counters["cold_fallback_experts"] += len(
                        cold_fallback_ids
                    )
            if uva_ids:
                scheduler_counters["uva_active_layers"] += 1
                scheduler_counters["uva_active_experts"] += len(uva_ids)
            if args.tier_scheduler_policy in PROMOTION_POLICIES and promoted_ids:
                for cold_id in promoted_ids:
                    event = promotion_events.get((layer, cold_id))
                    if event is not None:
                        compute_stream.wait_event(event)
            with torch.cuda.stream(compute_stream):
                if sync_miss_ids:
                    assert sync_miss_w13 is not None
                    assert sync_miss_w2 is not None
                    src_uva_w13, src_uva_w2 = layout["uva_cpu"][layer]
                    sync_miss_copy_start = torch.cuda.Event(enable_timing=True)
                    sync_miss_copy_end = torch.cuda.Event(enable_timing=True)
                    sync_miss_copy_start.record(compute_stream)
                    for local_idx, uva_id in enumerate(sync_miss_ids):
                        sync_miss_w13[local_idx].copy_(
                            src_uva_w13[uva_id], non_blocking=args.pin_memory
                        )
                        sync_miss_w2[local_idx].copy_(
                            src_uva_w2[uva_id], non_blocking=args.pin_memory
                        )
                    sync_miss_copy_end.record(compute_stream)
                    pending_sync_miss_copy_events.append(
                        (sync_miss_copy_start, sync_miss_copy_end)
                    )
                    scheduler_counters["sync_miss_copy_layers"] += 1
                    scheduler_counters["sync_miss_copy_experts"] += len(
                        sync_miss_ids
                    )
                parts = [
                    _select_and_sum(
                        layout["hot_w13"][layer],
                        layout["hot_w2"][layer],
                        hot_ids,
                        device=device,
                        touch_elements=args.touch_elements,
                    ),
                    _select_and_sum(
                        slot_w13[layer % args.slot_layers],
                        slot_w2[layer % args.slot_layers],
                        stream_slot_ids,
                        device=device,
                        touch_elements=args.touch_elements,
                    ),
                    _select_and_sum(
                        promote_w13[layer_to_cache_idx[layer]],
                        promote_w2[layer_to_cache_idx[layer]],
                        promoted_ids,
                        device=device,
                        touch_elements=args.touch_elements,
                    )
                    if promoted_ids and promote_w13 is not None and promote_w2 is not None
                    else None,
                    _select_and_sum(
                        layout["cold_uva_w13"][layer],
                        layout["cold_uva_w2"][layer],
                        cold_fallback_ids,
                        device=device,
                        touch_elements=args.touch_elements,
                    )
                    if cold_fallback_ids
                    else None,
                    _select_and_sum(
                        sync_miss_w13,
                        sync_miss_w2,
                        list(range(len(sync_miss_ids))),
                        device=device,
                        touch_elements=args.touch_elements,
                    )
                    if sync_miss_ids
                    and sync_miss_w13 is not None
                    and sync_miss_w2 is not None
                    else None,
                    _select_and_sum(
                        layout["uva_w13"][layer],
                        layout["uva_w2"][layer],
                        uva_read_ids,
                        device=device,
                        touch_elements=args.touch_elements,
                    )
                    if args.uva_experts and uva_read_ids and not args.disable_uva_access
                    else None,
                ]
                for part in parts:
                    if part is not None:
                        checksum = checksum + part
                uva_done_event: torch.cuda.Event | None = None
                uva_done_timing_event: torch.cuda.Event | None = None
                if (
                    args.tier_scheduler_policy in PROMOTION_POLICIES
                    and cold_fallback_ids
                ):
                    uva_done_event = torch.cuda.Event(enable_timing=False)
                    uva_done_timing_event = torch.cuda.Event(enable_timing=True)
                    uva_done_event.record(compute_stream)
                    uva_done_timing_event.record(compute_stream)
                sim_part = _simulate_token_compute(
                    sim_compute_state, args.sim_compute_iters
                )
                if sim_part is not None:
                    checksum = checksum + sim_part
                layer_end_event.record(compute_stream)

            if (
                args.tier_scheduler_policy in PROMOTION_POLICIES
                and cold_fallback_ids
                and uva_done_event is not None
            ):
                promotion_tail_event = submit_promotions_after_uva(
                    layer, cold_fallback_ids, uva_done_event
                )
                if promotion_tail_event is not None:
                    pending_promotion_timing_events.append(
                        (
                            layer_start_event,
                            uva_done_timing_event
                            if uva_done_timing_event is not None
                            else uva_done_event,
                            promotion_tail_event,
                            layer_end_event,
                            step,
                            layer,
                            len(cold_fallback_ids),
                        )
                    )
                    schedule_prefetch_after_promotion(
                        step, layer, promotion_tail_event
                    )
            if promotion_gated_prefetch_late:
                bank_release_events[layer % args.slot_layers] = (
                    promotion_gated_prefetch_events[layer]
                )
            else:
                bank_release_events[layer % args.slot_layers] = layer_end_event
            if args.timing_mode == "sync":
                torch.cuda.synchronize(device)
                layer_done = _now()
                elapsed = layer_done - layer_wall_start
                after_ready_elapsed = layer_done - layer_ready_wall_start
                before_ready_elapsed = max(0.0, layer_ready_wall_start - layer_wall_start)
                layer_times.append(elapsed)
                layer_after_ready_times.append(after_ready_elapsed)
                layer_before_ready_times.append(before_ready_elapsed)
                if len(records) < args.trace_records:
                    counts = _count_tiers(args, expert_ids)
                    records.append(
                        {
                            "step": step,
                            "layer": layer,
                            "active_experts": counts,
                            "layer_seconds": elapsed,
                            "layer_total_seconds": elapsed,
                            "layer_after_ready_seconds": after_ready_elapsed,
                            "layer_before_ready_seconds": before_ready_elapsed,
                        }
                    )
            else:
                counts = _count_tiers(args, expert_ids)
                pending_layer_events.append(
                    (
                        layer_start_event,
                        layer_ready_event,
                        layer_end_event,
                        step,
                        layer,
                        counts,
                    )
                )

            if args.copy_policy == "lookahead" and next_idx < len(layers):
                maybe_schedule_prefetch(step, layers[next_idx], events)
                next_idx += 1

        step_end_event.record(compute_stream)
        if args.timing_mode == "sync":
            torch.cuda.synchronize(device)
            step_times.append(_now() - step_wall_start)
        else:
            pending_step_events.append((step_start_event, step_end_event, step))

    torch.cuda.synchronize(device)
    step_records: list[dict[str, Any]] = []
    if args.timing_mode == "event":
        copy_elapsed_times.extend(
            _event_elapsed_seconds(start_event, end_event)
            for start_event, end_event in pending_copy_events
        )
        sync_miss_copy_elapsed_times.extend(
            _event_elapsed_seconds(start_event, end_event)
            for start_event, end_event in pending_sync_miss_copy_events
        )
        for (
            start_event,
            uva_done_event,
            promotion_tail_event,
            end_event,
            _step,
            _layer,
            _cold_count,
        ) in pending_promotion_timing_events:
            layer_elapsed = _event_elapsed_seconds(start_event, end_event)
            uva_done_offset = _event_elapsed_seconds(start_event, uva_done_event)
            promotion_tail_offset = _event_elapsed_seconds(
                start_event, promotion_tail_event
            )
            layer_remaining_after_uva = max(0.0, layer_elapsed - uva_done_offset)
            layer_remaining_after_promotion = max(
                0.0, layer_elapsed - promotion_tail_offset
            )
            promotion_tail_after_layer_end = max(
                0.0, promotion_tail_offset - layer_elapsed
            )
            uva_done_offset_times.append(uva_done_offset)
            layer_remaining_after_uva_times.append(layer_remaining_after_uva)
            promotion_tail_offset_times.append(promotion_tail_offset)
            layer_remaining_after_promotion_times.append(
                layer_remaining_after_promotion
            )
            promotion_tail_after_layer_end_times.append(
                promotion_tail_after_layer_end
            )
            if layer_elapsed > 0:
                promotion_tail_fraction_times.append(
                    promotion_tail_offset / layer_elapsed
                )
        for (
            start_event,
            ready_event,
            end_event,
            step,
            layer,
            counts,
        ) in pending_layer_events:
            total_elapsed = _event_elapsed_seconds(start_event, end_event)
            after_ready_elapsed = _event_elapsed_seconds(ready_event, end_event)
            before_ready_elapsed = _event_elapsed_seconds(start_event, ready_event)
            layer_times.append(total_elapsed)
            layer_after_ready_times.append(after_ready_elapsed)
            layer_before_ready_times.append(before_ready_elapsed)
            if len(records) < args.trace_records:
                records.append(
                    {
                        "step": step,
                        "layer": layer,
                        "active_experts": counts,
                        "layer_seconds": total_elapsed,
                        "layer_total_seconds": total_elapsed,
                        "layer_after_ready_seconds": after_ready_elapsed,
                        "layer_before_ready_seconds": before_ready_elapsed,
                    }
                )
        for start_event, end_event, step in pending_step_events:
            elapsed = _event_elapsed_seconds(start_event, end_event)
            step_times.append(elapsed)
            step_records.append({"step": step, "step_seconds": elapsed})
    else:
        step_records = [
            {"step": step, "step_seconds": elapsed}
            for step, elapsed in enumerate(step_times)
        ]
    return {
        "runtime_seconds": _now() - start,
        "step_latency": _stats(step_times),
        "layer_latency": _stats(layer_times),
        "layer_total_latency": _stats(layer_times),
        "layer_after_ready_latency": _stats(layer_after_ready_times),
        "layer_before_ready_latency": _stats(layer_before_ready_times),
        "copy_wait": _stats(wait_times),
        "copy_submit": _stats(copy_submit_times),
        "copy_elapsed": _stats(copy_elapsed_times),
        "sync_miss_copy_elapsed": _stats(sync_miss_copy_elapsed_times),
        "promotion_timing": {
            "uva_done_offset": _stats(uva_done_offset_times),
            "layer_remaining_after_uva": _stats(layer_remaining_after_uva_times),
            "promotion_tail_offset": _stats(promotion_tail_offset_times),
            "layer_remaining_after_promotion": _stats(
                layer_remaining_after_promotion_times
            ),
            "promotion_tail_after_layer_end": _stats(
                promotion_tail_after_layer_end_times
            ),
            "promotion_tail_fraction_of_layer": _float_stats(
                promotion_tail_fraction_times
            ),
        },
        "tier_scheduler": {
            "policy": args.tier_scheduler_policy,
            **scheduler_counters,
        },
        "checksum": float(checksum.detach().cpu().item()),
        "records_sample": records,
        "step_records": step_records,
    }


def _run_all_uva(
    args: argparse.Namespace,
    layers: list[int],
    layout: dict[str, Any],
    route_plan: dict[int, dict[int, list[int]]],
    device: torch.device,
) -> dict[str, Any]:
    checksum = torch.zeros((), dtype=torch.float32, device=device)
    layer_times: list[float] = []
    step_times: list[float] = []
    step_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    sim_compute_state = _build_sim_compute_state(args, device)
    start = _now()
    for step in range(args.decode_steps):
        step_start = _now()
        for layer in layers:
            layer_start = _now()
            expert_ids = route_plan[step][layer]
            with torch.cuda.stream(torch.cuda.current_stream(device=device)):
                part = _select_and_sum(
                    layout["uva_w13"][layer],
                    layout["uva_w2"][layer],
                    expert_ids,
                    device=device,
                    touch_elements=args.touch_elements,
                )
                if part is not None:
                    checksum = checksum + part
                sim_part = _simulate_token_compute(
                    sim_compute_state, args.sim_compute_iters
                )
                if sim_part is not None:
                    checksum = checksum + sim_part
            torch.cuda.synchronize(device)
            elapsed = _now() - layer_start
            layer_times.append(elapsed)
            if len(records) < args.trace_records:
                records.append(
                    {
                        "step": step,
                        "layer": layer,
                        "active_experts": _count_tiers(args, expert_ids),
                        "layer_seconds": elapsed,
                    }
                )
        elapsed = _now() - step_start
        step_times.append(elapsed)
        step_records.append({"step": step, "step_seconds": elapsed})
    torch.cuda.synchronize(device)
    return {
        "runtime_seconds": _now() - start,
        "step_latency": _stats(step_times),
        "layer_latency": _stats(layer_times),
        "checksum": float(checksum.detach().cpu().item()),
        "records_sample": records,
        "step_records": step_records,
    }


def _run_partial_uva(
    args: argparse.Namespace,
    layers: list[int],
    layout: dict[str, Any],
    route_plan: dict[int, dict[int, list[int]]],
    device: torch.device,
) -> dict[str, Any]:
    checksum = torch.zeros((), dtype=torch.float32, device=device)
    layer_times: list[float] = []
    step_times: list[float] = []
    step_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    gpu_experts = int(layout["gpu_experts"])
    sim_compute_state = _build_sim_compute_state(args, device)
    start = _now()
    for step in range(args.decode_steps):
        step_start = _now()
        for layer in layers:
            layer_start = _now()
            expert_ids = route_plan[step][layer]
            gpu_ids = [x for x in expert_ids if x < gpu_experts]
            uva_ids = [x - gpu_experts for x in expert_ids if x >= gpu_experts]
            with torch.cuda.stream(torch.cuda.current_stream(device=device)):
                parts = [
                    _select_and_sum(
                        layout["gpu_w13"][layer],
                        layout["gpu_w2"][layer],
                        gpu_ids,
                        device=device,
                        touch_elements=args.touch_elements,
                    ),
                    _select_and_sum(
                        layout["uva_w13"][layer],
                        layout["uva_w2"][layer],
                        uva_ids,
                        device=device,
                        touch_elements=args.touch_elements,
                    ),
                ]
                for part in parts:
                    if part is not None:
                        checksum = checksum + part
                sim_part = _simulate_token_compute(
                    sim_compute_state, args.sim_compute_iters
                )
                if sim_part is not None:
                    checksum = checksum + sim_part
            torch.cuda.synchronize(device)
            elapsed = _now() - layer_start
            layer_times.append(elapsed)
            if len(records) < args.trace_records:
                tier_counts = _count_tiers(args, expert_ids)
                records.append(
                    {
                        "step": step,
                        "layer": layer,
                        "active_experts": tier_counts,
                        "active_backing": {
                            "gpu": len(gpu_ids),
                            "uva": len(uva_ids),
                            "total": len(expert_ids),
                        },
                        "layer_seconds": elapsed,
                    }
                )
        elapsed = _now() - step_start
        step_times.append(elapsed)
        step_records.append({"step": step, "step_seconds": elapsed})
    torch.cuda.synchronize(device)
    return {
        "runtime_seconds": _now() - start,
        "step_latency": _stats(step_times),
        "layer_latency": _stats(layer_times),
        "checksum": float(checksum.detach().cpu().item()),
        "records_sample": records,
        "step_records": step_records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    layers = _parse_layers(args)
    if args.mode not in ("three_tier", "all_uva", "partial_uva"):
        raise ValueError(f"unknown mode {args.mode!r}")
    if args.slot_layers < 1:
        raise ValueError("slot_layers must be positive")
    if args.copy_policy == "lookahead" and args.slot_layers < 2:
        raise ValueError("lookahead copy_policy requires slot_layers >= 2")
    if args.tier_scheduler_policy != "naive" and args.mode != "three_tier":
        raise ValueError("tier_scheduler_policy is only supported in three_tier mode")
    if args.mode == "all_uva":
        args.hot_experts = 64
        args.stream_experts = 32
        args.uva_experts = 32
    if args.mode == "three_tier":
        args.uva_experts = args.num_experts - args.hot_experts - args.stream_experts
        if args.uva_experts < 0:
            raise ValueError("hot_experts + stream_experts must be <= num_experts")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    reader = CheckpointReader(args.checkpoint_dir)
    try:
        shapes = reader.expert_shapes(layers[0], 0)
        dtype = reader.expert_tensor(layers[0], 0, "gate_proj").dtype
        w13_shape = tuple(shapes["w13_shape"])
        w2_shape = tuple(shapes["w2_shape"])
        per_expert_bytes = (
            math.prod(w13_shape) + math.prod(w2_shape)
        ) * _dtype_bytes(dtype)
        route_plan = _build_route_plan(args, layers)
        route_summary = _summarize_route_plan(args, route_plan)

        total_start = _now()
        if args.mode == "three_tier":
            layout, layout_stats = _build_three_tier_layout(
                args, reader, layers, w13_shape, w2_shape, dtype, device
            )
            runtime = _run_three_tier(
                args, layers, layout, route_plan, w13_shape, w2_shape, dtype, device
            )
        elif args.mode == "all_uva":
            layout, layout_stats = _build_all_uva_layout(
                args, reader, layers, w13_shape, w2_shape, dtype
            )
            runtime = _run_all_uva(args, layers, layout, route_plan, device)
        else:
            layout, layout_stats = _build_partial_uva_layout(
                args, reader, layers, w13_shape, w2_shape, dtype, device
            )
            runtime = _run_partial_uva(args, layers, layout, route_plan, device)

        result = {
            "mode": args.mode,
            "config": _jsonable_config(args),
            "layers": layers,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "shapes": {
                **shapes,
                "per_expert_bytes": per_expert_bytes,
                "per_expert_mib": per_expert_bytes / (1024**2),
            },
            "route_summary": route_summary,
            "layout": layout_stats,
            "runtime": runtime,
            "total_seconds": _now() - total_start,
            "torch_cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "torch_cuda_max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "torch_cuda_max_memory_allocated_gib": _gib(
                torch.cuda.max_memory_allocated(device)
            ),
            "passed": True,
        }
        return result
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--mode", choices=("three_tier", "all_uva", "partial_uva"), required=True
    )
    parser.add_argument("--layers", default="")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hot-experts", type=int, default=64)
    parser.add_argument("--stream-experts", type=int, default=32)
    parser.add_argument("--uva-experts", type=int, default=32)
    parser.add_argument(
        "--partial-uva-experts",
        type=int,
        default=48,
        help="tail experts per layer backed by UVA in partial_uva mode",
    )
    parser.add_argument("--slot-layers", type=int, default=2)
    parser.add_argument("--copy-policy", choices=("sync", "lookahead"), default="lookahead")
    parser.add_argument(
        "--tier-scheduler-policy",
        choices=(
            "naive",
            "uva_priority",
            "uva_priority_promote",
            "uva_priority_promote_prefetch",
            "sync_miss_copy",
        ),
        default="naive",
        help=(
            "three_tier only: naive keeps lookahead copies unchanged; "
            "uva_priority skips P2 stream prefetch for layers with active UVA "
            "and reads skipped cold experts through pinned UVA fallback; "
            "uva_priority_promote promotes first-touch cold misses to a GPU cache "
            "after UVA fallback; uva_priority_promote_prefetch additionally "
            "starts next-layer P2 prefetch after the P1 promotion tail event and "
            "uses it only if ready before the target layer starts; "
            "sync_miss_copy synchronously copies active UVA-tier misses to a GPU "
            "miss cache after expert ids are known, then computes from the cache"
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("event", "sync"),
        default="event",
        help="event avoids per-layer device synchronize and preserves copy/compute overlap",
    )
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--hot-prob", type=float, default=0.80)
    parser.add_argument("--stream-prob", type=float, default=0.135)
    parser.add_argument("--uva-prob", type=float, default=0.065)
    parser.add_argument(
        "--touch-elements",
        type=int,
        default=0,
        help="0 touches full selected expert weights; positive values touch a prefix",
    )
    parser.add_argument(
        "--sim-compute-dim",
        type=int,
        default=0,
        help="adds a per-layer matmul over num_tokens rows when positive",
    )
    parser.add_argument(
        "--sim-compute-iters",
        type=int,
        default=1,
        help="number of simulated per-layer matmul iterations",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--route-mode",
        choices=("weighted", "fixed_counts"),
        default="weighted",
    )
    parser.add_argument("--fixed-hot", type=int, default=0)
    parser.add_argument("--fixed-stream", type=int, default=0)
    parser.add_argument("--fixed-uva", type=int, default=0)
    parser.add_argument(
        "--disable-runtime-copy",
        action="store_true",
        help="record ready events without issuing stream slot copies; only valid when stream ids are not consumed",
    )
    parser.add_argument(
        "--disable-uva-access",
        action="store_true",
        help="keep UVA ids in route stats but skip reading UVA tensors at runtime",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace-records", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
