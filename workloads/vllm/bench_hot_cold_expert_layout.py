#!/usr/bin/env python3
"""Standalone hot/cold MoE expert layout microbenchmark.

This script intentionally does not start vLLM. It mimics the part of vLLM
weight loading that matters for the current design decision:

  CPU checkpoint tensor -> GPU parameter tensor copy_

Hot experts are materialized as per-layer GPU tensors. Cold experts keep a
CPU canonical source and are copied into a fixed two-layer GPU slot pool.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))


SHAPES: dict[str, dict[str, tuple[int, int]]] = {
    "toy": {
        "w13": (256, 256),
        "w2": (256, 128),
    },
    # Qwen3-30B-A3B FP8 expert byte shapes observed in the current traces:
    # w13_weight.expert_i: [1536, 2048], w2_weight.expert_i: [2048, 768].
    "qwen3_fp8": {
        "w13": (1536, 2048),
        "w2": (2048, 768),
    },
}


DTYPES: dict[str, torch.dtype] = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _now() -> float:
    return time.perf_counter()


def _bytes(numel: int, dtype: torch.dtype) -> int:
    return numel * torch.empty((), dtype=dtype).element_size()


def _gib(value: int | float) -> float:
    return float(value) / (1024**3)


def _gbps(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(num_bytes) / seconds / 1e9


def _parse_shape(args: argparse.Namespace) -> tuple[tuple[int, int], tuple[int, int]]:
    if args.shape_name not in SHAPES:
        raise ValueError(f"unknown shape {args.shape_name!r}")
    shape = SHAPES[args.shape_name]
    return shape["w13"], shape["w2"]


def _pattern(layer_id: int, expert_id: int, role: str) -> int:
    role_bias = 17 if role == "w13" else 83
    return (layer_id * 37 + expert_id * 11 + role_bias) % 251


def _make_cpu_expert_batch(
    *,
    layer_id: int,
    expert_start: int,
    expert_count: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
    role: str,
    pin_memory: bool,
) -> torch.Tensor:
    try:
        tensor = torch.empty(
            (expert_count, shape[0], shape[1]),
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
    except RuntimeError:
        tensor = torch.empty(
            (expert_count, shape[0], shape[1]),
            dtype=dtype,
            device="cpu",
        )
    for local_id in range(expert_count):
        expert_id = expert_start + local_id
        tensor[local_id].fill_(_pattern(layer_id, expert_id, role))
    return tensor


def _maybe_uvm_context(enabled: bool) -> Iterator[None]:
    if not enabled:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        from vllm.device_allocator.uvm import uvm_local_memory_pool

        with uvm_local_memory_pool():
            yield

    return _ctx()


def _alloc_gpu_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    backing: str,
) -> tuple[torch.Tensor, str]:
    if backing == "cuda":
        return torch.empty(shape, dtype=dtype, device=device), "cuda"
    if backing == "uvm":
        with _maybe_uvm_context(True):
            return torch.empty(shape, dtype=dtype, device=device), "uvm"
    if backing == "auto":
        try:
            with _maybe_uvm_context(True):
                return torch.empty(shape, dtype=dtype, device=device), "uvm"
        except Exception as exc:
            print(
                f"[bench] hot UVM allocation failed, falling back to cudaMalloc: {exc}",
                file=sys.stderr,
            )
            return torch.empty(shape, dtype=dtype, device=device), "cuda"
    raise ValueError(f"unknown backing {backing!r}")


def _estimate(args: argparse.Namespace) -> dict[str, Any]:
    w13_shape, w2_shape = _parse_shape(args)
    dtype = DTYPES[args.dtype]
    per_expert_bytes = _bytes(math.prod(w13_shape) + math.prod(w2_shape), dtype)
    hot_bytes = args.num_layers * args.hot_experts * per_expert_bytes
    cold_slot_bytes = args.slot_layers * args.cold_experts * per_expert_bytes
    cold_layer_stage_bytes = args.cold_experts * per_expert_bytes
    hot_layer_stage_bytes = args.hot_experts * per_expert_bytes
    total_gpu_bytes = hot_bytes + cold_slot_bytes
    return {
        "w13_shape": list(w13_shape),
        "w2_shape": list(w2_shape),
        "dtype": args.dtype,
        "per_expert_bytes": per_expert_bytes,
        "per_expert_mib": per_expert_bytes / (1024**2),
        "hot_gpu_bytes": hot_bytes,
        "hot_gpu_gib": _gib(hot_bytes),
        "cold_slot_gpu_bytes": cold_slot_bytes,
        "cold_slot_gpu_gib": _gib(cold_slot_bytes),
        "cold_layer_stage_bytes": cold_layer_stage_bytes,
        "cold_layer_stage_gib": _gib(cold_layer_stage_bytes),
        "hot_layer_stage_bytes": hot_layer_stage_bytes,
        "hot_layer_stage_gib": _gib(hot_layer_stage_bytes),
        "estimated_total_gpu_bytes": total_gpu_bytes,
        "estimated_total_gpu_gib": _gib(total_gpu_bytes),
    }


def _jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def _verify_role(
    tensor: torch.Tensor,
    *,
    layer_id: int,
    expert_ids: list[int],
    expert_start: int,
    role: str,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for expert_id in expert_ids:
        local_id = expert_id - expert_start
        got = int(tensor[local_id, 0, 0].detach().cpu().item())
        expected = _pattern(layer_id, expert_id, role)
        if got != expected:
            errors.append(
                {
                    "expert_id": expert_id,
                    "local_id": local_id,
                    "role": role,
                    "got": got,
                    "expected": expected,
                }
            )
    return {"ok": not errors, "errors": errors}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_experts != args.hot_experts + args.cold_experts:
        raise ValueError("num_experts must equal hot_experts + cold_experts")
    if args.hot_experts <= 0 or args.cold_experts <= 0:
        raise ValueError("hot_experts and cold_experts must be positive")

    estimate = _estimate(args)
    if args.dry_run:
        return {"dry_run": True, "config": _jsonable_config(args), "estimate": estimate}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    dtype = DTYPES[args.dtype]
    w13_shape, w2_shape = _parse_shape(args)

    result: dict[str, Any] = {
        "dry_run": False,
        "config": _jsonable_config(args),
        "estimate": estimate,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "events": [],
    }

    start_total = _now()

    # 1. Hot experts: first 64 contiguous experts per layer, permanently GPU
    # resident. This mimics vLLM's load path by allocating the destination on
    # GPU, then copy_ from CPU checkpoint tensors.
    hot_w13: list[torch.Tensor] = []
    hot_w2: list[torch.Tensor] = []
    effective_hot_backings: set[str] = set()
    hot_copy_bytes = 0
    hot_start = _now()
    for layer_id in range(args.num_layers):
        dst_w13, backing_w13 = _alloc_gpu_tensor(
            (args.hot_experts, w13_shape[0], w13_shape[1]),
            dtype,
            device,
            args.hot_backing,
        )
        dst_w2, backing_w2 = _alloc_gpu_tensor(
            (args.hot_experts, w2_shape[0], w2_shape[1]),
            dtype,
            device,
            args.hot_backing,
        )
        effective_hot_backings.update([backing_w13, backing_w2])

        src_w13 = _make_cpu_expert_batch(
            layer_id=layer_id,
            expert_start=0,
            expert_count=args.hot_experts,
            shape=w13_shape,
            dtype=dtype,
            role="w13",
            pin_memory=args.pin_memory,
        )
        src_w2 = _make_cpu_expert_batch(
            layer_id=layer_id,
            expert_start=0,
            expert_count=args.hot_experts,
            shape=w2_shape,
            dtype=dtype,
            role="w2",
            pin_memory=args.pin_memory,
        )
        dst_w13.copy_(src_w13, non_blocking=args.pin_memory)
        dst_w2.copy_(src_w2, non_blocking=args.pin_memory)
        hot_copy_bytes += src_w13.nbytes + src_w2.nbytes
        hot_w13.append(dst_w13)
        hot_w2.append(dst_w2)

    torch.cuda.synchronize(device)
    hot_seconds = _now() - hot_start
    result["hot"] = {
        "effective_backings": sorted(effective_hot_backings),
        "copy_bytes": hot_copy_bytes,
        "copy_seconds": hot_seconds,
        "copy_gbps": _gbps(hot_copy_bytes, hot_seconds),
    }

    # 2. Cold slots: fixed two-layer GPU pool. expert_id 64 maps to slot 0,
    # expert_id 127 maps to slot 63. The bank is layer_id % slot_layers.
    slot_w13, effective_slot_w13 = _alloc_gpu_tensor(
        (args.slot_layers, args.cold_experts, w13_shape[0], w13_shape[1]),
        dtype,
        device,
        args.slot_backing,
    )
    slot_w2, effective_slot_w2 = _alloc_gpu_tensor(
        (args.slot_layers, args.cold_experts, w2_shape[0], w2_shape[1]),
        dtype,
        device,
        args.slot_backing,
    )

    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)
    copy_events: dict[int, torch.cuda.Event] = {}
    staging_refs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    cold_copy_bytes = 0
    cold_copy_seconds_accum = 0.0
    access_seconds_accum = 0.0
    checksum = torch.zeros((), dtype=torch.float32, device=device)

    def schedule_cold_layer(layer_id: int) -> None:
        nonlocal cold_copy_bytes, cold_copy_seconds_accum
        bank = layer_id % args.slot_layers
        src_w13 = _make_cpu_expert_batch(
            layer_id=layer_id,
            expert_start=args.hot_experts,
            expert_count=args.cold_experts,
            shape=w13_shape,
            dtype=dtype,
            role="w13",
            pin_memory=args.pin_memory,
        )
        src_w2 = _make_cpu_expert_batch(
            layer_id=layer_id,
            expert_start=args.hot_experts,
            expert_count=args.cold_experts,
            shape=w2_shape,
            dtype=dtype,
            role="w2",
            pin_memory=args.pin_memory,
        )
        t0 = _now()
        with torch.cuda.stream(copy_stream):
            slot_w13[bank].copy_(src_w13, non_blocking=args.pin_memory)
            slot_w2[bank].copy_(src_w2, non_blocking=args.pin_memory)
            event = torch.cuda.Event(enable_timing=False)
            event.record(copy_stream)
        copy_events[layer_id] = event
        staging_refs[layer_id] = (src_w13, src_w2)
        cold_copy_seconds_accum += _now() - t0
        cold_copy_bytes += src_w13.nbytes + src_w2.nbytes

    def access_layer(layer_id: int) -> dict[str, Any]:
        nonlocal checksum, access_seconds_accum
        bank = layer_id % args.slot_layers
        expert_ids = sorted(
            random.sample(
                range(args.hot_experts, args.num_experts),
                k=min(args.topk, args.cold_experts),
            )
        )
        local_ids = [expert_id - args.hot_experts for expert_id in expert_ids]
        ids_gpu = torch.tensor(local_ids, dtype=torch.long, device=device)
        touch = args.touch_elements
        t0 = _now()
        with torch.cuda.stream(compute_stream):
            flat_w13 = slot_w13[bank].reshape(args.cold_experts, -1)
            flat_w2 = slot_w2[bank].reshape(args.cold_experts, -1)
            selected_w13 = flat_w13.index_select(0, ids_gpu)
            selected_w2 = flat_w2.index_select(0, ids_gpu)
            if touch > 0:
                selected_w13 = selected_w13[:, :touch]
                selected_w2 = selected_w2[:, :touch]
            checksum = checksum + selected_w13.float().sum() + selected_w2.float().sum()
        torch.cuda.synchronize(device)
        elapsed = _now() - t0
        access_seconds_accum += elapsed
        record: dict[str, Any] = {
            "layer": layer_id,
            "bank": bank,
            "expert_ids": expert_ids,
            "access_seconds": elapsed,
        }
        if args.verify:
            errors: list[dict[str, Any]] = []
            for role, tensor in (("w13", slot_w13[bank]), ("w2", slot_w2[bank])):
                check = _verify_role(
                    tensor,
                    layer_id=layer_id,
                    expert_ids=expert_ids,
                    expert_start=args.hot_experts,
                    role=role,
                )
                errors.extend(check["errors"])
            record["verify_ok"] = not errors
            record["verify_errors"] = errors[:4]
        return record

    # Preload up to slot_layers layers. Then each layer access frees that bank
    # for the layer slot_layers steps ahead.
    next_to_schedule = 0
    preload = min(args.slot_layers, args.num_layers)
    for _ in range(preload):
        schedule_cold_layer(next_to_schedule)
        next_to_schedule += 1

    random_access_records: list[dict[str, Any]] = []
    cold_start = _now()
    for layer_id in range(args.num_layers):
        compute_stream.wait_event(copy_events[layer_id])
        if args.think_ms > 0:
            time.sleep(args.think_ms / 1000.0)
        random_access_records.append(access_layer(layer_id))
        staging_refs.pop(layer_id, None)

        if next_to_schedule < args.num_layers:
            schedule_cold_layer(next_to_schedule)
            next_to_schedule += 1

    torch.cuda.synchronize(device)
    cold_seconds = _now() - cold_start
    result["cold"] = {
        "slot_backings": sorted({effective_slot_w13, effective_slot_w2}),
        "slot_layers": args.slot_layers,
        "slot_experts_per_layer": args.cold_experts,
        "expert_id_to_slot": "slot = expert_id - hot_experts",
        "copy_bytes": cold_copy_bytes,
        "wall_seconds": cold_seconds,
        "copy_submit_seconds": cold_copy_seconds_accum,
        "copy_wall_gbps": _gbps(cold_copy_bytes, cold_seconds),
        "access_seconds": access_seconds_accum,
        "random_access_records_sample": random_access_records[: args.trace_layers],
        "checksum": float(checksum.detach().cpu().item()),
    }

    verify_errors: list[dict[str, Any]] = []
    if args.verify:
        hot_sample = [0, min(args.hot_experts - 1, 1), args.hot_experts - 1]
        cold_sample = [
            args.hot_experts,
            min(args.num_experts - 1, args.hot_experts + 1),
            args.num_experts - 1,
        ]
        last_hot_layer = args.num_layers - 1
        for role, tensors in (("w13", hot_w13), ("w2", hot_w2)):
            check = _verify_role(
                tensors[last_hot_layer],
                layer_id=last_hot_layer,
                expert_ids=hot_sample,
                expert_start=0,
                role=role,
            )
            verify_errors.extend(check["errors"])

        # Verify every resident cold bank corresponds to the most recent layer
        # mapped to that bank.
        for bank in range(args.slot_layers):
            resident_layers = [
                layer_id
                for layer_id in range(args.num_layers)
                if layer_id % args.slot_layers == bank
            ]
            if not resident_layers:
                continue
            layer_id = resident_layers[-1]
            for role, tensor in (("w13", slot_w13[bank]), ("w2", slot_w2[bank])):
                check = _verify_role(
                    tensor,
                    layer_id=layer_id,
                    expert_ids=cold_sample,
                    expert_start=args.hot_experts,
                    role=role,
                )
                verify_errors.extend(check["errors"])
        for record in random_access_records:
            verify_errors.extend(record.get("verify_errors") or [])

    result["verify"] = {"ok": not verify_errors, "errors": verify_errors[:16]}
    result["total_seconds"] = _now() - start_total
    result["torch_cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
    result["torch_cuda_max_memory_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    result["passed"] = result["verify"]["ok"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hot-experts", type=int, default=64)
    parser.add_argument("--cold-experts", type=int, default=64)
    parser.add_argument("--slot-layers", type=int, default=2)
    parser.add_argument("--shape-name", choices=sorted(SHAPES), default="toy")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="uint8")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--hot-backing",
        choices=("auto", "uvm", "cuda"),
        default="auto",
        help="auto tries local UVM first and falls back to cudaMalloc",
    )
    parser.add_argument(
        "--slot-backing",
        choices=("uvm", "cuda"),
        default="cuda",
        help="cold slot pool should normally be native cudaMalloc",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--touch-elements",
        type=int,
        default=4096,
        help="per selected expert; 0 means touch the full expert tensor",
    )
    parser.add_argument("--think-ms", type=float, default=0.0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace-layers", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    try:
        result = run(args)
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(text + "\n", encoding="utf-8")
        return 0 if result.get("passed", True) else 1
    finally:
        try:
            from vllm.device_allocator.uvm import cleanup_uvm_local_memory_pool

            cleanup_uvm_local_memory_pool()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
