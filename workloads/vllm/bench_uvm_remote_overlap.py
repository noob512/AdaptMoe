#!/usr/bin/env python3
"""Measure whether UVM remote reads slow down during UVM background promotion."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from bench_hot_cold_expert_checkpoint import DEFAULT_CHECKPOINT_DIR, CheckpointReader, _tiny_forward  # noqa: E402
from bench_uvm_remote_bg_vs_copy import (  # noqa: E402
    _load_expert_batch_cpu,
    _load_one_expert_gpu,
    _prepare_remote_bg,
    _prepare_remote_read,
    _prefetch_ranges_cpu,
    _ranges_for,
    _start_bg_promotion,
)


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[len(ordered) // 2],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _forward_event_timed(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    stream = torch.cuda.current_stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    if w13.dim() == 3:
        out = torch.zeros((hidden,), dtype=torch.float32, device=device)
        for local in range(w13.shape[0]):
            out = out + _tiny_forward(
                w13[local],
                w2[local],
                hidden=hidden,
                intermediate=intermediate,
                x=x,
            )
    else:
        out = _tiny_forward(w13, w2, hidden=hidden, intermediate=intermediate, x=x)
    end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)), out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    reader = CheckpointReader(args.checkpoint_dir)
    try:
        shapes = reader.expert_shapes(args.layer, args.read_expert)
        dtype = reader.expert_tensor(args.layer, args.read_expert, "gate_proj").dtype
        w13_shape = tuple(shapes["w13_shape"])
        w2_shape = tuple(shapes["w2_shape"])
        hidden = min(args.hidden, w13_shape[1], w2_shape[0])
        intermediate = min(args.intermediate, w13_shape[0] // 2, w2_shape[1])

        src_read_w13, src_read_w2, read_load_seconds = _load_expert_batch_cpu(
            reader,
            layer=args.layer,
            expert=args.read_expert,
            expert_count=args.read_expert_count,
            w13_shape=w13_shape,
            w2_shape=w2_shape,
            dtype=dtype,
            pin_memory=False,
        )
        src_bg_w13, src_bg_w2, bg_load_seconds = _load_expert_batch_cpu(
            reader,
            layer=args.layer,
            expert=args.bg_expert,
            expert_count=args.bg_expert_count,
            w13_shape=w13_shape,
            w2_shape=w2_shape,
            dtype=dtype,
            pin_memory=False,
        )

        read_w13, read_w2, read_backing, read_init_seconds = _load_one_expert_gpu(
            src_read_w13,
            src_read_w2,
            device=device,
            backing="uvm",
        )
        bg_w13, bg_w2, bg_backing, bg_init_seconds = _load_one_expert_gpu(
            src_bg_w13,
            src_bg_w2,
            device=device,
            backing="uvm",
        )

        read_ranges = _ranges_for(read_w13, read_w2)
        bg_ranges = _ranges_for(bg_w13, bg_w2)
        x = torch.linspace(-0.25, 0.25, steps=hidden, dtype=torch.float32, device=device)

        _prefetch_ranges_cpu(read_ranges)
        _prepare_remote_read(read_ranges, device=args.device)
        _prefetch_ranges_cpu(bg_ranges)
        _prepare_remote_bg(bg_ranges, device=args.device)
        torch.cuda.synchronize(device)

        # Warm both paths.
        _forward_event_timed(
            read_w13,
            read_w2,
            hidden=hidden,
            intermediate=intermediate,
            x=x,
            device=device,
        )
        _start_bg_promotion(bg_ranges, device=args.device)
        _forward_event_timed(
            read_w13,
            read_w2,
            hidden=hidden,
            intermediate=intermediate,
            x=x,
            device=device,
        )
        torch.cuda.synchronize(device)

        baseline_ms: list[float] = []
        overlap_ms: list[float] = []
        promote_submit_ms: list[float] = []

        for _ in range(args.repeats):
            _prefetch_ranges_cpu(read_ranges)
            _prepare_remote_read(read_ranges, device=args.device)
            torch.cuda.synchronize(device)
            elapsed_ms, _ = _forward_event_timed(
                read_w13,
                read_w2,
                hidden=hidden,
                intermediate=intermediate,
                x=x,
                device=device,
            )
            baseline_ms.append(elapsed_ms)

            _prefetch_ranges_cpu(read_ranges)
            _prepare_remote_read(read_ranges, device=args.device)
            _prefetch_ranges_cpu(bg_ranges)
            _prepare_remote_bg(bg_ranges, device=args.device)
            torch.cuda.synchronize(device)

            submit_start = time.perf_counter()
            _start_bg_promotion(bg_ranges, device=args.device)
            promote_submit_ms.append((time.perf_counter() - submit_start) * 1000.0)
            elapsed_ms, _ = _forward_event_timed(
                read_w13,
                read_w2,
                hidden=hidden,
                intermediate=intermediate,
                x=x,
                device=device,
            )
            overlap_ms.append(elapsed_ms)
            torch.cuda.synchronize(device)

        baseline_p50 = sorted(baseline_ms)[len(baseline_ms) // 2]
        overlap_p50 = sorted(overlap_ms)[len(overlap_ms) // 2]
        return {
            "config": {
                "checkpoint_dir": str(args.checkpoint_dir),
                "layer": args.layer,
                "read_expert": args.read_expert,
                "read_expert_count": args.read_expert_count,
                "bg_expert": args.bg_expert,
                "bg_expert_count": args.bg_expert_count,
                "hidden": hidden,
                "intermediate": intermediate,
                "repeats": args.repeats,
                "device": args.device,
            },
            "shapes": shapes,
            "effective_backing": {
                "read": read_backing,
                "background": bg_backing,
            },
            "load_seconds": {
                "read_cpu": read_load_seconds,
                "background_cpu": bg_load_seconds,
                "read_uvm_init": read_init_seconds,
                "background_uvm_init": bg_init_seconds,
            },
            "baseline_uvm_remote_ms": _stats(baseline_ms),
            "uvm_remote_with_bg_promotion_ms": _stats(overlap_ms),
            "bg_promotion_submit_ms": _stats(promote_submit_ms),
            "slowdown_ms_p50": overlap_p50 - baseline_p50,
            "slowdown_ratio_p50": overlap_p50 / baseline_p50,
        }
    finally:
        reader.close()
        try:
            from vllm.device_allocator.uvm import cleanup_uvm_local_memory_pool

            cleanup_uvm_local_memory_pool()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--read-expert", type=int, default=96)
    parser.add_argument("--read-expert-count", type=int, default=4)
    parser.add_argument("--bg-expert", type=int, default=64)
    parser.add_argument("--bg-expert-count", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
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
