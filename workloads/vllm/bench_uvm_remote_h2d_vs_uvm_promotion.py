#!/usr/bin/env python3
"""Compare UVM remote reads under ordinary H2D copy vs UVM promotion."""

from __future__ import annotations

import argparse
import ctypes
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

from bench_hot_cold_expert_checkpoint import DEFAULT_CHECKPOINT_DIR, CheckpointReader  # noqa: E402
from bench_uvm_remote_bg_vs_copy import (  # noqa: E402
    _load_expert_batch_cpu,
    _load_one_expert_gpu,
    _prepare_remote_bg,
    _prepare_remote_read,
    _prefetch_ranges_cpu,
    _ranges_for,
    _start_bg_promotion,
)
from bench_uvm_remote_overlap import _forward_event_timed  # noqa: E402
from bench_uvm_stage_boundary_remap import _check, _load_probe  # noqa: E402


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


def _probe_event(lib: ctypes.CDLL) -> ctypes.c_void_p:
    event = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_event(ctypes.byref(event), 1), "create_event")
    return event


def _probe_elapsed_ms(lib: ctypes.CDLL, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
    value = ctypes.c_float()
    _check(lib.uvm_stage_probe_elapsed_ms(start, end, ctypes.byref(value)), "elapsed_ms")
    return float(value.value)


def _alloc_pinned(lib: ctypes.CDLL, bytes_: int, pattern: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_alloc_pinned_host(ctypes.byref(ptr), bytes_), "alloc_pinned_host")
    ctypes.memset(int(ptr.value or 0), pattern, bytes_)
    return ptr


def _alloc_device(lib: ctypes.CDLL, bytes_: int, device: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_alloc_device(ctypes.byref(ptr), bytes_, device), "alloc_device")
    return ptr


def _h2d_copy_with_timing(
    lib: ctypes.CDLL,
    *,
    copy_stream: ctypes.c_void_p,
    copy_src: ctypes.c_void_p,
    copy_dst: ctypes.c_void_p,
    copy_bytes: int,
) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
    start = _probe_event(lib)
    end = _probe_event(lib)
    _check(lib.uvm_stage_probe_record_event(start, copy_stream), "record_copy_start")
    _check(
        lib.uvm_stage_probe_copy_host_to_device_async(
            copy_dst,
            copy_src,
            copy_bytes,
            copy_stream,
        ),
        "copy_host_to_device_async",
    )
    _check(lib.uvm_stage_probe_record_event(end, copy_stream), "record_copy_end")
    return start, end


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
            pin_memory=True,
        )

        read_w13, read_w2, read_backing, read_init_seconds = _load_one_expert_gpu(
            src_read_w13,
            src_read_w2,
            device=device,
            backing="uvm",
        )
        bg_uvm_w13, bg_uvm_w2, bg_uvm_backing, bg_uvm_init_seconds = _load_one_expert_gpu(
            src_bg_w13,
            src_bg_w2,
            device=device,
            backing="uvm",
        )
        torch.cuda.synchronize(device)

        background_bytes = int(
            src_bg_w13.numel() * src_bg_w13.element_size()
            + src_bg_w2.numel() * src_bg_w2.element_size()
        )
        lib = _load_probe()
        _check(lib.uvm_stage_probe_set_device(args.device), "probe_set_device")
        copy_stream = ctypes.c_void_p()
        _check(lib.uvm_stage_probe_create_stream(ctypes.byref(copy_stream)), "create_copy_stream")
        copy_src = _alloc_pinned(lib, background_bytes, 0x3C)
        copy_dst = _alloc_device(lib, background_bytes, args.device)

        read_ranges = _ranges_for(read_w13, read_w2)
        bg_uvm_ranges = _ranges_for(bg_uvm_w13, bg_uvm_w2)
        x = torch.linspace(-0.25, 0.25, steps=hidden, dtype=torch.float32, device=device)

        # Warm all paths once.
        _prefetch_ranges_cpu(read_ranges)
        _prepare_remote_read(read_ranges, device=args.device)
        _prefetch_ranges_cpu(bg_uvm_ranges)
        _prepare_remote_bg(bg_uvm_ranges, device=args.device)
        torch.cuda.synchronize(device)
        _forward_event_timed(
            read_w13,
            read_w2,
            hidden=hidden,
            intermediate=intermediate,
            x=x,
            device=device,
        )
        copy_start, copy_end = _h2d_copy_with_timing(
            lib,
            copy_stream=copy_stream,
            copy_src=copy_src,
            copy_dst=copy_dst,
            copy_bytes=background_bytes,
        )
        _forward_event_timed(
            read_w13,
            read_w2,
            hidden=hidden,
            intermediate=intermediate,
            x=x,
            device=device,
        )
        _check(lib.uvm_stage_probe_sync_event(copy_end), "sync_warm_copy")
        lib.uvm_stage_probe_destroy_event(copy_start)
        lib.uvm_stage_probe_destroy_event(copy_end)
        _start_bg_promotion(bg_uvm_ranges, device=args.device)
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
        h2d_overlap_ms: list[float] = []
        h2d_copy_ms: list[float] = []
        uvm_promotion_overlap_ms: list[float] = []
        uvm_promotion_submit_ms: list[float] = []

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
            torch.cuda.synchronize(device)
            copy_start, copy_end = _h2d_copy_with_timing(
                lib,
                copy_stream=copy_stream,
                copy_src=copy_src,
                copy_dst=copy_dst,
                copy_bytes=background_bytes,
            )
            elapsed_ms, _ = _forward_event_timed(
                read_w13,
                read_w2,
                hidden=hidden,
                intermediate=intermediate,
                x=x,
                device=device,
            )
            _check(lib.uvm_stage_probe_sync_event(copy_end), "sync_h2d_copy")
            h2d_overlap_ms.append(elapsed_ms)
            h2d_copy_ms.append(_probe_elapsed_ms(lib, copy_start, copy_end))
            lib.uvm_stage_probe_destroy_event(copy_start)
            lib.uvm_stage_probe_destroy_event(copy_end)

            _prefetch_ranges_cpu(read_ranges)
            _prepare_remote_read(read_ranges, device=args.device)
            _prefetch_ranges_cpu(bg_uvm_ranges)
            _prepare_remote_bg(bg_uvm_ranges, device=args.device)
            torch.cuda.synchronize(device)
            submit_start = time.perf_counter()
            _start_bg_promotion(bg_uvm_ranges, device=args.device)
            uvm_promotion_submit_ms.append((time.perf_counter() - submit_start) * 1000.0)
            elapsed_ms, _ = _forward_event_timed(
                read_w13,
                read_w2,
                hidden=hidden,
                intermediate=intermediate,
                x=x,
                device=device,
            )
            uvm_promotion_overlap_ms.append(elapsed_ms)
            torch.cuda.synchronize(device)

        baseline_p50 = sorted(baseline_ms)[len(baseline_ms) // 2]
        h2d_p50 = sorted(h2d_overlap_ms)[len(h2d_overlap_ms) // 2]
        uvm_p50 = sorted(uvm_promotion_overlap_ms)[len(uvm_promotion_overlap_ms) // 2]
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
            "bytes": {
                "read_total": int(read_w13.numel() * read_w13.element_size() + read_w2.numel() * read_w2.element_size()),
                "background_total": background_bytes,
            },
            "shapes": shapes,
            "effective_backing": {
                "read": read_backing,
                "background_uvm": bg_uvm_backing,
                "background_h2d": "raw_cuda_memcpy_async_pinned_to_device",
            },
            "load_seconds": {
                "read_cpu": read_load_seconds,
                "background_cpu": bg_load_seconds,
                "read_uvm_init": read_init_seconds,
                "background_uvm_init": bg_uvm_init_seconds,
            },
            "baseline_uvm_remote_ms": _stats(baseline_ms),
            "uvm_remote_with_h2d_copy_ms": _stats(h2d_overlap_ms),
            "background_h2d_copy_ms": _stats(h2d_copy_ms),
            "uvm_remote_with_uvm_promotion_ms": _stats(uvm_promotion_overlap_ms),
            "background_uvm_promotion_submit_ms": _stats(uvm_promotion_submit_ms),
            "h2d_slowdown_ms_p50": h2d_p50 - baseline_p50,
            "h2d_slowdown_ratio_p50": h2d_p50 / baseline_p50,
            "uvm_promotion_slowdown_ms_p50": uvm_p50 - baseline_p50,
            "uvm_promotion_slowdown_ratio_p50": uvm_p50 / baseline_p50,
        }
    finally:
        for name in ("copy_src", "copy_dst", "copy_stream", "lib"):
            if name not in locals():
                continue
            try:
                if name == "copy_src":
                    lib.uvm_stage_probe_free_pinned_host(copy_src)
                elif name == "copy_dst":
                    lib.uvm_stage_probe_free_device(copy_dst)
                elif name == "copy_stream":
                    lib.uvm_stage_probe_destroy_stream(copy_stream)
            except Exception:
                pass
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
