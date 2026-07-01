#!/usr/bin/env python3
"""Compare native cudaMemPrefetchAsync on UVM memory with ordinary H2D copy."""

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

from bench_hot_cold_expert_checkpoint import _alloc_gpu_tensor  # noqa: E402
from bench_uvm_promotion_vs_h2d_copy import _query_residency  # noqa: E402
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


def _event(lib: ctypes.CDLL) -> ctypes.c_void_p:
    event = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_event(ctypes.byref(event), 1), "create_event")
    return event


def _elapsed_ms(lib: ctypes.CDLL, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
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


def _run_h2d_once(
    lib: ctypes.CDLL,
    *,
    stream: ctypes.c_void_p,
    src: ctypes.c_void_p,
    dst: ctypes.c_void_p,
    bytes_: int,
) -> tuple[float, float]:
    start = _event(lib)
    end = _event(lib)
    try:
        submit_start = time.perf_counter()
        _check(lib.uvm_stage_probe_record_event(start, stream), "record_h2d_start")
        _check(
            lib.uvm_stage_probe_copy_host_to_device_async(dst, src, bytes_, stream),
            "copy_host_to_device_async",
        )
        _check(lib.uvm_stage_probe_record_event(end, stream), "record_h2d_end")
        submit_ms = (time.perf_counter() - submit_start) * 1000.0
        _check(lib.uvm_stage_probe_sync_event(end), "sync_h2d_end")
        return submit_ms, _elapsed_ms(lib, start, end)
    finally:
        lib.uvm_stage_probe_destroy_event(start)
        lib.uvm_stage_probe_destroy_event(end)


def _prefetch_ranges_cpu(ranges: list[tuple[int, int]]) -> int:
    from vllm.device_allocator.uvm import batch_prefetch_ranges_to_cpu

    return int(batch_prefetch_ranges_to_cpu(ranges))


def _prefetch_ranges_gpu(ranges: list[tuple[int, int]], device: int) -> int:
    from vllm.device_allocator.uvm import batch_prefetch_ranges_to_device

    return int(batch_prefetch_ranges_to_device(ranges, device=device))


def _run_uvm_prefetch_once(
    *,
    bytes_: int,
    device: torch.device,
    device_index: int,
) -> dict[str, Any]:
    tensor, backing = _alloc_gpu_tensor((bytes_,), torch.uint8, device, "uvm")
    if backing != "uvm":
        raise RuntimeError(f"expected UVM backing, got {backing}")

    tensor.fill_(0xA5)
    torch.cuda.synchronize(device)
    ranges = [(int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size()))]

    cpu_prefetch_start = time.perf_counter()
    cpu_prefetch_result = _prefetch_ranges_cpu(ranges)
    torch.cuda.synchronize(device)
    cpu_prefetch_ms = (time.perf_counter() - cpu_prefetch_start) * 1000.0
    before = _query_residency(ranges, device_index)

    stream = torch.cuda.current_stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    submit_start = time.perf_counter()
    start.record(stream)
    prefetch_result = _prefetch_ranges_gpu(ranges, device_index)
    end.record(stream)
    submit_ms = (time.perf_counter() - submit_start) * 1000.0
    end.synchronize()
    gpu_ms = float(start.elapsed_time(end))
    after = _query_residency(ranges, device_index)

    del tensor
    return {
        "cpu_prefetch_ms": cpu_prefetch_ms,
        "cpu_prefetch_result": cpu_prefetch_result,
        "before_residency": before,
        "prefetch_submit_ms": submit_ms,
        "prefetch_gpu_ms": gpu_ms,
        "prefetch_result": prefetch_result,
        "after_residency": after,
        "complete_ok": int(after.get("gpu_bytes", 0)) >= bytes_,
    }


def _parse_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.endswith("mib"):
            sizes.append(int(float(token[:-3]) * 1024 * 1024))
        elif token.endswith("mb"):
            sizes.append(int(float(token[:-2]) * 1000 * 1000))
        elif token.endswith("kib"):
            sizes.append(int(float(token[:-3]) * 1024))
        elif token.endswith("kb"):
            sizes.append(int(float(token[:-2]) * 1000))
        else:
            sizes.append(int(token))
    return sizes


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    lib = _load_probe()
    _check(lib.uvm_stage_probe_set_device(args.device), "set_device")
    h2d_stream = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_stream(ctypes.byref(h2d_stream)), "create_h2d_stream")

    rows: list[dict[str, Any]] = []
    try:
        for bytes_ in _parse_sizes(args.sizes):
            h2d_src = _alloc_pinned(lib, bytes_, 0x3C)
            h2d_dst = _alloc_device(lib, bytes_, args.device)
            try:
                _run_h2d_once(lib, stream=h2d_stream, src=h2d_src, dst=h2d_dst, bytes_=bytes_)

                h2d_submit_ms: list[float] = []
                h2d_gpu_ms: list[float] = []
                uvm_cpu_prefetch_ms: list[float] = []
                uvm_submit_ms: list[float] = []
                uvm_gpu_ms: list[float] = []
                uvm_ok = 0
                before_first: dict[str, Any] | None = None
                after_last: dict[str, Any] | None = None
                prefetch_results: list[int] = []

                for _ in range(args.repeats):
                    submit_ms, copy_ms = _run_h2d_once(
                        lib,
                        stream=h2d_stream,
                        src=h2d_src,
                        dst=h2d_dst,
                        bytes_=bytes_,
                    )
                    h2d_submit_ms.append(submit_ms)
                    h2d_gpu_ms.append(copy_ms)

                    uvm = _run_uvm_prefetch_once(
                        bytes_=bytes_,
                        device=device,
                        device_index=args.device,
                    )
                    uvm_cpu_prefetch_ms.append(float(uvm["cpu_prefetch_ms"]))
                    uvm_submit_ms.append(float(uvm["prefetch_submit_ms"]))
                    uvm_gpu_ms.append(float(uvm["prefetch_gpu_ms"]))
                    uvm_ok += int(bool(uvm["complete_ok"]))
                    if before_first is None:
                        before_first = uvm["before_residency"]
                    after_last = uvm["after_residency"]
                    prefetch_results.append(int(uvm["prefetch_result"]))

                h2d_p50 = sorted(h2d_gpu_ms)[len(h2d_gpu_ms) // 2]
                uvm_p50 = sorted(uvm_gpu_ms)[len(uvm_gpu_ms) // 2]
                rows.append({
                    "bytes": bytes_,
                    "mib": bytes_ / 1024 / 1024,
                    "repeats": args.repeats,
                    "h2d_submit_ms": _stats(h2d_submit_ms),
                    "h2d_copy_gpu_ms": _stats(h2d_gpu_ms),
                    "uvm_cpu_prefetch_setup_ms": _stats(uvm_cpu_prefetch_ms),
                    "uvm_prefetch_submit_ms": _stats(uvm_submit_ms),
                    "uvm_prefetch_gpu_ms": _stats(uvm_gpu_ms),
                    "uvm_over_h2d_gpu_ratio_p50": uvm_p50 / h2d_p50,
                    "uvm_complete_ok_count": uvm_ok,
                    "uvm_before_first": before_first,
                    "uvm_after_last": after_last,
                    "uvm_prefetch_results": prefetch_results,
                })
            finally:
                lib.uvm_stage_probe_free_pinned_host(h2d_src)
                lib.uvm_stage_probe_free_device(h2d_dst)
                torch.cuda.empty_cache()
    finally:
        lib.uvm_stage_probe_destroy_stream(h2d_stream)
        try:
            from vllm.device_allocator.uvm import cleanup_uvm_local_memory_pool

            cleanup_uvm_local_memory_pool()
        except Exception:
            pass

    return {
        "config": {
            "device": args.device,
            "sizes": args.sizes,
            "repeats": args.repeats,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="4.5MiB,18MiB,72MiB,144MiB,288MiB",
        help="Comma-separated byte sizes. Supports MiB/MB/KiB/KB suffixes.",
    )
    parser.add_argument("--repeats", type=int, default=5)
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
