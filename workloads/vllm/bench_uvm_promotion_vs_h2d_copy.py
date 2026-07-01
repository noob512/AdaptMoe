#!/usr/bin/env python3
"""Measure UVM background promotion cost against ordinary H2D copy."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
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
from bench_uvm_remote_bg_vs_copy import (  # noqa: E402
    _prepare_remote_bg,
    _prefetch_ranges_cpu,
    _start_bg_promotion,
)
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


def _query_residency(ranges: list[tuple[int, int]], device: int) -> dict[str, int | bool]:
    from vllm.device_allocator import uvm

    range_list = [(int(ptr), int(size)) for ptr, size in ranges if int(ptr) and int(size)]
    result: dict[str, int | bool] = {
        "ok": False,
        "status": -1,
        "queried_bytes": 0,
        "gpu_bytes": 0,
        "cpu_bytes": 0,
        "any_bytes": 0,
        "unknown_bytes": 0,
    }
    if not range_list:
        result["ok"] = True
        result["status"] = 0
        return result

    fd = -1
    try:
        gpu_uuid = uvm._phase_f_gpu_uuid(device)  # pylint: disable=protected-access
        fd = uvm._open_or_adopt_initialized_uvm_fd(  # pylint: disable=protected-access
            prefer_existing=True
        )
        params = uvm._PhaseFBatchExecutorParams()  # pylint: disable=protected-access
        params.action = 4  # UVM_PHASE_F_ACTION_QUERY_RESIDENCY
        params.count = len(range_list)
        params.gpu_uuid = gpu_uuid
        params.async_ = 0
        for index, (ptr, size) in enumerate(range_list):
            params.ranges[index].base = ptr
            params.ranges[index].length = size
        ret = uvm._libc_ioctl(  # pylint: disable=protected-access
            fd,
            uvm._UVM_PHASE_F_BATCH_EXECUTOR_IOCTL,  # pylint: disable=protected-access
            params,
        )
        result["status"] = int(ret if ret != 0 else params.rmStatus)
        result["ok"] = ret == 0 and int(params.rmStatus) == 0
        result["queried_bytes"] = sum(int(params.results[i].queried_bytes) for i in range(len(range_list)))
        result["gpu_bytes"] = sum(int(params.results[i].resident_gpu_bytes) for i in range(len(range_list)))
        result["cpu_bytes"] = sum(int(params.results[i].resident_cpu_bytes) for i in range(len(range_list)))
        result["any_bytes"] = sum(int(params.results[i].resident_any_bytes) for i in range(len(range_list)))
        result["unknown_bytes"] = sum(int(params.results[i].resident_unknown_bytes) for i in range(len(range_list)))
        return result
    except Exception:
        return result
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


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


def _run_uvm_promotion_once(
    *,
    bytes_: int,
    device: torch.device,
    device_index: int,
    timeout_ms: float,
    poll_sleep_ms: float,
) -> dict[str, Any]:
    tensor, backing = _alloc_gpu_tensor((bytes_,), torch.uint8, device, "uvm")
    if backing != "uvm":
        raise RuntimeError(f"expected UVM backing, got {backing}")
    tensor.fill_(0x5A)
    torch.cuda.synchronize(device)
    ranges = [(int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size()))]

    prefetch_start = time.perf_counter()
    prefetch_result = _prefetch_ranges_cpu(ranges)
    prefetch_cpu_ms = (time.perf_counter() - prefetch_start) * 1000.0
    before = _query_residency(ranges, device_index)

    prepare_start = time.perf_counter()
    prepare_result = _prepare_remote_bg(ranges, device=device_index)
    prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

    start = time.perf_counter()
    promotion_result = _start_bg_promotion(ranges, device=device_index)
    start_submit_ms = (time.perf_counter() - start) * 1000.0

    poll_count = 0
    deadline = start + timeout_ms / 1000.0
    last = _query_residency(ranges, device_index)
    while time.perf_counter() < deadline:
        poll_count += 1
        if int(last.get("gpu_bytes", 0)) >= bytes_:
            break
        if poll_sleep_ms > 0:
            time.sleep(poll_sleep_ms / 1000.0)
        last = _query_residency(ranges, device_index)

    complete_ms = (time.perf_counter() - start) * 1000.0
    ok = int(last.get("gpu_bytes", 0)) >= bytes_
    # Keep tensor alive until after final query, then let the pool reclaim later.
    del tensor
    return {
        "prefetch_cpu_ms": prefetch_cpu_ms,
        "prefetch_result": list(prefetch_result),
        "before_residency": before,
        "prepare_ms": prepare_ms,
        "prepare_result": list(prepare_result),
        "start_submit_ms": start_submit_ms,
        "start_result": list(promotion_result),
        "complete_ms": complete_ms,
        "complete_ok": ok,
        "poll_count": poll_count,
        "after_residency": last,
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
    stream = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_stream(ctypes.byref(stream)), "create_h2d_stream")

    rows: list[dict[str, Any]] = []
    try:
        for bytes_ in _parse_sizes(args.sizes):
            h2d_src = _alloc_pinned(lib, bytes_, 0x3C)
            h2d_dst = _alloc_device(lib, bytes_, args.device)
            try:
                # Warm ordinary copy once.
                _run_h2d_once(lib, stream=stream, src=h2d_src, dst=h2d_dst, bytes_=bytes_)

                h2d_submit_ms: list[float] = []
                h2d_copy_ms: list[float] = []
                uvm_prefetch_cpu_ms: list[float] = []
                uvm_prepare_ms: list[float] = []
                uvm_start_submit_ms: list[float] = []
                uvm_complete_ms: list[float] = []
                uvm_poll_counts: list[float] = []
                uvm_complete_ok = 0
                uvm_before: list[dict[str, int | bool]] = []
                uvm_after: list[dict[str, int | bool]] = []
                uvm_prepare_results: list[list[int]] = []
                uvm_start_results: list[list[int]] = []

                for _ in range(args.repeats):
                    submit_ms, copy_ms = _run_h2d_once(
                        lib,
                        stream=stream,
                        src=h2d_src,
                        dst=h2d_dst,
                        bytes_=bytes_,
                    )
                    h2d_submit_ms.append(submit_ms)
                    h2d_copy_ms.append(copy_ms)

                    uvm = _run_uvm_promotion_once(
                        bytes_=bytes_,
                        device=device,
                        device_index=args.device,
                        timeout_ms=args.timeout_ms,
                        poll_sleep_ms=args.poll_sleep_ms,
                    )
                    uvm_prefetch_cpu_ms.append(float(uvm["prefetch_cpu_ms"]))
                    uvm_prepare_ms.append(float(uvm["prepare_ms"]))
                    uvm_start_submit_ms.append(float(uvm["start_submit_ms"]))
                    uvm_complete_ms.append(float(uvm["complete_ms"]))
                    uvm_poll_counts.append(float(uvm["poll_count"]))
                    uvm_complete_ok += int(bool(uvm["complete_ok"]))
                    uvm_before.append(uvm["before_residency"])
                    uvm_after.append(uvm["after_residency"])
                    uvm_prepare_results.append(uvm["prepare_result"])
                    uvm_start_results.append(uvm["start_result"])

                h2d_copy_p50 = sorted(h2d_copy_ms)[len(h2d_copy_ms) // 2]
                uvm_complete_p50 = sorted(uvm_complete_ms)[len(uvm_complete_ms) // 2]
                rows.append({
                    "bytes": bytes_,
                    "mib": bytes_ / 1024 / 1024,
                    "repeats": args.repeats,
                    "h2d_submit_ms": _stats(h2d_submit_ms),
                    "h2d_copy_gpu_ms": _stats(h2d_copy_ms),
                    "uvm_prefetch_cpu_ms": _stats(uvm_prefetch_cpu_ms),
                    "uvm_prepare_remote_bg_ms": _stats(uvm_prepare_ms),
                    "uvm_start_submit_ms": _stats(uvm_start_submit_ms),
                    "uvm_complete_ms": _stats(uvm_complete_ms),
                    "uvm_poll_count": _stats(uvm_poll_counts),
                    "uvm_complete_ok_count": uvm_complete_ok,
                    "uvm_vs_h2d_complete_ratio_p50": uvm_complete_p50 / h2d_copy_p50,
                    "uvm_before_first": uvm_before[0],
                    "uvm_after_last": uvm_after[-1],
                    "uvm_prepare_results": uvm_prepare_results,
                    "uvm_start_results": uvm_start_results,
                })
            finally:
                lib.uvm_stage_probe_free_pinned_host(h2d_src)
                lib.uvm_stage_probe_free_device(h2d_dst)
                torch.cuda.empty_cache()
    finally:
        lib.uvm_stage_probe_destroy_stream(stream)
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
            "timeout_ms": args.timeout_ms,
            "poll_sleep_ms": args.poll_sleep_ms,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="4.5MiB,18MiB,72MiB,144MiB",
        help="Comma-separated byte sizes. Supports MiB/MB/KiB/KB suffixes.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout-ms", type=float, default=2000.0)
    parser.add_argument("--poll-sleep-ms", type=float, default=0.05)
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
