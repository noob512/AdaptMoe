#!/usr/bin/env python3
"""Measure whether background H2D copy slows a UVA host read kernel."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

from bench_uvm_stage_boundary_remap import _check, _load_probe


def _event(lib: ctypes.CDLL) -> ctypes.c_void_p:
    event = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_event(ctypes.byref(event), 1), "create_event")
    return event


def _elapsed_ms(lib: ctypes.CDLL, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
    value = ctypes.c_float()
    _check(lib.uvm_stage_probe_elapsed_ms(start, end, ctypes.byref(value)), "elapsed_ms")
    return float(value.value)


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": ordered[len(ordered) // 2],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _alloc_pinned(lib: ctypes.CDLL, bytes_: int, pattern: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_alloc_pinned_host(ctypes.byref(ptr), bytes_), "alloc_pinned_host")
    ctypes.memset(int(ptr.value or 0), pattern, bytes_)
    return ptr


def _alloc_device(lib: ctypes.CDLL, bytes_: int, device: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_alloc_device(ctypes.byref(ptr), bytes_, device), "alloc_device")
    return ptr


def _run_read(
    lib: ctypes.CDLL,
    *,
    read_stream: ctypes.c_void_p,
    read_src: ctypes.c_void_p,
    read_dst: ctypes.c_void_p,
    read_bytes: int,
    read_iters: int,
) -> float:
    start = _event(lib)
    end = _event(lib)
    try:
        _check(lib.uvm_stage_probe_record_event(start, read_stream), "record_read_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                read_src,
                read_dst,
                read_bytes,
                read_iters,
                read_stream,
            ),
            "launch_uva_read",
        )
        _check(lib.uvm_stage_probe_record_event(end, read_stream), "record_read_end")
        _check(lib.uvm_stage_probe_sync_event(end), "sync_read_end")
        return _elapsed_ms(lib, start, end)
    finally:
        lib.uvm_stage_probe_destroy_event(start)
        lib.uvm_stage_probe_destroy_event(end)


def _run_read_with_copy(
    lib: ctypes.CDLL,
    *,
    read_stream: ctypes.c_void_p,
    copy_stream: ctypes.c_void_p,
    read_src: ctypes.c_void_p,
    read_dst: ctypes.c_void_p,
    read_bytes: int,
    read_iters: int,
    copy_src: ctypes.c_void_p,
    copy_dst: ctypes.c_void_p,
    copy_bytes: int,
) -> tuple[float, float]:
    read_start = _event(lib)
    read_end = _event(lib)
    copy_start = _event(lib)
    copy_end = _event(lib)
    try:
        _check(lib.uvm_stage_probe_record_event(copy_start, copy_stream), "record_copy_start")
        _check(
            lib.uvm_stage_probe_copy_host_to_device_async(
                copy_dst,
                copy_src,
                copy_bytes,
                copy_stream,
            ),
            "copy_host_to_device_async",
        )
        _check(lib.uvm_stage_probe_record_event(copy_end, copy_stream), "record_copy_end")

        _check(lib.uvm_stage_probe_record_event(read_start, read_stream), "record_read_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                read_src,
                read_dst,
                read_bytes,
                read_iters,
                read_stream,
            ),
            "launch_uva_read_with_copy",
        )
        _check(lib.uvm_stage_probe_record_event(read_end, read_stream), "record_read_end")

        _check(lib.uvm_stage_probe_sync_event(read_end), "sync_read_end")
        _check(lib.uvm_stage_probe_sync_event(copy_end), "sync_copy_end")
        return _elapsed_ms(lib, read_start, read_end), _elapsed_ms(lib, copy_start, copy_end)
    finally:
        for event in (read_start, read_end, copy_start, copy_end):
            lib.uvm_stage_probe_destroy_event(event)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--read-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--copy-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--read-iters", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    lib = _load_probe()
    _check(lib.uvm_stage_probe_set_device(args.device), "set_device")

    read_stream = ctypes.c_void_p()
    copy_stream = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_stream(ctypes.byref(read_stream)), "create_read_stream")
    _check(lib.uvm_stage_probe_create_stream(ctypes.byref(copy_stream)), "create_copy_stream")

    read_src = _alloc_pinned(lib, args.read_bytes, 0xA5)
    copy_src = _alloc_pinned(lib, args.copy_bytes, 0x3C)
    read_dst = _alloc_device(lib, args.read_bytes, args.device)
    copy_dst = _alloc_device(lib, args.copy_bytes, args.device)

    try:
        # Warm up both paths once.
        _run_read(
            lib,
            read_stream=read_stream,
            read_src=read_src,
            read_dst=read_dst,
            read_bytes=args.read_bytes,
            read_iters=args.read_iters,
        )
        _run_read_with_copy(
            lib,
            read_stream=read_stream,
            copy_stream=copy_stream,
            read_src=read_src,
            read_dst=read_dst,
            read_bytes=args.read_bytes,
            read_iters=args.read_iters,
            copy_src=copy_src,
            copy_dst=copy_dst,
            copy_bytes=args.copy_bytes,
        )

        baseline: list[float] = []
        with_copy: list[float] = []
        copy_times: list[float] = []
        for _ in range(args.repeats):
            baseline.append(
                _run_read(
                    lib,
                    read_stream=read_stream,
                    read_src=read_src,
                    read_dst=read_dst,
                    read_bytes=args.read_bytes,
                    read_iters=args.read_iters,
                )
            )
            read_ms, copy_ms = _run_read_with_copy(
                lib,
                read_stream=read_stream,
                copy_stream=copy_stream,
                read_src=read_src,
                read_dst=read_dst,
                read_bytes=args.read_bytes,
                read_iters=args.read_iters,
                copy_src=copy_src,
                copy_dst=copy_dst,
                copy_bytes=args.copy_bytes,
            )
            with_copy.append(read_ms)
            copy_times.append(copy_ms)

        baseline_p50 = sorted(baseline)[len(baseline) // 2]
        with_copy_p50 = sorted(with_copy)[len(with_copy) // 2]
        result = {
            "device": args.device,
            "read_bytes": args.read_bytes,
            "copy_bytes": args.copy_bytes,
            "read_iters": args.read_iters,
            "repeats": args.repeats,
            "baseline_uva_read_ms": _stats(baseline),
            "uva_read_with_h2d_copy_ms": _stats(with_copy),
            "background_h2d_copy_ms": _stats(copy_times),
            "slowdown_ratio_p50": with_copy_p50 / baseline_p50,
            "slowdown_ms_p50": with_copy_p50 - baseline_p50,
        }
    finally:
        lib.uvm_stage_probe_free_device(read_dst)
        lib.uvm_stage_probe_free_device(copy_dst)
        lib.uvm_stage_probe_free_pinned_host(read_src)
        lib.uvm_stage_probe_free_pinned_host(copy_src)
        lib.uvm_stage_probe_destroy_stream(read_stream)
        lib.uvm_stage_probe_destroy_stream(copy_stream)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
