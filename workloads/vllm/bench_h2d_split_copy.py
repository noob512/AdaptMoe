#!/usr/bin/env python3
"""Measure ordinary H2D copy when a fixed total size is split into N submits."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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


def _parse_size(raw: str) -> int:
    token = raw.strip().lower()
    if token.endswith("mib"):
        return int(float(token[:-3]) * 1024 * 1024)
    if token.endswith("mb"):
        return int(float(token[:-2]) * 1000 * 1000)
    if token.endswith("kib"):
        return int(float(token[:-3]) * 1024)
    if token.endswith("kb"):
        return int(float(token[:-2]) * 1000)
    return int(token)


def _event(lib: ctypes.CDLL) -> ctypes.c_void_p:
    event = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_event(ctypes.byref(event), 1), "create_event")
    return event


def _elapsed_ms(lib: ctypes.CDLL, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
    value = ctypes.c_float()
    _check(lib.uvm_stage_probe_elapsed_ms(start, end, ctypes.byref(value)), "elapsed_ms")
    return float(value.value)


def _alloc_pinned(lib: ctypes.CDLL, bytes_: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_alloc_pinned_host(ctypes.byref(ptr), bytes_), "alloc_pinned_host")
    ctypes.memset(int(ptr.value or 0), 0x3C, bytes_)
    return ptr


def _alloc_device(lib: ctypes.CDLL, bytes_: int, device: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_alloc_device(ctypes.byref(ptr), bytes_, device), "alloc_device")
    return ptr


def _run_split_copy_once(
    lib: ctypes.CDLL,
    *,
    stream: ctypes.c_void_p,
    src: ctypes.c_void_p,
    dst: ctypes.c_void_p,
    total_bytes: int,
    split_count: int,
) -> tuple[float, float]:
    if total_bytes % split_count != 0:
        raise ValueError(f"{total_bytes=} is not divisible by {split_count=}")
    chunk = total_bytes // split_count
    start = _event(lib)
    end = _event(lib)
    src_base = int(src.value or 0)
    dst_base = int(dst.value or 0)
    try:
        submit_start = time.perf_counter()
        _check(lib.uvm_stage_probe_record_event(start, stream), "record_h2d_start")
        for index in range(split_count):
            offset = index * chunk
            _check(
                lib.uvm_stage_probe_copy_host_to_device_async(
                    ctypes.c_void_p(dst_base + offset),
                    ctypes.c_void_p(src_base + offset),
                    chunk,
                    stream,
                ),
                "copy_host_to_device_async",
            )
        _check(lib.uvm_stage_probe_record_event(end, stream), "record_h2d_end")
        submit_ms = (time.perf_counter() - submit_start) * 1000.0
        _check(lib.uvm_stage_probe_sync_event(end), "sync_h2d_end")
        return submit_ms, _elapsed_ms(lib, start, end)
    finally:
        lib.uvm_stage_probe_destroy_event(start)
        lib.uvm_stage_probe_destroy_event(end)


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_bytes = _parse_size(args.total_size)
    split_counts = [int(x) for x in args.split_counts.split(",") if x.strip()]
    lib = _load_probe()
    _check(lib.uvm_stage_probe_set_device(args.device), "set_device")
    stream = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_stream(ctypes.byref(stream)), "create_stream")
    src = _alloc_pinned(lib, total_bytes)
    dst = _alloc_device(lib, total_bytes, args.device)
    rows: list[dict[str, Any]] = []
    try:
        for split_count in split_counts:
            # Warm once for this split shape.
            _run_split_copy_once(
                lib,
                stream=stream,
                src=src,
                dst=dst,
                total_bytes=total_bytes,
                split_count=split_count,
            )
            submit_ms: list[float] = []
            gpu_ms: list[float] = []
            for _ in range(args.repeats):
                submit, elapsed = _run_split_copy_once(
                    lib,
                    stream=stream,
                    src=src,
                    dst=dst,
                    total_bytes=total_bytes,
                    split_count=split_count,
                )
                submit_ms.append(submit)
                gpu_ms.append(elapsed)
            rows.append({
                "total_bytes": total_bytes,
                "total_mib": total_bytes / 1024 / 1024,
                "split_count": split_count,
                "chunk_bytes": total_bytes // split_count,
                "chunk_mib": total_bytes / split_count / 1024 / 1024,
                "repeats": args.repeats,
                "h2d_split_submit_ms": _stats(submit_ms),
                "h2d_split_gpu_ms": _stats(gpu_ms),
            })
    finally:
        lib.uvm_stage_probe_free_device(dst)
        lib.uvm_stage_probe_free_pinned_host(src)
        lib.uvm_stage_probe_destroy_stream(stream)

    return {
        "config": {
            "device": args.device,
            "total_size": args.total_size,
            "split_counts": args.split_counts,
            "repeats": args.repeats,
        },
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [
        {
            "total_mib": row["total_mib"],
            "split_count": row["split_count"],
            "chunk_mib": row["chunk_mib"],
            "submit_p50_ms": row["h2d_split_submit_ms"]["p50_ms"],
            "submit_p95_ms": row["h2d_split_submit_ms"]["p95_ms"],
            "gpu_p50_ms": row["h2d_split_gpu_ms"]["p50_ms"],
            "gpu_p95_ms": row["h2d_split_gpu_ms"]["p95_ms"],
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--total-size", default="144MiB")
    parser.add_argument("--split-counts", default="1,2,4,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.out_csv, result["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
