#!/usr/bin/env python3
"""Compare contiguous cudaMemPrefetchAsync against explicit H2D copy.

The benchmark measures two shapes for the same byte count:

* one contiguous range
* several consecutive chunks inside the same contiguous allocation

The second shape matches the previous "serial prefetch" call pattern while
removing non-contiguous address effects.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from vllm.device_allocator.uvm import (  # noqa: E402
    batch_prefetch_ranges_to_cpu,
    batch_prefetch_ranges_to_device,
    cleanup_uvm_local_memory_pool,
    ensure_uvm_runtime_library,
    uvm_local_memory_pool,
)


def parse_size(raw: str) -> int:
    token = raw.strip().lower()
    scale = 1
    if token.endswith("gib"):
        scale = 1024**3
        token = token[:-3]
    elif token.endswith("gb"):
        scale = 1000**3
        token = token[:-2]
    elif token.endswith("mib"):
        scale = 1024**2
        token = token[:-3]
    elif token.endswith("mb"):
        scale = 1000**2
        token = token[:-2]
    elif token.endswith("kib"):
        scale = 1024
        token = token[:-3]
    elif token.endswith("kb"):
        scale = 1000
        token = token[:-2]
    return int(float(token) * scale)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(ordered, 0.50),
        "p90_ms": percentile(ordered, 0.90),
        "p95_ms": percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def gib_s(byte_count: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return byte_count / (ms / 1000.0) / (1024**3)


def make_ranges(base: int, total_bytes: int, chunks: int) -> list[tuple[int, int]]:
    if chunks <= 1:
        return [(base, total_bytes)]
    if total_bytes % chunks != 0:
        raise ValueError(f"total bytes {total_bytes} must be divisible by chunks {chunks}")
    chunk_bytes = total_bytes // chunks
    return [(base + index * chunk_bytes, chunk_bytes) for index in range(chunks)]


def time_submit_and_complete(
    submit: Callable[[], Any],
    *,
    device: torch.device,
) -> tuple[float, float, Any]:
    start_ns = time.perf_counter_ns()
    result = submit()
    submit_ns = time.perf_counter_ns()
    torch.cuda.synchronize(device)
    done_ns = time.perf_counter_ns()
    return (submit_ns - start_ns) / 1e6, (done_ns - start_ns) / 1e6, result


def time_with_events(
    submit: Callable[[], Any],
    *,
    device: torch.device,
) -> tuple[float, float, float, Any]:
    stream = torch.cuda.current_stream(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_ns = time.perf_counter_ns()
    start_event.record(stream)
    result = submit()
    submit_ns = time.perf_counter_ns()
    end_event.record(stream)
    end_event.synchronize()
    done_ns = time.perf_counter_ns()
    return (
        (submit_ns - start_ns) / 1e6,
        (done_ns - start_ns) / 1e6,
        float(start_event.elapsed_time(end_event)),
        result,
    )


def run_prefetch_case(
    *,
    byte_count: int,
    chunks: int,
    repeats: int,
    device: torch.device,
    device_index: int,
) -> dict[str, Any]:
    with uvm_local_memory_pool():
        tensor = torch.empty((byte_count,), dtype=torch.uint8, device=device)
    tensor.fill_(0x5A)
    torch.cuda.synchronize(device)

    ranges = make_ranges(int(tensor.data_ptr()), byte_count, chunks)
    submit_ms: list[float] = []
    complete_ms: list[float] = []
    event_ms: list[float] = []
    results: list[int] = []

    for _ in range(repeats):
        batch_prefetch_ranges_to_cpu([(int(tensor.data_ptr()), byte_count)])
        torch.cuda.synchronize(device)

        submit, complete, event_elapsed, result = time_with_events(
            lambda: batch_prefetch_ranges_to_device(ranges, device=device_index),
            device=device,
        )
        submit_ms.append(submit)
        complete_ms.append(complete)
        event_ms.append(event_elapsed)
        results.append(int(result))

    del tensor
    return {
        "case": f"cuda_prefetch_{chunks}x",
        "chunks": chunks,
        "range_count": len(ranges),
        "submit_ms": stats(submit_ms),
        "complete_ms": stats(complete_ms),
        "cuda_event_ms": stats(event_ms),
        "effective_gib_s_p50": gib_s(byte_count, stats(complete_ms)["p50_ms"]),
        "cuda_event_gib_s_p50": gib_s(byte_count, stats(event_ms)["p50_ms"]),
        "results": results,
    }


def run_copy_case(
    *,
    byte_count: int,
    chunks: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    src = torch.empty((byte_count,), dtype=torch.uint8, pin_memory=True)
    dst = torch.empty((byte_count,), dtype=torch.uint8, device=device)
    src.fill_(0x3C)
    torch.cuda.synchronize(device)

    chunk_bytes = byte_count // chunks

    def copy_submit() -> int:
        if chunks <= 1:
            dst.copy_(src, non_blocking=True)
            return 1
        for index in range(chunks):
            start = index * chunk_bytes
            end = start + chunk_bytes
            dst[start:end].copy_(src[start:end], non_blocking=True)
        return chunks

    # Warm copy once.
    copy_submit()
    torch.cuda.synchronize(device)

    submit_ms: list[float] = []
    complete_ms: list[float] = []
    event_ms: list[float] = []
    results: list[int] = []

    for _ in range(repeats):
        submit, complete, event_elapsed, result = time_with_events(copy_submit, device=device)
        submit_ms.append(submit)
        complete_ms.append(complete)
        event_ms.append(event_elapsed)
        results.append(int(result))

    del src
    del dst
    return {
        "case": f"cuda_copy_{chunks}x",
        "chunks": chunks,
        "copy_count": chunks,
        "submit_ms": stats(submit_ms),
        "complete_ms": stats(complete_ms),
        "cuda_event_ms": stats(event_ms),
        "effective_gib_s_p50": gib_s(byte_count, stats(complete_ms)["p50_ms"]),
        "cuda_event_gib_s_p50": gib_s(byte_count, stats(event_ms)["p50_ms"]),
        "results": results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    ensure_uvm_runtime_library()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    byte_count = parse_size(args.size)

    rows: list[dict[str, Any]] = []
    for chunks in args.chunks:
        rows.append(
            run_prefetch_case(
                byte_count=byte_count,
                chunks=chunks,
                repeats=args.repeats,
                device=device,
                device_index=args.device,
            )
        )
        rows.append(
            run_copy_case(
                byte_count=byte_count,
                chunks=chunks,
                repeats=args.repeats,
                device=device,
            )
        )
        torch.cuda.empty_cache()

    cleanup_uvm_local_memory_pool()

    return {
        "config": {
            "device": args.device,
            "size": args.size,
            "bytes": byte_count,
            "mib": byte_count / 1024 / 1024,
            "repeats": args.repeats,
            "chunks": args.chunks,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", default="20MiB")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
