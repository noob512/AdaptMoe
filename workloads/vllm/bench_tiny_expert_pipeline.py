#!/usr/bin/env python3
"""MoE expert prefetch pipeline experiment.

By default this simulates 90 experts with 6KiB logical weights each:
  page0: upper 4KiB
  page1: lower 2KiB payload + 2KiB padding

80 experts are treated as already GPU-resident and 10 experts as CPU-resident
missing experts. The experiment compares:
  1. synchronous prefetch of full logical expert ranges, then compute
  2. pipeline: prefetch upper range and lower range on a prefetch stream while
     compute windows run on a compute stream

This is a timing experiment for scheduling shape. Correctness is represented by
CUDA synchronization success; residency queries are intentionally not used for
sub-page ranges because they are page-granular and can reject non-page-aligned
small ranges.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from vllm.device_allocator.uvm import (  # noqa: E402
    batch_prefetch_ranges_to_cpu,
    batch_prefetch_ranges_to_device,
    cleanup_uvm_local_memory_pool,
    uvm_local_memory_pool,
)


PAGE_BYTES = 4096


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(ordered, 0.50),
        "p90_ms": percentile(ordered, 0.90),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


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


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def allocate_experts(count: int, device: torch.device, *, logical_bytes: int, padded_bytes: int) -> torch.Tensor:
    with uvm_local_memory_pool():
        storage = torch.empty((count * padded_bytes,), dtype=torch.uint8, device=device)

    for expert_id in range(count):
        base = expert_id * padded_bytes
        storage[base:base + logical_bytes].fill_((0x31 + expert_id) & 0xFF)
        storage[base + logical_bytes:base + padded_bytes].zero_()
    torch.cuda.synchronize(device)
    return storage


def expert_ranges(
    storage: torch.Tensor,
    expert_ids: list[int],
    *,
    padded_bytes: int,
    offset: int,
    size: int,
) -> list[tuple[int, int]]:
    base = int(storage.data_ptr())
    return [(base + expert_id * padded_bytes + offset, size) for expert_id in expert_ids]


def prefetch_cpu(ranges: list[tuple[int, int]], device: torch.device) -> None:
    batch_prefetch_ranges_to_cpu(ranges)
    torch.cuda.synchronize(device)


def prefetch_gpu(ranges: list[tuple[int, int]], device: torch.device) -> int:
    result = batch_prefetch_ranges_to_device(ranges, device.index or 0)
    return int(result)


def time_cuda_sleep(cycles: int, device: torch.device) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def calibrate_sleep_cycles(target_ms: float, device: torch.device) -> tuple[int, float]:
    cycles = 1_000_000
    measured = time_cuda_sleep(cycles, device)
    while measured < target_ms * 0.5 and cycles < 1_000_000_000:
        cycles = int(cycles * max(2.0, target_ms / max(measured, 0.001)))
        measured = time_cuda_sleep(cycles, device)

    for _ in range(4):
        scale = target_ms / max(measured, 0.001)
        cycles = max(1, int(cycles * scale))
        measured = time_cuda_sleep(cycles, device)

    return cycles, measured


def run_compute(cycles: int) -> None:
    torch.cuda._sleep(cycles)


def run_sync_case(
    *,
    storage: torch.Tensor,
    missing_ids: list[int],
    device: torch.device,
    compute_cycles: int,
    compute_windows: int,
    prefetch_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    logical_bytes: int,
    padded_bytes: int,
) -> dict[str, Any]:
    full_ranges = expert_ranges(storage, missing_ids, padded_bytes=padded_bytes, offset=0, size=logical_bytes)

    start_ns = time.perf_counter_ns()
    with torch.cuda.stream(prefetch_stream):
        prefetch_success = prefetch_gpu(full_ranges, device)
    submit_ns = time.perf_counter_ns()
    prefetch_stream.synchronize()
    prefetch_done_ns = time.perf_counter_ns()
    with torch.cuda.stream(compute_stream):
        for _ in range(compute_windows):
            run_compute(compute_cycles)
    compute_stream.synchronize()
    done_ns = time.perf_counter_ns()

    return {
        "prefetch_success": prefetch_success,
        "prefetch_submit_ms": (submit_ns - start_ns) / 1e6,
        "prefetch_complete_ms": (prefetch_done_ns - start_ns) / 1e6,
        "total_ms": (done_ns - start_ns) / 1e6,
    }


def run_pipeline_case(
    *,
    storage: torch.Tensor,
    missing_ids: list[int],
    device: torch.device,
    compute_cycles: int,
    compute_windows: int,
    prefetch_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    upper_bytes: int,
    lower_bytes: int,
    padded_bytes: int,
) -> dict[str, Any]:
    upper_ranges = expert_ranges(storage, missing_ids, padded_bytes=padded_bytes, offset=0, size=upper_bytes)
    lower_ranges = expert_ranges(storage, missing_ids, padded_bytes=padded_bytes, offset=upper_bytes, size=lower_bytes)

    start_ns = time.perf_counter_ns()
    with torch.cuda.stream(prefetch_stream):
        upper_success = prefetch_gpu(upper_ranges, device)
    upper_submit_ns = time.perf_counter_ns()
    with torch.cuda.stream(prefetch_stream):
        lower_success = prefetch_gpu(lower_ranges, device)
    lower_submit_ns = time.perf_counter_ns()
    with torch.cuda.stream(compute_stream):
        for _ in range(compute_windows):
            run_compute(compute_cycles)
    compute_stream.synchronize()
    prefetch_stream.synchronize()
    done_ns = time.perf_counter_ns()

    return {
        "upper_success": upper_success,
        "lower_success": lower_success,
        "upper_submit_ms": (upper_submit_ns - start_ns) / 1e6,
        "lower_submit_ms": (lower_submit_ns - start_ns) / 1e6,
        "total_ms": (done_ns - start_ns) / 1e6,
    }


def run_pipeline_threaded_case(
    *,
    storage: torch.Tensor,
    missing_ids: list[int],
    device: torch.device,
    compute_cycles: int,
    compute_windows: int,
    prefetch_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    upper_bytes: int,
    lower_bytes: int,
    padded_bytes: int,
) -> dict[str, Any]:
    upper_ranges = expert_ranges(storage, missing_ids, padded_bytes=padded_bytes, offset=0, size=upper_bytes)
    lower_ranges = expert_ranges(storage, missing_ids, padded_bytes=padded_bytes, offset=upper_bytes, size=lower_bytes)
    thread_result: dict[str, Any] = {
        "upper_success": 0,
        "lower_success": 0,
        "upper_submit_ms": 0.0,
        "lower_submit_ms": 0.0,
        "prefetch_done_ms": 0.0,
        "error": "",
    }

    start_ns = time.perf_counter_ns()

    def prefetch_worker() -> None:
        try:
            torch.cuda.set_device(device)
            with torch.cuda.stream(prefetch_stream):
                thread_result["upper_success"] = prefetch_gpu(upper_ranges, device)
            thread_result["upper_submit_ms"] = (time.perf_counter_ns() - start_ns) / 1e6
            with torch.cuda.stream(prefetch_stream):
                thread_result["lower_success"] = prefetch_gpu(lower_ranges, device)
            thread_result["lower_submit_ms"] = (time.perf_counter_ns() - start_ns) / 1e6
            prefetch_stream.synchronize()
            thread_result["prefetch_done_ms"] = (time.perf_counter_ns() - start_ns) / 1e6
        except Exception as exc:  # pragma: no cover - benchmark diagnostics
            thread_result["error"] = repr(exc)

    worker = threading.Thread(target=prefetch_worker)
    worker.start()

    with torch.cuda.stream(compute_stream):
        for _ in range(compute_windows):
            run_compute(compute_cycles)
    compute_stream.synchronize()
    compute_done_ns = time.perf_counter_ns()

    worker.join()
    if thread_result["error"]:
        raise RuntimeError(str(thread_result["error"]))

    done_ns = time.perf_counter_ns()
    return {
        **thread_result,
        "compute_done_ms": (compute_done_ns - start_ns) / 1e6,
        "total_ms": (done_ns - start_ns) / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=90)
    parser.add_argument("--resident-count", type=int, default=80)
    parser.add_argument("--missing", default="80,81,82,83,84,85,86,87,88,89")
    parser.add_argument("--target-compute-ms", type=float, default=3.0)
    parser.add_argument("--compute-windows", type=int, default=2)
    parser.add_argument("--logical-expert-size", default="6KiB")
    parser.add_argument("--upper-size", default="4KiB")
    parser.add_argument("--padded-expert-size", default="")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    missing_ids = [int(item) for item in args.missing.split(",") if item.strip()]
    if len(missing_ids) == 0:
        raise ValueError("--missing must contain at least one expert id")

    logical_bytes = parse_size(args.logical_expert_size)
    upper_bytes = parse_size(args.upper_size)
    if upper_bytes <= 0 or upper_bytes >= logical_bytes:
        raise ValueError("--upper-size must be in (0, logical expert size)")
    lower_bytes = logical_bytes - upper_bytes
    padded_bytes = parse_size(args.padded_expert_size) if args.padded_expert_size else align_up(logical_bytes, PAGE_BYTES)
    if padded_bytes < logical_bytes:
        raise ValueError("--padded-expert-size must be >= logical expert size")

    storage = allocate_experts(args.experts, device, logical_bytes=logical_bytes, padded_bytes=padded_bytes)
    all_full_ranges = expert_ranges(
        storage, list(range(args.experts)), padded_bytes=padded_bytes, offset=0, size=logical_bytes
    )
    missing_full_ranges = expert_ranges(storage, missing_ids, padded_bytes=padded_bytes, offset=0, size=logical_bytes)

    resident_ids = [expert_id for expert_id in range(args.experts) if expert_id not in set(missing_ids)]
    resident_full_ranges = expert_ranges(
        storage, resident_ids[:args.resident_count], padded_bytes=padded_bytes, offset=0, size=logical_bytes
    )

    # Establish the target state before each repeat: selected missing experts on
    # CPU, resident experts on GPU.
    compute_cycles, calibrated_ms = calibrate_sleep_cycles(args.target_compute_ms, device)
    prefetch_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    sync_samples: list[dict[str, Any]] = []
    pipeline_samples: list[dict[str, Any]] = []
    pipeline_threaded_samples: list[dict[str, Any]] = []

    for _ in range(args.repeats):
        prefetch_cpu(all_full_ranges, device)
        prefetch_gpu(resident_full_ranges, device)
        torch.cuda.synchronize(device)
        prefetch_cpu(missing_full_ranges, device)
        sync_samples.append(
            run_sync_case(
                storage=storage,
                missing_ids=missing_ids,
                device=device,
                compute_cycles=compute_cycles,
                compute_windows=args.compute_windows,
                prefetch_stream=prefetch_stream,
                compute_stream=compute_stream,
                logical_bytes=logical_bytes,
                padded_bytes=padded_bytes,
            )
        )

        prefetch_cpu(all_full_ranges, device)
        prefetch_gpu(resident_full_ranges, device)
        torch.cuda.synchronize(device)
        prefetch_cpu(missing_full_ranges, device)
        pipeline_samples.append(
            run_pipeline_case(
                storage=storage,
                missing_ids=missing_ids,
                device=device,
                compute_cycles=compute_cycles,
                compute_windows=args.compute_windows,
                prefetch_stream=prefetch_stream,
                compute_stream=compute_stream,
                upper_bytes=upper_bytes,
                lower_bytes=lower_bytes,
                padded_bytes=padded_bytes,
            )
        )

        prefetch_cpu(all_full_ranges, device)
        prefetch_gpu(resident_full_ranges, device)
        torch.cuda.synchronize(device)
        prefetch_cpu(missing_full_ranges, device)
        pipeline_threaded_samples.append(
            run_pipeline_threaded_case(
                storage=storage,
                missing_ids=missing_ids,
                device=device,
                compute_cycles=compute_cycles,
                compute_windows=args.compute_windows,
                prefetch_stream=prefetch_stream,
                compute_stream=compute_stream,
                upper_bytes=upper_bytes,
                lower_bytes=lower_bytes,
                padded_bytes=padded_bytes,
            )
        )

    result = {
        "config": {
            "experts": args.experts,
            "logical_expert_bytes": logical_bytes,
            "upper_bytes": upper_bytes,
            "lower_bytes": lower_bytes,
            "padded_expert_bytes": padded_bytes,
            "missing_ids": missing_ids,
            "resident_count": args.resident_count,
            "target_compute_ms": args.target_compute_ms,
            "compute_cycles": compute_cycles,
            "compute_windows": args.compute_windows,
            "calibrated_compute_ms": calibrated_ms,
            "repeats": args.repeats,
        },
        "summaries": {
            "sync_total_ms": summary([float(item["total_ms"]) for item in sync_samples]),
            "sync_prefetch_complete_ms": summary([float(item["prefetch_complete_ms"]) for item in sync_samples]),
            "pipeline_total_ms": summary([float(item["total_ms"]) for item in pipeline_samples]),
            "pipeline_upper_submit_ms": summary([float(item["upper_submit_ms"]) for item in pipeline_samples]),
            "pipeline_lower_submit_ms": summary([float(item["lower_submit_ms"]) for item in pipeline_samples]),
            "pipeline_threaded_total_ms": summary([float(item["total_ms"]) for item in pipeline_threaded_samples]),
            "pipeline_threaded_compute_done_ms": summary(
                [float(item["compute_done_ms"]) for item in pipeline_threaded_samples]
            ),
            "pipeline_threaded_prefetch_done_ms": summary(
                [float(item["prefetch_done_ms"]) for item in pipeline_threaded_samples]
            ),
        },
        "samples": {
            "sync": sync_samples,
            "pipeline": pipeline_samples,
            "pipeline_threaded": pipeline_threaded_samples,
        },
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    cleanup_uvm_local_memory_pool()


if __name__ == "__main__":
    main()
