#!/usr/bin/env python3
"""6MiB expert UVM batch pipeline experiment.

This benchmark uses the custom UVM expert fast paths, not native CUDA prefetch.
It simulates 90 experts with 6MiB weights. 80 experts are made GPU-resident and
10 selected experts are made CPU-resident before each measured case.

Cases:
  full_regions_sync:
    one region-id expert_fast_regions ioctl for 10 x 6MiB, sync wait, then two
    compute windows.

  full_regions_async:
    one region-id expert_fast_regions ioctl with async=true, then two compute
    windows, then a GPU page touch to force any remaining wait.

  split_batch_inline:
    base/length expert_fast_batch upper 4MiB async, compute, lower 2MiB async,
    compute, then GPU page touch.

  split_batch_threaded:
    a Python worker issues upper 4MiB async then lower 2MiB async while the main
    thread immediately runs two compute windows, then GPU page touch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from bench_uvm_expert_batch_prefetch import (  # noqa: E402
    ExpertRange,
    allocate_expert_ranges,
    cleanup_expert_ranges,
    force_cpu_resident,
    parse_size,
    selected_ranges,
)
from vllm.device_allocator.uvm import (  # noqa: E402
    batch_prefetch_ranges_to_device,
    cleanup_uvm_local_memory_pool,
    driver_expert_read_dup_fast,
    driver_expert_read_dup_fast_regions,
)


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
        cycles = max(1, int(cycles * target_ms / max(measured, 0.001)))
        measured = time_cuda_sleep(cycles, device)
    return cycles, measured


def run_compute(cycles: int, windows: int) -> None:
    for _ in range(windows):
        torch.cuda._sleep(cycles)


def touch_missing_pages(ranges: list[ExpertRange], page_bytes: int = 4096) -> int:
    acc = None
    for item in ranges:
        page_bytes_view = item.tensor[:item.size:page_bytes]
        value = page_bytes_view.sum(dtype=torch.int64)
        acc = value if acc is None else acc + value
    if acc is None:
        return 0
    return int(acc.item())


def reset_residency(all_ranges: list[ExpertRange], resident_ranges: list[ExpertRange], missing_ranges: list[ExpertRange], device: torch.device) -> None:
    force_cpu_resident(all_ranges, device)
    batch_prefetch_ranges_to_device([item.range_tuple for item in resident_ranges], device.index or 0)
    torch.cuda.synchronize(device)
    force_cpu_resident(missing_ranges, device)


def upper_ranges(ranges: list[ExpertRange], upper_bytes: int) -> list[tuple[int, int]]:
    return [(item.ptr, upper_bytes) for item in ranges]


def lower_ranges(ranges: list[ExpertRange], upper_bytes: int, lower_bytes: int) -> list[tuple[int, int]]:
    return [(item.ptr + upper_bytes, lower_bytes) for item in ranges]


def measure_case(
    name: str,
    *,
    all_ranges: list[ExpertRange],
    resident_ranges: list[ExpertRange],
    missing_ranges: list[ExpertRange],
    device: torch.device,
    compute_cycles: int,
    compute_windows: int,
    upper_bytes: int,
    lower_bytes: int,
) -> dict[str, Any]:
    reset_residency(all_ranges, resident_ranges, missing_ranges, device)

    start_ns = time.perf_counter_ns()
    submit_result: Any = None
    worker_result: dict[str, Any] = {}

    if name == "full_regions_sync":
        submit_result = driver_expert_read_dup_fast_regions(
            [item.region_id for item in missing_ranges],
            device=device.index or 0,
            async_=False,
        )
        submit_ns = time.perf_counter_ns()
        run_compute(compute_cycles, compute_windows)
        torch.cuda.synchronize(device)
    elif name == "full_regions_async":
        submit_result = driver_expert_read_dup_fast_regions(
            [item.region_id for item in missing_ranges],
            device=device.index or 0,
            async_=True,
        )
        submit_ns = time.perf_counter_ns()
        run_compute(compute_cycles, compute_windows)
        torch.cuda.synchronize(device)
    elif name == "split_batch_inline":
        upper_result = driver_expert_read_dup_fast(
            upper_ranges(missing_ranges, upper_bytes),
            device=device.index or 0,
            async_=True,
        )
        upper_ns = time.perf_counter_ns()
        run_compute(compute_cycles, 1)
        lower_result = driver_expert_read_dup_fast(
            lower_ranges(missing_ranges, upper_bytes, lower_bytes),
            device=device.index or 0,
            async_=True,
        )
        submit_ns = time.perf_counter_ns()
        run_compute(compute_cycles, max(0, compute_windows - 1))
        torch.cuda.synchronize(device)
        submit_result = {"upper": upper_result, "lower": lower_result, "upper_submit_ms": (upper_ns - start_ns) / 1e6}
    elif name == "split_batch_threaded":
        def worker() -> None:
            try:
                torch.cuda.set_device(device)
                upper_result = driver_expert_read_dup_fast(
                    upper_ranges(missing_ranges, upper_bytes),
                    device=device.index or 0,
                    async_=True,
                )
                upper_ns = time.perf_counter_ns()
                lower_result = driver_expert_read_dup_fast(
                    lower_ranges(missing_ranges, upper_bytes, lower_bytes),
                    device=device.index or 0,
                    async_=True,
                )
                lower_ns = time.perf_counter_ns()
                worker_result.update({
                    "upper": upper_result,
                    "lower": lower_result,
                    "upper_submit_ms": (upper_ns - start_ns) / 1e6,
                    "lower_submit_ms": (lower_ns - start_ns) / 1e6,
                })
            except Exception as exc:  # pragma: no cover - benchmark diagnostics
                worker_result["error"] = repr(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        run_compute(compute_cycles, compute_windows)
        torch.cuda.synchronize(device)
        thread.join()
        if worker_result.get("error"):
            raise RuntimeError(str(worker_result["error"]))
        submit_ns = time.perf_counter_ns()
        submit_result = worker_result
    else:
        raise ValueError(f"unknown case {name}")

    compute_done_ns = time.perf_counter_ns()
    checksum = touch_missing_pages(missing_ranges)
    touch_done_ns = time.perf_counter_ns()
    return {
        "case": name,
        "submit_result": submit_result,
        "submit_ms": (submit_ns - start_ns) / 1e6,
        "compute_done_ms": (compute_done_ns - start_ns) / 1e6,
        "touch_done_ms": (touch_done_ns - start_ns) / 1e6,
        "total_ms": (touch_done_ns - start_ns) / 1e6,
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=90)
    parser.add_argument("--missing", default="80,81,82,83,84,85,86,87,88,89")
    parser.add_argument("--expert-size", default="6MiB")
    parser.add_argument("--upper-size", default="4MiB")
    parser.add_argument("--target-compute-ms", type=float, default=3.0)
    parser.add_argument("--compute-windows", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "full_regions_sync",
            "full_regions_async",
            "split_batch_inline",
            "split_batch_threaded",
        ],
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    expert_bytes = parse_size(args.expert_size)
    upper_bytes = parse_size(args.upper_size)
    lower_bytes = expert_bytes - upper_bytes
    missing_ids = [int(item) for item in args.missing.split(",") if item.strip()]
    resident_ids = [expert_id for expert_id in range(args.experts) if expert_id not in set(missing_ids)]

    bench_args = SimpleNamespace(
        expert_bytes=expert_bytes,
        roles_per_expert=1,
        layer_id=0,
        region_id_salt=0x6D,
    )
    all_ranges = allocate_expert_ranges(
        args=bench_args,
        repeat=0,
        selected_ids=list(range(args.experts)),
        device=device,
        run_seed=0xBEEF,
    )
    missing_ranges = selected_ranges(all_ranges, missing_ids)
    resident_ranges = selected_ranges(all_ranges, resident_ids)

    compute_cycles, calibrated_ms = calibrate_sleep_cycles(args.target_compute_ms, device)

    # Warm up the page-touch kernel path outside measured samples. The first
    # PyTorch reduction over strided uint8 slices can otherwise dominate the
    # first measured case and hide UVM behavior.
    force_cpu_resident(all_ranges, device)
    batch_prefetch_ranges_to_device([item.range_tuple for item in missing_ranges], device.index or 0)
    torch.cuda.synchronize(device)
    _ = touch_missing_pages(missing_ranges)

    samples: list[dict[str, Any]] = []

    try:
        for _ in range(args.repeats):
            for case in args.cases:
                samples.append(
                    measure_case(
                        case,
                        all_ranges=all_ranges,
                        resident_ranges=resident_ranges,
                        missing_ranges=missing_ranges,
                        device=device,
                        compute_cycles=compute_cycles,
                        compute_windows=args.compute_windows,
                        upper_bytes=upper_bytes,
                        lower_bytes=lower_bytes,
                    )
                )
    finally:
        cleanup_expert_ranges(all_ranges, device)
        cleanup_uvm_local_memory_pool()

    summaries = {}
    for case in args.cases:
        case_samples = [sample for sample in samples if sample["case"] == case]
        summaries[case] = {
            "submit_ms": summary([float(sample["submit_ms"]) for sample in case_samples]),
            "compute_done_ms": summary([float(sample["compute_done_ms"]) for sample in case_samples]),
            "touch_done_ms": summary([float(sample["touch_done_ms"]) for sample in case_samples]),
            "total_ms": summary([float(sample["total_ms"]) for sample in case_samples]),
        }

    result = {
        "config": {
            "experts": args.experts,
            "expert_bytes": expert_bytes,
            "upper_bytes": upper_bytes,
            "lower_bytes": lower_bytes,
            "missing_ids": missing_ids,
            "resident_count": len(resident_ranges),
            "target_compute_ms": args.target_compute_ms,
            "calibrated_compute_ms": calibrated_ms,
            "compute_windows": args.compute_windows,
            "compute_cycles": compute_cycles,
            "repeats": args.repeats,
            "cases": args.cases,
        },
        "summaries": summaries,
        "samples": samples,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
