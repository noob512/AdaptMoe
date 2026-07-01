#!/usr/bin/env python3
"""Compare native UVM prefetch with expert-only read-duplicate promotion."""

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

from bench_hot_cold_expert_checkpoint import _alloc_gpu_tensor  # noqa: E402
from bench_uvm_expert_fault_tag_counter import (  # noqa: E402
    delta_stats,
    find_fault_stats,
    parse_size,
    read_fault_stats,
    register_expert,
    unregister_region,
)
from vllm.device_allocator.uvm import (  # noqa: E402
    UVM_REGION_ROLE_MOE_EXPERT_W2,
    batch_prefetch_ranges_to_cpu,
    batch_prefetch_ranges_to_device,
    cleanup_uvm_local_memory_pool,
    driver_batch_query_ranges_residency_bytes,
    driver_expert_read_dup_fast,
    driver_gpu_only_read_dup_soft,
    ensure_uvm_runtime_library,
)

INTERESTING_STATS = [
    "phase_n_expert_migrate_fastpath_ranges",
    "phase_n_expert_migrate_fastpath_pages",
    "phase_n_expert_migrate_fastpath_fallbacks",
]


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


def _parse_sizes(raw: str) -> list[int]:
    return [parse_size(item) for item in raw.split(",") if item.strip()]


def _alloc_cpu_resident_uvm(
    bytes_: int,
    device: torch.device,
    *,
    register: bool,
    region_id: int,
) -> tuple[torch.Tensor, list[tuple[int, int]], int | None, dict[str, int] | None]:
    tensor, backing = _alloc_gpu_tensor((bytes_,), torch.uint8, device, "uvm")
    if backing != "uvm":
        raise RuntimeError(f"expected UVM backing, got {backing}")
    tensor.fill_(0x6D)
    torch.cuda.synchronize(device)

    ptr = int(tensor.data_ptr())
    ranges = [(ptr, bytes_)]
    registered_region_id: int | None = None
    region_query: dict[str, int] | None = None
    if register:
        status, region_query = register_expert(
            ptr,
            bytes_,
            UVM_REGION_ROLE_MOE_EXPERT_W2,
            region_id,
        )
        if status < 0:
            raise RuntimeError(f"register_expert failed: {status}")
        registered_region_id = region_id

    cpu_result = int(batch_prefetch_ranges_to_cpu(ranges))
    torch.cuda.synchronize(device)
    if cpu_result <= 0:
        raise RuntimeError(f"CPU prefetch failed: {cpu_result}")
    return tensor, ranges, registered_region_id, region_query


def _time_call(fn) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start) * 1000.0, result


def _time_call_and_sync(fn, device: torch.device) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0, result


def _run_once(
    *,
    case: str,
    bytes_: int,
    device: torch.device,
    device_index: int,
    region_seed: int,
    fault_stats_path: Path,
) -> dict[str, Any]:
    register = case in ("native_prefetch_registered", "readdup_soft_registered", "expert_fast")
    tensor, ranges, region_id, region_query = _alloc_cpu_resident_uvm(
        bytes_,
        device,
        register=register,
        region_id=region_seed,
    )
    try:
        before = driver_batch_query_ranges_residency_bytes(ranges, device_index)
        stats_before = read_fault_stats(fault_stats_path)
        if case == "native_prefetch" or case == "native_prefetch_registered":
            elapsed_ms, result = _time_call_and_sync(
                lambda: int(
                    batch_prefetch_ranges_to_device(
                        ranges,
                        device=device_index,
                    )
                ),
                device,
            )
        elif case == "readdup_soft_registered":
            elapsed_ms, result = _time_call_and_sync(
                lambda: tuple(
                    int(x)
                    for x in driver_gpu_only_read_dup_soft(ranges, device=device_index)
                ),
                device,
            )
        elif case == "expert_fast":
            elapsed_ms, result = _time_call_and_sync(
                lambda: tuple(
                    int(x)
                    for x in driver_expert_read_dup_fast(ranges, device=device_index)
                ),
                device,
            )
        elif case == "expert_fast_unregistered":
            elapsed_ms, result = _time_call_and_sync(
                lambda: tuple(
                    int(x)
                    for x in driver_expert_read_dup_fast(ranges, device=device_index)
                ),
                device,
            )
        else:
            raise ValueError(f"unknown case {case}")

        stats_after = read_fault_stats(fault_stats_path)
        stats_delta = delta_stats(stats_before, stats_after)
        after = driver_batch_query_ranges_residency_bytes(ranges, device_index)
        return {
            "case": case,
            "bytes": bytes_,
            "elapsed_ms": elapsed_ms,
            "result": list(result) if isinstance(result, tuple) else result,
            "before_residency": before,
            "after_residency": after,
            "region_query": region_query,
            "delta_interesting": {key: stats_delta.get(key, 0) for key in INTERESTING_STATS},
        }
    finally:
        if region_id is not None:
            unregister_region(region_id)
        del tensor
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.synchronize(device)
    ensure_uvm_runtime_library()
    fault_stats_path = find_fault_stats()

    rows: list[dict[str, Any]] = []
    for bytes_ in _parse_sizes(args.sizes):
        case_rows: dict[str, list[dict[str, Any]]] = {
            "native_prefetch": [],
            "native_prefetch_registered": [],
            "readdup_soft_registered": [],
            "expert_fast": [],
        }
        if args.include_negative_control:
            case_rows["expert_fast_unregistered"] = []

        for repeat in range(args.repeats):
            for case in case_rows:
                row = _run_once(
                    case=case,
                    bytes_=bytes_,
                    device=device,
                    device_index=args.device,
                    region_seed=(time.time_ns() + repeat * 100 + len(case)) & 0x7FFFFFFFFFFFFFFF,
                    fault_stats_path=fault_stats_path,
                )
                case_rows[case].append(row)

        for case, samples in case_rows.items():
            values = [float(sample["elapsed_ms"]) for sample in samples]
            rows.append(
                {
                    "case": case,
                    "bytes": bytes_,
                    "mib": bytes_ / 1024 / 1024,
                    "repeats": args.repeats,
                    "elapsed_ms": _stats(values),
                    "first_result": samples[0]["result"],
                    "first_before_residency": samples[0]["before_residency"],
                    "last_after_residency": samples[-1]["after_residency"],
                    "first_region_query": samples[0]["region_query"],
                    "sum_delta": {
                        key: sum(sample["delta_interesting"].get(key, 0) for sample in samples)
                        for key in INTERESTING_STATS
                    },
                    "samples": samples if args.keep_samples else [],
                }
            )

    return {
        "config": {
            "device": args.device,
            "sizes": args.sizes,
            "repeats": args.repeats,
            "include_negative_control": args.include_negative_control,
            "fault_stats_path": str(fault_stats_path),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", default="4.5MiB,18MiB,72MiB,144MiB")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--include-negative-control", action="store_true")
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("log/uvm_expert_readdup_fast/result.json"),
    )
    args = parser.parse_args()

    try:
        result = run(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        cleanup_uvm_local_memory_pool()


if __name__ == "__main__":
    main()
