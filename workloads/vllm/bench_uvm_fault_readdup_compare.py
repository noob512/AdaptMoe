#!/usr/bin/env python3
"""Compare native and expert-tagged UVM read-duplicate fault service."""

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

from bench_uvm_expert_fault_tag_counter import (  # noqa: E402
    find_fault_stats,
    launch_read,
    make_managed_tensor,
    parse_size,
    prefetch_cpu,
    read_fault_stats,
    register_expert,
    unregister_region,
)
from bench_uvm_stage_boundary_remap import _check, _load_probe  # noqa: E402
from vllm.device_allocator.uvm import (  # noqa: E402
    UVM_REGION_ROLE_MOE_EXPERT_W2,
    batch_advise_ranges_read_mostly,
    cleanup_uvm_local_memory_pool,
    ensure_uvm_runtime_library,
)


INTERESTING_STATS = [
    "replayable_faults",
    "duplicates",
    "faults_by_access_type.read",
    "faults_by_phase_n_pool.pool_2",
    "faults_by_phase_n_role.role_4",
    "phase_n_duplicate_faults_by_pool.pool_2",
    "phase_n_duplicate_faults_by_role.role_4",
    "phase_n_unregistered_faults",
    "phase_n_unregistered_duplicate_faults",
    "phase_n_expert_dispatches",
    "phase_n_expert_faults_serviced",
    "phase_n_expert_forced_readdup_faults",
    "phase_n_expert_forced_readdup_pages",
    "phase_n_expert_fastpath_dispatches",
    "phase_n_expert_fastpath_faults_serviced",
    "phase_n_expert_fastpath_pages",
    "phase_n_expert_fastpath_fallbacks",
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


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


def _parse_sizes(raw: str) -> list[int]:
    return [parse_size(item) for item in raw.split(",") if item.strip()]


def _case_config(case: str) -> tuple[bool, bool]:
    if case == "native_readmostly":
        return False, True
    if case == "expert_forced_readdup":
        return True, False
    if case == "expert_fastpath":
        return True, False
    if case == "native_no_readmostly":
        return False, False
    if case == "expert_forced_plus_readmostly":
        return True, True
    raise ValueError(f"unknown case {case}")


def _run_once(
    *,
    lib,
    fault_stats_path: Path,
    case: str,
    byte_count: int,
    iters: int,
    device: torch.device,
    device_index: int,
    region_seed: int,
) -> dict[str, Any]:
    register, readmostly = _case_config(case)
    tensor = make_managed_tensor(byte_count, device)
    ptr = int(tensor.data_ptr())
    ctypes.memset(ptr, 0x5A, byte_count)
    ranges = [(ptr, byte_count)]
    region_id: int | None = None
    register_status: int | None = None
    readmostly_result: int | None = None

    try:
        if register:
            register_status, _ = register_expert(
                ptr,
                byte_count,
                UVM_REGION_ROLE_MOE_EXPERT_W2,
                region_seed,
            )
            if register_status < 0:
                raise RuntimeError(f"register_expert failed: {register_status}")
            region_id = region_seed

        if readmostly:
            readmostly_result = int(batch_advise_ranges_read_mostly(ranges, device=device_index))

        cpu_result = prefetch_cpu(ptr, byte_count, device)
        before = read_fault_stats(fault_stats_path)
        elapsed_ms = launch_read(lib, ptr, byte_count, iters, device_index)
        after = read_fault_stats(fault_stats_path)
        delta = _delta(before, after)

        return {
            "case": case,
            "bytes": byte_count,
            "elapsed_ms": elapsed_ms,
            "prefetch_cpu_result": cpu_result,
            "register_status": register_status,
            "readmostly_result": readmostly_result,
            "delta_interesting": {key: delta.get(key, 0) for key in INTERESTING_STATS},
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
    lib = _load_probe()
    _check(lib.uvm_stage_probe_set_device(args.device), "set_device")
    _check(lib.uvm_stage_probe_device_synchronize(), "device_synchronize")
    fault_stats_path = find_fault_stats()

    rows: list[dict[str, Any]] = []
    cases = [item for item in args.cases.split(",") if item.strip()]
    for byte_count in _parse_sizes(args.sizes):
        for case in cases:
            samples = [
                _run_once(
                    lib=lib,
                    fault_stats_path=fault_stats_path,
                    case=case,
                    byte_count=byte_count,
                    iters=args.iters,
                    device=device,
                    device_index=args.device,
                    region_seed=(time.time_ns() + repeat * 1000 + len(case)) & 0x7FFFFFFFFFFFFFFF,
                )
                for repeat in range(args.repeats)
            ]
            rows.append(
                {
                    "case": case,
                    "bytes": byte_count,
                    "mib": byte_count / 1024 / 1024,
                    "iters": args.iters,
                    "repeats": args.repeats,
                    "elapsed_ms": _stats([float(sample["elapsed_ms"]) for sample in samples]),
                    "first_delta": samples[0]["delta_interesting"],
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
            "iters": args.iters,
            "repeats": args.repeats,
            "cases": cases,
            "fault_stats_path": str(fault_stats_path),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", default="4.5MiB,18MiB,72MiB,144MiB")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--cases",
        default="native_readmostly,expert_forced_readdup,native_no_readmostly",
        help="comma-separated cases: native_readmostly,expert_forced_readdup,expert_fastpath,native_no_readmostly,expert_forced_plus_readmostly",
    )
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("log/uvm_fault_readdup_compare/result.json"))
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
