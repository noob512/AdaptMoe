#!/usr/bin/env python3
"""Standalone multi-expert UVM prefetch experiment.

This does not modify or import the vLLM MoE layer. It simulates a layer with
many UVM-resident MoE experts, selects a non-contiguous subset, and compares:

  * one Phase-F EXPERT_READ_DUP_FAST ioctl carrying all selected expert ranges
  * one Phase-F EXPERT_READ_DUP_FAST region-id ioctl carrying selected expert ids
  * serial Phase-F EXPERT_READ_DUP_FAST calls, one expert range at a time
  * serial cudaMemPrefetchAsync via the existing allocator shim

The goal is to validate the user-space shape needed by vLLM: after topk_ids are
known, collect the selected experts' UVM weight ranges and submit them together.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from vllm.device_allocator.uvm import (  # noqa: E402
    UVM_REGION_FLAG_EVICTABLE,
    UVM_REGION_FLAG_PREFETCHABLE,
    UVM_REGION_FLAG_STREAMING_SUBPOOL,
    UVM_REGION_POOL_WEIGHT,
    UVM_REGION_ROLE_MOE_EXPERT_W13,
    UVM_REGION_ROLE_MOE_EXPERT_W2,
    batch_prefetch_ranges_to_device,
    batch_prefetch_ranges_to_cpu,
    cleanup_uvm_local_memory_pool,
    driver_expert_read_dup_fast,
    driver_expert_read_dup_fast_native_regions,
    driver_expert_read_dup_fast_regions,
    ensure_uvm_runtime_library,
    probe_driver_capable_range,
    uvm_local_memory_pool,
)


LATENCY_STATS_PATH = Path("/proc/driver/nvidia-uvm/latency_stats")
LATENCY_STATS_RESET_PATH = Path("/proc/driver/nvidia-uvm/latency_stats_reset")
PARAM_DIR = Path("/sys/module/nvidia_uvm/parameters")

INTERESTING_LATENCY_KEYS = (
    "expert_batch_attempts",
    "expert_batch_successes",
    "expert_batch_fallbacks",
    "expert_batch_errors",
    "expert_batch_requested_entries",
    "expert_batch_active_entries",
    "expert_batch_committed_pages",
)


@dataclass
class ExpertRange:
    expert_id: int
    role_index: int
    role: int
    region_id: int
    tensor: torch.Tensor
    ptr: int
    size: int

    @property
    def range_tuple(self) -> tuple[int, int]:
        return self.ptr, self.size


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


def summarize(values: list[float]) -> dict[str, float | int]:
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


def read_int_param(name: str) -> int | None:
    path = PARAM_DIR / name
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def write_int_param(name: str, value: int) -> bool:
    path = PARAM_DIR / name
    try:
        path.write_text(f"{int(value)}\n", encoding="utf-8")
        return True
    except Exception:
        return False


def read_latency_stats() -> dict[str, int]:
    if not LATENCY_STATS_PATH.exists():
        return {}

    stats: dict[str, int] = {}
    for raw_line in LATENCY_STATS_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.endswith(":"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            stats[parts[0]] = int(parts[1])
    return stats


def reset_latency_stats() -> bool:
    try:
        LATENCY_STATS_RESET_PATH.write_text("1\n", encoding="utf-8")
        return True
    except Exception:
        return False


def stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


def make_region_id(run_seed: int, expert_id: int, role_index: int, repeat: int, salt: int) -> int:
    value = (
        (run_seed & 0xFFFF) << 48
        | (repeat & 0xFFFF) << 32
        | (expert_id & 0xFFFF) << 16
        | ((role_index & 0xFF) << 8)
        | (salt & 0xFF)
    )
    return value or 1


def register_region(
    *,
    region_id: int,
    ptr: int,
    size: int,
    layer_id: int,
    expert_id: int,
    role: int,
) -> int:
    lib = ensure_uvm_runtime_library()
    flags = UVM_REGION_FLAG_EVICTABLE | UVM_REGION_FLAG_PREFETCHABLE | UVM_REGION_FLAG_STREAMING_SUBPOOL
    return int(
        lib.uvm_driver_phase_f_register_region(
            region_id,
            ptr,
            size,
            UVM_REGION_POOL_WEIGHT,
            layer_id,
            expert_id,
            role,
            flags,
        )
    )


def unregister_region(region_id: int) -> int:
    lib = ensure_uvm_runtime_library()
    return int(lib.uvm_driver_phase_f_unregister_region(region_id))


def allocate_expert_ranges(
    *,
    args: argparse.Namespace,
    repeat: int,
    selected_ids: list[int],
    device: torch.device,
    run_seed: int,
) -> list[ExpertRange]:
    ranges: list[ExpertRange] = []
    roles = [UVM_REGION_ROLE_MOE_EXPERT_W13]
    if args.roles_per_expert >= 2:
        roles.append(UVM_REGION_ROLE_MOE_EXPERT_W2)

    bytes_per_role = args.expert_bytes // len(roles)
    if bytes_per_role <= 0:
        raise ValueError("expert bytes too small")

    for expert_id in selected_ids:
        for role_index, role in enumerate(roles):
            with uvm_local_memory_pool():
                tensor = torch.empty((bytes_per_role,), dtype=torch.uint8, device=device)
            tensor.fill_((0x41 + expert_id + role_index) & 0xFF)
            torch.cuda.synchronize(device)

            ptr = int(tensor.data_ptr())
            managed_ok, managed_reason, managed_code = probe_driver_capable_range(ptr, bytes_per_role)
            if not managed_ok:
                raise RuntimeError(
                    f"allocated pointer is not driver-capable managed memory: "
                    f"expert={expert_id} role_index={role_index} ptr=0x{ptr:x} "
                    f"size={bytes_per_role} reason={managed_reason} code={managed_code}"
                )

            region_id = make_region_id(run_seed, expert_id, role_index, repeat, args.region_id_salt)
            status = register_region(
                region_id=region_id,
                ptr=ptr,
                size=bytes_per_role,
                layer_id=args.layer_id,
                expert_id=expert_id,
                role=role,
            )
            if status < 0:
                raise RuntimeError(
                    f"register_region failed: status={status} expert={expert_id} "
                    f"role={role} ptr=0x{ptr:x} size={bytes_per_role}"
                )

            ranges.append(
                ExpertRange(
                    expert_id=expert_id,
                    role_index=role_index,
                    role=role,
                    region_id=region_id,
                    tensor=tensor,
                    ptr=ptr,
                    size=bytes_per_role,
                )
            )

    return ranges


def cleanup_expert_ranges(ranges: list[ExpertRange], device: torch.device) -> None:
    for item in ranges:
        try:
            unregister_region(item.region_id)
        except Exception:
            pass
    ranges.clear()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def force_cpu_resident(ranges: list[ExpertRange], device: torch.device) -> int:
    tuples = [item.range_tuple for item in ranges]
    result = int(batch_prefetch_ranges_to_cpu(tuples))
    torch.cuda.synchronize(device)
    return result


def verify_gpu_residency(
    ranges: list[ExpertRange],
    *,
    device_index: int,
) -> dict[str, int | bool]:
    result: dict[str, int | bool] = {
        "ok": False,
        "status": -1,
        "queried_bytes": 0,
        "gpu_bytes": 0,
        "cpu_bytes": 0,
        "any_bytes": 0,
        "unknown_bytes": 0,
    }
    if not ranges:
        result["ok"] = True
        result["status"] = 0
        return result

    lib = ensure_uvm_runtime_library()
    ptr_array = (ctypes.c_uint64 * len(ranges))(*(item.ptr for item in ranges))
    size_array = (ctypes.c_size_t * len(ranges))(*(item.size for item in ranges))
    gpu_bytes = ctypes.c_uint64(0)
    cpu_bytes = ctypes.c_uint64(0)
    any_bytes = ctypes.c_uint64(0)
    unknown_bytes = ctypes.c_uint64(0)
    queried_bytes = ctypes.c_uint64(0)
    status = int(
        lib.uvm_driver_batch_query_residency_bytes(
            ptr_array,
            size_array,
            ctypes.c_size_t(len(ranges)),
            ctypes.c_int(device_index),
            ctypes.byref(gpu_bytes),
            ctypes.byref(cpu_bytes),
            ctypes.byref(any_bytes),
            ctypes.byref(unknown_bytes),
            ctypes.byref(queried_bytes),
        )
    )
    result.update(
        {
            "ok": status >= 0,
            "status": status,
            "queried_bytes": int(queried_bytes.value),
            "gpu_bytes": int(gpu_bytes.value),
            "cpu_bytes": int(cpu_bytes.value),
            "any_bytes": int(any_bytes.value),
            "unknown_bytes": int(unknown_bytes.value),
        }
    )
    return result


def selected_ranges(all_ranges: list[ExpertRange], selected_ids: list[int]) -> list[ExpertRange]:
    selected = set(selected_ids)
    return [item for item in all_ranges if item.expert_id in selected]


def unselected_ranges(all_ranges: list[ExpertRange], selected_ids: list[int]) -> list[ExpertRange]:
    selected = set(selected_ids)
    return [item for item in all_ranges if item.expert_id not in selected]


def time_submit_and_sync(
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


def run_case(
    *,
    case: str,
    args: argparse.Namespace,
    repeat: int,
    selected_ids: list[int],
    device: torch.device,
    run_seed: int,
) -> dict[str, Any]:
    allocation_ids = selected_ids if args.selected_only_alloc else list(range(args.experts))
    all_ranges = allocate_expert_ranges(
        args=args,
        repeat=repeat,
        selected_ids=allocation_ids,
        device=device,
        run_seed=run_seed,
    )

    try:
        ranges = selected_ranges(all_ranges, selected_ids)
        cold_ranges = unselected_ranges(all_ranges, selected_ids)
        tuples = [item.range_tuple for item in ranges]
        cpu_prefetch_success = force_cpu_resident(all_ranges, device)
        before_residency = verify_gpu_residency(ranges, device_index=args.device)
        before_unselected_residency = verify_gpu_residency(cold_ranges, device_index=args.device)
        before_stats = read_latency_stats()

        if case == "expert_fast_batch":
            submit = lambda: driver_expert_read_dup_fast(tuples, device=args.device, async_=args.async_ioctl)
        elif case == "expert_fast_regions":
            region_ids = [item.region_id for item in ranges]
            submit = lambda: driver_expert_read_dup_fast_regions(region_ids, device=args.device, async_=args.async_ioctl)
        elif case == "expert_fast_native_regions":
            region_ids = [item.region_id for item in ranges]
            submit = lambda: driver_expert_read_dup_fast_native_regions(region_ids, device=args.device, async_=args.async_ioctl)
        elif case == "expert_fast_serial":
            def submit() -> list[tuple[int, int, int, int]]:
                return [
                    driver_expert_read_dup_fast([range_tuple], device=args.device, async_=args.async_ioctl)
                    for range_tuple in tuples
                ]
        elif case == "cuda_prefetch_serial":
            submit = lambda: batch_prefetch_ranges_to_device(tuples, device=args.device)
        else:
            raise ValueError(f"unknown case {case!r}")

        submit_ms, complete_ms, result = time_submit_and_sync(submit, device=device)
        after_stats = read_latency_stats()
        after_residency = verify_gpu_residency(ranges, device_index=args.device)
        after_unselected_residency = verify_gpu_residency(cold_ranges, device_index=args.device)

        requested_bytes = sum(item.size for item in ranges)
        gpu_bytes = int(after_residency.get("gpu_bytes") or 0)
        correct = gpu_bytes >= requested_bytes

        return {
            "case": case,
            "repeat": repeat,
            "selected_experts": selected_ids,
            "range_count": len(ranges),
            "allocated_experts": allocation_ids,
            "allocated_range_count": len(all_ranges),
            "requested_bytes": requested_bytes,
            "requested_mib": requested_bytes / 1024 / 1024,
            "cpu_prefetch_success": cpu_prefetch_success,
            "submit_ms": submit_ms,
            "complete_ms": complete_ms,
            "result": result,
            "before_residency": before_residency,
            "before_unselected_residency": before_unselected_residency,
            "after_residency": after_residency,
            "after_unselected_residency": after_unselected_residency,
            "correct": correct,
            "latency_delta": {
                key: stats_delta(before_stats, after_stats).get(key, 0)
                for key in INTERESTING_LATENCY_KEYS
            },
        }
    finally:
        cleanup_expert_ranges(all_ranges, device)


def parse_selected(raw: str, num_experts: int, select_count: int) -> list[int]:
    if raw:
        ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        stride = 3
        ids = []
        candidate = 1
        while len(ids) < select_count:
            value = candidate % num_experts
            if value not in ids:
                ids.append(value)
            candidate += stride
    if not ids:
        raise ValueError("no selected experts")
    if len(set(ids)) != len(ids):
        raise ValueError(f"selected experts contain duplicates: {ids}")
    bad = [expert_id for expert_id in ids if expert_id < 0 or expert_id >= num_experts]
    if bad:
        raise ValueError(f"selected experts outside [0, {num_experts}): {bad}")
    return ids


def maybe_configure_kernel_params() -> dict[str, Any]:
    requested = {
        "uvm_phase_n_range_tag_enable": 1,
        "uvm_phase_n_expert_migrate_fastpath_enable": 1,
        "uvm_phase_n_expert_migrate_batch_enable": 1,
        "uvm_phase_n_expert_migrate_fastpath_align_blocks_enable": 1,
    }
    result: dict[str, Any] = {}
    for name, value in requested.items():
        before = read_int_param(name)
        wrote = False
        if before is not None and before != value:
            wrote = write_int_param(name, value)
        after = read_int_param(name)
        result[name] = {
            "before": before,
            "requested": value,
            "write_attempted": before is not None and before != value,
            "write_ok": wrote,
            "after": after,
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.init()
    ensure_uvm_runtime_library()

    if args.expert_bytes <= 0:
        raise ValueError("expert size must be positive")
    if args.expert_bytes % args.roles_per_expert != 0:
        raise ValueError("--expert-size must be divisible by --roles-per-expert")

    selected_ids = parse_selected(args.selected, args.experts, args.select_count)
    kernel_params = maybe_configure_kernel_params() if args.configure_kernel else {}
    latency_reset_ok = reset_latency_stats() if args.reset_latency_stats else False

    cases = list(args.cases)
    rows: list[dict[str, Any]] = []
    run_seed = int(time.time_ns() & 0xFFFF)

    for repeat in range(args.repeats):
        for case in cases:
            rows.append(
                run_case(
                    case=case,
                    args=args,
                    repeat=repeat,
                    selected_ids=selected_ids,
                    device=device,
                    run_seed=run_seed,
                )
            )

    summaries: list[dict[str, Any]] = []
    for case in cases:
        samples = [row for row in rows if row["case"] == case]
        submit_values = [float(row["submit_ms"]) for row in samples]
        complete_values = [float(row["complete_ms"]) for row in samples]
        requested_bytes = int(samples[0]["requested_bytes"]) if samples else 0
        range_count = int(samples[0]["range_count"]) if samples else 0
        correct_count = sum(1 for row in samples if bool(row["correct"]))
        unselected_cold_count = sum(
            1
            for row in samples
            if int(row["after_unselected_residency"].get("gpu_bytes") or 0) == 0
        )
        summaries.append(
            {
                "case": case,
                "repeats": len(samples),
                "range_count": range_count,
                "requested_bytes": requested_bytes,
                "requested_mib": requested_bytes / 1024 / 1024,
                "correct_count": correct_count,
                "unselected_cold_count": unselected_cold_count,
                "submit_ms": summarize(submit_values),
                "complete_ms": summarize(complete_values),
                "effective_complete_gib_s_p50": (
                    (requested_bytes / 1024**3) / (summarize(complete_values)["p50_ms"] / 1000.0)
                    if requested_bytes > 0 and summarize(complete_values)["p50_ms"] > 0
                    else 0.0
                ),
                "latency_delta_sum": {
                    key: sum(int(row["latency_delta"].get(key, 0)) for row in samples)
                    for key in INTERESTING_LATENCY_KEYS
                },
                "first_result": samples[0]["result"] if samples else None,
                "last_after_residency": samples[-1]["after_residency"] if samples else None,
                "last_after_unselected_residency": (
                    samples[-1]["after_unselected_residency"] if samples else None
                ),
            }
        )

    return {
        "config": {
            "device": args.device,
            "experts": args.experts,
            "selected_experts": selected_ids,
            "select_count": len(selected_ids),
            "roles_per_expert": args.roles_per_expert,
            "expert_bytes": args.expert_bytes,
            "expert_mib": args.expert_bytes / 1024 / 1024,
            "range_count_per_repeat": len(selected_ids) * args.roles_per_expert,
            "allocated_range_count_per_repeat": (
                len(selected_ids) if args.selected_only_alloc else args.experts
            ) * args.roles_per_expert,
            "selected_only_alloc": args.selected_only_alloc,
            "repeats": args.repeats,
            "cases": cases,
            "async_ioctl": args.async_ioctl,
            "configure_kernel": args.configure_kernel,
            "kernel_params": kernel_params,
            "latency_reset_ok": latency_reset_ok,
            "pid": os.getpid(),
        },
        "summaries": summaries,
        "samples": [] if args.no_samples else rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--select-count", type=int, default=10)
    parser.add_argument(
        "--selected",
        default="1,4,7,10,13,16,19,22,25,28",
        help="Comma-separated selected expert ids. Empty string uses generated non-contiguous ids.",
    )
    parser.add_argument("--expert-size", default="2MiB")
    parser.add_argument("--roles-per-expert", type=int, choices=(1, 2), default=1)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--async-ioctl", action="store_true")
    parser.add_argument(
        "--selected-only-alloc",
        action="store_true",
        help="Allocate only selected experts. By default all --experts experts are allocated and CPU-resident.",
    )
    parser.add_argument("--configure-kernel", action="store_true")
    parser.add_argument("--reset-latency-stats", action="store_true")
    parser.add_argument("--region-id-salt", type=int, default=0xA5)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "expert_fast_batch",
            "expert_fast_regions",
            "expert_fast_native_regions",
            "expert_fast_serial",
            "cuda_prefetch_serial",
        ],
        choices=[
            "expert_fast_batch",
            "expert_fast_regions",
            "expert_fast_native_regions",
            "expert_fast_serial",
            "cuda_prefetch_serial",
        ],
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-samples", action="store_true")
    args = parser.parse_args()
    args.expert_bytes = parse_size(args.expert_size)

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_uvm_local_memory_pool()
