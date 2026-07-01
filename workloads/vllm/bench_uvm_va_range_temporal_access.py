#!/usr/bin/env python3
"""Measure temporal access order of tagged UVM expert VA ranges."""

from __future__ import annotations

import argparse
import ctypes
import json
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
    launch_read,
    parse_size,
    read_fault_stats,
    unregister_region,
)
from bench_uvm_stage_boundary_remap import _check, _load_probe  # noqa: E402
import vllm.device_allocator.uvm as uvm_mod  # noqa: E402
from vllm.device_allocator.uvm import (  # noqa: E402
    UVM_REGION_FLAG_EVICTABLE,
    UVM_REGION_FLAG_PREFETCHABLE,
    UVM_REGION_POOL_WEIGHT,
    UVM_REGION_ROLE_MOE_EXPERT_W13,
    UVM_REGION_ROLE_MOE_EXPERT_W2,
    batch_prefetch_ranges_to_cpu,
    cleanup_uvm_local_memory_pool,
)


def _make_region_id(base: int, index: int) -> int:
    return (base + index + 1) & 0x7FFFFFFFFFFFFFFF


def _interesting_delta(delta: dict[str, int]) -> dict[str, int]:
    keys = (
        "replayable_faults",
        "duplicates",
        "faults_by_phase_n_pool.pool_2",
        "faults_by_phase_n_role.role_3",
        "faults_by_phase_n_role.role_4",
        "phase_n_expert_dispatches",
        "phase_n_expert_faults_serviced",
        "phase_n_non_expert_managed_dispatches",
        "phase_n_non_expert_managed_faults_serviced",
        "phase_n_unregistered_faults",
    )
    return {key: int(delta.get(key, 0)) for key in keys}


def register_temporal_expert(
    ptr: int,
    byte_count: int,
    role: int,
    region_id: int,
    layer_id: int,
) -> tuple[int, dict[str, int] | None]:
    lib = uvm_mod.ensure_uvm_runtime_library()
    status = int(
        lib.uvm_driver_phase_f_register_region(
            ctypes.c_uint64(region_id),
            ctypes.c_uint64(ptr),
            ctypes.c_size_t(byte_count),
            ctypes.c_uint32(UVM_REGION_POOL_WEIGHT),
            ctypes.c_uint32(layer_id),
            ctypes.c_uint32(0),
            ctypes.c_uint32(role),
            ctypes.c_uint32(UVM_REGION_FLAG_EVICTABLE | UVM_REGION_FLAG_PREFETCHABLE),
        )
    )
    values = (ctypes.c_uint64 * 11)()
    query_status = int(
        lib.uvm_driver_phase_n_query_region_by_va(
            ctypes.c_uint64(ptr),
            values,
            ctypes.c_size_t(len(values)),
        )
    )
    if query_status < 0:
        return status, None

    keys = (
        "region_id",
        "base",
        "length",
        "pool",
        "layer_id",
        "expert_id",
        "role",
        "flags",
        "state",
        "priority",
        "last_access_epoch",
    )
    region_query = {key: int(values[index]) for index, key in enumerate(keys)}
    region_query["query_status"] = query_status
    return status, region_query


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.synchronize(device)
    uvm_mod.ensure_uvm_runtime_library()

    lib = _load_probe()
    _check(lib.uvm_stage_probe_set_device(args.device), "set_device")
    _check(lib.uvm_stage_probe_device_synchronize(), "device_synchronize")
    fault_stats_path = find_fault_stats()

    bytes_per_range = parse_size(args.bytes)
    range_count = args.layers * 2
    base_region_id = time.time_ns()
    tensors: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    region_ids: list[int] = []

    try:
        for index in range(range_count):
            layer_id = index // 2
            role_name = "w13" if index % 2 == 0 else "w2"
            role_id = (
                UVM_REGION_ROLE_MOE_EXPERT_W13
                if role_name == "w13"
                else UVM_REGION_ROLE_MOE_EXPERT_W2
            )
            tensor, backing = _alloc_gpu_tensor((bytes_per_range,), torch.uint8, device, "uvm")
            if backing != "uvm":
                raise RuntimeError(f"expected UVM backing, got {backing}")
            tensor.fill_(0x31 + (index % 67))
            torch.cuda.synchronize(device)
            ptr = int(tensor.data_ptr())
            region_id = _make_region_id(base_region_id, index)
            register_status, region_query = register_temporal_expert(
                ptr,
                bytes_per_range,
                role_id,
                region_id,
                layer_id,
            )
            if register_status < 0:
                raise RuntimeError(f"register expert range {index} failed: {register_status}")
            tensors.append(tensor)
            region_ids.append(region_id)
            records.append(
                {
                    "index": index,
                    "layer_id": layer_id,
                    "role": role_name,
                    "role_id": role_id,
                    "region_id": region_id,
                    "ptr": ptr,
                    "bytes": bytes_per_range,
                    "region_query": region_query,
                }
            )

        ranges = [(int(tensor.data_ptr()), bytes_per_range) for tensor in tensors]
        prefetch_result = int(batch_prefetch_ranges_to_cpu(ranges))
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        access_rows: list[dict[str, Any]] = []
        for record in records:
            before = read_fault_stats(fault_stats_path)
            host_start_ms = (time.perf_counter() - t0) * 1000.0
            elapsed_ms = launch_read(
                lib,
                int(record["ptr"]),
                bytes_per_range,
                args.iters,
                args.device,
            )
            host_end_ms = (time.perf_counter() - t0) * 1000.0
            after = read_fault_stats(fault_stats_path)
            delta = delta_stats(before, after)
            access_rows.append(
                {
                    **record,
                    "host_start_ms": host_start_ms,
                    "host_end_ms": host_end_ms,
                    "kernel_elapsed_ms": elapsed_ms,
                    "delta": _interesting_delta(delta),
                }
            )

        return {
            "device": args.device,
            "layers": args.layers,
            "range_count": range_count,
            "bytes_per_range": bytes_per_range,
            "iters": args.iters,
            "fault_stats_path": str(fault_stats_path),
            "prefetch_cpu_result": prefetch_result,
            "access_rows": access_rows,
            "summary": {
                "expert_dispatches_total": sum(
                    row["delta"]["phase_n_expert_dispatches"] for row in access_rows
                ),
                "non_expert_dispatches_total": sum(
                    row["delta"]["phase_n_non_expert_managed_dispatches"] for row in access_rows
                ),
                "unregistered_faults_total": sum(
                    row["delta"]["phase_n_unregistered_faults"] for row in access_rows
                ),
                "rows_with_faults": sum(
                    1 for row in access_rows if row["delta"]["replayable_faults"] > 0
                ),
            },
        }
    finally:
        for region_id in region_ids:
            try:
                unregister_region(region_id)
            except Exception:
                pass
        del tensors
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--bytes", default="1MiB")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("log/uvm_va_range_temporal_access/result.json"),
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
