#!/usr/bin/env python3
"""Probe expert-tagged UVM fault dispatch counters."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
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
    probe_driver_capable_range,
)

STAT_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([0-9]+)")
SECTION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*$")


def parse_size(raw: str) -> int:
    token = raw.strip().lower()
    scale = 1
    if token.endswith("mib"):
        scale = 1024 * 1024
        token = token[:-3]
    elif token.endswith("mb"):
        scale = 1000 * 1000
        token = token[:-2]
    elif token.endswith("kib"):
        scale = 1024
        token = token[:-3]
    elif token.endswith("kb"):
        scale = 1000
        token = token[:-2]
    return int(float(token) * scale)


def find_fault_stats() -> Path:
    root = Path("/proc/driver/nvidia-uvm")
    paths = sorted(root.rglob("fault_stats"))
    if not paths:
        raise RuntimeError(
            "No /proc/driver/nvidia-uvm/**/fault_stats found. "
            "Reload nvidia-uvm with uvm_enable_debug_procfs=1."
        )
    return paths[0]


def read_fault_stats(path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    section = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        section_match = SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1)
            continue
        match = STAT_LINE_RE.match(line)
        if not match:
            continue
        key = match.group(1)
        if key.startswith(("pool_", "role_")) and section:
            key = f"{section}.{key}"
        stats[key] = int(match.group(2))
    return stats


def delta_stats(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


def make_managed_tensor(byte_count: int, device: torch.device) -> torch.Tensor:
    tensor, backing = _alloc_gpu_tensor((byte_count,), torch.uint8, device, "uvm")
    if backing != "uvm":
        raise RuntimeError(f"expected UVM backing, got {backing}")
    return tensor


def prefetch_cpu(ptr: int, byte_count: int, device: torch.device) -> int:
    result = int(batch_prefetch_ranges_to_cpu([(ptr, byte_count)]))
    torch.cuda.synchronize(device)
    return result


def launch_read(lib: ctypes.CDLL, ptr: int, byte_count: int, iters: int, device_index: int) -> float:
    stream = ctypes.c_void_p()
    out = ctypes.c_void_p()
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    try:
        _check(lib.uvm_stage_probe_create_stream(ctypes.byref(stream)), "create_stream")
        _check(lib.uvm_stage_probe_alloc_device(ctypes.byref(out), byte_count, device_index), "alloc_output")
        _check(lib.uvm_stage_probe_create_event(ctypes.byref(start), 1), "create_start")
        _check(lib.uvm_stage_probe_create_event(ctypes.byref(end), 1), "create_end")
        _check(lib.uvm_stage_probe_record_event(start, stream), "record_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                ctypes.c_void_p(ptr),
                out,
                byte_count,
                iters,
                stream,
            ),
            "launch_read",
        )
        _check(lib.uvm_stage_probe_record_event(end, stream), "record_end")
        _check(lib.uvm_stage_probe_sync_event(end), "sync_end")
        elapsed = ctypes.c_float()
        _check(lib.uvm_stage_probe_elapsed_ms(start, end, ctypes.byref(elapsed)), "elapsed")
        return float(elapsed.value)
    finally:
        if start:
            lib.uvm_stage_probe_destroy_event(start)
        if end:
            lib.uvm_stage_probe_destroy_event(end)
        if out:
            lib.uvm_stage_probe_free_device(out)
        if stream:
            lib.uvm_stage_probe_destroy_stream(stream)


def register_expert(ptr: int, byte_count: int, role: int, region_id: int) -> tuple[int, dict[str, int] | None]:
    lib = uvm_mod.ensure_uvm_runtime_library()
    status = int(
        lib.uvm_driver_phase_f_register_region(
            ctypes.c_uint64(region_id),
            ctypes.c_uint64(ptr),
            ctypes.c_size_t(byte_count),
            ctypes.c_uint32(UVM_REGION_POOL_WEIGHT),
            ctypes.c_uint32(31),
            ctypes.c_uint32(7),
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
    region_query = None
    if query_status >= 0:
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


def unregister_region(region_id: int) -> int:
    lib = uvm_mod.ensure_uvm_runtime_library()
    return int(lib.uvm_driver_phase_f_unregister_region(ctypes.c_uint64(region_id)))


def run_case(
    *,
    lib: ctypes.CDLL,
    args: argparse.Namespace,
    fault_stats_path: Path,
    case_name: str,
    register: bool,
    role: int,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{args.device}")
    byte_count = parse_size(args.bytes)
    tensor = make_managed_tensor(byte_count, device)
    tensor.fill_(0x5A)
    torch.cuda.synchronize(device)

    ptr = int(tensor.data_ptr())
    managed_ok, managed_reason, managed_code = probe_driver_capable_range(ptr, byte_count)
    region_id = int(time.time_ns() & 0x7FFFFFFFFFFFFFFF)
    register_status = None
    region_query = None
    if register:
        register_status, region_query = register_expert(ptr, byte_count, role, region_id)

    prefetch_result = prefetch_cpu(ptr, byte_count, device)
    before = read_fault_stats(fault_stats_path)
    elapsed_ms = launch_read(lib, ptr, byte_count, args.iters, args.device)
    after = read_fault_stats(fault_stats_path)
    delta = delta_stats(before, after)

    if register:
        unregister_region(region_id)

    del tensor
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    interesting_keys = [
        "replayable_faults",
        "duplicates",
        "faults_by_phase_n_pool.pool_2",
        "faults_by_phase_n_role.role_3",
        "faults_by_phase_n_role.role_4",
        "phase_n_unregistered_faults",
        "phase_n_expert_dispatches",
        "phase_n_expert_faults_serviced",
        "phase_n_non_expert_managed_dispatches",
        "phase_n_non_expert_managed_faults_serviced",
    ]

    return {
        "case": case_name,
        "byte_count": byte_count,
        "ptr": ptr,
        "managed_probe": {
            "ok": bool(managed_ok),
            "reason": managed_reason,
            "code": int(managed_code),
        },
        "register_status": register_status,
        "region_query": region_query,
        "prefetch_cpu_result": prefetch_result,
        "read_elapsed_ms": elapsed_ms,
        "delta_interesting": {key: delta.get(key, 0) for key in interesting_keys},
        "delta_all": delta,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(args.device)
    torch.cuda.init()
    torch.cuda.synchronize()

    uvm_mod.ensure_uvm_runtime_library()
    lib = _load_probe()
    _check(lib.uvm_stage_probe_set_device(args.device), "set_device")
    _check(lib.uvm_stage_probe_device_synchronize(), "device_synchronize")
    fault_stats_path = find_fault_stats()

    cases: list[dict[str, Any]] = []
    cases.append(
        run_case(
            lib=lib,
            args=args,
            fault_stats_path=fault_stats_path,
            case_name="expert_only",
            register=True,
            role=UVM_REGION_ROLE_MOE_EXPERT_W13 if args.role == "w13" else UVM_REGION_ROLE_MOE_EXPERT_W2,
        )
    )
    if args.include_control:
        cases.append(
            run_case(
                lib=lib,
                args=args,
                fault_stats_path=fault_stats_path,
                case_name="unregistered_control",
                register=False,
                role=UVM_REGION_ROLE_MOE_EXPERT_W2,
            )
        )

    return {
        "fault_stats_path": str(fault_stats_path),
        "device": args.device,
        "bytes": args.bytes,
        "iters": args.iters,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bytes", default="4.5MiB")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--role", choices=("w13", "w2"), default="w13")
    parser.add_argument("--include-control", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("log/uvm_expert_fault_tag_counter/result.json"))
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
