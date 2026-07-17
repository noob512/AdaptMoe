#!/usr/bin/env python3
"""Pressure-test 512KiB UVM subchunk release against ordinary cudaMalloc."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_2mb_offload_release_slot_restore import (  # noqa: E402
    _pattern,
    _query_residency,
    _register_expert_region,
    _unregister_region,
)
from check_512k_subrange_release_phase_s2 import (  # noqa: E402
    _release_region_subchunk_64k,
)
from check_moe_slot_remap_ioctl_smoke import (  # noqa: E402
    UVM_VA_BLOCK_SIZE,
    ManagedAllocation,
    cuda_memcpy_host_to_device,
    gpu_uuid_for_device,
    initialize_cuda_device,
    ioctl,
    native_migrate,
    open_initialized_uvm_fd,
)
from check_uvm_pmm_512k_phase_s0 import CHUNK_512K, PAGE_64K  # noqa: E402

UVM_PHASE_N_QUERY_REGION_BLOCKS_IOCTL = 96
UVM_PHASE_N_REGION_BLOCK_TRACE_MAX = 16
ROUTE_LOG_PATH = Path("/proc/driver/nvidia-uvm/phase_f_route_log")
MIB = 1024 * 1024


class PhaseNQueryRegionBlocksParams(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("pool", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("priority", ctypes.c_uint32),
        ("va_range_count", ctypes.c_uint32),
        ("logical_block_count", ctypes.c_uint64),
        ("populated_block_count", ctypes.c_uint64),
        ("resident_block_count", ctypes.c_uint64),
        ("pending_block_count", ctypes.c_uint64),
        ("materialized_block_count", ctypes.c_uint64),
        ("mixed_block_count", ctypes.c_uint64),
        ("resident_gpu_bytes", ctypes.c_uint64),
        ("pending_gpu_bytes", ctypes.c_uint64),
        ("first_block_start", ctypes.c_uint64),
        ("last_block_end", ctypes.c_uint64),
        ("sampled_block_count", ctypes.c_uint32),
        ("reserved_out", ctypes.c_uint32),
        ("block_starts", ctypes.c_uint64 * UVM_PHASE_N_REGION_BLOCK_TRACE_MAX),
        ("block_ends", ctypes.c_uint64 * UVM_PHASE_N_REGION_BLOCK_TRACE_MAX),
        ("block_bytes", ctypes.c_uint64 * UVM_PHASE_N_REGION_BLOCK_TRACE_MAX),
        ("block_states", ctypes.c_uint32 * UVM_PHASE_N_REGION_BLOCK_TRACE_MAX),
        ("block_status", ctypes.c_uint32 * UVM_PHASE_N_REGION_BLOCK_TRACE_MAX),
        ("rmStatus", ctypes.c_uint32),
    ]


class CudaRuntime:
    def __init__(self) -> None:
        libname = ctypes.util.find_library("cudart") or "libcudart.so"
        self.lib = ctypes.CDLL(libname)

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.cudaMemGetInfo.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaGetLastError.argtypes = []
        self.lib.cudaGetLastError.restype = ctypes.c_int

    def error_string(self, ret: int) -> str:
        text = self.lib.cudaGetErrorString(ctypes.c_int(ret))
        if not text:
            return ""
        return text.decode("utf-8", errors="replace")

    def clear_last_error(self) -> int:
        return int(self.lib.cudaGetLastError())

    def mem_get_info(self) -> dict[str, int]:
        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        ret = int(self.lib.cudaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)))
        if ret != 0:
            self.clear_last_error()
            return {"ret": ret, "free": 0, "total": 0, "used": 0}
        return {
            "ret": ret,
            "free": int(free_bytes.value),
            "total": int(total_bytes.value),
            "used": int(total_bytes.value - free_bytes.value),
        }

    def malloc_raw(self, byte_count: int) -> tuple[int, int]:
        ptr = ctypes.c_void_p()
        ret = int(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(byte_count)))
        if ret != 0:
            self.clear_last_error()
            return 0, ret
        return int(ptr.value or 0), ret

    def free_raw(self, ptr: int) -> int:
        return int(self.lib.cudaFree(ctypes.c_void_p(ptr)))


class CudaAllocation:
    def __init__(self, runtime: CudaRuntime, ptr: int, byte_count: int, label: str) -> None:
        self.runtime = runtime
        self.ptr = int(ptr)
        self.byte_count = int(byte_count)
        self.label = label

    def close(self) -> int:
        if not self.ptr:
            return 0
        ret = self.runtime.free_raw(self.ptr)
        self.ptr = 0
        return ret


def _mem_snapshot(runtime: CudaRuntime, label: str) -> dict[str, int | str]:
    info = runtime.mem_get_info()
    info["label"] = label
    return info


def _try_cuda_malloc(runtime: CudaRuntime, byte_count: int, label: str) -> tuple[dict[str, Any], CudaAllocation | None]:
    before = runtime.mem_get_info()
    ptr, ret = runtime.malloc_raw(byte_count)
    after = runtime.mem_get_info()
    event: dict[str, Any] = {
        "label": label,
        "requested_bytes": int(byte_count),
        "ret": int(ret),
        "error": runtime.error_string(ret) if ret != 0 else "",
        "ok": ret == 0,
        "ptr": int(ptr),
        "mem_before": before,
        "mem_after": after,
    }
    if ret != 0:
        return event, None
    return event, CudaAllocation(runtime, ptr, byte_count, label)


def _align_down(value: int, alignment: int) -> int:
    return max(0, int(value) // int(alignment) * int(alignment))


def _fill_cuda_until_free_at_most(
    runtime: CudaRuntime,
    *,
    target_free_bytes: int,
    chunk_bytes: int,
    min_chunk_bytes: int,
    label: str,
) -> tuple[list[CudaAllocation], list[dict[str, Any]]]:
    allocations: list[CudaAllocation] = []
    events: list[dict[str, Any]] = []
    alignment = max(min_chunk_bytes, CHUNK_512K)

    while True:
        free_bytes = int(runtime.mem_get_info().get("free") or 0)
        if free_bytes <= target_free_bytes + min_chunk_bytes:
            break

        request = min(int(chunk_bytes), free_bytes - target_free_bytes)
        request = _align_down(request, alignment)
        if request < min_chunk_bytes:
            break

        while request >= min_chunk_bytes:
            event, allocation = _try_cuda_malloc(runtime, request, f"{label}:{len(allocations)}")
            events.append(event)
            if allocation is not None:
                allocations.append(allocation)
                break
            request = _align_down(request // 2, alignment)

        if request < min_chunk_bytes:
            break

    return allocations, events


def _fill_with_probe_allocations_until_failure(
    runtime: CudaRuntime,
    *,
    probe_bytes: int,
    max_successes: int,
    label: str,
) -> tuple[list[CudaAllocation], list[dict[str, Any]]]:
    allocations: list[CudaAllocation] = []
    events: list[dict[str, Any]] = []

    for index in range(max_successes + 1):
        event, allocation = _try_cuda_malloc(runtime, probe_bytes, f"{label}:{index}")
        events.append(event)
        if allocation is None:
            break
        allocations.append(allocation)

    return allocations, events


def _query_region_blocks(fd: int, region_id: int, gpu_id: int = 0) -> dict[str, Any]:
    params = PhaseNQueryRegionBlocksParams()
    params.region_id = int(region_id)
    params.gpu_id = int(gpu_id)
    ret, errno_value = ioctl(fd, UVM_PHASE_N_QUERY_REGION_BLOCKS_IOCTL, params)
    sampled = int(params.sampled_block_count)
    blocks = []
    for index in range(min(sampled, UVM_PHASE_N_REGION_BLOCK_TRACE_MAX)):
        blocks.append({
            "start": int(params.block_starts[index]),
            "end": int(params.block_ends[index]),
            "bytes": int(params.block_bytes[index]),
            "states": int(params.block_states[index]),
            "status": int(params.block_status[index]),
        })
    return {
        "ret": int(ret),
        "errno": int(errno_value),
        "rmStatus": int(params.rmStatus),
        "ok": ret == 0 and int(params.rmStatus) == 0,
        "region_id": int(region_id),
        "base": int(params.base),
        "length": int(params.length),
        "pool": int(params.pool),
        "layer_id": int(params.layer_id),
        "expert_id": int(params.expert_id),
        "role": int(params.role),
        "flags": int(params.flags),
        "state": int(params.state),
        "priority": int(params.priority),
        "va_range_count": int(params.va_range_count),
        "logical_block_count": int(params.logical_block_count),
        "populated_block_count": int(params.populated_block_count),
        "resident_block_count": int(params.resident_block_count),
        "pending_block_count": int(params.pending_block_count),
        "materialized_block_count": int(params.materialized_block_count),
        "mixed_block_count": int(params.mixed_block_count),
        "resident_gpu_bytes": int(params.resident_gpu_bytes),
        "pending_gpu_bytes": int(params.pending_gpu_bytes),
        "first_block_start": int(params.first_block_start),
        "last_block_end": int(params.last_block_end),
        "sampled_block_count": sampled,
        "blocks": blocks,
    }


def _aggregate_region_blocks(fd: int, region_ids: list[int]) -> dict[str, Any]:
    queries = [_query_region_blocks(fd, region_id) for region_id in region_ids]
    ok_queries = [query for query in queries if query.get("ok")]
    unique_blocks = {
        (block["start"], block["end"])
        for query in ok_queries
        for block in query.get("blocks", [])
    }
    return {
        "region_count": len(region_ids),
        "ok_region_count": len(ok_queries),
        "va_range_count_sum": sum(int(query.get("va_range_count") or 0) for query in ok_queries),
        "logical_block_count_sum": sum(int(query.get("logical_block_count") or 0) for query in ok_queries),
        "populated_block_count_sum": sum(int(query.get("populated_block_count") or 0) for query in ok_queries),
        "resident_block_count_sum": sum(int(query.get("resident_block_count") or 0) for query in ok_queries),
        "resident_gpu_bytes_sum": sum(int(query.get("resident_gpu_bytes") or 0) for query in ok_queries),
        "sampled_block_count_sum": sum(int(query.get("sampled_block_count") or 0) for query in ok_queries),
        "unique_sampled_block_count": len(unique_blocks),
        "queries": queries,
    }


def _read_route_log() -> str:
    try:
        return ROUTE_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _route_log_counts(region_id: int | None = None) -> dict[str, Any]:
    text = _read_route_log()
    if region_id is not None:
        escaped = re.escape(str(int(region_id)))
        region_text = "\n".join(
            line for line in text.splitlines() if re.search(rf"\bregion_id={escaped}\b|\bid={escaped}\b", line)
        )
    else:
        region_text = text

    return {
        "available": bool(text),
        "bytes": len(text.encode("utf-8", errors="replace")),
        "line_count": len(text.splitlines()),
        "region_id": int(region_id) if region_id is not None else None,
        "phase_f_register_region": region_text.count("phase_f_register_region"),
        "phase_n_native_range_tag_split": region_text.count("phase_n_native_range_tag_split"),
        "phase_n_native_range_tag_split_base": region_text.count("split=base"),
        "phase_n_native_range_tag_split_end": region_text.count("split=end"),
        "phase_n_native_range_tag_attach": region_text.count("phase_n_native_range_tag_attach"),
        "phase_n_native_block_init": region_text.count("phase_n_native_block_init"),
        "phase_n_native_block_resident": region_text.count("phase_n_native_block_resident"),
        "phase_n_region_blocks_map": region_text.count("phase_n_region_blocks_map"),
    }


def _delta_counts(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, int) and isinstance(before_value, int):
            delta[key] = after_value - before_value
    return delta


def _close_allocations(allocations: list[CudaAllocation]) -> list[dict[str, Any]]:
    closed = []
    for allocation in reversed(allocations):
        ptr = allocation.ptr
        ret = allocation.close()
        closed.append({
            "label": allocation.label,
            "ptr": ptr,
            "bytes": allocation.byte_count,
            "free_ret": ret,
        })
    allocations.clear()
    return closed


def run(args: argparse.Namespace) -> dict[str, Any]:
    region_count = int(args.region_count)
    if region_count <= 0:
        raise ValueError("--region-count must be positive")

    release_bytes = region_count * CHUNK_512K
    managed_alloc_bytes = release_bytes + UVM_VA_BLOCK_SIZE
    base_region_id = int(args.base_region_id)
    probe_bytes = int(args.probe_bytes)
    runtime = CudaRuntime()
    fd = -1
    target: ManagedAllocation | None = None
    registered_region_ids: list[int] = []
    pre_filler: list[CudaAllocation] = []
    post_filler: list[CudaAllocation] = []
    pressure_probe_filler: list[CudaAllocation] = []
    successful_release_probes: list[CudaAllocation] = []

    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(args.device),
        "region_count": region_count,
        "region_bytes": CHUNK_512K,
        "probe_bytes": probe_bytes,
        "physical_chunk_bytes": PAGE_64K,
        "expected_chunks_per_region": CHUNK_512K // PAGE_64K,
        "expected_total_released_bytes": release_bytes,
        "managed_alloc_bytes": managed_alloc_bytes,
        "block_bytes": UVM_VA_BLOCK_SIZE,
        "query_region_blocks_params_size": ctypes.sizeof(PhaseNQueryRegionBlocksParams),
        "mem_info": {},
        "pressure": {},
        "regions": [],
        "release_steps": [],
        "route_log_initial": _route_log_counts(),
    }

    try:
        initialize_cuda_device(args.device)
        result["mem_info"]["initial"] = _mem_snapshot(runtime, "initial")
        gpu_uuid = gpu_uuid_for_device(args.device)

        pre_reserve = max(int(args.pre_managed_reserve_mib) * MIB, managed_alloc_bytes + 64 * MIB)
        pre_filler, pre_events = _fill_cuda_until_free_at_most(
            runtime,
            target_free_bytes=pre_reserve,
            chunk_bytes=int(args.filler_chunk_mib) * MIB,
            min_chunk_bytes=int(args.min_filler_chunk_kib) * 1024,
            label="pre_managed_filler",
        )
        result["pressure"]["pre_managed_filler"] = {
            "target_free_bytes": pre_reserve,
            "allocation_count": len(pre_filler),
            "allocated_bytes": sum(allocation.byte_count for allocation in pre_filler),
            "events": pre_events,
        }
        result["mem_info"]["after_pre_managed_filler"] = _mem_snapshot(runtime, "after_pre_managed_filler")

        target = ManagedAllocation(args.device, managed_alloc_bytes)
        raw_base = target.address
        base = ((raw_base + UVM_VA_BLOCK_SIZE - 1) // UVM_VA_BLOCK_SIZE) * UVM_VA_BLOCK_SIZE
        allocation_end = raw_base + managed_alloc_bytes
        if base + release_bytes > allocation_end:
            raise RuntimeError("aligned release range does not fit managed allocation")

        payloads = [_pattern(CHUNK_512K, 0x41 + index * 13) for index in range(region_count)]
        cuda_memcpy_host_to_device(base, b"".join(payloads))

        result.update({
            "managed_raw_base": raw_base,
            "release_base": base,
            "release_length": release_bytes,
            "release_base_2mb_aligned": bool(base % UVM_VA_BLOCK_SIZE == 0),
            "release_base_512k_aligned": bool(base % CHUNK_512K == 0),
        })
        result["mem_info"]["after_managed_alloc_and_payload_copy"] = _mem_snapshot(
            runtime, "after_managed_alloc_and_payload_copy"
        )

        fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(
            prefer_existing=True
        )
        result["uvm_fd"] = {
            "fd_source": fd_source,
            "init_ret": int(init_ret),
            "init_errno": int(init_errno),
            "init_rmStatus": int(init_rm_status),
        }
        if fd < 0:
            result["passed"] = False
            result["blocked_reason"] = "failed to open initialized UVM fd"
            return result

        migrate = native_migrate(fd, base, release_bytes, gpu_uuid)
        result["native_migrate_to_gpu"] = migrate
        result["mem_info"]["after_native_migrate_to_gpu"] = _mem_snapshot(runtime, "after_native_migrate_to_gpu")
        result["residency_before_pressure_whole"] = _query_residency(
            fd, gpu_uuid, base, release_bytes
        )
        if not migrate.get("ok"):
            result["passed"] = False
            result["blocked_reason"] = "native migrate to GPU failed under pre-managed pressure"
            return result

        post_target_free = int(args.post_migrate_free_target_mib) * MIB
        if args.skip_post_migrate_filler:
            post_events = []
        else:
            post_filler, post_events = _fill_cuda_until_free_at_most(
                runtime,
                target_free_bytes=post_target_free,
                chunk_bytes=int(args.filler_chunk_mib) * MIB,
                min_chunk_bytes=int(args.min_filler_chunk_kib) * 1024,
                label="post_migrate_filler",
            )

        baseline_probe_event: dict[str, Any] | None = None
        if args.skip_pressure_probe_filler:
            if args.skip_baseline_probe:
                pressure_probe_events = []
            else:
                baseline_probe_event, baseline_allocation = _try_cuda_malloc(
                    runtime, probe_bytes, "baseline_probe_after_migrate"
                )
                if baseline_allocation is not None:
                    baseline_probe_event["freed_immediately"] = True
                    baseline_probe_event["free_ret"] = baseline_allocation.close()
                pressure_probe_events = [baseline_probe_event]
        else:
            pressure_probe_filler, pressure_probe_events = _fill_with_probe_allocations_until_failure(
                runtime,
                probe_bytes=probe_bytes,
                max_successes=int(args.max_probe_fill_count),
                label="pressure_probe_filler",
            )
        result["pressure"]["post_migrate_filler"] = {
            "skipped": bool(args.skip_post_migrate_filler),
            "target_free_bytes": post_target_free,
            "allocation_count": len(post_filler),
            "allocated_bytes": sum(allocation.byte_count for allocation in post_filler),
            "events": post_events,
        }
        result["pressure"]["probe_filler"] = {
            "skipped": bool(args.skip_pressure_probe_filler),
            "allocation_count": len(pressure_probe_filler),
            "allocated_bytes": sum(allocation.byte_count for allocation in pressure_probe_filler),
            "events": pressure_probe_events,
            "ended_with_probe_failure": bool(pressure_probe_events and not pressure_probe_events[-1]["ok"]),
            "baseline_probe_event": baseline_probe_event,
        }
        result["mem_info"]["after_pressure_saturation"] = _mem_snapshot(runtime, "after_pressure_saturation")
        result["residency_after_pressure_whole"] = _query_residency(fd, gpu_uuid, base, release_bytes)

        cumulative_released = 0
        for index in range(region_count):
            region_base = base + index * CHUNK_512K
            region_id = base_region_id + index
            route_before_register = _route_log_counts(region_id)
            aggregate_before_register = _aggregate_region_blocks(fd, registered_region_ids)

            register = _register_expert_region(
                fd,
                region_id=region_id,
                base=region_base,
                byte_count=CHUNK_512K,
            )
            registered_region_ids.append(region_id)
            query_after_register = _query_region_blocks(fd, region_id)
            aggregate_after_register = _aggregate_region_blocks(fd, registered_region_ids)
            route_after_register = _route_log_counts(region_id)

            before_region = _query_residency(fd, gpu_uuid, region_base, CHUNK_512K)
            before_whole = _query_residency(fd, gpu_uuid, base, release_bytes)
            route_before_release = _route_log_counts(region_id)
            release = _release_region_subchunk_64k(fd, gpu_uuid, region_id)
            cumulative_released += int(release.get("released_bytes") or 0)
            after_region = _query_residency(fd, gpu_uuid, region_base, CHUNK_512K)
            after_whole = _query_residency(fd, gpu_uuid, base, release_bytes)
            query_after_release = _query_region_blocks(fd, region_id)
            aggregate_after_release = _aggregate_region_blocks(fd, registered_region_ids)
            route_after_release = _route_log_counts(region_id)

            probe_event, probe_allocation = _try_cuda_malloc(
                runtime, probe_bytes, f"after_release_probe:{index}"
            )
            if probe_allocation is not None:
                successful_release_probes.append(probe_allocation)
            after_probe_region = _query_residency(fd, gpu_uuid, region_base, CHUNK_512K)
            after_probe_whole = _query_residency(fd, gpu_uuid, base, release_bytes)

            cpu_payload_ok = ctypes.string_at(region_base, CHUNK_512K) == payloads[index]
            step = {
                "index": index,
                "region_id": region_id,
                "base": region_base,
                "register": register,
                "query_after_register": query_after_register,
                "query_after_release": query_after_release,
                "aggregate_before_register": aggregate_before_register,
                "aggregate_after_register": aggregate_after_register,
                "aggregate_after_release": aggregate_after_release,
                "route_delta_register": _delta_counts(route_after_register, route_before_register),
                "route_delta_release": _delta_counts(route_after_release, route_before_release),
                "before_region": before_region,
                "before_whole": before_whole,
                "release": release,
                "after_region": after_region,
                "after_whole": after_whole,
                "cuda_malloc_512k_after_release": probe_event,
                "after_probe_region": after_probe_region,
                "after_probe_whole": after_probe_whole,
                "cumulative_released_bytes": cumulative_released,
                "expected_cumulative_released_bytes": (index + 1) * CHUNK_512K,
                "cpu_payload_ok": cpu_payload_ok,
                "checks": {
                    "register_ok": register.get("ok") is True,
                    "query_after_register_ok": query_after_register.get("ok") is True,
                    "release_ioctl_ok": release.get("ok") is True,
                    "release_attempted_512k": release.get("attempted_bytes") == CHUNK_512K,
                    "release_released_512k": release.get("released_bytes") == CHUNK_512K,
                    "release_no_failures": release.get("failure_blocks") == 0,
                    "after_region_gpu_zero": after_region.get("gpu_bytes") == 0,
                    "after_region_cpu_512k": after_region.get("cpu_bytes") == CHUNK_512K,
                    "after_whole_gpu_expected": (
                        after_whole.get("gpu_bytes")
                        == release_bytes - (index + 1) * CHUNK_512K
                    ),
                    "cumulative_released_expected": (
                        cumulative_released == (index + 1) * CHUNK_512K
                    ),
                    "cpu_payload_ok": cpu_payload_ok,
                },
            }
            step["release_path_passed"] = all(bool(value) for value in step["checks"].values())
            result["release_steps"].append(step)
            result["regions"].append({
                "index": index,
                "region_id": region_id,
                "base": region_base,
                "length": CHUNK_512K,
            })
            if args.stop_on_failure and not step["release_path_passed"]:
                break

        result["residency_after_release_whole"] = _query_residency(fd, gpu_uuid, base, release_bytes)
        result["aggregate_after_all_releases"] = _aggregate_region_blocks(fd, registered_region_ids)
        result["route_log_final"] = _route_log_counts()
        result["mem_info"]["after_all_releases_and_probes"] = _mem_snapshot(
            runtime, "after_all_releases_and_probes"
        )
        result["total_released_bytes"] = cumulative_released
        result["release_success_count"] = sum(
            1 for step in result["release_steps"] if step.get("release_path_passed")
        )
        result["cuda_malloc_probe_success_count"] = sum(
            1
            for step in result["release_steps"]
            if step.get("cuda_malloc_512k_after_release", {}).get("ok")
        )
        result["cuda_malloc_probe_success_indices"] = [
            int(step["index"])
            for step in result["release_steps"]
            if step.get("cuda_malloc_512k_after_release", {}).get("ok")
        ]
        pressure_baseline_required = not (
            args.skip_pressure_probe_filler and args.skip_baseline_probe
        )
        pressure_baseline_satisfied = (
            bool(result["pressure"]["probe_filler"]["ended_with_probe_failure"])
            if pressure_baseline_required
            else True
        )
        result["pressure_baseline_required"] = pressure_baseline_required
        result["pressure_baseline_satisfied"] = pressure_baseline_satisfied
        result["passed"] = (
            len(result["release_steps"]) == region_count
            and result["release_success_count"] == region_count
            and cumulative_released == release_bytes
            and pressure_baseline_satisfied
        )
        result["cuda_malloc_reuse_every_release"] = (
            result["cuda_malloc_probe_success_count"] == len(result["release_steps"])
        )
        if not result["passed"]:
            result["failure_reason"] = "one or more release path checks failed or pressure baseline did not fail"
        return result
    finally:
        if fd >= 0:
            for region_id in reversed(registered_region_ids):
                try:
                    _unregister_region(fd, region_id)
                except Exception:
                    pass
            os.close(fd)
        if target is not None:
            target.close()
        result["cleanup"] = {
            "successful_release_probes": _close_allocations(successful_release_probes),
            "pressure_probe_filler": _close_allocations(pressure_probe_filler),
            "post_filler": _close_allocations(post_filler),
            "pre_filler": _close_allocations(pre_filler),
        }
        result["mem_info"]["after_cleanup"] = _mem_snapshot(runtime, "after_cleanup")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--region-count", type=int, default=8)
    parser.add_argument("--base-region-id", type=lambda value: int(value, 0), default=0x5120002000)
    parser.add_argument("--probe-bytes", type=int, default=CHUNK_512K)
    parser.add_argument("--pre-managed-reserve-mib", type=int, default=1024)
    parser.add_argument("--post-migrate-free-target-mib", type=int, default=8)
    parser.add_argument("--filler-chunk-mib", type=int, default=512)
    parser.add_argument("--min-filler-chunk-kib", type=int, default=512)
    parser.add_argument("--max-probe-fill-count", type=int, default=4096)
    parser.add_argument("--skip-post-migrate-filler", action="store_true")
    parser.add_argument("--skip-pressure-probe-filler", action="store_true")
    parser.add_argument("--skip-baseline-probe", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    try:
        result = run(args)
    except Exception as exc:
        result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "passed": False,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")

    if result.get("passed") or args.allow_fail:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
