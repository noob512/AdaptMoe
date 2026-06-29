#!/usr/bin/env python3
"""Smoke-test the experimental UVM MoE slot pool metadata/query ioctl."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import subprocess
import sys
import uuid
from pathlib import Path

UVM_INITIALIZE_IOCTL = 0x30000001
UVM_MOE_SLOT_REMAP_BATCH_IOCTL = 91
UVM_MOE_SLOT_POOL_QUERY_IOCTL = 92
UVM_MOE_SLOT_POOL_READBACK_IOCTL = 93
UVM_MOE_SLOT_REMAP_MAX_ENTRIES = 128
UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES = 128

MODE_DRY_RUN = 0
MODE_VALIDATE = 1
MODE_COPY_ONLY = 3
MODE_GPU_COPY_ONLY = 4

ROLE_GATE_UP = 3
ROLE_DOWN = 4
ENTRY_OK = 0
ENTRY_SLOT_CONFLICT = 3
SLOT_STATE_READY = 3
BACKING_SYSMEM = 1
BACKING_VIDMEM = 3

QUERY_FLAG_FILTER_LAYER = 1 << 0
QUERY_FLAG_FILTER_SLOT = 1 << 1
QUERY_FLAG_RESET = 1 << 31


class UvmInitializeParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class UvmUuid(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)]


class MoeSlotRemapEntry(ctypes.Structure):
    _fields_ = [
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("va_base", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("slot_offset", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
    ]


class MoeSlotRemapBatchParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", UvmUuid),
        ("validate_managed", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("entries", MoeSlotRemapEntry * UVM_MOE_SLOT_REMAP_MAX_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class MoeSlotPoolQueryEntry(ctypes.Structure):
    _fields_ = [
        ("layer_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("owner_expert_id", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("range_count", ctypes.c_uint32),
        ("last_role", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("last_status", ctypes.c_uint32),
        ("epoch", ctypes.c_uint64),
        ("batch_id", ctypes.c_uint64),
        ("first_va_base", ctypes.c_uint64),
        ("last_va_base", ctypes.c_uint64),
        ("last_bytes", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),
        ("slot_bytes", ctypes.c_uint64),
        ("slot_physical_bytes", ctypes.c_uint64),
        ("copy_count", ctypes.c_uint64),
        ("copy_bytes", ctypes.c_uint64),
        ("copy_failed_count", ctypes.c_uint64),
        ("pte_remap_bytes", ctypes.c_uint64),
        ("pte_remap_count", ctypes.c_uint64),
        ("pte_clear_count", ctypes.c_uint64),
        ("pte_va_base", ctypes.c_uint64),
        ("pte_bytes", ctypes.c_uint64),
        ("pte_block_unmap_count", ctypes.c_uint64),
        ("pte_block_tracker_wait_count", ctypes.c_uint64),
        ("pte_rollback_count", ctypes.c_uint64),
        ("pte_rollback_bytes", ctypes.c_uint64),
        ("pte_failure_count", ctypes.c_uint64),
        ("pte_fault_clear_count", ctypes.c_uint64),
        ("pte_migrate_clear_count", ctypes.c_uint64),
        ("pte_unmap_clear_count", ctypes.c_uint64),
        ("pte_split_clear_count", ctypes.c_uint64),
        ("pte_release_clear_count", ctypes.c_uint64),
        ("pte_evict_clear_count", ctypes.c_uint64),
        ("pte_dangerous_op_reject_count", ctypes.c_uint64),
        ("backing_kind", ctypes.c_uint32),
        ("copy_status", ctypes.c_uint32),
        ("pte_mapped", ctypes.c_uint32),
        ("pte_status", ctypes.c_uint32),
        ("pte_mode", ctypes.c_uint32),
        ("pte_va_block_tracked", ctypes.c_uint32),
    ]


class MoeSlotPoolQueryParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("max_entries", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("total_slots", ctypes.c_uint32),
        ("returned_slots", ctypes.c_uint32),
        ("active_slots", ctypes.c_uint32),
        ("backed_slots", ctypes.c_uint32),
        ("current_epoch", ctypes.c_uint64),
        ("validated_entries", ctypes.c_uint64),
        ("overwrite_count", ctypes.c_uint64),
        ("range_update_count", ctypes.c_uint64),
        ("pool_full_count", ctypes.c_uint64),
        ("slot_backing_bytes", ctypes.c_uint64),
        ("copy_count", ctypes.c_uint64),
        ("copy_bytes", ctypes.c_uint64),
        ("copy_failed_count", ctypes.c_uint64),
        ("vidmem_evict_count", ctypes.c_uint64),
        ("pte_remap_count", ctypes.c_uint64),
        ("pte_remap_bytes", ctypes.c_uint64),
        ("pte_clear_count", ctypes.c_uint64),
        ("pte_block_unmap_count", ctypes.c_uint64),
        ("pte_block_tracker_wait_count", ctypes.c_uint64),
        ("pte_rollback_count", ctypes.c_uint64),
        ("pte_rollback_bytes", ctypes.c_uint64),
        ("pte_failure_count", ctypes.c_uint64),
        ("pte_fault_clear_count", ctypes.c_uint64),
        ("pte_migrate_clear_count", ctypes.c_uint64),
        ("pte_unmap_clear_count", ctypes.c_uint64),
        ("pte_split_clear_count", ctypes.c_uint64),
        ("pte_release_clear_count", ctypes.c_uint64),
        ("pte_evict_clear_count", ctypes.c_uint64),
        ("pte_dangerous_op_reject_count", ctypes.c_uint64),
        ("pte_residual_va_block_count", ctypes.c_uint64),
        ("pte_residual_page_count", ctypes.c_uint64),
        ("pte_residual_gpu_mask", ctypes.c_uint64),
        ("entries", MoeSlotPoolQueryEntry * UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class MoeSlotPoolReadbackParams(ctypes.Structure):
    _fields_ = [
        ("layer_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("backing_kind", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("slot_offset", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("buffer", ctypes.c_uint64),
        ("copied_bytes", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.ioctl.restype = ctypes.c_int


def ioctl(fd: int, request: int, params: ctypes.Structure) -> tuple[int, int]:
    ctypes.set_errno(0)
    ret = int(LIBC.ioctl(ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.byref(params)))
    return ret, int(ctypes.get_errno())


def gpu_uuid_from_nvidia_smi(device: int) -> UvmUuid:
    try:
        timeout = max(int(os.environ.get("UVM_MOE_NVIDIA_SMI_TIMEOUT", "5")), 1)
    except ValueError:
        timeout = 5

    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"nvidia-smi timed out after {timeout}s while querying GPU UUID; "
            "the node may have wedged NVIDIA/UVM state"
        ) from exc
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("failed to query GPU UUID with nvidia-smi") from exc

    first = output.strip().splitlines()[0].strip()
    if first.startswith("GPU-"):
        first = first[4:]
    parsed = uuid.UUID(first)
    result = UvmUuid()
    for index, byte in enumerate(parsed.bytes):
        result.uuid[index] = byte
    return result


def gpu_uuid_from_cuda_driver(device: int) -> UvmUuid:
    libname = ctypes.util.find_library("cuda") or "libcuda.so.1"
    libcuda = ctypes.CDLL(libname)
    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    cu_device_get = libcuda.cuDeviceGet
    cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cu_device_get.restype = ctypes.c_int
    cu_device_get_uuid = libcuda.cuDeviceGetUuid
    cu_device_get_uuid.argtypes = [ctypes.POINTER(ctypes.c_uint8 * 16), ctypes.c_int]
    cu_device_get_uuid.restype = ctypes.c_int

    ret = int(cu_init(0))
    if ret != 0:
        raise RuntimeError(f"cuInit failed: {ret}")
    cu_device = ctypes.c_int()
    ret = int(cu_device_get(ctypes.byref(cu_device), int(device)))
    if ret != 0:
        raise RuntimeError(f"cuDeviceGet({device}) failed: {ret}")
    uuid_bytes_array = (ctypes.c_uint8 * 16)()
    ret = int(cu_device_get_uuid(ctypes.byref(uuid_bytes_array), cu_device))
    if ret != 0:
        raise RuntimeError(f"cuDeviceGetUuid({device}) failed: {ret}")
    uuid_bytes = bytes(uuid_bytes_array)
    if len(uuid_bytes) != 16:
        raise RuntimeError(f"unexpected CUDA UUID size: {len(uuid_bytes)}")
    result = UvmUuid()
    for index, byte in enumerate(uuid_bytes):
        result.uuid[index] = byte
    return result


def gpu_uuid_for_device(device: int) -> UvmUuid:
    try:
        return gpu_uuid_from_cuda_driver(device)
    except Exception:
        if os.environ.get("UVM_MOE_ALLOW_NVIDIA_SMI_UUID", "0").lower() in (
            "1", "true", "yes", "on",
        ):
            return gpu_uuid_from_nvidia_smi(device)
        raise


def initialize_cuda_device(device: int) -> None:
    libname = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(libname)
    cuda_set_device = cudart.cudaSetDevice
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = ctypes.c_int
    cuda_free = cudart.cudaFree
    cuda_free.argtypes = [ctypes.c_void_p]
    cuda_free.restype = ctypes.c_int

    ret = int(cuda_set_device(ctypes.c_int(device)))
    if ret != 0:
        raise RuntimeError(f"cudaSetDevice({device}) failed: {ret}")
    ret = int(cuda_free(None))
    if ret != 0:
        raise RuntimeError(f"cudaFree(0) failed: {ret}")


def entry_to_dict(entry: MoeSlotPoolQueryEntry) -> dict[str, int]:
    return {
        "layer_id": int(entry.layer_id),
        "slot_id": int(entry.slot_id),
        "owner_expert_id": int(entry.owner_expert_id),
        "state": int(entry.state),
        "range_count": int(entry.range_count),
        "last_role": int(entry.last_role),
        "flags": int(entry.flags),
        "last_status": int(entry.last_status),
        "epoch": int(entry.epoch),
        "batch_id": int(entry.batch_id),
        "first_va_base": int(entry.first_va_base),
        "last_va_base": int(entry.last_va_base),
        "last_bytes": int(entry.last_bytes),
        "total_bytes": int(entry.total_bytes),
        "slot_bytes": int(entry.slot_bytes),
        "slot_physical_bytes": int(entry.slot_physical_bytes),
        "copy_count": int(entry.copy_count),
        "copy_bytes": int(entry.copy_bytes),
        "copy_failed_count": int(entry.copy_failed_count),
        "pte_remap_bytes": int(entry.pte_remap_bytes),
        "pte_remap_count": int(entry.pte_remap_count),
        "pte_clear_count": int(entry.pte_clear_count),
        "pte_va_base": int(entry.pte_va_base),
        "pte_bytes": int(entry.pte_bytes),
        "pte_block_unmap_count": int(entry.pte_block_unmap_count),
        "pte_block_tracker_wait_count": int(entry.pte_block_tracker_wait_count),
        "pte_rollback_count": int(entry.pte_rollback_count),
        "pte_rollback_bytes": int(entry.pte_rollback_bytes),
        "pte_failure_count": int(entry.pte_failure_count),
        "pte_fault_clear_count": int(entry.pte_fault_clear_count),
        "pte_migrate_clear_count": int(entry.pte_migrate_clear_count),
        "pte_unmap_clear_count": int(entry.pte_unmap_clear_count),
        "pte_split_clear_count": int(entry.pte_split_clear_count),
        "pte_release_clear_count": int(entry.pte_release_clear_count),
        "pte_evict_clear_count": int(entry.pte_evict_clear_count),
        "pte_dangerous_op_reject_count": int(entry.pte_dangerous_op_reject_count),
        "backing_kind": int(entry.backing_kind),
        "copy_status": int(entry.copy_status),
        "pte_mapped": int(entry.pte_mapped),
        "pte_status": int(entry.pte_status),
        "pte_mode": int(entry.pte_mode),
        "pte_va_block_tracked": int(entry.pte_va_block_tracked),
    }


def query_pool(
    fd: int,
    *,
    layer_id: int | None = None,
    slot_id: int | None = None,
    reset: bool = False,
    max_entries: int = 16,
) -> dict[str, object]:
    params = MoeSlotPoolQueryParams()
    params.max_entries = max_entries
    if layer_id is not None:
        params.flags |= QUERY_FLAG_FILTER_LAYER
        params.layer_id = layer_id
    if slot_id is not None:
        params.flags |= QUERY_FLAG_FILTER_SLOT
        params.slot_id = slot_id
    if reset:
        params.flags |= QUERY_FLAG_RESET

    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_POOL_QUERY_IOCTL, params)
    entries = [
        entry_to_dict(params.entries[index])
        for index in range(int(params.returned_slots))
    ]
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "ok": ret == 0 and int(params.rmStatus) == 0,
        "total_slots": int(params.total_slots),
        "returned_slots": int(params.returned_slots),
        "active_slots": int(params.active_slots),
        "backed_slots": int(params.backed_slots),
        "current_epoch": int(params.current_epoch),
        "validated_entries": int(params.validated_entries),
        "overwrite_count": int(params.overwrite_count),
        "range_update_count": int(params.range_update_count),
        "pool_full_count": int(params.pool_full_count),
        "slot_backing_bytes": int(params.slot_backing_bytes),
        "copy_count": int(params.copy_count),
        "copy_bytes": int(params.copy_bytes),
        "copy_failed_count": int(params.copy_failed_count),
        "vidmem_evict_count": int(params.vidmem_evict_count),
        "pte_remap_count": int(params.pte_remap_count),
        "pte_remap_bytes": int(params.pte_remap_bytes),
        "pte_clear_count": int(params.pte_clear_count),
        "pte_block_unmap_count": int(params.pte_block_unmap_count),
        "pte_block_tracker_wait_count": int(params.pte_block_tracker_wait_count),
        "pte_rollback_count": int(params.pte_rollback_count),
        "pte_rollback_bytes": int(params.pte_rollback_bytes),
        "pte_failure_count": int(params.pte_failure_count),
        "pte_fault_clear_count": int(params.pte_fault_clear_count),
        "pte_migrate_clear_count": int(params.pte_migrate_clear_count),
        "pte_unmap_clear_count": int(params.pte_unmap_clear_count),
        "pte_split_clear_count": int(params.pte_split_clear_count),
        "pte_release_clear_count": int(params.pte_release_clear_count),
        "pte_evict_clear_count": int(params.pte_evict_clear_count),
        "pte_dangerous_op_reject_count": int(params.pte_dangerous_op_reject_count),
        "pte_residual_va_block_count": int(params.pte_residual_va_block_count),
        "pte_residual_page_count": int(params.pte_residual_page_count),
        "pte_residual_gpu_mask": int(params.pte_residual_gpu_mask),
        "entries": entries,
    }


def readback_slot(
    fd: int,
    *,
    layer_id: int,
    slot_id: int,
    byte_count: int,
    slot_offset: int = 0,
) -> tuple[dict[str, object], bytes]:
    params = MoeSlotPoolReadbackParams()
    params.layer_id = layer_id
    params.slot_id = slot_id
    params.slot_offset = slot_offset
    params.bytes = byte_count
    buffer = ctypes.create_string_buffer(byte_count)
    params.buffer = ctypes.addressof(buffer)

    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_POOL_READBACK_IOCTL, params)
    copied = int(params.copied_bytes)
    return (
        {
            "ret": ret,
            "errno": errno_value,
            "rmStatus": int(params.rmStatus),
            "ok": ret == 0 and int(params.rmStatus) == 0,
            "backing_kind": int(params.backing_kind),
            "requested_bytes": byte_count,
            "copied_bytes": copied,
        },
        bytes(buffer.raw[:copied]),
    )


def submit_batch(
    fd: int,
    *,
    mode: int,
    gpu_uuid: UvmUuid | None = None,
    layer_id: int,
    expert_id: int,
    slot_id: int,
    base: int,
    page_size: int,
    ranges: int,
    slot_offset_stride: int | None = None,
) -> dict[str, object]:
    params = MoeSlotRemapBatchParams()
    params.mode = mode
    params.count = ranges
    if gpu_uuid is not None:
        params.gpu_uuid = gpu_uuid
    params.validate_managed = 0

    for index in range(ranges):
        params.entries[index].layer_id = layer_id
        params.entries[index].expert_id = expert_id
        params.entries[index].slot_id = slot_id
        params.entries[index].role = ROLE_GATE_UP if index == 0 else ROLE_DOWN
        params.entries[index].va_base = base + index * page_size
        params.entries[index].bytes = page_size
        params.entries[index].slot_offset = (
            index * page_size if slot_offset_stride is None
            else index * slot_offset_stride
        )
        params.entries[index].flags = 0x10 + index

    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
    entry_statuses = [int(params.entries[index].status) for index in range(ranges)]
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "conflict_count": int(params.conflict_count),
        "requested_bytes": int(params.requested_bytes),
        "accepted_bytes": int(params.accepted_bytes),
        "entry_statuses": entry_statuses,
    }


def check(condition: bool, checks: dict[str, bool], name: str) -> None:
    checks[name] = bool(condition)


def check_copy_backing_state(
    dump: dict[str, object],
    checks: dict[str, bool],
    *,
    prefix: str,
    expected_backing: int,
) -> None:
    entries = dump["entries"]  # type: ignore[index]
    backed_entries = [
        entry for entry in entries
        if isinstance(entry, dict)
        and int(entry.get("backing_kind") or 0) == expected_backing
    ]
    check(int(dump["active_slots"]) > 0, checks, f"{prefix}_active_slots_nonzero")
    check(int(dump["backed_slots"]) > 0, checks, f"{prefix}_backed_slots_nonzero")
    check(int(dump["slot_backing_bytes"]) > 0, checks, f"{prefix}_backing_bytes_nonzero")
    check(int(dump["copy_count"]) > 0, checks, f"{prefix}_copy_count_nonzero")
    check(int(dump["copy_bytes"]) > 0, checks, f"{prefix}_copy_bytes_nonzero")
    check(int(dump["copy_failed_count"]) == 0, checks, f"{prefix}_copy_failures_zero")
    check(int(dump["pte_failure_count"]) == 0, checks, f"{prefix}_pte_failures_zero")
    check(
        int(dump["pte_dangerous_op_reject_count"]) == 0,
        checks,
        f"{prefix}_dangerous_op_rejects_zero",
    )
    check(
        len(backed_entries) == int(dump["returned_slots"]),
        checks,
        f"{prefix}_returned_entries_backed_expected_kind",
    )


def check_slot_pte_state(dump: dict[str, object], checks: dict[str, bool]) -> None:
    entries = dump["entries"]  # type: ignore[index]
    mapped_entries = [
        entry for entry in entries
        if isinstance(entry, dict) and int(entry.get("pte_mapped") or 0) != 0
    ]
    lifecycle_cleared = (
        len(mapped_entries) == 0
        and int(dump["pte_clear_count"]) > 0
        and (
            int(dump["pte_unmap_clear_count"]) > 0
            or int(dump["pte_split_clear_count"]) > 0
            or int(dump["pte_release_clear_count"]) > 0
        )
    )
    vidmem_entries = [
        entry for entry in entries
        if isinstance(entry, dict)
        and int(entry.get("backing_kind") or 0) == BACKING_VIDMEM
    ]

    check(int(dump["active_slots"]) > 0, checks, "slot_pte_active_slots_nonzero")
    check(int(dump["backed_slots"]) > 0, checks, "slot_pte_backed_slots_nonzero")
    check(int(dump["slot_backing_bytes"]) > 0, checks, "slot_pte_backing_bytes_nonzero")
    check(int(dump["copy_count"]) > 0, checks, "slot_pte_copy_count_nonzero")
    check(int(dump["copy_failed_count"]) == 0, checks, "slot_pte_copy_failures_zero")
    check(int(dump["pte_remap_count"]) > 0, checks, "slot_pte_remap_count_nonzero")
    check(int(dump["pte_remap_bytes"]) > 0, checks, "slot_pte_remap_bytes_nonzero")
    check(int(dump["pte_failure_count"]) == 0, checks, "slot_pte_failures_zero")
    check(
        int(dump["pte_dangerous_op_reject_count"]) == 0,
        checks,
        "slot_pte_dangerous_op_rejects_zero",
    )
    check(
        len(mapped_entries) > 0 or lifecycle_cleared,
        checks,
        "slot_pte_mapped_or_lifecycle_cleared",
    )
    check(len(vidmem_entries) == int(dump["returned_slots"]), checks, "slot_pte_entries_vidmem_backed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-id", type=int, default=7)
    parser.add_argument("--slot-id", type=int, default=3)
    parser.add_argument("--expert-id", type=int, default=101)
    parser.add_argument("--replacement-expert-id", type=int, default=202)
    parser.add_argument("--base", type=lambda value: int(value, 0), default=0x1000000000)
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument("--query-layer-id", type=int)
    parser.add_argument("--query-slot-id", type=int)
    parser.add_argument("--max-entries", type=int, default=32)
    parser.add_argument("--reset-query", action="store_true")
    parser.add_argument("--expect-metadata-only-state", action="store_true")
    parser.add_argument("--expect-copy-only-state", action="store_true")
    parser.add_argument("--expect-gpu-copy-only-state", action="store_true")
    parser.add_argument("--expect-slot-pte-state", action="store_true")
    parser.add_argument("--include-gpu-copy-only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    expectation_count = sum(
        bool(value)
        for value in (
            args.expect_metadata_only_state,
            args.expect_copy_only_state,
            args.expect_gpu_copy_only_state,
            args.expect_slot_pte_state,
        )
    )
    if expectation_count > 1:
        raise SystemExit(
            "--expect-metadata-only-state, --expect-copy-only-state, and "
            "--expect-gpu-copy-only-state, and --expect-slot-pte-state are mutually exclusive"
        )

    page_size = os.sysconf("SC_PAGE_SIZE")
    base = args.base - (args.base % page_size)
    replacement_base = base + 16 * page_size

    result: dict[str, object] = {
        "ioctls": {
            "remap_batch": UVM_MOE_SLOT_REMAP_BATCH_IOCTL,
            "slot_pool_query": UVM_MOE_SLOT_POOL_QUERY_IOCTL,
            "slot_pool_readback": UVM_MOE_SLOT_POOL_READBACK_IOCTL,
        },
        "page_size": page_size,
        "checks": {},
    }
    checks: dict[str, bool] = result["checks"]  # type: ignore[assignment]

    fd = -1
    try:
        fd = os.open("/dev/nvidia-uvm", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        init_params = UvmInitializeParams()
        init_ret, init_errno = ioctl(fd, UVM_INITIALIZE_IOCTL, init_params)
        result["initialize"] = {
            "ret": init_ret,
            "errno": init_errno,
            "rmStatus": int(init_params.rmStatus),
        }
        check(init_ret == 0 and int(init_params.rmStatus) == 0, checks, "uvm_initialize_ok")
        if not checks["uvm_initialize_ok"]:
            result["passed"] = False
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1

        if args.dump_only:
            dump = query_pool(
                fd,
                layer_id=args.query_layer_id,
                slot_id=args.query_slot_id,
                reset=args.reset_query,
                max_entries=args.max_entries,
            )
            result["query"] = dump
            check(bool(dump["ok"]), checks, "dump_query_ok")
            if args.expect_metadata_only_state:
                check(
                    int(dump["backed_slots"]) == 0,
                    checks,
                    "metadata_only_backed_slots_zero",
                )
                check(
                    int(dump["slot_backing_bytes"]) == 0,
                    checks,
                    "metadata_only_backing_bytes_zero",
                )
                check(
                    int(dump["copy_count"]) == 0,
                    checks,
                    "metadata_only_copy_count_zero",
                )
                check(
                    int(dump["copy_bytes"]) == 0,
                    checks,
                    "metadata_only_copy_bytes_zero",
                )
                check(
                    int(dump["copy_failed_count"]) == 0,
                    checks,
                    "metadata_only_copy_failures_zero",
                )
                check(
                    int(dump["pte_failure_count"]) == 0,
                    checks,
                    "metadata_only_pte_failures_zero",
                )
                check(
                    int(dump["pte_dangerous_op_reject_count"]) == 0,
                    checks,
                    "metadata_only_dangerous_op_rejects_zero",
                )
            if args.expect_copy_only_state:
                check_copy_backing_state(
                    dump,
                    checks,
                    prefix="copy_only",
                    expected_backing=BACKING_SYSMEM,
                )
            if args.expect_gpu_copy_only_state:
                check_copy_backing_state(
                    dump,
                    checks,
                    prefix="gpu_copy_only",
                    expected_backing=BACKING_VIDMEM,
                )
            if args.expect_slot_pte_state:
                check_slot_pte_state(dump, checks)
            result["passed"] = all(checks.values())
            if args.out_json:
                args.out_json.parent.mkdir(parents=True, exist_ok=True)
                args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 1

        reset = query_pool(fd, reset=True)
        result["reset_query"] = reset
        check(bool(reset["ok"]), checks, "reset_query_ok")
        check(int(reset["active_slots"]) == 0, checks, "reset_clears_active_slots")

        overlap_conflict = submit_batch(
            fd,
            mode=MODE_VALIDATE,
            layer_id=args.layer_id,
            expert_id=args.expert_id,
            slot_id=args.slot_id,
            base=base,
            page_size=page_size,
            ranges=2,
            slot_offset_stride=0,
        )
        reset_after_overlap = query_pool(fd, reset=True)
        result["overlap_conflict_submit"] = overlap_conflict
        result["reset_after_overlap_conflict_query"] = reset_after_overlap
        check(overlap_conflict["success_count"] == 1, checks, "overlap_conflict_keeps_first_range")
        check(overlap_conflict["failure_count"] == 1, checks, "overlap_conflict_rejects_second_range")
        check(overlap_conflict["conflict_count"] == 1, checks, "overlap_conflict_count_increments")
        check(
            overlap_conflict["entry_statuses"] == [ENTRY_OK, ENTRY_SLOT_CONFLICT],
            checks,
            "overlap_conflict_entry_statuses",
        )
        check(bool(reset_after_overlap["ok"]), checks, "reset_after_overlap_conflict_ok")
        check(
            int(reset_after_overlap["active_slots"]) == 0,
            checks,
            "reset_after_overlap_conflict_clears_active_slots",
        )

        dry_run = submit_batch(
            fd,
            mode=MODE_DRY_RUN,
            layer_id=args.layer_id,
            expert_id=args.expert_id,
            slot_id=args.slot_id,
            base=base,
            page_size=page_size,
            ranges=2,
        )
        after_dry_run = query_pool(fd, layer_id=args.layer_id, slot_id=args.slot_id)
        result["dry_run_submit"] = dry_run
        result["after_dry_run_query"] = after_dry_run
        check(dry_run["ret"] == 0 and dry_run["success_count"] == 2, checks, "dry_run_submit_ok")
        check(int(after_dry_run["total_slots"]) == 0, checks, "dry_run_does_not_mutate_pool")

        validate = submit_batch(
            fd,
            mode=MODE_VALIDATE,
            layer_id=args.layer_id,
            expert_id=args.expert_id,
            slot_id=args.slot_id,
            base=base,
            page_size=page_size,
            ranges=2,
        )
        first_query = query_pool(fd, layer_id=args.layer_id, slot_id=args.slot_id)
        result["validate_submit"] = validate
        result["first_query"] = first_query
        first_entries = first_query["entries"]  # type: ignore[index]
        first_entry = first_entries[0] if first_entries else {}
        check(validate["ret"] == 0 and validate["success_count"] == 2, checks, "validate_submit_ok")
        check(int(first_query["total_slots"]) == 1, checks, "validate_creates_one_slot")
        check(first_entry.get("owner_expert_id") == args.expert_id, checks, "owner_expert_matches")
        check(first_entry.get("state") == SLOT_STATE_READY, checks, "slot_state_ready")
        check(first_entry.get("range_count") == 2, checks, "range_count_tracks_two_ranges")
        check(first_entry.get("total_bytes") == 2 * page_size, checks, "total_bytes_tracks_ranges")
        check(int(first_query["validated_entries"]) == 2, checks, "validated_entries_tracks_ranges")
        check(int(first_query["range_update_count"]) == 2, checks, "range_updates_track_ranges")
        check(int(first_query["pool_full_count"]) == 0, checks, "pool_full_zero")

        replacement = submit_batch(
            fd,
            mode=MODE_VALIDATE,
            layer_id=args.layer_id,
            expert_id=args.replacement_expert_id,
            slot_id=args.slot_id,
            base=replacement_base,
            page_size=page_size,
            ranges=1,
        )
        second_query = query_pool(fd, layer_id=args.layer_id, slot_id=args.slot_id)
        result["replacement_submit"] = replacement
        result["second_query"] = second_query
        second_entries = second_query["entries"]  # type: ignore[index]
        second_entry = second_entries[0] if second_entries else {}
        check(replacement["ret"] == 0 and replacement["success_count"] == 1, checks, "replacement_submit_ok")
        check(second_entry.get("owner_expert_id") == args.replacement_expert_id, checks, "replacement_overwrites_owner")
        check(second_entry.get("range_count") == 1, checks, "replacement_refreshes_range_count")
        check(int(second_query["overwrite_count"]) >= 1, checks, "overwrite_count_increments")
        check(int(second_query["validated_entries"]) == 3, checks, "validated_entries_after_replacement")
        check(int(second_query["current_epoch"]) == 3, checks, "epoch_tracks_validated_entries")

        copy_bytes = 2 * page_size
        copy_pattern = bytes((index % 251 for index in range(copy_bytes)))
        source = mmap.mmap(-1, copy_bytes)
        source.write(copy_pattern)
        source.seek(0)
        source_byte = ctypes.c_char.from_buffer(source)
        copy_base = ctypes.addressof(source_byte)
        del source_byte

        copy_only = submit_batch(
            fd,
            mode=MODE_COPY_ONLY,
            layer_id=args.layer_id,
            expert_id=args.replacement_expert_id,
            slot_id=args.slot_id,
            base=copy_base,
            page_size=page_size,
            ranges=2,
        )
        copy_query = query_pool(fd, layer_id=args.layer_id, slot_id=args.slot_id)
        copy_readback, copy_readback_bytes = readback_slot(
            fd,
            layer_id=args.layer_id,
            slot_id=args.slot_id,
            byte_count=copy_bytes,
        )
        source.close()

        result["copy_only_submit"] = copy_only
        result["copy_query"] = copy_query
        result["copy_readback"] = copy_readback
        copy_entries = copy_query["entries"]  # type: ignore[index]
        copy_entry = copy_entries[0] if copy_entries else {}
        check(copy_only["ret"] == 0 and copy_only["success_count"] == 2, checks, "copy_only_submit_ok")
        check(copy_only["failure_count"] == 0, checks, "copy_only_no_failures")
        check(copy_only["accepted_bytes"] == copy_bytes, checks, "copy_only_accepts_bytes")
        check(int(copy_query["backed_slots"]) == 1, checks, "copy_query_reports_backed_slot")
        check(int(copy_query["slot_backing_bytes"]) >= copy_bytes, checks, "copy_query_reports_backing_bytes")
        check(int(copy_query["copy_count"]) == 2, checks, "copy_query_tracks_copy_count")
        check(int(copy_query["copy_bytes"]) == copy_bytes, checks, "copy_query_tracks_copy_bytes")
        check(int(copy_query["copy_failed_count"]) == 0, checks, "copy_query_copy_failures_zero")
        check(copy_entry.get("backing_kind") == BACKING_SYSMEM, checks, "copy_entry_backing_sysmem")
        check(copy_entry.get("slot_bytes", 0) >= copy_bytes, checks, "copy_entry_slot_bytes")
        check(copy_entry.get("slot_physical_bytes", 0) >= copy_bytes, checks, "copy_entry_physical_bytes")
        check(copy_entry.get("copy_count") == 2, checks, "copy_entry_copy_count")
        check(copy_entry.get("copy_bytes") == copy_bytes, checks, "copy_entry_copy_bytes")
        check(copy_entry.get("copy_failed_count") == 0, checks, "copy_entry_copy_failures_zero")
        check(copy_entry.get("copy_status") == ENTRY_OK, checks, "copy_entry_status_ok")
        check(copy_entry.get("range_count") == 2, checks, "copy_refreshes_range_count")
        check(int(copy_query["validated_entries"]) == 5, checks, "copy_updates_validated_entries")
        check(bool(copy_readback["ok"]), checks, "copy_readback_ok")
        check(copy_readback["backing_kind"] == BACKING_SYSMEM, checks, "copy_readback_backing_sysmem")
        check(copy_readback["copied_bytes"] == copy_bytes, checks, "copy_readback_copied_bytes")
        check(copy_readback_bytes == copy_pattern, checks, "copy_readback_content_matches")

        if args.include_gpu_copy_only:
            initialize_cuda_device(args.device)
            gpu_copy_bytes = 2 * page_size
            gpu_copy_pattern = bytes(((index + 17) % 251 for index in range(gpu_copy_bytes)))
            gpu_source = mmap.mmap(-1, gpu_copy_bytes)
            gpu_source.write(gpu_copy_pattern)
            gpu_source.seek(0)
            gpu_source_byte = ctypes.c_char.from_buffer(gpu_source)
            gpu_copy_base = ctypes.addressof(gpu_source_byte)
            del gpu_source_byte

            gpu_copy = submit_batch(
                fd,
                mode=MODE_GPU_COPY_ONLY,
                gpu_uuid=gpu_uuid_for_device(args.device),
                layer_id=args.layer_id,
                expert_id=args.replacement_expert_id,
                slot_id=args.slot_id,
                base=gpu_copy_base,
                page_size=page_size,
                ranges=2,
            )
            gpu_copy_query = query_pool(fd, layer_id=args.layer_id, slot_id=args.slot_id)
            gpu_copy_readback, gpu_copy_readback_bytes = readback_slot(
                fd,
                layer_id=args.layer_id,
                slot_id=args.slot_id,
                byte_count=gpu_copy_bytes,
            )
            gpu_source.close()

            result["gpu_copy_only_submit"] = gpu_copy
            result["gpu_copy_query"] = gpu_copy_query
            result["gpu_copy_readback"] = gpu_copy_readback
            gpu_copy_entries = gpu_copy_query["entries"]  # type: ignore[index]
            gpu_copy_entry = gpu_copy_entries[0] if gpu_copy_entries else {}
            check(
                gpu_copy["ret"] == 0 and gpu_copy["success_count"] == 2,
                checks,
                "gpu_copy_only_submit_ok",
            )
            check(gpu_copy["failure_count"] == 0, checks, "gpu_copy_only_no_failures")
            check(gpu_copy["accepted_bytes"] == gpu_copy_bytes, checks, "gpu_copy_only_accepts_bytes")
            check(
                int(gpu_copy_query["backed_slots"]) == 1,
                checks,
                "gpu_copy_query_reports_backed_slot",
            )
            check(
                int(gpu_copy_query["slot_backing_bytes"]) >= gpu_copy_bytes,
                checks,
                "gpu_copy_query_reports_backing_bytes",
            )
            check(
                int(gpu_copy_query["copy_count"]) == 4,
                checks,
                "gpu_copy_query_tracks_copy_count",
            )
            check(
                int(gpu_copy_query["copy_bytes"]) == copy_bytes + gpu_copy_bytes,
                checks,
                "gpu_copy_query_tracks_copy_bytes",
            )
            check(
                int(gpu_copy_query["copy_failed_count"]) == 0,
                checks,
                "gpu_copy_query_copy_failures_zero",
            )
            check(
                gpu_copy_entry.get("backing_kind") == BACKING_VIDMEM,
                checks,
                "gpu_copy_entry_backing_vidmem",
            )
            check(
                gpu_copy_entry.get("slot_bytes", 0) >= gpu_copy_bytes,
                checks,
                "gpu_copy_entry_slot_bytes",
            )
            check(
                gpu_copy_entry.get("slot_physical_bytes", 0) >= gpu_copy_bytes,
                checks,
                "gpu_copy_entry_physical_bytes",
            )
            check(gpu_copy_entry.get("copy_count") == 2, checks, "gpu_copy_entry_copy_count")
            check(
                gpu_copy_entry.get("copy_bytes") == gpu_copy_bytes,
                checks,
                "gpu_copy_entry_copy_bytes",
            )
            check(
                gpu_copy_entry.get("copy_failed_count") == 0,
                checks,
                "gpu_copy_entry_copy_failures_zero",
            )
            check(gpu_copy_entry.get("copy_status") == ENTRY_OK, checks, "gpu_copy_entry_status_ok")
            check(gpu_copy_entry.get("range_count") == 2, checks, "gpu_copy_refreshes_range_count")
            check(
                int(gpu_copy_query["validated_entries"]) == 7,
                checks,
                "gpu_copy_updates_validated_entries",
            )
            check(bool(gpu_copy_readback["ok"]), checks, "gpu_copy_readback_ok")
            check(
                gpu_copy_readback["backing_kind"] == BACKING_VIDMEM,
                checks,
                "gpu_copy_readback_backing_vidmem",
            )
            check(
                gpu_copy_readback["copied_bytes"] == gpu_copy_bytes,
                checks,
                "gpu_copy_readback_copied_bytes",
            )
            check(
                gpu_copy_readback_bytes == gpu_copy_pattern,
                checks,
                "gpu_copy_readback_content_matches",
            )

        final_reset = query_pool(fd, reset=True)
        result["final_reset_query"] = final_reset
        check(bool(final_reset["ok"]), checks, "final_reset_query_ok")
        check(int(final_reset["active_slots"]) == 0, checks, "final_reset_clears_active_slots")
        check(int(final_reset["backed_slots"]) == 0, checks, "final_reset_clears_backing")
        check(int(final_reset["copy_count"]) == 0, checks, "final_reset_clears_copy_count")
        check(int(final_reset["copy_bytes"]) == 0, checks, "final_reset_clears_copy_bytes")

        result["passed"] = all(checks.values())
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    except OSError as exc:
        result.update({
            "passed": False,
            "error": str(exc),
            "errno": getattr(exc, "errno", None),
        })
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    finally:
        if fd >= 0:
            os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
