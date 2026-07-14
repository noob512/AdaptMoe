#!/usr/bin/env python3
"""Probe 2MiB-granularity expert release and CPU-to-slot restore.

This is intentionally smaller than a vLLM run. It models one 2MiB UVM VA block
containing two logical 1MiB slices:

  slice 0: target cold expert
  slice 1: neighboring expert/role data in the same 2MiB VA block

The release step registers the whole 2MiB envelope as the offload unit. The
restore step copies the whole 2MiB CPU payload into a driver-owned MoE slot and
remaps the original 2MiB VA block to that slot. Success means collateral data
can be repaired when destroy and restore use the same 2MiB granularity.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_moe_slot_remap_ioctl_smoke import (  # noqa: E402
    ENTRY_OK,
    MODE_VALUES,
    UVM_MOE_SLOT_REMAP_BATCH_IOCTL,
    UVM_VA_BLOCK_SIZE,
    UvmUuid,
    ManagedAllocation,
    MoeSlotRemapBatchParams,
    batch_to_result,
    cuda_memcpy_host_to_device,
    gpu_uuid_for_device,
    initialize_cuda_device,
    ioctl,
    kernel_readback,
    native_migrate,
    open_initialized_uvm_fd,
    query_slot_pool,
    read_uint_file,
    reset_slot_pool,
    slot_pool_health_ok,
    write_uint_file,
)

UVM_PHASE_F_BATCH_EXECUTOR_IOCTL = 81
UVM_PHASE_F_REGISTER_REGION_IOCTL = 82
UVM_PHASE_F_UNREGISTER_REGION_IOCTL = 83
UVM_EXPERT_RESERVED_POOL_IOCTL = 98
UVM_EXPERT_RESERVED_PREFETCH_IOCTL = 99
UVM_EXPERT_OFFLOAD_RELEASE_IOCTL = 100
UVM_PHASE_F_MAX_RANGES = 64

UVM_PHASE_F_ACTION_QUERY_RESIDENCY = 4
UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES = 128
UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES = 128

UVM_PHASE_N_POOL_WEIGHT = 2
UVM_REGION_ROLE_MOE_EXPERT_W13 = 3
UVM_REGION_FLAG_PREFETCHABLE = 1 << 3
UVM_REGION_FLAG_GPU_READ_ONLY = 1 << 9
UVM_REGION_FLAG_STREAMING_SUBPOOL = 1 << 12
UVM_REGION_FLAG_CPU_OFFLOADED_RESERVED = 1 << 14
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT = 1 << 0
UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC = 1
UVM_EXPERT_RESERVED_POOL_MODE_RESET = 2
UVM_EXPERT_RESERVED_PREFETCH_MODE_INVALIDATE = 2
UVM_EXPERT_RESERVED_PREFETCH_MODE_RESERVED_PREFETCH = 3
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT = 1 << 0

PTE_ENABLE_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable"
PTE_MAX_BYTES_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes"
VIDMEM_BUDGET_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"


class PhaseFRange(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
    ]


class PhaseFRangeResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("resident_on_count", ctypes.c_uint32),
        ("resident_on_cpu", ctypes.c_uint8),
        ("resident_on_gpu", ctypes.c_uint8),
        ("queried_bytes", ctypes.c_uint64),
        ("resident_gpu_bytes", ctypes.c_uint64),
        ("resident_cpu_bytes", ctypes.c_uint64),
        ("resident_any_bytes", ctypes.c_uint64),
        ("resident_unknown_bytes", ctypes.c_uint64),
    ]


class PhaseFBatchExecutorParams(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", UvmUuid),
        ("async_", ctypes.c_uint8),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("ranges", PhaseFRange * UVM_PHASE_F_MAX_RANGES),
        ("results", PhaseFRangeResult * UVM_PHASE_F_MAX_RANGES),
        ("rmStatus", ctypes.c_uint32),
    ]


class PhaseFRegion(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("pool", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class PhaseFRegisterRegionParams(ctypes.Structure):
    _fields_ = [
        ("region", PhaseFRegion),
        ("rmStatus", ctypes.c_uint32),
    ]


class PhaseFUnregisterRegionParams(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class ExpertOffloadReleaseParams(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("gpu_uuid", UvmUuid),
        ("flags", ctypes.c_uint32),
        ("max_blocks", ctypes.c_uint32),
        ("pool", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("block_count", ctypes.c_uint64),
        ("attempted_blocks", ctypes.c_uint32),
        ("success_blocks", ctypes.c_uint32),
        ("failure_blocks", ctypes.c_uint32),
        ("skipped_blocks", ctypes.c_uint32),
        ("attempted_bytes", ctypes.c_uint64),
        ("released_bytes", ctypes.c_uint64),
        ("failed_bytes", ctypes.c_uint64),
        ("first_failed_block", ctypes.c_uint64),
        ("first_error", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class ExpertReservedPoolEntry(ctypes.Structure):
    _fields_ = [
        ("pool_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("chunk_count", ctypes.c_uint32),
        ("slot_bytes", ctypes.c_uint64),
        ("physical_bytes", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
        ("first_chunk_address", ctypes.c_uint64),
        ("last_chunk_address", ctypes.c_uint64),
    ]


class ExpertReservedPoolParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", UvmUuid),
        ("max_entries", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("total_slots", ctypes.c_uint32),
        ("returned_slots", ctypes.c_uint32),
        ("active_slots", ctypes.c_uint32),
        ("backed_slots", ctypes.c_uint32),
        ("chunk_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("reserved_backing_bytes", ctypes.c_uint64),
        ("logical_slot_bytes", ctypes.c_uint64),
        ("current_epoch", ctypes.c_uint64),
        ("prealloc_count", ctypes.c_uint64),
        ("reset_count", ctypes.c_uint64),
        ("alloc_failed_count", ctypes.c_uint64),
        ("entries", ExpertReservedPoolEntry * UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class ExpertReservedPrefetchEntry(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("pool_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("slot_offset", ctypes.c_uint64),
        ("bytes_override", ctypes.c_uint64),
        ("resolved_base", ctypes.c_uint64),
        ("resolved_bytes", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("slot_epoch", ctypes.c_uint64),
        ("physical_bytes", ctypes.c_uint64),
    ]


class ExpertReservedPrefetchParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", UvmUuid),
        ("async_", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("resolved_regions", ctypes.c_uint64),
        ("assigned_slots", ctypes.c_uint64),
        ("invalidated_slots", ctypes.c_uint64),
        ("reserved_backing_bytes", ctypes.c_uint64),
        ("ordinary_resident_bytes", ctypes.c_uint64),
        ("pool_external_alloc_count", ctypes.c_uint64),
        ("footprint_contract_broken", ctypes.c_uint64),
        ("current_epoch", ctypes.c_uint64),
        ("entries", ExpertReservedPrefetchEntry * UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


def _pattern(byte_count: int, seed: int) -> bytes:
    return bytes(((index * 17 + seed) % 251 for index in range(byte_count)))


def _set_param(path: str, value: int, result: dict[str, object], key: str) -> None:
    old = read_uint_file(path)
    result[f"old_{key}"] = old
    if old != value:
        write_uint_file(path, value)
    result[f"current_{key}"] = read_uint_file(path)


def _query_residency(fd: int, gpu_uuid, base: int, byte_count: int) -> dict[str, object]:
    params = PhaseFBatchExecutorParams()
    params.action = UVM_PHASE_F_ACTION_QUERY_RESIDENCY
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.ranges[0].base = int(base)
    params.ranges[0].length = int(byte_count)
    ret, errno_value = ioctl(fd, UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, params)
    entry = params.results[0]
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "status": int(entry.status),
        "on_cpu": int(entry.resident_on_cpu),
        "on_gpu": int(entry.resident_on_gpu),
        "queried_bytes": int(entry.queried_bytes),
        "gpu_bytes": int(entry.resident_gpu_bytes),
        "cpu_bytes": int(entry.resident_cpu_bytes),
        "any_bytes": int(entry.resident_any_bytes),
        "unknown_bytes": int(entry.resident_unknown_bytes),
        "ok": ret == 0 and int(params.rmStatus) == 0 and int(entry.status) == 0,
    }


def _register_expert_region(
    fd: int,
    *,
    region_id: int,
    base: int,
    byte_count: int,
) -> dict[str, object]:
    params = PhaseFRegisterRegionParams()
    params.region.region_id = int(region_id)
    params.region.base = int(base)
    params.region.length = int(byte_count)
    params.region.pool = UVM_PHASE_N_POOL_WEIGHT
    params.region.layer_id = 0
    params.region.expert_id = 0
    params.region.role = UVM_REGION_ROLE_MOE_EXPERT_W13
    params.region.flags = (
        UVM_REGION_FLAG_PREFETCHABLE
        | UVM_REGION_FLAG_GPU_READ_ONLY
        | UVM_REGION_FLAG_STREAMING_SUBPOOL
        | UVM_REGION_FLAG_CPU_OFFLOADED_RESERVED
    )
    ret, errno_value = ioctl(fd, UVM_PHASE_F_REGISTER_REGION_IOCTL, params)
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "region_id": int(region_id),
        "base": int(base),
        "length": int(byte_count),
        "ok": ret == 0 and int(params.rmStatus) == 0,
    }


def _unregister_region(fd: int, region_id: int) -> dict[str, object]:
    params = PhaseFUnregisterRegionParams()
    params.region_id = int(region_id)
    ret, errno_value = ioctl(fd, UVM_PHASE_F_UNREGISTER_REGION_IOCTL, params)
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "ok": ret == 0 and int(params.rmStatus) == 0,
    }


def _release_region(fd: int, gpu_uuid, region_id: int) -> dict[str, object]:
    params = ExpertOffloadReleaseParams()
    params.region_id = int(region_id)
    params.gpu_uuid = gpu_uuid
    params.flags = UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT
    ret, errno_value = ioctl(fd, UVM_EXPERT_OFFLOAD_RELEASE_IOCTL, params)
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "region_id": int(params.region_id),
        "pool": int(params.pool),
        "layer_id": int(params.layer_id),
        "expert_id": int(params.expert_id),
        "role": int(params.role),
        "base": int(params.base),
        "length": int(params.length),
        "block_count": int(params.block_count),
        "attempted_blocks": int(params.attempted_blocks),
        "success_blocks": int(params.success_blocks),
        "failure_blocks": int(params.failure_blocks),
        "skipped_blocks": int(params.skipped_blocks),
        "attempted_bytes": int(params.attempted_bytes),
        "released_bytes": int(params.released_bytes),
        "failed_bytes": int(params.failed_bytes),
        "first_failed_block": int(params.first_failed_block),
        "first_error": int(params.first_error),
        "ok": ret == 0 and int(params.rmStatus) == 0 and int(params.released_bytes) > 0,
    }


def _reserved_pool_result(params: ExpertReservedPoolParams, ret: int, errno_value: int) -> dict[str, object]:
    entries = []
    for index in range(min(int(params.returned_slots), UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES)):
        entry = params.entries[index]
        entries.append({
            "pool_id": int(entry.pool_id),
            "slot_id": int(entry.slot_id),
            "status": int(entry.status),
            "state": int(entry.state),
            "chunk_count": int(entry.chunk_count),
            "slot_bytes": int(entry.slot_bytes),
            "physical_bytes": int(entry.physical_bytes),
            "epoch": int(entry.epoch),
        })
    return {
        "ret": int(ret),
        "errno": int(errno_value),
        "rmStatus": int(params.rmStatus),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "total_slots": int(params.total_slots),
        "returned_slots": int(params.returned_slots),
        "active_slots": int(params.active_slots),
        "backed_slots": int(params.backed_slots),
        "chunk_size": int(params.chunk_size),
        "reserved_backing_bytes": int(params.reserved_backing_bytes),
        "logical_slot_bytes": int(params.logical_slot_bytes),
        "prealloc_count": int(params.prealloc_count),
        "reset_count": int(params.reset_count),
        "alloc_failed_count": int(params.alloc_failed_count),
        "entries": entries,
        "ok": int(ret) == 0 and int(params.rmStatus) == 0,
    }


def _reserved_pool_reset(fd: int, gpu_uuid) -> dict[str, object]:
    params = ExpertReservedPoolParams()
    params.mode = UVM_EXPERT_RESERVED_POOL_MODE_RESET
    params.gpu_uuid = gpu_uuid
    params.max_entries = UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_POOL_IOCTL, params)
    return _reserved_pool_result(params, ret, errno_value)


def _reserved_pool_prealloc_2mb(fd: int, gpu_uuid, block_bytes: int) -> dict[str, object]:
    params = ExpertReservedPoolParams()
    params.mode = UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.max_entries = UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
    params.entries[0].pool_id = 0
    params.entries[0].slot_id = 0
    params.entries[0].slot_bytes = int(block_bytes)
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_POOL_IOCTL, params)
    return _reserved_pool_result(params, ret, errno_value)


def _reserved_prefetch_result(
    params: ExpertReservedPrefetchParams,
    ret: int,
    errno_value: int,
) -> dict[str, object]:
    entry = params.entries[0]
    return {
        "ret": int(ret),
        "errno": int(errno_value),
        "rmStatus": int(params.rmStatus),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "requested_bytes": int(params.requested_bytes),
        "accepted_bytes": int(params.accepted_bytes),
        "resolved_regions": int(params.resolved_regions),
        "assigned_slots": int(params.assigned_slots),
        "invalidated_slots": int(params.invalidated_slots),
        "reserved_backing_bytes": int(params.reserved_backing_bytes),
        "ordinary_resident_bytes": int(params.ordinary_resident_bytes),
        "pool_external_alloc_count": int(params.pool_external_alloc_count),
        "footprint_contract_broken": int(params.footprint_contract_broken),
        "entry": {
            "region_id": int(entry.region_id),
            "pool_id": int(entry.pool_id),
            "slot_id": int(entry.slot_id),
            "status": int(entry.status),
            "resolved_base": int(entry.resolved_base),
            "resolved_bytes": int(entry.resolved_bytes),
            "layer_id": int(entry.layer_id),
            "expert_id": int(entry.expert_id),
            "role": int(entry.role),
            "slot_epoch": int(entry.slot_epoch),
            "physical_bytes": int(entry.physical_bytes),
        },
        "ok": int(ret) == 0 and int(params.rmStatus) == 0,
    }


def _reserved_prefetch_restore(fd: int, gpu_uuid, *, region_id: int, block_bytes: int) -> dict[str, object]:
    params = ExpertReservedPrefetchParams()
    params.mode = UVM_EXPERT_RESERVED_PREFETCH_MODE_RESERVED_PREFETCH
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.async_ = 1
    params.entries[0].region_id = int(region_id)
    params.entries[0].pool_id = 0
    params.entries[0].slot_id = 0
    params.entries[0].flags = UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT
    params.entries[0].bytes_override = int(block_bytes)
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_PREFETCH_IOCTL, params)
    return _reserved_prefetch_result(params, ret, errno_value)


def _reserved_prefetch_invalidate(fd: int, gpu_uuid) -> dict[str, object]:
    params = ExpertReservedPrefetchParams()
    params.mode = UVM_EXPERT_RESERVED_PREFETCH_MODE_INVALIDATE
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.entries[0].pool_id = 0
    params.entries[0].slot_id = 0
    params.entries[0].flags = UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_PREFETCH_IOCTL, params)
    return _reserved_prefetch_result(params, ret, errno_value)


def _submit_slot_batch(fd: int, *, mode: str, gpu_uuid, va_base: int, byte_count: int) -> dict[str, object]:
    params = MoeSlotRemapBatchParams()
    params.mode = MODE_VALUES[mode]
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.validate_managed = 1 if mode == "slot_va_block_remap" else 0
    entry = params.entries[0]
    entry.layer_id = 77
    entry.expert_id = 1001
    entry.slot_id = 0
    entry.role = UVM_REGION_ROLE_MOE_EXPERT_W13
    entry.va_base = int(va_base)
    entry.bytes = int(byte_count)
    entry.slot_offset = 0
    entry.flags = 0
    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
    result = batch_to_result(params, ret, errno_value)
    result["mode"] = mode
    return result


def _prefill_slot_from_cpu_payload(fd: int, *, gpu_uuid, payload: bytes) -> dict[str, object]:
    source = mmap.mmap(-1, len(payload))
    try:
        source.write(payload)
        source.seek(0)
        source_byte = ctypes.c_char.from_buffer(source)
        source_base = ctypes.addressof(source_byte)
        del source_byte
        result = _submit_slot_batch(
            fd,
            mode="gpu_copy_only",
            gpu_uuid=gpu_uuid,
            va_base=source_base,
            byte_count=len(payload),
        )
        result["source_base"] = int(source_base)
        result["source_page_aligned"] = source_base % os.sysconf("SC_PAGE_SIZE") == 0
        return result
    finally:
        source.close()


def _batch_ok(result: dict[str, object]) -> bool:
    return (
        int(result.get("ret", -1)) == 0
        and int(result.get("rmStatus", -1)) == 0
        and int(result.get("success_count", -1)) == 1
        and int(result.get("failure_count", -1)) == 0
        and list(result.get("entry_statuses", [])) == [ENTRY_OK]
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    block_bytes = UVM_VA_BLOCK_SIZE
    slice_bytes = block_bytes // 2
    region_id = int(args.region_id)
    result: dict[str, object] = {
        "device": args.device,
        "block_bytes": block_bytes,
        "slice_bytes": slice_bytes,
        "region_id": region_id,
    }
    old_params: dict[str, int] = {}
    fd = -1
    target: ManagedAllocation | None = None

    try:
        old_params = {
            PTE_ENABLE_PARAM: read_uint_file(PTE_ENABLE_PARAM),
            PTE_MAX_BYTES_PARAM: read_uint_file(PTE_MAX_BYTES_PARAM),
            VIDMEM_BUDGET_PARAM: read_uint_file(VIDMEM_BUDGET_PARAM),
        }
        _set_param(PTE_ENABLE_PARAM, 1, result, "pte_enable")
        _set_param(PTE_MAX_BYTES_PARAM, block_bytes, result, "pte_max_bytes")
        _set_param(VIDMEM_BUDGET_PARAM, block_bytes, result, "vidmem_budget")

        initialize_cuda_device(args.device)
        gpu_uuid = gpu_uuid_for_device(args.device)
        target = ManagedAllocation(args.device, 2 * block_bytes)
        base = ((target.address + block_bytes - 1) // block_bytes) * block_bytes
        result.update({
            "managed_raw_base": target.address,
            "managed_base": base,
            "managed_base_2mb_aligned": base % block_bytes == 0,
        })

        target_payload = _pattern(slice_bytes, 11)
        neighbor_payload = _pattern(slice_bytes, 137)
        payload = target_payload + neighbor_payload
        cuda_memcpy_host_to_device(base, payload)

        fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(prefer_existing=True)
        result["uvm_fd"] = {
            "fd_source": fd_source,
            "init_ret": init_ret,
            "init_errno": init_errno,
            "init_rmStatus": init_rm_status,
        }
        if fd < 0:
            result["passed"] = False
            result["blocked_reason"] = "failed to open UVM fd"
            return result

        reset_before = reset_slot_pool(fd)
        result["reset_before"] = reset_before
        migrate_gpu = native_migrate(fd, base, block_bytes, gpu_uuid)
        result["native_migrate_to_gpu"] = migrate_gpu
        before_release = _query_residency(fd, gpu_uuid, base, block_bytes)
        result["residency_before_release"] = before_release
        gpu_read_before = kernel_readback(base, block_bytes, args.device)
        result["gpu_read_before_matches_payload"] = gpu_read_before == payload

        register = _register_expert_region(
            fd,
            region_id=region_id,
            base=base,
            byte_count=block_bytes,
        )
        result["register_2mb_envelope_region"] = register
        if not register["ok"]:
            result["passed"] = False
            result["blocked_reason"] = "2MiB envelope region registration failed"
            return result

        release = _release_region(fd, gpu_uuid, region_id)
        result["offload_release_2mb_envelope_region"] = release
        after_release = _query_residency(fd, gpu_uuid, base, block_bytes)
        result["residency_after_release"] = after_release

        cpu_payload_after_release = ctypes.string_at(base, block_bytes)
        result["cpu_read_after_release_matches_payload"] = cpu_payload_after_release == payload
        after_cpu_read = _query_residency(fd, gpu_uuid, base, block_bytes)
        result["residency_after_cpu_read"] = after_cpu_read

        reserved_reset_before = _reserved_pool_reset(fd, gpu_uuid)
        reserved_prealloc = _reserved_pool_prealloc_2mb(fd, gpu_uuid, block_bytes)
        reserved_prefetch = _reserved_prefetch_restore(
            fd,
            gpu_uuid,
            region_id=region_id,
            block_bytes=block_bytes,
        )
        gpu_read_after_reserved_restore = kernel_readback(base, block_bytes, args.device)
        reserved_invalidate = _reserved_prefetch_invalidate(fd, gpu_uuid)
        reserved_reset_after = _reserved_pool_reset(fd, gpu_uuid)
        result["expert_reserved_reset_before"] = reserved_reset_before
        result["expert_reserved_prealloc_2mb"] = reserved_prealloc
        result["expert_reserved_prefetch_restore_from_cpu"] = reserved_prefetch
        result["gpu_read_after_expert_reserved_restore_matches_full_payload"] = (
            gpu_read_after_reserved_restore == payload
        )
        result["gpu_read_after_expert_reserved_restore_matches_target_half"] = (
            gpu_read_after_reserved_restore[:slice_bytes] == target_payload
        )
        result["gpu_read_after_expert_reserved_restore_matches_neighbor_half"] = (
            gpu_read_after_reserved_restore[slice_bytes:] == neighbor_payload
        )
        result["expert_reserved_invalidate_after_restore"] = reserved_invalidate
        result["expert_reserved_reset_after"] = reserved_reset_after

        prefill = _prefill_slot_from_cpu_payload(
            fd,
            gpu_uuid=gpu_uuid,
            payload=cpu_payload_after_release,
        )
        result["slot_prefill_from_cpu_payload"] = prefill
        remap = _submit_slot_batch(
            fd,
            mode="slot_va_block_remap",
            gpu_uuid=gpu_uuid,
            va_base=base,
            byte_count=block_bytes,
        )
        result["slot_va_block_remap_restore"] = remap

        gpu_read_after_restore = kernel_readback(base, block_bytes, args.device)
        result["gpu_read_after_restore_matches_full_payload"] = gpu_read_after_restore == payload
        result["gpu_read_after_restore_matches_target_half"] = (
            gpu_read_after_restore[:slice_bytes] == target_payload
        )
        result["gpu_read_after_restore_matches_neighbor_half"] = (
            gpu_read_after_restore[slice_bytes:] == neighbor_payload
        )

        final_query = query_slot_pool(fd, max_entries=4)
        result["slot_pool_after_restore"] = final_query
        result["reset_after"] = reset_slot_pool(fd)

        result["checks"] = {
            "before_gpu_resident_2mb": before_release.get("gpu_bytes") == block_bytes,
            "release_freed_2mb": release.get("released_bytes") == block_bytes,
            "after_release_gpu_zero": after_release.get("gpu_bytes") == 0,
            "after_release_cpu_has_2mb": after_release.get("cpu_bytes") == block_bytes,
            "cpu_copy_available": result["cpu_read_after_release_matches_payload"],
            "cpu_read_did_not_refault_gpu": after_cpu_read.get("gpu_bytes") == 0,
            "expert_reserved_prealloc_ok": (
                reserved_prealloc.get("ok") is True
                and reserved_prealloc.get("success_count") == 1
                and reserved_prealloc.get("reserved_backing_bytes") == block_bytes
            ),
            "expert_reserved_prefetch_from_cpu_ok": (
                reserved_prefetch.get("ok") is True
                and reserved_prefetch.get("success_count") == 1
                and reserved_prefetch.get("accepted_bytes") == block_bytes
                and reserved_prefetch.get("assigned_slots") == 1
                and reserved_prefetch.get("entry", {}).get("status") == 0
            ),
            "expert_reserved_full_payload_restored": (
                result["gpu_read_after_expert_reserved_restore_matches_full_payload"]
            ),
            "expert_reserved_target_half_restored": (
                result["gpu_read_after_expert_reserved_restore_matches_target_half"]
            ),
            "expert_reserved_neighbor_half_restored": (
                result["gpu_read_after_expert_reserved_restore_matches_neighbor_half"]
            ),
            "expert_reserved_invalidate_ok": (
                reserved_invalidate.get("ok") is True
                and reserved_invalidate.get("success_count") == 1
            ),
            "expert_reserved_reset_ok": reserved_reset_after.get("ok") is True,
            "prefill_from_cpu_ok": _batch_ok(prefill),
            "remap_restore_ok": _batch_ok(remap) and int(remap.get("remap_count", -1)) == 1,
            "full_payload_restored": result["gpu_read_after_restore_matches_full_payload"],
            "target_half_restored": result["gpu_read_after_restore_matches_target_half"],
            "neighbor_half_restored": result["gpu_read_after_restore_matches_neighbor_half"],
            "slot_backing_2mb": final_query.get("slot_backing_bytes") == block_bytes,
            "slot_copy_count_1": final_query.get("copy_count") == 1,
            "slot_remap_count_1": final_query.get("pte_remap_count") == 1,
            "slot_pool_health_ok": slot_pool_health_ok(final_query),
        }
        result["passed"] = all(bool(value) for value in result["checks"].values())
        return result
    finally:
        if fd >= 0:
            try:
                _unregister_region(fd, region_id)
            except Exception:
                pass
            os.close(fd)
        if target is not None:
            target.close()
        if args.restore_params:
            for path, value in old_params.items():
                try:
                    write_uint_file(path, value)
                except Exception as exc:
                    result.setdefault("restore_errors", []).append({
                        "path": path,
                        "value": value,
                        "error": str(exc),
                    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--region-id", type=lambda value: int(value, 0), default=0x2B00000001)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--restore-params", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    result = run(args)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
