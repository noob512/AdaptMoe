#!/usr/bin/env python3
"""Phase S3 smoke for restoring a released 512KiB range through reserved slot."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_2mb_offload_release_slot_restore import (  # noqa: E402
    ExpertReservedPoolParams,
    ExpertReservedPrefetchParams,
    UVM_EXPERT_RESERVED_POOL_IOCTL,
    UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
    UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC,
    UVM_EXPERT_RESERVED_POOL_MODE_RESET,
    UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT,
    UVM_EXPERT_RESERVED_PREFETCH_IOCTL,
    UVM_EXPERT_RESERVED_PREFETCH_MODE_INVALIDATE,
    UVM_EXPERT_RESERVED_PREFETCH_MODE_RESERVED_PREFETCH,
    _pattern,
    _query_residency,
    _register_expert_region,
    _reserved_prefetch_result,
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
    kernel_readback,
    native_migrate,
    open_initialized_uvm_fd,
)
from check_uvm_pmm_512k_phase_s0 import CHUNK_512K, PAGE_64K  # noqa: E402


def _reserved_pool_result(params: ExpertReservedPoolParams,
                          ret: int,
                          errno_value: int) -> dict[str, Any]:
    entry_count = (
        int(params.count)
        if int(params.mode) == UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC
        else int(params.returned_slots)
    )
    entry_count = max(0, min(entry_count, UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES))
    entries: list[dict[str, int]] = []
    for index in range(entry_count):
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
            "first_chunk_address": int(entry.first_chunk_address),
            "last_chunk_address": int(entry.last_chunk_address),
        })

    return {
        "ret": int(ret),
        "errno": int(errno_value),
        "rmStatus": int(params.rmStatus),
        "ok": int(ret) == 0 and int(params.rmStatus) == 0,
        "mode": int(params.mode),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "total_slots": int(params.total_slots),
        "returned_slots": int(params.returned_slots),
        "active_slots": int(params.active_slots),
        "backed_slots": int(params.backed_slots),
        "chunk_size": int(params.chunk_size),
        "reserved_backing_bytes": int(params.reserved_backing_bytes),
        "logical_slot_bytes": int(params.logical_slot_bytes),
        "current_epoch": int(params.current_epoch),
        "prealloc_count": int(params.prealloc_count),
        "reset_count": int(params.reset_count),
        "alloc_failed_count": int(params.alloc_failed_count),
        "entries": entries,
    }


def _reserved_pool_reset(fd: int, gpu_uuid) -> dict[str, Any]:
    params = ExpertReservedPoolParams()
    params.mode = UVM_EXPERT_RESERVED_POOL_MODE_RESET
    params.gpu_uuid = gpu_uuid
    params.max_entries = UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_POOL_IOCTL, params)
    return _reserved_pool_result(params, ret, errno_value)


def _reserved_pool_prealloc_512k(fd: int,
                                 gpu_uuid,
                                 *,
                                 pool_id: int,
                                 slot_id: int,
                                 slot_bytes: int) -> dict[str, Any]:
    params = ExpertReservedPoolParams()
    params.mode = UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.max_entries = UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
    params.entries[0].pool_id = int(pool_id)
    params.entries[0].slot_id = int(slot_id)
    params.entries[0].slot_bytes = int(slot_bytes)
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_POOL_IOCTL, params)
    return _reserved_pool_result(params, ret, errno_value)


def _reserved_prefetch_restore_512k(fd: int,
                                    gpu_uuid,
                                    *,
                                    region_id: int,
                                    pool_id: int,
                                    slot_id: int,
                                    byte_count: int) -> dict[str, Any]:
    params = ExpertReservedPrefetchParams()
    params.mode = UVM_EXPERT_RESERVED_PREFETCH_MODE_RESERVED_PREFETCH
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.async_ = 1
    params.entries[0].region_id = int(region_id)
    params.entries[0].pool_id = int(pool_id)
    params.entries[0].slot_id = int(slot_id)
    params.entries[0].flags = UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT
    params.entries[0].bytes_override = int(byte_count)
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_PREFETCH_IOCTL, params)
    return _reserved_prefetch_result(params, ret, errno_value)


def _reserved_prefetch_invalidate(fd: int,
                                  gpu_uuid,
                                  *,
                                  pool_id: int,
                                  slot_id: int) -> dict[str, Any]:
    params = ExpertReservedPrefetchParams()
    params.mode = UVM_EXPERT_RESERVED_PREFETCH_MODE_INVALIDATE
    params.count = 1
    params.gpu_uuid = gpu_uuid
    params.entries[0].pool_id = int(pool_id)
    params.entries[0].slot_id = int(slot_id)
    params.entries[0].flags = UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_PREFETCH_IOCTL, params)
    return _reserved_prefetch_result(params, ret, errno_value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    block_bytes = UVM_VA_BLOCK_SIZE
    target_bytes = CHUNK_512K
    left_bytes = CHUNK_512K
    right_bytes = block_bytes - left_bytes - target_bytes
    pool_id = int(args.pool_id)
    slot_id = int(args.slot_id)
    region_id = int(args.region_id)
    fd = -1
    target: ManagedAllocation | None = None
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(args.device),
        "block_bytes": int(block_bytes),
        "logical_restore_bytes": int(target_bytes),
        "physical_chunk_bytes": int(PAGE_64K),
        "expected_slot_chunk_count": int(target_bytes // PAGE_64K),
        "pool_id": pool_id,
        "slot_id": slot_id,
        "region_id": region_id,
    }

    try:
        initialize_cuda_device(args.device)
        gpu_uuid = gpu_uuid_for_device(args.device)
        target = ManagedAllocation(args.device, 2 * block_bytes)
        base = ((target.address + block_bytes - 1) // block_bytes) * block_bytes
        target_base = base + left_bytes
        right_base = target_base + target_bytes
        result.update({
            "managed_raw_base": int(target.address),
            "managed_base": int(base),
            "target_base": int(target_base),
            "right_base": int(right_base),
            "managed_base_2mb_aligned": bool(base % block_bytes == 0),
            "target_512k_aligned": bool(target_base % target_bytes == 0),
        })

        left_payload = _pattern(left_bytes, 23)
        target_payload = _pattern(target_bytes, 83)
        right_payload = _pattern(right_bytes, 149)
        payload = left_payload + target_payload + right_payload
        cuda_memcpy_host_to_device(base, payload)

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

        result["reserved_reset_before"] = _reserved_pool_reset(fd, gpu_uuid)
        migrate_gpu = native_migrate(fd, base, block_bytes, gpu_uuid)
        result["native_migrate_to_gpu"] = migrate_gpu
        before_whole = _query_residency(fd, gpu_uuid, base, block_bytes)
        before_target = _query_residency(fd, gpu_uuid, target_base, target_bytes)
        result["residency_before_whole"] = before_whole
        result["residency_before_target"] = before_target
        result["gpu_read_before_matches_payload"] = (
            kernel_readback(base, block_bytes, args.device) == payload
        )

        register = _register_expert_region(
            fd,
            region_id=region_id,
            base=target_base,
            byte_count=target_bytes,
        )
        result["register_512k_region"] = register
        if not register["ok"]:
            result["passed"] = False
            result["blocked_reason"] = "512KiB region registration failed"
            return result

        release = _release_region_subchunk_64k(fd, gpu_uuid, region_id)
        result["offload_release_512k_subchunk_64k"] = release
        after_release_target = _query_residency(fd, gpu_uuid, target_base, target_bytes)
        after_release_whole = _query_residency(fd, gpu_uuid, base, block_bytes)
        result["residency_after_release_target"] = after_release_target
        result["residency_after_release_whole"] = after_release_whole
        result["cpu_read_after_release_matches_target_payload"] = (
            ctypes.string_at(target_base, target_bytes) == target_payload
        )

        reserved_prealloc = _reserved_pool_prealloc_512k(
            fd,
            gpu_uuid,
            pool_id=pool_id,
            slot_id=slot_id,
            slot_bytes=target_bytes,
        )
        result["expert_reserved_prealloc_512k"] = reserved_prealloc

        reserved_restore = _reserved_prefetch_restore_512k(
            fd,
            gpu_uuid,
            region_id=region_id,
            pool_id=pool_id,
            slot_id=slot_id,
            byte_count=target_bytes,
        )
        result["expert_reserved_prefetch_restore_512k"] = reserved_restore

        gpu_target_after_restore = kernel_readback(target_base, target_bytes, args.device)
        gpu_left_after_restore = kernel_readback(base, left_bytes, args.device)
        gpu_right_after_restore = kernel_readback(right_base, right_bytes, args.device)
        result["gpu_read_after_restore_matches_target"] = (
            gpu_target_after_restore == target_payload
        )
        result["gpu_read_after_restore_matches_left_neighbor"] = (
            gpu_left_after_restore == left_payload
        )
        result["gpu_read_after_restore_matches_right_neighbor"] = (
            gpu_right_after_restore == right_payload
        )
        result["residency_after_restore_target"] = _query_residency(
            fd, gpu_uuid, target_base, target_bytes
        )
        result["residency_after_restore_whole"] = _query_residency(
            fd, gpu_uuid, base, block_bytes
        )

        reserved_invalidate = _reserved_prefetch_invalidate(
            fd, gpu_uuid, pool_id=pool_id, slot_id=slot_id
        )
        reserved_reset_after = _reserved_pool_reset(fd, gpu_uuid)
        result["expert_reserved_invalidate_after_restore"] = reserved_invalidate
        result["reserved_reset_after"] = reserved_reset_after

        entry = (reserved_prealloc.get("entries") or [{}])[0]
        restore_entry = reserved_restore.get("entry", {})
        result["checks"] = {
            "before_whole_gpu_resident_2mb": before_whole.get("gpu_bytes") == block_bytes,
            "before_target_gpu_resident_512k": before_target.get("gpu_bytes") == target_bytes,
            "release_released_512k": release.get("released_bytes") == target_bytes,
            "after_release_target_gpu_zero": after_release_target.get("gpu_bytes") == 0,
            "after_release_target_cpu_512k": after_release_target.get("cpu_bytes") == target_bytes,
            "cpu_target_payload_available": result["cpu_read_after_release_matches_target_payload"],
            "slot_prealloc_ok": (
                reserved_prealloc.get("ok") is True
                and reserved_prealloc.get("success_count") == 1
                and reserved_prealloc.get("failure_count") == 0
            ),
            "slot_prealloc_512k_backing": (
                reserved_prealloc.get("reserved_backing_bytes") == target_bytes
                and reserved_prealloc.get("logical_slot_bytes") == target_bytes
            ),
            "slot_prealloc_64k_chunks": (
                reserved_prealloc.get("chunk_size") == PAGE_64K
                and entry.get("slot_bytes") == target_bytes
                and entry.get("physical_bytes") == target_bytes
                and entry.get("chunk_count") == target_bytes // PAGE_64K
            ),
            "reserved_restore_ok": (
                reserved_restore.get("ok") is True
                and reserved_restore.get("success_count") == 1
                and reserved_restore.get("failure_count") == 0
            ),
            "reserved_restore_accepted_512k": (
                reserved_restore.get("accepted_bytes") == target_bytes
                and reserved_restore.get("assigned_slots") == 1
                and restore_entry.get("status") == 0
                and restore_entry.get("resolved_bytes") == target_bytes
                and restore_entry.get("physical_bytes") == target_bytes
            ),
            "target_payload_restored": result["gpu_read_after_restore_matches_target"],
            "left_neighbor_payload_unchanged": result["gpu_read_after_restore_matches_left_neighbor"],
            "right_neighbor_payload_unchanged": result["gpu_read_after_restore_matches_right_neighbor"],
            "invalidate_ok": (
                reserved_invalidate.get("ok") is True
                and reserved_invalidate.get("success_count") == 1
            ),
            "reset_after_ok": reserved_reset_after.get("ok") is True,
        }
        result["passed"] = all(bool(value) for value in result["checks"].values())
        if not result["passed"]:
            result["failure_reason"] = "failed checks: " + ", ".join(
                key for key, value in result["checks"].items() if not value
            )
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--region-id", type=lambda value: int(value, 0), default=0x5120000003)
    parser.add_argument("--pool-id", type=int, default=5123)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if result.get("passed") or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
