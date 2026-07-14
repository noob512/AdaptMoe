#!/usr/bin/env python3
"""Phase S0 probe for 512KiB PMM/reserved-slot viability.

This script is intentionally a gate, not a workaround. It checks whether the
currently loaded UVM module natively supports 512KiB user PMM chunks and whether
the expert reserved slot pool can preallocate 48 logical 512KiB slots with only
24MiB of physical backing.
"""

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
    UVM_EXPERT_RESERVED_POOL_IOCTL,
    UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
    UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC,
    UVM_EXPERT_RESERVED_POOL_MODE_RESET,
)
from check_moe_slot_remap_ioctl_smoke import (  # noqa: E402
    ManagedAllocation,
    UvmInitializeParams,
    UvmUuid,
    gpu_uuid_for_device,
    initialize_cuda_device,
    ioctl,
    native_migrate,
    open_initialized_uvm_fd,
)

UVM_INITIALIZE_IOCTL = 0x30000001
UVM_TEST_PMM_QUERY_IOCTL = 221
UVM_TEST_CHUNK_SIZE_GET_USER_SIZE = 0

PAGE_4K = 4 * 1024
PAGE_64K = 64 * 1024
CHUNK_256K = 256 * 1024
CHUNK_512K = 512 * 1024
CHUNK_2M = 2 * 1024 * 1024


class UvmTestPmmQueryParams(ctypes.Structure):
    _fields_ = [
        ("gpu_uuid", UvmUuid),
        ("key", ctypes.c_uint64),
        ("value", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


def _chunk_sizes_from_mask(mask: int) -> list[int]:
    sizes: list[int] = []
    bit = 1
    while bit <= CHUNK_2M * 2:
        if mask & bit:
            sizes.append(bit)
        bit <<= 1
    return sizes


def _reserved_pool_result(params: ExpertReservedPoolParams,
                          ret: int,
                          errno_value: int) -> dict[str, Any]:
    entry_count = (
        int(params.count)
        if int(params.mode) == UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC
        else int(params.returned_slots)
    )
    entry_count = max(0, min(entry_count, UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES))
    entries = []
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


def _reset_reserved_pool(fd: int) -> dict[str, Any]:
    params = ExpertReservedPoolParams()
    params.mode = UVM_EXPERT_RESERVED_POOL_MODE_RESET
    params.max_entries = UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_POOL_IOCTL, params)
    return _reserved_pool_result(params, ret, errno_value)


def _prealloc_reserved_slots(fd: int,
                             gpu_uuid: UvmUuid,
                             *,
                             pool_id: int,
                             slot_count: int,
                             slot_bytes: int) -> dict[str, Any]:
    params = ExpertReservedPoolParams()
    params.mode = UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC
    params.count = int(slot_count)
    params.gpu_uuid = gpu_uuid
    params.max_entries = UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
    for index in range(slot_count):
        params.entries[index].pool_id = int(pool_id)
        params.entries[index].slot_id = int(index)
        params.entries[index].slot_bytes = int(slot_bytes)
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_POOL_IOCTL, params)
    return _reserved_pool_result(params, ret, errno_value)


def _query_user_pmm_chunk_mask(fd: int, gpu_uuid: UvmUuid) -> dict[str, Any]:
    params = UvmTestPmmQueryParams()
    params.gpu_uuid = gpu_uuid
    params.key = UVM_TEST_CHUNK_SIZE_GET_USER_SIZE
    ret, errno_value = ioctl(fd, UVM_TEST_PMM_QUERY_IOCTL, params)
    mask = int(params.value)
    return {
        "ret": int(ret),
        "errno": int(errno_value),
        "rmStatus": int(params.rmStatus),
        "ok": int(ret) == 0 and int(params.rmStatus) == 0,
        "mask": mask,
        "mask_hex": hex(mask),
        "sizes": _chunk_sizes_from_mask(mask),
        "has_4k": bool(mask & PAGE_4K),
        "has_64k": bool(mask & PAGE_64K),
        "has_256k": bool(mask & CHUNK_256K),
        "has_512k": bool(mask & CHUNK_512K),
        "has_2m": bool(mask & CHUNK_2M),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    initialize_cuda_device(args.device)
    alloc = ManagedAllocation(args.device, CHUNK_2M)
    fd = -1
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(args.device),
        "slot_count": int(args.slot_count),
        "slot_bytes": int(args.slot_bytes),
        "expected_logical_slot_bytes": int(args.slot_count * args.slot_bytes),
        "expected_reserved_backing_bytes": int(args.slot_count * args.slot_bytes),
        "checks": {},
    }

    try:
        fd, source, init_ret, init_errno, init_rm = open_initialized_uvm_fd(
            prefer_existing=True
        )
        result["uvm_fd"] = {
            "fd_source": source,
            "init_ret": int(init_ret),
            "init_errno": int(init_errno),
            "init_rmStatus": int(init_rm),
        }
        if fd < 0:
            result["passed"] = False
            result["failure_reason"] = "failed to open initialized UVM fd"
            return result

        init_params = UvmInitializeParams()
        init_again_ret, init_again_errno = ioctl(fd, UVM_INITIALIZE_IOCTL,
                                                 init_params)
        result["uvm_fd"]["init_again"] = {
            "ret": int(init_again_ret),
            "errno": int(init_again_errno),
            "rmStatus": int(init_params.rmStatus),
        }

        gpu_uuid = gpu_uuid_for_device(args.device)
        migrate = native_migrate(fd, alloc.address, alloc.byte_count, gpu_uuid)
        result["managed_probe"] = {
            "base": int(alloc.address),
            "bytes": int(alloc.byte_count),
            "migrate_to_gpu": migrate,
        }

        pmm_query = _query_user_pmm_chunk_mask(fd, gpu_uuid)
        result["pmm_user_chunk_sizes"] = pmm_query
        result["checks"]["pmm_has_512k"] = bool(pmm_query["has_512k"])
        result["checks"]["split_2m_to_512k_legal"] = bool(
            pmm_query["has_512k"] and pmm_query["has_2m"]
        )

        result["reset_before"] = _reset_reserved_pool(fd)
        prealloc = _prealloc_reserved_slots(
            fd,
            gpu_uuid,
            pool_id=args.pool_id,
            slot_count=args.slot_count,
            slot_bytes=args.slot_bytes,
        )
        result["prealloc_512k_slots"] = prealloc
        result["reset_after"] = _reset_reserved_pool(fd)

        all_entries_512k = all(
            int(entry["slot_bytes"]) == args.slot_bytes and
            int(entry["physical_bytes"]) == args.slot_bytes and
            int(entry["chunk_count"]) == 1
            for entry in prealloc.get("entries", [])
        )
        result["checks"]["prealloc_success"] = (
            prealloc["ok"] and
            int(prealloc["success_count"]) == args.slot_count and
            int(prealloc["failure_count"]) == 0
        )
        result["checks"]["reserved_backing_is_24m"] = (
            int(prealloc["reserved_backing_bytes"]) ==
            args.slot_count * args.slot_bytes
        )
        result["checks"]["logical_bytes_is_24m"] = (
            int(prealloc["logical_slot_bytes"]) ==
            args.slot_count * args.slot_bytes
        )
        result["checks"]["entries_are_single_512k_chunks"] = all_entries_512k

        result["passed"] = all(result["checks"].values())
        if not result["passed"]:
            failed = [
                key for key, value in result["checks"].items()
                if not value
            ]
            result["failure_reason"] = "failed checks: " + ", ".join(failed)
        return result
    finally:
        if fd >= 0:
            os.close(fd)
        alloc.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--slot-count", type=int, default=48)
    parser.add_argument("--slot-bytes", type=int, default=CHUNK_512K)
    parser.add_argument("--pool-id", type=int, default=5120)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="write the report but exit 0 even when S0 checks fail",
    )
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result.get("passed") or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
