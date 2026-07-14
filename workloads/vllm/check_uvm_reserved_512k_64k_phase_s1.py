#!/usr/bin/env python3
"""Phase S1 probe for 512KiB logical reserved slots backed by 64KiB chunks."""

from __future__ import annotations

import argparse
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
    gpu_uuid_for_device,
    initialize_cuda_device,
    ioctl,
    native_migrate,
    open_initialized_uvm_fd,
)
from check_uvm_pmm_512k_phase_s0 import (  # noqa: E402
    CHUNK_2M,
    CHUNK_512K,
    PAGE_64K,
    UVM_INITIALIZE_IOCTL,
    _query_user_pmm_chunk_mask,
)


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


def _reset_reserved_pool(fd: int) -> dict[str, Any]:
    params = ExpertReservedPoolParams()
    params.mode = UVM_EXPERT_RESERVED_POOL_MODE_RESET
    params.max_entries = UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES
    ret, errno_value = ioctl(fd, UVM_EXPERT_RESERVED_POOL_IOCTL, params)
    return _reserved_pool_result(params, ret, errno_value)


def _prealloc_reserved_slots(fd: int,
                             gpu_uuid,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    initialize_cuda_device(args.device)
    alloc = ManagedAllocation(args.device, CHUNK_2M)
    fd = -1
    expected_bytes = int(args.slot_count * args.slot_bytes)
    expected_chunks = int(args.slot_bytes // args.physical_chunk_bytes)
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(args.device),
        "slot_count": int(args.slot_count),
        "slot_bytes": int(args.slot_bytes),
        "physical_chunk_bytes": int(args.physical_chunk_bytes),
        "expected_chunk_count_per_slot": expected_chunks,
        "expected_logical_slot_bytes": expected_bytes,
        "expected_reserved_backing_bytes": expected_bytes,
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
        result["reset_before"] = _reset_reserved_pool(fd)
        prealloc = _prealloc_reserved_slots(
            fd,
            gpu_uuid,
            pool_id=args.pool_id,
            slot_count=args.slot_count,
            slot_bytes=args.slot_bytes,
        )
        result["prealloc_512k_logical_64k_physical"] = prealloc
        result["reset_after"] = _reset_reserved_pool(fd)

        entries = prealloc.get("entries", [])
        result["checks"]["prealloc_success"] = (
            prealloc["ok"] and
            int(prealloc["success_count"]) == args.slot_count and
            int(prealloc["failure_count"]) == 0
        )
        result["checks"]["logical_bytes_is_24m"] = (
            int(prealloc["logical_slot_bytes"]) == expected_bytes
        )
        result["checks"]["reserved_backing_is_24m"] = (
            int(prealloc["reserved_backing_bytes"]) == expected_bytes
        )
        result["checks"]["summary_chunk_size_is_64k"] = (
            int(prealloc["chunk_size"]) == args.physical_chunk_bytes
        )
        # UVM_TEST_PMM_QUERY is a diagnostic ioctl and can be unavailable in
        # some rebuilt module states. Treat it as advisory once reserved
        # prealloc itself proves that 64KiB chunks were allocated.
        result["checks"]["pmm_has_or_prealloc_uses_64k"] = bool(
            (pmm_query["ok"] and pmm_query["has_64k"]) or
            result["checks"]["summary_chunk_size_is_64k"]
        )
        result["checks"]["entry_count_matches"] = len(entries) == args.slot_count
        result["checks"]["entries_are_512k_slots_8x64k"] = all(
            int(entry["status"]) == 0 and
            int(entry["slot_bytes"]) == args.slot_bytes and
            int(entry["physical_bytes"]) == args.slot_bytes and
            int(entry["chunk_count"]) == expected_chunks
            for entry in entries
        )

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
    parser.add_argument("--physical-chunk-bytes", type=int, default=PAGE_64K)
    parser.add_argument("--pool-id", type=int, default=5121)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    if args.slot_bytes % args.physical_chunk_bytes != 0:
        raise SystemExit("slot-bytes must be a multiple of physical-chunk-bytes")

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result.get("passed") or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
