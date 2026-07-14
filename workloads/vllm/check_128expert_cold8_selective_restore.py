#!/usr/bin/env python3
"""Synthetic 128-expert smoke for 512KiB release + selective slot restore.

This test does not start vLLM. It models one MoE layer with 128 experts, each
represented by one 512KiB managed range. Eight cold experts are released to CPU
with the S2 64KiB-subchunk path, a small reserved-slot pool is preallocated, and
only the experts predicted to be accessed are restored through reserved slots.
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


DEFAULT_COLD_IDS = "3,17,31,47,63,79,95,111"
DEFAULT_SELECTED_IDS = "17,63,111"


def _parse_ids(value: str) -> list[int]:
    ids: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item, 0))
    return ids


def _expert_payload(expert_id: int, byte_count: int) -> bytes:
    return _pattern(byte_count, (expert_id * 37 + 11) % 251)


def _region_id(base_region_id: int,
               expert_id: int,
               chunk_id: int,
               chunks_per_expert: int) -> int:
    return int(base_region_id) + int(expert_id) * int(chunks_per_expert) + int(chunk_id)


def _expert_base(base: int, expert_id: int, expert_bytes: int) -> int:
    return int(base) + int(expert_id) * int(expert_bytes)


def _chunk_base(base: int,
                expert_id: int,
                chunk_id: int,
                expert_bytes: int,
                chunk_bytes: int) -> int:
    return _expert_base(base, expert_id, expert_bytes) + int(chunk_id) * int(chunk_bytes)


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


def _reserved_pool_prealloc(fd: int,
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


def _reserved_restore_one(fd: int,
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


def _reserved_invalidate_one(fd: int,
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


def _query_experts(fd: int,
                   gpu_uuid,
                   *,
                   base: int,
                   expert_ids: list[int],
                   expert_bytes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expert_id in expert_ids:
        query = _query_residency(
            fd,
            gpu_uuid,
            _expert_base(base, expert_id, expert_bytes),
            expert_bytes,
        )
        query["expert_id"] = int(expert_id)
        rows.append(query)
    return rows


def _summarize_rows(rows: list[dict[str, Any]],
                    expected_bytes: int) -> dict[str, Any]:
    return {
        "count": len(rows),
        "gpu_bytes": sum(int(row.get("gpu_bytes") or 0) for row in rows),
        "cpu_bytes": sum(int(row.get("cpu_bytes") or 0) for row in rows),
        "all_ok": all(bool(row.get("ok")) for row in rows),
        "all_gpu_zero": all(int(row.get("gpu_bytes") or 0) == 0 for row in rows),
        "all_cpu_expected": all(
            int(row.get("cpu_bytes") or 0) == expected_bytes for row in rows
        ),
        "all_gpu_expected": all(
            int(row.get("gpu_bytes") or 0) == expected_bytes for row in rows
        ),
    }


def _read_and_check_expert(device: int,
                           base: int,
                           expert_id: int,
                           expert_bytes: int) -> bool:
    addr = _expert_base(base, expert_id, expert_bytes)
    return kernel_readback(addr, expert_bytes, device) == _expert_payload(
        expert_id, expert_bytes
    )


def _cpu_read_and_check_expert(base: int,
                               expert_id: int,
                               expert_bytes: int) -> bool:
    addr = _expert_base(base, expert_id, expert_bytes)
    return ctypes.string_at(addr, expert_bytes) == _expert_payload(
        expert_id, expert_bytes
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    expert_count = int(args.expert_count)
    chunks_per_expert = int(args.chunks_per_expert)
    chunk_bytes = int(args.chunk_bytes)
    expert_bytes = chunks_per_expert * chunk_bytes
    total_bytes = expert_count * expert_bytes
    cold_ids = _parse_ids(args.cold_ids)
    selected_ids = _parse_ids(args.selected_ids)
    cold_set = set(cold_ids)
    selected_set = set(selected_ids)
    non_cold_ids = [expert_id for expert_id in range(expert_count)
                    if expert_id not in cold_set]
    cold_not_selected_ids = [expert_id for expert_id in cold_ids
                             if expert_id not in selected_set]
    pool_id = int(args.pool_id)
    slot_count = int(args.slot_count or (len(cold_ids) * chunks_per_expert))
    fd = -1
    alloc: ManagedAllocation | None = None
    registered_region_ids: list[int] = []
    restored_slot_ids: list[int] = []

    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(args.device),
        "expert_count": expert_count,
        "expert_bytes": expert_bytes,
        "chunks_per_expert": chunks_per_expert,
        "chunk_bytes": chunk_bytes,
        "total_bytes": total_bytes,
        "cold_ids": cold_ids,
        "selected_restore_ids": selected_ids,
        "cold_not_selected_ids": cold_not_selected_ids,
        "pool_id": pool_id,
        "slot_count": slot_count,
        "slot_bytes": chunk_bytes,
        "physical_chunk_bytes": int(PAGE_64K),
        "expected_slot_chunk_count": int(chunk_bytes // PAGE_64K),
        "base_region_id": int(args.base_region_id),
    }

    try:
        if expert_count != 128:
            raise ValueError("this smoke is intentionally fixed to 128 experts")
        if chunks_per_expert < 1:
            raise ValueError("chunks_per_expert must be positive")
        if chunk_bytes != CHUNK_512K:
            raise ValueError("this smoke uses 512KiB release/restore chunks")
        if expert_bytes % PAGE_64K != 0:
            raise ValueError("logical expert bytes must be 64KiB aligned")
        if any(expert_id < 0 or expert_id >= expert_count for expert_id in cold_ids):
            raise ValueError("cold expert id out of range")
        if len(cold_ids) != len(cold_set):
            raise ValueError("duplicate cold expert ids")
        if any(expert_id not in cold_set for expert_id in selected_ids):
            raise ValueError("selected experts must be a subset of cold experts")
        if len(selected_ids) * chunks_per_expert > slot_count:
            raise ValueError("slot_count must cover selected expert chunks")
        if slot_count > UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES:
            raise ValueError("slot_count exceeds ioctl max entries")

        initialize_cuda_device(args.device)
        gpu_uuid = gpu_uuid_for_device(args.device)
        alloc = ManagedAllocation(args.device, total_bytes + UVM_VA_BLOCK_SIZE)
        base = ((alloc.address + UVM_VA_BLOCK_SIZE - 1) // UVM_VA_BLOCK_SIZE) * UVM_VA_BLOCK_SIZE
        result.update({
            "managed_raw_base": int(alloc.address),
            "managed_base": int(base),
            "managed_base_2mb_aligned": bool(base % UVM_VA_BLOCK_SIZE == 0),
            "total_bytes_multiple_of_2mb": bool(total_bytes % UVM_VA_BLOCK_SIZE == 0),
        })

        payload = bytearray(total_bytes)
        for expert_id in range(expert_count):
            start = expert_id * expert_bytes
            payload[start:start + expert_bytes] = _expert_payload(
                expert_id, expert_bytes
            )
        cuda_memcpy_host_to_device(base, bytes(payload))
        del payload

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
        result["native_migrate_all_experts_to_gpu"] = native_migrate(
            fd, base, total_bytes, gpu_uuid
        )
        before_whole = _query_residency(fd, gpu_uuid, base, total_bytes)
        before_cold_rows = _query_experts(
            fd, gpu_uuid, base=base, expert_ids=cold_ids, expert_bytes=expert_bytes
        )
        before_non_cold_rows = _query_experts(
            fd, gpu_uuid, base=base, expert_ids=non_cold_ids,
            expert_bytes=expert_bytes
        )
        result["residency_before_whole"] = before_whole
        result["residency_before_cold_summary"] = _summarize_rows(
            before_cold_rows, expert_bytes
        )
        result["residency_before_non_cold_summary"] = _summarize_rows(
            before_non_cold_rows, expert_bytes
        )

        register_results: list[dict[str, Any]] = []
        for expert_id in cold_ids:
            for chunk_id in range(chunks_per_expert):
                region_id = _region_id(args.base_region_id, expert_id,
                                       chunk_id, chunks_per_expert)
                registered_region_ids.append(region_id)
                register = _register_expert_region(
                    fd,
                    region_id=region_id,
                    base=_chunk_base(base, expert_id, chunk_id,
                                     expert_bytes, chunk_bytes),
                    byte_count=chunk_bytes,
                )
                register["expert_id"] = int(expert_id)
                register["chunk_id"] = int(chunk_id)
                register_results.append(register)
                if not register["ok"]:
                    result["register_cold_regions"] = register_results
                    result["passed"] = False
                    result["blocked_reason"] = (
                        f"register failed for expert {expert_id} chunk {chunk_id}"
                    )
                    return result
        result["register_cold_regions"] = register_results

        release_results: list[dict[str, Any]] = []
        for expert_id in cold_ids:
            for chunk_id in range(chunks_per_expert):
                release = _release_region_subchunk_64k(
                    fd,
                    gpu_uuid,
                    _region_id(args.base_region_id, expert_id,
                               chunk_id, chunks_per_expert),
                )
                release["expert_id"] = int(expert_id)
                release["chunk_id"] = int(chunk_id)
                release_results.append(release)
                if not release["ok"]:
                    result["release_cold_regions"] = release_results
                    result["passed"] = False
                    result["blocked_reason"] = (
                        f"release failed for expert {expert_id} chunk {chunk_id}"
                    )
                    return result
        result["release_cold_regions"] = release_results

        after_release_whole = _query_residency(fd, gpu_uuid, base, total_bytes)
        after_release_cold_rows = _query_experts(
            fd, gpu_uuid, base=base, expert_ids=cold_ids, expert_bytes=expert_bytes
        )
        after_release_non_cold_rows = _query_experts(
            fd, gpu_uuid, base=base, expert_ids=non_cold_ids,
            expert_bytes=expert_bytes
        )
        result["residency_after_release_whole"] = after_release_whole
        result["residency_after_release_cold_summary"] = _summarize_rows(
            after_release_cold_rows, expert_bytes
        )
        result["residency_after_release_non_cold_summary"] = _summarize_rows(
            after_release_non_cold_rows, expert_bytes
        )
        result["cold_cpu_payload_after_release_ok"] = {
            str(expert_id): _cpu_read_and_check_expert(base, expert_id, expert_bytes)
            for expert_id in cold_ids
        }

        prealloc = _reserved_pool_prealloc(
            fd,
            gpu_uuid,
            pool_id=pool_id,
            slot_count=slot_count,
            slot_bytes=chunk_bytes,
        )
        result["expert_reserved_prealloc"] = prealloc
        if not (
            prealloc.get("ok") is True
            and prealloc.get("success_count") == slot_count
            and prealloc.get("failure_count") == 0
        ):
            result["passed"] = False
            result["blocked_reason"] = "reserved slot prealloc failed"
            return result

        restore_results: list[dict[str, Any]] = []
        next_slot_id = 0
        for expert_id in selected_ids:
            for chunk_id in range(chunks_per_expert):
                slot_id = next_slot_id
                next_slot_id += 1
                restore = _reserved_restore_one(
                    fd,
                    gpu_uuid,
                    region_id=_region_id(args.base_region_id, expert_id,
                                         chunk_id, chunks_per_expert),
                    pool_id=pool_id,
                    slot_id=slot_id,
                    byte_count=chunk_bytes,
                )
                restore["expert_id"] = int(expert_id)
                restore["chunk_id"] = int(chunk_id)
                restore["slot_id"] = int(slot_id)
                restore_results.append(restore)
                restored_slot_ids.append(slot_id)
                if not restore["ok"]:
                    result["selective_restore_results"] = restore_results
                    result["passed"] = False
                    result["blocked_reason"] = (
                        f"restore failed for expert {expert_id} chunk {chunk_id}"
                    )
                    return result
        result["selective_restore_results"] = restore_results

        result["selected_gpu_read_payload_ok"] = {
            str(expert_id): _read_and_check_expert(
                args.device, base, expert_id, expert_bytes
            )
            for expert_id in selected_ids
        }

        result["non_cold_gpu_read_payload_ok"] = {
            str(expert_id): _read_and_check_expert(
                args.device, base, expert_id, expert_bytes
            )
            for expert_id in non_cold_ids
        }

        after_selected_reads_cold_not_selected_rows = _query_experts(
            fd,
            gpu_uuid,
            base=base,
            expert_ids=cold_not_selected_ids,
            expert_bytes=expert_bytes,
        )
        after_selected_reads_non_cold_rows = _query_experts(
            fd,
            gpu_uuid,
            base=base,
            expert_ids=non_cold_ids,
            expert_bytes=expert_bytes,
        )
        after_restore_selected_rows = _query_experts(
            fd,
            gpu_uuid,
            base=base,
            expert_ids=selected_ids,
            expert_bytes=expert_bytes,
        )
        result["residency_after_selected_reads_cold_not_selected_summary"] = (
            _summarize_rows(after_selected_reads_cold_not_selected_rows, expert_bytes)
        )
        result["residency_after_selected_reads_non_cold_summary"] = (
            _summarize_rows(after_selected_reads_non_cold_rows, expert_bytes)
        )
        result["residency_after_restore_selected_summary"] = _summarize_rows(
            after_restore_selected_rows, expert_bytes
        )
        result["note"] = (
            "slot-backed readonly restore may not appear as ordinary GPU "
            "residency; selected experts are validated by restore counters and "
            "GPU readback from the original VA."
        )

        invalidate_results: list[dict[str, Any]] = []
        for slot_id in restored_slot_ids:
            invalidate = _reserved_invalidate_one(
                fd, gpu_uuid, pool_id=pool_id, slot_id=slot_id
            )
            invalidate["slot_id"] = int(slot_id)
            invalidate_results.append(invalidate)
        result["invalidate_restored_slots"] = invalidate_results
        result["reserved_reset_after"] = _reserved_pool_reset(fd, gpu_uuid)

        prealloc_entries = prealloc.get("entries") or []
        restore_success_bytes = sum(
            int(row.get("accepted_bytes") or 0) for row in restore_results
        )
        result["checks"] = {
            "before_whole_gpu_resident": before_whole.get("gpu_bytes") == total_bytes,
            "before_all_cold_gpu_resident": result["residency_before_cold_summary"]["all_gpu_expected"],
            "before_all_non_cold_gpu_resident": result["residency_before_non_cold_summary"]["all_gpu_expected"],
            "all_cold_release_ok": all(
                row.get("ok") is True
                and row.get("released_bytes") == chunk_bytes
                and row.get("failure_blocks") == 0
                for row in release_results
            ),
            "after_release_whole_gpu_matches_non_cold": (
                after_release_whole.get("gpu_bytes")
                == (expert_count - len(cold_ids)) * expert_bytes
            ),
            "after_release_whole_cpu_covers_cold": (
                int(after_release_whole.get("cpu_bytes") or 0)
                >= len(cold_ids) * expert_bytes
            ),
            "after_release_cold_all_cpu": result["residency_after_release_cold_summary"]["all_cpu_expected"],
            "after_release_cold_all_gpu_zero": result["residency_after_release_cold_summary"]["all_gpu_zero"],
            "after_release_non_cold_all_gpu": result["residency_after_release_non_cold_summary"]["all_gpu_expected"],
            "cold_cpu_payload_ok": all(result["cold_cpu_payload_after_release_ok"].values()),
            "slot_prealloc_ok": (
                prealloc.get("ok") is True
                and prealloc.get("success_count") == slot_count
                and prealloc.get("failure_count") == 0
            ),
            "slot_prealloc_512k_64k": (
                prealloc.get("reserved_backing_bytes") == slot_count * chunk_bytes
                and prealloc.get("logical_slot_bytes") == slot_count * chunk_bytes
                and prealloc.get("chunk_size") == PAGE_64K
                and len(prealloc_entries) == slot_count
                and all(
                    entry.get("slot_bytes") == chunk_bytes
                    and entry.get("physical_bytes") == chunk_bytes
                    and entry.get("chunk_count") == chunk_bytes // PAGE_64K
                    and entry.get("status") == 0
                    for entry in prealloc_entries
                )
            ),
            "selective_restore_only_selected_count": (
                len(restore_results) == len(selected_ids) * chunks_per_expert
            ),
            "selective_restore_all_ok": all(
                row.get("ok") is True
                and row.get("success_count") == 1
                and row.get("failure_count") == 0
                and row.get("accepted_bytes") == chunk_bytes
                and row.get("assigned_slots") == 1
                and (row.get("entry") or {}).get("resolved_bytes") == chunk_bytes
                and (row.get("entry") or {}).get("physical_bytes") == chunk_bytes
                for row in restore_results
            ),
            "selective_restore_bytes_match": restore_success_bytes == len(selected_ids) * expert_bytes,
            "selected_gpu_payload_ok": all(result["selected_gpu_read_payload_ok"].values()),
            "non_cold_gpu_payload_ok": all(result["non_cold_gpu_read_payload_ok"].values()),
            "cold_not_selected_still_gpu_zero": (
                result["residency_after_selected_reads_cold_not_selected_summary"]["all_gpu_zero"]
            ),
            "cold_not_selected_still_cpu": (
                result["residency_after_selected_reads_cold_not_selected_summary"]["all_cpu_expected"]
            ),
            "non_cold_still_gpu_after_reads": (
                result["residency_after_selected_reads_non_cold_summary"]["all_gpu_expected"]
            ),
            "invalidate_ok": all(row.get("ok") is True for row in invalidate_results),
            "reset_after_ok": result["reserved_reset_after"].get("ok") is True,
        }
        result["passed"] = all(bool(value) for value in result["checks"].values())
        if not result["passed"]:
            result["failure_reason"] = "failed checks: " + ", ".join(
                key for key, value in result["checks"].items() if not value
            )
        return result
    finally:
        if fd >= 0:
            for slot_id in restored_slot_ids:
                try:
                    _reserved_invalidate_one(fd, gpu_uuid, pool_id=pool_id,
                                             slot_id=slot_id)
                except Exception:
                    pass
            try:
                _reserved_pool_reset(fd, gpu_uuid)
            except Exception:
                pass
            for region_id in registered_region_ids:
                try:
                    _unregister_region(fd, region_id)
                except Exception:
                    pass
            os.close(fd)
        if alloc is not None:
            alloc.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expert-count", type=int, default=128)
    parser.add_argument("--chunks-per-expert", type=int, default=1)
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_512K)
    parser.add_argument("--cold-ids", default=DEFAULT_COLD_IDS)
    parser.add_argument("--selected-ids", default=DEFAULT_SELECTED_IDS)
    parser.add_argument("--pool-id", type=int, default=5128)
    parser.add_argument("--slot-count", type=int, default=0)
    parser.add_argument("--base-region-id", type=lambda value: int(value, 0),
                        default=0x5128000000)
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
