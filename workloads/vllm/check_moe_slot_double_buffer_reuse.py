#!/usr/bin/env python3
"""Validate 2x2MiB MoE slot reuse across six UVM VA blocks.

This probe models a 12MiB managed CPU/UVM allocation split into six 2MiB VA
blocks. It reuses two driver-owned 2MiB VIDMEM slots across three rounds:

  round 0: block 0,1 -> slot 0,1
  round 1: block 2,3 -> slot 0,1
  round 2: block 4,5 -> slot 0,1

For each round it:
  1. copies per-block payloads into the two slot backings with gpu_copy_only,
  2. remaps the corresponding managed VA blocks with slot_va_block_remap,
  3. reads back through the original managed VA with a GPU kernel,
  4. checks that the slot pool still uses only 4MiB of backing.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_moe_slot_remap_ioctl_smoke import (  # noqa: E402
    BACKING_VIDMEM,
    ENTRY_OK,
    MODE_VALUES,
    UVM_MOE_SLOT_REMAP_BATCH_IOCTL,
    UVM_VA_BLOCK_SIZE,
    ManagedAllocation,
    MoeSlotRemapBatchParams,
    batch_to_result,
    cuda_memcpy_host_to_device,
    gpu_uuid_for_device,
    initialize_cuda_device,
    ioctl,
    kernel_readback,
    open_initialized_uvm_fd,
    query_slot_pool,
    read_uint_file,
    reset_slot_pool,
    slot_pool_health_ok,
    write_uint_file,
)

PTE_ENABLE_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable"
PTE_MAX_BYTES_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes"
VIDMEM_BUDGET_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"


def _pattern(length: int, seed: int) -> bytes:
    return bytes(((index + seed) % 251 for index in range(length)))


def _new_batch(mode: str, gpu_uuid, *, validate_managed: bool, count: int = 2) -> MoeSlotRemapBatchParams:
    params = MoeSlotRemapBatchParams()
    params.mode = MODE_VALUES[mode]
    params.count = count
    params.gpu_uuid = gpu_uuid
    params.validate_managed = 1 if validate_managed else 0
    return params


def _fill_two_entries(
    params: MoeSlotRemapBatchParams,
    *,
    base0: int,
    block_bytes: int,
    round_index: int,
    layer_id: int = 42,
) -> None:
    for slot_id in range(2):
        entry = params.entries[slot_id]
        entry.layer_id = layer_id
        entry.expert_id = round_index * 2 + slot_id + 1
        entry.slot_id = slot_id
        entry.role = 3 if slot_id == 0 else 4
        entry.va_base = base0 + slot_id * block_bytes
        entry.bytes = block_bytes
        entry.slot_offset = 0
        entry.flags = 0


def _submit_two_entry_batch(
    fd: int,
    *,
    mode: str,
    gpu_uuid,
    base0: int,
    block_bytes: int,
    round_index: int,
    validate_managed: bool,
) -> dict[str, object]:
    params = _new_batch(mode, gpu_uuid, validate_managed=validate_managed)
    _fill_two_entries(
        params,
        base0=base0,
        block_bytes=block_bytes,
        round_index=round_index,
    )
    t0 = time.perf_counter()
    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = batch_to_result(params, ret, errno_value)
    result["latency_ms"] = elapsed_ms
    result["mode"] = mode
    return result


def _prefill_slots(fd: int, *, gpu_uuid, payload: bytes, block_bytes: int, round_index: int) -> dict[str, object]:
    source = mmap.mmap(-1, len(payload))
    try:
        source.write(payload)
        source.seek(0)
        source_byte = ctypes.c_char.from_buffer(source)
        source_base = ctypes.addressof(source_byte)
        del source_byte
        result = _submit_two_entry_batch(
            fd,
            mode="gpu_copy_only",
            gpu_uuid=gpu_uuid,
            base0=source_base,
            block_bytes=block_bytes,
            round_index=round_index,
            validate_managed=False,
        )
        result["source_base"] = source_base
        result["source_page_aligned"] = source_base % os.sysconf("SC_PAGE_SIZE") == 0
        return result
    finally:
        source.close()


def _assert_batch_ok(result: dict[str, object], *, expected_count: int = 2) -> bool:
    return (
        int(result.get("ret", -1)) == 0
        and int(result.get("rmStatus", -1)) == 0
        and int(result.get("success_count", -1)) == expected_count
        and int(result.get("failure_count", -1)) == 0
        and all(int(status) == ENTRY_OK for status in result.get("entry_statuses", []))
    )


def _round_query_checks(query: dict[str, object], *, block_bytes: int, round_index: int) -> dict[str, object]:
    entries = query.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    mapped_entries = [entry for entry in entries if int(entry.get("pte_mapped", 0)) != 0]
    expected_experts = {round_index * 2 + 1, round_index * 2 + 2}
    mapped_experts = {int(entry.get("owner_expert_id", -1)) for entry in mapped_entries}
    mapped_slots = {int(entry.get("slot_id", -1)) for entry in mapped_entries}
    return {
        "active_slots": int(query.get("active_slots", -1)),
        "backed_slots": int(query.get("backed_slots", -1)),
        "slot_backing_bytes": int(query.get("slot_backing_bytes", -1)),
        "mapped_entries": len(mapped_entries),
        "mapped_slots": sorted(mapped_slots),
        "mapped_experts": sorted(mapped_experts),
        "expected_experts": sorted(expected_experts),
        "slot_backing_is_4mb": int(query.get("slot_backing_bytes", -1)) == 2 * block_bytes,
        "mapped_current_round": mapped_experts == expected_experts and mapped_slots == {0, 1},
        "all_vidmem": all(int(entry.get("backing_kind", -1)) == BACKING_VIDMEM for entry in entries),
        "all_slot_2mb": all(int(entry.get("slot_bytes", -1)) == block_bytes for entry in entries),
        "all_va_block_tracked": all(int(entry.get("pte_va_block_tracked", 0)) == 1 for entry in mapped_entries),
        "all_mode_slot_va_block": all(
            int(entry.get("pte_mode", -1)) == MODE_VALUES["slot_va_block_remap"] for entry in mapped_entries
        ),
        "global_pte_clear_count": int(query.get("pte_clear_count", 0)),
        "global_pte_remap_count": int(query.get("pte_remap_count", 0)),
        "global_copy_count": int(query.get("copy_count", 0)),
        "global_vidmem_evict_count": int(query.get("vidmem_evict_count", 0)),
    }


def _set_param(path: str, value: int, result: dict[str, object], key: str) -> None:
    old = read_uint_file(path)
    result[f"old_{key}"] = old
    if old != value:
        write_uint_file(path, value)
        result[f"set_{key}"] = value
    result[f"current_{key}"] = read_uint_file(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    block_bytes = int(args.block_bytes)
    rounds = 3
    blocks = 6
    total_bytes = blocks * block_bytes
    result: dict[str, object] = {
        "device": args.device,
        "block_bytes": block_bytes,
        "blocks": blocks,
        "rounds": rounds,
        "slot_count": 2,
        "slot_budget_bytes": 2 * block_bytes,
        "round_results": [],
    }

    if block_bytes != UVM_VA_BLOCK_SIZE:
        result["warning"] = "probe is intended for 2MiB UVM VA blocks"
    if block_bytes <= 0 or block_bytes % os.sysconf("SC_PAGE_SIZE") != 0:
        raise ValueError("--block-bytes must be a positive page-size multiple")

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
        _set_param(VIDMEM_BUDGET_PARAM, 2 * block_bytes, result, "vidmem_budget")

        initialize_cuda_device(args.device)
        gpu_uuid = gpu_uuid_for_device(args.device)
        target = ManagedAllocation(args.device, total_bytes + block_bytes)
        base = ((target.address + block_bytes - 1) // block_bytes) * block_bytes
        result.update({
            "managed_alloc_bytes": target.byte_count,
            "managed_raw_base": target.address,
            "managed_base": base,
            "managed_base_2mb_aligned": base % block_bytes == 0,
            "managed_total_bytes": total_bytes,
        })

        old_payload = b"".join(_pattern(block_bytes, 181 + block_index * 7) for block_index in range(blocks))
        cuda_memcpy_host_to_device(base, old_payload)

        fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(prefer_existing=True)
        result.update({
            "fd_source": fd_source,
            "init_ret": init_ret,
            "init_errno": init_errno,
            "init_rmStatus": init_rm_status,
        })
        if fd < 0:
            result["passed"] = False
            result["blocked_reason"] = "failed to open initialized UVM fd"
            return result

        reset_before = reset_slot_pool(fd)
        result["reset_before"] = reset_before
        if not reset_before.get("ok"):
            result["passed"] = False
            result["blocked_reason"] = "slot pool reset failed"
            return result

        all_rounds_ok = True
        previous_clear_counts: list[int] = []
        for round_index in range(rounds):
            first_block = round_index * 2
            round_base = base + first_block * block_bytes
            slot_payload = b"".join(
                _pattern(block_bytes, 37 + (first_block + slot_id) * 19)
                for slot_id in range(2)
            )

            prefill = _prefill_slots(
                fd,
                gpu_uuid=gpu_uuid,
                payload=slot_payload,
                block_bytes=block_bytes,
                round_index=round_index,
            )
            remap = _submit_two_entry_batch(
                fd,
                mode="slot_va_block_remap",
                gpu_uuid=gpu_uuid,
                base0=round_base,
                block_bytes=block_bytes,
                round_index=round_index,
                validate_managed=True,
            )
            readback = kernel_readback(round_base, 2 * block_bytes, args.device)
            query = query_slot_pool(fd, max_entries=8)
            checks = _round_query_checks(query, block_bytes=block_bytes, round_index=round_index)
            readback_matches = readback == slot_payload
            old_matches = readback == old_payload[first_block * block_bytes:(first_block + 2) * block_bytes]
            if round_index > 0:
                previous_clear_counts.append(int(query.get("pte_clear_count", 0)))

            round_ok = (
                _assert_batch_ok(prefill)
                and _assert_batch_ok(remap)
                and int(remap.get("remap_count", -1)) == 2
                and int(remap.get("tlb_invalidate_count", -1)) == 2
                and readback_matches
                and not old_matches
                and bool(query.get("ok", False))
                and bool(checks["slot_backing_is_4mb"])
                and bool(checks["mapped_current_round"])
                and bool(checks["all_vidmem"])
                and bool(checks["all_slot_2mb"])
                and bool(checks["all_va_block_tracked"])
                and bool(checks["all_mode_slot_va_block"])
                and slot_pool_health_ok(query)
            )
            all_rounds_ok = all_rounds_ok and round_ok

            result["round_results"].append({
                "round": round_index,
                "blocks": [first_block + 1, first_block + 2],
                "target_va_base": round_base,
                "prefill": prefill,
                "remap": remap,
                "query": query,
                "checks": checks,
                "gpu_readback_matches_slot_payload": readback_matches,
                "gpu_readback_matches_old_payload": old_matches,
                "gpu_readback_prefix": readback[:64].hex(),
                "expected_prefix": slot_payload[:64].hex(),
                "round_ok": round_ok,
            })

        final_query = query_slot_pool(fd, max_entries=8)
        reset_after = reset_slot_pool(fd)
        result["final_query_before_reset"] = final_query
        result["reset_after"] = reset_after
        result["reuse_checks"] = {
            "final_slot_backing_bytes": int(final_query.get("slot_backing_bytes", -1)),
            "final_active_slots": int(final_query.get("active_slots", -1)),
            "final_backed_slots": int(final_query.get("backed_slots", -1)),
            "total_copy_count": int(final_query.get("copy_count", 0)),
            "total_pte_remap_count": int(final_query.get("pte_remap_count", 0)),
            "total_pte_clear_count": int(final_query.get("pte_clear_count", 0)),
            "clear_count_after_reuse_rounds": previous_clear_counts,
            "no_vidmem_eviction": int(final_query.get("vidmem_evict_count", 0)) == 0,
            "only_two_slots": int(final_query.get("active_slots", -1)) == 2,
            "fixed_4mb_backing": int(final_query.get("slot_backing_bytes", -1)) == 2 * block_bytes,
            "all_six_copies": int(final_query.get("copy_count", 0)) == 6,
            "all_six_remaps": int(final_query.get("pte_remap_count", 0)) == 6,
            "old_mappings_cleared": int(final_query.get("pte_clear_count", 0)) >= 4,
        }
        result["passed"] = (
            all_rounds_ok
            and bool(result["reuse_checks"]["no_vidmem_eviction"])
            and bool(result["reuse_checks"]["only_two_slots"])
            and bool(result["reuse_checks"]["fixed_4mb_backing"])
            and bool(result["reuse_checks"]["all_six_copies"])
            and bool(result["reuse_checks"]["all_six_remaps"])
            and bool(result["reuse_checks"]["old_mappings_cleared"])
            and bool(reset_after.get("ok", False))
        )
        return result
    finally:
        if fd >= 0:
            os.close(fd)
        if target is not None:
            target.close()
        if args.restore_params:
            for path, value in old_params.items():
                try:
                    write_uint_file(path, value)
                except Exception as exc:  # pragma: no cover - best-effort cleanup path
                    result.setdefault("restore_errors", []).append({"path": path, "value": value, "error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-bytes", type=int, default=UVM_VA_BLOCK_SIZE)
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
