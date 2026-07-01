#!/usr/bin/env python3
"""Minimal kernel-boundary UVM MoE slot VA remap probe.

The probe verifies the mechanism we want for stage-boundary expert promotion:

1. Allocate a managed target range and fill it with an "old" byte pattern.
2. Run a GPU read kernel against the target VA.
3. Prefill a MoE slot with a distinct "new" byte pattern.
4. Install a slot_va_block_remap for the same target VA.
5. Run the same GPU read kernel against the same VA and verify it observes
   the slot pattern.

This is intentionally outside vLLM and the MoE microbench. It only answers:
"Can a kernel boundary see a new backing through the same virtual address, and
what is the remap ioctl latency?"
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import time
from pathlib import Path

from check_moe_slot_remap_ioctl_smoke import (
    ENTRY_OK,
    MODE_VALUES,
    UVM_MOE_SLOT_REMAP_BATCH_IOCTL,
    UVM_VA_BLOCK_SIZE,
    MoeSlotRemapBatchParams,
    ManagedAllocation,
    align_up,
    batch_to_result,
    cuda_memcpy_host_to_device,
    fill_remap_entries,
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


def _pattern(length: int, seed: int) -> bytes:
    return bytes(((index + seed) % 251 for index in range(length)))


def _prefill_slot(
    fd: int,
    *,
    gpu_uuid,
    slot_payload: bytes,
    remap_bytes: int,
) -> dict[str, object]:
    source = mmap.mmap(-1, len(slot_payload))
    try:
        source.write(slot_payload)
        source.seek(0)
        source_byte = ctypes.c_char.from_buffer(source)
        source_base = ctypes.addressof(source_byte)
        del source_byte
        params = MoeSlotRemapBatchParams()
        fill_remap_entries(
            params,
            mode=MODE_VALUES["gpu_copy_only"],
            count=1,
            gpu_uuid=gpu_uuid,
            base=source_base,
            page_size=remap_bytes,
            validate_managed=False,
        )
        t0 = time.perf_counter()
        ret, errno_value = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
        elapsed = time.perf_counter() - t0
        result = batch_to_result(params, ret, errno_value)
        result["seconds"] = elapsed
        result["latency_ms"] = elapsed * 1000.0
        result["source_base"] = source_base
        result["source_page_aligned"] = source_base % os.sysconf("SC_PAGE_SIZE") == 0
        return result
    finally:
        source.close()


def _install_slot_va_block_remap(
    fd: int,
    *,
    gpu_uuid,
    base: int,
    remap_bytes: int,
) -> dict[str, object]:
    params = MoeSlotRemapBatchParams()
    fill_remap_entries(
        params,
        mode=MODE_VALUES["slot_va_block_remap"],
        count=1,
        gpu_uuid=gpu_uuid,
        base=base,
        page_size=remap_bytes,
        validate_managed=True,
    )
    t0 = time.perf_counter()
    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
    elapsed = time.perf_counter() - t0
    result = batch_to_result(params, ret, errno_value)
    result["seconds"] = elapsed
    result["latency_ms"] = elapsed * 1000.0
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    if args.remap_bytes <= 0 or args.remap_bytes % page_size != 0:
        raise ValueError("--remap-bytes must be a positive multiple of page size")

    result: dict[str, object] = {
        "device": args.device,
        "remap_bytes": args.remap_bytes,
        "page_size": page_size,
        "pte_enable_param": PTE_ENABLE_PARAM,
        "pte_max_bytes_param": PTE_MAX_BYTES_PARAM,
    }

    old_enable = read_uint_file(PTE_ENABLE_PARAM)
    old_max_bytes = read_uint_file(PTE_MAX_BYTES_PARAM)
    result["old_pte_enable"] = old_enable
    result["old_pte_max_bytes"] = old_max_bytes

    fd = -1
    target: ManagedAllocation | None = None
    try:
        if args.enable_pte_remap and old_enable == 0:
            write_uint_file(PTE_ENABLE_PARAM, 1)
            result["enabled_pte_remap_for_probe"] = True
        if old_max_bytes < args.remap_bytes:
            write_uint_file(PTE_MAX_BYTES_PARAM, args.remap_bytes)
            result["raised_pte_max_bytes_for_probe"] = True

        current_enable = read_uint_file(PTE_ENABLE_PARAM)
        current_max_bytes = read_uint_file(PTE_MAX_BYTES_PARAM)
        result["current_pte_enable"] = current_enable
        result["current_pte_max_bytes"] = current_max_bytes
        if current_enable == 0:
            result["passed"] = False
            result["blocked_reason"] = "uvm_moe_slot_pte_remap_enable is 0"
            return result

        initialize_cuda_device(args.device)
        gpu_uuid = gpu_uuid_for_device(args.device)
        alloc_bytes = args.remap_bytes + UVM_VA_BLOCK_SIZE + args.remap_bytes
        target = ManagedAllocation(args.device, alloc_bytes)
        base = align_up(target.address, args.remap_bytes)
        result.update(
            {
                "managed_alloc_bytes": alloc_bytes,
                "managed_raw_base": target.address,
                "managed_base": base,
                "managed_base_aligned": base % args.remap_bytes == 0,
                "managed_crosses_va_block": (
                    base // UVM_VA_BLOCK_SIZE
                    != (base + args.remap_bytes - 1) // UVM_VA_BLOCK_SIZE
                ),
            }
        )

        fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(
            prefer_existing=True,
        )
        result.update(
            {
                "fd_source": fd_source,
                "init_ret": init_ret,
                "init_errno": init_errno,
                "init_rmStatus": init_rm_status,
            }
        )
        if fd < 0:
            result["passed"] = False
            result["blocked_reason"] = "failed to open initialized UVM fd"
            return result

        result["reset_before"] = reset_slot_pool(fd)
        old_payload = _pattern(args.remap_bytes, 181)
        slot_payload = _pattern(args.remap_bytes, 37)

        cuda_memcpy_host_to_device(base, old_payload)

        t0 = time.perf_counter()
        before = kernel_readback(base, args.remap_bytes, args.device)
        result["kernel_read_before_ms"] = (time.perf_counter() - t0) * 1000.0
        result["before_matches_old"] = before == old_payload
        result["before_matches_slot"] = before == slot_payload
        result["before_prefix"] = before[:64].hex()

        prefill = _prefill_slot(
            fd,
            gpu_uuid=gpu_uuid,
            slot_payload=slot_payload,
            remap_bytes=args.remap_bytes,
        )
        result["slot_prefill_gpu_copy_only"] = prefill

        remap = _install_slot_va_block_remap(
            fd,
            gpu_uuid=gpu_uuid,
            base=base,
            remap_bytes=args.remap_bytes,
        )
        result["slot_va_block_remap"] = remap
        result["slot_pool_after_remap"] = query_slot_pool(fd)

        t0 = time.perf_counter()
        after = kernel_readback(base, args.remap_bytes, args.device)
        result["kernel_read_after_ms"] = (time.perf_counter() - t0) * 1000.0
        result["after_matches_old"] = after == old_payload
        result["after_matches_slot"] = after == slot_payload
        result["after_prefix"] = after[:64].hex()

        result["reset_after"] = reset_slot_pool(fd)
        t0 = time.perf_counter()
        after_reset = kernel_readback(base, args.remap_bytes, args.device)
        result["kernel_read_after_reset_ms"] = (time.perf_counter() - t0) * 1000.0
        result["after_reset_matches_old"] = after_reset == old_payload
        result["after_reset_prefix"] = after_reset[:64].hex()

        prefill_ok = (
            prefill["ret"] == 0
            and prefill["rmStatus"] == 0
            and prefill["success_count"] == 1
            and prefill["failure_count"] == 0
            and prefill["entry_statuses"] == [ENTRY_OK]
        )
        remap_ok = (
            remap["ret"] == 0
            and remap["rmStatus"] == 0
            and remap["success_count"] == 1
            and remap["failure_count"] == 0
            and remap["remap_count"] == 1
            and remap["tlb_invalidate_count"] == 1
            and remap["entry_statuses"] == [ENTRY_OK]
        )
        result["passed"] = (
            prefill_ok
            and remap_ok
            and result["before_matches_old"]
            and result["after_matches_slot"]
            and result["after_reset_matches_old"]
            and slot_pool_health_ok(result["slot_pool_after_remap"])
            and result["reset_after"]["ok"]
        )
        return result
    finally:
        if fd >= 0:
            os.close(fd)
        if target is not None:
            target.close()
        if args.restore_params:
            try:
                write_uint_file(PTE_ENABLE_PARAM, old_enable)
                write_uint_file(PTE_MAX_BYTES_PARAM, old_max_bytes)
            except Exception as exc:
                result["restore_error"] = str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--remap-bytes", type=int, default=65536)
    parser.add_argument("--enable-pte-remap", action="store_true")
    parser.add_argument("--restore-params", action="store_true")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
