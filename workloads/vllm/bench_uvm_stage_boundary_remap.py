#!/usr/bin/env python3
"""Two-stage kernel-boundary remap probe.

The probe simulates:

  w13 kernel on a managed/UVA range
  concurrent P1 copy into either a driver-owned VIDMEM slot or a managed
  user-backing range
  kernel boundary
  optional slot_va_block_remap
  w2 kernel through the same target VA
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import statistics
import subprocess
import time
from pathlib import Path

from check_moe_slot_remap_ioctl_smoke import (
    ENTRY_OK,
    MODE_VALUES,
    UVM_MOE_SLOT_REMAP_BATCH_IOCTL,
    UVM_MOE_SLOT_REMAP_MAX_ENTRIES,
    UVM_VA_BLOCK_SIZE,
    ManagedAllocation,
    MoeSlotRemapBatchParams,
    align_up,
    batch_to_result,
    gpu_uuid_for_device,
    initialize_cuda_device,
    ioctl,
    open_initialized_uvm_fd,
    query_slot_pool,
    reset_slot_pool,
    slot_pool_health_ok,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROBE_CU = SCRIPT_DIR / "uvm_stage_boundary_probe.cu"
PROBE_SO = SCRIPT_DIR / "uvm_stage_boundary_probe.so"
PTE_ENABLE_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable"
PTE_MAX_BYTES_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes"


def _uses_user_backing(args: argparse.Namespace) -> bool:
    return args.copy_backend == "async_user_backing"


def _uses_driver_vidmem_slot(args: argparse.Namespace) -> bool:
    return args.copy_backend in ("gpu_copy_only", "driver_vidmem_slot", "driver_vidmem_async")


def _uses_driver_vidmem_async(args: argparse.Namespace) -> bool:
    return args.copy_backend == "driver_vidmem_async"


def _allocate_w2_layout(
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    *,
    fill_backing: bool,
    allocate_sources: bool = True,
) -> tuple[
    list[ManagedAllocation],
    list[ManagedAllocation],
    list[int],
    list[int],
    list[object],
    list[int],
]:
    w2_allocs: list[ManagedAllocation] = []
    backing_allocs: list[ManagedAllocation] = []
    w2_bases: list[int] = []
    backing_bases: list[int] = []
    sources: list[object] = []
    source_addrs: list[int] = []

    if args.weight_layout == "layer_tensor":
        # vLLM stores MoE weights as layer-level tensors whose leading
        # dimension is expert id, so selected experts are offsets into one
        # contiguous allocation instead of independent allocations.
        layer_bytes = args.num_experts * args.w2_stride
        w2_alloc = ManagedAllocation(args.device, layer_bytes + 2 * UVM_VA_BLOCK_SIZE)
        w2_allocs.append(w2_alloc)
        w2_layer_base = align_up(w2_alloc.address, UVM_VA_BLOCK_SIZE)
        ctypes.memset(w2_layer_base, args.old_pattern, layer_bytes)
        if _uses_user_backing(args):
            backing_alloc = ManagedAllocation(args.device, layer_bytes + 2 * UVM_VA_BLOCK_SIZE)
            backing_allocs.append(backing_alloc)
            backing_layer_base = align_up(backing_alloc.address, UVM_VA_BLOCK_SIZE)
            if fill_backing:
                ctypes.memset(backing_layer_base, 0, layer_bytes)
        else:
            backing_layer_base = 0

        for index in range(args.batch_count):
            expert_id = args.start_expert + index
            w2_bases.append(w2_layer_base + expert_id * args.w2_stride)
            if _uses_user_backing(args):
                backing_bases.append(backing_layer_base + expert_id * args.w2_stride)
                if allocate_sources:
                    source_ptr = _make_pinned_host(lib, args.w2_bytes, args.slot_pattern)
                    sources.append(source_ptr)
                    source_addrs.append(int(source_ptr.value or 0))
            else:
                if allocate_sources:
                    source, source_addr = _make_mmap(args.w2_bytes, args.slot_pattern)
                    sources.append(source)
                    source_addrs.append(source_addr)
                backing_bases.append(0)
    else:
        for index in range(args.batch_count):
            w2_alloc = ManagedAllocation(args.device, args.w2_bytes + UVM_VA_BLOCK_SIZE)
            w2_allocs.append(w2_alloc)
            w2_base = align_up(w2_alloc.address, os.sysconf("SC_PAGE_SIZE"))
            ctypes.memset(w2_base, args.old_pattern, args.w2_bytes)
            w2_bases.append(w2_base)

            if _uses_user_backing(args):
                backing_alloc = ManagedAllocation(args.device, args.w2_bytes + UVM_VA_BLOCK_SIZE)
                backing_allocs.append(backing_alloc)
                backing_base = align_up(backing_alloc.address, os.sysconf("SC_PAGE_SIZE"))
                if fill_backing:
                    ctypes.memset(backing_base, 0, args.w2_bytes)
                backing_bases.append(backing_base)
                if allocate_sources:
                    source_ptr = _make_pinned_host(lib, args.w2_bytes, args.slot_pattern)
                    sources.append(source_ptr)
                    source_addrs.append(int(source_ptr.value or 0))
            else:
                if allocate_sources:
                    source, source_addr = _make_mmap(args.w2_bytes, args.slot_pattern)
                    sources.append(source)
                    source_addrs.append(source_addr)
                backing_bases.append(0)

    return w2_allocs, backing_allocs, w2_bases, backing_bases, sources, source_addrs


def _va_block_end_exclusive(address: int) -> int:
    return ((address // UVM_VA_BLOCK_SIZE) + 1) * UVM_VA_BLOCK_SIZE


def _split_one_expert_into_slices(
    *,
    expert_index: int,
    target_base: int,
    backing_base: int,
    byte_count: int,
    use_user_backing: bool,
) -> list[dict[str, int]]:
    slices: list[dict[str, int]] = []
    offset = 0
    while offset < byte_count:
        target = target_base + offset
        target_limit = _va_block_end_exclusive(target)
        slice_limit = min(byte_count, offset + (target_limit - target))
        if use_user_backing:
            backing = backing_base + offset
            backing_limit = _va_block_end_exclusive(backing)
            slice_limit = min(slice_limit, offset + (backing_limit - backing))
        else:
            backing = 0

        slice_bytes = slice_limit - offset
        if slice_bytes <= 0:
            raise RuntimeError("failed to split remap entry at VA block boundary")

        slices.append(
            {
                "expert_index": expert_index,
                "expert_id": 7 + expert_index,
                "slot_id": expert_index,
                "target_va": target,
                "backing_va": backing,
                "offset": offset,
                "bytes": slice_bytes,
                "target_block": target // UVM_VA_BLOCK_SIZE,
                "backing_block": backing // UVM_VA_BLOCK_SIZE if use_user_backing else 0,
            }
        )
        offset = slice_limit

    return slices


def _build_remap_slices(
    *,
    w2_bases: list[int],
    backing_bases: list[int],
    w2_bytes: int,
    use_user_backing: bool,
) -> list[dict[str, int]]:
    slices: list[dict[str, int]] = []
    for expert_index, w2_base in enumerate(w2_bases):
        backing_base = backing_bases[expert_index] if use_user_backing else 0
        slices.extend(
            _split_one_expert_into_slices(
                expert_index=expert_index,
                target_base=w2_base,
                backing_base=backing_base,
                byte_count=w2_bytes,
                use_user_backing=use_user_backing,
            )
        )
    if len(slices) > UVM_MOE_SLOT_REMAP_MAX_ENTRIES:
        raise RuntimeError(
            f"remap slice count {len(slices)} exceeds ioctl max entries "
            f"{UVM_MOE_SLOT_REMAP_MAX_ENTRIES}; reduce --batch-count or slice size"
        )
    return slices


def _build_probe_so() -> None:
    if PROBE_SO.exists() and PROBE_SO.stat().st_mtime >= PROBE_CU.stat().st_mtime:
        return
    cmd = [
        "/usr/local/cuda/bin/nvcc",
        "-O3",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        str(PROBE_CU),
        "-o",
        str(PROBE_SO),
    ]
    subprocess.run(cmd, check=True)


def _load_probe() -> ctypes.CDLL:
    _build_probe_so()
    lib = ctypes.CDLL(str(PROBE_SO))
    lib.uvm_stage_probe_set_device.argtypes = [ctypes.c_int]
    lib.uvm_stage_probe_set_device.restype = ctypes.c_int
    lib.uvm_stage_probe_create_stream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.uvm_stage_probe_create_stream.restype = ctypes.c_int
    lib.uvm_stage_probe_destroy_stream.argtypes = [ctypes.c_void_p]
    lib.uvm_stage_probe_destroy_stream.restype = ctypes.c_int
    lib.uvm_stage_probe_create_event.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.uvm_stage_probe_create_event.restype = ctypes.c_int
    lib.uvm_stage_probe_destroy_event.argtypes = [ctypes.c_void_p]
    lib.uvm_stage_probe_destroy_event.restype = ctypes.c_int
    lib.uvm_stage_probe_record_event.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.uvm_stage_probe_record_event.restype = ctypes.c_int
    lib.uvm_stage_probe_sync_event.argtypes = [ctypes.c_void_p]
    lib.uvm_stage_probe_sync_event.restype = ctypes.c_int
    lib.uvm_stage_probe_query_event.argtypes = [ctypes.c_void_p]
    lib.uvm_stage_probe_query_event.restype = ctypes.c_int
    lib.uvm_stage_probe_elapsed_ms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.uvm_stage_probe_elapsed_ms.restype = ctypes.c_int
    lib.uvm_stage_probe_alloc_device.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.uvm_stage_probe_alloc_device.restype = ctypes.c_int
    lib.uvm_stage_probe_free_device.argtypes = [ctypes.c_void_p]
    lib.uvm_stage_probe_free_device.restype = ctypes.c_int
    lib.uvm_stage_probe_copy_device_to_host.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.uvm_stage_probe_copy_device_to_host.restype = ctypes.c_int
    lib.uvm_stage_probe_alloc_pinned_host.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    lib.uvm_stage_probe_alloc_pinned_host.restype = ctypes.c_int
    lib.uvm_stage_probe_free_pinned_host.argtypes = [ctypes.c_void_p]
    lib.uvm_stage_probe_free_pinned_host.restype = ctypes.c_int
    lib.uvm_stage_probe_copy_host_to_device_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.uvm_stage_probe_copy_host_to_device_async.restype = ctypes.c_int
    lib.uvm_stage_probe_launch_read.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uvm_stage_probe_launch_read.restype = ctypes.c_int
    lib.uvm_stage_probe_device_synchronize.argtypes = []
    lib.uvm_stage_probe_device_synchronize.restype = ctypes.c_int
    return lib


def _check(ret: int, label: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{label} failed: {ret}")


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return None


def _make_mmap(byte_count: int, fill: int) -> tuple[mmap.mmap, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    mm = mmap.mmap(-1, byte_count)
    marker = ctypes.c_char.from_buffer(mm)
    addr = ctypes.addressof(marker)
    del marker
    if addr % page_size != 0:
        mm.close()
        raise RuntimeError("anonymous mmap is not page aligned")
    ctypes.memset(addr, fill, byte_count)
    return mm, addr


def _make_pinned_host(lib: ctypes.CDLL, byte_count: int, fill: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_alloc_pinned_host(ctypes.byref(ptr), byte_count), "alloc_pinned_host")
    ctypes.memset(ptr.value, fill, byte_count)
    return ptr


def _free_sources(lib: ctypes.CDLL, sources: list[object]) -> None:
    for source in sources:
        if isinstance(source, mmap.mmap):
            source.close()
        elif isinstance(source, ctypes.c_void_p) and source.value:
            lib.uvm_stage_probe_free_pinned_host(source)


def _event(lib: ctypes.CDLL, timing: bool = True) -> ctypes.c_void_p:
    event = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_event(ctypes.byref(event), int(timing)), "create_event")
    return event


def _elapsed_ms(lib: ctypes.CDLL, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
    value = ctypes.c_float()
    _check(lib.uvm_stage_probe_elapsed_ms(start, end, ctypes.byref(value)), "event_elapsed")
    return float(value.value)


def _fill_entry_at(
    params: MoeSlotRemapBatchParams,
    index: int,
    *,
    mode: int,
    gpu_uuid,
    va_base: int,
    remap_bytes: int,
    validate_managed: bool,
    layer_id: int = 31,
    expert_id: int = 7,
    slot_id: int = 0,
    role: int = 4,
    slot_offset: int = 0,
) -> None:
    params.mode = mode
    params.count = max(int(params.count), index + 1)
    params.gpu_uuid = gpu_uuid
    params.validate_managed = 1 if validate_managed else 0
    entry = params.entries[index]
    entry.layer_id = layer_id
    entry.expert_id = expert_id
    entry.slot_id = slot_id
    entry.role = role
    entry.va_base = va_base
    entry.bytes = remap_bytes
    entry.slot_offset = slot_offset
    entry.flags = 0
    entry.status = ENTRY_OK


def _fill_entry(
    params: MoeSlotRemapBatchParams,
    *,
    mode: int,
    gpu_uuid,
    va_base: int,
    remap_bytes: int,
    validate_managed: bool,
) -> None:
    _fill_entry_at(
        params,
        0,
        mode=mode,
        gpu_uuid=gpu_uuid,
        va_base=va_base,
        remap_bytes=remap_bytes,
        validate_managed=validate_managed,
    )


def _ioctl_batch(fd: int, params: MoeSlotRemapBatchParams) -> dict[str, object]:
    start = time.perf_counter()
    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
    elapsed = time.perf_counter() - start
    result = batch_to_result(params, ret, errno_value)
    result["latency_ms"] = elapsed * 1000.0
    return result


def _wait_driver_async_copy_ready(
    fd: int,
    *,
    expected_slots: int,
    timeout_ms: float = 5000.0,
) -> dict[str, object]:
    start = time.perf_counter()
    polls = 0
    last_query: dict[str, object] | None = None
    while True:
        polls += 1
        query = query_slot_pool(fd, max_entries=max(4, expected_slots + 1))
        last_query = query
        if int(query.get("copy_count", 0)) >= expected_slots:
            return {
                "ready": True,
                "polls": polls,
                "wait_ms": (time.perf_counter() - start) * 1000.0,
                "query": query,
            }
        if (time.perf_counter() - start) * 1000.0 > timeout_ms:
            return {
                "ready": False,
                "polls": polls,
                "wait_ms": (time.perf_counter() - start) * 1000.0,
                "query": last_query,
            }
        time.sleep(0.0001)


def _query_driver_async_copy_ready(
    fd: int,
    *,
    expected_slots: int,
) -> dict[str, object]:
    query = query_slot_pool(fd, max_entries=max(4, expected_slots + 1))
    ready = int(query.get("copy_count", 0)) >= expected_slots
    return {
        "ready": ready,
        "polls": 1,
        "wait_ms": 0.0,
        "query": query,
    }


def _copy_output_prefix(lib: ctypes.CDLL, device_ptr: ctypes.c_void_p, byte_count: int) -> bytes:
    sample = min(byte_count, 256)
    buf = ctypes.create_string_buffer(sample)
    _check(
        lib.uvm_stage_probe_copy_device_to_host(
            ctypes.c_void_p(ctypes.addressof(buf)),
            device_ptr,
            sample,
        ),
        "copy_device_to_host",
    )
    return bytes(buf.raw)


def _run_uva_baseline(
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    stream: ctypes.c_void_p,
    *,
    w13_base: int,
    w2_base: int,
    w13_bytes: int,
    w2_bytes: int,
    w2_out: ctypes.c_void_p,
) -> dict[str, object]:
    w13_start = _event(lib)
    w13_end = _event(lib)
    w2_start = _event(lib)
    w2_end = _event(lib)
    try:
        _check(lib.uvm_stage_probe_record_event(w13_start, stream), "record_w13_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                ctypes.c_void_p(w13_base),
                w2_out,
                w13_bytes,
                args.w13_iters,
                stream,
            ),
            "launch_w13_uva",
        )
        _check(lib.uvm_stage_probe_record_event(w13_end, stream), "record_w13_end")
        _check(lib.uvm_stage_probe_record_event(w2_start, stream), "record_w2_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                ctypes.c_void_p(w2_base),
                w2_out,
                w2_bytes,
                args.w2_iters,
                stream,
            ),
            "launch_w2_uva",
        )
        _check(lib.uvm_stage_probe_record_event(w2_end, stream), "record_w2_end")
        _check(lib.uvm_stage_probe_sync_event(w2_end), "sync_w2_uva")
        prefix = _copy_output_prefix(lib, w2_out, w2_bytes)
        expected = (args.old_pattern * args.w2_iters) & 0xFF
        return {
            "w13_ms": _elapsed_ms(lib, w13_start, w13_end),
            "w2_ms": _elapsed_ms(lib, w2_start, w2_end),
            "output_prefix": prefix[:32].hex(),
            "output_ok": all(byte == expected for byte in prefix),
        }
    finally:
        for event in (w13_start, w13_end, w2_start, w2_end):
            lib.uvm_stage_probe_destroy_event(event)


def _run_remap_case(
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    fd: int,
    gpu_uuid,
    stream: ctypes.c_void_p,
    *,
    w13_base: int,
    w2_base: int,
    source_addr: int,
    w13_bytes: int,
    w2_bytes: int,
    w2_out: ctypes.c_void_p,
) -> dict[str, object]:
    reset_before = reset_slot_pool(fd)
    w13_start = _event(lib)
    w13_end = _event(lib)
    w2_start = _event(lib)
    w2_end = _event(lib)
    try:
        _check(lib.uvm_stage_probe_record_event(w13_start, stream), "record_w13_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                ctypes.c_void_p(w13_base),
                w2_out,
                w13_bytes,
                args.w13_iters,
                stream,
            ),
            "launch_w13",
        )
        _check(lib.uvm_stage_probe_record_event(w13_end, stream), "record_w13_end")

        prefill_params = MoeSlotRemapBatchParams()
        _fill_entry(
            prefill_params,
            mode=MODE_VALUES["gpu_copy_only"],
            gpu_uuid=gpu_uuid,
            va_base=source_addr,
            remap_bytes=w2_bytes,
            validate_managed=False,
        )
        copy_start_host = time.perf_counter()
        prefill = _ioctl_batch(fd, prefill_params)
        copy_end_host = time.perf_counter()
        query_after_copy = lib.uvm_stage_probe_query_event(w13_end)
        copy_done_before_w13_end = query_after_copy == 1

        boundary_wait_start = time.perf_counter()
        _check(lib.uvm_stage_probe_sync_event(w13_end), "sync_w13_boundary")
        boundary_wait_ms = (time.perf_counter() - boundary_wait_start) * 1000.0

        remap_params = MoeSlotRemapBatchParams()
        _fill_entry(
            remap_params,
            mode=MODE_VALUES["slot_va_block_remap"],
            gpu_uuid=gpu_uuid,
            va_base=w2_base,
            remap_bytes=w2_bytes,
            validate_managed=True,
        )
        remap = _ioctl_batch(fd, remap_params)
        query_after_remap = query_slot_pool(fd, max_entries=4)

        _check(lib.uvm_stage_probe_record_event(w2_start, stream), "record_w2_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                ctypes.c_void_p(w2_base),
                w2_out,
                w2_bytes,
                args.w2_iters,
                stream,
            ),
            "launch_w2_slot",
        )
        _check(lib.uvm_stage_probe_record_event(w2_end, stream), "record_w2_end")
        _check(lib.uvm_stage_probe_sync_event(w2_end), "sync_w2_slot")
        prefix = _copy_output_prefix(lib, w2_out, w2_bytes)
        expected = (args.slot_pattern * args.w2_iters) & 0xFF
        reset_after = reset_slot_pool(fd)

        return {
            "reset_before": reset_before,
            "prefill": prefill,
            "copy_host_ms": (copy_end_host - copy_start_host) * 1000.0,
            "copy_done_before_w13_end": copy_done_before_w13_end,
            "w13_query_after_copy": query_after_copy,
            "boundary_wait_ms": boundary_wait_ms,
            "remap": remap,
            "slot_pool_after_remap": query_after_remap,
            "w13_ms": _elapsed_ms(lib, w13_start, w13_end),
            "w2_ms": _elapsed_ms(lib, w2_start, w2_end),
            "output_prefix": prefix[:32].hex(),
            "output_ok": all(byte == expected for byte in prefix),
            "reset_after": reset_after,
            "passed": (
                prefill.get("ret") == 0
                and prefill.get("rmStatus") == 0
                and prefill.get("success_count") == 1
                and remap.get("ret") == 0
                and remap.get("rmStatus") == 0
                and remap.get("success_count") == 1
                and query_after_remap.get("pte_remap_count") == 1
                and slot_pool_health_ok(query_after_remap)
                and all(byte == expected for byte in prefix)
                and reset_after.get("ok") is True
            ),
        }
    finally:
        for event in (w13_start, w13_end, w2_start, w2_end):
            lib.uvm_stage_probe_destroy_event(event)


def _run_uva_baseline_batch(
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    stream: ctypes.c_void_p,
    *,
    w13_base: int,
    w2_bases: list[int],
    w13_bytes: int,
    w2_bytes: int,
    w2_outs: list[ctypes.c_void_p],
) -> dict[str, object]:
    w13_start = _event(lib)
    w13_end = _event(lib)
    w2_start = _event(lib)
    w2_end = _event(lib)
    try:
        _check(lib.uvm_stage_probe_record_event(w13_start, stream), "record_w13_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                ctypes.c_void_p(w13_base),
                w2_outs[0],
                w13_bytes,
                args.w13_iters,
                stream,
            ),
            "launch_w13_uva_batch",
        )
        _check(lib.uvm_stage_probe_record_event(w13_end, stream), "record_w13_end")
        _check(lib.uvm_stage_probe_record_event(w2_start, stream), "record_w2_start")
        for index, w2_base in enumerate(w2_bases):
            _check(
                lib.uvm_stage_probe_launch_read(
                    ctypes.c_void_p(w2_base),
                    w2_outs[index],
                    w2_bytes,
                    args.w2_iters,
                    stream,
                ),
                f"launch_w2_uva_batch_{index}",
            )
        _check(lib.uvm_stage_probe_record_event(w2_end, stream), "record_w2_end")
        _check(lib.uvm_stage_probe_sync_event(w2_end), "sync_w2_uva_batch")
        expected = (args.old_pattern * args.w2_iters) & 0xFF
        prefixes = [_copy_output_prefix(lib, out, w2_bytes) for out in w2_outs]
        return {
            "w13_ms": _elapsed_ms(lib, w13_start, w13_end),
            "w2_ms": _elapsed_ms(lib, w2_start, w2_end),
            "output_prefixes": [prefix[:32].hex() for prefix in prefixes],
            "output_ok": all(all(byte == expected for byte in prefix) for prefix in prefixes),
        }
    finally:
        for event in (w13_start, w13_end, w2_start, w2_end):
            lib.uvm_stage_probe_destroy_event(event)


def _run_remap_case_batch(
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    fd: int,
    gpu_uuid,
    stream: ctypes.c_void_p,
    copy_stream: ctypes.c_void_p,
    *,
    w13_base: int,
    w2_bases: list[int],
    source_addrs: list[int],
    backing_bases: list[int],
    w13_bytes: int,
    w2_bytes: int,
    w2_outs: list[ctypes.c_void_p],
) -> dict[str, object]:
    reset_before = reset_slot_pool(fd)
    use_user_backing = _uses_user_backing(args)
    remap_slices = _build_remap_slices(
        w2_bases=w2_bases,
        backing_bases=backing_bases,
        w2_bytes=w2_bytes,
        use_user_backing=use_user_backing,
    )
    slices_per_expert = [0 for _ in w2_bases]
    for remap_slice in remap_slices:
        slices_per_expert[remap_slice["expert_index"]] += 1
    target_block_count = len({remap_slice["target_block"] for remap_slice in remap_slices})
    backing_block_count = (
        len({remap_slice["backing_block"] for remap_slice in remap_slices}) if use_user_backing else 0
    )
    w13_start = _event(lib)
    w13_end = _event(lib)
    copy_start = _event(lib)
    copy_end = _event(lib)
    w2_start = _event(lib)
    w2_end = _event(lib)
    try:
        validate_params = MoeSlotRemapBatchParams()
        for index, remap_slice in enumerate(remap_slices):
            _fill_entry_at(
                validate_params,
                index,
                mode=MODE_VALUES["validate"],
                gpu_uuid=gpu_uuid,
                va_base=remap_slice["target_va"],
                remap_bytes=remap_slice["bytes"],
                validate_managed=False,
                layer_id=31,
                expert_id=remap_slice["expert_id"],
                slot_id=remap_slice["slot_id"],
                role=4,
                slot_offset=remap_slice["offset"],
            )
        validate = _ioctl_batch(fd, validate_params)

        prealloc = None
        if args.copy_backend in ("driver_vidmem_slot", "driver_vidmem_async"):
            prealloc_params = MoeSlotRemapBatchParams()
            for index in range(len(w2_bases)):
                _fill_entry_at(
                    prealloc_params,
                    index,
                    mode=MODE_VALUES["vidmem_prealloc"],
                    gpu_uuid=gpu_uuid,
                    va_base=w2_bases[index],
                    remap_bytes=w2_bytes,
                    validate_managed=False,
                    layer_id=31,
                    expert_id=7 + index,
                    slot_id=index,
                    role=4,
                )
            prealloc = _ioctl_batch(fd, prealloc_params)

        _check(lib.uvm_stage_probe_record_event(w13_start, stream), "record_w13_start")
        _check(
            lib.uvm_stage_probe_launch_read(
                ctypes.c_void_p(w13_base),
                w2_outs[0],
                w13_bytes,
                args.w13_iters,
                stream,
            ),
            "launch_w13_batch",
        )
        _check(lib.uvm_stage_probe_record_event(w13_end, stream), "record_w13_end")

        copy_host_start = time.perf_counter()
        prefill = None
        if _uses_driver_vidmem_slot(args):
            prefill_params = MoeSlotRemapBatchParams()
            for index, source_addr in enumerate(source_addrs):
                _fill_entry_at(
                    prefill_params,
                    index,
                    mode=MODE_VALUES["gpu_copy_async_submit"]
                    if _uses_driver_vidmem_async(args)
                    else MODE_VALUES["gpu_copy_only"],
                    gpu_uuid=gpu_uuid,
                    va_base=source_addr,
                    remap_bytes=w2_bytes,
                    validate_managed=False,
                    layer_id=31,
                    expert_id=7 + index,
                    slot_id=index,
                    role=4,
                )
            prefill = _ioctl_batch(fd, prefill_params)
        else:
            _check(lib.uvm_stage_probe_record_event(copy_start, copy_stream), "record_copy_start")
            for index, source_addr in enumerate(source_addrs):
                _check(
                    lib.uvm_stage_probe_copy_host_to_device_async(
                        ctypes.c_void_p(backing_bases[index]),
                        ctypes.c_void_p(source_addr),
                        w2_bytes,
                        copy_stream,
                    ),
                    f"copy_host_to_backing_async_{index}",
                )
            _check(lib.uvm_stage_probe_record_event(copy_end, copy_stream), "record_copy_end")
            prefill = {
                "ret": 0,
                "rmStatus": 0,
                "success_count": len(w2_bases),
                "latency_ms": (time.perf_counter() - copy_host_start) * 1000.0,
                "mode": args.copy_backend,
            }
        copy_host_end = time.perf_counter()
        query_after_copy = lib.uvm_stage_probe_query_event(w13_end)
        copy_done_before_w13_end = query_after_copy == 1

        boundary_wait_start = time.perf_counter()
        _check(lib.uvm_stage_probe_sync_event(w13_end), "sync_w13_boundary_batch")
        boundary_wait_ms = (time.perf_counter() - boundary_wait_start) * 1000.0

        copy_wait_ms = 0.0
        async_copy_ready = None
        if _uses_user_backing(args):
            if args.boundary_policy == "wait_ready":
                copy_wait_start = time.perf_counter()
                _check(lib.uvm_stage_probe_sync_event(copy_end), "sync_async_copy_end")
                copy_wait_ms = (time.perf_counter() - copy_wait_start) * 1000.0
                copy_ready_at_boundary = True
            else:
                copy_ready_at_boundary = lib.uvm_stage_probe_query_event(copy_end) == 1
        elif _uses_driver_vidmem_async(args):
            if args.boundary_policy == "wait_ready":
                async_copy_ready = _wait_driver_async_copy_ready(
                    fd,
                    expected_slots=len(w2_bases),
                )
            else:
                async_copy_ready = _query_driver_async_copy_ready(
                    fd,
                    expected_slots=len(w2_bases),
                )
            copy_wait_ms = float(async_copy_ready["wait_ms"])
            query_after_copy = async_copy_ready["query"]
            copy_ready_at_boundary = bool(async_copy_ready["ready"])
            copy_done_before_w13_end = copy_ready_at_boundary
        else:
            copy_ready_at_boundary = True

        if not copy_ready_at_boundary:
            remap = {
                "ret": 0,
                "rmStatus": 0,
                "success_count": 0,
                "latency_ms": 0.0,
                "accepted_bytes": 0,
                "remap_count": 0,
                "tlb_invalidate_count": 0,
            }
            query_after_remap = query_after_copy
            _check(lib.uvm_stage_probe_record_event(w2_start, stream), "record_w2_start_uva_fallback")
            for index, w2_base in enumerate(w2_bases):
                _check(
                    lib.uvm_stage_probe_launch_read(
                        ctypes.c_void_p(w2_base),
                        w2_outs[index],
                        w2_bytes,
                        args.w2_iters,
                        stream,
                    ),
                    f"launch_w2_uva_fallback_{index}",
                )
            _check(lib.uvm_stage_probe_record_event(w2_end, stream), "record_w2_end_uva_fallback")
            _check(lib.uvm_stage_probe_sync_event(w2_end), "sync_w2_uva_fallback")
            prefixes = [_copy_output_prefix(lib, out, w2_bytes) for out in w2_outs]
            expected = (args.old_pattern * args.w2_iters) & 0xFF
            output_ok = all(all(byte == expected for byte in prefix) for prefix in prefixes)
            reset_after = reset_slot_pool(fd)
            return {
                "reset_before": reset_before,
                "validate": validate,
                "prealloc": prealloc,
                "prefill": prefill,
                "async_copy_ready": async_copy_ready,
                "copy_host_ms": (copy_host_end - copy_host_start) * 1000.0,
                "copy_wait_ms": copy_wait_ms,
                "copy_done_before_w13_end": copy_done_before_w13_end,
                "copy_ready_at_boundary": copy_ready_at_boundary,
                "used_remap": False,
                "fallback_uva": True,
                "boundary_policy": args.boundary_policy,
                "w13_query_after_copy": query_after_copy,
                "boundary_wait_ms": boundary_wait_ms,
                "remap": remap,
                "remap_host_gap_ms": 0.0,
                "remap_slice_count": 0,
                "slices_per_expert": slices_per_expert,
                "target_va_block_count": target_block_count,
                "backing_va_block_count": backing_block_count,
                "slot_pool_after_remap": query_after_remap,
                "w13_ms": _elapsed_ms(lib, w13_start, w13_end),
                "copy_ms": _elapsed_ms(lib, copy_start, copy_end) if _uses_user_backing(args) else None,
                "w2_ms": _elapsed_ms(lib, w2_start, w2_end),
                "output_prefixes": [prefix[:32].hex() for prefix in prefixes],
                "output_ok": output_ok,
                "reset_after": reset_after,
                "passed": output_ok and reset_after.get("ok") is True,
            }

        remap_params = MoeSlotRemapBatchParams()
        remap_mode = (
            "slot_va_block_user_backing_remap"
            if use_user_backing
            else "slot_va_block_remap"
        )
        for index, remap_slice in enumerate(remap_slices):
            _fill_entry_at(
                remap_params,
                index,
                mode=MODE_VALUES[remap_mode],
                gpu_uuid=gpu_uuid,
                va_base=remap_slice["target_va"],
                remap_bytes=remap_slice["bytes"],
                validate_managed=True,
                layer_id=31,
                expert_id=remap_slice["expert_id"],
                slot_id=remap_slice["slot_id"],
                role=4,
                slot_offset=remap_slice["backing_va"] if use_user_backing else remap_slice["offset"],
            )
        remap_start_host = time.perf_counter()
        remap = _ioctl_batch(fd, remap_params)
        remap_host_gap_ms = (time.perf_counter() - remap_start_host) * 1000.0
        query_after_remap = query_slot_pool(fd, max_entries=max(4, len(w2_bases) + 1))

        _check(lib.uvm_stage_probe_record_event(w2_start, stream), "record_w2_start")
        for index, w2_base in enumerate(w2_bases):
            _check(
                lib.uvm_stage_probe_launch_read(
                    ctypes.c_void_p(w2_base),
                    w2_outs[index],
                    w2_bytes,
                    args.w2_iters,
                    stream,
                ),
                f"launch_w2_slot_batch_{index}",
            )
        _check(lib.uvm_stage_probe_record_event(w2_end, stream), "record_w2_end")
        _check(lib.uvm_stage_probe_sync_event(w2_end), "sync_w2_slot_batch")
        prefixes = [_copy_output_prefix(lib, out, w2_bytes) for out in w2_outs]
        expected = (args.slot_pattern * args.w2_iters) & 0xFF
        reset_after = reset_slot_pool(fd)

        return {
            "reset_before": reset_before,
            "validate": validate,
            "prealloc": prealloc,
            "prefill": prefill,
            "async_copy_ready": async_copy_ready,
            "copy_host_ms": (copy_host_end - copy_host_start) * 1000.0,
            "copy_wait_ms": copy_wait_ms,
            "copy_done_before_w13_end": copy_done_before_w13_end,
            "copy_ready_at_boundary": copy_ready_at_boundary,
            "used_remap": True,
            "fallback_uva": False,
            "boundary_policy": args.boundary_policy,
            "w13_query_after_copy": query_after_copy,
            "boundary_wait_ms": boundary_wait_ms,
            "remap": remap,
            "remap_host_gap_ms": remap_host_gap_ms,
            "remap_slice_count": len(remap_slices),
            "slices_per_expert": slices_per_expert,
            "target_va_block_count": target_block_count,
            "backing_va_block_count": backing_block_count,
            "slot_pool_after_remap": query_after_remap,
            "w13_ms": _elapsed_ms(lib, w13_start, w13_end),
            "copy_ms": _elapsed_ms(lib, copy_start, copy_end) if _uses_user_backing(args) else None,
            "w2_ms": _elapsed_ms(lib, w2_start, w2_end),
            "output_prefixes": [prefix[:32].hex() for prefix in prefixes],
            "output_ok": all(all(byte == expected for byte in prefix) for prefix in prefixes),
            "reset_after": reset_after,
            "passed": (
                validate.get("ret") == 0
                and validate.get("rmStatus") == 0
                and validate.get("success_count") == len(remap_slices)
                and isinstance(prefill, dict)
                and prefill.get("ret") == 0
                and prefill.get("rmStatus") == 0
                and prefill.get("success_count") == len(w2_bases)
                and remap.get("ret") == 0
                and remap.get("rmStatus") == 0
                and remap.get("success_count") == len(remap_slices)
                and query_after_remap.get("pte_remap_count") == len(remap_slices)
                and slot_pool_health_ok(query_after_remap)
                and all(all(byte == expected for byte in prefix) for prefix in prefixes)
                and reset_after.get("ok") is True
            ),
        }
    finally:
        for event in (w13_start, w13_end, copy_start, copy_end, w2_start, w2_end):
            lib.uvm_stage_probe_destroy_event(event)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": ordered[len(ordered) // 2],
        "min": min(values),
        "max": max(values),
    }


def _summarize(results: list[dict[str, object]]) -> dict[str, object]:
    uva_w2 = [float(item["uva_baseline"]["w2_ms"]) for item in results]  # type: ignore[index]
    slot_w2 = [float(item["remap_case"]["w2_ms"]) for item in results]  # type: ignore[index]
    remap_ms = [float(item["remap_case"]["remap"]["latency_ms"]) for item in results]  # type: ignore[index]
    copy_ms = [float(item["remap_case"]["copy_host_ms"]) for item in results]  # type: ignore[index]
    copy_wait = [float(item["remap_case"].get("copy_wait_ms", 0.0)) for item in results]  # type: ignore[index]
    boundary_wait = [float(item["remap_case"]["boundary_wait_ms"]) for item in results]  # type: ignore[index]
    remap_gap = [float(item["remap_case"].get("remap_host_gap_ms", 0.0)) for item in results]  # type: ignore[index]
    boundary_tail = [
        float(item["remap_case"].get("copy_wait_ms", 0.0))
        + float(item["remap_case"].get("remap_host_gap_ms", 0.0))
        + float(item["remap_case"]["w2_ms"])  # type: ignore[index]
        for item in results
    ]
    remap_used_count = sum(1 for item in results if bool(item["remap_case"].get("used_remap")))  # type: ignore[index]
    fallback_count = sum(1 for item in results if bool(item["remap_case"].get("fallback_uva")))  # type: ignore[index]
    return {
        "runs": len(results),
        "all_passed": all(
            bool(item["uva_baseline"].get("output_ok")) and bool(item["remap_case"].get("passed"))
            for item in results
        ),
        "copy_done_before_w13_end_count": sum(
            1 for item in results if bool(item["remap_case"].get("copy_done_before_w13_end"))
        ),
        "copy_ready_at_boundary_count": sum(
            1 for item in results if bool(item["remap_case"].get("copy_ready_at_boundary"))
        ),
        "uva_w2_ms": _stats(uva_w2),
        "slot_w2_ms": _stats(slot_w2),
        "w2_saved_ms_p50": (sorted(uva_w2)[len(uva_w2) // 2] - sorted(slot_w2)[len(slot_w2) // 2]) if results else None,
        "remap_ms": _stats(remap_ms),
        "copy_host_ms": _stats(copy_ms),
        "copy_wait_ms": _stats(copy_wait),
        "boundary_wait_ms": _stats(boundary_wait),
        "remap_host_gap_ms": _stats(remap_gap),
        "boundary_tail_ms": _stats(boundary_tail),
        "remap_used_count": remap_used_count,
        "fallback_uva_count": fallback_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--w13-bytes", type=int, default=2 * 768 * 2048)
    parser.add_argument("--w2-bytes", type=int, default=2048 * 768)
    parser.add_argument("--w13-iters", type=int, default=32)
    parser.add_argument("--w2-iters", type=int, default=1)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument(
        "--weight-layout",
        choices=("layer_tensor", "separate"),
        default="layer_tensor",
        help="layer_tensor mimics vLLM [num_experts, ...] weight tensors; separate keeps one allocation per expert.",
    )
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--start-expert", type=int, default=0)
    parser.add_argument(
        "--w2-stride-bytes",
        type=int,
        default=0,
        help="Byte stride between adjacent experts in layer_tensor layout. Defaults to w2_bytes.",
    )
    parser.add_argument(
        "--copy-backend",
        choices=("gpu_copy_only", "driver_vidmem_slot", "driver_vidmem_async", "async_user_backing"),
        default="gpu_copy_only",
    )
    parser.add_argument(
        "--boundary-policy",
        choices=("wait_ready", "opportunistic"),
        default="opportunistic",
        help=(
            "wait_ready waits for async copy at the w13/w2 boundary before remap; "
            "opportunistic queries once and falls back to UVA if copy is not ready."
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--old-pattern", type=lambda value: int(value, 0), default=0xA5)
    parser.add_argument("--slot-pattern", type=lambda value: int(value, 0), default=0x3C)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if _read_int(PTE_ENABLE_PARAM) != 1:
        raise SystemExit("uvm_moe_slot_pte_remap_enable is not 1")
    max_bytes = _read_int(PTE_MAX_BYTES_PARAM)
    args.w13_bytes = align_up(args.w13_bytes, os.sysconf("SC_PAGE_SIZE"))
    args.w2_bytes = align_up(args.w2_bytes, os.sysconf("SC_PAGE_SIZE"))
    args.w2_stride = align_up(args.w2_stride_bytes or args.w2_bytes, os.sysconf("SC_PAGE_SIZE"))
    if args.batch_count < 1 or args.batch_count > 64:
        raise SystemExit("--batch-count must be in [1, 64]")
    if args.num_experts < 1:
        raise SystemExit("--num-experts must be positive")
    if args.start_expert < 0 or args.start_expert + args.batch_count > args.num_experts:
        raise SystemExit("--start-expert + --batch-count must be <= --num-experts")
    if args.w2_stride < args.w2_bytes:
        raise SystemExit("--w2-stride-bytes must be >= --w2-bytes")
    if max_bytes is not None and args.w2_bytes > max_bytes:
        raise SystemExit(f"uvm_moe_slot_pte_remap_max_bytes={max_bytes} too small for w2_bytes={args.w2_bytes}")

    initialize_cuda_device(args.device)
    lib = _load_probe()
    _check(lib.uvm_stage_probe_set_device(args.device), "set_device")
    gpu_uuid = gpu_uuid_for_device(args.device)
    fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(prefer_existing=True)
    if fd < 0:
        raise RuntimeError("failed to open initialized UVM fd")

    stream = ctypes.c_void_p()
    copy_stream = ctypes.c_void_p()
    _check(lib.uvm_stage_probe_create_stream(ctypes.byref(stream)), "create_stream")
    _check(lib.uvm_stage_probe_create_stream(ctypes.byref(copy_stream)), "create_copy_stream")
    results = []
    try:
        for repeat in range(args.repeats):
            alloc_bytes = args.w13_bytes + args.w2_bytes + 2 * UVM_VA_BLOCK_SIZE
            w13_alloc = None
            w13_alloc2 = None
            w2_allocs: list[ManagedAllocation] = []
            backing_allocs: list[ManagedAllocation] = []
            w2_allocs2: list[ManagedAllocation] = []
            backing_allocs2: list[ManagedAllocation] = []
            sources: list[object] = []
            w2_outs: list[ctypes.c_void_p] = []
            try:
                w13_alloc = ManagedAllocation(args.device, alloc_bytes)
                w13_base = align_up(w13_alloc.address, os.sysconf("SC_PAGE_SIZE"))
                ctypes.memset(w13_base, args.old_pattern, args.w13_bytes)
                (
                    w2_allocs,
                    backing_allocs,
                    w2_bases,
                    backing_bases,
                    sources,
                    source_addrs,
                ) = _allocate_w2_layout(args, lib, fill_backing=True)

                for index in range(args.batch_count):
                    w2_out = ctypes.c_void_p()
                    _check(
                        lib.uvm_stage_probe_alloc_device(
                            ctypes.byref(w2_out),
                            max(args.w13_bytes, args.w2_bytes),
                            args.device,
                        ),
                        f"alloc_w2_out_{index}",
                    )
                    w2_outs.append(w2_out)

                uva_baseline = _run_uva_baseline_batch(
                    args,
                    lib,
                    stream,
                    w13_base=w13_base,
                    w2_bases=w2_bases,
                    w13_bytes=args.w13_bytes,
                    w2_bytes=args.w2_bytes,
                    w2_outs=w2_outs,
                )

                w13_alloc2 = ManagedAllocation(args.device, alloc_bytes)
                w13_base2 = align_up(w13_alloc2.address, os.sysconf("SC_PAGE_SIZE"))
                ctypes.memset(w13_base2, args.old_pattern, args.w13_bytes)
                (
                    w2_allocs2,
                    backing_allocs2,
                    w2_bases2,
                    backing_bases2,
                    unused_sources2,
                    unused_source_addrs2,
                ) = _allocate_w2_layout(args, lib, fill_backing=True, allocate_sources=False)
                if unused_sources2 or unused_source_addrs2:
                    raise RuntimeError("remap layout unexpectedly allocated source buffers")

                remap_case = _run_remap_case_batch(
                    args,
                    lib,
                    fd,
                    gpu_uuid,
                    stream,
                    copy_stream,
                    w13_base=w13_base2,
                    w2_bases=w2_bases2,
                    source_addrs=source_addrs,
                    backing_bases=backing_bases2,
                    w13_bytes=args.w13_bytes,
                    w2_bytes=args.w2_bytes,
                    w2_outs=w2_outs,
                )

                item = {
                    "repeat": repeat,
                    "uva_baseline": uva_baseline,
                    "remap_case": remap_case,
                }
                results.append(item)
                print(
                    f"repeat={repeat} uva_w2_ms={uva_baseline['w2_ms']:.3f} "
                    f"slot_w2_ms={remap_case['w2_ms']:.3f} "
                    f"copy_ms={remap_case['copy_host_ms']:.3f} "
                    f"remap_ms={remap_case['remap']['latency_ms']:.3f} "
                    f"slices={remap_case['remap_slice_count']} "
                    f"target_blocks={remap_case['target_va_block_count']} "
                    f"batch={args.batch_count} layout={args.weight_layout} backend={args.copy_backend} "
                    f"copy_before_w13={remap_case['copy_done_before_w13_end']} "
                    f"passed={remap_case['passed']}",
                    flush=True,
                )
            finally:
                for w2_out in w2_outs:
                    if w2_out.value:
                        lib.uvm_stage_probe_free_device(w2_out)
                _free_sources(lib, sources)
                for alloc in backing_allocs2:
                    alloc.close()
                for alloc in w2_allocs2:
                    alloc.close()
                if w13_alloc2 is not None:
                    w13_alloc2.close()
                for alloc in backing_allocs:
                    alloc.close()
                for alloc in w2_allocs:
                    alloc.close()
                if w13_alloc is not None:
                    w13_alloc.close()
        output = {
            "device": args.device,
            "fd_source": fd_source,
            "init_ret": init_ret,
            "init_errno": init_errno,
            "init_rmStatus": init_rm_status,
            "w13_bytes": args.w13_bytes,
            "w2_bytes": args.w2_bytes,
            "w13_iters": args.w13_iters,
            "w2_iters": args.w2_iters,
            "batch_count": args.batch_count,
            "weight_layout": args.weight_layout,
            "boundary_policy": args.boundary_policy,
            "num_experts": args.num_experts,
            "start_expert": args.start_expert,
            "w2_stride": args.w2_stride,
            "copy_backend": args.copy_backend,
            "results": results,
            "summary": _summarize(results),
            "passed": all(
                bool(item["uva_baseline"].get("output_ok")) and bool(item["remap_case"].get("passed"))
                for item in results
            ),
        }
    finally:
        reset_slot_pool(fd)
        os.close(fd)
        lib.uvm_stage_probe_destroy_stream(stream)
        lib.uvm_stage_probe_destroy_stream(copy_stream)

    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
