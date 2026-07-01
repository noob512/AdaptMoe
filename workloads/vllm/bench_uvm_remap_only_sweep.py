#!/usr/bin/env python3
"""Batch remap-only upper-bound probe for controlled MoE slot remap."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import statistics
import time
from pathlib import Path

from check_moe_slot_remap_ioctl_smoke import (
    ENTRY_OK,
    MODE_VALUES,
    UVM_MOE_SLOT_REMAP_BATCH_IOCTL,
    UVM_VA_BLOCK_SIZE,
    ManagedAllocation,
    MoeSlotRemapBatchParams,
    align_up,
    batch_to_result,
    gpu_uuid_for_device,
    initialize_cuda_device,
    ioctl,
    kernel_readback,
    open_initialized_uvm_fd,
    query_slot_pool,
    reset_slot_pool,
    slot_pool_health_ok,
)

PTE_ENABLE_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable"
PTE_MAX_BYTES_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes"


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return None


def _size_arg(value: str) -> int:
    lower = value.strip().lower()
    scale = 1
    for suffix, multiplier in (
        ("kib", 1024),
        ("mib", 1024 * 1024),
        ("kb", 1000),
        ("mb", 1000 * 1000),
        ("k", 1024),
        ("m", 1024 * 1024),
    ):
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
            scale = multiplier
            break
    return int(float(lower) * scale)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


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


def _layout_bases(
    raw_base: int,
    *,
    remap_bytes: int,
    count: int,
    crosses_va_block: bool,
) -> tuple[list[int], int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    if remap_bytes % page_size != 0:
        raise ValueError("remap_bytes must be page aligned")
    if crosses_va_block:
        if remap_bytes >= UVM_VA_BLOCK_SIZE:
            stride = align_up(remap_bytes, UVM_VA_BLOCK_SIZE) + UVM_VA_BLOCK_SIZE
            first_block = align_up(raw_base + UVM_VA_BLOCK_SIZE, UVM_VA_BLOCK_SIZE)
            bases = [first_block + i * stride for i in range(count)]
        else:
            half = align_up(remap_bytes // 2, page_size)
            stride = 2 * UVM_VA_BLOCK_SIZE
            boundary = align_up(raw_base + UVM_VA_BLOCK_SIZE, UVM_VA_BLOCK_SIZE)
            bases = [boundary - half + i * stride for i in range(count)]
        end = bases[-1] + remap_bytes if bases else raw_base
        return bases, end - raw_base + UVM_VA_BLOCK_SIZE

    if remap_bytes > UVM_VA_BLOCK_SIZE:
        raise ValueError("non-cross layout is impossible for entries larger than one VA block")

    stride = UVM_VA_BLOCK_SIZE
    first = align_up(raw_base, UVM_VA_BLOCK_SIZE)
    bases = [first + i * stride for i in range(count)]
    end = bases[-1] + remap_bytes if bases else raw_base
    return bases, end - raw_base + UVM_VA_BLOCK_SIZE


def _fill_entries(
    params: MoeSlotRemapBatchParams,
    *,
    mode: int,
    gpu_uuid,
    bases: list[int],
    remap_bytes: int,
    validate_managed: bool,
    source_stride: int | None = None,
) -> None:
    params.mode = mode
    params.count = len(bases)
    params.gpu_uuid = gpu_uuid
    params.validate_managed = 1 if validate_managed else 0
    for index, base in enumerate(bases):
        entry = params.entries[index]
        entry.layer_id = 23
        entry.expert_id = 1000 + index
        entry.slot_id = index
        entry.role = 4
        entry.va_base = base
        entry.bytes = remap_bytes
        entry.slot_offset = 0 if source_stride is None else 0
        entry.flags = 0
        entry.status = ENTRY_OK


def _ioctl_batch(fd: int, params: MoeSlotRemapBatchParams) -> dict[str, object]:
    start = time.perf_counter()
    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
    elapsed = time.perf_counter() - start
    result = batch_to_result(params, ret, errno_value)
    result["latency_ms"] = elapsed * 1000.0
    return result


def _verify_prefixes(
    bases: list[int],
    *,
    remap_bytes: int,
    pattern: int,
    device: int,
    max_entries: int,
) -> dict[str, object]:
    checks = []
    sample_count = min(len(bases), max_entries)
    for index in range(sample_count):
        base = bases[index]
        head = kernel_readback(base, min(64, remap_bytes), device)
        tail_offset = max(remap_bytes - min(64, remap_bytes), 0)
        tail = kernel_readback(base + tail_offset, min(64, remap_bytes), device)
        checks.append({
            "index": index,
            "head_ok": all(byte == pattern for byte in head),
            "tail_ok": all(byte == pattern for byte in tail),
            "head_prefix": head[:16].hex(),
            "tail_prefix": tail[:16].hex(),
        })
    return {
        "sampled": sample_count,
        "ok": all(item["head_ok"] and item["tail_ok"] for item in checks),
        "checks": checks,
    }


def run_case(args: argparse.Namespace, size_name: str, remap_bytes: int, count: int, crosses: bool) -> dict[str, object]:
    initialize_cuda_device(args.device)
    gpu_uuid = gpu_uuid_for_device(args.device)
    fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(prefer_existing=True)
    if fd < 0:
        raise RuntimeError("failed to open initialized UVM fd")

    page_size = os.sysconf("SC_PAGE_SIZE")
    source = None
    target = None
    try:
        source_bytes = remap_bytes * count
        pattern = 0x30 + (count % 64)
        source, source_addr = _make_mmap(source_bytes, pattern)

        dummy_raw = 0x100000000
        _, needed_bytes = _layout_bases(
            dummy_raw,
            remap_bytes=remap_bytes,
            count=count,
            crosses_va_block=crosses,
        )
        target = ManagedAllocation(args.device, needed_bytes + 2 * UVM_VA_BLOCK_SIZE)
        bases, _ = _layout_bases(
            target.address,
            remap_bytes=remap_bytes,
            count=count,
            crosses_va_block=crosses,
        )
        if any(base < target.address or base + remap_bytes > target.address + target.byte_count for base in bases):
            raise RuntimeError("target layout escaped managed allocation")

        reset_before = reset_slot_pool(fd)
        prefill_params = MoeSlotRemapBatchParams()
        source_bases = [source_addr + index * remap_bytes for index in range(count)]
        _fill_entries(
            prefill_params,
            mode=MODE_VALUES["gpu_copy_only"],
            gpu_uuid=gpu_uuid,
            bases=source_bases,
            remap_bytes=remap_bytes,
            validate_managed=False,
        )
        prefill = _ioctl_batch(fd, prefill_params)

        remap_params = MoeSlotRemapBatchParams()
        _fill_entries(
            remap_params,
            mode=MODE_VALUES["slot_va_block_remap"],
            gpu_uuid=gpu_uuid,
            bases=bases,
            remap_bytes=remap_bytes,
            validate_managed=True,
        )
        remap = _ioctl_batch(fd, remap_params)
        query_after = query_slot_pool(fd, max_entries=min(count, 128))
        verify = {"ok": True, "sampled": 0, "checks": []}
        if args.verify:
            verify = _verify_prefixes(
                bases,
                remap_bytes=remap_bytes,
                pattern=pattern,
                device=args.device,
                max_entries=args.verify_entries,
            )
        reset_after = reset_slot_pool(fd)

        return {
            "size_name": size_name,
            "remap_bytes": remap_bytes,
            "count": count,
            "crosses_va_block": crosses,
            "entry_crosses": [
                base // UVM_VA_BLOCK_SIZE != (base + remap_bytes - 1) // UVM_VA_BLOCK_SIZE
                for base in bases[: min(len(bases), 8)]
            ],
            "fd_source": fd_source,
            "init_ret": init_ret,
            "init_errno": init_errno,
            "init_rmStatus": init_rm_status,
            "reset_before": reset_before,
            "prefill": prefill,
            "remap": remap,
            "slot_pool_after_remap": query_after,
            "verify": verify,
            "reset_after": reset_after,
            "passed": (
                prefill.get("ret") == 0
                and prefill.get("rmStatus") == 0
                and prefill.get("success_count") == count
                and remap.get("ret") == 0
                and remap.get("rmStatus") == 0
                and remap.get("success_count") == count
                and query_after.get("pte_remap_count") == count
                and query_after.get("pte_block_tracker_wait_count", 0) >= count
                and verify.get("ok") is True
                and reset_after.get("ok") is True
                and slot_pool_health_ok(query_after)
            ),
        }
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()
        os.close(fd)


def summarize(results: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, bool], list[dict[str, object]]] = {}
    for result in results:
        key = (
            str(result["size_name"]),
            int(result["remap_bytes"]),
            int(result["count"]),
            bool(result["crosses_va_block"]),
        )
        groups.setdefault(key, []).append(result)

    rows = []
    for (size_name, remap_bytes, count, crosses), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][3])):
        remap_latencies = [float(item["remap"]["latency_ms"]) for item in items]  # type: ignore[index]
        pte_counts = [int(item["slot_pool_after_remap"].get("pte_remap_count", 0)) for item in items]  # type: ignore[union-attr]
        tracker_counts = [
            int(item["slot_pool_after_remap"].get("pte_block_tracker_wait_count", 0))
            for item in items
        ]  # type: ignore[union-attr]
        tlb_counts = [int(item["remap"].get("tlb_invalidate_count", 0)) for item in items]  # type: ignore[union-attr]
        rows.append({
            "size_name": size_name,
            "remap_bytes": remap_bytes,
            "count": count,
            "crosses_va_block": crosses,
            "runs": len(items),
            "all_passed": all(bool(item.get("passed")) for item in items),
            "remap_latency_ms": _stats(remap_latencies),
            "latency_per_entry_p50_ms": (
                (_percentile(remap_latencies, 0.50) or 0.0) / count if count else None
            ),
            "pte_remap_count_values": pte_counts,
            "tracker_wait_count_values": tracker_counts,
            "tlb_invalidate_count_values": tlb_counts,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--counts", default="1,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-entries", type=int, default=2)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--sizes", default="")
    args = parser.parse_args()

    if _read_int(PTE_ENABLE_PARAM) != 1:
        raise SystemExit("uvm_moe_slot_pte_remap_enable is not 1")

    default_sizes = {
        "64KiB": 64 * 1024,
        "w2": 2048 * 768,
        "w13": 2 * 768 * 2048,
        "full": 3 * 768 * 2048,
    }
    if args.sizes:
        sizes = {}
        for item in args.sizes.split(","):
            name, value = item.split("=", 1)
            sizes[name] = _size_arg(value)
    else:
        sizes = default_sizes

    max_bytes = _read_int(PTE_MAX_BYTES_PARAM)
    if max_bytes is not None:
        too_large = {name: size for name, size in sizes.items() if size > max_bytes}
        if too_large:
            raise SystemExit(f"uvm_moe_slot_pte_remap_max_bytes={max_bytes} too small for {too_large}")

    counts = [int(item) for item in args.counts.split(",") if item]
    results = []
    for size_name, remap_bytes in sizes.items():
        remap_bytes = align_up(remap_bytes, os.sysconf("SC_PAGE_SIZE"))
        cross_options = [True] if remap_bytes > UVM_VA_BLOCK_SIZE else [False, True]
        for count in counts:
            for crosses in cross_options:
                for repeat in range(args.repeats):
                    result = run_case(args, size_name, remap_bytes, count, crosses)
                    result["repeat"] = repeat
                    results.append(result)
                    print(
                        f"{size_name} bytes={remap_bytes} count={count} cross={crosses} "
                        f"repeat={repeat} passed={result['passed']} "
                        f"remap_ms={result['remap']['latency_ms']:.3f}",
                        flush=True,
                    )

    output = {
        "device": args.device,
        "counts": counts,
        "repeats": args.repeats,
        "sizes": sizes,
        "results": results,
        "summary": summarize(results),
        "passed": all(bool(result.get("passed")) for result in results),
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
