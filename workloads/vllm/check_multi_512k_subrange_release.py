#!/usr/bin/env python3
"""Release multiple 512KiB managed regions through the subchunk-64K path."""

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
    _pattern,
    _query_residency,
    _register_expert_region,
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
    native_migrate,
    open_initialized_uvm_fd,
)
from check_uvm_pmm_512k_phase_s0 import CHUNK_512K, PAGE_64K  # noqa: E402


def _cuda_mem_get_info() -> dict[str, int]:
    import ctypes.util

    libname = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(libname)
    cuda_mem_get_info = cudart.cudaMemGetInfo
    cuda_mem_get_info.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    cuda_mem_get_info.restype = ctypes.c_int

    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    ret = int(cuda_mem_get_info(ctypes.byref(free_bytes), ctypes.byref(total_bytes)))
    if ret != 0:
        return {"ret": ret, "free": 0, "total": 0, "used": 0}

    return {
        "ret": ret,
        "free": int(free_bytes.value),
        "total": int(total_bytes.value),
        "used": int(total_bytes.value - free_bytes.value),
    }


def _region_payloads(region_count: int) -> list[bytes]:
    return [_pattern(CHUNK_512K, 0x31 + index * 17) for index in range(region_count)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    region_count = int(args.region_count)
    if region_count <= 0:
        raise ValueError("--region-count must be positive")

    release_bytes = region_count * CHUNK_512K
    alloc_bytes = release_bytes + UVM_VA_BLOCK_SIZE
    base_region_id = int(args.base_region_id)
    fd = -1
    target: ManagedAllocation | None = None
    registered_region_ids: list[int] = []

    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(args.device),
        "region_count": region_count,
        "region_bytes": CHUNK_512K,
        "physical_chunk_bytes": PAGE_64K,
        "expected_chunks_per_region": CHUNK_512K // PAGE_64K,
        "expected_total_released_bytes": release_bytes,
        "block_bytes": UVM_VA_BLOCK_SIZE,
        "mem_info": {},
        "regions": [],
        "release_steps": [],
    }

    try:
        initialize_cuda_device(args.device)
        result["mem_info"]["initial"] = _cuda_mem_get_info()
        gpu_uuid = gpu_uuid_for_device(args.device)

        target = ManagedAllocation(args.device, alloc_bytes)
        raw_base = target.address
        base = ((raw_base + UVM_VA_BLOCK_SIZE - 1) // UVM_VA_BLOCK_SIZE) * UVM_VA_BLOCK_SIZE
        allocation_end = raw_base + alloc_bytes
        if base + release_bytes > allocation_end:
            raise RuntimeError("aligned release range does not fit managed allocation")

        payloads = _region_payloads(region_count)
        payload = b"".join(payloads)
        cuda_memcpy_host_to_device(base, payload)

        result.update({
            "managed_raw_base": raw_base,
            "managed_alloc_bytes": alloc_bytes,
            "release_base": base,
            "release_length": release_bytes,
            "release_base_2mb_aligned": bool(base % UVM_VA_BLOCK_SIZE == 0),
            "release_base_512k_aligned": bool(base % CHUNK_512K == 0),
        })
        result["mem_info"]["after_allocation_and_payload_copy"] = _cuda_mem_get_info()

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

        migrate = native_migrate(fd, base, release_bytes, gpu_uuid)
        result["native_migrate_to_gpu"] = migrate
        result["mem_info"]["after_native_migrate_to_gpu"] = _cuda_mem_get_info()
        result["residency_before_release_whole"] = _query_residency(
            fd, gpu_uuid, base, release_bytes
        )

        for index in range(region_count):
            region_base = base + index * CHUNK_512K
            region_id = base_region_id + index
            register = _register_expert_region(
                fd,
                region_id=region_id,
                base=region_base,
                byte_count=CHUNK_512K,
            )
            result["regions"].append({
                "index": index,
                "region_id": region_id,
                "base": region_base,
                "length": CHUNK_512K,
                "register": register,
            })
            if not register["ok"]:
                result["passed"] = False
                result["blocked_reason"] = f"region registration failed at index {index}"
                return result
            registered_region_ids.append(region_id)

        cumulative_released = 0
        for index in range(region_count):
            region = result["regions"][index]
            region_id = int(region["region_id"])
            region_base = int(region["base"])
            before_region = _query_residency(fd, gpu_uuid, region_base, CHUNK_512K)
            release = _release_region_subchunk_64k(fd, gpu_uuid, region_id)
            cumulative_released += int(release.get("released_bytes") or 0)
            after_region = _query_residency(fd, gpu_uuid, region_base, CHUNK_512K)
            after_whole = _query_residency(fd, gpu_uuid, base, release_bytes)
            cpu_payload_ok = ctypes.string_at(region_base, CHUNK_512K) == payloads[index]
            mem_info = _cuda_mem_get_info()

            step = {
                "index": index,
                "region_id": region_id,
                "base": region_base,
                "before_region": before_region,
                "release": release,
                "after_region": after_region,
                "after_whole": after_whole,
                "cumulative_released_bytes": cumulative_released,
                "expected_cumulative_released_bytes": (index + 1) * CHUNK_512K,
                "cpu_payload_ok": cpu_payload_ok,
                "mem_info": mem_info,
                "checks": {
                    "before_region_gpu_512k": before_region.get("gpu_bytes") == CHUNK_512K,
                    "release_ioctl_ok": release.get("ok") is True,
                    "release_attempted_512k": release.get("attempted_bytes") == CHUNK_512K,
                    "release_released_512k": release.get("released_bytes") == CHUNK_512K,
                    "release_no_failures": release.get("failure_blocks") == 0,
                    "after_region_gpu_zero": after_region.get("gpu_bytes") == 0,
                    "after_region_cpu_512k": after_region.get("cpu_bytes") == CHUNK_512K,
                    "after_whole_gpu_expected": (
                        after_whole.get("gpu_bytes")
                        == release_bytes - (index + 1) * CHUNK_512K
                    ),
                    "after_whole_cpu_at_least_expected": (
                        int(after_whole.get("cpu_bytes") or 0)
                        >= (index + 1) * CHUNK_512K
                    ),
                    "cumulative_released_expected": (
                        cumulative_released == (index + 1) * CHUNK_512K
                    ),
                    "cpu_payload_ok": cpu_payload_ok,
                },
            }
            step["passed"] = all(bool(value) for value in step["checks"].values())
            result["release_steps"].append(step)
            if args.stop_on_failure and not step["passed"]:
                break

        result["residency_after_release_whole"] = _query_residency(
            fd, gpu_uuid, base, release_bytes
        )
        result["mem_info"]["after_all_releases"] = _cuda_mem_get_info()
        result["total_released_bytes"] = cumulative_released
        result["success_count"] = sum(1 for step in result["release_steps"] if step["passed"])
        result["passed"] = (
            len(result["release_steps"]) == region_count
            and result["success_count"] == region_count
            and cumulative_released == release_bytes
            and result["residency_after_release_whole"].get("gpu_bytes") == 0
            and int(result["residency_after_release_whole"].get("cpu_bytes") or 0) >= release_bytes
        )
        if not result["passed"]:
            result["failure_reason"] = "one or more 512KiB release steps failed"
        return result
    finally:
        if fd >= 0:
            for region_id in reversed(registered_region_ids):
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
    parser.add_argument("--region-count", type=int, default=8)
    parser.add_argument("--base-region-id", type=lambda value: int(value, 0), default=0x5120001000)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--stop-on-failure", action="store_true")
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
