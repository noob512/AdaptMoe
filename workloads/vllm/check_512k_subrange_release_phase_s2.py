#!/usr/bin/env python3
"""Phase S2 smoke for releasing one 512KiB logical range as 8x64KiB chunks."""

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
    ExpertOffloadReleaseParams,
    UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT,
    UVM_EXPERT_OFFLOAD_RELEASE_IOCTL,
    _pattern,
    _query_residency,
    _register_expert_region,
    _unregister_region,
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

UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K = 1 << 2


def _release_region_subchunk_64k(fd: int, gpu_uuid, region_id: int) -> dict[str, Any]:
    params = ExpertOffloadReleaseParams()
    params.region_id = int(region_id)
    params.gpu_uuid = gpu_uuid
    params.flags = (
        UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT
        | UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K
    )
    ret, errno_value = ioctl(fd, UVM_EXPERT_OFFLOAD_RELEASE_IOCTL, params)
    return {
        "ret": int(ret),
        "errno": int(errno_value),
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
        "ok": int(ret) == 0 and int(params.rmStatus) == 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    block_bytes = UVM_VA_BLOCK_SIZE
    target_bytes = CHUNK_512K
    left_bytes = CHUNK_512K
    right_bytes = block_bytes - left_bytes - target_bytes
    region_id = int(args.region_id)
    fd = -1
    target: ManagedAllocation | None = None
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(args.device),
        "block_bytes": int(block_bytes),
        "logical_release_bytes": int(target_bytes),
        "physical_chunk_bytes": int(PAGE_64K),
        "expected_released_chunks": int(target_bytes // PAGE_64K),
        "region_id": int(region_id),
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

        left_payload = _pattern(left_bytes, 17)
        target_payload = _pattern(target_bytes, 71)
        right_payload = _pattern(right_bytes, 139)
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

        migrate_gpu = native_migrate(fd, base, block_bytes, gpu_uuid)
        result["native_migrate_to_gpu"] = migrate_gpu
        before_whole = _query_residency(fd, gpu_uuid, base, block_bytes)
        before_target = _query_residency(fd, gpu_uuid, target_base, target_bytes)
        before_left = _query_residency(fd, gpu_uuid, base, left_bytes)
        before_right = _query_residency(fd, gpu_uuid, right_base, right_bytes)
        result["residency_before_whole"] = before_whole
        result["residency_before_target"] = before_target
        result["residency_before_left_neighbor"] = before_left
        result["residency_before_right_neighbor"] = before_right
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

        after_whole = _query_residency(fd, gpu_uuid, base, block_bytes)
        after_target = _query_residency(fd, gpu_uuid, target_base, target_bytes)
        after_left = _query_residency(fd, gpu_uuid, base, left_bytes)
        after_right = _query_residency(fd, gpu_uuid, right_base, right_bytes)
        result["residency_after_whole"] = after_whole
        result["residency_after_target"] = after_target
        result["residency_after_left_neighbor"] = after_left
        result["residency_after_right_neighbor"] = after_right

        cpu_target_after = ctypes.string_at(target_base, target_bytes)
        result["cpu_read_after_release_matches_target_payload"] = (
            cpu_target_after == target_payload
        )
        result["gpu_read_after_release_left_neighbor_matches"] = (
            kernel_readback(base, left_bytes, args.device) == left_payload
        )
        result["gpu_read_after_release_right_neighbor_matches"] = (
            kernel_readback(right_base, right_bytes, args.device) == right_payload
        )
        after_neighbor_reads = {
            "left": _query_residency(fd, gpu_uuid, base, left_bytes),
            "target": _query_residency(fd, gpu_uuid, target_base, target_bytes),
            "right": _query_residency(fd, gpu_uuid, right_base, right_bytes),
        }
        result["residency_after_neighbor_gpu_reads"] = after_neighbor_reads

        neighbor_gpu_after = int(after_left.get("gpu_bytes") or 0) + int(
            after_right.get("gpu_bytes") or 0
        )
        result["checks"] = {
            "before_whole_gpu_resident_2mb": before_whole.get("gpu_bytes") == block_bytes,
            "before_target_gpu_resident_512k": before_target.get("gpu_bytes") == target_bytes,
            "release_ioctl_ok": release.get("ok") is True,
            "release_attempted_512k": release.get("attempted_bytes") == target_bytes,
            "release_released_512k": release.get("released_bytes") == target_bytes,
            "release_no_failures": release.get("failure_blocks") == 0,
            "after_target_gpu_zero": after_target.get("gpu_bytes") == 0,
            "after_target_cpu_512k": after_target.get("cpu_bytes") == target_bytes,
            "after_neighbors_gpu_1p5mb": neighbor_gpu_after == block_bytes - target_bytes,
            "after_whole_gpu_1p5mb": after_whole.get("gpu_bytes") == block_bytes - target_bytes,
            "after_whole_cpu_at_least_512k": int(after_whole.get("cpu_bytes") or 0) >= target_bytes,
            "cpu_target_payload_ok": result["cpu_read_after_release_matches_target_payload"],
            "left_neighbor_payload_ok": result["gpu_read_after_release_left_neighbor_matches"],
            "right_neighbor_payload_ok": result["gpu_read_after_release_right_neighbor_matches"],
            "target_still_gpu_zero_after_neighbor_reads": (
                after_neighbor_reads["target"].get("gpu_bytes") == 0
            ),
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
    parser.add_argument("--region-id", type=lambda value: int(value, 0), default=0x5120000002)
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
