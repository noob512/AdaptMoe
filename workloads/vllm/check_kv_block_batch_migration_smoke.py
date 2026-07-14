#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from vllm.device_allocator.uvm import (
    batch_prefetch_ranges_to_cpu,
    batch_prefetch_ranges_to_device,
    cleanup_uvm_local_memory_pool,
    driver_batch_query_ranges_residency_bytes,
    uvm_local_memory_pool,
)


def now() -> float:
    return time.perf_counter()


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def query_ranges(
    ranges: list[tuple[int, int]],
    *,
    device: int,
) -> dict[str, Any]:
    result = dict(driver_batch_query_ranges_residency_bytes(ranges, device=device))
    queried = int(result.get("queried_bytes") or 0)
    result["supported"] = bool(result.get("ok")) and queried > 0
    result["expected_bytes"] = int(sum(length for _, length in ranges))
    result["gpu_ratio"] = (
        int(result.get("gpu_bytes") or 0) / queried if queried else 0.0
    )
    result["cpu_ratio"] = (
        int(result.get("cpu_bytes") or 0) / queried if queried else 0.0
    )
    return result


def block_ranges(
    tensors: list[torch.Tensor],
    block_ids: list[int],
    range_bytes: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for block_id in block_ids:
        offset = block_id * range_bytes
        for tensor in tensors:
            ranges.append((int(tensor.data_ptr()) + offset, range_bytes))
    return ranges


def contiguous_tensor_ranges(tensors: list[torch.Tensor]) -> list[tuple[int, int]]:
    return [(int(t.data_ptr()), int(t.numel() * t.element_size())) for t in tensors]


def prefetch_blocks(
    tensors: list[torch.Tensor],
    block_ids: list[int],
    range_bytes: int,
    *,
    device: int,
    mode: str,
    progress_every_blocks: int,
    stage_name: str,
) -> dict[str, Any]:
    expected_ranges = len(block_ids) * len(tensors)
    expected_bytes = expected_ranges * range_bytes
    submit_start = now()
    accepted = 0
    per_call_accepted: list[int] = []
    print(
        json.dumps(
            {
                "event": "prefetch_stage_begin",
                "stage": stage_name,
                "mode": mode,
                "device": device,
                "blocks": len(block_ids),
                "expected_ranges": expected_ranges,
                "bytes": expected_bytes,
                "mib": mib(expected_bytes),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if mode == "all":
        ranges = block_ranges(tensors, block_ids, range_bytes)
        accepted = (
            batch_prefetch_ranges_to_cpu(ranges)
            if device < 0
            else batch_prefetch_ranges_to_device(ranges, device=device)
        )
        per_call_accepted.append(int(accepted))
    elif mode == "block":
        for index, block_id in enumerate(block_ids, start=1):
            ranges = block_ranges(tensors, [block_id], range_bytes)
            current = (
                batch_prefetch_ranges_to_cpu(ranges)
                if device < 0
                else batch_prefetch_ranges_to_device(ranges, device=device)
            )
            accepted += int(current)
            per_call_accepted.append(int(current))
            if progress_every_blocks > 0 and (
                index == 1
                or index == len(block_ids)
                or index % progress_every_blocks == 0
            ):
                elapsed = now() - submit_start
                print(
                    json.dumps(
                        {
                            "event": "prefetch_stage_progress",
                            "stage": stage_name,
                            "device": device,
                            "blocks_done": index,
                            "blocks_total": len(block_ids),
                            "last_block_id": block_id,
                            "accepted_ranges": int(accepted),
                            "expected_ranges_so_far": index * len(tensors),
                            "elapsed_seconds": elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    submit_seconds = now() - submit_start
    sync_start = now()
    torch.cuda.synchronize()
    sync_seconds = now() - sync_start
    total_seconds = submit_seconds + sync_seconds
    print(
        json.dumps(
            {
                "event": "prefetch_stage_end",
                "stage": stage_name,
                "device": device,
                "accepted_ranges": int(accepted),
                "expected_ranges": expected_ranges,
                "all_ranges_accepted": int(accepted) == expected_ranges,
                "total_seconds": total_seconds,
                "mib_per_second": (
                    mib(expected_bytes) / total_seconds if total_seconds > 0 else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "mode": mode,
        "device": device,
        "blocks": len(block_ids),
        "expected_ranges": expected_ranges,
        "accepted_ranges": int(accepted),
        "all_ranges_accepted": int(accepted) == expected_ranges,
        "bytes": expected_bytes,
        "mib": mib(expected_bytes),
        "submit_seconds": submit_seconds,
        "sync_seconds": sync_seconds,
        "total_seconds": total_seconds,
        "mib_per_second": mib(expected_bytes) / total_seconds if total_seconds > 0 else None,
        "per_call_min_accepted": min(per_call_accepted) if per_call_accepted else 0,
        "per_call_max_accepted": max(per_call_accepted) if per_call_accepted else 0,
        "per_call_median_accepted": (
            statistics.median(per_call_accepted) if per_call_accepted else 0
        ),
    }


def fill_initial_payload(
    tensors: list[torch.Tensor],
    *,
    num_blocks: int,
    range_bytes: int,
) -> None:
    values = torch.arange(num_blocks, device="cuda", dtype=torch.int32)
    for layer, tensor in enumerate(tensors):
        pattern = ((values * 31 + layer * 17) % 251).to(torch.uint8)
        tensor.copy_(pattern[:, None].expand(num_blocks, range_bytes))
    torch.cuda.synchronize()


def verify_initial_payload(
    tensors: list[torch.Tensor],
    block_ids: list[int],
) -> dict[str, Any]:
    failures: list[dict[str, int]] = []
    for layer, tensor in enumerate(tensors):
        for block_id in block_ids:
            expected = int((block_id * 31 + layer * 17) % 251)
            if not bool(torch.all(tensor[block_id] == expected).item()):
                failures.append(
                    {
                        "layer": layer,
                        "block_id": block_id,
                        "expected": expected,
                    }
                )
                if len(failures) >= 8:
                    break
        if len(failures) >= 8:
            break
    return {
        "ok": not failures,
        "checked_layers": len(tensors),
        "checked_blocks": len(block_ids),
        "failures": failures,
    }


def write_selected_payload(
    tensors: list[torch.Tensor],
    block_ids: list[int],
) -> None:
    if not block_ids:
        return
    start = min(block_ids)
    end = max(block_ids) + 1
    if block_ids != list(range(start, end)):
        index = torch.tensor(block_ids, device="cuda", dtype=torch.long)
        for layer, tensor in enumerate(tensors):
            tensor[index].fill_(int((layer * 19 + 7) % 251))
    else:
        for layer, tensor in enumerate(tensors):
            tensor[start:end].fill_(int((layer * 19 + 7) % 251))
    torch.cuda.synchronize()


def verify_selected_written_payload(
    tensors: list[torch.Tensor],
    block_ids: list[int],
) -> dict[str, Any]:
    failures: list[dict[str, int]] = []
    if not block_ids:
        return {"ok": True, "failures": failures}
    start = min(block_ids)
    end = max(block_ids) + 1
    contiguous = block_ids == list(range(start, end))
    index = None if contiguous else torch.tensor(block_ids, device="cuda", dtype=torch.long)
    for layer, tensor in enumerate(tensors):
        expected = int((layer * 19 + 7) % 251)
        view = tensor[start:end] if contiguous else tensor[index]
        if not bool(torch.all(view == expected).item()):
            failures.append({"layer": layer, "expected": expected})
            if len(failures) >= 8:
                break
    return {
        "ok": not failures,
        "checked_layers": len(tensors),
        "checked_blocks": len(block_ids),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic writable KV block batch migration smoke."
    )
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--range-bytes", type=int, default=32768)
    parser.add_argument("--migrate-start", type=int, default=64)
    parser.add_argument("--migrate-count", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--progress-every-blocks", type=int, default=128)
    parser.add_argument(
        "--mode",
        choices=("block", "all"),
        default="block",
        help="block means one batch prefetch call per logical KV block.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_blocks <= 0 or args.layers <= 0 or args.range_bytes <= 0:
        raise SystemExit("num-blocks, layers, and range-bytes must be positive")
    if args.migrate_start < 0 or args.migrate_count <= 0:
        raise SystemExit("invalid migrate range")
    if args.migrate_start + args.migrate_count > args.num_blocks:
        raise SystemExit("migrate range exceeds num-blocks")

    torch.cuda.set_device(args.device)
    torch.cuda.synchronize()

    migrate_blocks = list(
        range(args.migrate_start, args.migrate_start + args.migrate_count)
    )
    neighbor_blocks = [
        block
        for block in (0, args.migrate_start - 1, args.migrate_start + args.migrate_count)
        if 0 <= block < args.num_blocks and block not in migrate_blocks
    ]

    result: dict[str, Any] = {
        "passed": False,
        "config": {
            "num_blocks": args.num_blocks,
            "layers": args.layers,
            "range_bytes": args.range_bytes,
            "bytes_per_block": args.layers * args.range_bytes,
            "migrate_start": args.migrate_start,
            "migrate_count": args.migrate_count,
            "migrate_bytes": args.migrate_count * args.layers * args.range_bytes,
            "mode": args.mode,
            "device": args.device,
            "progress_every_blocks": args.progress_every_blocks,
        },
        "stages": {},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        with uvm_local_memory_pool(enable_prefetch=True):
            alloc_start = now()
            tensors = [
                torch.empty(
                    (args.num_blocks, args.range_bytes),
                    device="cuda",
                    dtype=torch.uint8,
                )
                for _ in range(args.layers)
            ]
            torch.cuda.synchronize()
            result["stages"]["allocate"] = {
                "seconds": now() - alloc_start,
                "total_bytes": args.num_blocks * args.layers * args.range_bytes,
                "total_mib": mib(args.num_blocks * args.layers * args.range_bytes),
            }

            fill_start = now()
            fill_initial_payload(
                tensors,
                num_blocks=args.num_blocks,
                range_bytes=args.range_bytes,
            )
            result["stages"]["initial_fill"] = {"seconds": now() - fill_start}

            whole_ranges = contiguous_tensor_ranges(tensors)
            whole_gpu = batch_prefetch_ranges_to_device(whole_ranges, device=args.device)
            torch.cuda.synchronize()
            result["stages"]["prefetch_whole_gpu"] = {
                "accepted_ranges": int(whole_gpu),
                "expected_ranges": len(whole_ranges),
                "residency": query_ranges(whole_ranges, device=args.device),
            }

            selected_ranges = block_ranges(tensors, migrate_blocks, args.range_bytes)
            neighbor_ranges = block_ranges(tensors, neighbor_blocks, args.range_bytes)
            result["stages"]["selected_before_cpu_offload"] = query_ranges(
                selected_ranges,
                device=args.device,
            )
            result["stages"]["neighbor_before_cpu_offload"] = query_ranges(
                neighbor_ranges,
                device=args.device,
            )

            result["stages"]["selected_prefetch_cpu_1"] = prefetch_blocks(
                tensors,
                migrate_blocks,
                args.range_bytes,
                device=-1,
                mode=args.mode,
                progress_every_blocks=args.progress_every_blocks,
                stage_name="selected_prefetch_cpu_1",
            )
            result["stages"]["selected_after_cpu_offload_1"] = query_ranges(
                selected_ranges,
                device=args.device,
            )
            result["stages"]["neighbor_after_cpu_offload_1"] = query_ranges(
                neighbor_ranges,
                device=args.device,
            )

            result["stages"]["selected_prefetch_gpu_1"] = prefetch_blocks(
                tensors,
                migrate_blocks,
                args.range_bytes,
                device=args.device,
                mode=args.mode,
                progress_every_blocks=args.progress_every_blocks,
                stage_name="selected_prefetch_gpu_1",
            )
            result["stages"]["selected_after_gpu_restore_1"] = query_ranges(
                selected_ranges,
                device=args.device,
            )
            result["stages"]["verify_initial_after_restore"] = verify_initial_payload(
                tensors,
                migrate_blocks,
            )

            write_start = now()
            write_selected_payload(tensors, migrate_blocks)
            result["stages"]["selected_gpu_write"] = {"seconds": now() - write_start}
            result["stages"]["verify_written_on_gpu"] = (
                verify_selected_written_payload(tensors, migrate_blocks)
            )

            result["stages"]["selected_prefetch_cpu_2_after_write"] = prefetch_blocks(
                tensors,
                migrate_blocks,
                args.range_bytes,
                device=-1,
                mode=args.mode,
                progress_every_blocks=args.progress_every_blocks,
                stage_name="selected_prefetch_cpu_2_after_write",
            )
            result["stages"]["selected_after_cpu_offload_2"] = query_ranges(
                selected_ranges,
                device=args.device,
            )
            result["stages"]["selected_prefetch_gpu_2_after_write"] = prefetch_blocks(
                tensors,
                migrate_blocks,
                args.range_bytes,
                device=args.device,
                mode=args.mode,
                progress_every_blocks=args.progress_every_blocks,
                stage_name="selected_prefetch_gpu_2_after_write",
            )
            result["stages"]["selected_after_gpu_restore_2"] = query_ranges(
                selected_ranges,
                device=args.device,
            )
            result["stages"]["verify_written_after_roundtrip"] = (
                verify_selected_written_payload(tensors, migrate_blocks)
            )

            expected_selected_bytes = args.migrate_count * args.layers * args.range_bytes
            functional_checks = {
                "whole_gpu_prefetch_accepted": (
                    result["stages"]["prefetch_whole_gpu"]["accepted_ranges"]
                    == len(whole_ranges)
                ),
                "selected_cpu_1_accepted": result["stages"][
                    "selected_prefetch_cpu_1"
                ]["all_ranges_accepted"],
                "selected_gpu_1_accepted": result["stages"][
                    "selected_prefetch_gpu_1"
                ]["all_ranges_accepted"],
                "selected_cpu_2_accepted": result["stages"][
                    "selected_prefetch_cpu_2_after_write"
                ]["all_ranges_accepted"],
                "selected_gpu_2_accepted": result["stages"][
                    "selected_prefetch_gpu_2_after_write"
                ]["all_ranges_accepted"],
                "initial_payload_ok": result["stages"][
                    "verify_initial_after_restore"
                ]["ok"],
                "gpu_write_ok": result["stages"]["verify_written_on_gpu"]["ok"],
                "written_roundtrip_ok": result["stages"][
                    "verify_written_after_roundtrip"
                ]["ok"],
            }
            residency_diagnostics = {
                "expected_selected_bytes": expected_selected_bytes,
                "selected_cpu_after_offload_1_supported": result["stages"][
                    "selected_after_cpu_offload_1"
                ]["supported"],
                "selected_gpu_after_restore_1_supported": result["stages"][
                    "selected_after_gpu_restore_1"
                ]["supported"],
                "selected_cpu_after_offload_2_supported": result["stages"][
                    "selected_after_cpu_offload_2"
                ]["supported"],
                "selected_gpu_after_restore_2_supported": result["stages"][
                    "selected_after_gpu_restore_2"
                ]["supported"],
                "selected_cpu_residency_after_offload_1": (
                    int(result["stages"]["selected_after_cpu_offload_1"].get("cpu_bytes") or 0)
                    == expected_selected_bytes
                ),
                "selected_gpu_residency_after_restore_1": (
                    int(result["stages"]["selected_after_gpu_restore_1"].get("gpu_bytes") or 0)
                    == expected_selected_bytes
                ),
                "selected_cpu_residency_after_offload_2": (
                    int(result["stages"]["selected_after_cpu_offload_2"].get("cpu_bytes") or 0)
                    == expected_selected_bytes
                ),
                "selected_gpu_residency_after_restore_2": (
                    int(result["stages"]["selected_after_gpu_restore_2"].get("gpu_bytes") or 0)
                    == expected_selected_bytes
                ),
                "note": (
                    "Residency byte query is diagnostic for this synthetic KV layout; "
                    "functional pass requires batch prefetch acceptance plus data "
                    "correctness after GPU write and CPU/GPU roundtrip."
                ),
            }
            result["checks"] = functional_checks
            result["residency_diagnostics"] = residency_diagnostics
            result["passed"] = all(functional_checks.values())

            del tensors
    finally:
        cleanup_uvm_local_memory_pool()

    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
