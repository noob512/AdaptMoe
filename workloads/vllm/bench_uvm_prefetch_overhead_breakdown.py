#!/usr/bin/env python3
"""Break down native UVM prefetch overhead.

Experiments:
  1. no-op prefetch: range is already GPU resident, then prefetch to GPU again.
  2. range-count sweep: fixed total bytes, split into N ranges before prefetch.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from bench_hot_cold_expert_checkpoint import _alloc_gpu_tensor  # noqa: E402
from bench_uvm_promotion_vs_h2d_copy import _query_residency  # noqa: E402


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[len(ordered) // 2],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _parse_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.endswith("mib"):
            sizes.append(int(float(token[:-3]) * 1024 * 1024))
        elif token.endswith("mb"):
            sizes.append(int(float(token[:-2]) * 1000 * 1000))
        elif token.endswith("kib"):
            sizes.append(int(float(token[:-3]) * 1024))
        elif token.endswith("kb"):
            sizes.append(int(float(token[:-2]) * 1000))
        else:
            sizes.append(int(token))
    return sizes


def _prefetch_ranges_cpu(ranges: list[tuple[int, int]]) -> int:
    from vllm.device_allocator.uvm import batch_prefetch_ranges_to_cpu

    return int(batch_prefetch_ranges_to_cpu(ranges))


def _prefetch_ranges_gpu(ranges: list[tuple[int, int]], device: int) -> int:
    from vllm.device_allocator.uvm import batch_prefetch_ranges_to_device

    return int(batch_prefetch_ranges_to_device(ranges, device=device))


def _event_timed_prefetch_gpu(
    ranges: list[tuple[int, int]],
    *,
    device: torch.device,
    device_index: int,
) -> tuple[float, float, int]:
    stream = torch.cuda.current_stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    submit_start = time.perf_counter()
    start.record(stream)
    result = _prefetch_ranges_gpu(ranges, device_index)
    end.record(stream)
    submit_ms = (time.perf_counter() - submit_start) * 1000.0
    end.synchronize()
    gpu_ms = float(start.elapsed_time(end))
    return submit_ms, gpu_ms, result


def _alloc_uvm_bytes(bytes_: int, device: torch.device) -> torch.Tensor:
    tensor, backing = _alloc_gpu_tensor((bytes_,), torch.uint8, device, "uvm")
    if backing != "uvm":
        raise RuntimeError(f"expected UVM backing, got {backing}")
    tensor.fill_(0xC3)
    torch.cuda.synchronize(device)
    return tensor


def run_noop(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bytes_ in _parse_sizes(args.noop_sizes):
        tensor = _alloc_uvm_bytes(bytes_, device)
        ranges = [(int(tensor.data_ptr()), bytes_)]
        try:
            _prefetch_ranges_cpu(ranges)
            torch.cuda.synchronize(device)
            cpu_residency = _query_residency(ranges, args.device)

            _prefetch_ranges_gpu(ranges, args.device)
            torch.cuda.synchronize(device)
            gpu_residency = _query_residency(ranges, args.device)

            submit_ms: list[float] = []
            gpu_ms: list[float] = []
            results: list[int] = []
            for _ in range(args.repeats):
                submit, elapsed, result = _event_timed_prefetch_gpu(
                    ranges,
                    device=device,
                    device_index=args.device,
                )
                submit_ms.append(submit)
                gpu_ms.append(elapsed)
                results.append(result)
            final_residency = _query_residency(ranges, args.device)
            rows.append({
                "kind": "noop_gpu_resident",
                "bytes": bytes_,
                "mib": bytes_ / 1024 / 1024,
                "range_count": 1,
                "repeats": args.repeats,
                "cpu_residency_after_cpu_prefetch": cpu_residency,
                "gpu_residency_before_noop": gpu_residency,
                "final_residency": final_residency,
                "prefetch_submit_ms": _stats(submit_ms),
                "prefetch_gpu_ms": _stats(gpu_ms),
                "prefetch_results": results,
            })
        finally:
            del tensor
            torch.cuda.empty_cache()
    return rows


def _split_ranges(base: int, total_bytes: int, count: int) -> list[tuple[int, int]]:
    chunk = total_bytes // count
    if chunk <= 0 or chunk % 4096 != 0:
        raise ValueError(
            f"total_bytes={total_bytes} cannot be split into {count} page-aligned ranges"
        )
    return [(base + i * chunk, chunk) for i in range(count)]


def run_range_sweep(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    total_bytes = _parse_sizes(args.range_total_size)[0]
    rows: list[dict[str, Any]] = []
    for count in [int(x) for x in args.range_counts.split(",") if x.strip()]:
        tensor = _alloc_uvm_bytes(total_bytes, device)
        ranges = _split_ranges(int(tensor.data_ptr()), total_bytes, count)
        try:
            submit_ms: list[float] = []
            gpu_ms: list[float] = []
            cpu_setup_ms: list[float] = []
            results: list[int] = []
            before_first: dict[str, Any] | None = None
            after_last: dict[str, Any] | None = None

            for _ in range(args.repeats):
                cpu_start = time.perf_counter()
                _prefetch_ranges_cpu(ranges)
                torch.cuda.synchronize(device)
                cpu_setup_ms.append((time.perf_counter() - cpu_start) * 1000.0)
                before = _query_residency(ranges, args.device)
                if before_first is None:
                    before_first = before

                submit, elapsed, result = _event_timed_prefetch_gpu(
                    ranges,
                    device=device,
                    device_index=args.device,
                )
                submit_ms.append(submit)
                gpu_ms.append(elapsed)
                results.append(result)
                after_last = _query_residency(ranges, args.device)

            rows.append({
                "kind": "fixed_total_range_sweep",
                "bytes": total_bytes,
                "mib": total_bytes / 1024 / 1024,
                "range_count": count,
                "range_mib": total_bytes / count / 1024 / 1024,
                "repeats": args.repeats,
                "cpu_prefetch_setup_ms": _stats(cpu_setup_ms),
                "before_first": before_first,
                "after_last": after_last,
                "prefetch_submit_ms": _stats(submit_ms),
                "prefetch_gpu_ms": _stats(gpu_ms),
                "prefetch_results": results,
            })
        finally:
            del tensor
            torch.cuda.empty_cache()
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    rows = run_noop(args, device)
    rows.extend(run_range_sweep(args, device))
    return {
        "config": {
            "device": args.device,
            "repeats": args.repeats,
            "noop_sizes": args.noop_sizes,
            "range_total_size": args.range_total_size,
            "range_counts": args.range_counts,
        },
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_rows.append({
            "kind": row["kind"],
            "mib": row["mib"],
            "range_count": row["range_count"],
            "range_mib": row.get("range_mib", row["mib"]),
            "submit_p50_ms": row["prefetch_submit_ms"]["p50_ms"],
            "submit_p95_ms": row["prefetch_submit_ms"]["p95_ms"],
            "gpu_p50_ms": row["prefetch_gpu_ms"]["p50_ms"],
            "gpu_p95_ms": row["prefetch_gpu_ms"]["p95_ms"],
            "before_gpu_bytes": (
                row.get("before_first")
                or row.get("gpu_residency_before_noop")
                or {}
            ).get("gpu_bytes"),
            "before_cpu_bytes": (
                row.get("before_first")
                or row.get("gpu_residency_before_noop")
                or {}
            ).get("cpu_bytes"),
            "after_gpu_bytes": (row.get("after_last") or row.get("final_residency") or {}).get("gpu_bytes"),
            "after_cpu_bytes": (row.get("after_last") or row.get("final_residency") or {}).get("cpu_bytes"),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--noop-sizes", default="4.5MiB,18MiB,72MiB,144MiB,288MiB")
    parser.add_argument("--range-total-size", default="144MiB")
    parser.add_argument("--range-counts", default="1,2,4,8,16,32,64")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.out_csv, result["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
