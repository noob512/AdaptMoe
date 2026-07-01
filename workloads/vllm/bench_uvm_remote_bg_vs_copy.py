#!/usr/bin/env python3
"""A/B/C/D microbench for UVM remote-read background promotion vs copies.

Cases:
  sync_copy_miss: CPU checkpoint staging -> CUDA tensor -> wait -> tiny forward
  uva_remote:     CPU pinned checkpoint staging -> CUDA UVA view -> tiny forward
  uvm_fault:      UVM tensor is CPU-prefetched, tiny forward triggers GPU access
  uvm_prefetch:   UVM tensor is CPU-prefetched, then GPU-prefetched before forward
  uvm_remote_read: UVM tensor is CPU-prefetched, prepare remote-read PTEs, then
                  tiny forward reads remotely without background promotion
  uvm_remote_bg:  UVM tensor is CPU-prefetched, prepare remote-read PTEs, first
                  forward reads remotely, start background promotion, second
                  forward observes post-promotion behavior

The script runs one case per process. The shell runner aggregates results so
UVM allocator teardown crashes do not prevent the other cases from running.
"""

from __future__ import annotations

import argparse
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

from bench_hot_cold_expert_checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    CheckpointReader,
    _alloc_gpu_tensor,
    _copy_expert_to_batch,
    _dtype_bytes,
    _empty_cpu,
    _tiny_forward,
)


def _now() -> float:
    return time.perf_counter()


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    ordered = sorted(values)

    def pick(p: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[index] * 1000.0

    return {"p50_ms": pick(0.50), "p95_ms": pick(0.95), "p99_ms": pick(0.99)}


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _ranges_for(w13: torch.Tensor, w2: torch.Tensor) -> list[tuple[int, int]]:
    return [
        (int(w13.data_ptr()), _tensor_bytes(w13)),
        (int(w2.data_ptr()), _tensor_bytes(w2)),
    ]


def _load_expert_batch_cpu(
    reader: CheckpointReader,
    *,
    layer: int,
    expert: int,
    expert_count: int,
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    start = _now()
    w13 = _empty_cpu(
        (expert_count, w13_shape[0], w13_shape[1]), dtype, pin_memory=pin_memory
    )
    w2 = _empty_cpu(
        (expert_count, w2_shape[0], w2_shape[1]), dtype, pin_memory=pin_memory
    )
    for local in range(expert_count):
        _copy_expert_to_batch(
            reader,
            layer=layer,
            expert=expert + local,
            local=local,
            batch_w13=w13,
            batch_w2=w2,
        )
    return w13, w2, _now() - start


def _load_one_expert_gpu(
    src_w13: torch.Tensor,
    src_w2: torch.Tensor,
    *,
    device: torch.device,
    backing: str,
) -> tuple[torch.Tensor, torch.Tensor, str, float]:
    start = _now()
    dst_w13, backing_w13 = _alloc_gpu_tensor(
        tuple(src_w13.shape), src_w13.dtype, device, backing
    )
    dst_w2, backing_w2 = _alloc_gpu_tensor(
        tuple(src_w2.shape), src_w2.dtype, device, backing
    )
    dst_w13.copy_(src_w13)
    dst_w2.copy_(src_w2)
    torch.cuda.synchronize(device)
    effective = backing_w13 if backing_w13 == backing_w2 else f"{backing_w13},{backing_w2}"
    return dst_w13, dst_w2, effective, _now() - start


def _cuda_view_from_pinned_cpu(
    cpu_tensor: torch.Tensor,
) -> torch.Tensor:
    import vllm._C  # noqa: F401  # register torch.ops._C custom ops
    from vllm.utils.torch_utils import get_cuda_view_from_cpu_tensor

    if not cpu_tensor.is_pinned():
        raise RuntimeError("UVA remote access requires pinned CPU tensors")
    return get_cuda_view_from_cpu_tensor(cpu_tensor)


def _prefetch_ranges_cpu(ranges: list[tuple[int, int]]) -> tuple[int, str]:
    from vllm.device_allocator.uvm import batch_prefetch_ranges_to_cpu

    successes = batch_prefetch_ranges_to_cpu(ranges)
    torch.cuda.synchronize()
    return int(successes), "batch_prefetch_ranges_to_cpu"


def _prefetch_ranges_gpu(
    ranges: list[tuple[int, int]], *, device: int
) -> tuple[int, str]:
    from vllm.device_allocator.uvm import batch_prefetch_ranges_to_device

    successes = batch_prefetch_ranges_to_device(ranges, device)
    torch.cuda.synchronize(device)
    return int(successes), "batch_prefetch_ranges_to_device"


def _prepare_remote_bg(
    ranges: list[tuple[int, int]], *, device: int
) -> tuple[int, int, int, int]:
    from vllm.device_allocator.uvm import driver_prepare_ranges_remote_read_bg_deferred

    return tuple(
        int(x)
        for x in driver_prepare_ranges_remote_read_bg_deferred(ranges, device=device)
    )


def _prepare_remote_read(
    ranges: list[tuple[int, int]], *, device: int
) -> tuple[int, int, int, int]:
    from vllm.device_allocator.uvm import driver_prepare_ranges_remote_read

    return tuple(
        int(x) for x in driver_prepare_ranges_remote_read(ranges, device=device)
    )


def _start_bg_promotion(
    ranges: list[tuple[int, int]], *, device: int
) -> tuple[int, int, int, int]:
    from vllm.device_allocator.uvm import driver_start_ranges_bg_promotion

    return tuple(int(x) for x in driver_start_ranges_bg_promotion(ranges, device=device))


def _forward_timed(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    x: torch.Tensor,
    device: torch.device,
    repeats: int,
) -> tuple[list[float], torch.Tensor]:
    times: list[float] = []
    out = torch.empty((hidden,), dtype=torch.float32, device=device)
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = _now()
        if w13.dim() == 3:
            out = torch.zeros((hidden,), dtype=torch.float32, device=device)
            for local in range(w13.shape[0]):
                out = out + _tiny_forward(
                    w13[local],
                    w2[local],
                    hidden=hidden,
                    intermediate=intermediate,
                    x=x,
                )
        else:
            out = _tiny_forward(
                w13,
                w2,
                hidden=hidden,
                intermediate=intermediate,
                x=x,
            )
        torch.cuda.synchronize(device)
        times.append(_now() - start)
    return times, out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    reader = CheckpointReader(args.checkpoint_dir)
    try:
        shapes = reader.expert_shapes(args.layer, args.expert)
        dtype = reader.expert_tensor(args.layer, args.expert, "gate_proj").dtype
        w13_shape = tuple(shapes["w13_shape"])
        w2_shape = tuple(shapes["w2_shape"])
        hidden = min(args.hidden, w13_shape[1], w2_shape[0])
        intermediate = min(args.intermediate, w13_shape[0] // 2, w2_shape[1])

        src_w13, src_w2, checkpoint_read_seconds = _load_expert_batch_cpu(
            reader,
            layer=args.layer,
            expert=args.expert,
            expert_count=args.expert_count,
            w13_shape=w13_shape,
            w2_shape=w2_shape,
            dtype=dtype,
            pin_memory=args.pin_memory,
        )
        ref_w13, ref_w2, _, ref_copy_seconds = _load_one_expert_gpu(
            src_w13, src_w2, device=device, backing="cuda"
        )
        x = torch.linspace(-0.25, 0.25, steps=hidden, dtype=torch.float32, device=device)
        _, ref_out = _forward_timed(
            ref_w13,
            ref_w2,
            hidden=hidden,
            intermediate=intermediate,
            x=x,
            device=device,
            repeats=1,
        )

        result: dict[str, Any] = {
            "case": args.case,
            "config": {
                "checkpoint_dir": str(args.checkpoint_dir),
                "layer": args.layer,
                "expert": args.expert,
                "expert_count": args.expert_count,
                "device": args.device,
                "hidden": hidden,
                "intermediate": intermediate,
                "repeats": args.repeats,
                "pin_memory": args.pin_memory,
            },
            "shapes": shapes,
            "batch_bytes": _tensor_bytes(src_w13) + _tensor_bytes(src_w2),
            "per_expert_bytes": (_tensor_bytes(src_w13) + _tensor_bytes(src_w2))
            // max(args.expert_count, 1),
            "checkpoint_read_seconds": checkpoint_read_seconds,
            "reference_cuda_copy_seconds": ref_copy_seconds,
            "events": {},
        }

        setup_start = _now()
        if args.case == "sync_copy_miss":
            w13, w2, backing, copy_seconds = _load_one_expert_gpu(
                src_w13, src_w2, device=device, backing="cuda"
            )
            result["events"]["miss_copy_seconds"] = copy_seconds
            result["effective_backing"] = backing
        elif args.case == "uva_remote":
            view_start = _now()
            w13 = _cuda_view_from_pinned_cpu(src_w13)
            w2 = _cuda_view_from_pinned_cpu(src_w2)
            result["events"]["uva_view_seconds"] = _now() - view_start
            result["events"]["uva_cpu_w13_pinned"] = bool(src_w13.is_pinned())
            result["events"]["uva_cpu_w2_pinned"] = bool(src_w2.is_pinned())
            result["effective_backing"] = "pinned_uva"
        elif args.case in (
            "uvm_fault",
            "uvm_prefetch",
            "uvm_remote_read",
            "uvm_remote_bg",
        ):
            w13, w2, backing, copy_seconds = _load_one_expert_gpu(
                src_w13, src_w2, device=device, backing="uvm"
            )
            ranges = _ranges_for(w13, w2)
            cpu_successes, cpu_executor = _prefetch_ranges_cpu(ranges)
            result["effective_backing"] = backing
            result["events"]["uvm_initial_copy_seconds"] = copy_seconds
            result["events"]["prefetch_cpu_successes"] = cpu_successes
            result["events"]["prefetch_cpu_executor"] = cpu_executor
            if args.case == "uvm_prefetch":
                prefetch_start = _now()
                gpu_successes, gpu_executor = _prefetch_ranges_gpu(
                    ranges, device=args.device
                )
                result["events"]["prefetch_gpu_seconds"] = _now() - prefetch_start
                result["events"]["prefetch_gpu_successes"] = gpu_successes
                result["events"]["prefetch_gpu_executor"] = gpu_executor
            elif args.case == "uvm_remote_read":
                prepare_start = _now()
                prepare_result = _prepare_remote_read(ranges, device=args.device)
                result["events"]["remote_read_prepare_seconds"] = (
                    _now() - prepare_start
                )
                result["events"]["remote_read_prepare_result"] = list(prepare_result)
            elif args.case == "uvm_remote_bg":
                prepare_start = _now()
                prepare_result = _prepare_remote_bg(ranges, device=args.device)
                result["events"]["remote_bg_prepare_seconds"] = _now() - prepare_start
                result["events"]["remote_bg_prepare_result"] = list(prepare_result)
        else:
            raise ValueError(f"unknown case {args.case!r}")

        torch.cuda.synchronize(device)
        result["setup_seconds"] = _now() - setup_start

        first_times, first_out = _forward_timed(
            w13,
            w2,
            hidden=hidden,
            intermediate=intermediate,
            x=x,
            device=device,
            repeats=args.repeats,
        )

        if args.case == "uvm_remote_bg":
            ranges = _ranges_for(w13, w2)
            promote_start = _now()
            promote_result = _start_bg_promotion(ranges, device=args.device)
            result["events"]["remote_bg_start_promotion_seconds"] = (
                _now() - promote_start
            )
            result["events"]["remote_bg_start_promotion_result"] = list(promote_result)
            if args.promotion_wait_ms > 0:
                time.sleep(args.promotion_wait_ms / 1000.0)
            torch.cuda.synchronize(device)

        second_times, second_out = _forward_timed(
            w13,
            w2,
            hidden=hidden,
            intermediate=intermediate,
            x=x,
            device=device,
            repeats=args.repeats,
        )

        first_max_abs = float((ref_out - first_out).abs().max().detach().cpu().item())
        second_max_abs = float((ref_out - second_out).abs().max().detach().cpu().item())
        result["first_access"] = {
            "times_ms": [t * 1000.0 for t in first_times],
            **_percentiles(first_times),
            "mean_ms": statistics.fmean(first_times) * 1000.0,
        }
        result["second_access"] = {
            "times_ms": [t * 1000.0 for t in second_times],
            **_percentiles(second_times),
            "mean_ms": statistics.fmean(second_times) * 1000.0,
        }
        result["correctness"] = {
            "first_max_abs_diff": first_max_abs,
            "second_max_abs_diff": second_max_abs,
            "tolerance": args.tolerance,
            "passed": first_max_abs <= args.tolerance
            and second_max_abs <= args.tolerance,
        }
        result["torch_cuda_max_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        result["passed"] = bool(result["correctness"]["passed"])
        return result
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "sync_copy_miss",
            "uva_remote",
            "uvm_fault",
            "uvm_prefetch",
            "uvm_remote_read",
            "uvm_remote_bg",
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=113)
    parser.add_argument("--expert-count", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--promotion-wait-ms", type=float, default=20.0)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    try:
        result = run(args)
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(text + "\n", encoding="utf-8")
        return 0 if result.get("passed", False) else 1
    finally:
        try:
            from vllm.device_allocator.uvm import cleanup_uvm_local_memory_pool

            cleanup_uvm_local_memory_pool()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
