#!/usr/bin/env python3
"""Microbenchmark UVM expert-range migration latency.

This benchmark uses the repo's uvm_allocator.so directly. It allocates managed
ranges shaped like MoE expert weight slices, moves them to CPU, then measures
CPU/UVM -> GPU prefetch latency for different batch sizes and executors.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_uvm_allocator(repo_root: Path) -> Path:
    uvm_dir = repo_root / "workloads" / "vllm" / "vllm" / "uvm_test"
    subprocess.run(["make", "-B", "-C", str(uvm_dir), "uvm"], check=True)
    lib = uvm_dir / "uvm_allocator.so"
    if not lib.exists():
        raise FileNotFoundError(lib)
    return lib


def load_lib(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.uvm_malloc.restype = ctypes.c_void_p
    lib.uvm_malloc.argtypes = [
        ctypes.c_ssize_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uvm_free.restype = None
    lib.uvm_free.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ssize_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uvm_prefetch.restype = None
    lib.uvm_prefetch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uvm_batch_prefetch.restype = ctypes.c_int
    lib.uvm_batch_prefetch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uvm_driver_batch_prefetch_gpu.restype = ctypes.c_int
    lib.uvm_driver_batch_prefetch_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.uvm_driver_batch_prefetch_cpu.restype = ctypes.c_int
    lib.uvm_driver_batch_prefetch_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.uvm_driver_batch_query_residency_bytes.restype = ctypes.c_int
    lib.uvm_driver_batch_query_residency_bytes.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    if hasattr(lib, "uvm_is_managed_pointer"):
        lib.uvm_is_managed_pointer.restype = ctypes.c_int
        lib.uvm_is_managed_pointer.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
    return lib


def load_cudart() -> ctypes.CDLL:
    name = ctypes.util.find_library("cudart")
    candidates = [name] if name else []
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so",
        ]
    )
    last_error: OSError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            lib.cudaSetDevice.restype = ctypes.c_int
            lib.cudaSetDevice.argtypes = [ctypes.c_int]
            lib.cudaDeviceSynchronize.restype = ctypes.c_int
            lib.cudaDeviceSynchronize.argtypes = []
            return lib
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"failed to load libcudart: {last_error}")


def make_arrays(ranges: list[tuple[int, int]]) -> tuple[ctypes.Array, ctypes.Array]:
    ptrs = (ctypes.c_uint64 * len(ranges))()
    sizes = (ctypes.c_size_t * len(ranges))()
    for index, (ptr, size) in enumerate(ranges):
        ptrs[index] = int(ptr)
        sizes[index] = int(size)
    return ptrs, sizes


def sync(cudart: ctypes.CDLL) -> None:
    ret = int(cudart.cudaDeviceSynchronize())
    if ret != 0:
        raise RuntimeError(f"cudaDeviceSynchronize failed: {ret}")


def cpu_touch(ptr: int, size: int, value: int) -> None:
    buf = (ctypes.c_uint8 * size).from_address(ptr)
    step = 4096
    for offset in range(0, size, step):
        buf[offset] = value
    if size:
        buf[size - 1] = value


def query_residency(
    lib: ctypes.CDLL,
    ranges: list[tuple[int, int]],
    device: int,
) -> dict[str, int | bool]:
    if not ranges:
        return {
            "ok": True,
            "status": 0,
            "queried_bytes": 0,
            "gpu_bytes": 0,
            "cpu_bytes": 0,
            "any_bytes": 0,
            "unknown_bytes": 0,
        }
    ptrs, sizes = make_arrays(ranges)
    gpu = ctypes.c_uint64()
    cpu = ctypes.c_uint64()
    any_bytes = ctypes.c_uint64()
    unknown = ctypes.c_uint64()
    queried = ctypes.c_uint64()
    status = int(
        lib.uvm_driver_batch_query_residency_bytes(
            ptrs,
            sizes,
            len(ranges),
            device,
            ctypes.byref(gpu),
            ctypes.byref(cpu),
            ctypes.byref(any_bytes),
            ctypes.byref(unknown),
            ctypes.byref(queried),
        )
    )
    return {
        "ok": status >= 0,
        "status": status,
        "queried_bytes": int(queried.value),
        "gpu_bytes": int(gpu.value),
        "cpu_bytes": int(cpu.value),
        "any_bytes": int(any_bytes.value),
        "unknown_bytes": int(unknown.value),
    }


def time_prefetch(
    lib: ctypes.CDLL,
    cudart: ctypes.CDLL,
    ranges: list[tuple[int, int]],
    *,
    device: int,
    executor: str,
) -> tuple[float, int]:
    ptrs, sizes = make_arrays(ranges)
    start = time.perf_counter()
    if executor == "single":
        successes = 0
        for ptr, size in ranges:
            lib.uvm_prefetch(
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(size),
                ctypes.c_int(device),
                ctypes.c_void_p(),
            )
            successes += 1
    elif executor == "cpp_batch":
        successes = int(
            lib.uvm_batch_prefetch(
                ptrs,
                sizes,
                len(ranges),
                ctypes.c_int(device),
                ctypes.c_void_p(),
            )
        )
    elif executor == "driver_batch":
        if device < 0:
            successes = int(lib.uvm_driver_batch_prefetch_cpu(ptrs, sizes, len(ranges), 0))
        else:
            successes = int(
                lib.uvm_driver_batch_prefetch_gpu(
                    ptrs,
                    sizes,
                    len(ranges),
                    ctypes.c_int(device),
                    0,
                )
            )
    else:
        raise ValueError(executor)
    sync(cudart)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, successes


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9)))
    return ordered[index]


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def parse_role_bytes(raw: str, roles: list[str], default_bytes: int) -> dict[str, int]:
    values = {role: default_bytes for role in roles}
    if not raw:
        return values
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"invalid role byte mapping: {item!r}")
        role, value = item.split("=", 1)
        role = role.strip()
        if role not in values:
            raise ValueError(f"role {role!r} is not in --roles")
        values[role] = int(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expert-bytes", type=int, default=4718592)
    parser.add_argument(
        "--role-bytes",
        default="",
        help="Optional comma mapping, e.g. w13_weight=4718592,w2_weight=2359296.",
    )
    parser.add_argument("--max-experts", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,16,32,64,96,128")
    parser.add_argument("--executors", default="single,cpp_batch,driver_batch")
    parser.add_argument("--roles", default="w13_weight,w2_weight")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--touch-cpu-before-gpu", action="store_true", default=True)
    parser.add_argument("--no-touch-cpu-before-gpu", action="store_false", dest="touch_cpu_before_gpu")
    args = parser.parse_args()

    repo_root = find_repo_root()
    lib_path = (
        repo_root / "workloads" / "vllm" / "vllm" / "uvm_test" / "uvm_allocator.so"
    )
    if not args.skip_build:
        lib_path = build_uvm_allocator(repo_root)
    lib = load_lib(lib_path)
    cudart = load_cudart()
    set_device_result = int(cudart.cudaSetDevice(args.device))
    if set_device_result != 0:
        raise RuntimeError(f"cudaSetDevice({args.device}) failed: {set_device_result}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    role_bytes = parse_role_bytes(args.role_bytes, roles, args.expert_bytes)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    executors = [item.strip() for item in args.executors.split(",") if item.strip()]
    max_ranges_per_role = max(args.max_experts, max(batch_sizes))

    allocations_by_role: dict[str, list[tuple[int, int]]] = {}
    try:
        for role in roles:
            allocations_by_role[role] = []
            size_bytes = role_bytes[role]
            for index in range(max_ranges_per_role):
                ptr = lib.uvm_malloc(size_bytes, args.device, ctypes.c_void_p())
                if not ptr:
                    raise MemoryError(f"uvm_malloc failed for {role} range {index}")
                allocations_by_role[role].append((int(ptr), size_bytes))
        managed_probe = -999
        if hasattr(lib, "uvm_is_managed_pointer") and roles:
            first_role = roles[0]
            managed_probe = int(
                lib.uvm_is_managed_pointer(
                    allocations_by_role[first_role][0][0],
                    allocations_by_role[first_role][0][1],
                )
            )

        rows: list[dict[str, object]] = []
        for role in roles:
            role_allocations = allocations_by_role[role]
            expert_bytes = role_bytes[role]
            for batch_size in batch_sizes:
                ranges = role_allocations[:batch_size]
                for executor in executors:
                    latencies: list[float] = []
                    success_values: list[int] = []
                    before_last: dict[str, int | bool] = {}
                    after_last: dict[str, int | bool] = {}
                    for rep in range(args.repeat):
                        if args.touch_cpu_before_gpu:
                            for ptr, size in ranges:
                                cpu_touch(ptr, size, (rep + batch_size) & 0xFF)
                        cpu_ms, cpu_successes = time_prefetch(
                            lib,
                            cudart,
                            ranges,
                            device=-1,
                            executor="cpp_batch",
                        )
                        _ = (cpu_ms, cpu_successes)
                        before_last = query_residency(lib, ranges, args.device)
                        gpu_ms, successes = time_prefetch(
                            lib,
                            cudart,
                            ranges,
                            device=args.device,
                            executor=executor,
                        )
                        after_last = query_residency(lib, ranges, args.device)
                        latencies.append(gpu_ms)
                        success_values.append(successes)
                    total_bytes = batch_size * expert_bytes
                    med = median(latencies)
                    rows.append(
                        {
                            "role": role,
                            "executor": executor,
                            "batch_size": batch_size,
                            "expert_bytes": expert_bytes,
                            "total_bytes": total_bytes,
                            "total_mib": total_bytes / (1024 * 1024),
                            "repeat": args.repeat,
                            "managed_probe": managed_probe,
                            "success_min": min(success_values) if success_values else None,
                            "success_max": max(success_values) if success_values else None,
                            "latency_ms_mean": statistics.fmean(latencies),
                            "latency_ms_median": med,
                            "latency_ms_p90": p90(latencies),
                            "effective_gib_s_median": (
                                total_bytes / (1024**3) / (med / 1000.0)
                                if med and med > 0
                                else None
                            ),
                            "before_status": before_last.get("status"),
                            "before_gpu_bytes": before_last.get("gpu_bytes"),
                            "before_cpu_bytes": before_last.get("cpu_bytes"),
                            "before_unknown_bytes": before_last.get("unknown_bytes"),
                            "after_status": after_last.get("status"),
                            "after_gpu_bytes": after_last.get("gpu_bytes"),
                            "after_cpu_bytes": after_last.get("cpu_bytes"),
                            "after_unknown_bytes": after_last.get("unknown_bytes"),
                        }
                    )

        csv_path = args.out_dir / "uvm_expert_prefetch_microbench.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary_path = args.out_dir / "uvm_expert_prefetch_microbench.json"
        summary_path.write_text(
            json.dumps(
                {
                    "lib": str(lib_path),
                    "device": args.device,
                    "expert_bytes": args.expert_bytes,
                    "role_bytes": role_bytes,
                    "roles": roles,
                    "batch_sizes": batch_sizes,
                    "executors": executors,
                    "repeat": args.repeat,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        lines = [
            "# UVM Expert Prefetch Microbench",
            "",
            f"- lib: `{lib_path}`",
            f"- expert_bytes: `{args.expert_bytes}`",
            f"- role_bytes: `{role_bytes}`",
            f"- repeat: `{args.repeat}`",
            "",
            "| role | executor | batch | MiB | median ms | p90 ms | GiB/s | after GPU MiB | success |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            after_gpu = row.get("after_gpu_bytes")
            lines.append(
                "| {role} | {executor} | {batch} | {mib} | {median} | {p90} | {gibs} | {gpu_mib} | {success} |".format(
                    role=row["role"],
                    executor=row["executor"],
                    batch=row["batch_size"],
                    mib=fmt(row["total_mib"]),
                    median=fmt(row["latency_ms_median"]),
                    p90=fmt(row["latency_ms_p90"]),
                    gibs=fmt(row["effective_gib_s_median"]),
                    gpu_mib=fmt(
                        (float(after_gpu) / (1024 * 1024))
                        if isinstance(after_gpu, int)
                        else None
                    ),
                    success=fmt(row["success_min"]),
                )
            )
        report = "\n".join(lines) + "\n"
        report_path = args.out_dir / "uvm_expert_prefetch_microbench.md"
        report_path.write_text(report, encoding="utf-8")
        print(report)
    finally:
        for allocations in allocations_by_role.values():
            for ptr, size in allocations:
                lib.uvm_free(ctypes.c_void_p(ptr), size, args.device, ctypes.c_void_p())
        sync(cudart)


if __name__ == "__main__":
    main()
