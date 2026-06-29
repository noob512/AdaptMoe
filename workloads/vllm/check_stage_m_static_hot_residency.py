#!/usr/bin/env python3
"""Validate Stage-M static hot expert pre-residency sanity logs.

The checker is intentionally evidence-oriented: it emits a JSON and Markdown
report even when one acceptance criterion fails.  Use --strict when a non-zero
exit code is desired.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


GIB = 1024**3


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except FileNotFoundError:
        pass
    return records


def first_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def first_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def bytes_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / GIB


def close_enough(actual: int | None, expected: int, tolerance_ratio: float) -> bool:
    if actual is None:
        return False
    return abs(actual - expected) <= max(1, int(expected * tolerance_ratio))


def kv_covered_bytes(address_log: Path) -> tuple[int, int]:
    bytes_total = 0
    records = 0
    for line in read_text(address_log).splitlines():
        if not line.startswith("kv_cache,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            bytes_total += int(parts[4])
        except ValueError:
            continue
        records += 1
    return records, bytes_total


def allocator_kv_device_direct_stats(allocator_log: Path) -> dict[str, int]:
    text = read_text(allocator_log)
    records = 0
    bytes_total = 0
    managed_records = 0
    for line in text.splitlines():
        if "phase=initialize_kv_cache" not in line or "TRACE_POLICY" not in line:
            continue
        size = first_int(line, r"size_bytes=([0-9]+)") or 0
        if (
            "policy_source=kv_device_direct_policy" in line
            and "placement_backend=device_direct" in line
            and "action_success=1" in line
        ):
            records += 1
            bytes_total += size
        elif "placement_backend=managed" in line:
            managed_records += 1
    return {
        "kv_device_direct_records": records,
        "kv_device_direct_bytes": bytes_total,
        "kv_managed_policy_records": managed_records,
    }


def fixed_weight_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    materialized = [
        r for r in records
        if r.get("event") == "fixed_weight_device_direct_materialized"
    ]
    summaries = [
        r for r in records
        if r.get("event") == "fixed_weight_device_direct_summary"
    ]
    summary = summaries[-1] if summaries else {}
    expert_materialized = [r for r in materialized if r.get("is_moe_expert")]
    new_not_managed = [
        r for r in materialized
        if r.get("new_driver_capability_reason") == "not_cuda_managed"
    ]
    old_managed = [
        r for r in materialized
        if r.get("old_driver_capability_reason") == "cuda_managed"
    ]
    return {
        "trace_records": len(records),
        "materialized_records": len(materialized),
        "materialized_bytes": sum(int(r.get("bytes") or 0) for r in materialized),
        "summary_converted_count": int(summary.get("converted_count") or 0),
        "summary_converted_bytes": int(summary.get("converted_bytes") or 0),
        "summary_failed_count": int(summary.get("failed_count") or 0),
        "expert_materialized_records": len(expert_materialized),
        "new_not_managed_records": len(new_not_managed),
        "old_managed_records": len(old_managed),
        "all_materialized_new_not_managed": (
            bool(materialized) and len(new_not_managed) == len(materialized)
        ),
    }


def weight_map_stats(records: list[dict[str, Any]], target_roles: set[str]) -> dict[str, Any]:
    expert_slices = [
        r for r in records
        if r.get("is_moe_expert")
        and r.get("is_logical_slice")
        and str(r.get("role")) in target_roles
    ]
    expert_managed = [
        r for r in expert_slices
        if r.get("driver_capability_reason") == "cuda_managed"
    ]
    expert_not_managed = [
        r for r in expert_slices
        if r.get("driver_capability_reason") == "not_cuda_managed"
    ]
    fixed_weights = [
        r for r in records
        if r.get("fixed_weight")
        and not r.get("is_moe_expert")
        and r.get("kind") == "weight:param"
    ]
    fixed_not_managed = [
        r for r in fixed_weights
        if r.get("driver_capability_reason") == "not_cuda_managed"
    ]
    expert_fixed_flag = [r for r in expert_slices if r.get("fixed_weight")]
    return {
        "records": len(records),
        "expert_slice_records": len(expert_slices),
        "expert_slice_bytes": sum(
            int(r.get("size_bytes") or r.get("tensor_bytes") or 0)
            for r in expert_slices
        ),
        "expert_slice_managed_records": len(expert_managed),
        "expert_slice_not_managed_records": len(expert_not_managed),
        "expert_slice_managed_ratio": (
            len(expert_managed) / len(expert_slices) if expert_slices else None
        ),
        "expert_slice_fixed_flag_records": len(expert_fixed_flag),
        "fixed_weight_records": len(fixed_weights),
        "fixed_weight_bytes": sum(
            int(r.get("size_bytes") or r.get("tensor_bytes") or 0)
            for r in fixed_weights
        ),
        "fixed_weight_not_managed_records": len(fixed_not_managed),
        "fixed_weight_not_managed_ratio": (
            len(fixed_not_managed) / len(fixed_weights) if fixed_weights else None
        ),
    }


def init_hot_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    issued = [r for r in records if r.get("action") == "init_weight_hot_prefetch_issued"]
    skipped = [r for r in records if r.get("action") == "init_weight_hot_prefetch_skipped"]
    real_queries = [
        r for r in records
        if r.get("action") == "init_weight_hot_prefetch_real_query"
    ]
    stage_queries = [
        r for r in records
        if r.get("action") == "init_hot_residency_stage_query"
    ]
    summaries = [
        r for r in records
        if r.get("action") == "init_weight_offload_summary"
    ]
    offload_issued_actions = {
        "init_weight_offload_issued",
        "init_weight_offload_advise_cpu_issued",
        "init_weight_offload_prefetch_cpu_issued",
        "init_weight_offload_offload_cpu_issued",
    }
    offload_issued = [
        r for r in records
        if r.get("action") in offload_issued_actions
    ]
    queried = sum(int(r.get("queried_bytes") or 0) for r in real_queries)
    gpu = sum(int(r.get("gpu_bytes") or 0) for r in real_queries)
    stage_queried = sum(
        int(r.get("hot_residency_queried_bytes") or 0) for r in stage_queries
    )
    stage_gpu = sum(
        int(r.get("hot_residency_gpu_bytes") or 0) for r in stage_queries
    )
    summary_hot_issued = sum(
        int(r.get("hot_prefetch_issued_records") or 0) for r in summaries
    )
    summary_hot_bytes = sum(
        int(r.get("hot_prefetch_issued_bytes") or 0) for r in summaries
    )
    return {
        "trace_records": len(records),
        "hot_prefetch_issued_records": len(issued),
        "hot_prefetch_issued_bytes": sum(int(r.get("bytes") or 0) for r in issued),
        "hot_prefetch_skipped_records": len(skipped),
        "hot_real_query_records": len(real_queries),
        "hot_real_queried_bytes": queried,
        "hot_real_gpu_bytes": gpu,
        "hot_real_gpu_ratio": gpu / queried if queried else None,
        "stage_query_records": len(stage_queries),
        "stage_query_queried_bytes": stage_queried,
        "stage_query_gpu_bytes": stage_gpu,
        "stage_query_gpu_ratio": stage_gpu / stage_queried if stage_queried else None,
        "summary_records": len(summaries),
        "summary_hot_prefetch_issued_records": summary_hot_issued,
        "summary_hot_prefetch_issued_bytes": summary_hot_bytes,
        "init_weight_offload_issued_records": len(offload_issued),
        "init_weight_offload_issued_bytes": sum(
            int(r.get("bytes") or 0) for r in offload_issued
        ),
        "summary_init_weight_offload_issued_records": sum(
            int(r.get("issued_records") or 0) for r in summaries
        ),
        "summary_init_weight_offload_issued_bytes": sum(
            int(r.get("issued_bytes") or 0) for r in summaries
        ),
    }


def bench_stats(text: str) -> dict[str, Any]:
    return {
        "successful_requests": first_int(text, r"Successful requests:\s+([0-9,]+)"),
        "failed_requests": first_int(text, r"Failed requests:\s+([0-9,]+)"),
        "output_tok_s": first_float(text, r"Output token throughput \(tok/s\):\s+([0-9,.]+)"),
        "mean_ttft_ms": first_float(text, r"Mean TTFT \(ms\):\s+([0-9,.]+)"),
        "mean_tpot_ms": first_float(text, r"Mean TPOT \(ms\):\s+([0-9,.]+)"),
        "benchmark_exit_code": first_int(text, r"Benchmark phase 'main' exit code:\s+([0-9]+)"),
    }


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--hot-k", type=int, default=64)
    parser.add_argument("--kv-bytes", type=int, default=7516192768)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--roles", default="moe_gate_up,moe_down")
    parser.add_argument("--kv-tolerance-ratio", type=float, default=0.02)
    parser.add_argument("--min-hot-gpu-ratio", type=float, default=0.90)
    parser.add_argument("--expect-cold-offload", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir
    out_json = args.out_json or case_dir / "static_hot_residency_summary.json"
    out_md = args.out_md or case_dir / "static_hot_residency_report.md"
    roles = {role.strip() for role in args.roles.split(",") if role.strip()}

    server_log = case_dir / "vllm_server.log"
    bench_log = case_dir / "vllm_bench.log"
    address_log = case_dir / "vllm_uvm_address_regions.log"
    allocator_log = case_dir / "vllm_uvm_allocator.log"
    fixed_trace = case_dir / "vllm_uvm_fixed_weight_device_direct.jsonl"
    weight_map = case_dir / "vllm_uvm_weight_regions.jsonl"
    init_trace = case_dir / "vllm_uvm_init_weight_offload_stage_m.jsonl"

    server_text = read_text(server_log)
    bench_text = read_text(bench_log)
    kv_records, kv_bytes = kv_covered_bytes(address_log)
    kv_alloc = allocator_kv_device_direct_stats(allocator_log)
    fixed = fixed_weight_stats(jsonl(fixed_trace))
    weights = weight_map_stats(jsonl(weight_map), roles)
    hot = init_hot_stats(jsonl(init_trace))
    bench = bench_stats(bench_text)

    expected_hot_records = args.num_layers * args.hot_k * len(roles)
    expected_hot_bytes = None
    if weights["expert_slice_records"]:
        # Average over all expert slices is robust to mixed w13/w2 sizes.
        avg_role_slice_bytes = (
            int(weights["expert_slice_bytes"]) / int(weights["expert_slice_records"])
        )
        expected_hot_bytes = int(expected_hot_records * avg_role_slice_bytes)

    gpu_kv_tokens = first_int(server_text, r"GPU KV cache size:\s+([0-9,]+)\s+tokens")
    startup_ok = (
        "Application startup complete" in server_text
        or "Uvicorn running on" in server_text
        or (bench.get("successful_requests") or 0) > 0
    )

    checks: dict[str, bool] = {
        "server_started": startup_ok,
        "benchmark_completed": (
            (bench.get("successful_requests") or 0) > 0
            and (bench.get("failed_requests") or 0) == 0
            and bench.get("benchmark_exit_code") in (0, None)
        ),
        "kv_capacity_close_to_target": close_enough(
            kv_bytes, args.kv_bytes, args.kv_tolerance_ratio
        ),
        "kv_device_direct": (
            kv_alloc["kv_device_direct_records"] > 0
            and kv_alloc["kv_device_direct_bytes"] >= int(args.kv_bytes * 0.95)
        ),
        "fixed_weight_native": (
            fixed["summary_converted_count"] > 0
            and fixed["summary_failed_count"] == 0
            and fixed["all_materialized_new_not_managed"]
        ),
        "fixed_weight_no_expert_materialized": (
            fixed["expert_materialized_records"] == 0
            and weights["expert_slice_fixed_flag_records"] == 0
        ),
        "expert_weight_still_managed": (
            (weights["expert_slice_managed_ratio"] or 0.0) >= 0.99
            and weights["expert_slice_not_managed_records"] == 0
        ),
        "hot_prefetch_issued": (
            hot["hot_prefetch_issued_records"] >= expected_hot_records
            or hot["summary_hot_prefetch_issued_records"] >= expected_hot_records
        ),
        "hot_real_resident_ratio": (
            (hot["hot_real_gpu_ratio"] or 0.0) >= args.min_hot_gpu_ratio
        ),
    }
    if args.expect_cold_offload:
        checks["cold_offload_issued"] = (
            hot["init_weight_offload_issued_records"] > 0
            or hot["summary_init_weight_offload_issued_records"] > 0
        )
    else:
        checks["cold_offload_disabled"] = (
            hot["init_weight_offload_issued_records"] == 0
            and hot["summary_init_weight_offload_issued_records"] == 0
        )

    passed = all(checks.values())
    summary = {
        "case_dir": str(case_dir),
        "passed": passed,
        "checks": checks,
        "inputs": {
            "hot_k": args.hot_k,
            "kv_bytes": args.kv_bytes,
            "kv_gib": bytes_gib(args.kv_bytes),
            "num_layers": args.num_layers,
            "roles": sorted(roles),
            "expected_hot_records": expected_hot_records,
            "expected_hot_bytes": expected_hot_bytes,
            "expected_hot_gib": bytes_gib(expected_hot_bytes),
            "min_hot_gpu_ratio": args.min_hot_gpu_ratio,
        },
        "server": {
            "gpu_kv_cache_tokens": gpu_kv_tokens,
            "startup_ok": startup_ok,
        },
        "kv": {
            "address_records": kv_records,
            "covered_bytes": kv_bytes,
            "covered_gib": bytes_gib(kv_bytes),
            **kv_alloc,
            "kv_device_direct_gib": bytes_gib(kv_alloc["kv_device_direct_bytes"]),
        },
        "fixed_weight": {
            **fixed,
            "materialized_gib": bytes_gib(fixed["materialized_bytes"]),
            "summary_converted_gib": bytes_gib(fixed["summary_converted_bytes"]),
        },
        "weight_map": {
            **weights,
            "expert_slice_gib": bytes_gib(weights["expert_slice_bytes"]),
            "fixed_weight_gib": bytes_gib(weights["fixed_weight_bytes"]),
        },
        "hot_prefetch": {
            **hot,
            "hot_prefetch_issued_gib": bytes_gib(hot["hot_prefetch_issued_bytes"]),
            "summary_hot_prefetch_issued_gib": bytes_gib(
                hot["summary_hot_prefetch_issued_bytes"]
            ),
            "hot_real_queried_gib": bytes_gib(hot["hot_real_queried_bytes"]),
            "hot_real_gpu_gib": bytes_gib(hot["hot_real_gpu_bytes"]),
            "stage_query_queried_gib": bytes_gib(hot["stage_query_queried_bytes"]),
            "stage_query_gpu_gib": bytes_gib(hot["stage_query_gpu_bytes"]),
        },
        "benchmark": bench,
        "files": {
            "server_log": str(server_log),
            "bench_log": str(bench_log),
            "address_log": str(address_log),
            "allocator_log": str(allocator_log),
            "fixed_trace": str(fixed_trace),
            "weight_map": str(weight_map),
            "init_trace": str(init_trace),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Static Hot Expert Residency Sanity",
        "",
        f"- case_dir: `{case_dir}`",
        f"- overall: **{status(passed)}**",
        f"- target KV: `{args.kv_bytes}` bytes ({bytes_gib(args.kv_bytes):.3f} GiB)",
        f"- hotK: `{args.hot_k}`",
        f"- expected hot role-ranges: `{expected_hot_records}`",
        "",
        "## Checks",
        "",
        "| check | status |",
        "|---|---|",
    ]
    for name, ok in checks.items():
        lines.append(f"| `{name}` | {status(ok)} |")

    lines += [
        "",
        "## Evidence",
        "",
        "| item | value |",
        "|---|---:|",
        f"| GPU KV cache tokens | {fmt(gpu_kv_tokens)} |",
        f"| KV covered GiB | {fmt(bytes_gib(kv_bytes))} |",
        f"| KV device-direct records | {kv_alloc['kv_device_direct_records']} |",
        f"| KV device-direct GiB | {fmt(bytes_gib(kv_alloc['kv_device_direct_bytes']))} |",
        f"| fixed converted records | {fixed['summary_converted_count']} |",
        f"| fixed converted GiB | {fmt(bytes_gib(fixed['summary_converted_bytes']))} |",
        f"| fixed failed records | {fixed['summary_failed_count']} |",
        f"| expert slices managed ratio | {fmt(weights['expert_slice_managed_ratio'])} |",
        f"| expert slices not managed records | {weights['expert_slice_not_managed_records']} |",
        f"| hot prefetch issued records | {hot['hot_prefetch_issued_records']} |",
        f"| hot prefetch issued GiB | {fmt(bytes_gib(hot['hot_prefetch_issued_bytes']))} |",
        f"| hot real queried GiB | {fmt(bytes_gib(hot['hot_real_queried_bytes']))} |",
        f"| hot real GPU GiB | {fmt(bytes_gib(hot['hot_real_gpu_bytes']))} |",
        f"| hot real GPU ratio | {fmt(hot['hot_real_gpu_ratio'])} |",
        f"| cold offload issued records | {hot['init_weight_offload_issued_records']} |",
        f"| cold offload issued GiB | {fmt(bytes_gib(hot['init_weight_offload_issued_bytes']))} |",
        f"| summary cold offload issued records | {hot['summary_init_weight_offload_issued_records']} |",
        f"| summary cold offload issued GiB | {fmt(bytes_gib(hot['summary_init_weight_offload_issued_bytes']))} |",
        f"| successful requests | {fmt(bench.get('successful_requests'))} |",
        f"| failed requests | {fmt(bench.get('failed_requests'))} |",
        "",
        "## Files",
        "",
    ]
    for key, path in summary["files"].items():
        lines.append(f"- {key}: `{path}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"overall={status(passed)}")
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
