#!/usr/bin/env python3
"""Validate Stage-M cold expert streaming sanity logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


GIB = 1024**3
DRIVER_FLAG_EVICTABLE = 1 << 0
DRIVER_FLAG_PROTECTED = 1 << 1
DRIVER_FLAG_ACTIVE = 1 << 2
DRIVER_FLAG_HOT_SUBPOOL = 1 << 11
DRIVER_FLAG_STREAMING_SUBPOOL = 1 << 12
DRIVER_FLAG_COLD_SUBPOOL = 1 << 13


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


def bench_stats(text: str) -> dict[str, Any]:
    return {
        "successful_requests": first_int(text, r"Successful requests:\s+([0-9,]+)"),
        "failed_requests": first_int(text, r"Failed requests:\s+([0-9,]+)"),
        "output_tok_s": first_float(
            text, r"Output token throughput \(tok/s\):\s+([0-9,.]+)"
        ),
        "mean_ttft_ms": first_float(text, r"Mean TTFT \(ms\):\s+([0-9,.]+)"),
        "mean_tpot_ms": first_float(text, r"Mean TPOT \(ms\):\s+([0-9,.]+)"),
        "benchmark_exit_code": first_int(
            text, r"Benchmark phase 'main' exit code:\s+([0-9]+)"
        ),
    }


def sum_int(records: list[dict[str, Any]], key: str) -> int:
    total = 0
    for record in records:
        try:
            total += int(record.get(key) or 0)
        except (TypeError, ValueError):
            pass
    return total


def max_int(records: list[dict[str, Any]], key: str) -> int:
    value = 0
    for record in records:
        try:
            value = max(value, int(record.get(key) or 0))
        except (TypeError, ValueError):
            pass
    return value


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def driver_flags(record: dict[str, Any]) -> int:
    for key in ("driver_state", "driver_tag"):
        driver = record.get(key)
        if isinstance(driver, dict):
            try:
                return int(driver.get("flags") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def driver_tag_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    tagged_records = [
        r for r in records
        if r.get("action") in ("residency_register_region", "residency_state_update")
        and driver_flags(r) != 0
    ]
    hot = 0
    cold = 0
    streaming = 0
    streaming_evictable = 0
    streaming_protected = 0
    streaming_active = 0
    for record in tagged_records:
        flags = driver_flags(record)
        metadata = record_metadata(record)
        pool_class = str(
            metadata.get("pool_class")
            or metadata.get("driver_subpool")
            or ""
        )
        if flags & DRIVER_FLAG_HOT_SUBPOOL or pool_class == "hot":
            hot += 1
        if flags & DRIVER_FLAG_COLD_SUBPOOL or pool_class == "cold":
            cold += 1
        if flags & DRIVER_FLAG_STREAMING_SUBPOOL or pool_class == "streaming":
            streaming += 1
            if flags & DRIVER_FLAG_EVICTABLE:
                streaming_evictable += 1
            if flags & DRIVER_FLAG_PROTECTED:
                streaming_protected += 1
            if flags & DRIVER_FLAG_ACTIVE:
                streaming_active += 1
    return {
        "residency_trace_records": len(records),
        "driver_flagged_records": len(tagged_records),
        "hot_driver_tag_records": hot,
        "cold_driver_tag_records": cold,
        "streaming_driver_tag_records": streaming,
        "streaming_driver_evictable_records": streaming_evictable,
        "streaming_driver_protected_records": streaming_protected,
        "streaming_driver_active_records": streaming_active,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--streaming-budget-bytes", type=int, required=True)
    parser.add_argument("--expect-evictions", action="store_true")
    parser.add_argument("--expect-driver-tags", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir
    trace_file = args.trace_file or case_dir / "vllm_uvm_cold_streaming_stage_m.jsonl"
    out_json = args.out_json or case_dir / "cold_streaming_summary.json"
    out_md = args.out_md or case_dir / "cold_streaming_report.md"

    records = jsonl(trace_file)
    residency_trace = jsonl(case_dir / "vllm_uvm_pool_residency_stage_m.jsonl")
    summaries = [
        record
        for record in records
        if record.get("action") == "cold_streaming_step_summary"
    ]
    bench = bench_stats(read_text(case_dir / "vllm_bench.log"))
    tags = driver_tag_stats(residency_trace)

    total_cold_miss_experts = sum_int(summaries, "cold_miss_experts")
    total_cold_miss_regions = sum_int(summaries, "cold_miss_region_records")
    total_cold_miss_bytes = sum_int(summaries, "cold_miss_bytes")
    total_prefetch_records = sum_int(summaries, "streaming_prefetch_issued_records")
    total_prefetch_bytes = sum_int(summaries, "streaming_prefetch_issued_bytes")
    total_evict_records = sum_int(summaries, "streaming_eviction_count")
    total_evict_bytes = sum_int(summaries, "streaming_eviction_bytes")
    total_hot_evicted = sum_int(summaries, "hot_evicted_by_cold_count")
    total_budget_reject = sum_int(summaries, "streaming_budget_reject_records")
    total_protected_collision = sum_int(summaries, "protected_collision_records")
    max_live_bytes = max_int(summaries, "streaming_live_bytes_after")
    max_live_records = max_int(summaries, "streaming_live_records_after")
    max_live_within_budget = (
        args.streaming_budget_bytes <= 0 or max_live_bytes <= args.streaming_budget_bytes
    )

    checks = {
        "benchmark_completed": (
            (bench.get("successful_requests") or 0) > 0
            and (bench.get("failed_requests") or 0) == 0
            and bench.get("benchmark_exit_code") in (0, None)
        ),
        "streaming_trace_present": len(summaries) > 0,
        "cold_miss_observed": total_cold_miss_regions > 0,
        "cold_prefetch_issued": total_prefetch_records > 0,
        "hot_not_evicted_by_cold": total_hot_evicted == 0,
        "protected_collision_zero": total_protected_collision == 0,
        "streaming_live_within_budget": max_live_within_budget,
        "eviction_observed_when_expected": (
            total_evict_records > 0 if args.expect_evictions else True
        ),
    }
    if args.expect_driver_tags:
        checks["hot_driver_tag_observed"] = tags["hot_driver_tag_records"] > 0
        checks["cold_driver_tag_observed"] = tags["cold_driver_tag_records"] > 0
        checks["streaming_driver_tag_observed"] = (
            tags["streaming_driver_tag_records"] > 0
        )
        checks["streaming_driver_evictable"] = (
            tags["streaming_driver_evictable_records"] > 0
        )
        checks["streaming_not_driver_protected"] = (
            tags["streaming_driver_protected_records"] == 0
        )
    passed = all(checks.values())

    summary = {
        "case_dir": str(case_dir),
        "trace_file": str(trace_file),
        "passed": passed,
        "checks": checks,
        "inputs": {
            "streaming_budget_bytes": args.streaming_budget_bytes,
            "streaming_budget_gib": bytes_gib(args.streaming_budget_bytes),
            "expect_evictions": args.expect_evictions,
        },
        "cold_streaming": {
            "trace_records": len(records),
            "summary_records": len(summaries),
            "cold_miss_experts": total_cold_miss_experts,
            "cold_miss_region_records": total_cold_miss_regions,
            "cold_miss_bytes": total_cold_miss_bytes,
            "cold_miss_gib": bytes_gib(total_cold_miss_bytes),
            "streaming_prefetch_issued_records": total_prefetch_records,
            "streaming_prefetch_issued_bytes": total_prefetch_bytes,
            "streaming_prefetch_issued_gib": bytes_gib(total_prefetch_bytes),
            "streaming_eviction_count": total_evict_records,
            "streaming_eviction_bytes": total_evict_bytes,
            "streaming_eviction_gib": bytes_gib(total_evict_bytes),
            "hot_evicted_by_cold_count": total_hot_evicted,
            "protected_collision_records": total_protected_collision,
            "streaming_budget_reject_records": total_budget_reject,
            "max_streaming_live_bytes": max_live_bytes,
            "max_streaming_live_gib": bytes_gib(max_live_bytes),
            "max_streaming_live_records": max_live_records,
        },
        "driver_tags": tags,
        "benchmark": bench,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Cold Expert Streaming Sanity",
        "",
        f"- case_dir: `{case_dir}`",
        f"- trace_file: `{trace_file}`",
        f"- overall: **{status(passed)}**",
        f"- streaming budget: `{args.streaming_budget_bytes}` bytes "
        f"({bytes_gib(args.streaming_budget_bytes):.3f} GiB)",
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
        "| metric | value |",
        "|---|---:|",
        f"| summary records | {len(summaries)} |",
        f"| cold miss experts | {total_cold_miss_experts} |",
        f"| cold miss regions | {total_cold_miss_regions} |",
        f"| cold miss GiB | {fmt(bytes_gib(total_cold_miss_bytes))} |",
        f"| prefetch issued records | {total_prefetch_records} |",
        f"| prefetch issued GiB | {fmt(bytes_gib(total_prefetch_bytes))} |",
        f"| eviction count | {total_evict_records} |",
        f"| eviction GiB | {fmt(bytes_gib(total_evict_bytes))} |",
        f"| hot evicted by cold | {total_hot_evicted} |",
        f"| protected collisions | {total_protected_collision} |",
        f"| budget reject records | {total_budget_reject} |",
        f"| max live GiB | {fmt(bytes_gib(max_live_bytes))} |",
        f"| max live records | {max_live_records} |",
        f"| driver flagged records | {tags['driver_flagged_records']} |",
        f"| hot driver tag records | {tags['hot_driver_tag_records']} |",
        f"| cold driver tag records | {tags['cold_driver_tag_records']} |",
        f"| streaming driver tag records | {tags['streaming_driver_tag_records']} |",
        f"| streaming driver evictable records | {tags['streaming_driver_evictable_records']} |",
        f"| streaming driver protected records | {tags['streaming_driver_protected_records']} |",
        f"| successful requests | {fmt(bench.get('successful_requests'))} |",
        f"| failed requests | {fmt(bench.get('failed_requests'))} |",
        f"| output tok/s | {fmt(bench.get('output_tok_s'))} |",
        f"| mean TPOT ms | {fmt(bench.get('mean_tpot_ms'))} |",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"overall={status(passed)}")
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
