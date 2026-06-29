#!/usr/bin/env python3
"""Validate Stage-M cold expert batch slot allocation traces."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


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


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


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


def sum_key(records: list[dict[str, Any]], key: str) -> int:
    return sum(as_int(record.get(key)) for record in records)


def max_key(records: list[dict[str, Any]], key: str) -> int:
    value = 0
    for record in records:
        value = max(value, as_int(record.get(key)))
    return value


def count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--slots-per-layer", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir
    trace_file = args.trace_file
    out_json = args.out_json
    out_md = args.out_md
    if case_dir is not None:
        out_json = out_json or case_dir / "cold_slot_pool_summary.json"
        out_md = out_md or case_dir / "cold_slot_pool_report.md"
    else:
        out_json = out_json or trace_file.with_suffix(".summary.json")
        out_md = out_md or trace_file.with_suffix(".report.md")

    records = [
        record
        for record in jsonl(trace_file)
        if record.get("action") == "cold_slot_pool_step_summary"
    ]
    bench = bench_stats(read_text(case_dir / "vllm_bench.log")) if case_dir else {}

    duplicate_slot_records: list[dict[str, Any]] = []
    explicit_conflict_records: list[dict[str, Any]] = []
    protected_conflict_records: list[dict[str, Any]] = []
    capacity_violation_records: list[dict[str, Any]] = []
    overflow_mismatch_records: list[dict[str, Any]] = []
    parallel_mismatch_records: list[dict[str, Any]] = []
    miss_accounting_records: list[dict[str, Any]] = []
    live_capacity_records: list[dict[str, Any]] = []
    stale_read_records: list[dict[str, Any]] = []
    unexpected_capacity_records: list[dict[str, Any]] = []
    backend_failure_records: list[dict[str, Any]] = []
    trace_only_migration_records: list[dict[str, Any]] = []
    legacy_backend_assignment_records: list[dict[str, Any]] = []
    driver_backend_records: list[dict[str, Any]] = []
    driver_dry_run_failure_records: list[dict[str, Any]] = []
    driver_entry_mismatch_records: list[dict[str, Any]] = []
    driver_conflict_records: list[dict[str, Any]] = []
    driver_copy_only_mismatch_records: list[dict[str, Any]] = []
    driver_slot_pte_mismatch_records: list[dict[str, Any]] = []
    driver_metadata_copy_counter_records: list[dict[str, Any]] = []
    driver_copy_failure_records: list[dict[str, Any]] = []
    driver_failure_count_records: list[dict[str, Any]] = []

    for record in records:
        slot_ids = as_int_list(record.get("assignment_slot_ids"))
        assignment_count = as_int(record.get("assignment_count"))
        slot_capacity = as_int(record.get("slot_capacity"))
        slot_hits = as_int(record.get("slot_hits"))
        slot_misses = as_int(record.get("slot_misses"))
        overflow_experts = as_int_list(record.get("overflow_experts"))
        overflow_count = as_int(record.get("overflow_count"))
        parallel_copy_count = as_int(record.get("parallel_copy_count"))

        if len(slot_ids) != len(set(slot_ids)):
            duplicate_slot_records.append(record)
        if as_int(record.get("slot_id_conflicts")) != 0:
            explicit_conflict_records.append(record)
        if as_int(record.get("protected_slot_conflicts")) != 0:
            protected_conflict_records.append(record)
        if assignment_count > slot_capacity:
            capacity_violation_records.append(record)
        if overflow_count != len(overflow_experts):
            overflow_mismatch_records.append(record)
        if parallel_copy_count != assignment_count:
            parallel_mismatch_records.append(record)
        if slot_misses != assignment_count + overflow_count:
            miss_accounting_records.append(record)
        if slot_hits + assignment_count > slot_capacity:
            live_capacity_records.append(record)
        if bool(record.get("stale_read_detected")):
            stale_read_records.append(record)
        if args.slots_per_layer is not None and slot_capacity != args.slots_per_layer:
            unexpected_capacity_records.append(record)
        backend = str(record.get("backend") or "trace_only")
        if record.get("backend_success") is False:
            backend_failure_records.append(record)
        if backend == "trace_only" and (
            as_int(record.get("migration_issued_records"))
            or as_int(record.get("remap_count"))
            or as_int(record.get("tlb_invalidate_count"))
        ):
            trace_only_migration_records.append(record)
        if (
            backend == "legacy_prefetch"
            and as_int(record.get("migration_expert_count")) != assignment_count
        ):
            legacy_backend_assignment_records.append(record)
        if backend == "driver_slot_remap":
            driver_backend_records.append(record)
            mode = str(record.get("driver_slot_remap_mode") or "")
            status = str(record.get("backend_status") or "")
            if (
                mode in ("dry_run", "validate", "copy_only", "gpu_copy_only")
                and status
                not in (
                    "dry_run_ok",
                    "validate_ok",
                    "copy_only_ok",
                    "gpu_copy_only_ok",
                    "no_ranges",
                )
            ):
                driver_dry_run_failure_records.append(record)
            if as_int(record.get("driver_slot_remap_conflict_count")) != 0:
                driver_conflict_records.append(record)
            entry_count = as_int(record.get("driver_slot_remap_entry_count"))
            migration_range_count = as_int(record.get("migration_range_count"))
            if entry_count != migration_range_count:
                driver_entry_mismatch_records.append(record)
            copy_count = as_int(record.get("driver_slot_remap_copy_count"))
            copy_bytes = as_int(record.get("driver_slot_remap_copy_bytes"))
            copy_failed_count = as_int(
                record.get("driver_slot_remap_copy_failed_count")
            )
            failure_count = as_int(record.get("driver_slot_remap_failure_count"))
            migration_issued_bytes = as_int(record.get("migration_issued_bytes"))
            if copy_failed_count != 0:
                driver_copy_failure_records.append(record)
            if status not in ("no_ranges", "") and failure_count != 0:
                driver_failure_count_records.append(record)
            if mode in ("copy_only", "gpu_copy_only") and status == f"{mode}_ok":
                if (
                    copy_count != migration_range_count
                    or copy_bytes != migration_issued_bytes
                    or copy_failed_count != 0
                ):
                    driver_copy_only_mismatch_records.append(record)
            elif mode in ("slot_pte_remap", "slot_va_block_remap") and status == "ok":
                remap_count = as_int(record.get("remap_count"))
                tlb_invalidate_count = as_int(record.get("tlb_invalidate_count"))
                if (
                    remap_count != migration_range_count
                    or tlb_invalidate_count != migration_range_count
                    or copy_failed_count != 0
                ):
                    driver_slot_pte_mismatch_records.append(record)
            elif mode in ("dry_run", "validate") and (
                copy_count or copy_bytes or copy_failed_count
            ):
                driver_metadata_copy_counter_records.append(record)

    checks = {
        "trace_has_records": bool(records),
        "assignment_slot_ids_unique_per_record": not duplicate_slot_records,
        "slot_id_conflicts_zero": not explicit_conflict_records,
        "protected_slot_conflicts_zero": not protected_conflict_records,
        "assignment_count_within_capacity": not capacity_violation_records,
        "overflow_count_matches_list": not overflow_mismatch_records,
        "parallel_copy_count_matches_assignments": not parallel_mismatch_records,
        "misses_equal_assignments_plus_overflow": not miss_accounting_records,
        "hits_plus_assignments_within_capacity": not live_capacity_records,
        "stale_read_not_detected": not stale_read_records,
        "slot_capacity_matches_expected": not unexpected_capacity_records,
        "backend_success": not backend_failure_records,
        "trace_only_has_no_real_migration": not trace_only_migration_records,
        "legacy_prefetch_migrates_assigned_experts": (
            not legacy_backend_assignment_records
        ),
        "driver_slot_remap_dry_run_or_validate_ok": (
            not driver_dry_run_failure_records
        ),
        "driver_slot_remap_entry_count_matches_ranges": (
            not driver_entry_mismatch_records
        ),
        "driver_slot_remap_conflicts_zero": not driver_conflict_records,
        "driver_slot_remap_copy_only_counts_match_ranges": (
            not driver_copy_only_mismatch_records
        ),
        "driver_slot_remap_pte_counts_match_ranges": (
            not driver_slot_pte_mismatch_records
        ),
        "driver_slot_remap_metadata_modes_do_not_copy": (
            not driver_metadata_copy_counter_records
        ),
        "driver_slot_remap_copy_failures_zero": not driver_copy_failure_records,
        "driver_slot_remap_failure_count_zero": not driver_failure_count_records,
    }
    passed = all(checks.values())

    summary = {
        "passed": passed,
        "trace_file": str(trace_file),
        "records": len(records),
        "checks": checks,
        "bench": bench,
        "slot_pool": {
            "slot_hits": sum_key(records, "slot_hits"),
            "slot_misses": sum_key(records, "slot_misses"),
            "slot_assignments": sum_key(records, "assignment_count"),
            "overflow_count": sum_key(records, "overflow_count"),
            "parallel_copy_count": sum_key(records, "parallel_copy_count"),
            "remap_count": sum_key(records, "remap_count"),
            "planned_remap_count": sum_key(records, "planned_remap_count"),
            "tlb_invalidate_count": sum_key(records, "tlb_invalidate_count"),
            "planned_tlb_invalidate_count": sum_key(
                records,
                "planned_tlb_invalidate_count",
            ),
            "migration_range_count": sum_key(records, "migration_range_count"),
            "migration_expert_count": sum_key(records, "migration_expert_count"),
            "assigned_slot_count": sum_key(records, "assigned_slot_count"),
            "migration_bytes": sum_key(records, "migration_bytes"),
            "migration_issued_records": sum_key(records, "migration_issued_records"),
            "migration_failed_records": sum_key(records, "migration_failed_records"),
            "migration_issued_bytes": sum_key(records, "migration_issued_bytes"),
            "slot_id_conflicts": sum_key(records, "slot_id_conflicts"),
            "protected_slot_conflicts": sum_key(
                records,
                "protected_slot_conflicts",
            ),
            "stale_read_detected": len(stale_read_records),
            "max_assignments_per_record": max_key(records, "assignment_count"),
            "max_overflow_per_record": max_key(records, "overflow_count"),
            "max_slot_capacity": max_key(records, "slot_capacity"),
            "allocation_us_total": sum(
                float(record.get("allocation_us") or 0.0) for record in records
            ),
            "backend_migration_us_total": sum(
                float(record.get("backend_migration_us") or 0.0)
                for record in records
            ),
            "backend_counts": count_by_key(records, "backend"),
            "backend_status_counts": count_by_key(records, "backend_status"),
            "driver_slot_remap_records": len(driver_backend_records),
            "driver_slot_remap_entry_count": sum_key(
                records,
                "driver_slot_remap_entry_count",
            ),
            "driver_slot_remap_success_count": sum_key(
                records,
                "driver_slot_remap_success_count",
            ),
            "driver_slot_remap_copy_count": sum_key(
                records,
                "driver_slot_remap_copy_count",
            ),
            "driver_slot_remap_copy_bytes": sum_key(
                records,
                "driver_slot_remap_copy_bytes",
            ),
            "driver_slot_remap_copy_failed_count": sum_key(
                records,
                "driver_slot_remap_copy_failed_count",
            ),
            "driver_slot_remap_failure_count": sum_key(
                records,
                "driver_slot_remap_failure_count",
            ),
            "driver_slot_remap_conflict_count": sum_key(
                records,
                "driver_slot_remap_conflict_count",
            ),
            "driver_slot_remap_copy_failed_count": sum_key(
                records,
                "driver_slot_remap_copy_failed_count",
            ),
            "driver_slot_remap_requested_bytes": sum_key(
                records,
                "driver_slot_remap_requested_bytes",
            ),
            "driver_slot_remap_accepted_bytes": sum_key(
                records,
                "driver_slot_remap_accepted_bytes",
            ),
        },
        "failure_samples": {
            "duplicate_slot_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "assignment_slot_ids": r.get("assignment_slot_ids"),
                }
                for r in duplicate_slot_records[:8]
            ],
            "capacity_violation_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "assignment_count": r.get("assignment_count"),
                    "slot_capacity": r.get("slot_capacity"),
                }
                for r in capacity_violation_records[:8]
            ],
            "overflow_mismatch_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "overflow_count": r.get("overflow_count"),
                    "overflow_experts": r.get("overflow_experts"),
                }
                for r in overflow_mismatch_records[:8]
            ],
            "backend_failure_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "backend": r.get("backend"),
                    "backend_status": r.get("backend_status"),
                    "backend_error": r.get("backend_error"),
                }
                for r in backend_failure_records[:8]
            ],
            "legacy_backend_assignment_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "assignment_count": r.get("assignment_count"),
                    "migration_expert_count": r.get("migration_expert_count"),
                    "overflow_count": r.get("overflow_count"),
                }
                for r in legacy_backend_assignment_records[:8]
            ],
            "driver_dry_run_failure_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "backend_status": r.get("backend_status"),
                    "backend_error": r.get("backend_error"),
                    "driver_result": r.get("driver_result"),
                }
                for r in driver_dry_run_failure_records[:8]
            ],
            "driver_entry_mismatch_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "driver_slot_remap_entry_count": r.get(
                        "driver_slot_remap_entry_count"
                    ),
                    "migration_range_count": r.get("migration_range_count"),
                }
                for r in driver_entry_mismatch_records[:8]
            ],
            "driver_copy_only_mismatch_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "driver_slot_remap_copy_count": r.get(
                        "driver_slot_remap_copy_count"
                    ),
                    "migration_range_count": r.get("migration_range_count"),
                    "driver_slot_remap_copy_bytes": r.get(
                        "driver_slot_remap_copy_bytes"
                    ),
                    "migration_issued_bytes": r.get("migration_issued_bytes"),
                    "driver_slot_remap_copy_failed_count": r.get(
                        "driver_slot_remap_copy_failed_count"
                    ),
                }
                for r in driver_copy_only_mismatch_records[:8]
            ],
            "driver_metadata_copy_counter_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "driver_slot_remap_mode": r.get("driver_slot_remap_mode"),
                    "driver_slot_remap_copy_count": r.get(
                        "driver_slot_remap_copy_count"
                    ),
                    "driver_slot_remap_copy_bytes": r.get(
                        "driver_slot_remap_copy_bytes"
                    ),
                    "driver_slot_remap_copy_failed_count": r.get(
                        "driver_slot_remap_copy_failed_count"
                    ),
                }
                for r in driver_metadata_copy_counter_records[:8]
            ],
            "driver_copy_failure_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "backend_status": r.get("backend_status"),
                    "driver_slot_remap_copy_failed_count": r.get(
                        "driver_slot_remap_copy_failed_count"
                    ),
                    "driver_result": r.get("driver_result"),
                    "driver_prefill_result": r.get("driver_prefill_result"),
                }
                for r in driver_copy_failure_records[:8]
            ],
            "driver_failure_count_records": [
                {
                    "layer_id": r.get("layer_id"),
                    "step": r.get("step"),
                    "backend_status": r.get("backend_status"),
                    "driver_slot_remap_failure_count": r.get(
                        "driver_slot_remap_failure_count"
                    ),
                    "driver_result": r.get("driver_result"),
                }
                for r in driver_failure_count_records[:8]
            ],
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    stats = summary["slot_pool"]
    lines = [
        "# Cold Slot Pool Trace Report",
        "",
        f"overall={'PASS' if passed else 'FAIL'}",
        "",
        "## Checks",
        "",
    ]
    for key, ok in checks.items():
        lines.append(f"- {key}: {'PASS' if ok else 'FAIL'}")
    lines += [
        "",
        "## Counters",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "slot_hits",
        "slot_misses",
        "slot_assignments",
        "overflow_count",
        "parallel_copy_count",
        "planned_remap_count",
        "planned_tlb_invalidate_count",
        "migration_range_count",
        "migration_expert_count",
        "assigned_slot_count",
        "migration_bytes",
        "migration_issued_records",
        "migration_failed_records",
        "migration_issued_bytes",
        "slot_id_conflicts",
        "protected_slot_conflicts",
        "stale_read_detected",
        "max_assignments_per_record",
        "max_overflow_per_record",
        "max_slot_capacity",
        "allocation_us_total",
        "backend_migration_us_total",
        "driver_slot_remap_records",
        "driver_slot_remap_entry_count",
        "driver_slot_remap_success_count",
        "driver_slot_remap_failure_count",
        "driver_slot_remap_conflict_count",
        "driver_slot_remap_requested_bytes",
        "driver_slot_remap_accepted_bytes",
        "driver_slot_remap_copy_count",
        "driver_slot_remap_copy_bytes",
        "driver_slot_remap_copy_failed_count",
    ):
        lines.append(f"| {key} | {fmt(stats.get(key))} |")
    lines += [
        "",
        "## Backend",
        "",
        f"- backend_counts: `{json.dumps(stats.get('backend_counts'), sort_keys=True)}`",
        f"- backend_status_counts: `{json.dumps(stats.get('backend_status_counts'), sort_keys=True)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"overall={'PASS' if passed else 'FAIL'}")
    return 0 if passed or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
