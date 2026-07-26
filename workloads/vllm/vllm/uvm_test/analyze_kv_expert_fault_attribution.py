#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path
from typing import Any

PAGE_SIZE = 4096

FAULT_RE = re.compile(
    r"Replayable fault GPU .*?\braw=(0x[0-9a-fA-F]+)"
    r".*?\bpage=(0x[0-9a-fA-F]+)"
    r".*?\btype=(\S+).*?\baccess=(\S+)"
)
STATS_RE = re.compile(r"Replayable fault stats GPU .*")


def page_floor(value: int) -> int:
    return value - (value % PAGE_SIZE)


def page_ceil(value: int) -> int:
    return ((value + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE


def parse_metric(text: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    return float(match.group(1)) if match else None


def add_interval(
    intervals: list[tuple[int, int, str, dict[str, Any]]],
    start: int,
    end_inclusive: int,
    kind: str,
    record: dict[str, Any],
) -> None:
    if start <= 0 or end_inclusive < start:
        return
    intervals.append(
        (page_floor(start), page_ceil(end_inclusive + 1), kind, record)
    )


def load_kv_intervals(
    address_log: Path,
    kv_probe: Path,
) -> tuple[list[tuple[int, int, str, dict[str, Any]]], list[dict[str, Any]]]:
    intervals: list[tuple[int, int, str, dict[str, Any]]] = []
    records: list[dict[str, Any]] = []

    if address_log.exists():
        phase: str | None = None
        for line in address_log.read_text(errors="replace").splitlines():
            if line.startswith("["):
                match = re.search(r"phase=([^\s]+)", line)
                phase = match.group(1) if match else None
                continue
            if not line or line.startswith("kind,"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            kind, name, start_s, end_s, size_s = parts[:5]
            if not kind.startswith("kv_cache") or "range" in kind:
                continue
            try:
                start = int(start_s, 16)
                end = int(end_s, 16)
                size = int(size_s)
            except ValueError:
                continue
            record = {
                "source": "address_log",
                "phase": phase,
                "kind": kind,
                "name": name,
                "start": start,
                "end": end,
                "size_bytes": size,
            }
            records.append(record)
            add_interval(intervals, start, end, "kv", record)

    if kv_probe.exists():
        for line in kv_probe.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") not in ("kv_zeros_return", "kv_tensor_ready"):
                continue
            start = int(record.get("ptr") or 0)
            end = int(record.get("end") or 0)
            if start <= 0 or end < start:
                continue
            merged = {"source": "kv_init_probe", **record}
            records.append(merged)
            add_interval(intervals, start, end, "kv", merged)

    seen: set[tuple[int, int, str]] = set()
    deduped: list[tuple[int, int, str, dict[str, Any]]] = []
    for interval in intervals:
        key = (interval[0], interval[1], interval[2])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(interval)
    deduped.sort(key=lambda item: (item[0], item[1]))
    return deduped, records


def load_weight_intervals(
    weight_map: Path,
) -> tuple[
    list[tuple[int, int, str, dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, int],
]:
    intervals: list[tuple[int, int, str, dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    capability_reasons: dict[str, int] = {}
    semantic_classes: dict[str, int] = {}

    if not weight_map.exists():
        return intervals, records, capability_reasons, semantic_classes

    for line in weight_map.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        start = int(record.get("start_int") or 0)
        end = int(record.get("end_int") or 0)
        if start <= 0 or end < start:
            continue
        records.append(record)
        reason = str(record.get("driver_capability_reason") or "unknown")
        capability_reasons[reason] = capability_reasons.get(reason, 0) + 1
        semantic = str(record.get("semantic_type") or record.get("weight_class")
                       or "other")
        semantic_classes[semantic] = semantic_classes.get(semantic, 0) + 1

        if bool(record.get("is_moe_expert")):
            if bool(record.get("is_logical_expert_slice")):
                add_interval(intervals, start, end, "expert", record)
        else:
            add_interval(intervals, start, end, "other_weight", record)

    intervals.sort(key=lambda item: (item[0], item[1]))
    return intervals, records, capability_reasons, semantic_classes


def build_index(
    intervals: list[tuple[int, int, str, dict[str, Any]]],
) -> tuple[list[int], list[tuple[int, int, str, dict[str, Any]]]]:
    return [interval[0] for interval in intervals], intervals


def find_interval(
    address: int,
    index: tuple[list[int], list[tuple[int, int, str, dict[str, Any]]]],
) -> tuple[int, int, str, dict[str, Any]] | None:
    starts, intervals = index
    position = bisect.bisect_right(starts, address) - 1
    for cursor in range(position, max(position - 32, -1), -1):
        if cursor < 0:
            break
        start, end, kind, record = intervals[cursor]
        if start <= address < end:
            return start, end, kind, record
    return None


def classify_fault(
    page: int,
    kv_index: tuple[list[int], list[tuple[int, int, str, dict[str, Any]]]],
    expert_index: tuple[list[int], list[tuple[int, int, str, dict[str, Any]]]],
    other_weight_index: tuple[
        list[int],
        list[tuple[int, int, str, dict[str, Any]]],
    ],
) -> tuple[int, int, str, dict[str, Any]] | None:
    for index in (kv_index, expert_index, other_weight_index):
        match = find_interval(page, index)
        if match is not None:
            return match
    return None


def count_latency(path: Path) -> dict[str, int | None]:
    text = path.read_text(errors="replace") if path.exists() else ""
    keys = (
        "remote_first_faults",
        "remote_first_pages",
        "stage_a_cpu_faults_total",
        "stage_a_cpu_faults_weight",
        "stage_a_cpu_faults_kv",
        "stage_a_cpu_faults_scratch",
        "stage_a_cpu_faults_other",
        "gpu_only_cpu_faults",
    )
    result: dict[str, int | None] = {}
    for key in keys:
        match = re.search(rf"^{key}\s+(\d+)", text, re.MULTILINE)
        result[key] = int(match.group(1)) if match else None
    return result


def summarize_case(case_dir: Path) -> dict[str, Any]:
    kv_intervals, kv_records = load_kv_intervals(
        case_dir / "address_regions.log",
        case_dir / "kv_init_probe.jsonl",
    )
    weight_intervals, weight_records, capability_reasons, semantic_classes = (
        load_weight_intervals(case_dir / "weight_map.jsonl")
    )
    expert_intervals = [
        interval for interval in weight_intervals if interval[2] == "expert"
    ]
    other_weight_intervals = [
        interval for interval in weight_intervals if interval[2] == "other_weight"
    ]

    kv_index = build_index(kv_intervals)
    expert_index = build_index(expert_intervals)
    other_weight_index = build_index(other_weight_intervals)

    fault_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    access_counts: dict[str, int] = {}
    top_experts: dict[str, int] = {}
    top_unknown: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    fault_lines = 0
    stats_lines = 0

    fault_trace = case_dir / "fault_trace.log"
    if fault_trace.exists():
        for line in fault_trace.read_text(errors="replace").splitlines():
            if STATS_RE.search(line):
                stats_lines += 1
            match = FAULT_RE.search(line)
            if not match:
                continue
            fault_lines += 1
            raw = int(match.group(1), 16)
            page = int(match.group(2), 16)
            fault_type = match.group(3)
            access = match.group(4)
            type_counts[fault_type] = type_counts.get(fault_type, 0) + 1
            access_counts[access] = access_counts.get(access, 0) + 1

            hit = classify_fault(page, kv_index, expert_index, other_weight_index)
            if hit is None:
                fault_class = "unknown"
                top_unknown[f"0x{page:x}"] = top_unknown.get(f"0x{page:x}", 0) + 1
                record = None
            else:
                _, _, fault_class, record = hit
                if fault_class == "expert":
                    key = (
                        f"L{record.get('layer_id')}:"
                        f"E{record.get('expert_id')}:"
                        f"{record.get('role')}"
                    )
                    top_experts[key] = top_experts.get(key, 0) + 1
            fault_counts[fault_class] = fault_counts.get(fault_class, 0) + 1

            if len(examples) < 12:
                example: dict[str, Any] = {
                    "page": f"0x{page:x}",
                    "raw": f"0x{raw:x}",
                    "type": fault_type,
                    "access": access,
                    "class": fault_class,
                }
                if hit is not None and record is not None:
                    example.update(
                        {
                            "name": record.get("name") or record.get("path"),
                            "layer_id": record.get("layer_id"),
                            "expert_id": record.get("expert_id"),
                            "role": record.get("role"),
                            "interval_start": f"0x{hit[0]:x}",
                            "interval_end_exclusive": f"0x{hit[1]:x}",
                        }
                    )
                examples.append(example)

    bench = (
        (case_dir / "bench.log").read_text(errors="replace")
        if (case_dir / "bench.log").exists()
        else ""
    )
    server = (
        (case_dir / "server.log").read_text(errors="replace")
        if (case_dir / "server.log").exists()
        else ""
    )
    kv_tokens = None
    match = re.search(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", server)
    if match:
        kv_tokens = int(match.group(1).replace(",", ""))

    return {
        "case": case_dir.name,
        "status": (
            (case_dir / "status.txt").read_text(errors="replace").strip()
            if (case_dir / "status.txt").exists()
            else None
        ),
        "completed": parse_metric(bench, "Successful requests"),
        "failed": parse_metric(bench, "Failed requests"),
        "output_tok_s": parse_metric(bench, "Output token throughput (tok/s)"),
        "mean_tpot_ms": parse_metric(bench, "Mean TPOT (ms)"),
        "p99_tpot_ms": parse_metric(bench, "P99 TPOT (ms)"),
        "kv_tokens": kv_tokens,
        "intervals": {
            "kv": len(kv_intervals),
            "kv_records": len(kv_records),
            "expert": len(expert_intervals),
            "other_weight": len(other_weight_intervals),
            "weight_records_total": len(weight_records),
        },
        "weight_driver_capability_reasons": dict(
            sorted(capability_reasons.items(), key=lambda item: (-item[1], item[0]))
        ),
        "weight_semantic_classes": dict(
            sorted(semantic_classes.items(), key=lambda item: (-item[1], item[0]))
        ),
        "fault_trace": {
            "replayable_fault_lines": fault_lines,
            "replayable_fault_stats_lines": stats_lines,
            "by_class": dict(
                sorted(fault_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "by_type": dict(
                sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "by_access": dict(
                sorted(access_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "top_expert_regions": dict(
                sorted(top_experts.items(), key=lambda item: -item[1])[:20]
            ),
            "top_unknown_pages": dict(
                sorted(top_unknown.items(), key=lambda item: -item[1])[:20]
            ),
            "examples": examples,
        },
        "latency_after": count_latency(case_dir / "latency_after.log"),
        "server_flags": {
            "has_xid": "Xid" in server,
            "has_uvm_fatal": "UVM fatal" in server,
            "has_mmu_fault": "MMU" in server or "FAULT_PDE" in server,
            "has_illegal_memory_access": "illegal memory access" in server,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    summary = {
        "run_root": str(args.run_root),
        "cases": [
            summarize_case(case_dir)
            for case_dir in sorted(args.run_root.iterdir())
            if case_dir.is_dir()
        ],
    }
    out = args.run_root / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
