#!/usr/bin/env python3
"""Classify UVM replayable fault addresses against vLLM region logs.

This tool correlates:
1. vLLM tensor allocation regions from vllm_uvm_address_regions.log
2. UVM replayable fault addresses from uvm_kv_fault_addrs.log

It is designed to answer:
- whether the two logs are comparable in the same virtual address space
- which fault addresses hit weight regions
- which fault addresses hit KV cache regions
- which fault addresses remain unmatched
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SECTION_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+phase=(?P<phase>\S+)\s+pid=(?P<pid>\d+)\s+model=(?P<model>.+)$"
)
RAW_ADDR_RE = re.compile(r"(?:\braw|精确原始地址)=(0x[0-9a-fA-F]+)")
PAGE_ADDR_RE = re.compile(r"(?:\bpage|对齐到页地址)=(0x[0-9a-fA-F]+)")
STAT_INT_RE = re.compile(r"\b([A-Za-z_]+)=([0-9]+)")


@dataclass(frozen=True)
class Region:
    section_index: int
    timestamp: str
    phase: str
    pid: int
    model: str
    kind: str
    name: str
    start: int
    end: int
    size_bytes: int
    size_mb: float
    major_type: str
    is_summary: bool

    @property
    def start_hex(self) -> str:
        return f"0x{self.start:x}"

    @property
    def end_hex(self) -> str:
        return f"0x{self.end:x}"


@dataclass(frozen=True)
class Section:
    index: int
    timestamp: str
    phase: str
    pid: int
    model: str
    rows: tuple[Region, ...]


@dataclass(frozen=True)
class FaultRecord:
    line_no: int
    raw_address: int | None
    page_address: int | None

    def selected_address(self, use_raw: bool) -> int | None:
        if use_raw:
            return self.raw_address if self.raw_address is not None else self.page_address
        return self.page_address if self.page_address is not None else self.raw_address


class RegionLookup:
    def __init__(self, regions: Iterable[Region]):
        self.regions = sorted(regions, key=lambda region: (region.start, region.end))
        self.starts = [region.start for region in self.regions]

    def find(self, address: int) -> Region | None:
        idx = bisect.bisect_right(self.starts, address) - 1
        if idx < 0:
            return None
        region = self.regions[idx]
        if address <= region.end:
            return region
        return None

    def contains(self, address: int) -> bool:
        return self.find(address) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify UVM fault addresses using vLLM region logs."
    )
    parser.add_argument(
        "--address-log",
        default="/tmp/vllm_uvm_address_regions.log",
        help="Path to vLLM address region log.",
    )
    parser.add_argument(
        "--fault-log",
        default="/tmp/uvm_kv_fault_addrs.log",
        help="Path to UVM replayable fault address log.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Restrict address-log sections to a specific vLLM pid. Default: latest pid in the log.",
    )
    parser.add_argument(
        "--use-raw-address",
        action="store_true",
        help="Match faults using raw addresses instead of page-aligned addresses.",
    )
    parser.add_argument(
        "--regions-csv",
        default=None,
        help="Optional output CSV for normalized address regions.",
    )
    parser.add_argument(
        "--faults-csv",
        default=None,
        help="Optional output CSV for per-fault classification results.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional output JSON summary file.",
    )
    parser.add_argument(
        "--stats-log",
        default=None,
        help=(
            "Optional UVM replayable fault stats log. When provided, the report "
            "includes delta_after_dedup using the same baseline formula as "
            "run_kv_fault_ratio.sh."
        ),
    )
    parser.add_argument(
        "--dedup-mode",
        choices=("none", "global-page"),
        default="none",
        help=(
            "Deduplicate parsed fault records before classification. "
            "'global-page' keeps the first occurrence of each selected page address. "
            "This is an approximation of after-dedup attribution; exact after-dedup "
            "classification needs kernel-side logging after UVM duplicate removal."
        ),
    )
    parser.add_argument(
        "--max-faults",
        type=int,
        default=None,
        help="Stop after classifying N parsed fault records.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top matched regions to report.",
    )
    return parser.parse_args()


def classify_major_type(kind: str) -> tuple[str, bool]:
    if kind.startswith("weight:"):
        return "weight", False
    if kind in {"kv_cache:contiguous_range", "kv_cache:span_range"}:
        return "kv_cache", True
    if kind.startswith("kv_cache"):
        return "kv_cache", False
    return "other", False


def parse_address_log(path: Path) -> tuple[list[Section], list[str]]:
    warnings: list[str] = []
    sections: list[Section] = []
    current_meta: dict[str, object] | None = None
    current_rows: list[Region] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            header_match = SECTION_RE.match(stripped)
            if header_match:
                if current_meta is not None:
                    sections.append(
                        Section(
                            index=current_meta["index"],  # type: ignore[arg-type]
                            timestamp=current_meta["timestamp"],  # type: ignore[arg-type]
                            phase=current_meta["phase"],  # type: ignore[arg-type]
                            pid=current_meta["pid"],  # type: ignore[arg-type]
                            model=current_meta["model"],  # type: ignore[arg-type]
                            rows=tuple(current_rows),
                        )
                    )

                current_meta = {
                    "index": len(sections),
                    "timestamp": header_match.group("timestamp"),
                    "phase": header_match.group("phase"),
                    "pid": int(header_match.group("pid")),
                    "model": header_match.group("model"),
                }
                current_rows = []
                continue

            if stripped == "kind,name,start,end,size_bytes,size_mb":
                continue

            if current_meta is None:
                warnings.append(
                    f"line {line_no}: ignored data row before any section header: {stripped[:120]}"
                )
                continue

            parts = stripped.split(",", 5)
            if len(parts) != 6:
                warnings.append(
                    f"line {line_no}: expected 6 CSV fields in address log, got {len(parts)}"
                )
                continue

            kind, name, start_hex, end_hex, size_bytes_str, size_mb_str = parts
            try:
                start = int(start_hex, 16)
                end = int(end_hex, 16)
                size_bytes = int(size_bytes_str)
                size_mb = float(size_mb_str)
            except ValueError as exc:
                warnings.append(f"line {line_no}: failed to parse region row: {exc}")
                continue

            if end < start:
                warnings.append(
                    f"line {line_no}: region end is below start: {start_hex} .. {end_hex}"
                )
                continue

            major_type, is_summary = classify_major_type(kind)
            current_rows.append(
                Region(
                    section_index=current_meta["index"],  # type: ignore[arg-type]
                    timestamp=current_meta["timestamp"],  # type: ignore[arg-type]
                    phase=current_meta["phase"],  # type: ignore[arg-type]
                    pid=current_meta["pid"],  # type: ignore[arg-type]
                    model=current_meta["model"],  # type: ignore[arg-type]
                    kind=kind,
                    name=name,
                    start=start,
                    end=end,
                    size_bytes=size_bytes,
                    size_mb=size_mb,
                    major_type=major_type,
                    is_summary=is_summary,
                )
            )

    if current_meta is not None:
        sections.append(
            Section(
                index=current_meta["index"],  # type: ignore[arg-type]
                timestamp=current_meta["timestamp"],  # type: ignore[arg-type]
                phase=current_meta["phase"],  # type: ignore[arg-type]
                pid=current_meta["pid"],  # type: ignore[arg-type]
                model=current_meta["model"],  # type: ignore[arg-type]
                rows=tuple(current_rows),
            )
        )

    return sections, warnings


def select_sections(sections: list[Section], pid: int | None) -> tuple[int, list[Section]]:
    if not sections:
        raise ValueError("no sections were found in the address log")

    selected_pid = pid if pid is not None else sections[-1].pid
    same_pid = [section for section in sections if section.pid == selected_pid]
    if not same_pid:
        raise ValueError(f"no sections found for pid={selected_pid}")

    latest_by_phase: dict[str, Section] = {}
    for section in same_pid:
        latest_by_phase[section.phase] = section

    selected_sections = sorted(latest_by_phase.values(), key=lambda section: section.index)
    return selected_pid, selected_sections


def merge_intervals(regions: list[Region]) -> list[tuple[int, int]]:
    if not regions:
        return []

    merged: list[tuple[int, int]] = []
    sorted_regions = sorted(regions, key=lambda region: (region.start, region.end))
    current_start = sorted_regions[0].start
    current_end = sorted_regions[0].end

    for region in sorted_regions[1:]:
        if region.start <= current_end + 1:
            current_end = max(current_end, region.end)
            continue
        merged.append((current_start, current_end))
        current_start = region.start
        current_end = region.end

    merged.append((current_start, current_end))
    return merged


def interval_bytes(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


def find_overlaps(regions: list[Region]) -> list[dict[str, object]]:
    overlaps: list[dict[str, object]] = []
    if not regions:
        return overlaps

    ordered = sorted(regions, key=lambda region: (region.start, region.end))
    prev = ordered[0]
    for region in ordered[1:]:
        if region.start <= prev.end:
            overlaps.append(
                {
                    "left_kind": prev.kind,
                    "left_name": prev.name,
                    "left_start": prev.start_hex,
                    "left_end": prev.end_hex,
                    "right_kind": region.kind,
                    "right_name": region.name,
                    "right_start": region.start_hex,
                    "right_end": region.end_hex,
                }
            )
            if region.end > prev.end:
                prev = region
        else:
            prev = region
    return overlaps


def parse_fault_record(line: str, line_no: int) -> FaultRecord | None:
    raw_match = RAW_ADDR_RE.search(line)
    page_match = PAGE_ADDR_RE.search(line)
    if raw_match is None and page_match is None:
        return None
    raw_address = int(raw_match.group(1), 16) if raw_match else None
    page_address = int(page_match.group(1), 16) if page_match else None
    return FaultRecord(line_no=line_no, raw_address=raw_address, page_address=page_address)


def extract_stat_ints(line: str) -> dict[str, int]:
    return {key: int(value) for key, value in STAT_INT_RE.findall(line)}


def safe_delta(last: int | None, baseline: int | None) -> int | None:
    if last is None or baseline is None:
        return None
    return max(0, last - baseline)


def ratio_percent(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator * 100.0 / denominator


def parse_stats_delta(path: Path) -> dict[str, object]:
    stat_lines: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "uvm_parent_gpu_service_replayable_faults" not in line:
                continue
            values = extract_stat_ints(line)
            if "total_faults" in values and "batch_faults" in values:
                stat_lines.append(values)

    if not stat_lines:
        return {
            "stats_log": str(path),
            "parsed_stat_records": 0,
            "error": "no_replayable_fault_stats_found",
        }

    first = stat_lines[0]
    last = stat_lines[-1]
    base_faults = first.get("total_faults", 0) - first.get("batch_faults", 0)
    base_duplicates = first.get("total_duplicates", 0) - first.get("batch_duplicates", 0)
    base_after_dedup = first.get("total_after_dedup", 0) - first.get("batch_after_dedup", 0)
    base_kv_faults = first.get("total_kv_faults", 0) - first.get("batch_kv_faults", 0)
    base_kv_duplicates = first.get("total_kv_duplicates", 0) - first.get("batch_kv_duplicates", 0)
    base_kv_after_dedup = first.get("total_kv_after_dedup", 0) - first.get(
        "batch_kv_after_dedup", 0
    )

    delta_faults = safe_delta(last.get("total_faults"), base_faults)
    delta_duplicates = safe_delta(last.get("total_duplicates"), base_duplicates)
    delta_after_dedup = safe_delta(last.get("total_after_dedup"), base_after_dedup)
    delta_kv_faults = safe_delta(last.get("total_kv_faults"), base_kv_faults)
    delta_kv_duplicates = safe_delta(last.get("total_kv_duplicates"), base_kv_duplicates)
    delta_kv_after_dedup = safe_delta(
        last.get("total_kv_after_dedup"), base_kv_after_dedup
    )

    return {
        "stats_log": str(path),
        "parsed_stat_records": len(stat_lines),
        "delta_faults": delta_faults,
        "delta_duplicates": delta_duplicates,
        "delta_after_dedup": delta_after_dedup,
        "delta_kv_faults": delta_kv_faults,
        "delta_kv_duplicates": delta_kv_duplicates,
        "delta_kv_after_dedup": delta_kv_after_dedup,
        "delta_kv_ratio_pct": ratio_percent(delta_kv_faults, delta_faults),
        "delta_kv_after_dedup_ratio_pct": ratio_percent(
            delta_kv_after_dedup, delta_after_dedup
        ),
    }


def write_regions_csv(path: Path, regions: list[Region]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "section_index",
                "timestamp",
                "phase",
                "pid",
                "model",
                "major_type",
                "kind",
                "name",
                "start_hex",
                "end_hex",
                "size_bytes",
                "size_mb",
                "is_summary",
            ]
        )
        for region in sorted(regions, key=lambda item: (item.start, item.end, item.kind, item.name)):
            writer.writerow(
                [
                    region.section_index,
                    region.timestamp,
                    region.phase,
                    region.pid,
                    region.model,
                    region.major_type,
                    region.kind,
                    region.name,
                    region.start_hex,
                    region.end_hex,
                    region.size_bytes,
                    f"{region.size_mb:.3f}",
                    int(region.is_summary),
                ]
            )


def classify_faults(
    fault_log_path: Path,
    concrete_lookup: RegionLookup,
    kv_summary_lookup: RegionLookup,
    use_raw_address: bool,
    dedup_mode: str,
    top_n: int,
    faults_csv_path: Path | None,
    max_faults: int | None,
) -> dict[str, object]:
    parsed_lines = 0
    classified_records = 0
    duplicate_records_skipped = 0
    total_lines = 0
    ignored_lines = 0
    address_min: int | None = None
    address_max: int | None = None
    classification_counts: Counter[str] = Counter()
    matched_kind_counts: Counter[str] = Counter()
    matched_name_counts: Counter[str] = Counter()
    matched_phase_counts: Counter[str] = Counter()
    unknown_examples: list[dict[str, object]] = []
    seen_selected_addresses: set[int] = set()
    csv_handle = None
    csv_writer = None

    if faults_csv_path is not None:
        csv_handle = faults_csv_path.open("w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(
            [
                "line_no",
                "raw_address_hex",
                "page_address_hex",
                "selected_address_hex",
                "classification",
                "in_kv_summary_span",
                "matched_kind",
                "matched_name",
                "matched_phase",
                "matched_pid",
            ]
        )

    try:
        with fault_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for total_lines, line in enumerate(handle, start=1):
                record = parse_fault_record(line, total_lines)
                if record is None:
                    ignored_lines += 1
                    continue

                parsed_lines += 1
                selected_address = record.selected_address(use_raw_address)
                if selected_address is None:
                    classification_counts["unknown"] += 1
                    continue

                if dedup_mode == "global-page":
                    if selected_address in seen_selected_addresses:
                        duplicate_records_skipped += 1
                        if max_faults is not None and parsed_lines >= max_faults:
                            break
                        continue
                    seen_selected_addresses.add(selected_address)

                if address_min is None or selected_address < address_min:
                    address_min = selected_address
                if address_max is None or selected_address > address_max:
                    address_max = selected_address

                matched_region = concrete_lookup.find(selected_address)
                in_kv_summary_span = kv_summary_lookup.contains(selected_address)
                classified_records += 1

                if matched_region is None:
                    classification = (
                        "unknown_in_kv_summary_span" if in_kv_summary_span else "unknown"
                    )
                    classification_counts[classification] += 1
                    if len(unknown_examples) < 10:
                        unknown_examples.append(
                            {
                                "line_no": record.line_no,
                                "raw_address_hex": (
                                    f"0x{record.raw_address:x}" if record.raw_address is not None else None
                                ),
                                "page_address_hex": (
                                    f"0x{record.page_address:x}"
                                    if record.page_address is not None
                                    else None
                                ),
                                "selected_address_hex": f"0x{selected_address:x}",
                                "classification": classification,
                            }
                        )
                    if csv_writer is not None:
                        csv_writer.writerow(
                            [
                                record.line_no,
                                f"0x{record.raw_address:x}" if record.raw_address is not None else "",
                                f"0x{record.page_address:x}" if record.page_address is not None else "",
                                f"0x{selected_address:x}",
                                classification,
                                int(in_kv_summary_span),
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
                else:
                    classification = matched_region.major_type
                    classification_counts[classification] += 1
                    matched_kind_counts[matched_region.kind] += 1
                    matched_name_counts[matched_region.name] += 1
                    matched_phase_counts[matched_region.phase] += 1
                    if csv_writer is not None:
                        csv_writer.writerow(
                            [
                                record.line_no,
                                f"0x{record.raw_address:x}" if record.raw_address is not None else "",
                                f"0x{record.page_address:x}" if record.page_address is not None else "",
                                f"0x{selected_address:x}",
                                classification,
                                int(in_kv_summary_span),
                                matched_region.kind,
                                matched_region.name,
                                matched_region.phase,
                                matched_region.pid,
                            ]
                        )

                if max_faults is not None and classified_records >= max_faults:
                    break
    finally:
        if csv_handle is not None:
            csv_handle.close()

    return {
        "total_lines": total_lines,
        "parsed_fault_records": parsed_lines,
        "classified_fault_records": classified_records,
        "dedup_mode": dedup_mode,
        "duplicate_records_skipped": duplicate_records_skipped,
        "unique_selected_addresses": len(seen_selected_addresses)
        if dedup_mode == "global-page"
        else None,
        "ignored_lines": ignored_lines,
        "selected_address_min": f"0x{address_min:x}" if address_min is not None else None,
        "selected_address_max": f"0x{address_max:x}" if address_max is not None else None,
        "classification_counts": dict(classification_counts),
        "top_matched_kinds": matched_kind_counts.most_common(top_n),
        "top_matched_names": matched_name_counts.most_common(top_n),
        "matched_phase_counts": dict(matched_phase_counts),
        "unknown_examples": unknown_examples,
    }


def regions_summary(selected_sections: list[Section]) -> tuple[list[Region], dict[str, object]]:
    all_regions = [region for section in selected_sections for region in section.rows]
    concrete_regions = [region for region in all_regions if region.major_type in {"weight", "kv_cache"} and not region.is_summary]
    kv_summary_regions = [region for region in all_regions if region.kind in {"kv_cache:contiguous_range", "kv_cache:span_range"}]

    by_type: dict[str, list[Region]] = defaultdict(list)
    for region in concrete_regions:
        by_type[region.major_type].append(region)

    type_summary: dict[str, object] = {}
    for major_type, regions in sorted(by_type.items()):
        merged = merge_intervals(regions)
        type_summary[major_type] = {
            "region_count": len(regions),
            "merged_interval_count": len(merged),
            "total_region_bytes": sum(region.size_bytes for region in regions),
            "merged_covered_bytes": interval_bytes(merged),
            "min_start": f"0x{min(region.start for region in regions):x}",
            "max_end": f"0x{max(region.end for region in regions):x}",
        }

    overall_merged = merge_intervals(concrete_regions)
    overlaps = find_overlaps(concrete_regions)
    summary = {
        "section_count": len(selected_sections),
        "sections": [
            {
                "index": section.index,
                "timestamp": section.timestamp,
                "phase": section.phase,
                "pid": section.pid,
                "model": section.model,
                "row_count": len(section.rows),
            }
            for section in selected_sections
        ],
        "concrete_region_count": len(concrete_regions),
        "kv_summary_region_count": len(kv_summary_regions),
        "types": type_summary,
        "overall_merged_interval_count": len(overall_merged),
        "overall_covered_bytes": interval_bytes(overall_merged),
        "overall_min_start": f"0x{min(region.start for region in concrete_regions):x}" if concrete_regions else None,
        "overall_max_end": f"0x{max(region.end for region in concrete_regions):x}" if concrete_regions else None,
        "overlap_count": len(overlaps),
        "overlap_examples": overlaps[:10],
    }
    return all_regions, summary


def address_space_assessment(region_summary: dict[str, object], fault_summary: dict[str, object]) -> dict[str, object]:
    region_min_hex = region_summary.get("overall_min_start")
    region_max_hex = region_summary.get("overall_max_end")
    fault_min_hex = fault_summary.get("selected_address_min")
    fault_max_hex = fault_summary.get("selected_address_max")
    classification_counts = fault_summary["classification_counts"]

    region_min = int(region_min_hex, 16) if region_min_hex else None
    region_max = int(region_max_hex, 16) if region_max_hex else None
    fault_min = int(fault_min_hex, 16) if fault_min_hex else None
    fault_max = int(fault_max_hex, 16) if fault_max_hex else None

    span_overlap = False
    if None not in (region_min, region_max, fault_min, fault_max):
        span_overlap = not (fault_max < region_min or fault_min > region_max)

    matched_faults = classification_counts.get("weight", 0) + classification_counts.get("kv_cache", 0)
    if matched_faults > 0:
        conclusion = "same_virtual_address_space_very_likely"
    elif span_overlap:
        conclusion = "same_virtual_address_space_likely_but_no_direct_tensor_hit"
    else:
        conclusion = "not_confirmed_from_current_logs"

    return {
        "comparable_by_design": True,
        "reason": (
            "The fault log records UVM replayable fault addresses after page alignment, "
            "while the vLLM log records tensor data_ptr()-based UVA ranges. "
            "They are intended to be compared in the same GPU virtual address space."
        ),
        "selected_region_span_min": region_min_hex,
        "selected_region_span_max": region_max_hex,
        "fault_selected_address_min": fault_min_hex,
        "fault_selected_address_max": fault_max_hex,
        "fault_range_overlaps_region_span": span_overlap,
        "matched_faults": matched_faults,
        "matched_weight_faults": classification_counts.get("weight", 0),
        "matched_kv_faults": classification_counts.get("kv_cache", 0),
        "unknown_faults": classification_counts.get("unknown", 0),
        "unknown_in_kv_summary_span_faults": classification_counts.get(
            "unknown_in_kv_summary_span", 0
        ),
        "conclusion": conclusion,
    }


def print_report(report: dict[str, object]) -> None:
    print("Selected Address Sections")
    for section in report["region_summary"]["sections"]:
        print(
            f"- section#{section['index']} pid={section['pid']} phase={section['phase']} "
            f"timestamp={section['timestamp']} rows={section['row_count']}"
        )

    print("\nRegion Coverage")
    for major_type, info in report["region_summary"]["types"].items():
        print(
            f"- {major_type}: regions={info['region_count']} merged_intervals={info['merged_interval_count']} "
            f"covered_bytes={info['merged_covered_bytes']} span={info['min_start']}..{info['max_end']}"
        )
    print(
        f"- overall: intervals={report['region_summary']['overall_merged_interval_count']} "
        f"covered_bytes={report['region_summary']['overall_covered_bytes']} "
        f"span={report['region_summary']['overall_min_start']}..{report['region_summary']['overall_max_end']}"
    )
    print(f"- overlaps_detected: {report['region_summary']['overlap_count']}")

    print("\nFault Classification")
    fault_summary = report["fault_summary"]
    print(
        f"- parsed_fault_records={fault_summary['parsed_fault_records']} "
        f"classified_fault_records={fault_summary.get('classified_fault_records')} "
        f"dedup_mode={fault_summary.get('dedup_mode')} "
        f"duplicate_records_skipped={fault_summary.get('duplicate_records_skipped')} "
        f"ignored_lines={fault_summary['ignored_lines']} total_lines_seen={fault_summary['total_lines']}"
    )
    print(
        f"- selected_address_span={fault_summary['selected_address_min']}..{fault_summary['selected_address_max']}"
    )
    for name, count in sorted(fault_summary["classification_counts"].items()):
        print(f"- {name}: {count}")

    stats_summary = report.get("stats_summary")
    if stats_summary:
        print("\nReplayable Fault Delta")
        print(f"- stats_log: {stats_summary.get('stats_log')}")
        print(f"- parsed_stat_records: {stats_summary.get('parsed_stat_records')}")
        print(
            f"- delta_after_dedup: {stats_summary.get('delta_after_dedup')} "
            f"delta_kv_after_dedup: {stats_summary.get('delta_kv_after_dedup')}"
        )
        print(
            f"- delta_kv_ratio_pct: {stats_summary.get('delta_kv_ratio_pct')} "
            f"delta_kv_after_dedup_ratio_pct: {stats_summary.get('delta_kv_after_dedup_ratio_pct')}"
        )

    print("\nAddress Space Assessment")
    assessment = report["address_space_assessment"]
    print(f"- comparable_by_design: {assessment['comparable_by_design']}")
    print(f"- fault_range_overlaps_region_span: {assessment['fault_range_overlaps_region_span']}")
    print(f"- conclusion: {assessment['conclusion']}")

    top_names = fault_summary["top_matched_names"]
    if top_names:
        print("\nTop Matched Regions")
        for name, count in top_names:
            print(f"- {name}: {count}")

    unknown_examples = fault_summary["unknown_examples"]
    if unknown_examples:
        print("\nUnknown Examples")
        for example in unknown_examples:
            print(
                f"- line={example['line_no']} selected={example['selected_address_hex']} "
                f"classification={example['classification']}"
            )


def main() -> int:
    # 1. 解析命令行参数并获取日志文件路径
    args = parse_args()
    address_log_path = Path(args.address_log)
    fault_log_path = Path(args.fault_log)
    stats_log_path = Path(args.stats_log) if args.stats_log else None

    # 2. 前置检查：确保输入的 vLLM 地址日志和 UVM 缺页日志确实存在
    if not address_log_path.is_file():
        print(f"address log not found: {address_log_path}", file=sys.stderr)
        return 2
    if not fault_log_path.is_file():
        print(f"fault log not found: {fault_log_path}", file=sys.stderr)
        return 2

    # 3. 提取 (Extract)：解析 vLLM 打印的张量 UVA (统一虚拟地址) 分配图谱
    # sections 按时间和阶段(phase)划分了不同的地址快照
    sections, warnings = parse_address_log(address_log_path)
    
    # 4. 转换/过滤 (Transform)：锁定目标进程
    # 如果存在多个 vLLM worker，过滤出目标 PID（默认取日志中最后一个活跃的 PID）最新的地址快照
    selected_pid, selected_sections = select_sections(sections, args.pid)
    
    # 提取所有 region 并生成统计摘要（如各类型覆盖的总字节数、合并后的地址区间）
    all_regions, region_summary = regions_summary(selected_sections)
    
    # 剥离出具体的、拥有实际物理载荷的张量区域（排除掉 summary 类型的统计区间）
    # 这是后续高频执行二分查找的"精准靶库"
    concrete_regions = [
        region
        for region in all_regions
        if region.major_type in {"weight", "kv_cache"} and not region.is_summary
    ]
    
    # 提取 KV cache 的宏观跨度区域（用于辅助判断 unknown fault 是否落在了 KV 内存池的大盘子内）
    kv_summary_regions = [
        region for region in all_regions if region.kind in {"kv_cache:contiguous_range", "kv_cache:span_range"}
    ]

    # 5. 可选输出：导出对齐后的地址区间图谱到 CSV 供进一步分析
    if args.regions_csv:
        write_regions_csv(Path(args.regions_csv), all_regions)

    # 6. 核心比对 (Correlate/Classify)：用 UVM 底层故障地址去碰撞 vLLM 上层的张量地址
    fault_summary = classify_faults(
        fault_log_path=fault_log_path,
        concrete_lookup=RegionLookup(concrete_regions),     # 权重和 KV 的精确查找表 (内部用二分查找优化)
        kv_summary_lookup=RegionLookup(kv_summary_regions), # KV 宏观跨度的辅助查找表
        use_raw_address=args.use_raw_address,               # 是用原始地址还是按页对齐后的地址进行匹配
        dedup_mode=args.dedup_mode,
        top_n=args.top_n,                                   # 统计报错最频繁的 Top N 张量
        faults_csv_path=Path(args.faults_csv) if args.faults_csv else None,
        max_faults=args.max_faults,                         # 用于测试或限制处理规模，提前截断
    )

    stats_summary = parse_stats_delta(stats_log_path) if stats_log_path else None

    # 7. 交叉验证 (Validation)：评估两份日志的虚拟地址空间是否匹配
    # 如果 UVM fault 发生的区间跟 vLLM 张量分配的区间完全没有交集，说明日志抓串了或者环境脏了
    assessment = address_space_assessment(region_summary, fault_summary)
    
    # 8. 组装最终报告载荷
    report = {
        "selected_pid": selected_pid,
        "address_log": str(address_log_path),
        "fault_log": str(fault_log_path),
        "used_raw_addresses": bool(args.use_raw_address),
        "warnings": warnings,
        "region_summary": region_summary,
        "fault_summary": fault_summary,
        "stats_summary": stats_summary,
        "address_space_assessment": assessment,
    }

    # 9. 加载/输出 (Load)：持久化结构化报告（方便与其他 CI/CD 或看板工具集成）
    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    # 10. 终端呈现：在 stdout 打印人类可读的分析简报
    print_report(report)
    
    # 打印解析阶段抛出的异常或不规范日志警告（限制最多输出 20 条防刷屏）
    if warnings:
        print("\nWarnings")
        for warning in warnings[:20]:
            print(f"- {warning}")
        if len(warnings) > 20:
            print(f"- ... {len(warnings) - 20} more warnings omitted")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
