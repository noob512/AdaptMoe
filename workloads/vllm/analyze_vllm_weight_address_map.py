#!/usr/bin/env python3
"""Analyze vLLM semantic weight address maps and optional UVM fault addresses."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RAW_ADDR_RE = re.compile(r"(?:\braw|精确原始地址)=(0x[0-9a-fA-F]+)")
PAGE_ADDR_RE = re.compile(r"(?:\bpage|对齐到页地址)=(0x[0-9a-fA-F]+)")
STAT_INT_RE = re.compile(r"\b([A-Za-z_]+)=([0-9]+)")


@dataclass(frozen=True)
class WeightRegion:
    index: int
    name: str
    kind: str
    start: int
    end: int
    size_bytes: int
    accounted_size_bytes: int
    semantic_type: str
    role: str
    fixed_weight: bool
    is_moe_expert: bool
    protected: bool
    evictable: bool
    placement_policy: str
    layer_id: int | None
    expert_id: int | None
    dtype: str
    is_alias: bool
    alias_of: str | None
    is_logical_slice: bool
    logical_parent_name: str | None
    logical_slice_index: int | None
    logical_slice_count: int | None
    driver_capable: bool | None
    driver_capability_reason: str | None
    is_main_weight: bool
    is_moe_expert_main_weight: bool
    residency_query_enabled: bool
    residency_query_ok: bool | None
    residency_state: str | None
    residency_query_granularity: str | None
    residency_query_start: int | None
    residency_query_size_bytes: int
    residency_queried_bytes: int
    resident_gpu_bytes: int
    resident_cpu_bytes: int
    resident_any_bytes: int
    resident_unknown_bytes: int
    resident_gpu_ratio: float | None
    phase: str | None
    pid: int | None

    @property
    def start_hex(self) -> str:
        return f"0x{self.start:x}"

    @property
    def end_hex(self) -> str:
        return f"0x{self.end:x}"


class RegionLookup:
    def __init__(self, regions: Iterable[WeightRegion]) -> None:
        self.regions = sorted(regions, key=lambda item: (item.start, item.end))
        self.starts = [item.start for item in self.regions]

    def find(self, address: int) -> WeightRegion | None:
        idx = bisect.bisect_right(self.starts, address) - 1
        if idx < 0:
            return None
        region = self.regions[idx]
        if address <= region.end:
            return region
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM weight address map coverage and fault attribution."
    )
    parser.add_argument("--weight-map", required=True, help="vLLM weight map JSONL.")
    parser.add_argument("--fault-log", help="Optional per-fault address log.")
    parser.add_argument("--stats-log", help="Optional UVM replayable fault stats log.")
    parser.add_argument(
        "--dedup-mode",
        choices=("none", "global-page"),
        default="global-page",
        help="Deduplicate fault addresses before classifying them.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 16) if value.startswith("0x") else int(value)
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def weight_region_from_record(index: int, record: dict[str, Any]) -> WeightRegion | None:
    start = parse_int(record.get("start_int")) or parse_int(record.get("start"))
    end = parse_int(record.get("end_int")) or parse_int(record.get("end"))
    size_bytes = parse_int(record.get("size_bytes"))
    if start is None or end is None or size_bytes is None or size_bytes <= 0:
        return None
    if end < start:
        return None
    layer_id = parse_int(record.get("layer_id"))
    expert_id = parse_int(record.get("expert_id"))
    pid = parse_int(record.get("pid"))
    fixed_weight = parse_bool(record.get("fixed_weight"))
    is_moe_expert = parse_bool(record.get("is_moe_expert"))
    protected = parse_bool(record.get("protected"))
    evictable = parse_bool(record.get("evictable"))
    driver_capable = record.get("driver_capable")
    return WeightRegion(
        index=index,
        name=str(record.get("name") or "unknown"),
        kind=str(record.get("kind") or "unknown"),
        start=start,
        end=end,
        size_bytes=size_bytes,
        accounted_size_bytes=parse_int(record.get("accounted_size_bytes")) or size_bytes,
        semantic_type=str(record.get("semantic_type") or record.get("weight_class") or "other"),
        role=str(record.get("role") or "other"),
        fixed_weight=fixed_weight,
        is_moe_expert=is_moe_expert,
        protected=protected,
        evictable=evictable,
        placement_policy=str(record.get("placement_policy") or "unknown"),
        layer_id=layer_id,
        expert_id=expert_id,
        dtype=str(record.get("dtype") or "unknown"),
        is_alias=parse_bool(record.get("is_alias")),
        alias_of=str(record.get("alias_of")) if record.get("alias_of") is not None else None,
        is_logical_slice=parse_bool(
            record.get("is_logical_slice") or record.get("is_logical_expert_slice")
        ),
        logical_parent_name=(
            str(record.get("logical_parent_name"))
            if record.get("logical_parent_name") is not None
            else None
        ),
        logical_slice_index=parse_int(record.get("logical_slice_index")),
        logical_slice_count=parse_int(record.get("logical_slice_count")),
        driver_capable=parse_bool(driver_capable) if driver_capable is not None else None,
        driver_capability_reason=(
            str(record.get("driver_capability_reason"))
            if record.get("driver_capability_reason") is not None
            else None
        ),
        is_main_weight=parse_bool(record.get("is_main_weight")),
        is_moe_expert_main_weight=parse_bool(record.get("is_moe_expert_main_weight")),
        residency_query_enabled=parse_bool(record.get("residency_query_enabled")),
        residency_query_ok=(
            parse_bool(record.get("residency_query_ok"))
            if record.get("residency_query_ok") is not None
            else None
        ),
        residency_state=(
            str(record.get("residency_state"))
            if record.get("residency_state") is not None
            else None
        ),
        residency_query_granularity=(
            str(record.get("residency_query_granularity"))
            if record.get("residency_query_granularity") is not None
            else None
        ),
        residency_query_start=parse_int(record.get("residency_query_start_int")),
        residency_query_size_bytes=parse_int(
            record.get("residency_query_size_bytes")
        ) or 0,
        residency_queried_bytes=parse_int(record.get("residency_queried_bytes")) or 0,
        resident_gpu_bytes=parse_int(record.get("resident_gpu_bytes")) or 0,
        resident_cpu_bytes=parse_int(record.get("resident_cpu_bytes")) or 0,
        resident_any_bytes=parse_int(record.get("resident_any_bytes")) or 0,
        resident_unknown_bytes=parse_int(record.get("resident_unknown_bytes")) or 0,
        resident_gpu_ratio=parse_float(record.get("resident_gpu_ratio")),
        phase=str(record.get("phase")) if record.get("phase") is not None else None,
        pid=pid,
    )


def load_weight_regions(path: Path) -> list[WeightRegion]:
    regions: list[WeightRegion] = []
    for index, record in enumerate(read_jsonl(path)):
        region = weight_region_from_record(index, record)
        if region is not None:
            regions.append(region)
    return regions


def parse_fault_addresses(path: Path | None) -> tuple[list[int], int, int]:
    if path is None or not path.is_file():
        return [], 0, 0
    addresses: list[int] = []
    ignored = 0
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            total += 1
            page_match = PAGE_ADDR_RE.search(line)
            raw_match = RAW_ADDR_RE.search(line)
            match = page_match or raw_match
            if match is None:
                ignored += 1
                continue
            addresses.append(int(match.group(1), 16))
    return addresses, ignored, total


def parse_stats_delta(path: Path | None) -> dict[str, int | float | None]:
    if path is None or not path.is_file():
        return {
            "parsed_stat_records": 0,
            "delta_after_dedup": None,
            "delta_kv_after_dedup": None,
            "delta_kv_ratio_pct": None,
        }

    records: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = {key: int(value) for key, value in STAT_INT_RE.findall(line)}
            if fields:
                records.append(fields)
    if not records:
        return {
            "parsed_stat_records": 0,
            "delta_after_dedup": None,
            "delta_kv_after_dedup": None,
            "delta_kv_ratio_pct": None,
        }

    first = records[0]
    last = records[-1]
    total_keys = ("total_after_dedup", "after_dedup", "total_replayable_after_dedup")
    kv_keys = ("total_kv_after_dedup", "kv_after_dedup")
    total_key = next((key for key in total_keys if key in last), None)
    kv_key = next((key for key in kv_keys if key in last), None)
    total_delta = None
    kv_delta = None
    ratio = None
    if total_key is not None:
        total_delta = int(last.get(total_key, 0)) - int(first.get(total_key, 0))
    if kv_key is not None:
        kv_delta = int(last.get(kv_key, 0)) - int(first.get(kv_key, 0))
    if total_delta and total_delta > 0 and kv_delta is not None:
        ratio = 100.0 * float(kv_delta) / float(total_delta)
    return {
        "parsed_stat_records": len(records),
        "delta_after_dedup": total_delta,
        "delta_kv_after_dedup": kv_delta,
        "delta_kv_ratio_pct": ratio,
    }


def summarize_bytes(regions: list[WeightRegion]) -> dict[str, Any]:
    primary = [
        region for region in regions
        if not region.is_alias and not region.is_logical_slice
    ]
    logical_slices = [region for region in regions if region.is_logical_slice]
    named_aliases = [
        region for region in regions
        if region.is_alias and not region.is_logical_slice
    ]
    counters: dict[str, Counter[str]] = {
        "semantic_type": Counter(),
        "role": Counter(),
        "placement_policy": Counter(),
        "dtype": Counter(),
        "driver_capability_reason": Counter(),
        "residency_state": Counter(),
    }
    bytes_by_semantic: Counter[str] = Counter()
    bytes_by_role: Counter[str] = Counter()
    bytes_by_policy: Counter[str] = Counter()
    bytes_by_driver: Counter[str] = Counter()
    bytes_by_residency_state: Counter[str] = Counter()
    for region in primary:
        counters["semantic_type"][region.semantic_type] += 1
        counters["role"][region.role] += 1
        counters["placement_policy"][region.placement_policy] += 1
        counters["dtype"][region.dtype] += 1
        driver_reason = region.driver_capability_reason or "not_recorded"
        residency_state = region.residency_state or "not_recorded"
        counters["driver_capability_reason"][driver_reason] += 1
        counters["residency_state"][residency_state] += 1
        bytes_by_semantic[region.semantic_type] += region.accounted_size_bytes
        bytes_by_role[region.role] += region.accounted_size_bytes
        bytes_by_policy[region.placement_policy] += region.accounted_size_bytes
        bytes_by_driver[driver_reason] += region.accounted_size_bytes
        bytes_by_residency_state[residency_state] += region.accounted_size_bytes

    fixed_bytes = sum(region.accounted_size_bytes for region in primary if region.fixed_weight)
    expert_bytes = sum(region.accounted_size_bytes for region in primary if region.is_moe_expert)
    expert_main_bytes = sum(
        region.accounted_size_bytes
        for region in primary
        if region.is_moe_expert_main_weight
    )
    main_weight_bytes = sum(
        region.accounted_size_bytes for region in primary if region.is_main_weight
    )
    managed_bytes = sum(
        region.accounted_size_bytes for region in primary if region.driver_capable is True
    )
    unmanaged_bytes = sum(
        region.accounted_size_bytes for region in primary if region.driver_capable is False
    )
    residency_query_records = sum(
        1 for region in primary if region.residency_query_enabled
    )
    residency_query_ok_records = sum(
        1 for region in primary if region.residency_query_ok is True
    )
    residency_queried_bytes = sum(region.residency_queried_bytes for region in primary)
    resident_gpu_bytes = sum(region.resident_gpu_bytes for region in primary)
    resident_cpu_bytes = sum(region.resident_cpu_bytes for region in primary)
    resident_unknown_bytes = sum(region.resident_unknown_bytes for region in primary)
    residency_gpu_ratio = (
        float(resident_gpu_bytes) / float(residency_queried_bytes)
        if residency_queried_bytes > 0
        else None
    )
    logical_slice_query_records = sum(
        1 for region in logical_slices if region.residency_query_enabled
    )
    logical_slice_query_ok_records = sum(
        1 for region in logical_slices if region.residency_query_ok is True
    )
    logical_slice_queried_bytes = sum(
        region.residency_queried_bytes for region in logical_slices
    )
    logical_slice_gpu_bytes = sum(region.resident_gpu_bytes for region in logical_slices)
    logical_slice_gpu_ratio = (
        float(logical_slice_gpu_bytes) / float(logical_slice_queried_bytes)
        if logical_slice_queried_bytes > 0
        else None
    )
    logical_slice_parents = {
        region.logical_parent_name
        for region in logical_slices
        if region.logical_parent_name
    }

    return {
        "records": len(regions),
        "primary_records": len(primary),
        "alias_records": len(named_aliases),
        "logical_slice_records": len(logical_slices),
        "logical_slice_parent_records": len(logical_slice_parents),
        "total_accounted_bytes": sum(region.accounted_size_bytes for region in primary),
        "fixed_weight_bytes": fixed_bytes,
        "moe_expert_bytes": expert_bytes,
        "moe_expert_main_weight_bytes": expert_main_bytes,
        "main_weight_bytes": main_weight_bytes,
        "managed_bytes": managed_bytes,
        "unmanaged_or_nonmanaged_bytes": unmanaged_bytes,
        "residency_query_records": residency_query_records,
        "residency_query_ok_records": residency_query_ok_records,
        "residency_queried_bytes": residency_queried_bytes,
        "resident_gpu_bytes": resident_gpu_bytes,
        "resident_cpu_bytes": resident_cpu_bytes,
        "resident_unknown_bytes": resident_unknown_bytes,
        "residency_gpu_ratio": residency_gpu_ratio,
        "logical_slice_query_records": logical_slice_query_records,
        "logical_slice_query_ok_records": logical_slice_query_ok_records,
        "logical_slice_queried_bytes": logical_slice_queried_bytes,
        "logical_slice_resident_gpu_bytes": logical_slice_gpu_bytes,
        "logical_slice_residency_gpu_ratio": logical_slice_gpu_ratio,
        "semantic_counts": dict(counters["semantic_type"]),
        "role_counts": dict(counters["role"]),
        "placement_policy_counts": dict(counters["placement_policy"]),
        "dtype_counts": dict(counters["dtype"]),
        "driver_capability_reason_counts": dict(counters["driver_capability_reason"]),
        "residency_state_counts": dict(counters["residency_state"]),
        "bytes_by_semantic_type": dict(bytes_by_semantic),
        "bytes_by_role": dict(bytes_by_role),
        "bytes_by_placement_policy": dict(bytes_by_policy),
        "bytes_by_driver_capability_reason": dict(bytes_by_driver),
        "bytes_by_residency_state": dict(bytes_by_residency_state),
    }


def overlap_summary(regions: list[WeightRegion]) -> dict[str, Any]:
    primary = sorted(
        [
            region for region in regions
            if not region.is_alias and not region.is_logical_slice
        ],
        key=lambda item: (item.start, item.end),
    )
    overlaps = 0
    overlap_examples: list[dict[str, Any]] = []
    previous: WeightRegion | None = None
    for region in primary:
        if previous is not None and region.start <= previous.end:
            overlaps += 1
            if len(overlap_examples) < 10:
                overlap_examples.append(
                    {
                        "previous": previous.name,
                        "previous_start": previous.start_hex,
                        "previous_end": previous.end_hex,
                        "current": region.name,
                        "current_start": region.start_hex,
                        "current_end": region.end_hex,
                    }
                )
            if region.end > previous.end:
                previous = region
        else:
            previous = region
    return {
        "overlap_count": overlaps,
        "overlap_examples": overlap_examples,
    }


def classify_faults(
    addresses: list[int],
    regions: list[WeightRegion],
    dedup_mode: str,
    top_n: int,
) -> dict[str, Any]:
    duplicates_skipped = 0
    selected: list[int] = []
    seen: set[int] = set()
    for address in addresses:
        if dedup_mode == "global-page":
            if address in seen:
                duplicates_skipped += 1
                continue
            seen.add(address)
        selected.append(address)

    lookup = RegionLookup([
        region for region in regions
        if not region.is_alias and not region.is_logical_slice
    ])
    by_semantic: Counter[str] = Counter()
    by_role: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    by_fixed: Counter[str] = Counter()
    by_name: Counter[str] = Counter()
    unknown = 0
    matched = 0
    for address in selected:
        region = lookup.find(address)
        if region is None:
            unknown += 1
            continue
        matched += 1
        by_semantic[region.semantic_type] += 1
        by_role[region.role] += 1
        by_policy[region.placement_policy] += 1
        by_fixed["fixed_weight" if region.fixed_weight else "expert_or_evictable"] += 1
        by_name[region.name] += 1

    total = len(selected)
    def pct(value: int) -> float:
        return 100.0 * float(value) / float(total) if total else 0.0

    return {
        "raw_fault_records": len(addresses),
        "classified_fault_records": total,
        "duplicates_skipped": duplicates_skipped,
        "matched_weight_faults": matched,
        "unknown_faults": unknown,
        "matched_weight_pct": pct(matched),
        "unknown_pct": pct(unknown),
        "faults_by_semantic_type": dict(by_semantic),
        "faults_by_role": dict(by_role),
        "faults_by_placement_policy": dict(by_policy),
        "faults_by_fixed_or_expert": dict(by_fixed),
        "top_weight_fault_regions": by_name.most_common(top_n),
    }


def write_counter_csv(path: Path, rows: dict[str, int], total: int, value_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", value_name, "pct"])
        for key, value in sorted(rows.items(), key=lambda item: (-item[1], item[0])):
            pct = 100.0 * float(value) / float(total) if total else 0.0
            writer.writerow([key, value, f"{pct:.6f}"])


def write_region_csv(path: Path, regions: list[WeightRegion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name",
            "kind",
            "start",
            "end",
            "size_bytes",
            "accounted_size_bytes",
            "semantic_type",
            "role",
            "fixed_weight",
            "is_moe_expert",
            "protected",
            "evictable",
            "placement_policy",
            "layer_id",
            "expert_id",
            "dtype",
            "is_alias",
            "alias_of",
            "is_logical_slice",
            "logical_parent_name",
            "logical_slice_index",
            "logical_slice_count",
            "driver_capable",
            "driver_capability_reason",
            "is_main_weight",
            "is_moe_expert_main_weight",
            "residency_state",
            "residency_query_granularity",
            "residency_query_start",
            "residency_query_size_bytes",
            "residency_queried_bytes",
            "resident_gpu_bytes",
            "resident_cpu_bytes",
            "resident_unknown_bytes",
            "resident_gpu_ratio",
            "phase",
            "pid",
        ])
        for region in sorted(regions, key=lambda item: (item.start, item.end, item.name)):
            writer.writerow([
                region.name,
                region.kind,
                region.start_hex,
                region.end_hex,
                region.size_bytes,
                region.accounted_size_bytes,
                region.semantic_type,
                region.role,
                int(region.fixed_weight),
                int(region.is_moe_expert),
                int(region.protected),
                int(region.evictable),
                region.placement_policy,
                region.layer_id if region.layer_id is not None else "",
                region.expert_id if region.expert_id is not None else "",
                region.dtype,
                int(region.is_alias),
                region.alias_of or "",
                int(region.is_logical_slice),
                region.logical_parent_name or "",
                "" if region.logical_slice_index is None else region.logical_slice_index,
                "" if region.logical_slice_count is None else region.logical_slice_count,
                "" if region.driver_capable is None else int(region.driver_capable),
                region.driver_capability_reason or "",
                int(region.is_main_weight),
                int(region.is_moe_expert_main_weight),
                region.residency_state or "",
                region.residency_query_granularity or "",
                "" if region.residency_query_start is None else f"0x{region.residency_query_start:x}",
                region.residency_query_size_bytes,
                region.residency_queried_bytes,
                region.resident_gpu_bytes,
                region.resident_cpu_bytes,
                region.resident_unknown_bytes,
                "" if region.resident_gpu_ratio is None else f"{region.resident_gpu_ratio:.9f}",
                region.phase or "",
                region.pid if region.pid is not None else "",
            ])


def fmt_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 3):.3f} GiB"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    bytes_summary = summary["bytes"]
    fault_summary = summary["faults"]
    stats_summary = summary["stats_delta"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vLLM Weight Address Map Report\n\n")
        handle.write("## Coverage\n\n")
        handle.write(f"- records: {bytes_summary['records']}\n")
        handle.write(f"- primary_records: {bytes_summary['primary_records']}\n")
        handle.write(f"- alias_records: {bytes_summary['alias_records']}\n")
        handle.write(
            f"- logical_slice_records: {bytes_summary['logical_slice_records']}\n"
        )
        handle.write(
            "- logical_slice_parent_records: "
            f"{bytes_summary['logical_slice_parent_records']}\n"
        )
        handle.write(
            f"- total_accounted_bytes: {bytes_summary['total_accounted_bytes']} "
            f"({fmt_bytes(bytes_summary['total_accounted_bytes'])})\n"
        )
        handle.write(
            f"- fixed_weight_bytes: {bytes_summary['fixed_weight_bytes']} "
            f"({fmt_bytes(bytes_summary['fixed_weight_bytes'])})\n"
        )
        handle.write(
            f"- moe_expert_bytes: {bytes_summary['moe_expert_bytes']} "
            f"({fmt_bytes(bytes_summary['moe_expert_bytes'])})\n"
        )
        handle.write(
            "- moe_expert_main_weight_bytes: "
            f"{bytes_summary['moe_expert_main_weight_bytes']} "
            f"({fmt_bytes(bytes_summary['moe_expert_main_weight_bytes'])})\n"
        )
        handle.write(
            f"- main_weight_bytes: {bytes_summary['main_weight_bytes']} "
            f"({fmt_bytes(bytes_summary['main_weight_bytes'])})\n"
        )
        handle.write(
            f"- managed_bytes: {bytes_summary['managed_bytes']} "
            f"({fmt_bytes(bytes_summary['managed_bytes'])})\n"
        )
        handle.write(
            "- unmanaged_or_nonmanaged_bytes: "
            f"{bytes_summary['unmanaged_or_nonmanaged_bytes']} "
            f"({fmt_bytes(bytes_summary['unmanaged_or_nonmanaged_bytes'])})\n\n"
        )
        handle.write("## Residency Query\n\n")
        handle.write(f"- residency_query_records: {bytes_summary['residency_query_records']}\n")
        handle.write(
            f"- residency_query_ok_records: {bytes_summary['residency_query_ok_records']}\n"
        )
        handle.write(
            f"- residency_queried_bytes: {bytes_summary['residency_queried_bytes']} "
            f"({fmt_bytes(bytes_summary['residency_queried_bytes'])})\n"
        )
        handle.write(
            f"- resident_gpu_bytes: {bytes_summary['resident_gpu_bytes']} "
            f"({fmt_bytes(bytes_summary['resident_gpu_bytes'])})\n"
        )
        handle.write(
            f"- resident_cpu_bytes: {bytes_summary['resident_cpu_bytes']} "
            f"({fmt_bytes(bytes_summary['resident_cpu_bytes'])})\n"
        )
        handle.write(
            f"- resident_unknown_bytes: {bytes_summary['resident_unknown_bytes']} "
            f"({fmt_bytes(bytes_summary['resident_unknown_bytes'])})\n"
        )
        handle.write(
            f"- residency_gpu_ratio: {bytes_summary['residency_gpu_ratio']}\n\n"
        )
        handle.write(
            "- logical_slice_query_records: "
            f"{bytes_summary['logical_slice_query_records']}\n"
        )
        handle.write(
            "- logical_slice_query_ok_records: "
            f"{bytes_summary['logical_slice_query_ok_records']}\n"
        )
        handle.write(
            "- logical_slice_queried_bytes: "
            f"{bytes_summary['logical_slice_queried_bytes']} "
            f"({fmt_bytes(bytes_summary['logical_slice_queried_bytes'])})\n"
        )
        handle.write(
            "- logical_slice_resident_gpu_bytes: "
            f"{bytes_summary['logical_slice_resident_gpu_bytes']} "
            f"({fmt_bytes(bytes_summary['logical_slice_resident_gpu_bytes'])})\n"
        )
        handle.write(
            "- logical_slice_residency_gpu_ratio: "
            f"{bytes_summary['logical_slice_residency_gpu_ratio']}\n\n"
        )
        handle.write("| residency_state | bytes | GiB |\n")
        handle.write("|---|---:|---:|\n")
        for key, value in sorted(
            bytes_summary["bytes_by_residency_state"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            handle.write(f"| {key} | {value} | {value / (1024 ** 3):.3f} |\n")
        handle.write("\n")

        handle.write("## Bytes By Semantic Type\n\n")
        handle.write("| semantic_type | bytes | GiB |\n")
        handle.write("|---|---:|---:|\n")
        for key, value in sorted(
            bytes_summary["bytes_by_semantic_type"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            handle.write(f"| {key} | {value} | {value / (1024 ** 3):.3f} |\n")
        handle.write("\n")

        handle.write("## Fault Attribution\n\n")
        if fault_summary["raw_fault_records"] <= 0:
            handle.write("- no fault log provided, or no fault addresses parsed.\n\n")
        else:
            handle.write(f"- raw_fault_records: {fault_summary['raw_fault_records']}\n")
            handle.write(
                f"- classified_fault_records: {fault_summary['classified_fault_records']}\n"
            )
            handle.write(f"- duplicates_skipped: {fault_summary['duplicates_skipped']}\n")
            handle.write(
                "- matched_weight_faults: "
                f"{fault_summary['matched_weight_faults']} "
                f"({fault_summary['matched_weight_pct']:.3f}%)\n"
            )
            handle.write(
                f"- unknown_faults: {fault_summary['unknown_faults']} "
                f"({fault_summary['unknown_pct']:.3f}%)\n\n"
            )
            handle.write("| semantic_type | faults |\n")
            handle.write("|---|---:|\n")
            for key, value in sorted(
                fault_summary["faults_by_semantic_type"].items(),
                key=lambda item: (-item[1], item[0]),
            ):
                handle.write(f"| {key} | {value} |\n")
            handle.write("\n")

        handle.write("## UVM Stats Delta\n\n")
        handle.write(f"- parsed_stat_records: {stats_summary['parsed_stat_records']}\n")
        handle.write(f"- delta_after_dedup: {stats_summary['delta_after_dedup']}\n")
        handle.write(f"- delta_kv_after_dedup: {stats_summary['delta_kv_after_dedup']}\n")
        handle.write(f"- delta_kv_ratio_pct: {stats_summary['delta_kv_ratio_pct']}\n\n")

        handle.write("## Top Faulted Weight Regions\n\n")
        handle.write("| region | faults |\n")
        handle.write("|---|---:|\n")
        for name, count in fault_summary["top_weight_fault_regions"]:
            handle.write(f"| `{name}` | {count} |\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = load_weight_regions(Path(args.weight_map))
    addresses, ignored_fault_lines, total_fault_lines = parse_fault_addresses(
        Path(args.fault_log) if args.fault_log else None
    )
    byte_summary = summarize_bytes(regions)
    fault_summary = classify_faults(addresses, regions, args.dedup_mode, args.top_n)
    fault_summary["ignored_fault_lines"] = ignored_fault_lines
    fault_summary["total_fault_lines"] = total_fault_lines
    summary = {
        "weight_map": args.weight_map,
        "fault_log": args.fault_log,
        "stats_log": args.stats_log,
        "dedup_mode": args.dedup_mode,
        "bytes": byte_summary,
        "overlaps": overlap_summary(regions),
        "faults": fault_summary,
        "stats_delta": parse_stats_delta(Path(args.stats_log) if args.stats_log else None),
    }

    summary_json = out_dir / "weight_address_map_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    write_region_csv(out_dir / "weight_address_regions.csv", regions)
    write_counter_csv(
        out_dir / "weight_bytes_by_semantic_type.csv",
        byte_summary["bytes_by_semantic_type"],
        byte_summary["total_accounted_bytes"],
        "bytes",
    )
    write_counter_csv(
        out_dir / "weight_faults_by_semantic_type.csv",
        fault_summary["faults_by_semantic_type"],
        fault_summary["classified_fault_records"],
        "faults",
    )
    write_report(out_dir / "weight_address_map_report.md", summary)

    print("vLLM Weight Address Map Analysis")
    print(f"- weight_map={args.weight_map}")
    print(f"- records={byte_summary['records']} primary={byte_summary['primary_records']}")
    print(f"- total_accounted={fmt_bytes(byte_summary['total_accounted_bytes'])}")
    print(f"- fixed_weight={fmt_bytes(byte_summary['fixed_weight_bytes'])}")
    print(f"- moe_expert={fmt_bytes(byte_summary['moe_expert_bytes'])}")
    print(f"- matched_weight_faults={fault_summary['matched_weight_faults']}")
    print(f"- report={out_dir / 'weight_address_map_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
