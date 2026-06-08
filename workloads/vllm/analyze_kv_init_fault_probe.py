#!/usr/bin/env python3
"""Analyze KV cache init markers against UVM replayable fault addresses."""

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
ADDRESS_RE = re.compile(r"\baddress=(0x[0-9a-fA-F]+)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \t\n]+)")
ADDRESS_SECTION_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+phase=(?P<phase>\S+)\s+pid=(?P<pid>\d+)\s+model=(?P<model>.+)$"
)
MARKER_PREFIX = "VLLM_KV_INIT_PROBE"


@dataclass(frozen=True)
class Region:
    kind: str
    source: str
    name: str
    start: int
    end: int
    size_bytes: int


class RegionLookup:
    def __init__(self, regions: Iterable[Region]):
        self.regions = sorted(regions, key=lambda region: (region.start, region.end))
        self.starts = [region.start for region in self.regions]

    def find(self, address: int) -> Region | None:
        index = bisect.bisect_right(self.starts, address) - 1
        if index < 0:
            return None
        region = self.regions[index]
        if address <= region.end:
            return region
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify KV init replayable faults using probe markers."
    )
    parser.add_argument("--allocator-log", default="")
    parser.add_argument("--address-log", default="")
    parser.add_argument("--fault-log", required=True)
    parser.add_argument("--kv-init-probe-log", default="")
    parser.add_argument("--route-log", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--faults-csv", default="")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def parse_int(value: str, base: int = 10) -> int | None:
    try:
        return int(value, base)
    except Exception:
        return None


def parse_kv_pairs(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in KEY_VALUE_RE.finditer(line)}


def add_region(
    regions: list[Region],
    *,
    kind: str,
    source: str,
    name: str,
    start: int,
    size_bytes: int,
) -> None:
    if start == 0 or size_bytes <= 0:
        return
    regions.append(
        Region(
            kind=kind,
            source=source,
            name=name,
            start=start,
            end=start + size_bytes - 1,
            size_bytes=size_bytes,
        )
    )


def parse_allocator_regions(path: str) -> list[Region]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []

    regions: list[Region] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "TRACE_DRIVER_KV_TAG" not in line and "TRACE_DRIVER_SCRATCH_TAG" not in line:
                continue

            fields = parse_kv_pairs(line)
            ptr_text = fields.get("ptr", "0x0")
            size_text = fields.get("size_bytes", "0")
            registered = fields.get("registered", "1")
            start = parse_int(ptr_text, 16) or 0
            size_bytes = parse_int(size_text) or 0
            alloc_id = fields.get("alloc_id", "?")

            if registered == "0":
                continue
            if "TRACE_DRIVER_KV_TAG" in line:
                add_region(
                    regions,
                    kind="kv",
                    source="allocator_kv_tag",
                    name=f"alloc_id={alloc_id}",
                    start=start,
                    size_bytes=size_bytes,
                )
            else:
                add_region(
                    regions,
                    kind="scratch",
                    source="allocator_scratch_tag",
                    name=f"alloc_id={alloc_id}",
                    start=start,
                    size_bytes=size_bytes,
                )
    return regions


def address_log_kind_to_region_kind(kind: str) -> str:
    if kind.startswith("weight:"):
        return "weight"
    if kind.startswith("kv_cache"):
        return "kv"
    if kind.startswith("scratch") or kind.startswith("temp"):
        return "scratch"
    return "other"


def parse_address_regions(path: str) -> list[Region]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []

    regions: list[Region] = []
    current_section = "section=none"
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            section_match = ADDRESS_SECTION_RE.match(stripped)
            if section_match:
                current_section = (
                    f"pid={section_match.group('pid')},phase={section_match.group('phase')}"
                )
                continue
            if stripped == "kind,name,start,end,size_bytes,size_mb":
                continue

            parts = next(csv.reader([stripped]))
            if len(parts) != 6:
                continue
            kind, name, start_text, end_text, size_text, _size_mb = parts
            try:
                start = int(start_text, 16)
                end = int(end_text, 16)
                size_bytes = int(size_text)
            except Exception:
                continue
            if end < start:
                continue
            regions.append(
                Region(
                    kind=address_log_kind_to_region_kind(kind),
                    source="address_log",
                    name=f"{current_section},{kind},{name}",
                    start=start,
                    end=end,
                    size_bytes=size_bytes,
                )
            )
    return regions


def parse_probe_jsonl(path: str) -> tuple[list[dict[str, Any]], list[Region]]:
    if not path:
        return [], []
    file_path = Path(path)
    if not file_path.exists():
        return [], []

    records: list[dict[str, Any]] = []
    regions: list[Region] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            records.append(record)
            if record.get("event") not in {"kv_zeros_return", "kv_tensor_ready"}:
                continue
            ptr = int(record.get("ptr") or 0)
            size_bytes = int(record.get("size_bytes") or 0)
            name = f"path={record.get('path')},index={record.get('index')},event={record.get('event')}"
            add_region(
                regions,
                kind="kv",
                source="kv_init_probe_jsonl",
                name=name,
                start=ptr,
                size_bytes=size_bytes,
            )
    return records, regions


def build_lookups(regions: list[Region]) -> dict[str, RegionLookup]:
    by_kind: dict[str, list[Region]] = defaultdict(list)
    for region in regions:
        by_kind[region.kind].append(region)
    return {kind: RegionLookup(kind_regions) for kind, kind_regions in by_kind.items()}


def classify_address(address: int, lookups: dict[str, RegionLookup]) -> tuple[str, Region | None]:
    for kind in ("kv", "weight", "scratch", "other"):
        lookup = lookups.get(kind)
        if lookup is None:
            continue
        region = lookup.find(address)
        if region is not None:
            return kind, region
    return "unknown", None


def empty_counter() -> Counter[str]:
    return Counter(
        {
            "faults": 0,
            "unique_pages": 0,
            "kv": 0,
            "weight": 0,
            "scratch": 0,
            "other": 0,
            "unknown": 0,
        }
    )


def update_counter(counter: Counter[str], kind: str, page: int | None) -> None:
    counter["faults"] += 1
    counter[kind] += 1
    if page is not None:
        counter[f"page:{kind}:{page:x}"] += 1


def finalize_counter(counter: Counter[str]) -> dict[str, int]:
    result = {
        key: int(value)
        for key, value in counter.items()
        if not key.startswith("page:")
    }
    unique_by_kind = Counter()
    for key in counter:
        if not key.startswith("page:"):
            continue
        _prefix, kind, _page = key.split(":", 2)
        unique_by_kind[kind] += 1
    for kind in ("kv", "weight", "scratch", "other", "unknown"):
        result[f"unique_pages_{kind}"] = int(unique_by_kind[kind])
    result["unique_pages"] = int(sum(unique_by_kind.values()))
    return result


def marker_key(fields: dict[str, str]) -> str:
    return f"{fields.get('path', '?')}:{fields.get('index', '?')}"


def parse_marker(line: str) -> dict[str, str] | None:
    if MARKER_PREFIX not in line:
        return None
    marker_text = line.split(MARKER_PREFIX, 1)[1]
    fields = parse_kv_pairs(marker_text)
    fields["marker"] = MARKER_PREFIX
    return fields


def parse_fault_address(line: str) -> tuple[int | None, int | None]:
    raw_match = RAW_ADDR_RE.search(line)
    page_match = PAGE_ADDR_RE.search(line)
    raw = int(raw_match.group(1), 16) if raw_match else None
    page = int(page_match.group(1), 16) if page_match else None
    return raw, page


def analyze_fault_log(
    path: str,
    lookups: dict[str, RegionLookup],
    *,
    faults_csv: str = "",
    top_n: int = 20,
) -> dict[str, Any]:
    file_path = Path(path)
    total = empty_counter()
    active_windows: dict[str, dict[str, Any]] = {}
    completed_windows: list[dict[str, Any]] = []
    markers = Counter()
    top_regions = Counter()

    csv_handle = None
    csv_writer = None
    if faults_csv:
        csv_path = Path(faults_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = csv_path.open("w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(
            [
                "line_no",
                "raw",
                "page",
                "class",
                "region_source",
                "region_name",
                "region_start",
                "region_end",
            ]
        )

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                marker = parse_marker(line)
                if marker is not None:
                    event = marker.get("event", "unknown")
                    key = marker_key(marker)
                    markers[event] += 1
                    if event == "kv_zeros_begin":
                        active_windows[key] = {
                            "key": key,
                            "path": marker.get("path", "?"),
                            "index": marker.get("index", "?"),
                            "begin_line": line_no,
                            "begin_marker": marker,
                            "counts": empty_counter(),
                            "end_snapshots": [],
                        }
                    elif key in active_windows and event in {
                        "kv_zeros_return",
                        "kv_zeros_sync_end",
                        "kv_tensor_ready",
                    }:
                        window = active_windows[key]
                        window["end_snapshots"].append(
                            {
                                "event": event,
                                "line_no": line_no,
                                "counts": finalize_counter(window["counts"]),
                                "marker": marker,
                            }
                        )
                        if event == "kv_tensor_ready":
                            completed_windows.append(window)
                            del active_windows[key]
                    continue

                raw, page = parse_fault_address(line)
                selected = page if page is not None else raw
                if selected is None:
                    continue

                kind, region = classify_address(selected, lookups)
                update_counter(total, kind, page)
                for window in active_windows.values():
                    update_counter(window["counts"], kind, page)

                if region is not None:
                    top_regions[(kind, region.source, region.name)] += 1
                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            line_no,
                            f"0x{raw:x}" if raw is not None else "",
                            f"0x{page:x}" if page is not None else "",
                            kind,
                            region.source if region is not None else "",
                            region.name if region is not None else "",
                            f"0x{region.start:x}" if region is not None else "",
                            f"0x{region.end:x}" if region is not None else "",
                        ]
                    )
    finally:
        if csv_handle is not None:
            csv_handle.close()

    for window in active_windows.values():
        window["end_snapshots"].append(
            {
                "event": "unterminated",
                "line_no": None,
                "counts": finalize_counter(window["counts"]),
                "marker": {},
            }
        )
        completed_windows.append(window)

    simplified_windows = []
    for window in completed_windows:
        simplified_windows.append(
            {
                "key": window["key"],
                "path": window["path"],
                "index": window["index"],
                "begin_line": window["begin_line"],
                "end_snapshots": window["end_snapshots"],
            }
        )

    return {
        "fault_log": str(file_path),
        "totals": finalize_counter(total),
        "markers": {key: int(value) for key, value in markers.items()},
        "windows": simplified_windows,
        "top_regions": [
            {
                "class": key[0],
                "source": key[1],
                "name": key[2],
                "faults": int(value),
            }
            for key, value in top_regions.most_common(top_n)
        ],
    }


def analyze_route_log(path: str, lookups: dict[str, RegionLookup]) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}

    counters = empty_counter()
    line_counters = Counter()
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "phase_n_block_" in line:
                if "phase_n_block_materialize" in line:
                    line_counters["phase_n_block_materialize"] += 1
                if "phase_n_block_summary" in line:
                    line_counters["phase_n_block_summary"] += 1
                if "skip=no_tagged_region" in line:
                    line_counters["no_tagged_region"] += 1
            match = ADDRESS_RE.search(line)
            if not match:
                continue
            address = int(match.group(1), 16)
            kind, _region = classify_address(address, lookups)
            update_counter(counters, kind, address & ~0xFFF)

    return {
        "route_log": str(file_path),
        "address_totals": finalize_counter(counters),
        "line_counters": {key: int(value) for key, value in line_counters.items()},
    }


def summarize_regions(regions: list[Region]) -> dict[str, Any]:
    by_kind = Counter()
    bytes_by_kind = Counter()
    by_source = Counter()
    for region in regions:
        by_kind[region.kind] += 1
        bytes_by_kind[region.kind] += region.size_bytes
        by_source[region.source] += 1
    return {
        "total_regions": len(regions),
        "regions_by_kind": {key: int(value) for key, value in by_kind.items()},
        "bytes_by_kind": {key: int(value) for key, value in bytes_by_kind.items()},
        "regions_by_source": {key: int(value) for key, value in by_source.items()},
    }


def write_text_summary(summary: dict[str, Any], path: str) -> None:
    lines: list[str] = []
    lines.append("KV Init Fault Probe Summary")
    lines.append("===========================")
    lines.append(f"fault_log={summary['fault_analysis']['fault_log']}")
    lines.append(f"allocator_log={summary.get('allocator_log') or 'none'}")
    lines.append(f"address_log={summary.get('address_log') or 'none'}")
    lines.append(f"kv_init_probe_log={summary.get('kv_init_probe_log') or 'none'}")
    lines.append(f"route_log={summary.get('route_log') or 'none'}")
    lines.append("")

    lines.append("Regions")
    lines.append("-------")
    region_summary = summary["regions"]
    for key, value in region_summary.items():
        lines.append(f"{key}={json.dumps(value, sort_keys=True)}")
    lines.append("")

    lines.append("Whole-Run Fault Classification")
    lines.append("------------------------------")
    for key, value in summary["fault_analysis"]["totals"].items():
        lines.append(f"{key}={value}")
    lines.append("")

    lines.append("KV Init Markers")
    lines.append("---------------")
    markers = summary["fault_analysis"]["markers"]
    if markers:
        for key in sorted(markers):
            lines.append(f"{key}={markers[key]}")
    else:
        lines.append("no_marker_records=1")
    lines.append("")

    lines.append("KV Init Windows")
    lines.append("---------------")
    windows = summary["fault_analysis"]["windows"]
    if not windows:
        lines.append("no_completed_windows=1")
    for window in windows:
        lines.append(
            f"window key={window['key']} path={window['path']} "
            f"index={window['index']} begin_line={window['begin_line']}"
        )
        for snapshot in window["end_snapshots"]:
            counts = snapshot["counts"]
            lines.append(
                f"  end_event={snapshot['event']} line={snapshot['line_no']} "
                f"faults={counts.get('faults', 0)} kv={counts.get('kv', 0)} "
                f"weight={counts.get('weight', 0)} scratch={counts.get('scratch', 0)} "
                f"other={counts.get('other', 0)} unknown={counts.get('unknown', 0)} "
                f"unique_pages_kv={counts.get('unique_pages_kv', 0)}"
            )
    lines.append("")

    route = summary.get("route_analysis") or {}
    if route:
        lines.append("Route Log")
        lines.append("---------")
        for key, value in route.get("line_counters", {}).items():
            lines.append(f"{key}={value}")
        for key, value in route.get("address_totals", {}).items():
            lines.append(f"route_{key}={value}")
        lines.append("")

    lines.append("Top Matched Regions")
    lines.append("-------------------")
    for item in summary["fault_analysis"]["top_regions"]:
        lines.append(
            f"{item['faults']} class={item['class']} source={item['source']} "
            f"name={item['name']}"
        )

    output = "\n".join(lines) + "\n"
    if path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def main() -> int:
    args = parse_args()

    allocator_regions = parse_allocator_regions(args.allocator_log)
    address_regions = parse_address_regions(args.address_log)
    probe_records, probe_regions = parse_probe_jsonl(args.kv_init_probe_log)
    regions = allocator_regions + address_regions + probe_regions
    lookups = build_lookups(regions)

    summary: dict[str, Any] = {
        "allocator_log": args.allocator_log,
        "address_log": args.address_log,
        "kv_init_probe_log": args.kv_init_probe_log,
        "route_log": args.route_log,
        "regions": summarize_regions(regions),
        "probe_jsonl_records": len(probe_records),
        "fault_analysis": analyze_fault_log(
            args.fault_log,
            lookups,
            faults_csv=args.faults_csv,
            top_n=args.top_n,
        ),
        "route_analysis": analyze_route_log(args.route_log, lookups),
    }

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_text_summary(summary, args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
