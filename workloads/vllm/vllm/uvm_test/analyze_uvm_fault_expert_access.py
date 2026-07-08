#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAULT_RE = re.compile(
    r"Replayable fault GPU .*?\bpage=(0x[0-9a-fA-F]+)"
    r".*?\baccess=([A-Z_]+)"
)
MARKER_RE = re.compile(r"vllm_moe_stream\s+(\{.*\})")
LAYER_RE = re.compile(r"(?:^|\.)layers?\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    record: dict[str, Any]


@dataclass
class FaultEvent:
    order: int
    page: int
    access: str
    line: str
    marker: dict[str, Any] | None
    region: dict[str, Any] | None


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_layer_id(layer_name: Any) -> int | None:
    if not isinstance(layer_name, str):
        return None
    match = LAYER_RE.search(layer_name)
    if not match:
        return None
    return int(match.group(1))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: skip malformed json line={line_no}: {exc}")
    return records


def load_weight_intervals(
    path: Path,
    *,
    only_moe_expert: bool,
    only_managed: bool,
    include_parent_regions: bool,
) -> tuple[list[int], list[Interval], Counter[str]]:
    intervals: list[Interval] = []
    capability_reasons: Counter[str] = Counter()
    for record in load_jsonl(path):
        if only_moe_expert and not bool(record.get("is_moe_expert")):
            continue
        if not include_parent_regions and not bool(
            record.get("is_logical_expert_slice")
        ):
            continue
        reason = str(record.get("driver_capability_reason") or "unknown")
        capability_reasons[reason] += 1
        if only_managed and reason != "cuda_managed":
            continue
        start = parse_int(record.get("start_int"))
        end = parse_int(record.get("end_int"))
        if start is None or end is None or end < start:
            continue
        intervals.append(Interval(start=start, end=end + 1, record=record))
    intervals.sort(key=lambda interval: (interval.start, interval.end))
    starts = [interval.start for interval in intervals]
    return starts, intervals, capability_reasons


def find_interval(
    starts: list[int],
    intervals: list[Interval],
    address: int,
) -> dict[str, Any] | None:
    index = bisect.bisect_right(starts, address) - 1
    # Logical expert slices do not overlap in the normal case. Walk back a few
    # entries to tolerate parent/slice adjacency and page-rounded fault addresses.
    for cursor in range(index, max(index - 8, -1), -1):
        interval = intervals[cursor]
        if interval.start <= address < interval.end:
            return interval.record
    return None


def load_trace_events(
    fault_trace: Path,
    starts: list[int],
    intervals: list[Interval],
) -> tuple[list[FaultEvent], Counter[str], Counter[str]]:
    faults: list[FaultEvent] = []
    event_counts: Counter[str] = Counter()
    unmatched_fault_pages: Counter[str] = Counter()
    last_marker: dict[str, Any] | None = None
    with fault_trace.open("r", encoding="utf-8", errors="replace") as handle:
        for order, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            marker_match = MARKER_RE.search(line)
            if marker_match:
                event_counts["vllm_moe_stream_marker"] += 1
                try:
                    last_marker = json.loads(marker_match.group(1))
                except json.JSONDecodeError:
                    event_counts["bad_marker_json"] += 1
                continue
            fault_match = FAULT_RE.search(line)
            if not fault_match:
                continue
            event_counts["replayable_fault"] += 1
            page = int(fault_match.group(1), 16)
            access = fault_match.group(2)
            region = find_interval(starts, intervals, page)
            if region is None:
                unmatched_fault_pages[f"0x{page:x}"] += 1
            faults.append(
                FaultEvent(
                    order=order,
                    page=page,
                    access=access,
                    line=line,
                    marker=last_marker.copy() if last_marker is not None else None,
                    region=region,
                )
            )
    return faults, event_counts, unmatched_fault_pages


def region_key(region: dict[str, Any]) -> tuple[int | None, int | None, str]:
    return (
        parse_int(region.get("layer_id")),
        parse_int(region.get("expert_id")),
        str(region.get("role") or ""),
    )


def marker_layer(marker: dict[str, Any] | None) -> int | None:
    if marker is None:
        return None
    return parse_layer_id(marker.get("layer_name"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate UVM replayable fault pages with vLLM expert weight map "
            "and optional vLLM MoE stream trace markers."
        )
    )
    parser.add_argument("--fault-trace", required=True, type=Path)
    parser.add_argument("--weight-map", required=True, type=Path)
    parser.add_argument("--late-gap", type=int, default=2)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--include-parent-regions", action="store_true")
    parser.add_argument("--include-non-managed", action="store_true")
    args = parser.parse_args()

    starts, intervals, capability_reasons = load_weight_intervals(
        args.weight_map,
        only_moe_expert=True,
        only_managed=not args.include_non_managed,
        include_parent_regions=args.include_parent_regions,
    )
    faults, event_counts, unmatched_fault_pages = load_trace_events(
        args.fault_trace,
        starts,
        intervals,
    )

    matched_faults = [fault for fault in faults if fault.region is not None]
    by_region: Counter[tuple[int | None, int | None, str]] = Counter()
    by_access: Counter[str] = Counter()
    by_lateness: Counter[int] = Counter()
    late_faults: list[FaultEvent] = []

    for fault in matched_faults:
        assert fault.region is not None
        key = region_key(fault.region)
        by_region[key] += 1
        by_access[fault.access] += 1
        target_layer = key[0]
        current_layer = marker_layer(fault.marker)
        if target_layer is None or current_layer is None:
            continue
        delta = current_layer - target_layer
        by_lateness[delta] += 1
        if delta >= args.late_gap:
            late_faults.append(fault)

    print("inputs:")
    print(f"  fault_trace={args.fault_trace}")
    print(f"  weight_map={args.weight_map}")
    print(f"  intervals={len(intervals)}")
    print("weight_capability_reasons:")
    for reason, count in capability_reasons.most_common():
        print(f"  {reason}: {count}")
    if not intervals:
        print(
            "warning: no matching managed MoE expert intervals were loaded; "
            "fault access experiment is not observable for this weight map."
        )

    print("trace_events:")
    for key, count in event_counts.most_common():
        print(f"  {key}: {count}")
    print(f"  faults_total={len(faults)}")
    print(f"  faults_matched_to_expert={len(matched_faults)}")
    print(f"  faults_unmatched={len(faults) - len(matched_faults)}")

    print("fault_access:")
    for access, count in by_access.most_common():
        print(f"  {access}: {count}")

    print("top_faulted_expert_regions:")
    for (layer_id, expert_id, role), count in by_region.most_common(args.top):
        print(
            f"  layer={layer_id} expert={expert_id} role={role} faults={count}"
        )

    print("fault_current_layer_minus_weight_layer:")
    if by_lateness:
        for delta, count in sorted(by_lateness.items()):
            print(f"  delta={delta}: {count}")
    else:
        print("  none")

    print(f"late_faults_delta_ge_{args.late_gap}: {len(late_faults)}")
    for fault in late_faults[: args.top]:
        assert fault.region is not None
        layer_id, expert_id, role = region_key(fault.region)
        current_layer = marker_layer(fault.marker)
        action = fault.marker.get("action") if fault.marker else None
        current_layer_name = fault.marker.get("layer_name") if fault.marker else None
        delta = (
            current_layer - layer_id
            if current_layer is not None and layer_id is not None
            else None
        )
        print(
            "  late "
            f"order={fault.order} page=0x{fault.page:x} access={fault.access} "
            f"weight_layer={layer_id} expert={expert_id} role={role} "
            f"current_layer={current_layer} delta={delta} action={action} "
            f"current_layer_name={current_layer_name}"
        )

    if unmatched_fault_pages:
        print("top_unmatched_fault_pages:")
        for page, count in unmatched_fault_pages.most_common(min(args.top, 10)):
            print(f"  {page}: {count}")

    print("caveat:")
    print(
        "  Replayable faults prove observed misses/remote accesses. If an expert "
        "page already has a valid GPU PTE and does not fault, this script cannot "
        "prove the page was not read."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
