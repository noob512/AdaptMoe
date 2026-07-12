#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAULT_RE = re.compile(
    r"Replayable fault GPU .*?\bpage=(0x[0-9a-fA-F]+)"
    r".*?\baccess=([A-Z_]+)"
)


@dataclass(frozen=True)
class FaultEvent:
    order: int
    page: int
    access: str
    line: str
    after_compute_start: bool


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator * 100.0 / denominator


def _load_fault_events(
    trace_log: Path,
    compute_marker_prefix: str,
) -> tuple[list[FaultEvent], Counter[str]]:
    faults: list[FaultEvent] = []
    counts: Counter[str] = Counter()
    compute_started = False

    with trace_log.open("r", encoding="utf-8", errors="replace") as handle:
        for order, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if compute_marker_prefix in line:
                counts["compute_marker"] += 1
                compute_started = True
                continue

            fault_match = FAULT_RE.search(line)
            if not fault_match:
                continue

            counts["replayable_fault"] += 1
            faults.append(
                FaultEvent(
                    order=order,
                    page=int(fault_match.group(1), 16),
                    access=fault_match.group(2),
                    line=line,
                    after_compute_start=compute_started,
                )
            )

    return faults, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize UVM replayable faults before and after vLLM compute start."
    )
    parser.add_argument("--trace-log", required=True, type=Path)
    parser.add_argument(
        "--compute-marker-prefix",
        default="VLLM_UVM_COMPUTE_START",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.trace_log.is_file():
        raise SystemExit(f"trace log not found: {args.trace_log}")

    faults, counts = _load_fault_events(args.trace_log, args.compute_marker_prefix)
    total_faults = len(faults)
    before_faults = sum(1 for fault in faults if not fault.after_compute_start)
    after_faults = total_faults - before_faults

    total_pages = Counter(fault.page for fault in faults)
    after_pages = Counter(fault.page for fault in faults if fault.after_compute_start)
    total_access = Counter(fault.access for fault in faults)
    after_access = Counter(fault.access for fault in faults if fault.after_compute_start)

    summary: dict[str, Any] = {
        "trace_log": str(args.trace_log),
        "compute_marker_prefix": args.compute_marker_prefix,
        "compute_marker_count": counts["compute_marker"],
        "replayable_fault_count": total_faults,
        "faults_before_compute_start": before_faults,
        "faults_after_compute_start": after_faults,
        "faults_after_compute_start_ratio_pct": _ratio(after_faults, total_faults),
        "unique_pages_total": len(total_pages),
        "unique_pages_after_compute_start": len(after_pages),
        "unique_pages_after_compute_start_ratio_pct": _ratio(
            len(after_pages), len(total_pages)
        ),
        "top_pages_after_compute_start": [
            {"page_hex": f"0x{page:x}", "count": count}
            for page, count in after_pages.most_common(args.top)
        ],
        "fault_access_total": dict(total_access),
        "fault_access_after_compute_start": dict(after_access),
    }

    print("inputs:")
    print(f"  trace_log={summary['trace_log']}")
    print(f"  compute_marker_prefix={summary['compute_marker_prefix']}")
    print("trace_events:")
    print(f"  compute_marker_count={summary['compute_marker_count']}")
    print(f"  replayable_fault_count={summary['replayable_fault_count']}")
    print("compute_fault_window:")
    print(f"  faults_before_compute_start={summary['faults_before_compute_start']}")
    print(f"  faults_after_compute_start={summary['faults_after_compute_start']}")
    print(
        "  faults_after_compute_start_ratio_pct="
        f"{summary['faults_after_compute_start_ratio_pct']}"
    )
    print(f"  unique_pages_total={summary['unique_pages_total']}")
    print(
        "  unique_pages_after_compute_start="
        f"{summary['unique_pages_after_compute_start']}"
    )
    print(
        "  unique_pages_after_compute_start_ratio_pct="
        f"{summary['unique_pages_after_compute_start_ratio_pct']}"
    )
    print("fault_access_total:")
    for access, count in total_access.most_common():
        print(f"  {access}: {count}")
    print("fault_access_after_compute_start:")
    for access, count in after_access.most_common():
        print(f"  {access}: {count}")
    print("top_pages_after_compute_start:")
    for item in summary["top_pages_after_compute_start"]:
        print(f"  {item['page_hex']}: {item['count']}")

    if summary["compute_marker_count"] == 0:
        print("warning: compute start marker was not observed in the trace log.")
    if summary["faults_after_compute_start"] == 0:
        print("note: no replayable faults were observed after compute start.")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
