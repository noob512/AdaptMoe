#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: skip malformed json line={line_no}: {exc}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM reserved expert JSONL trace."
    )
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    records = load_records(args.trace)
    actions = Counter(str(record.get("action", "")) for record in records)
    current_streams = Counter(
        int(record["current_stream_cuda"])
        for record in records
        if "current_stream_cuda" in record
    )
    shared_streams = Counter(
        int(record["shared_experts_stream_cuda"])
        for record in records
        if "shared_experts_stream_cuda" in record
    )

    cleanup_lags: list[int] = []
    cleanup_by_layer: dict[str, list[tuple[int, int | None]]] = defaultdict(list)
    for record in records:
        if record.get("action") != "reserved_layer_done_cleanup":
            continue
        try:
            cleanup_step = int(record.get("step", -1))
            current_step = record.get("current_step")
            current_step_int = int(current_step) if current_step is not None else None
        except (TypeError, ValueError):
            continue
        if current_step_int is not None:
            cleanup_lags.append(current_step_int - cleanup_step)
        cleanup_by_layer[str(record.get("layer_name", ""))].append(
            (cleanup_step, current_step_int)
        )

    failed_reserved = [
        record
        for record in records
        if str(record.get("action", "")).startswith("reserved_prefetch")
        and record.get("success") is False
    ]

    print(f"records={len(records)}")
    print("actions:")
    for action, count in actions.most_common():
        print(f"  {action}: {count}")

    print("streams:")
    print(f"  current_stream_unique={len(current_streams)}")
    for stream, count in current_streams.most_common(8):
        print(f"    current_stream_cuda={stream} count={count}")
    print(f"  shared_stream_unique={len(shared_streams)}")
    for stream, count in shared_streams.most_common(8):
        print(f"    shared_experts_stream_cuda={stream} count={count}")

    if cleanup_lags:
        print("cleanup_lag_steps:")
        print(f"  samples={len(cleanup_lags)}")
        print(f"  min={min(cleanup_lags)} max={max(cleanup_lags)}")
        avg = sum(cleanup_lags) / len(cleanup_lags)
        print(f"  avg={avg:.3f}")
        print(f"  histogram={dict(sorted(Counter(cleanup_lags).items()))}")
    else:
        print("cleanup_lag_steps: none")

    print(f"reserved_prefetch_failed_records={len(failed_reserved)}")
    if failed_reserved:
        for record in failed_reserved[:8]:
            print(
                "  failed "
                f"layer={record.get('layer_name')} "
                f"step={record.get('step')} "
                f"region={record.get('residency_region_id')} "
                f"role={record.get('role')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
