#!/usr/bin/env python3
"""Summarize cold expert reserved-slot restore activity from vLLM traces."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROLE_NAMES = {
    3: "moe_gate_up",
    4: "moe_down",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which static cold experts triggered 512KiB reserved-slot "
            "restores in a cold8 vLLM run."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def layer_id_from_name(layer_name: Any) -> int | None:
    if not isinstance(layer_name, str):
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name)
    if not match:
        return None
    return int(match.group(1))


def find_mechanism_dir(run_root: Path) -> Path:
    candidates = [
        run_root / "cold8_offload_mechanism",
        run_root / "mechanism",
        run_root,
    ]
    for candidate in candidates:
        if (candidate / "prefetch_trace.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"no prefetch_trace.jsonl under {run_root}")


def load_plan(run_root: Path) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    plan = load_json(run_root / "cold8_offload_plan.json")
    token_hits: dict[tuple[int, int], int] = {}
    cold: set[tuple[int, int]] = set()
    for record in plan.get("offload_plan", []):
        try:
            layer_id = int(record["layer_id"])
            expert_id = int(record["expert_id"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (layer_id, expert_id)
        cold.add(key)
        token_hits[key] = int(record.get("token_hits") or 0)
    return token_hits, cold


def load_request_intervals(progress_path: Path) -> tuple[list[float], list[dict[str, Any]]]:
    intervals: list[dict[str, Any]] = []
    active: dict[int, dict[str, Any]] = {}
    if not progress_path.exists():
        return [], intervals
    with progress_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = record.get("action")
            request = record.get("request")
            timestamp = record.get("timestamp")
            if not isinstance(request, int) or not isinstance(timestamp, (int, float)):
                continue
            if action == "request_begin":
                active[request] = {
                    "request": request,
                    "begin": float(timestamp),
                    "end": None,
                    "prompt": record.get("prompt", ""),
                }
            elif action == "request_end":
                entry = active.pop(
                    request,
                    {"request": request, "begin": float(timestamp), "prompt": ""},
                )
                entry["end"] = float(timestamp)
                entry["output_tokens"] = record.get("output_tokens")
                entry["generate_seconds"] = record.get("generate_seconds")
                intervals.append(entry)
    intervals.sort(key=lambda item: float(item.get("begin") or 0.0))
    begins = [float(item.get("begin") or 0.0) for item in intervals]
    return begins, intervals


def request_for_timestamp(
    timestamp: Any, begins: list[float], intervals: list[dict[str, Any]]
) -> int | None:
    if not isinstance(timestamp, (int, float)) or not begins:
        return None
    idx = bisect.bisect_right(begins, float(timestamp)) - 1
    if idx < 0:
        return None
    interval = intervals[idx]
    end = interval.get("end")
    if isinstance(end, (int, float)) and float(timestamp) <= float(end):
        return int(interval["request"])
    return None


@dataclass
class ExpertStats:
    layer_id: int
    expert_id: int
    token_hits: int = 0
    restore_entry_count: int = 0
    restore_bytes: int = 0
    restore_steps: set[int] = field(default_factory=set)
    restore_requests: set[int] = field(default_factory=set)
    gate_up_entries: int = 0
    gate_up_bytes: int = 0
    down_entries: int = 0
    down_bytes: int = 0
    first_step: int | None = None
    last_step: int | None = None
    first_request: int | None = None
    last_request: int | None = None
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    def update(
        self,
        *,
        step: int | None,
        request: int | None,
        timestamp: float | None,
        role: str,
        bytes_value: int,
    ) -> None:
        self.restore_entry_count += 1
        self.restore_bytes += bytes_value
        if step is not None:
            self.restore_steps.add(step)
            self.first_step = step if self.first_step is None else min(self.first_step, step)
            self.last_step = step if self.last_step is None else max(self.last_step, step)
        if request is not None:
            self.restore_requests.add(request)
            self.first_request = (
                request if self.first_request is None else min(self.first_request, request)
            )
            self.last_request = (
                request if self.last_request is None else max(self.last_request, request)
            )
        if timestamp is not None:
            self.first_timestamp = (
                timestamp
                if self.first_timestamp is None
                else min(self.first_timestamp, timestamp)
            )
            self.last_timestamp = (
                timestamp
                if self.last_timestamp is None
                else max(self.last_timestamp, timestamp)
            )
        if role == "moe_gate_up":
            self.gate_up_entries += 1
            self.gate_up_bytes += bytes_value
        elif role == "moe_down":
            self.down_entries += 1
            self.down_bytes += bytes_value


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def role_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return ROLE_NAMES.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def analyze(run_root: Path, output_dir: Path) -> dict[str, Any]:
    mechanism_dir = find_mechanism_dir(run_root)
    prefetch_trace = mechanism_dir / "prefetch_trace.jsonl"
    progress_path = mechanism_dir / "progress.jsonl"
    begins, intervals = load_request_intervals(progress_path)
    token_hits, cold = load_plan(run_root)

    by_expert: dict[tuple[int, int], ExpertStats] = {}
    by_request: Counter[int] = Counter()
    by_layer: Counter[int] = Counter()
    by_step_expert_role: dict[tuple[int | None, int, int, str, int | None], dict[str, Any]] = {}
    ioctl = Counter()
    actions = Counter()
    failed_entries = 0
    first_failure: dict[str, Any] | None = None

    with prefetch_trace.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = record.get("action")
            actions[action] += 1
            if action == "reserved_debug_prefetch_ioctl_exit":
                ioctl["records"] += 1
                ioctl["success"] += int(record.get("success_count") or 0)
                ioctl["failure"] += int(record.get("failure_count") or 0)
                ioctl["accepted_bytes"] += int(record.get("accepted_bytes") or 0)
                continue
            if action != "reserved_debug_assign_item_enter":
                continue

            layer_id = int_or_none(record.get("layer_id"))
            if layer_id is None:
                layer_id = layer_id_from_name(record.get("layer_name"))
            expert_id = int_or_none(record.get("expert_id"))
            if layer_id is None or expert_id is None:
                continue
            role = role_name(record.get("role"))
            bytes_value = int(record.get("bytes") or 0)
            step = int_or_none(record.get("step"))
            timestamp = record.get("timestamp")
            ts_float = float(timestamp) if isinstance(timestamp, (int, float)) else None
            request = request_for_timestamp(ts_float, begins, intervals)

            key = (layer_id, expert_id)
            stats = by_expert.setdefault(
                key,
                ExpertStats(layer_id=layer_id, expert_id=expert_id),
            )
            stats.token_hits = token_hits.get(key, 0)
            stats.update(
                step=step,
                request=request,
                timestamp=ts_float,
                role=role,
                bytes_value=bytes_value,
            )
            by_layer[layer_id] += bytes_value
            if request is not None:
                by_request[request] += bytes_value

            event_key = (step, layer_id, expert_id, role, request)
            event = by_step_expert_role.setdefault(
                event_key,
                {
                    "step": step,
                    "request": request,
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "role": role,
                    "entries": 0,
                    "bytes": 0,
                    "first_timestamp": ts_float,
                    "last_timestamp": ts_float,
                },
            )
            event["entries"] += 1
            event["bytes"] += bytes_value
            if ts_float is not None:
                if event["first_timestamp"] is None:
                    event["first_timestamp"] = ts_float
                else:
                    event["first_timestamp"] = min(event["first_timestamp"], ts_float)
                if event["last_timestamp"] is None:
                    event["last_timestamp"] = ts_float
                else:
                    event["last_timestamp"] = max(event["last_timestamp"], ts_float)

            status = record.get("status")
            if status not in (None, 0, "0"):
                failed_entries += 1
                if first_failure is None:
                    first_failure = record

    # Include cold experts that never triggered a restore.
    for key in cold:
        by_expert.setdefault(
            key,
            ExpertStats(layer_id=key[0], expert_id=key[1], token_hits=token_hits.get(key, 0)),
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    by_expert_rows = []
    for stats in sorted(
        by_expert.values(),
        key=lambda item: (-item.restore_bytes, item.layer_id, item.expert_id),
    ):
        by_expert_rows.append(
            {
                "layer_id": stats.layer_id,
                "expert_id": stats.expert_id,
                "plan_token_hits": stats.token_hits,
                "was_in_cold_plan": (stats.layer_id, stats.expert_id) in cold,
                "restore_entry_count": stats.restore_entry_count,
                "restore_steps": len(stats.restore_steps),
                "restore_requests": len(stats.restore_requests),
                "restore_bytes": stats.restore_bytes,
                "restore_mib": stats.restore_bytes / (1024 * 1024),
                "gate_up_entries": stats.gate_up_entries,
                "gate_up_bytes": stats.gate_up_bytes,
                "down_entries": stats.down_entries,
                "down_bytes": stats.down_bytes,
                "first_step": stats.first_step,
                "last_step": stats.last_step,
                "first_request": stats.first_request,
                "last_request": stats.last_request,
                "first_timestamp": stats.first_timestamp,
                "last_timestamp": stats.last_timestamp,
            }
        )

    by_expert_csv = output_dir / "cold8_restore_by_expert.csv"
    with by_expert_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_expert_rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_expert_rows)

    event_rows = sorted(
        by_step_expert_role.values(),
        key=lambda item: (
            -int(item["bytes"]),
            item["step"] if item["step"] is not None else -1,
            item["layer_id"],
            item["expert_id"],
            item["role"],
        ),
    )
    events_csv = output_dir / "cold8_restore_events.csv"
    with events_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "request",
            "step",
            "layer_id",
            "expert_id",
            "role",
            "entries",
            "bytes",
            "first_timestamp",
            "last_timestamp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_rows)

    request_rows = [
        {
            "request": request,
            "restore_bytes": bytes_value,
            "restore_mib": bytes_value / (1024 * 1024),
        }
        for request, bytes_value in sorted(by_request.items())
    ]
    by_request_csv = output_dir / "cold8_restore_by_request.csv"
    with by_request_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["request", "restore_bytes", "restore_mib"])
        writer.writeheader()
        writer.writerows(request_rows)

    layer_rows = [
        {
            "layer_id": layer_id,
            "restore_bytes": bytes_value,
            "restore_mib": bytes_value / (1024 * 1024),
        }
        for layer_id, bytes_value in sorted(by_layer.items())
    ]
    by_layer_csv = output_dir / "cold8_restore_by_layer.csv"
    with by_layer_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer_id", "restore_bytes", "restore_mib"])
        writer.writeheader()
        writer.writerows(layer_rows)

    touched_cold = {
        (row["layer_id"], row["expert_id"])
        for row in by_expert_rows
        if row["was_in_cold_plan"] and row["restore_entry_count"] > 0
    }
    untouched_cold = sorted(cold - touched_cold)
    total_restore_bytes = sum(row["restore_bytes"] for row in by_expert_rows)
    total_restore_entries = sum(row["restore_entry_count"] for row in by_expert_rows)

    summary = {
        "run_root": str(run_root),
        "mechanism_dir": str(mechanism_dir),
        "prefetch_trace": str(prefetch_trace),
        "cold_plan_experts": len(cold),
        "touched_cold_experts": len(touched_cold),
        "untouched_cold_experts": len(untouched_cold),
        "restore_entry_count": total_restore_entries,
        "restore_bytes": total_restore_bytes,
        "restore_mib": total_restore_bytes / (1024 * 1024),
        "restore_event_rows": len(event_rows),
        "unique_restore_steps": len(
            {
                row["step"]
                for row in event_rows
                if row.get("step") is not None
            }
        ),
        "unique_restore_requests": len(by_request),
        "ioctl_records": ioctl["records"],
        "ioctl_success_entries": ioctl["success"],
        "ioctl_failure_entries": ioctl["failure"],
        "ioctl_accepted_bytes": ioctl["accepted_bytes"],
        "assign_failed_entries": failed_entries,
        "first_failure": first_failure,
        "top_experts": by_expert_rows[:25],
        "untouched_cold_expert_sample": [
            {"layer_id": layer_id, "expert_id": expert_id}
            for layer_id, expert_id in untouched_cold[:50]
        ],
        "outputs": {
            "by_expert_csv": str(by_expert_csv),
            "events_csv": str(events_csv),
            "by_request_csv": str(by_request_csv),
            "by_layer_csv": str(by_layer_csv),
        },
    }

    summary_json = output_dir / "cold8_restore_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = output_dir / "cold8_restore_report.md"
    top_rows = by_expert_rows[:20]
    lines = [
        "# Cold8 Reserved Restore Analysis",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Summary",
        "",
        "```text",
        f"cold_plan_experts = {len(cold)}",
        f"touched_cold_experts = {len(touched_cold)}",
        f"untouched_cold_experts = {len(untouched_cold)}",
        f"restore_entry_count = {total_restore_entries}",
        f"restore_bytes = {total_restore_bytes}",
        f"restore_mib = {total_restore_bytes / (1024 * 1024):.3f}",
        f"restore_event_rows = {len(event_rows)}",
        f"unique_restore_steps = {summary.get('unique_restore_steps')}",
        f"unique_restore_requests = {len(by_request)}",
        f"ioctl_records = {ioctl['records']}",
        f"ioctl_success_entries = {ioctl['success']}",
        f"ioctl_failure_entries = {ioctl['failure']}",
        "```",
        "",
        "## Top Restored Cold Experts",
        "",
        "| rank | layer | expert | plan hits | entries | MiB | steps | requests | gate_up entries | down entries | first request | last request |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(top_rows, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["layer_id"]),
                    str(row["expert_id"]),
                    str(row["plan_token_hits"]),
                    str(row["restore_entry_count"]),
                    f"{row['restore_mib']:.1f}",
                    str(row["restore_steps"]),
                    str(row["restore_requests"]),
                    str(row["gate_up_entries"]),
                    str(row["down_entries"]),
                    str(row["first_request"]),
                    str(row["last_request"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- by expert: `{by_expert_csv}`",
            f"- restore events: `{events_csv}`",
            f"- by request: `{by_request_csv}`",
            f"- by layer: `{by_layer_csv}`",
            f"- summary JSON: `{summary_json}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["outputs"]["report_md"] = str(report)
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.run_root / "cold8_restore_analysis")
    summary = analyze(args.run_root, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
