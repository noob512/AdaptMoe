#!/usr/bin/env python3
"""Join MoE routing top-K experts with semantic expert weight ranges.

The routing trace tells us which experts were active in each MoE invocation.
The weight map tells us where expert weights live.  This script verifies that
each per-event top-K expert can be resolved to concrete weight ranges such as:

    (layer_id, expert_id, moe_gate_up) -> ptr, bytes
    (layer_id, expert_id, moe_down)    -> ptr, bytes

If the weight map already contains logical expert slices, those are used.  If it
only contains fused per-layer tensors, slices are derived from shape/stride.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LAYER_RE = re.compile(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ExpertRange:
    layer_id: int
    expert_id: int
    role: str
    name: str
    start: int
    end: int
    size_bytes: int
    dtype: str
    source: str
    parent_name: str | None

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.layer_id, self.expert_id, self.role)

    def as_json(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "role": self.role,
            "name": self.name,
            "start": f"0x{self.start:x}",
            "end": f"0x{self.end:x}",
            "size_bytes": self.size_bytes,
            "dtype": self.dtype,
            "source": self.source,
            "parent_name": self.parent_name,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-trace", required=True, type=Path)
    parser.add_argument("--weight-map", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--roles",
        default="moe_gate_up,moe_down",
        help="Comma-separated expert roles that must be address-resolved.",
    )
    parser.add_argument(
        "--max-jsonl-events",
        type=int,
        default=0,
        help="Limit detailed JSONL event output. 0 means write every event.",
    )
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


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def extract_layer_id(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(statistics.fmean(vals))


def bucket_num_tokens(num_tokens: int) -> str:
    if num_tokens <= 1:
        return "1"
    if num_tokens <= 8:
        return "2-8"
    if num_tokens <= 32:
        return "9-32"
    if num_tokens <= 128:
        return "33-128"
    if num_tokens <= 512:
        return "129-512"
    return ">512"


def role_is_main_expert_weight(record: dict[str, Any], roles: set[str]) -> bool:
    if str(record.get("kind") or "") == "weight:expert_slice":
        return False
    if parse_bool(record.get("is_logical_slice")) or parse_bool(
        record.get("is_logical_expert_slice")
    ):
        return False
    role = str(record.get("role") or "")
    if role not in roles:
        return False
    name = str(record.get("name") or "")
    if parse_bool(record.get("is_moe_expert_main_weight")):
        return True
    shard_id = str(record.get("shard_id") or "")
    if shard_id in {"w13", "w2", "w1", "w3"}:
        return True
    return name.endswith(".w13_weight") or name.endswith(".w2_weight")


def direct_expert_range(record: dict[str, Any], roles: set[str]) -> ExpertRange | None:
    role = str(record.get("role") or "")
    if role not in roles:
        return None
    layer_id = parse_int(record.get("layer_id"))
    expert_id = parse_int(record.get("expert_id"))
    start = parse_int(record.get("start_int")) or parse_int(record.get("start"))
    end = parse_int(record.get("end_int")) or parse_int(record.get("end"))
    size_bytes = parse_int(record.get("size_bytes"))
    if None in (layer_id, expert_id, start, end, size_bytes):
        return None
    if int(size_bytes) <= 0 or int(end) < int(start):
        return None
    if not (
        parse_bool(record.get("is_logical_slice"))
        or parse_bool(record.get("is_logical_expert_slice"))
        or ".expert_" in str(record.get("name") or "")
    ):
        return None
    return ExpertRange(
        layer_id=int(layer_id),
        expert_id=int(expert_id),
        role=role,
        name=str(record.get("name") or ""),
        start=int(start),
        end=int(end),
        size_bytes=int(size_bytes),
        dtype=str(record.get("dtype") or "unknown"),
        source="direct_slice",
        parent_name=(
            str(record.get("logical_parent_name"))
            if record.get("logical_parent_name") is not None
            else str(record.get("alias_of"))
            if record.get("alias_of") is not None
            else None
        ),
    )


def derived_expert_ranges(record: dict[str, Any], roles: set[str]) -> list[ExpertRange]:
    if not role_is_main_expert_weight(record, roles):
        return []
    layer_id = parse_int(record.get("layer_id"))
    start = parse_int(record.get("data_start_int")) or parse_int(
        record.get("start_int")
    ) or parse_int(record.get("start"))
    total_size = parse_int(record.get("size_bytes"))
    shape = record.get("shape")
    stride = record.get("stride")
    element_size = parse_int(record.get("element_size"))
    if layer_id is None or start is None or total_size is None:
        return []
    if not isinstance(shape, list) or len(shape) < 2:
        return []
    try:
        shape_i = [int(item) for item in shape]
    except (TypeError, ValueError):
        return []
    expert_count = shape_i[0]
    if expert_count <= 0:
        return []
    if not element_size or element_size <= 0:
        tensor_bytes = parse_int(record.get("tensor_bytes")) or total_size
        numel = parse_int(record.get("numel"))
        element_size = max(int(tensor_bytes // numel), 1) if numel else 1
    if isinstance(stride, list) and len(stride) == len(shape_i):
        try:
            stride_i = [int(item) for item in stride]
        except (TypeError, ValueError):
            stride_i = []
    else:
        stride_i = []

    ranges: list[ExpertRange] = []
    role = str(record.get("role") or "")
    parent_name = str(record.get("name") or "")
    if stride_i:
        inner_min = 0
        inner_max = 0
        for dim, dim_stride in zip(shape_i[1:], stride_i[1:]):
            if dim <= 1:
                continue
            delta = (dim - 1) * dim_stride
            if delta >= 0:
                inner_max += delta
            else:
                inner_min += delta
        for expert_id in range(expert_count):
            base = int(start) + expert_id * stride_i[0] * int(element_size)
            slice_start = base + inner_min * int(element_size)
            slice_end = base + inner_max * int(element_size) + int(element_size) - 1
            ranges.append(
                ExpertRange(
                    layer_id=int(layer_id),
                    expert_id=expert_id,
                    role=role,
                    name=f"{parent_name}.expert_{expert_id}",
                    start=slice_start,
                    end=slice_end,
                    size_bytes=slice_end - slice_start + 1,
                    dtype=str(record.get("dtype") or "unknown"),
                    source="derived_from_parent",
                    parent_name=parent_name,
                )
            )
        return ranges

    slice_bytes = int(total_size) // expert_count
    if slice_bytes <= 0:
        return []
    for expert_id in range(expert_count):
        slice_start = int(start) + expert_id * slice_bytes
        slice_end = slice_start + slice_bytes - 1
        ranges.append(
            ExpertRange(
                layer_id=int(layer_id),
                expert_id=expert_id,
                role=role,
                name=f"{parent_name}.expert_{expert_id}",
                start=slice_start,
                end=slice_end,
                size_bytes=slice_bytes,
                dtype=str(record.get("dtype") or "unknown"),
                source="derived_equal_split",
                parent_name=parent_name,
            )
        )
    return ranges


def load_expert_ranges(
    weight_map: Path, roles: set[str]
) -> tuple[dict[tuple[int, int, str], ExpertRange], Counter[str]]:
    ranges: dict[tuple[int, int, str], ExpertRange] = {}
    source_counts: Counter[str] = Counter()
    parent_records: list[dict[str, Any]] = []
    for record in read_jsonl(weight_map):
        direct = direct_expert_range(record, roles)
        if direct is not None:
            ranges[direct.key] = direct
            source_counts[direct.source] += 1
        if role_is_main_expert_weight(record, roles):
            parent_records.append(record)

    for record in parent_records:
        for derived in derived_expert_ranges(record, roles):
            if derived.key not in ranges:
                ranges[derived.key] = derived
                source_counts[derived.source] += 1
    return ranges, source_counts


def as_int_dict(raw: object) -> dict[int, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, int] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def as_float_dict(raw: object) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def top_experts_for_event(record: dict[str, Any], k: int) -> tuple[list[int], str]:
    weights = as_float_dict(record.get("expert_weight_sums"))
    counts = as_int_dict(record.get("expert_token_counts"))
    active_raw = record.get("active_experts")
    active: set[int] = set()
    if isinstance(active_raw, list):
        for item in active_raw:
            try:
                active.add(int(item))
            except (TypeError, ValueError):
                continue
    if not active:
        active = set(weights) | set(counts)
    if not active:
        return [], "none"
    scores: dict[int, float]
    score_source = "expert_weight_sums" if weights else "expert_token_counts"
    if weights:
        scores = {expert_id: max(weights.get(expert_id, 0.0), 0.0) for expert_id in active}
    else:
        scores = {expert_id: float(counts.get(expert_id, 0)) for expert_id in active}
    ranked = sorted(active, key=lambda expert_id: (-scores.get(expert_id, 0.0), expert_id))
    return ranked[:k], score_source


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    args = parse_args()
    roles = {item.strip() for item in args.roles.split(",") if item.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expert_ranges, source_counts = load_expert_ranges(args.weight_map, roles)

    event_rows: list[dict[str, Any]] = []
    detail_count = 0
    expected_total = 0
    matched_total = 0
    full_match_events = 0
    score_sources: Counter[str] = Counter()
    layers_seen: set[int] = set()
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    detail_path = args.out_dir / "moe_top64_event_details.jsonl"
    with detail_path.open("w", encoding="utf-8") as detail_file:
        for event_index, record in enumerate(read_jsonl(args.routing_trace)):
            layer_name = str(record.get("layer_name") or "")
            layer_id = parse_int(record.get("layer_id"))
            if layer_id is None:
                layer_id = extract_layer_id(layer_name)
            if layer_id is None:
                continue
            top_experts, score_source = top_experts_for_event(record, args.k)
            if not top_experts:
                continue
            num_tokens = parse_int(record.get("num_tokens")) or 0
            step = parse_int(record.get("step")) or 0
            active_count = len(record.get("active_experts") or top_experts)
            expected = len(top_experts) * len(roles)
            matched = 0
            missing: list[dict[str, Any]] = []
            regions: list[dict[str, Any]] = []
            bytes_total = 0
            for expert_id in top_experts:
                for role in sorted(roles):
                    region = expert_ranges.get((int(layer_id), int(expert_id), role))
                    if region is None:
                        missing.append(
                            {
                                "layer_id": int(layer_id),
                                "expert_id": int(expert_id),
                                "role": role,
                            }
                        )
                        continue
                    matched += 1
                    bytes_total += int(region.size_bytes)
                    regions.append(region.as_json())

            expected_total += expected
            matched_total += matched
            if expected > 0 and matched == expected:
                full_match_events += 1
            score_sources[score_source] += 1
            layers_seen.add(int(layer_id))
            row = {
                "event_index": event_index,
                "layer_name": layer_name,
                "layer_id": int(layer_id),
                "step": step,
                "num_tokens": num_tokens,
                "num_tokens_bucket": bucket_num_tokens(num_tokens),
                "active_expert_count": active_count,
                "top_expert_count": len(top_experts),
                "score_source": score_source,
                "expected_role_records": expected,
                "matched_role_records": matched,
                "missing_role_records": expected - matched,
                "address_match_ratio": matched / expected if expected else None,
                "matched_bytes": bytes_total,
                "top_experts": " ".join(str(item) for item in top_experts),
                "missing": json.dumps(missing, sort_keys=True),
            }
            event_rows.append(row)
            bucket_rows[row["num_tokens_bucket"]].append(row)

            if args.max_jsonl_events <= 0 or detail_count < args.max_jsonl_events:
                detail_file.write(
                    json.dumps(
                        {
                            "event": {
                                key: row[key]
                                for key in (
                                    "event_index",
                                    "layer_name",
                                    "layer_id",
                                    "step",
                                    "num_tokens",
                                    "num_tokens_bucket",
                                    "active_expert_count",
                                    "top_expert_count",
                                    "score_source",
                                    "address_match_ratio",
                                    "matched_bytes",
                                )
                            },
                            "top_experts": top_experts,
                            "regions": regions,
                            "missing": missing,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                detail_count += 1

    fields = list(event_rows[0].keys()) if event_rows else []
    with (args.out_dir / "moe_top64_event_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(event_rows)

    bucket_summary: list[dict[str, Any]] = []
    for bucket in ("1", "2-8", "9-32", "33-128", "129-512", ">512"):
        rows = bucket_rows.get(bucket, [])
        if not rows:
            continue
        expected = sum(int(row["expected_role_records"]) for row in rows)
        matched = sum(int(row["matched_role_records"]) for row in rows)
        active_counts = [float(row["active_expert_count"]) for row in rows]
        top_counts = [float(row["top_expert_count"]) for row in rows]
        bucket_summary.append(
            {
                "bucket": bucket,
                "events": len(rows),
                "event_pct": len(rows) / len(event_rows) if event_rows else None,
                "active_mean": mean(active_counts),
                "active_p90": percentile(active_counts, 90),
                "active_p99": percentile(active_counts, 99),
                "topk_mean": mean(top_counts),
                "topk_p99": percentile(top_counts, 99),
                "expected_role_records": expected,
                "matched_role_records": matched,
                "address_match_ratio": matched / expected if expected else None,
                "full_match_event_pct": (
                    sum(
                        1
                        for row in rows
                        if int(row["expected_role_records"])
                        == int(row["matched_role_records"])
                    )
                    / len(rows)
                ),
            }
        )

    summary = {
        "routing_trace": str(args.routing_trace),
        "weight_map": str(args.weight_map),
        "k": args.k,
        "roles": sorted(roles),
        "events": len(event_rows),
        "layers_seen": len(layers_seen),
        "expert_range_entries": len(expert_ranges),
        "expert_range_source_counts": dict(source_counts),
        "score_source_counts": dict(score_sources),
        "expected_role_records": expected_total,
        "matched_role_records": matched_total,
        "missing_role_records": expected_total - matched_total,
        "address_match_ratio": (
            matched_total / expected_total if expected_total else None
        ),
        "full_match_events": full_match_events,
        "full_match_event_ratio": (
            full_match_events / len(event_rows) if event_rows else None
        ),
        "bucket_summary": bucket_summary,
    }
    (args.out_dir / "moe_top64_semantic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# MoE Top64 Expert Semantic Resolution",
        "",
        f"- routing_trace: `{args.routing_trace}`",
        f"- weight_map: `{args.weight_map}`",
        f"- k: `{args.k}`",
        f"- roles: `{','.join(sorted(roles))}`",
        f"- events: `{len(event_rows)}`",
        f"- layers_seen: `{len(layers_seen)}`",
        f"- expert_range_entries: `{len(expert_ranges)}`",
        f"- expert_range_source_counts: `{dict(source_counts)}`",
        f"- score_source_counts: `{dict(score_sources)}`",
        "",
        "## Overall",
        "",
        "| expected role records | matched role records | missing | match ratio | full-match events |",
        "|---:|---:|---:|---:|---:|",
        "| {expected} | {matched} | {missing} | {ratio} | {full} |".format(
            expected=expected_total,
            matched=matched_total,
            missing=expected_total - matched_total,
            ratio=fmt(summary["address_match_ratio"]),
            full=fmt(summary["full_match_event_ratio"]),
        ),
        "",
        "## By num_tokens Bucket",
        "",
        "| bucket | events | event_pct | active_mean | active_p90 | active_p99 | topK_mean | match_ratio | full_match_pct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_summary:
        lines.append(
            "| {bucket} | {events} | {event_pct} | {active_mean} | {active_p90} | {active_p99} | {topk_mean} | {match_ratio} | {full_match} |".format(
                bucket=row["bucket"],
                events=row["events"],
                event_pct=fmt(row["event_pct"]),
                active_mean=fmt(row["active_mean"]),
                active_p90=fmt(row["active_p90"]),
                active_p99=fmt(row["active_p99"]),
                topk_mean=fmt(row["topk_mean"]),
                match_ratio=fmt(row["address_match_ratio"]),
                full_match=fmt(row["full_match_event_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- event_csv: `{args.out_dir / 'moe_top64_event_summary.csv'}`",
            f"- detail_jsonl: `{detail_path}`",
            f"- summary_json: `{args.out_dir / 'moe_top64_semantic_summary.json'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (args.out_dir / "moe_top64_semantic_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
