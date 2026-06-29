#!/usr/bin/env python3
"""Measure overlap between actual MoE routing and per-layer hot experts.

This is an offline trace analysis tool.  It does not prefetch, offload, or
change residency.  It answers questions like:

  If each layer kept its Top-64 hottest experts resident, how many of the
  experts used by routing events with num_tokens around 32 would be covered?

The same-trace result is an optimistic upper signal.  The train/test result is
stricter: build hotsets from the first trace prefix and evaluate the suffix.
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
from typing import Iterable


@dataclass(frozen=True)
class Event:
    index: int
    step: int
    layer_id: int
    layer_name: str
    num_tokens: int
    active: frozenset[int]
    values: dict[int, float]
    total_weight: float
    num_experts: int


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
    return float(statistics.fmean(vals)) if vals else None


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        return f"{value:.4g}"
    return str(value)


def layer_id_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


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


def parse_events(trace: Path, fallback_num_experts: int) -> list[Event]:
    events: list[Event] = []
    with trace.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            layer_name = str(record.get("layer_name") or "")
            layer_id = layer_id_from_name(layer_name)
            if layer_id is None:
                continue

            active: set[int] = set()
            active_raw = record.get("active_experts")
            if isinstance(active_raw, list):
                for item in active_raw:
                    try:
                        active.add(int(item))
                    except (TypeError, ValueError):
                        continue

            counts = as_int_dict(record.get("expert_token_counts"))
            if not active:
                active = set(counts)
            if not active:
                continue

            values = as_float_dict(record.get("expert_weight_sums"))
            if not values:
                values = {expert_id: float(count) for expert_id, count in counts.items()}
            for expert_id in active:
                values.setdefault(expert_id, 0.0)

            total_weight = sum(max(values.get(expert_id, 0.0), 0.0)
                               for expert_id in active)
            if total_weight <= 0.0:
                total_weight = float(len(active))
                values = {expert_id: 1.0 for expert_id in active}

            try:
                num_experts = int(record.get("global_num_experts") or 0)
            except (TypeError, ValueError):
                num_experts = 0
            if num_experts <= 0:
                num_experts = fallback_num_experts

            events.append(
                Event(
                    index=index,
                    step=int(record.get("step") or 0),
                    layer_id=layer_id,
                    layer_name=layer_name,
                    num_tokens=int(record.get("num_tokens") or 0),
                    active=frozenset(active),
                    values=values,
                    total_weight=total_weight,
                    num_experts=num_experts,
                )
            )
    return events


def score_hotness(events: Iterable[Event], mode: str) -> dict[int, dict[int, float]]:
    scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for event in events:
        layer_scores = scores[event.layer_id]
        if mode == "weighted":
            for expert_id in event.active:
                layer_scores[expert_id] += max(event.values.get(expert_id, 0.0), 0.0)
        elif mode == "event_count":
            for expert_id in event.active:
                layer_scores[expert_id] += 1.0
        elif mode == "token_count":
            for expert_id in event.active:
                layer_scores[expert_id] += max(event.values.get(expert_id, 0.0), 0.0)
        else:
            raise ValueError(f"unknown score mode: {mode}")
    return {layer_id: dict(layer_scores) for layer_id, layer_scores in scores.items()}


def build_hotsets(
    scores: dict[int, dict[int, float]],
    layers: Iterable[int],
    num_experts: int,
    k: int,
) -> dict[int, set[int]]:
    all_experts = list(range(num_experts))
    hotsets: dict[int, set[int]] = {}
    for layer_id in layers:
        layer_scores = scores.get(layer_id, {})
        ranked = sorted(
            all_experts,
            key=lambda expert_id: (-layer_scores.get(expert_id, 0.0), expert_id),
        )
        hotsets[layer_id] = set(ranked[:k])
    return hotsets


def event_in_bucket(event: Event, bucket: str, around_min: int, around_max: int) -> bool:
    if bucket == "all":
        return True
    if bucket == "exact_32":
        return event.num_tokens == 32
    if bucket == "around":
        return around_min <= event.num_tokens <= around_max
    if bucket == "9_32":
        return 9 <= event.num_tokens <= 32
    if bucket == "17_48":
        return 17 <= event.num_tokens <= 48
    if bucket == "33_128":
        return 33 <= event.num_tokens <= 128
    if bucket == "gt_512":
        return event.num_tokens > 512
    raise ValueError(f"unknown bucket: {bucket}")


def stat_row(values: list[float], prefix: str) -> dict[str, float | None]:
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p90": percentile(values, 90),
        f"{prefix}_p99": percentile(values, 99),
        f"{prefix}_max": max(values) if values else None,
    }


def evaluate(
    *,
    events: list[Event],
    hotsets: dict[int, set[int]],
    label: str,
    bucket: str,
    k: int,
    expert_bytes: int,
) -> dict[str, object]:
    active_total = 0
    hit_total = 0
    weight_total = 0.0
    weight_hit = 0.0
    full_covered = 0
    active_counts: list[float] = []
    hit_counts: list[float] = []
    miss_counts: list[float] = []
    coverage_values: list[float] = []
    weighted_coverage_values: list[float] = []
    miss_bytes_one_role_by_step: dict[int, float] = defaultdict(float)
    miss_bytes_two_roles_by_step: dict[int, float] = defaultdict(float)
    miss_experts_by_step: dict[int, float] = defaultdict(float)

    layer_acc: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for event in events:
        hot = hotsets.get(event.layer_id, set())
        active = set(event.active)
        hits = active & hot
        misses = active - hot
        hit_weight = sum(max(event.values.get(expert_id, 0.0), 0.0)
                         for expert_id in hits)

        active_total += len(active)
        hit_total += len(hits)
        weight_total += event.total_weight
        weight_hit += hit_weight
        if not misses:
            full_covered += 1

        event_coverage = len(hits) / len(active) if active else 0.0
        event_weighted = hit_weight / event.total_weight if event.total_weight else 0.0
        active_counts.append(float(len(active)))
        hit_counts.append(float(len(hits)))
        miss_counts.append(float(len(misses)))
        coverage_values.append(event_coverage)
        weighted_coverage_values.append(event_weighted)
        miss_bytes_one_role_by_step[event.step] += len(misses) * expert_bytes
        miss_bytes_two_roles_by_step[event.step] += len(misses) * expert_bytes * 2
        miss_experts_by_step[event.step] += float(len(misses))

        acc = layer_acc[event.layer_id]
        acc["records"] += 1
        acc["active_total"] += len(active)
        acc["hit_total"] += len(hits)
        acc["weight_total"] += event.total_weight
        acc["weight_hit"] += hit_weight
        acc["miss_total"] += len(misses)
        acc["full_covered"] += 1 if not misses else 0

    rows = {
        "label": label,
        "bucket": bucket,
        "k": k,
        "records": len(events),
        "steps": len({event.step for event in events}),
        "layers": len({event.layer_id for event in events}),
        "coverage": hit_total / active_total if active_total else None,
        "weighted_coverage": weight_hit / weight_total if weight_total else None,
        "event_coverage_mean": mean(coverage_values),
        "event_weighted_coverage_mean": mean(weighted_coverage_values),
        "full_event_covered_rate": full_covered / len(events) if events else None,
    }
    rows.update(stat_row(active_counts, "active_experts"))
    rows.update(stat_row(hit_counts, "hit_experts"))
    rows.update(stat_row(miss_counts, "miss_experts"))
    rows.update(stat_row(coverage_values, "event_coverage"))
    rows.update(stat_row(weighted_coverage_values, "event_weighted_coverage"))
    rows.update(stat_row([value / (1024 * 1024)
                          for value in miss_bytes_one_role_by_step.values()],
                         "miss_mib_per_step_one_role"))
    rows.update(stat_row([value / (1024 * 1024)
                          for value in miss_bytes_two_roles_by_step.values()],
                         "miss_mib_per_step_two_roles"))
    rows.update(stat_row(list(miss_experts_by_step.values()), "miss_experts_per_step"))

    layer_rows: list[dict[str, object]] = []
    for layer_id in sorted(layer_acc):
        acc = layer_acc[layer_id]
        records = int(acc["records"])
        layer_rows.append(
            {
                "label": label,
                "bucket": bucket,
                "k": k,
                "layer_id": layer_id,
                "records": records,
                "coverage": acc["hit_total"] / acc["active_total"]
                if acc["active_total"] else None,
                "weighted_coverage": acc["weight_hit"] / acc["weight_total"]
                if acc["weight_total"] else None,
                "active_experts_mean": acc["active_total"] / records
                if records else None,
                "miss_experts_mean": acc["miss_total"] / records
                if records else None,
                "full_event_covered_rate": acc["full_covered"] / records
                if records else None,
            }
        )
    rows["_layer_rows"] = layer_rows
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")}
                  for row in rows]
    fieldnames = list(clean_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--score-mode",
                        choices=("weighted", "event_count", "token_count"),
                        default="weighted")
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument("--around-min", type=int, default=24)
    parser.add_argument("--around-max", type=int, default=40)
    parser.add_argument("--expert-bytes", type=int, default=4718592)
    parser.add_argument("--fallback-num-experts", type=int, default=128)
    args = parser.parse_args()

    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")
    if not 0.0 < args.split_fraction < 1.0:
        raise SystemExit("--split-fraction must be in (0, 1)")

    events = parse_events(args.trace, args.fallback_num_experts)
    if not events:
        raise SystemExit(f"no MoE routing events parsed from {args.trace}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    num_experts = max([args.fallback_num_experts] + [event.num_experts for event in events])
    layers = sorted({event.layer_id for event in events})
    split_index = max(1, min(len(events) - 1, int(len(events) * args.split_fraction)))
    train_events = events[:split_index]
    test_events = events[split_index:]

    all_scores = score_hotness(events, args.score_mode)
    train_scores = score_hotness(train_events, args.score_mode)
    same_hotsets = build_hotsets(all_scores, layers, num_experts, args.k)
    train_hotsets = build_hotsets(train_scores, layers, num_experts, args.k)

    buckets = ["exact_32", "around", "9_32", "17_48", "33_128", "gt_512", "all"]
    rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    for label, eval_events, hotsets in (
        ("same_trace_all_events", events, same_hotsets),
        ("train_test_heldout", test_events, train_hotsets),
    ):
        for bucket in buckets:
            subset = [
                event for event in eval_events
                if event_in_bucket(event, bucket, args.around_min, args.around_max)
            ]
            row = evaluate(
                events=subset,
                hotsets=hotsets,
                label=label,
                bucket=bucket,
                k=args.k,
                expert_bytes=args.expert_bytes,
            )
            layer_rows.extend(row.pop("_layer_rows"))
            rows.append(row)

    write_csv(args.out_dir / "top64_numtoken_overlap_summary.csv", rows)
    write_csv(args.out_dir / "top64_numtoken_overlap_layers.csv", layer_rows)

    num_token_counts = Counter(event.num_tokens for event in events)
    summary = {
        "trace": str(args.trace),
        "records": len(events),
        "train_records": len(train_events),
        "test_records": len(test_events),
        "layers": len(layers),
        "num_experts": num_experts,
        "k": args.k,
        "score_mode": args.score_mode,
        "around_min": args.around_min,
        "around_max": args.around_max,
        "expert_bytes": args.expert_bytes,
        "top_num_tokens": num_token_counts.most_common(32),
        "exact_32_records": num_token_counts.get(32, 0),
        "around_records": sum(
            count for num_tokens, count in num_token_counts.items()
            if args.around_min <= num_tokens <= args.around_max
        ),
        "summary_csv": str(args.out_dir / "top64_numtoken_overlap_summary.csv"),
        "layers_csv": str(args.out_dir / "top64_numtoken_overlap_layers.csv"),
    }
    (args.out_dir / "top64_numtoken_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    display_buckets = ["exact_32", "around", "9_32", "17_48", "all"]
    lines = [
        "# MoE Top-64 Num-Token Overlap",
        "",
        f"- trace: `{args.trace}`",
        f"- records: `{len(events)}`",
        f"- train/test split: `{len(train_events)}` / `{len(test_events)}`",
        f"- layers: `{len(layers)}`",
        f"- experts_per_layer: `{num_experts}`",
        f"- hotset: `per-layer top-{args.k}`",
        f"- score_mode: `{args.score_mode}`",
        f"- around bucket: `{args.around_min} <= num_tokens <= {args.around_max}`",
        f"- expert bytes assumption: `{args.expert_bytes}` per role",
        "",
        "## Num Tokens",
        "",
        "| num_tokens | records |",
        "|---:|---:|",
    ]
    for num_tokens, count in num_token_counts.most_common(24):
        lines.append(f"| {num_tokens} | {count} |")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| label | bucket | records | steps | coverage | weighted_coverage | event_cov_p50 | event_cov_p90 | active_mean | miss_mean | miss_p90 | full_event_rate | miss MiB/step two-role mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["bucket"] not in display_buckets:
            continue
        lines.append(
            "| {label} | {bucket} | {records} | {steps} | {coverage} | {weighted} | {event_cov_p50} | {event_cov_p90} | {active} | {miss_mean} | {miss_p90} | {full} | {miss_mib} |".format(
                label=row["label"],
                bucket=row["bucket"],
                records=row["records"],
                steps=row["steps"],
                coverage=fmt(row["coverage"]),
                weighted=fmt(row["weighted_coverage"]),
                event_cov_p50=fmt(row["event_coverage_p50"]),
                event_cov_p90=fmt(row["event_coverage_p90"]),
                active=fmt(row["active_experts_mean"]),
                miss_mean=fmt(row["miss_experts_mean"]),
                miss_p90=fmt(row["miss_experts_p90"]),
                full=fmt(row["full_event_covered_rate"]),
                miss_mib=fmt(row["miss_mib_per_step_two_roles_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- summary_csv: `{args.out_dir / 'top64_numtoken_overlap_summary.csv'}`",
            f"- layer_csv: `{args.out_dir / 'top64_numtoken_overlap_layers.csv'}`",
            f"- summary_json: `{args.out_dir / 'top64_numtoken_overlap_summary.json'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (args.out_dir / "top64_numtoken_overlap_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
