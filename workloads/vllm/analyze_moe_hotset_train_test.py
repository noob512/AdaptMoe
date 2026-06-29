#!/usr/bin/env python3
"""Train/test validation for per-layer MoE hot expert residency plans.

This is stricter than replaying a hotset on the same trace:

  1. Sort routing events by trace order.
  2. Use the first split fraction as train.
  3. Build per-layer top-K experts from train only.
  4. Evaluate coverage and miss bytes on the held-out test suffix.

The generated plan JSON is compatible with the Stage M weight plan format used
by build_stage_m_hot_cold_weight_plan.py / run_stage_m_phase_f_check.sh.
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
    layer_name: str
    layer_id: int
    step: int
    num_tokens: int
    active: frozenset[int]
    values: dict[int, float]
    total_weight: float
    global_num_experts: int


def parse_ks(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", ",").split(",") if item.strip()]


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
        return f"{value:.4g}"
    return str(value)


def layer_id_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


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


def num_token_bucket(num_tokens: int) -> str:
    if num_tokens == 1:
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


BUCKET_ORDER = ["all", "1", "2-8", "9-32", "33-128", "129-512", ">512"]


def load_events(path: Path, fallback_num_experts: int) -> list[Event]:
    events: list[Event] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            layer_name = str(record.get("layer_name") or "")
            layer_id = layer_id_from_name(layer_name)
            if layer_id is None:
                continue

            counts = as_int_dict(record.get("expert_token_counts"))
            weights = as_float_dict(record.get("expert_weight_sums"))

            active: set[int] = set()
            active_raw = record.get("active_experts")
            if isinstance(active_raw, list):
                for item in active_raw:
                    try:
                        active.add(int(item))
                    except (TypeError, ValueError):
                        continue
            if not active:
                active = set(counts)
            if not active:
                continue

            values = weights if weights else {
                expert_id: float(count) for expert_id, count in counts.items()
            }
            for expert_id in active:
                values.setdefault(expert_id, 0.0)

            total_weight = sum(max(values.get(expert_id, 0.0), 0.0)
                               for expert_id in active)
            if total_weight <= 0:
                total_weight = float(len(active))
                values = {expert_id: 1.0 for expert_id in active}

            try:
                global_num_experts = int(record.get("global_num_experts") or 0)
            except (TypeError, ValueError):
                global_num_experts = 0
            if global_num_experts <= 0:
                global_num_experts = fallback_num_experts

            events.append(
                Event(
                    index=index,
                    layer_name=layer_name,
                    layer_id=layer_id,
                    step=int(record.get("step") or 0),
                    num_tokens=int(record.get("num_tokens") or 0),
                    active=frozenset(active),
                    values=values,
                    total_weight=total_weight,
                    global_num_experts=global_num_experts,
                )
            )
    return events


def score_events(events: Iterable[Event], score_mode: str) -> dict[int, dict[int, float]]:
    by_layer: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for event in events:
        layer_scores = by_layer[event.layer_id]
        if score_mode == "count":
            for expert_id in event.active:
                layer_scores[expert_id] += 1.0
        elif score_mode == "token_count":
            for expert_id in event.active:
                layer_scores[expert_id] += max(event.values.get(expert_id, 0.0), 0.0)
        elif score_mode == "weighted":
            for expert_id in event.active:
                layer_scores[expert_id] += max(event.values.get(expert_id, 0.0), 0.0)
        else:
            raise ValueError(f"unknown score mode: {score_mode}")
    return {layer_id: dict(scores) for layer_id, scores in by_layer.items()}


def build_hotsets(
    *,
    scores_by_layer: dict[int, dict[int, float]],
    layers: list[int],
    num_experts: int,
    k: int,
) -> dict[int, set[int]]:
    hotsets: dict[int, set[int]] = {}
    all_experts = list(range(num_experts))
    for layer_id in layers:
        scores = scores_by_layer.get(layer_id, {})
        ranked = sorted(
            all_experts,
            key=lambda expert_id: (-scores.get(expert_id, 0.0), expert_id),
        )
        hotsets[layer_id] = set(ranked[:max(k, 0)])
    return hotsets


def event_subset(events: list[Event], bucket: str) -> list[Event]:
    if bucket == "all":
        return events
    return [event for event in events if num_token_bucket(event.num_tokens) == bucket]


def stats(values: list[float]) -> dict[str, object]:
    return {
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def evaluate(
    *,
    events: list[Event],
    hotsets: dict[int, set[int]],
    split: str,
    bucket: str,
    k: int,
    expert_bytes: int,
) -> dict[str, object]:
    active_total = 0
    hit_total = 0
    weighted_total = 0.0
    weighted_hit = 0.0
    miss_counts: list[float] = []
    miss_bytes_per_layer: list[float] = []
    active_counts: list[float] = []
    step_miss_bytes: dict[int, float] = defaultdict(float)
    step_miss_experts: dict[int, float] = defaultdict(float)

    for event in events:
        pred = hotsets.get(event.layer_id, set())
        active = set(event.active)
        hits = active & pred
        misses = active - pred

        active_total += len(active)
        hit_total += len(hits)
        weighted_total += event.total_weight
        weighted_hit += sum(max(event.values.get(expert_id, 0.0), 0.0)
                            for expert_id in hits)
        miss_counts.append(float(len(misses)))
        active_counts.append(float(len(active)))
        miss_bytes = float(len(misses) * expert_bytes)
        miss_bytes_per_layer.append(miss_bytes)
        step_miss_bytes[event.step] += miss_bytes
        step_miss_experts[event.step] += float(len(misses))

    miss_stat = stats(miss_counts)
    active_stat = stats(active_counts)
    miss_bytes_layer_stat = stats([value / (1024 * 1024) for value in miss_bytes_per_layer])
    miss_bytes_step_stat = stats([
        value / (1024 * 1024) for value in step_miss_bytes.values()
    ])
    miss_experts_step_stat = stats(list(step_miss_experts.values()))

    return {
        "split": split,
        "bucket": bucket,
        "k": k,
        "records": len(events),
        "steps": len(step_miss_bytes),
        "coverage": hit_total / active_total if active_total else None,
        "weighted_coverage": weighted_hit / weighted_total if weighted_total else None,
        "active_expert_mean": active_stat["mean"],
        "active_expert_p50": active_stat["p50"],
        "active_expert_p90": active_stat["p90"],
        "active_expert_p99": active_stat["p99"],
        "miss_experts_per_layer_mean": miss_stat["mean"],
        "miss_experts_per_layer_p50": miss_stat["p50"],
        "miss_experts_per_layer_p90": miss_stat["p90"],
        "miss_experts_per_layer_p99": miss_stat["p99"],
        "miss_mib_per_layer_mean": miss_bytes_layer_stat["mean"],
        "miss_mib_per_layer_p90": miss_bytes_layer_stat["p90"],
        "miss_mib_per_step_mean": miss_bytes_step_stat["mean"],
        "miss_mib_per_step_p90": miss_bytes_step_stat["p90"],
        "miss_mib_per_step_p99": miss_bytes_step_stat["p99"],
        "miss_experts_per_step_mean": miss_experts_step_stat["mean"],
        "miss_experts_per_step_p90": miss_experts_step_stat["p90"],
        "miss_experts_per_step_p99": miss_experts_step_stat["p99"],
    }


def write_plan(
    *,
    path: Path,
    report_path: Path,
    trace: Path,
    train_records: int,
    test_records: int,
    scores_by_layer: dict[int, dict[int, float]],
    layers: list[int],
    num_experts: int,
    k: int,
    score_mode: str,
    split_fraction: float,
    target_roles: list[str],
) -> None:
    prefetch_plan = []
    offload_plan = []
    all_experts = list(range(num_experts))

    for layer_id in layers:
        scores = scores_by_layer.get(layer_id, {})
        ranked = sorted(
            all_experts,
            key=lambda expert_id: (-scores.get(expert_id, 0.0), expert_id),
        )
        hot = set(ranked[:max(k, 0)])
        for expert_id in ranked[:max(k, 0)]:
            prefetch_plan.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "score": scores.get(expert_id, 0.0),
                    "reason": "train_test_top_hot",
                }
            )
        for expert_id in all_experts:
            if expert_id in hot:
                continue
            offload_plan.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "score": scores.get(expert_id, 0.0),
                    "reason": "train_test_non_hot",
                }
            )

    plan = {
        "metadata": {
            "policy": "train_test_static_hotset_v1",
            "routing_trace": str(trace),
            "split_fraction": split_fraction,
            "train_records": train_records,
            "test_records": test_records,
            "score_mode": score_mode,
            "hot_experts_per_layer": k,
            "cold_experts_per_layer": max(num_experts - k, 0),
            "num_layers": len(layers),
            "num_experts": num_experts,
            "target_roles": target_roles,
        },
        "prefetch_plan": prefetch_plan,
        "offload_plan": offload_plan,
        "prefetch_plan_records": len(prefetch_plan),
        "offload_plan_records": len(offload_plan),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Train/Test Static Hotset Weight Plan",
        "",
        f"- trace: `{trace}`",
        f"- split_fraction: `{split_fraction}`",
        f"- train_records: `{train_records}`",
        f"- test_records: `{test_records}`",
        f"- score_mode: `{score_mode}`",
        f"- hot_experts_per_layer: `{k}`",
        f"- layers: `{len(layers)}`",
        f"- experts_per_layer: `{num_experts}`",
        f"- prefetch_plan_records: `{len(prefetch_plan)}`",
        f"- offload_plan_records: `{len(offload_plan)}`",
        "",
        "## Layer Samples",
        "",
    ]
    for layer_id in layers[:12]:
        scores = scores_by_layer.get(layer_id, {})
        ranked = sorted(all_experts, key=lambda expert_id: (-scores.get(expert_id, 0.0), expert_id))
        lines.append(f"### Layer {layer_id}")
        lines.append(f"- hot_sample: `{ranked[:8]}`")
        lines.append(f"- score_sample: `{[scores.get(expert_id, 0.0) for expert_id in ranked[:8]]}`")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--k-values", default="32,48,64,72,80,96")
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument("--score-mode", choices=("weighted", "count", "token_count"), default="weighted")
    parser.add_argument("--expert-bytes", type=int, default=4718592)
    parser.add_argument("--fallback-num-experts", type=int, default=128)
    parser.add_argument("--fallback-num-layers", type=int, default=48)
    parser.add_argument("--target-roles", default="moe_gate_up,moe_down")
    parser.add_argument("--write-plans", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.split_fraction < 1.0:
        raise SystemExit("--split-fraction must be in (0, 1)")
    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.trace, args.fallback_num_experts)
    if not events:
        raise SystemExit(f"no routing events parsed from {args.trace}")

    split_index = max(1, min(len(events) - 1, int(len(events) * args.split_fraction)))
    train = events[:split_index]
    test = events[split_index:]
    num_experts = max([args.fallback_num_experts] + [event.global_num_experts for event in events])
    layers = sorted(set(event.layer_id for event in events))
    if not layers:
        layers = list(range(args.fallback_num_layers))
    target_roles = [role.strip() for role in args.target_roles.split(",") if role.strip()]
    scores_by_layer = score_events(train, args.score_mode)
    k_values = parse_ks(args.k_values)

    rows: list[dict[str, object]] = []
    for k in k_values:
        hotsets = build_hotsets(
            scores_by_layer=scores_by_layer,
            layers=layers,
            num_experts=num_experts,
            k=k,
        )
        train_hot_mib = len(layers) * k * args.expert_bytes / (1024 * 1024)
        for split_name, split_events in (("train", train), ("test", test)):
            for bucket in BUCKET_ORDER:
                subset = event_subset(split_events, bucket)
                row = evaluate(
                    events=subset,
                    hotsets=hotsets,
                    split=split_name,
                    bucket=bucket,
                    k=k,
                    expert_bytes=args.expert_bytes,
                )
                row["hot_mib"] = train_hot_mib
                row["hot_gib"] = train_hot_mib / 1024.0
                rows.append(row)

        if args.write_plans:
            plan_path = args.out_dir / "plans" / f"train_static_hotset_k{k}.json"
            plan_report = args.out_dir / "plans" / f"train_static_hotset_k{k}.md"
            write_plan(
                path=plan_path,
                report_path=plan_report,
                trace=args.trace,
                train_records=len(train),
                test_records=len(test),
                scores_by_layer=scores_by_layer,
                layers=layers,
                num_experts=num_experts,
                k=k,
                score_mode=args.score_mode,
                split_fraction=args.split_fraction,
                target_roles=target_roles,
            )

    metrics_csv = args.out_dir / "moe_hotset_train_test_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "trace": str(args.trace),
        "records": len(events),
        "train_records": len(train),
        "test_records": len(test),
        "split_fraction": args.split_fraction,
        "score_mode": args.score_mode,
        "expert_bytes": args.expert_bytes,
        "num_layers": len(layers),
        "num_experts": num_experts,
        "k_values": k_values,
        "metrics_csv": str(metrics_csv),
    }
    (args.out_dir / "moe_hotset_train_test_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    interesting = [
        row for row in rows
        if row["split"] == "test"
        and row["bucket"] in ("all", "1", "2-8", "9-32", ">512")
    ]
    lines = [
        "# MoE Train/Test Hotset Coverage",
        "",
        f"- trace: `{args.trace}`",
        f"- records: `{len(events)}`",
        f"- train_records: `{len(train)}`",
        f"- test_records: `{len(test)}`",
        f"- split_fraction: `{args.split_fraction}`",
        f"- score_mode: `{args.score_mode}`",
        f"- expert_bytes: `{args.expert_bytes}`",
        f"- layers: `{len(layers)}`",
        f"- experts_per_layer: `{num_experts}`",
        "",
        "## Test Coverage",
        "",
        "| bucket | K | records | coverage | weighted_coverage | active_mean | miss/layer mean | miss/layer p90 | miss MiB/step mean | miss MiB/step p90 | hot GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in interesting:
        lines.append(
            "| {bucket} | {k} | {records} | {coverage} | {weighted} | {active} | {miss_mean} | {miss_p90} | {miss_step} | {miss_step_p90} | {hot_gib} |".format(
                bucket=row["bucket"],
                k=row["k"],
                records=row["records"],
                coverage=fmt(row["coverage"]),
                weighted=fmt(row["weighted_coverage"]),
                active=fmt(row["active_expert_mean"]),
                miss_mean=fmt(row["miss_experts_per_layer_mean"]),
                miss_p90=fmt(row["miss_experts_per_layer_p90"]),
                miss_step=fmt(row["miss_mib_per_step_mean"]),
                miss_step_p90=fmt(row["miss_mib_per_step_p90"]),
                hot_gib=fmt(row["hot_gib"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- metrics_csv: `{metrics_csv}`",
            f"- summary_json: `{args.out_dir / 'moe_hotset_train_test_summary.json'}`",
        ]
    )
    if args.write_plans:
        lines.append(f"- plans_dir: `{args.out_dir / 'plans'}`")
    report = "\n".join(lines) + "\n"
    (args.out_dir / "moe_hotset_train_test_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
