#!/usr/bin/env python3
"""Analyze vLLM MoE routing traces and replay hot expert policies.

The input is the JSONL produced by VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1.
Outputs are intentionally simple CSV/JSON/Markdown so the experiment shell
scripts can aggregate them without depending on pandas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Event:
    index: int
    layer_name: str
    layer_id: int | None
    phase: str
    step: int
    num_tokens: int
    active: frozenset[int]
    values: dict[int, float]
    total_weight: float


def parse_ks(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


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


def extract_layer_id(layer_name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)", layer_name)
    if not match:
        return None
    return int(match.group(1))


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


def classify_phase(record: dict, decode_max_tokens: int) -> str:
    num_tokens = int(record.get("num_tokens") or 0)
    return "decode" if num_tokens <= decode_max_tokens else "prefill"


def load_events(path: Path, decode_max_tokens: int) -> list[Event]:
    events: list[Event] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            layer_name = str(record.get("layer_name") or "")
            if not layer_name:
                continue
            counts = as_int_dict(record.get("expert_token_counts"))
            weights = as_float_dict(record.get("expert_weight_sums"))
            active_raw = record.get("active_experts")
            active: set[int] = set()
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
            values = weights if weights else {key: float(value) for key, value in counts.items()}
            for expert_id in active:
                values.setdefault(expert_id, 0.0)
            total_weight = sum(max(values.get(expert_id, 0.0), 0.0) for expert_id in active)
            if total_weight <= 0:
                total_weight = float(len(active))
                values = {expert_id: 1.0 for expert_id in active}
            events.append(
                Event(
                    index=index,
                    layer_name=layer_name,
                    layer_id=extract_layer_id(layer_name),
                    phase=classify_phase(record, decode_max_tokens),
                    step=int(record.get("step") or 0),
                    num_tokens=int(record.get("num_tokens") or 0),
                    active=frozenset(active),
                    values=values,
                    total_weight=total_weight,
                )
            )
    return events


def topk(scores: dict[int, float], k: int) -> set[int]:
    if k <= 0:
        return set()
    return {
        expert_id
        for expert_id, _ in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:k]
    }


def aggregate_scores(events: Iterable[Event]) -> dict[str, dict[int, float]]:
    by_layer: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for event in events:
        layer_scores = by_layer[event.layer_name]
        for expert_id, value in event.values.items():
            layer_scores[expert_id] += value
    return {layer: dict(scores) for layer, scores in by_layer.items()}


def predictions_oracle(events: list[Event], k: int) -> tuple[dict[int, set[int]], set[int]]:
    return {event.index: topk(event.values, k) for event in events}, set()


def predictions_layer_static_full(
    events: list[Event], k: int
) -> tuple[dict[int, set[int]], set[int]]:
    scores = aggregate_scores(events)
    hotsets = {layer: topk(layer_scores, k) for layer, layer_scores in scores.items()}
    return {event.index: set(hotsets.get(event.layer_name, set())) for event in events}, set()


def predictions_layer_static_prefix(
    events: list[Event], k: int, warmup_records: int
) -> tuple[dict[int, set[int]], set[int]]:
    by_layer: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        by_layer[event.layer_name].append(event)
    scores: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    skip: set[int] = set()
    for layer_events in by_layer.values():
        for event in layer_events[:warmup_records]:
            skip.add(event.index)
            for expert_id, value in event.values.items():
                scores[event.layer_name][expert_id] += value
    hotsets = {layer: topk(dict(layer_scores), k) for layer, layer_scores in scores.items()}
    return {
        event.index: set(hotsets.get(event.layer_name, set()))
        for event in events
        if event.index not in skip
    }, skip


def predictions_last_step(events: list[Event], k: int) -> tuple[dict[int, set[int]], set[int]]:
    last_scores: dict[str, dict[int, float]] = {}
    predictions: dict[int, set[int]] = {}
    for event in events:
        predictions[event.index] = topk(last_scores.get(event.layer_name, {}), k)
        last_scores[event.layer_name] = dict(event.values)
    return predictions, set()


def predictions_ewma(
    events: list[Event], k: int, alpha: float
) -> tuple[dict[int, set[int]], set[int]]:
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    predictions: dict[int, set[int]] = {}
    for event in events:
        layer_scores = scores[event.layer_name]
        predictions[event.index] = topk(layer_scores, k)
        for expert_id in list(layer_scores):
            layer_scores[expert_id] *= 1.0 - alpha
        for expert_id, value in event.values.items():
            layer_scores[expert_id] = layer_scores.get(expert_id, 0.0) + alpha * value
    return predictions, set()


def predictions_hybrid(
    events: list[Event],
    k: int,
    alpha: float,
    recent_window: int,
    recent_bonus: float,
) -> tuple[dict[int, set[int]], set[int]]:
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    last_seen: dict[str, dict[int, int]] = defaultdict(dict)
    layer_positions: dict[str, int] = defaultdict(int)
    predictions: dict[int, set[int]] = {}
    for event in events:
        pos = layer_positions[event.layer_name]
        layer_scores = scores[event.layer_name]
        scored: dict[int, float] = dict(layer_scores)
        for expert_id, seen_pos in last_seen[event.layer_name].items():
            age = pos - seen_pos
            if age <= recent_window:
                scored[expert_id] = scored.get(expert_id, 0.0) + recent_bonus / (1.0 + age)
        predictions[event.index] = topk(scored, k)
        for expert_id in list(layer_scores):
            layer_scores[expert_id] *= 1.0 - alpha
        for expert_id, value in event.values.items():
            layer_scores[expert_id] = layer_scores.get(expert_id, 0.0) + alpha * value
            last_seen[event.layer_name][expert_id] = pos
        layer_positions[event.layer_name] += 1
    return predictions, set()


def evaluate(
    events: list[Event],
    predictions: dict[int, set[int]],
    *,
    strategy: str,
    split: str,
    k: int,
    expert_bytes: int,
    skip: set[int] | None = None,
) -> dict[str, object]:
    skip = skip or set()
    active_total = 0
    hit_total = 0
    weighted_total = 0.0
    weighted_hit = 0.0
    pred_total = 0
    false_total = 0
    misses: list[float] = []
    pred_counts: list[float] = []
    bytes_per_step: list[float] = []
    evaluated = 0

    for event in events:
        if event.index in skip:
            continue
        pred = predictions.get(event.index, set())
        active = set(event.active)
        hits = active & pred
        active_total += len(active)
        hit_total += len(hits)
        weighted_total += event.total_weight
        weighted_hit += sum(max(event.values.get(expert_id, 0.0), 0.0) for expert_id in hits)
        pred_total += len(pred)
        false_total += len(pred - active)
        misses.append(float(len(active - pred)))
        pred_counts.append(float(len(pred)))
        bytes_per_step.append(float(len(pred) * expert_bytes))
        evaluated += 1

    return {
        "split": split,
        "strategy": strategy,
        "k": k,
        "records": evaluated,
        "hit_rate": hit_total / active_total if active_total else None,
        "weighted_hit_rate": weighted_hit / weighted_total if weighted_total else None,
        "overprefetch_ratio": false_total / pred_total if pred_total else None,
        "active_expert_records": active_total,
        "hit_expert_records": hit_total,
        "predicted_expert_records": pred_total,
        "false_prefetch_records": false_total,
        "miss_experts_mean": mean(misses),
        "miss_experts_p50": percentile(misses, 50),
        "miss_experts_p90": percentile(misses, 90),
        "miss_experts_p99": percentile(misses, 99),
        "prefetch_experts_mean": mean(pred_counts),
        "prefetch_bytes_per_step_mean": mean(bytes_per_step),
        "prefetch_mib_per_step_mean": (
            mean(bytes_per_step) / (1024 * 1024) if mean(bytes_per_step) is not None else None
        ),
    }


def active_stats(events: list[Event], split: str) -> dict[str, object]:
    counts = [float(len(event.active)) for event in events]
    return {
        "split": split,
        "records": len(events),
        "active_expert_mean": mean(counts),
        "active_expert_p50": percentile(counts, 50),
        "active_expert_p90": percentile(counts, 90),
        "active_expert_p99": percentile(counts, 99),
        "num_tokens_mean": mean(float(event.num_tokens) for event in events),
    }


def layer_stats(events: list[Event], split: str) -> list[dict[str, object]]:
    by_layer: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        by_layer[event.layer_name].append(event)
    rows = []
    for layer_name, layer_events in sorted(by_layer.items()):
        counts = [float(len(event.active)) for event in layer_events]
        rows.append(
            {
                "split": split,
                "layer_name": layer_name,
                "layer_id": layer_events[0].layer_id,
                "records": len(layer_events),
                "active_expert_mean": mean(counts),
                "active_expert_p50": percentile(counts, 50),
                "active_expert_p90": percentile(counts, 90),
                "active_expert_p99": percentile(counts, 99),
            }
        )
    return rows


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--k-values", default="8,16,32,64,96,128")
    parser.add_argument("--decode-max-tokens", type=int, default=1)
    parser.add_argument("--expert-bytes", type=int, default=4718592)
    parser.add_argument("--ewma-alpha", type=float, default=0.2)
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--recent-bonus", type=float, default=1.0)
    parser.add_argument("--static-warmup-records", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.trace, args.decode_max_tokens)
    k_values = parse_ks(args.k_values)
    splits = {
        "all": events,
        "prefill": [event for event in events if event.phase == "prefill"],
        "decode": [event for event in events if event.phase == "decode"],
    }

    metric_rows: list[dict[str, object]] = []
    active_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    for split, split_events in splits.items():
        active_rows.append(active_stats(split_events, split))
        layer_rows.extend(layer_stats(split_events, split))
        for k in k_values:
            strategy_builders = [
                ("oracle", lambda evs, kk: predictions_oracle(evs, kk)),
                ("layer_static_full", lambda evs, kk: predictions_layer_static_full(evs, kk)),
                (
                    f"layer_static_prefix{args.static_warmup_records}",
                    lambda evs, kk: predictions_layer_static_prefix(
                        evs, kk, args.static_warmup_records
                    ),
                ),
                ("last_step", lambda evs, kk: predictions_last_step(evs, kk)),
                (
                    f"ewma_alpha{args.ewma_alpha:g}",
                    lambda evs, kk: predictions_ewma(evs, kk, args.ewma_alpha),
                ),
                (
                    "hybrid_recent_ewma",
                    lambda evs, kk: predictions_hybrid(
                        evs,
                        kk,
                        args.ewma_alpha,
                        args.recent_window,
                        args.recent_bonus,
                    ),
                ),
            ]
            for strategy, builder in strategy_builders:
                predictions, skip = builder(split_events, k)
                metric_rows.append(
                    evaluate(
                        split_events,
                        predictions,
                        strategy=strategy,
                        split=split,
                        k=k,
                        expert_bytes=args.expert_bytes,
                        skip=skip,
                    )
                )

    summary = {
        "trace": str(args.trace),
        "records": len(events),
        "decode_max_tokens": args.decode_max_tokens,
        "expert_bytes": args.expert_bytes,
        "active_stats": active_rows,
        "metrics": metric_rows,
    }
    (args.out_dir / "moe_routing_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    metric_fields = list(metric_rows[0].keys()) if metric_rows else []
    with (args.out_dir / "moe_routing_replay_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    active_fields = list(active_rows[0].keys()) if active_rows else []
    with (args.out_dir / "moe_routing_active_stats.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=active_fields)
        writer.writeheader()
        writer.writerows(active_rows)

    layer_fields = list(layer_rows[0].keys()) if layer_rows else []
    with (args.out_dir / "moe_routing_layer_stats.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_fields)
        writer.writeheader()
        writer.writerows(layer_rows)

    interesting = [
        row
        for row in metric_rows
        if row["split"] == "decode"
        and row["k"] in (32, 64, 96, 128)
        and row["strategy"] in ("layer_static_full", "last_step", "ewma_alpha0.2", "hybrid_recent_ewma")
    ]
    lines = [
        "# MoE Routing Hotset Replay",
        "",
        f"- trace: `{args.trace}`",
        f"- records: `{len(events)}`",
        f"- decode_max_tokens: `{args.decode_max_tokens}`",
        f"- expert_bytes: `{args.expert_bytes}`",
        "",
        "## Active Expert Stats",
        "",
        "| split | records | mean | p50 | p90 | p99 | num_tokens_mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in active_rows:
        lines.append(
            "| {split} | {records} | {mean} | {p50} | {p90} | {p99} | {tokens} |".format(
                split=row["split"],
                records=row["records"],
                mean=fmt(row["active_expert_mean"]),
                p50=fmt(row["active_expert_p50"]),
                p90=fmt(row["active_expert_p90"]),
                p99=fmt(row["active_expert_p99"]),
                tokens=fmt(row["num_tokens_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decode Hotset Signals",
            "",
            "| strategy | K | hit_rate | weighted_hit_rate | overprefetch | miss_mean | MiB/step |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in interesting:
        lines.append(
            "| {strategy} | {k} | {hit} | {weighted} | {over} | {miss} | {mib} |".format(
                strategy=row["strategy"],
                k=row["k"],
                hit=fmt(row["hit_rate"]),
                weighted=fmt(row["weighted_hit_rate"]),
                over=fmt(row["overprefetch_ratio"]),
                miss=fmt(row["miss_experts_mean"]),
                mib=fmt(row["prefetch_mib_per_step_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- metrics_csv: `{args.out_dir / 'moe_routing_replay_metrics.csv'}`",
            f"- active_csv: `{args.out_dir / 'moe_routing_active_stats.csv'}`",
            f"- layer_csv: `{args.out_dir / 'moe_routing_layer_stats.csv'}`",
            f"- summary_json: `{args.out_dir / 'moe_routing_replay_summary.json'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (args.out_dir / "moe_routing_replay_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
