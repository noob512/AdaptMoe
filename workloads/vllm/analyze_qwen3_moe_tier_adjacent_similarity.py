#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TIERS = ("hot", "warm", "cold")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(q * (len(ordered) - 1))
    return float(ordered[idx])


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def iter_token_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "qwen3_moe_token_expert_trace":
                continue

            layer_id = record.get("layer_id")
            if layer_id is None:
                continue
            layer_id = int(layer_id)
            request_ids = list(record.get("request_ids") or [])
            token_positions = list(record.get("token_positions") or [])
            topk_ids = list(record.get("topk_ids") or [])

            for token_idx, experts_raw in enumerate(topk_ids):
                if token_idx >= len(request_ids) or token_idx >= len(token_positions):
                    continue
                request_id = request_ids[token_idx]
                if request_id is None:
                    continue
                try:
                    token_position = int(token_positions[token_idx])
                except (TypeError, ValueError):
                    continue
                if token_position < 0:
                    continue

                experts = {
                    int(expert_id)
                    for expert_id in experts_raw
                    if int(expert_id) >= 0
                }
                if not experts:
                    continue
                yield str(request_id), token_position, layer_id, experts


def build_layer_tiers(
    path: Path,
) -> tuple[dict[int, dict[str, set[int]]], dict[int, Counter[int]]]:
    counters: dict[int, Counter[int]] = defaultdict(Counter)
    for _request_id, _token_position, layer_id, experts in iter_token_rows(path):
        counters[layer_id].update(experts)

    tiers: dict[int, dict[str, set[int]]] = {}
    for layer_id, counter in counters.items():
        ordered = [expert_id for expert_id, _count in counter.most_common()]
        if not ordered:
            tiers[layer_id] = {tier: set() for tier in TIERS}
            continue

        n = len(ordered)
        hot_end = (n + 2) // 3
        warm_end = hot_end + ((n - hot_end + 1) // 2)
        tiers[layer_id] = {
            "hot": set(ordered[:hot_end]),
            "warm": set(ordered[hot_end:warm_end]),
            "cold": set(ordered[warm_end:]),
        }
    return tiers, counters


def empty_stats() -> dict[str, Any]:
    return {
        "eligible_pairs": 0,
        "any_overlap_pairs": 0,
        "jaccards": [],
        "prev_selected": [],
        "curr_selected": [],
    }


def analyze(path: Path) -> dict[str, Any]:
    tiers, counters = build_layer_tiers(path)
    overall = {tier: empty_stats() for tier in TIERS}
    by_layer: dict[int, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {tier: empty_stats() for tier in TIERS}
    )
    last_seen: dict[tuple[str, int], tuple[int, set[int]]] = {}
    total_adjacent_pairs = 0
    skipped_non_adjacent = 0

    for request_id, token_position, layer_id, experts in iter_token_rows(path):
        key = (request_id, layer_id)
        previous = last_seen.get(key)
        if previous is not None:
            prev_position, prev_experts = previous
            if token_position == prev_position + 1:
                total_adjacent_pairs += 1
                layer_tiers = tiers.get(layer_id, {})
                for tier in TIERS:
                    tier_experts = layer_tiers.get(tier, set())
                    prev_tier = prev_experts & tier_experts
                    curr_tier = experts & tier_experts
                    if not prev_tier and not curr_tier:
                        continue
                    score = jaccard(prev_tier, curr_tier)
                    for bucket in (overall[tier], by_layer[layer_id][tier]):
                        bucket["eligible_pairs"] += 1
                        bucket["any_overlap_pairs"] += int(bool(prev_tier & curr_tier))
                        bucket["jaccards"].append(score)
                        bucket["prev_selected"].append(float(len(prev_tier)))
                        bucket["curr_selected"].append(float(len(curr_tier)))
            elif token_position > prev_position + 1:
                skipped_non_adjacent += 1
        current_previous = last_seen.get(key)
        if current_previous is None or token_position > current_previous[0]:
            last_seen[key] = (token_position, experts)

    def summarize_stats(stats: dict[str, Any]) -> dict[str, Any]:
        eligible = int(stats["eligible_pairs"])
        return {
            "eligible_pairs": eligible,
            "mean_jaccard": mean(stats["jaccards"]),
            "p50_jaccard": percentile(stats["jaccards"], 0.50),
            "p95_jaccard": percentile(stats["jaccards"], 0.95),
            "any_overlap_rate": (
                float(stats["any_overlap_pairs"]) / float(eligible)
                if eligible
                else 0.0
            ),
            "mean_prev_selected": mean(stats["prev_selected"]),
            "mean_curr_selected": mean(stats["curr_selected"]),
        }

    layer_rows = []
    for layer_id in sorted(by_layer):
        counter = counters[layer_id]
        row: dict[str, Any] = {
            "layer_id": layer_id,
            "unique_experts": len(counter),
            "hot_experts": len(tiers[layer_id]["hot"]),
            "warm_experts": len(tiers[layer_id]["warm"]),
            "cold_experts": len(tiers[layer_id]["cold"]),
            "hot_top": counters[layer_id].most_common(8),
        }
        for tier in TIERS:
            summary = summarize_stats(by_layer[layer_id][tier])
            for key, value in summary.items():
                row[f"{tier}_{key}"] = value
        layer_rows.append(row)

    return {
        "trace": str(path),
        "tiering": "per-layer equal-count terciles by expert token-hit count",
        "total_adjacent_pairs": total_adjacent_pairs,
        "skipped_non_adjacent_updates": skipped_non_adjacent,
        "layer_count": len(counters),
        "overall": {tier: summarize_stats(overall[tier]) for tier in TIERS},
        "layers": layer_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "layer_id",
        "unique_experts",
        "hot_experts",
        "warm_experts",
        "cold_experts",
        "hot_top",
    ]
    for tier in TIERS:
        fields.extend(
            [
                f"{tier}_eligible_pairs",
                f"{tier}_mean_jaccard",
                f"{tier}_p50_jaccard",
                f"{tier}_p95_jaccard",
                f"{tier}_any_overlap_rate",
                f"{tier}_mean_prev_selected",
                f"{tier}_mean_curr_selected",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["hot_top"] = " ".join(
                f"{expert}:{count}" for expert, count in out["hot_top"]
            )
            writer.writerow(out)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3 MoE Tiered Adjacent-Token Similarity",
        "",
        f"- trace: `{summary['trace']}`",
        f"- tiering: `{summary['tiering']}`",
        f"- total adjacent token pairs: `{summary['total_adjacent_pairs']}`",
        f"- layer count: `{summary['layer_count']}`",
        "",
        "## Overall",
        "",
        "| tier | eligible pairs | mean Jaccard | p50 | p95 | any-overlap rate | mean selected/token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        row = summary["overall"][tier]
        selected = (float(row["mean_prev_selected"]) + float(row["mean_curr_selected"])) / 2.0
        lines.append(
            "| {tier} | {pairs} | {mean} | {p50} | {p95} | {overlap} | {selected} |".format(
                tier=tier,
                pairs=row["eligible_pairs"],
                mean=fmt(row["mean_jaccard"]),
                p50=fmt(row["p50_jaccard"]),
                p95=fmt(row["p95_jaccard"]),
                overlap=fmt(row["any_overlap_rate"]),
                selected=fmt(selected),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Layer Summary",
            "",
            "| layer | hot J | warm J | cold J | hot overlap | warm overlap | cold overlap | hot top experts |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["layers"]:
        hot_top = " ".join(f"{expert}:{count}" for expert, count in row["hot_top"][:6])
        lines.append(
            "| {layer} | {hot_j} | {warm_j} | {cold_j} | {hot_o} | {warm_o} | {cold_o} | {hot_top} |".format(
                layer=row["layer_id"],
                hot_j=fmt(row["hot_mean_jaccard"]),
                warm_j=fmt(row["warm_mean_jaccard"]),
                cold_j=fmt(row["cold_mean_jaccard"]),
                hot_o=fmt(row["hot_any_overlap_rate"]),
                warm_o=fmt(row["warm_any_overlap_rate"]),
                cold_o=fmt(row["cold_any_overlap_rate"]),
                hot_top=hot_top,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `eligible pairs` excludes adjacent-token pairs where neither token selected an expert from that tier.",
            "- `mean selected/token` is the average number of selected top-k experts that fall into the tier.",
            "- Higher tier Jaccard means adjacent tokens tend to reuse the same experts inside that hot/warm/cold tier.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze adjacent-token expert similarity by hot/warm/cold expert tiers."
    )
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_path = args.trace_jsonl
    prefix = trace_path.with_suffix("")
    output_md = args.output_md or prefix.with_name(prefix.name + "_tier_adjacent.md")
    output_json = args.output_json or prefix.with_name(prefix.name + "_tier_adjacent.json")
    output_csv = args.output_csv or prefix.with_name(prefix.name + "_tier_adjacent.csv")

    summary = analyze(trace_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_csv, summary["layers"])
    write_markdown(output_md, summary)
    print(output_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
