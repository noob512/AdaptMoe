#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TIERS = ("hot", "stable", "cold")


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
    return float(len(left & right)) / float(len(union)) if union else 0.0


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def event_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "qwen3_moe_token_expert_trace":
                count += 1
    return count


def iter_token_rows(path: Path):
    event_idx = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "qwen3_moe_token_expert_trace":
                continue
            event_idx += 1

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

                experts = []
                for expert_id in experts_raw:
                    expert_id = int(expert_id)
                    if expert_id >= 0:
                        experts.append(expert_id)
                if not experts:
                    continue
                yield event_idx, str(request_id), token_position, layer_id, experts


def ranked_experts(counter: Counter[int], num_experts: int) -> list[int]:
    return sorted(range(num_experts), key=lambda expert: (-counter[expert], expert))


def make_tiers(
    counter: Counter[int],
    num_experts: int,
    hot_size: int,
    stable_size: int,
    cold_size: int,
) -> dict[str, set[int]]:
    ordered = ranked_experts(counter, num_experts)
    return {
        "hot": set(ordered[:hot_size]),
        "stable": set(ordered[hot_size : hot_size + stable_size]),
        "cold": set(ordered[num_experts - cold_size :]),
    }


def ranks(counter: Counter[int], num_experts: int) -> dict[int, int]:
    return {
        expert: rank
        for rank, expert in enumerate(ranked_experts(counter, num_experts), start=1)
    }


def spearman_rank_corr(
    left: Counter[int],
    right: Counter[int],
    num_experts: int,
) -> float:
    if num_experts <= 1:
        return 0.0
    left_ranks = ranks(left, num_experts)
    right_ranks = ranks(right, num_experts)
    diff_sq = sum(
        (left_ranks[expert] - right_ranks[expert]) ** 2
        for expert in range(num_experts)
    )
    n = float(num_experts)
    return 1.0 - (6.0 * float(diff_sq)) / (n * (n * n - 1.0))


def tier_name(
    expert: int,
    tier_sets: dict[str, set[int]],
) -> str:
    for tier in TIERS:
        if expert in tier_sets[tier]:
            return tier
    return "other"


def summarize_window_stability(
    full_tiers: dict[str, set[int]],
    full_counter: Counter[int],
    counters: list[Counter[int]],
    num_experts: int,
    hot_size: int,
    stable_size: int,
    cold_size: int,
) -> dict[str, Any]:
    retentions: dict[str, list[float]] = {tier: [] for tier in TIERS}
    jaccards: dict[str, list[float]] = {tier: [] for tier in TIERS}
    adjacent_jaccards: dict[str, list[float]] = {tier: [] for tier in TIERS}
    rank_corrs = []
    transitions: dict[str, Counter[str]] = {
        tier: Counter() for tier in TIERS
    }
    prev_tiers = None

    for counter in counters:
        if not counter:
            continue
        current_tiers = make_tiers(
            counter, num_experts, hot_size, stable_size, cold_size
        )
        rank_corrs.append(spearman_rank_corr(full_counter, counter, num_experts))

        for tier in TIERS:
            current = current_tiers[tier]
            full = full_tiers[tier]
            retentions[tier].append(float(len(current & full)) / float(len(full)))
            jaccards[tier].append(jaccard(current, full))
            if prev_tiers is not None:
                adjacent_jaccards[tier].append(jaccard(current, prev_tiers[tier]))

            for expert in full:
                transitions[tier][tier_name(expert, current_tiers)] += 1

        prev_tiers = current_tiers

    transition_rates: dict[str, dict[str, float]] = {}
    for source_tier, counts in transitions.items():
        total = float(sum(counts.values()))
        transition_rates[source_tier] = {
            target_tier: (float(counts[target_tier]) / total if total else 0.0)
            for target_tier in TIERS
        }

    return {
        "window_count": len([counter for counter in counters if counter]),
        "rank_spearman_mean": mean(rank_corrs),
        "rank_spearman_p05": percentile(rank_corrs, 0.05),
        "rank_spearman_p50": percentile(rank_corrs, 0.50),
        "rank_spearman_p95": percentile(rank_corrs, 0.95),
        "transition_rates": transition_rates,
        **{
            f"{tier}_retention_mean": mean(retentions[tier])
            for tier in TIERS
        },
        **{
            f"{tier}_retention_p05": percentile(retentions[tier], 0.05)
            for tier in TIERS
        },
        **{
            f"{tier}_jaccard_mean": mean(jaccards[tier])
            for tier in TIERS
        },
        **{
            f"{tier}_adjacent_jaccard_mean": mean(adjacent_jaccards[tier])
            for tier in TIERS
        },
    }


def analyze(
    path: Path,
    num_experts: int,
    hot_size: int,
    stable_size: int,
    cold_size: int,
    temporal_windows: int,
    request_buckets: int,
) -> dict[str, Any]:
    if hot_size + stable_size + cold_size != num_experts:
        raise ValueError("hot_size + stable_size + cold_size must equal num_experts")

    total_events = event_count(path)
    events_per_window = max(1, (total_events + temporal_windows - 1) // temporal_windows)

    overall: dict[int, Counter[int]] = defaultdict(Counter)
    temporal: dict[int, list[Counter[int]]] = defaultdict(
        lambda: [Counter() for _ in range(temporal_windows)]
    )
    request: dict[int, list[Counter[int]]] = defaultdict(
        lambda: [Counter() for _ in range(request_buckets)]
    )
    token_rows_by_layer: Counter[int] = Counter()
    assignment_rows_by_layer: Counter[int] = Counter()
    request_ids = set()

    for event_idx, request_id, _token_position, layer_id, experts in iter_token_rows(path):
        window_idx = min(event_idx // events_per_window, temporal_windows - 1)
        bucket_idx = stable_bucket(request_id, request_buckets)
        request_ids.add(request_id)
        token_rows_by_layer[layer_id] += 1
        assignment_rows_by_layer[layer_id] += len(experts)
        for expert_id in experts:
            if expert_id >= num_experts:
                continue
            overall[layer_id][expert_id] += 1
            temporal[layer_id][window_idx][expert_id] += 1
            request[layer_id][bucket_idx][expert_id] += 1

    layer_rows = []
    for layer_id in sorted(overall):
        counter = overall[layer_id]
        full_tiers = make_tiers(
            counter, num_experts, hot_size, stable_size, cold_size
        )
        total_assignments = float(sum(counter.values()))
        ordered = ranked_experts(counter, num_experts)

        tier_hits = {
            tier: sum(counter[expert] for expert in full_tiers[tier])
            for tier in TIERS
        }
        tier_shares = {
            tier: (float(tier_hits[tier]) / total_assignments if total_assignments else 0.0)
            for tier in TIERS
        }
        size_fracs = {
            "hot": float(hot_size) / float(num_experts),
            "stable": float(stable_size) / float(num_experts),
            "cold": float(cold_size) / float(num_experts),
        }

        hot_boundary_gap = counter[ordered[hot_size - 1]] - counter[ordered[hot_size]]
        stable_boundary_gap = (
            counter[ordered[hot_size + stable_size - 1]]
            - counter[ordered[hot_size + stable_size]]
        )
        hot_boundary_rel = (
            float(hot_boundary_gap) / float(counter[ordered[hot_size - 1]])
            if counter[ordered[hot_size - 1]]
            else 0.0
        )
        stable_boundary_rel = (
            float(stable_boundary_gap)
            / float(counter[ordered[hot_size + stable_size - 1]])
            if counter[ordered[hot_size + stable_size - 1]]
            else 0.0
        )

        temporal_summary = summarize_window_stability(
            full_tiers,
            counter,
            temporal[layer_id],
            num_experts,
            hot_size,
            stable_size,
            cold_size,
        )
        request_summary = summarize_window_stability(
            full_tiers,
            counter,
            request[layer_id],
            num_experts,
            hot_size,
            stable_size,
            cold_size,
        )

        row: dict[str, Any] = {
            "layer_id": layer_id,
            "token_rows": int(token_rows_by_layer[layer_id]),
            "assignments": int(assignment_rows_by_layer[layer_id]),
            "top8": counter.most_common(8),
            "hot_hit_share": tier_shares["hot"],
            "stable_hit_share": tier_shares["stable"],
            "cold_hit_share": tier_shares["cold"],
            "hot_lift_vs_uniform": tier_shares["hot"] / size_fracs["hot"],
            "stable_lift_vs_uniform": tier_shares["stable"] / size_fracs["stable"],
            "cold_lift_vs_uniform": tier_shares["cold"] / size_fracs["cold"],
            "hot_boundary_gap": int(hot_boundary_gap),
            "stable_boundary_gap": int(stable_boundary_gap),
            "hot_boundary_rel_gap": hot_boundary_rel,
            "stable_boundary_rel_gap": stable_boundary_rel,
        }
        for prefix, summary in (
            ("temporal", temporal_summary),
            ("request", request_summary),
        ):
            for key, value in summary.items():
                if key == "transition_rates":
                    for source_tier in TIERS:
                        for target_tier in TIERS:
                            row[
                                f"{prefix}_{source_tier}_to_{target_tier}_rate"
                            ] = value[source_tier][target_tier]
                else:
                    row[f"{prefix}_{key}"] = value
        layer_rows.append(row)

    def agg(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in layer_rows]
        return {
            "mean": mean(values),
            "p05": percentile(values, 0.05),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }

    overall_summary = {
        key: agg(key)
        for key in [
            "hot_hit_share",
            "stable_hit_share",
            "cold_hit_share",
            "hot_lift_vs_uniform",
            "stable_lift_vs_uniform",
            "cold_lift_vs_uniform",
            "hot_boundary_rel_gap",
            "stable_boundary_rel_gap",
            "temporal_rank_spearman_mean",
            "request_rank_spearman_mean",
            "temporal_hot_retention_mean",
            "temporal_stable_retention_mean",
            "temporal_cold_retention_mean",
            "request_hot_retention_mean",
            "request_stable_retention_mean",
            "request_cold_retention_mean",
        ]
    }

    return {
        "trace": str(path),
        "tiering": {
            "num_experts": num_experts,
            "hot_size": hot_size,
            "stable_size": stable_size,
            "cold_size": cold_size,
            "rule": "per-layer experts sorted by token-hit count: top hot_size, next stable_size, bottom cold_size",
        },
        "temporal_windows": temporal_windows,
        "request_buckets": request_buckets,
        "token_trace_events": total_events,
        "requests": len(request_ids),
        "layers": layer_rows,
        "overall": overall_summary,
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "layer_id",
        "token_rows",
        "assignments",
        "hot_hit_share",
        "stable_hit_share",
        "cold_hit_share",
        "hot_lift_vs_uniform",
        "stable_lift_vs_uniform",
        "cold_lift_vs_uniform",
        "hot_boundary_rel_gap",
        "stable_boundary_rel_gap",
        "temporal_rank_spearman_mean",
        "request_rank_spearman_mean",
        "temporal_hot_retention_mean",
        "temporal_stable_retention_mean",
        "temporal_cold_retention_mean",
        "request_hot_retention_mean",
        "request_stable_retention_mean",
        "request_cold_retention_mean",
        "temporal_hot_to_hot_rate",
        "temporal_stable_to_stable_rate",
        "temporal_cold_to_cold_rate",
        "request_hot_to_hot_rate",
        "request_stable_to_stable_rate",
        "request_cold_to_cold_rate",
        "top8",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["top8"] = " ".join(
                f"{expert}:{count}" for expert, count in row["top8"]
            )
            writer.writerow(out)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["layers"]
    overall = summary["overall"]
    lines = [
        "# Qwen3 MoE Fixed 80/32/16 Tier Stability",
        "",
        f"- trace: `{summary['trace']}`",
        f"- token trace events: `{summary['token_trace_events']}`",
        f"- requests: `{summary['requests']}`",
        f"- temporal windows: `{summary['temporal_windows']}`",
        f"- request buckets: `{summary['request_buckets']}`",
        f"- tiering: `{summary['tiering']['rule']}`",
        "",
        "## Overall Across Layers",
        "",
        "| metric | mean | p05 | p50 | p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in [
        "hot_hit_share",
        "stable_hit_share",
        "cold_hit_share",
        "hot_lift_vs_uniform",
        "stable_lift_vs_uniform",
        "cold_lift_vs_uniform",
        "temporal_rank_spearman_mean",
        "request_rank_spearman_mean",
        "temporal_hot_retention_mean",
        "temporal_stable_retention_mean",
        "temporal_cold_retention_mean",
        "request_hot_retention_mean",
        "request_stable_retention_mean",
        "request_cold_retention_mean",
        "hot_boundary_rel_gap",
        "stable_boundary_rel_gap",
    ]:
        item = overall[key]
        lines.append(
            f"| {key} | {fmt(item['mean'])} | {fmt(item['p05'])} | {fmt(item['p50'])} | {fmt(item['p95'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Layer Summary",
            "",
            "| layer | hot share | stable share | cold share | temp hot keep | temp stable keep | temp cold keep | req hot keep | req stable keep | req cold keep | temp rank rho | req rank rho | boundary rel gaps | top experts |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        top = " ".join(f"{expert}:{count}" for expert, count in row["top8"][:6])
        lines.append(
            "| {layer} | {hot} | {stable} | {cold} | {th} | {ts} | {tc} | {rh} | {rs} | {rc} | {tr} | {rr} | {bg} / {sg} | {top} |".format(
                layer=row["layer_id"],
                hot=fmt(row["hot_hit_share"]),
                stable=fmt(row["stable_hit_share"]),
                cold=fmt(row["cold_hit_share"]),
                th=fmt(row["temporal_hot_retention_mean"]),
                ts=fmt(row["temporal_stable_retention_mean"]),
                tc=fmt(row["temporal_cold_retention_mean"]),
                rh=fmt(row["request_hot_retention_mean"]),
                rs=fmt(row["request_stable_retention_mean"]),
                rc=fmt(row["request_cold_retention_mean"]),
                tr=fmt(row["temporal_rank_spearman_mean"]),
                rr=fmt(row["request_rank_spearman_mean"]),
                bg=fmt(row["hot_boundary_rel_gap"]),
                sg=fmt(row["stable_boundary_rel_gap"]),
                top=top,
            )
        )

    unstable_stable = sorted(rows, key=lambda row: row["temporal_stable_retention_mean"])[:8]
    strong_hot = sorted(rows, key=lambda row: row["temporal_hot_retention_mean"], reverse=True)[:8]
    lines.extend(
        [
            "",
            "## Extremes",
            "",
            "- strongest temporal hot retention: "
            + ", ".join(
                f"L{row['layer_id']}={row['temporal_hot_retention_mean']:.4f}"
                for row in strong_hot
            ),
            "- weakest temporal stable retention: "
            + ", ".join(
                f"L{row['layer_id']}={row['temporal_stable_retention_mean']:.4f}"
                for row in unstable_stable
            ),
            "",
            "## Reading The Metrics",
            "",
            "- `hit share` is the fraction of token-expert assignments captured by the fixed global tier.",
            "- `retention` recomputes the same 80/32/16 split inside each temporal window or request bucket and measures how many global-tier experts stay in that same tier.",
            "- `rank rho` is Spearman correlation between the full-run expert ranking and each split ranking.",
            "- `boundary rel gaps` are the relative count gaps at rank 80/81 and 112/113. Small gaps mean tier boundaries are easy to swap.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether Qwen3 MoE experts form stable fixed 80/32/16 hot/stable/cold tiers."
    )
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hot-size", type=int, default=80)
    parser.add_argument("--stable-size", type=int, default=32)
    parser.add_argument("--cold-size", type=int, default=16)
    parser.add_argument("--temporal-windows", type=int, default=8)
    parser.add_argument("--request-buckets", type=int, default=4)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefix = args.trace_jsonl.with_suffix("")
    output_md = args.output_md or prefix.with_name(prefix.name + "_fixed_tier_stability.md")
    output_json = args.output_json or prefix.with_name(prefix.name + "_fixed_tier_stability.json")
    output_csv = args.output_csv or prefix.with_name(prefix.name + "_fixed_tier_stability.csv")

    summary = analyze(
        args.trace_jsonl,
        args.num_experts,
        args.hot_size,
        args.stable_size,
        args.cold_size,
        args.temporal_windows,
        args.request_buckets,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_csv, summary["layers"])
    write_markdown(output_md, summary)
    print(output_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
