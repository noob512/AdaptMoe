#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and report per-layer hot/warm/cold MoE expert tiers.

The input is a vLLM Qwen3 MoE routing trace.  Newer traces can contain both
token-level events and layer aggregate events; token events are preferred so the
usage numbers count actual token-expert assignments exactly once.  Older traces
with only `expert_token_counts` are also supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

TIERS = ("hot", "warm", "cold")


def layer_id_from_name(layer_name: Any) -> int | None:
    if not isinstance(layer_name, str):
        return None
    match = re.search(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)", layer_name)
    return int(match.group(1)) if match else None


def as_int_dict(raw: Any) -> dict[int, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, int] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def as_float_dict(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(statistics.fmean(vals)) if vals else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    return float(vals[int(q * (len(vals) - 1))])


def normalized_entropy(counter: Counter[int], num_experts: int) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0 or num_experts <= 1:
        return 0.0
    entropy = 0.0
    for value in counter.values():
        if value <= 0:
            continue
        p = float(value) / total
        entropy -= p * math.log(p)
    return float(entropy / math.log(num_experts))


def read_trace_counters(
    trace_path: Path,
    *,
    score_mode: str,
    fallback_num_experts: int,
    mapped_only: bool,
) -> dict[str, Any]:
    token_counters: dict[int, Counter[int]] = defaultdict(Counter)
    token_weight_counters: dict[int, Counter[int]] = defaultdict(Counter)
    token_rows_by_layer: Counter[int] = Counter()
    token_events = 0
    token_assignments = 0
    token_requests: set[str] = set()

    layer_counters: dict[int, Counter[int]] = defaultdict(Counter)
    layer_weight_counters: dict[int, Counter[int]] = defaultdict(Counter)
    layer_rows_by_layer: Counter[int] = Counter()
    layer_events = 0
    layer_assignments = 0
    layer_requests: set[str] = set()
    num_experts = fallback_num_experts

    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            layer_id = record.get("layer_id")
            if layer_id is None:
                layer_id = layer_id_from_name(record.get("layer_name"))
            if layer_id is None:
                continue
            layer_id = int(layer_id)

            for key in ("num_experts", "global_num_experts", "logical_num_experts"):
                try:
                    candidate = int(record.get(key) or 0)
                except (TypeError, ValueError):
                    candidate = 0
                if candidate > num_experts:
                    num_experts = candidate

            event = record.get("event")
            if event == "qwen3_moe_token_expert_trace":
                token_events += 1
                topk_ids = list(record.get("topk_ids") or [])
                request_ids = list(record.get("request_ids") or [])
                token_positions = list(record.get("token_positions") or [])
                topk_weights = record.get("topk_weights")
                if not isinstance(topk_weights, list):
                    topk_weights = []

                for row_idx, experts_raw in enumerate(topk_ids):
                    request_id = (
                        request_ids[row_idx] if row_idx < len(request_ids) else None
                    )
                    token_position = (
                        token_positions[row_idx]
                        if row_idx < len(token_positions)
                        else None
                    )
                    mapped = request_id is not None
                    if mapped:
                        try:
                            mapped = int(token_position) >= 0
                        except (TypeError, ValueError):
                            mapped = False
                    if mapped_only and not mapped:
                        continue
                    if request_id is not None:
                        token_requests.add(str(request_id))
                    if not isinstance(experts_raw, list):
                        continue
                    weights_raw = (
                        topk_weights[row_idx]
                        if row_idx < len(topk_weights)
                        and isinstance(topk_weights[row_idx], list)
                        else []
                    )
                    valid = []
                    for pos, expert_raw in enumerate(experts_raw):
                        try:
                            expert_id = int(expert_raw)
                        except (TypeError, ValueError):
                            continue
                        if expert_id < 0:
                            continue
                        valid.append(expert_id)
                        token_counters[layer_id][expert_id] += 1
                        token_assignments += 1
                        try:
                            weight = float(weights_raw[pos])
                        except (TypeError, ValueError, IndexError):
                            weight = 1.0
                        token_weight_counters[layer_id][expert_id] += weight
                    if valid:
                        token_rows_by_layer[layer_id] += 1
                continue

            counts = as_int_dict(record.get("expert_token_counts"))
            weights = as_float_dict(record.get("expert_weight_sums"))
            active = record.get("active_experts")
            if not counts and isinstance(active, list):
                for item in active:
                    try:
                        counts[int(item)] += 1
                    except (TypeError, ValueError):
                        continue
            if not counts:
                continue

            layer_events += 1
            layer_rows_by_layer[layer_id] += 1
            request_id = record.get("request_id")
            if request_id is not None:
                layer_requests.add(str(request_id))
            for expert_id, count in counts.items():
                if expert_id < 0:
                    continue
                layer_counters[layer_id][expert_id] += count
                layer_assignments += count
                layer_weight_counters[layer_id][expert_id] += weights.get(
                    expert_id, float(count)
                )

    if token_events > 0 and token_assignments > 0:
        counters = token_weight_counters if score_mode == "weight_sum" else token_counters
        usage_counters = token_counters
        rows_by_layer = token_rows_by_layer
        return {
            "source": "token_trace",
            "events": token_events,
            "requests": len(token_requests),
            "assignments": token_assignments,
            "num_experts": num_experts,
            "score_counters": counters,
            "usage_counters": usage_counters,
            "rows_by_layer": rows_by_layer,
        }

    counters = layer_weight_counters if score_mode == "weight_sum" else layer_counters
    return {
        "source": "layer_aggregate_trace",
        "events": layer_events,
        "requests": len(layer_requests),
        "assignments": layer_assignments,
        "num_experts": num_experts,
        "score_counters": counters,
        "usage_counters": layer_counters,
        "rows_by_layer": layer_rows_by_layer,
    }


def ranked_experts(counter: Counter[int], num_experts: int) -> list[int]:
    return sorted(range(num_experts), key=lambda expert: (-counter[expert], expert))


def make_tiers(
    score_counter: Counter[int],
    num_experts: int,
    hot_size: int,
    warm_size: int,
    cold_size: int,
) -> dict[str, list[int]]:
    ordered = ranked_experts(score_counter, num_experts)
    return {
        "hot": ordered[:hot_size],
        "warm": ordered[hot_size : hot_size + warm_size],
        "cold": ordered[hot_size + warm_size : hot_size + warm_size + cold_size],
    }


def fmt_float(value: float) -> str:
    return f"{value:.4f}"


def analyze(
    trace_path: Path,
    *,
    score_mode: str,
    num_experts: int,
    hot_size: int,
    warm_size: int,
    cold_size: int,
    mapped_only: bool,
) -> dict[str, Any]:
    if hot_size + warm_size + cold_size != num_experts:
        raise ValueError("hot_size + warm_size + cold_size must equal num_experts")

    trace = read_trace_counters(
        trace_path,
        score_mode=score_mode,
        fallback_num_experts=num_experts,
        mapped_only=mapped_only,
    )
    actual_num_experts = int(trace["num_experts"])
    if actual_num_experts != num_experts:
        if hot_size + warm_size + cold_size != actual_num_experts:
            raise ValueError(
                "tier sizes must sum to detected num_experts="
                f"{actual_num_experts}, got {hot_size + warm_size + cold_size}"
            )
        num_experts = actual_num_experts

    score_counters: dict[int, Counter[int]] = trace["score_counters"]
    usage_counters: dict[int, Counter[int]] = trace["usage_counters"]
    rows_by_layer: Counter[int] = trace["rows_by_layer"]

    tier_assignment_counts = Counter[str]()
    tier_layer_counts = Counter[str]()
    layer_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []

    for layer_id in sorted(usage_counters):
        score_counter = score_counters.get(layer_id, Counter())
        usage_counter = usage_counters[layer_id]
        tiers = make_tiers(
            score_counter,
            num_experts,
            hot_size,
            warm_size,
            cold_size,
        )
        tier_sets = {tier: set(experts) for tier, experts in tiers.items()}
        total_assignments = float(sum(usage_counter.values()))

        row: dict[str, Any] = {
            "layer_id": layer_id,
            "records": int(rows_by_layer[layer_id]),
            "assignments": int(total_assignments),
            "unique_used_experts": len([v for v in usage_counter.values() if v > 0]),
            "entropy": normalized_entropy(usage_counter, num_experts),
            "top8": usage_counter.most_common(8),
        }

        for tier in TIERS:
            experts = tiers[tier]
            hits = sum(usage_counter[expert] for expert in experts)
            share = float(hits) / total_assignments if total_assignments else 0.0
            size = len(experts)
            size_frac = float(size) / float(num_experts) if num_experts else 0.0
            lift = share / size_frac if size_frac else 0.0
            tier_assignment_counts[tier] += hits
            tier_layer_counts[tier] += size
            row[f"{tier}_experts"] = experts
            row[f"{tier}_size"] = size
            row[f"{tier}_assignments"] = int(hits)
            row[f"{tier}_share"] = share
            row[f"{tier}_lift_vs_uniform"] = lift

            for rank_in_tier, expert_id in enumerate(experts, start=1):
                classification_rows.append(
                    {
                        "layer_id": layer_id,
                        "tier": tier,
                        "rank_in_tier": rank_in_tier,
                        "expert_id": expert_id,
                        "score": float(score_counter[expert_id]),
                        "assignments": int(usage_counter[expert_id]),
                        "assignment_share": (
                            float(usage_counter[expert_id]) / total_assignments
                            if total_assignments
                            else 0.0
                        ),
                    }
                )

        hot_set = tier_sets["hot"]
        warm_set = tier_sets["warm"]
        cold_set = tier_sets["cold"]
        row["management"] = {
            "gpu_resident": len(hot_set),
            "uvm_prefetch_or_compressed": len(warm_set),
            "cpu_offload_candidate": len(cold_set),
        }
        layer_rows.append(row)

    total = float(sum(tier_assignment_counts.values()))
    overall = {}
    for tier in TIERS:
        size_frac = (
            float(tier_layer_counts[tier])
            / float(sum(tier_layer_counts.values()))
            if sum(tier_layer_counts.values())
            else 0.0
        )
        share = float(tier_assignment_counts[tier]) / total if total else 0.0
        overall[tier] = {
            "experts_per_layer": {"hot": hot_size, "warm": warm_size, "cold": cold_size}[
                tier
            ],
            "total_layer_experts": int(tier_layer_counts[tier]),
            "assignments": int(tier_assignment_counts[tier]),
            "assignment_share": share,
            "size_share": size_frac,
            "lift_vs_uniform": share / size_frac if size_frac else 0.0,
        }

    return {
        "trace": str(trace_path),
        "source": trace["source"],
        "mapped_only": mapped_only,
        "events": int(trace["events"]),
        "requests": int(trace["requests"]),
        "assignments": int(trace["assignments"]),
        "tiering": {
            "rule": (
                f"per-layer experts ranked by {score_mode}; "
                "hot keeps highest ranks, warm keeps next ranks, cold keeps lowest ranks"
            ),
            "score_mode": score_mode,
            "num_experts": num_experts,
            "hot_size": hot_size,
            "warm_size": warm_size,
            "cold_size": cold_size,
            "management_mapping": {
                "hot": "GPU resident candidate",
                "warm": "UVM managed / prefetch / compressed candidate",
                "cold": "CPU offload candidate",
            },
        },
        "overall": overall,
        "layers": layer_rows,
        "classification_rows": classification_rows,
    }


def write_usage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "layer_id",
        "records",
        "assignments",
        "unique_used_experts",
        "entropy",
        "hot_size",
        "hot_assignments",
        "hot_share",
        "hot_lift_vs_uniform",
        "warm_size",
        "warm_assignments",
        "warm_share",
        "warm_lift_vs_uniform",
        "cold_size",
        "cold_assignments",
        "cold_share",
        "cold_lift_vs_uniform",
        "top8",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["top8"] = " ".join(
                f"{expert}:{count}" for expert, count in row["top8"]
            )
            writer.writerow(out)


def write_classification_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "layer_id",
        "tier",
        "rank_in_tier",
        "expert_id",
        "score",
        "assignments",
        "assignment_share",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    layers = summary["layers"]

    lines = [
        "# MoE Three-Tier Expert Usage",
        "",
        f"- trace: `{summary['trace']}`",
        f"- source: `{summary['source']}`",
        f"- mapped token rows only: `{summary['mapped_only']}`",
        f"- events: `{summary['events']}`",
        f"- requests: `{summary['requests']}`",
        f"- token-expert assignments: `{summary['assignments']}`",
        f"- tiering: `{summary['tiering']['rule']}`",
        "- management mapping: `hot=GPU resident`, `warm=UVM/prefetch`, `cold=CPU offload`",
        "",
        "## Overall",
        "",
        "| tier | experts/layer | assignment share | size share | lift vs uniform | assignments |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        row = overall[tier]
        lines.append(
            "| {tier} | {size} | {share} | {size_share} | {lift} | {assignments} |".format(
                tier=tier,
                size=row["experts_per_layer"],
                share=fmt_float(row["assignment_share"]),
                size_share=fmt_float(row["size_share"]),
                lift=fmt_float(row["lift_vs_uniform"]),
                assignments=row["assignments"],
            )
        )

    share_hot_warm = (
        float(overall["hot"]["assignment_share"])
        + float(overall["warm"]["assignment_share"])
    )
    lines.extend(
        [
            "",
            "## Result",
            "",
            (
                f"- hot+warm tiers cover `{share_hot_warm:.4f}` of observed "
                "token-expert assignments."
            ),
            (
                f"- cold tier accounts for only "
                f"`{float(overall['cold']['assignment_share']):.4f}` of assignments, "
                "so it is the natural CPU-offload candidate set."
            ),
            "",
            "## Per-Layer Usage",
            "",
            "| layer | hot share | warm share | cold share | entropy | top experts | hot sample | warm sample | cold sample |",
            "|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for row in layers:
        top = " ".join(f"{expert}:{count}" for expert, count in row["top8"][:6])
        hot_sample = ",".join(str(x) for x in row["hot_experts"][:8])
        warm_sample = ",".join(str(x) for x in row["warm_experts"][:8])
        cold_sample = ",".join(str(x) for x in row["cold_experts"][:8])
        lines.append(
            "| {layer} | {hot} | {warm} | {cold} | {entropy} | {top} | {hot_sample} | {warm_sample} | {cold_sample} |".format(
                layer=row["layer_id"],
                hot=fmt_float(row["hot_share"]),
                warm=fmt_float(row["warm_share"]),
                cold=fmt_float(row["cold_share"]),
                entropy=fmt_float(row["entropy"]),
                top=top,
                hot_sample=hot_sample,
                warm_sample=warm_sample,
                cold_sample=cold_sample,
            )
        )

    weakest_hot = sorted(layers, key=lambda row: row["hot_share"])[:5]
    strongest_cold = sorted(layers, key=lambda row: row["cold_share"], reverse=True)[:5]
    lines.extend(
        [
            "",
            "## Extremes",
            "",
            "- weakest hot coverage layers: "
            + ", ".join(
                f"L{row['layer_id']}={row['hot_share']:.4f}" for row in weakest_hot
            ),
            "- highest cold usage layers: "
            + ", ".join(
                f"L{row['layer_id']}={row['cold_share']:.4f}" for row in strongest_cold
            ),
            "",
            "## Files",
            "",
            "- `three_tier_usage_summary.json` contains full per-layer expert lists.",
            "- `three_tier_classification.csv` contains one row per `(layer, expert)`.",
            "- `three_tier_layer_usage.csv` contains per-layer tier hit shares.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze vLLM MoE expert heat as hot/warm/cold tiers."
    )
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hot-size", type=int, default=80)
    parser.add_argument("--warm-size", type=int, default=32)
    parser.add_argument("--cold-size", type=int, default=16)
    parser.add_argument(
        "--score-mode",
        choices=("token_count", "weight_sum"),
        default="token_count",
    )
    parser.add_argument(
        "--include-unmapped-token-rows",
        action="store_true",
        help="Count token trace rows even when vLLM request metadata is missing.",
    )
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.trace_jsonl.with_suffix("").with_name(
        args.trace_jsonl.with_suffix("").name + "_three_tier_usage"
    )
    summary = analyze(
        args.trace_jsonl,
        score_mode=args.score_mode,
        num_experts=args.num_experts,
        hot_size=args.hot_size,
        warm_size=args.warm_size,
        cold_size=args.cold_size,
        mapped_only=not args.include_unmapped_token_rows,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    json_summary = dict(summary)
    json_summary.pop("classification_rows")
    (out_dir / "three_tier_usage_summary.json").write_text(
        json.dumps(json_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_usage_csv(out_dir / "three_tier_layer_usage.csv", summary["layers"])
    write_classification_csv(
        out_dir / "three_tier_classification.csv",
        summary["classification_rows"],
    )
    write_markdown(out_dir / "three_tier_usage_report.md", summary)
    print((out_dir / "three_tier_usage_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
