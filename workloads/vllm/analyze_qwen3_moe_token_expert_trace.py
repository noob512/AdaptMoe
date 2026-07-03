#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(q * (len(ordered) - 1))
    return float(ordered[idx])


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def normalized_entropy(counter: Counter[int], num_experts: int | None) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [value / total for value in counter.values() if value > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    denom = math.log(num_experts or max(len(counter), 2))
    return float(entropy / denom) if denom > 0 else 0.0


def pairwise_jaccards(
    expert_sets: list[set[int]],
    *,
    max_pairs: int,
) -> list[float]:
    count = len(expert_sets)
    if count < 2:
        return []
    total_pairs = count * (count - 1) // 2
    if total_pairs <= max_pairs:
        return [
            jaccard(expert_sets[i], expert_sets[j])
            for i in range(count)
            for j in range(i + 1, count)
        ]

    rng = random.Random(0)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < max_pairs:
        i = rng.randrange(count)
        j = rng.randrange(count)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return [jaccard(expert_sets[i], expert_sets[j]) for i, j in sorted(pairs)]


def parse_trace(path: Path) -> tuple[list[dict[str, Any]], int, int | None]:
    rows: list[dict[str, Any]] = []
    token_events = 0
    num_experts: int | None = None

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "qwen3_moe_token_expert_trace":
            continue

        token_events += 1
        layer_id = record.get("layer_id")
        if layer_id is None:
            continue
        layer_id = int(layer_id)
        if record.get("num_experts") is not None:
            num_experts = int(record["num_experts"])

        request_ids = list(record.get("request_ids") or [])
        request_indices = list(record.get("request_indices") or [])
        token_positions = list(record.get("token_positions") or [])
        token_ids = list(record.get("token_ids") or [])
        topk_ids = list(record.get("topk_ids") or [])

        for token_idx, experts_raw in enumerate(topk_ids):
            request_id = request_ids[token_idx] if token_idx < len(request_ids) else None
            token_position = (
                token_positions[token_idx]
                if token_idx < len(token_positions)
                else -1
            )
            if request_id is None:
                continue
            try:
                token_position = int(token_position)
            except (TypeError, ValueError):
                continue
            if token_position < 0:
                continue

            experts = tuple(
                int(expert_id)
                for expert_id in experts_raw
                if int(expert_id) >= 0
            )
            if not experts:
                continue
            rows.append(
                {
                    "line_no": line_no,
                    "step": int(record.get("step") or 0),
                    "layer_id": layer_id,
                    "layer_name": record.get("layer_name"),
                    "request_id": str(request_id),
                    "request_index": (
                        int(request_indices[token_idx])
                        if token_idx < len(request_indices)
                        else -1
                    ),
                    "token_position": token_position,
                    "token_id": (
                        int(token_ids[token_idx]) if token_idx < len(token_ids) else -1
                    ),
                    "experts": experts,
                }
            )

    return rows, token_events, num_experts


def analyze(
    rows: list[dict[str, Any]],
    *,
    token_events: int,
    num_experts: int | None,
    max_pairs: int,
) -> dict[str, Any]:
    by_token: dict[tuple[str, int], dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    by_request_layer: dict[tuple[str, int], dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    by_layer_request: dict[int, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    by_layer_counter: dict[int, Counter[int]] = defaultdict(Counter)
    by_layer_unique_tokens: dict[int, set[tuple[str, int]]] = defaultdict(set)
    global_counter: Counter[int] = Counter()

    for row in rows:
        request_id = row["request_id"]
        token_position = int(row["token_position"])
        layer_id = int(row["layer_id"])
        experts = set(int(expert_id) for expert_id in row["experts"])

        by_token[(request_id, token_position)][layer_id].update(experts)
        by_request_layer[(request_id, layer_id)][token_position].update(experts)
        by_layer_request[layer_id][request_id].update(experts)
        by_layer_counter[layer_id].update(experts)
        by_layer_unique_tokens[layer_id].add((request_id, token_position))
        global_counter.update(experts)

    token_layer_jaccards: list[float] = []
    token_layer_any_overlap: list[float] = []
    token_layers_observed: list[float] = []
    layer_pair_jaccards: dict[tuple[int, int], list[float]] = defaultdict(list)

    for layer_map in by_token.values():
        layers = sorted(layer_map)
        token_layers_observed.append(float(len(layers)))
        for left_layer, right_layer in zip(layers, layers[1:]):
            left = layer_map[left_layer]
            right = layer_map[right_layer]
            score = jaccard(left, right)
            token_layer_jaccards.append(score)
            token_layer_any_overlap.append(1.0 if left & right else 0.0)
            layer_pair_jaccards[(left_layer, right_layer)].append(score)

    adjacent_token_jaccards: list[float] = []
    adjacent_token_by_layer: dict[int, list[float]] = defaultdict(list)
    request_layer_unique_experts: list[float] = []

    for (_request_id, layer_id), pos_map in by_request_layer.items():
        positions = sorted(pos_map)
        union: set[int] = set()
        for experts in pos_map.values():
            union.update(experts)
        request_layer_unique_experts.append(float(len(union)))
        for left_pos, right_pos in zip(positions, positions[1:]):
            score = jaccard(pos_map[left_pos], pos_map[right_pos])
            adjacent_token_jaccards.append(score)
            adjacent_token_by_layer[layer_id].append(score)

    inter_request_jaccards: list[float] = []
    inter_request_by_layer: dict[int, list[float]] = {}
    for layer_id, request_map in by_layer_request.items():
        scores = pairwise_jaccards(
            list(request_map.values()),
            max_pairs=max_pairs,
        )
        inter_request_by_layer[layer_id] = scores
        inter_request_jaccards.extend(scores)

    layer_rows = []
    for layer_id in sorted(by_layer_counter):
        counter = by_layer_counter[layer_id]
        request_count = len(by_layer_request[layer_id])
        layer_rows.append(
            {
                "layer_id": layer_id,
                "token_records": sum(counter.values()),
                "unique_tokens": len(by_layer_unique_tokens[layer_id]),
                "requests": request_count,
                "unique_experts": len(counter),
                "normalized_entropy": normalized_entropy(counter, num_experts),
                "mean_adjacent_token_jaccard": mean(
                    adjacent_token_by_layer.get(layer_id, [])
                ),
                "mean_inter_request_jaccard": mean(
                    inter_request_by_layer.get(layer_id, [])
                ),
                "top_experts": counter.most_common(16),
            }
        )

    layer_pair_rows = []
    for (left_layer, right_layer), scores in sorted(layer_pair_jaccards.items()):
        layer_pair_rows.append(
            {
                "left_layer": left_layer,
                "right_layer": right_layer,
                "pairs": len(scores),
                "mean_jaccard": mean(scores),
                "p50_jaccard": percentile(scores, 0.50),
                "p95_jaccard": percentile(scores, 0.95),
                "any_overlap_rate": mean([1.0 if value > 0 else 0.0 for value in scores]),
            }
        )

    return {
        "token_events": token_events,
        "token_records": len(rows),
        "request_count": len({row["request_id"] for row in rows}),
        "unique_token_count": len(by_token),
        "layer_count": len(by_layer_counter),
        "num_experts": num_experts,
        "global_unique_experts": len(global_counter),
        "global_top_experts": global_counter.most_common(32),
        "same_token_cross_layer": {
            "mean_layers_observed_per_token": mean(token_layers_observed),
            "p50_layers_observed_per_token": percentile(token_layers_observed, 0.50),
            "mean_adjacent_layer_jaccard": mean(token_layer_jaccards),
            "p50_adjacent_layer_jaccard": percentile(token_layer_jaccards, 0.50),
            "p95_adjacent_layer_jaccard": percentile(token_layer_jaccards, 0.95),
            "adjacent_layer_any_overlap_rate": mean(token_layer_any_overlap),
            "adjacent_layer_pair_count": len(token_layer_jaccards),
        },
        "same_request_different_tokens": {
            "mean_adjacent_token_jaccard": mean(adjacent_token_jaccards),
            "p50_adjacent_token_jaccard": percentile(adjacent_token_jaccards, 0.50),
            "p95_adjacent_token_jaccard": percentile(adjacent_token_jaccards, 0.95),
            "adjacent_token_pair_count": len(adjacent_token_jaccards),
            "mean_request_layer_unique_experts": mean(request_layer_unique_experts),
            "p95_request_layer_unique_experts": percentile(
                request_layer_unique_experts, 0.95
            ),
        },
        "different_requests": {
            "mean_inter_request_jaccard": mean(inter_request_jaccards),
            "p50_inter_request_jaccard": percentile(inter_request_jaccards, 0.50),
            "p95_inter_request_jaccard": percentile(inter_request_jaccards, 0.95),
            "sampled_request_pair_count": len(inter_request_jaccards),
        },
        "layers": layer_rows,
        "layer_pairs": layer_pair_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if "top_experts" in out:
                out["top_experts"] = " ".join(
                    f"{expert}:{count}" for expert, count in out["top_experts"]
                )
            writer.writerow(out)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, trace_path: Path, summary: dict[str, Any]) -> None:
    same_token = summary["same_token_cross_layer"]
    same_request = summary["same_request_different_tokens"]
    diff_requests = summary["different_requests"]
    lines = [
        "# Qwen3 MoE Token Expert Trace Analysis",
        "",
        f"- trace: `{trace_path}`",
        f"- token trace events: `{summary['token_events']}`",
        f"- mapped token records: `{summary['token_records']}`",
        f"- requests: `{summary['request_count']}`",
        f"- unique request tokens: `{summary['unique_token_count']}`",
        f"- MoE layers: `{summary['layer_count']}`",
        f"- unique experts observed: `{summary['global_unique_experts']}`",
        "",
        "## Same Token Across Layers",
        "",
        (
            "- mean adjacent-layer expert Jaccard: "
            f"`{fmt(same_token['mean_adjacent_layer_jaccard'])}`"
        ),
        (
            "- adjacent-layer any-overlap rate: "
            f"`{fmt(same_token['adjacent_layer_any_overlap_rate'])}`"
        ),
        (
            "- mean layers observed per token: "
            f"`{fmt(same_token['mean_layers_observed_per_token'])}`"
        ),
        "",
        "## Same Request, Different Tokens",
        "",
        (
            "- mean adjacent-token expert Jaccard within each request/layer: "
            f"`{fmt(same_request['mean_adjacent_token_jaccard'])}`"
        ),
        (
            "- mean unique experts per request/layer: "
            f"`{fmt(same_request['mean_request_layer_unique_experts'])}`"
        ),
        "",
        "## Different Requests",
        "",
        (
            "- mean inter-request expert-set Jaccard within each layer: "
            f"`{fmt(diff_requests['mean_inter_request_jaccard'])}`"
        ),
        (
            "- sampled inter-request pairs: "
            f"`{diff_requests['sampled_request_pair_count']}`"
        ),
        "",
        "## Layer Summary",
        "",
        (
            "| layer | token records | requests | unique experts | entropy | "
            "adjacent-token Jaccard | inter-request Jaccard | top experts |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["layers"][:64]:
        top = " ".join(f"{expert}:{count}" for expert, count in row["top_experts"][:8])
        lines.append(
            "| {layer_id} | {token_records} | {requests} | {unique_experts} | "
            "{entropy} | {adj_tok} | {inter_req} | {top} |".format(
                layer_id=row["layer_id"],
                token_records=row["token_records"],
                requests=row["requests"],
                unique_experts=row["unique_experts"],
                entropy=fmt(row["normalized_entropy"]),
                adj_tok=fmt(row["mean_adjacent_token_jaccard"]),
                inter_req=fmt(row["mean_inter_request_jaccard"]),
                top=top,
            )
        )
    if len(summary["layers"]) > 64:
        lines.append("| ... | ... | ... | ... | ... | ... | ... | ... |")

    lines.extend(
        [
            "",
            "## Reading The Numbers",
            "",
            (
                "- Low same-token cross-layer Jaccard means routers choose "
                "different experts at different layers for the same token."
            ),
            (
                "- High adjacent-token Jaccard means nearby tokens in the same "
                "request tend to reuse experts within a layer."
            ),
            (
                "- High inter-request Jaccard means different requests converge "
                "on similar layer-local expert sets."
            ),
            (
                "- Entropy close to 1.0 means token assignments are spread more "
                "evenly across observed experts."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Qwen3 MoE token/request/layer expert trace JSONL."
    )
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--layer-csv", type=Path)
    parser.add_argument("--layer-pair-csv", type=Path)
    parser.add_argument("--max-pairs", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_path = args.trace_jsonl
    default_prefix = trace_path.with_suffix("")
    output_md = args.output_md or default_prefix.with_name(
        default_prefix.name + "_token_analysis.md"
    )
    output_json = args.output_json or default_prefix.with_name(
        default_prefix.name + "_token_analysis.json"
    )
    layer_csv = args.layer_csv or default_prefix.with_name(
        default_prefix.name + "_token_layer_summary.csv"
    )
    layer_pair_csv = args.layer_pair_csv or default_prefix.with_name(
        default_prefix.name + "_token_layer_pair_summary.csv"
    )

    rows, token_events, num_experts = parse_trace(trace_path)
    summary = analyze(
        rows,
        token_events=token_events,
        num_experts=num_experts,
        max_pairs=args.max_pairs,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(
        layer_csv,
        summary["layers"],
        [
            "layer_id",
            "token_records",
            "unique_tokens",
            "requests",
            "unique_experts",
            "normalized_entropy",
            "mean_adjacent_token_jaccard",
            "mean_inter_request_jaccard",
            "top_experts",
        ],
    )
    write_csv(
        layer_pair_csv,
        summary["layer_pairs"],
        [
            "left_layer",
            "right_layer",
            "pairs",
            "mean_jaccard",
            "p50_jaccard",
            "p95_jaccard",
            "any_overlap_rate",
        ],
    )
    write_markdown(output_md, trace_path, summary)
    print(output_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
