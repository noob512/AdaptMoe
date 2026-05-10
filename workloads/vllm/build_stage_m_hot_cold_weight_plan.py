#!/usr/bin/env python3
"""Build a routing-aware Stage M hot/cold expert weight plan.

The input is the MoE routing JSONL emitted by
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1. The output is a Stage H/M-compatible plan
JSON containing:

  - prefetch_plan: top hot experts per layer
  - offload_plan: bottom cold experts per layer

This script intentionally makes the hot/cold decision in user space, where MoE
layer/expert semantics are available. The allocator/driver should only execute
the resulting actions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a routing-aware hot/cold expert weight plan.",
    )
    parser.add_argument("--routing-trace", required=True, type=Path)
    parser.add_argument("--output-plan", required=True, type=Path)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--hot-experts-per-layer", type=int, default=16)
    parser.add_argument("--cold-experts-per-layer", type=int, default=16)
    parser.add_argument("--fallback-layers", type=int, default=48)
    parser.add_argument("--fallback-experts", type=int, default=128)
    parser.add_argument(
        "--target-roles",
        default="moe_gate_up,moe_down",
        help="Comma-separated weight roles documented in the output metadata.",
    )
    parser.add_argument(
        "--cold-require-zero-hit",
        action="store_true",
        help=(
            "Only put zero-hit experts into offload_plan. This is more "
            "conservative and avoids offloading low-but-nonzero experts."
        ),
    )
    parser.add_argument(
        "--min-routing-records",
        type=int,
        default=1,
        help="Fail if fewer routing records are parsed.",
    )
    return parser.parse_args()


def layer_id_from_name(name: str) -> int | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    match = re.search(r"(?:layers|layer)\.(\d+)", name)
    return int(match.group(1)) if match else None


def read_routing_counts(
    routing_trace: Path,
) -> tuple[dict[int, Counter[int]], int, int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    num_records = 0
    max_experts_seen = 0

    with routing_trace.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            layer_id = layer_id_from_name(str(record.get("layer_name", "")))
            if layer_id is None:
                continue

            try:
                max_experts_seen = max(
                    max_experts_seen,
                    int(record.get("global_num_experts") or 0),
                )
            except (TypeError, ValueError):
                pass

            expert_counts = record.get("expert_token_counts") or {}
            if not isinstance(expert_counts, dict):
                continue
            for expert_id, count in expert_counts.items():
                try:
                    counts[layer_id][int(expert_id)] += int(count)
                except (TypeError, ValueError):
                    continue
            num_records += 1

    return counts, num_records, max_experts_seen


def build_plan(
    *,
    counts: dict[int, Counter[int]],
    num_records: int,
    num_experts: int,
    layers: list[int],
    hot_experts_per_layer: int,
    cold_experts_per_layer: int,
    target_roles: list[str],
    cold_require_zero_hit: bool,
    routing_trace: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefetch_plan: list[dict[str, Any]] = []
    offload_plan: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []

    for layer_id in layers:
        layer_counts = counts.get(layer_id, Counter())
        all_experts = list(range(num_experts))

        ranked_hot = sorted(
            all_experts,
            key=lambda expert_id: (-layer_counts.get(expert_id, 0), expert_id),
        )
        ranked_cold = sorted(
            all_experts,
            key=lambda expert_id: (layer_counts.get(expert_id, 0), expert_id),
        )

        hot = ranked_hot[: max(hot_experts_per_layer, 0)]
        hot_set = set(hot)
        cold: list[int] = []
        for expert_id in ranked_cold:
            if expert_id in hot_set:
                continue
            if cold_require_zero_hit and layer_counts.get(expert_id, 0) != 0:
                continue
            cold.append(expert_id)
            if len(cold) >= max(cold_experts_per_layer, 0):
                break

        for expert_id in hot:
            prefetch_plan.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "score": layer_counts.get(expert_id, 0),
                    "reason": "routing_heatmap_top_hot",
                }
            )
        for expert_id in cold:
            offload_plan.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "score": layer_counts.get(expert_id, 0),
                    "reason": "routing_heatmap_bottom_cold",
                }
            )

        layer_summaries.append(
            {
                "layer_id": layer_id,
                "observed_experts": len(layer_counts),
                "total_tokens": sum(layer_counts.values()),
                "hot": [
                    {"expert_id": expert_id, "score": layer_counts.get(expert_id, 0)}
                    for expert_id in hot[:8]
                ],
                "cold": [
                    {"expert_id": expert_id, "score": layer_counts.get(expert_id, 0)}
                    for expert_id in cold[:8]
                ],
            }
        )

    plan = {
        "metadata": {
            "policy": "routing_heatmap_v1",
            "routing_trace": str(routing_trace),
            "routing_records": num_records,
            "hot_experts_per_layer": hot_experts_per_layer,
            "cold_experts_per_layer": cold_experts_per_layer,
            "cold_require_zero_hit": cold_require_zero_hit,
            "num_layers": len(layers),
            "num_experts": num_experts,
            "target_roles": target_roles,
        },
        "prefetch_plan": prefetch_plan,
        "offload_plan": offload_plan,
        "prefetch_plan_records": len(prefetch_plan),
        "offload_plan_records": len(offload_plan),
    }
    return plan, layer_summaries


def write_report(
    *,
    report_path: Path,
    routing_trace: Path,
    plan: dict[str, Any],
    layer_summaries: list[dict[str, Any]],
) -> None:
    metadata = plan["metadata"]
    lines = [
        "# Routing-Aware Weight Plan Report",
        "",
        f"- routing_trace: `{routing_trace}`",
        f"- routing_records: {metadata['routing_records']}",
        f"- layers: {metadata['num_layers']}",
        f"- experts_per_layer: {metadata['num_experts']}",
        f"- hot_experts_per_layer: {metadata['hot_experts_per_layer']}",
        f"- cold_experts_per_layer: {metadata['cold_experts_per_layer']}",
        f"- cold_require_zero_hit: {metadata['cold_require_zero_hit']}",
        f"- prefetch_plan_records: {plan['prefetch_plan_records']}",
        f"- offload_plan_records: {plan['offload_plan_records']}",
        "",
        "## Layer Samples",
    ]
    for item in layer_summaries[:12]:
        lines.extend(
            [
                "",
                f"### Layer {item['layer_id']}",
                f"- observed_experts: {item['observed_experts']}",
                f"- total_tokens: {item['total_tokens']}",
                f"- hot_sample: {item['hot']}",
                f"- cold_sample: {item['cold']}",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.routing_trace.exists():
        raise SystemExit(f"routing trace not found: {args.routing_trace}")

    counts, num_records, max_experts_seen = read_routing_counts(args.routing_trace)
    if num_records < args.min_routing_records:
        raise SystemExit(
            "not enough routing records: "
            f"{num_records} < {args.min_routing_records}"
        )

    layers = sorted(counts) if counts else list(range(max(args.fallback_layers, 0)))
    num_experts = max(max_experts_seen, max(args.fallback_experts, 0))
    if not layers:
        raise SystemExit("no layers available for plan generation")
    if num_experts <= 0:
        raise SystemExit("num_experts must be positive")

    target_roles = [role.strip() for role in args.target_roles.split(",") if role.strip()]
    plan, layer_summaries = build_plan(
        counts=counts,
        num_records=num_records,
        num_experts=num_experts,
        layers=layers,
        hot_experts_per_layer=args.hot_experts_per_layer,
        cold_experts_per_layer=args.cold_experts_per_layer,
        target_roles=target_roles,
        cold_require_zero_hit=args.cold_require_zero_hit,
        routing_trace=args.routing_trace,
    )

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(
        json.dumps(plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.report_md is not None:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        write_report(
            report_path=args.report_md,
            routing_trace=args.routing_trace,
            plan=plan,
            layer_summaries=layer_summaries,
        )

    print(f"routing_records={num_records}")
    print(f"layers={len(layers)}")
    print(f"experts_per_layer={num_experts}")
    print(f"prefetch_plan_records={plan['prefetch_plan_records']}")
    print(f"offload_plan_records={plan['offload_plan_records']}")
    print(f"output_plan={args.output_plan}")
    if args.report_md is not None:
        print(f"report_md={args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
