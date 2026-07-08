#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def layer_id_from_name(layer_name: Any) -> int | None:
    if not isinstance(layer_name, str):
        return None
    parts = layer_name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a small UVM expert offload plan from a routing trace."
    )
    parser.add_argument("--routing-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--max-experts-per-layer", type=int, default=2)
    parser.add_argument("--min-layer", type=int, default=0)
    parser.add_argument("--max-layer", type=int, default=-1)
    args = parser.parse_args()

    experts_by_layer: dict[int, set[int]] = defaultdict(set)
    with args.routing_trace.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            layer_id = layer_id_from_name(record.get("layer_name"))
            if layer_id is None:
                continue
            if layer_id < args.min_layer:
                continue
            if args.max_layer >= 0 and layer_id > args.max_layer:
                continue
            active_experts = record.get("active_experts")
            if not isinstance(active_experts, list):
                continue
            for expert_id in active_experts:
                try:
                    experts_by_layer[layer_id].add(int(expert_id))
                except (TypeError, ValueError):
                    continue

    offload_plan: list[dict[str, object]] = []
    for layer_id in sorted(experts_by_layer)[: max(args.max_layers, 0)]:
        experts = sorted(experts_by_layer[layer_id])
        if args.max_experts_per_layer > 0:
            experts = experts[: args.max_experts_per_layer]
        for expert_id in experts:
            offload_plan.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "reason": "routing_trace_active_expert_probe",
                }
            )

    plan = {
        "metadata": {
            "policy": "routing_trace_active_expert_probe_v1",
            "routing_trace": str(args.routing_trace),
            "max_layers": args.max_layers,
            "max_experts_per_layer": args.max_experts_per_layer,
            "min_layer": args.min_layer,
            "max_layer": args.max_layer,
        },
        "prefetch_plan": [],
        "offload_plan": offload_plan,
        "prefetch_plan_records": 0,
        "offload_plan_records": len(offload_plan),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"offload_plan_records={len(offload_plan)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
