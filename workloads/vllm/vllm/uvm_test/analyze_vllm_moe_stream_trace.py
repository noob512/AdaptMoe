#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROUTED_EXPERT_ACTIONS = {
    "moe_forward_op_enter",
    "moe_forward_op_exit",
    "moe_quant_apply_enter",
    "moe_quant_apply_exit",
}
SHARED_EXPERT_ACTIONS = {
    "shared_experts_stream_compute_enter",
    "shared_experts_stream_compute_exit",
}
STREAM_SYNC_ACTIONS = {
    "reserved_stream_shared_experts_wait_current",
    "reserved_stream_current_wait_shared_experts",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skip malformed json line={line_no}: {exc}")
                continue
            if (
                record.get("mode") == "moe_stream_trace"
                or "current_stream_cuda" in record
                or "shared_experts_stream_cuda" in record
            ):
                records.append(record)
    return records


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def layer_id_from_name(layer_name: Any) -> int | None:
    if not isinstance(layer_name, str):
        return None
    match = re.search(r"(?:^|\.)layers?\.(\d+)(?:\.|$)", layer_name)
    if not match:
        return None
    return int(match.group(1))


def counter_for(
    records: list[dict[str, Any]],
    *,
    actions: set[str] | None = None,
    field: str = "current_stream_cuda",
) -> Counter[int]:
    counter: Counter[int] = Counter()
    for record in records:
        if actions is not None and str(record.get("action")) not in actions:
            continue
        stream = as_int(record.get(field))
        if stream is not None:
            counter[stream] += 1
    return counter


def print_counter(title: str, counter: Counter[int], limit: int) -> None:
    print(f"{title}_unique={len(counter)}")
    for stream, count in counter.most_common(limit):
        print(f"  {title}={stream} count={count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM MoE CUDA stream JSONL trace."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    records = load_records(args.trace)
    actions = Counter(str(record.get("action", "")) for record in records)
    current_streams = counter_for(records)
    shared_streams = counter_for(records, field="shared_experts_stream_cuda")
    routed_streams = counter_for(records, actions=ROUTED_EXPERT_ACTIONS)
    shared_compute_streams = counter_for(records, actions=SHARED_EXPERT_ACTIONS)
    sync_streams = counter_for(records, actions=STREAM_SYNC_ACTIONS)

    by_action: dict[str, Counter[int]] = defaultdict(Counter)
    by_layer: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        action = str(record.get("action", ""))
        stream = as_int(record.get("current_stream_cuda"))
        if stream is None:
            continue
        by_action[action][stream] += 1
        layer_name = str(record.get("layer_name", ""))
        by_layer[layer_name][stream] += 1

    layers = {
        layer_id_from_name(record.get("layer_name"))
        for record in records
    }
    layers.discard(None)

    shared_compute_mismatches = 0
    shared_compute_samples = 0
    for record in records:
        if str(record.get("action")) not in SHARED_EXPERT_ACTIONS:
            continue
        current = as_int(record.get("current_stream_cuda"))
        shared = as_int(record.get("shared_experts_stream_cuda"))
        if current is None or shared is None or shared == 0:
            continue
        shared_compute_samples += 1
        if current != shared:
            shared_compute_mismatches += 1

    print(f"records={len(records)}")
    print(f"layers_observed={len(layers)}")
    if layers:
        print(f"layer_id_min={min(layers)} layer_id_max={max(layers)}")
    print("actions:")
    for action, count in actions.most_common(args.top):
        print(f"  {action}: {count}")

    print("streams_overall:")
    print_counter("current_stream_cuda", current_streams, args.top)
    print_counter("shared_experts_stream_cuda", shared_streams, args.top)

    print("expert_compute_streams:")
    print_counter("routed_expert_current_stream", routed_streams, args.top)
    print_counter("shared_expert_current_stream", shared_compute_streams, args.top)
    print_counter("stream_sync_current_stream", sync_streams, args.top)
    if shared_compute_samples:
        print(
            "shared_compute_current_equals_shared="
            f"{shared_compute_samples - shared_compute_mismatches}/"
            f"{shared_compute_samples}"
        )
    else:
        print("shared_compute_current_equals_shared=not_observed")

    print("streams_by_action:")
    for action, streams in sorted(by_action.items()):
        stream_desc = ", ".join(
            f"{stream}:{count}" for stream, count in streams.most_common(args.top)
        )
        print(f"  {action}: unique={len(streams)} {stream_desc}")

    print("layers_with_multiple_current_streams:")
    multi = [
        (layer_name, streams)
        for layer_name, streams in sorted(by_layer.items())
        if len(streams) > 1
    ]
    if not multi:
        print("  none")
    else:
        for layer_name, streams in multi[: args.top]:
            stream_desc = ", ".join(
                f"{stream}:{count}" for stream, count in streams.most_common(args.top)
            )
            print(f"  {layer_name}: unique={len(streams)} {stream_desc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
