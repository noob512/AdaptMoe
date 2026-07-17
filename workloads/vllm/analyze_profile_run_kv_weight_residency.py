#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def summarize_weight_phase(records: list[dict[str, Any]]) -> dict[str, Any]:
    phase_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") != "weight:expert_slice":
            continue
        if not bool(record.get("is_moe_expert")):
            continue
        phase_records[str(record.get("phase") or "unknown")].append(record)

    summaries: dict[str, Any] = {}
    for phase, items in sorted(phase_records.items()):
        state_counts = Counter(str(item.get("residency_state") or "missing") for item in items)
        role_counts = Counter(str(item.get("role") or "unknown") for item in items)
        layer_ids = {
            int(item["layer_id"])
            for item in items
            if item.get("layer_id") is not None
        }
        expert_ids_by_layer: dict[int, set[int]] = defaultdict(set)
        queried_bytes = 0
        gpu_bytes = 0
        cpu_bytes = 0
        unknown_bytes = 0
        query_ok = 0
        driver_capable = 0
        non_driver_capable = 0
        examples_not_full_gpu: list[dict[str, Any]] = []
        for item in items:
            queried = int(item.get("residency_queried_bytes") or 0)
            gpu = int(item.get("resident_gpu_bytes") or 0)
            cpu = int(item.get("resident_cpu_bytes") or 0)
            unknown = int(item.get("resident_unknown_bytes") or 0)
            queried_bytes += queried
            gpu_bytes += gpu
            cpu_bytes += cpu
            unknown_bytes += unknown
            if bool(item.get("residency_query_ok")):
                query_ok += 1
            if bool(item.get("driver_capable")):
                driver_capable += 1
            else:
                non_driver_capable += 1
            if item.get("layer_id") is not None and item.get("expert_id") is not None:
                expert_ids_by_layer[int(item["layer_id"])].add(int(item["expert_id"]))
            state = str(item.get("residency_state") or "")
            if state != "gpu_full" and len(examples_not_full_gpu) < 12:
                examples_not_full_gpu.append(
                    {
                        "name": item.get("name"),
                        "layer_id": item.get("layer_id"),
                        "expert_id": item.get("expert_id"),
                        "role": item.get("role"),
                        "state": state,
                        "queried_bytes": queried,
                        "gpu_bytes": gpu,
                        "cpu_bytes": cpu,
                        "unknown_bytes": unknown,
                    }
                )
        expert_counts = [len(values) for values in expert_ids_by_layer.values()]
        all_full_gpu = bool(items) and state_counts.get("gpu_full", 0) == len(items)
        summaries[phase] = {
            "records": len(items),
            "query_ok_records": query_ok,
            "driver_capable_records": driver_capable,
            "non_driver_capable_records": non_driver_capable,
            "layers": len(layer_ids),
            "min_experts_per_layer": min(expert_counts) if expert_counts else 0,
            "max_experts_per_layer": max(expert_counts) if expert_counts else 0,
            "state_counts": dict(sorted(state_counts.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "queried_bytes": queried_bytes,
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "unknown_bytes": unknown_bytes,
            "gpu_ratio": (gpu_bytes / queried_bytes) if queried_bytes else None,
            "all_expert_slices_gpu_full": all_full_gpu,
            "examples_not_full_gpu": examples_not_full_gpu,
        }
    return summaries


def summarize_kv_events(records: list[dict[str, Any]]) -> dict[str, Any]:
    event_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        event_records[str(record.get("event") or "unknown")].append(record)

    summaries: dict[str, Any] = {}
    for event, items in sorted(event_records.items()):
        state_counts = Counter(str(item.get("residency_state") or "missing") for item in items)
        queried_bytes = 0
        gpu_bytes = 0
        cpu_bytes = 0
        unknown_bytes = 0
        tensor_bytes = 0
        query_ok = 0
        examples_not_full_gpu: list[dict[str, Any]] = []
        for item in items:
            queried = int(item.get("residency_queried_bytes") or 0)
            gpu = int(item.get("resident_gpu_bytes") or 0)
            cpu = int(item.get("resident_cpu_bytes") or 0)
            unknown = int(item.get("resident_unknown_bytes") or 0)
            queried_bytes += queried
            gpu_bytes += gpu
            cpu_bytes += cpu
            unknown_bytes += unknown
            tensor_bytes += int(item.get("size_bytes") or item.get("requested_size_bytes") or 0)
            if bool(item.get("residency_query_ok")):
                query_ok += 1
            state = str(item.get("residency_state") or "")
            if state != "gpu_full" and len(examples_not_full_gpu) < 12:
                examples_not_full_gpu.append(
                    {
                        "path": item.get("path"),
                        "index": item.get("index"),
                        "state": state,
                        "queried_bytes": queried,
                        "gpu_bytes": gpu,
                        "cpu_bytes": cpu,
                        "unknown_bytes": unknown,
                        "ptr_hex": item.get("ptr_hex"),
                        "size_bytes": item.get("size_bytes"),
                    }
                )
        all_full_gpu = bool(items) and state_counts.get("gpu_full", 0) == len(items)
        summaries[event] = {
            "records": len(items),
            "query_ok_records": query_ok,
            "state_counts": dict(sorted(state_counts.items())),
            "tensor_bytes": tensor_bytes,
            "queried_bytes": queried_bytes,
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "unknown_bytes": unknown_bytes,
            "gpu_ratio": (gpu_bytes / queried_bytes) if queried_bytes else None,
            "all_records_gpu_full": all_full_gpu,
            "examples_not_full_gpu": examples_not_full_gpu,
        }
    return summaries


def make_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# profile_run / KV torch.zeros Residency Probe",
        "",
        f"- run_status: {summary.get('run_status')}",
        f"- weight_map: `{summary.get('weight_map')}`",
        f"- kv_probe: `{summary.get('kv_probe')}`",
        "",
        "## Expert weight slices",
        "",
    ]
    weight = summary.get("weight_phases") or {}
    if not weight:
        lines.append("No expert-slice weight records were found.")
    for phase, item in weight.items():
        lines.extend(
            [
                f"### {phase}",
                "",
                f"- records: {item['records']}",
                f"- layers: {item['layers']}",
                (
                    "- experts_per_layer: "
                    f"{item['min_experts_per_layer']}..{item['max_experts_per_layer']}"
                ),
                f"- state_counts: `{item['state_counts']}`",
                f"- gpu_ratio: {item['gpu_ratio']}",
                f"- all_expert_slices_gpu_full: {item['all_expert_slices_gpu_full']}",
                "",
            ]
        )
    lines.extend(["## KV init tensors", ""])
    kv = summary.get("kv_events") or {}
    if not kv:
        lines.append("No KV init records were found.")
    for event, item in kv.items():
        lines.extend(
            [
                f"### {event}",
                "",
                f"- records: {item['records']}",
                f"- state_counts: `{item['state_counts']}`",
                f"- gpu_ratio: {item['gpu_ratio']}",
                f"- all_records_gpu_full: {item['all_records_gpu_full']}",
                "",
            ]
        )
    conclusions = summary.get("conclusions") or {}
    lines.extend(
        [
            "## Conclusions",
            "",
            f"- profile_run_expert_all_gpu_full: {conclusions.get('profile_run_expert_all_gpu_full')}",
            f"- load_model_expert_all_gpu_full: {conclusions.get('load_model_expert_all_gpu_full')}",
            f"- kv_zeros_return_all_gpu_full: {conclusions.get('kv_zeros_return_all_gpu_full')}",
            f"- kv_tensor_ready_all_gpu_full: {conclusions.get('kv_tensor_ready_all_gpu_full')}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--weight-map", type=Path, required=True)
    parser.add_argument("--kv-probe", type=Path, required=True)
    parser.add_argument("--run-status", type=int, default=0)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()

    weight_records = load_jsonl(args.weight_map)
    kv_records = load_jsonl(args.kv_probe)
    weight_phases = summarize_weight_phase(weight_records)
    kv_events = summarize_kv_events(kv_records)

    load_model = weight_phases.get("load_model") or {}
    after_profile = (
        weight_phases.get("after_profile_run")
        or weight_phases.get("after_determine_available_memory_profile_run")
        or {}
    )
    kv_zeros_return = kv_events.get("kv_zeros_return") or {}
    kv_tensor_ready = kv_events.get("kv_tensor_ready") or {}
    summary = {
        "run_root": str(args.run_root),
        "run_status": args.run_status,
        "weight_map": str(args.weight_map),
        "kv_probe": str(args.kv_probe),
        "weight_phases": weight_phases,
        "kv_events": kv_events,
        "conclusions": {
            "profile_run_expert_all_gpu_full": after_profile.get(
                "all_expert_slices_gpu_full"
            ),
            "load_model_expert_all_gpu_full": load_model.get(
                "all_expert_slices_gpu_full"
            ),
            "kv_zeros_return_all_gpu_full": kv_zeros_return.get(
                "all_records_gpu_full"
            ),
            "kv_tensor_ready_all_gpu_full": kv_tensor_ready.get(
                "all_records_gpu_full"
            ),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(make_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["conclusions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
