#!/usr/bin/env python3
"""Aggregate Qwen3-MoE per-layer CUDA timing traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_run_root(value: str) -> tuple[str, Path]:
    try:
        pattern, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "run roots must use PATTERN=PATH"
        ) from exc
    if not pattern or not path:
        raise argparse.ArgumentTypeError("run roots must use PATTERN=PATH")
    return pattern, Path(path)


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_manifest_layer_bytes(path: Path) -> dict[int, int]:
    layer_bytes: dict[int, int] = defaultdict(int)
    if not path.exists():
        return {}
    for record in load_json_lines(path):
        if record.get("event") != "cpu_offload_parameter":
            continue
        name = str(record.get("full_parameter_name", ""))
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match:
            layer_bytes[int(match.group(1))] += int(record.get("bytes", 0))
    return dict(layer_bytes)


def summarize_samples(samples: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": percentile(samples, 95.0),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        type=parse_run_root,
        help="A benchmark root in PATTERN=PATH form; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--burst-forwards", type=int, default=1)
    args = parser.parse_args()

    expected_forwards = args.warmups + args.iterations + args.burst_forwards
    layer_rows: list[dict[str, Any]] = []
    benchmark_results: dict[tuple[str, str, int], dict[str, Any]] = {}
    manifest_bytes: dict[tuple[str, str], dict[int, int]] = {}

    for pattern, root in args.run_root:
        root_summary = json.loads((root / "summary.json").read_text())
        for case in root_summary["cases"]:
            label = str(case["label"])
            for result in case["results"]:
                key = (pattern, label, int(result["num_tokens"]))
                benchmark_results[key] = result

            case_dir = root / label
            manifest_bytes[(pattern, label)] = load_manifest_layer_bytes(
                case_dir / "cpu_offload_manifest.jsonl"
            )
            timing_path = case_dir / "layer_timing.jsonl"
            grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for record in load_json_lines(timing_path):
                if record.get("event") != "qwen3_moe_layer_timing":
                    continue
                key = (int(record["num_tokens"]), int(record["layer_id"]))
                grouped[key].append(record)

            for (num_tokens, layer_id), records in sorted(grouped.items()):
                records.sort(key=lambda item: float(item["timestamp"]))
                if len(records) < expected_forwards:
                    raise RuntimeError(
                        f"{timing_path}: only {len(records)} records for "
                        f"tokens={num_tokens}, layer={layer_id}; "
                        f"expected at least {expected_forwards}"
                    )
                benchmark_records = records[-expected_forwards:]
                measured = benchmark_records[
                    args.warmups : args.warmups + args.iterations
                ]
                samples = [float(record["elapsed_ms"]) for record in measured]
                layer_rows.append(
                    {
                        "pattern": pattern,
                        "case": label,
                        "num_tokens": num_tokens,
                        "layer_id": layer_id,
                        "offload_bytes": manifest_bytes[(pattern, label)].get(
                            layer_id, 0
                        ),
                        **summarize_samples(samples),
                    }
                )

    expected_layers = set(range(48))
    rows_by_case: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        key = (row["pattern"], row["case"], row["num_tokens"])
        rows_by_case[key].append(row)

    case_summaries = []
    for key, rows in sorted(rows_by_case.items()):
        rows.sort(key=lambda row: int(row["layer_id"]))
        observed_layers = {int(row["layer_id"]) for row in rows}
        if observed_layers != expected_layers:
            raise RuntimeError(f"incomplete layer set for {key}: {observed_layers}")
        medians = [float(row["median_ms"]) for row in rows]
        slowest = max(rows, key=lambda row: float(row["median_ms"]))
        benchmark = benchmark_results[key]
        case_summaries.append(
            {
                "pattern": key[0],
                "case": key[1],
                "num_tokens": key[2],
                "sum_layer_median_ms": sum(medians),
                "mean_layer_median_ms": statistics.fmean(medians),
                "min_layer_median_ms": min(medians),
                "max_layer_median_ms": max(medians),
                "slowest_layer_id": int(slowest["layer_id"]),
                "layers_0_9_sum_ms": sum(medians[:10]),
                "layer_10_ms": medians[10],
                "layers_11_47_sum_ms": sum(medians[11:]),
                "instrumented_forward_median_ms": float(benchmark["median_ms"]),
            }
        )

    row_index = {
        (row["pattern"], row["case"], row["num_tokens"], row["layer_id"]): row
        for row in layer_rows
    }
    comparisons = []
    patterns = sorted({str(row["pattern"]) for row in layer_rows})
    num_tokens_values = sorted({int(row["num_tokens"]) for row in layer_rows})
    for pattern in patterns:
        labels = sorted(
            {str(row["case"]) for row in layer_rows if row["pattern"] == pattern}
        )
        if "local_hbm" not in labels:
            continue
        for label in (value for value in labels if value != "local_hbm"):
            for num_tokens in num_tokens_values:
                layer_comparisons = []
                for layer_id in range(48):
                    local = row_index[(pattern, "local_hbm", num_tokens, layer_id)]
                    remote = row_index[(pattern, label, num_tokens, layer_id)]
                    local_ms = float(local["median_ms"])
                    remote_ms = float(remote["median_ms"])
                    layer_comparisons.append(
                        {
                            "layer_id": layer_id,
                            "local_median_ms": local_ms,
                            "remote_median_ms": remote_ms,
                            "delta_ms": remote_ms - local_ms,
                            "overhead_pct": (remote_ms / local_ms - 1.0) * 100.0,
                            "offload_bytes": int(remote["offload_bytes"]),
                        }
                    )
                local_sum = sum(row["local_median_ms"] for row in layer_comparisons)
                remote_sum = sum(row["remote_median_ms"] for row in layer_comparisons)
                comparisons.append(
                    {
                        "pattern": pattern,
                        "remote_case": label,
                        "num_tokens": num_tokens,
                        "local_sum_layer_median_ms": local_sum,
                        "remote_sum_layer_median_ms": remote_sum,
                        "sum_layer_overhead_pct": (remote_sum / local_sum - 1.0)
                        * 100.0,
                        "layers_0_9_delta_ms": sum(
                            row["delta_ms"] for row in layer_comparisons[:10]
                        ),
                        "layer_10_delta_ms": layer_comparisons[10]["delta_ms"],
                        "layers_11_47_delta_ms": sum(
                            row["delta_ms"] for row in layer_comparisons[11:]
                        ),
                        "top_delta_layers": sorted(
                            layer_comparisons,
                            key=lambda row: row["delta_ms"],
                            reverse=True,
                        )[:10],
                        "layers": layer_comparisons,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_rows.sort(
        key=lambda row: (
            row["pattern"],
            row["case"],
            row["num_tokens"],
            row["layer_id"],
        )
    )
    with (args.output_dir / "layer_timings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)

    summary = {
        "config": {
            "warmups_discarded": args.warmups,
            "iterations": args.iterations,
            "burst_forwards_discarded": args.burst_forwards,
            "timing": "CUDA event around each Qwen3MoeDecoderLayer.forward",
        },
        "run_roots": [
            {"pattern": pattern, "path": str(path)}
            for pattern, path in args.run_root
        ],
        "case_summaries": case_summaries,
        "comparisons": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
