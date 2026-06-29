#!/usr/bin/env python3
"""Summarize Stage M cold streaming misses by layer and num_tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable


BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2-8", 2, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-512", 129, 512),
    (">512", 513, None),
)


@dataclass
class Record:
    layer_id: int
    step: int
    num_tokens: int
    active_experts: int
    active_hot_experts: int
    active_cold_experts: int
    cold_miss_experts: int
    cold_miss_regions: int
    cold_miss_bytes: int
    prefetch_bytes: int
    eviction_bytes: int


def bucket_num_tokens(num_tokens: int) -> str:
    for label, lo, hi in BUCKETS:
        if num_tokens >= lo and (hi is None or num_tokens <= hi):
            return label
    return "unknown"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def parse_records(path: str) -> list[Record]:
    records: list[Record] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("action") != "cold_streaming_step_summary":
                continue
            num_tokens = data.get("num_tokens")
            if num_tokens is None:
                # Older traces do not have this field; keep them out of the
                # num_tokens analysis instead of guessing from active experts.
                continue
            raw_layer_id = data.get("layer_id")
            layer_id = -1 if raw_layer_id is None else int(raw_layer_id)
            records.append(
                Record(
                    layer_id=layer_id,
                    step=int(data.get("step") or 0),
                    num_tokens=int(num_tokens),
                    active_experts=int(data.get("active_expert_records") or 0),
                    active_hot_experts=int(data.get("active_hot_experts") or 0),
                    active_cold_experts=int(data.get("active_cold_experts") or 0),
                    cold_miss_experts=int(data.get("cold_miss_experts") or 0),
                    cold_miss_regions=int(data.get("cold_miss_region_records") or 0),
                    cold_miss_bytes=int(data.get("cold_miss_bytes") or 0),
                    prefetch_bytes=int(data.get("streaming_prefetch_issued_bytes") or 0),
                    eviction_bytes=int(data.get("streaming_eviction_bytes") or 0),
                )
            )
    return records


def summarize(records: Iterable[Record]) -> dict[str, float | int]:
    rows = list(records)
    miss = [r.cold_miss_experts for r in rows]
    active = [r.active_experts for r in rows]
    tokens = [r.num_tokens for r in rows]
    miss_bytes = [r.cold_miss_bytes for r in rows]
    return {
        "records": len(rows),
        "num_tokens_mean": mean(tokens) if tokens else 0.0,
        "num_tokens_p50": percentile(tokens, 0.50),
        "num_tokens_p90": percentile(tokens, 0.90),
        "active_experts_mean": mean(active) if active else 0.0,
        "active_experts_p50": percentile(active, 0.50),
        "active_experts_p90": percentile(active, 0.90),
        "cold_miss_experts_sum": sum(miss),
        "cold_miss_experts_mean": mean(miss) if miss else 0.0,
        "cold_miss_experts_p50": percentile(miss, 0.50),
        "cold_miss_experts_p90": percentile(miss, 0.90),
        "cold_miss_experts_max": max(miss) if miss else 0,
        "cold_miss_bytes_sum": sum(miss_bytes),
        "cold_miss_bytes_mean": mean(miss_bytes) if miss_bytes else 0.0,
        "prefetch_bytes_sum": sum(r.prefetch_bytes for r in rows),
        "eviction_bytes_sum": sum(r.eviction_bytes for r in rows),
    }


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gib(value: float | int) -> float:
    return float(value) / float(1024**3)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = parse_records(args.trace)
    if not records:
        raise SystemExit("no cold_streaming_step_summary records with num_tokens")

    overall = summarize(records)
    xs = [math.log2(r.num_tokens + 1) for r in records]
    overall["corr_log2_num_tokens_miss_experts"] = corr(
        xs, [r.cold_miss_experts for r in records]
    )
    overall["corr_log2_num_tokens_active_experts"] = corr(
        xs, [r.active_experts for r in records]
    )

    by_bucket: dict[str, list[Record]] = defaultdict(list)
    by_layer: dict[int, list[Record]] = defaultdict(list)
    by_layer_bucket: dict[tuple[int, str], list[Record]] = defaultdict(list)
    for record in records:
        bucket = bucket_num_tokens(record.num_tokens)
        by_bucket[bucket].append(record)
        by_layer[record.layer_id].append(record)
        by_layer_bucket[(record.layer_id, bucket)].append(record)

    bucket_rows = []
    for label, _, _ in BUCKETS:
        summary = summarize(by_bucket.get(label, []))
        summary["num_tokens_bucket"] = label
        bucket_rows.append(summary)

    layer_rows = []
    for layer_id in sorted(by_layer):
        summary = summarize(by_layer[layer_id])
        summary["layer_id"] = layer_id
        layer_rows.append(summary)

    layer_bucket_rows = []
    for (layer_id, bucket), rows in sorted(by_layer_bucket.items()):
        summary = summarize(rows)
        summary["layer_id"] = layer_id
        summary["num_tokens_bucket"] = bucket
        layer_bucket_rows.append(summary)

    write_csv(os.path.join(args.out_dir, "cold_streaming_numtoken_buckets.csv"), bucket_rows)
    write_csv(os.path.join(args.out_dir, "cold_streaming_layer_miss.csv"), layer_rows)
    write_csv(
        os.path.join(args.out_dir, "cold_streaming_layer_numtoken_miss.csv"),
        layer_bucket_rows,
    )
    with open(
        os.path.join(args.out_dir, "cold_streaming_numtoken_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "trace": args.trace,
                "overall": overall,
                "records": len(records),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    report_path = os.path.join(args.out_dir, "cold_streaming_numtoken_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Cold Streaming Miss vs Num Tokens\n\n")
        handle.write(f"- trace: `{args.trace}`\n")
        handle.write(f"- records: `{len(records)}`\n")
        handle.write(
            "- corr(log2(num_tokens+1), miss_experts): "
            f"`{overall['corr_log2_num_tokens_miss_experts']:.4f}`\n"
        )
        handle.write(
            "- corr(log2(num_tokens+1), active_experts): "
            f"`{overall['corr_log2_num_tokens_active_experts']:.4f}`\n\n"
        )
        handle.write("## By Num Tokens\n\n")
        handle.write(
            "| bucket | records | active_mean | miss_mean | miss_p50 | miss_p90 | miss_max | miss GiB | prefetch GiB | eviction GiB |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in bucket_rows:
            handle.write(
                "| {bucket} | {records} | {active:.2f} | {mean:.2f} | {p50:.0f} | {p90:.0f} | {maxv:.0f} | {miss_gib:.3f} | {pref_gib:.3f} | {evict_gib:.3f} |\n".format(
                    bucket=row["num_tokens_bucket"],
                    records=row["records"],
                    active=float(row["active_experts_mean"]),
                    mean=float(row["cold_miss_experts_mean"]),
                    p50=float(row["cold_miss_experts_p50"]),
                    p90=float(row["cold_miss_experts_p90"]),
                    maxv=float(row["cold_miss_experts_max"]),
                    miss_gib=gib(row["cold_miss_bytes_sum"]),
                    pref_gib=gib(row["prefetch_bytes_sum"]),
                    evict_gib=gib(row["eviction_bytes_sum"]),
                )
            )
        handle.write("\n## Top Layers By Miss Bytes\n\n")
        handle.write(
            "| layer | records | num_tokens_mean | active_mean | miss_mean | miss_p90 | miss_max | miss GiB |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(layer_rows, key=lambda r: r["cold_miss_bytes_sum"], reverse=True)[:16]:
            handle.write(
                "| {layer} | {records} | {tokens:.2f} | {active:.2f} | {mean:.2f} | {p90:.0f} | {maxv:.0f} | {miss_gib:.3f} |\n".format(
                    layer=row["layer_id"],
                    records=row["records"],
                    tokens=float(row["num_tokens_mean"]),
                    active=float(row["active_experts_mean"]),
                    mean=float(row["cold_miss_experts_mean"]),
                    p90=float(row["cold_miss_experts_p90"]),
                    maxv=float(row["cold_miss_experts_max"]),
                    miss_gib=gib(row["cold_miss_bytes_sum"]),
                )
            )
        handle.write("\n## Files\n\n")
        handle.write("- buckets_csv: `cold_streaming_numtoken_buckets.csv`\n")
        handle.write("- layer_csv: `cold_streaming_layer_miss.csv`\n")
        handle.write("- layer_bucket_csv: `cold_streaming_layer_numtoken_miss.csv`\n")
        handle.write("- summary_json: `cold_streaming_numtoken_summary.json`\n")

    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
