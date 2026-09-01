#!/usr/bin/env python3
"""Verify and aggregate the E9 KTransformers one-factor formal matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate(values: list[float], *, seed: int) -> dict[str, Any]:
    require(values, "empty metric")
    rng = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(10_000)
    ]
    return {
        "count": count,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "median_bootstrap_95_ci": [percentile(medians, 0.025), percentile(medians, 0.975)],
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    config = load(root / "run_config.json")
    blocks = int(config["process_blocks"])
    minimum = 1 if config["mode"] == "qualification" else 5
    require(blocks >= minimum, "insufficient process blocks")
    expected_arms = set(config["arms"])

    rows = [
        line.split("\t")
        for line in (root / "execution_status.tsv").read_text(encoding="utf-8").splitlines()[1:]
        if line
    ]
    require(len(rows) == blocks * len(expected_arms), "case count mismatch")
    require(all(row[-1] == "0" for row in rows), "one or more E9 cases failed")
    seen: set[tuple[int, str]] = set()
    metrics: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    hashes: dict[str, set[str]] = defaultdict(set)
    samples: list[dict[str, Any]] = []
    for block_raw, _position, arm, case_label, _rc in rows:
        block = int(block_raw)
        require((block, arm) not in seen, f"duplicate block/arm: {block}/{arm}")
        seen.add((block, arm))
        case_root = root / "cases" / case_label
        benchmark = load(case_root / "benchmark.json")
        runtime = load(case_root / "runtime_verification.json")
        metadata = load(case_root / "case_metadata.json")
        require(benchmark.get("status") == "ok", f"{case_label}: benchmark status")
        require(runtime.get("status") == "ok", f"{case_label}: runtime verifier")
        require((case_root / "kernel_log_returncode.txt").read_text().strip() == "0", f"{case_label}: kernel log")
        kernel = (case_root / "kernel.since.log").read_text(encoding="utf-8", errors="replace").lower()
        require("nvrm: xid" not in kernel and "fault_pde" not in kernel, f"{case_label}: kernel fault")
        expected = config["arms"][arm]
        require(int(metadata["cpu_threads"]) == int(expected["threads"]), f"{case_label}: threads")
        require(bool(metadata["cuda_graph_enabled"]) is bool(expected["graph"]), f"{case_label}: graph")
        require(metadata["placement_strategy"] == expected["placement"], f"{case_label}: placement")
        require(bool(metadata["overlap_enabled"]) is bool(expected["overlap"]), f"{case_label}: overlap")
        require(bool(metadata["mixed_chunk_enabled"]) is bool(expected["mixed"]), f"{case_label}: mixed")
        phases = benchmark.get("phases")
        require(isinstance(phases, list) and len(phases) == 2, f"{case_label}: phases")
        for phase in phases:
            phase_name = str(phase["phase"])
            require(phase_name in {"c1", "c8"}, f"{case_label}: phase name")
            require(phase.get("all_requests_produced_fixed_output") is True, f"{case_label}/{phase_name}: output")
            output_hash = str(phase["output_fingerprints_sha256"])
            hashes[phase_name].add(output_hash)
            values = {
                "throughput_tps": float(phase["output_tokens_per_second"]),
                "mean_ttft_ms": float(phase["mean_ttft_ms"]),
                "p99_ttft_ms": float(phase["p99_ttft_ms"]),
                "mean_tpot_ms": float(phase["mean_tpot_ms"]),
                "p99_tpot_ms": float(phase["p99_tpot_ms"]),
            }
            for metric, value in values.items():
                require(math.isfinite(value) and value >= 0, f"{case_label}/{phase_name}/{metric}")
                metrics[(arm, phase_name, metric)].append(value)
            samples.append({"block": block, "arm": arm, "case": case_label, "phase": phase_name, "output_sha256": output_hash, **values})
    require(len(seen) == blocks * len(expected_arms), "incomplete block/arm coverage")
    require(all(len(values) == 1 for values in hashes.values()), "exact output drift across E9 arms")

    summary: dict[str, Any] = {}
    seed = 20260828
    for arm in sorted(expected_arms):
        summary[arm] = {}
        for phase in ("c1", "c8"):
            summary[arm][phase] = {}
            for metric in ("throughput_tps", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"):
                seed += 1
                values = metrics[(arm, phase, metric)]
                require(len(values) == blocks, f"{arm}/{phase}/{metric}: sample count")
                summary[arm][phase][metric] = aggregate(values, seed=seed)

    payload = {
        "schema": "eurosys_e9_ktransformers_formal_verification_v1",
        "status": "ok",
        "mode": config["mode"],
        "process_blocks": blocks,
        "case_count": len(rows),
        "arms": config["arms"],
        "numa_scope": config["numa_scope"],
        "exact_outputs_equal_across_arms": True,
        "output_sha256": {phase: next(iter(values)) for phase, values in hashes.items()},
        "samples": samples,
        "aggregate": summary,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
