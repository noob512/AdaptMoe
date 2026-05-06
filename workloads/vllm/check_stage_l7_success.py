#!/usr/bin/env python3
"""Validate Stage L7 Stage-K + Stage-L coordinated KV actions."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Stage L7 by running Stage L6 KV safe offload/prefetch-back "
            "through Stage K action-budget coordinator."
        )
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--mode", choices=("trace_only", "enforce"), default="trace_only")
    parser.add_argument("--coordinator-mode", choices=("trace_only", "enforce"), default="trace_only")
    parser.add_argument("--global-bytes-per-step", type=int, default=512 * 1024**2)
    parser.add_argument("--kv-bytes-per-step", type=int, default=512 * 1024**2)
    parser.add_argument("--weight-bytes-per-step", type=int, default=0)
    parser.add_argument("--scratch-bytes-per-step", type=int, default=0)
    parser.add_argument("--priority", default="kv,scratch,weights")
    parser.add_argument("--global-gpu-bytes", type=int, default=24 * 1024**3)
    parser.add_argument("--weight-gpu-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--kv-gpu-bytes", type=int, default=1 * 1024**3)
    parser.add_argument("--scratch-gpu-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--safety-margin-bytes", type=int, default=1 * 1024**3)
    parser.add_argument("--runtime-budget-blocks", type=int, default=1)
    parser.add_argument("--runtime-candidate-limit", type=int, default=16)
    parser.add_argument("--max-offload-blocks", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--gpu-run", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_stage_l7_success_check_{stamp}")


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_failed_requests(bench_log: Path | None) -> int | None:
    if bench_log is None or not bench_log.is_file():
        return None
    text = bench_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Failed requests:\s+(\d+)", text)
    return int(matches[-1]) if matches else None


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def paths(run_dir: Path) -> dict[str, Path]:
    return {
        "residency": run_dir / "vllm_uvm_pool_residency_stage_l7.jsonl",
        "runtime": run_dir / "vllm_uvm_kv_runtime_stage_l7.jsonl",
        "coordinator": run_dir / "vllm_uvm_pool_coordinator_stage_l7.jsonl",
        "pointer_map": run_dir / "vllm_uvm_kv_pointer_map_stage_l7.jsonl",
        "bench": run_dir / "vllm_bench_stage_l7.log",
        "runner": run_dir / "stage_l7_gpu_runner.log",
        "allocator": run_dir / "vllm_uvm_allocator_trace_stage_l7.log",
    }


def run_gpu_probe(args: argparse.Namespace, run_dir: Path) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = paths(run_dir)
    for path in (
        p["residency"],
        p["runtime"],
        p["coordinator"],
        p["pointer_map"],
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    cmd = [
        "./run_kv_fault_ratio.sh",
        "--model", args.model,
        "--mode", "trace",
        "--allocator-log", str(p["allocator"]),
        "--trace-log", str(run_dir / "uvm_kv_fault_stats_stage_l7.log"),
        "--address-log", str(run_dir / "vllm_uvm_address_regions_stage_l7.log"),
        "--server-log", str(run_dir / "vllm_server_stage_l7.log"),
        "--bench-log", str(p["bench"]),
        "--uvm-residency-enable", "1",
        "--uvm-residency-stage", "stage_l6",
        "--uvm-residency-mode", args.mode,
        "--uvm-residency-trace-file", str(p["residency"]),
        "--uvm-residency-global-gpu-bytes", str(args.global_gpu_bytes),
        "--uvm-residency-weight-gpu-bytes", str(args.weight_gpu_bytes),
        "--uvm-residency-kv-gpu-bytes", str(args.kv_gpu_bytes),
        "--uvm-residency-scratch-gpu-bytes", str(args.scratch_gpu_bytes),
        "--uvm-residency-safety-margin-bytes", str(args.safety_margin_bytes),
        "--uvm-residency-kv-metadata-enable", "1",
        "--uvm-residency-kv-data-offload-enable", "1",
        "--uvm-residency-kv-data-offload-mode", "prefetch_cpu",
        "--uvm-residency-kv-prefetch-back-enable", "1",
        "--uvm-residency-kv-prefetch-back-mode", "prefetch",
        "--uvm-residency-kv-max-offload-blocks-per-step", str(args.max_offload_blocks),
        "--uvm-residency-kv-pointer-map-file", str(p["pointer_map"]),
        "--uvm-kv-runtime-enable", "1",
        "--uvm-kv-runtime-mode", "trace_only",
        "--uvm-kv-runtime-budget-blocks", str(args.runtime_budget_blocks),
        "--uvm-kv-runtime-trace-file", str(p["runtime"]),
        "--uvm-kv-runtime-candidate-limit", str(args.runtime_candidate_limit),
        "--uvm-kv-runtime-prefix-evict-enable", "0",
        "--uvm-pool-coordinator-enable", "1",
        "--uvm-pool-coordinator-mode", args.coordinator_mode,
        "--uvm-pool-coordinator-trace-file", str(p["coordinator"]),
        "--uvm-pool-coordinator-global-bytes-per-step", str(args.global_bytes_per_step),
        "--uvm-pool-coordinator-kv-bytes-per-step", str(args.kv_bytes_per_step),
        "--uvm-pool-coordinator-weight-bytes-per-step", str(args.weight_bytes_per_step),
        "--uvm-pool-coordinator-scratch-bytes-per-step", str(args.scratch_bytes_per_step),
        "--uvm-pool-coordinator-priority", args.priority,
        "--prompts", str(args.prompts),
        "--request-rate", str(args.request_rate),
        "--output-len", str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L7 Success Check: running coordinated KV action probe")
    print(f" Output dir: {run_dir}")
    print(f" Residency trace: {p['residency']}")
    print(f" Runtime trace: {p['runtime']}")
    print(f" Coordinator trace: {p['coordinator']}")
    print(f" Pointer map: {p['pointer_map']}")
    print(f" Mode: {args.mode}, coordinator_mode: {args.coordinator_mode}")
    print("===========================================================")
    with p["runner"].open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        rc = process.wait()
    if rc != 0:
        raise SystemExit(f"Stage L7 GPU probe failed with exit code {rc}; log={p['runner']}")
    return p


def summarize_residency(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("action", "unknown")) for record in records)
    summary = {}
    unsafe = 0
    for record in records:
        if record.get("action") == "residency_summary":
            summary = record
        if record.get("action") == "residency_kv_data_offload_fallback":
            if (record.get("metadata") or {}).get("safe") is False:
                unsafe += 1
    return {
        "records": len(records),
        "action_counts": dict(counts),
        "summary": summary,
        "offload_issued": counts.get("residency_kv_data_offload_issued", 0),
        "prefetch_issued": counts.get("residency_kv_prefetch_back_issued", 0),
        "offload_fallback": counts.get("residency_kv_data_offload_fallback", 0),
        "unsafe_offloads": unsafe,
    }


def summarize_coordinator(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("action", "unknown")) for record in records)
    requests = [
        record for record in records
        if record.get("action") == "coordinator_request"
    ]
    kv_requests = [record for record in requests if record.get("pool") == "kv"]
    kv_stage_l7 = [
        record for record in kv_requests
        if (record.get("metadata") or {}).get("stage") == "stage_l7"
    ]
    kv_grants = [record for record in kv_stage_l7 if record.get("allowed") is True]
    kv_denies = [record for record in kv_stage_l7 if record.get("allowed") is False]
    kv_actions = Counter(str(record.get("requested_action", "unknown")) for record in kv_stage_l7)
    config = {}
    summary = {}
    for record in records:
        if record.get("action") == "coordinator_config":
            config = record
        elif record.get("action") == "coordinator_summary":
            summary = record
    return {
        "records": len(records),
        "action_counts": dict(counts),
        "config": config,
        "summary": summary,
        "kv_stage_l7_requests": len(kv_stage_l7),
        "kv_stage_l7_grants": len(kv_grants),
        "kv_stage_l7_denies": len(kv_denies),
        "kv_stage_l7_actions": dict(kv_actions),
    }


def validate(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    p = paths(run_dir)
    residency = summarize_residency(read_jsonl(p["residency"]))
    coordinator = summarize_coordinator(read_jsonl(p["coordinator"]))
    pointer_map_records = [
        record for record in read_jsonl(p["pointer_map"])
        if record.get("action") == "stage_l6_kv_pointer_map"
    ]
    pointer_map_entries = sum(
        int(record.get("mapped_blocks") or 0) for record in pointer_map_records
    )
    failed_requests = parse_failed_requests(p["bench"])
    checks = [
        check("residency_trace_present", p["residency"].is_file(), f"path={p['residency']}"),
        check("coordinator_trace_present", p["coordinator"].is_file(), f"path={p['coordinator']}"),
        check("pointer_map_present", p["pointer_map"].is_file(), f"entries={pointer_map_entries}"),
        check(
            "stage_k_config_present",
            bool(coordinator["config"]),
            f"config={coordinator['config']}",
        ),
        check(
            "coordinator_mode_matches",
            coordinator["config"].get("mode") == args.coordinator_mode,
            f"mode={coordinator['config'].get('mode')} expected={args.coordinator_mode}",
        ),
        check(
            "kv_stage_l7_coordinator_grants_present",
            coordinator["kv_stage_l7_grants"] > 0,
            f"kv_grants={coordinator['kv_stage_l7_grants']} actions={coordinator['kv_stage_l7_actions']}",
        ),
        check(
            "kv_stage_l7_coordinator_denies_zero",
            coordinator["kv_stage_l7_denies"] == 0,
            f"kv_denies={coordinator['kv_stage_l7_denies']}",
        ),
        check(
            "kv_safe_offload_issued_present",
            residency["offload_issued"] > 0,
            f"offload_issued={residency['offload_issued']}",
        ),
        check(
            "kv_prefetch_back_issued_present",
            residency["prefetch_issued"] > 0,
            f"prefetch_issued={residency['prefetch_issued']}",
        ),
        check(
            "unsafe_kv_offloads_zero",
            residency["unsafe_offloads"] == 0,
            f"unsafe_offloads={residency['unsafe_offloads']}",
        ),
        check(
            "benchmark_no_failed_requests",
            failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_l7_success_check.json"
    )
    result = {
        "passed": passed,
        "run_dir": str(run_dir),
        "paths": {key: str(value) for key, value in p.items()},
        "residency_summary": residency,
        "coordinator_summary": coordinator,
        "pointer_map_summary": {
            "records": len(pointer_map_records),
            "mapped_blocks": pointer_map_entries,
        },
        "checks": checks,
        "output_json": str(output_json),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def print_summary(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    residency = result["residency_summary"]
    coordinator = result["coordinator_summary"]
    print("===========================================================")
    print(f" Stage L7 Success Check: {status}")
    print(f" Run dir: {result['run_dir']}")
    print("===========================================================")
    print(
        "- kv_l7="
        f"coordinator_grants={coordinator['kv_stage_l7_grants']} "
        f"coordinator_denies={coordinator['kv_stage_l7_denies']} "
        f"actions={coordinator['kv_stage_l7_actions']} "
        f"offload_issued={residency['offload_issued']} "
        f"prefetch_issued={residency['prefetch_issued']} "
        f"unsafe_offloads={residency['unsafe_offloads']}"
    )
    print(f"- pointer_map={result['pointer_map_summary']}")
    print("- checks:")
    for item in result["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={result['output_json']}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    if not args.skip_run:
        if not args.gpu_run:
            raise SystemExit("Stage L7 check currently requires --gpu-run or --skip-run.")
        run_gpu_probe(args, run_dir)
    result = validate(args, run_dir)
    print_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
