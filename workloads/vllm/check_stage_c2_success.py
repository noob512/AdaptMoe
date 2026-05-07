#!/usr/bin/env python3
"""Run or validate a Stage C2 cuda_malloc_async success check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the current Stage C2 implementation is successful. "
            "By default this runs a small C1 cuda_malloc vs C2 cuda_malloc_async "
            "backend A/B experiment with an explicit CUDA mempool release threshold. "
            "Use --backend-json or --run-dir for offline validation."
        )
    )
    parser.add_argument(
        "--backend-json",
        help="Existing run_stage_c_attention_backend_ab.sh comparison JSON.",
    )
    parser.add_argument(
        "--run-dir",
        help="Existing backend A/B root directory containing the comparison JSON.",
    )
    parser.add_argument("--prompts", type=int, default=5)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="512")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--budget-bytes", type=int, default=1048576)
    parser.add_argument(
        "--pool-release-threshold",
        default="1048576",
        help=(
            "Expected async CUDA mempool release threshold. Empty string means "
            "validate CUDA default/no explicit pool config."
        ),
    )
    parser.add_argument("--out-dir", help="Output directory for a new check run.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not run a new experiment; requires --backend-json or --run-dir.",
    )
    parser.add_argument(
        "--no-require-effectiveness-vs-trace",
        action="store_true",
        help=(
            "Do not fail if the C2 run does not reduce gap/unknown faults versus "
            "its trace-only baseline. Correctness checks are still enforced."
        ),
    )
    parser.add_argument(
        "--require-not-worse-than-sync",
        action="store_true",
        help=(
            "Also require cuda_malloc_async to be no worse than cuda_malloc on the "
            "backend A/B comparison checks. This is intentionally off by default."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def pct_text(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:+.2f}%"


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def default_out_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_c2_success_check_{stamp}")


def report_path_for_run_dir(run_dir: Path, prompts: int) -> Path:
    return run_dir / f"vllm_stage_c_attention_backend_p{prompts}_comparison.json"


def run_experiment(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "stage_c2_success_check_runner.log"
    report_json = report_path_for_run_dir(out_dir, args.prompts)

    env = os.environ.copy()
    env.update(
        {
            "PROMPTS": str(args.prompts),
            "REQUEST_RATE": str(args.request_rate),
            "OUTPUT_LEN": str(args.output_len),
            "MODEL": str(args.model),
            "DEVICE_DIRECT_MAX_TOTAL_BYTES": str(args.budget_bytes),
            "ASYNC_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD": str(
                args.pool_release_threshold
            ),
            "ROOT_OUT_DIR": str(out_dir),
        }
    )

    cmd = ["./run_stage_c_attention_backend_ab.sh"]
    print("===========================================================")
    print(" Stage C2 Success Check: running backend A/B experiment")
    print(f" Output dir: {out_dir}")
    print(f" Model: {args.model}")
    print(f" Prompts: {args.prompts}")
    print(f" Budget bytes: {args.budget_bytes}")
    print(" Sync backend: cuda_malloc")
    print(" Async backend: cuda_malloc_async")
    print(
        " Async pool release threshold: "
        f"{args.pool_release_threshold or '<cuda-default>'}"
    )
    print("===========================================================")

    with run_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
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
        raise SystemExit(f"Stage C2 experiment failed with exit code {rc}; log={run_log}")
    if not report_json.is_file():
        raise SystemExit(f"Stage C2 report was not produced: {report_json}")
    return report_json, run_log


def backend_count(run: dict[str, Any], backend: str) -> int:
    counts = get_dict(run.get("device_direct_backend_counts"))
    return int(counts.get(backend) or 0)


def failed_requests(run: dict[str, Any]) -> Any:
    return run.get("failed_requests")


def validate_pool_config(
    *,
    async_run: dict[str, Any],
    expected_threshold: str,
) -> list[dict[str, Any]]:
    threshold_requested = expected_threshold != ""
    threshold_value = as_int(expected_threshold) if threshold_requested else None
    actual_set = async_run.get("device_direct_pool_release_threshold_set")
    actual_threshold = async_run.get("device_direct_pool_release_threshold")
    attempted = async_run.get("device_direct_pool_config_attempted")
    success = async_run.get("device_direct_pool_config_success")
    error = async_run.get("device_direct_pool_config_error")

    if threshold_requested:
        return [
            check(
                "async_pool_threshold_set",
                actual_set is True,
                f"device_direct_pool_release_threshold_set={actual_set}",
            ),
            check(
                "async_pool_threshold_matches",
                as_int(actual_threshold) == threshold_value,
                (
                    "device_direct_pool_release_threshold="
                    f"{actual_threshold} expected={threshold_value}"
                ),
            ),
            check(
                "async_pool_config_attempted",
                attempted == 1,
                f"device_direct_pool_config_attempted={attempted}",
            ),
            check(
                "async_pool_config_success",
                success == 1,
                f"device_direct_pool_config_success={success}",
            ),
            check(
                "async_pool_config_error_none",
                error in (None, "none"),
                f"device_direct_pool_config_error={error}",
            ),
        ]

    return [
        check(
            "async_pool_threshold_not_forced",
            actual_set in (False, None),
            f"device_direct_pool_release_threshold_set={actual_set}",
        ),
        check(
            "async_pool_config_not_required",
            attempted in (0, None),
            f"device_direct_pool_config_attempted={attempted}",
        ),
    ]


def validate_report(
    *,
    backend_json: Path,
    run_dir: Path | None,
    run_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report = load_json(backend_json)
    sync_run = get_dict(report.get("cuda_malloc"))
    async_run = get_dict(report.get("cuda_malloc_async"))
    comparison = get_dict(report.get("backend_comparison"))
    comparison_checks = get_dict(comparison.get("checks"))
    async_checks = get_dict(async_run.get("checks"))

    sync_peak_live = sync_run.get("device_direct_peak_live_bytes_observed")
    sync_max_total = sync_run.get("device_direct_max_total_bytes")
    async_peak_live = async_run.get("device_direct_peak_live_bytes_observed")
    async_max_total = async_run.get("device_direct_max_total_bytes")
    threshold_requested = str(args.pool_release_threshold) != ""
    direct_pool_config_ok = (
        not threshold_requested
        or (
            async_run.get("device_direct_pool_config_attempted") == 1
            and async_run.get("device_direct_pool_config_success") == 1
            and async_run.get("device_direct_pool_config_error") in (None, "none")
        )
    )

    checks: list[dict[str, Any]] = [
        check(
            "backend_report_has_sync_and_async_runs",
            bool(sync_run) and bool(async_run) and bool(comparison),
            f"sync_present={bool(sync_run)} async_present={bool(async_run)}",
        ),
        check(
            "backend_correctness_signal",
            comparison.get("correctness_signal") is True,
            f"correctness_signal={comparison.get('correctness_signal')}",
        ),
        check(
            "sync_success_signal",
            sync_run.get("success_signal") is True,
            f"sync_success_signal={sync_run.get('success_signal')}",
        ),
        check(
            "async_success_signal",
            async_run.get("success_signal") is True,
            f"async_success_signal={async_run.get('success_signal')}",
        ),
        check(
            "async_effectiveness_vs_trace",
            args.no_require_effectiveness_vs_trace
            or async_run.get("effectiveness_signal") is True,
            f"async_effectiveness_signal={async_run.get('effectiveness_signal')}",
        ),
        check(
            "sync_backend_used",
            backend_count(sync_run, "cuda_malloc") > 0,
            f"sync_backend_counts={sync_run.get('device_direct_backend_counts')}",
        ),
        check(
            "async_backend_used",
            backend_count(async_run, "cuda_malloc_async") > 0,
            f"async_backend_counts={async_run.get('device_direct_backend_counts')}",
        ),
        check(
            "async_did_not_use_sync_backend",
            backend_count(async_run, "cuda_malloc") == 0,
            f"async_backend_counts={async_run.get('device_direct_backend_counts')}",
        ),
        check(
            "sync_no_failed_requests",
            failed_requests(sync_run) == 0,
            f"sync_failed_requests={failed_requests(sync_run)}",
        ),
        check(
            "async_no_failed_requests",
            failed_requests(async_run) == 0,
            f"async_failed_requests={failed_requests(async_run)}",
        ),
        check(
            "sync_gap_policy_fail_zero",
            sync_run.get("gap_policy_fail") == 0,
            f"sync_gap_policy_fail={sync_run.get('gap_policy_fail')}",
        ),
        check(
            "async_gap_policy_fail_zero",
            async_run.get("gap_policy_fail") == 0,
            f"async_gap_policy_fail={async_run.get('gap_policy_fail')}",
        ),
        check(
            "sync_device_direct_records_present",
            int(sync_run.get("device_direct_actual_records") or 0) > 0,
            f"sync_actual_records={sync_run.get('device_direct_actual_records')}",
        ),
        check(
            "async_device_direct_records_present",
            int(async_run.get("device_direct_actual_records") or 0) > 0,
            f"async_actual_records={async_run.get('device_direct_actual_records')}",
        ),
        check(
            "sync_peak_live_within_budget",
            sync_max_total in (None, 0)
            or (
                sync_peak_live is not None
                and int(sync_peak_live) <= int(sync_max_total)
            ),
            f"sync_peak_live={sync_peak_live} sync_max_total={sync_max_total}",
        ),
        check(
            "async_peak_live_within_budget",
            async_max_total in (None, 0)
            or (
                async_peak_live is not None
                and int(async_peak_live) <= int(async_max_total)
            ),
            f"async_peak_live={async_peak_live} async_max_total={async_max_total}",
        ),
        check(
            "async_probe_main_same_gap",
            async_checks.get("device_probe_main_same_gap") is True,
            (
                "device_probe_main_same_gap="
                f"{async_checks.get('device_probe_main_same_gap')}"
            ),
        ),
        check(
            "async_pool_config_ok_if_requested",
            (
                comparison_checks.get("async_pool_config_ok_if_requested") is True
                or async_checks.get("device_pool_config_ok_if_requested") is True
                or direct_pool_config_ok
            ),
            (
                "backend_check="
                f"{comparison_checks.get('async_pool_config_ok_if_requested')} "
                "async_run_check="
                f"{async_checks.get('device_pool_config_ok_if_requested')} "
                f"direct_metrics_ok={direct_pool_config_ok}"
            ),
        ),
    ]
    checks.extend(
        validate_pool_config(
            async_run=async_run,
            expected_threshold=str(args.pool_release_threshold),
        )
    )

    if args.require_not_worse_than_sync:
        checks.extend(
            [
                check(
                    "async_gap_faults_not_higher_than_sync",
                    comparison_checks.get("async_gap_faults_not_higher_than_sync")
                    is True,
                    (
                        "async_gap_faults_not_higher_than_sync="
                        f"{comparison_checks.get('async_gap_faults_not_higher_than_sync')}"
                    ),
                ),
                check(
                    "async_tpot_not_more_than_10pct_worse_than_sync",
                    comparison_checks.get(
                        "async_tpot_not_more_than_10pct_worse_than_sync"
                    )
                    is True,
                    (
                        "async_tpot_not_more_than_10pct_worse_than_sync="
                        f"{comparison_checks.get('async_tpot_not_more_than_10pct_worse_than_sync')}"
                    ),
                ),
                check(
                    "async_throughput_not_more_than_10pct_worse_than_sync",
                    comparison_checks.get(
                        "async_throughput_not_more_than_10pct_worse_than_sync"
                    )
                    is True,
                    (
                        "async_throughput_not_more_than_10pct_worse_than_sync="
                        f"{comparison_checks.get('async_throughput_not_more_than_10pct_worse_than_sync')}"
                    ),
                ),
            ]
        )

    runner_log_check: dict[str, Any] | None = None
    if run_log and run_log.is_file():
        text = run_log.read_text(encoding="utf-8", errors="replace")
        runner_log_check = {
            "passed": "Failed to parse stats lines" not in text,
            "detail": f"runner_log={run_log}",
            "contains_parse_failure": "Failed to parse stats lines" in text,
            "contains_backend_comparison": "Stage C Backend Comparison" in text,
        }
        checks.append(
            check(
                "runner_log_parse_clean",
                bool(runner_log_check["passed"]),
                (
                    "no stats parse failure in runner log"
                    if runner_log_check["passed"]
                    else str(runner_log_check)
                ),
            )
        )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "backend_json": str(backend_json),
        "run_dir": str(run_dir) if run_dir else None,
        "run_log": str(run_log) if run_log else None,
        "summary": {
            "sync_gap_faults": sync_run.get("gap_faults"),
            "async_gap_faults": async_run.get("gap_faults"),
            "gap_fault_delta_pct_async_vs_sync": comparison.get(
                "gap_fault_delta_pct_async_vs_sync"
            ),
            "async_gap_fault_delta_pct_vs_trace": async_run.get(
                "gap_fault_delta_pct_vs_trace"
            ),
            "async_unknown_fault_delta_pct_vs_trace": async_run.get(
                "unknown_fault_delta_pct_vs_trace"
            ),
            "sync_output_tok_s": sync_run.get("output_throughput_tok_s"),
            "async_output_tok_s": async_run.get("output_throughput_tok_s"),
            "output_throughput_delta_pct_async_vs_sync": comparison.get(
                "output_throughput_delta_pct_async_vs_sync"
            ),
            "sync_mean_tpot_ms": sync_run.get("mean_tpot_ms"),
            "async_mean_tpot_ms": async_run.get("mean_tpot_ms"),
            "mean_tpot_delta_pct_async_vs_sync": comparison.get(
                "mean_tpot_delta_pct_async_vs_sync"
            ),
            "sync_actual_records": sync_run.get("device_direct_actual_records"),
            "async_actual_records": async_run.get("device_direct_actual_records"),
            "async_budget_reject_records": async_run.get(
                "device_direct_budget_reject_records"
            ),
            "async_peak_live_bytes_observed": async_peak_live,
            "async_max_total_bytes": async_max_total,
            "async_backend_counts": async_run.get("device_direct_backend_counts"),
            "async_pool_release_threshold_set": async_run.get(
                "device_direct_pool_release_threshold_set"
            ),
            "async_pool_release_threshold": async_run.get(
                "device_direct_pool_release_threshold"
            ),
            "async_pool_config_attempted": async_run.get(
                "device_direct_pool_config_attempted"
            ),
            "async_pool_config_success": async_run.get(
                "device_direct_pool_config_success"
            ),
            "async_pool_config_error": async_run.get(
                "device_direct_pool_config_error"
            ),
            "backend_correctness_signal": comparison.get("correctness_signal"),
            "backend_async_effectiveness_signal": comparison.get(
                "async_effectiveness_signal"
            ),
        },
        "checks": checks,
        "runner_log_check": runner_log_check,
    }


def print_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    summary = result["summary"]
    print("===========================================================")
    print(f" Stage C2 Success Check: {status}")
    print(f" Backend report: {result['backend_json']}")
    if result.get("run_dir"):
        print(f" Run dir: {result['run_dir']}")
    print("===========================================================")
    print(
        "- C2 vs trace: "
        f"gap_delta={pct_text(summary.get('async_gap_fault_delta_pct_vs_trace'))} "
        "unknown_delta="
        f"{pct_text(summary.get('async_unknown_fault_delta_pct_vs_trace'))}"
    )
    print(
        "- C2 vs C1 backend: "
        f"sync_gap={summary.get('sync_gap_faults')} "
        f"async_gap={summary.get('async_gap_faults')} "
        f"gap_delta={pct_text(summary.get('gap_fault_delta_pct_async_vs_sync'))}"
    )
    print(
        "- throughput/tpot vs C1: "
        f"sync_tok_s={summary.get('sync_output_tok_s')} "
        f"async_tok_s={summary.get('async_output_tok_s')} "
        "throughput_delta="
        f"{pct_text(summary.get('output_throughput_delta_pct_async_vs_sync'))} "
        f"sync_tpot={summary.get('sync_mean_tpot_ms')} "
        f"async_tpot={summary.get('async_mean_tpot_ms')} "
        f"tpot_delta={pct_text(summary.get('mean_tpot_delta_pct_async_vs_sync'))}"
    )
    print(
        "- async_device_direct: "
        f"actual={summary.get('async_actual_records')} "
        f"budget_rejects={summary.get('async_budget_reject_records')} "
        f"peak_live={summary.get('async_peak_live_bytes_observed')} "
        f"max_total={summary.get('async_max_total_bytes')}"
    )
    print(f"- async_backend_counts={summary.get('async_backend_counts')}")
    print(
        "- async_pool: "
        f"set={summary.get('async_pool_release_threshold_set')} "
        f"threshold={summary.get('async_pool_release_threshold')} "
        f"attempted={summary.get('async_pool_config_attempted')} "
        f"success={summary.get('async_pool_config_success')} "
        f"error={summary.get('async_pool_config_error')}"
    )
    print(
        "- backend_signals: "
        f"correctness={summary.get('backend_correctness_signal')} "
        f"async_effectiveness_vs_sync={summary.get('backend_async_effectiveness_signal')}"
    )
    print("- checks:")
    for item in result["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")


def main() -> int:
    args = parse_args()
    if args.pool_release_threshold != "" and as_int(args.pool_release_threshold) is None:
        raise SystemExit("--pool-release-threshold must be a non-negative integer or empty")

    run_dir: Path | None = Path(args.run_dir) if args.run_dir else None
    backend_json: Path | None = Path(args.backend_json) if args.backend_json else None
    run_log: Path | None = None

    if backend_json is None and run_dir is not None:
        backend_json = report_path_for_run_dir(run_dir, args.prompts)

    if backend_json is None:
        if args.skip_run:
            raise SystemExit("--skip-run requires --backend-json or --run-dir")
        run_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
        backend_json, run_log = run_experiment(args, run_dir)
    elif run_dir is None:
        run_dir = backend_json.parent

    result = validate_report(
        backend_json=backend_json,
        run_dir=run_dir,
        run_log=run_log,
        args=args,
    )

    output_json = (
        Path(args.output_json)
        if args.output_json
        else (run_dir or backend_json.parent) / "stage_c2_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_result(result)
    print(f"- check_json={output_json}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
