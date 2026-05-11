#!/usr/bin/env python3
"""Run or validate a Stage C1 attention-only cuda_malloc success check."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the current Stage C1 implementation is successful. "
            "By default this runs a small cuda_malloc Stage C1 A/B experiment; "
            "use --report-json or --run-dir for offline validation."
        )
    )
    parser.add_argument("--report-json", help="Existing Stage C A/B comparison JSON")
    parser.add_argument("--run-dir", help="Existing run directory containing the report")
    parser.add_argument("--prompts", type=int, default=5)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="512")
    parser.add_argument("--budget-bytes", type=int, default=1048576)
    parser.add_argument("--out-dir", help="Output directory for a new check run")
    parser.add_argument("--output-json", help="Where to write the check summary JSON")
    parser.add_argument("--max-tpot-regression-pct", type=float, default=10.0)
    parser.add_argument("--min-gap-reduction-pct", type=float, default=1.0)
    parser.add_argument("--min-unknown-reduction-pct", type=float, default=1.0)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not run a new experiment; requires --report-json or --run-dir.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def pct_text(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:+.2f}%"


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def report_path_for_run_dir(run_dir: Path, prompts: int) -> Path:
    return run_dir / f"vllm_stage_c_attention_p{prompts}_ab_comparison.json"


def default_out_dir() -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_c1_success_check_{stamp}")


def run_experiment(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "stage_c1_success_check_runner.log"
    report_json = report_path_for_run_dir(out_dir, args.prompts)

    env = os.environ.copy()
    env.update(
        {
            "PROMPTS": str(args.prompts),
            "REQUEST_RATE": str(args.request_rate),
            "OUTPUT_LEN": str(args.output_len),
            "DEVICE_DIRECT_MAX_TOTAL_BYTES": str(args.budget_bytes),
            "DEVICE_DIRECT_BACKEND": "cuda_malloc",
            "OUT_DIR": str(out_dir),
        }
    )

    cmd = ["./run_stage_c_attention_p20_ab.sh"]
    print("===========================================================")
    print(" Stage C1 Success Check: running experiment")
    print(f" Output dir: {out_dir}")
    print(f" Prompts: {args.prompts}")
    print(f" Budget bytes: {args.budget_bytes}")
    print(" Backend: cuda_malloc")
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
        raise SystemExit(f"Stage C1 experiment failed with exit code {rc}; log={run_log}")
    if not report_json.is_file():
        raise SystemExit(f"Stage C1 report was not produced: {report_json}")
    return report_json, run_log


def parse_localized_stats_line(line: str) -> dict[str, int]:
    localized_patterns = {
        "batch_faults": r"本批次总缺页实例数=([0-9]+)",
        "batch_after_dedup": r"本批次总缺页实例数=[0-9]+,去重后=([0-9]+)",
        "batch_kv_faults": r"KV类的总缺页数=([0-9]+)",
        "batch_kv_after_dedup": r"KV类的总缺页数=[0-9]+,去重后=([0-9]+)",
        "total_faults": r"\|\| 总缺页数=([0-9]+)",
        "total_after_dedup": r"\|\| 总缺页数=[0-9]+,去重后=([0-9]+)",
        "total_kv_faults": r"kv总错误数=([0-9]+)",
        "total_kv_after_dedup": r"kv总错误数=[0-9]+,去重后=([0-9]+)",
    }
    key_value_patterns = {
        "batch_faults": r"\bbatch_faults=([0-9]+)",
        "batch_after_dedup": r"\bbatch_after_dedup=([0-9]+)",
        "batch_kv_faults": r"\bbatch_kv_faults=([0-9]+)",
        "batch_kv_after_dedup": r"\bbatch_kv_after_dedup=([0-9]+)",
        "total_faults": r"\btotal_faults=([0-9]+)",
        "total_after_dedup": r"\btotal_after_dedup=([0-9]+)",
        "total_kv_faults": r"\btotal_kv_faults=([0-9]+)",
        "total_kv_after_dedup": r"\btotal_kv_after_dedup=([0-9]+)",
    }
    parsed: dict[str, int] = {}
    for key in localized_patterns:
        for pattern in (localized_patterns[key], key_value_patterns[key]):
            match = re.search(pattern, line)
            if match:
                parsed[key] = int(match.group(1))
                break
    return parsed


def check_stats_parser(run_dir: Path, prompts: int) -> dict[str, Any]:
    stats_log = run_dir / f"uvm_kv_fault_stats_gap2_stage_c_attention_p{prompts}.log"
    if not stats_log.is_file():
        return {
            "available": False,
            "passed": False,
            "detail": f"stats log not found: {stats_log}",
        }

    lines = [
        line
        for line in stats_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if "本批次总缺页实例数=" in line or "batch_faults=" in line
    ]
    if not lines:
        return {
            "available": True,
            "passed": False,
            "detail": f"no stats lines found in {stats_log}",
        }

    first = parse_localized_stats_line(lines[0])
    last = parse_localized_stats_line(lines[-1])
    required = {
        "batch_faults",
        "batch_after_dedup",
        "batch_kv_faults",
        "batch_kv_after_dedup",
        "total_faults",
        "total_after_dedup",
        "total_kv_faults",
        "total_kv_after_dedup",
    }
    missing = sorted((required - set(first)) | (required - set(last)))
    passed = not missing
    return {
        "available": True,
        "passed": passed,
        "detail": (
            f"parsed localized stats fields from {stats_log}"
            if passed
            else f"missing fields {missing} in {stats_log}"
        ),
        "first": first,
        "last": last,
    }


def validate_report(
    *,
    report_json: Path,
    run_dir: Path | None,
    run_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report = load_json(report_json)
    comparison = get_dict(report.get("comparison"))
    checks_from_report = get_dict(comparison.get("checks"))
    trace = get_dict(report.get("trace_only"))
    device = get_dict(report.get("stage_c_attention_device_direct"))
    metrics = get_dict(device.get("metrics"))
    bench = get_dict(device.get("bench_main"))

    backend_counts = get_dict(metrics.get("device_direct_backend_counts"))
    placement_counts = get_dict(metrics.get("placement_backend_counts"))
    gap_delta_pct = comparison.get("gap_fault_delta_pct")
    unknown_delta_pct = comparison.get("unknown_fault_delta_pct")
    tpot_delta_pct = comparison.get("mean_tpot_delta_pct")
    throughput_delta_pct = comparison.get("output_throughput_delta_pct")
    peak_live = metrics.get("device_direct_peak_live_bytes_observed")
    max_total = metrics.get("device_direct_max_total_bytes")

    checks: list[dict[str, Any]] = [
        check(
            "success_signal",
            comparison.get("success_signal") is True,
            f"success_signal={comparison.get('success_signal')}",
        ),
        check(
            "effectiveness_signal",
            comparison.get("effectiveness_signal") is True,
            f"effectiveness_signal={comparison.get('effectiveness_signal')}",
        ),
        check(
            "no_failed_requests",
            bench.get("failed_requests") == 0,
            f"failed_requests={bench.get('failed_requests')}",
        ),
        check(
            "cuda_malloc_backend_used",
            int(backend_counts.get("cuda_malloc") or 0) > 0,
            f"device_direct_backend_counts={backend_counts}",
        ),
        check(
            "async_backend_not_used",
            int(backend_counts.get("cuda_malloc_async") or 0) == 0,
            f"device_direct_backend_counts={backend_counts}",
        ),
        check(
            "device_direct_records_present",
            int(metrics.get("device_direct_actual_records") or 0) > 0,
            f"device_direct_actual_records={metrics.get('device_direct_actual_records')}",
        ),
        check(
            "device_direct_placement_present",
            int(placement_counts.get("device_direct") or 0) > 0,
            f"placement_backend_counts={placement_counts}",
        ),
        check(
            "gap_policy_fail_zero",
            metrics.get("gap_policy_fail") == 0,
            f"gap_policy_fail={metrics.get('gap_policy_fail')}",
        ),
        check(
            "peak_live_within_budget",
            max_total in (None, 0)
            or (peak_live is not None and int(peak_live) <= int(max_total)),
            f"peak_live={peak_live} max_total={max_total}",
        ),
        check(
            "gap_faults_reduced",
            gap_delta_pct is not None
            and float(gap_delta_pct) <= -(args.min_gap_reduction_pct / 100.0),
            f"gap_fault_delta_pct={pct_text(gap_delta_pct)}",
        ),
        check(
            "unknown_faults_reduced",
            unknown_delta_pct is not None
            and float(unknown_delta_pct) <= -(args.min_unknown_reduction_pct / 100.0),
            f"unknown_fault_delta_pct={pct_text(unknown_delta_pct)}",
        ),
        check(
            "tpot_not_too_much_worse",
            tpot_delta_pct is not None
            and float(tpot_delta_pct) <= (args.max_tpot_regression_pct / 100.0),
            f"mean_tpot_delta_pct={pct_text(tpot_delta_pct)}",
        ),
        check(
            "trace_probe_main_same_gap",
            checks_from_report.get("trace_probe_main_same_gap") is True,
            f"trace_probe_main_same_gap={checks_from_report.get('trace_probe_main_same_gap')}",
        ),
        check(
            "device_probe_main_same_gap",
            checks_from_report.get("device_probe_main_same_gap") is True,
            f"device_probe_main_same_gap={checks_from_report.get('device_probe_main_same_gap')}",
        ),
    ]

    runner_log_check: dict[str, Any] | None = None
    if run_log and run_log.is_file():
        text = run_log.read_text(encoding="utf-8", errors="replace")
        runner_log_check = {
            "passed": (
                "Failed to parse stats lines" not in text
                and "delta_faults=" in text
            ),
            "detail": f"runner_log={run_log}",
            "contains_delta_faults": "delta_faults=" in text,
            "contains_parse_failure": "Failed to parse stats lines" in text,
        }
        checks.append(
            check(
                "driver_stats_delta_printed",
                bool(runner_log_check["passed"]),
                (
                    "delta stats printed without parse failure"
                    if runner_log_check["passed"]
                    else str(runner_log_check)
                ),
            )
        )

    stats_parser_check = None
    if run_dir is not None:
        stats_parser_check = check_stats_parser(run_dir, args.prompts)
        checks.append(
            check(
                "localized_stats_fields_parseable",
                bool(stats_parser_check["passed"]),
                stats_parser_check["detail"],
            )
        )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "report_json": str(report_json),
        "run_dir": str(run_dir) if run_dir else None,
        "run_log": str(run_log) if run_log else None,
        "summary": {
            "trace_gap_faults": get_dict(trace.get("main_gap")).get("faults"),
            "device_gap_faults": get_dict(device.get("main_gap")).get("faults"),
            "gap_fault_delta_pct": gap_delta_pct,
            "unknown_fault_delta_pct": unknown_delta_pct,
            "output_throughput_delta_pct": throughput_delta_pct,
            "mean_tpot_delta_pct": tpot_delta_pct,
            "device_output_tok_s": bench.get("output_throughput_tok_s"),
            "device_mean_tpot_ms": bench.get("mean_tpot_ms"),
            "device_direct_actual_records": metrics.get("device_direct_actual_records"),
            "device_direct_budget_reject_records": metrics.get(
                "device_direct_budget_reject_records"
            ),
            "device_direct_peak_live_bytes_observed": peak_live,
            "device_direct_max_total_bytes": max_total,
            "device_direct_backend_counts": backend_counts,
            "placement_backend_counts": placement_counts,
        },
        "checks": checks,
        "runner_log_check": runner_log_check,
        "stats_parser_check": stats_parser_check,
    }


def print_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    summary = result["summary"]
    print("===========================================================")
    print(f" Stage C1 Success Check: {status}")
    print(f" Report: {result['report_json']}")
    if result.get("run_dir"):
        print(f" Run dir: {result['run_dir']}")
    print("===========================================================")
    print(
        "- gap_faults: "
        f"trace={summary.get('trace_gap_faults')} "
        f"device={summary.get('device_gap_faults')} "
        f"delta={pct_text(summary.get('gap_fault_delta_pct'))}"
    )
    print(f"- unknown_fault_delta={pct_text(summary.get('unknown_fault_delta_pct'))}")
    print(f"- output_throughput_delta={pct_text(summary.get('output_throughput_delta_pct'))}")
    print(f"- mean_tpot_delta={pct_text(summary.get('mean_tpot_delta_pct'))}")
    print(
        "- device_direct: "
        f"actual={summary.get('device_direct_actual_records')} "
        f"budget_rejects={summary.get('device_direct_budget_reject_records')} "
        f"peak_live={summary.get('device_direct_peak_live_bytes_observed')} "
        f"max_total={summary.get('device_direct_max_total_bytes')}"
    )
    print(f"- backend_counts={summary.get('device_direct_backend_counts')}")
    print("- checks:")
    for item in result["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")


def main() -> int:
    args = parse_args()

    run_dir: Path | None = Path(args.run_dir) if args.run_dir else None
    report_json: Path | None = Path(args.report_json) if args.report_json else None
    run_log: Path | None = None

    if report_json is None and run_dir is not None:
        report_json = report_path_for_run_dir(run_dir, args.prompts)

    if report_json is None:
        if args.skip_run:
            raise SystemExit("--skip-run requires --report-json or --run-dir")
        run_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
        report_json, run_log = run_experiment(args, run_dir)
    elif run_dir is None:
        run_dir = report_json.parent

    result = validate_report(
        report_json=report_json,
        run_dir=run_dir,
        run_log=run_log,
        args=args,
    )

    output_json = (
        Path(args.output_json)
        if args.output_json
        else (run_dir or report_json.parent) / "stage_c1_success_check.json"
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
