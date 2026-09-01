#!/usr/bin/env python3
"""Verify one no-HTTP E2-C16 direct-batch mechanism case."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FATAL = re.compile(
    r"CUDA error|illegal memory access|\bXid\b|PTE.*fault|PDE.*fault|"
    r"segmentation fault|out of memory|Traceback \(most recent call last\)",
    re.IGNORECASE,
)
KT_PREFILL = re.compile(
    r"Prefill batch.*#new-seq:\s*(\d+),\s*#new-token:\s*(\d+).*"
    r"#running-req:\s*(\d+)",
)


def peak_hbm(path: Path) -> int | None:
    peak: int | None = None
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            try:
                value = int(float(row[4].strip()))
            except ValueError:
                continue
            peak = value if peak is None else max(peak, value)
    return peak


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    result_path = args.case_root / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    run_log = (args.case_root / "run.log").read_text(
        encoding="utf-8", errors="replace"
    ) if (args.case_root / "run.log").is_file() else ""
    kernel_log = (args.case_root / "kernel.since.log").read_text(
        encoding="utf-8", errors="replace"
    ) if (args.case_root / "kernel.since.log").is_file() else ""
    measurement = result.get("measurement") or {}
    config = result.get("configuration") or {}
    warmups = result.get("warmups") or []
    returncode_path = args.case_root / "returncode.txt"
    returncode = int(returncode_path.read_text().strip()) \
        if returncode_path.is_file() else None

    kt_proof: dict[str, Any] | None = None
    if args.arm == "ktransformers":
        marker = "DIRECT_C16_MEASUREMENT_DISPATCH"
        post_marker = run_log.split(marker, 1)[1] if marker in run_log else ""
        matches = list(KT_PREFILL.finditer(post_marker))
        if matches:
            first = matches[0]
            kt_proof = {
                "prefill_log_entries": len(matches),
                "first_new_sequences": int(first.group(1)),
                "first_new_tokens": int(first.group(2)),
                "first_running_requests": int(first.group(3)),
                "total_new_sequences": sum(int(match.group(1)) for match in matches),
                "total_new_tokens": sum(int(match.group(2)) for match in matches),
            }
        scheduler_admission_ok = bool(
            kt_proof
            and kt_proof["total_new_sequences"] == 16
            and kt_proof["total_new_tokens"] == 16 * int(config.get("prompt_tokens", 0))
        )
        single_admission_ok = (
            measurement.get("api_batch_size") == 16
            and measurement.get("all_requests_in_one_generate_call") is True
            and measurement.get("single_admission_batch") is True
        )
        pure_phase_ok = config.get("prefill_decode_policy") == (
            "prefill-priority; mixed chunk disabled"
        )
    elif args.arm == "moe_infinity":
        scheduler_admission_ok = (
            measurement.get("prefill_sequences_before_first_decode") == 16
        )
        single_admission_ok = (
            measurement.get("admitted_request_count") == 16
            and measurement.get("all_requests_added_before_first_step") is True
            and measurement.get("single_admission_batch") is True
        )
        pure_phase_ok = (
            measurement.get("pure_phase_no_mixed_scheduler_output") is True
        )
    else:
        scheduler_admission_ok = (
            "UVM phase-batched cohort started: phase=prefill requests=16" in run_log
            and "UVM phase-batched cohort cutover: phase=decode requests=16" in run_log
        )
        single_admission_ok = (
            measurement.get("admitted_request_count") == 16
            and measurement.get("all_requests_added_before_first_step") is True
            and measurement.get("single_admission_batch") is True
        )
        pure_phase_ok = scheduler_admission_ok

    expected_output_tokens = int(config.get("output_tokens", 0))
    full_generation_ok = expected_output_tokens > 0 and all(
        request.get("success") is True
        and request.get("completion_tokens") == expected_output_tokens
        for request in measurement.get("requests", [])
    ) and len(measurement.get("requests", [])) == 16

    fatal_matches = [
        line[-2000:]
        for line in (run_log + "\n" + kernel_log).splitlines()
        if FATAL.search(line)
    ]
    checks = {
        "runner_returncode_zero": returncode == 0,
        "result_status_ok": result.get("status") == "ok",
        "three_warmups_succeeded": len(warmups) == 3 and all(
            warmup.get("successful_requests") == 1 for warmup in warmups
        ),
        "measurement_16_of_16_succeeded": (
            measurement.get("request_count") == 16
            and measurement.get("successful_requests") == 16
        ),
        "no_http_server": config.get("http_server_started") is False,
        "no_client": config.get("client_used") is False,
        "one_c16_measurement_admission": single_admission_ok,
        "scheduler_prefilled_all_16_before_decode": scheduler_admission_ok,
        "no_mixed_prefill_decode_step": pure_phase_ok,
        "full_configured_generation_completed": full_generation_ok,
        "no_fatal_log_pattern": not fatal_matches,
    }
    verification = {
        "schema": "eurosys_e2_c16_direct_batch_case_verification_v2",
        "arm": args.arm,
        "workload": config.get("workload"),
        "status": "ok" if all(checks.values()) else "error",
        "checks": checks,
        "ktransformers_scheduler_log_proof": kt_proof,
        "fatal_log_matches": fatal_matches[:20],
        "peak_hbm_mib": peak_hbm(args.case_root / "nvml.csv"),
    }
    write_json(args.case_root / "verification.json", verification)
    return 0 if verification["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
