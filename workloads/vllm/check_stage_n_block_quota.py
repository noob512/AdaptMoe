#!/usr/bin/env python3
"""Check Stage N P8 block-quota evidence from synthetic or vLLM run logs."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


POOL_NAMES = {
    "0": "unknown",
    "1": "kv",
    "2": "weights",
    "3": "scratch",
    "4": "activation",
    "kv": "kv",
    "weight": "weights",
    "weights": "weights",
    "scratch": "scratch",
}

LOG_NAMES = {
    "phase_f_route_log_snapshot.log",
    "kernel_since_start.log",
    "run.log",
    "synthetic.log",
    "summary.txt",
    "stage_n_p8d_block_quota_summary.txt",
    "stage_n_p8e_allocator_kv_tag_summary.txt",
    "stage_n_acceptance_summary.txt",
}

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")


def _int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except ValueError:
        return default


def _pool_name(value: str | None) -> str:
    if value is None:
        return "unknown"
    return POOL_NAMES.get(value.strip().lower(), value.strip().lower())


def _line_pool(fields: dict[str, str], line: str) -> str:
    pool = _pool_name(fields.get("pool"))
    if pool == "unknown" and "TRACE_DRIVER_KV_" in line:
        return "kv"
    return pool


def _fields(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def _candidate_logs(run_root: Path | None, extra_logs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        paths.append(path)

    if run_root:
        for name in LOG_NAMES:
            add(run_root / name)
        for path in run_root.rglob("*"):
            if path.name in LOG_NAMES or path.suffix in (".log", ".txt", ".jsonl", ".out", ".err"):
                add(path)
    for path in extra_logs:
        add(path)
    return paths


def parse_logs(paths: list[Path]) -> dict[str, object]:
    totals = defaultdict(int)
    pools: dict[str, defaultdict[str, int]] = {
        "kv": defaultdict(int),
        "weights": defaultdict(int),
        "scratch": defaultdict(int),
        "unknown": defaultdict(int),
    }
    max_by_pool: dict[str, dict[str, int]] = {
        pool: defaultdict(int) for pool in pools
    }
    failure_lines: list[str] = []
    first_kv_tag: tuple[str, int, str] | None = None
    first_kv_materialize: tuple[str, int, str] | None = None

    for path in paths:
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            line_no = 0
            for raw in handle:
                line_no += 1
                line = raw.rstrip("\n")
                fields = _fields(line)
                pool = _line_pool(fields, line)
                if pool not in pools:
                    pools[pool] = defaultdict(int)
                    max_by_pool[pool] = defaultdict(int)

                if "phase_n_tagged_range_register" in line or "phase_f_register_region" in line:
                    totals["tagged_range_register_records"] += 1
                    pools[pool]["tagged_range_register_records"] += 1
                    if pool == "kv" and first_kv_tag is None:
                        first_kv_tag = (path.name, line_no, line[:220])

                if "TRACE_DRIVER_KV_TAG" in line:
                    totals["allocator_kv_driver_tag_records"] += 1
                    pools["kv"]["allocator_kv_driver_tag_records"] += 1
                    if _int(fields.get("registered")) or _int(fields.get("status")) > 0:
                        totals["allocator_kv_driver_tag_success_records"] += 1
                        pools["kv"]["allocator_kv_driver_tag_success_records"] += 1

                if "TRACE_DRIVER_KV_UNTAG" in line:
                    totals["allocator_kv_driver_untag_records"] += 1
                    pools["kv"]["allocator_kv_driver_untag_records"] += 1

                if "skip=no_tagged_region" in line:
                    totals["no_tagged_region_records"] += 1
                    pools[pool]["no_tagged_region_records"] += 1

                if "phase_n_block_materialize" in line:
                    totals["block_materialize_records"] += 1
                    pools[pool]["block_materialize_records"] += 1
                    if pool == "kv" and first_kv_materialize is None:
                        first_kv_materialize = (path.name, line_no, line[:220])
                    if _int(fields.get("mixed_pool")):
                        totals["mixed_pool_records"] += 1
                        pools[pool]["mixed_pool_records"] += 1

                if "phase_n_native_block_resident" in line:
                    totals["native_block_resident_records"] += 1
                    pools[pool]["native_block_resident_records"] += 1
                    totals["block_summary_records"] += 1
                    pools[pool]["block_summary_records"] += 1
                    if _int(fields.get("materialized")):
                        totals["native_block_materialize_records"] += 1
                        pools[pool]["native_block_materialize_records"] += 1
                        totals["block_materialize_records"] += 1
                        pools[pool]["block_materialize_records"] += 1
                        if pool == "kv" and first_kv_materialize is None:
                            first_kv_materialize = (path.name, line_no, line[:220])
                    for key in ("budget_blocks", "resident_blocks", "materialized_blocks", "over_blocks"):
                        max_by_pool[pool][key] = max(max_by_pool[pool][key], _int(fields.get(key)))

                if "phase_n_native_range_tag_attach" in line:
                    totals["native_range_tag_attach_records"] += 1
                    pools[pool]["native_range_tag_attach_records"] += 1

                if "phase_n_block_summary" in line:
                    totals["block_summary_records"] += 1
                    pools[pool]["block_summary_records"] += 1
                    for key in ("budget_blocks", "resident_blocks", "materialized_blocks", "mixed_blocks", "over_blocks"):
                        max_by_pool[pool][key] = max(max_by_pool[pool][key], _int(fields.get(key)))

                if "phase_n_block_over_budget" in line:
                    totals["block_over_budget_records"] += 1
                    pools[pool]["block_over_budget_records"] += 1
                    for key in ("budget_blocks", "resident_blocks", "over_blocks"):
                        max_by_pool[pool][key] = max(max_by_pool[pool][key], _int(fields.get(key)))

                if "phase_n_block_auto_hook" in line:
                    totals["block_auto_hook_records"] += 1
                    pools[pool]["block_auto_hook_records"] += 1
                    if _int(fields.get("would_reclaim")):
                        totals["block_auto_hook_would_reclaim_records"] += 1
                        pools[pool]["block_auto_hook_would_reclaim_records"] += 1
                    if _int(fields.get("execute")):
                        totals["block_auto_hook_execute_records"] += 1
                        pools[pool]["block_auto_hook_execute_records"] += 1
                    if _int(fields.get("success")) > 0:
                        totals["block_auto_hook_success_records"] += 1
                        pools[pool]["block_auto_hook_success_records"] += 1
                    if _int(fields.get("reclaimed")) > 0:
                        totals["block_auto_hook_reclaimed_bytes"] += _int(fields.get("reclaimed"))
                        pools[pool]["block_auto_hook_reclaimed_bytes"] += _int(fields.get("reclaimed"))
                    if _int(fields.get("reclaim_to_budget")):
                        totals["block_auto_hook_reclaim_to_budget_records"] += 1
                        pools[pool]["block_auto_hook_reclaim_to_budget_records"] += 1
                    for key in ("budget_blocks", "resident_blocks", "resident_before_blocks", "resident_after_blocks", "over_blocks"):
                        max_by_pool[pool][key] = max(max_by_pool[pool][key], _int(fields.get(key)))

                if "phase_n_block_reclaim pool=" in line:
                    totals["block_reclaim_records"] += 1
                    pools[pool]["block_reclaim_records"] += 1
                    if _int(fields.get("success")):
                        totals["block_reclaim_success_records"] += 1
                        pools[pool]["block_reclaim_success_records"] += 1
                    else:
                        totals["block_reclaim_failure_records"] += 1
                        pools[pool]["block_reclaim_failure_records"] += 1

                if "phase_n_native_block_reclaim pool=" in line:
                    totals["native_block_reclaim_records"] += 1
                    pools[pool]["native_block_reclaim_records"] += 1
                    totals["block_reclaim_records"] += 1
                    pools[pool]["block_reclaim_records"] += 1
                    if _int(fields.get("success")):
                        totals["block_reclaim_success_records"] += 1
                        pools[pool]["block_reclaim_success_records"] += 1
                    else:
                        totals["block_reclaim_failure_records"] += 1
                        pools[pool]["block_reclaim_failure_records"] += 1

                if "phase_n_block_reclaim_skip" in line:
                    totals["block_reclaim_skip_records"] += 1
                    pools[pool]["block_reclaim_skip_records"] += 1
                    if "reason=wrong_pool" in line:
                        totals["cross_pool_errors"] += 1

                if "phase_n_block_mixed_pool" in line:
                    totals["mixed_pool_records"] += 1
                    pools[pool]["mixed_pool_records"] += 1

                if "phase_n_get_pool_state block_quota=1" in line:
                    totals["get_pool_state_block_quota_records"] += 1
                    pools[pool]["get_pool_state_block_quota_records"] += 1
                    for key in ("budget_blocks", "resident_blocks", "materialized_blocks", "mixed_blocks", "over_blocks", "max_over_blocks"):
                        max_by_pool[pool][key] = max(max_by_pool[pool][key], _int(fields.get(key)))

                if "phase_n_reclaim_pool block_quota=1" in line:
                    totals["reclaim_pool_block_quota_records"] += 1
                    pools[pool]["reclaim_pool_block_quota_records"] += 1
                    for key in ("budget_blocks", "resident_blocks", "over_blocks"):
                        max_by_pool[pool][key] = max(max_by_pool[pool][key], _int(fields.get(key)))

                if "phase_n_reclaim_pool native_block=1" in line:
                    totals["reclaim_pool_block_quota_records"] += 1
                    totals["reclaim_pool_native_block_records"] += 1
                    pools[pool]["reclaim_pool_block_quota_records"] += 1
                    pools[pool]["reclaim_pool_native_block_records"] += 1
                    for key in ("budget_blocks", "resident_blocks", "over_blocks"):
                        max_by_pool[pool][key] = max(max_by_pool[pool][key], _int(fields.get(key)))

                failed_match = re.search(r"Failed requests:\s*([0-9]+)", line)
                if (
                    any(token in line for token in ("Traceback", "RuntimeError:", "CUDA error", "out of memory"))
                    or (failed_match and _int(failed_match.group(1)) > 0)
                ):
                    failure_lines.append(f"{path.name}:{line[:220]}")

    kv_tag_before_materialize = 0
    if first_kv_tag and first_kv_materialize:
        kv_tag_before_materialize = int(
            first_kv_tag[0] == first_kv_materialize[0]
            and first_kv_tag[1] <= first_kv_materialize[1]
        )
        totals["kv_tag_before_materialize_records"] = kv_tag_before_materialize

    return {
        "totals": totals,
        "pools": pools,
        "max_by_pool": max_by_pool,
        "failure_lines": failure_lines,
        "first_kv_tag": first_kv_tag,
        "first_kv_materialize": first_kv_materialize,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", nargs="?", type=Path)
    parser.add_argument("--log", action="append", default=[], type=Path)
    parser.add_argument("--require-pool", action="append", default=[], choices=["kv", "weights", "scratch"])
    parser.add_argument("--require-materialize", action="store_true")
    parser.add_argument("--require-over-budget", action="store_true")
    parser.add_argument("--require-auto-hook", action="store_true")
    parser.add_argument("--require-reclaim", action="store_true")
    parser.add_argument("--require-block-fastpath", action="store_true")
    parser.add_argument("--require-allocator-kv-tag", action="store_true")
    parser.add_argument("--require-kv-tag-before-materialize", action="store_true")
    parser.add_argument("--require-kv-resident-blocks", action="store_true")
    parser.add_argument("--allow-mixed-pool", action="store_true")
    parser.add_argument("--allowed-post-fault-slack-blocks", type=int)
    parser.add_argument("--check-failures", action="store_true")
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()

    paths = _candidate_logs(args.run_root, args.log)
    parsed = parse_logs(paths)
    totals: defaultdict[str, int] = parsed["totals"]  # type: ignore[assignment]
    pools: dict[str, defaultdict[str, int]] = parsed["pools"]  # type: ignore[assignment]
    max_by_pool: dict[str, dict[str, int]] = parsed["max_by_pool"]  # type: ignore[assignment]
    failure_lines: list[str] = parsed["failure_lines"]  # type: ignore[assignment]
    first_kv_tag: tuple[str, int, str] | None = parsed["first_kv_tag"]  # type: ignore[assignment]
    first_kv_materialize: tuple[str, int, str] | None = parsed["first_kv_materialize"]  # type: ignore[assignment]

    violations: list[str] = []
    if not paths:
        violations.append("no_log_files_found")
    if args.require_materialize and totals["block_materialize_records"] == 0:
        violations.append("missing_phase_n_block_materialize")
    if args.require_over_budget and totals["block_over_budget_records"] == 0:
        violations.append("missing_phase_n_block_over_budget")
    if args.require_auto_hook and totals["block_auto_hook_records"] == 0:
        violations.append("missing_phase_n_block_auto_hook")
    if args.require_reclaim and totals["block_reclaim_success_records"] == 0:
        violations.append("missing_successful_phase_n_block_reclaim")
    if args.require_block_fastpath and totals["get_pool_state_block_quota_records"] == 0:
        violations.append("missing_get_pool_state_block_quota_fastpath")
    if args.require_allocator_kv_tag and totals["allocator_kv_driver_tag_success_records"] == 0:
        violations.append("missing_successful_allocator_kv_driver_tag")
    if args.require_kv_tag_before_materialize and totals["kv_tag_before_materialize_records"] == 0:
        violations.append("missing_kv_tag_before_materialize")
    if args.require_kv_resident_blocks and max_by_pool.get("kv", {}).get("resident_blocks", 0) == 0:
        violations.append("missing_kv_resident_blocks")
    if not args.allow_mixed_pool and totals["mixed_pool_records"] > 0:
        violations.append("unexpected_mixed_pool_records")
    if totals["cross_pool_errors"] > 0:
        violations.append("cross_pool_reclaim_errors")
    if args.allowed_post_fault_slack_blocks is not None:
        max_over = max((values.get("over_blocks", 0) for values in max_by_pool.values()), default=0)
        if max_over > args.allowed_post_fault_slack_blocks:
            violations.append(
                f"post_fault_slack_exceeded:max_over_blocks={max_over}:allowed={args.allowed_post_fault_slack_blocks}"
            )
    for pool in args.require_pool:
        if pools.get(pool, {}).get("block_materialize_records", 0) == 0:
            violations.append(f"missing_materialize_pool_{pool}")
    if args.check_failures and failure_lines:
        violations.append("application_or_cuda_failure_lines")

    print("stage_n_block_quota_check=1")
    print(f"log_files={len(paths)}")
    for key in sorted(totals):
        print(f"{key}={totals[key]}")
    for pool in ("kv", "weights", "scratch", "unknown"):
        counters = pools.get(pool, {})
        max_values = max_by_pool.get(pool, {})
        print(
            f"pool_{pool}="
            f"materialize:{counters.get('block_materialize_records', 0)},"
            f"native_materialize:{counters.get('native_block_materialize_records', 0)},"
            f"summary:{counters.get('block_summary_records', 0)},"
            f"native_resident:{counters.get('native_block_resident_records', 0)},"
            f"over_budget:{counters.get('block_over_budget_records', 0)},"
            f"auto_hook:{counters.get('block_auto_hook_records', 0)},"
            f"auto_execute:{counters.get('block_auto_hook_execute_records', 0)},"
            f"auto_success:{counters.get('block_auto_hook_success_records', 0)},"
            f"reclaim:{counters.get('block_reclaim_records', 0)},"
            f"native_reclaim:{counters.get('native_block_reclaim_records', 0)},"
            f"reclaim_success:{counters.get('block_reclaim_success_records', 0)},"
            f"get_state_fastpath:{counters.get('get_pool_state_block_quota_records', 0)},"
            f"native_range_tag:{counters.get('native_range_tag_attach_records', 0)},"
            f"allocator_kv_tag:{counters.get('allocator_kv_driver_tag_records', 0)},"
            f"allocator_kv_tag_success:{counters.get('allocator_kv_driver_tag_success_records', 0)},"
            f"no_tagged_region:{counters.get('no_tagged_region_records', 0)},"
            f"max_budget_blocks:{max_values.get('budget_blocks', 0)},"
            f"max_resident_blocks:{max_values.get('resident_blocks', 0)},"
            f"max_over_blocks:{max_values.get('over_blocks', 0)},"
            f"max_mixed_blocks:{max_values.get('mixed_blocks', 0)}"
        )
    print(f"first_kv_tag={first_kv_tag[0]}:{first_kv_tag[1]}" if first_kv_tag else "first_kv_tag=none")
    print(
        f"first_kv_materialize={first_kv_materialize[0]}:{first_kv_materialize[1]}"
        if first_kv_materialize
        else "first_kv_materialize=none"
    )
    if first_kv_tag:
        print(f"first_kv_tag_sample={first_kv_tag[2]}")
    if first_kv_materialize:
        print(f"first_kv_materialize_sample={first_kv_materialize[2]}")
    if failure_lines:
        print("failure_line_samples=" + " | ".join(failure_lines[:5]))
    print("block_quota_pass=" + ("1" if not violations else "0"))
    print("block_quota_violations=" + (",".join(violations) if violations else "none"))

    if violations and args.fail_on_violation:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
