#!/usr/bin/env python3
"""Check Stage N hard partition evidence from one vLLM run root."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


POOLS = ("kv", "weights", "scratch")
POOL_IDS = {"1": "kv", "2": "weights", "3": "scratch"}


def _empty_pool_counters() -> dict[str, int]:
    return {pool: 0 for pool in POOLS}


def _int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in _read_text(path).splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _parse_key_values(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ \n]+)", line))


def _latest_driver_pool_states(
    path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    int,
    int,
    int,
    int,
    dict[str, int],
    dict[str, int],
    int,
]:
    latest: dict[str, dict[str, Any]] = {}
    records = 0
    failures = 0
    max_denies = 0
    max_would_denies = 0
    per_pool_denies = _empty_pool_counters()
    per_pool_would_denies = _empty_pool_counters()
    scratch_rejects = 0
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return (
            latest,
            records,
            failures,
            max_denies,
            max_would_denies,
            per_pool_denies,
            per_pool_would_denies,
            scratch_rejects,
        )

    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            action = record.get("action")
            if action == "residency_driver_pool_state":
                records += 1
                failures += _int(record.get("failures"))
                states = record.get("states") or {}
                for pool, state in states.items():
                    if isinstance(state, dict):
                        latest[pool] = state
            elif action == "residency_summary":
                max_denies = max(max_denies, _int(record.get("denies")))
                max_would_denies = max(max_would_denies, _int(record.get("would_denies")))
                scratch_rejects = max(
                    scratch_rejects,
                    _int(record.get("scratch_budget_reject_records")),
                )
            elif action == "residency_scratch_summary":
                scratch_rejects = max(
                    scratch_rejects,
                    _int(record.get("scratch_budget_reject_records")),
                )
            elif action in ("residency_request", "residency_deny"):
                pool = record.get("pool")
                if pool in per_pool_would_denies and record.get("would_deny"):
                    per_pool_would_denies[pool] += 1
                if pool in per_pool_denies and (
                    action == "residency_deny" or record.get("allowed") is False
                ):
                    per_pool_denies[pool] += 1
    return (
        latest,
        records,
        failures,
        max_denies,
        max_would_denies,
        per_pool_denies,
        per_pool_would_denies,
        scratch_rejects,
    )


def _scratch_trace_summary(
    path: Path,
    fallback_budget: int,
    enforce_warmup: bool,
) -> dict[str, int]:
    live: dict[str, int] = {}
    all_live: dict[str, int] = {}
    warmup_live: dict[str, int] = {}
    live_bytes = 0
    all_live_bytes = 0
    warmup_live_bytes = 0
    peak = 0
    all_peak = 0
    warmup_peak = 0
    max_alloc = 0
    all_max_alloc = 0
    warmup_max_alloc = 0
    over_records = 0
    all_over_records = 0
    warmup_over_records = 0
    over_peak = 0
    all_over_peak = 0
    warmup_over_peak = 0
    reject_records = 0
    budget = fallback_budget

    if not path.exists():
        return {
            "budget": budget,
            "enforce_warmup": 1 if enforce_warmup else 0,
            "live_peak": 0,
            "live_final": 0,
            "max_alloc": 0,
            "over_records": 0,
            "over_peak": 0,
            "reject_records": 0,
            "all_live_peak": 0,
            "all_live_final": 0,
            "all_max_alloc": 0,
            "all_over_records": 0,
            "all_over_peak": 0,
            "warmup_live_peak": 0,
            "warmup_live_final": 0,
            "warmup_max_alloc": 0,
            "warmup_over_records": 0,
            "warmup_over_peak": 0,
        }

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = _parse_key_values(line)
            if "TRACE_DRIVER_SCRATCH_TAG" in line:
                alloc_id = fields.get("alloc_id")
                size = _int(fields.get("size_bytes"))
                predicted_class = fields.get("predicted_class", "")
                is_warmup = predicted_class == "warmup_workspace"
                if alloc_id and size > 0:
                    all_live[alloc_id] = size
                    all_live_bytes += size
                    all_peak = max(all_peak, all_live_bytes)
                    all_max_alloc = max(all_max_alloc, size)
                    if budget > 0 and all_live_bytes > budget:
                        all_over_records += 1
                        all_over_peak = max(all_over_peak, all_live_bytes - budget)

                    if is_warmup:
                        warmup_live[alloc_id] = size
                        warmup_live_bytes += size
                        warmup_peak = max(warmup_peak, warmup_live_bytes)
                        warmup_max_alloc = max(warmup_max_alloc, size)
                        if budget > 0 and warmup_live_bytes > budget:
                            warmup_over_records += 1
                            warmup_over_peak = max(
                                warmup_over_peak,
                                warmup_live_bytes - budget,
                            )

                    if enforce_warmup or not is_warmup:
                        live[alloc_id] = size
                        live_bytes += size
                        peak = max(peak, live_bytes)
                        max_alloc = max(max_alloc, size)
                        if budget > 0 and live_bytes > budget:
                            over_records += 1
                            over_peak = max(over_peak, live_bytes - budget)
            elif "TRACE_DRIVER_SCRATCH_UNTAG" in line:
                alloc_id = fields.get("alloc_id")
                size = live.pop(alloc_id, 0) if alloc_id else 0
                if size:
                    live_bytes = max(live_bytes - size, 0)
                all_size = all_live.pop(alloc_id, 0) if alloc_id else 0
                if all_size:
                    all_live_bytes = max(all_live_bytes - all_size, 0)
                warmup_size = warmup_live.pop(alloc_id, 0) if alloc_id else 0
                if warmup_size:
                    warmup_live_bytes = max(warmup_live_bytes - warmup_size, 0)
            elif "TRACE_DRIVER_SCRATCH_REJECT" in line:
                reject_records += 1

    return {
        "budget": budget,
        "enforce_warmup": 1 if enforce_warmup else 0,
        "live_peak": peak,
        "live_final": live_bytes,
        "max_alloc": max_alloc,
        "over_records": over_records,
        "over_peak": over_peak,
        "reject_records": reject_records,
        "all_live_peak": all_peak,
        "all_live_final": all_live_bytes,
        "all_max_alloc": all_max_alloc,
        "all_over_records": all_over_records,
        "all_over_peak": all_over_peak,
        "warmup_live_peak": warmup_peak,
        "warmup_live_final": warmup_live_bytes,
        "warmup_max_alloc": warmup_max_alloc,
        "warmup_over_records": warmup_over_records,
        "warmup_over_peak": warmup_over_peak,
    }


def _route_summary(paths: list[Path]) -> dict[str, int]:
    out = {
        "auto_hook_records": 0,
        "auto_hook_would_reclaim_records": 0,
        "auto_hook_execute_records": 0,
        "auto_hook_success_records": 0,
        "auto_reclaimed_bytes": 0,
        "auto_attempted_regions": 0,
        "auto_success_regions": 0,
        "auto_failure_regions": 0,
        "auto_block_records": 0,
        "auto_deny_records": 0,
        "auto_deny_or_block_records": 0,
        "max_auto_over_bytes": 0,
        "max_auto_target_bytes": 0,
        "pool_reclaim_records": 0,
        "pool_reclaimed_bytes": 0,
    }
    for pool in POOLS:
        out[f"{pool}_auto_hook_records"] = 0
        out[f"{pool}_auto_hook_execute_records"] = 0
        out[f"{pool}_auto_reclaimed_bytes"] = 0
        out[f"{pool}_auto_attempted_bytes"] = 0
        out[f"{pool}_auto_success_regions"] = 0
        out[f"{pool}_auto_failure_regions"] = 0
        out[f"{pool}_auto_block_records"] = 0
        out[f"{pool}_auto_deny_records"] = 0
        out[f"{pool}_auto_deny_or_block_records"] = 0
        out[f"{pool}_auto_reclaim_to_budget_records"] = 0
        out[f"{pool}_max_auto_over_bytes"] = 0
        out[f"{pool}_max_auto_target_bytes"] = 0
    seen_lines: set[str] = set()

    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line in seen_lines:
                    continue
                seen_lines.add(line)
                fields = _parse_key_values(line)
                if "phase_n_auto_reclaim_hook" in line:
                    pool = POOL_IDS.get(fields.get("pool", ""), "")
                    out["auto_hook_records"] += 1
                    if fields.get("would_reclaim") == "1":
                        out["auto_hook_would_reclaim_records"] += 1
                    if fields.get("execute") == "1":
                        out["auto_hook_execute_records"] += 1
                    if _int(fields.get("success")) > 0:
                        out["auto_hook_success_records"] += 1
                    if fields.get("block") == "1":
                        out["auto_block_records"] += 1
                    if fields.get("deny") == "1":
                        out["auto_deny_records"] += 1
                    if (
                        fields.get("block") == "1"
                        or fields.get("deny") == "1"
                        or fields.get("deny_or_block") == "1"
                    ):
                        out["auto_deny_or_block_records"] += 1
                    out["auto_reclaimed_bytes"] += _int(fields.get("reclaimed"))
                    out["auto_attempted_regions"] += _int(fields.get("attempted"))
                    out["auto_success_regions"] += _int(fields.get("success"))
                    out["auto_failure_regions"] += _int(fields.get("failure"))
                    out["max_auto_over_bytes"] = max(out["max_auto_over_bytes"], _int(fields.get("over")))
                    out["max_auto_target_bytes"] = max(out["max_auto_target_bytes"], _int(fields.get("target")))
                    if pool in POOLS:
                        out[f"{pool}_auto_hook_records"] += 1
                        if fields.get("execute") == "1":
                            out[f"{pool}_auto_hook_execute_records"] += 1
                        out[f"{pool}_auto_reclaimed_bytes"] += _int(fields.get("reclaimed"))
                        out[f"{pool}_auto_attempted_bytes"] += _int(fields.get("attempted_bytes"))
                        out[f"{pool}_auto_success_regions"] += _int(fields.get("success"))
                        out[f"{pool}_auto_failure_regions"] += _int(fields.get("failure"))
                        if fields.get("block") == "1":
                            out[f"{pool}_auto_block_records"] += 1
                        if fields.get("deny") == "1":
                            out[f"{pool}_auto_deny_records"] += 1
                        if (
                            fields.get("block") == "1"
                            or fields.get("deny") == "1"
                            or fields.get("deny_or_block") == "1"
                        ):
                            out[f"{pool}_auto_deny_or_block_records"] += 1
                        if fields.get("reclaim_to_budget") == "1":
                            out[f"{pool}_auto_reclaim_to_budget_records"] += 1
                        out[f"{pool}_max_auto_over_bytes"] = max(
                            out[f"{pool}_max_auto_over_bytes"],
                            _int(fields.get("over")),
                        )
                        out[f"{pool}_max_auto_target_bytes"] = max(
                            out[f"{pool}_max_auto_target_bytes"],
                            _int(fields.get("target")),
                        )
                elif "phase_n_reclaim_pool" in line:
                    out["pool_reclaim_records"] += 1
                    out["pool_reclaimed_bytes"] += _int(fields.get("reclaimed"))
    return out


def _pool_line(prefix: str, pool: str, state: dict[str, Any], tolerance: int) -> tuple[list[str], bool, int]:
    budget = _int(state.get("budget_bytes"))
    resident = _int(state.get("resident_gpu_bytes"))
    explicit_over = _int(state.get("over_budget_bytes"))
    computed_over = max(resident - budget, 0) if budget > 0 else 0
    over = max(explicit_over, computed_over)
    under = budget > 0 and resident <= budget + tolerance and over <= tolerance
    lines = [
        f"{prefix}_{pool}_budget_bytes={budget}",
        f"{prefix}_{pool}_resident_gpu_bytes={resident}",
        f"{prefix}_{pool}_over_budget_bytes={over}",
        f"{prefix}_{pool}_candidate_bytes={_int(state.get('victim_candidate_bytes'))}",
        f"{prefix}_{pool}_candidate_regions={_int(state.get('victim_candidate_regions'))}",
        f"{prefix}_{pool}_selected_victim_bytes={_int(state.get('selected_victim_bytes'))}",
        f"{prefix}_{pool}_victim_count={_int(state.get('victim_count'))}",
        f"{prefix}_{pool}_under_budget={1 if under else 0}",
    ]
    return lines, under, over


def check_run(run_root: Path, tolerance: int) -> tuple[dict[str, int | str], list[str]]:
    env = _parse_env(run_root / "env_stage_n_tag_only.txt")
    residency_path = run_root / "vllm_uvm_pool_residency_stage_n.jsonl"
    allocator_path = run_root / "vllm_uvm_allocator_trace.log"
    (
        states,
        state_records,
        state_failures,
        denies,
        would_denies,
        per_pool_denies,
        per_pool_would_denies,
        scratch_rejects,
    ) = _latest_driver_pool_states(residency_path)
    scratch_budget = _int(
        (states.get("scratch") or {}).get("budget_bytes"),
        _int(env.get("SCRATCH_TEMP_BUDGET_BYTES")),
    )
    scratch = _scratch_trace_summary(
        allocator_path,
        scratch_budget,
        env.get("VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP") == "1",
    )
    scratch_rejects = max(scratch_rejects, scratch["reject_records"])
    route = _route_summary([
        run_root / "phase_f_route_log_snapshot.log",
        run_root / "kernel_since_start.log",
    ])

    values: dict[str, int | str] = {
        "hard_partition_tolerance_bytes": tolerance,
        "hard_partition_driver_pool_state_records": state_records,
        "hard_partition_driver_pool_state_failures": state_failures,
        "hard_partition_denies": denies,
        "hard_partition_would_denies": would_denies,
        "hard_partition_scratch_budget_reject_records": scratch_rejects,
        "hard_partition_auto_hook_records": route["auto_hook_records"],
        "hard_partition_auto_hook_would_reclaim_records": route["auto_hook_would_reclaim_records"],
        "hard_partition_auto_hook_execute_records": route["auto_hook_execute_records"],
        "hard_partition_auto_hook_success_records": route["auto_hook_success_records"],
        "hard_partition_auto_reclaimed_bytes": route["auto_reclaimed_bytes"],
        "hard_partition_auto_attempted_regions": route["auto_attempted_regions"],
        "hard_partition_auto_success_regions": route["auto_success_regions"],
        "hard_partition_auto_failure_regions": route["auto_failure_regions"],
        "hard_partition_auto_block_records": route["auto_block_records"],
        "hard_partition_auto_deny_records": route["auto_deny_records"],
        "hard_partition_auto_deny_or_block_records": route["auto_deny_or_block_records"],
        "hard_partition_max_auto_over_bytes": route["max_auto_over_bytes"],
        "hard_partition_max_auto_target_bytes": route["max_auto_target_bytes"],
        "hard_partition_pool_reclaim_records": route["pool_reclaim_records"],
        "hard_partition_pool_reclaimed_bytes": route["pool_reclaimed_bytes"],
        "hard_partition_scratch_budget_bytes": scratch["budget"],
        "hard_partition_scratch_enforce_warmup": scratch["enforce_warmup"],
        "hard_partition_scratch_live_peak_bytes": scratch["live_peak"],
        "hard_partition_scratch_live_final_bytes": scratch["live_final"],
        "hard_partition_scratch_max_alloc_bytes": scratch["max_alloc"],
        "hard_partition_scratch_over_budget_records": scratch["over_records"],
        "hard_partition_scratch_over_budget_peak_bytes": scratch["over_peak"],
        "hard_partition_scratch_driver_reject_records": scratch["reject_records"],
        "hard_partition_scratch_all_live_peak_bytes": scratch["all_live_peak"],
        "hard_partition_scratch_all_over_budget_records": scratch["all_over_records"],
        "hard_partition_scratch_all_over_budget_peak_bytes": scratch["all_over_peak"],
        "hard_partition_scratch_warmup_live_peak_bytes": scratch["warmup_live_peak"],
        "hard_partition_scratch_warmup_over_budget_records": scratch[
            "warmup_over_records"
        ],
        "hard_partition_scratch_warmup_over_budget_peak_bytes": scratch[
            "warmup_over_peak"
        ],
    }

    failures: list[str] = []
    all_driver_under = True
    max_driver_over = 0
    for pool in POOLS:
        state = states.get(pool)
        if not state:
            all_driver_under = False
            failures.append(f"{pool}:missing_driver_pool_state")
            for key in (
                "budget_bytes",
                "resident_gpu_bytes",
                "over_budget_bytes",
                "candidate_bytes",
                "candidate_regions",
                "selected_victim_bytes",
                "victim_count",
                "under_budget",
            ):
                values[f"hard_partition_pool_{pool}_{key}"] = 0
            continue
        _, under, over = _pool_line("hard_partition_pool", pool, state, tolerance)
        budget = _int(state.get("budget_bytes"))
        resident = _int(state.get("resident_gpu_bytes"))
        values[f"hard_partition_pool_{pool}_budget_bytes"] = budget
        values[f"hard_partition_pool_{pool}_resident_gpu_bytes"] = resident
        values[f"hard_partition_pool_{pool}_over_budget_bytes"] = over
        values[f"hard_partition_pool_{pool}_candidate_bytes"] = _int(state.get("victim_candidate_bytes"))
        values[f"hard_partition_pool_{pool}_candidate_regions"] = _int(state.get("victim_candidate_regions"))
        values[f"hard_partition_pool_{pool}_selected_victim_bytes"] = _int(state.get("selected_victim_bytes"))
        values[f"hard_partition_pool_{pool}_victim_count"] = _int(state.get("victim_count"))
        values[f"hard_partition_pool_{pool}_under_budget"] = 1 if under else 0
        values[f"hard_partition_pool_{pool}_deny_or_block_records"] = per_pool_denies[pool]
        values[f"hard_partition_pool_{pool}_would_deny_records"] = per_pool_would_denies[pool]
        values[f"hard_partition_pool_{pool}_auto_hook_records"] = route[f"{pool}_auto_hook_records"]
        values[f"hard_partition_pool_{pool}_auto_hook_execute_records"] = route[
            f"{pool}_auto_hook_execute_records"
        ]
        values[f"hard_partition_pool_{pool}_auto_reclaimed_bytes"] = route[
            f"{pool}_auto_reclaimed_bytes"
        ]
        values[f"hard_partition_pool_{pool}_auto_attempted_bytes"] = route[
            f"{pool}_auto_attempted_bytes"
        ]
        values[f"hard_partition_pool_{pool}_auto_success_regions"] = route[
            f"{pool}_auto_success_regions"
        ]
        values[f"hard_partition_pool_{pool}_auto_failure_regions"] = route[
            f"{pool}_auto_failure_regions"
        ]
        values[f"hard_partition_pool_{pool}_auto_block_records"] = route[
            f"{pool}_auto_block_records"
        ]
        values[f"hard_partition_pool_{pool}_auto_deny_records"] = route[
            f"{pool}_auto_deny_records"
        ]
        values[f"hard_partition_pool_{pool}_auto_deny_or_block_records"] = route[
            f"{pool}_auto_deny_or_block_records"
        ]
        values[f"hard_partition_pool_{pool}_auto_reclaim_to_budget_records"] = route[
            f"{pool}_auto_reclaim_to_budget_records"
        ]
        values[f"hard_partition_pool_{pool}_max_auto_over_bytes"] = route[
            f"{pool}_max_auto_over_bytes"
        ]
        values[f"hard_partition_pool_{pool}_max_auto_target_bytes"] = route[
            f"{pool}_max_auto_target_bytes"
        ]
        max_driver_over = max(max_driver_over, over)
        if not under:
            all_driver_under = False
            pool_block_records = (
                per_pool_denies[pool]
                + route[f"{pool}_auto_deny_or_block_records"]
            )
            if pool == "scratch":
                pool_block_records += scratch_rejects
            pool_reclaim_to_budget_records = route[
                f"{pool}_auto_reclaim_to_budget_records"
            ]
            if pool_block_records <= 0:
                failures.append(
                    f"{pool}:resident_gpu_bytes:{resident}:budget_bytes:{budget}:"
                    f"over_budget_bytes:{over}:auto_reclaimed_bytes:"
                    f"{route[f'{pool}_auto_reclaimed_bytes']}:"
                    f"reclaim_to_budget_records:{pool_reclaim_to_budget_records}:"
                    f"deny_or_block_records:0"
                )

    scratch_under = (
        scratch["budget"] > 0
        and scratch["live_peak"] <= scratch["budget"] + tolerance
        and scratch["over_records"] == 0
    )
    if not scratch_under:
        if (
            scratch_rejects <= 0
            and per_pool_denies["scratch"] <= 0
            and route["scratch_auto_deny_or_block_records"] <= 0
        ):
            failures.append(
                "scratch:live_peak_bytes:"
                f"{scratch['live_peak']}:budget_bytes:{scratch['budget']}:"
                f"over_budget_records:{scratch['over_records']}:"
                f"reject_records:{scratch_rejects}"
            )

    driver_pools_blocked = all(
        values.get(f"hard_partition_pool_{pool}_under_budget") == 1
        or per_pool_denies[pool] > 0
        or route[f"{pool}_auto_deny_or_block_records"] > 0
        or (pool == "scratch" and scratch_rejects > 0)
        for pool in POOLS
    )
    scratch_blocked = (
        scratch_under
        or scratch_rejects > 0
        or per_pool_denies["scratch"] > 0
        or route["scratch_auto_deny_or_block_records"] > 0
    )
    sufficient_block = driver_pools_blocked and scratch_blocked and (
        any(per_pool_denies.values())
        or scratch_rejects > 0
        or route["auto_deny_or_block_records"] > 0
    )
    sufficient_reclaim = all_driver_under and scratch_under
    hard_pass = len(failures) == 0 and (sufficient_reclaim or sufficient_block)

    values["hard_partition_all_driver_pools_under_budget"] = 1 if all_driver_under else 0
    values["hard_partition_all_transient_pools_under_budget"] = 1 if scratch_under else 0
    values["hard_partition_max_driver_over_budget_bytes"] = max_driver_over
    values["hard_partition_sufficient_reclaim_to_budget"] = 1 if sufficient_reclaim else 0
    values["hard_partition_sufficient_deny_or_block"] = 1 if sufficient_block else 0
    values["hard_partition_pass"] = 1 if hard_pass else 0
    values["hard_partition_failure_reasons"] = "none" if hard_pass else ";".join(failures)
    return values, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--tolerance-bytes", type=int, default=0)
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()

    values, _ = check_run(args.run_root, max(args.tolerance_bytes, 0))
    print("Hard Partition Proof Summary")
    print("----------------------------")
    for key in sorted(values):
        print(f"{key}={values[key]}")
    if args.fail_on_violation and values.get("hard_partition_pass") != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
