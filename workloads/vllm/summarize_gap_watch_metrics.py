#!/usr/bin/env python3
"""Summarize gap-watch policy effectiveness from allocator trace logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TRACE_POLICY_RE = re.compile(
    r"TRACE_POLICY alloc_id=(?P<alloc_id>\d+) ptr=(?P<ptr>0x[0-9a-fA-F]+) "
    r"end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_bucket=(?P<size_bucket>[^ ]+) device=(?P<device>-?\d+) "
    r"phase=(?P<phase>\S+) predicted_class=(?P<predicted_class>[^ ]+) "
    r"action=(?P<action>[^ ]+) policy_source=(?P<policy_source>[^ ]+) "
    r"gap_watch_class_match=(?P<gap_watch_class_match>\d+) "
    r"gap_overlap_bytes=(?P<gap_overlap_bytes>\d+) "
    r"action_success=(?P<action_success>\d+) action_error=(?P<action_error>[^ ]+)"
)
TRACE_GAP_ALLOC_RE = re.compile(
    r"TRACE_GAP_WATCH_ALLOC alloc_id=(?P<alloc_id>\d+) watch_name=(?P<watch_name>[^ ]+) "
    r"ptr=(?P<ptr>0x[0-9a-fA-F]+) end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_bucket=(?P<size_bucket>[^ ]+) device=(?P<device>-?\d+) phase=(?P<phase>\S+) "
    r"predicted_class=(?P<predicted_class>[^ ]+) action=(?P<action>[^ ]+) "
    r"policy_source=(?P<policy_source>[^ ]+) gap_watch_target_class=(?P<target_class>[^ ]+) "
    r"gap_watch_class_match=(?P<class_match>\d+) .* overlap_bytes=(?P<overlap_bytes>\d+)"
)
TRACE_GAP_FREE_RE = re.compile(
    r"TRACE_GAP_WATCH_FREE free_id=(?P<free_id>\d+) watch_name=(?P<watch_name>[^ ]+) "
    r"ptr=(?P<ptr>0x[0-9a-fA-F]+) end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"device=(?P<device>-?\d+) phase=(?P<phase>\S+) alloc_id=(?P<alloc_id>\d+) "
    r"alloc_phase=(?P<alloc_phase>\S+) alloc_predicted_class=(?P<predicted_class>[^ ]+) "
    r"alloc_action=(?P<action>[^ ]+) alloc_policy_source=(?P<policy_source>[^ ]+) "
    r"alloc_policy_success=(?P<policy_success>\d+) alloc_policy_error=(?P<policy_error>[^ ]+) "
    r"gap_watch_target_class=(?P<target_class>[^ ]+) gap_watch_policy_action=(?P<policy_action>[^ ]+) "
    r".* overlap_bytes=(?P<overlap_bytes>\d+) .* lifetime_s=(?P<lifetime_s>-?[0-9.]+)"
)
SUMMARY_KEY_RE = re.compile(r"^\s{2}(?P<key>[^:]+): (?P<value>.+)$")
KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^ ]+)")


def parse_kv_fields(line: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value").strip()
        for match in KV_RE.finditer(line)
    }


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_bool_from_int(value: object) -> bool | None:
    parsed = optional_int(value)
    if parsed is None:
        return None
    return bool(parsed)


def counter_ratios(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize whether gap-watch policy hit and succeeded."
    )
    parser.add_argument(
        "--allocator-log",
        required=True,
        help="Path to the allocator trace log.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allocator_log = Path(args.allocator_log)
    if not allocator_log.is_file():
        raise SystemExit(f"allocator log not found: {allocator_log}")

    policy_records = 0
    gap_overlap_records = 0
    gap_policy_records = 0
    gap_policy_success = 0
    gap_policy_fail = 0
    gap_overlap_bytes = 0
    gap_policy_overlap_bytes = 0
    class_match_records = 0
    action_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    target_class_counter: Counter[str] = Counter()
    policy_source_counter: Counter[str] = Counter()
    placement_backend_counter: Counter[str] = Counter()
    device_direct_backend_counter: Counter[str] = Counter()
    device_direct_reason_counter: Counter[str] = Counter()
    lifetime_values: list[float] = []
    session_summary: dict[str, str] = {}
    current_session_summary: dict[str, str] = {}
    session_summaries: list[dict[str, str]] = []
    in_summary = False
    device_direct_trace_records = 0
    device_direct_eligible_records = 0
    hot_gap_match_records = 0
    device_direct_max_total_bytes_values: list[int] = []
    device_direct_live_bytes_values: list[int] = []
    device_direct_budget_remaining_values: list[int] = []
    device_direct_pool_release_threshold_set_values: list[int] = []
    device_direct_pool_release_threshold_values: list[int] = []
    device_direct_pool_config_attempted_values: list[int] = []
    device_direct_pool_config_success_values: list[int] = []
    device_direct_pool_config_device_values: list[int] = []
    kv_budget_reason_counter: Counter[str] = Counter()
    kv_budget_tracked_records = 0
    kv_budget_over_records = 0
    kv_budget_bytes_values: list[int] = []
    kv_live_bytes_values: list[int] = []
    kv_budget_remaining_values: list[int] = []
    kv_budget_mode_values: list[str] = []
    weight_budget_reason_counter: Counter[str] = Counter()
    weight_budget_tracked_records = 0
    weight_budget_over_records = 0
    weight_budget_bytes_values: list[int] = []
    weight_live_bytes_values: list[int] = []
    weight_budget_remaining_values: list[int] = []
    weight_budget_mode_values: list[str] = []
    pool_alloc_records = 0
    pool_registry_enabled_values: list[int] = []
    pool_free_records = 0
    pool_kind_counter: Counter[str] = Counter()
    pool_class_counter: Counter[str] = Counter()
    pool_phase_counter: Counter[str] = Counter()
    pool_placement_counter: Counter[str] = Counter()
    pool_alloc_bytes_counter: Counter[str] = Counter()
    pool_free_bytes_counter: Counter[str] = Counter()
    pool_live_object_values: list[int] = []
    pool_peak_live_object_values: list[int] = []
    pool_kv_live_values: list[int] = []
    pool_weight_live_values: list[int] = []
    pool_runtime_scratch_live_values: list[int] = []
    scratch_pool_reason_counter: Counter[str] = Counter()
    scratch_pool_tracked_records = 0
    scratch_pool_eligible_records = 0
    scratch_pool_device_direct_records = 0
    scratch_pool_over_budget_records = 0
    scratch_pool_enabled_values: list[int] = []
    scratch_pool_budget_bytes_values: list[int] = []
    scratch_pool_live_values: list[int] = []
    scratch_pool_budget_remaining_values: list[int] = []
    scratch_pool_mode_values: list[str] = []

    def record_scratch_pool_fields(kv_fields: dict[str, str]) -> None:
        nonlocal scratch_pool_tracked_records
        nonlocal scratch_pool_eligible_records
        nonlocal scratch_pool_device_direct_records
        nonlocal scratch_pool_over_budget_records
        if kv_fields.get("scratch_pool_tracked") != "1":
            return
        scratch_pool_tracked_records += 1
        if kv_fields.get("scratch_pool_eligible") == "1":
            scratch_pool_eligible_records += 1
        if kv_fields.get("scratch_pool_device_direct") == "1":
            scratch_pool_device_direct_records += 1
        if kv_fields.get("scratch_pool_over_budget") == "1":
            scratch_pool_over_budget_records += 1
        scratch_pool_reason_counter[
            kv_fields.get("scratch_pool_reason", "not_recorded")
        ] += 1
        if "scratch_pool_enable" in kv_fields:
            scratch_pool_enabled_values.append(int(kv_fields["scratch_pool_enable"]))
        if "scratch_pool_budget_bytes" in kv_fields:
            scratch_pool_budget_bytes_values.append(
                int(kv_fields["scratch_pool_budget_bytes"])
            )
        if "scratch_pool_live_bytes" in kv_fields:
            scratch_pool_live_values.append(int(kv_fields["scratch_pool_live_bytes"]))
        if "scratch_pool_budget_remaining" in kv_fields:
            scratch_pool_budget_remaining_values.append(
                int(kv_fields["scratch_pool_budget_remaining"])
            )
        if "scratch_pool_mode" in kv_fields:
            scratch_pool_mode_values.append(kv_fields["scratch_pool_mode"])

    def record_kv_budget_fields(kv_fields: dict[str, str]) -> None:
        nonlocal kv_budget_tracked_records, kv_budget_over_records
        if kv_fields.get("kv_budget_tracked") != "1":
            return
        kv_budget_tracked_records += 1
        if kv_fields.get("kv_budget_over_budget") == "1":
            kv_budget_over_records += 1
        kv_budget_reason_counter[kv_fields.get("kv_budget_reason", "not_recorded")] += 1
        if "kv_budget_bytes" in kv_fields:
            kv_budget_bytes_values.append(int(kv_fields["kv_budget_bytes"]))
        if "kv_live_bytes" in kv_fields:
            kv_live_bytes_values.append(int(kv_fields["kv_live_bytes"]))
        if "kv_budget_remaining" in kv_fields:
            kv_budget_remaining_values.append(int(kv_fields["kv_budget_remaining"]))
        if "kv_budget_mode" in kv_fields:
            kv_budget_mode_values.append(kv_fields["kv_budget_mode"])

    def record_weight_budget_fields(kv_fields: dict[str, str]) -> None:
        nonlocal weight_budget_tracked_records, weight_budget_over_records
        if kv_fields.get("weight_budget_tracked") != "1":
            return
        weight_budget_tracked_records += 1
        if kv_fields.get("weight_budget_over_budget") == "1":
            weight_budget_over_records += 1
        weight_budget_reason_counter[
            kv_fields.get("weight_budget_reason", "not_recorded")
        ] += 1
        if "weight_budget_bytes" in kv_fields:
            weight_budget_bytes_values.append(int(kv_fields["weight_budget_bytes"]))
        if "weight_live_bytes" in kv_fields:
            weight_live_bytes_values.append(int(kv_fields["weight_live_bytes"]))
        if "weight_budget_remaining" in kv_fields:
            weight_budget_remaining_values.append(
                int(kv_fields["weight_budget_remaining"])
            )
        if "weight_budget_mode" in kv_fields:
            weight_budget_mode_values.append(kv_fields["weight_budget_mode"])

    with allocator_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Session Summary" in line:
                in_summary = True
                current_session_summary = {}
                continue
            if in_summary:
                if line.startswith("========================================"):
                    in_summary = False
                    if current_session_summary:
                        session_summaries.append(current_session_summary)
                    continue
                summary_match = SUMMARY_KEY_RE.search(line)
                if summary_match is not None:
                    current_session_summary[summary_match.group("key")] = (
                        summary_match.group("value")
                    )

            policy_match = TRACE_POLICY_RE.search(line)
            if policy_match is not None:
                kv_fields = parse_kv_fields(line)
                record_kv_budget_fields(kv_fields)
                record_weight_budget_fields(kv_fields)
                record_scratch_pool_fields(kv_fields)
                policy_records += 1
                continue

            gap_alloc_match = TRACE_GAP_ALLOC_RE.search(line)
            if gap_alloc_match is not None:
                kv_fields = parse_kv_fields(line)
                record_weight_budget_fields(kv_fields)
                record_scratch_pool_fields(kv_fields)
                gap_overlap_records += 1
                overlap_bytes = int(gap_alloc_match.group("overlap_bytes"))
                gap_overlap_bytes += overlap_bytes
                predicted_class = gap_alloc_match.group("predicted_class")
                phase = gap_alloc_match.group("phase")
                action = gap_alloc_match.group("action")
                policy_source = gap_alloc_match.group("policy_source")
                target_class = gap_alloc_match.group("target_class")
                class_match = gap_alloc_match.group("class_match") == "1"
                class_counter[predicted_class] += 1
                phase_counter[phase] += 1
                action_counter[action] += 1
                policy_source_counter[policy_source] += 1
                target_class_counter[target_class] += 1
                placement_backend_counter[kv_fields.get("placement_backend", "unknown")] += 1
                device_direct_backend_counter[
                    kv_fields.get("device_direct_backend", "none")
                ] += 1
                device_direct_reason_counter[
                    kv_fields.get("device_direct_reason", "not_recorded")
                ] += 1
                if action in {"device_direct_trace", "device_direct"}:
                    device_direct_trace_records += 1
                if kv_fields.get("device_direct_eligible") == "1":
                    device_direct_eligible_records += 1
                if kv_fields.get("hot_gap_match") == "1":
                    hot_gap_match_records += 1
                if "device_direct_max_total_bytes" in kv_fields:
                    device_direct_max_total_bytes_values.append(
                        int(kv_fields["device_direct_max_total_bytes"])
                    )
                if "device_direct_live_bytes" in kv_fields:
                    device_direct_live_bytes_values.append(
                        int(kv_fields["device_direct_live_bytes"])
                    )
                if "device_direct_budget_remaining" in kv_fields:
                    device_direct_budget_remaining_values.append(
                        int(kv_fields["device_direct_budget_remaining"])
                    )
                if "device_direct_pool_release_threshold_set" in kv_fields:
                    device_direct_pool_release_threshold_set_values.append(
                        int(kv_fields["device_direct_pool_release_threshold_set"])
                    )
                if "device_direct_pool_release_threshold" in kv_fields:
                    device_direct_pool_release_threshold_values.append(
                        int(kv_fields["device_direct_pool_release_threshold"])
                    )
                if "device_direct_pool_config_attempted" in kv_fields:
                    device_direct_pool_config_attempted_values.append(
                        int(kv_fields["device_direct_pool_config_attempted"])
                    )
                if "device_direct_pool_config_success" in kv_fields:
                    device_direct_pool_config_success_values.append(
                        int(kv_fields["device_direct_pool_config_success"])
                    )
                if "device_direct_pool_config_device" in kv_fields:
                    device_direct_pool_config_device_values.append(
                        int(kv_fields["device_direct_pool_config_device"])
                    )
                if class_match:
                    class_match_records += 1
                if policy_source == "gap_watch_policy" and action != "managed_default":
                    gap_policy_records += 1
                    gap_policy_overlap_bytes += overlap_bytes
                continue

            gap_free_match = TRACE_GAP_FREE_RE.search(line)
            if gap_free_match is not None:
                if gap_free_match.group("policy_source") == "gap_watch_policy" and gap_free_match.group("action") != "managed_default":
                    if gap_free_match.group("policy_success") == "1":
                        gap_policy_success += 1
                    else:
                        gap_policy_fail += 1
                lifetime_s = float(gap_free_match.group("lifetime_s"))
                if lifetime_s >= 0:
                    lifetime_values.append(lifetime_s)

            if "TRACE_POOL_ALLOC" in line:
                kv_fields = parse_kv_fields(line)
                pool_kind = kv_fields.get("pool_kind", "unknown")
                pool_alloc_records += 1
                pool_kind_counter[pool_kind] += 1
                pool_class_counter[kv_fields.get("predicted_class", "unknown")] += 1
                pool_phase_counter[kv_fields.get("phase", "unknown")] += 1
                pool_placement_counter[
                    kv_fields.get("placement_backend", "unknown")
                ] += 1
                if "pool_registry_enabled" in kv_fields:
                    pool_registry_enabled_values.append(
                        int(kv_fields["pool_registry_enabled"])
                    )
                if "size_bytes" in kv_fields:
                    pool_alloc_bytes_counter[pool_kind] += int(kv_fields["size_bytes"])
                if "pool_registry_live_objects" in kv_fields:
                    pool_live_object_values.append(
                        int(kv_fields["pool_registry_live_objects"])
                    )
                if "pool_registry_peak_live_objects" in kv_fields:
                    pool_peak_live_object_values.append(
                        int(kv_fields["pool_registry_peak_live_objects"])
                    )
                if "pool_kv_live_bytes" in kv_fields:
                    pool_kv_live_values.append(int(kv_fields["pool_kv_live_bytes"]))
                if "pool_weight_live_bytes" in kv_fields:
                    pool_weight_live_values.append(
                        int(kv_fields["pool_weight_live_bytes"])
                    )
                if "pool_runtime_scratch_live_bytes" in kv_fields:
                    pool_runtime_scratch_live_values.append(
                        int(kv_fields["pool_runtime_scratch_live_bytes"])
                    )
                continue

            if "TRACE_POOL_FREE" in line:
                kv_fields = parse_kv_fields(line)
                pool_kind = kv_fields.get("pool_kind", "unknown")
                pool_free_records += 1
                if "size_bytes" in kv_fields:
                    pool_free_bytes_counter[pool_kind] += int(kv_fields["size_bytes"])
                if "pool_registry_live_objects" in kv_fields:
                    pool_live_object_values.append(
                        int(kv_fields["pool_registry_live_objects"])
                    )
                if "pool_registry_peak_live_objects" in kv_fields:
                    pool_peak_live_object_values.append(
                        int(kv_fields["pool_registry_peak_live_objects"])
                    )
                if "pool_kv_live_bytes" in kv_fields:
                    pool_kv_live_values.append(int(kv_fields["pool_kv_live_bytes"]))
                if "pool_weight_live_bytes" in kv_fields:
                    pool_weight_live_values.append(
                        int(kv_fields["pool_weight_live_bytes"])
                    )
                if "pool_runtime_scratch_live_bytes" in kv_fields:
                    pool_runtime_scratch_live_values.append(
                        int(kv_fields["pool_runtime_scratch_live_bytes"])
                    )
                continue

    if in_summary and current_session_summary:
        session_summaries.append(current_session_summary)
    if session_summaries:
        session_summary = max(
            session_summaries,
            key=lambda item: optional_int(item.get("Total allocations")) or 0,
        )

    summary = {
        "allocator_log": str(allocator_log),
        "policy_records": policy_records,
        "gap_overlap_records": gap_overlap_records,
        "gap_overlap_bytes": gap_overlap_bytes,
        "gap_policy_records": gap_policy_records,
        "gap_policy_overlap_bytes": gap_policy_overlap_bytes,
        "gap_policy_success": gap_policy_success,
        "gap_policy_fail": gap_policy_fail,
        "gap_watch_class_match_records": class_match_records,
        "dominant_predicted_class": class_counter.most_common(1)[0][0] if class_counter else None,
        "dominant_phase": phase_counter.most_common(1)[0][0] if phase_counter else None,
        "dominant_action": action_counter.most_common(1)[0][0] if action_counter else None,
        "dominant_target_class": target_class_counter.most_common(1)[0][0] if target_class_counter else None,
        "policy_source_counts": dict(policy_source_counter),
        "predicted_class_counts": dict(class_counter),
        "phase_counts": dict(phase_counter),
        "phase_record_ratios": counter_ratios(phase_counter),
        "action_counts": dict(action_counter),
        "target_class_counts": dict(target_class_counter),
        "placement_backend_counts": dict(placement_backend_counter),
        "device_direct_backend_counts": dict(device_direct_backend_counter),
        "device_direct_reason_counts": dict(device_direct_reason_counter),
        "device_direct_trace_records": device_direct_trace_records,
        "device_direct_eligible_records": device_direct_eligible_records,
        "device_direct_actual_records": placement_backend_counter.get(
            "device_direct", 0
        ),
        "device_direct_budget_reject_records": device_direct_reason_counter.get(
            "device_direct_budget_exceeded", 0
        ),
        "device_direct_max_total_bytes": (
            device_direct_max_total_bytes_values[-1]
            if device_direct_max_total_bytes_values
            else None
        ),
        "device_direct_peak_live_bytes_observed": (
            max(device_direct_live_bytes_values)
            if device_direct_live_bytes_values
            else None
        ),
        "device_direct_min_budget_remaining_observed": (
            min(device_direct_budget_remaining_values)
            if device_direct_budget_remaining_values
            else None
        ),
        "device_direct_pool_release_threshold_set": (
            bool(device_direct_pool_release_threshold_set_values[-1])
            if device_direct_pool_release_threshold_set_values
            else optional_bool_from_int(
                session_summary.get("Device-direct pool release threshold set")
            )
        ),
        "device_direct_pool_release_threshold": (
            device_direct_pool_release_threshold_values[-1]
            if device_direct_pool_release_threshold_values
            else optional_int(
                session_summary.get("Device-direct pool release threshold")
            )
        ),
        "device_direct_pool_config_attempted": session_summary.get(
            "Device-direct pool config attempted"
        ),
        "device_direct_pool_config_success": session_summary.get(
            "Device-direct pool config success"
        ),
        "device_direct_pool_config_device": session_summary.get(
            "Device-direct pool config device"
        ),
        "device_direct_pool_config_error": session_summary.get(
            "Device-direct pool config error"
        ),
        "kv_budget_bytes": (
            kv_budget_bytes_values[-1]
            if kv_budget_bytes_values
            else optional_int(session_summary.get("KV budget bytes"))
        ),
        "kv_budget_mode": (
            kv_budget_mode_values[-1]
            if kv_budget_mode_values
            else session_summary.get("KV budget mode")
        ),
        "kv_trace_allocations": (
            kv_budget_tracked_records
            if kv_budget_tracked_records
            else optional_int(session_summary.get("KV trace allocations"))
        ),
        "kv_requested_bytes": optional_int(
            session_summary.get("KV requested bytes")
        ),
        "kv_live_bytes": (
            kv_live_bytes_values[-1]
            if kv_live_bytes_values
            else optional_int(session_summary.get("KV live bytes"))
        ),
        "kv_peak_live_bytes_observed": (
            max(kv_live_bytes_values)
            if kv_live_bytes_values
            else optional_int(session_summary.get("KV peak live bytes"))
        ),
        "kv_min_budget_remaining_observed": (
            min(kv_budget_remaining_values)
            if kv_budget_remaining_values
            else optional_int(session_summary.get("KV budget remaining"))
        ),
        "kv_budget_over_records": (
            kv_budget_over_records
            if kv_budget_tracked_records
            else optional_int(session_summary.get("KV budget over allocations"))
        ),
        "kv_budget_reject_records": (
            kv_budget_reason_counter.get("kv_budget_exceeded_soft_enforce", 0)
            if kv_budget_tracked_records
            else optional_int(session_summary.get("KV budget reject allocations"))
        ),
        "kv_budget_reason_counts": dict(kv_budget_reason_counter),
        "weight_budget_bytes": (
            weight_budget_bytes_values[-1]
            if weight_budget_bytes_values
            else optional_int(session_summary.get("Weight budget bytes"))
        ),
        "weight_budget_mode": (
            weight_budget_mode_values[-1]
            if weight_budget_mode_values
            else session_summary.get("Weight budget mode")
        ),
        "weight_trace_allocations": (
            weight_budget_tracked_records
            if weight_budget_tracked_records
            else optional_int(session_summary.get("Weight trace allocations"))
        ),
        "weight_requested_bytes": optional_int(
            session_summary.get("Weight requested bytes")
        ),
        "weight_live_bytes": (
            weight_live_bytes_values[-1]
            if weight_live_bytes_values
            else optional_int(session_summary.get("Weight live bytes"))
        ),
        "weight_peak_live_bytes_observed": (
            max(weight_live_bytes_values)
            if weight_live_bytes_values
            else optional_int(session_summary.get("Weight peak live bytes"))
        ),
        "weight_min_budget_remaining_observed": (
            min(weight_budget_remaining_values)
            if weight_budget_remaining_values
            else optional_int(session_summary.get("Weight budget remaining"))
        ),
        "weight_budget_over_records": (
            weight_budget_over_records
            if weight_budget_tracked_records
            else optional_int(session_summary.get("Weight budget over allocations"))
        ),
        "weight_budget_reject_records": (
            weight_budget_reason_counter.get(
                "weight_budget_exceeded_soft_enforce", 0
            )
            if weight_budget_tracked_records
            else optional_int(session_summary.get("Weight budget reject allocations"))
        ),
        "weight_budget_reason_counts": dict(weight_budget_reason_counter),
        "pool_registry_enabled": optional_bool_from_int(
            pool_registry_enabled_values[-1]
            if pool_registry_enabled_values
            else session_summary.get("Pool registry enabled")
        ),
        "pool_registry_alloc_records": (
            pool_alloc_records
            if pool_alloc_records
            else optional_int(session_summary.get("Pool registry tracked allocations"))
        ),
        "pool_registry_free_records": (
            pool_free_records
            if pool_free_records
            else optional_int(session_summary.get("Pool registry free success"))
        ),
        "pool_registry_live_objects": (
            pool_live_object_values[-1]
            if pool_live_object_values
            else optional_int(session_summary.get("Pool registry live objects"))
        ),
        "pool_registry_peak_live_objects": (
            max(pool_peak_live_object_values)
            if pool_peak_live_object_values
            else optional_int(session_summary.get("Pool registry peak live objects"))
        ),
        "pool_kind_counts": dict(pool_kind_counter),
        "pool_class_counts": dict(pool_class_counter),
        "pool_phase_counts": dict(pool_phase_counter),
        "pool_placement_backend_counts": dict(pool_placement_counter),
        "pool_alloc_bytes_by_kind": dict(pool_alloc_bytes_counter),
        "pool_free_bytes_by_kind": dict(pool_free_bytes_counter),
        "pool_kv_allocations": optional_int(
            session_summary.get("Pool kv allocations")
        ),
        "pool_kv_requested_bytes": optional_int(
            session_summary.get("Pool kv requested bytes")
        ),
        "pool_kv_live_bytes": (
            pool_kv_live_values[-1]
            if pool_kv_live_values
            else optional_int(session_summary.get("Pool kv live bytes"))
        ),
        "pool_kv_peak_live_bytes": (
            max(pool_kv_live_values)
            if pool_kv_live_values
            else optional_int(session_summary.get("Pool kv peak live bytes"))
        ),
        "pool_kv_free_success": optional_int(
            session_summary.get("Pool kv free success")
        ),
        "pool_weight_allocations": optional_int(
            session_summary.get("Pool weight allocations")
        ),
        "pool_weight_requested_bytes": optional_int(
            session_summary.get("Pool weight requested bytes")
        ),
        "pool_weight_live_bytes": (
            pool_weight_live_values[-1]
            if pool_weight_live_values
            else optional_int(session_summary.get("Pool weight live bytes"))
        ),
        "pool_weight_peak_live_bytes": (
            max(pool_weight_live_values)
            if pool_weight_live_values
            else optional_int(session_summary.get("Pool weight peak live bytes"))
        ),
        "pool_weight_free_success": optional_int(
            session_summary.get("Pool weight free success")
        ),
        "pool_runtime_scratch_allocations": optional_int(
            session_summary.get("Pool runtime scratch allocations")
        ),
        "pool_runtime_scratch_requested_bytes": optional_int(
            session_summary.get("Pool runtime scratch requested bytes")
        ),
        "pool_runtime_scratch_live_bytes": (
            pool_runtime_scratch_live_values[-1]
            if pool_runtime_scratch_live_values
            else optional_int(session_summary.get("Pool runtime scratch live bytes"))
        ),
        "pool_runtime_scratch_peak_live_bytes": (
            max(pool_runtime_scratch_live_values)
            if pool_runtime_scratch_live_values
            else optional_int(
                session_summary.get("Pool runtime scratch peak live bytes")
            )
        ),
        "pool_runtime_scratch_free_success": optional_int(
            session_summary.get("Pool runtime scratch free success")
        ),
        "scratch_pool_enabled": (
            optional_bool_from_int(scratch_pool_enabled_values[-1])
            if scratch_pool_enabled_values
            else optional_bool_from_int(session_summary.get("Scratch pool enabled"))
        ),
        "scratch_pool_budget_bytes": (
            scratch_pool_budget_bytes_values[-1]
            if scratch_pool_budget_bytes_values
            else optional_int(session_summary.get("Scratch pool budget bytes"))
        ),
        "scratch_pool_mode": (
            scratch_pool_mode_values[-1]
            if scratch_pool_mode_values
            else session_summary.get("Scratch pool mode")
        ),
        "scratch_pool_trace_records": (
            scratch_pool_tracked_records
            if scratch_pool_tracked_records
            else optional_int(session_summary.get("Scratch pool trace allocations"))
        ),
        "scratch_pool_eligible_records": (
            scratch_pool_eligible_records
            if scratch_pool_tracked_records
            else optional_int(session_summary.get("Scratch pool eligible allocations"))
        ),
        "scratch_pool_device_direct_records": (
            scratch_pool_device_direct_records
            if scratch_pool_tracked_records
            else optional_int(
                session_summary.get("Scratch pool device-direct allocations")
            )
        ),
        "scratch_pool_device_direct_bytes": optional_int(
            session_summary.get("Scratch pool device-direct bytes")
        ),
        "scratch_pool_live_bytes": (
            scratch_pool_live_values[-1]
            if scratch_pool_live_values
            else optional_int(
                session_summary.get("Scratch pool device-direct live bytes")
            )
        ),
        "scratch_pool_peak_live_bytes": (
            max(scratch_pool_live_values)
            if scratch_pool_live_values
            else optional_int(
                session_summary.get("Scratch pool device-direct peak live bytes")
            )
        ),
        "scratch_pool_min_budget_remaining_observed": (
            min(scratch_pool_budget_remaining_values)
            if scratch_pool_budget_remaining_values
            else optional_int(session_summary.get("Scratch pool budget remaining"))
        ),
        "scratch_pool_budget_over_records": (
            scratch_pool_over_budget_records
            if scratch_pool_tracked_records
            else optional_int(
                session_summary.get("Scratch pool budget over allocations")
            )
        ),
        "scratch_pool_budget_reject_records": (
            scratch_pool_reason_counter.get(
                "scratch_pool_budget_exceeded_fallback_managed", 0
            )
            if scratch_pool_tracked_records
            else optional_int(
                session_summary.get("Scratch pool budget reject allocations")
            )
        ),
        "scratch_pool_reason_counts": dict(scratch_pool_reason_counter),
        "scratch_pool_reuse_cache_enabled": optional_bool_from_int(
            session_summary.get("Scratch pool reuse cache enabled")
        ),
        "scratch_pool_reuse_cache_max_bytes": optional_int(
            session_summary.get("Scratch pool reuse cache max bytes")
        ),
        "scratch_pool_reuse_cache_max_entries_per_bucket": optional_int(
            session_summary.get(
                "Scratch pool reuse cache max entries per bucket"
            )
        ),
        "scratch_pool_reuse_cache_hits": optional_int(
            session_summary.get("Scratch pool reuse cache hits")
        ),
        "scratch_pool_reuse_cache_misses": optional_int(
            session_summary.get("Scratch pool reuse cache misses")
        ),
        "scratch_pool_reuse_cache_fresh_backend_allocations": optional_int(
            session_summary.get(
                "Scratch pool reuse cache fresh backend allocations"
            )
        ),
        "scratch_pool_reuse_cache_inserts": optional_int(
            session_summary.get("Scratch pool reuse cache inserts")
        ),
        "scratch_pool_reuse_cache_insert_rejects": optional_int(
            session_summary.get("Scratch pool reuse cache insert rejects")
        ),
        "scratch_pool_reuse_cache_hit_bytes": optional_int(
            session_summary.get("Scratch pool reuse cache hit bytes")
        ),
        "scratch_pool_reuse_cache_insert_bytes": optional_int(
            session_summary.get("Scratch pool reuse cache insert bytes")
        ),
        "scratch_pool_reuse_cache_cached_bytes": optional_int(
            session_summary.get("Scratch pool reuse cache cached bytes")
        ),
        "scratch_pool_reuse_cache_peak_cached_bytes": optional_int(
            session_summary.get("Scratch pool reuse cache peak cached bytes")
        ),
        "scratch_pool_reuse_cache_cached_entries": optional_int(
            session_summary.get("Scratch pool reuse cache cached entries")
        ),
        "scratch_pool_reuse_cache_event_not_ready": optional_int(
            session_summary.get("Scratch pool reuse cache event not ready")
        ),
        "scratch_pool_reuse_cache_event_errors": optional_int(
            session_summary.get("Scratch pool reuse cache event errors")
        ),
        "hot_gap_match_records": hot_gap_match_records,
        "median_lifetime_s": (
            sorted(lifetime_values)[len(lifetime_values) // 2]
            if lifetime_values
            else None
        ),
        "session_summary": session_summary,
    }

    summary["device_direct_pool_config_attempted"] = (
        optional_int(summary["device_direct_pool_config_attempted"])
        if summary["device_direct_pool_config_attempted"] is not None
        else (
            device_direct_pool_config_attempted_values[-1]
            if device_direct_pool_config_attempted_values
            else None
        )
    )
    summary["device_direct_pool_config_success"] = (
        optional_int(summary["device_direct_pool_config_success"])
        if summary["device_direct_pool_config_success"] is not None
        else (
            device_direct_pool_config_success_values[-1]
            if device_direct_pool_config_success_values
            else None
        )
    )
    summary["device_direct_pool_config_device"] = (
        optional_int(summary["device_direct_pool_config_device"])
        if summary["device_direct_pool_config_device"] is not None
        else (
            device_direct_pool_config_device_values[-1]
            if device_direct_pool_config_device_values
            else None
        )
    )

    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print("Gap Watch Metrics Summary")
    print(f"- gap_overlap_records={gap_overlap_records}")
    print(f"- gap_policy_records={gap_policy_records}")
    print(f"- gap_policy_success={gap_policy_success}")
    print(f"- gap_policy_fail={gap_policy_fail}")
    print(f"- gap_overlap_bytes={gap_overlap_bytes}")
    print(f"- gap_policy_overlap_bytes={gap_policy_overlap_bytes}")
    print(f"- dominant_predicted_class={summary['dominant_predicted_class']}")
    print(f"- dominant_phase={summary['dominant_phase']}")
    print(f"- dominant_action={summary['dominant_action']}")
    print(f"- dominant_target_class={summary['dominant_target_class']}")
    print(f"- phase_record_ratios={summary['phase_record_ratios']}")
    print(f"- placement_backend_counts={summary['placement_backend_counts']}")
    print(f"- device_direct_backend_counts={summary['device_direct_backend_counts']}")
    print(f"- device_direct_trace_records={device_direct_trace_records}")
    print(f"- device_direct_eligible_records={device_direct_eligible_records}")
    print(f"- device_direct_actual_records={summary['device_direct_actual_records']}")
    print(f"- device_direct_budget_reject_records={summary['device_direct_budget_reject_records']}")
    print(f"- device_direct_max_total_bytes={summary['device_direct_max_total_bytes']}")
    print(f"- device_direct_peak_live_bytes_observed={summary['device_direct_peak_live_bytes_observed']}")
    print(f"- device_direct_min_budget_remaining_observed={summary['device_direct_min_budget_remaining_observed']}")
    print(f"- device_direct_pool_release_threshold_set={summary['device_direct_pool_release_threshold_set']}")
    print(f"- device_direct_pool_release_threshold={summary['device_direct_pool_release_threshold']}")
    print(f"- device_direct_pool_config_attempted={summary['device_direct_pool_config_attempted']}")
    print(f"- device_direct_pool_config_success={summary['device_direct_pool_config_success']}")
    print(f"- device_direct_pool_config_error={summary['device_direct_pool_config_error']}")
    print(f"- kv_budget_bytes={summary['kv_budget_bytes']}")
    print(f"- kv_budget_mode={summary['kv_budget_mode']}")
    print(f"- kv_trace_allocations={summary['kv_trace_allocations']}")
    print(f"- kv_live_bytes={summary['kv_live_bytes']}")
    print(f"- kv_peak_live_bytes_observed={summary['kv_peak_live_bytes_observed']}")
    print(f"- kv_min_budget_remaining_observed={summary['kv_min_budget_remaining_observed']}")
    print(f"- kv_budget_over_records={summary['kv_budget_over_records']}")
    print(f"- kv_budget_reject_records={summary['kv_budget_reject_records']}")
    print(f"- kv_budget_reason_counts={summary['kv_budget_reason_counts']}")
    print(f"- weight_budget_bytes={summary['weight_budget_bytes']}")
    print(f"- weight_budget_mode={summary['weight_budget_mode']}")
    print(f"- weight_trace_allocations={summary['weight_trace_allocations']}")
    print(f"- weight_live_bytes={summary['weight_live_bytes']}")
    print(f"- weight_peak_live_bytes_observed={summary['weight_peak_live_bytes_observed']}")
    print(f"- weight_min_budget_remaining_observed={summary['weight_min_budget_remaining_observed']}")
    print(f"- weight_budget_over_records={summary['weight_budget_over_records']}")
    print(f"- weight_budget_reject_records={summary['weight_budget_reject_records']}")
    print(f"- weight_budget_reason_counts={summary['weight_budget_reason_counts']}")
    print(f"- pool_registry_enabled={summary['pool_registry_enabled']}")
    print(f"- pool_registry_alloc_records={summary['pool_registry_alloc_records']}")
    print(f"- pool_registry_free_records={summary['pool_registry_free_records']}")
    print(f"- pool_registry_live_objects={summary['pool_registry_live_objects']}")
    print(f"- pool_registry_peak_live_objects={summary['pool_registry_peak_live_objects']}")
    print(f"- pool_kind_counts={summary['pool_kind_counts']}")
    print(f"- pool_alloc_bytes_by_kind={summary['pool_alloc_bytes_by_kind']}")
    print(f"- pool_kv_live_bytes={summary['pool_kv_live_bytes']}")
    print(f"- pool_weight_live_bytes={summary['pool_weight_live_bytes']}")
    print(f"- pool_runtime_scratch_live_bytes={summary['pool_runtime_scratch_live_bytes']}")
    print(f"- scratch_pool_enabled={summary['scratch_pool_enabled']}")
    print(f"- scratch_pool_budget_bytes={summary['scratch_pool_budget_bytes']}")
    print(f"- scratch_pool_mode={summary['scratch_pool_mode']}")
    print(f"- scratch_pool_trace_records={summary['scratch_pool_trace_records']}")
    print(f"- scratch_pool_eligible_records={summary['scratch_pool_eligible_records']}")
    print(f"- scratch_pool_device_direct_records={summary['scratch_pool_device_direct_records']}")
    print(f"- scratch_pool_live_bytes={summary['scratch_pool_live_bytes']}")
    print(f"- scratch_pool_peak_live_bytes={summary['scratch_pool_peak_live_bytes']}")
    print(f"- scratch_pool_budget_over_records={summary['scratch_pool_budget_over_records']}")
    print(f"- scratch_pool_budget_reject_records={summary['scratch_pool_budget_reject_records']}")
    print(f"- scratch_pool_reason_counts={summary['scratch_pool_reason_counts']}")
    print(f"- scratch_pool_reuse_cache_enabled={summary['scratch_pool_reuse_cache_enabled']}")
    print(f"- scratch_pool_reuse_cache_hits={summary['scratch_pool_reuse_cache_hits']}")
    print(f"- scratch_pool_reuse_cache_misses={summary['scratch_pool_reuse_cache_misses']}")
    print(f"- scratch_pool_reuse_cache_fresh_backend_allocations={summary['scratch_pool_reuse_cache_fresh_backend_allocations']}")
    print(f"- scratch_pool_reuse_cache_inserts={summary['scratch_pool_reuse_cache_inserts']}")
    print(f"- scratch_pool_reuse_cache_insert_rejects={summary['scratch_pool_reuse_cache_insert_rejects']}")
    print(f"- scratch_pool_reuse_cache_peak_cached_bytes={summary['scratch_pool_reuse_cache_peak_cached_bytes']}")
    print(f"- scratch_pool_reuse_cache_cached_entries={summary['scratch_pool_reuse_cache_cached_entries']}")
    print(f"- hot_gap_match_records={hot_gap_match_records}")
    print(f"- median_lifetime_s={summary['median_lifetime_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
