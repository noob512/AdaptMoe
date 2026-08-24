#!/usr/bin/env python3
"""Verify the real-model GPU/free, GPU/backup, CPU/offline qualification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BAD_KERNEL = re.compile(
    r"NVRM:\s*Xid|FAULT_PDE|manual[_ -]migration[_ -]fault|"
    r"\bOops\b|\bBUG:|kernel panic|general protection fault|Call Trace",
    re.IGNORECASE,
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def verify_queue(snapshot: dict[str, Any], label: str) -> None:
    require(snapshot.get("invariant_ok") is True, f"{label}: invariant")
    total = int(snapshot["total_blocks_including_null"])
    groups = [
        snapshot["gpu_free_block_ids"],
        snapshot["gpu_backup_block_ids"],
        snapshot["cpu_offline_block_ids"],
        snapshot["active_block_ids"],
        snapshot["null_block_ids"],
    ]
    flattened = [int(block_id) for group in groups for block_id in group]
    require(len(flattened) == total, f"{label}: conservation")
    require(len(flattened) == len(set(flattened)), f"{label}: queue overlap")
    require(set(flattened) == set(range(total)), f"{label}: missing block")
    require(snapshot["null_block_ids"] == [0], f"{label}: null block")
    target = int(snapshot["applied_target_blocks"])
    gpu_or_active = {
        int(value)
        for key in (
            "gpu_free_block_ids",
            "gpu_backup_block_ids",
            "active_block_ids",
        )
        for value in snapshot[key]
    }
    require(
        all(0 < block_id <= target for block_id in gpu_or_active),
        f"{label}: non-GPU-target block is allocatable",
    )
    require(
        all(int(block_id) > target for block_id in snapshot["cpu_offline_block_ids"]),
        f"{label}: offline block lies in active prefix",
    )


def verify_physical(snapshot: dict[str, Any], target: int, label: str) -> None:
    require(int(snapshot["target_blocks"]) == target, f"{label}: target")
    for part in ("full", "head", "tail"):
        record = snapshot[part]
        require(record.get("ok") is True, f"{label}: {part} query")
        require(int(record.get("unknown_bytes", -1)) == 0, f"{label}: {part} unknown")
    head = snapshot["head"]
    tail = snapshot["tail"]
    full = snapshot["full"]
    require(
        int(head["gpu_bytes"]) == int(head["requested_bytes"])
        and int(head["cpu_bytes"]) == 0,
        f"{label}: GPU prefix residency",
    )
    require(
        int(tail["cpu_bytes"]) == int(tail["requested_bytes"])
        and int(tail["gpu_bytes"]) == 0,
        f"{label}: CPU tail residency",
    )
    require(
        int(full["gpu_bytes"]) + int(full["cpu_bytes"])
        == int(full["requested_bytes"]),
        f"{label}: physical accounting",
    )


def verify_shutdown(result: dict[str, Any], label: str) -> None:
    shutdown = result.get("shutdown_contract") or {}
    require(not shutdown.get("cleanup_errors"), f"{label}: cleanup errors")
    require(
        shutdown.get("controlled_hard_exit_after_cleanup") is True,
        f"{label}: controlled cleanup",
    )
    diagnostics = shutdown.get("diagnostics_after_shutdown") or {}
    model = diagnostics.get("model") or {}
    driver = diagnostics.get("driver") or {}
    require(int(model.get("session_id", -1)) == 0, f"{label}: model session")
    require(
        int(model.get("active_driver_backing_bytes", -1)) == 0,
        f"{label}: model backing",
    )
    require(int(driver.get("session_count", -1)) == 0, f"{label}: driver session")


def one_rpc(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("method") == method]
    require(len(matches) == 1, f"expected one {method} RPC, found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    baseline = load_json(root / "baseline" / "result.json")
    queue = load_json(root / "three_queue" / "result.json")
    for label, result in (("baseline", baseline), ("three_queue", queue)):
        require(
            (root / label / "returncode.txt").read_text().strip() == "0",
            f"{label}: runner return code",
        )
        require(result.get("status") == "ok", f"{label}: runner status")
        require(result.get("arm") == "D", f"{label}: graph arm")
        verify_shutdown(result, label)
        require(
            int(result["graph_wrapper_calls_after_load"].get("full_capture", -1))
            == int(result["graph_wrapper_calls_final"].get("full_capture", -2)),
            f"{label}: Graph recapture",
        )
        require(
            int(
                result["compilation_counter_after_load"][
                    "num_cudagraph_captured"
                ]
            )
            == int(
                result["compilation_counter_final"]["num_cudagraph_captured"]
            ),
            f"{label}: capture counter",
        )

    require(
        baseline["config"].get("three_queue_qualification") is False,
        "baseline unexpectedly enabled three queues",
    )
    require(
        queue["config"].get("three_queue_qualification") is True,
        "queue arm did not enable qualification",
    )
    require(
        baseline["output_token_ids_sha256"] == queue["output_token_ids_sha256"],
        "queue output differs from baseline",
    )
    require(
        [item["output_token_ids"] for item in baseline["outputs"]]
        == [item["output_token_ids"] for item in queue["outputs"]]
        and [item["prompt_tokens"] for item in baseline["outputs"]]
        == [item["prompt_tokens"] for item in queue["outputs"]]
        and all(item.get("finished") is True for item in queue["outputs"]),
        "output records differ",
    )

    baseline_roundtrip = baseline.get("kv_autoscale_roundtrip") or {}
    queue_roundtrip = queue.get("kv_autoscale_roundtrip") or {}
    for label, record, expected_enabled in (
        ("baseline", baseline_roundtrip, False),
        ("three_queue", queue_roundtrip, True),
    ):
        require(
            record.get("schema") == "vllm_uvm_kv_autoscale_roundtrip_v1",
            f"{label}: autoscale roundtrip schema",
        )
        require(
            record.get("enabled") is expected_enabled,
            f"{label}: autoscale enablement",
        )
        require(len(record.get("steps") or []) >= 4, f"{label}: low steps")
        steps = record["steps"]
        require(
            steps[0].get("phase") == "prefill"
            and all(item.get("phase") == "decode" for item in steps[1:]),
            f"{label}: Prefill/Decode phase sequence",
        )
        require(
            all(
                int(item["graph_wrapper_delta"].get("full_replay", 0)) == 1
                for item in steps
            ),
            f"{label}: FULL Graph replay sequence",
        )
        require(
            int(record["compilation_counter_before"]["num_cudagraph_captured"])
            == int(record["compilation_counter_after"]["num_cudagraph_captured"]),
            f"{label}: autoscale Graph recapture",
        )
        require(
            int(record["graph_wrapper_calls_before"].get("full_capture", -1))
            == int(record["graph_wrapper_calls_after"].get("full_capture", -2)),
            f"{label}: autoscale Graph wrapper recapture",
        )
        resume = [
            item
            for item in record.get("resume_transitions") or []
            if item.get("event")
            == "cpu_offload_uvm_original_va_alias_prefill_activated"
            and item.get("ok") is True
        ]
        repark = [
            item
            for item in record.get("repark_transitions") or []
            if item.get("event")
            == "cpu_offload_uvm_original_va_remote_transition"
            and item.get("post_state") == "remote_parked"
            and item.get("ok") is True
        ]
        require(len(resume) == 1, f"{label}: alias resume transition")
        require(len(repark) == 1, f"{label}: remote re-park transition")
        parameter_snapshots = record["parameter_va_snapshots"]
        parameter_vas = [
            parameter_snapshots[key]["vas"]
            for key in (
                "before",
                "after_prefill",
                "after_first_decode",
                "after",
            )
        ]
        require(
            all(value == parameter_vas[0] for value in parameter_vas[1:]),
            f"{label}: Parameter VA changed",
        )
        verify_queue(record["queue_before"], f"{label}: low queue before")
        for index, step in enumerate(steps):
            verify_queue(
                step["queue_after_step"],
                f"{label}: low queue step {index}",
            )
        verify_queue(record["queue_after"], f"{label}: low queue after")
        before_target = int(record["target_before_blocks"])
        after_target = int(record["target_after_blocks"])
        verify_physical(
            record["physical_before"],
            before_target,
            f"{label}: low physical before",
        )
        verify_physical(
            record["physical_after"],
            after_target,
            f"{label}: low physical after",
        )
        require(
            record["physical_before"]["va_sha256"]
            == record["physical_after"]["va_sha256"],
            f"{label}: autoscale KV VA changed",
        )

    require(
        baseline_roundtrip["output_token_ids_sha256"]
        == queue_roundtrip["output_token_ids_sha256"]
        and [
            item["output_token_ids"]
            for item in baseline_roundtrip["outputs"]
        ]
        == [item["output_token_ids"] for item in queue_roundtrip["outputs"]],
        "autoscale low-pressure output differs from baseline",
    )
    require(
        int(baseline_roundtrip["target_before_blocks"])
        == int(baseline_roundtrip["target_after_blocks"]),
        "disabled baseline changed its KV target",
    )
    require(
        queue_roundtrip.get("shrink_observation") is not None
        and int(queue_roundtrip["target_before_blocks"]) == 40
        and int(queue_roundtrip["target_after_blocks"]) == 20,
        "enabled autoscale did not perform the expected 40 -> 20 shrink",
    )
    shrink = queue_roundtrip["shrink_observation"]
    require(
        int(shrink["step"]) >= 2
        and int(shrink["old_target_blocks"]) == 40
        and int(shrink["new_target_blocks"]) == 20,
        "shrink occurred before the third low-pressure round or used an "
        "unexpected target",
    )
    verify_queue(shrink["queue"], "autoscale shrink queue")
    verify_physical(shrink["physical"], 20, "autoscale shrink physical")
    low_actions = [
        str(step["autoscale_after_step"]["last_decision"].get("action"))
        for step in queue_roundtrip["steps"][:3]
    ]
    require(
        low_actions
        == ["observe_low_pressure", "observe_low_pressure", "shrink_pending"],
        "low-pressure streak actions differ",
    )
    require(
        all(
            int(step["target_after_step_blocks"]) == 40
            for step in queue_roundtrip["steps"][:2]
        ),
        "physical shrink occurred before three low-pressure rounds",
    )
    require(
        queue_roundtrip["autoscale_after"].get("expert_restore_pending") is False,
        "expert restore remains pending",
    )

    record = queue.get("three_queue_qualification") or {}
    require(
        record.get("schema") == "vllm_uvm_kv_three_queue_e2e_v1",
        "three-queue record schema",
    )
    before = record["queue_before_request"]
    scheduled = record["prefill_schedule"]["uvm_three_queue_after_schedule"]
    after_prefill = record["queue_after_prefill"]
    after_request = record["queue_after_request"]
    for label, snapshot in (
        ("before", before),
        ("after_schedule", scheduled),
        ("after_prefill", after_prefill),
        ("after_request", after_request),
    ):
        verify_queue(snapshot, label)

    initial_free = set(int(value) for value in before["gpu_free_block_ids"])
    initial_backup = set(int(value) for value in before["gpu_backup_block_ids"])
    scheduled_active = set(int(value) for value in scheduled["active_block_ids"])
    consumed_backup_ids = initial_backup & scheduled_active
    pressure_scheduled = record["prefill_schedule"][
        "uvm_backup_pressure_after_schedule"
    ]
    consumed_blocks = int(pressure_scheduled["pending_blocks"])
    require(initial_free <= scheduled_active, "backup was used before ordinary free")
    require(consumed_backup_ids, "no backup block became active")
    require(
        len(consumed_backup_ids) == consumed_blocks,
        "backup membership and pressure count differ",
    )
    require(
        int(scheduled["gpu_free_blocks"]) == 0,
        "ordinary free remained while backup was active",
    )
    require(int(pressure_scheduled["pending_events"]) == 1, "pressure event count")

    pressure_after = record["pressure_after_prefill"]
    require(
        int(pressure_after["pending_blocks"]) == 0
        and int(pressure_after["pending_events"]) == 0,
        "pressure event was not consumed",
    )
    require(
        int(pressure_after["total_activated_blocks"])
        == int(pressure_after["total_consumed_blocks"])
        == consumed_blocks,
        "activation/consumption accounting",
    )

    initial_target = int(before["applied_target_blocks"])
    grown_target = int(record["target_after_prefill_blocks"])
    final_target = int(record["final_target_blocks"])
    grow_blocks = grown_target - initial_target
    require(grow_blocks >= consumed_blocks, "physical KV grow is too small")
    require(final_target == grown_target, "target changed after pressure grow")
    require(
        int(before["cpu_offline_blocks"])
        - int(after_prefill["cpu_offline_blocks"])
        == grow_blocks,
        "offline-to-GPU queue migration count",
    )
    require(
        int(after_request["gpu_backup_blocks"])
        == int(after_request["backup_target_blocks"])
        == int(before["gpu_backup_blocks"]),
        "backup reserve was not refilled",
    )
    verify_physical(record["physical_before_request"], initial_target, "physical before")
    verify_physical(record["physical_after_prefill"], grown_target, "physical after prefill")
    verify_physical(record["physical_after_request"], final_target, "physical after request")
    require(
        record["physical_before_request"]["va_sha256"]
        == record["physical_after_prefill"]["va_sha256"]
        == record["physical_after_request"]["va_sha256"],
        "KV virtual addresses changed",
    )

    expert_before = record["expert_before_request"]
    expert_after = record["expert_after_prefill"]
    released_experts = (
        int(expert_after["remote_expert_count"])
        - int(expert_before["remote_expert_count"])
    )
    require(released_experts == 48, "expert wave did not release one expert per layer")
    require(
        int(expert_after["remote_region_bytes"]) > 0,
        "expert wave released no audited bytes",
    )
    released_region_bytes = int(expert_after["remote_region_bytes"])
    expert_before_low = queue_roundtrip["expert_before"]
    expert_after_low = queue_roundtrip["expert_after"]
    restored_experts = (
        int(expert_before_low["remote_expert_count"])
        - int(expert_after_low["remote_expert_count"])
    )
    require(
        restored_experts == 48,
        "KV shrink did not restore one expert per layer",
    )
    shrink_step = int(shrink["step"])
    require(
        shrink_step == int(queue_roundtrip["steps"][-1]["step"])
        and all(
            int(step["expert_after_step"]["remote_expert_count"])
            == int(expert_before_low["remote_expert_count"])
            for step in queue_roundtrip["steps"]
            if int(step["step"]) < shrink_step
        ),
        "experts were restored before the request-completion step",
    )
    require(
        int(expert_after_low["remote_expert_count"])
        == int(record["expert_before_request"]["remote_expert_count"]),
        "expert count did not return to the pre-pressure baseline",
    )
    before_low_layers = expert_before_low["layers"]
    after_low_layers = expert_after_low["layers"]
    initial_layers = record["expert_before_request"]["layers"]
    require(
        set(before_low_layers) == set(after_low_layers) == set(initial_layers),
        "expert layer sets differ across autoscale roundtrip",
    )
    for layer_key in sorted(before_low_layers):
        before_ids = set(before_low_layers[layer_key]["remote_expert_ids"])
        after_ids = set(after_low_layers[layer_key]["remote_expert_ids"])
        initial_ids = set(initial_layers[layer_key]["remote_expert_ids"])
        require(
            len(before_ids - after_ids) == 1
            and after_ids == initial_ids,
            f"{layer_key}: expert restore is not exactly one-per-layer",
        )

    restore_rpc_records = queue_roundtrip["rpc_records"]
    shrink_rpc = one_rpc(restore_rpc_records, "resize_uvm_kv_cache_target")
    restore_candidate_rpc = one_rpc(
        restore_rpc_records,
        "get_uvm_incremental_expert_restore_candidates",
    )
    restore_rpc = one_rpc(
        restore_rpc_records,
        "run_uvm_incremental_expert_restore",
    )
    require(
        shrink_rpc["result"].get("ok") is True
        and shrink_rpc["result"].get("made_progress") is True,
        "physical KV shrink RPC failed",
    )
    require(
        len(restore_candidate_rpc["result"].get("candidates") or {}) == 48,
        "restore candidate wave layer count",
    )
    require(
        restore_rpc["result"].get("ok") is True
        and int(restore_rpc["result"].get("restored_expert_count", -1)) == 48,
        "expert restore RPC failed",
    )
    restore_methods = [str(item["method"]) for item in restore_rpc_records]
    require(
        restore_methods.index("resize_uvm_kv_cache_target")
        < restore_methods.index("get_uvm_incremental_expert_restore_candidates")
        < restore_methods.index("run_uvm_incremental_expert_restore"),
        "expert restoration did not follow physical KV shrink",
    )
    cutover = queue.get("cutover_transition") or {}
    require(cutover.get("ok") is True, "remote Decode cutover failed")
    require(
        int(cutover.get("runtime_remote_range_count", 0)) > 0,
        "cutover did not receive runtime remote expert ranges",
    )
    require(
        int(cutover.get("runtime_remote_bytes", -1))
        == released_region_bytes,
        "cutover runtime remote bytes differ from the expert wave",
    )
    require(
        int(cutover.get("hot_runtime_remote_bytes", -1))
        == released_region_bytes,
        "runtime remote experts were not removed from the hot-GPU audit",
    )
    hot_physical = cutover.get("hot_physical_residency") or {}
    require(
        int(hot_physical.get("cpu_bytes", -1)) == released_region_bytes
        and int(hot_physical.get("gpu_bytes", -1))
        == int(cutover.get("hot_expected_gpu_bytes", -2)),
        "post-cutover hot/runtime-remote physical residency is incorrect",
    )
    rpc_records = record["uvm_rpc_records"]
    candidate_rpc = one_rpc(
        rpc_records, "get_uvm_incremental_expert_offload_candidates"
    )
    release_rpc = one_rpc(rpc_records, "run_uvm_incremental_expert_offload")
    resize_rpc = one_rpc(rpc_records, "resize_uvm_kv_cache_target")
    require(
        len(candidate_rpc["result"].get("candidates") or {}) == 48,
        "candidate wave layer count",
    )
    release_result = release_rpc["result"]
    require(release_result.get("ok") is True, "expert wave RPC failed")
    require(
        int(release_result.get("released_expert_count", -1)) == 48,
        "expert wave RPC release count",
    )
    require(
        resize_rpc["result"].get("ok") is True
        and resize_rpc["result"].get("made_progress") is True,
        "physical KV grow RPC failed",
    )
    methods = [str(item["method"]) for item in rpc_records]
    require(
        methods.index("get_uvm_incremental_expert_offload_candidates")
        < methods.index("run_uvm_incremental_expert_offload")
        < methods.index("resize_uvm_kv_cache_target"),
        "expert wave did not precede physical grow",
    )

    kernel_text = (root / "kernel.since.log").read_text(encoding="utf-8")
    require(BAD_KERNEL.search(kernel_text) is None, "kernel fault signature")
    for label in ("baseline", "three_queue"):
        log_path = root / label / "uvm.log"
        if log_path.is_file():
            require(
                BAD_KERNEL.search(log_path.read_text(encoding="utf-8")) is None,
                f"{label}: UVM fault signature",
            )
        state = load_json(root / label / "gpu.after.json")
        require(state.get("gpu_clean") is True, f"{label}: GPU not clean")
        require(not state.get("compute_pids"), f"{label}: compute clients remain")
        require(not state.get("uvm_client_pids"), f"{label}: UVM clients remain")
    module_after = load_json(root / "module.after.json")
    module_before = load_json(root / "module.before.json")
    require(module_after.get("gpu_clean") is True, "module restore GPU state")
    require(
        module_after.get("loaded_srcversion") == module_before.get("loaded_srcversion"),
        "stock nvidia_uvm was not restored",
    )

    verification = {
        "schema": "vllm_uvm_three_queue_e2e_verification_v1",
        "status": "ok",
        "model": queue["config"]["model"],
        "initial_target_blocks": initial_target,
        "backup_reserve_blocks": int(before["gpu_backup_blocks"]),
        "backup_consumed_blocks": consumed_blocks,
        "grown_target_blocks": grown_target,
        "grown_blocks": grow_blocks,
        "shrunken_target_blocks": int(queue_roundtrip["target_after_blocks"]),
        "offline_blocks_migrated_to_gpu": grow_blocks,
        "released_experts": released_experts,
        "restored_experts": restored_experts,
        "released_expert_region_bytes": released_region_bytes,
        "cutover_runtime_remote_bytes": int(
            cutover["runtime_remote_bytes"]
        ),
        "output_token_ids_sha256": queue["output_token_ids_sha256"],
        "autoscale_output_token_ids_sha256": queue_roundtrip[
            "output_token_ids_sha256"
        ],
        "cuda_graph_capture_count": int(
            queue["compilation_counter_final"]["num_cudagraph_captured"]
        ),
        "prefill_graph_replay_verified": True,
        "decode_graph_replay_verified": True,
        "prefill_decode_roundtrip_verified": True,
        "prefill_step_ms": float(queue["timing"]["prefill_step_ms"]),
        "cycle_e2e_ms": float(queue["timing"]["cycle_e2e_ms"]),
        "kernel_faults": 0,
        "shutdown_clean": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
