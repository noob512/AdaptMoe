# SPDX-License-Identifier: Apache-2.0
"""Application-side explicit UVM Region registration and placement."""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from vllm.device_allocator.uvm import (
    batch_prefetch_ranges_to_device_with_executor,
    driver_register_region,
    driver_release_regions,
    driver_update_region_state,
)
from vllm.device_allocator.uvm_region import (
    ExplicitRegionRegistry,
    KernelRegion,
    RegionState,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


VALID_STATES = {
    "unknown",
    "gpu",
    "cpu",
    "prefetching_gpu",
    "prefetching_cpu",
    "released",
}

DRIVER_REGION_STATES = {
    "unknown": 0,
    "cpu": 1,
    "released": 1,
    "prefetching_cpu": 2,
    "gpu": 3,
    "prefetching_gpu": 4,
}

_EXPLICIT_REGION_REGISTRY = ExplicitRegionRegistry()
_EXPLICIT_REGION_LOCK = threading.Lock()


def _normalize_state(state: str) -> str:
    state = (state or "unknown").strip().lower()
    return state if state in VALID_STATES else "unknown"


def _stable_driver_region_id(region_id: str) -> int:
    text = str(region_id or "")
    if not text:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    # Keep the key positive for radix-tree users and reserve 0 as invalid.
    return (int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF) or 1


def _driver_region_state_id(state: str) -> int:
    return DRIVER_REGION_STATES.get(_normalize_state(state), 0)


def register_uvm_region(
    region_id: str,
    base: int,
    length: int,
    *,
    state: str = "unknown",
) -> KernelRegion:
    """Register geometry with an UNKNOWN initial state in both tables."""
    with _EXPLICIT_REGION_LOCK:
        region = _EXPLICIT_REGION_REGISTRY.register_region(
            region_id,
            base,
            length,
            RegionState.UNKNOWN,
        )
        driver_region_id = _stable_driver_region_id(region.region_id)
        try:
            status = driver_register_region(
                region_id=driver_region_id,
                ptr=region.base,
                nbytes=region.length,
            )
        except Exception as exc:
            region.state = RegionState.UNKNOWN
            _EXPLICIT_REGION_REGISTRY.regions.pop(region.region_id, None)
            raise RuntimeError(
                "Explicit UVM driver region registration raised an exception: "
                f"region_id={region.region_id!r}, "
                f"driver_region_id={driver_region_id}, "
                f"base=0x{region.base:x}, length={region.length}"
            ) from exc

        # Some lightweight unit-test mocks are side-effect-only and return
        # None.  The production boundary always returns an integer status.
        status_code = 0 if status is None else int(status)
        if status_code < 0:
            region.state = RegionState.UNKNOWN
            # The application table must contain only driver-registered
            # regions.  Otherwise a later raw-range Hot prefetch could hide
            # this registration failure and incorrectly commit GPU state.
            _EXPLICIT_REGION_REGISTRY.regions.pop(region.region_id, None)
            raise RuntimeError(
                "Explicit UVM driver region registration failed: "
                f"region_id={region.region_id!r}, "
                f"driver_region_id={driver_region_id}, "
                f"base=0x{region.base:x}, length={region.length}, "
                f"status={status_code}"
            )

        # Kernel registration itself initializes the state to UNKNOWN. Keep
        # registration single-phase so a redundant update cannot leave the
        # application and driver tables disagreeing about initial placement.
        return region


def release_uvm_regions(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    remote_read: bool = False,
) -> dict[str, Any]:
    """Release only the explicitly named regions, with idempotent state checks."""
    requested = [str(region_id) for region_id in region_ids]
    statuses: dict[str, bool] = {}
    pending: list[KernelRegion] = []
    with _EXPLICIT_REGION_LOCK:
        for region_id in requested:
            region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            if region is None:
                statuses[region_id] = False
            elif region.state == RegionState.CPU and not remote_read:
                statuses[region_id] = True
            else:
                # UNKNOWN is not a proof that the range is already on either
                # device. CPU placement also does not prove that remote-read
                # mappings exist. Submit the conservative operation and only
                # commit CPU after the driver reports success.
                region.state = RegionState.PREFETCHING_CPU
                pending.append(region)

    if pending:
        numeric_to_region = {
            _stable_driver_region_id(region.region_id): region for region in pending
        }
        try:
            result = driver_release_regions(
                list(numeric_to_region),
                device=device,
                remote_read=remote_read,
            )
        except Exception:
            logger.warning("Failed to release explicit UVM regions", exc_info=True)
            result = {"ok": False, "statuses": {}}
        driver_statuses = result.get("statuses")
        if not isinstance(driver_statuses, dict):
            driver_statuses = {}
        with _EXPLICIT_REGION_LOCK:
            for numeric_id, region in numeric_to_region.items():
                ok = bool(driver_statuses.get(numeric_id, False))
                region.state = RegionState.CPU if ok else RegionState.UNKNOWN
                statuses[region.region_id] = ok
                driver_update_region_state(
                    region_id=numeric_id,
                    state=_driver_region_state_id(region.state.value),
                )

    return {
        "ok": all(statuses.get(region_id, False) for region_id in requested),
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses.get(region_id, False)
        ],
    }


def restore_uvm_regions(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    executor: str = "auto",
) -> dict[str, Any]:
    """Restore only the explicitly named regions using their registered ranges."""
    requested = [str(region_id) for region_id in region_ids]
    statuses: dict[str, bool] = {}
    pending: list[KernelRegion] = []
    with _EXPLICIT_REGION_LOCK:
        for region_id in requested:
            region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            if region is None:
                statuses[region_id] = False
            elif region.state == RegionState.GPU:
                statuses[region_id] = True
            else:
                # UNKNOWN must be restored conservatively instead of being
                # treated as an implicit failure or an already-resident range.
                region.state = RegionState.PREFETCHING_GPU
                pending.append(region)

    if pending:
        ranges = [(region.base, region.length) for region in pending]
        try:
            successes, _, _ = batch_prefetch_ranges_to_device_with_executor(
                ranges,
                device,
                executor=executor,
            )
        except Exception:
            logger.warning("Failed to restore explicit UVM regions", exc_info=True)
            successes = 0
        complete = int(successes) == len(pending)
        with _EXPLICIT_REGION_LOCK:
            if complete:
                complete = _set_uvm_region_state_locked(
                    pending,
                    RegionState.GPU,
                )
            else:
                _set_uvm_region_state_locked(pending, RegionState.UNKNOWN)
            for region in pending:
                statuses[region.region_id] = complete

    return {
        "ok": all(statuses.get(region_id, False) for region_id in requested),
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses.get(region_id, False)
        ],
    }


def _merge_overlapping_uvm_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping positive ranges without crossing allocation borders.

    Two independent ``cudaMallocManaged`` allocations can be virtually
    adjacent.  CUDA rejects one prefetch spanning that shared boundary, so
    touching ranges must remain separate unless allocation identity is known.
    """
    normalized = sorted(
        (int(base), int(length))
        for base, length in ranges
        if int(base) > 0 and int(length) > 0
    )
    if not normalized:
        return []

    merged: list[tuple[int, int]] = []
    current_base, current_length = normalized[0]
    current_end = current_base + current_length
    for base, length in normalized[1:]:
        end = base + length
        if base < current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_base, current_end - current_base))
        current_base = base
        current_end = end
    merged.append((current_base, current_end - current_base))
    return merged


def _set_uvm_region_state_locked(
    regions: list[KernelRegion],
    state: RegionState,
) -> bool:
    update_ok = True
    for region in regions:
        try:
            result = driver_update_region_state(
                region_id=_stable_driver_region_id(region.region_id),
                state=_driver_region_state_id(state.value),
            )
            status = 0 if result is None else int(result)
        except Exception:
            logger.warning(
                "Explicit UVM driver state update raised: region_id=%s state=%s",
                region.region_id,
                state.value,
                exc_info=True,
            )
            update_ok = False
            break
        if status < 0:
            logger.warning(
                "Explicit UVM driver state update failed: "
                "region_id=%s state=%s status=%s",
                region.region_id,
                state.value,
                status,
            )
            update_ok = False
            break

    if update_ok:
        for region in regions:
            region.state = state
        return True

    # A partially updated batch cannot support a placement claim. Degrade the
    # complete application batch first, then converge every driver entry to
    # UNKNOWN on a best-effort basis.
    for region in regions:
        region.state = RegionState.UNKNOWN
    for region in regions:
        try:
            driver_update_region_state(
                region_id=_stable_driver_region_id(region.region_id),
                state=_driver_region_state_id(RegionState.UNKNOWN.value),
            )
        except Exception:
            logger.debug(
                "Failed to roll explicit UVM driver state back to UNKNOWN: %s",
                region.region_id,
                exc_info=True,
            )
    return False


def prepare_uvm_regions_for_gpu_placement(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    executor: str = "auto",
) -> dict[str, Any]:
    """Submit one coalesced GPU prefetch batch without committing GPU state.

    A successful return leaves every named region in ``PREFETCHING_GPU``.
    The caller must synchronize CUDA and then call
    :func:`commit_uvm_regions_gpu_placement`.  Any validation, submission, or
    partial-batch failure atomically degrades all present regions to UNKNOWN.
    """
    requested = list(dict.fromkeys(str(region_id) for region_id in region_ids))
    if not requested:
        return {"ok": True, "reason": "no_regions"}

    with _EXPLICIT_REGION_LOCK:
        regions = [
            region
            for region_id in requested
            if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id)) is not None
        ]
        missing = [
            region_id
            for region_id in requested
            if region_id not in _EXPLICIT_REGION_REGISTRY.regions
        ]
        if missing:
            _set_uvm_region_state_locked(regions, RegionState.UNKNOWN)
            return {"ok": False, "reason": "unregistered_regions"}
        ranges = _merge_overlapping_uvm_ranges(
            [(region.base, region.length) for region in regions]
        )
        if not ranges:
            _set_uvm_region_state_locked(regions, RegionState.UNKNOWN)
            return {"ok": False, "reason": "no_valid_ranges"}
        if not _set_uvm_region_state_locked(
            regions,
            RegionState.PREFETCHING_GPU,
        ):
            return {"ok": False, "reason": "driver_state_update_failed"}

    try:
        successes, _, _ = (
            batch_prefetch_ranges_to_device_with_executor(
                ranges,
                device,
                executor=executor,
            )
        )
    except Exception:
        logger.warning("Failed to submit final Hot Expert GPU placement", exc_info=True)
        successes = 0

    complete = int(successes) == len(ranges)
    if not complete:
        with _EXPLICIT_REGION_LOCK:
            current = [
                region
                for region_id in requested
                if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id))
                is not None
            ]
            _set_uvm_region_state_locked(current, RegionState.UNKNOWN)

    return {
        "ok": complete,
        "reason": "prefetch_submitted" if complete else "prefetch_incomplete",
    }


def commit_uvm_regions_gpu_placement(
    region_ids: list[str] | tuple[str, ...],
    *,
    synchronized: bool,
) -> dict[str, Any]:
    """Commit GPU only after the caller's single CUDA synchronization."""
    requested = list(dict.fromkeys(str(region_id) for region_id in region_ids))
    with _EXPLICIT_REGION_LOCK:
        regions = [
            region
            for region_id in requested
            if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id)) is not None
        ]
        missing = [
            region_id
            for region_id in requested
            if region_id not in _EXPLICIT_REGION_REGISTRY.regions
        ]
        ready = (
            bool(requested)
            and not missing
            and all(region.state == RegionState.PREFETCHING_GPU for region in regions)
        )
        committed = bool(synchronized) and ready
        reason = "placement_not_synchronized"
        if committed:
            committed = _set_uvm_region_state_locked(
                regions,
                RegionState.GPU,
            )
            reason = "committed" if committed else "driver_state_update_failed"
        else:
            _set_uvm_region_state_locked(regions, RegionState.UNKNOWN)

    return {"ok": committed, "reason": reason}


def query_uvm_region_states(
    region_ids: list[str] | tuple[str, ...],
) -> dict[str, str]:
    with _EXPLICIT_REGION_LOCK:
        return {
            region_id: region.state.value
            for region_id in (str(value) for value in region_ids)
            if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id)) is not None
        }
