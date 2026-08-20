# SPDX-License-Identifier: Apache-2.0
"""Userspace wrapper for the experimental UVM static-alias UAPI.

The static-alias commands are initialization and teardown operations. They are
never issued during CUDA Graph capture or replay. A caller maps one or more
original managed VAs, seals the immutable session, captures Graphs which use
those VAs, and destroys the session only after all such Graphs and CUDA work
have been destroyed or synchronized.

This module deliberately owns state independently from ``uvm.py``'s dynamic
cmd96 slot-transition session. It reuses only that module's low-level ioctl,
GPU-UUID, and initialized-UVM-fd discovery helpers so MAP targets the UVM VA
space which owns the submitted managed addresses.
"""

from __future__ import annotations

import atexit
import ctypes
import dataclasses
import os
import threading
import time
from contextlib import suppress
from typing import Any, Sequence

import torch

from vllm.device_allocator import uvm as _uvm

UVM_MOE_STATIC_ALIAS_ABI_VERSION = 1
UVM_MOE_STATIC_ALIAS_MAX_ENTRIES = 1152

_UVM_MOE_STATIC_ALIAS_MAP_IOCTL = 97
_UVM_MOE_STATIC_ALIAS_SEAL_IOCTL = 98
_UVM_MOE_STATIC_ALIAS_DESTROY_IOCTL = 99

UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW = 1 << 0
UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K = 1 << 1
_UVM_MOE_STATIC_ALIAS_ENTRY_FLAGS_ALL = (
    UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
    | UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K
)
UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING = 1 << 0

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_PAGE_BYTES = 4096
_PTE_64K_BYTES = 64 * 1024

# Keep these aliases patchable by CPU-only tests. They are the only coupling to
# the older wrapper and do not share any cmd96 owner, plan, or generation state.
_libc_ioctl = _uvm._libc_ioctl
_phase_f_gpu_uuid = _uvm._phase_f_gpu_uuid
_iter_existing_uvm_fd_sources = _uvm._iter_existing_uvm_fd_sources
_initialize_duplicated_uvm_fd = _uvm._initialize_duplicated_uvm_fd


class _StaticAliasUuid(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)]


class _StaticAliasEntry(ctypes.Structure):
    _fields_ = [
        ("backing_layer_id", ctypes.c_uint32),
        ("backing_expert_id", ctypes.c_uint32),
        ("backing_slot_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("va_base", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("slot_offset", ctypes.c_uint64),
        ("status", ctypes.c_uint32),
    ]


class _StaticAliasMapParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("gpu_uuid", _StaticAliasUuid),
        ("requested_bytes", ctypes.c_uint64),
        ("mapped_bytes", ctypes.c_uint64),
        ("total_alias_bytes", ctypes.c_uint64),
        ("backing_bytes", ctypes.c_uint64),
        ("count", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        (
            "entries",
            _StaticAliasEntry * UVM_MOE_STATIC_ALIAS_MAX_ENTRIES,
        ),
        ("rmStatus", ctypes.c_uint32),
    ]


class _StaticAliasSealParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("expected_alias_bytes", ctypes.c_uint64),
        ("alias_bytes", ctypes.c_uint64),
        ("backing_bytes", ctypes.c_uint64),
        ("expected_alias_count", ctypes.c_uint32),
        ("sealed", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _StaticAliasDestroyParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("released_alias_bytes", ctypes.c_uint64),
        ("remaining_alias_bytes", ctypes.c_uint64),
        ("released_backing_bytes", ctypes.c_uint64),
        ("remaining_backing_bytes", ctypes.c_uint64),
        ("released_alias_count", ctypes.c_uint32),
        ("remaining_alias_count", ctypes.c_uint32),
        ("session_cleared", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


def _assert_static_alias_abi() -> None:
    """Fail before an ioctl if the hand-written ctypes layout drifts."""
    expected = {
        "entry_size": (ctypes.sizeof(_StaticAliasEntry), 48),
        "entry_va_base": (_StaticAliasEntry.va_base.offset, 16),
        "entry_status": (_StaticAliasEntry.status.offset, 40),
        "map_size": (ctypes.sizeof(_StaticAliasMapParams), 55400),
        "map_session_id": (_StaticAliasMapParams.session_id.offset, 16),
        "map_requested_bytes": (
            _StaticAliasMapParams.requested_bytes.offset,
            40,
        ),
        "map_count": (_StaticAliasMapParams.count.offset, 72),
        "map_entries": (_StaticAliasMapParams.entries.offset, 96),
        "map_rm_status": (_StaticAliasMapParams.rmStatus.offset, 55392),
        "seal_size": (ctypes.sizeof(_StaticAliasSealParams), 72),
        "seal_session_id": (_StaticAliasSealParams.session_id.offset, 16),
        "seal_expected_count": (
            _StaticAliasSealParams.expected_alias_count.offset,
            48,
        ),
        "seal_rm_status": (_StaticAliasSealParams.rmStatus.offset, 64),
        "destroy_size": (ctypes.sizeof(_StaticAliasDestroyParams), 80),
        "destroy_session_id": (
            _StaticAliasDestroyParams.session_id.offset,
            16,
        ),
        "destroy_released_aliases": (
            _StaticAliasDestroyParams.released_alias_count.offset,
            56,
        ),
        "destroy_rm_status": (
            _StaticAliasDestroyParams.rmStatus.offset,
            72,
        ),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError(
            "UVM MoE static-alias ioctl ABI mismatch: " + ", ".join(mismatches)
        )


_assert_static_alias_abi()


@dataclasses.dataclass
class _StaticAliasSession:
    fd: int
    uuid_bytes: bytes
    session_id: int
    owner_pid: int
    sealed: bool = False
    poisoned: bool = False
    alias_count: int = 0
    alias_bytes: int = 0
    backing_bytes: int = 0
    target_ranges: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    backing_identities: set[tuple[int, int, int]] = dataclasses.field(
        default_factory=set
    )


_static_alias_sessions: dict[int, _StaticAliasSession] = {}
_static_alias_lock = threading.RLock()
_static_alias_close_registered = False


def _capture_rejection() -> tuple[str, str] | None:
    try:
        capturing = bool(torch.cuda.is_current_stream_capturing())
    except Exception as exc:
        return ("cuda_graph_capture_query_failed", repr(exc))
    if capturing:
        return (
            "cuda_graph_capture_unsupported",
            "static-alias ioctls are initialization/teardown operations",
        )
    return None


def _normalize_device(device: int) -> tuple[int, str]:
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        return 0, "device must be a non-negative integer"
    return int(device), ""


def _normalize_entries(
    entries: Sequence[Sequence[int]],
) -> tuple[list[tuple[int, int, int, int, int, int, int]], str]:
    if isinstance(entries, (str, bytes)):
        return [], "entries must be a sequence of seven-integer records"
    try:
        records = list(entries)
    except TypeError:
        return [], "entries must be a sequence"
    if not records:
        return [], "at least one alias entry is required"
    if len(records) > UVM_MOE_STATIC_ALIAS_MAX_ENTRIES:
        return [], "one MAP exceeds the 1152-entry ABI"

    normalized: list[tuple[int, int, int, int, int, int, int]] = []
    target_ranges: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        if isinstance(record, (str, bytes)):
            return [], f"entry {index} is not a seven-integer record"
        try:
            values = tuple(record)
        except TypeError:
            return [], f"entry {index} is not a sequence"
        if len(values) != 7:
            return [], f"entry {index} must contain seven integers"
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            return [], f"entry {index} contains a non-integer field"
        (
            backing_layer_id,
            backing_expert_id,
            backing_slot_id,
            flags,
            va_base,
            length,
            slot_offset,
        ) = (int(value) for value in values)
        if any(
            value < 0 or value > _UINT32_MAX
            for value in (
                backing_layer_id,
                backing_expert_id,
                backing_slot_id,
                flags,
            )
        ):
            return [], f"entry {index} contains a uint32 overflow"
        if flags & ~_UVM_MOE_STATIC_ALIAS_ENTRY_FLAGS_ALL:
            return [], f"entry {index} contains unknown flags"
        if va_base <= 0 or length <= 0 or slot_offset < 0:
            return [], f"entry {index} has an invalid VA, length, or offset"
        if any(value > _UINT64_MAX for value in (va_base, length, slot_offset)):
            return [], f"entry {index} contains a uint64 overflow"
        if va_base > _UINT64_MAX - length or slot_offset > _UINT64_MAX - length:
            return [], f"entry {index} range overflows uint64"
        alignment = (
            _PTE_64K_BYTES
            if flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K
            else _PAGE_BYTES
        )
        if any(value % alignment for value in (va_base, length, slot_offset)):
            return [], f"entry {index} is not {alignment}-byte aligned"
        end = va_base + length
        target_ranges.append((va_base, end))
        normalized.append(
            (
                backing_layer_id,
                backing_expert_id,
                backing_slot_id,
                flags,
                va_base,
                length,
                slot_offset,
            )
        )

    ordered = sorted(target_ranges)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        return [], "target alias VA ranges overlap"
    pte_modes = {
        bool(record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K)
        for record in normalized
    }
    if len(pte_modes) != 1:
        return [], "one MAP cannot mix 4K and 64K PTE entries"
    return normalized, ""


def _make_uuid(device: int) -> tuple[_StaticAliasUuid | None, bytes, str]:
    try:
        source = _phase_f_gpu_uuid(int(device))
        uuid_bytes = bytes(source.uuid)
    except Exception as exc:
        return None, b"", repr(exc)
    if len(uuid_bytes) != 16:
        return None, b"", f"unexpected GPU UUID size {len(uuid_bytes)}"
    value = _StaticAliasUuid()
    value.uuid[:] = uuid_bytes
    return value, uuid_bytes, ""


def _base_result(operation: str, **values: Any) -> dict[str, object]:
    return {
        "ok": False,
        "operation": operation,
        "abi_version": UVM_MOE_STATIC_ALIAS_ABI_VERSION,
        **values,
    }


def _session_block_reason(
    session: _StaticAliasSession,
    *,
    allow_poisoned: bool = False,
) -> tuple[str, str] | None:
    if session.owner_pid != os.getpid():
        return (
            "fork_inherited_static_alias_session",
            f"session belongs to pid {session.owner_pid}, current pid is {os.getpid()}",
        )
    if session.poisoned and not allow_poisoned:
        return ("static_alias_session_poisoned", "session is destroy-only")
    return None


def _ranges_overlap_existing(
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    existing: Sequence[tuple[int, int]],
) -> bool:
    for record in normalized:
        start = int(record[4])
        end = start + int(record[5])
        if any(start < old_end and old_start < end for old_start, old_end in existing):
            return True
    return False


def _fill_map_params(
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    *,
    session_id: int,
    gpu_uuid: _StaticAliasUuid,
    flags: int,
) -> _StaticAliasMapParams:
    params = _StaticAliasMapParams()
    params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
    params.struct_size = ctypes.sizeof(_StaticAliasMapParams)
    params.flags = int(flags)
    params.session_id = int(session_id)
    params.gpu_uuid = gpu_uuid
    params.count = len(normalized)
    for index, record in enumerate(normalized):
        (
            backing_layer_id,
            backing_expert_id,
            backing_slot_id,
            entry_flags,
            va_base,
            length,
            slot_offset,
        ) = record
        entry = params.entries[index]
        entry.backing_layer_id = backing_layer_id
        entry.backing_expert_id = backing_expert_id
        entry.backing_slot_id = backing_slot_id
        entry.flags = entry_flags
        entry.va_base = va_base
        entry.bytes = length
        entry.slot_offset = slot_offset
    return params


def _submit_map(
    fd: int,
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    *,
    session_id: int,
    gpu_uuid: _StaticAliasUuid,
    flags: int,
    prior_alias_count: int,
    prior_alias_bytes: int,
) -> dict[str, object]:
    params = _fill_map_params(
        normalized,
        session_id=session_id,
        gpu_uuid=gpu_uuid,
        flags=flags,
    )
    expected_bytes = sum(record[5] for record in normalized)
    started_ns = time.perf_counter_ns()
    ctypes.set_errno(0)
    try:
        ret = int(_libc_ioctl(fd, _UVM_MOE_STATIC_ALIAS_MAP_IOCTL, params))
    except Exception as exc:
        return _base_result(
            "map",
            reason="map_ioctl_exception",
            error=repr(exc),
            requested_session_id=int(session_id),
            requested_count=len(normalized),
            expected_requested_bytes=expected_bytes,
            ioctl_wall_ns=time.perf_counter_ns() - started_ns,
            side_effect_indeterminate=True,
            made_progress=True,
            poisoned=True,
        )
    wall_ns = time.perf_counter_ns() - started_ns
    errno_value = ctypes.get_errno() if ret != 0 else 0
    returned_session_id = int(params.session_id)
    statuses = [int(params.entries[index].status) for index in range(len(normalized))]
    poisoned_raw = int(params.poisoned)
    boolean_fields_ok = poisoned_raw in (0, 1)
    expected_total_alias_bytes = prior_alias_bytes + expected_bytes
    expected_total_alias_count = prior_alias_count + len(normalized)
    returned_alias_count = int(params.alias_count)
    returned_total_alias_count = prior_alias_count + returned_alias_count
    success_contract_ok = (
        ret == 0
        and int(params.rmStatus) == 0
        and boolean_fields_ok
        and poisoned_raw == 0
        and returned_session_id > 0
        and (session_id == 0 or returned_session_id == session_id)
        and int(params.requested_bytes) == expected_bytes
        and int(params.mapped_bytes) == expected_bytes
        and int(params.total_alias_bytes) == expected_total_alias_bytes
        and int(params.backing_bytes) > 0
        and int(params.success_count) == len(normalized)
        and int(params.failure_count) == 0
        and int(params.alias_count) == len(normalized)
        and statuses == [0] * len(normalized)
    )
    # A failed probe may come from an initialized UVM file description which
    # does not own the CUDA VA space. A bare returned session id has no PTE or
    # backing side effect and must not stop candidate probing. Any mutation is
    # represented by one of the counters below or by the poisoned bit.
    made_progress = (
        int(params.mapped_bytes) != 0
        or int(params.total_alias_bytes) != 0
        or int(params.backing_bytes) != 0
        or int(params.success_count) != 0
        or int(params.alias_count) != 0
        or int(params.tlb_invalidate_count) != 0
        or poisoned_raw != 0
    )
    ioctl_completed = ret == 0 and int(params.rmStatus) == 0
    contract_violation = ioctl_completed and not success_contract_ok
    return _base_result(
        "map",
        ok=success_contract_ok,
        reason=(
            "ok"
            if success_contract_ok
            else "map_contract_mismatch"
            if contract_violation
            else "map_ioctl_failed"
        ),
        requested_session_id=int(session_id),
        session_id=returned_session_id,
        requested_count=len(normalized),
        requested_bytes=int(params.requested_bytes),
        expected_requested_bytes=expected_bytes,
        mapped_bytes=int(params.mapped_bytes),
        total_alias_bytes=int(params.total_alias_bytes),
        backing_bytes=int(params.backing_bytes),
        success_count=int(params.success_count),
        failure_count=int(params.failure_count),
        alias_count=returned_alias_count,
        # The v1 UAPI reports alias_count for this MAP. Combining it with the
        # cached pre-call count also describes a failed MAP which committed
        # its complete topology before a later PTE operation failed.
        total_alias_count=returned_total_alias_count,
        tlb_invalidate_count=int(params.tlb_invalidate_count),
        entry_statuses=statuses,
        poisoned=(poisoned_raw == 1 or contract_violation),
        kernel_poisoned=poisoned_raw == 1,
        boolean_fields_ok=boolean_fields_ok,
        success_contract_ok=success_contract_ok,
        contract_violation=contract_violation,
        made_progress=made_progress,
        side_effect_indeterminate=False,
        ioctl_ret=ret,
        errno=errno_value,
        rm_status=int(params.rmStatus),
        ioctl_calls=1,
        ioctl_wall_ns=wall_ns,
    )


def _sync_cached_map_result(
    cached: _StaticAliasSession,
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    result: dict[str, object],
) -> None:
    """Mirror kernel-visible MAP progress into the destroy contract cache."""
    alias_count = int(result.get("alias_count") or 0)
    total_alias_count = result.get("total_alias_count")
    if total_alias_count is None:
        total_alias_count = cached.alias_count + alias_count

    cached.alias_count = int(total_alias_count)
    if "total_alias_bytes" in result:
        cached.alias_bytes = int(result["total_alias_bytes"])
    if "backing_bytes" in result:
        cached.backing_bytes = int(result["backing_bytes"])

    # MAP v1 commits a whole request topology at once. Only claim its ranges
    # and backing identities when the kernel confirms the full batch count.
    # An indeterminate/partial result remains destroy-only without inventing
    # which subset became active.
    if alias_count == len(normalized):
        cached.target_ranges.extend(
            (record[4], record[4] + record[5]) for record in normalized
        )
        cached.backing_identities.update(
            (record[0], record[1], record[2]) for record in normalized
        )


def _register_close_once() -> None:
    global _static_alias_close_registered
    if not _static_alias_close_registered:
        atexit.register(_close_static_alias_sessions)
        _static_alias_close_registered = True


def driver_static_alias_map(
    entries: Sequence[Sequence[int]],
    *,
    session_id: int = 0,
    device: int = 0,
    flags: int = 0,
) -> dict[str, object]:
    """Create or append to one unsealed static-alias session."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    normalized, validation_error = _normalize_entries(entries)
    if (
        not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id < 0
        or session_id > _UINT64_MAX
    ):
        validation_error = "session_id is outside uint64"
    if (
        not isinstance(flags, int)
        or isinstance(flags, bool)
        or flags != 0
    ):
        validation_error = "MAP flags must be zero for ABI v1"
    if device_error:
        validation_error = device_error
    if validation_error:
        return _base_result(
            "map",
            reason="validation_failed",
            error=validation_error,
            session_id=int(session_id) if type(session_id) is int else session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "map",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    gpu_uuid, uuid_bytes, uuid_error = _make_uuid(normalized_device)
    if gpu_uuid is None:
        return _base_result(
            "map",
            reason="gpu_uuid_failed",
            error=uuid_error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if session_id != 0:
            if cached is None or cached.session_id != int(session_id):
                return _base_result(
                    "map",
                    reason="static_alias_session_not_found",
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            blocked = _session_block_reason(cached)
            if blocked is not None:
                reason, error = blocked
                return _base_result(
                    "map",
                    reason=reason,
                    error=error,
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            if cached.sealed:
                return _base_result(
                    "map",
                    reason="static_alias_session_sealed",
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            if cached.uuid_bytes != uuid_bytes:
                return _base_result(
                    "map",
                    reason="cached_gpu_uuid_mismatch",
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            if _ranges_overlap_existing(normalized, cached.target_ranges):
                return _base_result(
                    "map",
                    reason="validation_failed",
                    error="target alias VA overlaps the existing session",
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            result = _submit_map(
                cached.fd,
                normalized,
                session_id=int(session_id),
                gpu_uuid=gpu_uuid,
                flags=int(flags),
                prior_alias_count=cached.alias_count,
                prior_alias_bytes=cached.alias_bytes,
            )
            if bool(result["ok"]) or bool(result.get("made_progress")):
                _sync_cached_map_result(cached, normalized, result)
            if not bool(result["ok"]) and (
                bool(result.get("made_progress")) or bool(result.get("poisoned"))
            ):
                cached.poisoned = True
            result["wall_ns"] = time.perf_counter_ns() - started_ns
            return result

        if cached is not None:
            return _base_result(
                "map",
                reason="static_alias_session_already_active",
                session_id=cached.session_id,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        sources = _iter_existing_uvm_fd_sources()
        # Slot backing is currently created through uvm.py's mode-8 command.
        # Prefer that command's retained file description so MAP targets the
        # exact same UVM VA space. Keep generic discovery as a fallback for
        # callers which arrange backing without the Python slot wrapper.
        legacy_cached = _uvm._uvm_slot_direct_sessions.get(normalized_device)
        if legacy_cached is not None:
            preferred_fd = int(legacy_cached[0])
            sources = [preferred_fd] + [fd for fd in sources if fd != preferred_fd]
        if not sources:
            return _base_result(
                "map",
                reason="no_existing_uvm_fd",
                session_id=0,
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        last_result: dict[str, object] | None = None
        candidate_attempts = 0
        candidate_ioctl_calls = 0
        for source_fd in sources:
            fd = _initialize_duplicated_uvm_fd(source_fd)
            if fd is None:
                continue
            candidate_attempts += 1
            result = _submit_map(
                fd,
                normalized,
                session_id=0,
                gpu_uuid=gpu_uuid,
                flags=int(flags),
                prior_alias_count=0,
                prior_alias_bytes=0,
            )
            candidate_ioctl_calls += int(result.get("ioctl_calls", 0))
            if bool(result["ok"]) or bool(result.get("made_progress")):
                returned_session_id = int(result.get("session_id") or 0)
                cached = _StaticAliasSession(
                    fd=fd,
                    uuid_bytes=uuid_bytes,
                    session_id=returned_session_id,
                    owner_pid=os.getpid(),
                    poisoned=not bool(result["ok"]),
                )
                _sync_cached_map_result(cached, normalized, result)
                _static_alias_sessions[normalized_device] = cached
                _register_close_once()
                result["fd_candidate_attempts"] = candidate_attempts
                result["fd_candidate_ioctl_calls"] = candidate_ioctl_calls
                result["wall_ns"] = time.perf_counter_ns() - started_ns
                return result
            last_result = result
            with suppress(OSError):
                os.close(fd)

        if last_result is None:
            last_result = _base_result(
                "map",
                reason="no_initialized_uvm_fd",
                session_id=0,
            )
        else:
            last_result["reason"] = "no_matching_existing_uvm_fd"
        last_result["fd_candidate_attempts"] = candidate_attempts
        last_result["fd_candidate_ioctl_calls"] = candidate_ioctl_calls
        last_result["wall_ns"] = time.perf_counter_ns() - started_ns
        return last_result


def driver_static_alias_seal(
    session_id: int,
    *,
    expected_alias_count: int | None = None,
    expected_alias_bytes: int | None = None,
    device: int = 0,
) -> dict[str, object]:
    """Seal an immutable session before any CUDA Graph captures its VAs."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    if (
        device_error
        or not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id <= 0
        or session_id > _UINT64_MAX
    ):
        return _base_result(
            "seal",
            reason="validation_failed",
            error=device_error or "session_id must be a non-zero uint64",
            session_id=session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "seal",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if cached is None or cached.session_id != int(session_id):
            return _base_result(
                "seal",
                reason="static_alias_session_not_found",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        blocked = _session_block_reason(cached)
        if blocked is not None:
            reason, error = blocked
            return _base_result(
                "seal",
                reason=reason,
                error=error,
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        if cached.sealed:
            return _base_result(
                "seal",
                reason="static_alias_session_already_sealed",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        expected_count = (
            cached.alias_count
            if expected_alias_count is None
            else int(expected_alias_count)
        )
        expected_bytes = (
            cached.alias_bytes
            if expected_alias_bytes is None
            else int(expected_alias_bytes)
        )
        if expected_count != cached.alias_count or expected_bytes != cached.alias_bytes:
            return _base_result(
                "seal",
                reason="validation_failed",
                error="seal expectations disagree with the userspace session",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        params = _StaticAliasSealParams()
        params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
        params.struct_size = ctypes.sizeof(_StaticAliasSealParams)
        params.session_id = int(session_id)
        params.expected_alias_count = expected_count
        params.expected_alias_bytes = expected_bytes
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = int(_libc_ioctl(cached.fd, _UVM_MOE_STATIC_ALIAS_SEAL_IOCTL, params))
        except Exception as exc:
            cached.poisoned = True
            return _base_result(
                "seal",
                reason="seal_ioctl_exception",
                error=repr(exc),
                session_id=int(session_id),
                poisoned=True,
                side_effect_indeterminate=True,
                ioctl_wall_ns=time.perf_counter_ns() - ioctl_started_ns,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0
        sealed_raw = int(params.sealed)
        poisoned_raw = int(params.poisoned)
        boolean_fields_ok = sealed_raw in (0, 1) and poisoned_raw in (0, 1)
        ok = (
            ret == 0
            and int(params.rmStatus) == 0
            and boolean_fields_ok
            and sealed_raw == 1
            and poisoned_raw == 0
            and int(params.alias_count) == expected_count
            and int(params.alias_bytes) == expected_bytes
            and int(params.backing_bytes) == cached.backing_bytes
        )
        contract_violation = ret == 0 and int(params.rmStatus) == 0 and not ok
        if ok:
            cached.sealed = True
        elif poisoned_raw == 1 or contract_violation:
            cached.poisoned = True
        return _base_result(
            "seal",
            ok=ok,
            reason=(
                "ok"
                if ok
                else "seal_contract_mismatch"
                if contract_violation
                else "seal_ioctl_failed"
            ),
            session_id=int(session_id),
            sealed=sealed_raw == 1,
            alias_count=int(params.alias_count),
            alias_bytes=int(params.alias_bytes),
            backing_bytes=int(params.backing_bytes),
            poisoned=cached.poisoned,
            kernel_poisoned=poisoned_raw == 1,
            boolean_fields_ok=boolean_fields_ok,
            contract_violation=contract_violation,
            ioctl_ret=ret,
            errno=errno_value,
            rm_status=int(params.rmStatus),
            ioctl_calls=1,
            ioctl_wall_ns=ioctl_wall_ns,
            wall_ns=time.perf_counter_ns() - started_ns,
        )


def driver_static_alias_destroy(
    session_id: int,
    *,
    device: int = 0,
    release_backing: bool = True,
    synchronize: bool = False,
) -> dict[str, object]:
    """Destroy aliases after Graph teardown and an optional device barrier."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    if (
        device_error
        or not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id <= 0
        or session_id > _UINT64_MAX
        or not isinstance(release_backing, bool)
        or not isinstance(synchronize, bool)
    ):
        return _base_result(
            "destroy",
            reason="validation_failed",
            error=device_error or "invalid destroy argument",
            session_id=session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "destroy",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    if synchronize:
        try:
            torch.cuda.synchronize(normalized_device)
        except Exception as exc:
            return _base_result(
                "destroy",
                reason="cuda_synchronize_failed",
                error=repr(exc),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if cached is None or cached.session_id != int(session_id):
            return _base_result(
                "destroy",
                reason="static_alias_session_not_found",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        blocked = _session_block_reason(cached, allow_poisoned=True)
        if blocked is not None:
            reason, error = blocked
            return _base_result(
                "destroy",
                reason=reason,
                error=error,
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        params = _StaticAliasDestroyParams()
        params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
        params.struct_size = ctypes.sizeof(_StaticAliasDestroyParams)
        params.flags = (
            UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING
            if release_backing
            else 0
        )
        params.session_id = int(session_id)
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = int(
                _libc_ioctl(cached.fd, _UVM_MOE_STATIC_ALIAS_DESTROY_IOCTL, params)
            )
        except Exception as exc:
            cached.poisoned = True
            return _base_result(
                "destroy",
                reason="destroy_ioctl_exception",
                error=repr(exc),
                session_id=int(session_id),
                poisoned=True,
                side_effect_indeterminate=True,
                ioctl_wall_ns=time.perf_counter_ns() - ioctl_started_ns,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0
        cleared_raw = int(params.session_cleared)
        poisoned_raw = int(params.poisoned)
        boolean_fields_ok = cleared_raw in (0, 1) and poisoned_raw in (0, 1)
        aliases_removed = (
            int(params.released_alias_count) == cached.alias_count
            and int(params.released_alias_bytes) == cached.alias_bytes
            and int(params.remaining_alias_count) == 0
            and int(params.remaining_alias_bytes) == 0
        )
        if release_backing:
            backing_contract_ok = (
                int(params.released_backing_bytes) == cached.backing_bytes
                and int(params.remaining_backing_bytes) == 0
                and cleared_raw == 1
            )
        else:
            backing_contract_ok = (
                int(params.released_backing_bytes) == 0
                and int(params.remaining_backing_bytes) == cached.backing_bytes
                and cleared_raw == 0
            )
        kernel_session_cleared = (
            ret == 0
            and int(params.rmStatus) == 0
            and cleared_raw == 1
            and int(params.remaining_alias_count) == 0
            and int(params.remaining_alias_bytes) == 0
            and int(params.remaining_backing_bytes) == 0
        )
        ok = (
            ret == 0
            and int(params.rmStatus) == 0
            and boolean_fields_ok
            and poisoned_raw == 0
            and aliases_removed
            and backing_contract_ok
        )
        contract_violation = ret == 0 and int(params.rmStatus) == 0 and not ok
        result = _base_result(
            "destroy",
            ok=ok,
            reason=(
                "ok"
                if ok
                else "destroy_contract_mismatch"
                if contract_violation
                else "destroy_ioctl_failed"
            ),
            session_id=int(session_id),
            release_backing=release_backing,
            released_alias_count=int(params.released_alias_count),
            remaining_alias_count=int(params.remaining_alias_count),
            released_alias_bytes=int(params.released_alias_bytes),
            remaining_alias_bytes=int(params.remaining_alias_bytes),
            released_backing_bytes=int(params.released_backing_bytes),
            remaining_backing_bytes=int(params.remaining_backing_bytes),
            session_cleared=cleared_raw == 1,
            cache_cleared=kernel_session_cleared,
            poisoned=(poisoned_raw == 1 or contract_violation),
            kernel_poisoned=poisoned_raw == 1,
            boolean_fields_ok=boolean_fields_ok,
            contract_violation=contract_violation,
            ioctl_ret=ret,
            errno=errno_value,
            rm_status=int(params.rmStatus),
            ioctl_calls=1,
            ioctl_wall_ns=ioctl_wall_ns,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
        if kernel_session_cleared:
            # released_* is a per-call value and can legitimately disagree
            # with stale userspace totals after a prior partially successful
            # append. The authoritative cleared+remaining tuple means there
            # is no kernel session left to retry, so always retire our fd and
            # cache even while returning the contract mismatch to the caller.
            legacy_owners = _uvm._uvm_slot_direct_owners.get(normalized_device)
            if legacy_owners is not None:
                for layer_id, expert_id, slot_id in cached.backing_identities:
                    key = (layer_id, slot_id)
                    if legacy_owners.get(key) == expert_id:
                        legacy_owners.pop(key, None)
            _static_alias_sessions.pop(normalized_device, None)
            with suppress(OSError):
                os.close(cached.fd)
        elif ok:
            cached.alias_count = 0
            cached.alias_bytes = 0
            cached.backing_bytes = int(params.remaining_backing_bytes)
            cached.target_ranges.clear()
        else:
            cached.alias_count = int(params.remaining_alias_count)
            cached.alias_bytes = int(params.remaining_alias_bytes)
            cached.backing_bytes = int(params.remaining_backing_bytes)
            cached.poisoned = poisoned_raw == 1 or contract_violation
        return result


def get_static_alias_diagnostics() -> dict[str, object]:
    """Return process-local lifecycle state without issuing an ioctl."""
    with _static_alias_lock:
        sessions = [
            {
                "device": device,
                "session_id": session.session_id,
                "owner_pid": session.owner_pid,
                "sealed": session.sealed,
                "poisoned": session.poisoned,
                "alias_count": session.alias_count,
                "alias_bytes": session.alias_bytes,
                "backing_bytes": session.backing_bytes,
            }
            for device, session in sorted(_static_alias_sessions.items())
        ]
    return {"session_count": len(sessions), "sessions": sessions}


def _close_static_alias_sessions() -> None:
    """Best-effort process-exit cleanup; never destroy a fork-inherited session."""
    with _static_alias_lock:
        snapshot = [
            (device, session.session_id, session.owner_pid, session.fd)
            for device, session in _static_alias_sessions.items()
        ]
    for device, session_id, owner_pid, fd in snapshot:
        if owner_pid != os.getpid() or session_id == 0:
            with suppress(OSError):
                os.close(fd)
            continue
        try:
            result = driver_static_alias_destroy(
                session_id,
                device=device,
                release_backing=True,
                synchronize=True,
            )
            if not bool(result.get("ok")):
                with suppress(OSError):
                    os.close(fd)
        except Exception:
            with suppress(OSError):
                os.close(fd)
