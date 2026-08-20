from __future__ import annotations

import ctypes
import errno
import os
from collections.abc import Callable

import pytest

from vllm.device_allocator import uvm_static_alias as static_alias

UUID_BYTES = bytes(range(16))
PAGE_BYTES = 4096


@pytest.fixture(autouse=True)
def reset_static_alias_state(monkeypatch: pytest.MonkeyPatch):
    static_alias._static_alias_sessions.clear()
    static_alias._uvm._uvm_slot_direct_owners.clear()
    monkeypatch.setattr(
        static_alias.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    yield
    static_alias._static_alias_sessions.clear()
    static_alias._uvm._uvm_slot_direct_owners.clear()


def _uuid() -> static_alias._StaticAliasUuid:
    value = static_alias._StaticAliasUuid()
    value.uuid[:] = UUID_BYTES
    return value


def _install_fd_candidates(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_fds: tuple[int, ...] = (10,),
    duplicate_fds: tuple[int, ...] = (110,),
) -> list[int]:
    assert len(source_fds) == len(duplicate_fds)
    mapping = dict(zip(source_fds, duplicate_fds))
    closed: list[int] = []
    monkeypatch.setattr(static_alias, "_phase_f_gpu_uuid", lambda _device: _uuid())
    monkeypatch.setattr(
        static_alias,
        "_iter_existing_uvm_fd_sources",
        lambda: list(source_fds),
    )
    monkeypatch.setattr(
        static_alias,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: mapping[source_fd],
    )
    monkeypatch.setattr(static_alias.os, "close", lambda fd: closed.append(int(fd)))
    return closed


def _entries() -> list[tuple[int, int, int, int, int, int, int]]:
    return [
        (
            0,
            11,
            7,
            static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
            0x20000000,
            PAGE_BYTES,
            0,
        ),
        (0, 11, 7, 0, 0x30000000, PAGE_BYTES, 0),
    ]


def _complete_map(
    params: static_alias._StaticAliasMapParams,
    *,
    session_id: int = 41,
    prior_alias_bytes: int = 0,
    backing_bytes: int = PAGE_BYTES,
) -> None:
    requested = sum(int(params.entries[index].bytes) for index in range(params.count))
    params.session_id = session_id
    params.requested_bytes = requested
    params.mapped_bytes = requested
    params.total_alias_bytes = prior_alias_bytes + requested
    params.backing_bytes = backing_bytes
    params.success_count = params.count
    params.failure_count = 0
    params.alias_count = params.count
    params.tlb_invalidate_count = params.count
    params.poisoned = 0
    for index in range(params.count):
        params.entries[index].status = 0
    params.rmStatus = 0


def _create_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ioctl: Callable[[int, int, object], int] | None = None,
) -> dict[str, object]:
    _install_fd_candidates(monkeypatch)

    if ioctl is None:

        def ioctl(_fd: int, _request: int, raw: object) -> int:
            assert isinstance(raw, static_alias._StaticAliasMapParams)
            _complete_map(raw)
            return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    return static_alias.driver_static_alias_map(_entries())


def test_static_alias_ioctl_abi_matches_kernel_header() -> None:
    assert static_alias.UVM_MOE_STATIC_ALIAS_ABI_VERSION == 1
    assert static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES == 1152
    assert static_alias._UVM_MOE_STATIC_ALIAS_MAP_IOCTL == 97
    assert static_alias._UVM_MOE_STATIC_ALIAS_SEAL_IOCTL == 98
    assert static_alias._UVM_MOE_STATIC_ALIAS_DESTROY_IOCTL == 99

    assert ctypes.sizeof(static_alias._StaticAliasEntry) == 48
    assert static_alias._StaticAliasEntry.va_base.offset == 16
    assert static_alias._StaticAliasEntry.status.offset == 40

    assert ctypes.sizeof(static_alias._StaticAliasMapParams) == 55400
    assert static_alias._StaticAliasMapParams.session_id.offset == 16
    assert static_alias._StaticAliasMapParams.requested_bytes.offset == 40
    assert static_alias._StaticAliasMapParams.count.offset == 72
    assert static_alias._StaticAliasMapParams.entries.offset == 96
    assert static_alias._StaticAliasMapParams.rmStatus.offset == 55392

    assert ctypes.sizeof(static_alias._StaticAliasSealParams) == 72
    assert static_alias._StaticAliasSealParams.expected_alias_count.offset == 48
    assert static_alias._StaticAliasSealParams.rmStatus.offset == 64

    assert ctypes.sizeof(static_alias._StaticAliasDestroyParams) == 80
    assert static_alias._StaticAliasDestroyParams.released_alias_count.offset == 56
    assert static_alias._StaticAliasDestroyParams.rmStatus.offset == 72


def test_map_create_marshals_entries_and_caches_independent_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fd_candidates(monkeypatch)
    calls: list[tuple[int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append((fd, request))
        assert params.abi_version == 1
        assert params.struct_size == 55400
        assert params.flags == 0
        assert params.reserved == 0
        assert params.session_id == 0
        assert bytes(params.gpu_uuid.uuid) == UUID_BYTES
        assert params.count == 2
        first = params.entries[0]
        assert (
            first.backing_layer_id,
            first.backing_expert_id,
            first.backing_slot_id,
            first.flags,
            first.va_base,
            first.bytes,
            first.slot_offset,
        ) == (0, 11, 7, 1, 0x20000000, PAGE_BYTES, 0)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is True
    assert result["session_id"] == 41
    assert result["requested_bytes"] == 2 * PAGE_BYTES
    assert result["mapped_bytes"] == 2 * PAGE_BYTES
    assert result["total_alias_bytes"] == 2 * PAGE_BYTES
    assert result["backing_bytes"] == PAGE_BYTES
    assert result["entry_statuses"] == [0, 0]
    assert calls == [(110, 97)]
    diagnostics = static_alias.get_static_alias_diagnostics()
    assert diagnostics["session_count"] == 1
    assert diagnostics["sessions"][0] == {
        "device": 0,
        "session_id": 41,
        "owner_pid": os.getpid(),
        "sealed": False,
        "poisoned": False,
        "alias_count": 2,
        "alias_bytes": 2 * PAGE_BYTES,
        "backing_bytes": PAGE_BYTES,
    }


def test_map_create_probes_zero_progress_fd_then_keeps_matching_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(
        monkeypatch,
        source_fds=(10, 11),
        duplicate_fds=(110, 111),
    )
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append(fd)
        if fd == 110:
            ctypes.set_errno(errno.EINVAL)
            return -1
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is True
    assert result["fd_candidate_attempts"] == 2
    assert result["fd_candidate_ioctl_calls"] == 2
    assert calls == [110, 111]
    assert closed == [110]
    assert static_alias._static_alias_sessions[0].fd == 111


def test_map_create_retries_bare_failed_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(
        monkeypatch,
        source_fds=(10, 11),
        duplicate_fds=(110, 111),
    )
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append(fd)
        if fd == 110:
            params.session_id = 7
            params.failure_count = params.count
            params.rmStatus = 0x26
            return 0
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is True
    assert result["fd_candidate_attempts"] == 2
    assert calls == [110, 111]
    assert closed == [110]
    assert static_alias._static_alias_sessions[0].fd == 111


def test_map_create_prefers_slot_preallocation_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(
        monkeypatch,
        source_fds=(10, 11),
        duplicate_fds=(110, 111),
    )
    static_alias._uvm._uvm_slot_direct_sessions[0] = (11, UUID_BYTES)
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append(fd)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    try:
        result = static_alias.driver_static_alias_map(_entries())
    finally:
        static_alias._uvm._uvm_slot_direct_sessions.pop(0, None)

    assert result["ok"] is True
    assert calls == [111]
    assert closed == []
    assert static_alias._static_alias_sessions[0].fd == 111


def test_map_append_uses_cached_fd_and_updates_total_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    calls: list[tuple[int, int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append((fd, request, int(params.session_id)))
        _complete_map(
            params,
            session_id=41,
            prior_alias_bytes=2 * PAGE_BYTES,
        )
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(
        [(0, 11, 7, 0, 0x40000000, PAGE_BYTES, 0)],
        session_id=41,
    )

    assert result["ok"] is True
    assert result["total_alias_bytes"] == 3 * PAGE_BYTES
    assert result["total_alias_count"] == 3
    assert calls == [(110, 97, 41)]
    assert static_alias._static_alias_sessions[0].alias_count == 3


def test_map_append_failure_after_topology_commit_syncs_destroy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    appended = [(0, 12, 8, 0, 0x40000000, PAGE_BYTES, 0)]

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(
            params,
            session_id=41,
            prior_alias_bytes=2 * PAGE_BYTES,
            backing_bytes=2 * PAGE_BYTES,
        )
        # This mirrors the driver path where the complete alias topology is
        # committed before a later PTE operation fails and poisons the session.
        params.mapped_bytes = 0
        params.success_count = 0
        params.failure_count = params.count
        params.poisoned = 1
        params.entries[0].status = 8
        params.rmStatus = 0x56
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(appended, session_id=41)

    assert result["ok"] is False
    assert result["made_progress"] is True
    assert result["alias_count"] == 1
    assert result["total_alias_count"] == 3
    cached = static_alias._static_alias_sessions[0]
    assert cached.poisoned is True
    assert cached.alias_count == 3
    assert cached.alias_bytes == 3 * PAGE_BYTES
    assert cached.backing_bytes == 2 * PAGE_BYTES
    assert cached.target_ranges[-1] == (0x40000000, 0x40000000 + PAGE_BYTES)
    assert cached.backing_identities == {(0, 11, 7), (0, 12, 8)}


@pytest.mark.parametrize(
    "entries,error_fragment",
    [
        (
            [
                (0, 0, 0, 0, 0x20000000, 2 * PAGE_BYTES, 0),
                (0, 0, 0, 0, 0x20001000, PAGE_BYTES, 0),
            ],
            "overlap",
        ),
        (
            [
                (
                    0,
                    0,
                    0,
                    static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K,
                    0x20001000,
                    PAGE_BYTES,
                    0,
                )
            ],
            "65536-byte aligned",
        ),
        ([(0, 0, 0, 4, 0x20000000, PAGE_BYTES, 0)], "unknown flags"),
        (
            [
                (0, 0, 0, 0, 0x20000000, PAGE_BYTES, 0),
                (
                    0,
                    0,
                    0,
                    static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K,
                    0x30000000,
                    64 * 1024,
                    0,
                ),
            ],
            "mix 4K and 64K",
        ),
    ],
)
def test_map_validation_rejects_before_fd_or_ioctl(
    monkeypatch: pytest.MonkeyPatch,
    entries: list[tuple[int, ...]],
    error_fragment: str,
) -> None:
    monkeypatch.setattr(
        static_alias,
        "_phase_f_gpu_uuid",
        lambda _device: (_ for _ in ()).throw(AssertionError("UUID must not run")),
    )
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ioctl must not run")),
    )

    result = static_alias.driver_static_alias_map(entries)

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"
    assert error_fragment in str(result["error"])


@pytest.mark.parametrize("operation", ["map", "seal", "destroy"])
def test_all_static_alias_ioctls_reject_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(
        static_alias.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        static_alias,
        "_phase_f_gpu_uuid",
        lambda _device: (_ for _ in ()).throw(AssertionError("UUID must not run")),
    )
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ioctl must not run")),
    )
    if operation == "map":
        result = static_alias.driver_static_alias_map(_entries())
    else:
        static_alias._static_alias_sessions[0] = static_alias._StaticAliasSession(
            fd=110,
            uuid_bytes=UUID_BYTES,
            session_id=41,
            owner_pid=os.getpid(),
            alias_count=2,
            alias_bytes=2 * PAGE_BYTES,
            backing_bytes=PAGE_BYTES,
        )
        if operation == "seal":
            result = static_alias.driver_static_alias_seal(41)
        else:
            result = static_alias.driver_static_alias_destroy(41)

    assert result["ok"] is False
    assert result["reason"] == "cuda_graph_capture_unsupported"


def test_seal_checks_kernel_counts_and_freezes_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    calls: list[tuple[int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasSealParams)
        calls.append((fd, request))
        assert params.abi_version == 1
        assert params.struct_size == 72
        assert params.flags == params.reserved == 0
        assert params.session_id == 41
        assert params.expected_alias_count == 2
        assert params.expected_alias_bytes == 2 * PAGE_BYTES
        params.sealed = 1
        params.alias_count = 2
        params.alias_bytes = 2 * PAGE_BYTES
        params.backing_bytes = PAGE_BYTES
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_seal(41)

    assert result["ok"] is True
    assert result["sealed"] is True
    assert calls == [(110, 98)]
    append = static_alias.driver_static_alias_map(
        [(0, 11, 7, 0, 0x40000000, PAGE_BYTES, 0)],
        session_id=41,
    )
    assert append["ok"] is False
    assert append["reason"] == "static_alias_session_sealed"


def test_map_contract_mismatch_poison_is_destroy_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fd_candidates(monkeypatch)

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params, session_id=77)
        params.mapped_bytes -= PAGE_BYTES
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is False
    assert result["reason"] == "map_contract_mismatch"
    assert result["contract_violation"] is True
    assert static_alias._static_alias_sessions[0].poisoned is True
    assert static_alias._static_alias_sessions[0].backing_identities == {
        (0, 11, 7)
    }
    append = static_alias.driver_static_alias_map(
        [(0, 11, 7, 0, 0x40000000, PAGE_BYTES, 0)],
        session_id=77,
    )
    assert append["reason"] == "static_alias_session_poisoned"


def test_destroy_release_backing_closes_fd_and_clears_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(monkeypatch)

    def create_ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", create_ioctl)
    assert static_alias.driver_static_alias_map(_entries())["ok"] is True
    static_alias._uvm._uvm_slot_direct_owners[0] = {(0, 7): 11}

    def destroy_ioctl(fd: int, request: int, params: object) -> int:
        assert fd == 110 and request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        assert params.abi_version == 1
        assert params.struct_size == 80
        assert params.flags == 1
        assert params.reserved == 0
        assert params.session_id == 41
        params.released_alias_count = 2
        params.remaining_alias_count = 0
        params.released_alias_bytes = 2 * PAGE_BYTES
        params.remaining_alias_bytes = 0
        params.released_backing_bytes = PAGE_BYTES
        params.remaining_backing_bytes = 0
        params.session_cleared = 1
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", destroy_ioctl)
    result = static_alias.driver_static_alias_destroy(41)

    assert result["ok"] is True
    assert result["session_cleared"] is True
    assert static_alias.get_static_alias_diagnostics()["session_count"] == 0
    assert static_alias._uvm._uvm_slot_direct_owners[0] == {}
    assert closed == [110]


def test_destroy_kernel_cleared_retires_stale_userspace_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(monkeypatch)

    def create_ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", create_ioctl)
    assert static_alias.driver_static_alias_map(_entries())["ok"] is True
    static_alias._uvm._uvm_slot_direct_owners[0] = {(0, 7): 11}

    def destroy_ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        # Model an older stale wrapper cache after a committed append: the
        # kernel releases more aliases/backing than userspace expected, but
        # authoritatively reports that nothing remains.
        params.released_alias_count = 3
        params.released_alias_bytes = 3 * PAGE_BYTES
        params.released_backing_bytes = 2 * PAGE_BYTES
        params.remaining_alias_count = 0
        params.remaining_alias_bytes = 0
        params.remaining_backing_bytes = 0
        params.session_cleared = 1
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", destroy_ioctl)
    result = static_alias.driver_static_alias_destroy(41)

    assert result["ok"] is False
    assert result["reason"] == "destroy_contract_mismatch"
    assert result["contract_violation"] is True
    assert result["session_cleared"] is True
    assert result["cache_cleared"] is True
    assert static_alias.get_static_alias_diagnostics()["session_count"] == 0
    assert static_alias._uvm._uvm_slot_direct_owners[0] == {}
    assert closed == [110]


def test_destroy_can_unmap_then_release_retained_backing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(monkeypatch)

    def create_ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", create_ioctl)
    assert static_alias.driver_static_alias_map(_entries())["ok"] is True

    def unmap_ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        assert params.flags == 0
        params.released_alias_count = 2
        params.released_alias_bytes = 2 * PAGE_BYTES
        params.remaining_backing_bytes = PAGE_BYTES
        params.session_cleared = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", unmap_ioctl)
    first = static_alias.driver_static_alias_destroy(41, release_backing=False)
    assert first["ok"] is True
    assert static_alias._static_alias_sessions[0].alias_count == 0
    assert closed == []

    def release_ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        assert params.flags == 1
        params.released_backing_bytes = PAGE_BYTES
        params.session_cleared = 1
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", release_ioctl)
    second = static_alias.driver_static_alias_destroy(41, release_backing=True)
    assert second["ok"] is True
    assert closed == [110]


def test_fork_inherited_session_cannot_seal_or_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    owner_pid = os.getpid()
    monkeypatch.setattr(static_alias.os, "getpid", lambda: owner_pid + 1)
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ioctl must not run")),
    )

    sealed = static_alias.driver_static_alias_seal(41)
    destroyed = static_alias.driver_static_alias_destroy(41)

    assert sealed["reason"] == "fork_inherited_static_alias_session"
    assert destroyed["reason"] == "fork_inherited_static_alias_session"
