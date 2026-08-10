# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace

import pytest

from vllm.device_allocator import uvm as uvm_module


def test_region_state_ioctl_is_available_for_local_uvm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    runtime = SimpleNamespace(
        uvm_driver_update_region_state=lambda region_id, state: (
            calls.append((int(region_id.value), int(state.value))) or 1
        ),
        uvm_driver_query_region=lambda *_args: 1,
        uvm_driver_query_region_by_va=lambda *_args: 1,
    )
    monkeypatch.setattr(uvm_module, "_uvm_enabled", False)
    monkeypatch.setattr(uvm_module, "_uvm_lib", runtime)

    assert uvm_module.has_region_state_ioctl() is True
    assert uvm_module.driver_update_region_state(region_id=7, state=4) == 1
    assert calls == [(7, 4)]


def _release_params(*, remote_read: bool = False):
    params = uvm_module._ExpertOffloadReleaseParams()
    params.region_id = 7
    params.base = 0x100000
    params.length = 0x80000
    if remote_read:
        params.flags = uvm_module.UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ
    return params


def _patch_driver_fds(
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing_sources: list[int] | None = None,
    new_fd: int = 200,
) -> list[int]:
    closed: list[int] = []
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(
        uvm_module,
        "_iter_existing_uvm_fd_sources",
        lambda: list(existing_sources or []),
    )
    monkeypatch.setattr(
        uvm_module,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: source_fd + 100,
    )
    monkeypatch.setattr(uvm_module, "_open_initialized_uvm_fd", lambda: new_fd)
    monkeypatch.setattr(
        uvm_module.os,
        "close",
        lambda fd: closed.append(fd),
    )
    return closed


def test_region_release_accepts_complete_release() -> None:
    params = _release_params()
    params.block_count = 1
    params.attempted_blocks = 1
    params.released_bytes = params.length

    assert uvm_module._region_release_outcome(params, ret=0) == (True, True)


def test_region_release_accepts_zero_count_cpu_idempotency() -> None:
    params = _release_params()

    assert uvm_module._region_release_outcome(params, ret=0) == (True, False)


def test_region_release_rejects_zero_count_without_valid_region() -> None:
    params = uvm_module._ExpertOffloadReleaseParams()

    assert uvm_module._region_release_outcome(params, ret=0) == (False, False)


def test_remote_release_requires_complete_verified_mapping() -> None:
    params = _release_params(remote_read=True)
    params.block_count = 1
    params.cpu_resident_bytes = params.length
    params.remote_mapped_bytes = params.length - 1

    assert uvm_module._region_release_outcome(params, ret=0) == (False, False)

    params.remote_mapped_bytes = params.length
    assert uvm_module._region_release_outcome(params, ret=0) == (True, False)


def test_new_remote_mapping_counts_as_progress() -> None:
    params = _release_params(remote_read=True)
    params.block_count = 1
    params.cpu_resident_bytes = params.length
    params.remote_mapped_bytes = params.length
    params.newly_remote_mapped_bytes = params.length

    assert uvm_module._region_release_outcome(params, ret=0) == (True, True)


def test_driver_release_returns_only_per_region_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_driver_fds(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.length = 0x80000
        params.block_count = 1
        if params.region_id == 11:
            params.attempted_blocks = 1
            params.released_bytes = params.length
            return 0
        params.rmStatus = 5
        return -1

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_release_regions([11, 12], prefer_existing=False)

    assert result == {"ok": False, "statuses": {11: True, 12: False}}


def test_existing_fd_without_progress_falls_back_to_new_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _patch_driver_fds(monkeypatch, existing_sources=[5], new_fd=200)
    ioctl_fds: list[int] = []

    def ioctl(fd: int, _request: int, params: object) -> int:
        ioctl_fds.append(fd)
        params.length = 0x80000
        if fd == 105:
            params.rmStatus = 5
            return -1
        params.block_count = 1
        params.attempted_blocks = 1
        params.released_bytes = params.length
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_release_regions([11])

    assert result == {"ok": True, "statuses": {11: True}}
    assert ioctl_fds == [105, 200]
    assert closed == [105, 200]


def test_existing_fd_with_partial_progress_does_not_retry_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_driver_fds(monkeypatch, existing_sources=[5])
    opened_new = False

    def open_new() -> int:
        nonlocal opened_new
        opened_new = True
        return 200

    monkeypatch.setattr(uvm_module, "_open_initialized_uvm_fd", open_new)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.length = 0x80000
        if params.region_id == 11:
            params.block_count = 1
            params.attempted_blocks = 1
            params.released_bytes = params.length
            return 0
        params.rmStatus = 5
        return -1

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_release_regions([11, 12])

    assert result == {"ok": False, "statuses": {11: True, 12: False}}
    assert opened_new is False


@pytest.mark.parametrize("failure_stage", ["uuid", "open", "ioctl"])
def test_driver_release_exceptions_fail_every_requested_region(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    _patch_driver_fds(monkeypatch)
    if failure_stage == "uuid":
        monkeypatch.setattr(
            uvm_module,
            "_phase_f_gpu_uuid",
            lambda _device: (_ for _ in ()).throw(RuntimeError("uuid")),
        )
    elif failure_stage == "open":
        monkeypatch.setattr(
            uvm_module,
            "_open_initialized_uvm_fd",
            lambda: (_ for _ in ()).throw(OSError("open")),
        )
    else:
        monkeypatch.setattr(
            uvm_module,
            "_libc_ioctl",
            lambda *_args: (_ for _ in ()).throw(OSError("ioctl")),
        )

    result = uvm_module.driver_release_regions([11, 12], prefer_existing=False)

    assert result == {"ok": False, "statuses": {11: False, 12: False}}


def test_empty_release_is_idempotent_and_api_has_no_partial_limit() -> None:
    assert uvm_module.driver_release_regions([]) == {"ok": True, "statuses": {}}
    parameters = inspect.signature(uvm_module.driver_release_regions).parameters
    assert "max_blocks" not in parameters
    assert "return_entries" not in parameters
    assert "max_blocks" not in dict(uvm_module._ExpertOffloadReleaseParams._fields_)
    assert uvm_module._ExpertOffloadReleaseParams.base.offset == 32
    assert uvm_module._ExpertOffloadReleaseParams.rmStatus.offset == 140
