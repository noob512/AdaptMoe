# SPDX-License-Identifier: Apache-2.0

from vllm.device_allocator import uvm as uvm_module


def _idempotent_release_params(*, remote_read: bool):
    params = uvm_module._ExpertOffloadReleaseParams()
    params.region_id = 7
    params.base = 0x100000
    params.length = 0x80000
    if remote_read:
        params.flags = (
            uvm_module.UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ
        )
    return params


def test_region_release_accepts_zero_count_cpu_idempotency() -> None:
    params = _idempotent_release_params(remote_read=False)

    entry = uvm_module._region_release_entry_to_dict(
        params,
        ret=0,
        saved_errno=0,
    )

    assert entry["ok"] is True
    assert entry["idempotent"] is True
    assert entry["reason"] is None


def test_remote_release_rejects_zero_count_cpu_idempotency() -> None:
    params = _idempotent_release_params(remote_read=True)

    entry = uvm_module._region_release_entry_to_dict(
        params,
        ret=0,
        saved_errno=0,
    )

    assert entry["ok"] is False
    assert entry["idempotent"] is False
    assert entry["reason"] == "remote_mapping_incomplete"


def test_repeated_remote_release_accepts_verified_mapping() -> None:
    params = _idempotent_release_params(remote_read=True)
    params.block_count = 1
    params.remote_map_blocks = 1
    params.cpu_resident_bytes = params.length
    params.remote_mapped_bytes = params.length

    entry = uvm_module._region_release_entry_to_dict(
        params,
        ret=0,
        saved_errno=0,
    )

    assert entry["ok"] is True
    assert entry["idempotent"] is False
    assert entry["reason"] is None


def test_region_release_rejects_zero_count_without_valid_region() -> None:
    params = uvm_module._ExpertOffloadReleaseParams()

    entry = uvm_module._region_release_entry_to_dict(
        params,
        ret=0,
        saved_errno=0,
    )

    assert entry["ok"] is False
    assert entry["idempotent"] is False
    assert entry["reason"] == "no_va_blocks"
