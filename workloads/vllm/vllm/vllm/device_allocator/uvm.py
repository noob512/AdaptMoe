# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA UVM (Unified Virtual Memory) allocator for vLLM.

This module provides a custom CUDA memory allocator that uses cudaMallocManaged
to enable memory oversubscription - allocating more GPU memory than physically
available by using CPU memory as backing store.

Usage:
    # Enable UVM before any CUDA operations
    from vllm.device_allocator.uvm import enable_uvm_allocator

    # Option 1: Call explicitly
    enable_uvm_allocator()

    # Option 2: Set environment variable VLLM_USE_UVM=1

Example:
    VLLM_USE_UVM=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf

Note:
    UVM allows running models larger than GPU memory, but with significant
    performance overhead due to page migration between CPU and GPU.
    This is useful for:
    - Testing large models on smaller GPUs
    - Development and debugging
    - Scenarios where memory > throughput

Warning:
    UVM mode is NOT recommended for production use due to:
    - Unpredictable latency from page faults
    - Lower throughput compared to native GPU memory
    - Potential compatibility issues with CUDA graphs
"""

import atexit
import ctypes
import gc
import os
import uuid as uuidlib
from contextlib import contextmanager
from collections.abc import Sequence
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global state
_uvm_enabled = False
_uvm_lib: Optional[ctypes.CDLL] = None
_uvm_log_close_registered = False
_uvm_local_allocator = None
_uvm_local_mem_pool = None
_uvm_phase_stack: list[str] = []
_UVM_DRIVER_BATCH_MAX_RANGES = 64
_UVM_INITIALIZE_IOCTL = 0x30000001
_UVM_PHASE_F_BATCH_EXECUTOR_IOCTL = 81
_UVM_PHASE_F_MAX_RANGES = 64
_UVM_MOE_SLOT_REMAP_BATCH_IOCTL = 91
_UVM_MOE_SLOT_POOL_QUERY_IOCTL = 92
_UVM_EXPERT_RESERVED_POOL_IOCTL = 98
_UVM_EXPERT_RESERVED_PREFETCH_BATCH_IOCTL = 99
_UVM_MOE_SLOT_REMAP_MAX_ENTRIES = 128
_UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES = 128
_UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES = 128
_UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES = 128
_UVM_MOE_SLOT_REMAP_STATS_COUNT = 8
_UVM_PAGE_SIZE = 0

UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ = 5
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG = 6
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED = 7
UVM_PHASE_F_ACTION_START_BG_PROMOTION = 8
UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT = 14
UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION = 16
UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION = 17
UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST = 18
UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS = 19

UVM_REGION_POOL_UNKNOWN = 0
UVM_REGION_POOL_KV = 1
UVM_REGION_POOL_WEIGHT = 2
UVM_REGION_POOL_SCRATCH = 3
UVM_REGION_POOL_ACTIVATION = 4

UVM_REGION_ROLE_UNKNOWN = 0
UVM_REGION_ROLE_KV_BLOCK = 1
UVM_REGION_ROLE_DENSE_WEIGHT = 2
UVM_REGION_ROLE_MOE_EXPERT_W13 = 3
UVM_REGION_ROLE_MOE_EXPERT_W2 = 4
UVM_REGION_ROLE_SCRATCH = 5
UVM_REGION_ROLE_WORKSPACE = 6

UVM_REGION_FLAG_EVICTABLE = 1 << 0
UVM_REGION_FLAG_PROTECTED = 1 << 1
UVM_REGION_FLAG_ACTIVE = 1 << 2
UVM_REGION_FLAG_PREFETCHABLE = 1 << 3
UVM_REGION_FLAG_PINNED = 1 << 4
UVM_REGION_FLAG_TRANSIENT = 1 << 5
UVM_REGION_FLAG_HOT_SUBPOOL = 1 << 11
UVM_REGION_FLAG_STREAMING_SUBPOOL = 1 << 12
UVM_REGION_FLAG_COLD_SUBPOOL = 1 << 13
UVM_REGION_FLAG_CPU_OFFLOADED_RESERVED = 1 << 14

UVM_REGION_STATE_UNKNOWN = 0
UVM_REGION_STATE_COLD = 1
UVM_REGION_STATE_WARM = 2
UVM_REGION_STATE_HOT = 3
UVM_REGION_STATE_ACTIVE = 4
UVM_REGION_STATE_INACTIVE = 5

UVM_POOL_BUDGET_MODE_TRACE_ONLY = 0
UVM_POOL_BUDGET_MODE_SOFT = 1
UVM_POOL_BUDGET_MODE_ENFORCE = 2

UVM_MOE_SLOT_REMAP_MODE_DRY_RUN = 0
UVM_MOE_SLOT_REMAP_MODE_VALIDATE = 1
UVM_MOE_SLOT_REMAP_MODE_REMAP = 2
UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY = 3
UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY = 4
UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP = 5
UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP = 6

UVM_MOE_SLOT_REMAP_ROLE_UNKNOWN = 0
UVM_MOE_SLOT_REMAP_ROLE_GATE_UP = 3
UVM_MOE_SLOT_REMAP_ROLE_DOWN = 4

UVM_MOE_SLOT_POOL_QUERY_FLAG_FILTER_LAYER = 1 << 0
UVM_MOE_SLOT_POOL_QUERY_FLAG_FILTER_SLOT = 1 << 1
UVM_MOE_SLOT_POOL_QUERY_FLAG_RESET = 1 << 31

UVM_MOE_SLOT_POOL_STATE_EMPTY = 0
UVM_MOE_SLOT_POOL_STATE_RESERVED = 1
UVM_MOE_SLOT_POOL_STATE_LOADING = 2
UVM_MOE_SLOT_POOL_STATE_READY = 3
UVM_MOE_SLOT_POOL_STATE_IN_USE = 4
UVM_MOE_SLOT_POOL_STATE_RELEASED = 5
UVM_MOE_SLOT_POOL_STATE_ERROR = 6

UVM_MOE_SLOT_POOL_BACKING_NONE = 0
UVM_MOE_SLOT_POOL_BACKING_SYSMEM = 1
UVM_MOE_SLOT_POOL_BACKING_SYSMEM_DMA = 2
UVM_MOE_SLOT_POOL_BACKING_VIDMEM = 3

UVM_EXPERT_RESERVED_POOL_MODE_QUERY = 0
UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC = 1
UVM_EXPERT_RESERVED_POOL_MODE_RESET = 2

UVM_EXPERT_RESERVED_POOL_ENTRY_FLAG_ALLOW_EVICT = 1 << 0

UVM_EXPERT_RESERVED_POOL_SLOT_EMPTY = 0
UVM_EXPERT_RESERVED_POOL_SLOT_RESERVED = 1
UVM_EXPERT_RESERVED_POOL_SLOT_LOADING = 2
UVM_EXPERT_RESERVED_POOL_SLOT_ASSIGNED = 3
UVM_EXPERT_RESERVED_POOL_SLOT_ERROR = 4

UVM_EXPERT_RESERVED_PREFETCH_MODE_DRY_RUN = 0
UVM_EXPERT_RESERVED_PREFETCH_MODE_RESOLVE = 1
UVM_EXPERT_RESERVED_PREFETCH_MODE_INVALIDATE = 2
UVM_EXPERT_RESERVED_PREFETCH_MODE_RESERVED_PREFETCH = 3

UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT = 1 << 0

UVM_EXPERT_RESERVED_PREFETCH_ENTRY_OK = 0
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_INVALID_ARGUMENT = 1
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_INVALID_DEVICE = 2
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_REGION_NOT_FOUND = 3
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_REGION_UNSUPPORTED = 4
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_SLOT_NOT_FOUND = 5
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_SLOT_UNBACKED = 6
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_SLOT_TOO_SMALL = 7
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_SLOT_IN_USE = 8
UVM_EXPERT_RESERVED_PREFETCH_ENTRY_NATIVE_POPULATE_UNSUPPORTED = 9


class _UvmInitializeParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class _PhaseFUuid(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)]


class _PhaseFRange(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
    ]


class _PhaseFRangeResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("resident_on_count", ctypes.c_uint32),
        ("resident_on_cpu", ctypes.c_uint8),
        ("resident_on_gpu", ctypes.c_uint8),
        ("queried_bytes", ctypes.c_uint64),
        ("resident_gpu_bytes", ctypes.c_uint64),
        ("resident_cpu_bytes", ctypes.c_uint64),
        ("resident_any_bytes", ctypes.c_uint64),
        ("resident_unknown_bytes", ctypes.c_uint64),
    ]


class _PhaseFBatchExecutorParams(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("async_", ctypes.c_uint8),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("ranges", _PhaseFRange * _UVM_PHASE_F_MAX_RANGES),
        ("results", _PhaseFRangeResult * _UVM_PHASE_F_MAX_RANGES),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MoeSlotRemapEntry(ctypes.Structure):
    _fields_ = [
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("va_base", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("slot_offset", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
    ]


class _MoeSlotRemapBatchParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("validate_managed", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("entries", _MoeSlotRemapEntry * _UVM_MOE_SLOT_REMAP_MAX_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MoeSlotPoolQueryEntry(ctypes.Structure):
    _fields_ = [
        ("layer_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("owner_expert_id", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("range_count", ctypes.c_uint32),
        ("last_role", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("last_status", ctypes.c_uint32),
        ("epoch", ctypes.c_uint64),
        ("batch_id", ctypes.c_uint64),
        ("first_va_base", ctypes.c_uint64),
        ("last_va_base", ctypes.c_uint64),
        ("last_bytes", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),
        ("slot_bytes", ctypes.c_uint64),
        ("slot_physical_bytes", ctypes.c_uint64),
        ("copy_count", ctypes.c_uint64),
        ("copy_bytes", ctypes.c_uint64),
        ("copy_failed_count", ctypes.c_uint64),
        ("pte_remap_bytes", ctypes.c_uint64),
        ("pte_remap_count", ctypes.c_uint64),
        ("pte_clear_count", ctypes.c_uint64),
        ("pte_va_base", ctypes.c_uint64),
        ("pte_bytes", ctypes.c_uint64),
        ("pte_block_unmap_count", ctypes.c_uint64),
        ("pte_block_tracker_wait_count", ctypes.c_uint64),
        ("pte_rollback_count", ctypes.c_uint64),
        ("pte_rollback_bytes", ctypes.c_uint64),
        ("pte_failure_count", ctypes.c_uint64),
        ("pte_fault_clear_count", ctypes.c_uint64),
        ("pte_migrate_clear_count", ctypes.c_uint64),
        ("pte_unmap_clear_count", ctypes.c_uint64),
        ("pte_split_clear_count", ctypes.c_uint64),
        ("pte_release_clear_count", ctypes.c_uint64),
        ("pte_evict_clear_count", ctypes.c_uint64),
        ("pte_dangerous_op_reject_count", ctypes.c_uint64),
        ("backing_kind", ctypes.c_uint32),
        ("copy_status", ctypes.c_uint32),
        ("pte_mapped", ctypes.c_uint32),
        ("pte_status", ctypes.c_uint32),
        ("pte_mode", ctypes.c_uint32),
        ("pte_va_block_tracked", ctypes.c_uint32),
    ]


class _MoeSlotPoolQueryParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("max_entries", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("total_slots", ctypes.c_uint32),
        ("returned_slots", ctypes.c_uint32),
        ("active_slots", ctypes.c_uint32),
        ("backed_slots", ctypes.c_uint32),
        ("current_epoch", ctypes.c_uint64),
        ("validated_entries", ctypes.c_uint64),
        ("overwrite_count", ctypes.c_uint64),
        ("range_update_count", ctypes.c_uint64),
        ("pool_full_count", ctypes.c_uint64),
        ("slot_backing_bytes", ctypes.c_uint64),
        ("copy_count", ctypes.c_uint64),
        ("copy_bytes", ctypes.c_uint64),
        ("copy_failed_count", ctypes.c_uint64),
        ("vidmem_evict_count", ctypes.c_uint64),
        ("pte_remap_count", ctypes.c_uint64),
        ("pte_remap_bytes", ctypes.c_uint64),
        ("pte_clear_count", ctypes.c_uint64),
        ("pte_block_unmap_count", ctypes.c_uint64),
        ("pte_block_tracker_wait_count", ctypes.c_uint64),
        ("pte_rollback_count", ctypes.c_uint64),
        ("pte_rollback_bytes", ctypes.c_uint64),
        ("pte_failure_count", ctypes.c_uint64),
        ("pte_fault_clear_count", ctypes.c_uint64),
        ("pte_migrate_clear_count", ctypes.c_uint64),
        ("pte_unmap_clear_count", ctypes.c_uint64),
        ("pte_split_clear_count", ctypes.c_uint64),
        ("pte_release_clear_count", ctypes.c_uint64),
        ("pte_evict_clear_count", ctypes.c_uint64),
        ("pte_dangerous_op_reject_count", ctypes.c_uint64),
        ("pte_residual_va_block_count", ctypes.c_uint64),
        ("pte_residual_page_count", ctypes.c_uint64),
        ("pte_residual_gpu_mask", ctypes.c_uint64),
        ("entries", _MoeSlotPoolQueryEntry * _UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class _ExpertReservedPoolEntry(ctypes.Structure):
    _fields_ = [
        ("pool_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("chunk_count", ctypes.c_uint32),
        ("slot_bytes", ctypes.c_uint64),
        ("physical_bytes", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
        ("first_chunk_address", ctypes.c_uint64),
        ("last_chunk_address", ctypes.c_uint64),
    ]


class _ExpertReservedPoolParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("max_entries", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("total_slots", ctypes.c_uint32),
        ("returned_slots", ctypes.c_uint32),
        ("active_slots", ctypes.c_uint32),
        ("backed_slots", ctypes.c_uint32),
        ("chunk_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("reserved_backing_bytes", ctypes.c_uint64),
        ("logical_slot_bytes", ctypes.c_uint64),
        ("current_epoch", ctypes.c_uint64),
        ("prealloc_count", ctypes.c_uint64),
        ("reset_count", ctypes.c_uint64),
        ("alloc_failed_count", ctypes.c_uint64),
        (
            "entries",
            _ExpertReservedPoolEntry * _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
        ),
        ("rmStatus", ctypes.c_uint32),
    ]


class _ExpertReservedPrefetchEntry(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("pool_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("slot_offset", ctypes.c_uint64),
        ("bytes_override", ctypes.c_uint64),
        ("resolved_base", ctypes.c_uint64),
        ("resolved_bytes", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("slot_epoch", ctypes.c_uint64),
        ("physical_bytes", ctypes.c_uint64),
    ]


class _ExpertReservedPrefetchBatchParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("async_", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("resolved_regions", ctypes.c_uint64),
        ("assigned_slots", ctypes.c_uint64),
        ("invalidated_slots", ctypes.c_uint64),
        ("reserved_backing_bytes", ctypes.c_uint64),
        ("ordinary_resident_bytes", ctypes.c_uint64),
        ("pool_external_alloc_count", ctypes.c_uint64),
        ("footprint_contract_broken", ctypes.c_uint64),
        ("current_epoch", ctypes.c_uint64),
        (
            "entries",
            _ExpertReservedPrefetchEntry *
            _UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES,
        ),
        ("rmStatus", ctypes.c_uint32),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.ioctl.restype = ctypes.c_int


def _libc_ioctl(fd: int, request: int, params: ctypes.Structure) -> int:
    return int(_LIBC.ioctl(
        ctypes.c_int(fd),
        ctypes.c_ulong(request),
        ctypes.byref(params),
    ))


def _phase_f_gpu_uuid(device: int) -> _PhaseFUuid:
    if device < 0:
        device = torch.cuda.current_device()
    raw_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if raw_uuid is None:
        raise RuntimeError("torch.cuda device properties do not expose UUID")
    if isinstance(raw_uuid, bytes):
        uuid_bytes = raw_uuid
    else:
        uuid_text = str(raw_uuid)
        if uuid_text.startswith("GPU-"):
            uuid_text = uuid_text[4:]
        uuid_bytes = uuidlib.UUID(uuid_text).bytes
    if len(uuid_bytes) != 16:
        raise RuntimeError(f"Unexpected CUDA UUID size: {len(uuid_bytes)}")
    return _PhaseFUuid((ctypes.c_uint8 * 16).from_buffer_copy(uuid_bytes))


def _open_initialized_uvm_fd() -> int:
    fd = os.open("/dev/nvidia-uvm", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    params = _UvmInitializeParams()
    ret = _libc_ioctl(fd, _UVM_INITIALIZE_IOCTL, params)
    if ret != 0 or int(params.rmStatus) != 0:
        saved_errno = ctypes.get_errno()
        os.close(fd)
        raise RuntimeError(
            "UVM_INITIALIZE failed: "
            f"ret={ret} errno={saved_errno} rmStatus={int(params.rmStatus)}"
        )
    return fd


def _fd_link_is_uvm(link_target: str) -> bool:
    base = os.path.basename(link_target)
    return base == "nvidia-uvm" or base.startswith("nvidia-uvm ")


def _iter_existing_uvm_fd_sources() -> list[int]:
    try:
        fd_names = sorted(
            os.listdir("/proc/self/fd"),
            key=lambda value: int(value) if value.isdigit() else -1,
        )
    except OSError:
        return []

    sources: list[int] = []
    for name in fd_names:
        if not name.isdigit():
            continue
        source_fd = int(name)
        if source_fd < 3:
            continue
        try:
            link_target = os.readlink(f"/proc/self/fd/{source_fd}")
        except OSError:
            continue
        if not _fd_link_is_uvm(link_target):
            continue
        sources.append(source_fd)
    return sources


def _initialize_duplicated_uvm_fd(source_fd: int) -> Optional[int]:
    try:
        fd = os.dup(source_fd)
    except OSError:
        return None

    params = _UvmInitializeParams()
    ret = _libc_ioctl(fd, _UVM_INITIALIZE_IOCTL, params)
    if ret == 0 and int(params.rmStatus) == 0:
        return fd

    try:
        os.close(fd)
    except OSError:
        pass
    return None


def _adopt_existing_uvm_fd() -> Optional[int]:
    for source_fd in _iter_existing_uvm_fd_sources():
        fd = _initialize_duplicated_uvm_fd(source_fd)
        if fd is not None:
            return fd
    return None


def _iter_initialized_existing_uvm_fds():
    for source_fd in _iter_existing_uvm_fd_sources():
        fd = _initialize_duplicated_uvm_fd(source_fd)
        if fd is not None:
            yield fd


def _open_or_adopt_initialized_uvm_fd(*, prefer_existing: bool = False) -> int:
    if prefer_existing:
        fd = _adopt_existing_uvm_fd()
        if fd is not None:
            return fd

    return _open_initialized_uvm_fd()


def _fill_moe_slot_remap_params(
    params: _MoeSlotRemapBatchParams,
    *,
    chunk: list[dict[str, int]],
    mode_value: int,
    gpu_uuid: _PhaseFUuid,
    validate_managed: bool,
) -> None:
    params.mode = mode_value
    params.count = len(chunk)
    params.gpu_uuid = gpu_uuid
    params.validate_managed = 1 if validate_managed else 0
    for index, entry in enumerate(chunk):
        params.entries[index].layer_id = entry["layer_id"]
        params.entries[index].expert_id = entry["expert_id"]
        params.entries[index].slot_id = entry["slot_id"]
        params.entries[index].role = entry["role"]
        params.entries[index].va_base = entry["va_base"]
        params.entries[index].bytes = entry["bytes"]
        params.entries[index].slot_offset = entry["slot_offset"]
        params.entries[index].flags = entry["flags"]


def _moe_slot_fd_validates_targets(
    fd: int,
    normalized: list[dict[str, int]],
    *,
    gpu_uuid: _PhaseFUuid,
) -> bool:
    for start in range(0, len(normalized), _UVM_MOE_SLOT_REMAP_MAX_ENTRIES):
        chunk = normalized[start:start + _UVM_MOE_SLOT_REMAP_MAX_ENTRIES]
        params = _MoeSlotRemapBatchParams()
        _fill_moe_slot_remap_params(
            params,
            chunk=chunk,
            mode_value=UVM_MOE_SLOT_REMAP_MODE_DRY_RUN,
            gpu_uuid=gpu_uuid,
            validate_managed=True,
        )
        ctypes.set_errno(0)
        ret = _libc_ioctl(fd, _UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
        if (
            ret != 0
            or int(params.rmStatus) != 0
            or int(params.success_count) != len(chunk)
            or int(params.failure_count) != 0
        ):
            return False
    return True


def _open_moe_slot_uvm_fd_for_targets(
    normalized: list[dict[str, int]],
    *,
    mode_value: int,
    gpu_uuid: _PhaseFUuid,
    prefer_existing: bool,
) -> int:
    needs_target_owner = mode_value in (
        UVM_MOE_SLOT_REMAP_MODE_REMAP,
        UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY,
        UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY,
        UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP,
        UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP,
    )

    if prefer_existing:
        fallback_fd: Optional[int] = None
        for fd in _iter_initialized_existing_uvm_fds():
            if needs_target_owner:
                if _moe_slot_fd_validates_targets(
                    fd,
                    normalized,
                    gpu_uuid=gpu_uuid,
                ):
                    if fallback_fd is not None:
                        os.close(fallback_fd)
                    return fd
                os.close(fd)
                continue
            elif fallback_fd is None:
                fallback_fd = fd
                continue

            if fallback_fd is None:
                fallback_fd = fd
            else:
                os.close(fd)

        if fallback_fd is not None:
            return fallback_fd

    return _open_initialized_uvm_fd()


def find_uvm_allocator_library() -> Optional[str]:
    """
    Find the UVM allocator shared library.

    Returns:
        Path to the library if found, None otherwise.
    """
    # Try to find the library in common locations. In local editable layouts,
    # ``import vllm`` may resolve as a namespace package with ``__file__`` set
    # to None, so derive the package path from this module first.
    vllm_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )

    # Possible library names and locations
    lib_names = [
        "uvm_allocator.abi3.so",  # ABI3 build (most common)
        "uvm_allocator.so",
        "libuvm_allocator.so",
    ]
    search_paths = [
        os.path.join(vllm_path, "..", "uvm_test"),  # Local development build
        vllm_path,  # Installed location
        os.path.join(vllm_path, ".."),  # Development location
        os.path.dirname(os.path.abspath(__file__)),  # Same directory
    ]

    for path in search_paths:
        for lib_name in lib_names:
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                return lib_path

    # Try to find using /proc/self/maps (similar to cumem allocator)
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "uvm_allocator" in line:
                    start = line.index("/")
                    return line[start:].strip()
    except (FileNotFoundError, ValueError):
        pass

    return None


def load_uvm_library() -> ctypes.CDLL:
    """
    Load the UVM allocator library and set up function signatures.

    Returns:
        CDLL handle to the library.

    Raises:
        RuntimeError: If library cannot be found or loaded.
    """
    lib_path = find_uvm_allocator_library()

    if lib_path is None:
        raise RuntimeError(
            "UVM allocator library not found. "
            "Please ensure vLLM is built with UVM support enabled. "
            "The library should be at: <vllm_path>/uvm_allocator.so"
        )

    logger.info("Loading UVM allocator library from: %s", lib_path)

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load UVM allocator library: {e}") from e

    # Define function signatures for statistics API
    lib.uvm_get_allocated_bytes.restype = ctypes.c_size_t
    lib.uvm_get_allocated_bytes.argtypes = []

    lib.uvm_get_peak_allocated_bytes.restype = ctypes.c_size_t
    lib.uvm_get_peak_allocated_bytes.argtypes = []

    lib.uvm_get_num_allocs.restype = ctypes.c_size_t
    lib.uvm_get_num_allocs.argtypes = []

    lib.uvm_get_num_frees.restype = ctypes.c_size_t
    lib.uvm_get_num_frees.argtypes = []

    lib.uvm_reset_peak_stats.restype = None
    lib.uvm_reset_peak_stats.argtypes = []

    lib.uvm_reset_all_stats.restype = None
    lib.uvm_reset_all_stats.argtypes = []

    lib.uvm_set_prefetch.restype = None
    lib.uvm_set_prefetch.argtypes = [ctypes.c_int]

    lib.uvm_set_verbose.restype = None
    lib.uvm_set_verbose.argtypes = [ctypes.c_int]

    lib.uvm_set_phase.restype = None
    lib.uvm_set_phase.argtypes = [ctypes.c_char_p]

    lib.uvm_mark_phase_event.restype = None
    lib.uvm_mark_phase_event.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

    lib.uvm_prefetch.restype = None
    lib.uvm_prefetch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    lib.uvm_batch_prefetch.restype = ctypes.c_int
    lib.uvm_batch_prefetch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    lib.uvm_advise.restype = None
    lib.uvm_advise.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.uvm_batch_advise.restype = ctypes.c_int
    lib.uvm_batch_advise.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.uvm_driver_batch_prefetch_cpu.restype = ctypes.c_int
    lib.uvm_driver_batch_prefetch_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    lib.uvm_driver_batch_prefetch_gpu.restype = ctypes.c_int
    lib.uvm_driver_batch_prefetch_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.uvm_driver_batch_advise_cpu.restype = ctypes.c_int
    lib.uvm_driver_batch_advise_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]

    lib.uvm_driver_batch_query_residency.restype = ctypes.c_int
    lib.uvm_driver_batch_query_residency.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    for helper_name in (
        "uvm_driver_batch_prepare_remote_read",
        "uvm_driver_batch_prepare_remote_read_bg_deferred",
        "uvm_driver_batch_start_bg_promotion",
        "uvm_driver_batch_gpu_only_read_dup_soft",
        "uvm_driver_batch_expert_read_dup_fast",
        "uvm_driver_batch_pause_bg_promotion",
        "uvm_driver_batch_resume_bg_promotion",
    ):
        if hasattr(lib, helper_name):
            helper = getattr(lib, helper_name)
            helper.restype = ctypes.c_int
            helper.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ]

    lib.uvm_driver_batch_query_residency_bytes.restype = ctypes.c_int
    lib.uvm_driver_batch_query_residency_bytes.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]

    lib.uvm_driver_phase_f_register_region.restype = ctypes.c_int
    lib.uvm_driver_phase_f_register_region.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]

    lib.uvm_driver_phase_f_unregister_region.restype = ctypes.c_int
    lib.uvm_driver_phase_f_unregister_region.argtypes = [ctypes.c_uint64]

    lib.uvm_driver_batch_query_residency_regions.restype = ctypes.c_int
    lib.uvm_driver_batch_query_residency_regions.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    if hasattr(lib, "uvm_driver_batch_expert_read_dup_fast_regions"):
        lib.uvm_driver_batch_expert_read_dup_fast_regions.restype = ctypes.c_int
        lib.uvm_driver_batch_expert_read_dup_fast_regions.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
        ]

    if hasattr(lib, "uvm_driver_batch_expert_read_dup_fast_native_regions"):
        lib.uvm_driver_batch_expert_read_dup_fast_native_regions.restype = ctypes.c_int
        lib.uvm_driver_batch_expert_read_dup_fast_native_regions.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]

    lib.uvm_driver_phase_f_register_query_region.restype = ctypes.c_int
    lib.uvm_driver_phase_f_register_query_region.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    if hasattr(lib, "uvm_driver_moe_slot_remap_batch"):
        lib.uvm_driver_moe_slot_remap_batch.restype = ctypes.c_int
        lib.uvm_driver_moe_slot_remap_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

    if hasattr(lib, "uvm_driver_phase_n_set_pool_budget"):
        lib.uvm_driver_phase_n_set_pool_budget.restype = ctypes.c_int
        lib.uvm_driver_phase_n_set_pool_budget.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]

    if hasattr(lib, "uvm_driver_phase_n_get_pool_state"):
        lib.uvm_driver_phase_n_get_pool_state.restype = ctypes.c_int
        lib.uvm_driver_phase_n_get_pool_state.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

    if hasattr(lib, "uvm_driver_phase_n_reclaim_pool"):
        lib.uvm_driver_phase_n_reclaim_pool.restype = ctypes.c_int
        lib.uvm_driver_phase_n_reclaim_pool.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

    if hasattr(lib, "uvm_driver_phase_n_update_region_state"):
        lib.uvm_driver_phase_n_update_region_state.restype = ctypes.c_int
        lib.uvm_driver_phase_n_update_region_state.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]

    if hasattr(lib, "uvm_driver_phase_n_query_region"):
        lib.uvm_driver_phase_n_query_region.restype = ctypes.c_int
        lib.uvm_driver_phase_n_query_region.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

    if hasattr(lib, "uvm_driver_phase_n_query_region_by_va"):
        lib.uvm_driver_phase_n_query_region_by_va.restype = ctypes.c_int
        lib.uvm_driver_phase_n_query_region_by_va.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

    lib.uvm_is_managed_pointer.restype = ctypes.c_int
    lib.uvm_is_managed_pointer.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]

    lib.uvm_close_log.restype = None
    lib.uvm_close_log.argtypes = []

    return lib


def _close_uvm_log_at_exit() -> None:
    if _uvm_lib is None:
        return
    try:
        _uvm_lib.uvm_close_log()
    except Exception:
        logger.debug("Failed to close UVM allocator log", exc_info=True)


def ensure_uvm_runtime_library() -> ctypes.CDLL:
    """Load the UVM helper library without enabling the global CUDA allocator.

    This is used by isolated experiments that allocate only selected tensors
    through a local PyTorch MemPool while the rest of vLLM keeps the ordinary
    CUDA allocator.
    """
    global _uvm_lib
    if _uvm_lib is None:
        _uvm_lib = load_uvm_library()
        _uvm_lib.uvm_set_prefetch(0)
        _uvm_lib.uvm_set_verbose(0)
    return _uvm_lib


def cleanup_uvm_local_memory_pool() -> None:
    """Drop local UVM MemPool objects before interpreter/CUDA teardown.

    PyTorch's CUDAPluggableAllocator/MemPool destructors can run very late
    during Python shutdown. Keeping the local UVM allocator cached globally
    until then has caused process-exit segfaults in isolated microbenchmarks,
    even after all tensors were explicitly deleted. Clearing the cached pool
    while CUDA and Python runtime state are still healthy avoids that teardown
    ordering hazard.
    """
    global _uvm_local_allocator, _uvm_local_mem_pool

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Failed to synchronize before UVM local pool cleanup", exc_info=True)

    _uvm_local_mem_pool = None
    _uvm_local_allocator = None
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        logger.debug("Failed to synchronize after UVM local pool cleanup", exc_info=True)


@contextmanager
def uvm_local_memory_pool():
    """Temporarily allocate CUDA tensors from cudaMallocManaged.

    Unlike ``enable_uvm_allocator()``, this does not call
    ``change_current_allocator``. Only allocations inside this context use the
    UVM pluggable allocator, which lets CPU-offload experiments replace just
    the offloaded weight tensors.
    """
    global _uvm_local_allocator, _uvm_local_mem_pool
    ensure_uvm_runtime_library()
    lib_path = find_uvm_allocator_library()
    if lib_path is None:
        raise RuntimeError("UVM allocator library not found")

    if _uvm_local_allocator is None or _uvm_local_mem_pool is None:
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            lib_path, "uvm_malloc", "uvm_free"
        )
        _uvm_local_allocator = allocator
        _uvm_local_mem_pool = torch.cuda.memory.MemPool(allocator._allocator)

    with torch.cuda.memory.use_mem_pool(_uvm_local_mem_pool):
        yield


def enable_uvm_allocator(
    enable_prefetch: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Enable the UVM allocator for PyTorch CUDA operations.

    This must be called before any CUDA memory allocations occur.

    Args:
        enable_prefetch: If True, prefetch allocations to GPU after allocation.
            This can improve performance but uses more GPU memory.
        verbose: If True, log large allocations to stderr.

    Returns:
        True if UVM was enabled successfully, False otherwise.

    Raises:
        RuntimeError: If UVM allocator library is not available.
    """
    global _uvm_enabled, _uvm_lib, _uvm_log_close_registered

    if _uvm_enabled:
        logger.warning("UVM allocator is already enabled")
        return True

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, UVM allocator cannot be enabled")
        return False

    # Load the library
    _uvm_lib = load_uvm_library()

    # Configure the allocator
    _uvm_lib.uvm_set_prefetch(1 if enable_prefetch else 0)
    _uvm_lib.uvm_set_verbose(1 if verbose else 0)

    # Find library path for PyTorch pluggable allocator
    lib_path = find_uvm_allocator_library()
    if lib_path is None:
        raise RuntimeError("UVM allocator library not found")

    try:
        # Create and enable the pluggable allocator
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            lib_path, 'uvm_malloc', 'uvm_free'
        )
        torch.cuda.memory.change_current_allocator(allocator)
        _uvm_enabled = True
        if not _uvm_log_close_registered:
            atexit.register(_close_uvm_log_at_exit)
            _uvm_log_close_registered = True
        logger.info("UVM allocator enabled successfully")
        _set_uvm_phase("enabled")
        mark_uvm_phase_event("allocator_enabled", "enabled")

        # Pre-initialize cuBLAS after UVM is enabled
        # This ensures cuBLAS workspace is allocated via UVM
        # and won't fail later due to native memory exhaustion
        try:
            logger.info("Pre-initializing cuBLAS with UVM...")
            with uvm_allocation_phase("uvm_enable:cublas_preinit"):
                a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
                b = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                del a, b
                torch.cuda.empty_cache()
            logger.info("cuBLAS pre-initialized successfully with UVM")
        except Exception as e:
            logger.warning("Failed to pre-initialize cuBLAS: %s", e)

        return True

    except Exception as e:
        logger.error("Failed to enable UVM allocator: %s", e)
        raise RuntimeError(f"Failed to enable UVM allocator: {e}") from e


def is_uvm_enabled() -> bool:
    """Check if UVM allocator is currently enabled."""
    return _uvm_enabled


def _set_uvm_phase(phase: str) -> None:
    if _uvm_lib is None:
        return
    _uvm_lib.uvm_set_phase(phase.encode("utf-8"))


def mark_uvm_phase_event(event: str, phase: str | None = None) -> None:
    if _uvm_lib is None:
        return
    event_bytes = event.encode("utf-8")
    phase_bytes = phase.encode("utf-8") if phase is not None else None
    _uvm_lib.uvm_mark_phase_event(event_bytes, phase_bytes)


@contextmanager
def uvm_allocation_phase(phase: str):
    """Attach allocator traces to a high-level execution phase."""
    if not _uvm_enabled or _uvm_lib is None:
        yield
        return

    _uvm_phase_stack.append(phase)
    mark_uvm_phase_event("enter", phase)
    _set_uvm_phase(phase)
    try:
        yield
    finally:
        mark_uvm_phase_event("exit", phase)
        _uvm_phase_stack.pop()
        restored = _uvm_phase_stack[-1] if _uvm_phase_stack else "enabled"
        _set_uvm_phase(restored)


@contextmanager
def uvm_enabled_allocation_phase(phase: str):
    """Attach a finer phase only while serving is in enabled mode."""
    if not _uvm_enabled or _uvm_lib is None:
        yield
        return

    current_phase = _uvm_phase_stack[-1] if _uvm_phase_stack else "enabled"
    if current_phase == "enabled" or current_phase.startswith("enabled:"):
        with uvm_allocation_phase(phase):
            yield
    else:
        yield


def get_uvm_stats() -> dict:
    """
    Get UVM memory statistics.

    Returns:
        Dictionary with memory statistics:
        - allocated_bytes: Current allocated bytes
        - peak_bytes: Peak allocated bytes
        - num_allocs: Total number of allocations
        - num_frees: Total number of frees
    """
    if not _uvm_enabled or _uvm_lib is None:
        return {
            "allocated_bytes": 0,
            "peak_bytes": 0,
            "num_allocs": 0,
            "num_frees": 0,
            "enabled": False,
        }

    return {
        "allocated_bytes": _uvm_lib.uvm_get_allocated_bytes(),
        "peak_bytes": _uvm_lib.uvm_get_peak_allocated_bytes(),
        "num_allocs": _uvm_lib.uvm_get_num_allocs(),
        "num_frees": _uvm_lib.uvm_get_num_frees(),
        "enabled": True,
    }


def reset_uvm_peak_stats() -> None:
    """Reset peak memory statistics to current allocation."""
    if _uvm_lib is not None:
        _uvm_lib.uvm_reset_peak_stats()


def reset_uvm_all_stats() -> None:
    """Reset all memory statistics."""
    if _uvm_lib is not None:
        _uvm_lib.uvm_reset_all_stats()


def prefetch_to_device(tensor: torch.Tensor, device: int = 0) -> None:
    """
    Prefetch a UVM tensor to the specified GPU device.

    This can improve performance by proactively moving data to GPU
    before it's needed.

    Args:
        tensor: The tensor to prefetch (must be on CUDA)
        device: Target GPU device ID
    """
    if not _uvm_enabled:
        return

    if not tensor.is_cuda:
        logger.warning("Cannot prefetch non-CUDA tensor")
        return

    # Use cudaMemPrefetchAsync via the library
    if _uvm_lib is not None:
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        prefetch_range_to_device(ptr, size, device)


def prefetch_range_to_device(ptr: int, size: int, device: int = 0) -> bool:
    """
    Prefetch a raw UVM address range to the specified GPU device.

    Returns True when a prefetch call was issued. This is intentionally a thin
    wrapper around the allocator .so so higher-level policies can prefetch
    logical tensor slices without constructing tensor views.
    """
    if uvm_residency_tag_only_enabled():
        return False
    if _uvm_lib is None:
        return False
    if ptr <= 0 or size <= 0:
        return False
    stream = torch.cuda.current_stream().cuda_stream
    _uvm_lib.uvm_prefetch(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        ctypes.c_int(device),
        ctypes.c_void_p(stream),
    )
    return True


def prefetch_range_to_cpu(ptr: int, size: int) -> bool:
    """Prefetch a raw UVM address range to CPU memory."""
    return prefetch_range_to_device(ptr, size, device=-1)


def probe_driver_capable_range(ptr: int, size: int) -> tuple[bool, str, int]:
    """Return whether a raw range is CUDA managed and suitable for driver ioctls.

    The Phase F driver executor only works on ranges visible to UVM as managed
    VA blocks. This probe records the CUDA runtime's pointer classification so
    weight traces can distinguish non-managed tensors from driver plumbing bugs.
    """
    if _uvm_lib is None:
        return False, "uvm_library_unavailable", -2
    if ptr <= 0 or size <= 0:
        return False, "invalid_range", -3
    try:
        result = int(
            _uvm_lib.uvm_is_managed_pointer(
                ctypes.c_uint64(ptr),
                ctypes.c_size_t(size),
            )
        )
    except Exception:
        logger.debug("Failed to probe UVM pointer attributes", exc_info=True)
        return False, "probe_exception", -4
    if result == 1:
        return True, "cuda_managed", result
    if result == 0:
        return False, "not_cuda_managed", result
    return False, f"cuda_pointer_probe_failed:{result}", result


def _prepare_uvm_range_arrays(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[ctypes.Array[ctypes.c_uint64], ctypes.Array[ctypes.c_size_t], int]:
    valid_ranges = [
        (int(ptr), int(size))
        for ptr, size in ranges
        if int(ptr) > 0 and int(size) > 0
    ]
    count = len(valid_ranges)
    ptr_array = (ctypes.c_uint64 * count)(
        *(ptr for ptr, _ in valid_ranges)
    )
    size_array = (ctypes.c_size_t * count)(
        *(size for _, size in valid_ranges)
    )
    return ptr_array, size_array, count


def get_uvm_page_size() -> int:
    global _UVM_PAGE_SIZE
    if _UVM_PAGE_SIZE > 0:
        return _UVM_PAGE_SIZE

    raw = os.environ.get("VLLM_UVM_HYBRID_PAGE_SIZE", "").strip()
    if raw:
        try:
            value = int(raw, 0)
            if value > 0:
                _UVM_PAGE_SIZE = value
                return _UVM_PAGE_SIZE
        except ValueError:
            logger.warning("Invalid VLLM_UVM_HYBRID_PAGE_SIZE=%r", raw)

    try:
        _UVM_PAGE_SIZE = int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        _UVM_PAGE_SIZE = 4096
    if _UVM_PAGE_SIZE <= 0:
        _UVM_PAGE_SIZE = 4096
    return _UVM_PAGE_SIZE


def round_ranges_to_pages(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    page_size: int | None = None,
) -> list[tuple[int, int]]:
    """Round raw byte ranges to page ranges and merge overlaps.

    This is the intentionally aggressive experimental normalization used by
    the hybrid Phase-F ioctl path. It assumes the rounded pages are safe to
    treat as readonly managed weight pages.
    """
    page = int(page_size or get_uvm_page_size())
    if page <= 0:
        page = 4096

    rounded: list[tuple[int, int]] = []
    for ptr, size in ranges:
        start = int(ptr)
        length = int(size)
        if start <= 0 or length <= 0:
            continue
        end = start + length
        rounded_start = start - (start % page)
        rounded_end = ((end + page - 1) // page) * page
        if rounded_end <= rounded_start:
            continue
        rounded.append((rounded_start, rounded_end - rounded_start))

    if not rounded:
        return []

    rounded.sort()
    merged: list[tuple[int, int]] = []
    cur_start, cur_size = rounded[0]
    cur_end = cur_start + cur_size
    for start, size in rounded[1:]:
        end = start + size
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end - cur_start))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end - cur_start))
    return merged


def _phase_f_cpp_helper_name(action: int) -> str | None:
    if action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ:
        return "uvm_driver_batch_prepare_remote_read"
    if action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED:
        return "uvm_driver_batch_prepare_remote_read_bg_deferred"
    if action == UVM_PHASE_F_ACTION_START_BG_PROMOTION:
        return "uvm_driver_batch_start_bg_promotion"
    if action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT:
        return "uvm_driver_batch_gpu_only_read_dup_soft"
    if action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST:
        return "uvm_driver_batch_expert_read_dup_fast"
    if action == UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION:
        return "uvm_driver_batch_pause_bg_promotion"
    if action == UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION:
        return "uvm_driver_batch_resume_bg_promotion"
    return None


def _phase_f_cpp_helper_enabled() -> bool:
    raw = os.environ.get("VLLM_UVM_PHASE_F_CPP_HELPER")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _driver_phase_f_batch_execute_ranges_cpp_helper(
    ranges: list[tuple[int, int]],
    action: int,
    *,
    device: int = 0,
    async_: bool = False,
) -> tuple[int, int, int, int] | None:
    if not _phase_f_cpp_helper_enabled() or _uvm_lib is None:
        return None

    helper_name = _phase_f_cpp_helper_name(action)
    if helper_name is None or not hasattr(_uvm_lib, helper_name):
        return None

    helper = getattr(_uvm_lib, helper_name)
    total_success = 0
    total_failure = 0
    last_rm_status = 0
    for start in range(0, len(ranges), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = ranges[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        result = int(
            helper(
                ptr_array,
                size_array,
                ctypes.c_size_t(count),
                ctypes.c_int(device),
                ctypes.c_int(1 if async_ else 0),
            )
        )
        if result < 0:
            if result < -6:
                last_rm_status = -result
            total_failure += count
            logger.warning(
                "Phase-F C++ helper %s failed: result=%s rmStatus=%s "
                "success=%s failure=%s",
                helper_name,
                result,
                last_rm_status,
                total_success,
                total_failure,
            )
            return total_success, total_failure, last_rm_status, result
        total_success += result
        if result < count:
            total_failure += count - result

    return total_success, total_failure, last_rm_status, 0


def driver_phase_f_batch_execute_ranges(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    action: int,
    *,
    device: int = 0,
    async_: bool = False,
) -> tuple[int, int, int, int]:
    """Submit raw VA ranges to the modified UVM Phase-F executor.

    This pure-Python ioctl path is intentionally independent from the
    pluggable allocator .so exports, so isolated vLLM experiments can exercise
    newly added kernel actions without rebuilding vLLM or the allocator helper.

    Returns (success_count, failure_count, rm_status, ioctl_ret). Negative
    ``ioctl_ret`` values are local wrapper failures.
    """
    if uvm_residency_tag_only_enabled():
        return 0, 0, 0, 0

    range_list = [
        (int(ptr), int(size))
        for ptr, size in ranges
        if int(ptr) > 0 and int(size) > 0
    ]
    if not range_list:
        return 0, 0, 0, 0

    helper_result = _driver_phase_f_batch_execute_ranges_cpp_helper(
        range_list,
        action,
        device=device,
        async_=async_,
    )
    if helper_result is not None:
        return helper_result

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for Phase-F ioctl",
                     exc_info=True)
        return 0, len(range_list), 0, -3

    fd = -1
    try:
        fd = _open_or_adopt_initialized_uvm_fd(prefer_existing=True)
        total_success = 0
        total_failure = 0
        last_rm_status = 0

        for start in range(0, len(range_list), _UVM_PHASE_F_MAX_RANGES):
            chunk = range_list[start:start + _UVM_PHASE_F_MAX_RANGES]
            params = _PhaseFBatchExecutorParams()
            params.action = int(action)
            params.count = len(chunk)
            params.gpu_uuid = gpu_uuid
            params.async_ = 1 if async_ else 0
            for index, (ptr, size) in enumerate(chunk):
                params.ranges[index].base = int(ptr)
                params.ranges[index].length = int(size)

            ret = _libc_ioctl(fd, _UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, params)
            if ret != 0:
                saved_errno = ctypes.get_errno()
                logger.warning(
                    "Phase-F ioctl action=%s failed: ret=%s errno=%s "
                    "rmStatus=%s success=%s failure=%s",
                    action,
                    ret,
                    saved_errno,
                    int(params.rmStatus),
                    int(params.success_count),
                    int(params.failure_count),
                )
                total_success += int(params.success_count)
                total_failure += int(params.failure_count)
                return total_success, total_failure, int(params.rmStatus), -6

            total_success += int(params.success_count)
            total_failure += int(params.failure_count)
            last_rm_status = int(params.rmStatus)
            if int(params.rmStatus) != 0 and int(params.success_count) == 0:
                return total_success, total_failure, last_rm_status, -last_rm_status

        return total_success, total_failure, last_rm_status, 0
    except Exception:
        logger.debug("Phase-F direct ioctl failed", exc_info=True)
        return 0, len(range_list), 0, -4
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _moe_slot_remap_mode_value(mode: str | int) -> int:
    if isinstance(mode, int):
        return int(mode)
    normalized = str(mode or "dry_run").strip().lower()
    if normalized in ("dry_run", "dry-run", "trace", "trace_only"):
        return UVM_MOE_SLOT_REMAP_MODE_DRY_RUN
    if normalized in ("validate", "validate_only"):
        return UVM_MOE_SLOT_REMAP_MODE_VALIDATE
    if normalized in ("remap", "pte_remap", "real"):
        return UVM_MOE_SLOT_REMAP_MODE_REMAP
    if normalized in ("copy_only", "copy-only", "copy", "migration", "migrate"):
        return UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY
    if normalized in (
        "gpu_copy_only",
        "gpu-copy-only",
        "gpu_copy",
        "vidmem_copy_only",
        "vidmem-copy-only",
    ):
        return UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY
    if normalized in (
        "slot_pte_remap",
        "slot-pte-remap",
        "pte_remap",
        "pte-remap",
        "slot_pte",
    ):
        return UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP
    if normalized in (
        "slot_va_block_remap",
        "slot-va-block-remap",
        "slot_va_block",
        "va_block_remap",
        "va-block-remap",
    ):
        return UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP
    return UVM_MOE_SLOT_REMAP_MODE_DRY_RUN


def _moe_slot_role_value(role: str | int | None) -> int:
    if isinstance(role, int):
        return int(role)
    normalized = str(role or "").strip().lower()
    if normalized in ("moe_gate_up", "gate_up", "w13", "w13_weight"):
        return UVM_MOE_SLOT_REMAP_ROLE_GATE_UP
    if normalized in ("moe_down", "down", "w2", "w2_weight"):
        return UVM_MOE_SLOT_REMAP_ROLE_DOWN
    return UVM_MOE_SLOT_REMAP_ROLE_UNKNOWN


def _normalize_moe_slot_remap_entries(
    entries: Sequence[dict[str, object]],
    *,
    round_to_pages: bool = True,
) -> list[dict[str, int]]:
    page = get_uvm_page_size()
    normalized: list[dict[str, int]] = []
    for entry in entries:
        try:
            va_base = int(entry.get("va_base", entry.get("ptr", 0)))
            size_bytes = int(entry.get("bytes", entry.get("size", 0)))
            layer_id = int(entry.get("layer_id", 0))
            expert_id = int(entry.get("expert_id", 0))
            slot_id = int(entry.get("slot_id", 0))
            slot_offset = int(entry.get("slot_offset", 0))
            flags = int(entry.get("flags", 0))
        except (TypeError, ValueError, AttributeError):
            continue

        if va_base <= 0 or size_bytes <= 0:
            continue
        if round_to_pages:
            end = va_base + size_bytes
            va_base = va_base - (va_base % page)
            rounded_end = ((end + page - 1) // page) * page
            size_bytes = max(rounded_end - va_base, 0)
        if size_bytes <= 0:
            continue
        normalized.append({
            "layer_id": layer_id,
            "expert_id": expert_id,
            "slot_id": slot_id,
            "role": _moe_slot_role_value(entry.get("role")),
            "va_base": va_base,
            "bytes": size_bytes,
            "slot_offset": slot_offset,
            "flags": flags,
        })
    return normalized


def _empty_moe_slot_remap_result(
    *,
    mode: str | int,
    ioctl_ret: int = 0,
    rm_status: int = 0,
) -> dict[str, object]:
    mode_value = _moe_slot_remap_mode_value(mode)
    return {
        "mode": mode_value,
        "mode_name": {
            UVM_MOE_SLOT_REMAP_MODE_DRY_RUN: "dry_run",
            UVM_MOE_SLOT_REMAP_MODE_VALIDATE: "validate",
            UVM_MOE_SLOT_REMAP_MODE_REMAP: "remap",
            UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY: "copy_only",
            UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY: "gpu_copy_only",
            UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP: "slot_pte_remap",
            UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP: "slot_va_block_remap",
        }.get(mode_value, str(mode)),
        "entry_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": 0,
        "accepted_bytes": 0,
        "remap_count": 0,
        "tlb_invalidate_count": 0,
        "copy_count": 0,
        "copy_bytes": 0,
        "copy_failed_count": 0,
        "rm_status": int(rm_status),
        "ioctl_ret": int(ioctl_ret),
        "ok": int(ioctl_ret) == 0 and int(rm_status) == 0,
        "backend": "none",
    }


def _driver_moe_slot_remap_batch_cpp_helper(
    entries: list[dict[str, int]],
    *,
    mode: str | int = "dry_run",
    device: int = 0,
    validate_managed: bool = False,
) -> dict[str, int | str | bool] | None:
    if _uvm_lib is None or not hasattr(_uvm_lib, "uvm_driver_moe_slot_remap_batch"):
        return None
    if not entries:
        return _empty_moe_slot_remap_result(mode=mode)

    helper = _uvm_lib.uvm_driver_moe_slot_remap_batch
    mode_value = _moe_slot_remap_mode_value(mode)
    if (mode_value in (
            UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY,
            UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY,
    ) or
            any(int(entry.get("slot_offset") or 0) != 0 for entry in entries)):
        return None
    total = _empty_moe_slot_remap_result(mode=mode)
    total["backend"] = "cpp_helper"
    total["ok"] = True

    for start in range(0, len(entries), _UVM_MOE_SLOT_REMAP_MAX_ENTRIES):
        chunk = entries[start:start + _UVM_MOE_SLOT_REMAP_MAX_ENTRIES]
        count = len(chunk)
        layer_ids = (ctypes.c_uint32 * count)(*(e["layer_id"] for e in chunk))
        expert_ids = (ctypes.c_uint32 * count)(*(e["expert_id"] for e in chunk))
        slot_ids = (ctypes.c_uint32 * count)(*(e["slot_id"] for e in chunk))
        roles = (ctypes.c_uint32 * count)(*(e["role"] for e in chunk))
        ptrs = (ctypes.c_uint64 * count)(*(e["va_base"] for e in chunk))
        sizes = (ctypes.c_size_t * count)(*(e["bytes"] for e in chunk))
        stats = (ctypes.c_uint64 * _UVM_MOE_SLOT_REMAP_STATS_COUNT)()
        ret = int(helper(
            layer_ids,
            expert_ids,
            slot_ids,
            roles,
            ptrs,
            sizes,
            ctypes.c_size_t(count),
            ctypes.c_uint32(mode_value),
            ctypes.c_int(device),
            ctypes.c_uint32(1 if validate_managed else 0),
            stats,
            ctypes.c_size_t(_UVM_MOE_SLOT_REMAP_STATS_COUNT),
        ))

        total["entry_count"] = int(total["entry_count"]) + count
        total["rm_status"] = int(stats[0])
        total["success_count"] = int(total["success_count"]) + int(stats[1])
        total["failure_count"] = int(total["failure_count"]) + int(stats[2])
        total["conflict_count"] = int(total["conflict_count"]) + int(stats[3])
        total["requested_bytes"] = int(total["requested_bytes"]) + int(stats[4])
        total["accepted_bytes"] = int(total["accepted_bytes"]) + int(stats[5])
        total["remap_count"] = int(total["remap_count"]) + int(stats[6])
        total["tlb_invalidate_count"] = (
            int(total["tlb_invalidate_count"]) + int(stats[7])
        )
        total["ioctl_ret"] = ret if ret < 0 else 0
        if ret < 0 or int(stats[0]) != 0:
            total["ok"] = False
            break

    if mode_value in (
            UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY,
            UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY,
    ):
        total["copy_count"] = int(total["success_count"])
        total["copy_bytes"] = int(total["accepted_bytes"])
        total["copy_failed_count"] = int(total["failure_count"])

    return total


def driver_moe_slot_remap_batch(
    entries: Sequence[dict[str, object]],
    *,
    mode: str | int = "dry_run",
    device: int = 0,
    validate_managed: bool = False,
    round_to_pages: bool = True,
    fd_target_entries: Sequence[dict[str, object]] | None = None,
    fd_target_round_to_pages: bool | None = None,
    reset_slot_pool_after: bool = False,
) -> dict[str, object]:
    """Submit MoE slot remap assignments to the modified UVM driver."""
    normalized = _normalize_moe_slot_remap_entries(
        entries,
        round_to_pages=round_to_pages,
    )
    if not normalized or uvm_residency_tag_only_enabled():
        return _empty_moe_slot_remap_result(mode=mode)

    fd_targets = normalized
    if fd_target_entries is not None:
        fd_targets = _normalize_moe_slot_remap_entries(
            fd_target_entries,
            round_to_pages=(
                round_to_pages
                if fd_target_round_to_pages is None
                else fd_target_round_to_pages
            ),
        )
        if not fd_targets:
            result = _empty_moe_slot_remap_result(mode=mode, ioctl_ret=-5)
            result["entry_count"] = len(normalized)
            result["failure_count"] = len(normalized)
            result["requested_bytes"] = sum(e["bytes"] for e in normalized)
            result["backend"] = "python_ioctl"
            result["ok"] = False
            return result
    else:
        helper_result = _driver_moe_slot_remap_batch_cpp_helper(
            normalized,
            mode=mode,
            device=device,
            validate_managed=validate_managed,
        ) if not reset_slot_pool_after else None
        if helper_result is not None:
            return helper_result

    try:
        gpu_uuid = _phase_f_gpu_uuid(device) if int(device) >= 0 else _PhaseFUuid()
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for MoE slot remap ioctl",
                     exc_info=True)
        result = _empty_moe_slot_remap_result(mode=mode, ioctl_ret=-3)
        result["entry_count"] = len(normalized)
        result["failure_count"] = len(normalized)
        result["requested_bytes"] = sum(e["bytes"] for e in normalized)
        result["backend"] = "python_ioctl"
        result["ok"] = False
        return result

    mode_value = _moe_slot_remap_mode_value(mode)
    fd = -1
    total = _empty_moe_slot_remap_result(mode=mode)
    total["backend"] = "python_ioctl"
    total["ok"] = True
    try:
        prefer_existing_fd = (
            validate_managed
            or mode_value in (
                UVM_MOE_SLOT_REMAP_MODE_REMAP,
                UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY,
                UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY,
                UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP,
                UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP,
            )
        )
        fd = _open_moe_slot_uvm_fd_for_targets(
            fd_targets,
            mode_value=mode_value,
            gpu_uuid=gpu_uuid,
            prefer_existing=prefer_existing_fd,
        )
        for start in range(0, len(normalized), _UVM_MOE_SLOT_REMAP_MAX_ENTRIES):
            chunk = normalized[start:start + _UVM_MOE_SLOT_REMAP_MAX_ENTRIES]
            params = _MoeSlotRemapBatchParams()
            _fill_moe_slot_remap_params(
                params,
                chunk=chunk,
                mode_value=mode_value,
                gpu_uuid=gpu_uuid,
                validate_managed=validate_managed,
            )

            ret = _libc_ioctl(fd, _UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
            total["entry_count"] = int(total["entry_count"]) + len(chunk)
            total["success_count"] = (
                int(total["success_count"]) + int(params.success_count)
            )
            total["failure_count"] = (
                int(total["failure_count"]) + int(params.failure_count)
            )
            total["conflict_count"] = (
                int(total["conflict_count"]) + int(params.conflict_count)
            )
            total["requested_bytes"] = (
                int(total["requested_bytes"]) + int(params.requested_bytes)
            )
            total["accepted_bytes"] = (
                int(total["accepted_bytes"]) + int(params.accepted_bytes)
            )
            total["remap_count"] = int(total["remap_count"]) + int(params.remap_count)
            total["tlb_invalidate_count"] = (
                int(total["tlb_invalidate_count"]) + int(params.tlb_invalidate_count)
            )
            total["rm_status"] = int(params.rmStatus)
            total["ioctl_ret"] = int(ret)

            if ret != 0 or int(params.rmStatus) != 0:
                saved_errno = ctypes.get_errno()
                logger.warning(
                    "MoE slot remap ioctl failed: mode=%s ret=%s errno=%s "
                    "rmStatus=%s success=%s failure=%s conflicts=%s",
                    mode,
                    ret,
                    saved_errno,
                    int(params.rmStatus),
                    int(params.success_count),
                    int(params.failure_count),
                    int(params.conflict_count),
                )
                total["ok"] = False
                break
        if reset_slot_pool_after and total.get("ok") is True:
            reset_result = _driver_moe_slot_pool_reset_after_remap_on_fd(fd)
            total["slot_pool_reset_after_remap"] = reset_result
            if reset_result.get("ok") is not True:
                total["ok"] = False
                total["reset_after_remap_failed"] = True
        if mode_value in (
                UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY,
                UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY,
        ):
            total["copy_count"] = int(total["success_count"])
            total["copy_bytes"] = int(total["accepted_bytes"])
            total["copy_failed_count"] = int(total["failure_count"])
        return total
    except Exception:
        logger.debug("MoE slot remap direct ioctl failed", exc_info=True)
        result = _empty_moe_slot_remap_result(mode=mode, ioctl_ret=-4)
        result["entry_count"] = len(normalized)
        result["failure_count"] = len(normalized)
        result["requested_bytes"] = sum(e["bytes"] for e in normalized)
        result["backend"] = "python_ioctl"
        result["ok"] = False
        return result
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _moe_slot_pool_entry_to_dict(entry: _MoeSlotPoolQueryEntry) -> dict[str, int]:
    return {
        "layer_id": int(entry.layer_id),
        "slot_id": int(entry.slot_id),
        "owner_expert_id": int(entry.owner_expert_id),
        "state": int(entry.state),
        "range_count": int(entry.range_count),
        "last_role": int(entry.last_role),
        "flags": int(entry.flags),
        "last_status": int(entry.last_status),
        "epoch": int(entry.epoch),
        "batch_id": int(entry.batch_id),
        "first_va_base": int(entry.first_va_base),
        "last_va_base": int(entry.last_va_base),
        "last_bytes": int(entry.last_bytes),
        "total_bytes": int(entry.total_bytes),
        "slot_bytes": int(entry.slot_bytes),
        "slot_physical_bytes": int(entry.slot_physical_bytes),
        "copy_count": int(entry.copy_count),
        "copy_bytes": int(entry.copy_bytes),
        "copy_failed_count": int(entry.copy_failed_count),
        "pte_remap_bytes": int(entry.pte_remap_bytes),
        "pte_remap_count": int(entry.pte_remap_count),
        "pte_clear_count": int(entry.pte_clear_count),
        "pte_va_base": int(entry.pte_va_base),
        "pte_bytes": int(entry.pte_bytes),
        "pte_block_unmap_count": int(entry.pte_block_unmap_count),
        "pte_block_tracker_wait_count": int(entry.pte_block_tracker_wait_count),
        "pte_rollback_count": int(entry.pte_rollback_count),
        "pte_rollback_bytes": int(entry.pte_rollback_bytes),
        "pte_failure_count": int(entry.pte_failure_count),
        "pte_fault_clear_count": int(entry.pte_fault_clear_count),
        "pte_migrate_clear_count": int(entry.pte_migrate_clear_count),
        "pte_unmap_clear_count": int(entry.pte_unmap_clear_count),
        "pte_split_clear_count": int(entry.pte_split_clear_count),
        "pte_release_clear_count": int(entry.pte_release_clear_count),
        "pte_evict_clear_count": int(entry.pte_evict_clear_count),
        "pte_dangerous_op_reject_count": int(entry.pte_dangerous_op_reject_count),
        "backing_kind": int(entry.backing_kind),
        "copy_status": int(entry.copy_status),
        "pte_mapped": int(entry.pte_mapped),
        "pte_status": int(entry.pte_status),
        "pte_mode": int(entry.pte_mode),
        "pte_va_block_tracked": int(entry.pte_va_block_tracked),
    }


_MOE_SLOT_POOL_QUERY_SUMMARY_KEYS = (
    "ok",
    "ioctl_ret",
    "errno",
    "rm_status",
    "total_slots",
    "returned_slots",
    "active_slots",
    "backed_slots",
    "current_epoch",
    "validated_entries",
    "overwrite_count",
    "range_update_count",
    "pool_full_count",
    "slot_backing_bytes",
    "copy_count",
    "copy_bytes",
    "copy_failed_count",
    "vidmem_evict_count",
    "pte_remap_count",
    "pte_remap_bytes",
    "pte_clear_count",
    "pte_block_unmap_count",
    "pte_block_tracker_wait_count",
    "pte_rollback_count",
    "pte_rollback_bytes",
    "pte_failure_count",
    "pte_fault_clear_count",
    "pte_migrate_clear_count",
    "pte_unmap_clear_count",
    "pte_split_clear_count",
    "pte_release_clear_count",
    "pte_evict_clear_count",
    "pte_dangerous_op_reject_count",
    "pte_residual_va_block_count",
    "pte_residual_page_count",
    "pte_residual_gpu_mask",
)


def _compact_moe_slot_pool_query_result(
    result: dict[str, object],
) -> dict[str, int | bool]:
    return {
        key: bool(result.get(key)) if key == "ok" else int(result.get(key) or 0)
        for key in _MOE_SLOT_POOL_QUERY_SUMMARY_KEYS
    }


def _make_moe_slot_pool_query_params(
    *,
    layer_id: int | None = None,
    slot_id: int | None = None,
    max_entries: int = _UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES,
    reset: bool = False,
) -> tuple[_MoeSlotPoolQueryParams, int]:
    flags = 0
    if layer_id is not None:
        flags |= UVM_MOE_SLOT_POOL_QUERY_FLAG_FILTER_LAYER
    if slot_id is not None:
        flags |= UVM_MOE_SLOT_POOL_QUERY_FLAG_FILTER_SLOT
    if reset:
        flags |= UVM_MOE_SLOT_POOL_QUERY_FLAG_RESET

    params = _MoeSlotPoolQueryParams()
    params.flags = flags
    params.max_entries = max(0, min(
        int(max_entries),
        _UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES,
    ))
    params.layer_id = 0 if layer_id is None else int(layer_id)
    params.slot_id = 0 if slot_id is None else int(slot_id)
    return params, flags


def _empty_moe_slot_pool_query_result(flags: int = 0) -> dict[str, object]:
    return {
        "ok": False,
        "ioctl": _UVM_MOE_SLOT_POOL_QUERY_IOCTL,
        "ioctl_ret": -1,
        "errno": 0,
        "rm_status": 0,
        "flags": flags,
        "entries": [],
    }


def _driver_moe_slot_pool_query_on_fd(
    fd: int,
    *,
    layer_id: int | None = None,
    slot_id: int | None = None,
    max_entries: int = _UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES,
    reset: bool = False,
) -> dict[str, object]:
    params, flags = _make_moe_slot_pool_query_params(
        layer_id=layer_id,
        slot_id=slot_id,
        max_entries=max_entries,
        reset=reset,
    )
    result = _empty_moe_slot_pool_query_result(flags)

    ctypes.set_errno(0)
    ret = _libc_ioctl(fd, _UVM_MOE_SLOT_POOL_QUERY_IOCTL, params)
    saved_errno = ctypes.get_errno()
    entries = [
        _moe_slot_pool_entry_to_dict(params.entries[index])
        for index in range(int(params.returned_slots))
    ]
    result.update({
        "ok": ret == 0 and int(params.rmStatus) == 0,
        "ioctl_ret": int(ret),
        "errno": int(saved_errno),
        "rm_status": int(params.rmStatus),
        "total_slots": int(params.total_slots),
        "returned_slots": int(params.returned_slots),
        "active_slots": int(params.active_slots),
        "backed_slots": int(params.backed_slots),
        "current_epoch": int(params.current_epoch),
        "validated_entries": int(params.validated_entries),
        "overwrite_count": int(params.overwrite_count),
        "range_update_count": int(params.range_update_count),
        "pool_full_count": int(params.pool_full_count),
        "slot_backing_bytes": int(params.slot_backing_bytes),
        "copy_count": int(params.copy_count),
        "copy_bytes": int(params.copy_bytes),
        "copy_failed_count": int(params.copy_failed_count),
        "vidmem_evict_count": int(params.vidmem_evict_count),
        "pte_remap_count": int(params.pte_remap_count),
        "pte_remap_bytes": int(params.pte_remap_bytes),
        "pte_clear_count": int(params.pte_clear_count),
        "pte_block_unmap_count": int(params.pte_block_unmap_count),
        "pte_block_tracker_wait_count": int(params.pte_block_tracker_wait_count),
        "pte_rollback_count": int(params.pte_rollback_count),
        "pte_rollback_bytes": int(params.pte_rollback_bytes),
        "pte_failure_count": int(params.pte_failure_count),
        "pte_fault_clear_count": int(params.pte_fault_clear_count),
        "pte_migrate_clear_count": int(params.pte_migrate_clear_count),
        "pte_unmap_clear_count": int(params.pte_unmap_clear_count),
        "pte_split_clear_count": int(params.pte_split_clear_count),
        "pte_release_clear_count": int(params.pte_release_clear_count),
        "pte_evict_clear_count": int(params.pte_evict_clear_count),
        "pte_dangerous_op_reject_count": int(params.pte_dangerous_op_reject_count),
        "pte_residual_va_block_count": int(params.pte_residual_va_block_count),
        "pte_residual_page_count": int(params.pte_residual_page_count),
        "pte_residual_gpu_mask": int(params.pte_residual_gpu_mask),
        "entries": entries,
    })
    return result


def _driver_moe_slot_pool_reset_after_remap_on_fd(
    fd: int,
) -> dict[str, object]:
    before = _driver_moe_slot_pool_query_on_fd(fd, max_entries=0)
    reset = _driver_moe_slot_pool_query_on_fd(fd, max_entries=0, reset=True)
    after = _driver_moe_slot_pool_query_on_fd(fd, max_entries=0)

    before_summary = _compact_moe_slot_pool_query_result(before)
    reset_summary = _compact_moe_slot_pool_query_result(reset)
    after_summary = _compact_moe_slot_pool_query_result(after)
    ok = (
        bool(before_summary["ok"])
        and bool(reset_summary["ok"])
        and bool(after_summary["ok"])
        and int(after_summary["active_slots"]) == 0
        and int(after_summary["backed_slots"]) == 0
        and int(after_summary["slot_backing_bytes"]) == 0
        and int(reset_summary["pte_residual_va_block_count"]) == 0
        and int(reset_summary["pte_residual_page_count"]) == 0
        and int(after_summary["pte_residual_va_block_count"]) == 0
        and int(after_summary["pte_residual_page_count"]) == 0
    )
    return {
        "ok": ok,
        "before": before_summary,
        "reset": reset_summary,
        "after": after_summary,
    }


def driver_moe_slot_pool_query(
    *,
    layer_id: int | None = None,
    slot_id: int | None = None,
    max_entries: int = _UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES,
    reset: bool = False,
    prefer_existing: bool = False,
) -> dict[str, object]:
    """Query the experimental UVM driver-side MoE slot pool metadata."""
    _, flags = _make_moe_slot_pool_query_params(
        layer_id=layer_id,
        slot_id=slot_id,
        max_entries=max_entries,
        reset=reset,
    )
    result = _empty_moe_slot_pool_query_result(flags)

    fd = -1
    try:
        fd = _open_or_adopt_initialized_uvm_fd(prefer_existing=prefer_existing)
        return _driver_moe_slot_pool_query_on_fd(
            fd,
            layer_id=layer_id,
            slot_id=slot_id,
            max_entries=max_entries,
            reset=reset,
        )
    except Exception:
        logger.debug("MoE slot pool query ioctl failed", exc_info=True)
        result["ioctl_ret"] = -4
        return result
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _expert_reserved_pool_mode_value(mode: str | int) -> int:
    if isinstance(mode, int):
        return int(mode)
    normalized = str(mode or "query").strip().lower()
    if normalized in ("query", "status", "state"):
        return UVM_EXPERT_RESERVED_POOL_MODE_QUERY
    if normalized in ("prealloc", "preallocate", "alloc", "reserve"):
        return UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC
    if normalized in ("reset", "release", "clear"):
        return UVM_EXPERT_RESERVED_POOL_MODE_RESET
    return UVM_EXPERT_RESERVED_POOL_MODE_QUERY


def _expert_reserved_pool_entry_to_dict(
    entry: _ExpertReservedPoolEntry,
) -> dict[str, int]:
    return {
        "pool_id": int(entry.pool_id),
        "slot_id": int(entry.slot_id),
        "flags": int(entry.flags),
        "status": int(entry.status),
        "state": int(entry.state),
        "chunk_count": int(entry.chunk_count),
        "slot_bytes": int(entry.slot_bytes),
        "physical_bytes": int(entry.physical_bytes),
        "epoch": int(entry.epoch),
        "first_chunk_address": int(entry.first_chunk_address),
        "last_chunk_address": int(entry.last_chunk_address),
    }


def _empty_expert_reserved_pool_result(
    *,
    mode: str | int = "query",
    ioctl_ret: int = -1,
    rm_status: int = 0,
) -> dict[str, object]:
    mode_value = _expert_reserved_pool_mode_value(mode)
    return {
        "ok": False,
        "ioctl": _UVM_EXPERT_RESERVED_POOL_IOCTL,
        "ioctl_ret": int(ioctl_ret),
        "errno": 0,
        "rm_status": int(rm_status),
        "mode": mode_value,
        "mode_name": {
            UVM_EXPERT_RESERVED_POOL_MODE_QUERY: "query",
            UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC: "prealloc",
            UVM_EXPERT_RESERVED_POOL_MODE_RESET: "reset",
        }.get(mode_value, str(mode)),
        "success_count": 0,
        "failure_count": 0,
        "total_slots": 0,
        "returned_slots": 0,
        "active_slots": 0,
        "backed_slots": 0,
        "chunk_size": 0,
        "reserved_backing_bytes": 0,
        "logical_slot_bytes": 0,
        "current_epoch": 0,
        "prealloc_count": 0,
        "reset_count": 0,
        "alloc_failed_count": 0,
        "entries": [],
    }


def _fill_expert_reserved_pool_params(
    params: _ExpertReservedPoolParams,
    *,
    mode_value: int,
    entries: Sequence[dict[str, object]],
    gpu_uuid: _PhaseFUuid,
    max_entries: int,
    allow_evict: bool,
) -> None:
    params.mode = mode_value
    params.count = len(entries) if mode_value == UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC else 0
    params.gpu_uuid = gpu_uuid
    params.max_entries = max(0, min(
        int(max_entries),
        _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
    ))
    for index, entry in enumerate(entries[:_UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES]):
        flags = int(entry.get("flags", 0))
        if allow_evict:
            flags |= UVM_EXPERT_RESERVED_POOL_ENTRY_FLAG_ALLOW_EVICT
        params.entries[index].pool_id = int(entry.get("pool_id", 0))
        params.entries[index].slot_id = int(entry.get("slot_id", index))
        params.entries[index].flags = flags
        params.entries[index].slot_bytes = int(
            entry.get("slot_bytes", entry.get("bytes", 0))
        )


def _expert_reserved_pool_result_from_params(
    params: _ExpertReservedPoolParams,
    *,
    mode: str | int,
    ret: int,
    saved_errno: int,
) -> dict[str, object]:
    returned_slots = max(0, min(
        int(params.returned_slots),
        _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
    ))
    entries = [
        _expert_reserved_pool_entry_to_dict(params.entries[index])
        for index in range(returned_slots)
    ]
    if int(params.mode) == UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC:
        entries = [
            _expert_reserved_pool_entry_to_dict(params.entries[index])
            for index in range(max(0, min(
                int(params.count),
                _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
            )))
        ]

    result = _empty_expert_reserved_pool_result(
        mode=mode,
        ioctl_ret=ret,
        rm_status=int(params.rmStatus),
    )
    result.update({
        "ok": int(ret) == 0 and int(params.rmStatus) == 0,
        "errno": int(saved_errno),
        "mode": int(params.mode),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "total_slots": int(params.total_slots),
        "returned_slots": int(params.returned_slots),
        "active_slots": int(params.active_slots),
        "backed_slots": int(params.backed_slots),
        "chunk_size": int(params.chunk_size),
        "reserved_backing_bytes": int(params.reserved_backing_bytes),
        "logical_slot_bytes": int(params.logical_slot_bytes),
        "current_epoch": int(params.current_epoch),
        "prealloc_count": int(params.prealloc_count),
        "reset_count": int(params.reset_count),
        "alloc_failed_count": int(params.alloc_failed_count),
        "entries": entries,
    })
    return result


def driver_expert_reserved_pool(
    *,
    mode: str | int = "query",
    entries: Sequence[dict[str, object]] = (),
    device: int = 0,
    max_entries: int = _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
    allow_evict: bool = False,
    prefer_existing: bool = False,
) -> dict[str, object]:
    """Manage the Phase-1 reserved PMM chunk pool in the modified UVM driver."""
    mode_value = _expert_reserved_pool_mode_value(mode)
    normalized_entries = list(entries)
    if len(normalized_entries) > _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES:
        result = _empty_expert_reserved_pool_result(mode=mode, ioctl_ret=-2)
        result["failure_count"] = len(normalized_entries)
        return result

    try:
        gpu_uuid = (
            _phase_f_gpu_uuid(device)
            if mode_value == UVM_EXPERT_RESERVED_POOL_MODE_PREALLOC
            else _PhaseFUuid()
        )
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for expert reserved pool",
                     exc_info=True)
        result = _empty_expert_reserved_pool_result(mode=mode, ioctl_ret=-3)
        result["failure_count"] = len(normalized_entries)
        return result

    params = _ExpertReservedPoolParams()
    _fill_expert_reserved_pool_params(
        params,
        mode_value=mode_value,
        entries=normalized_entries,
        gpu_uuid=gpu_uuid,
        max_entries=max_entries,
        allow_evict=allow_evict,
    )

    fd = -1
    try:
        fd = _open_or_adopt_initialized_uvm_fd(prefer_existing=prefer_existing)
        ctypes.set_errno(0)
        ret = _libc_ioctl(fd, _UVM_EXPERT_RESERVED_POOL_IOCTL, params)
        saved_errno = ctypes.get_errno()
        return _expert_reserved_pool_result_from_params(
            params,
            mode=mode,
            ret=int(ret),
            saved_errno=int(saved_errno),
        )
    except Exception:
        logger.debug("Expert reserved pool ioctl failed", exc_info=True)
        result = _empty_expert_reserved_pool_result(mode=mode, ioctl_ret=-4)
        result["failure_count"] = len(normalized_entries)
        return result
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def driver_expert_reserved_pool_prealloc(
    slots: Sequence[dict[str, object]],
    *,
    device: int = 0,
    allow_evict: bool = False,
    prefer_existing: bool = False,
) -> dict[str, object]:
    return driver_expert_reserved_pool(
        mode="prealloc",
        entries=slots,
        device=device,
        allow_evict=allow_evict,
        prefer_existing=prefer_existing,
    )


def driver_expert_reserved_pool_query(
    *,
    max_entries: int = _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
    prefer_existing: bool = False,
) -> dict[str, object]:
    return driver_expert_reserved_pool(
        mode="query",
        max_entries=max_entries,
        prefer_existing=prefer_existing,
    )


def driver_expert_reserved_pool_reset(
    *,
    max_entries: int = _UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES,
    prefer_existing: bool = False,
) -> dict[str, object]:
    return driver_expert_reserved_pool(
        mode="reset",
        max_entries=max_entries,
        prefer_existing=prefer_existing,
    )


def _expert_reserved_prefetch_mode_value(mode: str | int) -> int:
    if isinstance(mode, int):
        return int(mode)
    normalized = str(mode or "dry_run").strip().lower()
    if normalized in ("dry_run", "dry-run", "validate", "validate_only"):
        return UVM_EXPERT_RESERVED_PREFETCH_MODE_DRY_RUN
    if normalized in ("resolve", "assign", "bind"):
        return UVM_EXPERT_RESERVED_PREFETCH_MODE_RESOLVE
    if normalized in ("invalidate", "clear", "release"):
        return UVM_EXPERT_RESERVED_PREFETCH_MODE_INVALIDATE
    if normalized in (
        "reserved_prefetch",
        "reserved-prefetch",
        "prefetch",
        "native",
        "native_prefetch",
    ):
        return UVM_EXPERT_RESERVED_PREFETCH_MODE_RESERVED_PREFETCH
    return UVM_EXPERT_RESERVED_PREFETCH_MODE_DRY_RUN


def _expert_reserved_prefetch_entry_to_dict(
    entry: _ExpertReservedPrefetchEntry,
) -> dict[str, int]:
    return {
        "region_id": int(entry.region_id),
        "pool_id": int(entry.pool_id),
        "slot_id": int(entry.slot_id),
        "flags": int(entry.flags),
        "status": int(entry.status),
        "slot_offset": int(entry.slot_offset),
        "bytes_override": int(entry.bytes_override),
        "resolved_base": int(entry.resolved_base),
        "resolved_bytes": int(entry.resolved_bytes),
        "layer_id": int(entry.layer_id),
        "expert_id": int(entry.expert_id),
        "role": int(entry.role),
        "slot_epoch": int(entry.slot_epoch),
        "physical_bytes": int(entry.physical_bytes),
    }


def _empty_expert_reserved_prefetch_result(
    *,
    mode: str | int = "dry_run",
    ioctl_ret: int = -1,
    rm_status: int = 0,
) -> dict[str, object]:
    mode_value = _expert_reserved_prefetch_mode_value(mode)
    return {
        "ok": False,
        "ioctl": _UVM_EXPERT_RESERVED_PREFETCH_BATCH_IOCTL,
        "ioctl_ret": int(ioctl_ret),
        "errno": 0,
        "rm_status": int(rm_status),
        "mode": mode_value,
        "mode_name": {
            UVM_EXPERT_RESERVED_PREFETCH_MODE_DRY_RUN: "dry_run",
            UVM_EXPERT_RESERVED_PREFETCH_MODE_RESOLVE: "resolve",
            UVM_EXPERT_RESERVED_PREFETCH_MODE_INVALIDATE: "invalidate",
            UVM_EXPERT_RESERVED_PREFETCH_MODE_RESERVED_PREFETCH:
                "reserved_prefetch",
        }.get(mode_value, str(mode)),
        "entry_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "requested_bytes": 0,
        "accepted_bytes": 0,
        "resolved_regions": 0,
        "assigned_slots": 0,
        "invalidated_slots": 0,
        "reserved_backing_bytes": 0,
        "ordinary_resident_bytes": 0,
        "pool_external_alloc_count": 0,
        "footprint_contract_broken": 0,
        "current_epoch": 0,
        "entries": [],
    }


def _normalize_expert_reserved_prefetch_assignments(
    region_ids: Sequence[int] | None,
    slot_ids: Sequence[int] | None,
    *,
    assignments: Sequence[dict[str, object]] | None,
    pool_id: int,
    slot_offsets: Sequence[int] | None,
    bytes_overrides: Sequence[int] | None,
    strict_footprint: bool,
) -> list[dict[str, int]]:
    normalized: list[dict[str, int]] = []
    if assignments is not None:
        source = assignments
        for index, entry in enumerate(source):
            try:
                flags = int(entry.get("flags", 0))
                if bool(entry.get("strict_footprint", strict_footprint)):
                    flags |= UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT
                normalized.append({
                    "region_id": int(entry.get("region_id", 0)),
                    "pool_id": int(entry.get("pool_id", pool_id)),
                    "slot_id": int(entry.get("slot_id", index)),
                    "flags": flags,
                    "slot_offset": int(entry.get("slot_offset", 0)),
                    "bytes_override": int(entry.get("bytes_override",
                                                    entry.get("bytes", 0))),
                })
            except (TypeError, ValueError, AttributeError):
                continue
        return normalized

    if region_ids is None:
        return normalized

    region_list = list(region_ids)
    slot_list = list(slot_ids) if slot_ids is not None else list(range(len(region_list)))
    offset_list = list(slot_offsets) if slot_offsets is not None else []
    bytes_list = list(bytes_overrides) if bytes_overrides is not None else []
    for index, region_id in enumerate(region_list):
        if index >= len(slot_list):
            break
        flags = (
            UVM_EXPERT_RESERVED_PREFETCH_ENTRY_FLAG_STRICT_FOOTPRINT
            if strict_footprint else 0
        )
        normalized.append({
            "region_id": int(region_id),
            "pool_id": int(pool_id),
            "slot_id": int(slot_list[index]),
            "flags": flags,
            "slot_offset": int(offset_list[index]) if index < len(offset_list) else 0,
            "bytes_override": int(bytes_list[index]) if index < len(bytes_list) else 0,
        })
    return normalized


def _fill_expert_reserved_prefetch_params(
    params: _ExpertReservedPrefetchBatchParams,
    *,
    mode_value: int,
    entries: Sequence[dict[str, int]],
    gpu_uuid: _PhaseFUuid,
    async_: bool,
) -> None:
    params.mode = mode_value
    params.count = len(entries)
    params.gpu_uuid = gpu_uuid
    params.async_ = 1 if async_ else 0
    for index, entry in enumerate(entries[:_UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES]):
        params.entries[index].region_id = int(entry.get("region_id", 0))
        params.entries[index].pool_id = int(entry.get("pool_id", 0))
        params.entries[index].slot_id = int(entry.get("slot_id", index))
        params.entries[index].flags = int(entry.get("flags", 0))
        params.entries[index].slot_offset = int(entry.get("slot_offset", 0))
        params.entries[index].bytes_override = int(entry.get("bytes_override", 0))


def _expert_reserved_prefetch_result_from_params(
    params: _ExpertReservedPrefetchBatchParams,
    *,
    mode: str | int,
    ret: int,
    saved_errno: int,
) -> dict[str, object]:
    entry_count = max(0, min(
        int(params.count),
        _UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES,
    ))
    result = _empty_expert_reserved_prefetch_result(
        mode=mode,
        ioctl_ret=ret,
        rm_status=int(params.rmStatus),
    )
    result.update({
        "ok": int(ret) == 0 and int(params.rmStatus) == 0,
        "errno": int(saved_errno),
        "mode": int(params.mode),
        "entry_count": entry_count,
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "requested_bytes": int(params.requested_bytes),
        "accepted_bytes": int(params.accepted_bytes),
        "resolved_regions": int(params.resolved_regions),
        "assigned_slots": int(params.assigned_slots),
        "invalidated_slots": int(params.invalidated_slots),
        "reserved_backing_bytes": int(params.reserved_backing_bytes),
        "ordinary_resident_bytes": int(params.ordinary_resident_bytes),
        "pool_external_alloc_count": int(params.pool_external_alloc_count),
        "footprint_contract_broken": int(params.footprint_contract_broken),
        "current_epoch": int(params.current_epoch),
        "entries": [
            _expert_reserved_prefetch_entry_to_dict(params.entries[index])
            for index in range(entry_count)
        ],
    })
    return result


def driver_expert_reserved_prefetch_regions(
    region_ids: Sequence[int] | None = None,
    slot_ids: Sequence[int] | None = None,
    *,
    assignments: Sequence[dict[str, object]] | None = None,
    pool_id: int = 0,
    slot_offsets: Sequence[int] | None = None,
    bytes_overrides: Sequence[int] | None = None,
    device: int = 0,
    async_: bool = True,
    strict_footprint: bool = True,
    mode: str | int = "resolve",
    prefer_existing: bool = True,
) -> dict[str, object]:
    """Bind Phase-F expert regions to Phase-1 reserved slots.

    Phase 2 validates and records region->slot assignment. The reserved_prefetch
    mode copies CPU-resident expert pages into reserved PMM chunks and installs
    readonly slot-backed GPU PTEs without attaching the chunks to ordinary UVM
    residency metadata.
    """
    mode_value = _expert_reserved_prefetch_mode_value(mode)
    normalized_entries = _normalize_expert_reserved_prefetch_assignments(
        region_ids,
        slot_ids,
        assignments=assignments,
        pool_id=pool_id,
        slot_offsets=slot_offsets,
        bytes_overrides=bytes_overrides,
        strict_footprint=strict_footprint,
    )
    if len(normalized_entries) > _UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES:
        result = _empty_expert_reserved_prefetch_result(mode=mode, ioctl_ret=-2)
        result["failure_count"] = len(normalized_entries)
        return result

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for expert reserved prefetch",
                     exc_info=True)
        result = _empty_expert_reserved_prefetch_result(mode=mode, ioctl_ret=-3)
        result["failure_count"] = len(normalized_entries)
        return result

    params = _ExpertReservedPrefetchBatchParams()
    _fill_expert_reserved_prefetch_params(
        params,
        mode_value=mode_value,
        entries=normalized_entries,
        gpu_uuid=gpu_uuid,
        async_=async_,
    )

    fd = -1
    try:
        fd = _open_or_adopt_initialized_uvm_fd(prefer_existing=prefer_existing)
        ctypes.set_errno(0)
        ret = _libc_ioctl(fd, _UVM_EXPERT_RESERVED_PREFETCH_BATCH_IOCTL, params)
        saved_errno = ctypes.get_errno()
        return _expert_reserved_prefetch_result_from_params(
            params,
            mode=mode,
            ret=int(ret),
            saved_errno=int(saved_errno),
        )
    except Exception:
        logger.debug("Expert reserved prefetch ioctl failed", exc_info=True)
        result = _empty_expert_reserved_prefetch_result(mode=mode, ioctl_ret=-4)
        result["failure_count"] = len(normalized_entries)
        return result
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def driver_prepare_ranges_remote_read_bg_deferred(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    """Prepare remote READ PTEs and mark background promotion pending."""
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED,
        device=device,
    )


def driver_prepare_ranges_remote_read(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    """Prepare remote READ PTEs without marking background promotion pending."""
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ,
        device=device,
    )


def driver_start_ranges_bg_promotion(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_START_BG_PROMOTION,
        device=device,
    )


def driver_gpu_only_read_dup_soft(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
    async_: bool = False,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT,
        device=device,
        async_=async_,
    )


def driver_expert_read_dup_fast(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
    async_: bool = False,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST,
        device=device,
        async_=async_,
    )


def driver_expert_read_dup_fast_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    async_: bool = False,
) -> int:
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return -1

    if not hasattr(lib, "uvm_driver_batch_expert_read_dup_fast_regions"):
        return -2

    valid_ids = [int(region_id) for region_id in region_ids if int(region_id) > 0]
    if not valid_ids:
        return 0

    total = 0
    for start in range(0, len(valid_ids), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = valid_ids[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        id_array = (ctypes.c_uint64 * len(chunk))(*chunk)
        try:
            result = int(
                lib.uvm_driver_batch_expert_read_dup_fast_regions(
                    id_array,
                    ctypes.c_size_t(len(chunk)),
                    ctypes.c_int(device),
                    ctypes.c_int(1 if async_ else 0),
                )
            )
        except Exception:
            logger.debug("Failed to expert-read-duplicate Phase F driver regions", exc_info=True)
            return -3
        if result < 0:
            return result
        total += result

    return total


def driver_expert_read_dup_fast_native_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    async_: bool = False,
) -> int:
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return -1

    if not hasattr(lib, "uvm_driver_batch_expert_read_dup_fast_native_regions"):
        return -2

    valid_ids = [int(region_id) for region_id in region_ids if int(region_id) > 0]
    if not valid_ids:
        return 0

    total = 0
    stream = torch.cuda.current_stream().cuda_stream
    for start in range(0, len(valid_ids), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = valid_ids[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        id_array = (ctypes.c_uint64 * len(chunk))(*chunk)
        try:
            result = int(
                lib.uvm_driver_batch_expert_read_dup_fast_native_regions(
                    id_array,
                    ctypes.c_size_t(len(chunk)),
                    ctypes.c_int(device),
                    ctypes.c_int(1 if async_ else 0),
                    ctypes.c_void_p(stream),
                )
            )
        except Exception:
            logger.debug("Failed to native expert-read-duplicate Phase F driver regions", exc_info=True)
            return -3
        if result < 0:
            return result
        total += result

    return total


def driver_pause_ranges_bg_promotion(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION,
        device=device,
    )


def driver_resume_ranges_bg_promotion(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION,
        device=device,
    )


def uvm_residency_tag_only_enabled() -> bool:
    raw = os.environ.get("VLLM_UVM_RESIDENCY_TAG_ONLY")
    return raw is not None and raw.strip().lower() in ("1", "true", "yes", "on")


def batch_prefetch_ranges_to_device(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
) -> int:
    """Batch prefetch raw UVM ranges to a GPU or CPU device.

    Returns the number of CUDA prefetch calls accepted by the allocator shim.
    """
    if uvm_residency_tag_only_enabled():
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    stream = torch.cuda.current_stream().cuda_stream
    return int(
        _uvm_lib.uvm_batch_prefetch(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(device),
            ctypes.c_void_p(stream),
        )
    )


def batch_prefetch_ranges_to_cpu(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> int:
    """Batch prefetch raw UVM ranges to CPU memory."""
    return batch_prefetch_ranges_to_device(ranges, device=-1)


def advise_range_preferred_location(ptr: int, size: int, device: int) -> bool:
    """
    Set cudaMemAdviseSetPreferredLocation for a raw UVM address range.

    Returns True when the advise call was issued. The allocator shim currently
    logs CUDA errors internally instead of returning them, so benchmark health
    and Stage I trace records are used for end-to-end validation.
    """
    if uvm_residency_tag_only_enabled():
        return False
    if _uvm_lib is None:
        return False
    if ptr <= 0 or size <= 0:
        return False
    _uvm_lib.uvm_advise(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        ctypes.c_int(2),  # cudaMemAdviseSetPreferredLocation
        ctypes.c_int(device),
    )
    return True


def batch_advise_ranges_preferred_location(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int,
) -> int:
    """Batch set cudaMemAdviseSetPreferredLocation for raw UVM ranges."""
    if uvm_residency_tag_only_enabled():
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    return int(
        _uvm_lib.uvm_batch_advise(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(2),  # cudaMemAdviseSetPreferredLocation
            ctypes.c_int(device),
        )
    )


def batch_advise_ranges_read_mostly(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
) -> int:
    """Batch set cudaMemAdviseSetReadMostly for raw UVM ranges."""
    if uvm_residency_tag_only_enabled():
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    return int(
        _uvm_lib.uvm_batch_advise(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(1),  # uvm_allocator shim: ReadMostly
            ctypes.c_int(device),
        )
    )


def batch_advise_ranges_accessed_by(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
) -> int:
    """Batch set cudaMemAdviseSetAccessedBy for raw UVM ranges."""
    if uvm_residency_tag_only_enabled():
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    return int(
        _uvm_lib.uvm_batch_advise(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(3),  # uvm_allocator shim: AccessedBy
            ctypes.c_int(device),
        )
    )


def driver_batch_prefetch_ranges_to_device(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
    *,
    async_: bool = False,
) -> int:
    """Try Phase F driver batch prefetch. Negative return means unavailable."""
    if uvm_residency_tag_only_enabled():
        return 0
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    range_list = list(ranges)
    if not range_list:
        return 0
    total = 0
    for start in range(0, len(range_list), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = range_list[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        if device < 0:
            result = int(
                _uvm_lib.uvm_driver_batch_prefetch_cpu(
                    ptr_array,
                    size_array,
                    ctypes.c_size_t(count),
                    ctypes.c_int(1 if async_ else 0),
                )
            )
        else:
            result = int(
                _uvm_lib.uvm_driver_batch_prefetch_gpu(
                    ptr_array,
                    size_array,
                    ctypes.c_size_t(count),
                    ctypes.c_int(device),
                    ctypes.c_int(1 if async_ else 0),
                )
            )
        if result < 0:
            return result
        total += result
    return total


def driver_batch_advise_ranges_cpu(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> int:
    """Try Phase F driver batch CPU preferred-location advise."""
    if uvm_residency_tag_only_enabled():
        return 0
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    range_list = list(ranges)
    if not range_list:
        return 0
    total = 0
    for start in range(0, len(range_list), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = range_list[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        result = int(_uvm_lib.uvm_driver_batch_advise_cpu(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
        ))
        if result < 0:
            return result
        total += result
    return total


def driver_batch_query_ranges_residency(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
) -> int:
    """Try Phase F driver batch residency query. Returns successes or negative."""
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    range_list = list(ranges)
    if not range_list:
        return 0
    total = 0
    for start in range(0, len(range_list), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = range_list[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        result = int(_uvm_lib.uvm_driver_batch_query_residency(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(device),
        ))
        if result < 0:
            return result
        total += result
    return total


def driver_batch_query_ranges_residency_bytes(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
) -> dict[str, int | bool]:
    """Query page-granular real UVM residency bytes for managed ranges."""
    result: dict[str, int | bool] = {
        "ok": False,
        "status": -1,
        "queried_bytes": 0,
        "gpu_bytes": 0,
        "cpu_bytes": 0,
        "any_bytes": 0,
        "unknown_bytes": 0,
    }
    if not _uvm_enabled or _uvm_lib is None:
        return result

    range_list = list(ranges)
    if not range_list:
        result["ok"] = True
        result["status"] = 0
        return result

    final_status = 0
    total_gpu_bytes = 0
    total_cpu_bytes = 0
    total_any_bytes = 0
    total_unknown_bytes = 0
    total_queried_bytes = 0

    for start in range(0, len(range_list), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = range_list[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        gpu_bytes = ctypes.c_uint64(0)
        cpu_bytes = ctypes.c_uint64(0)
        any_bytes = ctypes.c_uint64(0)
        unknown_bytes = ctypes.c_uint64(0)
        queried_bytes = ctypes.c_uint64(0)
        status = int(
            _uvm_lib.uvm_driver_batch_query_residency_bytes(
                ptr_array,
                size_array,
                ctypes.c_size_t(count),
                ctypes.c_int(device),
                ctypes.byref(gpu_bytes),
                ctypes.byref(cpu_bytes),
                ctypes.byref(any_bytes),
                ctypes.byref(unknown_bytes),
                ctypes.byref(queried_bytes),
            )
        )
        if status < 0:
            final_status = status
            break
        final_status += status
        total_gpu_bytes += int(gpu_bytes.value)
        total_cpu_bytes += int(cpu_bytes.value)
        total_any_bytes += int(any_bytes.value)
        total_unknown_bytes += int(unknown_bytes.value)
        total_queried_bytes += int(queried_bytes.value)

    result.update(
        {
            "ok": final_status >= 0,
            "status": final_status,
            "queried_bytes": total_queried_bytes,
            "gpu_bytes": total_gpu_bytes,
            "cpu_bytes": total_cpu_bytes,
            "any_bytes": total_any_bytes,
            "unknown_bytes": total_unknown_bytes,
        }
    )
    return result


def driver_register_region(
    *,
    region_id: int,
    ptr: int,
    nbytes: int,
    pool: int,
    layer_id: int = 0,
    expert_id: int = 0,
    role: int = UVM_REGION_ROLE_UNKNOWN,
    flags: int = 0,
) -> int:
    """Register a tagged managed range in the Phase F driver region table."""
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    if region_id <= 0 or ptr <= 0 or nbytes <= 0:
        return -2
    try:
        return int(
            _uvm_lib.uvm_driver_phase_f_register_region(
                ctypes.c_uint64(region_id),
                ctypes.c_uint64(ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_uint32(max(int(pool), 0)),
                ctypes.c_uint32(max(int(layer_id), 0)),
                ctypes.c_uint32(max(int(expert_id), 0)),
                ctypes.c_uint32(max(int(role), 0)),
                ctypes.c_uint32(max(int(flags), 0)),
            )
        )
    except Exception:
        logger.debug("Failed to register Phase F driver region", exc_info=True)
        return -3


def driver_unregister_region(region_id: int) -> int:
    """Unregister a Phase F driver region by numeric region id."""
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    if region_id <= 0:
        return -2
    try:
        return int(
            _uvm_lib.uvm_driver_phase_f_unregister_region(
                ctypes.c_uint64(region_id)
            )
        )
    except Exception:
        logger.debug("Failed to unregister Phase F driver region", exc_info=True)
        return -3


def driver_batch_query_regions_residency(
    region_ids: Sequence[int],
    device: int = 0,
) -> int:
    """Query residency by Phase F registered region ids."""
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    valid_ids = [int(region_id) for region_id in region_ids if int(region_id) > 0]
    if not valid_ids:
        return 0
    total = 0
    for start in range(0, len(valid_ids), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = valid_ids[start:start + _UVM_DRIVER_BATCH_MAX_RANGES]
        id_array = (ctypes.c_uint64 * len(chunk))(*chunk)
        try:
            result = int(
                _uvm_lib.uvm_driver_batch_query_residency_regions(
                    id_array,
                    ctypes.c_size_t(len(chunk)),
                    ctypes.c_int(device),
                )
            )
        except Exception:
            logger.debug("Failed to query Phase F driver regions", exc_info=True)
            return -3
        if result < 0:
            return result
        total += result
    return total


def driver_register_query_region(
    *,
    region_id: int,
    ptr: int,
    nbytes: int,
    device: int = 0,
) -> int:
    """Register, query, and unregister a region using the C++ smoke helper."""
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    if region_id <= 0 or ptr <= 0 or nbytes <= 0:
        return -2
    try:
        return int(
            _uvm_lib.uvm_driver_phase_f_register_query_region(
                ctypes.c_uint64(region_id),
                ctypes.c_uint64(ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(device),
            )
        )
    except Exception:
        logger.debug("Failed to register/query Phase F driver region",
                     exc_info=True)
        return -3


def driver_set_pool_budget(
    *,
    pool: int,
    budget_bytes: int,
    gpu_id: int = 0,
    mode: int = UVM_POOL_BUDGET_MODE_TRACE_ONLY,
    flags: int = 0,
) -> int:
    """Set a Stage N per-pool GPU residency budget in the driver."""
    if not has_pool_budget_ioctl():
        return -1
    if pool < 0 or budget_bytes < 0:
        return -2
    try:
        return int(
            _uvm_lib.uvm_driver_phase_n_set_pool_budget(
                ctypes.c_uint32(int(pool)),
                ctypes.c_uint32(max(int(gpu_id), 0)),
                ctypes.c_uint64(int(budget_bytes)),
                ctypes.c_uint32(max(int(mode), 0)),
                ctypes.c_uint32(max(int(flags), 0)),
            )
        )
    except Exception:
        logger.debug("Failed to set Phase N driver pool budget", exc_info=True)
        return -3


def driver_get_pool_state(pool: int, gpu_id: int = 0) -> dict[str, int] | None:
    """Return Stage N driver pool state, or None when unavailable/failing."""
    if not has_pool_budget_ioctl() or pool < 0:
        return None
    values = (ctypes.c_uint64 * 36)()
    try:
        status = int(
            _uvm_lib.uvm_driver_phase_n_get_pool_state(
                ctypes.c_uint32(int(pool)),
                ctypes.c_uint32(max(int(gpu_id), 0)),
                values,
                ctypes.c_size_t(len(values)),
            )
        )
    except Exception:
        logger.debug("Failed to query Phase N driver pool state", exc_info=True)
        return None
    if status < 0:
        return None
    keys = (
        "budget_bytes",
        "resident_gpu_bytes",
        "resident_cpu_bytes",
        "reclaim_bytes",
        "protected_bytes",
        "evictable_bytes",
        "over_budget_bytes",
        "reclaim_target_bytes",
        "victim_candidate_bytes",
        "victim_candidate_regions",
        "selected_victim_region_id",
        "selected_victim_bytes",
        "victim_count",
        "metadata_only",
        "victim_region_id_0",
        "victim_region_id_1",
        "victim_region_id_2",
        "victim_region_id_3",
        "victim_region_id_4",
        "victim_region_id_5",
        "victim_region_id_6",
        "victim_region_id_7",
        "victim_bytes_0",
        "victim_bytes_1",
        "victim_bytes_2",
        "victim_bytes_3",
        "victim_bytes_4",
        "victim_bytes_5",
        "victim_bytes_6",
        "victim_bytes_7",
        "over_budget_events",
        "reclaim_success_events",
        "reclaim_failed_events",
        "accounting_corrections",
        "mode",
        "flags",
    )
    return {key: int(values[i]) for i, key in enumerate(keys)}


def driver_reclaim_pool(
    pool: int,
    *,
    gpu_id: int = 0,
    target_bytes: int = 0,
    current_epoch: int = 0,
    cooldown_epochs: int = 0,
    max_victims: int = 8,
    async_reclaim: bool = False,
    flags: int = 0,
) -> dict[str, int] | None:
    """Trigger Stage N driver-owned soft reclaim for one pool."""
    if not has_pool_reclaim_ioctl() or pool < 0:
        return None
    values = (ctypes.c_uint64 * 45)()
    try:
        status = int(
            _uvm_lib.uvm_driver_phase_n_reclaim_pool(
                ctypes.c_uint32(int(pool)),
                ctypes.c_uint32(max(int(gpu_id), 0)),
                ctypes.c_uint64(max(int(target_bytes), 0)),
                ctypes.c_uint64(max(int(current_epoch), 0)),
                ctypes.c_uint64(max(int(cooldown_epochs), 0)),
                ctypes.c_uint32(max(int(max_victims), 0)),
                ctypes.c_uint32(1 if async_reclaim else 0),
                ctypes.c_uint32(max(int(flags), 0)),
                values,
                ctypes.c_size_t(len(values)),
            )
        )
    except Exception:
        logger.debug("Failed to trigger Phase N driver pool reclaim",
                     exc_info=True)
        return None
    if status < 0:
        return None

    keys = (
        "attempted_regions",
        "success_regions",
        "failure_regions",
        "victim_count",
        "budget_bytes",
        "resident_gpu_bytes_before",
        "over_budget_bytes",
        "reclaim_target_bytes",
        "attempted_bytes",
        "reclaimed_bytes",
        "failed_bytes",
        "victim_candidate_bytes",
        "victim_candidate_regions",
        "cooldown_skipped_bytes",
        "cooldown_skipped_regions",
        "victim_region_id_0",
        "victim_region_id_1",
        "victim_region_id_2",
        "victim_region_id_3",
        "victim_region_id_4",
        "victim_region_id_5",
        "victim_region_id_6",
        "victim_region_id_7",
        "victim_bytes_0",
        "victim_bytes_1",
        "victim_bytes_2",
        "victim_bytes_3",
        "victim_bytes_4",
        "victim_bytes_5",
        "victim_bytes_6",
        "victim_bytes_7",
        "victim_status_0",
        "victim_status_1",
        "victim_status_2",
        "victim_status_3",
        "victim_status_4",
        "victim_status_5",
        "victim_status_6",
        "victim_status_7",
        "reclaim_success_events",
        "reclaim_failed_events",
        "mode",
        "pool_flags",
        "rmStatus",
        "flags",
    )
    result = {key: int(values[i]) for i, key in enumerate(keys)}
    victim_count = int(result.get("victim_count", 0))
    result["victims"] = [
        {
            "region_id": int(result.get(f"victim_region_id_{i}", 0)),
            "bytes": int(result.get(f"victim_bytes_{i}", 0)),
            "status": int(result.get(f"victim_status_{i}", 0)),
        }
        for i in range(min(max(victim_count, 0), 8))
        if int(result.get(f"victim_region_id_{i}", 0)) != 0
    ]
    return result


def driver_update_region_state(
    *,
    region_id: int,
    state: int = UVM_REGION_STATE_UNKNOWN,
    flags_set: int = 0,
    flags_clear: int = 0,
    priority: int = 0,
    last_access_epoch: int = 0,
) -> int:
    """Update Stage N region state metadata in the driver."""
    if not has_region_state_ioctl():
        return -1
    if region_id <= 0:
        return -2
    try:
        return int(
            _uvm_lib.uvm_driver_phase_n_update_region_state(
                ctypes.c_uint64(int(region_id)),
                ctypes.c_uint32(max(int(state), 0)),
                ctypes.c_uint32(max(int(flags_set), 0)),
                ctypes.c_uint32(max(int(flags_clear), 0)),
                ctypes.c_uint32(max(int(priority), 0)),
                ctypes.c_uint64(max(int(last_access_epoch), 0)),
            )
        )
    except Exception:
        logger.debug("Failed to update Phase N driver region state",
                     exc_info=True)
        return -3


def _driver_region_values_to_dict(values: ctypes.Array) -> dict[str, int]:
    keys = (
        "region_id",
        "base",
        "length",
        "pool",
        "layer_id",
        "expert_id",
        "role",
        "flags",
        "state",
        "priority",
        "last_access_epoch",
    )
    return {key: int(values[i]) for i, key in enumerate(keys)}


def driver_query_region(region_id: int) -> dict[str, int] | None:
    """Query Stage N region metadata by region id."""
    if not has_region_state_ioctl() or region_id <= 0:
        return None
    values = (ctypes.c_uint64 * 11)()
    try:
        status = int(
            _uvm_lib.uvm_driver_phase_n_query_region(
                ctypes.c_uint64(int(region_id)),
                values,
                ctypes.c_size_t(len(values)),
            )
        )
    except Exception:
        logger.debug("Failed to query Phase N driver region", exc_info=True)
        return None
    if status < 0:
        return None
    return _driver_region_values_to_dict(values)


def driver_query_region_by_va(address: int) -> dict[str, int] | None:
    """Query Stage N region metadata by a virtual address."""
    if not has_region_state_ioctl() or address <= 0:
        return None
    values = (ctypes.c_uint64 * 11)()
    try:
        status = int(
            _uvm_lib.uvm_driver_phase_n_query_region_by_va(
                ctypes.c_uint64(int(address)),
                values,
                ctypes.c_size_t(len(values)),
            )
        )
    except Exception:
        logger.debug("Failed to query Phase N driver region by VA",
                     exc_info=True)
        return None
    if status < 0:
        return None
    return _driver_region_values_to_dict(values)


def has_driver_region_ioctl() -> bool:
    return _uvm_enabled and _uvm_lib is not None and all(
        hasattr(_uvm_lib, name)
        for name in (
            "uvm_driver_phase_f_register_region",
            "uvm_driver_phase_f_unregister_region",
            "uvm_driver_batch_query_residency_regions",
        )
    )


def has_pool_budget_ioctl() -> bool:
    return _uvm_enabled and _uvm_lib is not None and all(
        hasattr(_uvm_lib, name)
        for name in (
            "uvm_driver_phase_n_set_pool_budget",
            "uvm_driver_phase_n_get_pool_state",
        )
    )


def has_pool_reclaim_ioctl() -> bool:
    return has_pool_budget_ioctl() and hasattr(
        _uvm_lib, "uvm_driver_phase_n_reclaim_pool")


def has_region_state_ioctl() -> bool:
    return _uvm_enabled and _uvm_lib is not None and all(
        hasattr(_uvm_lib, name)
        for name in (
            "uvm_driver_phase_n_update_region_state",
            "uvm_driver_phase_n_query_region",
            "uvm_driver_phase_n_query_region_by_va",
        )
    )


def batch_prefetch_ranges_to_device_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
    *,
    executor: str = "cpp_batch",
) -> tuple[int, str, int]:
    """Batch prefetch using cpp_batch, driver_batch, or auto fallback.

    Returns (successes, executor_used, driver_result). driver_result is negative
    when the Phase F driver path was unavailable or failed.
    """
    if executor == "driver_batch":
        driver_result = driver_batch_prefetch_ranges_to_device(ranges, device)
        return max(driver_result, 0), "driver_batch", driver_result
    if executor == "auto":
        driver_result = driver_batch_prefetch_ranges_to_device(ranges, device)
        if driver_result >= 0:
            return driver_result, "driver_batch", driver_result
        return (
            batch_prefetch_ranges_to_device(ranges, device),
            "cpp_batch_fallback",
            driver_result,
        )
    return batch_prefetch_ranges_to_device(ranges, device), "cpp_batch", 0


def batch_offload_ranges_to_cpu_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
    executor: str = "cpp_batch",
) -> tuple[int, str, int, int]:
    """Batch CPU offload using cpp_batch, driver_batch, or auto fallback.

    Returns (successes, executor_used, driver_advise_result,
    driver_prefetch_result).
    """
    if executor == "driver_batch":
        advise_result = driver_batch_advise_ranges_cpu(ranges)
        prefetch_result = driver_batch_prefetch_ranges_to_device(ranges, -1)
        successes = min(max(advise_result, 0), max(prefetch_result, 0))
        if successes > 0 and sync:
            try:
                torch.cuda.synchronize()
            except Exception:
                logger.debug("Failed to synchronize after driver CPU offload",
                             exc_info=True)
        return successes, "driver_batch", advise_result, prefetch_result
    if executor == "auto":
        advise_result = driver_batch_advise_ranges_cpu(ranges)
        prefetch_result = driver_batch_prefetch_ranges_to_device(ranges, -1)
        if advise_result >= 0 and prefetch_result >= 0:
            successes = min(advise_result, prefetch_result)
            if successes > 0 and sync:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    logger.debug("Failed to synchronize after driver CPU offload",
                                 exc_info=True)
            return successes, "driver_batch", advise_result, prefetch_result
        return (
            batch_offload_ranges_to_cpu(ranges, sync=sync),
            "cpp_batch_fallback",
            advise_result,
            prefetch_result,
        )
    return batch_offload_ranges_to_cpu(ranges, sync=sync), "cpp_batch", 0, 0


def batch_advise_ranges_cpu_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    executor: str = "cpp_batch",
) -> tuple[int, str, int]:
    """Batch CPU preferred-location advise with optional Phase F driver path."""
    if executor == "driver_batch":
        driver_result = driver_batch_advise_ranges_cpu(ranges)
        return max(driver_result, 0), "driver_batch", driver_result
    if executor == "auto":
        driver_result = driver_batch_advise_ranges_cpu(ranges)
        if driver_result >= 0:
            return driver_result, "driver_batch", driver_result
        return (
            batch_advise_ranges_preferred_location(ranges, -1),
            "cpp_batch_fallback",
            driver_result,
        )
    return batch_advise_ranges_preferred_location(ranges, -1), "cpp_batch", 0


def batch_prepare_ranges_for_remote_read_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
    executor: str = "cpp_batch",
    device: int = 0,
    accessed_by: bool = False,
) -> tuple[int, str, int, int, int, int]:
    """Prepare UVM ranges for CPU-resident GPU remote reads.

    The range is marked ReadMostly, moved to CPU/preferred CPU, and optionally
    tagged AccessedBy for the GPU. With the modified driver remote-first path
    enabled, the ReadMostly + CPU-resident state lets first GPU reads install
    remote CPU mappings instead of migrating the page to GPU.

    Returns (successes, executor_used, readmostly_result, accessed_by_result,
    driver_advise_result, driver_prefetch_result).
    """
    if not ranges:
        return 0, executor, 0, 0, 0, 0

    readmostly_result = batch_advise_ranges_read_mostly(ranges, device=device)
    (
        offload_successes,
        executor_used,
        driver_advise_result,
        driver_prefetch_result,
    ) = batch_offload_ranges_to_cpu_with_executor(
        ranges,
        sync=sync,
        executor=executor,
    )
    accessed_by_result = 0
    if accessed_by:
        accessed_by_result = batch_advise_ranges_accessed_by(
            ranges,
            device=device,
        )

    successes = min(readmostly_result, offload_successes)
    if accessed_by:
        successes = min(successes, accessed_by_result)
    return (
        successes,
        executor_used,
        readmostly_result,
        accessed_by_result,
        driver_advise_result,
        driver_prefetch_result,
    )


def batch_prepare_ranges_for_native_remote_access_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
    executor: str = "cpp_batch",
    device: int = 0,
) -> tuple[int, str, int, int, int]:
    """Prepare UVM ranges for native pure GPU remote access.

    This path matches the microbenchmark's ``uvm_pure_remote`` mechanism:
    place managed ranges on CPU, set PreferredLocation(CPU), then set
    AccessedBy(GPU) so native UVM can pre-map GPU->CPU remote READ mappings.
    It deliberately does not use ReadMostly, modified-driver remote-first, or
    repository-specific driver-batch ioctls.

    Returns (successes, executor_used, accessed_by_result,
    prefetch_cpu_successes, preferred_cpu_result).
    """
    if not ranges:
        return 0, executor, 0, 0, 0

    # Keep this experiment on native CUDA UVM APIs. The driver-batch executor
    # can use repository-specific Phase F ioctls, which would change the test.
    prefetch_cpu_successes = batch_prefetch_ranges_to_device(ranges, -1)
    executor_used = "cpp_batch_native_cuda"
    if prefetch_cpu_successes > 0 and sync:
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.debug("Failed to synchronize after native remote CPU prefetch",
                         exc_info=True)

    preferred_cpu_result = batch_advise_ranges_preferred_location(
        ranges,
        device=-1,
    )
    accessed_by_result = batch_advise_ranges_accessed_by(
        ranges,
        device=device,
    )
    successes = min(
        prefetch_cpu_successes,
        preferred_cpu_result,
        accessed_by_result,
    )
    return (
        successes,
        executor_used,
        accessed_by_result,
        prefetch_cpu_successes,
        preferred_cpu_result,
    )


def offload_range_to_cpu(
    ptr: int,
    size: int,
    *,
    sync: bool = False,
) -> bool:
    """Set CPU preferred location and prefetch a raw UVM range to CPU.

    This is the common Stage G/M weight enforcement primitive. It still uses
    CUDA/UVM as the execution layer, but keeps the semantic decision in vLLM.
    """
    advise_success = advise_range_preferred_location(ptr, size, -1)
    prefetch_success = prefetch_range_to_cpu(ptr, size)
    success = advise_success and prefetch_success
    if success and sync:
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.debug("Failed to synchronize after UVM CPU offload",
                         exc_info=True)
    return success


def batch_offload_ranges_to_cpu(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
) -> int:
    """Batch set CPU preferred location and prefetch UVM ranges to CPU."""
    if not ranges:
        return 0
    advise_successes = batch_advise_ranges_preferred_location(ranges, -1)
    prefetch_successes = batch_prefetch_ranges_to_cpu(ranges)
    successes = min(advise_successes, prefetch_successes)
    if successes > 0 and sync:
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.debug("Failed to synchronize after batch UVM CPU offload",
                         exc_info=True)
    return successes


def set_preferred_location(tensor: torch.Tensor, device: int) -> None:
    """
    Set the preferred location for a UVM tensor.

    Args:
        tensor: The tensor to configure
        device: Preferred device (-1 for CPU, >= 0 for GPU)
    """
    if not _uvm_enabled or _uvm_lib is None:
        return

    if not tensor.is_cuda:
        return

    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    advise_range_preferred_location(ptr, size, device)


@contextmanager
def uvm_memory_tracking():
    """
    Context manager for tracking UVM memory usage within a block.

    Example:
        with uvm_memory_tracking() as tracker:
            # Do some operations
            pass
        print(f"Peak memory: {tracker['peak_bytes'] / 1e9:.2f} GB")
    """
    if not _uvm_enabled:
        yield {"enabled": False}
        return

    reset_uvm_peak_stats()
    stats_before = get_uvm_stats()

    result = {
        "enabled": True,
        "start_bytes": stats_before["allocated_bytes"],
    }

    try:
        yield result
    finally:
        stats_after = get_uvm_stats()
        result["end_bytes"] = stats_after["allocated_bytes"]
        result["peak_bytes"] = stats_after["peak_bytes"]
        result["delta_bytes"] = (
            stats_after["allocated_bytes"] - stats_before["allocated_bytes"]
        )
        result["allocations"] = (
            stats_after["num_allocs"] - stats_before["num_allocs"]
        )


def maybe_enable_uvm_from_env() -> bool:
    """
    Enable UVM allocator if VLLM_USE_UVM environment variable is set.

    This is called automatically during vLLM initialization.

    Returns:
        True if UVM was enabled, False otherwise.
    """
    if os.environ.get("VLLM_USE_UVM", "0").lower() in ("1", "true", "yes"):
        enable_prefetch = os.environ.get(
            "VLLM_UVM_PREFETCH", "0"
        ).lower() in ("1", "true", "yes")
        verbose = os.environ.get(
            "VLLM_UVM_VERBOSE", "0"
        ).lower() in ("1", "true", "yes")

        try:
            enable_uvm_allocator(enable_prefetch=enable_prefetch, verbose=verbose)
            return True
        except Exception as e:
            logger.error("Failed to enable UVM allocator from environment: %s", e)
            return False

    return False
