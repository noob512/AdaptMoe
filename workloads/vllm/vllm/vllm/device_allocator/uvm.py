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
import threading
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
_uvm_local_prefetch_stack: list[bool] = []
_uvm_phase_stack: list[str] = []
_UVM_DRIVER_BATCH_MAX_RANGES = 64
_UVM_INITIALIZE_IOCTL = 0x30000001
_UVM_PHASE_F_BATCH_EXECUTOR_IOCTL = 81
_UVM_PHASE_F_MAX_RANGES = 64
_UVM_MOE_SLOT_POOL_QUERY_IOCTL = 92
_UVM_EXPERT_RESERVED_POOL_IOCTL = 98
_UVM_EXPERT_RESERVED_PREFETCH_BATCH_IOCTL = 99
_UVM_EXPERT_OFFLOAD_RELEASE_IOCTL = 100
_UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES = 128
_UVM_EXPERT_RESERVED_POOL_MAX_ENTRIES = 128
_UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES = 128
_UVM_EXPERT_RESERVED_PREFETCH_STATS_COUNT = 16
_UVM_PAGE_SIZE = 0

UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ = 5
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG = 6
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED = 7
UVM_PHASE_F_ACTION_START_BG_PROMOTION = 8
UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION = 16
UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION = 17

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
UVM_REGION_FLAG_EXPERT_ENVELOPE = 1 << 15

UVM_REGION_STATE_UNKNOWN = 0
UVM_REGION_STATE_COLD = 1
UVM_REGION_STATE_WARM = 2
UVM_REGION_STATE_HOT = 3
UVM_REGION_STATE_ACTIVE = 4
UVM_REGION_STATE_INACTIVE = 5

UVM_POOL_BUDGET_MODE_TRACE_ONLY = 0
UVM_POOL_BUDGET_MODE_SOFT = 1
UVM_POOL_BUDGET_MODE_ENFORCE = 2



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

UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT = 1 << 0
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_STRICT_2M_BLOCKS = 1 << 1
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K = 1 << 2
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ = 1 << 3
UVM_EXPERT_REMOTE_PREFETCH_ENVELOPE_BYTES = 512 * 1024


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


class _ExpertOffloadReleaseParams(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("gpu_uuid", _PhaseFUuid),
        ("flags", ctypes.c_uint32),
        ("max_blocks", ctypes.c_uint32),
        ("pool", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("block_count", ctypes.c_uint64),
        ("attempted_blocks", ctypes.c_uint32),
        ("success_blocks", ctypes.c_uint32),
        ("failure_blocks", ctypes.c_uint32),
        ("skipped_blocks", ctypes.c_uint32),
        ("attempted_bytes", ctypes.c_uint64),
        ("released_bytes", ctypes.c_uint64),
        ("failed_bytes", ctypes.c_uint64),
        ("remote_map_blocks", ctypes.c_uint32),
        ("remote_map_failure_blocks", ctypes.c_uint32),
        ("cpu_resident_bytes", ctypes.c_uint64),
        ("remote_mapped_bytes", ctypes.c_uint64),
        ("newly_remote_mapped_bytes", ctypes.c_uint64),
        ("first_failed_block", ctypes.c_uint64),
        ("first_error", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _ExpertReservedPrefetchHelperBuffers:
    def __init__(self) -> None:
        max_entries = _UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES
        self.region_ids = (ctypes.c_uint64 * max_entries)()
        self.pool_ids = (ctypes.c_uint32 * max_entries)()
        self.slot_ids = (ctypes.c_uint32 * max_entries)()
        self.slot_offsets = (ctypes.c_uint64 * max_entries)()
        self.bytes_overrides = (ctypes.c_uint64 * max_entries)()
        self.flags = (ctypes.c_uint32 * max_entries)()
        self.stats = (ctypes.c_uint64 *
                      _UVM_EXPERT_RESERVED_PREFETCH_STATS_COUNT)()
        self.out_entries = (
            _ExpertReservedPrefetchEntry *
            _UVM_EXPERT_RESERVED_PREFETCH_MAX_ENTRIES
        )()


_expert_reserved_prefetch_tls = threading.local()


def _expert_reserved_prefetch_helper_buffers(
) -> _ExpertReservedPrefetchHelperBuffers:
    buffers = getattr(_expert_reserved_prefetch_tls, "buffers", None)
    if buffers is None:
        buffers = _ExpertReservedPrefetchHelperBuffers()
        _expert_reserved_prefetch_tls.buffers = buffers
    return buffers


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
    if hasattr(lib, "uvm_driver_batch_query_residency_bytes_nowait"):
        lib.uvm_driver_batch_query_residency_bytes_nowait.restype = ctypes.c_int
        lib.uvm_driver_batch_query_residency_bytes_nowait.argtypes = (
            lib.uvm_driver_batch_query_residency_bytes.argtypes
        )

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






    if hasattr(lib, "uvm_driver_expert_reserved_prefetch_regions"):
        lib.uvm_driver_expert_reserved_prefetch_regions.restype = ctypes.c_int
        lib.uvm_driver_expert_reserved_prefetch_regions.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(_ExpertReservedPrefetchEntry),
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
    _uvm_lib.uvm_set_prefetch(1 if _uvm_local_prefetch_enabled() else 0)
    _uvm_lib.uvm_set_verbose(0)
    return _uvm_lib




def _uvm_local_prefetch_enabled() -> bool:
    return _uvm_local_prefetch_stack[-1] if _uvm_local_prefetch_stack else False


@contextmanager
def uvm_local_memory_pool(enable_prefetch: bool | None = None):
    """Temporarily allocate CUDA tensors from cudaMallocManaged.

    Unlike ``enable_uvm_allocator()``, this does not call
    ``change_current_allocator``. Only allocations inside this context use the
    UVM pluggable allocator, which lets CPU-offload experiments replace just
    the offloaded weight tensors.

    If ``enable_prefetch`` is omitted, the current nested prefetch state is
    inherited. This keeps outer pre-service UVM scopes intact even when helper
    code opens its own local pool.
    """
    global _uvm_local_allocator, _uvm_local_mem_pool
    if enable_prefetch is None:
        enable_prefetch = _uvm_local_prefetch_enabled()
    _uvm_local_prefetch_stack.append(bool(enable_prefetch))
    try:
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

        if _uvm_lib is not None:
            _uvm_lib.uvm_set_prefetch(1 if enable_prefetch else 0)

        with torch.cuda.memory.use_mem_pool(_uvm_local_mem_pool):
            yield
    finally:
        if _uvm_local_prefetch_stack:
            _uvm_local_prefetch_stack.pop()
        if _uvm_lib is not None:
            _uvm_lib.uvm_set_prefetch(
                1 if _uvm_local_prefetch_enabled() else 0
            )


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
    return batch_prefetch_ranges_to_device(((ptr, size),), device=device) == 1




def probe_driver_capable_range(ptr: int, size: int) -> tuple[bool, str, int]:
    """Return whether a raw range is CUDA managed and suitable for driver ioctls.

    The Phase F driver executor only works on ranges visible to UVM as managed
    VA blocks. This probe records the CUDA runtime's pointer classification so
    weight traces can distinguish non-managed tensors from driver plumbing bugs.
    """
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return False, "uvm_library_unavailable", -2
    if ptr <= 0 or size <= 0:
        return False, "invalid_range", -3
    try:
        result = int(
            lib.uvm_is_managed_pointer(
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




def _expert_offload_release_entry_to_dict(
    params: _ExpertOffloadReleaseParams,
    *,
    ret: int,
    saved_errno: int,
    fd_source: str | None = None,
) -> dict[str, object]:
    raw_ok = int(ret) == 0 and int(params.rmStatus) == 0
    remote_read = bool(
        int(params.flags) & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ
    )
    effective_ok = raw_ok and (
        int(params.block_count) > 0
        or int(params.attempted_blocks) > 0
        or int(params.skipped_blocks) > 0
        or int(params.released_bytes) > 0
    )
    if remote_read:
        effective_ok = (
            effective_ok
            and int(params.remote_map_failure_blocks) == 0
            and int(params.cpu_resident_bytes) >= int(params.length)
            and int(params.remote_mapped_bytes) >= int(params.length)
        )
    reason = None
    if raw_ok and not effective_ok:
        reason = (
            "remote_mapping_incomplete"
            if remote_read
            else "no_va_blocks"
        )
    elif not raw_ok:
        reason = "ioctl_or_rm_error"
    return {
        "ok": effective_ok,
        "raw_ok": raw_ok,
        "reason": reason,
        "fd_source": fd_source,
        "ioctl_ret": int(ret),
        "errno": int(saved_errno),
        "rm_status": int(params.rmStatus),
        "flags": int(params.flags),
        "remote_read": remote_read,
        "region_id": int(params.region_id),
        "pool": int(params.pool),
        "layer_id": int(params.layer_id),
        "expert_id": int(params.expert_id),
        "role": int(params.role),
        "base": int(params.base),
        "length": int(params.length),
        "block_count": int(params.block_count),
        "attempted_blocks": int(params.attempted_blocks),
        "success_blocks": int(params.success_blocks),
        "failure_blocks": int(params.failure_blocks),
        "skipped_blocks": int(params.skipped_blocks),
        "attempted_bytes": int(params.attempted_bytes),
        "released_bytes": int(params.released_bytes),
        "failed_bytes": int(params.failed_bytes),
        "remote_map_blocks": int(params.remote_map_blocks),
        "remote_map_failure_blocks": int(params.remote_map_failure_blocks),
        "cpu_resident_bytes": int(params.cpu_resident_bytes),
        "remote_mapped_bytes": int(params.remote_mapped_bytes),
        "newly_remote_mapped_bytes": int(params.newly_remote_mapped_bytes),
        "first_failed_block": int(params.first_failed_block),
        "first_error": int(params.first_error),
    }


def driver_expert_offload_release_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    require_expert: bool = True,
    strict_2m_blocks: bool = False,
    subchunk_64k: bool = False,
    remote_read: bool = False,
    max_blocks: int = 0,
    prefer_existing: bool = True,
    return_entries: bool = True,
) -> dict[str, object]:
    """Targeted offload registered expert regions and release GPU PMM chunks.

    The region ids are Phase-F driver region ids, not Python residency keys.
    The managed VA ranges remain allocated. With ``remote_read=True``, each
    released range is also synchronously mapped read-only from the GPU to its
    CPU backing before the ioctl commits success.
    """
    normalized: list[int] = []
    for region_id in region_ids:
        try:
            value = int(region_id)
        except (TypeError, ValueError):
            continue
        if value > 0:
            normalized.append(value)
    result: dict[str, object] = {
        "ok": False,
        "ioctl": _UVM_EXPERT_OFFLOAD_RELEASE_IOCTL,
        "entry_count": len(normalized),
        "success_count": 0,
        "failure_count": 0,
        "skipped_blocks": 0,
        "attempted_blocks": 0,
        "success_blocks": 0,
        "failure_blocks": 0,
        "attempted_bytes": 0,
        "released_bytes": 0,
        "failed_bytes": 0,
        "remote_map_blocks": 0,
        "remote_map_failure_blocks": 0,
        "cpu_resident_bytes": 0,
        "remote_mapped_bytes": 0,
        "newly_remote_mapped_bytes": 0,
        "first_error": 0,
        "first_failed_region_id": 0,
        "first_failed_block": 0,
        "entries": [],
    }
    if not normalized:
        result["ok"] = True
        return result

    flags = 0
    if require_expert:
        flags |= UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT
    if strict_2m_blocks:
        flags |= UVM_EXPERT_OFFLOAD_RELEASE_FLAG_STRICT_2M_BLOCKS
    if subchunk_64k:
        flags |= UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K
    if remote_read:
        flags |= UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for expert offload release",
                     exc_info=True)
        result["failure_count"] = len(normalized)
        result["first_error"] = -3
        return result

    def _run_release_on_fd(fd: int, fd_source: str) -> dict[str, object]:
        candidate: dict[str, object] = {
            "ok": False,
            "ioctl": _UVM_EXPERT_OFFLOAD_RELEASE_IOCTL,
            "entry_count": len(normalized),
            "success_count": 0,
            "failure_count": 0,
            "skipped_blocks": 0,
            "attempted_blocks": 0,
            "success_blocks": 0,
            "failure_blocks": 0,
            "attempted_bytes": 0,
            "released_bytes": 0,
            "failed_bytes": 0,
            "remote_map_blocks": 0,
            "remote_map_failure_blocks": 0,
            "cpu_resident_bytes": 0,
            "remote_mapped_bytes": 0,
            "newly_remote_mapped_bytes": 0,
            "first_error": 0,
            "first_failed_region_id": 0,
            "first_failed_block": 0,
            "fd_source": fd_source,
            "entries": [],
        }
        entries: list[dict[str, object]] = []
        for region_id in normalized:
            params = _ExpertOffloadReleaseParams()
            params.region_id = int(region_id)
            params.gpu_uuid = gpu_uuid
            params.flags = int(flags)
            params.max_blocks = max(int(max_blocks or 0), 0)
            ctypes.set_errno(0)
            ret = _libc_ioctl(fd, _UVM_EXPERT_OFFLOAD_RELEASE_IOCTL, params)
            saved_errno = ctypes.get_errno()
            entry = _expert_offload_release_entry_to_dict(
                params,
                ret=int(ret),
                saved_errno=int(saved_errno),
                fd_source=fd_source,
            )
            if return_entries:
                entries.append(entry)

            ok = bool(entry["ok"])
            candidate["success_count"] = int(candidate["success_count"]) + (1 if ok else 0)
            candidate["failure_count"] = int(candidate["failure_count"]) + (0 if ok else 1)
            for key in (
                "attempted_blocks",
                "success_blocks",
                "failure_blocks",
                "skipped_blocks",
                "attempted_bytes",
                "released_bytes",
                "failed_bytes",
                "remote_map_blocks",
                "remote_map_failure_blocks",
                "cpu_resident_bytes",
                "remote_mapped_bytes",
                "newly_remote_mapped_bytes",
            ):
                candidate[key] = int(candidate[key]) + int(entry.get(key) or 0)
            if not ok and int(candidate["first_error"]) == 0:
                candidate["first_error"] = int(entry.get("rm_status") or 0) or int(
                    entry.get("first_error") or 0
                ) or -5
                candidate["first_failed_region_id"] = int(region_id)
                candidate["first_failed_block"] = int(
                    entry.get("first_failed_block") or 0
                )

        candidate["ok"] = int(candidate["failure_count"]) == 0
        candidate["entries"] = entries
        return candidate

    candidate_results: list[dict[str, object]] = []
    try:
        if prefer_existing:
            for source_fd in _iter_existing_uvm_fd_sources():
                fd = _initialize_duplicated_uvm_fd(source_fd)
                if fd is None:
                    continue
                try:
                    candidate = _run_release_on_fd(
                        fd,
                        f"existing:{source_fd}",
                    )
                finally:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                candidate_results.append(candidate)
                if bool(candidate.get("ok")) or int(
                    candidate.get("released_bytes") or 0
                ) > 0 or int(candidate.get("attempted_blocks") or 0) > 0:
                    return candidate

        fd = _open_initialized_uvm_fd()
        try:
            candidate = _run_release_on_fd(fd, "new")
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        candidate_results.append(candidate)
        if bool(candidate.get("ok")):
            return candidate

        if candidate_results:
            best = max(
                candidate_results,
                key=lambda item: (
                    int(item.get("released_bytes") or 0),
                    int(item.get("attempted_blocks") or 0),
                    int(item.get("success_count") or 0),
                    -int(item.get("failure_count") or 0),
                ),
            )
            best["attempted_fd_count"] = len(candidate_results)
            return best

        return result
    except Exception:
        logger.debug("Expert offload release ioctl failed", exc_info=True)
        result["failure_count"] = len(normalized)
        if int(result["first_error"]) == 0:
            result["first_error"] = -4
        result["entries"] = (
            candidate_results[-1].get("entries", [])
            if candidate_results
            else []
        )
        return result


def driver_expert_offload_release_remote_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    require_expert: bool = True,
    strict_2m_blocks: bool = False,
    subchunk_64k: bool = False,
    max_blocks: int = 0,
    prefer_existing: bool = True,
    return_entries: bool = True,
) -> dict[str, object]:
    """Release expert GPU backing and establish remote-read PTEs atomically.

    Atomicity here is per registered driver region: the ioctl reports success
    only after all bytes in that region are CPU resident and GPU remote mapped.
    The aggregate result remains failed if any submitted region is incomplete.
    """
    return driver_expert_offload_release_regions(
        region_ids,
        device=device,
        require_expert=require_expert,
        strict_2m_blocks=strict_2m_blocks,
        subchunk_64k=subchunk_64k,
        remote_read=True,
        max_blocks=max_blocks,
        prefer_existing=prefer_existing,
        return_entries=return_entries,
    )


def driver_prefetch_cpu_release_remote_512k_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    prefer_existing: bool = True,
) -> dict[str, object]:
    """Move registered 512 KiB envelopes to CPU and free their GPU backing.

    This is the driver-backed equivalent of CPU prefetch for expert envelopes.
    One ioctl path migrates each envelope to CPU, releases its GPU PMM backing,
    and installs GPU remote-read PTEs. Success requires all four effects for
    every submitted envelope; an accepted ioctl alone is not sufficient.
    """
    normalized = [int(region_id) for region_id in region_ids if int(region_id) > 0]
    result = driver_expert_offload_release_remote_regions(
        normalized,
        device=device,
        subchunk_64k=True,
        prefer_existing=prefer_existing,
        return_entries=True,
    )
    validated_entries: list[dict[str, object]] = []
    complete_count = 0
    for raw_entry in result.get("entries") or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        length = int(entry.get("length") or 0)
        complete = (
            bool(entry.get("ok"))
            and length == UVM_EXPERT_REMOTE_PREFETCH_ENVELOPE_BYTES
            and int(entry.get("released_bytes") or 0) >= length
            and int(entry.get("cpu_resident_bytes") or 0) >= length
            and int(entry.get("remote_mapped_bytes") or 0) >= length
            and int(entry.get("remote_map_failure_blocks") or 0) == 0
        )
        if complete:
            reason = None
            complete_count += 1
        elif length != UVM_EXPERT_REMOTE_PREFETCH_ENVELOPE_BYTES:
            reason = "not_512k_envelope"
        elif int(entry.get("released_bytes") or 0) < length:
            reason = "gpu_backing_release_incomplete"
        elif int(entry.get("cpu_resident_bytes") or 0) < length:
            reason = "cpu_migration_incomplete"
        elif int(entry.get("remote_mapped_bytes") or 0) < length:
            reason = "remote_mapping_incomplete"
        else:
            reason = str(entry.get("reason") or "driver_operation_failed")
        entry["prefetch_release_complete"] = complete
        entry["prefetch_release_reason"] = reason
        validated_entries.append(entry)

    requested_count = len(normalized)
    aggregate_ok = (
        bool(result.get("ok"))
        and len(validated_entries) == requested_count
        and complete_count == requested_count
    )
    return {
        **result,
        "ok": aggregate_ok,
        "entries": validated_entries,
        "operation": "prefetch_cpu_release_remote_512k",
        "envelope_bytes": UVM_EXPERT_REMOTE_PREFETCH_ENVELOPE_BYTES,
        "requested_region_count": requested_count,
        "validated_success_count": complete_count,
        "validated_failure_count": requested_count - complete_count,
    }


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


def _ioctl_expert_reserved_prefetch_on_fd(
    fd: int,
    *,
    mode: str | int,
    mode_value: int,
    entries: Sequence[dict[str, int]],
    gpu_uuid: _PhaseFUuid,
    async_: bool,
) -> dict[str, object]:
    params = _ExpertReservedPrefetchBatchParams()
    _fill_expert_reserved_prefetch_params(
        params,
        mode_value=mode_value,
        entries=entries,
        gpu_uuid=gpu_uuid,
        async_=async_,
    )
    ctypes.set_errno(0)
    ret = _libc_ioctl(fd, _UVM_EXPERT_RESERVED_PREFETCH_BATCH_IOCTL, params)
    saved_errno = ctypes.get_errno()
    return _expert_reserved_prefetch_result_from_params(
        params,
        mode=mode,
        ret=int(ret),
        saved_errno=int(saved_errno),
    )


def _expert_reserved_prefetch_via_helper(
    *,
    mode: str | int,
    mode_value: int,
    entries: Sequence[dict[str, int]],
    device: int,
    async_: bool,
    return_entries: bool = True,
) -> Optional[dict[str, object]]:
    if _uvm_lib is None or not hasattr(
        _uvm_lib, "uvm_driver_expert_reserved_prefetch_regions"
    ):
        return None

    count = len(entries)
    buffers = _expert_reserved_prefetch_helper_buffers()
    for index, entry in enumerate(entries):
        buffers.region_ids[index] = int(entry.get("region_id", 0))
        buffers.pool_ids[index] = int(entry.get("pool_id", 0))
        buffers.slot_ids[index] = int(entry.get("slot_id", index))
        buffers.slot_offsets[index] = int(entry.get("slot_offset", 0))
        buffers.bytes_overrides[index] = int(entry.get("bytes_override", 0))
        buffers.flags[index] = int(entry.get("flags", 0))

    for index in range(_UVM_EXPERT_RESERVED_PREFETCH_STATS_COUNT):
        buffers.stats[index] = 0

    if return_entries:
        for index in range(count):
            buffers.out_entries[index] = _ExpertReservedPrefetchEntry()

    try:
        status = int(
            _uvm_lib.uvm_driver_expert_reserved_prefetch_regions(
                buffers.region_ids,
                buffers.pool_ids,
                buffers.slot_ids,
                buffers.slot_offsets,
                buffers.bytes_overrides,
                buffers.flags,
                ctypes.c_size_t(count),
                ctypes.c_uint32(mode_value),
                ctypes.c_int(device),
                ctypes.c_uint32(1 if async_ else 0),
                buffers.stats,
                ctypes.c_size_t(_UVM_EXPERT_RESERVED_PREFETCH_STATS_COUNT),
                buffers.out_entries,
                ctypes.c_size_t(count if return_entries else 0),
            )
        )
    except Exception:
        logger.debug("Expert reserved prefetch helper failed", exc_info=True)
        return None

    ret = 0 if status >= 0 else -1
    saved_errno = int(buffers.stats[13]) if status == -6 else 0
    result = _empty_expert_reserved_prefetch_result(
        mode=mode,
        ioctl_ret=ret,
        rm_status=int(buffers.stats[0]),
    )
    result.update({
        "ok": int(ret) == 0 and int(buffers.stats[0]) == 0,
        "errno": int(saved_errno),
        "mode": int(mode_value),
        "entry_count": int(count),
        "success_count": int(buffers.stats[1]),
        "failure_count": int(buffers.stats[2]),
        "requested_bytes": int(buffers.stats[3]),
        "accepted_bytes": int(buffers.stats[4]),
        "resolved_regions": int(buffers.stats[5]),
        "assigned_slots": int(buffers.stats[6]),
        "invalidated_slots": int(buffers.stats[7]),
        "reserved_backing_bytes": int(buffers.stats[8]),
        "ordinary_resident_bytes": int(buffers.stats[9]),
        "pool_external_alloc_count": int(buffers.stats[10]),
        "footprint_contract_broken": int(buffers.stats[11]),
        "current_epoch": int(buffers.stats[12]),
    })
    if return_entries:
        result["entries"] = [
            _expert_reserved_prefetch_entry_to_dict(buffers.out_entries[index])
            for index in range(count)
        ]
    else:
        result["entries"] = []
    result["helper_status"] = status
    result["helper_path"] = True
    result["compact_result"] = not return_entries
    return result


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
    return_entries: bool = True,
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

    helper_result = _expert_reserved_prefetch_via_helper(
        mode=mode,
        mode_value=mode_value,
        entries=normalized_entries,
        device=device,
        async_=async_,
        return_entries=return_entries,
    )
    if helper_result is not None:
        return helper_result

    fd = -1
    last_result: dict[str, object] | None = None
    try:
        if prefer_existing:
            for candidate_fd in _iter_initialized_existing_uvm_fds():
                try:
                    result = _ioctl_expert_reserved_prefetch_on_fd(
                        candidate_fd,
                        mode=mode,
                        mode_value=mode_value,
                        entries=normalized_entries,
                        gpu_uuid=gpu_uuid,
                        async_=async_,
                    )
                finally:
                    os.close(candidate_fd)

                last_result = result
                if result.get("ok") is True:
                    return result

                # A partial operation may have changed slot owner state. Return
                # it directly instead of probing more VA spaces.
                if int(result.get("success_count", 0) or 0) > 0:
                    return result

        fd = _open_initialized_uvm_fd()
        return _ioctl_expert_reserved_prefetch_on_fd(
            fd,
            mode=mode,
            mode_value=mode_value,
            entries=normalized_entries,
            gpu_uuid=gpu_uuid,
            async_=async_,
        )
    except Exception:
        logger.debug("Expert reserved prefetch ioctl failed", exc_info=True)
        if last_result is not None:
            return last_result
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
    *,
    allow_tag_only: bool = False,
) -> int:
    """Batch prefetch raw UVM ranges to a GPU or CPU device.

    Returns the number of CUDA prefetch calls accepted by the allocator shim.
    """
    if uvm_residency_tag_only_enabled() and not allow_tag_only:
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
    *,
    allow_tag_only: bool = False,
) -> int:
    """Batch prefetch raw UVM ranges to CPU memory."""
    return batch_prefetch_ranges_to_device(
        ranges,
        device=-1,
        allow_tag_only=allow_tag_only,
    )


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




def driver_batch_query_ranges_residency_bytes(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
    *,
    wait_for_trackers: bool = True,
) -> dict[str, int | bool]:
    """Query page-granular real UVM residency bytes for managed ranges.

    A no-wait query is intended for callers which have already synchronized
    CUDA work. It reads the protected resident masks without waiting on each
    VA block tracker, avoiding an unbounded wait on stale experimental tracker
    entries.
    """
    result: dict[str, int | bool] = {
        "ok": False,
        "status": -1,
        "queried_bytes": 0,
        "gpu_bytes": 0,
        "cpu_bytes": 0,
        "any_bytes": 0,
        "unknown_bytes": 0,
    }
    try:
        ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return result

    range_list = list(ranges)
    if not range_list:
        result["ok"] = True
        result["status"] = 0
        return result

    query_fn = _uvm_lib.uvm_driver_batch_query_residency_bytes
    if not wait_for_trackers:
        query_fn = getattr(
            _uvm_lib,
            "uvm_driver_batch_query_residency_bytes_nowait",
            None,
        )
        if query_fn is None:
            result["status"] = -2
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
            query_fn(
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
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return -1
    if region_id <= 0 or ptr <= 0 or nbytes <= 0:
        return -2
    try:
        return int(
            lib.uvm_driver_phase_f_register_region(
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
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return -1
    if region_id <= 0:
        return -2
    try:
        return int(
            lib.uvm_driver_phase_f_unregister_region(
                ctypes.c_uint64(region_id)
            )
        )
    except Exception:
        logger.debug("Failed to unregister Phase F driver region", exc_info=True)
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
