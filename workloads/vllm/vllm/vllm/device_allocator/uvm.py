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
_uvm_local_prefetch_stack: list[bool] = []
_uvm_phase_stack: list[str] = []
_UVM_DRIVER_BATCH_MAX_RANGES = 64
_UVM_INITIALIZE_IOCTL = 0x30000001
_UVM_PHASE_F_BATCH_EXECUTOR_IOCTL = 81
_UVM_PHASE_F_MAX_RANGES = 64
_UVM_EXPERT_OFFLOAD_RELEASE_IOCTL = 100
_UVM_PAGE_SIZE = 0

UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ = 5
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG = 6
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED = 7
UVM_PHASE_F_ACTION_START_BG_PROMOTION = 8
UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION = 16
UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION = 17

UVM_REGION_STATE_UNKNOWN = 0
UVM_REGION_STATE_COLD = 1
UVM_REGION_STATE_WARM = 2
UVM_REGION_STATE_HOT = 3
UVM_REGION_STATE_ACTIVE = 4
UVM_REGION_STATE_INACTIVE = 5

UVM_POOL_BUDGET_MODE_TRACE_ONLY = 0
UVM_POOL_BUDGET_MODE_SOFT = 1
UVM_POOL_BUDGET_MODE_ENFORCE = 2



UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K = 1 << 2
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ = 1 << 3


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






class _ExpertOffloadReleaseParams(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("gpu_uuid", _PhaseFUuid),
        ("flags", ctypes.c_uint32),
        ("max_blocks", ctypes.c_uint32),
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
    ]

    lib.uvm_driver_phase_f_unregister_region.restype = ctypes.c_int
    lib.uvm_driver_phase_f_unregister_region.argtypes = [ctypes.c_uint64]

    if hasattr(lib, "uvm_driver_phase_n_update_region_state"):
        lib.uvm_driver_phase_n_update_region_state.restype = ctypes.c_int
        lib.uvm_driver_phase_n_update_region_state.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
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


def _region_release_entry_to_dict(
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
    activity_counters = (
        int(params.block_count),
        int(params.attempted_blocks),
        int(params.success_blocks),
        int(params.failure_blocks),
        int(params.skipped_blocks),
        int(params.attempted_bytes),
        int(params.released_bytes),
        int(params.failed_bytes),
        int(params.remote_map_blocks),
        int(params.remote_map_failure_blocks),
        int(params.cpu_resident_bytes),
        int(params.remote_mapped_bytes),
        int(params.newly_remote_mapped_bytes),
    )
    idempotent = (
        raw_ok
        and not remote_read
        and int(params.length) > 0
        and int(params.first_error) == 0
        and not any(activity_counters)
    )
    effective_ok = raw_ok and (
        idempotent
        or int(params.block_count) > 0
        or int(params.attempted_blocks) > 0
        or int(params.skipped_blocks) > 0
        or int(params.released_bytes) > 0
    )
    if remote_read and not idempotent:
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
        "idempotent": idempotent,
        "reason": reason,
        "fd_source": fd_source,
        "ioctl_ret": int(ret),
        "errno": int(saved_errno),
        "rm_status": int(params.rmStatus),
        "remote_read": remote_read,
        "region_id": int(params.region_id),
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


def driver_release_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    remote_read: bool = False,
    max_blocks: int = 0,
    prefer_existing: bool = True,
    return_entries: bool = True,
) -> dict[str, object]:
    """Release explicitly named regions at 64 KiB subchunk granularity.

    The managed VA ranges remain allocated. With ``remote_read=True``, the
    released ranges are also synchronously mapped read-only from the GPU to
    their CPU backing before the ioctl commits success.
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

    flags = UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K
    if remote_read:
        flags |= UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for region release",
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
            entry = _region_release_entry_to_dict(
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
        logger.debug("Region release ioctl failed", exc_info=True)
        result["failure_count"] = len(normalized)
        if int(result["first_error"]) == 0:
            result["first_error"] = -4
        result["entries"] = (
            candidate_results[-1].get("entries", [])
            if candidate_results
            else []
        )
        return result


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
) -> int:
    """Register an opaque explicit region by id and VA geometry."""
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
    """Deprecated: driver-owned pool budgets are not used by explicit regions."""
    return -95


def driver_get_pool_state(pool: int, gpu_id: int = 0) -> dict[str, int] | None:
    """Deprecated: Phase-N pool snapshots are unreachable in explicit mode."""
    return None


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
    """Deprecated: reclaim is application-selected explicit release only."""
    return None


def driver_update_region_state(
    *,
    region_id: int,
    state: int = UVM_REGION_STATE_UNKNOWN,
) -> int:
    """Update the minimal explicit-region state in the driver."""
    if not has_region_state_ioctl():
        return -1
    if region_id <= 0:
        return -2
    try:
        return int(
            _uvm_lib.uvm_driver_phase_n_update_region_state(
                ctypes.c_uint64(int(region_id)),
                ctypes.c_uint32(max(int(state), 0)),
            )
        )
    except Exception:
        logger.debug("Failed to update Phase N driver region state",
                     exc_info=True)
        return -3


def _driver_region_values_to_dict(values: ctypes.Array) -> dict[str, int]:
    return {
        "region_id": int(values[0]),
        "base": int(values[1]),
        "length": int(values[2]),
        "state": int(values[3]),
    }


def driver_query_region(region_id: int) -> dict[str, int] | None:
    """Query Stage N region metadata by region id."""
    if not has_region_state_ioctl() or region_id <= 0:
        return None
    values = (ctypes.c_uint64 * 4)()
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


def driver_query_region_states(region_ids: Sequence[int]) -> dict[int, int]:
    """Return only the state of explicitly queried numeric region IDs."""
    result: dict[int, int] = {}
    for value in region_ids:
        try:
            region_id = int(value)
        except (TypeError, ValueError):
            continue
        snapshot = driver_query_region(region_id)
        if snapshot is not None:
            result[region_id] = int(snapshot.get("state", UVM_REGION_STATE_UNKNOWN))
    return result


def has_pool_budget_ioctl() -> bool:
    return False


def has_pool_reclaim_ioctl() -> bool:
    return False


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


def driver_restore_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    executor: str = "auto",
) -> dict[str, object]:
    """Restore explicitly registered regions by their recorded VA ranges."""
    ranges: list[tuple[int, int]] = []
    valid_ids: list[int] = []
    for value in region_ids:
        try:
            region_id = int(value)
        except (TypeError, ValueError):
            continue
        snapshot = driver_query_region(region_id)
        if snapshot is None:
            continue
        base = int(snapshot.get("base") or 0)
        length = int(snapshot.get("length") or 0)
        if base > 0 and length > 0:
            valid_ids.append(region_id)
            ranges.append((base, length))
    successes, executor_used, driver_status = (
        batch_prefetch_ranges_to_device_with_executor(
            ranges,
            device,
            executor=executor,
        )
        if ranges
        else (0, executor, 0)
    )
    # The range executor does not know region identity. Reflect completed
    # restores back into the explicit-region state machine in submission order.
    for region_id in valid_ids[:max(int(successes), 0)]:
        driver_update_region_state(
            region_id=region_id,
            state=UVM_REGION_STATE_HOT,
        )
    complete = int(successes) == len(ranges)
    return {
        "ok": complete,
        "region_ids": valid_ids,
        "success_count": int(successes),
        "failure_count": len(ranges) - int(successes),
        "executor": executor_used,
        "driver_status": driver_status,
    }
