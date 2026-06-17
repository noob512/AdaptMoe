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

UVM_REGION_STATE_UNKNOWN = 0
UVM_REGION_STATE_COLD = 1
UVM_REGION_STATE_WARM = 2
UVM_REGION_STATE_HOT = 3
UVM_REGION_STATE_ACTIVE = 4
UVM_REGION_STATE_INACTIVE = 5

UVM_POOL_BUDGET_MODE_TRACE_ONLY = 0
UVM_POOL_BUDGET_MODE_SOFT = 1
UVM_POOL_BUDGET_MODE_ENFORCE = 2


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

    lib.uvm_driver_phase_f_register_query_region.restype = ctypes.c_int
    lib.uvm_driver_phase_f_register_query_region.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_int,
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

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for Phase-F ioctl",
                     exc_info=True)
        return 0, len(range_list), 0, -3

    fd = -1
    try:
        fd = _open_initialized_uvm_fd()
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

    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        result["ok"] = True
        result["status"] = 0
        return result

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
    result.update(
        {
            "ok": status >= 0,
            "status": status,
            "queried_bytes": int(queried_bytes.value),
            "gpu_bytes": int(gpu_bytes.value),
            "cpu_bytes": int(cpu_bytes.value),
            "any_bytes": int(any_bytes.value),
            "unknown_bytes": int(unknown_bytes.value),
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
