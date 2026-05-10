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

import ctypes
import os
import atexit
from contextlib import contextmanager
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global state
_uvm_enabled = False
_uvm_lib: Optional[ctypes.CDLL] = None
_uvm_phase_stack: list[str] = []


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

    lib.uvm_advise.restype = None
    lib.uvm_advise.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
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
    global _uvm_enabled, _uvm_lib

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
        atexit.register(_close_uvm_log_at_exit)
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
    if not _uvm_enabled or _uvm_lib is None:
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


def advise_range_preferred_location(ptr: int, size: int, device: int) -> bool:
    """
    Set cudaMemAdviseSetPreferredLocation for a raw UVM address range.

    Returns True when the advise call was issued. The allocator shim currently
    logs CUDA errors internally instead of returning them, so benchmark health
    and Stage I trace records are used for end-to-end validation.
    """
    if not _uvm_enabled or _uvm_lib is None:
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
