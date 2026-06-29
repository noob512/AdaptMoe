#!/usr/bin/env python3
"""Smoke-test the experimental UVM_MOE_SLOT_REMAP_BATCH ioctl."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

UVM_INITIALIZE_IOCTL = 0x30000001
UVM_MIGRATE_IOCTL = 51
UVM_MM_INITIALIZE_IOCTL = 75
UVM_TEST_VA_RANGE_SPLIT_IOCTL = 208
UVM_MOE_SLOT_REMAP_BATCH_IOCTL = 91
UVM_MOE_SLOT_POOL_QUERY_IOCTL = 92
UVM_MOE_SLOT_REMAP_MAX_ENTRIES = 128
UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES = 128
NV_ERR_INVALID_STATE = 0x00000040
NV_WARN_NOTHING_TO_DO = 0x00010006

MODE_VALUES = {
    "dry_run": 0,
    "validate": 1,
    "remap": 2,
    "copy_only": 3,
    "gpu_copy_only": 4,
    "slot_pte_remap": 5,
    "slot_va_block_remap": 6,
}

ENTRY_OK = 0
ENTRY_REMAP_NOT_SUPPORTED = 4
ENTRY_REMAP_FAILED = 7
ENTRY_ROLLED_BACK = 8
BACKING_VIDMEM = 3
QUERY_FLAG_RESET = 1 << 31
UVM_VA_BLOCK_SIZE = 2 * 1024 * 1024
VIDMEM_BUDGET_PARAM = "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"


class UvmInitializeParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class UvmMmInitializeParams(ctypes.Structure):
    _fields_ = [
        ("uvmFd", ctypes.c_int32),
        ("rmStatus", ctypes.c_uint32),
    ]


class UvmUuid(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)]


class UvmMigrateParams(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("destinationUuid", UvmUuid),
        ("flags", ctypes.c_uint32),
        ("semaphoreAddress", ctypes.c_uint64),
        ("semaphorePayload", ctypes.c_uint32),
        ("cpuNumaNode", ctypes.c_int32),
        ("userSpaceStart", ctypes.c_uint64),
        ("userSpaceLength", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class UvmTestVaRangeSplitParams(ctypes.Structure):
    _fields_ = [
        ("split_address", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class MoeSlotRemapEntry(ctypes.Structure):
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


class MoeSlotRemapBatchParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", UvmUuid),
        ("validate_managed", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("entries", MoeSlotRemapEntry * UVM_MOE_SLOT_REMAP_MAX_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class MoeSlotPoolQueryEntry(ctypes.Structure):
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


class MoeSlotPoolQueryParams(ctypes.Structure):
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
        ("entries", MoeSlotPoolQueryEntry * UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.ioctl.restype = ctypes.c_int


def ioctl(fd: int, request: int, params: ctypes.Structure) -> tuple[int, int]:
    ctypes.set_errno(0)
    ret = int(LIBC.ioctl(ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.byref(params)))
    return ret, int(ctypes.get_errno())


def fd_link_is_uvm(link_target: str) -> bool:
    base = os.path.basename(link_target)
    return base == "nvidia-uvm" or base.startswith("nvidia-uvm ")


def duplicate_existing_uvm_fd() -> tuple[int, str]:
    for name in sorted(os.listdir("/proc/self/fd"), key=lambda value: int(value) if value.isdigit() else -1):
        if not name.isdigit():
            continue
        source_fd = int(name)
        if source_fd < 3:
            continue
        try:
            link_target = os.readlink(f"/proc/self/fd/{source_fd}")
        except OSError:
            continue
        if not fd_link_is_uvm(link_target):
            continue
        try:
            return os.dup(source_fd), f"adopted:{source_fd}:{link_target}"
        except OSError:
            continue
    return -1, ""


def open_initialized_uvm_fd(*, prefer_existing: bool = False) -> tuple[int, str, int, int, int]:
    candidates: list[tuple[int, str]] = []
    if prefer_existing:
        fd, source = duplicate_existing_uvm_fd()
        if fd >= 0:
            candidates.append((fd, source))

    fd = os.open("/dev/nvidia-uvm", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    candidates.append((fd, "new:/dev/nvidia-uvm"))

    last_ret = -1
    last_errno = 0
    last_rm_status = 0
    for candidate_fd, source in candidates:
        init_params = UvmInitializeParams()
        init_ret, init_errno = ioctl(candidate_fd, UVM_INITIALIZE_IOCTL, init_params)
        last_ret = init_ret
        last_errno = init_errno
        last_rm_status = int(init_params.rmStatus)
        if init_ret == 0 and last_rm_status == 0:
            return candidate_fd, source, init_ret, init_errno, last_rm_status
        os.close(candidate_fd)

    return -1, "none", last_ret, last_errno, last_rm_status


def open_initialized_uvm_mm_fd(uvm_fd: int) -> tuple[int, int, int, int]:
    fd = os.open("/dev/nvidia-uvm", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    params = UvmMmInitializeParams()
    params.uvmFd = uvm_fd
    ret, errno_value = ioctl(fd, UVM_MM_INITIALIZE_IOCTL, params)
    if ret == 0 and int(params.rmStatus) == 0:
        return fd, ret, errno_value, int(params.rmStatus)

    os.close(fd)
    return -1, ret, errno_value, int(params.rmStatus)


def gpu_uuid_from_nvidia_smi(device: int) -> UvmUuid:
    try:
        timeout = max(int(os.environ.get("UVM_MOE_NVIDIA_SMI_TIMEOUT", "5")), 1)
    except ValueError:
        timeout = 5

    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"nvidia-smi timed out after {timeout}s while querying GPU UUID; "
            "the node may have wedged NVIDIA/UVM state"
        ) from exc
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("failed to query GPU UUID with nvidia-smi") from exc

    first = output.strip().splitlines()[0].strip()
    if first.startswith("GPU-"):
        first = first[4:]
    parsed = uuid.UUID(first)
    result = UvmUuid()
    for index, byte in enumerate(parsed.bytes):
        result.uuid[index] = byte
    return result


def gpu_uuid_from_cuda_driver(device: int) -> UvmUuid:
    libname = ctypes.util.find_library("cuda") or "libcuda.so.1"
    libcuda = ctypes.CDLL(libname)
    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    cu_device_get = libcuda.cuDeviceGet
    cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cu_device_get.restype = ctypes.c_int
    cu_device_get_uuid = libcuda.cuDeviceGetUuid
    cu_device_get_uuid.argtypes = [ctypes.POINTER(ctypes.c_uint8 * 16), ctypes.c_int]
    cu_device_get_uuid.restype = ctypes.c_int

    ret = int(cu_init(0))
    if ret != 0:
        raise RuntimeError(f"cuInit failed: {ret}")
    cu_device = ctypes.c_int()
    ret = int(cu_device_get(ctypes.byref(cu_device), int(device)))
    if ret != 0:
        raise RuntimeError(f"cuDeviceGet({device}) failed: {ret}")
    uuid_bytes_array = (ctypes.c_uint8 * 16)()
    ret = int(cu_device_get_uuid(ctypes.byref(uuid_bytes_array), cu_device))
    if ret != 0:
        raise RuntimeError(f"cuDeviceGetUuid({device}) failed: {ret}")
    uuid_bytes = bytes(uuid_bytes_array)
    if len(uuid_bytes) != 16:
        raise RuntimeError(f"unexpected CUDA UUID size: {len(uuid_bytes)}")
    result = UvmUuid()
    for index, byte in enumerate(uuid_bytes):
        result.uuid[index] = byte
    return result


def gpu_uuid_for_device(device: int) -> UvmUuid:
    try:
        return gpu_uuid_from_cuda_driver(device)
    except Exception:
        if os.environ.get("UVM_MOE_ALLOW_NVIDIA_SMI_UUID", "0").lower() in (
            "1", "true", "yes", "on",
        ):
            return gpu_uuid_from_nvidia_smi(device)
        raise


def initialize_cuda_device(device: int) -> None:
    libname = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(libname)
    cuda_set_device = cudart.cudaSetDevice
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = ctypes.c_int
    cuda_free = cudart.cudaFree
    cuda_free.argtypes = [ctypes.c_void_p]
    cuda_free.restype = ctypes.c_int

    ret = int(cuda_set_device(ctypes.c_int(device)))
    if ret != 0:
        raise RuntimeError(f"cudaSetDevice({device}) failed: {ret}")
    ret = int(cuda_free(None))
    if ret != 0:
        raise RuntimeError(f"cudaFree(0) failed: {ret}")


class ManagedAllocation:
    def __init__(self, device: int, byte_count: int) -> None:
        libname = ctypes.util.find_library("cudart") or "libcudart.so"
        self._cudart = ctypes.CDLL(libname)
        self._ptr = ctypes.c_void_p()
        self.byte_count = int(byte_count)

        cuda_malloc_managed = self._cudart.cudaMallocManaged
        cuda_malloc_managed.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cuda_malloc_managed.restype = ctypes.c_int
        self._cuda_free = self._cudart.cudaFree
        self._cuda_free.argtypes = [ctypes.c_void_p]
        self._cuda_free.restype = ctypes.c_int
        cuda_device_synchronize = self._cudart.cudaDeviceSynchronize
        cuda_device_synchronize.argtypes = []
        cuda_device_synchronize.restype = ctypes.c_int

        initialize_cuda_device(device)
        ret = int(cuda_malloc_managed(ctypes.byref(self._ptr), self.byte_count, 1))
        if ret != 0:
            raise RuntimeError(f"cudaMallocManaged({self.byte_count}) failed: {ret}")

        ctypes.memset(self._ptr, 0x5A, self.byte_count)
        ret = int(cuda_device_synchronize())
        if ret != 0:
            raise RuntimeError(f"cudaDeviceSynchronize() failed: {ret}")

    @property
    def address(self) -> int:
        return int(self._ptr.value or 0)

    def close(self) -> None:
        if self._ptr.value:
            self._cuda_free(self._ptr)
            self._ptr = ctypes.c_void_p()


class AnonymousMmapAllocation:
    def __init__(self, byte_count: int, fill: int = 0x5A) -> None:
        self.byte_count = int(byte_count)
        self._mmap = mmap.mmap(-1, self.byte_count)
        self._mmap.write(bytes((fill for _ in range(self.byte_count))))
        self._mmap.seek(0)
        mmap_byte = ctypes.c_char.from_buffer(self._mmap)
        self._address = ctypes.addressof(mmap_byte)
        del mmap_byte

    @property
    def address(self) -> int:
        return self._address

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
            self._address = 0


def cuda_memcpy_device_to_host(device_ptr: int, byte_count: int) -> bytes:
    libname = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(libname)
    cuda_malloc_host = cudart.cudaMallocHost
    cuda_malloc_host.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda_malloc_host.restype = ctypes.c_int
    cuda_free_host = cudart.cudaFreeHost
    cuda_free_host.argtypes = [ctypes.c_void_p]
    cuda_free_host.restype = ctypes.c_int
    cuda_memcpy = cudart.cudaMemcpy
    cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cuda_memcpy.restype = ctypes.c_int

    host_ptr = ctypes.c_void_p()
    ret = int(cuda_malloc_host(ctypes.byref(host_ptr), byte_count))
    if ret != 0:
        raise RuntimeError(f"cudaMallocHost({byte_count}) failed: {ret}")
    try:
        ret = int(cuda_memcpy(host_ptr, ctypes.c_void_p(device_ptr), byte_count, 2))
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed: {ret}")
        return ctypes.string_at(host_ptr, byte_count)
    finally:
        cuda_free_host(host_ptr)


def cuda_memcpy_host_to_device(device_ptr: int, payload: bytes) -> None:
    libname = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(libname)
    cuda_malloc_host = cudart.cudaMallocHost
    cuda_malloc_host.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda_malloc_host.restype = ctypes.c_int
    cuda_free_host = cudart.cudaFreeHost
    cuda_free_host.argtypes = [ctypes.c_void_p]
    cuda_free_host.restype = ctypes.c_int
    cuda_memcpy = cudart.cudaMemcpy
    cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cuda_memcpy.restype = ctypes.c_int

    host_ptr = ctypes.c_void_p()
    byte_count = len(payload)
    ret = int(cuda_malloc_host(ctypes.byref(host_ptr), byte_count))
    if ret != 0:
        raise RuntimeError(f"cudaMallocHost({byte_count}) failed: {ret}")
    try:
        ctypes.memmove(host_ptr, payload, byte_count)
        ret = int(cuda_memcpy(ctypes.c_void_p(device_ptr), host_ptr, byte_count, 1))
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed: {ret}")
    finally:
        cuda_free_host(host_ptr)


def load_uvm_shim() -> ctypes.CDLL:
    candidates = [
        Path(__file__).resolve().parent / "vllm" / "uvm_test" / "uvm_allocator.so",
        Path(__file__).resolve().parent / "vllm" / "vllm" / "uvm_allocator.abi3.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            lib = ctypes.CDLL(str(candidate))
            lib.uvm_smoke_gpu_readback.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.uvm_smoke_gpu_readback.restype = ctypes.c_int
            lib.uvm_smoke_gpu_fill.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.uvm_smoke_gpu_fill.restype = ctypes.c_int
            return lib
    raise RuntimeError("uvm_allocator shim not found")


def kernel_readback(device_ptr: int, byte_count: int, device: int) -> bytes:
    lib = load_uvm_shim()
    buffer = ctypes.create_string_buffer(byte_count)
    ret = int(lib.uvm_smoke_gpu_readback(
        ctypes.c_void_p(device_ptr),
        ctypes.c_void_p(ctypes.addressof(buffer)),
        ctypes.c_size_t(byte_count),
        ctypes.c_int(device),
    ))
    if ret != 0:
        raise RuntimeError(f"uvm_smoke_gpu_readback failed: {ret}")
    return bytes(buffer.raw)


def kernel_fill(device_ptr: int, value: int, byte_count: int, device: int) -> None:
    lib = load_uvm_shim()
    ret = int(lib.uvm_smoke_gpu_fill(
        ctypes.c_void_p(device_ptr),
        ctypes.c_uint(value),
        ctypes.c_size_t(byte_count),
        ctypes.c_int(device),
    ))
    if ret != 0:
        raise RuntimeError(f"uvm_smoke_gpu_fill failed: {ret}")


def reset_slot_pool(fd: int) -> dict[str, object]:
    params = MoeSlotPoolQueryParams()
    params.flags = QUERY_FLAG_RESET
    params.max_entries = 0
    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_POOL_QUERY_IOCTL, params)
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "ok": ret == 0 and int(params.rmStatus) == 0,
        "active_slots": int(params.active_slots),
        "backed_slots": int(params.backed_slots),
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
    }


def query_slot_pool(fd: int, max_entries: int = 8) -> dict[str, object]:
    params = MoeSlotPoolQueryParams()
    params.max_entries = max(0, min(max_entries, UVM_MOE_SLOT_POOL_MAX_QUERY_ENTRIES))
    ret, errno_value = ioctl(fd, UVM_MOE_SLOT_POOL_QUERY_IOCTL, params)
    entries = []
    for index in range(int(params.returned_slots)):
        entry = params.entries[index]
        entries.append({
            "layer_id": int(entry.layer_id),
            "slot_id": int(entry.slot_id),
            "owner_expert_id": int(entry.owner_expert_id),
            "state": int(entry.state),
            "range_count": int(entry.range_count),
            "last_status": int(entry.last_status),
            "slot_bytes": int(entry.slot_bytes),
            "slot_physical_bytes": int(entry.slot_physical_bytes),
            "backing_kind": int(entry.backing_kind),
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
            "pte_mapped": int(entry.pte_mapped),
            "pte_status": int(entry.pte_status),
            "pte_mode": int(entry.pte_mode),
            "pte_va_block_tracked": int(entry.pte_va_block_tracked),
        })
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "ok": ret == 0 and int(params.rmStatus) == 0,
        "total_slots": int(params.total_slots),
        "returned_slots": int(params.returned_slots),
        "active_slots": int(params.active_slots),
        "backed_slots": int(params.backed_slots),
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
    }


def slot_pool_health_ok(query: dict[str, object]) -> bool:
    return (
        int(query.get("copy_failed_count", 0)) == 0
        and int(query.get("pte_failure_count", 0)) == 0
        and int(query.get("pte_dangerous_op_reject_count", 0)) == 0
        and int(query.get("pte_residual_va_block_count", 0)) == 0
        and int(query.get("pte_residual_page_count", 0)) == 0
    )


def read_uint_file(path: str) -> int:
    return int(Path(path).read_text(encoding="utf-8").strip())


def write_uint_file(path: str, value: int) -> None:
    payload = f"{int(value)}\n"
    try:
        Path(path).write_text(payload, encoding="utf-8")
        return
    except PermissionError:
        pass

    subprocess.run(
        ["sudo", "tee", path],
        input=payload,
        text=True,
        stdout=subprocess.DEVNULL,
        check=True,
    )


def write_json_result(result: dict[str, object], out_json: Path | None) -> None:
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def run_release_parent(args: argparse.Namespace, remap_bytes: int) -> int:
    result: dict[str, object] = {
        "mode": args.mode,
        "count": args.count,
        "validate_managed": bool(args.validate_managed),
        "verify_gpu_read": bool(args.verify_gpu_read),
        "clear_via_release": True,
        "remap_bytes": remap_bytes,
    }
    fd = -1
    try:
        fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(prefer_existing=False)
        result["parent_pre_reset"] = {
            "fd_source": fd_source,
            "init_ret": init_ret,
            "init_errno": init_errno,
            "init_rmStatus": init_rm_status,
        }
        if fd < 0:
            result["passed"] = False
            write_json_result(result, args.out_json)
            return 1
        result["reset_before"] = reset_slot_pool(fd)
        if fd >= 0:
            os.close(fd)
            fd = -1
        if not result["reset_before"]["ok"]:  # type: ignore[index]
            result["passed"] = False
            write_json_result(result, args.out_json)
            return 1

        child_out = None
        if args.out_json:
            child_out = args.out_json.with_suffix(args.out_json.suffix + ".child.json")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            args.mode,
            "--count",
            str(args.count),
            "--device",
            str(args.device),
            "--base",
            hex(args.base),
            "--remap-bytes",
            str(remap_bytes),
            "--source-kind",
            args.source_kind,
            "--release-child",
        ]
        if args.validate_managed:
            cmd.append("--validate-managed")
        if args.verify_gpu_read:
            cmd.append("--verify-gpu-read")
        if args.cross_va_block:
            cmd.append("--cross-va-block")
        if child_out:
            cmd.extend(["--out-json", str(child_out)])

        child = subprocess.run(cmd, text=True, capture_output=True, check=False)
        result["release_child"] = {
            "returncode": int(child.returncode),
            "stdout_tail": child.stdout[-4096:],
            "stderr_tail": child.stderr[-4096:],
            "out_json": str(child_out) if child_out else "",
        }

        query_after = None
        reset_after = None
        for _ in range(20):
            fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(prefer_existing=False)
            result["parent_query_fd"] = {
                "fd_source": fd_source,
                "init_ret": init_ret,
                "init_errno": init_errno,
                "init_rmStatus": init_rm_status,
            }
            if fd < 0:
                break
            query_after = query_slot_pool(fd, max_entries=max(8, args.count))
            if int(query_after.get("pte_release_clear_count", 0)) >= args.count:
                break
            os.close(fd)
            fd = -1
            time.sleep(0.1)

        if fd >= 0:
            reset_after = reset_slot_pool(fd)
            os.close(fd)
            fd = -1

        result["slot_pool_after_release"] = query_after or {}
        result["reset_after"] = reset_after or {}
        entries = query_after.get("entries", []) if isinstance(query_after, dict) else []
        mapped_entries = [
            entry for entry in entries
            if isinstance(entry, dict) and int(entry.get("pte_mapped", 0)) != 0
        ]
        release_clear_count = int(query_after.get("pte_release_clear_count", 0)) if isinstance(query_after, dict) else 0
        result["release_clear_checks"] = {
            "mapped_entries": len(mapped_entries),
            "global_release_clear_count": release_clear_count,
            "global_clear_count": int(query_after.get("pte_clear_count", 0)) if isinstance(query_after, dict) else 0,
            "copy_failed_count": int(query_after.get("copy_failed_count", 0)) if isinstance(query_after, dict) else 0,
            "pte_failure_count": int(query_after.get("pte_failure_count", 0)) if isinstance(query_after, dict) else 0,
            "dangerous_op_reject_count": int(query_after.get("pte_dangerous_op_reject_count", 0)) if isinstance(query_after, dict) else 0,
        }
        passed = (
            child.returncode == 0
            and isinstance(query_after, dict)
            and query_after.get("ok", False)
            and len(mapped_entries) == 0
            and release_clear_count >= args.count
            and slot_pool_health_ok(query_after)
            and isinstance(reset_after, dict)
            and reset_after.get("ok", False)
        )
        result["passed"] = passed
        write_json_result(result, args.out_json)
        return 0 if passed else 1
    finally:
        if fd >= 0:
            os.close(fd)


def native_migrate(fd: int, base: int, byte_count: int, gpu_uuid: UvmUuid) -> dict[str, int]:
    params = UvmMigrateParams()
    params.base = base
    params.length = byte_count
    params.destinationUuid = gpu_uuid
    params.flags = 0
    params.semaphoreAddress = 0
    params.semaphorePayload = 0
    params.cpuNumaNode = -1
    ret, errno_value = ioctl(fd, UVM_MIGRATE_IOCTL, params)
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "userSpaceStart": int(params.userSpaceStart),
        "userSpaceLength": int(params.userSpaceLength),
        "ok": ret == 0 and int(params.rmStatus) == 0,
    }


def native_va_range_split(fd: int, split_address: int) -> dict[str, int]:
    params = UvmTestVaRangeSplitParams()
    params.split_address = split_address
    ret, errno_value = ioctl(fd, UVM_TEST_VA_RANGE_SPLIT_IOCTL, params)
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "split_address": int(params.split_address),
        "ok": ret == 0 and int(params.rmStatus) == 0,
    }


def batch_to_result(params: MoeSlotRemapBatchParams, ret: int, errno_value: int) -> dict[str, object]:
    count = int(params.count)
    return {
        "ret": ret,
        "errno": errno_value,
        "rmStatus": int(params.rmStatus),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "conflict_count": int(params.conflict_count),
        "requested_bytes": int(params.requested_bytes),
        "accepted_bytes": int(params.accepted_bytes),
        "remap_count": int(params.remap_count),
        "tlb_invalidate_count": int(params.tlb_invalidate_count),
        "entry_statuses": [int(params.entries[i].status) for i in range(count)],
    }


def fill_remap_entries(
    params: MoeSlotRemapBatchParams,
    *,
    mode: int,
    count: int,
    gpu_uuid: UvmUuid,
    base: int,
    page_size: int,
    validate_managed: bool,
    start_slot: int = 0,
    start_expert: int = 100,
    same_slot_ranges: bool = False,
    slot_offset_stride: int | None = None,
) -> None:
    params.mode = mode
    params.count = count
    params.gpu_uuid = gpu_uuid
    params.validate_managed = 1 if validate_managed else 0
    for index in range(count):
        params.entries[index].layer_id = 7
        params.entries[index].expert_id = start_expert if same_slot_ranges else start_expert + index
        params.entries[index].slot_id = start_slot if same_slot_ranges else start_slot + index
        params.entries[index].role = 3 if index % 2 == 0 else 4
        params.entries[index].va_base = base + index * page_size
        params.entries[index].bytes = page_size
        params.entries[index].slot_offset = (
            index * slot_offset_stride if slot_offset_stride is not None else 0
        )


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_VALUES), default="dry_run")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--validate-managed", action="store_true")
    parser.add_argument("--verify-gpu-read", action="store_true")
    parser.add_argument("--base", type=lambda value: int(value, 0), default=0x1000000000)
    parser.add_argument("--remap-bytes", type=int)
    parser.add_argument("--cross-va-block", action="store_true")
    parser.add_argument(
        "--same-slot-ranges",
        action="store_true",
        help="Submit all entries against one layer/expert/slot with distinct slot offsets.",
    )
    parser.add_argument("--induce-rollback-failure", action="store_true")
    parser.add_argument("--clear-via-native-migrate", action="store_true")
    parser.add_argument("--clear-via-native-unmap", action="store_true")
    parser.add_argument("--clear-via-native-split", action="store_true")
    parser.add_argument("--clear-via-gpu-write-fault", action="store_true")
    parser.add_argument("--clear-via-release", action="store_true")
    parser.add_argument("--release-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--evict-active-via-gpu-copy",
        action="store_true",
        help=(
            "After slot_va_block_remap succeeds, submit more gpu_copy_only slots "
            "to pressure the VIDMEM slot budget and verify active slot PTE eviction."
        ),
    )
    parser.add_argument(
        "--evict-active-temp-budget",
        type=int,
        help=(
            "Temporarily set the UVM MoE slot VIDMEM budget after active slot PTEs "
            "are installed and before the gpu_copy_only eviction pressure is submitted."
        ),
    )
    parser.add_argument(
        "--source-kind",
        choices=("managed", "hmm"),
        default="managed",
        help="Target source allocation for slot PTE modes; hmm uses anonymous mmap to exercise mmu notifier invalidation.",
    )
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if args.count < 0 or args.count > UVM_MOE_SLOT_REMAP_MAX_ENTRIES:
        raise SystemExit(f"--count must be in [0, {UVM_MOE_SLOT_REMAP_MAX_ENTRIES}]")
    slot_pte_modes = ("slot_pte_remap", "slot_va_block_remap")
    if args.induce_rollback_failure and (args.mode != "slot_va_block_remap" or args.count < 2):
        raise SystemExit("--induce-rollback-failure requires --mode slot_va_block_remap --count >= 2")
    if args.cross_va_block and args.mode != "slot_va_block_remap":
        raise SystemExit("--cross-va-block currently applies to --mode slot_va_block_remap")
    if args.clear_via_native_migrate and args.mode not in slot_pte_modes:
        raise SystemExit("--clear-via-native-migrate requires a slot PTE mode")
    if args.clear_via_native_unmap and args.mode not in slot_pte_modes:
        raise SystemExit("--clear-via-native-unmap requires a slot PTE mode")
    if args.clear_via_native_split and args.mode != "slot_va_block_remap":
        raise SystemExit("--clear-via-native-split requires --mode slot_va_block_remap")
    if args.clear_via_gpu_write_fault and args.mode not in slot_pte_modes:
        raise SystemExit("--clear-via-gpu-write-fault requires a slot PTE mode")
    if args.clear_via_release and args.mode not in slot_pte_modes:
        raise SystemExit("--clear-via-release requires a slot PTE mode")
    if args.evict_active_via_gpu_copy and args.mode != "slot_va_block_remap":
        raise SystemExit("--evict-active-via-gpu-copy requires --mode slot_va_block_remap")
    if args.evict_active_temp_budget is not None and not args.evict_active_via_gpu_copy:
        raise SystemExit("--evict-active-temp-budget requires --evict-active-via-gpu-copy")
    if args.evict_active_temp_budget is not None and args.evict_active_temp_budget <= 0:
        raise SystemExit("--evict-active-temp-budget must be positive")
    if args.release_child and args.mode not in slot_pte_modes:
        raise SystemExit("--release-child requires a slot PTE mode")
    if args.release_child and args.clear_via_release:
        raise SystemExit("--release-child is internal and cannot be combined with --clear-via-release")
    if args.induce_rollback_failure and (args.clear_via_release or args.release_child):
        raise SystemExit("release clear smoke requires a successful mapped slot PTE")
    if (args.clear_via_release or args.release_child) and args.count <= 0:
        raise SystemExit("release clear smoke requires --count > 0")
    clear_modes = [
        args.clear_via_native_migrate,
        args.clear_via_native_unmap,
        args.clear_via_native_split,
        args.clear_via_gpu_write_fault,
        args.clear_via_release,
        args.evict_active_via_gpu_copy,
    ]
    if sum(1 for enabled in clear_modes if enabled) > 1:
        raise SystemExit("clear-via options are mutually exclusive")
    if args.source_kind == "hmm" and args.mode != "slot_va_block_remap":
        raise SystemExit("--source-kind hmm currently requires --mode slot_va_block_remap")
    if args.source_kind == "hmm" and args.validate_managed:
        raise SystemExit("--source-kind hmm cannot be combined with --validate-managed")
    if args.source_kind == "hmm" and args.clear_via_native_migrate:
        raise SystemExit("--source-kind hmm currently supports --clear-via-native-unmap, not native migrate")
    if args.source_kind == "hmm" and args.clear_via_native_split:
        raise SystemExit("--source-kind hmm cannot be combined with --clear-via-native-split")
    if args.source_kind == "hmm" and args.clear_via_gpu_write_fault:
        raise SystemExit("--source-kind hmm cannot be combined with --clear-via-gpu-write-fault")
    if args.source_kind == "hmm" and (args.clear_via_release or args.release_child):
        raise SystemExit("--source-kind hmm cannot be combined with release clear smoke")
    if args.clear_via_native_split and args.cross_va_block:
        raise SystemExit("--clear-via-native-split currently uses a single managed VA block; omit --cross-va-block")

    page_size = os.sysconf("SC_PAGE_SIZE")
    remap_bytes = args.remap_bytes if args.remap_bytes is not None else (65536 if args.mode in slot_pte_modes else page_size)
    if remap_bytes <= 0 or remap_bytes % page_size != 0:
        raise SystemExit(f"--remap-bytes must be a positive multiple of {page_size}")
    if args.cross_va_block and remap_bytes < 2 * page_size:
        raise SystemExit("--cross-va-block needs --remap-bytes >= 2 pages")
    if args.same_slot_ranges and args.count < 2:
        raise SystemExit("--same-slot-ranges needs --count >= 2")
    if args.clear_via_release:
        return run_release_parent(args, remap_bytes)
    managed_modes = ("remap", "slot_pte_remap", "slot_va_block_remap")
    gpu_uuid_modes = ("remap", "gpu_copy_only", "slot_pte_remap", "slot_va_block_remap")
    base = args.base - (args.base % page_size)
    source = None
    target_source = None
    slot_source = None
    reset_needed = args.mode in slot_pte_modes
    if args.mode in ("copy_only", "gpu_copy_only") and args.count > 0:
        source = mmap.mmap(-1, args.count * remap_bytes)
        source.write(bytes((index % 251 for index in range(args.count * remap_bytes))))
        source.seek(0)
        source_byte = ctypes.c_char.from_buffer(source)
        base = ctypes.addressof(source_byte)
        del source_byte

    result: dict[str, object] = {
        "mode": args.mode,
        "count": args.count,
        "validate_managed": args.validate_managed,
        "remap_bytes": remap_bytes,
        "cross_va_block": bool(args.cross_va_block),
        "same_slot_ranges": bool(args.same_slot_ranges),
        "induce_rollback_failure": bool(args.induce_rollback_failure),
        "clear_via_native_migrate": bool(args.clear_via_native_migrate),
        "clear_via_native_unmap": bool(args.clear_via_native_unmap),
        "clear_via_native_split": bool(args.clear_via_native_split),
        "clear_via_gpu_write_fault": bool(args.clear_via_gpu_write_fault),
        "clear_via_release": bool(args.clear_via_release),
        "release_child": bool(args.release_child),
        "evict_active_via_gpu_copy": bool(args.evict_active_via_gpu_copy),
        "evict_active_temp_budget": args.evict_active_temp_budget,
        "source_kind": args.source_kind,
        "ioctl": UVM_MOE_SLOT_REMAP_BATCH_IOCTL,
    }

    fd = -1
    mm_fd = -1
    try:
        gpu_uuid = UvmUuid()
        if args.mode in gpu_uuid_modes:
            try:
                initialize_cuda_device(args.device)
                gpu_uuid = gpu_uuid_for_device(args.device)
            except Exception as exc:
                result.update({
                    "passed": False,
                    "gpu_uuid_error": str(exc),
                    "device": args.device,
                })
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1

        if args.mode in managed_modes and args.count > 0:
            alloc_bytes = args.count * remap_bytes
            if args.mode in slot_pte_modes:
                alloc_bytes += UVM_VA_BLOCK_SIZE + remap_bytes
            if args.source_kind == "hmm":
                target_source = AnonymousMmapAllocation(alloc_bytes)
            else:
                target_source = ManagedAllocation(args.device, alloc_bytes)
            raw_base = target_source.address
            if args.cross_va_block:
                block_boundary = align_up(raw_base + remap_bytes, UVM_VA_BLOCK_SIZE)
                cross_prefix = max(page_size, (remap_bytes // 2) // page_size * page_size)
                if cross_prefix >= remap_bytes:
                    cross_prefix = page_size
                base = block_boundary - cross_prefix
                if base < raw_base:
                    base += UVM_VA_BLOCK_SIZE
            else:
                base = align_up(raw_base, remap_bytes) if args.mode in slot_pte_modes else raw_base
            result.update({
                "managed_alloc_bytes": alloc_bytes,
                "managed_raw_base": raw_base,
                "managed_base": base,
                "managed_base_alignment": remap_bytes,
                "managed_base_aligned": base % remap_bytes == 0,
                "managed_crosses_va_block": (
                    base // UVM_VA_BLOCK_SIZE
                    != (base + remap_bytes - 1) // UVM_VA_BLOCK_SIZE
                ),
            })

        fd, fd_source, init_ret, init_errno, init_rm_status = open_initialized_uvm_fd(
            prefer_existing=args.mode in managed_modes,
        )
        result.update({
            "fd_source": fd_source,
            "init_ret": init_ret,
            "init_errno": init_errno,
            "init_rmStatus": init_rm_status,
        })
        if fd < 0:
            result["passed"] = False
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1

        if args.source_kind == "hmm":
            mm_fd, mm_init_ret, mm_init_errno, mm_init_rm_status = open_initialized_uvm_mm_fd(fd)
            mm_already_initialized = (
                mm_init_ret == 0
                and mm_init_rm_status == NV_ERR_INVALID_STATE
                and fd_source.startswith("adopted:")
            )
            mm_not_required = (
                mm_init_ret == 0
                and mm_init_rm_status == NV_WARN_NOTHING_TO_DO
            )
            result.update({
                "mm_fd_opened": mm_fd >= 0,
                "mm_fd_already_initialized": mm_already_initialized,
                "mm_fd_not_required": mm_not_required,
                "mm_init_ret": mm_init_ret,
                "mm_init_errno": mm_init_errno,
                "mm_init_rmStatus": mm_init_rm_status,
            })
            if mm_fd < 0 and not mm_already_initialized and not mm_not_required:
                result["passed"] = False
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1

        if args.mode in slot_pte_modes and args.count > 0:
            reset_before = reset_slot_pool(fd)
            result["reset_before"] = reset_before
            if not reset_before["ok"]:
                result["passed"] = False
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1

            slot_pattern = bytes(((index + 37) % 251 for index in range(args.count * remap_bytes)))
            slot_source = mmap.mmap(-1, len(slot_pattern))
            slot_source.write(slot_pattern)
            slot_source.seek(0)
            slot_source_byte = ctypes.c_char.from_buffer(slot_source)
            slot_base = ctypes.addressof(slot_source_byte)
            del slot_source_byte

            prefill_params = MoeSlotRemapBatchParams()
            fill_remap_entries(
                prefill_params,
                mode=MODE_VALUES["gpu_copy_only"],
                count=args.count,
                gpu_uuid=gpu_uuid,
                base=slot_base,
                page_size=remap_bytes,
                validate_managed=False,
                same_slot_ranges=args.same_slot_ranges,
                slot_offset_stride=remap_bytes if args.same_slot_ranges else None,
            )
            prefill_ret, prefill_errno = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, prefill_params)
            prefill_result = batch_to_result(prefill_params, prefill_ret, prefill_errno)
            result["slot_prefill_gpu_copy_only"] = prefill_result
            prefill_ok = (
                prefill_ret == 0
                and int(prefill_params.rmStatus) == 0
                and int(prefill_params.success_count) == args.count
                and int(prefill_params.failure_count) == 0
                and all(status == ENTRY_OK for status in prefill_result["entry_statuses"])
            )
            if not prefill_ok:
                result["passed"] = False
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1

            cuda_memcpy_host_to_device(base, bytes((0xB5 for _ in range(args.count * remap_bytes))))

        params = MoeSlotRemapBatchParams()
        fill_remap_entries(
            params,
            mode=MODE_VALUES[args.mode],
            count=args.count,
            gpu_uuid=gpu_uuid,
            base=base,
            page_size=remap_bytes,
            validate_managed=args.validate_managed,
            same_slot_ranges=args.same_slot_ranges,
            slot_offset_stride=remap_bytes if args.same_slot_ranges else None,
        )
        if args.induce_rollback_failure and args.count > 1:
            params.entries[args.count - 1].expert_id += 100000
            result["rollback_failure_entry"] = args.count - 1

        remap_ret, remap_errno = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
        entry_statuses = [int(params.entries[i].status) for i in range(args.count)]
        result.update({
            "remap_ret": remap_ret,
            "remap_errno": remap_errno,
            "rmStatus": int(params.rmStatus),
            "success_count": int(params.success_count),
            "failure_count": int(params.failure_count),
            "conflict_count": int(params.conflict_count),
            "requested_bytes": int(params.requested_bytes),
            "accepted_bytes": int(params.accepted_bytes),
            "remap_count": int(params.remap_count),
            "tlb_invalidate_count": int(params.tlb_invalidate_count),
            "entry_statuses": entry_statuses,
        })
        if args.mode in slot_pte_modes:
            result["slot_pool_after_slot_pte_remap"] = query_slot_pool(fd)

        if args.mode in ("dry_run", "validate", "copy_only", "gpu_copy_only") and not args.validate_managed:
            passed = (
                remap_ret == 0
                and int(params.rmStatus) == 0
                and int(params.success_count) == args.count
                and int(params.failure_count) == 0
                and int(params.conflict_count) == 0
                and all(status == ENTRY_OK for status in entry_statuses)
            )
        elif args.mode == "remap":
            passed = (
                remap_ret == 0
                and int(params.rmStatus) == 0
                and int(params.success_count) == args.count
                and int(params.failure_count) == 0
                and int(params.remap_count) == args.count
                and int(params.tlb_invalidate_count) == args.count
                and all(status == ENTRY_OK for status in entry_statuses)
            )
            result["expected_native_uvm_remap_path"] = True
        elif args.mode in slot_pte_modes:
            result["expected_slot_pte_remap_path"] = args.mode == "slot_pte_remap"
            result["expected_slot_va_block_remap_path"] = args.mode == "slot_va_block_remap"
            query_after = result.get("slot_pool_after_slot_pte_remap", {})
            entries = query_after.get("entries", []) if isinstance(query_after, dict) else []

            if args.induce_rollback_failure:
                mapped_entries = [entry for entry in entries if int(entry.get("pte_mapped", 0)) != 0]
                result["rollback_query_checks"] = {
                    "mapped_entries_after_failure": len(mapped_entries),
                    "rollback_count": int(query_after.get("pte_rollback_count", 0)) if isinstance(query_after, dict) else 0,
                    "rollback_bytes": int(query_after.get("pte_rollback_bytes", 0)) if isinstance(query_after, dict) else 0,
                    "failure_count": int(query_after.get("pte_failure_count", 0)) if isinstance(query_after, dict) else 0,
                    "rolled_back_statuses": entry_statuses.count(ENTRY_ROLLED_BACK),
                }
                passed = (
                    int(params.rmStatus) != 0
                    and int(params.failure_count) >= 1
                    and int(params.success_count) == 0
                    and int(params.remap_count) == 0
                    and len(mapped_entries) == 0
                    and entry_statuses.count(ENTRY_ROLLED_BACK) >= 1
                    and int(query_after.get("pte_rollback_count", 0)) >= 1
                    and int(query_after.get("pte_rollback_bytes", 0)) >= remap_bytes
                    and int(query_after.get("pte_failure_count", 0)) >= 1
                )
                if args.verify_gpu_read and args.count > 0:
                    host_bytes = kernel_readback(base, args.count * remap_bytes, args.device)
                    expected_original = bytes((0xB5 for _ in range(args.count * remap_bytes)))
                    result["gpu_readback_after_failed_batch_matches_original"] = host_bytes == expected_original
                    result["gpu_readback_after_failed_batch_prefix"] = host_bytes[:64].hex()
                    passed = passed and bool(result["gpu_readback_after_failed_batch_matches_original"])
            else:
                passed = (
                    remap_ret == 0
                    and int(params.rmStatus) == 0
                    and int(params.success_count) == args.count
                    and int(params.failure_count) == 0
                    and int(params.remap_count) == args.count
                    and int(params.tlb_invalidate_count) == args.count
                    and all(status == ENTRY_OK for status in entry_statuses)
                )
            if not args.induce_rollback_failure and args.mode == "slot_va_block_remap":
                first_entry = entries[0] if entries else {}
                mapped_entries = [entry for entry in entries if int(entry.get("pte_mapped", 0)) != 0]
                total_block_unmaps = sum(int(entry.get("pte_block_unmap_count", 0)) for entry in entries)
                total_tracker_waits = sum(int(entry.get("pte_block_tracker_wait_count", 0)) for entry in entries)
                expected_mapped_slots = 1 if args.same_slot_ranges else args.count
                result["slot_va_block_query_checks"] = {
                    "tracked": bool(first_entry.get("pte_va_block_tracked")),
                    "mode": int(first_entry.get("pte_mode", -1)),
                    "mapped_entries": len(mapped_entries),
                    "expected_mapped_slots": expected_mapped_slots,
                    "same_slot_pte_remap_count": int(first_entry.get("pte_remap_count", 0)),
                    "same_slot_pte_bytes": int(first_entry.get("pte_bytes", 0)),
                    "block_unmap_count": total_block_unmaps,
                    "block_tracker_wait_count": total_tracker_waits,
                }
                passed = (
                    passed
                    and len(mapped_entries) == expected_mapped_slots
                    and all(bool(entry.get("pte_va_block_tracked")) for entry in entries[:expected_mapped_slots])
                    and all(int(entry.get("pte_mode", -1)) == MODE_VALUES["slot_va_block_remap"] for entry in entries[:expected_mapped_slots])
                    and total_block_unmaps >= args.count
                    and total_tracker_waits >= args.count
                    and slot_pool_health_ok(query_after)
                )
                if args.same_slot_ranges:
                    passed = (
                        passed
                        and int(first_entry.get("pte_remap_count", 0)) == args.count
                        and int(first_entry.get("pte_bytes", 0)) == args.count * remap_bytes
                        and int(first_entry.get("copy_count", 0)) == args.count
                        and int(first_entry.get("copy_bytes", 0)) == args.count * remap_bytes
                    )
                if args.cross_va_block:
                    passed = passed and total_block_unmaps >= 2 and total_tracker_waits >= 2
            if passed and not args.induce_rollback_failure and args.verify_gpu_read and args.count > 0:
                host_bytes = kernel_readback(base, args.count * remap_bytes, args.device)
                expected = bytes(((index + 37) % 251 for index in range(args.count * remap_bytes)))
                result["gpu_readback_matches_slot_fill"] = host_bytes == expected
                result["gpu_readback_prefix"] = host_bytes[:64].hex()
                passed = passed and bool(result["gpu_readback_matches_slot_fill"])
            if passed and args.evict_active_via_gpu_copy and not args.induce_rollback_failure and args.count > 0:
                evict_count = args.count
                evict_pattern = bytes(((index + 113) % 251 for index in range(evict_count * remap_bytes)))
                evict_source = mmap.mmap(-1, len(evict_pattern))
                old_vidmem_budget: int | None = None
                try:
                    if args.evict_active_temp_budget is not None:
                        old_vidmem_budget = read_uint_file(VIDMEM_BUDGET_PARAM)
                        write_uint_file(VIDMEM_BUDGET_PARAM, args.evict_active_temp_budget)
                        result["active_evict_temp_budget_applied"] = {
                            "param": VIDMEM_BUDGET_PARAM,
                            "old_budget": old_vidmem_budget,
                            "temp_budget": int(args.evict_active_temp_budget),
                        }

                    evict_source.write(evict_pattern)
                    evict_source.seek(0)
                    evict_source_byte = ctypes.c_char.from_buffer(evict_source)
                    evict_base = ctypes.addressof(evict_source_byte)
                    del evict_source_byte

                    evict_params = MoeSlotRemapBatchParams()
                    fill_remap_entries(
                        evict_params,
                        mode=MODE_VALUES["gpu_copy_only"],
                        count=evict_count,
                        gpu_uuid=gpu_uuid,
                        base=evict_base,
                        page_size=remap_bytes,
                        validate_managed=False,
                        start_slot=args.count,
                        start_expert=1000,
                    )
                    evict_ret, evict_errno = ioctl(fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, evict_params)
                    evict_result = batch_to_result(evict_params, evict_ret, evict_errno)
                    query_after_evict = query_slot_pool(fd, max_entries=max(8, args.count * 2))
                    entries_after_evict = query_after_evict.get("entries", []) if isinstance(query_after_evict, dict) else []
                    mapped_after_evict = [
                        entry for entry in entries_after_evict
                        if int(entry.get("pte_mapped", 0)) != 0
                    ]
                    total_evict_clears = sum(
                        int(entry.get("pte_evict_clear_count", 0))
                        for entry in entries_after_evict
                    )
                    evicted_readback = kernel_readback(base, args.count * remap_bytes, args.device)
                    expected_original = bytes((0xB5 for _ in range(args.count * remap_bytes)))
                    result["active_evict_gpu_copy_only"] = evict_result
                    result["slot_pool_after_active_evict"] = query_after_evict
                    result["active_evict_clear_checks"] = {
                        "mapped_entries": len(mapped_after_evict),
                        "entry_evict_clear_count": total_evict_clears,
                        "global_evict_clear_count": int(query_after_evict.get("pte_evict_clear_count", 0)),
                        "global_clear_count": int(query_after_evict.get("pte_clear_count", 0)),
                        "vidmem_evict_count": int(query_after_evict.get("vidmem_evict_count", 0)),
                        "copy_failed_count": int(query_after_evict.get("copy_failed_count", 0)),
                        "pte_failure_count": int(query_after_evict.get("pte_failure_count", 0)),
                        "dangerous_op_reject_count": int(query_after_evict.get("pte_dangerous_op_reject_count", 0)),
                    }
                    result["gpu_readback_after_active_evict_matches_original"] = evicted_readback == expected_original
                    result["gpu_readback_after_active_evict_prefix"] = evicted_readback[:64].hex()
                    evict_ok = (
                        evict_ret == 0
                        and int(evict_params.rmStatus) == 0
                        and int(evict_params.success_count) == evict_count
                        and int(evict_params.failure_count) == 0
                        and all(status == ENTRY_OK for status in evict_result["entry_statuses"])
                        and query_after_evict.get("ok", False)
                        and int(query_after_evict.get("vidmem_evict_count", 0)) > 0
                        and int(query_after_evict.get("pte_evict_clear_count", 0)) >= args.count
                        and int(query_after_evict.get("copy_failed_count", 0)) == 0
                        and int(query_after_evict.get("pte_failure_count", 0)) == 0
                        and int(query_after_evict.get("pte_dangerous_op_reject_count", 0)) == 0
                        and slot_pool_health_ok(query_after_evict)
                        and bool(result["gpu_readback_after_active_evict_matches_original"])
                    )
                    passed = passed and evict_ok
                finally:
                    if old_vidmem_budget is not None:
                        try:
                            write_uint_file(VIDMEM_BUDGET_PARAM, old_vidmem_budget)
                            result["active_evict_temp_budget_restored"] = old_vidmem_budget
                        except Exception as exc:
                            result["active_evict_temp_budget_restore_error"] = str(exc)
                            passed = False
                    evict_source.close()
            if passed and args.clear_via_native_migrate and not args.induce_rollback_failure and args.count > 0:
                native_result = native_migrate(fd, base, args.count * remap_bytes, gpu_uuid)
                query_after_native = query_slot_pool(fd)
                entries_after_native = query_after_native.get("entries", []) if isinstance(query_after_native, dict) else []
                mapped_after_native = [
                    entry for entry in entries_after_native
                    if int(entry.get("pte_mapped", 0)) != 0
                ]
                total_migrate_clears = sum(
                    int(entry.get("pte_migrate_clear_count", 0))
                    for entry in entries_after_native
                )
                result["native_migrate_after_slot_pte"] = native_result
                result["slot_pool_after_native_migrate"] = query_after_native
                result["native_migrate_clear_checks"] = {
                    "mapped_entries": len(mapped_after_native),
                    "entry_migrate_clear_count": total_migrate_clears,
                    "global_migrate_clear_count": int(query_after_native.get("pte_migrate_clear_count", 0)),
                    "global_clear_count": int(query_after_native.get("pte_clear_count", 0)),
                }
                native_clear_ok = (
                    native_result["ok"]
                    and query_after_native.get("ok", False)
                    and len(mapped_after_native) == 0
                    and total_migrate_clears >= args.count
                    and int(query_after_native.get("pte_migrate_clear_count", 0)) >= args.count
                    and slot_pool_health_ok(query_after_native)
                )
                if args.verify_gpu_read:
                    after_native_bytes = kernel_readback(base, args.count * remap_bytes, args.device)
                    expected_original = bytes((0xB5 for _ in range(args.count * remap_bytes)))
                    result["gpu_readback_after_native_migrate_matches_original"] = after_native_bytes == expected_original
                    result["gpu_readback_after_native_migrate_prefix"] = after_native_bytes[:64].hex()
                    native_clear_ok = native_clear_ok and bool(result["gpu_readback_after_native_migrate_matches_original"])
                passed = passed and native_clear_ok
            if passed and args.clear_via_native_split and not args.induce_rollback_failure and args.count > 0:
                split_address = base + page_size - 1
                native_result = native_va_range_split(fd, split_address)
                query_after_native = query_slot_pool(fd)
                entries_after_native = query_after_native.get("entries", []) if isinstance(query_after_native, dict) else []
                mapped_after_native = [
                    entry for entry in entries_after_native
                    if int(entry.get("pte_mapped", 0)) != 0
                ]
                total_split_clears = sum(
                    int(entry.get("pte_split_clear_count", 0))
                    for entry in entries_after_native
                )
                result["native_split_after_slot_pte"] = native_result
                result["slot_pool_after_native_split"] = query_after_native
                result["native_split_clear_checks"] = {
                    "mapped_entries": len(mapped_after_native),
                    "entry_split_clear_count": total_split_clears,
                    "global_split_clear_count": int(query_after_native.get("pte_split_clear_count", 0)),
                    "global_clear_count": int(query_after_native.get("pte_clear_count", 0)),
                }
                native_clear_ok = (
                    native_result["ok"]
                    and query_after_native.get("ok", False)
                    and len(mapped_after_native) == 0
                    and total_split_clears >= args.count
                    and int(query_after_native.get("pte_split_clear_count", 0)) >= args.count
                    and slot_pool_health_ok(query_after_native)
                )
                if args.verify_gpu_read:
                    after_native_bytes = kernel_readback(base, args.count * remap_bytes, args.device)
                    expected_original = bytes((0xB5 for _ in range(args.count * remap_bytes)))
                    result["gpu_readback_after_native_split_matches_original"] = after_native_bytes == expected_original
                    result["gpu_readback_after_native_split_prefix"] = after_native_bytes[:64].hex()
                    native_clear_ok = native_clear_ok and bool(result["gpu_readback_after_native_split_matches_original"])
                passed = passed and native_clear_ok
            if passed and args.clear_via_gpu_write_fault and not args.induce_rollback_failure and args.count > 0:
                write_bytes = args.count * remap_bytes
                kernel_fill(base, 0xC3, write_bytes, args.device)
                query_after_fault = query_slot_pool(fd)
                entries_after_fault = query_after_fault.get("entries", []) if isinstance(query_after_fault, dict) else []
                mapped_after_fault = [
                    entry for entry in entries_after_fault
                    if int(entry.get("pte_mapped", 0)) != 0
                ]
                total_fault_clears = sum(
                    int(entry.get("pte_fault_clear_count", 0))
                    for entry in entries_after_fault
                )
                result["slot_pool_after_gpu_write_fault"] = query_after_fault
                result["gpu_write_fault_clear_checks"] = {
                    "mapped_entries": len(mapped_after_fault),
                    "entry_fault_clear_count": total_fault_clears,
                    "global_fault_clear_count": int(query_after_fault.get("pte_fault_clear_count", 0)),
                    "global_clear_count": int(query_after_fault.get("pte_clear_count", 0)),
                }
                fault_clear_ok = (
                    query_after_fault.get("ok", False)
                    and len(mapped_after_fault) == 0
                    and total_fault_clears >= args.count
                    and int(query_after_fault.get("pte_fault_clear_count", 0)) >= args.count
                    and slot_pool_health_ok(query_after_fault)
                )
                if args.verify_gpu_read:
                    after_fault_bytes = kernel_readback(base, write_bytes, args.device)
                    expected_written = bytes((0xC3 for _ in range(write_bytes)))
                    result["gpu_readback_after_gpu_write_fault_matches_written"] = after_fault_bytes == expected_written
                    result["gpu_readback_after_gpu_write_fault_prefix"] = after_fault_bytes[:64].hex()
                    fault_clear_ok = fault_clear_ok and bool(result["gpu_readback_after_gpu_write_fault_matches_written"])
                passed = passed and fault_clear_ok
            if passed and args.clear_via_native_unmap and not args.induce_rollback_failure and args.count > 0:
                if target_source is None:
                    result["native_unmap_after_slot_pte"] = {
                        "ok": False,
                        "error": "target allocation is not available",
                    }
                    passed = False
                else:
                    target_source.close()
                    target_source = None
                    query_after_native = query_slot_pool(fd)
                    entries_after_native = query_after_native.get("entries", []) if isinstance(query_after_native, dict) else []
                    mapped_after_native = [
                        entry for entry in entries_after_native
                        if int(entry.get("pte_mapped", 0)) != 0
                    ]
                    total_unmap_clears = sum(
                        int(entry.get("pte_unmap_clear_count", 0))
                        for entry in entries_after_native
                    )
                    result["native_unmap_after_slot_pte"] = {
                        "ok": True,
                    }
                    result["slot_pool_after_native_unmap"] = query_after_native
                    result["native_unmap_clear_checks"] = {
                        "mapped_entries": len(mapped_after_native),
                        "entry_unmap_clear_count": total_unmap_clears,
                        "global_unmap_clear_count": int(query_after_native.get("pte_unmap_clear_count", 0)),
                        "global_clear_count": int(query_after_native.get("pte_clear_count", 0)),
                    }
                    passed = (
                        passed
                        and query_after_native.get("ok", False)
                        and len(mapped_after_native) == 0
                        and total_unmap_clears >= args.count
                        and int(query_after_native.get("pte_unmap_clear_count", 0)) >= args.count
                        and slot_pool_health_ok(query_after_native)
                    )
        else:
            passed = remap_ret == 0

        if args.release_child:
            result["release_child_leaving_slot_pte_mapped"] = bool(passed)
            result["passed"] = passed
            write_json_result(result, args.out_json)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if passed else 1)

        if fd >= 0 and reset_needed:
            try:
                reset_after = reset_slot_pool(fd)
                result["reset_after"] = reset_after
                passed = passed and bool(reset_after["ok"])
                passed = passed and slot_pool_health_ok(reset_after)
                if (
                    passed
                    and args.mode == "slot_va_block_remap"
                    and args.verify_gpu_read
                    and not args.clear_via_native_unmap
                    and args.count > 0
                ):
                    after_reset_bytes = kernel_readback(base, args.count * remap_bytes, args.device)
                    if args.clear_via_gpu_write_fault:
                        expected_after_reset = bytes((0xC3 for _ in range(args.count * remap_bytes)))
                    else:
                        expected_after_reset = bytes((0xB5 for _ in range(args.count * remap_bytes)))
                    result["gpu_readback_after_reset_matches_original"] = after_reset_bytes == expected_after_reset
                    result["gpu_readback_after_reset_prefix"] = after_reset_bytes[:64].hex()
                    passed = passed and bool(result["gpu_readback_after_reset_matches_original"])
                reset_needed = False
            except Exception as exc:
                result["reset_after_error"] = str(exc)
                passed = False
                reset_needed = False

        result["passed"] = passed
        write_json_result(result, args.out_json)
        return 0 if passed else 1
    except OSError as exc:
        result.update({
            "passed": False,
            "error": str(exc),
            "errno": getattr(exc, "errno", None),
        })
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    finally:
        if fd >= 0 and reset_needed:
            try:
                result["reset_after"] = reset_slot_pool(fd)
            except Exception as exc:
                result["reset_after_error"] = str(exc)
        if source is not None:
            source.close()
        if slot_source is not None:
            slot_source.close()
        if target_source is not None:
            target_source.close()
        if mm_fd >= 0:
            os.close(mm_fd)
        if fd >= 0:
            os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
