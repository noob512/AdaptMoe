# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

import torch

_LOG_LOCK = threading.Lock()


def kv_init_probe_enabled() -> bool:
    return os.environ.get("VLLM_UVM_KV_INIT_FAULT_PROBE", "0") == "1"


def _sync_enabled() -> bool:
    return os.environ.get("VLLM_UVM_KV_INIT_FAULT_PROBE_SYNC", "0") == "1"


def _query_residency_enabled() -> bool:
    return os.environ.get(
        "VLLM_UVM_KV_INIT_PROBE_QUERY_RESIDENCY", "0"
    ) == "1"


def _query_residency_round_pages() -> bool:
    return os.environ.get(
        "VLLM_UVM_KV_INIT_PROBE_QUERY_RESIDENCY_ROUND_PAGES", "1"
    ) != "0"


def _query_residency_device(tensor: torch.Tensor) -> int:
    raw = os.environ.get("VLLM_UVM_KV_INIT_PROBE_QUERY_RESIDENCY_DEVICE", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return 0
    try:
        return int(tensor.device.index or 0)
    except Exception:
        return 0


def _probe_log_path() -> str | None:
    path = os.environ.get("VLLM_UVM_KV_INIT_FAULT_PROBE_FILE", "")
    return path or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {}

    try:
        ptr = int(tensor.data_ptr())
    except Exception:
        ptr = 0

    try:
        numel = int(tensor.numel())
    except Exception:
        numel = 0

    try:
        element_size = int(tensor.element_size())
    except Exception:
        element_size = 0

    size_bytes = numel * element_size
    end = ptr + size_bytes - 1 if ptr and size_bytes > 0 else 0

    return {
        "ptr": ptr,
        "ptr_hex": f"0x{ptr:x}" if ptr else "0x0",
        "end": end,
        "end_hex": f"0x{end:x}" if end else "0x0",
        "numel": numel,
        "element_size": element_size,
        "size_bytes": size_bytes,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "is_cuda": bool(tensor.is_cuda),
    }


def _tensor_residency_metadata(tensor: torch.Tensor | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "residency_query_enabled": bool(_query_residency_enabled()),
        "residency_query_ok": False,
        "residency_query_status": -1,
        "residency_query_device": None,
        "residency_queried_bytes": 0,
        "resident_gpu_bytes": 0,
        "resident_cpu_bytes": 0,
        "resident_any_bytes": 0,
        "resident_unknown_bytes": 0,
        "resident_gpu_ratio": None,
        "resident_cpu_ratio": None,
        "resident_unknown_ratio": None,
        "residency_state": "not_queried",
    }
    if not _query_residency_enabled():
        return result
    if tensor is None:
        result["residency_state"] = "no_tensor"
        return result
    if not bool(tensor.is_cuda):
        result["residency_state"] = "not_cuda"
        return result

    meta = _tensor_metadata(tensor)
    ptr = int(meta.get("ptr") or 0)
    size_bytes = int(meta.get("size_bytes") or 0)
    if ptr <= 0 or size_bytes <= 0:
        result["residency_state"] = "invalid_range"
        return result

    query_ptr = ptr
    query_size = size_bytes
    if _query_residency_round_pages():
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError, AttributeError):
            page_size = 4096
        if page_size <= 0:
            page_size = 4096
        query_start = ptr - (ptr % page_size)
        query_end = ((ptr + size_bytes + page_size - 1) // page_size) * page_size
        query_ptr = query_start
        query_size = max(query_end - query_start, size_bytes)

    device = _query_residency_device(tensor)
    result["residency_query_device"] = device
    result["residency_query_start_int"] = query_ptr
    result["residency_query_size_bytes"] = query_size
    result["residency_query_granularity"] = (
        "page_rounded" if _query_residency_round_pages() else "exact"
    )
    try:
        from vllm.device_allocator.uvm import (
            driver_batch_query_ranges_residency_bytes,
        )

        residency = driver_batch_query_ranges_residency_bytes(
            [(query_ptr, query_size)], device=device
        )
    except Exception as exc:
        result["residency_state"] = "query_exception"
        result["residency_query_error"] = repr(exc)
        return result

    queried_bytes = int(residency.get("queried_bytes") or 0)
    gpu_bytes = int(residency.get("gpu_bytes") or 0)
    cpu_bytes = int(residency.get("cpu_bytes") or 0)
    unknown_bytes = int(residency.get("unknown_bytes") or 0)
    result.update(
        {
            "residency_query_ok": bool(residency.get("ok")),
            "residency_query_status": int(residency.get("status") or 0),
            "residency_queried_bytes": queried_bytes,
            "resident_gpu_bytes": gpu_bytes,
            "resident_cpu_bytes": cpu_bytes,
            "resident_any_bytes": int(residency.get("any_bytes") or 0),
            "resident_unknown_bytes": unknown_bytes,
        }
    )
    denom = queried_bytes if queried_bytes > 0 else query_size
    if denom > 0:
        result["resident_gpu_ratio"] = float(gpu_bytes) / float(denom)
        result["resident_cpu_ratio"] = float(cpu_bytes) / float(denom)
        result["resident_unknown_ratio"] = float(unknown_bytes) / float(denom)
    if not bool(residency.get("ok")):
        result["residency_state"] = "query_failed"
    elif queried_bytes <= 0:
        result["residency_state"] = "query_empty"
    elif gpu_bytes >= queried_bytes:
        result["residency_state"] = "gpu_full"
    elif gpu_bytes > 0:
        result["residency_state"] = "gpu_partial"
    elif cpu_bytes >= queried_bytes:
        result["residency_state"] = "cpu_full"
    elif cpu_bytes > 0:
        result["residency_state"] = "cpu_partial"
    elif unknown_bytes > 0:
        result["residency_state"] = "unknown"
    else:
        result["residency_state"] = "not_resident_reported"
    return result


def _append_jsonl(record: dict[str, Any]) -> None:
    path = _probe_log_path()
    if path is None:
        return

    with suppress(Exception):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _LOG_LOCK:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        except Exception:
            pass






def record_kv_init_probe_event(
    event: str,
    *,
    path: str,
    index: int | str,
    tensor: torch.Tensor | None = None,
    size_bytes: int | None = None,
    device: Any = None,
    shape: Sequence[int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    if not kv_init_probe_enabled():
        return

    record: dict[str, Any] = {
        "event": event,
        "path": path,
        "index": index,
        "ts": time.time(),
        "pid": os.getpid(),
    }
    if size_bytes is not None:
        record["requested_size_bytes"] = int(size_bytes)
    if device is not None:
        record["requested_device"] = str(device)
    if shape is not None:
        record["requested_shape"] = [int(item) for item in shape]
    if extra:
        record.update({str(key): _json_safe(val) for key, val in extra.items()})

    record.update(_tensor_metadata(tensor))
    record.update(_tensor_residency_metadata(tensor))
    _append_jsonl(record)

def synchronize_kv_init_probe_tensor(
    *,
    path: str,
    index: int | str,
    tensor: torch.Tensor,
) -> None:
    if not kv_init_probe_enabled() or not _sync_enabled():
        return

    record_kv_init_probe_event(
        "kv_zeros_sync_begin",
        path=path,
        index=index,
        tensor=tensor,
    )
    try:
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        else:
            torch.cuda.synchronize()
        status = "ok"
    except Exception as exc:
        status = f"{type(exc).__name__}:{exc}"

    record_kv_init_probe_event(
        "kv_zeros_sync_end",
        path=path,
        index=index,
        tensor=tensor,
        extra={"sync_status": status},
    )
