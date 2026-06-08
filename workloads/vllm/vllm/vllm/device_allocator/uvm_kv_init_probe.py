# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch

_TRACE_MARKER_PATHS = (
    "/sys/kernel/tracing/trace_marker",
    "/sys/kernel/debug/tracing/trace_marker",
)
_LOG_LOCK = threading.Lock()
_TRACE_MARKER_PATH: str | None = None
_TRACE_MARKER_MISSING = False


def kv_init_probe_enabled() -> bool:
    return os.environ.get("VLLM_UVM_KV_INIT_FAULT_PROBE", "0") == "1"


def _sync_enabled() -> bool:
    return os.environ.get("VLLM_UVM_KV_INIT_FAULT_PROBE_SYNC", "0") == "1"


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


def _append_jsonl(record: dict[str, Any]) -> None:
    path = _probe_log_path()
    if path is None:
        return

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _LOG_LOCK:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        except Exception:
            pass


def _find_trace_marker_path() -> str | None:
    global _TRACE_MARKER_MISSING, _TRACE_MARKER_PATH

    if _TRACE_MARKER_MISSING:
        return None
    if _TRACE_MARKER_PATH is not None:
        return _TRACE_MARKER_PATH

    for path in _TRACE_MARKER_PATHS:
        if os.path.exists(path):
            _TRACE_MARKER_PATH = path
            return path

    _TRACE_MARKER_MISSING = True
    return None


def _write_trace_marker(fields: dict[str, Any]) -> None:
    path = _find_trace_marker_path()
    if path is None:
        return

    parts = ["VLLM_KV_INIT_PROBE"]
    for key in sorted(fields):
        value = fields[key]
        if isinstance(value, (list, tuple)):
            text = "x".join(str(item) for item in value)
        else:
            text = str(value)
        text = text.replace(" ", "_")
        parts.append(f"{key}={text}")

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(" ".join(parts))
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
    _append_jsonl(record)

    marker_fields = {
        "event": record.get("event"),
        "path": record.get("path"),
        "index": record.get("index"),
        "pid": record.get("pid"),
    }
    for key in (
        "requested_size_bytes",
        "size_bytes",
        "ptr_hex",
        "end_hex",
        "dtype",
        "device",
    ):
        if key in record:
            marker_fields[key] = record[key]
    _write_trace_marker(marker_fields)


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
