#!/usr/bin/env python3
"""Standalone real-checkpoint hot/cold MoE expert layout prototype.

This script does not start vLLM. It reads Qwen3 FP8 expert weights directly
from safetensors and validates the proposed explicit layout:

  hot experts 0..63: per-layer GPU resident tensors
  cold experts 64..127: copied layer-by-layer into a 2-layer GPU slot pool

The script also runs a tiny isolated MoE-style pointer check. For sampled
experts, it computes the same small forward from the original checkpoint tensor
and from the split hot/cold GPU pointer path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

DEFAULT_CHECKPOINT_DIR = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/"
    "snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
)


def _now() -> float:
    return time.perf_counter()


def _gib(value: int | float) -> float:
    return float(value) / (1024**3)


def _gbps(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(num_bytes) / seconds / 1e9


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def _parse_layers(args: argparse.Namespace) -> list[int]:
    if args.layers:
        layers: list[int] = []
        for part in args.layers.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                if end < start:
                    raise ValueError(f"invalid layer range {part!r}")
                layers.extend(range(start, end + 1))
            else:
                layers.append(int(part))
        return layers
    return list(range(args.layer_start, args.layer_start + args.layer_count))


def _maybe_uvm_context(enabled: bool) -> Iterator[None]:
    if not enabled:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        from vllm.device_allocator.uvm import uvm_local_memory_pool

        with uvm_local_memory_pool():
            yield

    return _ctx()


def _prefetch_tensor_to_device(tensor: torch.Tensor, device: int) -> bool:
    try:
        from vllm.device_allocator.uvm import prefetch_range_to_device

        return bool(
            prefetch_range_to_device(
                int(tensor.data_ptr()),
                int(tensor.numel() * tensor.element_size()),
                device,
            )
        )
    except Exception as exc:
        print(f"[checkpoint-bench] UVM prefetch failed: {exc}", file=sys.stderr)
        return False


def _alloc_gpu_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    backing: str,
) -> tuple[torch.Tensor, str]:
    if backing == "cuda":
        return torch.empty(shape, dtype=dtype, device=device), "cuda"
    if backing == "uvm":
        with _maybe_uvm_context(True):
            return torch.empty(shape, dtype=dtype, device=device), "uvm"
    if backing == "auto":
        try:
            with _maybe_uvm_context(True):
                return torch.empty(shape, dtype=dtype, device=device), "uvm"
        except Exception as exc:
            print(
                f"[checkpoint-bench] UVM allocation failed, falling back to cuda: {exc}",
                file=sys.stderr,
            )
            return torch.empty(shape, dtype=dtype, device=device), "cuda"
    raise ValueError(f"unknown backing {backing!r}")


def _cuda_view_from_pinned_cpu(cpu_tensor: torch.Tensor) -> torch.Tensor:
    import vllm._C  # noqa: F401  # register torch.ops._C custom ops
    from vllm.utils.torch_utils import get_cuda_view_from_cpu_tensor

    if not cpu_tensor.is_pinned():
        raise RuntimeError("UVA backing requires pinned CPU tensors")
    return get_cuda_view_from_cpu_tensor(cpu_tensor)


def _empty_cpu(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    try:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    except RuntimeError:
        return torch.empty(shape, dtype=dtype, device="cpu")


class CheckpointReader:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        index_path = checkpoint_dir / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._stack = contextlib.ExitStack()
        self._handles: dict[str, Any] = {}

    def close(self) -> None:
        self._stack.close()

    def _handle(self, filename: str) -> Any:
        handle = self._handles.get(filename)
        if handle is None:
            handle = self._stack.enter_context(
                safe_open(self.checkpoint_dir / filename, framework="pt", device="cpu")
            )
            self._handles[filename] = handle
        return handle

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        return self._handle(filename).get_tensor(name)

    def expert_tensor(self, layer: int, expert: int, proj: str) -> torch.Tensor:
        name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        return self.tensor(name)

    def expert_shapes(self, layer: int, expert: int) -> dict[str, Any]:
        gate = self.expert_tensor(layer, expert, "gate_proj")
        up = self.expert_tensor(layer, expert, "up_proj")
        down = self.expert_tensor(layer, expert, "down_proj")
        return {
            "gate_shape": list(gate.shape),
            "up_shape": list(up.shape),
            "down_shape": list(down.shape),
            "dtype": str(gate.dtype),
            "w13_shape": [gate.shape[0] + up.shape[0], gate.shape[1]],
            "w2_shape": list(down.shape),
        }


def _copy_expert_to_batch(
    reader: CheckpointReader,
    *,
    layer: int,
    expert: int,
    local: int,
    batch_w13: torch.Tensor,
    batch_w2: torch.Tensor,
) -> None:
    gate = reader.expert_tensor(layer, expert, "gate_proj")
    up = reader.expert_tensor(layer, expert, "up_proj")
    down = reader.expert_tensor(layer, expert, "down_proj")
    gate_rows = gate.shape[0]
    batch_w13[local, :gate_rows].copy_(gate)
    batch_w13[local, gate_rows : gate_rows + up.shape[0]].copy_(up)
    batch_w2[local].copy_(down)


def _load_expert_batch(
    reader: CheckpointReader,
    *,
    layer: int,
    expert_start: int,
    expert_count: int,
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    start = _now()
    batch_w13 = _empty_cpu(
        (expert_count, w13_shape[0], w13_shape[1]), dtype, pin_memory=pin_memory
    )
    batch_w2 = _empty_cpu(
        (expert_count, w2_shape[0], w2_shape[1]), dtype, pin_memory=pin_memory
    )
    for local in range(expert_count):
        expert = expert_start + local
        _copy_expert_to_batch(
            reader,
            layer=layer,
            expert=expert,
            local=local,
            batch_w13=batch_w13,
            batch_w2=batch_w2,
        )
    return batch_w13, batch_w2, _now() - start


def _tiny_forward(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    x: torch.Tensor,
) -> torch.Tensor:
    w13_f = w13[: 2 * intermediate, :hidden].float()
    w2_f = w2[:hidden, :intermediate].float()
    gate = w13_f[:intermediate]
    up = w13_f[intermediate : 2 * intermediate]
    hidden_in = x[:hidden].float().to(w13_f.device)
    interm = torch.matmul(gate, hidden_in) * torch.matmul(up, hidden_in)
    return torch.matmul(w2_f, interm)


def _pointer_check_one(
    reader: CheckpointReader,
    *,
    layer: int,
    expert: int,
    local: int,
    split_w13: torch.Tensor,
    split_w2: torch.Tensor,
    w13_shape: tuple[int, int],
    w2_shape: tuple[int, int],
    dtype: torch.dtype,
    pin_memory: bool,
    hidden: int,
    intermediate: int,
    x_cpu: torch.Tensor,
) -> dict[str, Any]:
    ref_w13 = _empty_cpu((1, w13_shape[0], w13_shape[1]), dtype, pin_memory=False)
    ref_w2 = _empty_cpu((1, w2_shape[0], w2_shape[1]), dtype, pin_memory=False)
    _copy_expert_to_batch(
        reader,
        layer=layer,
        expert=expert,
        local=0,
        batch_w13=ref_w13,
        batch_w2=ref_w2,
    )
    ref_w13_gpu = ref_w13[0].to(split_w13.device)
    ref_w2_gpu = ref_w2[0].to(split_w13.device)
    ref_out = _tiny_forward(
        ref_w13_gpu,
        ref_w2_gpu,
        hidden=hidden,
        intermediate=intermediate,
        x=x_cpu.to(split_w13.device),
    )
    split_out = _tiny_forward(
        split_w13[local],
        split_w2[local],
        hidden=hidden,
        intermediate=intermediate,
        x=x_cpu.to(split_w13.device),
    )
    max_abs = float((ref_out - split_out).abs().max().detach().cpu().item())
    checksum_abs = float(split_out.abs().sum().item())
    return {
        "layer": layer,
        "expert": expert,
        "local": local,
        "max_abs_diff": max_abs,
        "split_checksum_abs": checksum_abs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    # =========================================================================
    # 1. 参数校验与冷、热、UVM 专家边界计算
    # =========================================================================
    # 解析出当前需要执行测试的层编号列表
    layers = _parse_layers(args)
    if not layers:
        raise ValueError("at least one layer is required")
        
    # 如果未显式指定流式专家数量，默认将所有冷专家都作为流式专家处理
    stream_experts = args.stream_experts
    if stream_experts is None:
        stream_experts = args.cold_experts
        
    # 如果未显式指定 UVM 专家数量，则通过：总专家数 - 热专家数 - 流式专家数 自动推导
    uvm_experts = args.num_experts - args.hot_experts - stream_experts
    if uvm_experts is None:
        uvm_experts = args.num_experts - args.hot_experts - stream_experts
        
    # 确保专家数量非负
    if stream_experts < 0 or uvm_experts < 0:
        raise ValueError("stream_experts and uvm_experts must be non-negative")
        
    # 校验守恒定律 1：冷专家总数必须等于 流式专家 + UVM 专家
    if args.cold_experts != stream_experts + uvm_experts:
        raise ValueError("cold_experts must equal stream_experts + uvm_experts")
        
    # 校验守恒定律 2：总专家数必须等于 热专家 + 流式专家 + UVM 专家
    if args.num_experts != args.hot_experts + stream_experts + uvm_experts:
        raise ValueError("num_experts must equal hot_experts + stream_experts + uvm_experts")
        
    # 缓存插槽的层数空间必须大于 0
    if args.slot_layers <= 0:
        raise ValueError("slot_layers must be positive")
        
    # 计算三类专家在全局索引（0 ~ num_experts）中的起始边界
    hot_start_expert = 0                               # 热专家从 0 开始
    stream_start_expert = args.hot_experts             # 流式专家紧随热专家之后
    uvm_start_expert = args.hot_experts + stream_experts # UVM 专家在最后

    # =========================================================================
    # 2. 检查点（模型权重）元数据读取与显存占用预估
    # =========================================================================
    # 初始化检查点读取器
    reader = CheckpointReader(args.checkpoint_dir)
    try:
        # 读取第一层、第一个专家的张量形状（MoE 中各专家形状通常一致）
        shapes = reader.expert_shapes(layers[0], 0)
        # 获取门控投影权重（gate_proj）的数据类型（如 float16, bfloat16）
        dtype = reader.expert_tensor(layers[0], 0, "gate_proj").dtype
        
        # 提取各个投影矩阵的维度（w13 通常合并了 gate_proj 和 up_proj，w2 为 down_proj）
        w13_shape = tuple(shapes["w13_shape"])
        w2_shape = tuple(shapes["w2_shape"])
        
        # 计算单个专家占用的字节数：(w13元素数 + w2元素数) * 每个元素的字节数
        per_expert_bytes = (
            math.prod(w13_shape) + math.prod(w2_shape)
        ) * _dtype_bytes(dtype)
        
        # 构建显存预估字典
        estimate = {
            **shapes,
            "per_expert_bytes": per_expert_bytes,
            "per_expert_mib": per_expert_bytes / (1024**2),
            # 热专家常驻显存：总层数 * 每层热专家数 * 单个专家字节数
            "hot_gpu_bytes": len(layers) * args.hot_experts * per_expert_bytes,
            "hot_gpu_gib": _gib(len(layers) * args.hot_experts * per_expert_bytes),
            # 流式插槽预留显存：插槽层数 * 流式专家数 * 单个专家字节数
            "cold_slot_gpu_bytes": args.slot_layers * stream_experts * per_expert_bytes,
            "cold_slot_gpu_gib": _gib(
                args.slot_layers * stream_experts * per_expert_bytes
            ),
            # UVM/UVA 占用的系统内存或显存空间
            "uvm_or_uva_bytes": len(layers) * uvm_experts * per_expert_bytes,
            "uvm_or_uva_gib": _gib(
                len(layers) * uvm_experts * per_expert_bytes
            ),
            # 全量流式检查点的总大小
            "streamed_checkpoint_bytes": len(layers) * args.num_experts * per_expert_bytes,
            "streamed_checkpoint_gib": _gib(
                len(layers) * args.num_experts * per_expert_bytes
            ),
        }
        
        # 如果是空运行（--dry-run），直接返回配置和预估数据，不初始化 GPU 资源
        if args.dry_run:
            return {
                "dry_run": True,
                "config": _jsonable_config(args),
                "layers": layers,
                "estimate": estimate,
                "tiers": {
                    "hot": [hot_start_expert, stream_start_expert],
                    "stream": [stream_start_expert, uvm_start_expert],
                    "uvm": [uvm_start_expert, args.num_experts],
                },
                "passed": True,
            }

        # =========================================================================
        # 3. 硬件环境初始化与环境设置
        # =========================================================================
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # 设置随机种子与 PyTorch 种子以确保路由模拟可复现
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # 绑定并设置当前指定的 GPU 设备
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)

        # 初始化正式的结果报告字典
        result: dict[str, Any] = {
            "dry_run": False,
            "config": _jsonable_config(args),
            "layers": layers,
            "estimate": estimate,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "events": [],
            "tiers": {
                "hot": {
                    "expert_start": hot_start_expert,
                    "expert_end_exclusive": stream_start_expert,
                    "count": args.hot_experts,
                    "backing": args.hot_backing,
                },
                "stream": {
                    "expert_start": stream_start_expert,
                    "expert_end_exclusive": uvm_start_expert,
                    "count": stream_experts,
                    "slot_layers": args.slot_layers,
                    "backing": args.slot_backing,
                },
                "uvm": {
                    "expert_start": uvm_start_expert,
                    "expert_end_exclusive": args.num_experts,
                    "count": uvm_experts,
                    "mode": args.uvm_mode,
                    "backing": args.uvm_backing,
                },
            },
        }
        start_total = _now()

        # =========================================================================
        # 4. 模拟生成专家访问计划（模拟 Top-K 路由机制）
        # =========================================================================
        # 冷专家访问计划：为每一层随机挑选 K 个冷专家（包含流式专家和 UVM 专家）作为激活对象
        cold_access_plan = {
            layer: sorted(
                random.sample(
                    range(stream_start_expert, args.num_experts),
                    k=min(args.topk, stream_experts + uvm_experts),
                )
            )
            for layer in layers
        }
        
        # 指针正确性校验计划：为了后面验证内存映射是否正确，在每层随机抽取特定数量的
        # 热专家、流式专家和 UVM 专家来进行前向计算比对
        pointer_plan: dict[int, list[int]] = {}
        for layer in layers:
            hot_k = min(args.pointer_checks_per_layer // 2, args.hot_experts)
            remaining_k = args.pointer_checks_per_layer - hot_k
            stream_k = min(remaining_k // 2, stream_experts)
            uvm_k = min(remaining_k - stream_k, uvm_experts)
            pointer_plan[layer] = sorted(
                random.sample(range(0, args.hot_experts), k=hot_k)
                + random.sample(range(stream_start_expert, uvm_start_expert), k=stream_k)
                + random.sample(range(uvm_start_expert, args.num_experts), k=uvm_k)
            )

        # =========================================================================
        # 5. 常驻“热专家”权重的初始化与加载 (常驻 GPU 显存)
        # =========================================================================
        hot_w13: dict[int, torch.Tensor] = {}
        hot_w2: dict[int, torch.Tensor] = {}
        effective_hot_backings: set[str] = set()
        hot_read_seconds = 0.0
        hot_copy_bytes = 0
        hot_start = _now()
        
        for layer in layers:
            # 5.1 分配热专家在 GPU 端的存储张量空间
            dst_w13, backing_w13 = _alloc_gpu_tensor(
                (args.hot_experts, w13_shape[0], w13_shape[1]),
                dtype,
                device,
                args.hot_backing,
            )
            dst_w2, backing_w2 = _alloc_gpu_tensor(
                (args.hot_experts, w2_shape[0], w2_shape[1]),
                dtype,
                device,
                args.hot_backing,
            )
            # 5.2 从检查点将一整批热专家权重加载至主机内存（src_w13, src_w2）
            src_w13, src_w2, read_seconds = _load_expert_batch(
                reader,
                layer=layer,
                expert_start=0,
                expert_count=args.hot_experts,
                w13_shape=w13_shape,
                w2_shape=w2_shape,
                dtype=dtype,
                pin_memory=args.pin_memory,
            )
            # 5.3 复制权重到 GPU。若开启锁页内存（pin_memory），则使用非阻塞（non_blocking）异步传输
            dst_w13.copy_(src_w13, non_blocking=args.pin_memory)
            dst_w2.copy_(src_w2, non_blocking=args.pin_memory)
            
            effective_hot_backings.update([backing_w13, backing_w2])
            hot_read_seconds += read_seconds
            hot_copy_bytes += src_w13.nbytes + src_w2.nbytes
            hot_w13[layer] = dst_w13
            hot_w2[layer] = dst_w2
            
        # 同步等待所有层热专家数据拷贝完毕
        torch.cuda.synchronize(device)
        hot_seconds = _now() - hot_start
        
        # 记录热专家部分的运行基准指标
        result["hot"] = {
            "effective_backings": sorted(effective_hot_backings),
            "read_seconds": hot_read_seconds,
            "copy_bytes": hot_copy_bytes,
            "wall_seconds": hot_seconds,
            "copy_wall_gbps": _gbps(hot_copy_bytes, hot_seconds),
        }

        # =========================================================================
        # 6. “UVM/UVA 专家”权重的初始化与加载 (通过零复制或页面调度间接访问)
        # =========================================================================
        uvm_w13: dict[int, torch.Tensor] = {}
        uvm_w2: dict[int, torch.Tensor] = {}
        uva_cpu_refs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        effective_uvm_backings: set[str] = set()
        uvm_read_seconds = 0.0
        uvm_copy_bytes = 0
        uvm_view_seconds = 0.0
        uvm_start = _now()
        
        if uvm_experts > 0:
            for layer in layers:
                # 6.1 从磁盘读取 UVM 专家批次
                src_w13, src_w2, read_seconds = _load_expert_batch(
                    reader,
                    layer=layer,
                    expert_start=uvm_start_expert,
                    expert_count=uvm_experts,
                    w13_shape=w13_shape,
                    w2_shape=w2_shape,
                    dtype=dtype,
                    # 如果使用 UVA（统一虚拟寻址），必须强制开启锁页内存
                    pin_memory=args.pin_memory or args.uvm_backing == "uva",
                )
                
                if args.uvm_backing == "uva":
                    # UVA 模式：通过 CUDA API 直接将锁页的主机内存地址映射为 GPU 可访问的虚拟地址
                    view_start = _now()
                    dst_w13 = _cuda_view_from_pinned_cpu(src_w13)
                    dst_w2 = _cuda_view_from_pinned_cpu(src_w2)
                    uvm_view_seconds += _now() - view_start
                    # 保持 CPU 张量引用防止被垃圾回收
                    uva_cpu_refs[layer] = (src_w13, src_w2)
                    backing_w13 = backing_w2 = "pinned_uva"
                else:
                    # 标准 UVM 模式：在系统分配 UVM 虚拟内存空间，并将数据拷贝进去
                    dst_w13, backing_w13 = _alloc_gpu_tensor(
                        (uvm_experts, w13_shape[0], w13_shape[1]),
                        dtype,
                        device,
                        args.uvm_backing,
                    )
                    dst_w2, backing_w2 = _alloc_gpu_tensor(
                        (uvm_experts, w2_shape[0], w2_shape[1]),
                        dtype,
                        device,
                        args.uvm_backing,
                    )
                    dst_w13.copy_(src_w13, non_blocking=args.pin_memory)
                    dst_w2.copy_(src_w2, non_blocking=args.pin_memory)
                    
                uvm_read_seconds += read_seconds
                uvm_copy_bytes += src_w13.nbytes + src_w2.nbytes
                effective_uvm_backings.update([backing_w13, backing_w2])
                uvm_w13[layer] = dst_w13
                uvm_w2[layer] = dst_w2
                
            torch.cuda.synchronize(device)
            
        uvm_seconds = _now() - uvm_start
        # 记录 UVM 部分的运行指标
        result["uvm"] = {
            "effective_backings": sorted(effective_uvm_backings),
            "mode": args.uvm_mode,
            "expert_start": uvm_start_expert,
            "expert_count": uvm_experts,
            "read_seconds": uvm_read_seconds,
            "copy_bytes": uvm_copy_bytes,
            "wall_seconds": uvm_seconds,
            "view_seconds": uvm_view_seconds,
            "copy_wall_gbps": _gbps(uvm_copy_bytes, uvm_seconds),
            "prefetch_success_count": 0,
            "prefetch_failure_count": 0,
        }

        # =========================================================================
        # 7. “流式专家插槽连接池(Streaming Slots)”显存分配
        # =========================================================================
        # 在 GPU 上只开辟具有 slot_layers 层空间大小的常驻显存循环池
        slot_w13, effective_slot_w13 = _alloc_gpu_tensor(
            (args.slot_layers, stream_experts, w13_shape[0], w13_shape[1]),
            dtype,
            device,
            args.slot_backing,
        )
        slot_w2, effective_slot_w2 = _alloc_gpu_tensor(
            (args.slot_layers, stream_experts, w2_shape[0], w2_shape[1]),
            dtype,
            device,
            args.slot_backing,
        )

        # =========================================================================
        # 8. 定义多 CUDA 流异步流水线函数 (Schedule vs Access)
        # =========================================================================
        # 创建独立的专用数据拷贝流（copy_stream），用于异步跨 PCIe 传输权重
        copy_stream = torch.cuda.Stream(device=device)
        # 获取当前默认的计算流（compute_stream），用于模拟矩阵乘法计算
        compute_stream = torch.cuda.current_stream(device=device)
        
        copy_events: dict[int, torch.cuda.Event] = {}
        staging_refs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        cold_read_seconds = 0.0
        cold_copy_bytes = 0
        cold_copy_submit_seconds = 0.0
        cold_access_seconds = 0.0
        checksum = torch.zeros((), dtype=torch.float32, device=device)
        latest_layer_by_bank: dict[int, int] = {}

        # 内部函数：安排调度某一层的冷专家（负责加载数据并提交异步拷贝）
        def schedule_cold_layer(layer: int) -> None:
            nonlocal cold_read_seconds, cold_copy_bytes, cold_copy_submit_seconds
            # 核心：通过取模运算（% slot_layers）决定当前层复用显存池中的哪一个 Bank 槽
            bank = layer % args.slot_layers
            
            # 异步读取：首先将当前层的流式专家数据从磁盘拉到主机内存
            src_w13, src_w2, read_seconds = _load_expert_batch(
                reader,
                layer=layer,
                expert_start=stream_start_expert,
                expert_count=stream_experts,
                w13_shape=w13_shape,
                w2_shape=w2_shape,
                dtype=dtype,
                pin_memory=args.pin_memory,
            )
            t0 = _now()
            # 切换到非阻塞的拷贝专属 CUDA 流上执行内存搬运
            with torch.cuda.stream(copy_stream):
                slot_w13[bank].copy_(src_w13, non_blocking=args.pin_memory)
                slot_w2[bank].copy_(src_w2, non_blocking=args.pin_memory)
                # 并在拷贝流中放置一个 CUDA 事件（Event），标志着该层的数据已经成功送出
                event = torch.cuda.Event(enable_timing=False)
                event.record(copy_stream)
                
            copy_events[layer] = event
            staging_refs[layer] = (src_w13, src_w2) # 临时封存防释放
            latest_layer_by_bank[bank] = layer
            cold_read_seconds += read_seconds
            cold_copy_submit_seconds += _now() - t0
            cold_copy_bytes += src_w13.nbytes + src_w2.nbytes

        # 内部函数：访问并模拟计算某一层的冷专家
        def access_cold_layer(layer: int) -> dict[str, Any]:
            nonlocal checksum, cold_access_seconds
            bank = layer % args.slot_layers
            expert_ids = cold_access_plan[layer]
            
            # 将该层规划访问的冷专家划分出哪些属于流式专家，哪些属于 UVM 专家
            stream_ids = [
                expert_id for expert_id in expert_ids
                if stream_start_expert <= expert_id < uvm_start_expert
            ]
            uvm_ids = [
                expert_id for expert_id in expert_ids
                if uvm_start_expert <= expert_id < args.num_experts
            ]
            touch = args.touch_elements
            t0 = _now()
            
            # 切换至当前的计算流中执行模拟计算操作
            with torch.cuda.stream(compute_stream):
                # 8.1 提取并触及流式槽中的专家数据
                if stream_ids:
                    # 换算为槽内的相对索引
                    local_ids = [expert_id - stream_start_expert for expert_id in stream_ids]
                    ids_gpu = torch.tensor(local_ids, dtype=torch.long, device=device)
                    # 将高维张量展平后利用 index_select 模拟动态路由的索引选择
                    flat_w13 = slot_w13[bank].reshape(stream_experts, -1)
                    flat_w2 = slot_w2[bank].reshape(stream_experts, -1)
                    selected_w13 = flat_w13.index_select(0, ids_gpu)
                    selected_w2 = flat_w2.index_select(0, ids_gpu)
                    if touch > 0:
                        selected_w13 = selected_w13[:, :touch]
                        selected_w2 = selected_w2[:, :touch]
                    # 累加求和校验码，强制促使 GPU 触及并拉取这些被索引的显存数据
                    checksum = (
                        checksum
                        + selected_w13.float().sum()
                        + selected_w2.float().sum()
                    )
                    
                # 8.2 提取并触及系统内存中的 UVM 专家数据
                for expert_id in uvm_ids:
                    local = expert_id - uvm_start_expert
                    w13_view = uvm_w13[layer][local]
                    w2_view = uvm_w2[layer][local]
                    # 如果配置为主动预引模式，则在计算前向底层硬件驱动下达异步预引到 GPU 的指令
                    if args.uvm_mode == "prefetch":
                        ok_w13 = _prefetch_tensor_to_device(w13_view, args.device)
                        ok_w2 = _prefetch_tensor_to_device(w2_view, args.device)
                        result["uvm"]["prefetch_success_count"] += int(ok_w13)
                        result["uvm"]["prefetch_success_count"] += int(ok_w2)
                        result["uvm"]["prefetch_failure_count"] += int(not ok_w13)
                        result["uvm"]["prefetch_failure_count"] += int(not ok_w2)
                    flat_w13 = w13_view.reshape(-1)
                    flat_w2 = w2_view.reshape(-1)
                    if touch > 0:
                        flat_w13 = flat_w13[:touch]
                        flat_w2 = flat_w2[:touch]
                    # 累加求和。在 fault 模式下，此处将当场触发操作系统的缺页中断，动态按页加载权重
                    checksum = checksum + flat_w13.float().sum() + flat_w2.float().sum()
                    
            # 强行同步等待当前层模拟计算完毕，计算真实的耗时
            torch.cuda.synchronize(device)
            elapsed = _now() - t0
            cold_access_seconds += elapsed
            return {
                "layer": layer,
                "bank": bank,
                "expert_ids": expert_ids,
                "stream_expert_ids": stream_ids,
                "uvm_expert_ids": uvm_ids,
                "access_seconds": elapsed,
            }

        # =========================================================================
        # 9. 启动流水线窗口的预热加载（Warm-up / Prefill pipeline）
        # =========================================================================
        layer_queue = list(layers)
        next_to_schedule = 0
        # 预先铺满插槽池（例如：如果池容量为 2 层，则先提前开始异步搬运前 2 层的数据）
        for _ in range(min(args.slot_layers, len(layer_queue))):
            schedule_cold_layer(layer_queue[next_to_schedule])
            next_to_schedule += 1

        # =========================================================================
        # 10. 顺序消费层队列，执行主循环 (Compute-Copy Overlap Main Loop)
        # =========================================================================
        cold_records: list[dict[str, Any]] = []
        pointer_records: list[dict[str, Any]] = []
        max_pointer_diff = 0.0
        # 为接下来的高精度数学计算校验初始化一组线性的输入张量
        x_cpu = torch.linspace(
            -0.25, 0.25, steps=args.pointer_hidden, dtype=torch.float32
        )
        cold_start = _now()
        
        for layer in layer_queue:
            # 【核心同步点】：让默认计算流等待该层拷贝流的事件通知。
            # 这确保了在进入 access_cold_layer 之前，该层所需的数据拷贝已经通过 PCIe 搬运就绪
            compute_stream.wait_event(copy_events[layer])
            # 执行计算，完成后代表该层已经完美消费
            cold_records.append(access_cold_layer(layer))
            # 释放主机侧的零时锁页缓冲块引用，保持内存整洁
            staging_refs.pop(layer, None)

            # 10.1 进行前向前向计算指针的高精度严苛校验
            if args.pointer_checks_per_layer > 0:
                bank = layer % args.slot_layers
                for expert in pointer_plan[layer]:
                    # 定位被检测专家的物理内存池切片
                    if expert < args.hot_experts:
                        split_w13 = hot_w13[layer]
                        split_w2 = hot_w2[layer]
                        local = expert
                    elif expert < uvm_start_expert:
                        split_w13 = slot_w13[bank]
                        split_w2 = slot_w2[bank]
                        local = expert - stream_start_expert
                    else:
                        split_w13 = uvm_w13[layer]
                        split_w2 = uvm_w2[layer]
                        local = expert - uvm_start_expert
                        if args.uvm_mode == "prefetch":
                            _prefetch_tensor_to_device(split_w13[local], args.device)
                            _prefetch_tensor_to_device(split_w2[local], args.device)
                            
                    # 调用更底层的数学函数对输入 `x_cpu` 跑一次该专家的 MLP 映射，比对磁盘原始数据
                    record = _pointer_check_one(
                        reader,
                        layer=layer,
                        expert=expert,
                        local=local,
                        split_w13=split_w13,
                        split_w2=split_w2,
                        w13_shape=w13_shape,
                        w2_shape=w2_shape,
                        dtype=dtype,
                        pin_memory=args.pin_memory,
                        hidden=min(args.pointer_hidden, w13_shape[1], w2_shape[0]),
                        intermediate=min(
                            args.pointer_intermediate,
                            w13_shape[0] // 2,
                            w2_shape[1],
                        ),
                        x_cpu=x_cpu,
                    )
                    max_pointer_diff = max(max_pointer_diff, record["max_abs_diff"])
                    pointer_records.append(record)

            # 【滑窗推进】：当当前层开始计算时，空出来的显存 Bank 立即开始拉取后续未来的层数据
            if next_to_schedule < len(layer_queue):
                schedule_cold_layer(layer_queue[next_to_schedule])
                next_to_schedule += 1

        # =========================================================================
        # 11. 收尾工作、最终指标聚合与资源清理
        # =========================================================================
        # 强制整卡同步，结束耗时统计
        torch.cuda.synchronize(device)
        cold_seconds = _now() - cold_start
        
        # 填充完整的冷专家及流水线统计报告
        result["cold"] = {
            "slot_backings": sorted({effective_slot_w13, effective_slot_w2}),
            "slot_layers": args.slot_layers,
            "slot_experts_per_layer": stream_experts,
            "expert_id_to_slot": "slot = expert_id - stream_start_expert",
            "bank_mapping": "bank = layer_id % slot_layers",
            "expert_start": stream_start_expert,
            "read_seconds": cold_read_seconds,
            "copy_bytes": cold_copy_bytes,
            "wall_seconds": cold_seconds,
            "copy_submit_seconds": cold_copy_submit_seconds,
            "copy_wall_gbps": _gbps(cold_copy_bytes, cold_seconds),
            "access_seconds": cold_access_seconds,
            "latest_layer_by_bank": latest_layer_by_bank,
            "random_access_records_sample": cold_records[: args.trace_layers],
            "checksum": float(checksum.detach().cpu().item()),
        }
        
        # 填充高精度数学校验报告，判定本次基准测试是否通过
        result["pointer_check"] = {
            "records": len(pointer_records),
            "max_abs_diff": max_pointer_diff,
            "tolerance": args.pointer_tolerance,
            "passed": max_pointer_diff <= args.pointer_tolerance,
            "records_sample": pointer_records[: args.trace_layers],
        }

        # 填入宏观层面的系统信息和显存峰值开销
        result["total_seconds"] = _now() - start_total
        result["torch_cuda_memory_allocated_bytes"] = int(
            torch.cuda.memory_allocated(device)
        )
        result["torch_cuda_max_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        # 测试全面通过的标志：取决于高精度绝对误差是否小于设定的容忍门限（pointer_tolerance）
        result["passed"] = bool(result["pointer_check"]["passed"])
        return result
        
    finally:
        # 在最外层 `finally` 块中关闭检查点文件句柄，防止底层文件句柄泄露
        reader.close()

def main() -> int:
    # 初始化命令行参数解析器，用于配置 MoE (Mixture of Experts) 专家的显存与流式传输测试
    parser = argparse.ArgumentParser()
    
    # =========================================================================
    # 1. 基础配置与模型结构参数
    # =========================================================================
    # 模型权重或检查点的存储路径，默认为预设的默认路径
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    
    # 需要测试或处理的具体层编号列表（字符串格式，如 "1,2,3"），留空则默认按范围处理
    parser.add_argument("--layers", default="")
    
    # 专家层处理的起始索引（从第几层开始测试）
    parser.add_argument("--layer-start", type=int, default=0)
    
    # 本次测试需要覆盖的连续专家层数量
    parser.add_argument("--layer-count", type=int, default=2)
    
    # 该 MoE 层中的专家总数（例如 128 个专家）
    parser.add_argument("--num-experts", type=int, default=128)
    
    # =========================================================================
    # 2. 专家冷热分级与流式管理配置
    # =========================================================================
    # 常驻显存（GPU High-bandwidth Memory）的高频激活“热专家”数量
    parser.add_argument("--hot-experts", type=int, default=64)
    
    # 留驻在内存或低速存储中的低频激活“冷专家”数量
    parser.add_argument("--cold-experts", type=int, default=64)
    
    # 显式使用“流式插槽连接池(Streaming Slot Pool)”进行动态加载的冷专家数量
    # 开启后，冷专家会在推理时通过高速通道（如 PCIe）流式置换到 GPU 预留插槽中
    parser.add_argument(
        "--stream-experts",
        type=int,
        default=None,
        help="cold experts backed by the explicit streaming slot pool",
    )
    
    # 使用 NVIDIA 统一内存（UVM - Unified Virtual Memory）进行管理的冷专家数量
    # 允许 GPU 直接通过虚拟内存地址访问位于系统内存（CPU RAM）中的专家权重
    parser.add_argument(
        "--uvm-experts",
        type=int,
        default=None,
        help="cold experts backed by managed UVM tensors",
    )
    
    # 为动态流式置换专家预留的插槽所覆盖的层数上限
    parser.add_argument("--slot-layers", type=int, default=2)
    
    # =========================================================================
    # 3. 硬件存储介质与驱动模式配置 (Backing & Device)
    # =========================================================================
    # 指定运行该测试的 GPU 设备 ID（如 GPU 0）
    parser.add_argument("--device", type=int, default=0)
    
    # 热专家权重的底层存储后端：
    # 'cuda': 强制驻留常规显存；'uvm': 使用统一内存；'auto': 自动选择
    parser.add_argument(
        "--hot-backing",
        choices=("auto", "uvm", "cuda"),
        default="cuda",
    )
    
    # 流式插槽预留空间的底层存储后端（通常为 'cuda' 强行开辟显存空间以保证速度）
    parser.add_argument(
        "--slot-backing",
        choices=("uvm", "cuda"),
        default="cuda",
    )
    
    # UVM 专家的底层后端拓扑结构：
    # 'uvm': 标准统一内存；'uva': 统一虚拟寻址（多卡独热延伸）；'cuda': 强制转显存
    parser.add_argument(
        "--uvm-backing",
        choices=("auto", "uvm", "cuda", "uva"),
        default="uvm",
    )
    
    # UVM 统一内存的页面调度模式：
    # 'fault': 缺页中断加载（按需按页延迟加载）；'prefetch': 异步预引（提前推送到 GPU）
    parser.add_argument(
        "--uvm-mode",
        choices=("fault", "prefetch"),
        default="fault",
    )
    
    # =========================================================================
    # 4. 路由算法、微基准测试与容错参数
    # =========================================================================
    # Top-K 专家路由选择：每个 Token 会被同时分发给多少个专家（常见如 top-2 或 top-8）
    parser.add_argument("--topk", type=int, default=8)
    
    # 微基准测试中，每次测试试图触及/热身的张量元素数量（评估内存吞吐吞吐量）
    parser.add_argument("--touch-elements", type=int, default=4096)
    
    # 每层进行指针有效性或内存越界检查的次数
    parser.add_argument("--pointer-checks-per-layer", type=int, default=4)
    
    # 指针测试中模拟的隐藏层维度（Hidden Dimension Size）
    parser.add_argument("--pointer-hidden", type=int, default=32)
    
    # 指针测试中模拟的 MoE 中间全连接层维度（Intermediate Dimension Size）
    parser.add_argument("--pointer-intermediate", type=int, default=32)
    
    # 指针校验或数值精度比对时的容忍度误差门限（Tolerance）
    parser.add_argument("--pointer-tolerance", type=float, default=1e-2)
    
    # 是否开启锁页内存（Pin Memory）。开启后能显著加快 CPU 到 GPU 之间的 PCIe 传输速率
    parser.add_argument("--pin-memory", action="store_true")
    
    # 是否开启空运行模拟模式（不真正分配大显存或加载权重，仅进行逻辑跑通）
    parser.add_argument("--dry-run", action="store_true")
    
    # 随机数种子，用于固定专家路由模拟或随机张量初始化，确保结果可复现
    parser.add_argument("--seed", type=int, default=1234)
    
    # 性能追踪（Trace）分析所捕获的层数深度
    parser.add_argument("--trace-layers", type=int, default=8)
    
    # 测试结果输出的 JSON 文件路径，若指定则自动创建目录并写入
    parser.add_argument("--out-json", type=Path, default=None)
    
    # 解析命令行输入的参数
    args = parser.parse_args()

    try:
        # 执行核心测试逻辑，返回包含测试指标和是否通过的字典
        result = run(args)
        
        # 将结果美化转换为 JSON 字符串（缩进 2 空格，键名排序）
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        
        # 如果指定了输出路径，则确保父目录存在，并以 UTF-8 编码安全写入结果
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(text + "\n", encoding="utf-8")
        
        # 根据结果中的 "passed" 字段决定退出码：True 返回 0 (成功), False 返回 1 (失败)
        return 0 if result.get("passed", True) else 1
        
    finally:
        # 无论程序正常结束还是遭遇异常崩塌，都在 `finally` 块中执行底层资源回收
        try:
            # 延时导入 vLLM 的设备分配器组件，清理前面注册的 UVM 本地内存池
            from vllm.device_allocator.uvm import cleanup_uvm_local_memory_pool

            cleanup_uvm_local_memory_pool()
        except Exception:
            # 即使清理过程报错（例如未成功初始化 UVM）也静默跳过，确保程序安全退出
            pass


if __name__ == "__main__":
    # 使用系统的标准退出方法返回 main 函数的执行状态码
    raise SystemExit(main())
