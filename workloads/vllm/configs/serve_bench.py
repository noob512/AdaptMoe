#!/usr/bin/env python3
"""vLLM serve + benchmark atomic runner: single mode, single run, JSON output.

用法示例:
    uv run python configs/serve_bench.py --mode cpu_offload --output results/cpu_offload.json
    uv run python configs/serve_bench.py --mode uvm --output results/uvm_baseline.json

此脚本支持 vLLM 原生 CPU offload 与项目的 UVM 模式。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ==========================================
# 1. 路径与依赖解析设置
# ==========================================
# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = Path(__file__).resolve().parent
# 获取工作负载的根目录（通常是脚本所在目录的上一级）
WORKLOAD_DIR = SCRIPT_DIR.parent
# 获取所有工作负载的顶层目录
WORKLOADS_DIR = WORKLOAD_DIR.parent

# 将公共脚本目录插入到 sys.path 的最前面，以便能够导入项目自定义的公共模块 (common.py)
sys.path.insert(0, str(WORKLOADS_DIR / "scripts"))
# 从外部导入自定义的辅助函数：清理GPU、等待服务器启动、停止服务器、解析vLLM跑分结果
from common import cleanup_gpu, wait_for_server, stop_server, parse_vllm_bench_output

# ==========================================
# 2. 全局环境变量与常量配置
# ==========================================
# 获取 vLLM 源码/可执行文件 目录，如果未设置则使用默认路径
VLLM_SERVER_DIR = os.environ.get("VLLM_SERVER_DIR", str(WORKLOAD_DIR / "vllm"))
# 获取数据集路径，默认使用 ShareGPT 的清洗版本（常用于大模型基准测试）
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    str(WORKLOAD_DIR / "datasets" / "ShareGPT_V3_unfiltered_cleaned_split.json"),
)
# 指定要进行压测的模型，这里使用的是 Qwen 30B FP8 量化版本
MODEL = "Qwen/Qwen3-30B-A3B-FP8"

# 定义不同模式下的服务器启动配置
MODE_CONFIGS = {
    # 模式1: CPU Offload (vLLM 原生的机制，将部分权重或 KV cache 卸载到 CPU 内存)
    "cpu_offload": {
        "server_cmd": f"uv run vllm serve {MODEL} --enforce-eager --cpu-offload-gb 8",
        "env": {}, # 无特殊环境变量
    },
    # 模式2: UVM (Unified Virtual Memory，使用项目内的 allocator 和驱动机制)
    "uvm": {
        "server_cmd": f"uv run vllm serve {MODEL} --enforce-eager --max-num-seqs 16",
        "env": {"VLLM_USE_UVM": "1"}, # 注入特定的环境变量以开启 UVM 特性
    },
}
#,"VLLM_UVM_KV_CACHE_SIZE_GB": "5"  # (被注释掉的历史配置项)

# 服务器启动的超时时间（秒），大模型加载到 GPU 需要较长时间，这里设为 10 分钟
SERVER_STARTUP_TIMEOUT = 600
# 轮询检查服务器是否启动成功的时间间隔（秒）
SERVER_CHECK_INTERVAL = 5


# ==========================================
# 3. 核心压测执行逻辑
# ==========================================
def run_serve_bench(mode: str, prompts: int, port: int = 8000) -> dict:
    """启动 vLLM 服务器，运行压测工具，停止服务器，最后返回格式化的结果字典。"""
    
    # 获取对应模式的启动命令和环境变量
    config = MODE_CONFIGS[mode]

    # --- 阶段 A：启动 vLLM Server 进程 ---
    
    # 复制当前系统环境变量，并更新特定模式需要的环境变量，避免污染全局环境
    env = os.environ.copy()
    env.update(config["env"])

    # 拼接完整的服务器启动命令，指定监听端口
    server_cmd = config["server_cmd"] + f" --port {port}"
    print(f"Starting vLLM server (mode={mode}, port={port})...", file=sys.stderr)
    
    # 创建专门的日志文件来记录当前模式下 Server 端的输出
    server_log_file = open(os.path.join(WORKLOAD_DIR, f"vllm_server_{mode}.log"), "w")
    
    # 使用 subprocess 以后台进程的方式启动 vLLM 服务器
    server_proc = subprocess.Popen(
        server_cmd,
        shell=True,
        cwd=str(WORKLOAD_DIR),
        env=env,
        stdout=server_log_file,            # 将标准输出重定向到日志文件
        stderr=subprocess.STDOUT,          # 将标准错误合并到标准输出，同样写入日志文件
        # preexec_fn=os.setsid 非常关键：它会让子进程在一个新的进程组中运行。
        # 这样在后续杀死服务器时，可以发送信号给整个进程组，确保彻底杀掉其派生的所有子进程，防止显存泄漏。
        preexec_fn=os.setsid, 
    )

    try:
        # 阻塞等待服务器启动完毕（通常是不断探测指定端口是否已开放）
        if not wait_for_server(port=port, timeout=SERVER_STARTUP_TIMEOUT,
                              check_interval=SERVER_CHECK_INTERVAL, process=server_proc):
            print("ERROR: Server failed to start", file=sys.stderr)
            stop_server(server_proc) # 如果启动失败，清理僵尸进程
            sys.exit(1)

        print(f"Server ready. Running benchmark ({prompts} prompts)...", file=sys.stderr)

        # --- 阶段 B：运行 vLLM Benchmark 客户端 ---
        
        # 组装压测客户端的命令
        # --sharegpt-output-len 512: 强制生成的最大长度为 512
        # --request-rate 5: 每秒发送 5 个请求 (QPS=5)
        bench_cmd = (
            f"uv run vllm bench serve "
            f"--model {MODEL} "
            f"--dataset-name sharegpt "
            f"--dataset-path {DATASET_PATH} "
            f"--num-prompts {prompts} "
            f"--sharegpt-output-len 512 --seed 42 --request-rate 5 "
            f"--port {port}"
        )

        start = time.time()
        # 同步执行压测命令（会阻塞直到压测跑完）
        bench_result = subprocess.run(
            bench_cmd, shell=True, cwd=str(WORKLOAD_DIR),
            capture_output=True, text=True, # 捕获标准输出和错误到变量中，并解码为字符串
        )
        elapsed = time.time() - start

        # 检查压测是否执行成功
        if bench_result.returncode != 0:
            print(f"Benchmark failed (exit {bench_result.returncode}):", file=sys.stderr)
            # 如果失败，打印最后 2000 个字符的错误日志帮助排查
            print(bench_result.stderr[-2000:], file=sys.stderr)
            stop_server(server_proc)
            sys.exit(1)

        # 合并客户端的输出和错误，并调用外部工具将其解析为结构化的指标数据 (metrics)
        bench_output = bench_result.stdout + bench_result.stderr
        metrics = parse_vllm_bench_output(bench_output)

    finally:
        # --- 阶段 C：资源清理 ---
        # 无论压测成功与否，或者代码抛出什么异常，finally 块确保 vLLM 服务器一定会被关闭释放 VRAM
        print("Stopping server...", file=sys.stderr)
        stop_server(server_proc)
        time.sleep(3) # 留出缓冲时间让操作系统回收 GPU 显存和端口

    # 整合所有的测试参数和结果，构建 JSON 结构
    return {
        "workload": "vllm",
        "config": f"serve_{mode}",
        "params": {
            "mode": mode,
            "model": MODEL,
            "prompts": prompts,
        },
        "metrics": metrics,                            # 解析出的关键吞吐、延迟等指标
        "timestamp": datetime.now().isoformat(),       # 记录测试发生的具体时间
        "duration_s": round(elapsed, 2),               # 压测客户端实际运行花费的总秒数
        "raw": {"bench_output": bench_output},         # 保留原始终端输出，用于未来可能的 debug
    }


# ==========================================
# 4. 命令行入口与参数解析
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="vLLM serve + benchmark single run")
    # choices 限定了只能输入预设的两种模式
    parser.add_argument("--mode", default="cpu_offload", choices=list(MODE_CONFIGS.keys()),
                        help="Server mode")
    parser.add_argument("--prompts", type=int, default=50, help="Number of prompts")
    # 如果不指定 output，结果会直接打印在终端上
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    # 允许用户通过 flag 跳过 GPU 清理步骤（例如多卡复用环境中）
    parser.add_argument("--no-cleanup", action="store_true", help="Skip GPU cleanup")
    args = parser.parse_args()

    # 目录合法性校验
    if not Path(VLLM_SERVER_DIR).exists():
        print(f"ERROR: vLLM not found at {VLLM_SERVER_DIR}", file=sys.stderr)
        sys.exit(1)

    # 默认情况下，在启动 vLLM 之前执行 kill 操作或清理缓存，确保环境干净
    if not args.no_cleanup:
        cleanup_gpu()

    # 核心调用
    result = run_serve_bench(args.mode, args.prompts, port=args.port)

    # 将 Python 字典序列化为漂亮的 JSON 字符串
    output_json = json.dumps(result, indent=2)
    
    # 结果输出处理
    if args.output:
        # 如果提供了输出路径，确保其父级目录存在 (mkdir -p)，然后写入文件
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        # 否则直接打印到控制台 (stdout)
        print(output_json)


# 标准的 Python 脚本入口判断
if __name__ == "__main__":
    main()
