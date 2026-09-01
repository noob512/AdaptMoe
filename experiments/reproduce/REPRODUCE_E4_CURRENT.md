# 复现 E4 并发与PCIe饱和qualification

当前协议使用固定always-remote exact-coordinate计划，5个独立进程按
`asc/desc/asc/desc/asc`扫描C1、C4、C8、C16、C24、C32，每档输出128 tokens。

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e4-repro-NEW-attempt-01" \
PROCESS_BLOCKS=5 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e4.sh

workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/verify_formal_e4.py "$RUN_ROOT"
```

输出目录必须不存在。开始前要求GPU 0 MiB、无compute/UVM client；失败attempt必须保留。
当前参考artifact为
`../eurosys_moe_offload_evaluation/artifacts/e4-repro-20260827-attempt-01/`。
CPU DRAM带宽在当前KVM环境为N/A，不能用PCIe RX替代。
