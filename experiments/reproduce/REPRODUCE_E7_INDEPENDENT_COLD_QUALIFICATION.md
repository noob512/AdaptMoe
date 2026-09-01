# 复现 E7 六臂独立冷进程workload-shift qualification

本流程对应2026-08-29最新E7结果。六个arm分别使用独立模型/CUDA冷进程，每个arm运行
一次32-request shift trace。冻结target来自已验证的早期calibration结果，不能现场重新
选择或使用其他future route。

## 前置检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/eurosys_moe_offload_evaluation/\
run_e7_independent_cold_qualification.sh preflight

nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
```

GPU运行前必须为0 MiB、无compute/UVM client，并加载stock UVM。source result SHA256必须
为`3190d36d37a0ca9eaccb5bbeb35d4dcfc3b9f553815e0ea7240df1ade917187c`。

## 单轮运行与复核

```bash
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e7-independent-cold-NEW-attempt-01-qualification \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e7_independent_cold_qualification.sh qualify

bash experiments/eurosys_moe_offload_evaluation/\
run_e7_independent_cold_qualification.sh verify "$RUN_ROOT"
```

必须满足：六臂各32请求、输出hash和route-count hash一致、Graph capture `2 -> 2`、
adaptive至少一次真实swap、oracle动态远端assignment最少、hot-adversarial最多、cleanup
成功且最终恢复stock UVM和GPU 0 MiB。

这是每arm一次的qualification，不提供方差或置信区间。同步maintenance必须计入cycle
throughput；不能用排除维护后的请求延迟声称端到端无回退。
