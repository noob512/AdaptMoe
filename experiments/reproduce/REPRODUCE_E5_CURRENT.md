# 复现 E5 当前模块stable/non-stable VA配对

这是2026-08-31当前模块可直接复现的E5部分。两个arm使用相同Qwen3模型、layers 0--3、
8个Parameter、2,415,919,104 B逻辑alias和128 MiB物理KV，各运行10个独立冷进程。

```bash
cd /home/ubuntu/nvidia-uvm-gpu

STABLE_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e5-destroy-rebuild-current-module-NEW-attempt-01-formal
NONSTABLE_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e5-nonstable-va-current-module-NEW-attempt-01-formal

RUN_ROOT="$STABLE_ROOT" REPEATS=10 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e5_destroy_rebuild_formal.sh

RUN_ROOT="$NONSTABLE_ROOT" REPEATS=10 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e5_nonstable_va_recapture_formal.sh

workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/compare_e5_stable_vs_nonstable_va.py \
  "$STABLE_ROOT" "$NONSTABLE_ROOT" \
  --output /tmp/e5-stable-vs-nonstable-new.json
```

两个verifier必须为`status=ok`；stable臂8个Parameter VA保持不变，non-stable臂每代
8/8 VA改变并重捕获Graph；B1/B4输出hash必须跨臂一致。

Graph/eager、PARK/RESUME和sequential/grouped PTE的完整消融仍来自较早模块，尚未在
当前模块整组重跑。旧协议命令见
`../eurosys_moe_offload_evaluation/REPORT_E2_E12_REPRO_20260827.md`第7节，不能与当前
两臂直接合并成同轮消融。
