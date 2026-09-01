# 复现 E12 Graph replay与双页表PARK/RESUME qualification

## 1,000次Graph replay smoke

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
env CUDA_VISIBLE_DEVICES=0 taskset -c 0-13 .venv/bin/python \
  ../../../experiments/vllm_p3_static_alias_graph_integration/smoke_static_alias_graph.py \
  --output ../../../experiments/eurosys_moe_offload_evaluation/artifacts/\
e12-graph-soak-1000-NEW-attempt-01.json \
  --manifest ../../../experiments/eurosys_moe_offload_evaluation/artifacts/\
e12-graph-soak-1000-NEW-attempt-01.manifest.jsonl \
  --layer-mib 576 --warmup-replays 3 --measured-replays 1000
```

## 1,000次完整双页表往返

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e12-dual-view-soak-NEW-attempt-01-qualification \
CYCLES=1000 VERIFY_EVERY=1 CHECKPOINT_EVERY=100 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e12_dual_view_soak_qualification.sh
```

验收要求1000/1000次PARK和RESUME逐轮数据检查通过、双页表父块覆盖100%、无range-map
回退、Xid、PDE/PTE fault或资源泄漏，退出后恢复stock UVM和GPU 0 MiB。

该循环只满足机制次数门，不是6/24小时整模型服务soak，也不包含KV峰谷和barrier故障注入。
