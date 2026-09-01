# E8 MoE-Infinity cache/prefetch复现状态

当前只有provenance preflight可用：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/broad_offload_compare/run_moe_infinity_qualification.sh preflight
```

截至2026-08-31，该命令因缺少与活动MoE-Infinity commit、扩展ELF和约30 GiB store一致的
fresh build receipt而返回1。不得绕过commit、patch、ELF或receipt hash门运行性能实验。

因此E8目前没有成功性能实验可复现。本文件记录的是合法阻塞入口，不代表E8算法性能失败。
