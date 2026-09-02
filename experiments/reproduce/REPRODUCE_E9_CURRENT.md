# 复现 E9 KTransformers placement/CPU消融qualification

最后核验：2026-09-02。没有新的通过总体验收的E9 artifact。活动KTransformers HEAD为
`5a9c7d4cfaaed0136930e4b629ce5d55cec71a54`，父仓库gitlink仍指向
`8d9325af331d916721dfe7e2104f93f9b268678b`；更新gitlink前，以下流程只代表当前工作区，
不能保证从main fresh clone得到相同代码。

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/\
e9-formal-NEW-attempt-01-qualification" \
ALLOW_GPU_RUN=1 MODE=qualification PROCESS_BLOCKS=1 \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e9.sh
```

该流程运行baseline、threads7、threads28、eager、uniform、no-overlap、no-mixed共7个case，
每臂C1+C8。当前参考运行7/7执行完成，但顶层correctness verifier因uniform和no-mixed的
greedy token漂移不通过，因此只可复现诊断结果，不能启动或声称5-block formal性能结论。

冻结selection reference为
`experiments/eurosys_moe_offload_evaluation/inputs/e9_sharegpt_selection_reference_20260828.json`。
