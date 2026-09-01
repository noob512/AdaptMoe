# E7 独立冷进程单轮 qualification 结果（2026-08-29）

## 1. 实验状态

本轮 `status=ok`。六个 arm 分别在独立的新模型/CUDA 进程中运行，每个 arm
只执行一次 32-request shift trace，没有统计重复，也没有为本轮重新执行 calibration
或 Oracle 计划生成请求。

- 实测请求数：`6 × 32 = 192`；本轮 calibration 请求数：`0`。
- 六个进程 PID：`12386, 12929, 13426, 14066, 14600, 15111`。
- 每个 arm 启动前和退出后均为 GPU `0 MiB`、无 compute/UVM client。
- 六臂输出 hash 和路由计数 hash 完全一致。
- 每个进程 CUDA Graph 计数均为 `2 -> 2`，没有重捕获。
- 六臂 cleanup 均成功；未发现 Xid、`FAULT_PTE`、`FAULT_PDE` 或 illegal-memory 记录。
- 实验结束后已恢复 stock UVM；GPU `0 MiB`，随后恢复了监控服务。

## 2. 冻结协议

六臂 target 在 GPU 运行前从此前唯一成功的 E7 qualification 结果离线冻结：

- source result SHA256：
  `3190d36d37a0ca9eaccb5bbeb35d4dcfc3b9f553815e0ea7240df1ade917187c`
- frozen targets SHA256：
  `1c80ef62ebf8fc926f50b67a05d4b3e9d5d3a18643de192a63d1b33ec7db6e5f`
- Oracle 使用历史完整 shift route 的未来信息，只作为固定放置上界。

冻结参数为 Qwen3-30B-A3B-FP8、B1、128-token prompt、32-token output、
32 个 shift 请求、4 requests/window、EWMA alpha `0.25`、每层每窗口最多
交换 1 对专家、最小 count delta `1`、每层 4 个动态远端槽、256 MiB KV、
CUDA Graph `FULL_SEPARATE [1, 128]`、seed `20260829`。

模型加载、四轮动态槽初始化和起始 target 切换不计入请求延迟。表中的
`request tok/s` 只包括请求执行；`cycle tok/s` 包括每四个请求后的同步 heat/swap
维护，但仍不包括模型加载和 arm setup。

## 3. 单轮结果

| Arm | 请求中位延迟 (ms) | 相对 frequency-cold | request tok/s | cycle tok/s | 动态远端命中率 | 总远端命中率 | Oracle hit（终态） | 同步维护 (ms) | 在线 swaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frequency-cold | 1797.7 | 0.00% | 17.44 | 17.42 | 1.427% | 14.723% | 36.98% | 77.4 | 0 |
| random | 1602.6 | -10.85% | 20.04 | 20.02 | 3.186% | 16.481% | 4.17% | 57.6 | 0 |
| frozen static | 1741.0 | -3.15% | 17.54 | 17.52 | 1.688% | 14.983% | 18.23% | 69.2 | 0 |
| GPU heat + CPU EWMA | 1744.4 | -2.96% | 17.64 | 14.23 | 0.712% | 14.007% | 88.54% | 13885.6 | 221 |
| oracle | 1743.8 | -3.00% | 17.70 | 17.68 | 0.178% | 13.474% | 100.00% | 67.9 | 0 |
| hot-adversarial | 1935.9 | +7.69% | 16.50 | 16.48 | 12.484% | 25.780% | 0.00% | 71.1 | 0 |

所有 arm 的 expert assignment 总数相同。动态远端 assignments 分别为
`27889, 62239, 32974, 13910, 3485, 243916`。

## 4. 结论

1. **自适应放置有效。** Adaptive 的 Oracle-slot hit 从 `36.98%` 提升到
   `88.54%`，动态远端 assignments 相比 frequency-cold 减少 `50.12%`。
   32 个请求内仍未达到 95% 收敛门。
2. **本轮没有观察到请求路径性能下降。** 排除同步维护后，adaptive 请求中位延迟
   比 frequency-cold 低 `2.96%`，与 oracle 只差约 `0.03%`。
3. **仍不能证明端到端性能稳定。** Adaptive 同步维护累计 `13.886 s`，包含维护的
   cycle throughput 为 `14.23 tok/s`，相比 frequency-cold 的 `17.42 tok/s`
   下降 `18.28%`。因此当前实现未通过“维护成本计入后的性能无回退”目标。
4. **Oracle 最少远端命中，但不保证本次延迟排名第一。** Oracle 确实取得固定策略中
   最低的动态远端命中率；但静态远端专家仍使其总远端命中率为 `13.47%`，相比
   frequency-cold 只减少 `1.25` 个百分点。请求延迟还包含 attention、非 MoE kernel、
   调度和系统噪声。本轮 random 虽然远端命中更多却最快，也表明单次观测不能建立
   严格的延迟因果或稳定性结论。

所以，本轮可以证明独立冷进程下机制链路正确、放置方向正确，并且**没有观察到纯请求
路径回退**；但不能证明统计意义的性能稳定，也不能证明包含同步维护后的端到端性能
无下降。

## 5. 证据

- 完整 artifact：
  `artifacts/e7-independent-cold-REPRO-20260829-attempt-01-qualification/`
- 聚合校验：`verification.json`，SHA256
  `65cce3f7668be108a9cb8235c22266aa014312ab5a9b5095d62123b2f726e641`
- 跨臂输出 SHA256：
  `ac7558d1855c91bcff2847e1fa299a9d5642986abb2a70fb3bc33b0250b017cd`
- 跨臂路由计数 SHA256：
  `99157cfb2484ca71428e89cbd37c291b3855f17807bdfb43c0a3272ef129fb88`

这是每 arm 一次的 qualification，不提供方差、置信区间或显著性结论。
