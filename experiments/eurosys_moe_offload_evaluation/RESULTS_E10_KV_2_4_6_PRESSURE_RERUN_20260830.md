# E10 KV 2→4→6 GiB pressure qualification rerun（2026-08-30）

## 结论

经用户明确授权，修复后只重新启动了一次 GPU qualification，产物为 `attempt-02`。该轮验证了第一次失败的专家恢复问题已经修复，也成功建立了 8 GiB 稳定 KV 虚拟地址池及初始 2 GiB GPU / 6 GiB CPU 物理布局；但首个 2 GiB 请求结束后的协议检查失败，因此仍未得到可验证的三档 Prefill/Decode 性能结果。

离线 verifier 状态为 `failed`，runner 返回码为 `1`，失败阶段为 `first_2g_request_unexpected_early_growth`。没有进行第三次 GPU 运行。

## 本轮成功验证的部分

- 8,589,410,304 bytes KV 稳定虚拟地址池分配成功。
- 初始物理布局正确：
  - GPU 头部：2,146,959,360 bytes，target=1,364 blocks；
  - CPU 尾部：6,442,450,944 bytes；
  - unknown bytes=0。
- 三档 CUDA Graph 均完成预捕获。
- immutable CPU shadow 覆盖 1,824 个专家参数范围，共 4,303,355,904 bytes。
- 专家档位切换：
  - cold70→cold80 捕获切换成功：1,864.18 ms；
  - cold80→cold89 捕获切换成功：2,522.81 ms；
  - cold89→cold70 恢复成功：2,933.94 ms。
- 上轮失败的 full REBIND 顺序问题未再出现；`startup_plan_class_restore_succeeded=true`。
- 内核 Xid=0，MMU fault=0。

## 新的失败原因

协议要求首个 2 GiB 请求期间关闭统一 autoscale，只记录 backup 队列压力；等该请求完成并记录 2 GiB 性能后，再开启压力驱动的 2→4→6 GiB 切换。

但 EngineCore 还保留了一条旧的“独立 backup-pressure 扩容”路径。即使 `VLLM_UVM_KV_AUTO_SCALE_ENABLE=0`，它仍会消费 backup 压力并执行扩容。本轮首个请求先激活 4 个 backup blocks，随后系统提前执行 cold70→cold80；又激活 1 个 backup block，并继续执行 cold80→cold89。

这两次非预期运行期专家切换分别耗时：

- cold70→cold80：2,131.23 ms；
- cold80→cold89：1,661.16 ms。

因此 runner 在准备写入 2 GiB tier 结果时发现当前 KV target 已不是 1,364 blocks，触发 `ContractError('initial KV tier is not 2 GiB')`。

首个请求已经执行到该协议检查点，但 Prefill/Decode 计时尚未写入 `result.json`。为避免把不完整的内存中数据当作正式结果，verifier 将 `requests_measured` 记为 0，Prefill/Decode 均为 `null`。

## 运行后的修复

新增 `VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE`：

- E10 首个 2 GiB 请求期间设为 `0`，保留 backup 压力但不提前扩容；
- runner 完成 2 GiB 记录后开启统一 autoscale；
- 后续压力再按“专家档位优先、KV 物理前缀随后”的顺序触发 2→4→6 GiB。

同时，runner 预检现在强制检查该隔离开关，verifier 也能正确区分此次失败阶段。修复后 3 项针对性测试和完整 48 项 KV/plan-class 回归测试均通过。该修复尚未进行真实 GPU 验证。

## 清理状态

- GPU 最终显存占用：0 MiB；
- 无残留计算进程或 UVM client；
- stock UVM 已恢复，srcversion=`85A79790636BBD99BA3E43B`。

## 产物

- 唯一授权重跑目录：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-02-qualification/`
- 离线校验：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-02-qualification/verification.json`
- runner 结果：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-02-qualification/sequence/result.json`
- 完整日志：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-02-qualification/sequence/run.log`
