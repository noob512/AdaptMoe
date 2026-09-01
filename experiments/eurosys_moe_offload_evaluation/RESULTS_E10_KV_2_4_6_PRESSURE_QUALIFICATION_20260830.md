# E10 KV 2→4→6 GiB pressure qualification（2026-08-30）

## 结论

本次只启动了一次正式 GPU 实验，没有重复运行。唯一一次运行在启动期预编译三档专家 Graph 后、恢复初始专家档位时失败，尚未进入请求执行。因此，本次没有 Prefill、Decode 或 rebalance 端到端性能数据，不能作为 E10 性能结论。

离线 verifier 状态为 `failed`，runner 返回码为 `134`，完成请求数为 `0`。

## 实验协议

- KV 使用一个固定的 8 GiB managed 虚拟地址池：8,589,410,304 bytes，共 5,461 blocks。
- 计划物理驻留档位为 1,364 / 2,730 / 4,095 blocks，近似 2 / 4 / 6 GiB。
- 压力只来自真实 scheduler backup 队列消费，不启用预测。
- 每次 rebalance 的顺序固定为：先切换 cold-frequency 专家档位，再扩大 KV 物理驻留前缀。
- 专家档位为 cold70 / cold80 / cold89：
  - cold70→cold80：增加 2,264,924,160 bytes 冷专家；
  - cold80→cold89：增加 2,038,431,744 bytes 冷专家。
- 每个 KV 档位原计划执行 1 个请求，5 个 Decode 预热 step 和 16 个 Decode 计时 step；统计重复数为 0。

## 唯一一次运行实际完成的部分

- 8 GiB KV managed 地址池已经分配，vLLM 报告 87,376 token 的 KV 容量。
- 未来两档新增专家的 4,303,355,904 bytes pinned CPU source 已在初始化期填充。
- 同样大小的 immutable CPU shadow 已建立，覆盖 1,824 个专家参数范围。
- cold70→cold80 成功，耗时 2,501.20 ms。
- cold80→cold89 成功，耗时 2,713.29 ms。
- 三个专家档位的 Graph 均完成预捕获，过程中 Parameter VA 保持稳定。

注意：运行在 `compile_or_warm_up_model()` 内失败。虽然 8 GiB KV 地址池已分配，但随后把非活动 KV 页搬回 CPU、建立初始 2 GiB 物理布局的 pre-service rebalance 尚未执行。因此不能声称“KV 2→4→6 GiB 动态切换”已经跑通。

## 失败位置与原因

预捕获结束后需要从 cold89 恢复到初始 cold70，一次恢复 1,824 个专家参数范围。该变化超过增量接口单次 1,152-entry 的上限，因此进入 full REBIND 路径。

旧路径在解除 cold89 的远端 alias 之前，先尝试让这些 Parameter VA 恢复 GPU 驻留。此时 VA 仍指向 immutable CPU shadow，导致恢复结果为 `advised=0, prefetched=1824`，但驻留校验没有看到 1,824 个范围回到 GPU。随后的回滚又触发非法 GPU 地址访问，进程被标记为 `poisoned`；内核记录 1 次 Xid 31 MMU fault。

失败切换 cold89→cold70 耗时 1,414.55 ms。该时间是失败路径耗时，不是有效的 rebalance 性能数据。

## 失败后的修复状态

代码已调整为：对于大型 remove-only 档位变化，先解除旧 alias，再恢复专家 GPU 驻留。针对增量和 full REBIND 两条路径的 4 项测试通过，KV/plan-class 相关的 47 项回归测试也通过。

这只是 CPU mock/单元测试验证。按照“不重复实验”的要求，没有再次启动 GPU，因此修复尚未获得真实 GPU 验证。

## 清理状态

- 实验进程和 UVM client 均已退出；
- GPU 最终显存占用为 0 MiB；
- stock UVM 已恢复，srcversion 为 `85A79790636BBD99BA3E43B`。

## 产物

- 唯一运行目录：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-01-qualification/`
- 离线校验：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-01-qualification/verification.json`
- runner 结果：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-01-qualification/sequence/result.json`
- 完整日志：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-01-qualification/sequence/run.log`
