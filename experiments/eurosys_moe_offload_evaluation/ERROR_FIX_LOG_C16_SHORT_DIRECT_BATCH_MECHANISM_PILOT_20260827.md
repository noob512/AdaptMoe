# C16 无 HTTP 单批跨机制实验错误与修复记录

本文只记录 2026-08-27 C16 direct-batch pilot 中已解决的错误。失败 artifact 全部
保留，不把失败尝试计入最终性能表。

## F1：公共 harness 覆盖验证器变量

- 现象：attempt-02 的三个 native arm 完成后调用了旧 verifier。
- 根因：公共 harness 中的 `VERIFY` 变量覆盖了本实验同名变量。
- 修复：本实验使用独立变量 `C16_VERIFY`。
- 验证：attempt-04 六类最终 case 均使用
  `verify_direct_batch_c16_case.py`。

## F2：运行中在线修改脚本导致 Bash 读取混合版本

- 现象：attempt-02 在已有 shell 继续读取脚本时退出，artifact 不完整。
- 根因：脚本执行期间发生在线编辑，Bash 后续读取到了不同版本。
- 修复：终止该 attempt；冻结脚本后从新 artifact 根目录重新运行。
- 影响：attempt-02 不计入结果；已完成原始文件保留。

## F3：KTransformers direct API 的 NUMA 节点类型错误

- 现象：启动阶段报
  `TypeError: object of type 'int' has no len()`，峰值 HBM 仅 2,529 MiB。
- 根因：CLI 会把 `--kt-numa-nodes 0` 规范化成 list，但 direct runner 直接构造
  `ServerArgs`，错误传入 `kt_numa_nodes=0`。
- 修复：`direct_batch_c16_ktransformers_runner.py` 改为
  `kt_numa_nodes=[0]`。
- 验证：定向复测 return code 0；3 warmups 和 16/16 measurement 成功。
- 证据：
  `artifacts/c16-short-direct-batch-pilot-20260827-attempt-04/retries/ktransformers-kt-numa-list-fix-attempt-01/`。

## F4：KTransformers direct-batch 验证器误读 running-req

- 现象：计算 16/16 成功，但 verification 唯一失败项为首步 batch。
- 根因：日志为 `new-seq=16,new-token=2048,running-req=0`；旧验证器误把
  `running-req` 当成本步加入后的总数。
- 修复：验证器按当前 SGLang 语义要求 16 个 new sequences、2,048 new tokens、
  0 个此前 running requests。
- 验证：同一不可变 result/run.log 复验 status 为 `ok`。

## F5：MoE-Infinity 比例变化使 KV 从 192 向下取整为 191 blocks

- 现象：`device_ratio=0.829,kv_ratio=0.014036` 在 warmup 前被容量门禁拒绝。
- 根因：MoE serving 的绝对 KV 比例是 `device_ratio * kv_ratio`；只降低前者会使
  绝对 KV 字节数减少并跨过 block floor 边界。
- 修复公式：
  `new_kv_ratio = target_absolute_kv_ratio / new_device_ratio`。
- 验证：0.829/0.827/0.80 后续尝试均恢复 192 blocks；最终 0.80 使用
  0.0145625，实测 339,762,207 B。
- 影响：请求容量保持 3,072 tokens，不通过缩小 KV 绕过 OOM。

## F6：MoE-Infinity non-paged logits 重复峰值

- 实验：Short C16、16 x 128-token Prefill。
- 现象与边界：
  - ratio 0.83：第一条序列的 50 MiB FP32 logits 分配 OOM；
  - ratio 0.829：仍在 50 MiB logits 处分配失败；
  - ratio 0.827：attention softmax 2 MiB 分配失败；
  - ratio 0.80：16 条 forward 完成，最终 `torch.cat` 申请 800 MiB 时 OOM。
- 根因：non-paged fallback 逐序列返回 `[128,102400]` FP32 logits，每份约
  50 MiB。旧实现先把 16 份保存在 list，再由 `torch.cat` 创建另一份
  `[2048,102400]`、约 800 MiB 的输出，峰值约 1.6 GiB。
- 修复：`MoE-Infinity/moe_infinity/serving/model_runner.py` 预分配最终 tensor，
  每次 forward 后直接 `copy_` 到最终 slice；删除最终 `torch.cat`。
- 契约：返回 shape 仍为 `[batch.total_tokens,vocab]`，dtype/device 不变；engine
  仍按 token offsets 选择每条请求最后一个 token。
- 测试：`tests/python/serving/test_model_runner.py`，5 passed。
- 复测：相同 ratio 0.80、192 KV blocks、1 x 16 请求，16/16 成功；峰值
  32,109 MiB，无 fatal/kernel fault。
- 证据：
  `artifacts/c16-short-direct-batch-pilot-20260827-attempt-04/retries/moe-infinity-preallocated-pack-ratio-0.800-attempt-05/`。

## 最终回归

- Python syntax：通过。
- MoE ModelRunner tests：5 passed。
- 六个最终 arms：96/96 measurement 请求完成。
- direct-batch gate：六个 arms 全部通过。
- fatal/kernel fault：最终 arms 为 0。
- GPU 0 MiB、无 compute process、stock UVM、两把 lock 均已释放。
