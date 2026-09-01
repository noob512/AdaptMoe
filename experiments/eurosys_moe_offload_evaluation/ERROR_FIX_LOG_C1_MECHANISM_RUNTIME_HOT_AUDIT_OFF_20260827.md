# C1 跨机制实验错误与修复记录

本文只记录 2026-08-27 C1 runtime-hot-audit-off matrix 中已解决的问题。所有路径均
相对于仓库根目录。

## F1：ours 首个 Decode 被热门专家审计主导

- 实验：ours / Short cold1。
- 现象：audit-on transition 426.84 ms，首个 ITL 446.55 ms。
- 根因：PARK 前、PARK 后、remote-map 后三轮扫描/抽样读取 28.34 GiB hot experts，
  transition 尾部又做全量 hot physical-residency 查询。
- 修复：增加 `VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT`，默认开启；本实验同时设置
  `VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0` 和
  `VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0`。
- 代码：`workloads/vllm/vllm/vllm/model_executor/models/utils.py`；测试：
  `workloads/vllm/vllm/tests/test_cpu_offload_exact_static.py`。
- 验证：相关单测 2 passed；Short transition 37.78 ms，下降 91.1%；hot audit 字段
  为 false/null，cold residency、stable VA 和 fault gate 保留。
- 影响：只改变显式实验环境；默认行为不变。

## F2：原生 vLLM RAG MLA workspace OOM

- 实验：all-GPU、pinned-UVA、copy / RAG。
- 现象：8K 单步 Prefill 在 512 MiB MLA 临时分配处 OOM；expandable allocator 无效，
  4K chunk 仍 OOM。
- 根因：32 GiB HBM 上权重、KV 和 Prefill workspace 的瞬时总量不兼容。
- 修复：保持同一 8192/128 token 请求，设置 `max-num-batched-tokens=2048`，执行
  2K x 4 Prefill。
- 验证：三个机制均 1/1 成功，TTFT 390.40/465.81/390.71 ms。
- 影响：请求未改变；后续 F3 也把 ours 配平为 2K x 4。

## F3：ours RAG 把请求长度当作单个 Graph shape，造成过度卸载

- 实验：ours / RAG。
- 现象：单步 8K 配置下 cold1、cold2、cold3 均无法同时建立完整 KV、8K Graph/workspace
  和 hot placement，cold4 才通过。
- 根因：8,192 是完整请求长度，不要求成为一个 CUDA Graph shape。原生 vLLM 已因
  512 MiB MLA workspace 使用 2K x 4；ours 保留单步 8K 会额外占用 Graph/workspace，
  与“精确预留并尽可能少卸载”目标不一致。
- 修复：KV 仍精确预留 8,320 tokens / 260 MiB，只把 ours 的 Prefill Graph、warmup
  reserve、MoE chunk 和 `max-num-batched-tokens` 统一为 2,048。
- 验证：cold1 完成 8192/128 请求，TTFT 385.00 ms、TPOT 11.74 ms、E2E 1875.39 ms、
  transition 34.91 ms；cold residency 429 MiB 全在 CPU，无 CUDA/PTE fault。
- 影响：最终四种 ours workload 均只需 cold1；旧 cold1–cold3 失败只代表单步 8K
  配置的容量边界。

## F4：KTransformers RAG 少生成两个 token

- 实验：KTransformers / RAG cold1。
- 现象：`max-total-tokens=8320` 时只生成 126/128 tokens；增加 cold experts 无效。
- 根因：SGLang 内部预留两个 token 槽。
- 修复：只把 RAG `max-total-tokens` 调整为 8322。
- 验证：cold1 完成 128/128，TTFT 1335.15 ms、TPOT 10.79 ms。
- 影响：不改变输入、目标输出或专家布局。

## F5：MoE-Infinity KV ratio 不能跨请求复用

- 实验：MoE-Infinity / Ordinary、RAG、Long。
- 现象：Ordinary 使用 `kv_ratio=0.001` 时，admin stats 为
  `pending_requests=1,num_steps=0`，GPU 利用率 0；请求不进入 Prefill。
- 根因：KV block 为 16 tokens、1,769,472 B；0.001 只得到 13–14 blocks，低于
  Ordinary 所需 42 blocks。原先 RAG=0.012、Long=0.002 也不足。
- 修复：矩阵冻结 12/42/520/66 blocks；对每个 device ratio 用参考机 available bytes
  反算 `kv_cache_ratio`，向上取整到 1e-6，使绝对 block 数不随 expert ratio 改变。
- 代码/配置：`configs/c1_mechanism_runtime_hot_audit_off_matrix.json` 和
  `run_c1_mechanism_runtime_hot_audit_off_matrix.sh`。
- 验证：Ordinary 0.85 实际建立 42 blocks 并完成；RAG 精确建立 520 blocks；Long
  建立 66 blocks 并完成。
- 旧证据：`ordinary_chat/ratio_85` 的 `stall_diagnosis.json`、`admin_stats.json`，以及
  `ratio_84/ABORTED_INVALID_KV.json`。

## F6：MoE-Infinity RAG 永久停在 scheduler waiting

- 实验：MoE-Infinity / RAG，KV 已修复为 520 blocks。
- 现象：520/520 blocks 空闲，仍为 `pending=1,num_steps=0`。
- 根因：serving engine 把 `max_tokens_per_step` 固定为 2048；scheduler 对 8192-token
  Prefill 执行 admission `break`，实现没有 chunked-prefill continuation。
- 修复：把 `max_tokens_per_step` 作为可选参数从 `MoE.serve` 传到 engine config；实验
  launcher 和 runner 对 RAG 显式传 8192。
- 代码：`MoE-Infinity/moe_infinity/entrypoints/big_modeling.py`、
  `MoE-Infinity/moe_infinity/entrypoints/openai/api_server_v2.py`、
  `launch_moe_infinity_server.py`。
- 测试：新增 API config 定向单测，1 passed。
- 验证：RAG 能进入真实 Prefill；ratio 0.36 OOM、0.35 完整成功，形成可解释边界。
- 影响：默认仍为 2048；只有显式 RAG 实验使用 8192。

## F7：KTransformers 正常关停 traceback 被误报为 fatal

- 实验：结果汇总。
- 现象：请求成功后，SGLang scheduler/detokenizer 接收 SIGINT，两个 traceback 可能
  交错并以 `KeyboardInterrupt` 结束；旧正则把首行 `Traceback` 标为 fatal。
- 根因：汇总器没有区分运行期异常和成功请求后的正常关停。
- 修复：仅对“otherwise successful KTransformers case”，过滤在 timestamped
  `INFO: Shutting down` 前以 KeyboardInterrupt 结束的 traceback；失败 case 和其他
  fatal 不过滤。
- 代码：`summarize_c1_mechanism_runtime_hot_audit_off.py`。
- 验证：24 个 final cells 的 `fatal_log_excerpt` 全部为 null；MoE OOM 失败仍被捕获。

## F8：人工中断留下 NVML sampler，阻塞 UVM 模块切换

- 实验：ours / RAG 2K x 4 配对复测的启动前检查。
- 现象：GPU 为 0 MiB、无 compute process，但 `wait_gpu_clean` 一直等待；`fuser`
  显示三个 `/dev/nvidia-uvm` client。
- 根因：早先人工中断的 MoE-Infinity ratio 84、83 和 RAG ratio 85 attempt 留下三个
  父进程为 PID 1 的 `nvidia-smi --loop-ms=200`。其 stdout FD 分别指向对应 case 的
  `nvml.csv`，所以它们不是模型进程或系统 exporter。
- 修复：逐一核验 PID、cmdline 和 `/proc/<pid>/fd/1` 后，仅发送 `SIGTERM` 给三个孤儿
  sampler；未杀死来源不明的 UVM client，也未跳过 clean gate。
- 验证：custom UVM 正常装载，RAG cold1 复测通过；退出后 stock UVM srcversion
  `85A79790636BBD99BA3E43B`、GPU 0 MiB、无 compute process、端口和 lock 均释放。
- 影响：没有改变模型、请求或测量；复现文档新增人工中断后的定向清理方法。

## 最终回归

- ours audit 开关定向测试：2 passed；
- MoE `max_tokens_per_step` 定向测试：1 passed；
- shell/JSON/Python syntax 检查通过；
- 24/24 final cells 请求成功，78 attempts；
- 成功日志 0 fatal；
- GPU、端口、lock 和 stock-UVM cleanup 通过。

## F9：Decode-only 早退后遗留 sampler 和 custom UVM

- 实验：RAG `FULL_DECODE_ONLY` attempt-02/03 的诊断基础设施清理。
- 现象：attempt-02 的 EngineCore 早退后，case 的 NVML sampler 仍持有
  `/dev/nvidia-uvm`，loaded srcversion 仍为 custom `5BA0A4BFB3AD96A55780A4C`；因此
  attempt-03 的 preflight 正确拒绝启动并报告 `initial UVM is not stock`。
- 根因：异常/人工终止发生在 runner 写出正常 cleanup artifact 前，孤儿 sampler 阻止
  custom UVM 卸载。EngineCore 的 `pure virtual method called` 根因尚未解决，单列 U5。
- 修复：通过 sampler 的 cmdline 和 stdout FD 定向确认其属于 attempt-02 后终止该
  sampler，再执行正式模块恢复；没有跳过 initial-module gate，也没有杀死来源不明进程。
- 验证：attempt-04 的 `module.before.json` 和 `module.after.json` 均为 stock
  `85A79790636BBD99BA3E43B`；1/1 请求成功，GPU 0 MiB、30404 空闲、无 UVM client。
- 影响：只清理失败诊断遗留资源，不改变模型、Graph 模式、请求或输出。
