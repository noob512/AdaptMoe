# E2-C16 direct-batch 错误与修复记录

日期：2026-08-28

## F1：HTTP 到达顺序形成 `1+(N-1)`，不是真正的一次性 C16 admission

- 现象：首请求先进入 Prefill/Decode，其余请求等待，导致高并发 TTFT 约 2.2–2.5 s。
- 根因：客户端、HTTP 和请求预处理使第一个请求先触发 engine step。
- 修复：使用 `Processor.process_inputs + LLMEngine.add_request`，16 次 admission 全部完成后才调用第一次 `EngineCore.step()`；不启动 server/client/socket。
- 变更：`direct_batch_phase_runner.py`、`direct_batch_c16_vllm_runner.py` 及两个跨系统 runner。
- 验证：所有有效 cell 的 `one_c16_measurement_admission=true`；Short/Ordinary/Long 首 step 为 16。RAG 因 token budget 首 step 为 2，但 16 条仍在该 step 前全部入队。

## F2：Prefill 与 Decode mixed step

- 现象：长短请求并存时，新 Prefill 可能与已有 Decode 同一步执行，改变实验含义。
- 根因：默认 continuous batching 允许 mixed scheduling。
- 修复：启用 phase-batched scheduler 和 KV-aware cohort；KTransformers 禁用 mixed chunk；MoE-Infinity 先清空 waiting Prefill 再 Decode。
- 验证：20 个有效 cell 的 `scheduler_prefilled_all_16_before_decode` 与 `no_mixed_prefill_decode_step` 均为 true。

## F3：KV/临时缓冲区建立时驱逐热门专家

- 现象：Graph/KV 成功建立后，热门专家仍有页面落在 CPU，后续访问可能失败；单纯再次 rebalance 不能解决总容量不足。
- 根因：KV、MoE workspace、Prefill allocator reserve 与 Graph pool 的永久/临时分配顺序会占用专家原本可恢复的 GPU 空间。
- 修复：
  1. KV 初始化前建立与 KV 同尺寸的 native hole sentinel；
  2. Decode Graph 捕获后释放 hole，给 Prefill Graph/临时区连续空间；
  3. warmup 阶段保留与正式 token shape 一致的 Prefill allocator reserve；
  4. Graph 建立期间专家先保持 remote，完成后恢复热点专家并强制检查驻留和 192 MiB headroom；
  5. 空间仍不足时增加 cold experts，不缩短请求或 KV。
- 涉及：`workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py`、`model_executor/models/utils.py`、`run_e2_c16_direct_batch_matrix.sh`。
- 验证：ours Short/Ordinary/RAG/Long 均无 CUDA fault；GPU 和 stock UVM 正常清理。

## F4：ours RAG 卸载量不足

- 现象：cold16/17 热点专家未全部回到 GPU；cold18 恢复后 headroom 不足。
- 根因：4,160 MiB KV、Prefill allocator reserve 和 `{1,16,16384}` Graph pool 的固定占用超过较小 cold 配置释放的空间。
- 修复：Graph shape 从通用集合缩减到实验实际需要的 `{1,16,prefill}`，候选扩展至 cold24；保留驻留断言，继续扫描。
- 验证：cold19 最小成功，16/16 完整生成，峰值 31,705 MiB；cold16/17/18 失败证据均保留在 attempt-04。
- 潜在影响：cold19 比较小 workload 卸载更多专家；这是容量约束，不应跨 workload 直接比较卸载量。

## F5：copy-on-forward 通用遍历误卸载 attention 参数

- 现象：旧扫描把 attention 参数迁到 CPU，缓存 `W_UK_T` 仍引用 CPU storage；选择整个 `w13_weight` 又会产生 704 MiB 单次复制并触发临时 OOM。
- 根因：按模型参数遍历顺序达到 `cpu_offload_gb`，没有限制到可安全包装的目标张量。
- 修复：使用 `VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE`，只选择各层完整 `model.layers.N.mlp.experts.w2_weight`；每层精确 369,098,752 bytes（352 MiB）。
- 配置扫描：Ordinary 2–4 层、RAG 13–20 层、Long 3–6 层。
- 验证：相关 exact-name 单测 5 passed；Ordinary 2 层、RAG 15 层、Long 3 层均通过。RAG 13/14 的 OOM 完整保留，证明 15 为候选集合内最小成功点。
- 涉及：`configs/e2_c16_direct_batch_matrix.json`、`run_e2_c16_direct_batch_matrix.sh`、`direct_batch_c16_vllm_runner.py`。

## F6：MoE-Infinity 全序列 logits 重复分配

- 现象：C16 时全序列 FP32 logits 拼接产生约 800 MiB 峰值，长请求 OOM。
- 根因：serving 只需每条序列最后 token logits，却投影并拼接所有 token。
- 修复：仅对最后 hidden state 执行 `lm_head`；逐序列释放临时 logits，避免两份 8K logits 并存。
- 涉及：`MoE-Infinity/moe_infinity/serving/model_runner.py`、`serving/engine.py`。
- 验证：MoE-Infinity Short、Ordinary、Long 完成 16/16；定向测试 24 passed，完整 serving 测试 144 passed。

## F7：MoE-Infinity 重复物化未使用的 PagedKV GPU tensor

- 现象：DeepSeek non-paged backend 已使用 HF `past_key_values`，仍额外分配第二份从未读取的 PagedKV storage。
- 根因：逻辑 block allocator 与物理 PagedKV storage 被绑定创建。
- 修复：增加 `materialize_paged_kv_cache=False`；保留逻辑 block 容量/所有权记账，但 `kv_cache_storage_bytes=0`。
- 涉及：`serving/kv_cache.py`、`serving/engine.py`、`api_server_v2.py`、`direct_batch_c16_moe_infinity_runner.py`。
- 验证：有效结果记录 `paged_kv_storage_materialized=false`；Short/Ordinary/Long 通过。RAG 的剩余 expert-cache OOM 已单独列入未解决报告。

## F8：较小 KTransformers placement 容量不足

- 现象：部分候选在 activation/MoE 临时分配处 CUDA OOM，父进程随后以 137 退出。
- 修复：保持请求与 KV 不变，增加每层 CPU experts，逐候选重启独立进程。
- 验证：Short cold1、Ordinary cold2、RAG cold12、Long cold2 均完成 16/16；失败候选未计入结果。

## F9：早期主扫描误调用通用 verifier

- 现象：主扫描的 all-GPU Short 首次执行在 `execution.tsv` 中记录 verification return code 2；`verify.stderr.txt` 显示通用 `verify_experiment.py` 不接受 `--arm native_all_gpu`。
- 根因：早期 launcher 与被 `source` 的基础 harness 存在 verifier 变量命名/调用冲突。
- 修复：使用独立变量 `C16_VERIFY`，固定调用 `verify_direct_batch_c16_case.py <case> --arm <arm>`。
- 验证：对未修改的原始 `result.json`、`run.log`、kernel log 和 NVML 数据重新执行专用 verifier；现有 `verification.json` 为 `status=ok` 且 11 项检查全 true。原始 return code 2 和 stderr 保留，未篡改历史执行记录。
