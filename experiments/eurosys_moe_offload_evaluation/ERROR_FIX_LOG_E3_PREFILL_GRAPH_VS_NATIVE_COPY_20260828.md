# E3 Prefill-only Graph vs native：错误与修复记录

## EF-1：错误缩小 max_model_len

- 发生范围：此前 `e3-prefill-only-four-mechanism-20260828-attempt-02`；
- 现象：runner 将 `max_model_len` 设为 `prompt+1`，Long workload 从冻结的 1024 错误缩为 513；
- 根因：把“只采样一个 token”误当成可以缩小原请求 context envelope；
- 修复：`run_e3_prefill_only_four_mechanism.sh` 使用 `prompt+original_output`，只将 measured output 固定为 1；
- 验证：新 artifact 的每个 `result.json` 均记录正确 `max_model_len`；Long/B1/B8 分别为 1024；
- 处理：attempt-02 全部不计入正式性能数据。

## EF-2：四机制旧矩阵被用户收窄为双机制

- 发生范围：`e3-prefill-only-four-mechanism-20260828-attempt-03`；
- 处理：按用户要求终止，不把其不完整数据并入新双机制矩阵；
- 最小变更：runner shell 增加 `CUSTOM_ARM_FILTER`、`RUN_CUSTOM`、`RUN_NATIVE`，不改变 cell 参数；
- 验证：新 `configuration.json` 明确 `custom_arm_filter=graph_copy`，execution 只含 Graph-copy 与 native copy-on-forward。

## EF-3：防止 B8 被拆成 1+7

- 风险：客户端或逐条请求接收会让 scheduler 先处理 1 条、后处理剩余 7 条；
- 方案：in-process 调用 `_validate_and_add_requests()` 一次加入完整 list，再执行一次 `llm_engine.step()`；
- 防回归：`phase.validate_prefill()`、request count、scheduled token count 和无 cached/resumed 请求均为强制 contract；
- 验证：全部 B8 结果的首步 `new_request_ids` 长度为 8，总 token 数精确等于 `prompt*8`。

## EF-4：容量随大 Prefill自适应

- 风险：固定 cold10 无法同时容纳大 KV、Prefill workspace、Graph pool 与热门专家；
- 方案：不改变 batch，根据冻结空间账目仅增加卸载专家数：Medium/B8、RAG/B1 用 cold38，RAG/B8 用 cold63；
- 验证：三个大 Graph cell 均成功，最大 65,536-token 单步 Prefill 通过，峰值 HBM 30,829 MiB；无热门专家访问失效和 CUDA fault。
