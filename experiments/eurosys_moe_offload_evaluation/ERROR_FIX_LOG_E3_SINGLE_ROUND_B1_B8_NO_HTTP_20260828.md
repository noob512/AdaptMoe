# E3 B1/B8 无 HTTP：错误与修复记录（2026-08-28）

## EF-E3-001：KTransformers direct Engine 参数名不兼容

对应失败 artifact：

`artifacts/e3-repro-20260828-attempt-02-single-round-no-http`

现象：custom 阶段 21/21 有效 runs 已通过，但每个 KT workload 都在 GPU 分配前退出：

```text
TypeError: ServerArgs.__init__() got an unexpected keyword argument 'fp8_gemm_backend'
```

根因：`--fp8-gemm-backend` 是 CLI 选项名；当前 SGLang 的 in-process `Engine` 接口接收 `ServerArgs` dataclass 字段名 `fp8_gemm_runner_backend`。direct runner 错把旧/CLI 名称传给 Python API。

最小修复：

- 文件：`direct_batch_e3_ktransformers_runner.py`；
- 将 `fp8_gemm_backend="triton"` 改为 `fp8_gemm_runner_backend="triton"`；
- FP8 后端的数值设置仍为 Triton，未改变实验机制、模型、专家坐标、KV 或 batch。

验证：

- Python 编译、shell 语法与 `git diff --check` 通过；
- 用 AST 提取全部 31 个 `Engine` keyword，与当前 `inspect.signature(ServerArgs)` 比较，未知字段为 0；
- 新 artifact `attempt-03` 中 5 个 KT 进程均成功初始化；
- 9/9 个 KT measurement cells、37/37 请求通过；
- verifier 返回 `status=ok`、`errors=[]`。

潜在影响：仅修正 direct Python API 的参数映射；不影响既有 CLI runner，也不改变 FP8 kernel 选择。

## EF-E3-002：UVM 模块切换窗口内 NVML 短暂不可用

现象：attempt-02 custom 完成、从 custom UVM 恢复 stock UVM 的瞬间，一次 `nvidia-smi` 返回无法连接 driver。

根因：采样命令恰好落在 NVIDIA/UVM 模块卸载和重新加载窗口内；不是 cell 内 CUDA fault。

处理：launcher 的 `wait_gpu_clean` 等待模块恢复后才启动 KT，没有并发访问 GPU。后续 `nvidia-smi` 正常，attempt-03 的 before/custom/after module snapshot 完整，最终 GPU 0 MiB。

潜在影响：无实验数据影响；监控脚本不应把模块切换窗口的一次 NVML 失败误记为 workload 失败。
