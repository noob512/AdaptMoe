# E3 CUDA Graph 单轮错误与修复记录（2026-08-28）

## F1：原 E3 H/A/R 实际为 eager

- 现象：虽传入 capture sizes，运行时为 `enforce_eager=True`、
  `cudagraph_mode=NONE`。
- 根因：H/A/R contract 的 `graph=false`。
- 修复：常规 cell 改用 G/B/C；大 Prefill 使用 D。verifier 强制检查实际 mode、
  capture sizes 和每步 `full_replay=1`。
- 验证：聚合 24/24 runs 通过 Graph contract。

## F2：RAG/B1 首次 D 启动丢失 PARK/RESUME contract

- Artifact：`artifacts/e3-graph-rag-b1-D-tier8-protected-20260828-attempt-01/`。
- 错误：`preserved_graph_contract_reason=alias_prefill_graphs_are_still_live`。
- 根因：漏传 `VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1` 和
  `VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1`，启动时销毁 session，
  Prefill capture 又创建新 session，post-capture transition 正确拒绝销毁活 Graph。
- 修复：补齐 PARK/RESUME、capture-before-hot-restore、all-remote profile 和
  192 MiB minimum-free contract。
- 验证：attempt-02 通过，B1 完整输出，双 Graph replay，session_count=0。

## F3：RAG/B8 cold57/60/62 显存门槛失败

| Attempt | 配置 | 错误证据 |
|---|---|---|
| `...tier12...attempt-01` | cold57、65K reserve | 313 MiB hot experts 留在 CPU |
| `...cold60...attempt-02` | cold60、65K reserve | 75 MiB hot experts 留在 CPU |
| `...cold62...attempt-03` | cold62、65K reserve | hot 全恢复，但 free=160.75 MiB < 192 MiB |

根因是 65K reserve warmup 的 `reserved_delta=3,491,758,080 B`（3.252 GiB）
占用过大；只增加 cold experts 会牺牲越来越多 GPU-local专家，同时未消除过度
reserve。

最小修复：保留 cold63，将 reserve warmup 从65,536降到20,480 tokens。实际
`reserved_delta=1,094,713,344 B`（1.020 GiB），而正式 Prefill 仍由已捕获的
65,536-token Graph replay承担。

验证：`artifacts/e3-graph-rag-b8-D-cold63-reserve1g-20260828-attempt-05/`：
8/8 完整输出；Prefill、first Decode、121 measured Decode 全部 replay；hot恢复后
free=2.012 GiB；无 fault；session_count=0；GPU结束为0 MiB。

潜在影响：1 GiB reserve 仅单轮验证，尚不能证明跨进程 allocator 稳定性；已列入
未解决问题。

## F4：缺失 cold60/cold62/cold63 计划

- 现象：冻结输入只有 cold57（12g），无法按实测缺口做最小逐层扩容。
- 补充：使用 `freeze_qwen_e3_plan.py` 从同一 routing source
  SHA256 `912732...d072f81` 派生计划；tie-break、排序和专家字节几何不变。
- 最终 cold63：每层63个专家，selected main=14,269,022,208 B，plan SHA256
  `05686bd2c4df0f5c052c90902e5945bee9d921a8d36b7b1009712dd314cc8620`。
- 位置：`artifacts/e3-derived-plan-cold63-20260828/`。

## F5：被替换配置的人工终止

`artifacts/e3-graph-rag-b8-D-cold63-protected-20260828-attempt-04/` 在用户要求把
reserve降至1 GiB后被 SIGINT 终止。GPU和stock UVM均恢复，未完成结果不计入
正式汇总；这不是算法失败。
