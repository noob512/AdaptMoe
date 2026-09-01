# E10 KV 2→4→6 GiB pressure qualification attempt-03（2026-08-30）

## 结论

按用户重新授权，调整请求压力后只执行了一次新的 GPU qualification，没有统计重复。本轮 runner 返回码为 `0`，`result.json` 和修复后的离线 verifier 均为 `status=ok`；2、4、6 GiB 三档请求全部完成 Prefill、首次 Decode、5 步 Decode warmup 和 16 步 steady Decode 测量。

本轮证明了以下机制可以完整运行：

- 初始化时一次性保留 8,589,410,304 bytes（约 8 GiB）KV 虚拟地址；
- 初始仅让约 2 GiB KV 物理页驻留 GPU，其余页位于 CPU；
- 根据真实 scheduler backup 队列压力，将 KV GPU 驻留量依次调整为约 4 GiB、6 GiB；
- 每次先切换 cold-frequency 专家档，再调整 KV 物理布局；
- KV 虚拟地址不变，CUDA Graph 不重捕获，三档输出一致。

但这次单轮 qualification 不能证明性能收益。随着 KV 档位增加，需要放到 CPU 的专家更多，steady Decode 中位延迟从 17.93 ms 增至 21.79 ms、26.39 ms；运行期 rebalance 和首次 Decode 拓扑切换也仍有秒级同步开销。

## 请求压力调整

- 2 GiB 请求：关闭独立 backup 自动扩容，只记录压力；请求结束后清除剩余压力事件，避免首个请求跨档。
- 4 GiB、6 GiB 请求：只在 Prefill schedule 窗口开启 autoscale，完成本档切换后立即关闭；Decode 不再触发下一档。
- 每档只运行一个请求；没有预测机制。
- 每次增加 1,365 blocks，实际目标为 1,364、2,730、4,095 blocks。

## 性能结果

| KV GPU 驻留档 | 专家档 | Prefill E2E | 其中 rebalance | Prefill 拓扑恢复 | 纯 Prefill 计算 | steady Decode 中位数 | steady Decode tok/s | 请求 E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 GiB | cold70 | 335.60 ms | 0 ms | 0 ms | 335.60 ms | 17.93 ms | 55.76 | 3016.86 ms |
| 4 GiB | cold80 | 1934.16 ms | 1172.11 ms | 371.28 ms | 390.78 ms | 21.79 ms | 45.88 | 4654.29 ms |
| 6 GiB | cold89 | 2815.23 ms | 1859.04 ms | 521.57 ms | 434.61 ms | 26.39 ms | 37.89 | 6074.64 ms |

`纯 Prefill 计算 = Prefill E2E - rebalance - Prefill 拓扑恢复`。所有数字均来自本次唯一一轮，不能作为具有置信区间的正式性能结果。

## Rebalance 分解

| 切换 | 专家档切换 | KV 物理布局调整 | 总 rebalance |
|---|---:|---:|---:|
| 2→4 GiB / cold70→cold80 | 1108.72 ms | 63.39 ms | 1172.11 ms |
| 4→6 GiB / cold80→cold89 | 1758.75 ms | 100.29 ms | 1859.04 ms |

首次 Decode 还会从 Prefill 拓扑切回 remote Decode 拓扑：

- 2 GiB：首次 Decode 总计 2244.78 ms；其中初始化切换 2208.56 ms，物理审计约 1891.13 ms。
- 4 GiB：首次 Decode 2223.52 ms，其中 PARK 2179.35 ms。
- 6 GiB：首次 Decode 2630.40 ms，其中 PARK 2567.23 ms。

因此当前主要瓶颈不是 KV 页的 2 GiB 增量调整（63–100 ms），而是专家档切换以及 Prefill→Decode 的全量 PARK/物理审计。

## 物理布局和正确性

| 档位 | GPU KV bytes | CPU KV bytes | unknown bytes |
|---|---:|---:|---:|
| 2 GiB | 2,146,959,360 | 6,442,450,944 | 0 |
| 4 GiB | 4,295,491,584 | 4,293,918,720 | 0 |
| 6 GiB | 6,442,450,944 | 2,146,959,360 | 0 |

- 三档 KV VA hash 均为 `cdd0a700b61af3756ee70482cc3a920be1724cd40068d1a403073140aebdf055`。
- Graph capture 计数保持 `6→6`，无重捕获。
- 三档输出 token hash 均为 `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- verifier：`status=ok`，完整序列 1，统计重复 0。
- Xid=0，MMU Fault=0。

## Wrapper 校验误报

GPU runner 成功后，外层脚本最初返回 `1`，原因不是实验失败：旧 verifier 将正常内核日志中的小写 `debug:` 转成大写后，误匹配为 fatal pattern `BUG:`。已把匹配改为区分大小写，并针对同一产物重新执行离线校验；新的 `verification.json` 为 `status=ok`。没有因此重跑 GPU 实验。

## 清理状态

- runner 返回码：`0`；
- 最终 GPU 占用：0 MiB；
- 无残留 compute PID 或 UVM client；
- stock UVM 已恢复，srcversion=`85A79790636BBD99BA3E43B`。

## 产物

- 运行目录：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-03-qualification/`
- runner 结果：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-03-qualification/sequence/result.json`
- 离线校验：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-03-qualification/verification.json`
- 完整日志：`artifacts/e10-kv-2-4-6-pressure-20260830-attempt-03-qualification/sequence/run.log`
