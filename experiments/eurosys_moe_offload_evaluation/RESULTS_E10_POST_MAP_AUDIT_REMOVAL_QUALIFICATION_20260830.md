# E10 去除后置全量驻留查询 qualification（2026-08-30）

## 结论

按授权仅执行了一次完整 GPU qualification，没有失败重试和统计重复。
KV 物理驻留按 `2 -> 4 -> 6 GiB` 调整，专家档位按
`cold70 -> cold80 -> cold89` 调整。runner 和离线 verifier 均为
`status=ok`。

本轮确认：保留远程映射操作自身的逐范围验证，同时关闭建立映射后的
冷、热专家全量驻留查询，可以消除首次 Decode 中约 1.8--2.1 秒的重复
扫描。4 GiB 和 6 GiB 首次 Decode 拓扑切换分别为 363.26 ms 和
482.96 ms，与修改前预计的约 365 ms 和 486 ms 一致。

## 性能结果

| KV GPU 驻留档 | 专家档 | Prefill E2E | Prefill 恢复 | Rebalance（不含恢复） | 纯 Prefill | 首次 Decode E2E | 首次 Decode 拓扑切换 | steady Decode 中位数 | Decode tok/s | 请求 E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 GiB | cold70 | 335.60 ms | 0.00 ms | 0.00 ms | 335.60 ms | 353.68 ms | 321.75 ms | 17.89 ms | 55.90 | 1117.51 ms |
| 4 GiB | cold80 | 1264.17 ms | 327.66 ms | 546.67 ms | 389.85 ms | 401.29 ms | 363.26 ms | 21.77 ms | 45.94 | 2159.04 ms |
| 6 GiB | cold89 | 1606.93 ms | 376.56 ms | 796.08 ms | 434.29 ms | 536.54 ms | 482.96 ms | 26.30 ms | 38.02 | 2768.36 ms |

每档一个请求、5 步 Decode warmup 和 16 步 steady Decode 测量；这些是
单次 qualification 数值，不具有置信区间。

## 首次 Decode 切换分解

| KV 档位 | 暂停旧目录 | 建立并验证 CPU 读取关系 | 后置阶段（查询关闭） | 切换总时间 |
|---|---:|---:|---:|---:|
| 2 GiB | 259.93 ms | 61.52 ms | 0.157 ms | 321.75 ms |
| 4 GiB | 280.14 ms | 82.78 ms | 0.139 ms | 363.26 ms |
| 6 GiB | 356.39 ms | 126.21 ms | 0.185 ms | 482.96 ms |

后置查询字段为 `physical_residency=null` 和
`hot_physical_residency=null`。这不表示没有验证：建立 CPU 读取关系的
操作仍逐范围确认 CPU 驻留、远程映射、覆盖字节以及 unknown bytes，并且
三个档位均为完整成功、`unknown_bytes=0`。

## 与修改前一轮对比

对比对象是紧邻本次修改前的单轮
`20260830T040827Z_qualification`。这是方向性比较，不是统计结论。

| 指标 | 2 GiB | 4 GiB | 6 GiB |
|---|---:|---:|---:|
| 首次 Decode 拓扑切换（旧） | 2268.35 ms | 2173.08 ms | 2567.96 ms |
| 首次 Decode 拓扑切换（新） | 321.75 ms | 363.26 ms | 482.96 ms |
| 降幅 | 85.81% | 83.29% | 81.19% |
| 请求 E2E（旧） | 3067.57 ms | 4146.67 ms | 4715.00 ms |
| 请求 E2E（新） | 1117.51 ms | 2159.04 ms | 2768.36 ms |
| 降幅 | 63.57% | 47.93% | 41.29% |

steady Decode 中位数相对上一轮分别变化约 -0.53%、-0.71%、+0.21%，
没有观察到关闭后置查询影响正常 Decode 性能。6 GiB 本轮 Prefill
rebalance 比上一轮慢，主要来自单次 4,050-entry 差量提交的运行波动；
它与本次删除的首次 Decode 后置查询不是同一条路径。

## Rebalance 路径

- `cold70 -> cold80`：预生成指令命中；4,500-entry 差量由一个 ioctl
  提交，差量操作 390.27 ms；专家档切换 458.61 ms；KV 物理布局调整
  79.62 ms。
- `cold80 -> cold89`：预生成指令命中；4,050-entry 差量由一个 ioctl
  提交，差量操作 588.71 ms；专家档切换 653.89 ms；KV 物理布局调整
  129.10 ms。
- 两次档位切换内部均没有额外 PARK/RESUME；Prefill 目录恢复在切档前
  单独完成。

## 正确性和清理

- 三档输出 token hash 完全一致：
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- Parameter VA 始终不变。
- CUDA Graph capture 数保持 `6 -> 6`，完成 69 次 replay，没有重捕获。
- 自定义 UVM 运行时 `manual_fault_mode=2`，access-counter、prefetch、
  thrashing 均为 `0`。
- verifier `status=ok`，内核错误检查通过。
- 清理无错误，stock UVM 已恢复；最终 GPU 使用 0 MiB，无 compute PID
  和 UVM client 残留。

## 产物

- 唯一运行目录：
  `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-04-post-audit-removal-qualification/`
- runner 结果：上述目录的 `sequence/result.json`。
- 离线校验：上述目录的 `verification.json`。
- 完整日志：上述目录的 `sequence/run.log`。
