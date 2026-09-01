# E3 6.5 Prefill/Decode 三机制完整数据复跑（2026-08-31）

## 结论

本轮重新采集了 native copy/UVA、Custom Prefill Graph-copy 和 Custom Decode
remote Graph。Prefill 10/10、native FLASH 20/20、原协议 `FLASH_ATTN` Custom Decode
8/8 均成功。旧表缺失的 Medium/B8、RAG/B8 由当前档全覆盖双页表 D 路径补齐；又对称
重跑这两点的 native TRITON copy/UVA 4/4，最终得到逐 cell 同 backend、完整输出 hash
一致的 10/10 三机制数据。

完整 Decode 矩阵分为两个协议层：8 个 C/FLASH 点和 2 个 D/TRITON 点。每个 cell 内
Custom、copy、UVA 的 attention backend 一致，可作该 cell 的 exact-generation 数值比较；
但不能把跨层差异当成 attention backend 已被全局控制的单变量实验。

## Prefill-only

Custom 为 `TRITON_ATTN`、exact expert slices、Graph replay；native 为
`FLASH_ATTN`、whole Parameters、eager。`Graph/copy` 和 `Graph/UVA` 均定义为
`native latency / Custom latency`。单位为 ms。

| 请求 | B | Custom Graph-copy | Native copy | Native UVA | Graph/copy | Graph/UVA |
|---|---:|---:|---:|---:|---:|---:|
| Short 128 | 1 | 67.93 | 68.19 | 70.18 | 1.004× | 1.033× |
| Short 128 | 8 | 92.14 | 76.82 | 196.81 | 0.834× | 2.136× |
| Ordinary 512 | 1 | 73.96 | 70.62 | 121.89 | 0.955× | 1.648× |
| Ordinary 512 | 8 | 151.54 | 137.93 | 826.48 | 0.910× | 5.454× |
| Medium 2048 | 1 | 113.44 | 103.48 | 418.44 | 0.912× | 3.689× |
| Medium 2048 | 8 | 651.61 | 563.05 | 7508.83 | 0.864× | 11.524× |
| RAG 8192 | 1 | 588.59 | 461.11 | 3754.00 | 0.783× | 6.378× |
| RAG 8192 | 8 | 3696.11 | 2728.73 | 48291.27 | 0.738× | 13.065× |
| Long 512 | 1 | 75.03 | 71.55 | 122.66 | 0.954× | 1.635× |
| Long 512 | 8 | 151.16 | 138.14 | 834.24 | 0.914× | 5.519× |

Custom Graph-copy 相对 UVA 为 10/10 更快，范围 1.033×--13.065×。相对 native
copy，Short/B1 基本持平并快 1.004×，其余 9/10 点更慢。该表混入 attention backend
和卸载几何差异，不能把差值单独归因于 Graph。

首 token 在 9/10 logical cell 一致。Medium/B8 中 Custom 第 3 条请求首 token 为
264，native copy/UVA 为 279，因此该 Prefill cell 仍不能进入跨三臂 exact-generation
排名。

## Decode-only

延迟为 whole-batch 单 token steady Decode step 中位数。为与 native 排除前5个物理
Decode step 对齐，Custom 使用 `warmup_decode[-1] + measured_decode` 重新计算中位数。
倍率定义为 `native latency / Custom latency`。

| 请求 | B | 三臂协议 | Custom Graph remote | Native copy | Native UVA | Custom/copy | Custom/UVA | 完整hash |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Short 128→32 | 1 | C/FLASH | 10.20 | 70.56 | 68.80 | 6.92× | 6.75× | 一致 |
| Short 128→32 | 8 | C/FLASH | 15.37 | 70.64 | 69.48 | 4.60× | 4.52× | 一致 |
| Ordinary 512→128 | 1 | C/FLASH | 10.13 | 69.90 | 67.36 | 6.90× | 6.65× | 一致 |
| Ordinary 512→128 | 8 | C/FLASH | 17.95 | 71.87 | 69.41 | 4.00× | 3.87× | 一致 |
| Medium 2048→128 | 1 | C/FLASH | 13.98 | 69.32 | 68.97 | 4.96× | 4.93× | 一致 |
| Medium 2048→128 | 8 | D/TRITON | 42.01 | 167.77 | 82.78 | 3.99× | 1.97× | 一致 |
| RAG 8192→128 | 1 | C/FLASH | 24.01 | 161.39 | 68.01 | 6.72× | 2.83× | 一致 |
| RAG 8192→128 | 8 | D/TRITON | 64.21 | 280.88 | 125.06 | 4.37× | 1.95× | 一致 |
| Long 512→512 | 1 | C/FLASH | 11.06 | 69.65 | 67.34 | 6.30× | 6.09× | 一致 |
| Long 512→512 | 8 | C/FLASH | 31.51 | 72.38 | 70.49 | 2.30× | 2.24× | 一致 |

逐 cell 同 backend 的 10 个点中，Custom 全部最快：对 native copy 快
2.30×--6.92×，对 native UVA 快 1.95×--6.75×，完整输出 hash 10/10 一致。
这些范围合并了 FLASH 和 TRITON 两个协议层，只是矩阵概览，不能解释为纯 remote
访问或纯 Graph 的单项贡献。

两个补充点保持原请求大小和B8整批入队，没有拆batch或启用chunked Prefill：

| Cell | cold/层 | Prefill Graph | P→D | steady Decode | E2E | 双页表覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| Medium/B8-D | 38 | 649.66 ms | 28.66 ms | 42.01 ms | 6239.56 ms | 100% |
| RAG/B8-D | 63 | 3691.49 ms | 49.32 ms | 64.21 ms | 12126.53 ms | 100% |

D 两点的 Prefill/Decode Graph 均 replay，无服务期重捕获；`remote_range_map_ioctl_calls=0`。
与最初 FLASH native 表交叉比较时两点完整 hash 不同；改用本轮对称 TRITON native
后，Medium/B8、RAG/B8 均为 Custom/copy/UVA 三臂完整 hash 一致。

## 为什么没有得到全10点统一FLASH

本轮另做三次 Medium/B8-G/FLASH 定向尝试，均保留16,384-token单步Prefill：

1. cold38、4,096-token autotune：startup warmup触发CUBLAS错误，随后CUDA illegal-address；
2. cold38、2,048-token autotune：仍在startup阶段触发CUDA illegal-address；
3. cold63、2,048-token autotune：startup通过，但正式16K Prefill触发CUDA illegal-address。

这重新确认统一 FLASH 下的旧N/A来自 original-VA 大 Prefill 路径能力边界，不是简单
增大卸载量即可修复。最终补全使用 D/TRITON Graph Prefill 绕过该故障，并对称使用
TRITON native arms，没有把跨 backend hash 当作通过。
三次失败均为 `rc=134`，未混入性能表；没有Xid/PDE/PTE fault，但异常退出后的custom模块
卸载记录了 slab-cache cleanup warning。最终成功矩阵结束后GPU为0 MiB、无compute进程，
并已恢复系统stock `nvidia-uvm`。

## 验收

- native FLASH copy/UVA：20/20 `status=ok`；
- 补缺 native TRITON copy/UVA：4/4 `status=ok`；
- Prefill Graph-copy：10/10 `status=ok`，正式Prefill均为Graph replay；
- 原协议 C/FLASH Decode：8/8成功，调度、Graph replay、cleanup和三臂完整hash均通过；
- 当前档双页表矩阵：verifier `status=ok`，10/10 cell、24/24 run；
- D/TRITON补充点：2/2完整生成，双页表100%覆盖，CPU range-map回退为0，与对称
  native TRITON arms完整hash 2/2一致；
- 最终GPU 0 MiB、compute process 0、stock UVM已恢复。

本轮仍是每点一个独立冷进程的 qualification，只给出点估计，不提供统计显著性。

## Artifacts

- native copy/UVA：`artifacts/e3-section65-native-20260831-attempt-01/`
- 补缺 native TRITON：`artifacts/e3-section65-native-triton-large-b8-20260831-attempt-01/`
- Prefill Graph-copy：`artifacts/e3-section65-prefill-20260831-attempt-01/`
- C/FLASH Decode：`artifacts/e3-section65-decode-flash-20260831-attempt-01/`
- D/TRITON补充及双页表verifier：`artifacts/e3-section65-decode-complete-20260831-attempt-01/`
- G/FLASH失败诊断：`artifacts/e3-section65-flash-g-medium-b8-20260831-attempt-01/`、
  `attempt-02/`、`attempt-03-cold63/`。
