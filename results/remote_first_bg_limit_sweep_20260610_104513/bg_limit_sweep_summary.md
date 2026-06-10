# Remote-First BG Promotion Limit Sweep

- Run root: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513`
- Metadata: `/home/ubuntu/nvidia-uvm-gpu/results/remote_first_bg_limit_sweep_20260610_104513/metadata.txt`

## Performance And Counters

| BG_MAX_PAGES | Pass0 ms | Last-pass ms | Median ms | Last GiB/s | RF pages | Remote map | BG enq | Demand promoted | Prefetch promoted | BG committed | PTE switch | Worker rounds | Retries | Cancelled | Fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 12.857 | 0.047 | 0.044 | 331.413 | 4163 | 4163 | 32768 | 3595 | 29173 | 32768 | 32768 | 876 | 0 | 0 | 384 |
| 128 | 13.239 | 0.043 | 0.044 | 359.771 | 5592 | 5592 | 32768 | 5560 | 27208 | 32768 | 32768 | 713 | 0 | 0 | 384 |
| 256 | 11.036 | 0.044 | 0.045 | 357.197 | 5171 | 5171 | 32768 | 5095 | 27673 | 32768 | 32768 | 645 | 0 | 0 | 384 |
| 512 | 13.293 | 0.043 | 0.044 | 366.761 | 5418 | 5418 | 32768 | 5394 | 27374 | 32768 | 32768 | 669 | 0 | 0 | 384 |
| 1024 | 12.468 | 0.043 | 0.044 | 363.713 | 5074 | 5074 | 32768 | 5058 | 27710 | 32768 | 32768 | 638 | 0 | 0 | 384 |
| 0 | 12.274 | 0.044 | 0.044 | 357.930 | 4867 | 4867 | 32768 | 4855 | 27913 | 32768 | 32768 | 638 | 0 | 0 | 384 |

## How To Read This

- If `BG_MAX_PAGES` is too large, pass0 may rise because background copy competes with urgent remote reads.
- If `BG_MAX_PAGES` is too small, worker rounds rise and later passes may remain remote for longer.
- A useful setting keeps pass0 low while making `BG committed` and `PTE switch` reach the working set quickly.
