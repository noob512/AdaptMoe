// Reproduce the CTA geometry used by vLLM's Triton fused MoE GEMM and record
// which SM executes each output tile for a single expert.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                             \
        if (status != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,           \
                         cudaGetErrorString(status));                            \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

struct CtaRecord {
    uint32_t smid;
    uint32_t pid_m;
    uint32_t pid_n;
    uint32_t unused;
    uint64_t start_ns;
    uint64_t end_ns;
};

__device__ __forceinline__ uint32_t read_smid() {
    uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void fused_moe_geometry_probe(CtaRecord* records, int num_pid_m,
                                         int num_pid_n, int group_size_m,
                                         uint64_t hold_ns) {
    const int pid = static_cast<int>(blockIdx.x);
    const int num_pid_in_group = group_size_m * num_pid_n;
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * group_size_m;
    const int actual_group_m = min(num_pid_m - first_pid_m, group_size_m);
    const int pid_in_group = pid % num_pid_in_group;
    const int pid_m = first_pid_m + (pid_in_group % actual_group_m);
    const int pid_n = pid_in_group / actual_group_m;

    __shared__ uint64_t block_start_ns;
    if (threadIdx.x == 0) {
        records[pid].smid = read_smid();
        records[pid].pid_m = static_cast<uint32_t>(pid_m);
        records[pid].pid_n = static_cast<uint32_t>(pid_n);
        block_start_ns = read_globaltimer();
        records[pid].start_ns = block_start_ns;
    }
    __syncthreads();

    // Keep the CTA resident long enough to distinguish actual concurrent SM
    // execution from sequential reuse of several SMs.
    while (read_globaltimer() - block_start_ns < hold_ns) {
        // The volatile special-register read itself prevents loop removal.
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        records[pid].end_ns = read_globaltimer();
    }
}

struct Event {
    uint64_t timestamp;
    int delta;
    uint32_t smid;
};

static int peak_concurrent_sms(const std::vector<CtaRecord>& records) {
    std::vector<Event> events;
    events.reserve(records.size() * 2);
    for (const auto& record : records) {
        events.push_back({record.start_ns, 1, record.smid});
        events.push_back({record.end_ns, -1, record.smid});
    }
    std::sort(events.begin(), events.end(), [](const Event& left, const Event& right) {
        if (left.timestamp != right.timestamp) {
            return left.timestamp < right.timestamp;
        }
        return left.delta < right.delta;
    });

    std::map<uint32_t, int> active_ctas;
    int peak = 0;
    for (const auto& event : events) {
        int& count = active_ctas[event.smid];
        count += event.delta;
        if (count == 0) {
            active_ctas.erase(event.smid);
        }
        peak = std::max(peak, static_cast<int>(active_ctas.size()));
    }
    return peak;
}

int main(int argc, char** argv) {
    const uint64_t hold_ns = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 200000;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
    constexpr int block_size_m = 64;
    constexpr int block_size_n = 128;
    constexpr int group_size_m = 32;
    constexpr int threads_per_cta = 128;  // vLLM default num_warps=4.
    const int token_counts[] = {1, 32, 64, 65, 128, 256, 512};

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("# gpu=%s sm_count=%d block_m=%d block_n=%d group_m=%d "
                "threads=%d hold_ns=%llu repeats=%d\n",
                properties.name, properties.multiProcessorCount, block_size_m,
                block_size_n, group_size_m, threads_per_cta,
                static_cast<unsigned long long>(hold_ns), repeats);
    std::printf("phase,tokens,m_tiles,n_tiles,ctas,repeat,unique_sms,"
                "peak_concurrent_sms,min_ctas_per_sm,max_ctas_per_sm\n");

    // Qwen3-30B-A3B: w13 output is 2 * 768; w2 output is hidden_size 2048.
    const std::tuple<const char*, int> phases[] = {{"w13", 1536}, {"w2", 2048}};
    for (const auto& [phase, output_width] : phases) {
        const int n_tiles = (output_width + block_size_n - 1) / block_size_n;
        for (int tokens : token_counts) {
            const int m_tiles = (tokens + block_size_m - 1) / block_size_m;
            const int ctas = m_tiles * n_tiles;
            CtaRecord* device_records = nullptr;
            CUDA_CHECK(cudaMalloc(&device_records, ctas * sizeof(CtaRecord)));
            std::vector<CtaRecord> host_records(ctas);

            for (int repeat = 0; repeat < repeats; ++repeat) {
                fused_moe_geometry_probe<<<ctas, threads_per_cta>>>(
                    device_records, m_tiles, n_tiles, group_size_m, hold_ns);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(host_records.data(), device_records,
                                      ctas * sizeof(CtaRecord), cudaMemcpyDeviceToHost));

                std::map<uint32_t, int> ctas_by_sm;
                for (const auto& record : host_records) {
                    ++ctas_by_sm[record.smid];
                }
                int min_ctas = ctas;
                int max_ctas = 0;
                for (const auto& [smid, count] : ctas_by_sm) {
                    (void)smid;
                    min_ctas = std::min(min_ctas, count);
                    max_ctas = std::max(max_ctas, count);
                }
                std::printf("%s,%d,%d,%d,%d,%d,%zu,%d,%d,%d\n", phase,
                            tokens, m_tiles, n_tiles, ctas, repeat,
                            ctas_by_sm.size(), peak_concurrent_sms(host_records),
                            min_ctas, max_ctas);
            }
            CUDA_CHECK(cudaFree(device_records));
        }
    }
    return 0;
}
