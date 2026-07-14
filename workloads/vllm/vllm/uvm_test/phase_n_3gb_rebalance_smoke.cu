#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
int uvm_driver_phase_f_register_region(uint64_t region_id,
                                       uintptr_t base,
                                       size_t length,
                                       uint32_t pool,
                                       uint32_t layer_id,
                                       uint32_t expert_id,
                                       uint32_t role,
                                       uint32_t flags);
int uvm_driver_phase_f_unregister_region(uint64_t region_id);
int uvm_driver_phase_n_set_dynamic_budgets(uint32_t gpu_id,
                                           uint64_t epoch,
                                           uint64_t managed_hbm_budget_bytes,
                                           uint64_t safety_margin_bytes,
                                           uint64_t scratch_workspace_bytes,
                                           uint64_t kv_budget_bytes,
                                           uint64_t expert_hot_budget_bytes,
                                           uint64_t expert_streaming_budget_bytes,
                                           uint32_t mode,
                                           uint32_t flags,
                                           uint64_t *values_out,
                                           size_t values_count);
int uvm_driver_phase_n_get_pool_state(uint32_t pool,
                                      uint32_t gpu_id,
                                      uint64_t *values_out,
                                      size_t values_count);
int uvm_driver_phase_n_query_region_blocks(uint64_t region_id,
                                           uint32_t gpu_id,
                                           uint64_t *values_out,
                                           size_t values_count);
int uvm_driver_phase_n_reclaim_region(uint64_t region_id,
                                      uint32_t gpu_id,
                                      uint32_t async,
                                      uint32_t flags,
                                      uint32_t max_blocks,
                                      uint64_t current_epoch,
                                      uint64_t *values_out,
                                      size_t values_count);
}

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                \
                         "CUDA error %s:%d: %s returned %s\n",                 \
                         __FILE__,                                              \
                         __LINE__,                                              \
                         #expr,                                                 \
                         cudaGetErrorString(_err));                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace {

constexpr uint32_t kPoolKv = 1;
constexpr uint32_t kPoolExpert = 2;
constexpr uint32_t kModeEnforce = 2;
constexpr uint32_t kFlagEvictable = 1u << 0;
constexpr uint32_t kFlagPrefetchable = 1u << 3;
constexpr uint32_t kFlagGpuReadOnly = 1u << 9;
constexpr uint32_t kRoleKv = 1;
constexpr uint32_t kRoleExpert = 2;
constexpr uint64_t kKvRegionBase = 0x530000000000ULL;
constexpr uint64_t kExpertRegionBase = 0x540000000000ULL;
constexpr size_t kQueryValues = 87;
constexpr size_t kReclaimValues = 79;
constexpr size_t kPoolValues = 38;
constexpr int kThreads = 256;
constexpr int kBlocks = 1024;

struct Config {
    int device = 0;
    size_t region_mib = 512;
    int expert_regions = 5;
    int kv_regions = 2;
    int async = 0;
    int low_rebalance = 0;
};

__global__ void fill_u4_kernel(uint4 *data, size_t count, unsigned int seed)
{
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
        data[i] = make_uint4((unsigned int)i ^ seed, seed + 1, seed + 2, seed + 3);
}

__global__ void read_two_ranges_kernel(const uint4 *expert,
                                       size_t expert_count,
                                       const uint4 *kv,
                                       size_t kv_count,
                                       unsigned long long *partial)
{
    __shared__ unsigned long long scratch[kThreads];
    unsigned long long acc = 0;
    const size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < expert_count; i += stride) {
        uint4 v = expert[i];
        acc += (unsigned long long)v.x;
        acc ^= (unsigned long long)v.y << 7;
        acc += (unsigned long long)v.z << 13;
        acc ^= (unsigned long long)v.w << 19;
    }

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < kv_count; i += stride) {
        uint4 v = kv[i];
        acc += (unsigned long long)v.x;
        acc ^= (unsigned long long)v.y << 5;
        acc += (unsigned long long)v.z << 11;
        acc ^= (unsigned long long)v.w << 17;
    }

    scratch[threadIdx.x] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = scratch[0];
}

bool parse_size(const char *arg, size_t *out)
{
    char *end = nullptr;
    unsigned long long value = std::strtoull(arg, &end, 0);
    if (end == arg || *end != '\0')
        return false;
    *out = static_cast<size_t>(value);
    return true;
}

bool parse_int(const char *arg, int *out)
{
    char *end = nullptr;
    long value = std::strtol(arg, &end, 0);
    if (end == arg || *end != '\0')
        return false;
    *out = static_cast<int>(value);
    return true;
}

void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--device N] [--region-mib N] "
                 "[--expert-regions N] [--kv-regions N] [--async 0|1] "
                 "[--low-rebalance 0|1]\n",
                 argv0);
}

uintptr_t align_up(uintptr_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
}

int run_step(const char *name, int result)
{
    std::printf("phase_n_3gb_step name=%s result=%d\n", name, result);
    std::fflush(stdout);
    return result < 0 ? 1 : 0;
}

float elapsed_event(cudaEvent_t start, cudaEvent_t end)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

float prefetch_region(void *ptr, size_t bytes, int dst, cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, dst, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_event(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

float fill_region(uintptr_t base, size_t bytes, unsigned int seed, cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    fill_u4_kernel<<<kBlocks, kThreads, 0, stream>>>(
        reinterpret_cast<uint4 *>(base), bytes / sizeof(uint4), seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_event(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

float read_active(uintptr_t expert_base,
                  size_t expert_bytes,
                  uintptr_t kv_base,
                  size_t kv_bytes,
                  unsigned long long *partial,
                  cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    read_two_ranges_kernel<<<kBlocks, kThreads, 0, stream>>>(
        reinterpret_cast<uint4 *>(expert_base),
        expert_bytes / sizeof(uint4),
        reinterpret_cast<uint4 *>(kv_base),
        kv_bytes / sizeof(uint4),
        partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_event(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

void print_query(const char *label, const uint64_t *q)
{
    std::printf("phase_n_3gb_query label=%s region_id=%llu pool=%llu object=%llu flags=0x%llx logical=%llu populated=%llu resident=%llu pending=%llu resident_bytes=%llu block0_state=0x%llx\n",
                label,
                static_cast<unsigned long long>(q[0]),
                static_cast<unsigned long long>(q[3]),
                static_cast<unsigned long long>(q[5]),
                static_cast<unsigned long long>(q[7]),
                static_cast<unsigned long long>(q[11]),
                static_cast<unsigned long long>(q[12]),
                static_cast<unsigned long long>(q[13]),
                static_cast<unsigned long long>(q[14]),
                static_cast<unsigned long long>(q[17]),
                static_cast<unsigned long long>(q[71]));
}

void print_reclaim(const char *label, const uint64_t *r)
{
    std::printf("phase_n_3gb_reclaim label=%s region_id=%llu pool=%llu resident_before=%llu resident_after=%llu pending=%llu attempted=%llu success=%llu failure=%llu reclaimed_bytes=%llu status=%llu\n",
                label,
                static_cast<unsigned long long>(r[0]),
                static_cast<unsigned long long>(r[1]),
                static_cast<unsigned long long>(r[4]),
                static_cast<unsigned long long>(r[5]),
                static_cast<unsigned long long>(r[6]),
                static_cast<unsigned long long>(r[7]),
                static_cast<unsigned long long>(r[8]),
                static_cast<unsigned long long>(r[9]),
                static_cast<unsigned long long>(r[12]),
                static_cast<unsigned long long>(r[14]));
}

void print_pool(const char *label, const uint64_t *kv, const uint64_t *expert)
{
    std::printf("phase_n_3gb_pool label=%s kv_budget=%llu kv_resident=%llu kv_pending=%llu kv_over=%llu expert_budget=%llu expert_resident=%llu expert_pending=%llu expert_over=%llu total_resident=%llu\n",
                label,
                static_cast<unsigned long long>(kv[0]),
                static_cast<unsigned long long>(kv[1]),
                static_cast<unsigned long long>(kv[36]),
                static_cast<unsigned long long>(kv[6]),
                static_cast<unsigned long long>(expert[0]),
                static_cast<unsigned long long>(expert[1]),
                static_cast<unsigned long long>(expert[36]),
                static_cast<unsigned long long>(expert[6]),
                static_cast<unsigned long long>(kv[1] + expert[1]));
}

} // namespace

int main(int argc, char **argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.device))
                return 2;
        }
        else if (std::strcmp(argv[i], "--region-mib") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &cfg.region_mib))
                return 2;
        }
        else if (std::strcmp(argv[i], "--expert-regions") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.expert_regions))
                return 2;
        }
        else if (std::strcmp(argv[i], "--kv-regions") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.kv_regions))
                return 2;
        }
        else if (std::strcmp(argv[i], "--async") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.async))
                return 2;
        }
        else if (std::strcmp(argv[i], "--low-rebalance") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.low_rebalance))
                return 2;
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
        else {
            usage(argv[0]);
            return 2;
        }
    }

    if (cfg.region_mib == 0 || cfg.expert_regions < 2 || cfg.kv_regions < 2) {
        usage(argv[0]);
        return 2;
    }

    const size_t region_bytes = cfg.region_mib * 1024ull * 1024ull;
    const size_t expert_span = (size_t)cfg.expert_regions * region_bytes;
    const size_t kv_span = (size_t)cfg.kv_regions * region_bytes;
    const size_t expert_alloc = expert_span + region_bytes;
    const size_t kv_alloc = kv_span + region_bytes;
    const uint64_t total_budget = 3ull * 1024ull * 1024ull * 1024ull;
    const uint64_t kv_initial_budget = region_bytes;
    const uint64_t expert_initial_budget = total_budget - kv_initial_budget;
    const uint64_t kv_final_budget = kv_span;
    const uint64_t expert_final_budget = total_budget - kv_final_budget;

    int failed = 0;
    void *expert_raw = nullptr;
    void *kv_raw = nullptr;
    uintptr_t expert_base = 0;
    uintptr_t kv_base = 0;
    cudaStream_t stream = nullptr;
    unsigned long long *partial = nullptr;
    std::vector<uint64_t> region_ids;

    uint64_t dyn[8] = {};
    uint64_t dyn_after[8] = {};
    uint64_t dyn_low[8] = {};
    uint64_t kv_pool_initial[kPoolValues] = {};
    uint64_t expert_pool_initial[kPoolValues] = {};
    uint64_t kv_pool_after[kPoolValues] = {};
    uint64_t expert_pool_after[kPoolValues] = {};
    uint64_t kv_pool_low[kPoolValues] = {};
    uint64_t expert_pool_low[kPoolValues] = {};
    uint64_t expert_victim_before[kQueryValues] = {};
    uint64_t expert_victim_after[kQueryValues] = {};
    uint64_t kv_tail_before[kQueryValues] = {};
    uint64_t kv_tail_after[kQueryValues] = {};
    uint64_t reclaim[kReclaimValues] = {};
    uint64_t kv_reclaim_low[kReclaimValues] = {};

    std::printf("phase_n_3gb_begin device=%d region_mib=%zu expert_regions=%d kv_regions=%d total_budget=%llu\n",
                cfg.device,
                cfg.region_mib,
                cfg.expert_regions,
                cfg.kv_regions,
                static_cast<unsigned long long>(total_budget));

    CUDA_CHECK(cudaSetDevice(cfg.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMallocManaged(&expert_raw, expert_alloc, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&kv_raw, kv_alloc, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMalloc(&partial, sizeof(unsigned long long) * kBlocks));

    expert_base = align_up(reinterpret_cast<uintptr_t>(expert_raw), region_bytes);
    kv_base = align_up(reinterpret_cast<uintptr_t>(kv_raw), region_bytes);
    if (expert_base + expert_span > reinterpret_cast<uintptr_t>(expert_raw) + expert_alloc ||
        kv_base + kv_span > reinterpret_cast<uintptr_t>(kv_raw) + kv_alloc) {
        std::fprintf(stderr, "aligned managed subrange overflow\n");
        failed = 1;
        goto cleanup;
    }

    std::printf("phase_n_3gb_alloc expert_raw=0x%llx expert_base=0x%llx expert_span=%zu kv_raw=0x%llx kv_base=0x%llx kv_span=%zu\n",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(expert_raw)),
                static_cast<unsigned long long>(expert_base),
                expert_span,
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(kv_raw)),
                static_cast<unsigned long long>(kv_base),
                kv_span);

    for (int i = 0; !failed && i < cfg.expert_regions; ++i) {
        uint64_t id = kExpertRegionBase + (uint64_t)i;
        region_ids.push_back(id);
        failed = run_step("register-expert",
                          uvm_driver_phase_f_register_region(id,
                                                             expert_base + (size_t)i * region_bytes,
                                                             region_bytes,
                                                             kPoolExpert,
                                                             0,
                                                             (uint32_t)i,
                                                             kRoleExpert,
                                                             kFlagEvictable | kFlagPrefetchable | kFlagGpuReadOnly));
    }

    for (int i = 0; !failed && i < cfg.kv_regions; ++i) {
        uint64_t id = kKvRegionBase + (uint64_t)i;
        region_ids.push_back(id);
        failed = run_step("register-kv",
                          uvm_driver_phase_f_register_region(id,
                                                             kv_base + (size_t)i * region_bytes,
                                                             region_bytes,
                                                             kPoolKv,
                                                             0,
                                                             (uint32_t)i,
                                                             kRoleKv,
                                                             kFlagEvictable | kFlagPrefetchable));
    }

    if (!failed) {
        failed = run_step("set-initial-budget",
                          uvm_driver_phase_n_set_dynamic_budgets(0,
                                                                 1,
                                                                 total_budget,
                                                                 0,
                                                                 0,
                                                                 kv_initial_budget,
                                                                 expert_initial_budget,
                                                                 0,
                                                                 kModeEnforce,
                                                                 0,
                                                                 dyn,
                                                                 8));
    }

    if (!failed) {
        CUDA_CHECK(cudaMemAdvise(reinterpret_cast<void *>(expert_base),
                                 expert_span,
                                 cudaMemAdviseSetReadMostly,
                                 cfg.device));
        CUDA_CHECK(cudaMemAdvise(reinterpret_cast<void *>(expert_base),
                                 expert_span,
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
        std::printf("phase_n_3gb_readmostly expert_bytes=%zu status=0\n", expert_span);
    }

    if (!failed) {
        float expert_prefetch_ms = prefetch_region(reinterpret_cast<void *>(expert_base),
                                                  expert_span,
                                                  cfg.device,
                                                  stream);
        float kv_head_prefetch_ms = prefetch_region(reinterpret_cast<void *>(kv_base),
                                                   region_bytes,
                                                   cfg.device,
                                                   stream);
        float kv_tail_cpu_ms = prefetch_region(reinterpret_cast<void *>(kv_base + region_bytes),
                                              kv_span - region_bytes,
                                              cudaCpuDeviceId,
                                              stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("phase_n_3gb_initial_residency expert_prefetch_ms=%.3f kv_head_prefetch_ms=%.3f kv_tail_cpu_ms=%.3f\n",
                    expert_prefetch_ms,
                    kv_head_prefetch_ms,
                    kv_tail_cpu_ms);
    }

    if (!failed) {
        for (int i = 0; i < cfg.expert_regions; ++i)
            (void)fill_region(expert_base + (size_t)i * region_bytes, region_bytes, 0xE000u + (unsigned)i, stream);
        (void)fill_region(kv_base, region_bytes, 0xA000u, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (!failed)
        failed = run_step("query-expert-victim-before", uvm_driver_phase_n_query_region_blocks(kExpertRegionBase, 0, expert_victim_before, kQueryValues));
    if (!failed)
        failed = run_step("query-kv-tail-before", uvm_driver_phase_n_query_region_blocks(kKvRegionBase + 1, 0, kv_tail_before, kQueryValues));
    if (!failed) {
        print_query("expert0_before_rebalance", expert_victim_before);
        print_query("kv_tail_before_rebalance", kv_tail_before);
    }

    if (!failed)
        failed = run_step("pool-kv-initial", uvm_driver_phase_n_get_pool_state(kPoolKv, 0, kv_pool_initial, kPoolValues));
    if (!failed)
        failed = run_step("pool-expert-initial", uvm_driver_phase_n_get_pool_state(kPoolExpert, 0, expert_pool_initial, kPoolValues));
    if (!failed)
        print_pool("initial", kv_pool_initial, expert_pool_initial);

    if (!failed) {
        failed = run_step("set-final-budget",
                          uvm_driver_phase_n_set_dynamic_budgets(0,
                                                                 2,
                                                                 total_budget,
                                                                 0,
                                                                 0,
                                                                 kv_final_budget,
                                                                 expert_final_budget,
                                                                 0,
                                                                 kModeEnforce,
                                                                 0,
                                                                 dyn_after,
                                                                 8));
    }

    if (!failed) {
        failed = run_step("reclaim-expert0",
                          uvm_driver_phase_n_reclaim_region(kExpertRegionBase,
                                                            0,
                                                            cfg.async ? 1 : 0,
                                                            0,
                                                            0,
                                                            3,
                                                            reclaim,
                                                            kReclaimValues));
    }
    if (!failed)
        print_reclaim("expert0", reclaim);

    if (!failed) {
        float kv_tail_promote_ms = prefetch_region(reinterpret_cast<void *>(kv_base + region_bytes),
                                                  kv_span - region_bytes,
                                                  cfg.device,
                                                  stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("phase_n_3gb_kv_tail_promote bytes=%zu ms=%.3f\n",
                    kv_span - region_bytes,
                    kv_tail_promote_ms);
    }

    if (!failed)
        failed = run_step("query-expert-victim-after", uvm_driver_phase_n_query_region_blocks(kExpertRegionBase, 0, expert_victim_after, kQueryValues));
    if (!failed)
        failed = run_step("query-kv-tail-after", uvm_driver_phase_n_query_region_blocks(kKvRegionBase + 1, 0, kv_tail_after, kQueryValues));
    if (!failed) {
        print_query("expert0_after_rebalance", expert_victim_after);
        print_query("kv_tail_after_rebalance", kv_tail_after);
    }

    if (!failed)
        failed = run_step("pool-kv-after", uvm_driver_phase_n_get_pool_state(kPoolKv, 0, kv_pool_after, kPoolValues));
    if (!failed)
        failed = run_step("pool-expert-after", uvm_driver_phase_n_get_pool_state(kPoolExpert, 0, expert_pool_after, kPoolValues));
    if (!failed)
        print_pool("after_rebalance", kv_pool_after, expert_pool_after);

    if (!failed && cfg.low_rebalance) {
        failed = run_step("set-low-budget",
                          uvm_driver_phase_n_set_dynamic_budgets(0,
                                                                 3,
                                                                 total_budget,
                                                                 0,
                                                                 0,
                                                                 kv_initial_budget,
                                                                 expert_initial_budget,
                                                                 0,
                                                                 kModeEnforce,
                                                                 0,
                                                                 dyn_low,
                                                                 8));
    }

    if (!failed && cfg.low_rebalance) {
        failed = run_step("reclaim-kv-tail-low",
                          uvm_driver_phase_n_reclaim_region(kKvRegionBase + 1,
                                                            0,
                                                            cfg.async ? 1 : 0,
                                                            0,
                                                            0,
                                                            4,
                                                            kv_reclaim_low,
                                                            kReclaimValues));
    }
    if (!failed && cfg.low_rebalance)
        print_reclaim("kv_tail_low", kv_reclaim_low);

    if (!failed && cfg.low_rebalance)
        failed = run_step("pool-kv-low", uvm_driver_phase_n_get_pool_state(kPoolKv, 0, kv_pool_low, kPoolValues));
    if (!failed && cfg.low_rebalance)
        failed = run_step("pool-expert-low", uvm_driver_phase_n_get_pool_state(kPoolExpert, 0, expert_pool_low, kPoolValues));
    if (!failed && cfg.low_rebalance)
        print_pool("low_rebalance", kv_pool_low, expert_pool_low);

    if (!failed) {
        float compute_ms = read_active(expert_base + region_bytes,
                                       expert_span - region_bytes,
                                       kv_base,
                                       kv_span,
                                       partial,
                                       stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("phase_n_3gb_compute active_expert_bytes=%zu active_kv_bytes=%zu compute_ms=%.3f\n",
                    expert_span - region_bytes,
                    kv_span,
                    compute_ms);
    }

    {
        const uint64_t expected_region_bytes = static_cast<uint64_t>(region_bytes);
        const bool initial_budget_ok = kv_pool_initial[0] == kv_initial_budget &&
                                       expert_pool_initial[0] == expert_initial_budget;
        const bool final_budget_ok = kv_pool_after[0] == kv_final_budget &&
                                     expert_pool_after[0] == expert_final_budget;
        const bool expert_before_ok = expert_victim_before[13] >= 1 &&
                                      expert_victim_before[17] >= expected_region_bytes;
        const bool kv_tail_before_ok = kv_tail_before[13] == 0;
        const bool expert_reclaim_ok = reclaim[4] >= 1 &&
                                       reclaim[5] == 0 &&
                                       reclaim[8] >= 1 &&
                                       reclaim[9] == 0 &&
                                       reclaim[12] >= expected_region_bytes &&
                                       expert_victim_after[13] == 0;
        const bool kv_tail_after_ok = kv_tail_after[13] >= 1 &&
                                      kv_tail_after[17] >= expected_region_bytes;
        const bool total_budget_ok = kv_pool_after[1] + expert_pool_after[1] <= total_budget;
        const bool total_ok = initial_budget_ok &&
                              final_budget_ok &&
                              expert_before_ok &&
                              kv_tail_before_ok &&
                              expert_reclaim_ok &&
                              kv_tail_after_ok &&
                              total_budget_ok &&
                              !failed;

        std::printf("phase_n_3gb_final initial_budget_ok=%u final_budget_ok=%u expert_before_ok=%u kv_tail_before_ok=%u expert_reclaim_ok=%u kv_tail_after_ok=%u total_budget_ok=%u total_ok=%u kv_resident_after=%llu expert_resident_after=%llu total_resident_after=%llu\n",
                    initial_budget_ok ? 1 : 0,
                    final_budget_ok ? 1 : 0,
                    expert_before_ok ? 1 : 0,
                    kv_tail_before_ok ? 1 : 0,
                    expert_reclaim_ok ? 1 : 0,
                    kv_tail_after_ok ? 1 : 0,
                    total_budget_ok ? 1 : 0,
                    total_ok ? 1 : 0,
                    static_cast<unsigned long long>(kv_pool_after[1]),
                    static_cast<unsigned long long>(expert_pool_after[1]),
                    static_cast<unsigned long long>(kv_pool_after[1] + expert_pool_after[1]));
        if (cfg.low_rebalance) {
            const bool low_budget_ok = kv_pool_low[0] == kv_initial_budget &&
                                       expert_pool_low[0] == expert_initial_budget;
            const bool kv_low_after_ok = kv_pool_low[1] < kv_pool_after[1];
            const bool u2_total_ok = total_ok && low_budget_ok && kv_low_after_ok;
            std::printf("phase_n_3gb_u2_final low_budget_ok=%u kv_low_after_ok=%u total_ok=%u kv_resident_initial=%llu kv_resident_high=%llu kv_resident_low=%llu expert_resident_initial=%llu expert_resident_high=%llu expert_resident_low=%llu\n",
                        low_budget_ok ? 1 : 0,
                        kv_low_after_ok ? 1 : 0,
                        u2_total_ok ? 1 : 0,
                        static_cast<unsigned long long>(kv_pool_initial[1]),
                        static_cast<unsigned long long>(kv_pool_after[1]),
                        static_cast<unsigned long long>(kv_pool_low[1]),
                        static_cast<unsigned long long>(expert_pool_initial[1]),
                        static_cast<unsigned long long>(expert_pool_after[1]),
                        static_cast<unsigned long long>(expert_pool_low[1]));
            if (!u2_total_ok)
                failed = 1;
        }
        if (!total_ok)
            failed = 1;
    }

cleanup:
    for (uint64_t id : region_ids)
        (void)uvm_driver_phase_f_unregister_region(id);
    if (partial)
        cudaFree(partial);
    if (kv_raw)
        cudaFree(kv_raw);
    if (expert_raw)
        cudaFree(expert_raw);
    if (stream)
        cudaStreamDestroy(stream);
    return failed ? 1 : 0;
}
