#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

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

static constexpr int kThreads = 256;
static constexpr int kFillBlocks = 2048;
static constexpr int kReadBlocks = 2048;

__global__ void fill_u4_kernel(uint4 *data, size_t count, unsigned int seed)
{
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
        data[i] = make_uint4((unsigned int)i ^ seed, seed + 1, seed + 2, seed + 3);
}

__global__ void vllm_like_decode_kernel(const uint4 *kv_head,
                                        size_t kv_count,
                                        const uint4 *expert,
                                        size_t expert_count,
                                        unsigned long long *partial)
{
    __shared__ unsigned long long scratch[kThreads];
    unsigned long long acc = 0;
    const size_t stride = (size_t)blockDim.x * gridDim.x;

    // Simulate the compute phase after residency has already been established:
    // first read the active KV window, then read the active expert weights.
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < kv_count; i += stride) {
        uint4 v = kv_head[i];
        acc += (unsigned long long)v.x;
        acc ^= (unsigned long long)v.y << 5;
        acc += (unsigned long long)v.z << 11;
        acc ^= (unsigned long long)v.w << 17;
    }

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < expert_count; i += stride) {
        uint4 v = expert[i];
        acc += (unsigned long long)v.x;
        acc ^= (unsigned long long)v.y << 7;
        acc += (unsigned long long)v.z << 13;
        acc ^= (unsigned long long)v.w << 19;
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

struct Args {
    int device = 0;
    int repeats = 5;
    int warmups = 2;
    size_t expert_mib = 1024;
    size_t kv_total_mib = 1536;
    size_t kv_active_mib = 512;
    bool skip_native = false;
    bool skip_uvm = false;
};

struct Stats {
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double mean_ms = 0.0;
    double p90_ms = 0.0;
    double max_ms = 0.0;
};

static void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "usage: %s [--device N] [--expert-mib N] [--kv-total-mib N] "
                 "[--kv-active-mib N] [--repeats N] [--warmups N] "
                 "[--skip-native] [--skip-uvm]\n",
                 argv0);
}

static Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--expert-mib") == 0 && i + 1 < argc)
            args.expert_mib = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--kv-total-mib") == 0 && i + 1 < argc)
            args.kv_total_mib = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--kv-active-mib") == 0 && i + 1 < argc)
            args.kv_active_mib = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            args.repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmups") == 0 && i + 1 < argc)
            args.warmups = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--skip-native") == 0)
            args.skip_native = true;
        else if (std::strcmp(argv[i], "--skip-uvm") == 0)
            args.skip_uvm = true;
        else {
            usage(argv[0]);
            std::exit(2);
        }
    }

    if (args.device < 0 ||
        args.repeats <= 0 ||
        args.warmups < 0 ||
        args.expert_mib == 0 ||
        args.kv_total_mib == 0 ||
        args.kv_active_mib == 0 ||
        args.kv_active_mib > args.kv_total_mib ||
        (args.skip_native && args.skip_uvm)) {
        usage(argv[0]);
        std::exit(2);
    }
    return args;
}

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static size_t mib_to_bytes(size_t mib)
{
    return mib * 1024ull * 1024ull;
}

static size_t align_u4(size_t bytes)
{
    return (bytes / sizeof(uint4)) * sizeof(uint4);
}

static float elapsed_event(cudaEvent_t start, cudaEvent_t end)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

static Stats stats_of(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    Stats s;
    s.min_ms = values.front();
    s.p50_ms = values[values.size() / 2];
    s.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    s.p90_ms = values[(size_t)((values.size() - 1) * 0.90)];
    s.max_ms = values.back();
    return s;
}

static double gib_per_s(size_t bytes, double ms)
{
    if (ms <= 0.0)
        return 0.0;
    return (double)bytes / (1024.0 * 1024.0 * 1024.0) / (ms / 1000.0);
}

static void print_mem(const char *label)
{
    size_t free_b = 0;
    size_t total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("mem label=%s free_mib=%.1f used_mib=%.1f total_mib=%.1f\n",
                label,
                (double)free_b / (1024.0 * 1024.0),
                (double)(total_b - free_b) / (1024.0 * 1024.0),
                (double)total_b / (1024.0 * 1024.0));
}

static float time_fill(uint4 *ptr, size_t bytes, unsigned int seed, cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    fill_u4_kernel<<<kFillBlocks, kThreads, 0, stream>>>(ptr, bytes / sizeof(uint4), seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_event(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static float time_prefetch(void *ptr, size_t bytes, int dst, cudaStream_t stream)
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

static float time_compute(uint4 *kv_head,
                          size_t kv_active_bytes,
                          uint4 *expert,
                          size_t expert_bytes,
                          unsigned long long *partial,
                          cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    vllm_like_decode_kernel<<<kReadBlocks, kThreads, 0, stream>>>(
        kv_head,
        kv_active_bytes / sizeof(uint4),
        expert,
        expert_bytes / sizeof(uint4),
        partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_event(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static void run_native_case(const Args &args,
                            size_t expert_bytes,
                            size_t kv_total_bytes,
                            size_t kv_active_bytes)
{
    std::printf("case_begin name=native_all_cudaMalloc expert_mib=%zu kv_total_mib=%zu kv_active_mib=%zu\n",
                (size_t)(expert_bytes / (1024ull * 1024ull)),
                (size_t)(kv_total_bytes / (1024ull * 1024ull)),
                (size_t)(kv_active_bytes / (1024ull * 1024ull)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    print_mem("native_before_alloc");

    void *expert_raw = nullptr;
    void *kv_raw = nullptr;
    double alloc_start = now_ms();
    CUDA_CHECK(cudaMalloc(&expert_raw, expert_bytes));
    CUDA_CHECK(cudaMalloc(&kv_raw, kv_total_bytes));
    double alloc_ms = now_ms() - alloc_start;
    print_mem("native_after_alloc");

    unsigned long long *partial = nullptr;
    CUDA_CHECK(cudaMalloc(&partial, sizeof(unsigned long long) * kReadBlocks));
    float expert_init_ms = time_fill(static_cast<uint4 *>(expert_raw), expert_bytes, 0xE500u, stream);
    float kv_init_ms = time_fill(static_cast<uint4 *>(kv_raw), kv_total_bytes, 0xA500u, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    print_mem("native_after_init");

    std::vector<double> compute_ms;
    for (int iter = 0; iter < args.warmups + args.repeats; ++iter) {
        float ms = time_compute(static_cast<uint4 *>(kv_raw),
                                kv_active_bytes,
                                static_cast<uint4 *>(expert_raw),
                                expert_bytes,
                                partial,
                                stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (iter >= args.warmups) {
            compute_ms.push_back(ms);
            std::printf("sample case=native iter=%d compute_only_ms=%.3f compute_gib_s=%.3f\n",
                        iter - args.warmups,
                        ms,
                        gib_per_s(expert_bytes + kv_active_bytes, ms));
        }
    }

    Stats c = stats_of(compute_ms);
    std::printf("summary case=native alloc_ms=%.3f expert_init_ms=%.3f kv_init_ms=%.3f "
                "compute_only_p50_ms=%.3f compute_only_mean_ms=%.3f compute_only_p90_ms=%.3f "
                "compute_only_p50_gib_s=%.3f\n",
                alloc_ms,
                expert_init_ms,
                kv_init_ms,
                c.p50_ms,
                c.mean_ms,
                c.p90_ms,
                gib_per_s(expert_bytes + kv_active_bytes, c.p50_ms));

    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaFree(kv_raw));
    CUDA_CHECK(cudaFree(expert_raw));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("native_after_free");
    std::printf("case_end name=native_all_cudaMalloc\n");
}

static void run_uvm_case(const Args &args,
                         size_t expert_bytes,
                         size_t kv_total_bytes,
                         size_t kv_active_bytes)
{
    std::printf("case_begin name=uvm_expert_uvm_kv_head_resident expert_mib=%zu kv_total_mib=%zu kv_active_mib=%zu\n",
                (size_t)(expert_bytes / (1024ull * 1024ull)),
                (size_t)(kv_total_bytes / (1024ull * 1024ull)),
                (size_t)(kv_active_bytes / (1024ull * 1024ull)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    print_mem("uvm_before_alloc");

    void *expert_raw = nullptr;
    void *kv_raw = nullptr;
    double alloc_start = now_ms();
    CUDA_CHECK(cudaMallocManaged(&expert_raw, expert_bytes, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&kv_raw, kv_total_bytes, cudaMemAttachGlobal));
    double alloc_ms = now_ms() - alloc_start;
    print_mem("uvm_after_alloc");

    unsigned long long *partial = nullptr;
    CUDA_CHECK(cudaMalloc(&partial, sizeof(unsigned long long) * kReadBlocks));

    float expert_first_gpu_ms = time_prefetch(expert_raw, expert_bytes, args.device, stream);
    float kv_first_gpu_ms = time_prefetch(kv_raw, kv_total_bytes, args.device, stream);
    float expert_init_ms = time_fill(static_cast<uint4 *>(expert_raw), expert_bytes, 0xE500u, stream);
    float kv_init_ms = time_fill(static_cast<uint4 *>(kv_raw), kv_total_bytes, 0xA500u, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    print_mem("uvm_after_init");

    float expert_initial_cpu_ms = time_prefetch(expert_raw, expert_bytes, cudaCpuDeviceId, stream);
    float kv_tail_initial_cpu_ms = 0.0f;
    if (kv_total_bytes > kv_active_bytes) {
        void *kv_tail = static_cast<void *>(static_cast<char *>(kv_raw) + kv_active_bytes);
        kv_tail_initial_cpu_ms = time_prefetch(kv_tail, kv_total_bytes - kv_active_bytes, cudaCpuDeviceId, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    print_mem("uvm_after_initial_offload");

    std::vector<double> expert_promote_ms;
    std::vector<double> kv_head_promote_ms;
    std::vector<double> compute_ms;
    std::vector<double> promote_plus_compute_ms;

    for (int iter = 0; iter < args.warmups + args.repeats; ++iter) {
        if (iter > 0) {
            CUDA_CHECK(cudaMemPrefetchAsync(expert_raw, expert_bytes, cudaCpuDeviceId, stream));
            if (kv_total_bytes > kv_active_bytes) {
                void *kv_tail = static_cast<void *>(static_cast<char *>(kv_raw) + kv_active_bytes);
                CUDA_CHECK(cudaMemPrefetchAsync(kv_tail, kv_total_bytes - kv_active_bytes, cudaCpuDeviceId, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        double wall_start = now_ms();
        float e_ms = time_prefetch(expert_raw, expert_bytes, args.device, stream);
        float kv_ms = time_prefetch(kv_raw, kv_active_bytes, args.device, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double compute_start = now_ms();
        float c_ms = time_compute(static_cast<uint4 *>(kv_raw),
                                  kv_active_bytes,
                                  static_cast<uint4 *>(expert_raw),
                                  expert_bytes,
                                  partial,
                                  stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double compute_wall_ms = now_ms() - compute_start;
        double total_wall_ms = now_ms() - wall_start;

        if (iter >= args.warmups) {
            expert_promote_ms.push_back(e_ms);
            kv_head_promote_ms.push_back(kv_ms);
            compute_ms.push_back(c_ms);
            promote_plus_compute_ms.push_back(total_wall_ms);
            std::printf("sample case=uvm iter=%d expert_promote_ms=%.3f kv_head_promote_ms=%.3f "
                        "compute_only_ms=%.3f compute_only_wall_ms=%.3f compute_gib_s=%.3f "
                        "promote_plus_compute_wall_ms=%.3f\n",
                        iter - args.warmups,
                        e_ms,
                        kv_ms,
                        c_ms,
                        compute_wall_ms,
                        gib_per_s(expert_bytes + kv_active_bytes, c_ms),
                        total_wall_ms);
        }
    }

    Stats ep = stats_of(expert_promote_ms);
    Stats kp = stats_of(kv_head_promote_ms);
    Stats c = stats_of(compute_ms);
    Stats e2e = stats_of(promote_plus_compute_ms);
    std::printf("summary case=uvm alloc_ms=%.3f expert_first_gpu_ms=%.3f kv_first_gpu_ms=%.3f "
                "expert_init_ms=%.3f kv_init_ms=%.3f expert_initial_cpu_ms=%.3f kv_tail_initial_cpu_ms=%.3f "
                "expert_promote_p50_ms=%.3f kv_head_promote_p50_ms=%.3f "
                "compute_only_p50_ms=%.3f compute_only_mean_ms=%.3f compute_only_p90_ms=%.3f "
                "compute_only_p50_gib_s=%.3f promote_plus_compute_p50_ms=%.3f promote_plus_compute_mean_ms=%.3f\n",
                alloc_ms,
                expert_first_gpu_ms,
                kv_first_gpu_ms,
                expert_init_ms,
                kv_init_ms,
                expert_initial_cpu_ms,
                kv_tail_initial_cpu_ms,
                ep.p50_ms,
                kp.p50_ms,
                c.p50_ms,
                c.mean_ms,
                c.p90_ms,
                gib_per_s(expert_bytes + kv_active_bytes, c.p50_ms),
                e2e.p50_ms,
                e2e.mean_ms);

    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaFree(kv_raw));
    CUDA_CHECK(cudaFree(expert_raw));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("uvm_after_free");
    std::printf("case_end name=uvm_expert_uvm_kv_head_resident\n");
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    const size_t expert_bytes = align_u4(mib_to_bytes(args.expert_mib));
    const size_t kv_total_bytes = align_u4(mib_to_bytes(args.kv_total_mib));
    const size_t kv_active_bytes = align_u4(mib_to_bytes(args.kv_active_mib));

    std::printf("config device=%d name=%s expert_mib=%zu kv_total_mib=%zu kv_active_mib=%zu repeats=%d warmups=%d\n",
                args.device,
                prop.name,
                (size_t)(expert_bytes / (1024ull * 1024ull)),
                (size_t)(kv_total_bytes / (1024ull * 1024ull)),
                (size_t)(kv_active_bytes / (1024ull * 1024ull)),
                args.repeats,
                args.warmups);
    print_mem("start");

    if (!args.skip_native)
        run_native_case(args, expert_bytes, kv_total_bytes, kv_active_bytes);
    if (!args.skip_uvm)
        run_uvm_case(args, expert_bytes, kv_total_bytes, kv_active_bytes);

    print_mem("end");
    return 0;
}
