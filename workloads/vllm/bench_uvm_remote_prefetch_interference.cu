#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
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
static constexpr int kBlocks = 1024;

__global__ void read_u4_kernel(const uint4 *data, size_t count, unsigned long long *partial)
{
    __shared__ unsigned long long scratch[kThreads];
    unsigned long long acc = 0;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        uint4 value = data[i];
        acc += (unsigned long long)value.x;
        acc += (unsigned long long)value.y;
        acc += (unsigned long long)value.z;
        acc += (unsigned long long)value.w;
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

struct Stats {
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double mean_ms = 0.0;
    double max_ms = 0.0;
};

struct Args {
    int device = 0;
    int repeats = 7;
    size_t bytes = 144ull * 1024ull * 1024ull;
};

struct PinnedMapped {
    uint4 *host = nullptr;
    uint4 *device = nullptr;
};

static Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            args.device = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            args.repeats = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--bytes-mib") == 0 && i + 1 < argc) {
            args.bytes = (size_t)std::strtoull(argv[++i], nullptr, 10) * 1024ull * 1024ull;
        }
        else {
            std::fprintf(stderr, "usage: %s [--device N] [--repeats N] [--bytes-mib N]\n", argv[0]);
            std::exit(2);
        }
    }

    args.bytes = (args.bytes / sizeof(uint4)) * sizeof(uint4);
    if (args.bytes == 0 || args.repeats <= 0) {
        std::fprintf(stderr, "invalid bytes or repeats\n");
        std::exit(2);
    }
    return args;
}

static Stats stats_of(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    Stats s;
    s.min_ms = values.front();
    s.max_ms = values.back();
    s.p50_ms = values[values.size() / 2];
    s.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return s;
}

static double gbps(size_t bytes, double ms)
{
    if (ms <= 0.0)
        return 0.0;
    return (double)bytes / (ms / 1000.0) / 1.0e9;
}

static void fill_cpu(uint4 *ptr, size_t count, unsigned int seed)
{
    for (size_t i = 0; i < count; ++i)
        ptr[i] = make_uint4((unsigned int)i ^ seed, seed + 1, seed + 2, seed + 3);
}

static void setup_cpu_resident(uint4 *ptr, size_t count, size_t bytes, cudaStream_t stream, unsigned int seed)
{
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fill_cpu(ptr, count, seed);
}

static void set_remote_advise(uint4 *ptr, size_t bytes, int device)
{
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device));
}

static PinnedMapped alloc_pinned_mapped(size_t bytes)
{
    PinnedMapped mapped;
    void *host = nullptr;
    void *device = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
    mapped.host = static_cast<uint4 *>(host);
    mapped.device = static_cast<uint4 *>(device);
    return mapped;
}

static float elapsed_between(cudaEvent_t start, cudaEvent_t end)
{
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

static float time_prefetch_gpu(uint4 *ptr, size_t bytes, int device, cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    float ms = elapsed_between(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static float time_h2d_copy(uint4 *dst, const uint4 *src, size_t bytes, cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    float ms = elapsed_between(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static float time_read_kernel(uint4 *ptr,
                              size_t count,
                              unsigned long long *partial,
                              cudaStream_t stream,
                              cudaEvent_t *start_out = nullptr,
                              cudaEvent_t *end_out = nullptr)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    read_u4_kernel<<<kBlocks, kThreads, 0, stream>>>(ptr, count, partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));

    if (start_out && end_out) {
        *start_out = start;
        *end_out = end;
        return 0.0f;
    }

    float ms = elapsed_between(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static uint4 *alloc_managed(size_t bytes)
{
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
    return static_cast<uint4 *>(ptr);
}

static void print_stats(const char *name, const Stats &s, size_t bytes)
{
    std::printf("  \"%s\": {\"min_ms\": %.6f, \"p50_ms\": %.6f, \"mean_ms\": %.6f, "
                "\"max_ms\": %.6f, \"p50_gbps\": %.3f}",
                name,
                s.min_ms,
                s.p50_ms,
                s.mean_ms,
                s.max_ms,
                gbps(bytes, s.p50_ms));
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flags_err != cudaSuccess && flags_err != cudaErrorSetOnActiveProcess) {
        std::fprintf(stderr, "cudaSetDeviceFlags(cudaDeviceMapHost) failed: %s\n", cudaGetErrorString(flags_err));
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    cudaStream_t setup_stream, read_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&setup_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&read_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));

    unsigned long long *partial = nullptr;
    CUDA_CHECK(cudaMalloc(&partial, kBlocks * sizeof(*partial)));

    const size_t count = args.bytes / sizeof(uint4);

    uint4 *warm = alloc_managed(4 * 1024 * 1024);
    setup_cpu_resident(warm, (4 * 1024 * 1024) / sizeof(uint4), 4 * 1024 * 1024, setup_stream, 0xabc);
    CUDA_CHECK(cudaMemPrefetchAsync(warm, 4 * 1024 * 1024, args.device, setup_stream));
    CUDA_CHECK(cudaStreamSynchronize(setup_stream));
    (void)time_read_kernel(warm, (4 * 1024 * 1024) / sizeof(uint4), partial, read_stream);
    CUDA_CHECK(cudaFree(warm));

    std::vector<float> fault_read_ms;
    std::vector<float> remote_read_ms;
    std::vector<float> gpu_prefetched_read_ms;
    std::vector<float> prefetch_plain_ms;
    std::vector<float> prefetch_after_accessedby_ms;
    std::vector<float> prefetch_after_remote_read_ms;
    std::vector<float> read_after_remote_then_prefetch_ms;
    std::vector<float> overlap_remote_read_base_ms;
    std::vector<float> overlap_prefetch_base_ms;
    std::vector<float> overlap_remote_read_ms;
    std::vector<float> overlap_prefetch_ms;
    std::vector<float> uva_read_ms;
    std::vector<float> h2d_copy_ms;
    std::vector<float> h2d_copy_after_uva_read_ms;
    std::vector<float> read_after_h2d_copy_ms;
    std::vector<float> overlap_uva_read_base_ms;
    std::vector<float> overlap_h2d_copy_base_ms;
    std::vector<float> overlap_uva_read_ms;
    std::vector<float> overlap_h2d_copy_ms;

    for (int r = 0; r < args.repeats; ++r) {
        uint4 *plain = alloc_managed(args.bytes);
        setup_cpu_resident(plain, count, args.bytes, setup_stream, 0x1000 + r);
        fault_read_ms.push_back(time_read_kernel(plain, count, partial, read_stream));
        CUDA_CHECK(cudaFree(plain));

        uint4 *remote = alloc_managed(args.bytes);
        set_remote_advise(remote, args.bytes, args.device);
        setup_cpu_resident(remote, count, args.bytes, setup_stream, 0x2000 + r);
        remote_read_ms.push_back(time_read_kernel(remote, count, partial, read_stream));
        CUDA_CHECK(cudaFree(remote));

        uint4 *pref_gpu = alloc_managed(args.bytes);
        setup_cpu_resident(pref_gpu, count, args.bytes, setup_stream, 0x3000 + r);
        (void)time_prefetch_gpu(pref_gpu, args.bytes, args.device, copy_stream);
        gpu_prefetched_read_ms.push_back(time_read_kernel(pref_gpu, count, partial, read_stream));
        CUDA_CHECK(cudaFree(pref_gpu));

        uint4 *pref_plain = alloc_managed(args.bytes);
        setup_cpu_resident(pref_plain, count, args.bytes, setup_stream, 0x4000 + r);
        prefetch_plain_ms.push_back(time_prefetch_gpu(pref_plain, args.bytes, args.device, copy_stream));
        CUDA_CHECK(cudaFree(pref_plain));

        uint4 *pref_remote = alloc_managed(args.bytes);
        set_remote_advise(pref_remote, args.bytes, args.device);
        setup_cpu_resident(pref_remote, count, args.bytes, setup_stream, 0x5000 + r);
        prefetch_after_accessedby_ms.push_back(time_prefetch_gpu(pref_remote, args.bytes, args.device, copy_stream));
        CUDA_CHECK(cudaFree(pref_remote));

        uint4 *remote_then_pref = alloc_managed(args.bytes);
        set_remote_advise(remote_then_pref, args.bytes, args.device);
        setup_cpu_resident(remote_then_pref, count, args.bytes, setup_stream, 0x6000 + r);
        (void)time_read_kernel(remote_then_pref, count, partial, read_stream);
        prefetch_after_remote_read_ms.push_back(time_prefetch_gpu(remote_then_pref, args.bytes, args.device, copy_stream));
        read_after_remote_then_prefetch_ms.push_back(time_read_kernel(remote_then_pref, count, partial, read_stream));
        CUDA_CHECK(cudaFree(remote_then_pref));

        uint4 *overlap_remote = alloc_managed(args.bytes);
        uint4 *overlap_promote = alloc_managed(args.bytes);
        set_remote_advise(overlap_remote, args.bytes, args.device);
        setup_cpu_resident(overlap_remote, count, args.bytes, setup_stream, 0x7000 + r);
        setup_cpu_resident(overlap_promote, count, args.bytes, setup_stream, 0x8000 + r);
        overlap_remote_read_base_ms.push_back(time_read_kernel(overlap_remote, count, partial, read_stream));
        CUDA_CHECK(cudaMemPrefetchAsync(overlap_remote, args.bytes, cudaCpuDeviceId, setup_stream));
        CUDA_CHECK(cudaStreamSynchronize(setup_stream));
        fill_cpu(overlap_remote, count, 0x7100 + r);
        overlap_prefetch_base_ms.push_back(time_prefetch_gpu(overlap_promote, args.bytes, args.device, copy_stream));

        CUDA_CHECK(cudaMemPrefetchAsync(overlap_remote, args.bytes, cudaCpuDeviceId, setup_stream));
        CUDA_CHECK(cudaMemPrefetchAsync(overlap_promote, args.bytes, cudaCpuDeviceId, setup_stream));
        CUDA_CHECK(cudaStreamSynchronize(setup_stream));
        fill_cpu(overlap_remote, count, 0x7200 + r);
        fill_cpu(overlap_promote, count, 0x8200 + r);

        cudaEvent_t read_start, read_end, pref_start, pref_end;
        CUDA_CHECK(cudaEventCreate(&pref_start));
        CUDA_CHECK(cudaEventCreate(&pref_end));
        CUDA_CHECK(cudaEventRecord(pref_start, copy_stream));
        CUDA_CHECK(cudaMemPrefetchAsync(overlap_promote, args.bytes, args.device, copy_stream));
        CUDA_CHECK(cudaEventRecord(pref_end, copy_stream));
        (void)time_read_kernel(overlap_remote, count, partial, read_stream, &read_start, &read_end);

        overlap_remote_read_ms.push_back(elapsed_between(read_start, read_end));
        overlap_prefetch_ms.push_back(elapsed_between(pref_start, pref_end));

        CUDA_CHECK(cudaEventDestroy(read_start));
        CUDA_CHECK(cudaEventDestroy(read_end));
        CUDA_CHECK(cudaEventDestroy(pref_start));
        CUDA_CHECK(cudaEventDestroy(pref_end));
        CUDA_CHECK(cudaFree(overlap_remote));
        CUDA_CHECK(cudaFree(overlap_promote));

        PinnedMapped uva = alloc_pinned_mapped(args.bytes);
        fill_cpu(uva.host, count, 0x9000 + r);
        uva_read_ms.push_back(time_read_kernel(uva.device, count, partial, read_stream));

        uint4 *h2d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&h2d_dst, args.bytes));
        h2d_copy_ms.push_back(time_h2d_copy(h2d_dst, uva.host, args.bytes, copy_stream));

        fill_cpu(uva.host, count, 0x9100 + r);
        (void)time_read_kernel(uva.device, count, partial, read_stream);
        h2d_copy_after_uva_read_ms.push_back(time_h2d_copy(h2d_dst, uva.host, args.bytes, copy_stream));
        read_after_h2d_copy_ms.push_back(time_read_kernel(h2d_dst, count, partial, read_stream));
        CUDA_CHECK(cudaFree(h2d_dst));
        CUDA_CHECK(cudaFreeHost(uva.host));

        PinnedMapped overlap_uva = alloc_pinned_mapped(args.bytes);
        PinnedMapped overlap_copy_src = alloc_pinned_mapped(args.bytes);
        uint4 *overlap_h2d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&overlap_h2d_dst, args.bytes));
        fill_cpu(overlap_uva.host, count, 0xa000 + r);
        fill_cpu(overlap_copy_src.host, count, 0xb000 + r);
        overlap_uva_read_base_ms.push_back(time_read_kernel(overlap_uva.device, count, partial, read_stream));
        overlap_h2d_copy_base_ms.push_back(time_h2d_copy(overlap_h2d_dst, overlap_copy_src.host, args.bytes, copy_stream));

        fill_cpu(overlap_uva.host, count, 0xa100 + r);
        fill_cpu(overlap_copy_src.host, count, 0xb100 + r);
        cudaEvent_t uva_read_start, uva_read_end, h2d_start, h2d_end;
        CUDA_CHECK(cudaEventCreate(&h2d_start));
        CUDA_CHECK(cudaEventCreate(&h2d_end));
        CUDA_CHECK(cudaEventRecord(h2d_start, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(overlap_h2d_dst, overlap_copy_src.host, args.bytes, cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaEventRecord(h2d_end, copy_stream));
        (void)time_read_kernel(overlap_uva.device, count, partial, read_stream, &uva_read_start, &uva_read_end);

        overlap_uva_read_ms.push_back(elapsed_between(uva_read_start, uva_read_end));
        overlap_h2d_copy_ms.push_back(elapsed_between(h2d_start, h2d_end));

        CUDA_CHECK(cudaEventDestroy(uva_read_start));
        CUDA_CHECK(cudaEventDestroy(uva_read_end));
        CUDA_CHECK(cudaEventDestroy(h2d_start));
        CUDA_CHECK(cudaEventDestroy(h2d_end));
        CUDA_CHECK(cudaFree(overlap_h2d_dst));
        CUDA_CHECK(cudaFreeHost(overlap_uva.host));
        CUDA_CHECK(cudaFreeHost(overlap_copy_src.host));
    }

    Stats fault_read = stats_of(fault_read_ms);
    Stats remote_read = stats_of(remote_read_ms);
    Stats gpu_read = stats_of(gpu_prefetched_read_ms);
    Stats pref_plain = stats_of(prefetch_plain_ms);
    Stats pref_accessed = stats_of(prefetch_after_accessedby_ms);
    Stats pref_after_remote = stats_of(prefetch_after_remote_read_ms);
    Stats read_after_pref = stats_of(read_after_remote_then_prefetch_ms);
    Stats overlap_remote_base = stats_of(overlap_remote_read_base_ms);
    Stats overlap_pref_base = stats_of(overlap_prefetch_base_ms);
    Stats overlap_remote = stats_of(overlap_remote_read_ms);
    Stats overlap_pref = stats_of(overlap_prefetch_ms);
    Stats uva_read = stats_of(uva_read_ms);
    Stats h2d_copy = stats_of(h2d_copy_ms);
    Stats h2d_after_uva = stats_of(h2d_copy_after_uva_read_ms);
    Stats read_after_h2d = stats_of(read_after_h2d_copy_ms);
    Stats overlap_uva_base = stats_of(overlap_uva_read_base_ms);
    Stats overlap_h2d_base = stats_of(overlap_h2d_copy_base_ms);
    Stats overlap_uva = stats_of(overlap_uva_read_ms);
    Stats overlap_h2d = stats_of(overlap_h2d_copy_ms);

    std::printf("{\n");
    std::printf("  \"device\": {\"id\": %d, \"name\": \"%s\"},\n", args.device, prop.name);
    std::printf("  \"bytes\": %zu,\n", args.bytes);
    std::printf("  \"bytes_mib\": %.3f,\n", (double)args.bytes / 1024.0 / 1024.0);
    std::printf("  \"repeats\": %d,\n", args.repeats);
    std::printf("  \"stats\": {\n");
    print_stats("fault_read_from_cpu_no_accessedby", fault_read, args.bytes);
    std::printf(",\n");
    print_stats("remote_read_cpu_preferred_accessedby", remote_read, args.bytes);
    std::printf(",\n");
    print_stats("gpu_prefetched_read", gpu_read, args.bytes);
    std::printf(",\n");
    print_stats("prefetch_cpu_to_gpu_plain", pref_plain, args.bytes);
    std::printf(",\n");
    print_stats("prefetch_cpu_to_gpu_after_accessedby", pref_accessed, args.bytes);
    std::printf(",\n");
    print_stats("prefetch_cpu_to_gpu_after_remote_read", pref_after_remote, args.bytes);
    std::printf(",\n");
    print_stats("read_after_remote_then_prefetch", read_after_pref, args.bytes);
    std::printf(",\n");
    print_stats("overlap_remote_read_baseline", overlap_remote_base, args.bytes);
    std::printf(",\n");
    print_stats("overlap_prefetch_baseline", overlap_pref_base, args.bytes);
    std::printf(",\n");
    print_stats("overlap_remote_read_during_prefetch", overlap_remote, args.bytes);
    std::printf(",\n");
    print_stats("overlap_prefetch_during_remote_read", overlap_pref, args.bytes);
    std::printf(",\n");
    print_stats("uva_pinned_host_read", uva_read, args.bytes);
    std::printf(",\n");
    print_stats("h2d_copy_pinned_to_device", h2d_copy, args.bytes);
    std::printf(",\n");
    print_stats("h2d_copy_after_uva_read", h2d_after_uva, args.bytes);
    std::printf(",\n");
    print_stats("read_after_h2d_copy", read_after_h2d, args.bytes);
    std::printf(",\n");
    print_stats("overlap_uva_read_baseline", overlap_uva_base, args.bytes);
    std::printf(",\n");
    print_stats("overlap_h2d_copy_baseline", overlap_h2d_base, args.bytes);
    std::printf(",\n");
    print_stats("overlap_uva_read_during_h2d_copy", overlap_uva, args.bytes);
    std::printf(",\n");
    print_stats("overlap_h2d_copy_during_uva_read", overlap_h2d, args.bytes);
    std::printf("\n  },\n");
    std::printf("  \"ratios\": {\n");
    std::printf("    \"remote_read_vs_gpu_prefetched_read_p50\": %.6f,\n",
                remote_read.p50_ms / std::max(gpu_read.p50_ms, 1e-9));
    std::printf("    \"prefetch_after_accessedby_vs_plain_p50\": %.6f,\n",
                pref_accessed.p50_ms / std::max(pref_plain.p50_ms, 1e-9));
    std::printf("    \"prefetch_after_remote_read_vs_plain_p50\": %.6f,\n",
                pref_after_remote.p50_ms / std::max(pref_plain.p50_ms, 1e-9));
    std::printf("    \"remote_read_overlap_slowdown_p50\": %.6f,\n",
                overlap_remote.p50_ms / std::max(overlap_remote_base.p50_ms, 1e-9));
    std::printf("    \"prefetch_overlap_slowdown_p50\": %.6f,\n",
                overlap_pref.p50_ms / std::max(overlap_pref_base.p50_ms, 1e-9));
    std::printf("    \"uva_read_vs_gpu_h2d_read_p50\": %.6f,\n",
                uva_read.p50_ms / std::max(read_after_h2d.p50_ms, 1e-9));
    std::printf("    \"h2d_after_uva_read_vs_plain_p50\": %.6f,\n",
                h2d_after_uva.p50_ms / std::max(h2d_copy.p50_ms, 1e-9));
    std::printf("    \"uva_read_overlap_slowdown_p50\": %.6f,\n",
                overlap_uva.p50_ms / std::max(overlap_uva_base.p50_ms, 1e-9));
    std::printf("    \"h2d_overlap_slowdown_p50\": %.6f\n",
                overlap_h2d.p50_ms / std::max(overlap_h2d_base.p50_ms, 1e-9));
    std::printf("  }\n");
    std::printf("}\n");

    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaStreamDestroy(setup_stream));
    CUDA_CHECK(cudaStreamDestroy(read_stream));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    return 0;
}
