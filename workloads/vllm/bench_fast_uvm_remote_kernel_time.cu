#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <stdint.h>
#include <sys/types.h>

extern "C" void *uvm_malloc(ssize_t size, int device, cudaStream_t stream);
extern "C" void uvm_free(void *ptr, ssize_t size, int device, cudaStream_t stream);
extern "C" int uvm_driver_batch_prepare_remote_read(const uintptr_t *ptrs,
                                                     const size_t *sizes,
                                                     size_t count,
                                                     int device,
                                                     int async);
extern "C" int uvm_driver_phase_f_register_region(uint64_t region_id,
                                                   uintptr_t base,
                                                   size_t length,
                                                   uint32_t pool,
                                                   uint32_t layer_id,
                                                   uint32_t expert_id,
                                                   uint32_t role,
                                                   uint32_t flags);
extern "C" int uvm_driver_phase_f_unregister_region(uint64_t region_id);

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
static constexpr uint32_t kUvmRegionPoolWeight = 2;
static constexpr uint32_t kUvmRegionRoleMoeExpertW13 = 3;
static constexpr uint32_t kUvmRegionFlagEvictable = 1u << 0;
static constexpr uint32_t kUvmRegionFlagPrefetchable = 1u << 3;

__global__ void read_u4_kernel(const uint4 *data, size_t count, unsigned long long *partial)
{
    __shared__ unsigned long long scratch[kThreads];
    unsigned long long acc = 0;
    const size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const uint4 value = data[i];
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

struct Args {
    int device = 0;
    int repeats = 9;
    std::vector<size_t> sizes = {
        4ull * 1024ull * 1024ull + 512ull * 1024ull,
        18ull * 1024ull * 1024ull,
        72ull * 1024ull * 1024ull,
        144ull * 1024ull * 1024ull,
    };
};

struct Stats {
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double mean_ms = 0.0;
    double max_ms = 0.0;
};

static size_t parse_mib(const char *text)
{
    return (size_t)std::strtoull(text, nullptr, 10) * 1024ull * 1024ull;
}

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
        else if (std::strcmp(argv[i], "--sizes-mib") == 0 && i + 1 < argc) {
            args.sizes.clear();
            char *raw = argv[++i];
            char *save = nullptr;
            for (char *tok = strtok_r(raw, ",", &save); tok; tok = strtok_r(nullptr, ",", &save))
                args.sizes.push_back(parse_mib(tok));
        }
        else {
            std::fprintf(stderr, "usage: %s [--device N] [--repeats N] [--sizes-mib a,b,c]\n", argv[0]);
            std::exit(2);
        }
    }

    if (args.repeats <= 0 || args.sizes.empty()) {
        std::fprintf(stderr, "invalid repeats or sizes\n");
        std::exit(2);
    }

    for (size_t &bytes : args.sizes) {
        bytes = (bytes / sizeof(uint4)) * sizeof(uint4);
        if (bytes == 0) {
            std::fprintf(stderr, "invalid zero size\n");
            std::exit(2);
        }
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

static float time_read_kernel(const uint4 *ptr,
                              size_t bytes,
                              unsigned long long *partial,
                              cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    read_u4_kernel<<<kBlocks, kThreads, 0, stream>>>(ptr, bytes / sizeof(uint4), partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static float time_prefetch_then_read_kernel(const uint4 *ptr,
                                            size_t bytes,
                                            int device,
                                            unsigned long long *partial,
                                            cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(const_cast<uint4 *>(ptr), bytes, device, stream));
    read_u4_kernel<<<kBlocks, kThreads, 0, stream>>>(ptr, bytes / sizeof(uint4), partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static void prepare_native_uvm_remote(uint4 **ptr_out, size_t bytes, int device, cudaStream_t stream, unsigned int seed)
{
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fill_cpu(static_cast<uint4 *>(ptr), bytes / sizeof(uint4), seed);
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device));
    *ptr_out = static_cast<uint4 *>(ptr);
}

static void prepare_native_uvm_fault_migrate(uint4 **ptr_out, size_t bytes, cudaStream_t stream, unsigned int seed)
{
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fill_cpu(static_cast<uint4 *>(ptr), bytes / sizeof(uint4), seed);
    *ptr_out = static_cast<uint4 *>(ptr);
}

static void prepare_fast_uvm_remote(uint4 **ptr_out, size_t bytes, int device, cudaStream_t stream, unsigned int seed)
{
    void *ptr = uvm_malloc((ssize_t)bytes, device, stream);
    if (!ptr) {
        std::fprintf(stderr, "uvm_malloc failed for %zu bytes\n", bytes);
        std::exit(1);
    }

    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fill_cpu(static_cast<uint4 *>(ptr), bytes / sizeof(uint4), seed);
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

    uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    size_t size = bytes;
    int result = uvm_driver_batch_prepare_remote_read(&base, &size, 1, device, 0);
    if (result <= 0) {
        std::fprintf(stderr, "uvm_driver_batch_prepare_remote_read failed: %d\n", result);
        std::exit(1);
    }

    *ptr_out = static_cast<uint4 *>(ptr);
}

static void prepare_fast_uvm_remote_first_fault(uint4 **ptr_out,
                                                size_t bytes,
                                                int device,
                                                cudaStream_t stream,
                                                unsigned int seed)
{
    void *ptr = uvm_malloc((ssize_t)bytes, device, stream);
    if (!ptr) {
        std::fprintf(stderr, "uvm_malloc failed for %zu bytes\n", bytes);
        std::exit(1);
    }

    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fill_cpu(static_cast<uint4 *>(ptr), bytes / sizeof(uint4), seed);
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly, device));

    *ptr_out = static_cast<uint4 *>(ptr);
}

static void prepare_expert_readdup_fault_fastpath(uint4 **ptr_out,
                                                  uint64_t *region_id_out,
                                                  size_t bytes,
                                                  int device,
                                                  cudaStream_t stream,
                                                  unsigned int seed,
                                                  uint64_t region_id)
{
    void *ptr = uvm_malloc((ssize_t)bytes, device, stream);
    if (!ptr) {
        std::fprintf(stderr, "uvm_malloc failed for %zu bytes\n", bytes);
        std::exit(1);
    }

    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fill_cpu(static_cast<uint4 *>(ptr), bytes / sizeof(uint4), seed);

    int result = uvm_driver_phase_f_register_region(region_id,
                                                    reinterpret_cast<uintptr_t>(ptr),
                                                    bytes,
                                                    kUvmRegionPoolWeight,
                                                    31,
                                                    7,
                                                    kUvmRegionRoleMoeExpertW13,
                                                    kUvmRegionFlagEvictable | kUvmRegionFlagPrefetchable);
    if (result < 0) {
        std::fprintf(stderr, "uvm_driver_phase_f_register_region failed: %d\n", result);
        std::exit(1);
    }

    *ptr_out = static_cast<uint4 *>(ptr);
    *region_id_out = region_id;
}

static void prepare_uva_remote(uint4 **ptr_out, uint4 **host_out, size_t bytes, unsigned int seed)
{
    void *host = nullptr;
    void *device_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr, host, 0));
    fill_cpu(static_cast<uint4 *>(host), bytes / sizeof(uint4), seed);
    *ptr_out = static_cast<uint4 *>(device_ptr);
    *host_out = static_cast<uint4 *>(host);
}

static void print_row(const char *name, const Stats &stats, size_t bytes)
{
    std::printf("  %-24s p50=%8.3f ms  mean=%8.3f ms  min=%8.3f ms  max=%8.3f ms  p50_bw=%6.1f GB/s\n",
                name,
                stats.p50_ms,
                stats.mean_ms,
                stats.min_ms,
                stats.max_ms,
                gbps(bytes, stats.p50_ms));
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

    cudaStream_t setup_stream, read_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&setup_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&read_stream, cudaStreamNonBlocking));

    unsigned long long *partial = nullptr;
    CUDA_CHECK(cudaMalloc(&partial, kBlocks * sizeof(*partial)));

    uint4 *warm = nullptr;
    CUDA_CHECK(cudaMallocManaged(&warm, 4ull * 1024ull * 1024ull));
    CUDA_CHECK(cudaMemPrefetchAsync(warm, 4ull * 1024ull * 1024ull, cudaCpuDeviceId, setup_stream));
    CUDA_CHECK(cudaStreamSynchronize(setup_stream));
    fill_cpu(warm, (4ull * 1024ull * 1024ull) / sizeof(uint4), 0xabc);
    CUDA_CHECK(cudaMemAdvise(warm, 4ull * 1024ull * 1024ull, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(warm, 4ull * 1024ull * 1024ull, cudaMemAdviseSetAccessedBy, args.device));
    (void)time_read_kernel(warm, 4ull * 1024ull * 1024ull, partial, read_stream);
    CUDA_CHECK(cudaFree(warm));

    std::printf("device=%d name=\"%s\" repeats=%d metric=cuda_event_kernel_elapsed\n",
                args.device,
                prop.name,
                args.repeats);

    for (size_t bytes : args.sizes) {
        std::vector<float> uva_ms;
        std::vector<float> native_fault_ms;
        std::vector<float> native_sync_prefetch_then_compute_ms;
        std::vector<float> native_uvm_ms;
        std::vector<float> fast_uvm_ms;
        std::vector<float> fast_uvm_fault_ms;
        std::vector<float> expert_readdup_fault_ms;

        for (int r = 0; r < args.repeats; ++r) {
            uint4 *ptr = nullptr;
            uint4 *host = nullptr;
            prepare_uva_remote(&ptr, &host, bytes, 0x1000 + r);
            uva_ms.push_back(time_read_kernel(ptr, bytes, partial, read_stream));
            CUDA_CHECK(cudaFreeHost(host));

            prepare_native_uvm_fault_migrate(&ptr, bytes, setup_stream, 0x1800 + r);
            native_fault_ms.push_back(time_read_kernel(ptr, bytes, partial, read_stream));
            CUDA_CHECK(cudaFree(ptr));

            prepare_native_uvm_fault_migrate(&ptr, bytes, setup_stream, 0x1900 + r);
            native_sync_prefetch_then_compute_ms.push_back(
                time_prefetch_then_read_kernel(ptr, bytes, args.device, partial, read_stream));
            CUDA_CHECK(cudaFree(ptr));

            prepare_native_uvm_remote(&ptr, bytes, args.device, setup_stream, 0x2000 + r);
            native_uvm_ms.push_back(time_read_kernel(ptr, bytes, partial, read_stream));
            CUDA_CHECK(cudaFree(ptr));

            prepare_fast_uvm_remote(&ptr, bytes, args.device, setup_stream, 0x3000 + r);
            fast_uvm_ms.push_back(time_read_kernel(ptr, bytes, partial, read_stream));
            uvm_free(ptr, (ssize_t)bytes, args.device, setup_stream);

            prepare_fast_uvm_remote_first_fault(&ptr, bytes, args.device, setup_stream, 0x4000 + r);
            fast_uvm_fault_ms.push_back(time_read_kernel(ptr, bytes, partial, read_stream));
            uvm_free(ptr, (ssize_t)bytes, args.device, setup_stream);

            uint64_t region_id = 0;
            prepare_expert_readdup_fault_fastpath(&ptr,
                                                  &region_id,
                                                  bytes,
                                                  args.device,
                                                  setup_stream,
                                                  0x5000 + r,
                                                  0x500000000ull + bytes + (uint64_t)r);
            expert_readdup_fault_ms.push_back(time_read_kernel(ptr, bytes, partial, read_stream));
            if (region_id)
                (void)uvm_driver_phase_f_unregister_region(region_id);
            uvm_free(ptr, (ssize_t)bytes, args.device, setup_stream);
        }

        std::printf("size=%.3f MiB bytes=%zu\n", (double)bytes / 1024.0 / 1024.0, bytes);
        print_row("native_uva_remote", stats_of(uva_ms), bytes);
        print_row("native_uvm_fault_migrate", stats_of(native_fault_ms), bytes);
        print_row("native_prefetch_then_compute", stats_of(native_sync_prefetch_then_compute_ms), bytes);
        print_row("native_uvm_remote", stats_of(native_uvm_ms), bytes);
        print_row("fast_uvm_remote", stats_of(fast_uvm_ms), bytes);
        print_row("fast_uvm_fault_remote", stats_of(fast_uvm_fault_ms), bytes);
        print_row("expert_readdup_fault_fastpath", stats_of(expert_readdup_fault_ms), bytes);
    }

    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaStreamDestroy(setup_stream));
    CUDA_CHECK(cudaStreamDestroy(read_stream));
    return 0;
}
