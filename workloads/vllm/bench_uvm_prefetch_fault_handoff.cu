#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
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
    const size_t stride = (size_t)blockDim.x * gridDim.x;

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

struct Args {
    int device = 0;
    int repeats = 7;
    int delay_us = 0;
    size_t bytes = 144ull * 1024ull * 1024ull;
    size_t small_bytes = 4ull * 1024ull * 1024ull;
    size_t chunk_bytes = 10ull * 1024ull * 1024ull;
};

struct Stats {
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double mean_ms = 0.0;
    double max_ms = 0.0;
};

struct OverlapSample {
    float read_ms = 0.0f;
    float prefetch_ms = 0.0f;
    float total_ms = 0.0f;
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
        else if (std::strcmp(argv[i], "--small-mib") == 0 && i + 1 < argc) {
            args.small_bytes = (size_t)std::strtoull(argv[++i], nullptr, 10) * 1024ull * 1024ull;
        }
        else if (std::strcmp(argv[i], "--chunk-mib") == 0 && i + 1 < argc) {
            args.chunk_bytes = (size_t)std::strtoull(argv[++i], nullptr, 10) * 1024ull * 1024ull;
        }
        else if (std::strcmp(argv[i], "--delay-us") == 0 && i + 1 < argc) {
            args.delay_us = std::atoi(argv[++i]);
        }
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--repeats N] [--bytes-mib N] "
                         "[--small-mib N] [--chunk-mib N] [--delay-us N]\n",
                         argv[0]);
            std::exit(2);
        }
    }

    args.bytes = (args.bytes / sizeof(uint4)) * sizeof(uint4);
    args.small_bytes = (args.small_bytes / sizeof(uint4)) * sizeof(uint4);
    args.chunk_bytes = (args.chunk_bytes / sizeof(uint4)) * sizeof(uint4);
    if (args.bytes == 0 ||
        args.small_bytes == 0 ||
        args.chunk_bytes == 0 ||
        args.repeats <= 0 ||
        args.delay_us < 0) {
        std::fprintf(stderr, "invalid arguments\n");
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

static uint4 *alloc_managed(size_t bytes)
{
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
    return static_cast<uint4 *>(ptr);
}

static void setup_cpu_resident(uint4 *ptr, size_t bytes, cudaStream_t stream, unsigned int seed)
{
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fill_cpu(ptr, bytes / sizeof(uint4), seed);
}

static float elapsed_between(cudaEvent_t start, cudaEvent_t end)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(end));
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
    const float ms = elapsed_between(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static float time_chunked_prefetch_gpu(uint4 *ptr, size_t bytes, size_t chunk_bytes, int device, cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));

    for (size_t offset = 0; offset < bytes; offset += chunk_bytes) {
        const size_t this_chunk = std::min(chunk_bytes, bytes - offset);
        void *chunk_ptr = static_cast<void *>(reinterpret_cast<char *>(ptr) + offset);

        CUDA_CHECK(cudaMemPrefetchAsync(chunk_ptr, this_chunk, device, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_between(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static float time_queued_chunked_prefetch_gpu(uint4 *ptr,
                                              size_t bytes,
                                              size_t chunk_bytes,
                                              int device,
                                              cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));

    for (size_t offset = 0; offset < bytes; offset += chunk_bytes) {
        const size_t this_chunk = std::min(chunk_bytes, bytes - offset);
        void *chunk_ptr = static_cast<void *>(reinterpret_cast<char *>(ptr) + offset);

        CUDA_CHECK(cudaMemPrefetchAsync(chunk_ptr, this_chunk, device, stream));
    }

    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_between(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static float time_read_kernel(uint4 *ptr, size_t bytes, unsigned long long *partial, cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    read_u4_kernel<<<kBlocks, kThreads, 0, stream>>>(ptr, bytes / sizeof(uint4), partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    const float ms = elapsed_between(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

static OverlapSample time_prefetch_and_read(uint4 *prefetch_ptr,
                                            size_t prefetch_bytes,
                                            uint4 *read_ptr,
                                            size_t read_bytes,
                                            int device,
                                            unsigned long long *partial,
                                            cudaStream_t control_stream,
                                            cudaStream_t prefetch_stream,
                                            cudaStream_t read_stream,
                                            int delay_us)
{
    cudaEvent_t go;
    cudaEvent_t prefetch_start, prefetch_end;
    cudaEvent_t read_start, read_end;

    CUDA_CHECK(cudaEventCreateWithFlags(&go, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&prefetch_start));
    CUDA_CHECK(cudaEventCreate(&prefetch_end));
    CUDA_CHECK(cudaEventCreate(&read_start));
    CUDA_CHECK(cudaEventCreate(&read_end));

    CUDA_CHECK(cudaStreamWaitEvent(prefetch_stream, go, 0));
    CUDA_CHECK(cudaEventRecord(prefetch_start, prefetch_stream));
    CUDA_CHECK(cudaMemPrefetchAsync(prefetch_ptr, prefetch_bytes, device, prefetch_stream));
    CUDA_CHECK(cudaEventRecord(prefetch_end, prefetch_stream));

    CUDA_CHECK(cudaEventRecord(go, control_stream));
    CUDA_CHECK(cudaStreamSynchronize(control_stream));

    if (delay_us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));

    CUDA_CHECK(cudaEventRecord(read_start, read_stream));
    read_u4_kernel<<<kBlocks, kThreads, 0, read_stream>>>(read_ptr, read_bytes / sizeof(uint4), partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(read_end, read_stream));

    CUDA_CHECK(cudaEventSynchronize(prefetch_end));
    CUDA_CHECK(cudaEventSynchronize(read_end));

    OverlapSample sample;
    CUDA_CHECK(cudaEventElapsedTime(&sample.prefetch_ms, prefetch_start, prefetch_end));
    CUDA_CHECK(cudaEventElapsedTime(&sample.read_ms, read_start, read_end));
    sample.total_ms = std::max(sample.prefetch_ms, sample.read_ms + (float)delay_us / 1000.0f);

    CUDA_CHECK(cudaEventDestroy(go));
    CUDA_CHECK(cudaEventDestroy(prefetch_start));
    CUDA_CHECK(cudaEventDestroy(prefetch_end));
    CUDA_CHECK(cudaEventDestroy(read_start));
    CUDA_CHECK(cudaEventDestroy(read_end));
    return sample;
}

static OverlapSample time_chunked_prefetch_and_read(uint4 *prefetch_ptr,
                                                    size_t prefetch_bytes,
                                                    size_t chunk_bytes,
                                                    uint4 *read_ptr,
                                                    size_t read_bytes,
                                                    int device,
                                                    unsigned long long *partial,
                                                    cudaStream_t control_stream,
                                                    cudaStream_t prefetch_stream,
                                                    cudaStream_t read_stream,
                                                    int delay_us)
{
    cudaEvent_t go;
    cudaEvent_t prefetch_start, prefetch_end;
    cudaEvent_t read_start, read_end;

    CUDA_CHECK(cudaEventCreateWithFlags(&go, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&prefetch_start));
    CUDA_CHECK(cudaEventCreate(&prefetch_end));
    CUDA_CHECK(cudaEventCreate(&read_start));
    CUDA_CHECK(cudaEventCreate(&read_end));

    std::thread prefetch_thread([=]() {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamWaitEvent(prefetch_stream, go, 0));
        CUDA_CHECK(cudaEventRecord(prefetch_start, prefetch_stream));

        for (size_t offset = 0; offset < prefetch_bytes; offset += chunk_bytes) {
            const size_t this_chunk = std::min(chunk_bytes, prefetch_bytes - offset);
            void *chunk_ptr = static_cast<void *>(reinterpret_cast<char *>(prefetch_ptr) + offset);

            CUDA_CHECK(cudaMemPrefetchAsync(chunk_ptr, this_chunk, device, prefetch_stream));
            CUDA_CHECK(cudaStreamSynchronize(prefetch_stream));
        }

        CUDA_CHECK(cudaEventRecord(prefetch_end, prefetch_stream));
        CUDA_CHECK(cudaStreamSynchronize(prefetch_stream));
    });

    CUDA_CHECK(cudaEventRecord(go, control_stream));
    CUDA_CHECK(cudaStreamSynchronize(control_stream));

    if (delay_us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));

    CUDA_CHECK(cudaEventRecord(read_start, read_stream));
    read_u4_kernel<<<kBlocks, kThreads, 0, read_stream>>>(read_ptr, read_bytes / sizeof(uint4), partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(read_end, read_stream));

    CUDA_CHECK(cudaEventSynchronize(read_end));
    prefetch_thread.join();

    OverlapSample sample;
    CUDA_CHECK(cudaEventElapsedTime(&sample.prefetch_ms, prefetch_start, prefetch_end));
    CUDA_CHECK(cudaEventElapsedTime(&sample.read_ms, read_start, read_end));
    sample.total_ms = std::max(sample.prefetch_ms, sample.read_ms + (float)delay_us / 1000.0f);

    CUDA_CHECK(cudaEventDestroy(go));
    CUDA_CHECK(cudaEventDestroy(prefetch_start));
    CUDA_CHECK(cudaEventDestroy(prefetch_end));
    CUDA_CHECK(cudaEventDestroy(read_start));
    CUDA_CHECK(cudaEventDestroy(read_end));
    return sample;
}

static OverlapSample time_queued_chunked_prefetch_and_read(uint4 *prefetch_ptr,
                                                           size_t prefetch_bytes,
                                                           size_t chunk_bytes,
                                                           uint4 *read_ptr,
                                                           size_t read_bytes,
                                                           int device,
                                                           unsigned long long *partial,
                                                           cudaStream_t control_stream,
                                                           cudaStream_t prefetch_stream,
                                                           cudaStream_t read_stream,
                                                           int delay_us)
{
    cudaEvent_t go;
    cudaEvent_t prefetch_start, prefetch_end;
    cudaEvent_t read_start, read_end;

    CUDA_CHECK(cudaEventCreateWithFlags(&go, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&prefetch_start));
    CUDA_CHECK(cudaEventCreate(&prefetch_end));
    CUDA_CHECK(cudaEventCreate(&read_start));
    CUDA_CHECK(cudaEventCreate(&read_end));

    CUDA_CHECK(cudaStreamWaitEvent(prefetch_stream, go, 0));
    CUDA_CHECK(cudaEventRecord(prefetch_start, prefetch_stream));

    for (size_t offset = 0; offset < prefetch_bytes; offset += chunk_bytes) {
        const size_t this_chunk = std::min(chunk_bytes, prefetch_bytes - offset);
        void *chunk_ptr = static_cast<void *>(reinterpret_cast<char *>(prefetch_ptr) + offset);

        CUDA_CHECK(cudaMemPrefetchAsync(chunk_ptr, this_chunk, device, prefetch_stream));
    }

    CUDA_CHECK(cudaEventRecord(prefetch_end, prefetch_stream));
    CUDA_CHECK(cudaEventRecord(go, control_stream));
    CUDA_CHECK(cudaStreamSynchronize(control_stream));

    if (delay_us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));

    CUDA_CHECK(cudaEventRecord(read_start, read_stream));
    read_u4_kernel<<<kBlocks, kThreads, 0, read_stream>>>(read_ptr, read_bytes / sizeof(uint4), partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(read_end, read_stream));

    CUDA_CHECK(cudaEventSynchronize(prefetch_end));
    CUDA_CHECK(cudaEventSynchronize(read_end));

    OverlapSample sample;
    CUDA_CHECK(cudaEventElapsedTime(&sample.prefetch_ms, prefetch_start, prefetch_end));
    CUDA_CHECK(cudaEventElapsedTime(&sample.read_ms, read_start, read_end));
    sample.total_ms = std::max(sample.prefetch_ms, sample.read_ms + (float)delay_us / 1000.0f);

    CUDA_CHECK(cudaEventDestroy(go));
    CUDA_CHECK(cudaEventDestroy(prefetch_start));
    CUDA_CHECK(cudaEventDestroy(prefetch_end));
    CUDA_CHECK(cudaEventDestroy(read_start));
    CUDA_CHECK(cudaEventDestroy(read_end));
    return sample;
}

static void print_stats_line(const char *name, const Stats &stats, size_t bytes)
{
    std::printf("%-42s p50=%8.3f ms  mean=%8.3f ms  min=%8.3f ms  max=%8.3f ms  p50_bw=%6.1f GB/s\n",
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

    CUDA_CHECK(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    cudaStream_t setup_stream, control_stream, prefetch_stream, read_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&setup_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&control_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&read_stream, cudaStreamNonBlocking));

    unsigned long long *partial = nullptr;
    CUDA_CHECK(cudaMalloc(&partial, kBlocks * sizeof(*partial)));

    uint4 *warm = alloc_managed(4ull * 1024ull * 1024ull);
    setup_cpu_resident(warm, 4ull * 1024ull * 1024ull, setup_stream, 0xabc);
    CUDA_CHECK(cudaMemPrefetchAsync(warm, 4ull * 1024ull * 1024ull, args.device, setup_stream));
    CUDA_CHECK(cudaStreamSynchronize(setup_stream));
    (void)time_read_kernel(warm, 4ull * 1024ull * 1024ull, partial, read_stream);
    CUDA_CHECK(cudaFree(warm));

    std::vector<float> prefetch_only_ms;
    std::vector<float> chunked_prefetch_only_ms;
    std::vector<float> queued_chunked_prefetch_only_ms;
    std::vector<float> fault_read_only_ms;
    std::vector<float> same_read_ms;
    std::vector<float> same_prefetch_ms;
    std::vector<float> same_total_ms;
    std::vector<float> two_range_read_ms;
    std::vector<float> two_range_prefetch_ms;
    std::vector<float> two_range_total_ms;
    std::vector<float> small_fault_only_ms;
    std::vector<float> small_during_large_read_ms;
    std::vector<float> small_during_large_prefetch_ms;
    std::vector<float> small_during_large_total_ms;
    std::vector<float> small_during_chunked_read_ms;
    std::vector<float> chunked_prefetch_ms;
    std::vector<float> small_during_chunked_total_ms;
    std::vector<float> small_during_queued_chunked_read_ms;
    std::vector<float> queued_chunked_prefetch_ms;
    std::vector<float> small_during_queued_chunked_total_ms;

    for (int r = 0; r < args.repeats; ++r) {
        uint4 *prefetch_only = alloc_managed(args.bytes);
        setup_cpu_resident(prefetch_only, args.bytes, setup_stream, 0x1000 + r);
        prefetch_only_ms.push_back(time_prefetch_gpu(prefetch_only, args.bytes, args.device, prefetch_stream));
        CUDA_CHECK(cudaFree(prefetch_only));

        uint4 *chunked_only = alloc_managed(args.bytes);
        setup_cpu_resident(chunked_only, args.bytes, setup_stream, 0x1100 + r);
        chunked_prefetch_only_ms.push_back(time_chunked_prefetch_gpu(chunked_only,
                                                                     args.bytes,
                                                                     args.chunk_bytes,
                                                                     args.device,
                                                                     prefetch_stream));
        CUDA_CHECK(cudaFree(chunked_only));

        uint4 *queued_chunked_only = alloc_managed(args.bytes);
        setup_cpu_resident(queued_chunked_only, args.bytes, setup_stream, 0x1200 + r);
        queued_chunked_prefetch_only_ms.push_back(time_queued_chunked_prefetch_gpu(queued_chunked_only,
                                                                                   args.bytes,
                                                                                   args.chunk_bytes,
                                                                                   args.device,
                                                                                   prefetch_stream));
        CUDA_CHECK(cudaFree(queued_chunked_only));

        uint4 *fault_only = alloc_managed(args.bytes);
        setup_cpu_resident(fault_only, args.bytes, setup_stream, 0x2000 + r);
        fault_read_only_ms.push_back(time_read_kernel(fault_only, args.bytes, partial, read_stream));
        CUDA_CHECK(cudaFree(fault_only));

        uint4 *same = alloc_managed(args.bytes);
        setup_cpu_resident(same, args.bytes, setup_stream, 0x3000 + r);
        OverlapSample same_sample = time_prefetch_and_read(same,
                                                           args.bytes,
                                                           same,
                                                           args.bytes,
                                                           args.device,
                                                           partial,
                                                           control_stream,
                                                           prefetch_stream,
                                                           read_stream,
                                                           args.delay_us);
        same_read_ms.push_back(same_sample.read_ms);
        same_prefetch_ms.push_back(same_sample.prefetch_ms);
        same_total_ms.push_back(same_sample.total_ms);
        CUDA_CHECK(cudaFree(same));

        uint4 *fault_range = alloc_managed(args.bytes);
        uint4 *prefetch_range = alloc_managed(args.bytes);
        setup_cpu_resident(fault_range, args.bytes, setup_stream, 0x4000 + r);
        setup_cpu_resident(prefetch_range, args.bytes, setup_stream, 0x5000 + r);
        OverlapSample two_sample = time_prefetch_and_read(prefetch_range,
                                                          args.bytes,
                                                          fault_range,
                                                          args.bytes,
                                                          args.device,
                                                          partial,
                                                          control_stream,
                                                          prefetch_stream,
                                                          read_stream,
                                                          args.delay_us);
        two_range_read_ms.push_back(two_sample.read_ms);
        two_range_prefetch_ms.push_back(two_sample.prefetch_ms);
        two_range_total_ms.push_back(two_sample.total_ms);
        CUDA_CHECK(cudaFree(fault_range));
        CUDA_CHECK(cudaFree(prefetch_range));

        uint4 *small_fault = alloc_managed(args.small_bytes);
        setup_cpu_resident(small_fault, args.small_bytes, setup_stream, 0x6000 + r);
        small_fault_only_ms.push_back(time_read_kernel(small_fault, args.small_bytes, partial, read_stream));
        CUDA_CHECK(cudaFree(small_fault));

        uint4 *small = alloc_managed(args.small_bytes);
        uint4 *large_prefetch = alloc_managed(args.bytes);
        setup_cpu_resident(small, args.small_bytes, setup_stream, 0x7000 + r);
        setup_cpu_resident(large_prefetch, args.bytes, setup_stream, 0x8000 + r);
        OverlapSample small_sample = time_prefetch_and_read(large_prefetch,
                                                            args.bytes,
                                                            small,
                                                            args.small_bytes,
                                                            args.device,
                                                            partial,
                                                            control_stream,
                                                            prefetch_stream,
                                                            read_stream,
                                                            args.delay_us);
        small_during_large_read_ms.push_back(small_sample.read_ms);
        small_during_large_prefetch_ms.push_back(small_sample.prefetch_ms);
        small_during_large_total_ms.push_back(small_sample.total_ms);
        CUDA_CHECK(cudaFree(small));
        CUDA_CHECK(cudaFree(large_prefetch));

        uint4 *small_chunked = alloc_managed(args.small_bytes);
        uint4 *chunked_prefetch = alloc_managed(args.bytes);
        setup_cpu_resident(small_chunked, args.small_bytes, setup_stream, 0x9000 + r);
        setup_cpu_resident(chunked_prefetch, args.bytes, setup_stream, 0xa000 + r);
        OverlapSample chunked_sample = time_chunked_prefetch_and_read(chunked_prefetch,
                                                                      args.bytes,
                                                                      args.chunk_bytes,
                                                                      small_chunked,
                                                                      args.small_bytes,
                                                                      args.device,
                                                                      partial,
                                                                      control_stream,
                                                                      prefetch_stream,
                                                                      read_stream,
                                                                      args.delay_us);
        small_during_chunked_read_ms.push_back(chunked_sample.read_ms);
        chunked_prefetch_ms.push_back(chunked_sample.prefetch_ms);
        small_during_chunked_total_ms.push_back(chunked_sample.total_ms);
        CUDA_CHECK(cudaFree(small_chunked));
        CUDA_CHECK(cudaFree(chunked_prefetch));

        uint4 *small_queued_chunked = alloc_managed(args.small_bytes);
        uint4 *queued_chunked_prefetch = alloc_managed(args.bytes);
        setup_cpu_resident(small_queued_chunked, args.small_bytes, setup_stream, 0xb000 + r);
        setup_cpu_resident(queued_chunked_prefetch, args.bytes, setup_stream, 0xc000 + r);
        OverlapSample queued_chunked_sample = time_queued_chunked_prefetch_and_read(queued_chunked_prefetch,
                                                                                    args.bytes,
                                                                                    args.chunk_bytes,
                                                                                    small_queued_chunked,
                                                                                    args.small_bytes,
                                                                                    args.device,
                                                                                    partial,
                                                                                    control_stream,
                                                                                    prefetch_stream,
                                                                                    read_stream,
                                                                                    args.delay_us);
        small_during_queued_chunked_read_ms.push_back(queued_chunked_sample.read_ms);
        queued_chunked_prefetch_ms.push_back(queued_chunked_sample.prefetch_ms);
        small_during_queued_chunked_total_ms.push_back(queued_chunked_sample.total_ms);
        CUDA_CHECK(cudaFree(small_queued_chunked));
        CUDA_CHECK(cudaFree(queued_chunked_prefetch));
    }

    std::printf("device=%d name=\"%s\" repeats=%d bytes=%.1f MiB small=%.1f MiB chunk=%.1f MiB delay=%d us\n",
                args.device,
                prop.name,
                args.repeats,
                (double)args.bytes / 1024.0 / 1024.0,
                (double)args.small_bytes / 1024.0 / 1024.0,
                (double)args.chunk_bytes / 1024.0 / 1024.0,
                args.delay_us);
    print_stats_line("prefetch_only_large", stats_of(prefetch_only_ms), args.bytes);
    print_stats_line("chunked_prefetch_only_large", stats_of(chunked_prefetch_only_ms), args.bytes);
    print_stats_line("queued_chunked_prefetch_only_large", stats_of(queued_chunked_prefetch_only_ms), args.bytes);
    print_stats_line("fault_read_only_large", stats_of(fault_read_only_ms), args.bytes);
    print_stats_line("same_range_overlap_read", stats_of(same_read_ms), args.bytes);
    print_stats_line("same_range_overlap_prefetch", stats_of(same_prefetch_ms), args.bytes);
    print_stats_line("same_range_overlap_total", stats_of(same_total_ms), args.bytes);
    print_stats_line("two_range_overlap_fault_read", stats_of(two_range_read_ms), args.bytes);
    print_stats_line("two_range_overlap_prefetch", stats_of(two_range_prefetch_ms), args.bytes);
    print_stats_line("two_range_overlap_total", stats_of(two_range_total_ms), args.bytes);
    print_stats_line("small_fault_only", stats_of(small_fault_only_ms), args.small_bytes);
    print_stats_line("small_fault_during_large_prefetch", stats_of(small_during_large_read_ms), args.small_bytes);
    print_stats_line("large_prefetch_during_small_fault", stats_of(small_during_large_prefetch_ms), args.bytes);
    print_stats_line("small_fault_large_prefetch_total", stats_of(small_during_large_total_ms), args.bytes);
    print_stats_line("small_fault_during_chunked_prefetch", stats_of(small_during_chunked_read_ms), args.small_bytes);
    print_stats_line("chunked_prefetch_during_small_fault", stats_of(chunked_prefetch_ms), args.bytes);
    print_stats_line("small_fault_chunked_prefetch_total", stats_of(small_during_chunked_total_ms), args.bytes);
    print_stats_line("small_fault_during_queued_chunked", stats_of(small_during_queued_chunked_read_ms), args.small_bytes);
    print_stats_line("queued_chunked_during_small_fault", stats_of(queued_chunked_prefetch_ms), args.bytes);
    print_stats_line("small_fault_queued_chunked_total", stats_of(small_during_queued_chunked_total_ms), args.bytes);

    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaStreamDestroy(setup_stream));
    CUDA_CHECK(cudaStreamDestroy(control_stream));
    CUDA_CHECK(cudaStreamDestroy(prefetch_stream));
    CUDA_CHECK(cudaStreamDestroy(read_stream));
    return 0;
}
