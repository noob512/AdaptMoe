#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct Args {
    int device = 0;
    int regions = 10;
    size_t mib_per_region = 2;
    int repeats = 30;
    int warmups = 5;
    bool pinned = true;
    bool single_alloc = false;
};

static Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--regions") == 0 && i + 1 < argc)
            args.regions = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mib-per-region") == 0 && i + 1 < argc)
            args.mib_per_region = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            args.repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmups") == 0 && i + 1 < argc)
            args.warmups = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pageable") == 0)
            args.pinned = false;
        else if (std::strcmp(argv[i], "--pinned") == 0)
            args.pinned = true;
        else if (std::strcmp(argv[i], "--single-alloc") == 0)
            args.single_alloc = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--regions N] [--mib-per-region N] "
                         "[--repeats N] [--warmups N] [--pinned|--pageable] [--single-alloc]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    return args;
}

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static double percentile(std::vector<double> values, double p)
{
    std::sort(values.begin(), values.end());
    size_t idx = (size_t)((values.size() - 1) * p);
    return values[idx];
}

static void touch_cpu(unsigned char *ptr, size_t bytes, unsigned seed)
{
    for (size_t i = 0; i < bytes; i += 4096)
        ptr[i] = (unsigned char)(seed + i);
    if (bytes)
        ptr[bytes - 1] = (unsigned char)(seed ^ 0xa5);
}

static float event_elapsed(cudaEvent_t start, cudaEvent_t end)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    const size_t bytes_per_region = args.mib_per_region * 1024ull * 1024ull;
    const size_t total_bytes = bytes_per_region * (size_t)args.regions;

    std::vector<unsigned char *> host_regions;
    std::vector<unsigned char *> device_regions;
    std::vector<void *> host_allocs;
    std::vector<void *> device_allocs;

    auto alloc_host = [&](size_t bytes) -> unsigned char * {
        void *ptr = nullptr;
        if (args.pinned)
            CUDA_CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
        else {
            ptr = std::malloc(bytes);
            if (!ptr) {
                std::fprintf(stderr, "malloc failed\n");
                std::exit(1);
            }
        }
        host_allocs.push_back(ptr);
        return static_cast<unsigned char *>(ptr);
    };

    auto alloc_device = [&](size_t bytes) -> unsigned char * {
        void *ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        device_allocs.push_back(ptr);
        return static_cast<unsigned char *>(ptr);
    };

    if (args.single_alloc) {
        unsigned char *h = alloc_host(total_bytes);
        unsigned char *d = alloc_device(total_bytes);
        for (int i = 0; i < args.regions; ++i) {
            host_regions.push_back(h + (size_t)i * bytes_per_region);
            device_regions.push_back(d + (size_t)i * bytes_per_region);
        }
    }
    else {
        for (int i = 0; i < args.regions; ++i) {
            host_regions.push_back(alloc_host(bytes_per_region));
            device_regions.push_back(alloc_device(bytes_per_region));
        }
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<double> event_samples;
    std::vector<double> wall_samples;
    const int total_iters = args.warmups + args.repeats;
    for (int iter = 0; iter < total_iters; ++iter) {
        for (int i = 0; i < args.regions; ++i)
            touch_cpu(host_regions[i], bytes_per_region, (unsigned)(iter * 31 + i));

        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));

        const double wall_start = now_ms();
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < args.regions; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(device_regions[i],
                                       host_regions[i],
                                       bytes_per_region,
                                       cudaMemcpyHostToDevice,
                                       stream));
        }
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double wall_ms = now_ms() - wall_start;
        const double event_ms = event_elapsed(start, end);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));

        if (iter >= args.warmups) {
            event_samples.push_back(event_ms);
            wall_samples.push_back(wall_ms);
            std::printf("sample iter=%d event_ms=%.3f wall_ms=%.3f event_gib_s=%.3f wall_gib_s=%.3f\n",
                        iter - args.warmups,
                        event_ms,
                        wall_ms,
                        (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (event_ms / 1000.0),
                        (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (wall_ms / 1000.0));
        }
    }

    const double event_p50 = percentile(event_samples, 0.50);
    const double wall_p50 = percentile(wall_samples, 0.50);
    const double event_mean = std::accumulate(event_samples.begin(), event_samples.end(), 0.0) / event_samples.size();
    const double wall_mean = std::accumulate(wall_samples.begin(), wall_samples.end(), 0.0) / wall_samples.size();

    std::printf("summary device=%s regions=%d mib_per_region=%zu total_mib=%.2f pinned=%d single_alloc=%d "
                "event_p50_ms=%.3f event_mean_ms=%.3f event_p90_ms=%.3f "
                "wall_p50_ms=%.3f wall_mean_ms=%.3f wall_p90_ms=%.3f "
                "event_p50_gib_s=%.3f wall_p50_gib_s=%.3f\n",
                prop.name,
                args.regions,
                args.mib_per_region,
                (double)total_bytes / (1024.0 * 1024.0),
                args.pinned ? 1 : 0,
                args.single_alloc ? 1 : 0,
                event_p50,
                event_mean,
                percentile(event_samples, 0.90),
                wall_p50,
                wall_mean,
                percentile(wall_samples, 0.90),
                (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (event_p50 / 1000.0),
                (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (wall_p50 / 1000.0));

    CUDA_CHECK(cudaStreamDestroy(stream));
    for (void *ptr : device_allocs)
        CUDA_CHECK(cudaFree(ptr));
    for (void *ptr : host_allocs) {
        if (args.pinned)
            CUDA_CHECK(cudaFreeHost(ptr));
        else
            std::free(ptr);
    }
    return 0;
}
