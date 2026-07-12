#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct Args {
    int device = 0;
    int expert_count = 8;
    size_t expert_bytes = 64ull * 1024ull * 1024ull;
    size_t probe_bytes = 256ull * 1024ull * 1024ull;
    size_t target_free_before_probe = 0;
    size_t pressure_chunk_bytes = 256ull * 1024ull * 1024ull;
    bool skip_before_probe = false;
    bool keep_after_probe = false;
};

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

static size_t parse_size(const char *value)
{
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 0);
    if (end == value) {
        std::fprintf(stderr, "invalid size: %s\n", value);
        std::exit(2);
    }

    if (*end == 'k' || *end == 'K')
        parsed *= 1024ull;
    else if (*end == 'm' || *end == 'M')
        parsed *= 1024ull * 1024ull;
    else if (*end == 'g' || *end == 'G')
        parsed *= 1024ull * 1024ull * 1024ull;
    else if (*end != '\0') {
        std::fprintf(stderr, "invalid size suffix: %s\n", value);
        std::exit(2);
    }

    return (size_t)parsed;
}

static Args parse_args(int argc, char **argv)
{
    Args args;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--expert-count") == 0 && i + 1 < argc)
            args.expert_count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--expert-bytes") == 0 && i + 1 < argc)
            args.expert_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--probe-bytes") == 0 && i + 1 < argc)
            args.probe_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--target-free-before-probe") == 0 &&
                 i + 1 < argc)
            args.target_free_before_probe = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--pressure-chunk-bytes") == 0 &&
                 i + 1 < argc)
            args.pressure_chunk_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--skip-before-probe") == 0)
            args.skip_before_probe = true;
        else if (std::strcmp(argv[i], "--keep-after-probe") == 0)
            args.keep_after_probe = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--expert-count N] "
                         "[--expert-bytes N|NM|NG] [--probe-bytes N|NM|NG] "
                         "[--target-free-before-probe N|NM|NG] "
                         "[--pressure-chunk-bytes N|NM|NG] "
                         "[--skip-before-probe] [--keep-after-probe]\n",
                         argv[0]);
            std::exit(2);
        }
    }

    if (args.target_free_before_probe == 0)
        args.target_free_before_probe = std::max<size_t>(
            16ull * 1024ull * 1024ull,
            args.probe_bytes / 2);

    if (args.expert_count <= 0 ||
        args.expert_bytes == 0 ||
        args.probe_bytes == 0 ||
        args.pressure_chunk_bytes == 0 ||
        (size_t)args.expert_count * args.expert_bytes < args.probe_bytes) {
        std::fprintf(stderr,
                     "invalid arguments: offloaded expert bytes should be >= "
                     "probe bytes for a clean before/after test\n");
        std::exit(2);
    }

    return args;
}

static void print_mem(const char *label)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("cuda_mem label=%s free_bytes=%zu total_bytes=%zu used_bytes=%zu\n",
                label,
                free_bytes,
                total_bytes,
                total_bytes - free_bytes);
}

static unsigned int expert_seed(int expert)
{
    return 0x41c50000u ^ (0x01010101u * (unsigned int)(expert + 1));
}

static void fill_expert(unsigned int *ptr, size_t bytes, int expert)
{
    const size_t words = bytes / sizeof(unsigned int);
    const unsigned int seed = expert_seed(expert);
    for (size_t i = 0; i < words; ++i)
        ptr[i] = seed ^ (unsigned int)i;
}

static size_t allocate_pressure(size_t target_free,
                                size_t max_chunk_bytes,
                                std::vector<void *> *pressure_allocs)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const size_t min_chunk = 1ull * 1024ull * 1024ull;
    size_t chunk_limit = std::max(max_chunk_bytes, min_chunk);
    size_t allocated = 0;

    while (true) {
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes <= target_free + min_chunk)
            break;

        size_t wanted = free_bytes - target_free;
        size_t chunk = std::min(wanted, chunk_limit);
        chunk = (chunk / min_chunk) * min_chunk;
        if (chunk < min_chunk)
            break;

        void *ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, chunk);
        if (err == cudaSuccess) {
            pressure_allocs->push_back(ptr);
            allocated += chunk;
            continue;
        }

        cudaGetLastError();
        if (chunk_limit <= min_chunk)
            break;
        chunk_limit = std::max(min_chunk, chunk_limit / 2);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("pressure_alloc target_free=%zu allocated_bytes=%zu chunks=%zu "
                "free_after=%zu total_bytes=%zu\n",
                target_free,
                allocated,
                pressure_allocs->size(),
                free_bytes,
                total_bytes);
    return allocated;
}

static bool probe_malloc(const char *label, size_t bytes, void **out_ptr)
{
    size_t free_before = 0;
    size_t total_before = 0;
    size_t free_after = 0;
    size_t total_after = 0;
    void *ptr = nullptr;

    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));
    double start = now_ms();
    cudaError_t err = cudaMalloc(&ptr, bytes);
    double elapsed_ms = now_ms() - start;
    bool ok = err == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        ptr = nullptr;
    }
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));

    std::printf("probe_malloc label=%s requested_bytes=%zu success=%u "
                "cuda_error=%d error_name=%s elapsed_ms=%.3f "
                "free_before=%zu free_after=%zu\n",
                label,
                bytes,
                ok ? 1u : 0u,
                (int)err,
                cudaGetErrorName(err),
                elapsed_ms,
                free_before,
                free_after);

    *out_ptr = ptr;
    return ok;
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    const size_t total_expert_bytes =
        (size_t)args.expert_count * args.expert_bytes;

    CUDA_CHECK(cudaSetDevice(args.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    std::printf("expert_offload_release_smoke_begin device=%d gpu=%s "
                "expert_count=%d expert_bytes=%zu total_expert_bytes=%zu "
                "probe_bytes=%zu target_free_before_probe=%zu "
                "pressure_chunk_bytes=%zu\n",
                args.device,
                prop.name,
                args.expert_count,
                args.expert_bytes,
                total_expert_bytes,
                args.probe_bytes,
                args.target_free_before_probe,
                args.pressure_chunk_bytes);
    print_mem("start");

    std::vector<unsigned int *> experts(args.expert_count, nullptr);
    std::vector<void *> pressure_allocs;
    void *before_probe = nullptr;
    void *after_probe = nullptr;

    double alloc_start = now_ms();
    for (int expert = 0; expert < args.expert_count; ++expert) {
        CUDA_CHECK(cudaMallocManaged(&experts[expert], args.expert_bytes));
        fill_expert(experts[expert], args.expert_bytes, expert);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_alloc_managed experts=%d bytes=%zu elapsed_ms=%.3f\n",
                args.expert_count,
                total_expert_bytes,
                now_ms() - alloc_start);
    print_mem("after_managed_alloc_cpu_fill");

    double gpu_prefetch_start = now_ms();
    for (int expert = 0; expert < args.expert_count; ++expert) {
        CUDA_CHECK(cudaMemPrefetchAsync(experts[expert],
                                        args.expert_bytes,
                                        args.device));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_prefetch_gpu experts=%d bytes=%zu elapsed_ms=%.3f\n",
                args.expert_count,
                total_expert_bytes,
                now_ms() - gpu_prefetch_start);
    print_mem("after_expert_gpu_prefetch");

    allocate_pressure(args.target_free_before_probe,
                      args.pressure_chunk_bytes,
                      &pressure_allocs);
    print_mem("after_pressure_before_probe");

    bool before_ok = false;
    bool before_skipped = args.skip_before_probe;
    if (args.skip_before_probe) {
        std::printf("probe_malloc label=before_offload skipped=1\n");
    }
    else {
        before_ok = probe_malloc("before_offload", args.probe_bytes, &before_probe);
        if (before_probe) {
            CUDA_CHECK(cudaFree(before_probe));
            before_probe = nullptr;
        }
    }
    print_mem("after_before_probe_cleanup");

    double cpu_prefetch_start = now_ms();
    for (int expert = 0; expert < args.expert_count; ++expert) {
        CUDA_CHECK(cudaMemPrefetchAsync(experts[expert],
                                        args.expert_bytes,
                                        cudaCpuDeviceId));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_prefetch_cpu experts=%d bytes=%zu elapsed_ms=%.3f\n",
                args.expert_count,
                total_expert_bytes,
                now_ms() - cpu_prefetch_start);
    print_mem("after_expert_cpu_offload");

    bool after_ok = probe_malloc("after_offload", args.probe_bytes, &after_probe);
    if (after_probe && !args.keep_after_probe) {
        CUDA_CHECK(cudaFree(after_probe));
        after_probe = nullptr;
    }
    print_mem("after_after_probe");

    bool strong_evidence = (!before_ok || before_skipped) && after_ok;
    bool weak_evidence = after_ok;
    std::printf("expert_offload_release_smoke_final before_probe_success=%u "
                "before_probe_skipped=%u after_probe_success=%u "
                "strong_evidence=%u weak_evidence=%u offloaded_expert_bytes=%zu "
                "probe_bytes=%zu\n",
                before_ok ? 1u : 0u,
                before_skipped ? 1u : 0u,
                after_ok ? 1u : 0u,
                strong_evidence ? 1u : 0u,
                weak_evidence ? 1u : 0u,
                total_expert_bytes,
                args.probe_bytes);

    if (after_probe) {
        CUDA_CHECK(cudaFree(after_probe));
        after_probe = nullptr;
    }
    for (void *ptr : pressure_allocs)
        CUDA_CHECK(cudaFree(ptr));
    for (unsigned int *ptr : experts)
        CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("after_cleanup");

    return weak_evidence ? 0 : 1;
}
