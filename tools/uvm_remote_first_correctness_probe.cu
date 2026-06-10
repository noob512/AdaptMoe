#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr,                                             \
                         "CUDA error %s:%d: %s\n",                         \
                         __FILE__,                                           \
                         __LINE__,                                           \
                         cudaGetErrorString(err__));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

struct Options {
    size_t pages = 4096;
    size_t write_pages = 512;
    int passes = 4;
    int ops_per_page = 16;
    int streams = 4;
    int device = 0;
    int threads = 256;
    int blocks_per_sm = 4;
};

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static size_t parse_size(const char *s)
{
    char *end = nullptr;
    unsigned long long value = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "Invalid integer: %s\n", s);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(value);
}

static void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--pages N] [--write-pages N] [--passes N]\n"
                 "          [--ops-per-page N] [--streams N] [--device N]\n"
                 "          [--threads N] [--blocks-per-sm N]\n",
                 argv0);
}

static Options parse_args(int argc, char **argv)
{
    Options opt;

    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--pages") == 0)
            opt.pages = parse_size(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--write-pages") == 0)
            opt.write_pages = parse_size(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--passes") == 0)
            opt.passes = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--ops-per-page") == 0)
            opt.ops_per_page = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--streams") == 0)
            opt.streams = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--device") == 0)
            opt.device = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--threads") == 0)
            opt.threads = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--blocks-per-sm") == 0)
            opt.blocks_per_sm = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (opt.pages == 0 || opt.passes <= 0 || opt.ops_per_page <= 0 || opt.streams <= 0) {
        std::fprintf(stderr, "pages, passes, ops_per_page, and streams must be positive\n");
        std::exit(EXIT_FAILURE);
    }

    opt.write_pages = std::min(opt.write_pages, opt.pages);
    return opt;
}

static __host__ __device__ uint64_t initial_value(size_t page)
{
    return 0x9e3779b97f4a7c15ull ^ static_cast<uint64_t>(page * 11400714819323198485ull);
}

static __host__ __device__ uint64_t cpu_update_value(size_t page)
{
    return 0xfeed000000000000ull ^ static_cast<uint64_t>(page * 0x100000001b3ull);
}

static __host__ __device__ uint64_t gpu_update_value(size_t page)
{
    return 0xbeef000000000000ull ^ static_cast<uint64_t>(page * 0x9e3779b185ebca87ull);
}

static void init_page_heads(uint64_t *data, size_t pages, size_t elems_per_page)
{
    for (size_t page = 0; page < pages; ++page)
        data[page * elems_per_page] = initial_value(page);
}

static void cpu_update_page_heads(uint64_t *data, size_t pages, size_t elems_per_page)
{
    for (size_t page = 0; page < pages; ++page)
        data[page * elems_per_page] = cpu_update_value(page);
}

static uint64_t expected_checksum(const uint64_t *data, size_t pages, size_t elems_per_page, int ops_per_page)
{
    uint64_t sum = 0;
    for (size_t page = 0; page < pages; ++page) {
        const uint64_t value = data[page * elems_per_page];
        for (int op = 0; op < ops_per_page; ++op)
            sum += value;
    }
    return sum;
}

static uint64_t expected_generated_checksum(size_t pages,
                                            size_t write_pages,
                                            int ops_per_page,
                                            uint64_t (*written_value)(size_t))
{
    uint64_t sum = 0;

    for (size_t page = 0; page < pages; ++page) {
        const uint64_t value = page < write_pages ? written_value(page) : initial_value(page);
        for (int op = 0; op < ops_per_page; ++op)
            sum += value;
    }

    return sum;
}

__global__ void checksum_page_heads_kernel(const uint64_t *data,
                                           uint64_t *partials,
                                           size_t pages,
                                           size_t elems_per_page,
                                           int ops_per_page)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = gridDim.x * blockDim.x;
    const volatile uint64_t *volatile_data = data;
    uint64_t acc = 0;

    for (size_t page = tid; page < pages; page += grid) {
        const size_t index = page * elems_per_page;
        for (int op = 0; op < ops_per_page; ++op)
            acc += volatile_data[index];
    }

    partials[tid] = acc;
}

__global__ void gpu_update_page_heads_kernel(uint64_t *data, size_t pages, size_t elems_per_page)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = gridDim.x * blockDim.x;

    for (size_t page = tid; page < pages; page += grid)
        data[page * elems_per_page] = gpu_update_value(page);
}

__global__ void warmup_kernel(uint64_t *partials, size_t count)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count)
        partials[tid] = tid;
}

static uint64_t reduce_partials(uint64_t *partials, size_t partial_count)
{
    std::vector<uint64_t> host(partial_count);
    CHECK_CUDA(cudaMemcpy(host.data(), partials, partial_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    uint64_t sum = 0;
    for (uint64_t value : host)
        sum += value;
    return sum;
}

static uint64_t run_gpu_checksum(const uint64_t *data,
                                 uint64_t *partials,
                                 size_t partial_count,
                                 int blocks,
                                 int threads,
                                 size_t pages,
                                 size_t elems_per_page,
                                 int ops_per_page,
                                 cudaStream_t stream = nullptr)
{
    checksum_page_heads_kernel<<<blocks, threads, 0, stream>>>(data, partials, pages, elems_per_page, ops_per_page);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return reduce_partials(partials, partial_count);
}

static bool report(const char *test, uint64_t observed, uint64_t expected, double wall_ms)
{
    const bool passed = observed == expected;
    std::printf("%s,%s,%llu,%llu,%.6f\n",
                test,
                passed ? "PASS" : "FAIL",
                static_cast<unsigned long long>(observed),
                static_cast<unsigned long long>(expected),
                wall_ms);
    return passed;
}

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);

    CHECK_CUDA(cudaSetDevice(opt.device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, opt.device));

    const long os_page_size = sysconf(_SC_PAGESIZE);
    if (os_page_size <= 0 || (os_page_size & (os_page_size - 1)) != 0) {
        std::fprintf(stderr, "Unsupported OS page size: %ld\n", os_page_size);
        return EXIT_FAILURE;
    }

    const size_t page_size = static_cast<size_t>(os_page_size);
    const size_t elems_per_page = page_size / sizeof(uint64_t);
    const size_t elems = opt.pages * elems_per_page;
    const size_t bytes = elems * sizeof(uint64_t);
    const int blocks = prop.multiProcessorCount * opt.blocks_per_sm;
    const size_t partial_count = static_cast<size_t>(blocks) * opt.threads;

    uint64_t *data = nullptr;
    uint64_t *partials = nullptr;
    bool all_passed = true;

    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    init_page_heads(data, opt.pages, elems_per_page);
    CHECK_CUDA(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMalloc(&partials, partial_count * sizeof(uint64_t)));
    warmup_kernel<<<blocks, opt.threads>>>(partials, partial_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, opt.device));

    std::printf("test,status,observed,expected,wall_ms\n");

    for (int pass = 0; pass < opt.passes; ++pass) {
        const uint64_t expected = expected_generated_checksum(opt.pages,
                                                              0,
                                                              opt.ops_per_page,
                                                              initial_value);
        const double start = now_ms();
        const uint64_t observed = run_gpu_checksum(data,
                                                   partials,
                                                   partial_count,
                                                   blocks,
                                                   opt.threads,
                                                   opt.pages,
                                                   elems_per_page,
                                                   opt.ops_per_page);
        char name[64];
        std::snprintf(name, sizeof(name), "gpu_read_pass_%d", pass);
        all_passed &= report(name, observed, expected, now_ms() - start);
    }

    {
        const double start = now_ms();
        const uint64_t observed = expected_checksum(data, opt.pages, elems_per_page, opt.ops_per_page);
        const uint64_t expected = expected_generated_checksum(opt.pages,
                                                              0,
                                                              opt.ops_per_page,
                                                              initial_value);
        all_passed &= report("cpu_read_after_promotion", observed, expected, now_ms() - start);
    }

    {
        const double start = now_ms();
        cpu_update_page_heads(data, opt.write_pages, elems_per_page);
        const uint64_t expected = expected_generated_checksum(opt.pages,
                                                              opt.write_pages,
                                                              opt.ops_per_page,
                                                              cpu_update_value);
        const uint64_t observed = run_gpu_checksum(data,
                                                   partials,
                                                   partial_count,
                                                   blocks,
                                                   opt.threads,
                                                   opt.pages,
                                                   elems_per_page,
                                                   opt.ops_per_page);
        all_passed &= report("cpu_write_then_gpu_read", observed, expected, now_ms() - start);
    }

    {
        const double start = now_ms();
        gpu_update_page_heads_kernel<<<blocks, opt.threads>>>(data, opt.write_pages, elems_per_page);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        const uint64_t observed = expected_checksum(data, opt.pages, elems_per_page, opt.ops_per_page);
        const uint64_t expected = expected_generated_checksum(opt.pages,
                                                              opt.write_pages,
                                                              opt.ops_per_page,
                                                              gpu_update_value);
        all_passed &= report("gpu_write_then_cpu_read", observed, expected, now_ms() - start);
    }

    {
        std::vector<cudaStream_t> streams(opt.streams);
        uint64_t *multi_partials = nullptr;
        CHECK_CUDA(cudaMalloc(&multi_partials, partial_count * opt.streams * sizeof(uint64_t)));

        for (int i = 0; i < opt.streams; ++i)
            CHECK_CUDA(cudaStreamCreate(&streams[i]));

        const uint64_t expected = expected_generated_checksum(opt.pages,
                                                              opt.write_pages,
                                                              opt.ops_per_page,
                                                              gpu_update_value);
        const double start = now_ms();
        for (int i = 0; i < opt.streams; ++i) {
            checksum_page_heads_kernel<<<blocks, opt.threads, 0, streams[i]>>>(
                data,
                multi_partials + partial_count * i,
                opt.pages,
                elems_per_page,
                opt.ops_per_page);
            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        bool multi_passed = true;
        uint64_t first_observed = 0;
        for (int i = 0; i < opt.streams; ++i) {
            const uint64_t observed = reduce_partials(multi_partials + partial_count * i, partial_count);
            if (i == 0)
                first_observed = observed;
            multi_passed &= observed == expected;
        }

        std::printf("multi_stream_reads,%s,%llu,%llu,%.6f\n",
                    multi_passed ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(first_observed),
                    static_cast<unsigned long long>(expected),
                    now_ms() - start);
        all_passed &= multi_passed;

        for (cudaStream_t stream : streams)
            CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_CUDA(cudaFree(multi_partials));
    }

    CHECK_CUDA(cudaFree(partials));
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaDeviceReset());

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
