#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

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
    std::string profile = "remote_first_bg";
    size_t pages = 16384;
    int passes = 6;
    int ops_per_page = 128;
    int repeat = 0;
    int device = 0;
    int threads = 256;
    int blocks_per_sm = 4;
    bool print_header = true;
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
                 "Usage: %s --profile NAME [--pages N] [--passes N]\n"
                 "          [--ops-per-page N] [--repeat N] [--device N]\n"
                 "          [--threads N] [--blocks-per-sm N] [--no-header]\n"
                 "\n"
                 "Profiles:\n"
                 "  remote_first_bg    CPU-resident ReadMostly; driver remote-first + bg promotion\n"
                 "  remote_first_only  Same CUDA setup; script disables bg promotion\n"
                 "  readonly_dup_sync  Same CUDA setup; script disables remote-first for native read-dup\n"
                 "  readmostly_fault   Backward-compatible alias of remote_first_bg\n"
                 "  counter_migrate    CPU preferred + AccessedBy; access-counter migration candidate\n"
                 "  local_prefetch     ReadMostly plus synchronous GPU prefetch; local baseline\n"
                 "  default_uvm        No advice; original on-demand migration control\n",
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

        if (std::strcmp(argv[i], "--profile") == 0)
            opt.profile = need_value(argv[i]);
        else if (std::strcmp(argv[i], "--pages") == 0)
            opt.pages = parse_size(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--passes") == 0)
            opt.passes = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--ops-per-page") == 0)
            opt.ops_per_page = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--repeat") == 0)
            opt.repeat = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--device") == 0)
            opt.device = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--threads") == 0)
            opt.threads = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--blocks-per-sm") == 0)
            opt.blocks_per_sm = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--no-header") == 0)
            opt.print_header = false;
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

    if (opt.profile != "remote_first_bg" &&
        opt.profile != "remote_first_only" &&
        opt.profile != "readonly_dup_sync" &&
        opt.profile != "readmostly_fault" &&
        opt.profile != "counter_migrate" &&
        opt.profile != "local_prefetch" &&
        opt.profile != "default_uvm") {
        std::fprintf(stderr, "Unknown profile: %s\n", opt.profile.c_str());
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    if (opt.pages == 0 || opt.passes <= 0 || opt.ops_per_page <= 0) {
        std::fprintf(stderr, "pages, passes, and ops_per_page must be positive\n");
        std::exit(EXIT_FAILURE);
    }

    return opt;
}

__global__ void readonly_probe_kernel(const uint64_t *data,
                                      uint64_t *partials,
                                      size_t pages,
                                      size_t elems_per_page,
                                      int ops_per_page,
                                      uint64_t salt)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = gridDim.x * blockDim.x;
    const volatile uint64_t *volatile_data = data;
    uint64_t acc = salt + tid;

    for (size_t page = tid; page < pages; page += grid) {
        const size_t base = page * elems_per_page;
        for (int op = 0; op < ops_per_page; ++op) {
            const size_t offset = (static_cast<size_t>(op) * 131u + tid) & (elems_per_page - 1u);
            acc += volatile_data[base + offset];
        }
    }

    partials[tid] = acc;
}

__global__ void warmup_kernel(uint64_t *partials, size_t count)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count)
        partials[tid] = tid;
}

static void init_host_data(uint64_t *data, size_t elems)
{
    for (size_t i = 0; i < elems; ++i)
        data[i] = 0x9e3779b97f4a7c15ull ^ static_cast<uint64_t>(i * 11400714819323198485ull);
}

static bool is_readmostly_fault_profile(const std::string &profile)
{
    return profile == "remote_first_bg" ||
           profile == "remote_first_only" ||
           profile == "readonly_dup_sync" ||
           profile == "readmostly_fault";
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
    if (elems_per_page == 0 || (elems_per_page & (elems_per_page - 1)) != 0) {
        std::fprintf(stderr, "Unsupported elems_per_page: %zu\n", elems_per_page);
        return EXIT_FAILURE;
    }

    const size_t elems = opt.pages * elems_per_page;
    const size_t bytes = elems * sizeof(uint64_t);
    const int blocks = prop.multiProcessorCount * opt.blocks_per_sm;
    const size_t partial_count = static_cast<size_t>(blocks) * opt.threads;

    uint64_t *data = nullptr;
    uint64_t *partials = nullptr;

    const double setup_start = now_ms();

    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    init_host_data(data, elems);
    CHECK_CUDA(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMalloc(&partials, partial_count * sizeof(uint64_t)));
    warmup_kernel<<<blocks, opt.threads>>>(partials, partial_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (is_readmostly_fault_profile(opt.profile) || opt.profile == "local_prefetch")
        CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, opt.device));

    if (opt.profile == "counter_migrate") {
        CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, opt.device));
    }

    if (opt.profile == "local_prefetch") {
        CHECK_CUDA(cudaMemPrefetchAsync(data, bytes, opt.device));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    const double setup_ms = now_ms() - setup_start;

    if (opt.print_header) {
        std::printf("profile,repeat,pass,pages,ops_per_page,setup_ms,wall_ms,event_ms,mloads_s,gib_s\n");
    }

    const uint64_t logical_loads = static_cast<uint64_t>(opt.pages) * static_cast<uint64_t>(opt.ops_per_page);
    const double logical_gib = (static_cast<double>(logical_loads) * sizeof(uint64_t)) / (1024.0 * 1024.0 * 1024.0);

    for (int pass = 0; pass < opt.passes; ++pass) {
        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        const double wall_start = now_ms();
        CHECK_CUDA(cudaEventRecord(ev_start));
        readonly_probe_kernel<<<blocks, opt.threads>>>(
            data,
            partials,
            opt.pages,
            elems_per_page,
            opt.ops_per_page,
            0x12340000ull + static_cast<uint64_t>(opt.repeat * 131 + pass));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaDeviceSynchronize());
        const double wall_ms = now_ms() - wall_start;

        float event_ms_f = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&event_ms_f, ev_start, ev_stop));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));

        const double mloads_s = wall_ms > 0.0 ? static_cast<double>(logical_loads) / (wall_ms * 1000.0) : 0.0;
        const double gib_s = wall_ms > 0.0 ? logical_gib / (wall_ms / 1000.0) : 0.0;

        std::printf("%s,%d,%d,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    opt.profile.c_str(),
                    opt.repeat,
                    pass,
                    opt.pages,
                    opt.ops_per_page,
                    setup_ms,
                    wall_ms,
                    static_cast<double>(event_ms_f),
                    mloads_s,
                    gib_s);
    }

    CHECK_CUDA(cudaFree(partials));
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaDeviceReset());
    return EXIT_SUCCESS;
}
