#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

#include "uvm_linux_ioctl.h"

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

#ifndef UVM_PHASE_F_ACTION_QUERY_RESIDENCY
#define UVM_PHASE_F_ACTION_QUERY_RESIDENCY 4
#endif

namespace {

struct Options {
    std::string mode = "remote_cpu";
    size_t pages = 8192;
    int passes = 3;
    int device = 0;
    int threads = 256;
    int blocks_per_sm = 4;
    int repeat = 0;
    bool print_header = true;
};

struct QueryResult {
    bool ok = false;
    uint32_t rm_status = 0;
    uint32_t range_status = 0;
    uint64_t resident_gpu_bytes = 0;
    uint64_t resident_cpu_bytes = 0;
    uint32_t resident_on_gpu = 0;
    uint32_t resident_on_cpu = 0;
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
                 "Usage: %s --mode remote_cpu|prefetch_gpu|fault_migrate [--pages N]\n"
                 "          [--passes N] [--device N] [--threads N]\n"
                 "          [--blocks-per-sm N] [--repeat N] [--no-header]\n"
                 "\n"
                 "Modes:\n"
                 "  remote_cpu     CPU resident + PreferredLocation(CPU) + AccessedBy(GPU); GPU writes remotely\n"
                 "  prefetch_gpu   CPU resident, then timed cudaMemPrefetchAsync(..., GPU), then GPU writes locally\n"
                 "  fault_migrate  CPU resident, no hints; first GPU write faults and migrates to GPU\n",
                 argv0);
}

static bool valid_mode(const std::string &mode)
{
    return mode == "remote_cpu" || mode == "prefetch_gpu" || mode == "fault_migrate";
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

        if (std::strcmp(argv[i], "--mode") == 0)
            opt.mode = need_value(argv[i]);
        else if (std::strcmp(argv[i], "--pages") == 0)
            opt.pages = parse_size(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--passes") == 0)
            opt.passes = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--device") == 0)
            opt.device = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--threads") == 0)
            opt.threads = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--blocks-per-sm") == 0)
            opt.blocks_per_sm = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--repeat") == 0)
            opt.repeat = static_cast<int>(parse_size(need_value(argv[i])));
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

    if (!valid_mode(opt.mode)) {
        std::fprintf(stderr, "Invalid --mode: %s\n", opt.mode.c_str());
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    if (opt.pages == 0 || opt.passes <= 0 || opt.threads <= 0 || opt.blocks_per_sm <= 0) {
        std::fprintf(stderr, "--pages, --passes, --threads, and --blocks-per-sm must be positive\n");
        std::exit(EXIT_FAILURE);
    }

    return opt;
}

static bool get_device_uuid(int device, NvProcessorUuid *uuid)
{
    cudaDeviceProp prop{};
    cudaError_t err;

    if (!uuid)
        return false;

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
        return false;

    static_assert(sizeof(prop.uuid.bytes) == sizeof(uuid->uuid), "Unexpected CUDA UUID size");
    std::memcpy(uuid->uuid, prop.uuid.bytes, sizeof(uuid->uuid));
    return true;
}

static bool is_nvidia_uvm_fd_target(const char *target)
{
    const char *name;
    size_t len;

    if (!target)
        return false;

    name = std::strrchr(target, '/');
    name = name ? name + 1 : target;
    len = std::strlen("nvidia-uvm");

    return std::strncmp(name, "nvidia-uvm", len) == 0 && (name[len] == '\0' || name[len] == ' ');
}

static int duplicate_existing_uvm_fd()
{
    DIR *dir = opendir("/proc/self/fd");
    struct dirent *entry;

    if (!dir)
        return -1;

    while ((entry = readdir(dir)) != nullptr) {
        char *end = nullptr;
        long fd_long = std::strtol(entry->d_name, &end, 10);
        char link_path[64];
        char target[PATH_MAX];
        ssize_t len;

        if (end == entry->d_name || *end != '\0')
            continue;

        std::snprintf(link_path, sizeof(link_path), "/proc/self/fd/%ld", fd_long);
        len = readlink(link_path, target, sizeof(target) - 1);
        if (len < 0)
            continue;

        target[len] = '\0';
        if (!is_nvidia_uvm_fd_target(target))
            continue;

        int dup_fd = fcntl(static_cast<int>(fd_long), F_DUPFD_CLOEXEC, 3);
        if (dup_fd >= 0) {
            closedir(dir);
            return dup_fd;
        }
    }

    closedir(dir);
    return -1;
}

static QueryResult query_residency(int fd, void *ptr, size_t bytes, int device)
{
    QueryResult query{};
    UVM_PHASE_F_BATCH_EXECUTOR_PARAMS params{};

    if (fd < 0)
        return query;

    params.action = UVM_PHASE_F_ACTION_QUERY_RESIDENCY;
    params.count = 1;
    params.async = NV_FALSE;
    params.ranges[0].base = reinterpret_cast<uint64_t>(ptr);
    params.ranges[0].length = static_cast<uint64_t>(bytes);

    if (!get_device_uuid(device, &params.gpu_uuid)) {
        query.range_status = 0xffffffffu;
        return query;
    }

    if (ioctl(fd, UVM_PHASE_F_BATCH_EXECUTOR, &params) != 0) {
        query.rm_status = params.rmStatus;
        query.range_status = errno;
        return query;
    }

    query.rm_status = params.rmStatus;
    query.range_status = params.results[0].status;
    query.ok = params.rmStatus == 0 && params.failure_count == 0 && params.results[0].status == 0;
    query.resident_gpu_bytes = params.results[0].resident_gpu_bytes;
    query.resident_cpu_bytes = params.results[0].resident_cpu_bytes;
    query.resident_on_gpu = params.results[0].resident_on_gpu ? 1 : 0;
    query.resident_on_cpu = params.results[0].resident_on_cpu ? 1 : 0;

    return query;
}

static void init_host_data(uint64_t *data, size_t elems)
{
    for (size_t i = 0; i < elems; ++i)
        data[i] = 0x1234000000000000ULL ^ static_cast<uint64_t>(i * 1315423911ULL);
}

__global__ void write_stream_kernel(uint64_t *data, size_t elems, uint64_t salt)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = blockDim.x * gridDim.x;

    for (size_t i = tid; i < elems; i += grid)
        data[i] = salt + static_cast<uint64_t>(i);
}

} // namespace

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);

    CHECK_CUDA(cudaSetDevice(opt.device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, opt.device));

    const long os_page_size = sysconf(_SC_PAGESIZE);
    if (os_page_size <= 0) {
        std::fprintf(stderr, "Invalid OS page size: %ld\n", os_page_size);
        return EXIT_FAILURE;
    }

    const size_t page_size = static_cast<size_t>(os_page_size);
    const size_t elems = (opt.pages * page_size) / sizeof(uint64_t);
    const size_t bytes = elems * sizeof(uint64_t);
    const int blocks = std::max(1, prop.multiProcessorCount * opt.blocks_per_sm);

    uint64_t *data = nullptr;
    CHECK_CUDA(cudaMallocManaged(&data, bytes));

    const double setup_start_ms = now_ms();
    init_host_data(data, elems);
    CHECK_CUDA(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());

    double prepare_ms = 0.0;
    if (opt.mode == "remote_cpu") {
        const double prepare_start_ms = now_ms();
        CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, opt.device));
        CHECK_CUDA(cudaDeviceSynchronize());
        prepare_ms = now_ms() - prepare_start_ms;
    }
    else if (opt.mode == "prefetch_gpu") {
        const double prepare_start_ms = now_ms();
        CHECK_CUDA(cudaMemPrefetchAsync(data, bytes, opt.device));
        CHECK_CUDA(cudaDeviceSynchronize());
        prepare_ms = now_ms() - prepare_start_ms;
    }
    const double setup_ms = now_ms() - setup_start_ms;

    int uvm_fd = duplicate_existing_uvm_fd();
    QueryResult query_before = query_residency(uvm_fd, data, bytes, opt.device);

    if (opt.print_header) {
        std::printf("mode,repeat,pass,pages,bytes,setup_ms,prepare_ms,wall_ms,event_ms,write_gib,wall_gib_s,event_gib_s,query_before_ok,query_before_cpu_bytes,query_before_gpu_bytes,query_before_on_cpu,query_before_on_gpu,query_after_ok,query_after_cpu_bytes,query_after_gpu_bytes,query_after_on_cpu,query_after_on_gpu\n");
    }

    const double write_gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

    for (int pass = 0; pass < opt.passes; ++pass) {
        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        const uint64_t salt = 0xfeed000000000000ULL ^
                              (static_cast<uint64_t>(opt.repeat) << 32) ^
                              static_cast<uint64_t>(pass);

        const double wall_start_ms = now_ms();
        CHECK_CUDA(cudaEventRecord(ev_start));
        write_stream_kernel<<<blocks, opt.threads>>>(data, elems, salt);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaDeviceSynchronize());
        const double wall_ms = now_ms() - wall_start_ms;

        float event_ms_f = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&event_ms_f, ev_start, ev_stop));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));

        QueryResult query_after = query_residency(uvm_fd, data, bytes, opt.device);

        const double event_ms = static_cast<double>(event_ms_f);
        const double wall_gib_s = wall_ms > 0.0 ? write_gib / (wall_ms / 1000.0) : 0.0;
        const double event_gib_s = event_ms > 0.0 ? write_gib / (event_ms / 1000.0) : 0.0;

        std::printf("%s,%d,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%u,%llu,%llu,%u,%u,%u,%llu,%llu,%u,%u\n",
                    opt.mode.c_str(),
                    opt.repeat,
                    pass,
                    opt.pages,
                    bytes,
                    setup_ms,
                    prepare_ms,
                    wall_ms,
                    event_ms,
                    write_gib,
                    wall_gib_s,
                    event_gib_s,
                    query_before.ok ? 1 : 0,
                    static_cast<unsigned long long>(query_before.resident_cpu_bytes),
                    static_cast<unsigned long long>(query_before.resident_gpu_bytes),
                    query_before.resident_on_cpu,
                    query_before.resident_on_gpu,
                    query_after.ok ? 1 : 0,
                    static_cast<unsigned long long>(query_after.resident_cpu_bytes),
                    static_cast<unsigned long long>(query_after.resident_gpu_bytes),
                    query_after.resident_on_cpu,
                    query_after.resident_on_gpu);

        query_before = query_after;
    }

    if (uvm_fd >= 0)
        close(uvm_fd);

    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
