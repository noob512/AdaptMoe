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

#ifndef UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT
#define UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT 14
#endif

#ifndef UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY
#define UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY 13
#endif

#ifndef UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT
#define UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT       (1U << 6)
#define UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING (1U << 8)
#define UVM_PHASE_N_REGION_FLAG_GPU_READ_ONLY       (1U << 9)
#endif

namespace {

struct Options {
    std::string mode = "readdup_soft";
    size_t pages = 8192;
    int device = 0;
    int iterations = 10;
    bool clear_after = true;
};

struct RegionSpec {
    uint64_t region_id = 0;
    void *ptr = nullptr;
    size_t bytes = 0;
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
                 "Usage: %s --mode baseline_prefetch|readdup_soft [--pages N]\n"
                 "          [--iterations N] [--device N] [--no-clear-after]\n",
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

        if (std::strcmp(argv[i], "--mode") == 0) {
            opt.mode = need_value(argv[i]);
        }
        else if (std::strcmp(argv[i], "--pages") == 0) {
            opt.pages = parse_size(need_value(argv[i]));
        }
        else if (std::strcmp(argv[i], "--iterations") == 0) {
            opt.iterations = static_cast<int>(parse_size(need_value(argv[i])));
        }
        else if (std::strcmp(argv[i], "--device") == 0) {
            opt.device = static_cast<int>(parse_size(need_value(argv[i])));
        }
        else if (std::strcmp(argv[i], "--no-clear-after") == 0) {
            opt.clear_after = false;
        }
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

    if (opt.mode != "baseline_prefetch" && opt.mode != "readdup_soft") {
        std::fprintf(stderr, "Invalid --mode: %s\n", opt.mode.c_str());
        std::exit(EXIT_FAILURE);
    }

    if (opt.pages == 0 || opt.iterations <= 0) {
        std::fprintf(stderr, "--pages and --iterations must be positive\n");
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

static uint64_t region_id_for(const void *ptr)
{
    uint64_t value = reinterpret_cast<uintptr_t>(ptr);
    value ^= 0x5700000300000000ULL + (value << 6) + (value >> 2);
    value &= 0x7fffffffffffffffULL;
    return value == 0 ? 0x57000003ULL : value;
}

static void register_weight_region_or_die(int fd, const RegionSpec &region)
{
    UVM_PHASE_F_REGISTER_REGION_PARAMS params{};

    params.region.region_id = region.region_id;
    params.region.base = reinterpret_cast<uint64_t>(region.ptr);
    params.region.length = static_cast<uint64_t>(region.bytes);
    params.region.pool = UVM_PHASE_N_POOL_WEIGHT;
    params.region.role = 1;
    params.region.flags = UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT |
                          UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING |
                          UVM_PHASE_N_REGION_FLAG_GPU_READ_ONLY;

    if (ioctl(fd, UVM_PHASE_F_REGISTER_REGION, &params) != 0 || params.rmStatus != 0) {
        std::fprintf(stderr,
                     "register_region failed errno=%d rmStatus=%u region_id=%llu base=0x%llx bytes=%zu\n",
                     errno,
                     params.rmStatus,
                     static_cast<unsigned long long>(region.region_id),
                     static_cast<unsigned long long>(params.region.base),
                     region.bytes);
        std::exit(EXIT_FAILURE);
    }
}

static void unregister_region(int fd, uint64_t region_id)
{
    UVM_PHASE_F_UNREGISTER_REGION_PARAMS params{};

    params.region_id = region_id;
    (void)ioctl(fd, UVM_PHASE_F_UNREGISTER_REGION, &params);
}

static UVM_PHASE_F_RANGE_RESULT run_region_action_or_die(int fd,
                                                         const RegionSpec &region,
                                                         uint32_t action,
                                                         int device)
{
    UVM_PHASE_F_BATCH_REGIONS_PARAMS params{};

    params.action = action;
    params.count = 1;
    params.async = NV_FALSE;
    params.region_ids[0] = region.region_id;

    if (!get_device_uuid(device, &params.gpu_uuid)) {
        std::fprintf(stderr, "Could not get CUDA device UUID for device=%d\n", device);
        std::exit(EXIT_FAILURE);
    }

    if (ioctl(fd, UVM_PHASE_F_BATCH_REGIONS, &params) != 0 ||
        params.rmStatus != 0 ||
        params.failure_count != 0 ||
        params.results[0].status != 0) {
        std::fprintf(stderr,
                     "phase_f_action failed action=%u errno=%d rmStatus=%u result_status=%u success=%u failure=%u\n",
                     action,
                     errno,
                     params.rmStatus,
                     params.results[0].status,
                     params.success_count,
                     params.failure_count);
        std::exit(EXIT_FAILURE);
    }

    return params.results[0];
}

__global__ void read_kernel(const uint64_t *data, size_t elems, uint64_t *out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = blockDim.x * gridDim.x;
    uint64_t acc = 0;

    for (size_t i = tid; i < elems; i += grid)
        acc += data[i] ^ static_cast<uint64_t>(i);

    if (acc)
        atomicAdd(reinterpret_cast<unsigned long long *>(out),
                  static_cast<unsigned long long>(acc & 0xffffULL));
}

static double run_gpu_read(uint64_t *data, size_t elems, uint64_t *out_device, uint64_t *checksum)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms = 0.0f;
    uint64_t out = 0;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaMemset(out_device, 0, sizeof(uint64_t)));
    CHECK_CUDA(cudaEventRecord(start));
    read_kernel<<<128, 256>>>(data, elems, out_device);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaMemcpy(&out, out_device, sizeof(out), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    if (checksum)
        *checksum += out;

    return static_cast<double>(ms);
}

static void cpu_fill(uint64_t *data, size_t elems)
{
    for (size_t i = 0; i < elems; ++i)
        data[i] = 0x100000000ULL + i;
}

static uint64_t cpu_touch_one_word_per_page(uint64_t *data, size_t pages)
{
    volatile uint64_t *volatile_data = data;
    const size_t stride = 4096 / sizeof(uint64_t);
    uint64_t acc = 0;

    for (size_t page = 0; page < pages; ++page) {
        const size_t index = page * stride;
        uint64_t value = volatile_data[index] + 1;
        volatile_data[index] = value;
        acc += value;
    }

    return acc;
}

} // namespace

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);
    const size_t page_size = 4096;
    const size_t elems_per_page = page_size / sizeof(uint64_t);
    const size_t elems = opt.pages * elems_per_page;
    const size_t bytes = elems * sizeof(uint64_t);
    uint64_t *data = nullptr;
    uint64_t *out_device = nullptr;
    uint64_t checksum = 0;
    uint64_t cpu_checksum = 0;
    double init_ms;
    double apply_ms = 0.0;
    double avg_cpu_restore_ms = 0.0;
    double avg_gpu_prefetch_ms = 0.0;
    double avg_gpu_read_ms = 0.0;
    double min_gpu_read_ms = 1.0e30;
    int fd = -1;
    bool adopted_existing_uvm_fd = false;
    UVM_PHASE_F_RANGE_RESULT apply_result{};
    UVM_PHASE_F_RANGE_RESULT query_result{};

    CHECK_CUDA(cudaSetDevice(opt.device));
    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    CHECK_CUDA(cudaMalloc(&out_device, sizeof(uint64_t)));

    double start = now_ms();
    cpu_fill(data, elems);
    init_ms = now_ms() - start;

    if (opt.mode == "readdup_soft") {
        RegionSpec region{};

        fd = duplicate_existing_uvm_fd();
        adopted_existing_uvm_fd = fd >= 0;
        if (fd < 0) {
            std::fprintf(stderr, "Could not duplicate CUDA /dev/nvidia-uvm fd\n");
            return EXIT_FAILURE;
        }

        region.region_id = region_id_for(data);
        region.ptr = data;
        region.bytes = bytes;
        register_weight_region_or_die(fd, region);

        start = now_ms();
        apply_result = run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT, opt.device);
        apply_ms = now_ms() - start;
        query_result = run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_QUERY_RESIDENCY, opt.device);

        for (int iter = 0; iter < opt.iterations; ++iter) {
            double read_ms = run_gpu_read(data, elems, out_device, &checksum);
            avg_gpu_read_ms += read_ms;
            min_gpu_read_ms = std::min(min_gpu_read_ms, read_ms);
        }

        if (opt.clear_after)
            (void)run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY, opt.device);

        unregister_region(fd, region.region_id);
        close(fd);
    }
    else {
        for (int iter = 0; iter < opt.iterations; ++iter) {
            start = now_ms();
            cpu_checksum += cpu_touch_one_word_per_page(data, opt.pages);
            avg_cpu_restore_ms += now_ms() - start;

            start = now_ms();
            CHECK_CUDA(cudaMemPrefetchAsync(data, bytes, opt.device));
            CHECK_CUDA(cudaDeviceSynchronize());
            avg_gpu_prefetch_ms += now_ms() - start;

            double read_ms = run_gpu_read(data, elems, out_device, &checksum);
            avg_gpu_read_ms += read_ms;
            min_gpu_read_ms = std::min(min_gpu_read_ms, read_ms);
        }
    }

    avg_gpu_read_ms /= opt.iterations;
    if (opt.mode == "baseline_prefetch") {
        avg_cpu_restore_ms /= opt.iterations;
        avg_gpu_prefetch_ms /= opt.iterations;
    }

    std::printf("stage_c_weight_unmap_bench mode=%s pages=%zu bytes=%zu iterations=%d adopted_existing_uvm_fd=%u init_ms=%.6f apply_ms=%.6f avg_cpu_restore_ms=%.6f avg_gpu_prefetch_ms=%.6f avg_gpu_read_ms=%.6f min_gpu_read_ms=%.6f apply_gpu_bytes=%llu apply_cpu_bytes=%llu query_gpu_bytes=%llu query_cpu_bytes=%llu query_on_cpu=%u query_on_gpu=%u checksum=%llu cpu_checksum=%llu\n",
                opt.mode.c_str(),
                opt.pages,
                bytes,
                opt.iterations,
                adopted_existing_uvm_fd ? 1 : 0,
                init_ms,
                apply_ms,
                avg_cpu_restore_ms,
                avg_gpu_prefetch_ms,
                avg_gpu_read_ms,
                min_gpu_read_ms,
                static_cast<unsigned long long>(apply_result.resident_gpu_bytes),
                static_cast<unsigned long long>(apply_result.resident_cpu_bytes),
                static_cast<unsigned long long>(query_result.resident_gpu_bytes),
                static_cast<unsigned long long>(query_result.resident_cpu_bytes),
                query_result.resident_on_cpu ? 1 : 0,
                query_result.resident_on_gpu ? 1 : 0,
                static_cast<unsigned long long>(checksum),
                static_cast<unsigned long long>(cpu_checksum));

    CHECK_CUDA(cudaFree(out_device));
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
