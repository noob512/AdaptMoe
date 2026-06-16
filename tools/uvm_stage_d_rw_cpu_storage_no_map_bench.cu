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
#include <vector>

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

#ifndef UVM_PHASE_F_ACTION_PREFETCH_CPU
#define UVM_PHASE_F_ACTION_PREFETCH_CPU          1
#define UVM_PHASE_F_ACTION_QUERY_RESIDENCY       4
#define UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT     10
#define UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY       13
#define UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP  15
#endif

#ifndef UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT
#define UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT       (1U << 6)
#define UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING (1U << 8)
#define UVM_PHASE_N_REGION_FLAG_GPU_READ_WRITE      (1U << 10)
#endif

namespace {

struct Options {
    std::string mode = "cpu_no_map";
    size_t pages = 8192;
    int device = 0;
    int iterations = 10;
    bool clear_after = true;
    std::string iteration_csv;
};

struct RegionSpec {
    uint64_t region_id = 0;
    void *ptr = nullptr;
    size_t bytes = 0;
};

struct Stats {
    double avg = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static Stats summarize(std::vector<double> values)
{
    Stats stats{};
    double sum = 0.0;

    if (values.empty())
        return stats;

    std::sort(values.begin(), values.end());
    for (double value : values)
        sum += value;

    auto percentile = [&](double q) {
        size_t index = static_cast<size_t>((values.size() - 1) * q);
        return values[index];
    };

    stats.avg = sum / static_cast<double>(values.size());
    stats.min = values.front();
    stats.p50 = percentile(0.50);
    stats.p90 = percentile(0.90);
    stats.p99 = percentile(0.99);
    stats.max = values.back();
    return stats;
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
                 "Usage: %s --mode baseline_cpu_map|cpu_no_map [--pages N]\n"
                 "          [--iterations N] [--device N] [--iteration-csv PATH]\n"
                 "          [--no-clear-after]\n",
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
        else if (std::strcmp(argv[i], "--iteration-csv") == 0) {
            opt.iteration_csv = need_value(argv[i]);
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

    if (opt.mode != "baseline_cpu_map" && opt.mode != "cpu_no_map") {
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
    value ^= 0x6b00000400000000ULL + (value << 6) + (value >> 2);
    value &= 0x7fffffffffffffffULL;
    return value == 0 ? 0x6b000004ULL : value;
}

static void register_kv_region_or_die(int fd, const RegionSpec &region)
{
    UVM_PHASE_F_REGISTER_REGION_PARAMS params{};

    params.region.region_id = region.region_id;
    params.region.base = reinterpret_cast<uint64_t>(region.ptr);
    params.region.length = static_cast<uint64_t>(region.bytes);
    params.region.pool = UVM_PHASE_N_POOL_KV;
    params.region.role = 2;
    params.region.flags = UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT |
                          UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING |
                          UVM_PHASE_N_REGION_FLAG_GPU_READ_WRITE;

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
                                                         int device,
                                                         const char *name)
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
                     "phase_f_action failed name=%s action=%u errno=%d rmStatus=%u result_status=%u success=%u failure=%u\n",
                     name,
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

__global__ void gpu_write_kernel(uint64_t *data, size_t elems, uint64_t addend, uint64_t *out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = blockDim.x * gridDim.x;
    uint64_t acc = 0;

    for (size_t i = tid; i < elems; i += grid) {
        uint64_t value = data[i] + addend + static_cast<uint64_t>(i & 7);
        data[i] = value;
        acc += value;
    }

    if (acc)
        atomicAdd(reinterpret_cast<unsigned long long *>(out),
                  static_cast<unsigned long long>(acc & 0xffffULL));
}

static double run_gpu_write(uint64_t *data, size_t elems, uint64_t addend, uint64_t *out_device, uint64_t *checksum)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms = 0.0f;
    uint64_t out = 0;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaMemset(out_device, 0, sizeof(uint64_t)));
    CHECK_CUDA(cudaEventRecord(start));
    gpu_write_kernel<<<128, 256>>>(data, elems, addend, out_device);
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
        data[i] = 0x200000000ULL + i;
}

static UVM_PHASE_F_RANGE_RESULT timed_action(int fd,
                                             const RegionSpec &region,
                                             uint32_t action,
                                             int device,
                                             const char *name,
                                             double *elapsed_ms)
{
    double start = now_ms();
    UVM_PHASE_F_RANGE_RESULT result = run_region_action_or_die(fd, region, action, device, name);
    if (elapsed_ms)
        *elapsed_ms = now_ms() - start;
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);
    const uint32_t cpu_migrate_action = opt.mode == "baseline_cpu_map" ?
                                            UVM_PHASE_F_ACTION_PREFETCH_CPU :
                                            UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP;
    const size_t page_size = 4096;
    const size_t elems_per_page = page_size / sizeof(uint64_t);
    const size_t elems = opt.pages * elems_per_page;
    const size_t bytes = elems * sizeof(uint64_t);
    uint64_t *data = nullptr;
    uint64_t *out_device = nullptr;
    uint64_t checksum = 0;
    uint64_t setup_checksum = 0;
    double init_ms = 0.0;
    double setup_apply_ms = 0.0;
    double setup_write_ms = 0.0;
    double setup_cpu_migrate_ms = 0.0;
    double clear_ms = 0.0;
    UVM_PHASE_F_RANGE_RESULT setup_apply{};
    UVM_PHASE_F_RANGE_RESULT setup_query_cpu{};
    UVM_PHASE_F_RANGE_RESULT final_query_cpu{};
    UVM_PHASE_F_RANGE_RESULT clear{};
    std::vector<double> apply_ms;
    std::vector<double> gpu_write_ms;
    std::vector<double> cpu_migrate_ms;
    uint64_t total_apply_gpu_bytes = 0;
    uint64_t total_apply_cpu_unmap_bytes = 0;
    FILE *iter_file = nullptr;

    CHECK_CUDA(cudaSetDevice(opt.device));
    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    CHECK_CUDA(cudaMalloc(&out_device, sizeof(uint64_t)));

    double start = now_ms();
    cpu_fill(data, elems);
    init_ms = now_ms() - start;

    int fd = duplicate_existing_uvm_fd();
    const bool adopted_existing_uvm_fd = fd >= 0;
    if (fd < 0) {
        std::fprintf(stderr, "Could not duplicate CUDA /dev/nvidia-uvm fd\n");
        return EXIT_FAILURE;
    }

    RegionSpec region{};
    region.region_id = region_id_for(data);
    region.ptr = data;
    region.bytes = bytes;
    register_kv_region_or_die(fd, region);

    setup_apply = timed_action(fd,
                               region,
                               UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT,
                               opt.device,
                               "setup_gpu_only_rw_soft",
                               &setup_apply_ms);
    setup_write_ms = run_gpu_write(data, elems, 1, out_device, &setup_checksum);
    (void)timed_action(fd, region, cpu_migrate_action, opt.device, "setup_cpu_migrate", &setup_cpu_migrate_ms);
    setup_query_cpu = run_region_action_or_die(fd,
                                               region,
                                               UVM_PHASE_F_ACTION_QUERY_RESIDENCY,
                                               opt.device,
                                               "query_after_setup_cpu_migrate");

    if (!opt.iteration_csv.empty()) {
        iter_file = std::fopen(opt.iteration_csv.c_str(), "w");
        if (!iter_file) {
            std::fprintf(stderr, "Could not open iteration CSV %s errno=%d\n", opt.iteration_csv.c_str(), errno);
            std::exit(EXIT_FAILURE);
        }

        std::fprintf(iter_file,
                     "iteration,apply_ms,apply_gpu_bytes,apply_cpu_unmap_bytes,gpu_write_ms,cpu_migrate_ms\n");
    }

    apply_ms.reserve(opt.iterations);
    gpu_write_ms.reserve(opt.iterations);
    cpu_migrate_ms.reserve(opt.iterations);

    for (int iter = 0; iter < opt.iterations; ++iter) {
        double iter_apply_ms = 0.0;
        double iter_write_ms = 0.0;
        double iter_migrate_ms = 0.0;
        UVM_PHASE_F_RANGE_RESULT apply_result =
            timed_action(fd,
                         region,
                         UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT,
                         opt.device,
                         "iter_gpu_only_rw_soft",
                         &iter_apply_ms);

        iter_write_ms = run_gpu_write(data,
                                      elems,
                                      static_cast<uint64_t>(100 + iter),
                                      out_device,
                                      &checksum);

        (void)timed_action(fd, region, cpu_migrate_action, opt.device, "iter_cpu_migrate", &iter_migrate_ms);

        apply_ms.push_back(iter_apply_ms);
        gpu_write_ms.push_back(iter_write_ms);
        cpu_migrate_ms.push_back(iter_migrate_ms);
        total_apply_gpu_bytes += apply_result.resident_gpu_bytes;
        total_apply_cpu_unmap_bytes += apply_result.resident_cpu_bytes;

        if (iter_file) {
            std::fprintf(iter_file,
                         "%d,%.6f,%llu,%llu,%.6f,%.6f\n",
                         iter,
                         iter_apply_ms,
                         static_cast<unsigned long long>(apply_result.resident_gpu_bytes),
                         static_cast<unsigned long long>(apply_result.resident_cpu_bytes),
                         iter_write_ms,
                         iter_migrate_ms);
        }
    }

    if (iter_file)
        std::fclose(iter_file);

    final_query_cpu = run_region_action_or_die(fd,
                                               region,
                                               UVM_PHASE_F_ACTION_QUERY_RESIDENCY,
                                               opt.device,
                                               "query_after_final_cpu_migrate");

    if (opt.clear_after)
        clear = timed_action(fd,
                             region,
                             UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY,
                             opt.device,
                             "clear_gpu_only",
                             &clear_ms);

    Stats apply_stats = summarize(apply_ms);
    Stats write_stats = summarize(gpu_write_ms);
    Stats migrate_stats = summarize(cpu_migrate_ms);

    std::printf("stage_d_rw_cpu_storage_no_map_bench mode=%s pages=%zu bytes=%zu iterations=%d adopted_existing_uvm_fd=%u init_ms=%.6f setup_apply_ms=%.6f setup_apply_gpu_bytes=%llu setup_apply_cpu_unmap_bytes=%llu setup_write_ms=%.6f setup_cpu_migrate_ms=%.6f setup_query_cpu_bytes=%llu setup_query_gpu_bytes=%llu setup_query_on_cpu=%u setup_query_on_gpu=%u avg_apply_ms=%.6f min_apply_ms=%.6f p50_apply_ms=%.6f p90_apply_ms=%.6f p99_apply_ms=%.6f max_apply_ms=%.6f avg_gpu_write_ms=%.6f min_gpu_write_ms=%.6f max_gpu_write_ms=%.6f avg_cpu_migrate_ms=%.6f min_cpu_migrate_ms=%.6f p50_cpu_migrate_ms=%.6f p90_cpu_migrate_ms=%.6f p99_cpu_migrate_ms=%.6f max_cpu_migrate_ms=%.6f total_apply_gpu_bytes=%llu total_apply_cpu_unmap_bytes=%llu final_query_cpu_bytes=%llu final_query_gpu_bytes=%llu final_query_on_cpu=%u final_query_on_gpu=%u clear_gpu_bytes=%llu clear_ms=%.6f setup_checksum=%llu checksum=%llu\n",
                opt.mode.c_str(),
                opt.pages,
                bytes,
                opt.iterations,
                adopted_existing_uvm_fd ? 1 : 0,
                init_ms,
                setup_apply_ms,
                static_cast<unsigned long long>(setup_apply.resident_gpu_bytes),
                static_cast<unsigned long long>(setup_apply.resident_cpu_bytes),
                setup_write_ms,
                setup_cpu_migrate_ms,
                static_cast<unsigned long long>(setup_query_cpu.resident_cpu_bytes),
                static_cast<unsigned long long>(setup_query_cpu.resident_gpu_bytes),
                setup_query_cpu.resident_on_cpu ? 1 : 0,
                setup_query_cpu.resident_on_gpu ? 1 : 0,
                apply_stats.avg,
                apply_stats.min,
                apply_stats.p50,
                apply_stats.p90,
                apply_stats.p99,
                apply_stats.max,
                write_stats.avg,
                write_stats.min,
                write_stats.max,
                migrate_stats.avg,
                migrate_stats.min,
                migrate_stats.p50,
                migrate_stats.p90,
                migrate_stats.p99,
                migrate_stats.max,
                static_cast<unsigned long long>(total_apply_gpu_bytes),
                static_cast<unsigned long long>(total_apply_cpu_unmap_bytes),
                static_cast<unsigned long long>(final_query_cpu.resident_cpu_bytes),
                static_cast<unsigned long long>(final_query_cpu.resident_gpu_bytes),
                final_query_cpu.resident_on_cpu ? 1 : 0,
                final_query_cpu.resident_on_gpu ? 1 : 0,
                static_cast<unsigned long long>(clear.resident_gpu_bytes),
                clear_ms,
                static_cast<unsigned long long>(setup_checksum),
                static_cast<unsigned long long>(checksum));

    unregister_region(fd, region.region_id);
    close(fd);

    CHECK_CUDA(cudaFree(out_device));
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
