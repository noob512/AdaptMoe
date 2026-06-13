#include <cuda_runtime.h>

#include <cerrno>
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

#ifndef UVM_PHASE_F_ACTION_PREFETCH_CPU
#define UVM_PHASE_F_ACTION_PREFETCH_CPU      1
#define UVM_PHASE_F_ACTION_QUERY_RESIDENCY   4
#define UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT 10
#define UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY   13
#define UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP 15
#endif

#ifndef UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT
#define UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT       (1U << 6)
#define UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING (1U << 8)
#define UVM_PHASE_N_REGION_FLAG_GPU_READ_WRITE      (1U << 10)
#endif

namespace {

struct Options {
    size_t pages = 1024;
    int device = 0;
    bool cpu_touch_after_cpu_migrate = false;
    bool clear_after = true;
};

struct RegionSpec {
    uint64_t region_id = 0;
    void *ptr = nullptr;
    size_t bytes = 0;
};

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
                 "Usage: %s [--pages N] [--device N]\n"
                 "          [--cpu-touch-after-cpu-migrate] [--no-clear-after]\n",
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

        if (std::strcmp(argv[i], "--pages") == 0) {
            opt.pages = parse_size(need_value(argv[i]));
        }
        else if (std::strcmp(argv[i], "--device") == 0) {
            opt.device = static_cast<int>(parse_size(need_value(argv[i])));
        }
        else if (std::strcmp(argv[i], "--cpu-touch-after-cpu-migrate") == 0) {
            opt.cpu_touch_after_cpu_migrate = true;
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

    if (opt.pages == 0) {
        std::fprintf(stderr, "--pages must be positive\n");
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
    value ^= 0x6b00000300000000ULL + (value << 6) + (value >> 2);
    value &= 0x7fffffffffffffffULL;
    return value == 0 ? 0x6b000003ULL : value;
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

    std::printf("phase_f_action name=%s status=%u on_cpu=%u on_gpu=%u gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu unknown_bytes=%llu\n",
                name,
                params.results[0].status,
                params.results[0].resident_on_cpu ? 1 : 0,
                params.results[0].resident_on_gpu ? 1 : 0,
                static_cast<unsigned long long>(params.results[0].resident_gpu_bytes),
                static_cast<unsigned long long>(params.results[0].resident_cpu_bytes),
                static_cast<unsigned long long>(params.results[0].resident_any_bytes),
                static_cast<unsigned long long>(params.results[0].resident_unknown_bytes));

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

static uint64_t run_gpu_write(uint64_t *data, size_t elems, uint64_t addend, uint64_t *out_device)
{
    uint64_t out = 0;

    CHECK_CUDA(cudaMemset(out_device, 0, sizeof(uint64_t)));
    gpu_write_kernel<<<128, 256>>>(data, elems, addend, out_device);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&out, out_device, sizeof(out), cudaMemcpyDeviceToHost));

    return out;
}

static uint64_t cpu_read_one_word_per_page(uint64_t *data, size_t pages)
{
    volatile uint64_t *volatile_data = data;
    const size_t stride = 4096 / sizeof(uint64_t);
    uint64_t acc = 0;

    for (size_t page = 0; page < pages; ++page)
        acc += volatile_data[page * stride];

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
    uint64_t gpu_out_first = 0;
    uint64_t gpu_out_second = 0;
    uint64_t cpu_sum = 0;

    CHECK_CUDA(cudaSetDevice(opt.device));
    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    CHECK_CUDA(cudaMalloc(&out_device, sizeof(uint64_t)));

    for (size_t i = 0; i < elems; ++i)
        data[i] = 0x2000 + i;

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

    UVM_PHASE_F_RANGE_RESULT first_apply =
        run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT, opt.device, "first_gpu_only_rw_soft");

    gpu_out_first = run_gpu_write(data, elems, 1, out_device);

    UVM_PHASE_F_RANGE_RESULT prefetch_cpu =
        run_region_action_or_die(fd,
                                 region,
                                 UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP,
                                 opt.device,
                                 "prefetch_cpu_no_map_after_gpu_write");

    UVM_PHASE_F_RANGE_RESULT query_cpu =
        run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_QUERY_RESIDENCY, opt.device, "query_after_prefetch_cpu");

    if (opt.cpu_touch_after_cpu_migrate)
        cpu_sum = cpu_read_one_word_per_page(data, opt.pages);

    UVM_PHASE_F_RANGE_RESULT second_apply =
        run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT, opt.device, "second_gpu_only_rw_soft");

    gpu_out_second = run_gpu_write(data, elems, 2, out_device);

    UVM_PHASE_F_RANGE_RESULT query_gpu =
        run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_QUERY_RESIDENCY, opt.device, "query_after_second_gpu_write");

    UVM_PHASE_F_RANGE_RESULT clear{};
    if (opt.clear_after)
        clear = run_region_action_or_die(fd, region, UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY, opt.device, "clear_gpu_only");

    std::printf("stage_d_rw_cpu_storage_no_map_probe pages=%zu bytes=%zu region_id=%llu data=0x%llx adopted_existing_uvm_fd=%u cpu_touch_after_cpu_migrate=%u first_apply_gpu_bytes=%llu first_apply_cpu_unmap_bytes=%llu prefetch_cpu_gpu_bytes=%llu prefetch_cpu_cpu_bytes=%llu query_cpu_gpu_bytes=%llu query_cpu_cpu_bytes=%llu query_cpu_on_cpu=%u query_cpu_on_gpu=%u second_apply_gpu_bytes=%llu second_apply_cpu_unmap_bytes=%llu query_gpu_gpu_bytes=%llu query_gpu_cpu_bytes=%llu query_gpu_on_cpu=%u query_gpu_on_gpu=%u clear_gpu_bytes=%llu gpu_out_first=%llu gpu_out_second=%llu cpu_sum=%llu\n",
                opt.pages,
                bytes,
                static_cast<unsigned long long>(region.region_id),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(data)),
                adopted_existing_uvm_fd ? 1 : 0,
                opt.cpu_touch_after_cpu_migrate ? 1 : 0,
                static_cast<unsigned long long>(first_apply.resident_gpu_bytes),
                static_cast<unsigned long long>(first_apply.resident_cpu_bytes),
                static_cast<unsigned long long>(prefetch_cpu.resident_gpu_bytes),
                static_cast<unsigned long long>(prefetch_cpu.resident_cpu_bytes),
                static_cast<unsigned long long>(query_cpu.resident_gpu_bytes),
                static_cast<unsigned long long>(query_cpu.resident_cpu_bytes),
                query_cpu.resident_on_cpu ? 1 : 0,
                query_cpu.resident_on_gpu ? 1 : 0,
                static_cast<unsigned long long>(second_apply.resident_gpu_bytes),
                static_cast<unsigned long long>(second_apply.resident_cpu_bytes),
                static_cast<unsigned long long>(query_gpu.resident_gpu_bytes),
                static_cast<unsigned long long>(query_gpu.resident_cpu_bytes),
                query_gpu.resident_on_cpu ? 1 : 0,
                query_gpu.resident_on_gpu ? 1 : 0,
                static_cast<unsigned long long>(clear.resident_gpu_bytes),
                static_cast<unsigned long long>(gpu_out_first),
                static_cast<unsigned long long>(gpu_out_second),
                static_cast<unsigned long long>(cpu_sum));

    unregister_region(fd, region.region_id);
    close(fd);

    CHECK_CUDA(cudaFree(out_device));
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
