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

#ifndef UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT
#define UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT       (1U << 6)
#define UVM_PHASE_N_REGION_FLAG_GPU_ONLY_HARD       (1U << 7)
#define UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING (1U << 8)
#define UVM_PHASE_N_REGION_FLAG_GPU_READ_ONLY       (1U << 9)
#define UVM_PHASE_N_REGION_FLAG_GPU_READ_WRITE      (1U << 10)
#endif

namespace {

struct Options {
    size_t pages = 1024;
    int device = 0;
    bool cpu_touch_weight = false;
    bool cpu_touch_kv = false;
    bool gpu_touch = true;
    bool allow_new_uvm_fd = false;
};

struct RegionSpec {
    uint64_t region_id = 0;
    void *ptr = nullptr;
    size_t bytes = 0;
    uint32_t pool = 0;
    uint32_t role = 0;
    uint32_t flags = 0;
    const char *name = "";
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
                 "          [--cpu-touch none|weight|kv|both] [--no-gpu-touch]\n"
                 "          [--allow-new-uvm-fd]\n",
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
        else if (std::strcmp(argv[i], "--cpu-touch") == 0) {
            const char *value = need_value(argv[i]);
            if (std::strcmp(value, "none") == 0) {
                opt.cpu_touch_weight = false;
                opt.cpu_touch_kv = false;
            }
            else if (std::strcmp(value, "weight") == 0) {
                opt.cpu_touch_weight = true;
                opt.cpu_touch_kv = false;
            }
            else if (std::strcmp(value, "kv") == 0) {
                opt.cpu_touch_weight = false;
                opt.cpu_touch_kv = true;
            }
            else if (std::strcmp(value, "both") == 0) {
                opt.cpu_touch_weight = true;
                opt.cpu_touch_kv = true;
            }
            else {
                std::fprintf(stderr, "Invalid --cpu-touch value: %s\n", value);
                std::exit(EXIT_FAILURE);
            }
        }
        else if (std::strcmp(argv[i], "--no-gpu-touch") == 0) {
            opt.gpu_touch = false;
        }
        else if (std::strcmp(argv[i], "--allow-new-uvm-fd") == 0) {
            opt.allow_new_uvm_fd = true;
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

static int open_initialized_uvm_fd()
{
    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);

    if (fd < 0)
        return -1;

    UVM_INITIALIZE_PARAMS init{};
    if (ioctl(fd, UVM_INITIALIZE, &init) != 0 || init.rmStatus != 0) {
        close(fd);
        return -1;
    }

    return fd;
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

static uint64_t region_id_for(const void *ptr, uint64_t salt)
{
    uint64_t value = reinterpret_cast<uintptr_t>(ptr);
    value ^= salt + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    value &= 0x7fffffffffffffffULL;
    return value == 0 ? salt : value;
}

static void register_region_or_die(int fd, const RegionSpec &region)
{
    UVM_PHASE_F_REGISTER_REGION_PARAMS params{};

    params.region.region_id = region.region_id;
    params.region.base = reinterpret_cast<uint64_t>(region.ptr);
    params.region.length = static_cast<uint64_t>(region.bytes);
    params.region.pool = region.pool;
    params.region.layer_id = 0;
    params.region.expert_id = 0;
    params.region.role = region.role;
    params.region.flags = region.flags;

    if (ioctl(fd, UVM_PHASE_F_REGISTER_REGION, &params) != 0 || params.rmStatus != 0) {
        std::fprintf(stderr,
                     "register_region failed name=%s errno=%d rmStatus=%u region_id=%llu base=0x%llx bytes=%zu\n",
                     region.name,
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

__global__ void gpu_touch_kernel(uint64_t *weight, uint64_t *kv, size_t elems, uint64_t *out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = blockDim.x * gridDim.x;
    uint64_t acc = 0;

    for (size_t i = tid; i < elems; i += grid) {
        acc += weight[i];
        kv[i] = kv[i] + 1;
    }

    if (tid == 0)
        *out = acc;
}

static uint64_t cpu_touch(uint64_t *data, size_t elems, bool write)
{
    volatile uint64_t *volatile_data = data;
    uint64_t acc = 0;
    const size_t stride = 4096 / sizeof(uint64_t);

    for (size_t i = 0; i < elems; i += stride) {
        if (write) {
            const uint64_t value = 0x5a5a000000000000ULL ^ static_cast<uint64_t>(i);
            volatile_data[i] = value;
            acc += value;
        }
        else {
            acc += volatile_data[i];
        }
    }

    return acc;
}

} // namespace

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);

    CHECK_CUDA(cudaSetDevice(opt.device));

    const size_t page_size = 4096;
    const size_t elems_per_page = page_size / sizeof(uint64_t);
    const size_t elems = opt.pages * elems_per_page;
    const size_t bytes = elems * sizeof(uint64_t);

    uint64_t *weight = nullptr;
    uint64_t *kv = nullptr;
    uint64_t *out_device = nullptr;
    uint64_t gpu_out = 0;

    CHECK_CUDA(cudaMallocManaged(&weight, bytes));
    CHECK_CUDA(cudaMallocManaged(&kv, bytes));

    for (size_t i = 0; i < elems; ++i) {
        weight[i] = 0x1000 + i;
        kv[i] = 0x2000 + i;
    }

    CHECK_CUDA(cudaMemPrefetchAsync(weight, bytes, opt.device));
    CHECK_CUDA(cudaMemPrefetchAsync(kv, bytes, opt.device));
    CHECK_CUDA(cudaDeviceSynchronize());

    int fd = duplicate_existing_uvm_fd();
    const bool adopted_existing_uvm_fd = fd >= 0;
    if (fd < 0 && opt.allow_new_uvm_fd)
        fd = open_initialized_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr,
                     "Could not duplicate an existing CUDA /dev/nvidia-uvm fd. "
                     "The probe needs the same UVM VA space as the managed allocations; "
                     "use --allow-new-uvm-fd only for ioctl debugging.\n");
        return EXIT_FAILURE;
    }

    RegionSpec weight_region{};
    weight_region.region_id = region_id_for(weight, 0x57000001ULL);
    weight_region.ptr = weight;
    weight_region.bytes = bytes;
    weight_region.pool = UVM_PHASE_N_POOL_WEIGHT;
    weight_region.role = 1;
    weight_region.flags = UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT |
                          UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING |
                          UVM_PHASE_N_REGION_FLAG_GPU_READ_ONLY;
    weight_region.name = "weight";

    RegionSpec kv_region{};
    kv_region.region_id = region_id_for(kv, 0x6b000001ULL);
    kv_region.ptr = kv;
    kv_region.bytes = bytes;
    kv_region.pool = UVM_PHASE_N_POOL_KV;
    kv_region.role = 2;
    kv_region.flags = UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT |
                      UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING |
                      UVM_PHASE_N_REGION_FLAG_GPU_READ_WRITE;
    kv_region.name = "kv";

    register_region_or_die(fd, weight_region);
    register_region_or_die(fd, kv_region);

    CHECK_CUDA(cudaMalloc(&out_device, sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(out_device, 0, sizeof(uint64_t)));

    if (opt.gpu_touch) {
        gpu_touch_kernel<<<128, 256>>>(weight, kv, elems, out_device);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&gpu_out, out_device, sizeof(gpu_out), cudaMemcpyDeviceToHost));
    }

    uint64_t weight_cpu_sum = 0;
    uint64_t kv_cpu_sum = 0;
    if (opt.cpu_touch_weight)
        weight_cpu_sum = cpu_touch(weight, elems, false);
    if (opt.cpu_touch_kv)
        kv_cpu_sum = cpu_touch(kv, elems, true);

    std::printf("stage_a_probe pages=%zu bytes=%zu weight_region_id=%llu kv_region_id=%llu weight=0x%llx kv=0x%llx adopted_existing_uvm_fd=%u gpu_touch=%u cpu_touch_weight=%u cpu_touch_kv=%u gpu_out=%llu weight_cpu_sum=%llu kv_cpu_sum=%llu\n",
                opt.pages,
                bytes,
                static_cast<unsigned long long>(weight_region.region_id),
                static_cast<unsigned long long>(kv_region.region_id),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(weight)),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(kv)),
                adopted_existing_uvm_fd ? 1 : 0,
                opt.gpu_touch ? 1 : 0,
                opt.cpu_touch_weight ? 1 : 0,
                opt.cpu_touch_kv ? 1 : 0,
                static_cast<unsigned long long>(gpu_out),
                static_cast<unsigned long long>(weight_cpu_sum),
                static_cast<unsigned long long>(kv_cpu_sum));

    unregister_region(fd, weight_region.region_id);
    unregister_region(fd, kv_region.region_id);
    close(fd);

    CHECK_CUDA(cudaFree(out_device));
    CHECK_CUDA(cudaFree(kv));
    CHECK_CUDA(cudaFree(weight));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
