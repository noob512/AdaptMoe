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

#ifndef UVM_PHASE_F_ACTION_QUERY_RESIDENCY
#define UVM_PHASE_F_ACTION_QUERY_RESIDENCY    4
#endif

#ifndef UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT
#define UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT 9
#define UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT   10
#define UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY     13
#endif

#ifndef UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT
#define UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT 14
#endif

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
    bool first_gpu_touch = true;
    bool second_gpu_touch = true;
    bool clear_after = true;
    bool allow_new_uvm_fd = false;
    bool weight_readdup = false;
    bool query_after_apply = false;
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
                 "          [--cpu-touch none|weight|kv|both]\n"
                 "          [--no-first-gpu-touch] [--no-second-gpu-touch]\n"
                 "          [--no-clear-after] [--allow-new-uvm-fd]\n"
                 "          [--weight-readdup] [--query-after-apply]\n",
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
        else if (std::strcmp(argv[i], "--no-first-gpu-touch") == 0) {
            opt.first_gpu_touch = false;
        }
        else if (std::strcmp(argv[i], "--no-second-gpu-touch") == 0) {
            opt.second_gpu_touch = false;
        }
        else if (std::strcmp(argv[i], "--no-clear-after") == 0) {
            opt.clear_after = false;
        }
        else if (std::strcmp(argv[i], "--allow-new-uvm-fd") == 0) {
            opt.allow_new_uvm_fd = true;
        }
        else if (std::strcmp(argv[i], "--weight-readdup") == 0) {
            opt.weight_readdup = true;
        }
        else if (std::strcmp(argv[i], "--query-after-apply") == 0) {
            opt.query_after_apply = true;
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

static UVM_PHASE_F_RANGE_RESULT run_region_action_or_die(int fd,
                                                         const RegionSpec &region,
                                                         uint32_t action,
                                                         int device,
                                                         const char *action_name)
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
                     "phase_f_action failed name=%s region=%s errno=%d rmStatus=%u result_status=%u success=%u failure=%u\n",
                     action_name,
                     region.name,
                     errno,
                     params.rmStatus,
                     params.results[0].status,
                     params.success_count,
                     params.failure_count);
        std::exit(EXIT_FAILURE);
    }

    std::printf("phase_f_action name=%s region=%s status=%u on_count=%u on_cpu=%u on_gpu=%u queried_bytes=%llu gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu unknown_bytes=%llu\n",
                action_name,
                region.name,
                params.results[0].status,
                params.results[0].resident_on_count,
                params.results[0].resident_on_cpu ? 1 : 0,
                params.results[0].resident_on_gpu ? 1 : 0,
                static_cast<unsigned long long>(params.results[0].queried_bytes),
                static_cast<unsigned long long>(params.results[0].resident_gpu_bytes),
                static_cast<unsigned long long>(params.results[0].resident_cpu_bytes),
                static_cast<unsigned long long>(params.results[0].resident_any_bytes),
                static_cast<unsigned long long>(params.results[0].resident_unknown_bytes));

    return params.results[0];
}

__global__ void gpu_touch_kernel(uint64_t *weight, uint64_t *kv, size_t elems, uint64_t *out, uint64_t addend)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = blockDim.x * gridDim.x;
    uint64_t acc = 0;

    for (size_t i = tid; i < elems; i += grid) {
        uint64_t w = weight[i];
        uint64_t v = kv[i];
        kv[i] = v + addend;
        acc += (w ^ (v + i));
    }

    if (acc)
        atomicAdd(reinterpret_cast<unsigned long long *>(out),
                  static_cast<unsigned long long>(acc & 0xffffULL));
}

static uint64_t run_gpu_touch(uint64_t *weight, uint64_t *kv, size_t elems, uint64_t *out_device, uint64_t addend)
{
    uint64_t out = 0;

    CHECK_CUDA(cudaMemset(out_device, 0, sizeof(uint64_t)));
    gpu_touch_kernel<<<128, 256>>>(weight, kv, elems, out_device, addend);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&out, out_device, sizeof(out), cudaMemcpyDeviceToHost));
    return out;
}

static uint64_t cpu_touch(uint64_t *data, size_t elems, bool write)
{
    volatile uint64_t *volatile_data = data;
    uint64_t acc = 0;
    const size_t stride = 4096 / sizeof(uint64_t);

    for (size_t i = 0; i < elems; i += stride) {
        if (write) {
            const uint64_t value = 0x5b5b000000000000ULL ^ static_cast<uint64_t>(i);
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
    uint64_t gpu_out_first = 0;
    uint64_t gpu_out_second = 0;
    uint64_t weight_cpu_sum = 0;
    uint64_t kv_cpu_sum = 0;
    UVM_PHASE_F_RANGE_RESULT weight_query_after_apply{};

    CHECK_CUDA(cudaMallocManaged(&weight, bytes));
    CHECK_CUDA(cudaMallocManaged(&kv, bytes));

    for (size_t i = 0; i < elems; ++i) {
        weight[i] = 0x1000 + i;
        kv[i] = 0x2000 + i;
    }

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
    weight_region.region_id = region_id_for(weight, 0x57000002ULL);
    weight_region.ptr = weight;
    weight_region.bytes = bytes;
    weight_region.pool = UVM_PHASE_N_POOL_WEIGHT;
    weight_region.role = 1;
    weight_region.flags = UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT |
                          UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING |
                          UVM_PHASE_N_REGION_FLAG_GPU_READ_ONLY;
    weight_region.name = "weight";

    RegionSpec kv_region{};
    kv_region.region_id = region_id_for(kv, 0x6b000002ULL);
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

    const uint32_t weight_action = opt.weight_readdup ?
        UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT :
        UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT;
    const char *weight_action_name = opt.weight_readdup ?
        "gpu_only_read_dup_soft" :
        "gpu_only_read_soft";

    UVM_PHASE_F_RANGE_RESULT weight_apply =
        run_region_action_or_die(fd, weight_region, weight_action, opt.device, weight_action_name);
    UVM_PHASE_F_RANGE_RESULT kv_apply =
        run_region_action_or_die(fd, kv_region, UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT, opt.device, "gpu_only_rw_soft");

    if (opt.query_after_apply) {
        weight_query_after_apply =
            run_region_action_or_die(fd,
                                     weight_region,
                                     UVM_PHASE_F_ACTION_QUERY_RESIDENCY,
                                     opt.device,
                                     "query_residency_after_apply");
    }

    CHECK_CUDA(cudaMalloc(&out_device, sizeof(uint64_t)));

    if (opt.first_gpu_touch)
        gpu_out_first = run_gpu_touch(weight, kv, elems, out_device, 1);

    if (opt.cpu_touch_weight)
        weight_cpu_sum = cpu_touch(weight, elems, false);
    if (opt.cpu_touch_kv)
        kv_cpu_sum = cpu_touch(kv, elems, true);

    if (opt.second_gpu_touch)
        gpu_out_second = run_gpu_touch(weight, kv, elems, out_device, 2);

    UVM_PHASE_F_RANGE_RESULT weight_clear{};
    UVM_PHASE_F_RANGE_RESULT kv_clear{};
    if (opt.clear_after) {
        weight_clear =
            run_region_action_or_die(fd, weight_region, UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY, opt.device, "clear_gpu_only");
        kv_clear =
            run_region_action_or_die(fd, kv_region, UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY, opt.device, "clear_gpu_only");
    }

    std::printf("stage_b_probe pages=%zu bytes=%zu weight_region_id=%llu kv_region_id=%llu weight=0x%llx kv=0x%llx adopted_existing_uvm_fd=%u first_gpu_touch=%u second_gpu_touch=%u clear_after=%u cpu_touch_weight=%u cpu_touch_kv=%u weight_readdup=%u query_after_apply=%u weight_apply_gpu_bytes=%llu weight_apply_cpu_bytes=%llu kv_apply_gpu_bytes=%llu kv_apply_cpu_bytes=%llu weight_query_gpu_bytes=%llu weight_query_cpu_bytes=%llu weight_query_on_cpu=%u weight_query_on_gpu=%u weight_clear_gpu_bytes=%llu kv_clear_gpu_bytes=%llu gpu_out_first=%llu gpu_out_second=%llu weight_cpu_sum=%llu kv_cpu_sum=%llu\n",
                opt.pages,
                bytes,
                static_cast<unsigned long long>(weight_region.region_id),
                static_cast<unsigned long long>(kv_region.region_id),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(weight)),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(kv)),
                adopted_existing_uvm_fd ? 1 : 0,
                opt.first_gpu_touch ? 1 : 0,
                opt.second_gpu_touch ? 1 : 0,
                opt.clear_after ? 1 : 0,
                opt.cpu_touch_weight ? 1 : 0,
                opt.cpu_touch_kv ? 1 : 0,
                opt.weight_readdup ? 1 : 0,
                opt.query_after_apply ? 1 : 0,
                static_cast<unsigned long long>(weight_apply.resident_gpu_bytes),
                static_cast<unsigned long long>(weight_apply.resident_cpu_bytes),
                static_cast<unsigned long long>(kv_apply.resident_gpu_bytes),
                static_cast<unsigned long long>(kv_apply.resident_cpu_bytes),
                static_cast<unsigned long long>(weight_query_after_apply.resident_gpu_bytes),
                static_cast<unsigned long long>(weight_query_after_apply.resident_cpu_bytes),
                weight_query_after_apply.resident_on_cpu ? 1 : 0,
                weight_query_after_apply.resident_on_gpu ? 1 : 0,
                static_cast<unsigned long long>(weight_clear.resident_gpu_bytes),
                static_cast<unsigned long long>(kv_clear.resident_gpu_bytes),
                static_cast<unsigned long long>(gpu_out_first),
                static_cast<unsigned long long>(gpu_out_second),
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
