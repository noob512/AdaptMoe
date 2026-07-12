#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

#include <cuda.h>
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

#define CU_CHECK(expr)                                                          \
    do {                                                                        \
        CUresult err__ = (expr);                                                \
        if (err__ != CUDA_SUCCESS) {                                            \
            const char *name__ = nullptr;                                       \
            const char *str__ = nullptr;                                        \
            cuGetErrorName(err__, &name__);                                     \
            cuGetErrorString(err__, &str__);                                    \
            std::fprintf(stderr, "CUDA driver error %s:%d: %s %s\n", __FILE__,  \
                         __LINE__, name__ ? name__ : "?", str__ ? str__ : "?"); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static constexpr unsigned int kUvmInitializeIoctl = 0x30000001;
static constexpr unsigned int kUvmPhaseFRegisterRegionIoctl = 82;
static constexpr unsigned int kUvmPhaseFUnregisterRegionIoctl = 83;
static constexpr unsigned int kExpertReservedPoolIoctl = 98;
static constexpr unsigned int kExpertReservedPrefetchIoctl = 99;

static constexpr unsigned int kPoolWeight = 2;
static constexpr unsigned int kRoleW13 = 3;
static constexpr unsigned int kRegionFlagPrefetchable = 1u << 3;
static constexpr unsigned int kRegionFlagGpuReadOnly = 1u << 9;
static constexpr unsigned int kRegionFlagStreamingSubpool = 1u << 12;
static constexpr unsigned int kRegionFlagCpuOffloadedReserved = 1u << 14;
static constexpr unsigned int kStrictFootprint = 1u << 0;

static constexpr unsigned int kReservedPoolModePrealloc = 1;
static constexpr unsigned int kReservedPoolModeReset = 2;
static constexpr unsigned int kReservedPrefetchModeInvalidate = 2;
static constexpr unsigned int kReservedPrefetchModeReservedPrefetch = 3;

struct UvmInitializeParams {
    unsigned long long flags = 0;
    unsigned int rmStatus = 0;
};

struct PhaseFUuid {
    unsigned char uuid[16] = {};
};

struct PhaseFRegion {
    unsigned long long region_id = 0;
    unsigned long long base = 0;
    unsigned long long length = 0;
    unsigned int pool = 0;
    unsigned int layer_id = 0;
    unsigned int expert_id = 0;
    unsigned int role = 0;
    unsigned int flags = 0;
};

struct PhaseFRegisterRegionParams {
    PhaseFRegion region;
    unsigned int rmStatus = 0;
};

struct PhaseFUnregisterRegionParams {
    unsigned long long region_id = 0;
    unsigned int rmStatus = 0;
};

struct ExpertReservedPoolEntry {
    unsigned int pool_id = 0;
    unsigned int slot_id = 0;
    unsigned int flags = 0;
    unsigned int status = 0;
    unsigned int state = 0;
    unsigned int chunk_count = 0;
    unsigned long long slot_bytes = 0;
    unsigned long long physical_bytes = 0;
    unsigned long long epoch = 0;
    unsigned long long first_chunk_address = 0;
    unsigned long long last_chunk_address = 0;
};

struct ExpertReservedPoolParams {
    unsigned int mode = 0;
    unsigned int count = 0;
    PhaseFUuid gpu_uuid = {};
    unsigned int max_entries = 0;
    unsigned int success_count = 0;
    unsigned int failure_count = 0;
    unsigned int total_slots = 0;
    unsigned int returned_slots = 0;
    unsigned int active_slots = 0;
    unsigned int backed_slots = 0;
    unsigned int chunk_size = 0;
    unsigned int reserved = 0;
    unsigned long long reserved_backing_bytes = 0;
    unsigned long long logical_slot_bytes = 0;
    unsigned long long current_epoch = 0;
    unsigned long long prealloc_count = 0;
    unsigned long long reset_count = 0;
    unsigned long long alloc_failed_count = 0;
    ExpertReservedPoolEntry entries[128] = {};
    unsigned int rmStatus = 0;
};

struct ExpertReservedPrefetchEntry {
    unsigned long long region_id = 0;
    unsigned int pool_id = 0;
    unsigned int slot_id = 0;
    unsigned int flags = 0;
    unsigned int status = 0;
    unsigned long long slot_offset = 0;
    unsigned long long bytes_override = 0;
    unsigned long long resolved_base = 0;
    unsigned long long resolved_bytes = 0;
    unsigned int layer_id = 0;
    unsigned int expert_id = 0;
    unsigned int role = 0;
    unsigned int reserved = 0;
    unsigned long long slot_epoch = 0;
    unsigned long long physical_bytes = 0;
};

struct ExpertReservedPrefetchParams {
    unsigned int mode = 0;
    unsigned int count = 0;
    PhaseFUuid gpu_uuid = {};
    unsigned int async = 0;
    unsigned int success_count = 0;
    unsigned int failure_count = 0;
    unsigned int reserved = 0;
    unsigned long long requested_bytes = 0;
    unsigned long long accepted_bytes = 0;
    unsigned long long resolved_regions = 0;
    unsigned long long assigned_slots = 0;
    unsigned long long invalidated_slots = 0;
    unsigned long long reserved_backing_bytes = 0;
    unsigned long long ordinary_resident_bytes = 0;
    unsigned long long pool_external_alloc_count = 0;
    unsigned long long footprint_contract_broken = 0;
    unsigned long long current_epoch = 0;
    ExpertReservedPrefetchEntry entries[128] = {};
    unsigned int rmStatus = 0;
};

struct Args {
    int device = 0;
    int expert_count = 160;
    int offload_count = 16;
    unsigned int slot_count = 8;
    unsigned int pool_id = 0;
    size_t expert_bytes = 4ull * 1024ull * 1024ull;
    size_t slot_bytes = 0;
    size_t target_free_after_experts = 0;
    size_t pressure_chunk_bytes = 256ull * 1024ull * 1024ull;
    bool control_before_offload = true;
    bool require_full_after = false;
    bool reset_after = true;
};

struct PoolAttempt {
    const char *label = "";
    int ioctl_ret = -1;
    unsigned int rm_status = 0;
    ExpertReservedPoolParams params;
    std::vector<unsigned int> backed_slot_ids;
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
        else if (std::strcmp(argv[i], "--offload-count") == 0 && i + 1 < argc)
            args.offload_count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--expert-bytes") == 0 && i + 1 < argc)
            args.expert_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--slot-count") == 0 && i + 1 < argc)
            args.slot_count = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--slot-bytes") == 0 && i + 1 < argc)
            args.slot_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--pool-id") == 0 && i + 1 < argc)
            args.pool_id = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--target-free-after-experts") == 0 &&
                 i + 1 < argc)
            args.target_free_after_experts = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--pressure-chunk-bytes") == 0 &&
                 i + 1 < argc)
            args.pressure_chunk_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--no-control-before-offload") == 0)
            args.control_before_offload = false;
        else if (std::strcmp(argv[i], "--require-full-after") == 0)
            args.require_full_after = true;
        else if (std::strcmp(argv[i], "--no-reset-after") == 0)
            args.reset_after = false;
        else {
            std::fprintf(
                stderr,
                "usage: %s [--device N] [--expert-count N] "
                "[--offload-count N] [--expert-bytes N|NM|NG] "
                "[--slot-count N] [--slot-bytes N|NM|NG] [--pool-id N] "
                "[--target-free-after-experts N|NM|NG] "
                "[--pressure-chunk-bytes N|NM|NG] "
                "[--no-control-before-offload] [--require-full-after] "
                "[--no-reset-after]\n",
                argv[0]);
            std::exit(2);
        }
    }

    if (args.slot_bytes == 0)
        args.slot_bytes = args.expert_bytes;
    if (args.expert_count <= 0 ||
        args.offload_count <= 0 ||
        args.offload_count > args.expert_count ||
        args.slot_count == 0 ||
        args.slot_count > 128 ||
        args.expert_bytes == 0 ||
        args.slot_bytes < args.expert_bytes ||
        args.expert_bytes % sizeof(unsigned int) != 0 ||
        args.pressure_chunk_bytes == 0) {
        std::fprintf(stderr, "invalid arguments\n");
        std::exit(2);
    }

    return args;
}

static unsigned int expert_seed(int expert)
{
    return 0x6e710000u ^ (0x01010101u * (unsigned int)(expert + 1));
}

static void fill_pattern(unsigned int *data, size_t words, int expert)
{
    unsigned int seed = expert_seed(expert);
    for (size_t i = 0; i < words; ++i)
        data[i] = seed ^ (unsigned int)i;
}

__global__ void check_pattern_kernel(const unsigned int *data,
                                     size_t words,
                                     unsigned int seed,
                                     unsigned int *errors)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;
    if (data[i] != (seed ^ (unsigned int)i))
        atomicAdd(errors, 1u);
}

static bool fd_target_is_uvm(const char *target)
{
    if (!target)
        return false;

    const char *base = std::strrchr(target, '/');
    base = base ? base + 1 : target;
    return std::strcmp(base, "nvidia-uvm") == 0 ||
           std::strncmp(base, "nvidia-uvm ", std::strlen("nvidia-uvm ")) == 0;
}

static int open_uvm_fd()
{
    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0)
        return -1;

    UvmInitializeParams params{};
    if (ioctl(fd, kUvmInitializeIoctl, &params) != 0 || params.rmStatus != 0) {
        close(fd);
        return -1;
    }

    return fd;
}

static int duplicate_existing_uvm_fd()
{
    DIR *dir = opendir("/proc/self/fd");
    if (!dir)
        return -1;

    int dir_fd = dirfd(dir);
    int result = -1;
    struct dirent *entry = nullptr;

    while ((entry = readdir(dir)) != nullptr) {
        char *end = nullptr;
        long fd_long = std::strtol(entry->d_name, &end, 10);
        char fd_path[64];
        char link_target[PATH_MAX];
        ssize_t link_len;

        if (end == entry->d_name || *end != '\0' || fd_long < 0 ||
            fd_long > INT_MAX)
            continue;
        if ((int)fd_long == dir_fd)
            continue;

        std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%ld", fd_long);
        link_len = readlink(fd_path, link_target, sizeof(link_target) - 1);
        if (link_len < 0)
            continue;
        link_target[link_len] = '\0';
        if (!fd_target_is_uvm(link_target))
            continue;

        int dup_fd = fcntl((int)fd_long, F_DUPFD_CLOEXEC, 3);
        if (dup_fd < 0)
            continue;

        UvmInitializeParams params{};
        if (ioctl(dup_fd, kUvmInitializeIoctl, &params) == 0 &&
            params.rmStatus == 0) {
            result = dup_fd;
            break;
        }

        close(dup_fd);
    }

    closedir(dir);
    return result;
}

static int open_registration_uvm_fd()
{
    int fd = duplicate_existing_uvm_fd();
    return fd >= 0 ? fd : open_uvm_fd();
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

static bool reset_pool(int fd, const PhaseFUuid &gpu_uuid, const char *label)
{
    ExpertReservedPoolParams reset{};
    reset.mode = kReservedPoolModeReset;
    reset.gpu_uuid = gpu_uuid;
    int ret = ioctl(fd, kExpertReservedPoolIoctl, &reset);
    bool ok = ret == 0 && reset.rmStatus == 0;
    std::printf("reserved_pool_reset label=%s ret=%d rmStatus=%u ok=%u "
                "reset_count=%llu backing_bytes=%llu\n",
                label,
                ret,
                reset.rmStatus,
                ok ? 1u : 0u,
                reset.reset_count,
                reset.reserved_backing_bytes);
    return ok;
}

static PoolAttempt prealloc_slots(int fd,
                                  const PhaseFUuid &gpu_uuid,
                                  unsigned int pool_id,
                                  unsigned int slot_count,
                                  size_t slot_bytes,
                                  const char *label)
{
    PoolAttempt attempt;
    attempt.label = label;
    attempt.params.mode = kReservedPoolModePrealloc;
    attempt.params.count = slot_count;
    attempt.params.gpu_uuid = gpu_uuid;
    attempt.params.max_entries = slot_count;
    for (unsigned int slot = 0; slot < slot_count; ++slot) {
        attempt.params.entries[slot].pool_id = pool_id;
        attempt.params.entries[slot].slot_id = slot;
        attempt.params.entries[slot].slot_bytes = slot_bytes;
    }

    double start = now_ms();
    attempt.ioctl_ret = ioctl(fd, kExpertReservedPoolIoctl, &attempt.params);
    double elapsed_ms = now_ms() - start;
    attempt.rm_status = attempt.params.rmStatus;

    for (unsigned int i = 0; i < slot_count; ++i) {
        const ExpertReservedPoolEntry &entry = attempt.params.entries[i];
        if (entry.status == 0 && entry.physical_bytes >= slot_bytes)
            attempt.backed_slot_ids.push_back(entry.slot_id);
    }
    if (attempt.backed_slot_ids.empty() && attempt.params.success_count > 0) {
        for (unsigned int slot = 0;
             slot < std::min(slot_count, attempt.params.success_count);
             ++slot)
            attempt.backed_slot_ids.push_back(slot);
    }

    std::printf("reserved_pool_prealloc label=%s ret=%d rmStatus=%u "
                "requested_slots=%u success=%u failure=%u backed_slots=%u "
                "usable_entries=%zu backing_bytes=%llu logical_bytes=%llu "
                "alloc_failed_count=%llu current_epoch=%llu elapsed_ms=%.3f\n",
                label,
                attempt.ioctl_ret,
                attempt.rm_status,
                slot_count,
                attempt.params.success_count,
                attempt.params.failure_count,
                attempt.params.backed_slots,
                attempt.backed_slot_ids.size(),
                attempt.params.reserved_backing_bytes,
                attempt.params.logical_slot_bytes,
                attempt.params.alloc_failed_count,
                attempt.params.current_epoch,
                elapsed_ms);
    return attempt;
}

static size_t apply_device_pressure(size_t target_free_after_experts,
                                    size_t expert_gpu_bytes,
                                    size_t max_chunk_bytes,
                                    std::vector<void *> *pressure_allocs)
{
    if (target_free_after_experts == 0)
        return 0;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    const size_t target_before_expert_prefetch =
        expert_gpu_bytes + target_free_after_experts;
    if (free_bytes <= target_before_expert_prefetch) {
        std::printf("pressure_alloc skipped current_free=%zu "
                    "target_before_expert_prefetch=%zu\n",
                    free_bytes,
                    target_before_expert_prefetch);
        return 0;
    }

    size_t allocated = 0;
    size_t chunk_limit = max_chunk_bytes;
    const size_t min_chunk = 1ull * 1024ull * 1024ull;

    while (true) {
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes <= target_before_expert_prefetch + min_chunk)
            break;

        size_t wanted = free_bytes - target_before_expert_prefetch;
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
    std::printf("pressure_alloc target_free_after_experts=%zu "
                "target_free_before_expert_prefetch=%zu allocated_bytes=%zu "
                "chunks=%zu free_after_pressure=%zu total_bytes=%zu\n",
                target_free_after_experts,
                target_before_expert_prefetch,
                allocated,
                pressure_allocs->size(),
                free_bytes,
                total_bytes);
    return allocated;
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    const size_t words = args.expert_bytes / sizeof(unsigned int);
    const size_t total_expert_bytes =
        (size_t)args.expert_count * args.expert_bytes;
    const size_t offload_bytes =
        (size_t)args.offload_count * args.expert_bytes;
    const size_t requested_slot_bytes =
        (size_t)args.slot_count * args.slot_bytes;
    const int offload_start = args.expert_count - args.offload_count;
    const unsigned long long region_base_id = 0xABC700000001ull;

    CUDA_CHECK(cudaSetDevice(args.device));
    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
    CUuuid cuda_uuid{};
    CU_CHECK(cuDeviceGetUuid(&cuda_uuid, cu_device));

    std::printf("expert_reclaim_slot_smoke_begin device=%d gpu=%s "
                "expert_count=%d offload_count=%d expert_bytes=%zu "
                "total_expert_bytes=%zu offload_bytes=%zu slot_count=%u "
                "slot_bytes=%zu requested_slot_bytes=%zu "
                "target_free_after_experts=%zu control_before_offload=%u\n",
                args.device,
                prop.name,
                args.expert_count,
                args.offload_count,
                args.expert_bytes,
                total_expert_bytes,
                offload_bytes,
                args.slot_count,
                args.slot_bytes,
                requested_slot_bytes,
                args.target_free_after_experts,
                args.control_before_offload ? 1u : 0u);
    print_mem("start");

    std::vector<unsigned int *> experts(args.expert_count, nullptr);
    std::vector<unsigned long long> region_ids(args.expert_count, 0);
    std::vector<void *> pressure_allocs;

    double alloc_start = now_ms();
    for (int expert = 0; expert < args.expert_count; ++expert) {
        CUDA_CHECK(cudaMallocManaged(&experts[expert], args.expert_bytes));
        fill_pattern(experts[expert], words, expert);
        region_ids[expert] = region_base_id + (unsigned long long)expert;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_alloc_and_fill experts=%d bytes=%zu elapsed_ms=%.3f\n",
                args.expert_count,
                total_expert_bytes,
                now_ms() - alloc_start);
    print_mem("after_managed_alloc_cpu_fill");

    apply_device_pressure(args.target_free_after_experts,
                          total_expert_bytes,
                          args.pressure_chunk_bytes,
                          &pressure_allocs);
    print_mem("after_optional_pressure");

    double gpu_prefetch_start = now_ms();
    for (int expert = 0; expert < args.expert_count; ++expert)
        CUDA_CHECK(cudaMemPrefetchAsync(experts[expert],
                                        args.expert_bytes,
                                        args.device));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_prefetch_gpu experts=%d bytes=%zu elapsed_ms=%.3f\n",
                args.expert_count,
                total_expert_bytes,
                now_ms() - gpu_prefetch_start);
    print_mem("after_expert_gpu_prefetch");

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr, "failed to open or duplicate initialized UVM fd\n");
        return 1;
    }
    PhaseFUuid gpu_uuid{};
    std::memcpy(gpu_uuid.uuid, cuda_uuid.bytes, sizeof(gpu_uuid.uuid));

    bool final_ok = true;
    if (!reset_pool(fd, gpu_uuid, "initial"))
        final_ok = false;

    if (args.control_before_offload) {
        PoolAttempt before = prealloc_slots(fd,
                                            gpu_uuid,
                                            args.pool_id,
                                            args.slot_count,
                                            args.slot_bytes,
                                            "before_offload");
        if (before.params.success_count > 0)
            reset_pool(fd, gpu_uuid, "after_before_offload_control");
    }

    double offload_start_ms = now_ms();
    for (int expert = offload_start; expert < args.expert_count; ++expert)
        CUDA_CHECK(cudaMemPrefetchAsync(experts[expert],
                                        args.expert_bytes,
                                        cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_offload_cpu experts=%d first_expert=%d bytes=%zu "
                "elapsed_ms=%.3f\n",
                args.offload_count,
                offload_start,
                offload_bytes,
                now_ms() - offload_start_ms);
    print_mem("after_expert_cpu_offload");

    reset_pool(fd, gpu_uuid, "before_after_offload_prealloc");
    PoolAttempt after = prealloc_slots(fd,
                                       gpu_uuid,
                                       args.pool_id,
                                       args.slot_count,
                                       args.slot_bytes,
                                       "after_offload");

    if (args.require_full_after &&
        after.params.success_count != args.slot_count) {
        std::fprintf(stderr,
                     "after_offload did not allocate all requested slots: "
                     "requested=%u success=%u\n",
                     args.slot_count,
                     after.params.success_count);
        final_ok = false;
    }

    unsigned int remap_count = (unsigned int)std::min(
        std::min(after.backed_slot_ids.size(), (size_t)args.offload_count),
        (size_t)args.slot_count);

    std::vector<unsigned long long> registered_region_ids;
    registered_region_ids.reserve(remap_count);
    for (unsigned int i = 0; i < remap_count; ++i) {
        int expert = offload_start + (int)i;
        PhaseFRegisterRegionParams reg{};
        reg.region.region_id = region_ids[expert];
        reg.region.base = reinterpret_cast<unsigned long long>(experts[expert]);
        reg.region.length = args.expert_bytes;
        reg.region.pool = kPoolWeight;
        reg.region.layer_id = 0;
        reg.region.expert_id = (unsigned int)expert;
        reg.region.role = kRoleW13;
        reg.region.flags = kRegionFlagPrefetchable |
                           kRegionFlagGpuReadOnly |
                           kRegionFlagStreamingSubpool |
                           kRegionFlagCpuOffloadedReserved;
        int ret = ioctl(fd, kUvmPhaseFRegisterRegionIoctl, &reg);
        bool ok = ret == 0 && reg.rmStatus == 0;
        std::printf("register_offloaded_region index=%u expert=%d region_id=%llu "
                    "ret=%d rmStatus=%u ok=%u\n",
                    i,
                    expert,
                    reg.region.region_id,
                    ret,
                    reg.rmStatus,
                    ok ? 1u : 0u);
        if (!ok) {
            final_ok = false;
            break;
        }
        registered_region_ids.push_back(reg.region.region_id);
    }

    unsigned int *errors = nullptr;
    CUDA_CHECK(cudaMallocManaged(&errors, sizeof(unsigned int)));

    bool remap_ok = remap_count > 0 && registered_region_ids.size() == remap_count;
    if (remap_ok) {
        ExpertReservedPrefetchParams prefetch{};
        prefetch.mode = kReservedPrefetchModeReservedPrefetch;
        prefetch.count = remap_count;
        prefetch.gpu_uuid = gpu_uuid;
        prefetch.async = 1;
        for (unsigned int i = 0; i < remap_count; ++i) {
            prefetch.entries[i].region_id = registered_region_ids[i];
            prefetch.entries[i].pool_id = args.pool_id;
            prefetch.entries[i].slot_id = after.backed_slot_ids[i];
            prefetch.entries[i].flags = kStrictFootprint;
            prefetch.entries[i].bytes_override = args.expert_bytes;
        }

        double prefetch_start = now_ms();
        int ret = ioctl(fd, kExpertReservedPrefetchIoctl, &prefetch);
        double prefetch_ms = now_ms() - prefetch_start;
        remap_ok = ret == 0 &&
                   prefetch.rmStatus == 0 &&
                   prefetch.success_count == remap_count &&
                   prefetch.failure_count == 0 &&
                   prefetch.accepted_bytes ==
                       (unsigned long long)remap_count * args.expert_bytes &&
                   prefetch.footprint_contract_broken == 0;
        std::printf("reserved_prefetch_after_offload ret=%d rmStatus=%u "
                    "requested=%u success=%u failure=%u requested_bytes=%llu "
                    "accepted_bytes=%llu assigned_slots=%llu ordinary_bytes=%llu "
                    "external_allocs=%llu footprint_broken=%llu elapsed_ms=%.3f "
                    "ok=%u\n",
                    ret,
                    prefetch.rmStatus,
                    remap_count,
                    prefetch.success_count,
                    prefetch.failure_count,
                    prefetch.requested_bytes,
                    prefetch.accepted_bytes,
                    prefetch.assigned_slots,
                    prefetch.ordinary_resident_bytes,
                    prefetch.pool_external_alloc_count,
                    prefetch.footprint_contract_broken,
                    prefetch_ms,
                    remap_ok ? 1u : 0u);

        int block = 256;
        int grid = (int)((words + block - 1) / block);
        unsigned int total_errors = 0;
        for (unsigned int i = 0; i < remap_count; ++i) {
            int expert = offload_start + (int)i;
            *errors = 0;
            CUDA_CHECK(cudaDeviceSynchronize());
            check_pattern_kernel<<<grid, block>>>(
                experts[expert],
                words,
                expert_seed(expert),
                errors);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            total_errors += *errors;
            std::printf("reserved_slot_validate index=%u expert=%d errors=%u\n",
                        i,
                        expert,
                        *errors);
        }
        if (total_errors != 0)
            remap_ok = false;

        ExpertReservedPrefetchParams invalidate{};
        invalidate.mode = kReservedPrefetchModeInvalidate;
        invalidate.count = remap_count;
        invalidate.gpu_uuid = gpu_uuid;
        for (unsigned int i = 0; i < remap_count; ++i) {
            invalidate.entries[i].pool_id = args.pool_id;
            invalidate.entries[i].slot_id = after.backed_slot_ids[i];
            invalidate.entries[i].flags = kStrictFootprint;
        }
        int inv_ret = ioctl(fd, kExpertReservedPrefetchIoctl, &invalidate);
        std::printf("reserved_prefetch_invalidate ret=%d rmStatus=%u "
                    "requested=%u success=%u failure=%u invalidated=%llu\n",
                    inv_ret,
                    invalidate.rmStatus,
                    remap_count,
                    invalidate.success_count,
                    invalidate.failure_count,
                    invalidate.invalidated_slots);
    } else {
        std::printf("reserved_prefetch_after_offload skipped remap_count=%u "
                    "registered=%zu\n",
                    remap_count,
                    registered_region_ids.size());
        final_ok = false;
    }

    for (unsigned long long region_id : registered_region_ids) {
        PhaseFUnregisterRegionParams unreg{};
        unreg.region_id = region_id;
        (void)ioctl(fd, kUvmPhaseFUnregisterRegionIoctl, &unreg);
    }

    if (args.reset_after)
        reset_pool(fd, gpu_uuid, "final");
    close(fd);

    CUDA_CHECK(cudaFree(errors));
    for (void *ptr : pressure_allocs)
        CUDA_CHECK(cudaFree(ptr));
    for (unsigned int *ptr : experts)
        CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("after_cleanup");

    final_ok = final_ok && remap_ok;
    std::printf("expert_reclaim_slot_smoke_final ok=%u after_success=%u "
                "after_failure=%u remap_count=%u remap_ok=%u "
                "offload_bytes=%zu requested_slot_bytes=%zu\n",
                final_ok ? 1u : 0u,
                after.params.success_count,
                after.params.failure_count,
                remap_count,
                remap_ok ? 1u : 0u,
                offload_bytes,
                requested_slot_bytes);
    return final_ok ? 0 : 1;
}
