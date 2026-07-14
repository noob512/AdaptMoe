#include <algorithm>
#include <cerrno>
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

static constexpr unsigned int kUvmInitializeIoctl = 0x30000001;
static constexpr unsigned int kPhaseFBatchExecutorIoctl = 81;
static constexpr unsigned int kPhaseFRegisterRegionIoctl = 82;
static constexpr unsigned int kPhaseFUnregisterRegionIoctl = 83;
static constexpr unsigned int kExpertReservedPoolIoctl = 98;
static constexpr unsigned int kExpertOffloadReleaseIoctl = 100;

static constexpr unsigned int kMaxRanges = 64;
static constexpr unsigned int kMaxReservedEntries = 128;
static constexpr unsigned int kQueryResidencyAction = 4;
static constexpr unsigned int kReservedPoolModeQuery = 0;
static constexpr unsigned int kReservedPoolModePrealloc = 1;
static constexpr unsigned int kReservedPoolModeReset = 2;
static constexpr unsigned int kReservedPoolEntryFlagAllowEvict = 1u << 0;
static constexpr unsigned int kPoolWeight = 2;
static constexpr unsigned int kRoleGateUp = 3;
static constexpr unsigned int kRegionFlagPrefetchable = 1u << 3;
static constexpr unsigned int kRegionFlagGpuReadOnly = 1u << 9;
static constexpr unsigned int kRegionFlagStreamingSubpool = 1u << 12;
static constexpr unsigned int kRegionFlagCpuOffloadedReserved = 1u << 14;
static constexpr unsigned int kExpertReleaseRequireExpert = 1u << 0;

struct UvmInitializeParams {
    unsigned long long flags = 0;
    unsigned int rmStatus = 0;
};

struct PhaseFUuid {
    unsigned char uuid[16] = {};
};

struct PhaseFRange {
    unsigned long long base = 0;
    unsigned long long length = 0;
};

struct PhaseFRangeResult {
    unsigned int status = 0;
    unsigned int resident_on_count = 0;
    unsigned char resident_on_cpu = 0;
    unsigned char resident_on_gpu = 0;
    unsigned long long queried_bytes = 0;
    unsigned long long resident_gpu_bytes = 0;
    unsigned long long resident_cpu_bytes = 0;
    unsigned long long resident_any_bytes = 0;
    unsigned long long resident_unknown_bytes = 0;
};

struct PhaseFBatchExecutorParams {
    unsigned int action = 0;
    unsigned int count = 0;
    PhaseFUuid gpu_uuid = {};
    unsigned char async = 0;
    unsigned int success_count = 0;
    unsigned int failure_count = 0;
    PhaseFRange ranges[kMaxRanges] = {};
    PhaseFRangeResult results[kMaxRanges] = {};
    unsigned int rmStatus = 0;
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
    PhaseFRegion region = {};
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
    ExpertReservedPoolEntry entries[kMaxReservedEntries] = {};
    unsigned int rmStatus = 0;
};

struct ExpertOffloadReleaseParams {
    unsigned long long region_id = 0;
    PhaseFUuid gpu_uuid = {};
    unsigned int flags = 0;
    unsigned int max_blocks = 0;
    unsigned int pool = 0;
    unsigned int layer_id = 0;
    unsigned int expert_id = 0;
    unsigned int role = 0;
    unsigned long long base = 0;
    unsigned long long length = 0;
    unsigned long long block_count = 0;
    unsigned int attempted_blocks = 0;
    unsigned int success_blocks = 0;
    unsigned int failure_blocks = 0;
    unsigned int skipped_blocks = 0;
    unsigned long long attempted_bytes = 0;
    unsigned long long released_bytes = 0;
    unsigned long long failed_bytes = 0;
    unsigned long long first_failed_block = 0;
    unsigned int first_error = 0;
    unsigned int rmStatus = 0;
};

struct Args {
    int device = 0;
    unsigned int expert_count = 8;
    size_t expert_bytes = 64ull * 1024ull * 1024ull;
    unsigned int slot_count = 48;
    size_t slot_bytes = 4ull * 1024ull * 1024ull;
    unsigned int pool_id = 0;
    size_t target_free_before_slots = 128ull * 1024ull * 1024ull;
    size_t pressure_chunk_bytes = 256ull * 1024ull * 1024ull;
    bool allow_evict = false;
    bool skip_before_prealloc = false;
    bool use_driver_release = false;
    unsigned long long region_id_base = 0x0ff1000000000000ull;
};

struct ResidencySummary {
    unsigned long long queried = 0;
    unsigned long long gpu_bytes = 0;
    unsigned long long cpu_bytes = 0;
    unsigned long long any_bytes = 0;
    unsigned long long unknown_bytes = 0;
    unsigned int success_count = 0;
    unsigned int failure_count = 0;
};

struct PreallocAttempt {
    int ioctl_ret = -1;
    int saved_errno = 0;
    unsigned int rm_status = 0;
    unsigned int success_count = 0;
    unsigned int failure_count = 0;
    unsigned int backed_slots = 0;
    unsigned long long reserved_backing_bytes = 0;
    unsigned long long logical_slot_bytes = 0;
    unsigned long long alloc_failed_count = 0;
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

static void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "usage: %s [--device N] [--expert-count N] "
                 "[--expert-bytes N|NM|NG] [--slot-count N] "
                 "[--slot-bytes N|NM|NG] [--pool-id N] "
                 "[--target-free-before-slots N|NM|NG] "
                 "[--pressure-chunk-bytes N|NM|NG] [--allow-evict] "
                 "[--skip-before-prealloc] [--use-driver-release] "
                 "[--region-id-base N]\n",
                 argv0);
}

static Args parse_args(int argc, char **argv)
{
    Args args;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--expert-count") == 0 && i + 1 < argc)
            args.expert_count = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--expert-bytes") == 0 && i + 1 < argc)
            args.expert_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--slot-count") == 0 && i + 1 < argc)
            args.slot_count = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--slot-bytes") == 0 && i + 1 < argc)
            args.slot_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--pool-id") == 0 && i + 1 < argc)
            args.pool_id = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--target-free-before-slots") == 0 &&
                 i + 1 < argc)
            args.target_free_before_slots = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--pressure-chunk-bytes") == 0 &&
                 i + 1 < argc)
            args.pressure_chunk_bytes = parse_size(argv[++i]);
        else if (std::strcmp(argv[i], "--allow-evict") == 0)
            args.allow_evict = true;
        else if (std::strcmp(argv[i], "--skip-before-prealloc") == 0)
            args.skip_before_prealloc = true;
        else if (std::strcmp(argv[i], "--use-driver-release") == 0)
            args.use_driver_release = true;
        else if (std::strcmp(argv[i], "--region-id-base") == 0 && i + 1 < argc)
            args.region_id_base = std::strtoull(argv[++i], nullptr, 0);
        else {
            usage(argv[0]);
            std::exit(2);
        }
    }

    if (args.expert_count == 0 ||
        args.expert_count > kMaxRanges ||
        args.expert_bytes == 0 ||
        args.slot_count == 0 ||
        args.slot_count > kMaxReservedEntries ||
        args.slot_bytes == 0 ||
        args.pressure_chunk_bytes == 0) {
        usage(argv[0]);
        std::exit(2);
    }

    const size_t requested_slot_bytes = (size_t)args.slot_count * args.slot_bytes;
    if ((size_t)args.expert_count * args.expert_bytes < requested_slot_bytes) {
        std::fprintf(stderr,
                     "invalid arguments: total expert bytes should be >= "
                     "requested slot bytes for a clean reuse test\n");
        std::exit(2);
    }

    return args;
}

static unsigned long long region_id_for(const Args &args, unsigned int expert)
{
    return args.region_id_base + expert;
}

static int open_uvm_fd()
{
    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0)
        return -1;

    UvmInitializeParams params{};
    if (ioctl(fd, kUvmInitializeIoctl, &params) != 0 || params.rmStatus != 0) {
        std::fprintf(stderr, "uvm initialize failed rmStatus=%u\n", params.rmStatus);
        close(fd);
        return -1;
    }

    return fd;
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

        if (end == entry->d_name || *end != '\0' ||
            fd_long < 0 || fd_long > INT_MAX)
            continue;
        if ((int)fd_long == dir_fd)
            continue;

        std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%ld", fd_long);
        ssize_t link_len = readlink(fd_path, link_target, sizeof(link_target) - 1);
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

static PhaseFUuid gpu_uuid_for_device(int device)
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    PhaseFUuid uuid{};
    std::memcpy(uuid.uuid, prop.uuid.bytes, sizeof(uuid.uuid));
    return uuid;
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

static ResidencySummary query_residency(int fd,
                                        const PhaseFUuid &gpu_uuid,
                                        const std::vector<void *> &experts,
                                        size_t expert_bytes,
                                        const char *label)
{
    ResidencySummary summary{};
    size_t offset = 0;

    while (offset < experts.size()) {
        const size_t batch_count = std::min<size_t>(kMaxRanges,
                                                   experts.size() - offset);
        PhaseFBatchExecutorParams params{};
        params.action = kQueryResidencyAction;
        params.count = (unsigned int)batch_count;
        params.gpu_uuid = gpu_uuid;

        for (size_t i = 0; i < batch_count; ++i) {
            params.ranges[i].base =
                reinterpret_cast<unsigned long long>(experts[offset + i]);
            params.ranges[i].length = expert_bytes;
        }

        int ret = ioctl(fd, kPhaseFBatchExecutorIoctl, &params);
        if (ret != 0 || params.rmStatus != 0) {
            std::perror("query_residency");
            std::printf("residency_summary label=%s ret=%d rmStatus=%u "
                        "success=%u failure=%u queried_bytes=%llu "
                        "gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu "
                        "unknown_bytes=%llu\n",
                        label,
                        ret,
                        params.rmStatus,
                        summary.success_count,
                        summary.failure_count + (unsigned int)batch_count,
                        summary.queried,
                        summary.gpu_bytes,
                        summary.cpu_bytes,
                        summary.any_bytes,
                        summary.unknown_bytes);
            summary.failure_count += (unsigned int)batch_count;
            return summary;
        }

        summary.success_count += params.success_count;
        summary.failure_count += params.failure_count;
        for (size_t i = 0; i < batch_count; ++i) {
            summary.queried += params.results[i].queried_bytes;
            summary.gpu_bytes += params.results[i].resident_gpu_bytes;
            summary.cpu_bytes += params.results[i].resident_cpu_bytes;
            summary.any_bytes += params.results[i].resident_any_bytes;
            summary.unknown_bytes += params.results[i].resident_unknown_bytes;
        }
        offset += batch_count;
    }

    std::printf("residency_summary label=%s success=%u failure=%u "
                "queried_bytes=%llu gpu_bytes=%llu cpu_bytes=%llu "
                "any_bytes=%llu unknown_bytes=%llu\n",
                label,
                summary.success_count,
                summary.failure_count,
                summary.queried,
                summary.gpu_bytes,
                summary.cpu_bytes,
                summary.any_bytes,
                summary.unknown_bytes);
    return summary;
}

static void pool_query_or_reset(int fd,
                                const PhaseFUuid &gpu_uuid,
                                unsigned int mode,
                                const char *label)
{
    ExpertReservedPoolParams params{};
    params.mode = mode;
    params.gpu_uuid = gpu_uuid;
    params.max_entries = kMaxReservedEntries;
    int ret = ioctl(fd, kExpertReservedPoolIoctl, &params);
    std::printf("reserved_pool_%s ret=%d rmStatus=%u active_slots=%u "
                "backed_slots=%u returned_slots=%u backing_bytes=%llu "
                "logical_bytes=%llu alloc_failed_count=%llu reset_count=%llu "
                "prealloc_count=%llu\n",
                label,
                ret,
                params.rmStatus,
                params.active_slots,
                params.backed_slots,
                params.returned_slots,
                params.reserved_backing_bytes,
                params.logical_slot_bytes,
                params.alloc_failed_count,
                params.reset_count,
                params.prealloc_count);
}

static void register_region(int fd,
                            const Args &args,
                            unsigned int expert,
                            const void *ptr)
{
    PhaseFRegisterRegionParams params{};
    params.region.region_id = region_id_for(args, expert);
    params.region.base = reinterpret_cast<unsigned long long>(ptr);
    params.region.length = args.expert_bytes;
    params.region.pool = kPoolWeight;
    params.region.layer_id = expert;
    params.region.expert_id = expert;
    params.region.role = kRoleGateUp;
    params.region.flags = kRegionFlagPrefetchable |
                          kRegionFlagGpuReadOnly |
                          kRegionFlagStreamingSubpool |
                          kRegionFlagCpuOffloadedReserved;

    int ret = ioctl(fd, kPhaseFRegisterRegionIoctl, &params);
    std::printf("register_region expert=%u region_id=%llu ret=%d rmStatus=%u "
                "base=0x%llx length=%zu\n",
                expert,
                params.region.region_id,
                ret,
                params.rmStatus,
                params.region.base,
                args.expert_bytes);
    if (ret != 0 || params.rmStatus != 0)
        std::exit(1);
}

static void unregister_region(int fd, unsigned long long region_id)
{
    PhaseFUnregisterRegionParams params{};
    params.region_id = region_id;
    (void)ioctl(fd, kPhaseFUnregisterRegionIoctl, &params);
}

static bool release_region(int fd,
                           const PhaseFUuid &gpu_uuid,
                           const Args &args,
                           unsigned int expert,
                           unsigned long long *released_bytes_out)
{
    ExpertOffloadReleaseParams params{};
    params.region_id = region_id_for(args, expert);
    params.gpu_uuid = gpu_uuid;
    params.flags = kExpertReleaseRequireExpert;

    int ret = ioctl(fd, kExpertOffloadReleaseIoctl, &params);
    bool ok = ret == 0 && params.rmStatus == 0;
    std::printf("expert_offload_release expert=%u region_id=%llu ret=%d "
                "rmStatus=%u blocks=%llu attempted=%u success=%u failure=%u "
                "skipped=%u attempted_bytes=%llu released_bytes=%llu "
                "failed_bytes=%llu first_error=%u first_failed=0x%llx\n",
                expert,
                params.region_id,
                ret,
                params.rmStatus,
                params.block_count,
                params.attempted_blocks,
                params.success_blocks,
                params.failure_blocks,
                params.skipped_blocks,
                params.attempted_bytes,
                params.released_bytes,
                params.failed_bytes,
                params.first_error,
                params.first_failed_block);
    if (released_bytes_out)
        *released_bytes_out += params.released_bytes;
    return ok;
}

static PreallocAttempt prealloc_slots(int fd,
                                      const PhaseFUuid &gpu_uuid,
                                      const Args &args,
                                      const char *label)
{
    ExpertReservedPoolParams params{};
    params.mode = kReservedPoolModePrealloc;
    params.count = args.slot_count;
    params.gpu_uuid = gpu_uuid;
    params.max_entries = kMaxReservedEntries;

    for (unsigned int slot = 0; slot < args.slot_count; ++slot) {
        params.entries[slot].pool_id = args.pool_id;
        params.entries[slot].slot_id = slot;
        params.entries[slot].slot_bytes = args.slot_bytes;
        if (args.allow_evict)
            params.entries[slot].flags |= kReservedPoolEntryFlagAllowEvict;
    }

    double start = now_ms();
    errno = 0;
    int ret = ioctl(fd, kExpertReservedPoolIoctl, &params);
    int saved_errno = errno;
    double elapsed_ms = now_ms() - start;

    unsigned int status_hist[8] = {};
    for (unsigned int i = 0; i < args.slot_count; ++i) {
        unsigned int status = params.entries[i].status;
        if (status < 8)
            ++status_hist[status];
    }

    PreallocAttempt attempt{};
    attempt.ioctl_ret = ret;
    attempt.saved_errno = saved_errno;
    attempt.rm_status = params.rmStatus;
    attempt.success_count = params.success_count;
    attempt.failure_count = params.failure_count;
    attempt.backed_slots = params.backed_slots;
    attempt.reserved_backing_bytes = params.reserved_backing_bytes;
    attempt.logical_slot_bytes = params.logical_slot_bytes;
    attempt.alloc_failed_count = params.alloc_failed_count;

    std::printf("reserved_pool_prealloc label=%s ret=%d errno=%d rmStatus=%u "
                "requested_slots=%u slot_bytes=%zu required_bytes=%zu "
                "success=%u failure=%u active_slots=%u backed_slots=%u "
                "backing_bytes=%llu logical_bytes=%llu alloc_failed_count=%llu "
                "allow_evict=%u elapsed_ms=%.3f status_ok=%u "
                "status_invalid_arg=%u status_invalid_device=%u "
                "status_in_use=%u status_backing_failed=%u "
                "status_pool_full=%u\n",
                label,
                ret,
                saved_errno,
                params.rmStatus,
                args.slot_count,
                args.slot_bytes,
                (size_t)args.slot_count * args.slot_bytes,
                params.success_count,
                params.failure_count,
                params.active_slots,
                params.backed_slots,
                params.reserved_backing_bytes,
                params.logical_slot_bytes,
                params.alloc_failed_count,
                args.allow_evict ? 1u : 0u,
                elapsed_ms,
                status_hist[0],
                status_hist[1],
                status_hist[2],
                status_hist[3],
                status_hist[4],
                status_hist[5]);
    return attempt;
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

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);

    Args args = parse_args(argc, argv);
    const size_t total_expert_bytes = (size_t)args.expert_count * args.expert_bytes;
    const size_t required_slot_bytes = (size_t)args.slot_count * args.slot_bytes;

    CUDA_CHECK(cudaSetDevice(args.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
    PhaseFUuid gpu_uuid = gpu_uuid_for_device(args.device);

    std::printf("expert_offload_slot_reuse_smoke_begin device=%d gpu=%s "
                "expert_count=%u expert_bytes=%zu total_expert_bytes=%zu "
                "slot_count=%u slot_bytes=%zu required_slot_bytes=%zu "
                "pool_id=%u target_free_before_slots=%zu "
                "pressure_chunk_bytes=%zu allow_evict=%u "
                "skip_before_prealloc=%u use_driver_release=%u "
                "region_id_base=%llu\n",
                args.device,
                prop.name,
                args.expert_count,
                args.expert_bytes,
                total_expert_bytes,
                args.slot_count,
                args.slot_bytes,
                required_slot_bytes,
                args.pool_id,
                args.target_free_before_slots,
                args.pressure_chunk_bytes,
                args.allow_evict ? 1u : 0u,
                args.skip_before_prealloc ? 1u : 0u,
                args.use_driver_release ? 1u : 0u,
                args.region_id_base);

    print_mem("start");

    void *uvm_seed = nullptr;
    CUDA_CHECK(cudaMallocManaged(&uvm_seed, 4096));
    CUDA_CHECK(cudaDeviceSynchronize());

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::perror("open_registration_uvm_fd");
        CUDA_CHECK(cudaFree(uvm_seed));
        return 1;
    }

    pool_query_or_reset(fd, gpu_uuid, kReservedPoolModeReset, "reset_start");
    pool_query_or_reset(fd, gpu_uuid, kReservedPoolModeQuery, "query_start");
    CUDA_CHECK(cudaFree(uvm_seed));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<void *> experts(args.expert_count, nullptr);
    std::vector<unsigned long long> expert_bases(args.expert_count, 0);
    std::vector<void *> pressure_allocs;

    double alloc_start = now_ms();
    for (unsigned int expert = 0; expert < args.expert_count; ++expert) {
        CUDA_CHECK(cudaMallocManaged(&experts[expert], args.expert_bytes));
        expert_bases[expert] = reinterpret_cast<unsigned long long>(experts[expert]);
        std::printf("expert_va_alloc expert=%u region_id=%llu base=0x%llx "
                    "bytes=%zu\n",
                    expert,
                    region_id_for(args, expert),
                    expert_bases[expert],
                    args.expert_bytes);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_alloc_managed experts=%u bytes=%zu elapsed_ms=%.3f\n",
                args.expert_count,
                total_expert_bytes,
                now_ms() - alloc_start);
    print_mem("after_managed_alloc");

    double gpu_prefetch_start = now_ms();
    for (unsigned int expert = 0; expert < args.expert_count; ++expert) {
        CUDA_CHECK(cudaMemPrefetchAsync(experts[expert],
                                        args.expert_bytes,
                                        args.device));
        CUDA_CHECK(cudaMemsetAsync(experts[expert],
                                   0x5a,
                                   args.expert_bytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("expert_prefetch_gpu experts=%u bytes=%zu elapsed_ms=%.3f\n",
                args.expert_count,
                total_expert_bytes,
                now_ms() - gpu_prefetch_start);
    print_mem("after_expert_gpu_prefetch");
    ResidencySummary after_gpu = query_residency(fd,
                                                 gpu_uuid,
                                                 experts,
                                                 args.expert_bytes,
                                                 "after_expert_gpu_prefetch");

    allocate_pressure(args.target_free_before_slots,
                      args.pressure_chunk_bytes,
                      &pressure_allocs);
    print_mem("after_pressure_before_prealloc");

    PreallocAttempt before{};
    bool before_skipped = args.skip_before_prealloc;
    if (before_skipped) {
        std::printf("reserved_pool_prealloc label=before_offload skipped=1\n");
    }
    else {
        before = prealloc_slots(fd, gpu_uuid, args, "before_offload");
        pool_query_or_reset(fd, gpu_uuid, kReservedPoolModeReset,
                            "reset_after_before_prealloc");
    }
    print_mem("after_before_prealloc_reset");

    double cpu_prefetch_start = now_ms();
    unsigned long long driver_released_bytes = 0;
    if (args.use_driver_release) {
        for (unsigned int expert = 0; expert < args.expert_count; ++expert)
            register_region(fd, args, expert, experts[expert]);

        bool release_ok = true;
        for (unsigned int expert = 0; expert < args.expert_count; ++expert) {
            if (!release_region(fd,
                                gpu_uuid,
                                args,
                                expert,
                                &driver_released_bytes)) {
                release_ok = false;
            }
        }
        if (!release_ok)
            std::printf("expert_offload_release aggregate_ok=0\n");
    }
    else {
        for (unsigned int expert = 0; expert < args.expert_count; ++expert) {
            CUDA_CHECK(cudaMemPrefetchAsync(experts[expert],
                                            args.expert_bytes,
                                            cudaCpuDeviceId));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::printf("expert_offload_cpu method=%s experts=%u bytes=%zu "
                "driver_released_bytes=%llu elapsed_ms=%.3f\n",
                args.use_driver_release ? "driver_release" : "cuda_prefetch",
                args.expert_count,
                total_expert_bytes,
                driver_released_bytes,
                now_ms() - cpu_prefetch_start);
    print_mem("after_expert_cpu_offload_before_slots");
    ResidencySummary after_cpu = query_residency(fd,
                                                 gpu_uuid,
                                                 experts,
                                                 args.expert_bytes,
                                                 "after_expert_cpu_offload");

    PreallocAttempt after = prealloc_slots(fd, gpu_uuid, args, "after_offload");
    pool_query_or_reset(fd, gpu_uuid, kReservedPoolModeQuery,
                        "query_after_offload_prealloc");

    bool va_stable = true;
    for (unsigned int expert = 0; expert < args.expert_count; ++expert) {
        const unsigned long long current_base =
            reinterpret_cast<unsigned long long>(experts[expert]);
        if (current_base != expert_bases[expert])
            va_stable = false;
    }
    std::printf("expert_va_stability experts=%u stable=%u first_base=0x%llx "
                "last_base=0x%llx\n",
                args.expert_count,
                va_stable ? 1u : 0u,
                expert_bases.empty() ? 0ull : expert_bases.front(),
                expert_bases.empty() ? 0ull : expert_bases.back());

    const bool before_all_success =
        !before_skipped && before.success_count == args.slot_count &&
        before.reserved_backing_bytes >= required_slot_bytes;
    const bool after_all_success =
        after.success_count == args.slot_count &&
        after.reserved_backing_bytes >= required_slot_bytes;
    const bool offload_released =
        after_gpu.gpu_bytes >= required_slot_bytes &&
        after_cpu.gpu_bytes + required_slot_bytes <= after_gpu.gpu_bytes &&
        after_cpu.cpu_bytes >= required_slot_bytes;
    const bool strong_evidence =
        (!before_all_success || before_skipped) && after_all_success &&
        offload_released;
    const bool weak_evidence = after_all_success && offload_released && va_stable;

    std::printf("expert_offload_slot_reuse_smoke_final before_prealloc_skipped=%u "
                "before_all_success=%u before_success=%u before_failure=%u "
                "after_all_success=%u after_success=%u after_failure=%u "
                "offload_released=%u strong_evidence=%u weak_evidence=%u "
                "va_stable=%u "
                "gpu_resident_before_offload=%llu gpu_resident_after_offload=%llu "
                "cpu_resident_after_offload=%llu required_slot_bytes=%zu "
                "after_reserved_backing_bytes=%llu\n",
                before_skipped ? 1u : 0u,
                before_all_success ? 1u : 0u,
                before.success_count,
                before.failure_count,
                after_all_success ? 1u : 0u,
                after.success_count,
                after.failure_count,
                offload_released ? 1u : 0u,
                strong_evidence ? 1u : 0u,
                weak_evidence ? 1u : 0u,
                va_stable ? 1u : 0u,
                after_gpu.gpu_bytes,
                after_cpu.gpu_bytes,
                after_cpu.cpu_bytes,
                required_slot_bytes,
                after.reserved_backing_bytes);

    pool_query_or_reset(fd, gpu_uuid, kReservedPoolModeReset, "reset_final");

    if (args.use_driver_release) {
        for (unsigned int expert = 0; expert < args.expert_count; ++expert)
            unregister_region(fd, region_id_for(args, expert));
    }

    for (void *ptr : pressure_allocs)
        CUDA_CHECK(cudaFree(ptr));
    for (void *ptr : experts)
        CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("after_cleanup");
    close(fd);

    return weak_evidence ? 0 : 1;
}
