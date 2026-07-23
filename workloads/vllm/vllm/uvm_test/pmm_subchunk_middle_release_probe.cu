#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
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
static constexpr unsigned int kPhaseNQueryRegionBlocksIoctl = 96;
static constexpr unsigned int kExpertOffloadReleaseIoctl = 100;

static constexpr size_t kMiB = 1024ull * 1024ull;
static constexpr size_t kTotalBytes = 6ull * kMiB;
static constexpr size_t kReleaseOffset = 2ull * kMiB;
static constexpr size_t kReleaseBytes = 1536ull * 1024ull;
static constexpr size_t kEnvelopeBytes = 512ull * 1024ull;
static constexpr unsigned int kEnvelopeCount = 3;
static constexpr unsigned int kMaxRanges = 64;
static constexpr unsigned int kTraceMax = 16;
static constexpr unsigned int kQueryResidencyAction = 4;

static constexpr unsigned int kPoolWeight = 2;
static constexpr unsigned int kRoleMoeExpertW13 = 3;
static constexpr unsigned int kRegionFlagPrefetchable = 1u << 3;
static constexpr unsigned int kRegionFlagGpuReadOnly = 1u << 9;
static constexpr unsigned int kRegionFlagStreamingSubpool = 1u << 12;
static constexpr unsigned int kRegionFlagCpuOffloadedReserved = 1u << 14;
static constexpr unsigned int kRegionFlagExpertEnvelope = 1u << 15;
static constexpr unsigned int kReleaseRequireExpert = 1u << 0;
static constexpr unsigned int kReleaseSubchunk64K = 1u << 2;

static constexpr unsigned long long kObserverRegionId = 0x6dd0000000000001ull;
static constexpr unsigned long long kEnvelopeRegionBase = 0x6dd0000000001000ull;

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

struct PhaseNQueryRegionBlocksParams {
    unsigned long long region_id = 0;
    unsigned int gpu_id = 0;
    unsigned int reserved = 0;
    unsigned long long base = 0;
    unsigned long long length = 0;
    unsigned int pool = 0;
    unsigned int layer_id = 0;
    unsigned int expert_id = 0;
    unsigned int role = 0;
    unsigned int flags = 0;
    unsigned int state = 0;
    unsigned int priority = 0;
    unsigned int va_range_count = 0;
    unsigned long long logical_block_count = 0;
    unsigned long long populated_block_count = 0;
    unsigned long long resident_block_count = 0;
    unsigned long long pending_block_count = 0;
    unsigned long long materialized_block_count = 0;
    unsigned long long mixed_block_count = 0;
    unsigned long long resident_gpu_bytes = 0;
    unsigned long long pending_gpu_bytes = 0;
    unsigned long long first_block_start = 0;
    unsigned long long last_block_end = 0;
    unsigned int sampled_block_count = 0;
    unsigned int reserved_out = 0;
    unsigned long long block_starts[kTraceMax] = {};
    unsigned long long block_ends[kTraceMax] = {};
    unsigned long long block_bytes[kTraceMax] = {};
    unsigned int block_states[kTraceMax] = {};
    unsigned int block_status[kTraceMax] = {};
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
    bool envelope = true;
    bool retouch = true;
};

__global__ void verify_pattern_kernel(const unsigned int *ptr,
                                      size_t words,
                                      unsigned int *error_count)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (; index < words; index += stride) {
        unsigned int expected = 0x6d000000u ^ static_cast<unsigned int>(index);
        if (ptr[index] != expected)
            atomicAdd(error_count, 1u);
    }
}

static bool parse_args(int argc, char **argv, Args *args)
{
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            args->device = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--envelope") == 0) {
            args->envelope = true;
        }
        else if (std::strcmp(argv[i], "--legacy") == 0) {
            args->envelope = false;
        }
        else if (std::strcmp(argv[i], "--no-retouch") == 0) {
            args->retouch = false;
        }
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--envelope|--legacy] "
                         "[--no-retouch]\n",
                         argv[0]);
            return false;
        }
    }
    return true;
}

static int open_uvm_fd()
{
    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0)
        return -1;

    UvmInitializeParams params{};
    if (ioctl(fd, kUvmInitializeIoctl, &params) != 0 || params.rmStatus != 0) {
        std::fprintf(stderr, "uvm initialize failed rmStatus=%u errno=%d\n",
                     params.rmStatus, errno);
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
        if (static_cast<int>(fd_long) == dir_fd)
            continue;

        std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%ld", fd_long);
        ssize_t link_len = readlink(fd_path, link_target, sizeof(link_target) - 1);
        if (link_len < 0)
            continue;
        link_target[link_len] = '\0';
        if (!fd_target_is_uvm(link_target))
            continue;

        int dup_fd = fcntl(static_cast<int>(fd_long), F_DUPFD_CLOEXEC, 3);
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
    if (fd >= 0) {
        std::printf("uvm_fd source=duplicated fd=%d\n", fd);
        return fd;
    }

    fd = open_uvm_fd();
    if (fd >= 0)
        std::printf("uvm_fd source=new fd=%d\n", fd);
    return fd;
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

static void fill_pattern(unsigned int *ptr, size_t bytes)
{
    const size_t words = bytes / sizeof(unsigned int);
    for (size_t i = 0; i < words; ++i)
        ptr[i] = 0x6d000000u ^ static_cast<unsigned int>(i);
}

static PhaseFRangeResult query_residency(int fd,
                                         const PhaseFUuid &gpu_uuid,
                                         const void *ptr,
                                         size_t bytes,
                                         const char *label)
{
    PhaseFBatchExecutorParams params{};
    params.action = kQueryResidencyAction;
    params.count = 1;
    params.gpu_uuid = gpu_uuid;
    params.ranges[0].base =
        reinterpret_cast<unsigned long long>(ptr);
    params.ranges[0].length =
        static_cast<unsigned long long>(bytes);

    int ret = ioctl(fd, kPhaseFBatchExecutorIoctl, &params);
    int saved_errno = errno;
    const PhaseFRangeResult &result = params.results[0];
    std::printf("range_residency label=%s ret=%d errno=%d rmStatus=%u "
                "status=%u success=%u failure=%u base=0x%llx bytes=%zu "
                "queried=%llu gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu "
                "unknown_bytes=%llu resident_on_gpu=%u resident_on_cpu=%u\n",
                label,
                ret,
                saved_errno,
                params.rmStatus,
                result.status,
                params.success_count,
                params.failure_count,
                params.ranges[0].base,
                bytes,
                result.queried_bytes,
                result.resident_gpu_bytes,
                result.resident_cpu_bytes,
                result.resident_any_bytes,
                result.resident_unknown_bytes,
                result.resident_on_gpu ? 1u : 0u,
                result.resident_on_cpu ? 1u : 0u);
    if (ret != 0 || params.rmStatus != 0 || result.status != 0)
        std::exit(1);
    return result;
}

static bool gpu_verify_all(const unsigned int *ptr, size_t bytes, const char *label)
{
    unsigned int *error_count = nullptr;
    CUDA_CHECK(cudaMallocManaged(&error_count, sizeof(*error_count)));
    *error_count = 0;
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t words = bytes / sizeof(unsigned int);
    const int threads = 256;
    const int blocks = 256;
    verify_pattern_kernel<<<blocks, threads>>>(ptr, words, error_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = *error_count == 0;
    std::printf("gpu_verify_all label=%s bytes=%zu words=%zu errors=%u ok=%u\n",
                label,
                bytes,
                words,
                *error_count,
                ok ? 1u : 0u);
    CUDA_CHECK(cudaFree(error_count));
    return ok;
}

static void register_region(int fd,
                            unsigned long long region_id,
                            uintptr_t base,
                            size_t length,
                            unsigned int flags,
                            unsigned int expert_id,
                            const char *label)
{
    PhaseFRegisterRegionParams params{};
    params.region.region_id = region_id;
    params.region.base = static_cast<unsigned long long>(base);
    params.region.length = static_cast<unsigned long long>(length);
    params.region.pool = kPoolWeight;
    params.region.layer_id = 0;
    params.region.expert_id = expert_id;
    params.region.role = kRoleMoeExpertW13;
    params.region.flags = flags;

    int ret = ioctl(fd, kPhaseFRegisterRegionIoctl, &params);
    int saved_errno = errno;
    std::printf("register_region label=%s region_id=%llu ret=%d rmStatus=%u "
                "errno=%d base=0x%llx length=%llu flags=0x%x\n",
                label,
                region_id,
                ret,
                params.rmStatus,
                saved_errno,
                params.region.base,
                params.region.length,
                flags);
    if (ret != 0 || params.rmStatus != 0)
        std::exit(1);
}

static void unregister_region(int fd,
                              unsigned long long region_id,
                              const char *label)
{
    PhaseFUnregisterRegionParams params{};
    params.region_id = region_id;
    int ret = ioctl(fd, kPhaseFUnregisterRegionIoctl, &params);
    std::printf("unregister_region label=%s region_id=%llu ret=%d rmStatus=%u\n",
                label,
                region_id,
                ret,
                params.rmStatus);
}

static PhaseNQueryRegionBlocksParams query_region_blocks(int fd,
                                                         unsigned long long region_id,
                                                         const char *label)
{
    PhaseNQueryRegionBlocksParams params{};
    params.region_id = region_id;
    int ret = ioctl(fd, kPhaseNQueryRegionBlocksIoctl, &params);
    std::printf("query_region_blocks label=%s region_id=%llu ret=%d rmStatus=%u "
                "base=0x%llx length=%llu flags=0x%x va_ranges=%u "
                "logical_blocks=%llu populated_blocks=%llu resident_blocks=%llu "
                "materialized_blocks=%llu mixed_blocks=%llu resident_gpu_bytes=%llu "
                "first=0x%llx last=0x%llx sampled=%u\n",
                label,
                region_id,
                ret,
                params.rmStatus,
                params.base,
                params.length,
                params.flags,
                params.va_range_count,
                params.logical_block_count,
                params.populated_block_count,
                params.resident_block_count,
                params.materialized_block_count,
                params.mixed_block_count,
                params.resident_gpu_bytes,
                params.first_block_start,
                params.last_block_end,
                params.sampled_block_count);
    for (unsigned int i = 0; i < params.sampled_block_count; ++i) {
        std::printf("query_region_block_sample label=%s index=%u start=0x%llx "
                    "end=0x%llx bytes=%llu state=0x%x status=%u\n",
                    label,
                    i,
                    params.block_starts[i],
                    params.block_ends[i],
                    params.block_bytes[i],
                    params.block_states[i],
                    params.block_status[i]);
    }
    if (ret != 0 || params.rmStatus != 0)
        std::exit(1);
    return params;
}

static PhaseNQueryRegionBlocksParams observer_query(int fd,
                                                    uintptr_t base,
                                                    const char *label)
{
    const unsigned int flags = kRegionFlagPrefetchable |
                               kRegionFlagGpuReadOnly |
                               kRegionFlagStreamingSubpool |
                               kRegionFlagCpuOffloadedReserved;
    register_region(fd, kObserverRegionId, base, kTotalBytes, flags, 777, label);
    PhaseNQueryRegionBlocksParams query =
        query_region_blocks(fd, kObserverRegionId, label);
    unregister_region(fd, kObserverRegionId, label);
    return query;
}

static ExpertOffloadReleaseParams release_region(int fd,
                                                 const PhaseFUuid &gpu_uuid,
                                                 unsigned long long region_id,
                                                 const char *label)
{
    ExpertOffloadReleaseParams params{};
    params.region_id = region_id;
    params.gpu_uuid = gpu_uuid;
    params.flags = kReleaseRequireExpert | kReleaseSubchunk64K;

    int ret = ioctl(fd, kExpertOffloadReleaseIoctl, &params);
    std::printf("expert_offload_release label=%s region_id=%llu ret=%d rmStatus=%u "
                "base=0x%llx length=%llu blocks=%llu attempted=%u success=%u "
                "failure=%u skipped=%u attempted_bytes=%llu released_bytes=%llu "
                "failed_bytes=%llu first_error=%u first_failed=0x%llx\n",
                label,
                region_id,
                ret,
                params.rmStatus,
                params.base,
                params.length,
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
    if (ret != 0 || params.rmStatus != 0)
        std::exit(1);
    return params;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);

    Args args{};
    if (!parse_args(argc, argv, &args))
        return 2;

    CUDA_CHECK(cudaSetDevice(args.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
    PhaseFUuid gpu_uuid = gpu_uuid_for_device(args.device);

    std::printf("pmm_subchunk_middle_release_probe_begin device=%d gpu=%s mode=%s "
                "total_bytes=%zu release_offset=%zu release_bytes=%zu "
                "envelope_bytes=%zu envelope_count=%u\n",
                args.device,
                prop.name,
                args.envelope ? "envelope" : "legacy",
                kTotalBytes,
                kReleaseOffset,
                kReleaseBytes,
                kEnvelopeBytes,
                kEnvelopeCount);
    print_mem("start");

    unsigned int *allocation = nullptr;
    CUDA_CHECK(cudaMallocManaged(&allocation, kTotalBytes));
    uintptr_t base = reinterpret_cast<uintptr_t>(allocation);
    std::printf("managed_alloc base=0x%llx bytes=%zu align_512k=%u align_2m=%u\n",
                static_cast<unsigned long long>(base),
                kTotalBytes,
                (base % kEnvelopeBytes) == 0 ? 1u : 0u,
                (base % (2ull * kMiB)) == 0 ? 1u : 0u);
    if ((base % kEnvelopeBytes) != 0) {
        std::fprintf(stderr,
                     "managed allocation is not 512KiB aligned; cannot run "
                     "SUBCHUNK_64K envelope release safely\n");
        cudaFree(allocation);
        return 2;
    }

    fill_pattern(allocation, kTotalBytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemPrefetchAsync(allocation, kTotalBytes, args.device));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("after_gpu_prefetch");
    unsigned char *allocation_bytes =
        reinterpret_cast<unsigned char *>(allocation);

    const size_t left_offset = 0;
    const size_t left_bytes = 2ull * kMiB;
    const size_t target_offset = kReleaseOffset;
    const size_t target_bytes = kReleaseBytes;
    const size_t tail_offset = kReleaseOffset + kReleaseBytes;
    const size_t tail_bytes = 4ull * kMiB - tail_offset;
    const size_t right_offset = 4ull * kMiB;
    const size_t right_bytes = 2ull * kMiB;

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::perror("open /dev/nvidia-uvm");
        cudaFree(allocation);
        return 1;
    }

    PhaseNQueryRegionBlocksParams before =
        observer_query(fd, base, "observer_before_release");
    PhaseFRangeResult before_left =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + left_offset,
                        left_bytes,
                        "before_left_0_2m");
    PhaseFRangeResult before_target =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + target_offset,
                        target_bytes,
                        "before_target_2_3p5m");
    PhaseFRangeResult before_tail =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + tail_offset,
                        tail_bytes,
                        "before_tail_3p5_4m");
    PhaseFRangeResult before_right =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + right_offset,
                        right_bytes,
                        "before_right_4_6m");

    const unsigned int base_flags = kRegionFlagPrefetchable |
                                    kRegionFlagGpuReadOnly |
                                    kRegionFlagStreamingSubpool |
                                    kRegionFlagCpuOffloadedReserved;
    const unsigned int release_flags =
        base_flags | (args.envelope ? kRegionFlagExpertEnvelope : 0u);

    std::vector<unsigned long long> region_ids;
    for (unsigned int i = 0; i < kEnvelopeCount; ++i) {
        unsigned long long region_id = kEnvelopeRegionBase + i;
        uintptr_t region_base = base + kReleaseOffset + i * kEnvelopeBytes;
        char label[64];
        std::snprintf(label, sizeof(label), "middle_%u_register", i);
        register_region(fd,
                        region_id,
                        region_base,
                        kEnvelopeBytes,
                        release_flags,
                        i,
                        label);
        region_ids.push_back(region_id);
    }

    unsigned long long total_released = 0;
    unsigned long long total_attempted = 0;
    bool release_ok = true;
    print_mem("before_middle_release");
    for (unsigned int i = 0; i < region_ids.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "middle_%u_release", i);
        ExpertOffloadReleaseParams rel =
            release_region(fd, gpu_uuid, region_ids[i], label);
        total_released += rel.released_bytes;
        total_attempted += rel.attempted_bytes;
        release_ok = release_ok &&
                     rel.rmStatus == 0 &&
                     rel.failure_blocks == 0 &&
                     rel.released_bytes == kEnvelopeBytes;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("after_middle_release");

    PhaseFRangeResult after_left =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + left_offset,
                        left_bytes,
                        "after_release_left_0_2m");
    PhaseFRangeResult after_target =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + target_offset,
                        target_bytes,
                        "after_release_target_2_3p5m");
    PhaseFRangeResult after_tail =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + tail_offset,
                        tail_bytes,
                        "after_release_tail_3p5_4m");
    PhaseFRangeResult after_right =
        query_residency(fd,
                        gpu_uuid,
                        allocation_bytes + right_offset,
                        right_bytes,
                        "after_release_right_4_6m");

    for (unsigned int i = 0; i < region_ids.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "middle_%u_unregister", i);
        unregister_region(fd, region_ids[i], label);
    }

    PhaseNQueryRegionBlocksParams after =
        observer_query(fd, base, "observer_after_release");

    PhaseNQueryRegionBlocksParams retouch{};
    PhaseFRangeResult restored_target{};
    bool restore_ok = !args.retouch;
    bool restored_neighbors_ok = !args.retouch;
    bool gpu_verify_ok = !args.retouch;
    if (args.retouch) {
        CUDA_CHECK(cudaMemPrefetchAsync(allocation_bytes + target_offset,
                                        target_bytes,
                                        args.device));
        CUDA_CHECK(cudaDeviceSynchronize());
        print_mem("after_target_restore_prefetch");
        PhaseFRangeResult restored_left =
            query_residency(fd,
                            gpu_uuid,
                            allocation_bytes + left_offset,
                            left_bytes,
                            "after_restore_left_0_2m");
        restored_target =
            query_residency(fd,
                            gpu_uuid,
                            allocation_bytes + target_offset,
                            target_bytes,
                            "after_restore_target_2_3p5m");
        PhaseFRangeResult restored_tail =
            query_residency(fd,
                            gpu_uuid,
                            allocation_bytes + tail_offset,
                            tail_bytes,
                            "after_restore_tail_3p5_4m");
        PhaseFRangeResult restored_right =
            query_residency(fd,
                            gpu_uuid,
                            allocation_bytes + right_offset,
                            right_bytes,
                            "after_restore_right_4_6m");
        restore_ok = restored_target.resident_gpu_bytes == target_bytes;
        restored_neighbors_ok =
            restored_left.resident_gpu_bytes == left_bytes &&
            restored_tail.resident_gpu_bytes == tail_bytes &&
            restored_right.resident_gpu_bytes == right_bytes;
        gpu_verify_ok = gpu_verify_all(allocation, kTotalBytes, "after_restore");
        retouch = observer_query(fd, base, "observer_after_retouch");
    }

    const bool before_residency_ok =
        before_left.resident_gpu_bytes == left_bytes &&
        before_target.resident_gpu_bytes == target_bytes &&
        before_tail.resident_gpu_bytes == tail_bytes &&
        before_right.resident_gpu_bytes == right_bytes;
    const bool target_offloaded =
        after_target.resident_gpu_bytes == 0 &&
        after_target.resident_cpu_bytes == target_bytes;
    const bool neighbors_unaffected =
        after_left.resident_gpu_bytes == left_bytes &&
        after_tail.resident_gpu_bytes == tail_bytes &&
        after_right.resident_gpu_bytes == right_bytes;
    const bool va_range_stable = before.va_range_count == after.va_range_count;
    const bool populated_stable =
        before.populated_block_count == after.populated_block_count;
    const bool logical_stable =
        before.logical_block_count == after.logical_block_count;
    const bool sampled_first_stable =
        before.first_block_start == after.first_block_start &&
        before.last_block_end == after.last_block_end;
    const bool retouch_stable =
        !args.retouch ||
        (after.va_range_count == retouch.va_range_count &&
         after.populated_block_count == retouch.populated_block_count &&
         after.logical_block_count == retouch.logical_block_count);
    const bool object_stable = va_range_stable &&
                               populated_stable &&
                               logical_stable &&
                               sampled_first_stable &&
                               retouch_stable;

    std::printf("pmm_subchunk_middle_release_probe_final mode=%s "
                "release_ok=%u total_attempted=%llu total_released=%llu "
                "expected_released=%zu before_residency_ok=%u "
                "target_offloaded=%u neighbors_unaffected=%u restore_ok=%u "
                "restored_neighbors_ok=%u gpu_verify_ok=%u "
                "before_va_ranges=%u after_va_ranges=%u "
                "before_populated=%llu after_populated=%llu "
                "before_logical=%llu after_logical=%llu object_stable=%u "
                "retouch_stable=%u\n",
                args.envelope ? "envelope" : "legacy",
                release_ok ? 1u : 0u,
                total_attempted,
                total_released,
                kReleaseBytes,
                before_residency_ok ? 1u : 0u,
                target_offloaded ? 1u : 0u,
                neighbors_unaffected ? 1u : 0u,
                restore_ok ? 1u : 0u,
                restored_neighbors_ok ? 1u : 0u,
                gpu_verify_ok ? 1u : 0u,
                before.va_range_count,
                after.va_range_count,
                before.populated_block_count,
                after.populated_block_count,
                before.logical_block_count,
                after.logical_block_count,
                object_stable ? 1u : 0u,
                retouch_stable ? 1u : 0u);

    close(fd);
    CUDA_CHECK(cudaFree(allocation));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_mem("after_cleanup");

    if (!release_ok ||
        !before_residency_ok ||
        !target_offloaded ||
        !neighbors_unaffected ||
        !restore_ok ||
        !restored_neighbors_ok ||
        !gpu_verify_ok) {
        return 1;
    }
    if (args.envelope && !object_stable)
        return 1;
    return 0;
}
