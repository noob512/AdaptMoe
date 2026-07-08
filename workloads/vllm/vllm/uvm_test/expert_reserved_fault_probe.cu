#include <algorithm>
#include <chrono>
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

#define IOCTL_CHECK(fd, request, params, what)                                  \
    do {                                                                        \
        if (ioctl((fd), (request), &(params)) != 0 || (params).rmStatus != 0) {  \
            std::perror((what));                                                \
            std::fprintf(stderr, "%s failed rmStatus=%u\n", (what),             \
                         (unsigned int)(params).rmStatus);                      \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static constexpr unsigned int kUvmInitializeIoctl = 0x30000001;
static constexpr unsigned int kUvmPhaseFRegisterRegionIoctl = 82;
static constexpr unsigned int kUvmPhaseFUnregisterRegionIoctl = 83;
static constexpr unsigned int kExpertReservedPoolIoctl = 98;
static constexpr unsigned int kExpertReservedPrefetchIoctl = 99;

static constexpr unsigned int kPoolWeight = 2;
static constexpr unsigned int kRoleGateUp = 3;
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

enum class CaseKind {
    AlignmentGuard,
    CleanupReaccess,
    UnrelatedWriteAfterCleanup,
    NoInvalidateUnrelatedWrite,
    BigPteSplitCleanup,
    ReadonlyWrite,
    CrossStreamCleanup,
};

enum class SyncMode {
    None,
    Current,
    Side,
    Device,
};

struct Args {
    CaseKind case_kind = CaseKind::AlignmentGuard;
    SyncMode sync_mode = SyncMode::Device;
    int device = 0;
    size_t bytes = 3ull * 1024 * 1024;
    size_t base_offset = 1ull * 1024 * 1024;
    size_t guard_bytes = 1ull * 1024 * 1024;
    size_t slot_bytes = 4ull * 1024 * 1024;
    int spin_iters = 200000;
    bool destructive = false;
};

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
        int dup_fd;

        if (end == entry->d_name || *end != '\0' || fd_long < 0 || fd_long > INT_MAX)
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

        dup_fd = fcntl((int)fd_long, F_DUPFD_CLOEXEC, 3);
        if (dup_fd < 0)
            continue;

        UvmInitializeParams params{};
        if (ioctl(dup_fd, kUvmInitializeIoctl, &params) == 0 && params.rmStatus == 0) {
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

static const char *case_name(CaseKind kind)
{
    switch (kind) {
        case CaseKind::AlignmentGuard: return "alignment_guard";
        case CaseKind::CleanupReaccess: return "cleanup_reaccess";
        case CaseKind::UnrelatedWriteAfterCleanup: return "unrelated_write_after_cleanup";
        case CaseKind::NoInvalidateUnrelatedWrite: return "no_invalidate_unrelated_write";
        case CaseKind::BigPteSplitCleanup: return "big_pte_split_cleanup";
        case CaseKind::ReadonlyWrite: return "readonly_write";
        case CaseKind::CrossStreamCleanup: return "cross_stream_cleanup";
    }
    return "unknown";
}

static CaseKind parse_case(const char *value)
{
    if (std::strcmp(value, "alignment_guard") == 0)
        return CaseKind::AlignmentGuard;
    if (std::strcmp(value, "cleanup_reaccess") == 0)
        return CaseKind::CleanupReaccess;
    if (std::strcmp(value, "unrelated_write_after_cleanup") == 0)
        return CaseKind::UnrelatedWriteAfterCleanup;
    if (std::strcmp(value, "no_invalidate_unrelated_write") == 0)
        return CaseKind::NoInvalidateUnrelatedWrite;
    if (std::strcmp(value, "big_pte_split_cleanup") == 0)
        return CaseKind::BigPteSplitCleanup;
    if (std::strcmp(value, "readonly_write") == 0)
        return CaseKind::ReadonlyWrite;
    if (std::strcmp(value, "cross_stream_cleanup") == 0)
        return CaseKind::CrossStreamCleanup;

    std::fprintf(stderr, "unknown --case %s\n", value);
    std::exit(2);
}

static SyncMode parse_sync_mode(const char *value)
{
    if (std::strcmp(value, "none") == 0)
        return SyncMode::None;
    if (std::strcmp(value, "current") == 0)
        return SyncMode::Current;
    if (std::strcmp(value, "side") == 0)
        return SyncMode::Side;
    if (std::strcmp(value, "device") == 0)
        return SyncMode::Device;

    std::fprintf(stderr, "unknown --sync-mode %s\n", value);
    std::exit(2);
}

static Args parse_args(int argc, char **argv)
{
    Args args;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc)
            args.case_kind = parse_case(argv[++i]);
        else if (std::strcmp(argv[i], "--sync-mode") == 0 && i + 1 < argc)
            args.sync_mode = parse_sync_mode(argv[++i]);
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc)
            args.bytes = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--base-offset") == 0 && i + 1 < argc)
            args.base_offset = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--guard-bytes") == 0 && i + 1 < argc)
            args.guard_bytes = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--slot-bytes") == 0 && i + 1 < argc)
            args.slot_bytes = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--spin-iters") == 0 && i + 1 < argc)
            args.spin_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--destructive") == 0)
            args.destructive = true;
        else {
            std::fprintf(stderr,
                         "usage: %s --case alignment_guard|cleanup_reaccess|"
                         "unrelated_write_after_cleanup|no_invalidate_unrelated_write|"
                         "big_pte_split_cleanup|readonly_write|cross_stream_cleanup "
                         "[--sync-mode none|current|side|device] [--device N] "
                         "[--bytes N] [--base-offset N] [--guard-bytes N] "
                         "[--slot-bytes N] [--spin-iters N] [--destructive]\n",
                         argv[0]);
            std::exit(2);
        }
    }

    if (args.bytes == 0 ||
        args.slot_bytes < args.bytes ||
        args.bytes % sizeof(unsigned int) != 0 ||
        args.base_offset % sizeof(unsigned int) != 0 ||
        args.guard_bytes % sizeof(unsigned int) != 0 ||
        args.slot_bytes % 4096 != 0 ||
        args.bytes % 4096 != 0 ||
        args.base_offset % 4096 != 0) {
        std::fprintf(stderr, "invalid arguments\n");
        std::exit(2);
    }

    return args;
}

static unsigned int seed_for(int tag)
{
    return 0x9e370000u ^ (0x01010101u * (unsigned int)(tag + 1));
}

static void fill_pattern(unsigned int *data, size_t words, int tag)
{
    unsigned int seed = seed_for(tag);
    for (size_t i = 0; i < words; ++i)
        data[i] = seed ^ (unsigned int)i;
}

static void print_addr(const char *label, const void *ptr)
{
    unsigned long long va = reinterpret_cast<unsigned long long>(ptr);
    std::printf("%s=0x%llx mod4K=0x%llx mod64K=0x%llx mod2M=0x%llx mod4M=0x%llx\n",
                label,
                va,
                va % (4ull * 1024),
                va % (64ull * 1024),
                va % (2ull * 1024 * 1024),
                va % (4ull * 1024 * 1024));
}

__global__ void check_pattern_kernel(const unsigned int *data,
                                     size_t words,
                                     unsigned int seed,
                                     unsigned int *errors)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;

    unsigned int expected = seed ^ (unsigned int)i;
    if (data[i] != expected)
        atomicAdd(errors, 1u);
}

__global__ void write_pattern_kernel(unsigned int *data,
                                     size_t words,
                                     unsigned int seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;

    data[i] = seed ^ (unsigned int)i;
}

__global__ void spin_read_kernel(const unsigned int *data,
                                 size_t words,
                                 unsigned int seed,
                                 unsigned int *sink,
                                 int iters)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;

    unsigned int acc = 0;
    for (int iter = 0; iter < iters; ++iter) {
        size_t index = (i + (size_t)iter * 131u) % words;
        acc ^= data[index] ^ seed ^ (unsigned int)iter;
    }

    if ((acc & 0x80000000u) != 0)
        atomicAdd(sink, 1u);
}

static void launch_check(const char *label,
                         const unsigned int *ptr,
                         size_t bytes,
                         int tag,
                         unsigned int *errors)
{
    size_t words = bytes / sizeof(unsigned int);
    int block = 256;
    int grid = (int)((words + block - 1) / block);

    *errors = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    check_pattern_kernel<<<grid, block>>>(ptr, words, seed_for(tag), errors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (*errors != 0) {
        std::fprintf(stderr, "%s failed errors=%u\n", label, *errors);
        std::exit(3);
    }
    std::printf("%s ok words=%zu\n", label, words);
}

static void launch_write(unsigned int *ptr, size_t bytes, int tag)
{
    size_t words = bytes / sizeof(unsigned int);
    int block = 256;
    int grid = (int)((words + block - 1) / block);

    write_pattern_kernel<<<grid, block>>>(ptr, words, seed_for(tag));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct ProbeState {
    int fd = -1;
    PhaseFUuid gpu_uuid = {};
    unsigned long long region_id = 0xABC2F00D0001ull;
};

static void reset_pool(int fd, const PhaseFUuid &gpu_uuid)
{
    ExpertReservedPoolParams reset{};
    reset.mode = kReservedPoolModeReset;
    reset.gpu_uuid = gpu_uuid;
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, reset, "reserved_pool_reset");
}

static void prealloc_slot(int fd, const PhaseFUuid &gpu_uuid, size_t slot_bytes)
{
    ExpertReservedPoolParams pool{};
    pool.mode = kReservedPoolModePrealloc;
    pool.count = 1;
    pool.gpu_uuid = gpu_uuid;
    pool.entries[0].pool_id = 0;
    pool.entries[0].slot_id = 0;
    pool.entries[0].slot_bytes = slot_bytes;
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, pool, "reserved_pool_prealloc");
    std::printf("prealloc success=%u failure=%u slot_bytes=%llu physical_bytes=%llu backing=%llu\n",
                pool.success_count,
                pool.failure_count,
                pool.entries[0].slot_bytes,
                pool.entries[0].physical_bytes,
                pool.reserved_backing_bytes);
}

static void register_region(int fd,
                            unsigned long long region_id,
                            unsigned int *region,
                            size_t bytes)
{
    PhaseFRegisterRegionParams reg{};
    reg.region.region_id = region_id;
    reg.region.base = reinterpret_cast<unsigned long long>(region);
    reg.region.length = bytes;
    reg.region.pool = kPoolWeight;
    reg.region.layer_id = 5;
    reg.region.expert_id = 1;
    reg.region.role = kRoleGateUp;
    reg.region.flags = kRegionFlagPrefetchable |
                       kRegionFlagGpuReadOnly |
                       kRegionFlagStreamingSubpool |
                       kRegionFlagCpuOffloadedReserved;
    IOCTL_CHECK(fd, kUvmPhaseFRegisterRegionIoctl, reg, "register_region");
}

static void unregister_region(int fd, unsigned long long region_id)
{
    PhaseFUnregisterRegionParams unreg{};
    unreg.region_id = region_id;
    (void)ioctl(fd, kUvmPhaseFUnregisterRegionIoctl, &unreg);
}

static void reserved_prefetch(int fd,
                              const PhaseFUuid &gpu_uuid,
                              unsigned long long region_id,
                              size_t bytes)
{
    ExpertReservedPrefetchParams prefetch{};
    prefetch.mode = kReservedPrefetchModeReservedPrefetch;
    prefetch.count = 1;
    prefetch.gpu_uuid = gpu_uuid;
    prefetch.async = 1;
    prefetch.entries[0].region_id = region_id;
    prefetch.entries[0].pool_id = 0;
    prefetch.entries[0].slot_id = 0;
    prefetch.entries[0].flags = kStrictFootprint;

    IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, prefetch, "reserved_prefetch");
    std::printf("reserved_prefetch success=%u failure=%u accepted=%llu assigned=%llu "
                "ordinary=%llu external=%llu broken=%llu status=%u physical=%llu\n",
                prefetch.success_count,
                prefetch.failure_count,
                prefetch.accepted_bytes,
                prefetch.assigned_slots,
                prefetch.ordinary_resident_bytes,
                prefetch.pool_external_alloc_count,
                prefetch.footprint_contract_broken,
                prefetch.entries[0].status,
                prefetch.entries[0].physical_bytes);

    if (prefetch.success_count != 1 ||
        prefetch.failure_count != 0 ||
        prefetch.accepted_bytes != bytes ||
        prefetch.assigned_slots != 1 ||
        prefetch.ordinary_resident_bytes != 0 ||
        prefetch.pool_external_alloc_count != 0 ||
        prefetch.footprint_contract_broken != 0 ||
        prefetch.entries[0].status != 0) {
        std::fprintf(stderr, "reserved_prefetch unexpected result\n");
        std::exit(3);
    }
}

static void invalidate_slot(int fd, const PhaseFUuid &gpu_uuid)
{
    ExpertReservedPrefetchParams invalidate{};
    invalidate.mode = kReservedPrefetchModeInvalidate;
    invalidate.count = 1;
    invalidate.gpu_uuid = gpu_uuid;
    invalidate.entries[0].pool_id = 0;
    invalidate.entries[0].slot_id = 0;
    invalidate.entries[0].flags = kStrictFootprint;
    IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, invalidate, "reserved_prefetch_invalidate");
    std::printf("invalidate success=%u failure=%u invalidated=%llu status=%u\n",
                invalidate.success_count,
                invalidate.failure_count,
                invalidate.invalidated_slots,
                invalidate.entries[0].status);
}

static void maybe_require_destructive(const Args &args)
{
    bool destructive_case =
        args.case_kind == CaseKind::ReadonlyWrite ||
        (args.case_kind == CaseKind::CrossStreamCleanup &&
         args.sync_mode != SyncMode::Device &&
         args.sync_mode != SyncMode::Side);

    if (destructive_case && !args.destructive) {
        std::fprintf(stderr,
                     "case=%s sync_mode=%d may intentionally trigger an MMU fault; "
                     "rerun with --destructive to execute it\n",
                     case_name(args.case_kind),
                     (int)args.sync_mode);
        std::exit(77);
    }
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    maybe_require_destructive(args);

    CUDA_CHECK(cudaSetDevice(args.device));
    CU_CHECK(cuInit(0));

    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, args.device));
    CUuuid cuda_uuid{};
    CU_CHECK(cuDeviceGetUuid(&cuda_uuid, cu_device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    PhaseFUuid gpu_uuid{};
    std::memcpy(gpu_uuid.uuid, cuda_uuid.bytes, sizeof(gpu_uuid.uuid));

    size_t allocation_bytes = args.base_offset + args.bytes + args.guard_bytes;
    size_t unrelated_bytes = 2ull * 1024 * 1024;
    unsigned int *raw = nullptr;
    unsigned int *unrelated = nullptr;
    unsigned int *errors = nullptr;

    CUDA_CHECK(cudaMallocManaged(&raw, allocation_bytes, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&unrelated, unrelated_bytes, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&errors, sizeof(unsigned int), cudaMemAttachGlobal));

    unsigned int *region = reinterpret_cast<unsigned int *>(
        reinterpret_cast<unsigned char *>(raw) + args.base_offset);
    unsigned int *after_guard = reinterpret_cast<unsigned int *>(
        reinterpret_cast<unsigned char *>(region) + args.bytes);

    size_t before_words = args.base_offset / sizeof(unsigned int);
    size_t region_words = args.bytes / sizeof(unsigned int);
    size_t after_words = args.guard_bytes / sizeof(unsigned int);

    if (before_words)
        fill_pattern(raw, before_words, 10);
    fill_pattern(region, region_words, 20);
    if (after_words)
        fill_pattern(after_guard, after_words, 30);
    fill_pattern(unrelated, unrelated_bytes / sizeof(unsigned int), 40);
    *errors = 0;
    CUDA_CHECK(cudaDeviceSynchronize());

    if (args.case_kind == CaseKind::BigPteSplitCleanup) {
        CUDA_CHECK(cudaMemPrefetchAsync(raw, allocation_bytes, args.device));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemPrefetchAsync(raw, allocation_bytes, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (before_words)
            fill_pattern(raw, before_words, 10);
        fill_pattern(region, region_words, 20);
        if (after_words)
            fill_pattern(after_guard, after_words, 30);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("primed native GPU PTEs then returned allocation to CPU\n");
    }

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr, "failed to open or duplicate initialized UVM fd\n");
        return 1;
    }

    reset_pool(fd, gpu_uuid);
    prealloc_slot(fd, gpu_uuid, args.slot_bytes);
    register_region(fd, 0xABC2F00D0001ull, region, args.bytes);

    std::printf("fault_probe_begin case=%s sync_mode=%d gpu=%s bytes=%zu "
                "base_offset=%zu guard_bytes=%zu slot_bytes=%zu allocation_bytes=%zu\n",
                case_name(args.case_kind),
                (int)args.sync_mode,
                prop.name,
                args.bytes,
                args.base_offset,
                args.guard_bytes,
                args.slot_bytes,
                allocation_bytes);
    print_addr("raw", raw);
    print_addr("region", region);
    print_addr("region_end", reinterpret_cast<unsigned char *>(region) + args.bytes);
    print_addr("unrelated", unrelated);

    reserved_prefetch(fd, gpu_uuid, 0xABC2F00D0001ull, args.bytes);
    launch_check("reserved_region_read_check", region, args.bytes, 20, errors);

    if (args.case_kind == CaseKind::ReadonlyWrite) {
        std::printf("readonly_write launching write into readonly slot-backed region; "
                    "an illegal access or Xid is an expected reproduction\n");
        launch_write(region, args.bytes, 21);
        std::printf("readonly_write unexpectedly completed without CUDA error\n");
    }
    else if (args.case_kind == CaseKind::CrossStreamCleanup) {
        cudaStream_t side_stream;
        CUDA_CHECK(cudaStreamCreate(&side_stream));

        size_t words = args.bytes / sizeof(unsigned int);
        int block = 256;
        int grid = (int)std::min<size_t>((words + block - 1) / block, 4096);
        spin_read_kernel<<<grid, block, 0, side_stream>>>(region,
                                                          words,
                                                          seed_for(20),
                                                          errors,
                                                          args.spin_iters);
        CUDA_CHECK(cudaGetLastError());

        if (args.sync_mode == SyncMode::Current)
            CUDA_CHECK(cudaStreamSynchronize(0));
        else if (args.sync_mode == SyncMode::Side)
            CUDA_CHECK(cudaStreamSynchronize(side_stream));
        else if (args.sync_mode == SyncMode::Device)
            CUDA_CHECK(cudaDeviceSynchronize());

        invalidate_slot(fd, gpu_uuid);
        CUDA_CHECK(cudaStreamSynchronize(side_stream));
        CUDA_CHECK(cudaStreamDestroy(side_stream));
    }
    else {
        bool invalidate = args.case_kind != CaseKind::NoInvalidateUnrelatedWrite;
        if (invalidate)
            invalidate_slot(fd, gpu_uuid);
        else
            std::printf("skipping invalidate by request\n");

        if (args.case_kind == CaseKind::CleanupReaccess) {
            launch_check("same_region_after_cleanup_read_check", region, args.bytes, 20, errors);
        }
        else if (args.case_kind == CaseKind::AlignmentGuard ||
                 args.case_kind == CaseKind::BigPteSplitCleanup) {
            if (before_words)
                launch_check("before_guard_check", raw, args.base_offset, 10, errors);
            if (after_words)
                launch_check("after_guard_check", after_guard, args.guard_bytes, 30, errors);
            launch_write(raw, args.base_offset, 11);
            launch_write(after_guard, args.guard_bytes, 31);
            if (before_words)
                launch_check("before_guard_write_check", raw, args.base_offset, 11, errors);
            if (after_words)
                launch_check("after_guard_write_check", after_guard, args.guard_bytes, 31, errors);
        }
        else if (args.case_kind == CaseKind::UnrelatedWriteAfterCleanup ||
                 args.case_kind == CaseKind::NoInvalidateUnrelatedWrite) {
            launch_write(unrelated, unrelated_bytes, 41);
            launch_check("unrelated_write_check", unrelated, unrelated_bytes, 41, errors);
        }
    }

    unregister_region(fd, 0xABC2F00D0001ull);
    reset_pool(fd, gpu_uuid);
    close(fd);

    CUDA_CHECK(cudaFree(errors));
    CUDA_CHECK(cudaFree(unrelated));
    CUDA_CHECK(cudaFree(raw));

    std::printf("CASE_RESULT case=%s status=pass\n", case_name(args.case_kind));
    return 0;
}
