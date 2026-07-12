#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <unistd.h>

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
static constexpr unsigned int kUvmTestDrainReplayableFaultsIoctl = 274;
static constexpr unsigned int kPhaseFBatchExecutorIoctl = 81;
static constexpr unsigned int kPhaseFRegisterRegionIoctl = 82;
static constexpr unsigned int kPhaseFUnregisterRegionIoctl = 83;
static constexpr unsigned int kExpertReservedPoolIoctl = 98;
static constexpr unsigned int kExpertReservedPrefetchIoctl = 99;

static constexpr unsigned int kMaxRanges = 64;
static constexpr unsigned int kMaxReservedEntries = 128;
static constexpr unsigned int kQueryResidencyAction = 4;
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

struct DrainReplayableFaultsParams {
    PhaseFUuid gpu_uuid = {};
    unsigned long long timeout_ns = 0;
    unsigned int rmStatus = 0;
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
    ExpertReservedPrefetchEntry entries[kMaxReservedEntries] = {};
    unsigned int rmStatus = 0;
};

struct Residency {
    unsigned int status = 0;
    bool on_cpu = false;
    bool on_gpu = false;
    unsigned long long queried = 0;
    unsigned long long gpu_bytes = 0;
    unsigned long long cpu_bytes = 0;
    unsigned long long any_bytes = 0;
    unsigned long long unknown_bytes = 0;
};

struct Args {
    int device = 0;
    size_t bytes = 4ull * 1024 * 1024;
    size_t slot_bytes = 4ull * 1024 * 1024;
    unsigned int pool_id = 0;
    unsigned int slot_id = 0;
    unsigned long long region_id = 0xFACE00000001ull;
    const char *mode = "both";
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
        if (end == entry->d_name || *end != '\0' || fd_long < 0 || fd_long > INT_MAX)
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

static Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc)
            args.bytes = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--slot-bytes") == 0 && i + 1 < argc)
            args.slot_bytes = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--pool-id") == 0 && i + 1 < argc)
            args.pool_id = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--slot-id") == 0 && i + 1 < argc)
            args.slot_id = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--region-id") == 0 && i + 1 < argc)
            args.region_id = std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            args.mode = argv[++i];
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--bytes N] [--slot-bytes N] "
                         "[--pool-id N] [--slot-id N] [--region-id N] "
                         "[--mode native|reserved|both]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    if (args.bytes == 0 || args.slot_bytes < args.bytes ||
        args.bytes % 4096 != 0 || args.slot_bytes % 4096 != 0 ||
        args.region_id == 0 ||
        (std::strcmp(args.mode, "native") != 0 &&
         std::strcmp(args.mode, "reserved") != 0 &&
         std::strcmp(args.mode, "both") != 0)) {
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

static PhaseFUuid get_gpu_uuid(int device)
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    PhaseFUuid uuid{};
    std::memcpy(uuid.uuid, prop.uuid.bytes, sizeof(uuid.uuid));
    return uuid;
}

static Residency query_residency(int fd, const PhaseFUuid &gpu_uuid, const void *ptr, size_t bytes)
{
    PhaseFBatchExecutorParams params{};
    params.action = kQueryResidencyAction;
    params.count = 1;
    params.gpu_uuid = gpu_uuid;
    params.ranges[0].base = reinterpret_cast<unsigned long long>(ptr);
    params.ranges[0].length = bytes;
    IOCTL_CHECK(fd, kPhaseFBatchExecutorIoctl, params, "query_residency");
    Residency result{};
    result.status = params.results[0].status;
    result.on_cpu = params.results[0].resident_on_cpu != 0;
    result.on_gpu = params.results[0].resident_on_gpu != 0;
    result.queried = params.results[0].queried_bytes;
    result.gpu_bytes = params.results[0].resident_gpu_bytes;
    result.cpu_bytes = params.results[0].resident_cpu_bytes;
    result.any_bytes = params.results[0].resident_any_bytes;
    result.unknown_bytes = params.results[0].resident_unknown_bytes;
    return result;
}

static void print_residency(const char *label, const Residency &r)
{
    std::printf("residency %-28s status=%u on_cpu=%u on_gpu=%u queried=%llu "
                "gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu unknown_bytes=%llu\n",
                label,
                r.status,
                r.on_cpu ? 1 : 0,
                r.on_gpu ? 1 : 0,
                r.queried,
                r.gpu_bytes,
                r.cpu_bytes,
                r.any_bytes,
                r.unknown_bytes);
}

static void reset_pool(int fd, const PhaseFUuid &gpu_uuid)
{
    ExpertReservedPoolParams reset{};
    reset.mode = kReservedPoolModeReset;
    reset.gpu_uuid = gpu_uuid;
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, reset, "reserved_pool_reset");
}

static void prealloc_slot(int fd, const PhaseFUuid &gpu_uuid, const Args &args)
{
    ExpertReservedPoolParams pool{};
    pool.mode = kReservedPoolModePrealloc;
    pool.count = 1;
    pool.gpu_uuid = gpu_uuid;
    pool.entries[0].pool_id = args.pool_id;
    pool.entries[0].slot_id = args.slot_id;
    pool.entries[0].slot_bytes = args.slot_bytes;
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, pool, "reserved_pool_prealloc");
    std::printf("reserved_pool_prealloc success=%u failure=%u slot_bytes=%llu physical_bytes=%llu backing=%llu\n",
                pool.success_count,
                pool.failure_count,
                pool.entries[0].slot_bytes,
                pool.entries[0].physical_bytes,
                pool.reserved_backing_bytes);
}

static void register_region(int fd, const Args &args, const void *ptr)
{
    PhaseFRegisterRegionParams reg{};
    reg.region.region_id = args.region_id;
    reg.region.base = reinterpret_cast<unsigned long long>(ptr);
    reg.region.length = args.bytes;
    reg.region.pool = kPoolWeight;
    reg.region.layer_id = 0;
    reg.region.expert_id = 0;
    reg.region.role = kRoleGateUp;
    reg.region.flags = kRegionFlagPrefetchable |
                       kRegionFlagGpuReadOnly |
                       kRegionFlagStreamingSubpool |
                       kRegionFlagCpuOffloadedReserved;
    IOCTL_CHECK(fd, kPhaseFRegisterRegionIoctl, reg, "register_region");
}

static void unregister_region(int fd, unsigned long long region_id)
{
    PhaseFUnregisterRegionParams unreg{};
    unreg.region_id = region_id;
    (void)ioctl(fd, kPhaseFUnregisterRegionIoctl, &unreg);
}

static ExpertReservedPrefetchParams reserved_prefetch(int fd, const PhaseFUuid &gpu_uuid, const Args &args)
{
    ExpertReservedPrefetchParams prefetch{};
    prefetch.mode = kReservedPrefetchModeReservedPrefetch;
    prefetch.count = 1;
    prefetch.gpu_uuid = gpu_uuid;
    prefetch.async = 1;
    prefetch.entries[0].region_id = args.region_id;
    prefetch.entries[0].pool_id = args.pool_id;
    prefetch.entries[0].slot_id = args.slot_id;
    prefetch.entries[0].flags = kStrictFootprint;
    prefetch.entries[0].bytes_override = args.bytes;
    IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, prefetch, "reserved_prefetch");
    std::printf("reserved_prefetch success=%u failure=%u accepted=%llu assigned=%llu "
                "ordinary=%llu external=%llu broken=%llu entry_status=%u "
                "resolved_base=0x%llx resolved_bytes=%llu physical=%llu\n",
                prefetch.success_count,
                prefetch.failure_count,
                prefetch.accepted_bytes,
                prefetch.assigned_slots,
                prefetch.ordinary_resident_bytes,
                prefetch.pool_external_alloc_count,
                prefetch.footprint_contract_broken,
                prefetch.entries[0].status,
                prefetch.entries[0].resolved_base,
                prefetch.entries[0].resolved_bytes,
                prefetch.entries[0].physical_bytes);
    return prefetch;
}

static void invalidate_slot(int fd, const PhaseFUuid &gpu_uuid, const Args &args)
{
    ExpertReservedPrefetchParams invalidate{};
    invalidate.mode = kReservedPrefetchModeInvalidate;
    invalidate.count = 1;
    invalidate.gpu_uuid = gpu_uuid;
    invalidate.entries[0].pool_id = args.pool_id;
    invalidate.entries[0].slot_id = args.slot_id;
    invalidate.entries[0].flags = kStrictFootprint;
    IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, invalidate, "reserved_prefetch_invalidate");
    std::printf("reserved_prefetch_invalidate success=%u failure=%u invalidated=%llu status=%u\n",
                invalidate.success_count,
                invalidate.failure_count,
                invalidate.invalidated_slots,
                invalidate.entries[0].status);
}

__global__ void read_one_word_per_page_kernel(const unsigned int *data,
                                              size_t bytes,
                                              unsigned int seed,
                                              unsigned int *errors,
                                              unsigned int *sink)
{
    size_t page = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pages = bytes / 4096;
    if (page >= pages)
        return;
    size_t index = page * (4096 / sizeof(unsigned int));
    unsigned int value = data[index];
    unsigned int expected = seed ^ (unsigned int)index;
    if (value != expected)
        atomicAdd(errors, 1u);
    atomicXor(sink, value);
}

static void launch_page_read(const char *label,
                             const unsigned int *ptr,
                             size_t bytes,
                             int tag,
                             unsigned int *errors,
                             unsigned int *sink)
{
    size_t pages = bytes / 4096;
    int block = 256;
    int grid = (int)((pages + block - 1) / block);
    unsigned int host_errors = 0;
    unsigned int host_sink = 0;
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(unsigned int)));
    read_one_word_per_page_kernel<<<grid, block>>>(ptr, bytes, seed_for(tag), errors, sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_sink, sink, sizeof(host_sink), cudaMemcpyDeviceToHost));
    if (host_errors != 0) {
        std::fprintf(stderr, "%s failed errors=%u\n", label, host_errors);
        std::exit(3);
    }
    std::printf("%s ok pages=%zu sink=0x%x\n", label, pages, host_sink);
}

static void prefetch_to_cpu(const char *label, void *ptr, size_t bytes)
{
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%s cpu_prefetch_done bytes=%zu\n", label, bytes);
}

static void drain_replayable_faults(int fd, const PhaseFUuid &gpu_uuid, const char *label)
{
    DrainReplayableFaultsParams params{};
    params.gpu_uuid = gpu_uuid;
    params.timeout_ns = 5ull * 1000ull * 1000ull * 1000ull;
    if (ioctl(fd, kUvmTestDrainReplayableFaultsIoctl, &params) != 0) {
        std::printf("%s drain_replayable_faults_unavailable errno=%d\n", label, errno);
        return;
    }
    if (params.rmStatus != 0) {
        std::printf("%s drain_replayable_faults_status=%u\n", label, params.rmStatus);
        return;
    }
    std::printf("%s drain_replayable_faults_done\n", label);
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
    PhaseFUuid gpu_uuid = get_gpu_uuid(args.device);

    unsigned int *native = nullptr;
    unsigned int *reserved = nullptr;
    unsigned int *errors = nullptr;
    unsigned int *sink = nullptr;
    CUDA_CHECK(cudaMallocManaged(&native, args.bytes, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&reserved, args.bytes, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMalloc(&errors, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned int)));

    fill_pattern(native, args.bytes / sizeof(unsigned int), 10);
    fill_pattern(reserved, args.bytes / sizeof(unsigned int), 20);
    CUDA_CHECK(cudaDeviceSynchronize());

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr, "failed to open or duplicate initialized UVM fd\n");
        return 1;
    }

    std::printf("probe_begin gpu=%s device=%d mode=%s bytes=%zu slot_bytes=%zu pool=%u slot=%u region_id=%llu\n",
                prop.name,
                args.device,
                args.mode,
                args.bytes,
                args.slot_bytes,
                args.pool_id,
                args.slot_id,
                args.region_id);
    std::printf("native_ptr=0x%llx reserved_ptr=0x%llx\n",
                reinterpret_cast<unsigned long long>(native),
                reinterpret_cast<unsigned long long>(reserved));

    bool native_migrated = true;
    bool reserved_ordinary_unchanged = true;

    if (std::strcmp(args.mode, "reserved") != 0) {
        prefetch_to_cpu("native_control", native, args.bytes);
        Residency native_before = query_residency(fd, gpu_uuid, native, args.bytes);
        print_residency("native_before_gpu_read", native_before);
        launch_page_read("native_gpu_read", native, args.bytes, 10, errors, sink);
        drain_replayable_faults(fd, gpu_uuid, "native_gpu_read");
        Residency native_after = query_residency(fd, gpu_uuid, native, args.bytes);
        print_residency("native_after_gpu_read", native_after);
        native_migrated =
            native_after.gpu_bytes > native_before.gpu_bytes &&
            native_after.gpu_bytes > 0;
    }

    if (std::strcmp(args.mode, "native") != 0) {
        reset_pool(fd, gpu_uuid);
        prealloc_slot(fd, gpu_uuid, args);
        register_region(fd, args, reserved);

        prefetch_to_cpu("reserved_path", reserved, args.bytes);
        Residency reserved_before = query_residency(fd, gpu_uuid, reserved, args.bytes);
        print_residency("reserved_before_prefetch", reserved_before);
        ExpertReservedPrefetchParams prefetch = reserved_prefetch(fd, gpu_uuid, args);
        CUDA_CHECK(cudaDeviceSynchronize());
        Residency reserved_after_prefetch = query_residency(fd, gpu_uuid, reserved, args.bytes);
        print_residency("reserved_after_prefetch", reserved_after_prefetch);
        launch_page_read("reserved_gpu_read", reserved, args.bytes, 20, errors, sink);
        drain_replayable_faults(fd, gpu_uuid, "reserved_gpu_read");
        Residency reserved_after_read = query_residency(fd, gpu_uuid, reserved, args.bytes);
        print_residency("reserved_after_gpu_read", reserved_after_read);

        reserved_ordinary_unchanged =
            prefetch.success_count == 1 &&
            prefetch.failure_count == 0 &&
            prefetch.accepted_bytes == args.bytes &&
            reserved_after_read.gpu_bytes == reserved_after_prefetch.gpu_bytes &&
            reserved_after_read.cpu_bytes == reserved_after_prefetch.cpu_bytes &&
            reserved_after_read.gpu_bytes == 0 &&
            reserved_after_read.cpu_bytes >= args.bytes;

        invalidate_slot(fd, gpu_uuid, args);
        unregister_region(fd, args.region_id);
        reset_pool(fd, gpu_uuid);
    }

    std::printf("verdict native_control_migrated=%u reserved_ordinary_residency_unchanged=%u\n",
                native_migrated ? 1 : 0,
                reserved_ordinary_unchanged ? 1 : 0);

    close(fd);

    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(errors));
    CUDA_CHECK(cudaFree(reserved));
    CUDA_CHECK(cudaFree(native));

    return (native_migrated && reserved_ordinary_unchanged) ? 0 : 4;
}
