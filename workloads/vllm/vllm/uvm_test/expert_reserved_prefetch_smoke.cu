#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <numeric>
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
static constexpr unsigned int kUvmMigrateIoctl = 51;
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

struct UvmMigrateParams {
    unsigned long long base = 0;
    unsigned long long length = 0;
    PhaseFUuid destinationUuid = {};
    unsigned int flags = 0;
    unsigned long long semaphoreAddress = 0;
    unsigned int semaphorePayload = 0;
    int cpuNumaNode = -1;
    unsigned long long userSpaceStart = 0;
    unsigned long long userSpaceLength = 0;
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

struct Args {
    int device = 0;
    int experts = 1;
    int repeats = 1;
    size_t bytes = 2ull * 1024 * 1024;
    bool native_baseline = false;
    bool strict_migrate_guard_test = false;
};

static Args parse_args(int argc, char **argv)
{
    Args args;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--experts") == 0 && i + 1 < argc)
            args.experts = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            args.repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc)
            args.bytes = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--native-baseline") == 0)
            args.native_baseline = true;
        else if (std::strcmp(argv[i], "--strict-migrate-guard-test") == 0)
            args.strict_migrate_guard_test = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--experts N] [--repeats N] "
                         "[--bytes N] [--native-baseline] [--strict-migrate-guard-test]\n",
                         argv[0]);
            std::exit(2);
        }
    }

    if (args.experts <= 0 || args.repeats <= 0 || args.bytes == 0 || args.bytes % sizeof(unsigned int) != 0) {
        std::fprintf(stderr, "invalid arguments\n");
        std::exit(2);
    }

    return args;
}

static unsigned int expert_seed(int expert)
{
    return 0x5a170000u ^ (0x01010101u * (unsigned int)(expert + 1));
}

static void fill_pattern(unsigned int *data, size_t words, int expert)
{
    unsigned int seed = expert_seed(expert);
    for (size_t i = 0; i < words; ++i)
        data[i] = seed ^ (unsigned int)i;
}

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void print_latency_summary(const char *label, const std::vector<double> &samples, size_t bytes)
{
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double avg = sum / (double)samples.size();
    double min_v = sorted.front();
    double max_v = sorted.back();
    double p50 = sorted[sorted.size() / 2];
    double p95 = sorted[(size_t)((sorted.size() - 1) * 0.95)];
    double gib = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    double avg_bw = gib / (avg / 1000.0);

    std::printf("%s samples=%zu bytes=%zu avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f "
                "min_ms=%.3f max_ms=%.3f avg_GiBps=%.3f\n",
                label,
                samples.size(),
                bytes,
                avg,
                p50,
                p95,
                min_v,
                max_v,
                avg_bw);
}

__global__ void check_pattern_kernel(const unsigned int *data, size_t words, unsigned int seed, unsigned int *errors)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;

    unsigned int expected = seed ^ (unsigned int)i;
    if (data[i] != expected)
        atomicAdd(errors, 1u);
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    const int device = args.device;
    const size_t bytes = args.bytes;
    const size_t words = bytes / sizeof(unsigned int);
    const unsigned long long region_base_id = 0xABC200000001ull;
    std::vector<unsigned int *> experts(args.experts, nullptr);
    std::vector<unsigned long long> region_ids(args.experts, 0);
    std::vector<double> reserved_ms;
    std::vector<double> native_ms;
    unsigned int total_errors = 0;

    CUDA_CHECK(cudaSetDevice(device));
    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUuuid cuda_uuid{};
    CU_CHECK(cuDeviceGetUuid(&cuda_uuid, cu_device));

    for (int expert = 0; expert < args.experts; ++expert) {
        CUDA_CHECK(cudaMallocManaged(&experts[expert], bytes));
        fill_pattern(experts[expert], words, expert);
        region_ids[expert] = region_base_id + (unsigned long long)expert;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (args.native_baseline) {
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            for (int expert = 0; expert < args.experts; ++expert) {
                CUDA_CHECK(cudaMemPrefetchAsync(experts[expert], bytes, cudaCpuDeviceId));
                CUDA_CHECK(cudaDeviceSynchronize());
                fill_pattern(experts[expert], words, expert);
                CUDA_CHECK(cudaDeviceSynchronize());

                double start = now_ms();
                CUDA_CHECK(cudaMemPrefetchAsync(experts[expert], bytes, device));
                CUDA_CHECK(cudaDeviceSynchronize());
                native_ms.push_back(now_ms() - start);
            }
        }

        print_latency_summary("native_prefetch_baseline", native_ms, bytes);

        for (int expert = 0; expert < args.experts; ++expert) {
            CUDA_CHECK(cudaMemPrefetchAsync(experts[expert], bytes, cudaCpuDeviceId));
            CUDA_CHECK(cudaDeviceSynchronize());
            fill_pattern(experts[expert], words, expert);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr, "failed to open or duplicate initialized UVM fd\n");
        return 1;
    }

    PhaseFUuid gpu_uuid{};
    std::memcpy(gpu_uuid.uuid, cuda_uuid.bytes, sizeof(gpu_uuid.uuid));

    ExpertReservedPoolParams reset_pool{};
    reset_pool.mode = kReservedPoolModeReset;
    reset_pool.gpu_uuid = gpu_uuid;
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, reset_pool, "reserved_pool_reset");

    ExpertReservedPoolParams pool{};
    pool.mode = kReservedPoolModePrealloc;
    pool.count = 1;
    pool.gpu_uuid = gpu_uuid;
    pool.entries[0].pool_id = 0;
    pool.entries[0].slot_id = 0;
    pool.entries[0].slot_bytes = bytes;
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, pool, "reserved_pool_prealloc");

    for (int expert = 0; expert < args.experts; ++expert) {
        PhaseFRegisterRegionParams reg{};
        reg.region.region_id = region_ids[expert];
        reg.region.base = reinterpret_cast<unsigned long long>(experts[expert]);
        reg.region.length = bytes;
        reg.region.pool = kPoolWeight;
        reg.region.layer_id = 0;
        reg.region.expert_id = (unsigned int)expert;
        reg.region.role = kRoleW13;
        reg.region.flags = kRegionFlagPrefetchable | kRegionFlagGpuReadOnly | kRegionFlagStreamingSubpool;
        if (args.strict_migrate_guard_test)
            reg.region.flags |= kRegionFlagCpuOffloadedReserved;
        IOCTL_CHECK(fd, kUvmPhaseFRegisterRegionIoctl, reg, "register_region");
    }

    if (args.strict_migrate_guard_test) {
        UvmMigrateParams migrate{};
        migrate.base = reinterpret_cast<unsigned long long>(experts[0]);
        migrate.length = bytes;
        migrate.destinationUuid = gpu_uuid;
        migrate.cpuNumaNode = -1;

        int ret = ioctl(fd, kUvmMigrateIoctl, &migrate);
        bool rejected = ret != 0 || migrate.rmStatus != 0;
        std::printf("strict_migrate_guard ret=%d rmStatus=%u rejected=%u\n",
                    ret,
                    migrate.rmStatus,
                    rejected ? 1u : 0u);
        if (!rejected) {
            std::fprintf(stderr, "strict migrate guard failed: native UVM_MIGRATE was allowed\n");
            std::exit(3);
        }
    }

    std::printf("slot_reuse_begin experts=%d repeats=%d bytes=%zu slot_count=1 slot_bytes=%llu gpu=%s\n",
                args.experts,
                args.repeats,
                bytes,
                pool.entries[0].physical_bytes,
                prop.name);

    unsigned int *errors = nullptr;
    CUDA_CHECK(cudaMallocManaged(&errors, sizeof(unsigned int)));
    int block = 256;
    int grid = (int)((words + block - 1) / block);

    for (int repeat = 0; repeat < args.repeats; ++repeat) {
        for (int expert = 0; expert < args.experts; ++expert) {
            ExpertReservedPrefetchParams prefetch{};
            prefetch.mode = kReservedPrefetchModeReservedPrefetch;
            prefetch.count = 1;
            prefetch.gpu_uuid = gpu_uuid;
            prefetch.async = 1;
            prefetch.entries[0].region_id = region_ids[expert];
            prefetch.entries[0].pool_id = 0;
            prefetch.entries[0].slot_id = 0;
            prefetch.entries[0].flags = kStrictFootprint;

            double start = now_ms();
            IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, prefetch, "reserved_prefetch");
            reserved_ms.push_back(now_ms() - start);

            if (prefetch.success_count != 1 ||
                prefetch.failure_count != 0 ||
                prefetch.accepted_bytes != bytes ||
                prefetch.assigned_slots != 1 ||
                prefetch.ordinary_resident_bytes != 0 ||
                prefetch.pool_external_alloc_count != 0 ||
                prefetch.footprint_contract_broken != 0 ||
                prefetch.entries[0].status != 0) {
                std::fprintf(stderr,
                             "unexpected reserved_prefetch result repeat=%d expert=%d success=%u "
                             "failure=%u accepted=%llu assigned=%llu ordinary=%llu external=%llu "
                             "broken=%llu entry_status=%u\n",
                             repeat,
                             expert,
                             prefetch.success_count,
                             prefetch.failure_count,
                             prefetch.accepted_bytes,
                             prefetch.assigned_slots,
                             prefetch.ordinary_resident_bytes,
                             prefetch.pool_external_alloc_count,
                             prefetch.footprint_contract_broken,
                             prefetch.entries[0].status);
                std::exit(1);
            }

            *errors = 0;
            CUDA_CHECK(cudaDeviceSynchronize());
            check_pattern_kernel<<<grid, block>>>(experts[expert], words, expert_seed(expert), errors);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            if (*errors != 0) {
                std::fprintf(stderr,
                             "kernel validation failed repeat=%d expert=%d errors=%u\n",
                             repeat,
                             expert,
                             *errors);
                total_errors += *errors;
                std::exit(2);
            }
        }
    }

    print_latency_summary("reserved_slot_reuse_prefetch", reserved_ms, bytes);
    std::printf("slot_reuse_correctness experts=%d repeats=%d remaps=%zu kernel_errors=%u\n",
                args.experts,
                args.repeats,
                reserved_ms.size(),
                total_errors);

    ExpertReservedPrefetchParams invalidate{};
    invalidate.mode = kReservedPrefetchModeInvalidate;
    invalidate.count = 1;
    invalidate.gpu_uuid = gpu_uuid;
    invalidate.entries[0].pool_id = 0;
    invalidate.entries[0].slot_id = 0;
    invalidate.entries[0].flags = kStrictFootprint;
    IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, invalidate, "reserved_prefetch_invalidate");
    std::printf("invalidate success=%u invalidated=%llu entry_status=%u\n",
                invalidate.success_count,
                invalidate.invalidated_slots,
                invalidate.entries[0].status);

    for (int expert = 0; expert < args.experts; ++expert) {
        PhaseFUnregisterRegionParams unreg{};
        unreg.region_id = region_ids[expert];
        (void)ioctl(fd, kUvmPhaseFUnregisterRegionIoctl, &unreg);
    }

    ExpertReservedPoolParams reset_after{};
    reset_after.mode = kReservedPoolModeReset;
    reset_after.gpu_uuid = gpu_uuid;
    (void)ioctl(fd, kExpertReservedPoolIoctl, &reset_after);

    unsigned int host_errors = *errors;
    close(fd);
    CUDA_CHECK(cudaFree(errors));
    for (int expert = 0; expert < args.experts; ++expert)
        CUDA_CHECK(cudaFree(experts[expert]));

    return host_errors == 0 ? 0 : 2;
}
