#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static constexpr unsigned int kUvmPhaseFRegisterRegionIoctl = 82;
static constexpr unsigned int kUvmPhaseFUnregisterRegionIoctl = 83;
static constexpr unsigned int kExpertReservedPoolIoctl = 98;
static constexpr unsigned int kExpertReservedPrefetchIoctl = 99;

static constexpr unsigned int kPoolWeight = 2;
static constexpr unsigned int kRoleGateUp = 3;
static constexpr unsigned int kRoleDown = 4;
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
    int layers = 20;
    int offloaded_experts = 3;
    int bank_count = 2;
    unsigned int pool_id = 0;
    size_t gate_bytes = 3ull * 1024 * 1024;
    size_t down_bytes = 1536ull * 1024;
    size_t gate_offset = 1ull * 1024 * 1024;
    size_t down_offset = 1536ull * 1024;
    size_t guard_bytes = 1ull * 1024 * 1024;
    size_t slot_bytes = 4ull * 1024 * 1024;
    int spin_iters = 2000;
    bool quiet_layers = false;
};

struct RegionAlloc {
    unsigned int *raw = nullptr;
    unsigned int *ptr = nullptr;
    size_t alloc_bytes = 0;
    size_t bytes = 0;
    unsigned long long region_id = 0;
    unsigned int layer = 0;
    unsigned int expert = 0;
    unsigned int role = 0;
};

struct BankState {
    cudaEvent_t done = nullptr;
    bool has_owner = false;
    bool has_event = false;
    int last_layer = -1;
};

struct LayerSample {
    double reuse_wait_ms = 0.0;
    double invalidate_ms = 0.0;
    double prefetch_ms = 0.0;
    double submit_ms = 0.0;
};

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static unsigned long long parse_ull(const char *value)
{
    return std::strtoull(value, nullptr, 0);
}

static Args parse_args(int argc, char **argv)
{
    Args args;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--layers") == 0 && i + 1 < argc)
            args.layers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--offloaded-experts") == 0 && i + 1 < argc)
            args.offloaded_experts = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bank-count") == 0 && i + 1 < argc)
            args.bank_count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pool-id") == 0 && i + 1 < argc)
            args.pool_id = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--gate-bytes") == 0 && i + 1 < argc)
            args.gate_bytes = (size_t)parse_ull(argv[++i]);
        else if (std::strcmp(argv[i], "--down-bytes") == 0 && i + 1 < argc)
            args.down_bytes = (size_t)parse_ull(argv[++i]);
        else if (std::strcmp(argv[i], "--gate-offset") == 0 && i + 1 < argc)
            args.gate_offset = (size_t)parse_ull(argv[++i]);
        else if (std::strcmp(argv[i], "--down-offset") == 0 && i + 1 < argc)
            args.down_offset = (size_t)parse_ull(argv[++i]);
        else if (std::strcmp(argv[i], "--guard-bytes") == 0 && i + 1 < argc)
            args.guard_bytes = (size_t)parse_ull(argv[++i]);
        else if (std::strcmp(argv[i], "--slot-bytes") == 0 && i + 1 < argc)
            args.slot_bytes = (size_t)parse_ull(argv[++i]);
        else if (std::strcmp(argv[i], "--spin-iters") == 0 && i + 1 < argc)
            args.spin_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--quiet-layers") == 0)
            args.quiet_layers = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--layers 20] "
                         "[--offloaded-experts 3] [--bank-count 2] "
                         "[--slot-bytes 4194304] [--spin-iters N] "
                         "[--quiet-layers]\n",
                         argv[0]);
            std::exit(2);
        }
    }

    if (args.layers <= 0 ||
        args.offloaded_experts <= 0 ||
        args.bank_count <= 0 ||
        args.bank_count * args.offloaded_experts * 2 > 128 ||
        args.slot_bytes < args.gate_bytes ||
        args.slot_bytes < args.down_bytes ||
        args.gate_bytes % sizeof(unsigned int) != 0 ||
        args.down_bytes % sizeof(unsigned int) != 0 ||
        args.gate_offset % 4096 != 0 ||
        args.down_offset % 4096 != 0) {
        std::fprintf(stderr, "invalid arguments\n");
        std::exit(2);
    }

    return args;
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

static __host__ __device__ unsigned int pattern_seed(unsigned int layer,
                                                     unsigned int expert,
                                                     unsigned int role)
{
    return 0x5a000000u ^
           (layer * 0x00010001u) ^
           (expert * 0x00000111u) ^
           (role * 0x0000003du);
}

static __host__ __device__ unsigned int pattern_value(unsigned int layer,
                                                      unsigned int expert,
                                                      unsigned int role,
                                                      size_t index)
{
    return pattern_seed(layer, expert, role) ^ (unsigned int)(index * 2654435761u);
}

static void fill_region(RegionAlloc &region)
{
    size_t words = region.bytes / sizeof(unsigned int);

    for (size_t i = 0; i < words; ++i)
        region.ptr[i] = pattern_value(region.layer, region.expert, region.role, i);
}

static void print_summary(const char *label, const std::vector<double> &values)
{
    if (values.empty())
        return;

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    double avg = sum / (double)sorted.size();
    double p50 = sorted[sorted.size() / 2];
    double p95 = sorted[(size_t)((sorted.size() - 1) * 0.95)];

    std::printf("%s samples=%zu avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f "
                "min_ms=%.3f max_ms=%.3f\n",
                label,
                sorted.size(),
                avg,
                p50,
                p95,
                sorted.front(),
                sorted.back());
}

__global__ void simulate_moe_weight_kernel(unsigned int **gate_ptrs,
                                           unsigned int **down_ptrs,
                                           unsigned int *errors,
                                           int layer,
                                           int offloaded_experts,
                                           size_t gate_words,
                                           size_t down_words,
                                           int spin_iters)
{
    int task = blockIdx.x;
    int expert = task / 2;
    bool down = (task & 1) != 0;
    unsigned int role = down ? kRoleDown : kRoleGateUp;
    unsigned int *ptr = down ?
        down_ptrs[layer * offloaded_experts + expert] :
        gate_ptrs[layer * offloaded_experts + expert];
    size_t words = down ? down_words : gate_words;
    unsigned int local_errors = 0;

    for (size_t i = threadIdx.x; i < words; i += blockDim.x) {
        unsigned int expected = pattern_value((unsigned int)layer,
                                              (unsigned int)expert,
                                              role,
                                              i);
        if (ptr[i] != expected)
            ++local_errors;
    }

    volatile unsigned int sink = pattern_seed((unsigned int)layer, (unsigned int)expert, role);
    for (int i = 0; i < spin_iters; ++i)
        sink = (sink << 5) ^ (sink >> 2) ^ (unsigned int)i;

    if (sink == 0x12345678u)
        ++local_errors;

    if (local_errors)
        atomicAdd(errors, local_errors);
}

static RegionAlloc allocate_region(size_t bytes,
                                   size_t offset,
                                   size_t guard_bytes,
                                   unsigned int layer,
                                   unsigned int expert,
                                   unsigned int role,
                                   unsigned long long region_id)
{
    RegionAlloc region{};
    region.alloc_bytes = offset + bytes + guard_bytes;
    region.bytes = bytes;
    region.region_id = region_id;
    region.layer = layer;
    region.expert = expert;
    region.role = role;

    CUDA_CHECK(cudaMallocManaged(&region.raw, region.alloc_bytes, cudaMemAttachGlobal));
    region.ptr = reinterpret_cast<unsigned int *>(
        reinterpret_cast<unsigned char *>(region.raw) + offset);
    fill_region(region);
    return region;
}

static void register_region(int fd, const RegionAlloc &region)
{
    PhaseFRegisterRegionParams reg{};
    reg.region.region_id = region.region_id;
    reg.region.base = reinterpret_cast<unsigned long long>(region.ptr);
    reg.region.length = region.bytes;
    reg.region.pool = kPoolWeight;
    reg.region.layer_id = region.layer;
    reg.region.expert_id = region.expert;
    reg.region.role = region.role;
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

static int slot_id_for(const Args &args, int bank, int expert, unsigned int role)
{
    int role_slot = role == kRoleGateUp ? 0 : 1;
    return bank * args.offloaded_experts * 2 + expert * 2 + role_slot;
}

static void prealloc_slots(int fd, const PhaseFUuid &gpu_uuid, const Args &args)
{
    ExpertReservedPoolParams reset{};
    reset.mode = kReservedPoolModeReset;
    reset.gpu_uuid = gpu_uuid;
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, reset, "reserved_pool_reset");

    ExpertReservedPoolParams pool{};
    pool.mode = kReservedPoolModePrealloc;
    pool.count = (unsigned int)(args.bank_count * args.offloaded_experts * 2);
    pool.gpu_uuid = gpu_uuid;

    for (unsigned int slot = 0; slot < pool.count; ++slot) {
        pool.entries[slot].pool_id = args.pool_id;
        pool.entries[slot].slot_id = slot;
        pool.entries[slot].slot_bytes = args.slot_bytes;
    }

    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, pool, "reserved_pool_prealloc");
    std::printf("reserved_pool_prealloc banks=%d slots_per_bank=%d total_slots=%u "
                "success=%u failure=%u backing_bytes=%llu logical_bytes=%llu\n",
                args.bank_count,
                args.offloaded_experts * 2,
                pool.count,
                pool.success_count,
                pool.failure_count,
                pool.reserved_backing_bytes,
                pool.logical_slot_bytes);

    if (pool.success_count != pool.count || pool.failure_count != 0) {
        std::fprintf(stderr, "reserved pool prealloc did not allocate all slots\n");
        std::exit(1);
    }
}

static void invalidate_bank(int fd,
                            const PhaseFUuid &gpu_uuid,
                            const Args &args,
                            int bank)
{
    ExpertReservedPrefetchParams invalidate{};
    invalidate.mode = kReservedPrefetchModeInvalidate;
    invalidate.count = (unsigned int)(args.offloaded_experts * 2);
    invalidate.gpu_uuid = gpu_uuid;

    for (int expert = 0; expert < args.offloaded_experts; ++expert) {
        int gate_index = expert * 2;
        int down_index = gate_index + 1;

        invalidate.entries[gate_index].pool_id = args.pool_id;
        invalidate.entries[gate_index].slot_id = (unsigned int)slot_id_for(args, bank, expert, kRoleGateUp);
        invalidate.entries[gate_index].flags = kStrictFootprint;

        invalidate.entries[down_index].pool_id = args.pool_id;
        invalidate.entries[down_index].slot_id = (unsigned int)slot_id_for(args, bank, expert, kRoleDown);
        invalidate.entries[down_index].flags = kStrictFootprint;
    }

    IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, invalidate, "reserved_prefetch_invalidate_bank");
    if (invalidate.success_count != invalidate.count ||
        invalidate.failure_count != 0 ||
        invalidate.invalidated_slots != invalidate.count) {
        std::fprintf(stderr,
                     "bad invalidate bank=%d success=%u failure=%u invalidated=%llu expected=%u\n",
                     bank,
                     invalidate.success_count,
                     invalidate.failure_count,
                     invalidate.invalidated_slots,
                     invalidate.count);
        std::exit(1);
    }
}

static void prefetch_layer_to_bank(int fd,
                                   const PhaseFUuid &gpu_uuid,
                                   const Args &args,
                                   const std::vector<RegionAlloc> &gate_regions,
                                   const std::vector<RegionAlloc> &down_regions,
                                   int layer,
                                   int bank)
{
    ExpertReservedPrefetchParams prefetch{};
    prefetch.mode = kReservedPrefetchModeReservedPrefetch;
    prefetch.count = (unsigned int)(args.offloaded_experts * 2);
    prefetch.gpu_uuid = gpu_uuid;
    prefetch.async = 1;

    for (int expert = 0; expert < args.offloaded_experts; ++expert) {
        int region_index = layer * args.offloaded_experts + expert;
        int gate_entry = expert * 2;
        int down_entry = gate_entry + 1;

        prefetch.entries[gate_entry].region_id = gate_regions[region_index].region_id;
        prefetch.entries[gate_entry].pool_id = args.pool_id;
        prefetch.entries[gate_entry].slot_id = (unsigned int)slot_id_for(args, bank, expert, kRoleGateUp);
        prefetch.entries[gate_entry].flags = kStrictFootprint;
        prefetch.entries[gate_entry].bytes_override = args.gate_bytes;

        prefetch.entries[down_entry].region_id = down_regions[region_index].region_id;
        prefetch.entries[down_entry].pool_id = args.pool_id;
        prefetch.entries[down_entry].slot_id = (unsigned int)slot_id_for(args, bank, expert, kRoleDown);
        prefetch.entries[down_entry].flags = kStrictFootprint;
        prefetch.entries[down_entry].bytes_override = args.down_bytes;
    }

    IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, prefetch, "reserved_prefetch_layer_bank");

    unsigned long long expected_bytes =
        (unsigned long long)args.offloaded_experts *
        (unsigned long long)(args.gate_bytes + args.down_bytes);
    if (prefetch.success_count != prefetch.count ||
        prefetch.failure_count != 0 ||
        prefetch.accepted_bytes != expected_bytes ||
        prefetch.assigned_slots != prefetch.count ||
        prefetch.ordinary_resident_bytes != 0 ||
        prefetch.pool_external_alloc_count != 0 ||
        prefetch.footprint_contract_broken != 0) {
        std::fprintf(stderr,
                     "bad prefetch layer=%d bank=%d success=%u failure=%u accepted=%llu "
                     "assigned=%llu ordinary=%llu external=%llu broken=%llu expected_bytes=%llu\n",
                     layer,
                     bank,
                     prefetch.success_count,
                     prefetch.failure_count,
                     prefetch.accepted_bytes,
                     prefetch.assigned_slots,
                     prefetch.ordinary_resident_bytes,
                     prefetch.pool_external_alloc_count,
                     prefetch.footprint_contract_broken,
                     expected_bytes);
        std::exit(1);
    }
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    const int regions_per_role = args.layers * args.offloaded_experts;
    const int regions_total = regions_per_role * 2;
    const size_t gate_words = args.gate_bytes / sizeof(unsigned int);
    const size_t down_words = args.down_bytes / sizeof(unsigned int);
    const unsigned long long region_base = 0xABC4E0000000ull;

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

    std::printf("double_slot_event_probe_begin gpu=%s layers=%d offloaded_experts=%d "
                "bank_count=%d slots_per_bank=%d total_slots=%d gate_bytes=%zu "
                "down_bytes=%zu slot_bytes=%zu gate_offset=%zu down_offset=%zu "
                "spin_iters=%d\n",
                prop.name,
                args.layers,
                args.offloaded_experts,
                args.bank_count,
                args.offloaded_experts * 2,
                args.bank_count * args.offloaded_experts * 2,
                args.gate_bytes,
                args.down_bytes,
                args.slot_bytes,
                args.gate_offset,
                args.down_offset,
                args.spin_iters);

    std::vector<RegionAlloc> gate_regions;
    std::vector<RegionAlloc> down_regions;
    std::vector<unsigned int *> gate_ptrs;
    std::vector<unsigned int *> down_ptrs;
    gate_regions.reserve(regions_per_role);
    down_regions.reserve(regions_per_role);
    gate_ptrs.reserve(regions_per_role);
    down_ptrs.reserve(regions_per_role);

    double alloc_start = now_ms();
    for (int layer = 0; layer < args.layers; ++layer) {
        for (int expert = 0; expert < args.offloaded_experts; ++expert) {
            unsigned long long gate_id = region_base +
                                         (unsigned long long)layer * 1000ull +
                                         (unsigned long long)expert * 10ull +
                                         3ull;
            unsigned long long down_id = region_base +
                                         (unsigned long long)layer * 1000ull +
                                         (unsigned long long)expert * 10ull +
                                         4ull;
            RegionAlloc gate = allocate_region(args.gate_bytes,
                                               args.gate_offset,
                                               args.guard_bytes,
                                               (unsigned int)layer,
                                               (unsigned int)expert,
                                               kRoleGateUp,
                                               gate_id);
            RegionAlloc down = allocate_region(args.down_bytes,
                                               args.down_offset,
                                               args.guard_bytes,
                                               (unsigned int)layer,
                                               (unsigned int)expert,
                                               kRoleDown,
                                               down_id);
            gate_ptrs.push_back(gate.ptr);
            down_ptrs.push_back(down.ptr);
            gate_regions.push_back(gate);
            down_regions.push_back(down);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("managed_region_alloc_and_cpu_fill regions=%d total_payload_bytes=%llu ms=%.3f\n",
                regions_total,
                (unsigned long long)args.layers *
                    (unsigned long long)args.offloaded_experts *
                    (unsigned long long)(args.gate_bytes + args.down_bytes),
                now_ms() - alloc_start);

    unsigned int **d_gate_ptrs = nullptr;
    unsigned int **d_down_ptrs = nullptr;
    unsigned int *errors = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gate_ptrs, gate_ptrs.size() * sizeof(unsigned int *)));
    CUDA_CHECK(cudaMalloc(&d_down_ptrs, down_ptrs.size() * sizeof(unsigned int *)));
    CUDA_CHECK(cudaMemcpy(d_gate_ptrs,
                          gate_ptrs.data(),
                          gate_ptrs.size() * sizeof(unsigned int *),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_down_ptrs,
                          down_ptrs.data(),
                          down_ptrs.size() * sizeof(unsigned int *),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocManaged(&errors, sizeof(unsigned int), cudaMemAttachGlobal));
    *errors = 0;
    CUDA_CHECK(cudaMemPrefetchAsync(errors, sizeof(unsigned int), args.device));
    CUDA_CHECK(cudaDeviceSynchronize());

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr, "failed to open or duplicate initialized UVM fd\n");
        return 1;
    }

    prealloc_slots(fd, gpu_uuid, args);

    double register_start = now_ms();
    for (const RegionAlloc &region : gate_regions)
        register_region(fd, region);
    for (const RegionAlloc &region : down_regions)
        register_region(fd, region);
    std::printf("registered_offloaded_regions count=%d ms=%.3f\n",
                regions_total,
                now_ms() - register_start);

    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

    std::vector<BankState> banks(args.bank_count);
    for (BankState &bank : banks)
        CUDA_CHECK(cudaEventCreateWithFlags(&bank.done, cudaEventDisableTiming));

    std::vector<LayerSample> samples;
    samples.reserve(args.layers);
    std::vector<double> wait_values;
    std::vector<double> invalidate_values;
    std::vector<double> prefetch_values;
    std::vector<double> submit_values;

    for (int layer = 0; layer < args.layers; ++layer) {
        int bank = layer % args.bank_count;
        LayerSample sample{};

        if (banks[bank].has_owner) {
            double wait_start = now_ms();
            if (banks[bank].has_event)
                CUDA_CHECK(cudaEventSynchronize(banks[bank].done));
            sample.reuse_wait_ms = now_ms() - wait_start;

            double invalidate_start = now_ms();
            invalidate_bank(fd, gpu_uuid, args, bank);
            sample.invalidate_ms = now_ms() - invalidate_start;
        }

        double prefetch_start = now_ms();
        prefetch_layer_to_bank(fd,
                               gpu_uuid,
                               args,
                               gate_regions,
                               down_regions,
                               layer,
                               bank);
        sample.prefetch_ms = now_ms() - prefetch_start;

        double submit_start = now_ms();
        dim3 grid(args.offloaded_experts * 2);
        dim3 block(256);
        simulate_moe_weight_kernel<<<grid, block, 0, compute_stream>>>(d_gate_ptrs,
                                                                       d_down_ptrs,
                                                                       errors,
                                                                       layer,
                                                                       args.offloaded_experts,
                                                                       gate_words,
                                                                       down_words,
                                                                       args.spin_iters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(banks[bank].done, compute_stream));
        sample.submit_ms = now_ms() - submit_start;

        banks[bank].has_owner = true;
        banks[bank].has_event = true;
        banks[bank].last_layer = layer;

        wait_values.push_back(sample.reuse_wait_ms);
        invalidate_values.push_back(sample.invalidate_ms);
        prefetch_values.push_back(sample.prefetch_ms);
        submit_values.push_back(sample.submit_ms);
        samples.push_back(sample);

        if (!args.quiet_layers) {
            std::printf("layer_submit layer=%d bank=%d previous_layer=%d "
                        "event_wait_ms=%.3f invalidate_ms=%.3f "
                        "prefetch_ms=%.3f submit_ms=%.3f\n",
                        layer,
                        bank,
                        layer >= args.bank_count ? layer - args.bank_count : -1,
                        sample.reuse_wait_ms,
                        sample.invalidate_ms,
                        sample.prefetch_ms,
                        sample.submit_ms);
        }
    }

    double final_cleanup_start = now_ms();
    for (int bank = 0; bank < args.bank_count; ++bank) {
        if (!banks[bank].has_owner)
            continue;

        CUDA_CHECK(cudaEventSynchronize(banks[bank].done));
        invalidate_bank(fd, gpu_uuid, args, bank);
        banks[bank].has_owner = false;
    }
    double final_cleanup_ms = now_ms() - final_cleanup_start;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemPrefetchAsync(errors, sizeof(unsigned int), cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    print_summary("reuse_event_wait", wait_values);
    print_summary("deferred_invalidate", invalidate_values);
    print_summary("reserved_prefetch", prefetch_values);
    print_summary("kernel_submit", submit_values);

    unsigned int final_errors = *errors;

    std::printf("double_slot_event_result layers=%d bank_count=%d remaps=%d "
                "final_cleanup_ms=%.3f kernel_errors=%u status=%s\n",
                args.layers,
                args.bank_count,
                args.layers,
                final_cleanup_ms,
                final_errors,
                final_errors == 0 ? "pass" : "fail");

    for (const RegionAlloc &region : gate_regions)
        unregister_region(fd, region.region_id);
    for (const RegionAlloc &region : down_regions)
        unregister_region(fd, region.region_id);

    ExpertReservedPoolParams reset_after{};
    reset_after.mode = kReservedPoolModeReset;
    reset_after.gpu_uuid = gpu_uuid;
    (void)ioctl(fd, kExpertReservedPoolIoctl, &reset_after);

    for (BankState &bank : banks)
        CUDA_CHECK(cudaEventDestroy(bank.done));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));

    CUDA_CHECK(cudaFree(errors));
    CUDA_CHECK(cudaFree(d_gate_ptrs));
    CUDA_CHECK(cudaFree(d_down_ptrs));
    for (RegionAlloc &region : gate_regions)
        CUDA_CHECK(cudaFree(region.raw));
    for (RegionAlloc &region : down_regions)
        CUDA_CHECK(cudaFree(region.raw));
    close(fd);

    return final_errors == 0 ? 0 : 4;
}
