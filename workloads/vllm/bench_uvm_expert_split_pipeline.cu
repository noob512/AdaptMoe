#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <numeric>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                \
                         "CUDA error %s:%d: %s returned %s\n",                 \
                         __FILE__,                                              \
                         __LINE__,                                              \
                         #expr,                                                 \
                         cudaGetErrorString(_err));                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static constexpr int kThreads = 256;
static constexpr unsigned int kUvmInitializeIoctl = 0x30000001;
static constexpr unsigned int kUvmPhaseFRegisterRegionIoctl = 82;
static constexpr unsigned int kUvmPhaseFUnregisterRegionIoctl = 83;
static constexpr unsigned int kPoolWeight = 2;
static constexpr unsigned int kRoleW13 = 3;
static constexpr unsigned int kRoleW2 = 4;
static constexpr unsigned int kRegionFlagPrefetchable = 1u << 3;
static constexpr unsigned int kRegionFlagGpuReadOnly = 1u << 9;
static constexpr unsigned int kRegionFlagStreamingSubpool = 1u << 12;

struct UvmInitializeParams {
    unsigned long long flags = 0;
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
    PhaseFRegion region;
    unsigned int rmStatus = 0;
};

struct PhaseFUnregisterRegionParams {
    unsigned long long region_id = 0;
    unsigned int rmStatus = 0;
};

struct TaggedRegion {
    unsigned long long id = 0;
};

__global__ void expert_phase_kernel(uint4 *const *experts,
                                    int num_experts,
                                    int blocks_per_expert,
                                    size_t half_offset,
                                    size_t count,
                                    unsigned long long target_cycles,
                                    unsigned long long *partial)
{
    __shared__ unsigned long long scratch[kThreads];
    const int expert = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x - expert * blocks_per_expert;
    unsigned long long acc = 0;
    const size_t stride = (size_t)blockDim.x * blocks_per_expert;

    if (expert >= num_experts)
        return;

    const uint4 *data = experts[expert] + half_offset;

    for (size_t i = (size_t)local_block * blockDim.x + threadIdx.x; i < count; i += stride) {
        uint4 value = data[i];
        acc += (unsigned long long)value.x;
        acc ^= (unsigned long long)value.y << 7;
        acc += (unsigned long long)value.z << 13;
        acc ^= (unsigned long long)value.w << 19;
    }

    const unsigned long long start = clock64();
    while (clock64() - start < target_cycles) {
        acc = acc * 2862933555777941757ULL + 3037000493ULL;
        acc ^= acc >> 23;
    }

    scratch[threadIdx.x] = acc;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        partial[blockIdx.x] = scratch[0];
}

__global__ void activation_phase_kernel(unsigned long long target_cycles,
                                        unsigned long long *partial)
{
    unsigned long long acc =
        ((unsigned long long)blockIdx.x << 32) ^ (unsigned long long)threadIdx.x;

    const unsigned long long start = clock64();
    while (clock64() - start < target_cycles) {
        acc = acc * 2862933555777941757ULL + 3037000493ULL;
        acc ^= acc >> 23;
    }

    if (threadIdx.x == 0)
        partial[blockIdx.x] ^= acc;
}

struct Args {
    int device = 0;
    int experts = 10;
    int repeats = 5;
    int warmups = 1;
    int blocks_per_sm = 1;
    int upper_us = 350;
    int activation_us = 50;
    int lower_us = 350;
    size_t expert_bytes = 4ull * 1024ull * 1024ull;
    std::string mode = "all";
    bool advise_read_mostly = false;
    bool tag_expert_regions = false;
};

struct ModeSample {
    std::string mode;
    int iter = 0;
    double total_ms = 0.0;
    double prefetch_ms = 0.0;
    double submit_ms = 0.0;
    double wait_ms = 0.0;
    double upper_prefetch_ms = 0.0;
    double lower_prefetch_ms = 0.0;
    double upper_kernel_ms = 0.0;
    double activation_kernel_ms = 0.0;
    double lower_kernel_ms = 0.0;
};

struct Stats {
    double min = 0.0;
    double p50 = 0.0;
    double mean = 0.0;
    double p90 = 0.0;
    double max = 0.0;
};

static void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "usage: %s [--device N] [--experts N] [--repeats N] [--warmups N]\n"
                 "          [--expert-mib N] [--upper-us N] [--activation-us N] [--lower-us N]\n"
                 "          [--blocks-per-sm N] [--mode all|compare_handoff|gpu_resident|fault_only|sync_full|split_overlap|async_full|prefetch_split_timing]\n"
                 "          [--advise-read-mostly] [--tag-expert-regions]\n",
                 argv0);
}

static Args parse_args(int argc, char **argv)
{
    Args args;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            args.device = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            args.experts = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            args.repeats = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            args.warmups = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--expert-mib") == 0 && i + 1 < argc) {
            args.expert_bytes = (size_t)std::strtoull(argv[++i], nullptr, 10) * 1024ull * 1024ull;
        }
        else if (std::strcmp(argv[i], "--upper-us") == 0 && i + 1 < argc) {
            args.upper_us = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--activation-us") == 0 && i + 1 < argc) {
            args.activation_us = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--lower-us") == 0 && i + 1 < argc) {
            args.lower_us = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--blocks-per-sm") == 0 && i + 1 < argc) {
            args.blocks_per_sm = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            args.mode = argv[++i];
        }
        else if (std::strcmp(argv[i], "--advise-read-mostly") == 0) {
            args.advise_read_mostly = true;
        }
        else if (std::strcmp(argv[i], "--tag-expert-regions") == 0) {
            args.tag_expert_regions = true;
        }
        else {
            usage(argv[0]);
            std::exit(2);
        }
    }

    args.expert_bytes = (args.expert_bytes / (2 * sizeof(uint4))) * (2 * sizeof(uint4));
    if (args.device < 0 ||
        args.experts <= 0 ||
        args.repeats <= 0 ||
        args.warmups < 0 ||
        args.blocks_per_sm <= 0 ||
        args.upper_us < 0 ||
        args.activation_us < 0 ||
        args.lower_us < 0 ||
        args.expert_bytes < 2 * sizeof(uint4)) {
        usage(argv[0]);
        std::exit(2);
    }

    return args;
}

static unsigned long long cycles_for_us(int us, int clock_rate_khz)
{
    if (us <= 0)
        return 0;
    return ((unsigned long long)clock_rate_khz * (unsigned long long)us) / 1000ull;
}

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

static float elapsed_event(cudaEvent_t start, cudaEvent_t end)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
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

static bool register_region(int fd,
                            unsigned long long region_id,
                            void *base,
                            size_t bytes,
                            unsigned int expert_id,
                            unsigned int role)
{
    PhaseFRegisterRegionParams params{};
    params.region.region_id = region_id;
    params.region.base = reinterpret_cast<unsigned long long>(base);
    params.region.length = static_cast<unsigned long long>(bytes);
    params.region.pool = kPoolWeight;
    params.region.layer_id = 0;
    params.region.expert_id = expert_id;
    params.region.role = role;
    params.region.flags = kRegionFlagPrefetchable | kRegionFlagGpuReadOnly | kRegionFlagStreamingSubpool;

    if (ioctl(fd, kUvmPhaseFRegisterRegionIoctl, &params) != 0)
        return false;

    return params.rmStatus == 0;
}

static void unregister_region(int fd, unsigned long long region_id)
{
    PhaseFUnregisterRegionParams params{};
    params.region_id = region_id;
    (void)ioctl(fd, kUvmPhaseFUnregisterRegionIoctl, &params);
}

static std::vector<TaggedRegion> tag_expert_regions(const std::vector<uint4 *> &experts,
                                                    const Args &args)
{
    std::vector<TaggedRegion> tagged;

    if (!args.tag_expert_regions)
        return tagged;

    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr,
                     "warning: failed to open initialized /dev/nvidia-uvm for region tagging; "
                     "expert prefetch handoff will not be exercised\n");
        return tagged;
    }

    const size_t half_bytes = args.expert_bytes / 2;
    const unsigned long long base_id = 0xE900000000ull;
    tagged.reserve(experts.size() * 2);

    for (size_t i = 0; i < experts.size(); ++i) {
        unsigned long long upper_id = base_id + (unsigned long long)i * 2ull;
        unsigned long long lower_id = upper_id + 1ull;
        void *upper = experts[i];
        void *lower = reinterpret_cast<char *>(experts[i]) + half_bytes;

        if (!register_region(fd, upper_id, upper, half_bytes, (unsigned int)i, kRoleW13) ||
            !register_region(fd, lower_id, lower, half_bytes, (unsigned int)i, kRoleW2)) {
            std::fprintf(stderr,
                         "warning: failed to register expert %zu W13/W2 regions; "
                         "disable TAG_EXPERT_REGIONS or load the modified UVM driver\n",
                         i);
            for (const TaggedRegion &region : tagged)
                unregister_region(fd, region.id);
            tagged.clear();
            close(fd);
            return tagged;
        }

        tagged.push_back({upper_id});
        tagged.push_back({lower_id});
    }

    close(fd);
    std::printf("tagged_expert_regions count=%zu pool=WEIGHT roles=W13,W2\n", tagged.size());
    return tagged;
}

static void untag_expert_regions(const std::vector<TaggedRegion> &tagged)
{
    if (tagged.empty())
        return;

    int fd = open_registration_uvm_fd();
    if (fd < 0)
        return;

    for (const TaggedRegion &region : tagged)
        unregister_region(fd, region.id);

    close(fd);
}

static float prefetch_many_sync(const std::vector<uint4 *> &experts,
                                size_t offset_bytes,
                                size_t bytes,
                                int device,
                                cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (uint4 *expert : experts) {
        void *ptr = reinterpret_cast<char *>(expert) + offset_bytes;
        CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, stream));
    }
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const float ms = elapsed_event(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

struct PrefetchTiming {
    double event_ms = 0.0;
    double submit_ms = 0.0;
    double wait_ms = 0.0;
    double wall_ms = 0.0;
};

static PrefetchTiming prefetch_many_sync_timing(const std::vector<uint4 *> &experts,
                                                size_t offset_bytes,
                                                size_t bytes,
                                                int device,
                                                cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    const double wall_start = now_ms();
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (uint4 *expert : experts) {
        void *ptr = reinterpret_cast<char *>(expert) + offset_bytes;
        CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, stream));
    }
    CUDA_CHECK(cudaEventRecord(end, stream));

    const double submit_done = now_ms();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double wait_done = now_ms();

    PrefetchTiming timing;
    timing.event_ms = elapsed_event(start, end);
    timing.submit_ms = submit_done - wall_start;
    timing.wait_ms = wait_done - submit_done;
    timing.wall_ms = wait_done - wall_start;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return timing;
}

static void prefetch_many_async_timed(const std::vector<uint4 *> &experts,
                                      size_t offset_bytes,
                                      size_t bytes,
                                      int device,
                                      cudaStream_t stream,
                                      cudaEvent_t *start_out,
                                      cudaEvent_t *end_out)
{
    CUDA_CHECK(cudaEventCreate(start_out));
    CUDA_CHECK(cudaEventCreate(end_out));
    CUDA_CHECK(cudaEventRecord(*start_out, stream));
    for (uint4 *expert : experts) {
        void *ptr = reinterpret_cast<char *>(expert) + offset_bytes;
        CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, stream));
    }
    CUDA_CHECK(cudaEventRecord(*end_out, stream));
}

static void launch_phase_timed(uint4 **expert_ptrs,
                               int num_experts,
                               int blocks_per_expert,
                               size_t half_offset,
                               size_t half_count,
                               unsigned long long target_cycles,
                               unsigned long long *partial,
                               cudaStream_t stream,
                               cudaEvent_t *start_out,
                               cudaEvent_t *end_out)
{
    CUDA_CHECK(cudaEventCreate(start_out));
    CUDA_CHECK(cudaEventCreate(end_out));
    CUDA_CHECK(cudaEventRecord(*start_out, stream));
    expert_phase_kernel<<<num_experts * blocks_per_expert, kThreads, 0, stream>>>(
        expert_ptrs, num_experts, blocks_per_expert, half_offset, half_count, target_cycles, partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(*end_out, stream));
}

static void launch_activation_timed(int grid,
                                    unsigned long long target_cycles,
                                    unsigned long long *partial,
                                    cudaStream_t stream,
                                    cudaEvent_t *start_out,
                                    cudaEvent_t *end_out)
{
    CUDA_CHECK(cudaEventCreate(start_out));
    CUDA_CHECK(cudaEventCreate(end_out));
    CUDA_CHECK(cudaEventRecord(*start_out, stream));
    activation_phase_kernel<<<grid, kThreads, 0, stream>>>(target_cycles, partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(*end_out, stream));
}

static void destroy_pair(cudaEvent_t start, cudaEvent_t end)
{
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
}

static void fill_cpu(uint4 *ptr, size_t count, unsigned int seed)
{
    for (size_t i = 0; i < count; ++i)
        ptr[i] = make_uint4((unsigned int)i ^ seed, seed + 1, seed + 2, seed + 3);
}

static uint4 *alloc_expert(const Args &args, unsigned int seed)
{
    void *raw = nullptr;
    CUDA_CHECK(cudaMallocManaged(&raw, args.expert_bytes, cudaMemAttachGlobal));
    if (args.advise_read_mostly)
        CUDA_CHECK(cudaMemAdvise(raw, args.expert_bytes, cudaMemAdviseSetReadMostly, args.device));
    fill_cpu(static_cast<uint4 *>(raw), args.expert_bytes / sizeof(uint4), seed);
    return static_cast<uint4 *>(raw);
}

static void setup_cpu_resident(const std::vector<uint4 *> &experts,
                               const Args &args,
                               cudaStream_t stream,
                               unsigned int seed)
{
    for (uint4 *expert : experts)
        CUDA_CHECK(cudaMemPrefetchAsync(expert, args.expert_bytes, cudaCpuDeviceId, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < experts.size(); ++i)
        fill_cpu(experts[i], args.expert_bytes / sizeof(uint4), seed + (unsigned int)i * 17u);
}

static ModeSample run_gpu_resident(const std::vector<uint4 *> &experts,
                                   const Args &args,
                                   int iter,
                                   uint4 **expert_ptrs,
                                   int blocks_per_expert,
                                   unsigned long long upper_cycles,
                                   unsigned long long activation_cycles,
                                   unsigned long long lower_cycles,
                                   unsigned long long *partial,
                                   cudaStream_t setup_stream,
                                   cudaStream_t prefetch_stream,
                                   cudaStream_t compute_stream)
{
    const size_t half_bytes = args.expert_bytes / 2;
    const int phase_grid = args.experts * blocks_per_expert;
    ModeSample sample;
    sample.mode = "gpu_resident";
    sample.iter = iter;

    setup_cpu_resident(experts, args, setup_stream, 0x10000u + (unsigned int)iter * 97u);
    sample.prefetch_ms += prefetch_many_sync(experts, 0, args.expert_bytes, args.device, prefetch_stream);

    const double start = now_ms();
    cudaEvent_t upper_start, upper_end, activation_start, activation_end, lower_start, lower_end;
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       0,
                       half_bytes / sizeof(uint4),
                       upper_cycles,
                       partial,
                       compute_stream,
                       &upper_start,
                       &upper_end);
    launch_activation_timed(phase_grid,
                            activation_cycles,
                            partial,
                            compute_stream,
                            &activation_start,
                            &activation_end);
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       half_bytes / sizeof(uint4),
                       half_bytes / sizeof(uint4),
                       lower_cycles,
                       partial,
                       compute_stream,
                       &lower_start,
                       &lower_end);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    sample.upper_kernel_ms += elapsed_event(upper_start, upper_end);
    sample.activation_kernel_ms += elapsed_event(activation_start, activation_end);
    sample.lower_kernel_ms += elapsed_event(lower_start, lower_end);
    destroy_pair(upper_start, upper_end);
    destroy_pair(activation_start, activation_end);
    destroy_pair(lower_start, lower_end);
    sample.total_ms = now_ms() - start;
    return sample;
}

static ModeSample run_fault_only(const std::vector<uint4 *> &experts,
                                 const Args &args,
                                 int iter,
                                 uint4 **expert_ptrs,
                                 int blocks_per_expert,
                                 unsigned long long upper_cycles,
                                 unsigned long long activation_cycles,
                                 unsigned long long lower_cycles,
                                 unsigned long long *partial,
                                 cudaStream_t setup_stream,
                                 cudaStream_t compute_stream)
{
    const size_t half_bytes = args.expert_bytes / 2;
    const int phase_grid = args.experts * blocks_per_expert;
    ModeSample sample;
    sample.mode = "fault_only";
    sample.iter = iter;

    setup_cpu_resident(experts, args, setup_stream, 0x20000u + (unsigned int)iter * 97u);

    const double start = now_ms();
    cudaEvent_t upper_start, upper_end, activation_start, activation_end, lower_start, lower_end;
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       0,
                       half_bytes / sizeof(uint4),
                       upper_cycles,
                       partial,
                       compute_stream,
                       &upper_start,
                       &upper_end);
    launch_activation_timed(phase_grid,
                            activation_cycles,
                            partial,
                            compute_stream,
                            &activation_start,
                            &activation_end);
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       half_bytes / sizeof(uint4),
                       half_bytes / sizeof(uint4),
                       lower_cycles,
                       partial,
                       compute_stream,
                       &lower_start,
                       &lower_end);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    sample.upper_kernel_ms += elapsed_event(upper_start, upper_end);
    sample.activation_kernel_ms += elapsed_event(activation_start, activation_end);
    sample.lower_kernel_ms += elapsed_event(lower_start, lower_end);
    destroy_pair(upper_start, upper_end);
    destroy_pair(activation_start, activation_end);
    destroy_pair(lower_start, lower_end);
    sample.total_ms = now_ms() - start;
    return sample;
}

static ModeSample run_sync_full(const std::vector<uint4 *> &experts,
                                const Args &args,
                                int iter,
                                uint4 **expert_ptrs,
                                int blocks_per_expert,
                                unsigned long long upper_cycles,
                                unsigned long long activation_cycles,
                                unsigned long long lower_cycles,
                                unsigned long long *partial,
                                cudaStream_t setup_stream,
                                cudaStream_t prefetch_stream,
                                cudaStream_t compute_stream)
{
    const size_t half_bytes = args.expert_bytes / 2;
    const int phase_grid = args.experts * blocks_per_expert;
    ModeSample sample;
    sample.mode = "sync_full";
    sample.iter = iter;

    setup_cpu_resident(experts, args, setup_stream, 0x30000u + (unsigned int)iter * 97u);

    const double start = now_ms();
    sample.prefetch_ms += prefetch_many_sync(experts, 0, args.expert_bytes, args.device, prefetch_stream);

    cudaEvent_t upper_start, upper_end, activation_start, activation_end, lower_start, lower_end;
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       0,
                       half_bytes / sizeof(uint4),
                       upper_cycles,
                       partial,
                       compute_stream,
                       &upper_start,
                       &upper_end);
    launch_activation_timed(phase_grid,
                            activation_cycles,
                            partial,
                            compute_stream,
                            &activation_start,
                            &activation_end);
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       half_bytes / sizeof(uint4),
                       half_bytes / sizeof(uint4),
                       lower_cycles,
                       partial,
                       compute_stream,
                       &lower_start,
                       &lower_end);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    sample.upper_kernel_ms += elapsed_event(upper_start, upper_end);
    sample.activation_kernel_ms += elapsed_event(activation_start, activation_end);
    sample.lower_kernel_ms += elapsed_event(lower_start, lower_end);
    destroy_pair(upper_start, upper_end);
    destroy_pair(activation_start, activation_end);
    destroy_pair(lower_start, lower_end);
    sample.total_ms = now_ms() - start;
    return sample;
}

static ModeSample run_split_overlap(const std::vector<uint4 *> &experts,
                                    const Args &args,
                                    int iter,
                                    uint4 **expert_ptrs,
                                    int blocks_per_expert,
                                    unsigned long long upper_cycles,
                                    unsigned long long activation_cycles,
                                    unsigned long long lower_cycles,
                                    unsigned long long *partial,
                                    cudaStream_t setup_stream,
                                    cudaStream_t upper_prefetch_stream,
                                    cudaStream_t lower_prefetch_stream,
                                    cudaStream_t compute_stream)
{
    const size_t half_bytes = args.expert_bytes / 2;
    const int phase_grid = args.experts * blocks_per_expert;
    ModeSample sample;
    sample.mode = "split_overlap";
    sample.iter = iter;

    setup_cpu_resident(experts, args, setup_stream, 0x40000u + (unsigned int)iter * 97u);

    const double start = now_ms();
    sample.upper_prefetch_ms += prefetch_many_sync(experts, 0, half_bytes, args.device, upper_prefetch_stream);

    cudaEvent_t lower_prefetch_start, lower_prefetch_end;
    cudaEvent_t upper_start, upper_end, activation_start, activation_end, lower_start, lower_end;

    prefetch_many_async_timed(experts,
                              half_bytes,
                              half_bytes,
                              args.device,
                              lower_prefetch_stream,
                              &lower_prefetch_start,
                              &lower_prefetch_end);

    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       0,
                       half_bytes / sizeof(uint4),
                       upper_cycles,
                       partial,
                       compute_stream,
                       &upper_start,
                       &upper_end);
    launch_activation_timed(phase_grid,
                            activation_cycles,
                            partial,
                            compute_stream,
                            &activation_start,
                            &activation_end);
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       half_bytes / sizeof(uint4),
                       half_bytes / sizeof(uint4),
                       lower_cycles,
                       partial,
                       compute_stream,
                       &lower_start,
                       &lower_end);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaEventSynchronize(lower_prefetch_end));

    sample.lower_prefetch_ms += elapsed_event(lower_prefetch_start, lower_prefetch_end);
    sample.upper_kernel_ms += elapsed_event(upper_start, upper_end);
    sample.activation_kernel_ms += elapsed_event(activation_start, activation_end);
    sample.lower_kernel_ms += elapsed_event(lower_start, lower_end);

    destroy_pair(lower_prefetch_start, lower_prefetch_end);
    destroy_pair(upper_start, upper_end);
    destroy_pair(activation_start, activation_end);
    destroy_pair(lower_start, lower_end);
    sample.total_ms = now_ms() - start;
    sample.prefetch_ms = sample.upper_prefetch_ms + sample.lower_prefetch_ms;
    return sample;
}

static ModeSample run_prefetch_split_timing(const std::vector<uint4 *> &experts,
                                            const Args &args,
                                            int iter,
                                            cudaStream_t setup_stream,
                                            cudaStream_t prefetch_stream)
{
    const size_t half_bytes = args.expert_bytes / 2;
    ModeSample sample;
    sample.mode = "prefetch_split_timing";
    sample.iter = iter;

    setup_cpu_resident(experts, args, setup_stream, 0x60000u + (unsigned int)iter * 97u);

    const double start = now_ms();
    PrefetchTiming upper = prefetch_many_sync_timing(experts, 0, half_bytes, args.device, prefetch_stream);
    PrefetchTiming lower = prefetch_many_sync_timing(experts, half_bytes, half_bytes, args.device, prefetch_stream);

    sample.upper_prefetch_ms = upper.event_ms;
    sample.lower_prefetch_ms = lower.event_ms;
    sample.prefetch_ms = upper.event_ms + lower.event_ms;
    sample.submit_ms = upper.submit_ms + lower.submit_ms;
    sample.wait_ms = upper.wait_ms + lower.wait_ms;
    sample.total_ms = now_ms() - start;
    return sample;
}

static ModeSample run_async_full(const std::vector<uint4 *> &experts,
                                 const Args &args,
                                 int iter,
                                 uint4 **expert_ptrs,
                                 int blocks_per_expert,
                                 unsigned long long upper_cycles,
                                 unsigned long long activation_cycles,
                                 unsigned long long lower_cycles,
                                 unsigned long long *partial,
                                 cudaStream_t setup_stream,
                                 cudaStream_t prefetch_stream,
                                 cudaStream_t compute_stream)
{
    const size_t half_bytes = args.expert_bytes / 2;
    const int phase_grid = args.experts * blocks_per_expert;
    ModeSample sample;
    sample.mode = "async_full";
    sample.iter = iter;

    setup_cpu_resident(experts, args, setup_stream, 0x50000u + (unsigned int)iter * 97u);

    const double start = now_ms();
    cudaEvent_t prefetch_start, prefetch_end;
    cudaEvent_t upper_start, upper_end, activation_start, activation_end, lower_start, lower_end;

    prefetch_many_async_timed(experts,
                              0,
                              args.expert_bytes,
                              args.device,
                              prefetch_stream,
                              &prefetch_start,
                              &prefetch_end);
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       0,
                       half_bytes / sizeof(uint4),
                       upper_cycles,
                       partial,
                       compute_stream,
                       &upper_start,
                       &upper_end);
    launch_activation_timed(phase_grid,
                            activation_cycles,
                            partial,
                            compute_stream,
                            &activation_start,
                            &activation_end);
    launch_phase_timed(expert_ptrs,
                       args.experts,
                       blocks_per_expert,
                       half_bytes / sizeof(uint4),
                       half_bytes / sizeof(uint4),
                       lower_cycles,
                       partial,
                       compute_stream,
                       &lower_start,
                       &lower_end);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaEventSynchronize(prefetch_end));

    sample.prefetch_ms += elapsed_event(prefetch_start, prefetch_end);
    sample.upper_kernel_ms += elapsed_event(upper_start, upper_end);
    sample.activation_kernel_ms += elapsed_event(activation_start, activation_end);
    sample.lower_kernel_ms += elapsed_event(lower_start, lower_end);

    destroy_pair(prefetch_start, prefetch_end);
    destroy_pair(upper_start, upper_end);
    destroy_pair(activation_start, activation_end);
    destroy_pair(lower_start, lower_end);
    sample.total_ms = now_ms() - start;
    return sample;
}

static std::vector<std::string> selected_modes(const std::string &mode)
{
    if (mode == "all")
        return {"gpu_resident", "fault_only", "sync_full", "split_overlap", "async_full"};
    if (mode == "compare_handoff")
        return {"gpu_resident", "sync_full", "split_overlap"};
    if (mode == "prefetch_split_timing")
        return {"prefetch_split_timing"};
    return {mode};
}

static Stats stats_of(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    Stats s;
    s.min = values.front();
    s.p50 = values[values.size() / 2];
    s.p90 = values[(values.size() * 9) / 10 < values.size() ? (values.size() * 9) / 10 : values.size() - 1];
    s.max = values.back();
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return s;
}

static void print_sample(const ModeSample &s)
{
    std::printf("sample mode=%s iter=%d total_ms=%.3f prefetch_ms=%.3f "
                "submit_ms=%.3f wait_ms=%.3f "
                "upper_prefetch_ms=%.3f lower_prefetch_ms=%.3f "
                "upper_kernel_ms=%.3f activation_kernel_ms=%.3f lower_kernel_ms=%.3f\n",
                s.mode.c_str(),
                s.iter,
                s.total_ms,
                s.prefetch_ms,
                s.submit_ms,
                s.wait_ms,
                s.upper_prefetch_ms,
                s.lower_prefetch_ms,
                s.upper_kernel_ms,
                s.activation_kernel_ms,
                s.lower_kernel_ms);
}

static void print_summary(const std::string &mode, const std::vector<ModeSample> &samples, double sync_p50)
{
    std::vector<double> total;
    std::vector<double> upper_kernel;
    std::vector<double> activation_kernel;
    std::vector<double> lower_kernel;
    std::vector<double> prefetch;
    std::vector<double> submit;
    std::vector<double> wait;

    for (const ModeSample &s : samples) {
        if (s.mode != mode)
            continue;
        total.push_back(s.total_ms);
        upper_kernel.push_back(s.upper_kernel_ms);
        activation_kernel.push_back(s.activation_kernel_ms);
        lower_kernel.push_back(s.lower_kernel_ms);
        prefetch.push_back(s.prefetch_ms);
        submit.push_back(s.submit_ms);
        wait.push_back(s.wait_ms);
    }

    if (total.empty())
        return;

    const Stats total_s = stats_of(total);
    const Stats upper_s = stats_of(upper_kernel);
    const Stats activation_s = stats_of(activation_kernel);
    const Stats lower_s = stats_of(lower_kernel);
    const Stats prefetch_s = stats_of(prefetch);
    const Stats submit_s = stats_of(submit);
    const Stats wait_s = stats_of(wait);
    const double speedup = (sync_p50 > 0.0 && mode != "sync_full") ? sync_p50 / total_s.p50 : 1.0;

    std::printf("summary mode=%s total_p50_ms=%.3f total_mean_ms=%.3f total_p90_ms=%.3f "
                "prefetch_p50_ms=%.3f submit_p50_ms=%.3f wait_p50_ms=%.3f "
                "upper_kernel_p50_ms=%.3f activation_kernel_p50_ms=%.3f "
                "lower_kernel_p50_ms=%.3f "
                "speedup_vs_sync_full=%.3f\n",
                mode.c_str(),
                total_s.p50,
                total_s.mean,
                total_s.p90,
                prefetch_s.p50,
                submit_s.p50,
                wait_s.p50,
                upper_s.p50,
                activation_s.p50,
                lower_s.p50,
                speedup);
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    CUDA_CHECK(cudaSetDevice(args.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    const int target_phase_blocks = std::max(args.experts, prop.multiProcessorCount * args.blocks_per_sm);
    const int blocks_per_expert = std::max(1, target_phase_blocks / args.experts);
    const int grid = args.experts * blocks_per_expert;
    const size_t half_bytes = args.expert_bytes / 2;
    const unsigned long long upper_cycles = cycles_for_us(args.upper_us, prop.clockRate);
    const unsigned long long activation_cycles = cycles_for_us(args.activation_us, prop.clockRate);
    const unsigned long long lower_cycles = cycles_for_us(args.lower_us, prop.clockRate);
    const std::vector<std::string> modes = selected_modes(args.mode);

    cudaStream_t setup_stream, upper_prefetch_stream, lower_prefetch_stream, compute_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&setup_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&upper_prefetch_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&lower_prefetch_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

    std::vector<uint4 *> experts;
    experts.reserve(args.experts);
    for (int i = 0; i < args.experts; ++i)
        experts.push_back(alloc_expert(args, 0xabc000u + (unsigned int)i * 31u));

    std::vector<TaggedRegion> tagged_regions = tag_expert_regions(experts, args);

    uint4 **expert_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&expert_ptrs, experts.size() * sizeof(*expert_ptrs)));
    CUDA_CHECK(cudaMemcpy(expert_ptrs,
                          experts.data(),
                          experts.size() * sizeof(*expert_ptrs),
                          cudaMemcpyHostToDevice));

    unsigned long long *partial = nullptr;
    CUDA_CHECK(cudaMalloc(&partial, (size_t)grid * sizeof(*partial)));

    std::printf("config device=%d name=\"%s\" sms=%d execution=grouped_phase_w13_then_w2 "
                "target_phase_blocks=%d grid=%d blocks_per_expert=%d experts=%d expert_mib=%.2f "
                "upper_mib=%.2f lower_mib=%.2f upper_us=%d activation_us=%d lower_us=%d "
                "repeats=%d warmups=%d "
                "advise_read_mostly=%d tag_expert_regions=%d tagged_region_count=%zu mode=%s\n",
                args.device,
                prop.name,
                prop.multiProcessorCount,
                target_phase_blocks,
                grid,
                blocks_per_expert,
                args.experts,
                (double)args.expert_bytes / 1024.0 / 1024.0,
                (double)half_bytes / 1024.0 / 1024.0,
                (double)half_bytes / 1024.0 / 1024.0,
                args.upper_us,
                args.activation_us,
                args.lower_us,
                args.repeats,
                args.warmups,
                args.advise_read_mostly ? 1 : 0,
                args.tag_expert_regions ? 1 : 0,
                tagged_regions.size(),
                args.mode.c_str());

    std::vector<ModeSample> samples;
    const int first_iter = -args.warmups;
    for (int iter = first_iter; iter < args.repeats; ++iter) {
        for (const std::string &mode : modes) {
            ModeSample sample;
            if (mode == "gpu_resident") {
                sample = run_gpu_resident(experts,
                                          args,
                                          iter,
                                          expert_ptrs,
                                          blocks_per_expert,
                                          upper_cycles,
                                          activation_cycles,
                                          lower_cycles,
                                          partial,
                                          setup_stream,
                                          upper_prefetch_stream,
                                          compute_stream);
            }
            else if (mode == "fault_only") {
                sample = run_fault_only(experts,
                                        args,
                                        iter,
                                        expert_ptrs,
                                        blocks_per_expert,
                                        upper_cycles,
                                        activation_cycles,
                                        lower_cycles,
                                        partial,
                                        setup_stream,
                                        compute_stream);
            }
            else if (mode == "sync_full") {
                sample = run_sync_full(experts,
                                       args,
                                       iter,
                                       expert_ptrs,
                                       blocks_per_expert,
                                       upper_cycles,
                                       activation_cycles,
                                       lower_cycles,
                                       partial,
                                       setup_stream,
                                       upper_prefetch_stream,
                                       compute_stream);
            }
            else if (mode == "split_overlap") {
                sample = run_split_overlap(experts,
                                           args,
                                           iter,
                                           expert_ptrs,
                                           blocks_per_expert,
                                           upper_cycles,
                                           activation_cycles,
                                           lower_cycles,
                                           partial,
                                           setup_stream,
                                           upper_prefetch_stream,
                                           lower_prefetch_stream,
                                           compute_stream);
            }
            else if (mode == "prefetch_split_timing") {
                sample = run_prefetch_split_timing(experts,
                                                   args,
                                                   iter,
                                                   setup_stream,
                                                   upper_prefetch_stream);
            }
            else if (mode == "async_full") {
                sample = run_async_full(experts,
                                        args,
                                        iter,
                                        expert_ptrs,
                                        blocks_per_expert,
                                        upper_cycles,
                                        activation_cycles,
                                        lower_cycles,
                                        partial,
                                        setup_stream,
                                        lower_prefetch_stream,
                                        compute_stream);
            }
            else {
                std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
                std::exit(2);
            }

            if (iter >= 0) {
                print_sample(sample);
                samples.push_back(sample);
            }
        }
    }

    double sync_p50 = 0.0;
    {
        std::vector<double> sync_totals;
        for (const ModeSample &s : samples) {
            if (s.mode == "sync_full")
                sync_totals.push_back(s.total_ms);
        }
        if (!sync_totals.empty())
            sync_p50 = stats_of(sync_totals).p50;
    }

    for (const std::string &mode : modes)
        print_summary(mode, samples, sync_p50);

    untag_expert_regions(tagged_regions);
    for (uint4 *expert : experts)
        CUDA_CHECK(cudaFree(expert));
    CUDA_CHECK(cudaFree(expert_ptrs));
    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaStreamDestroy(setup_stream));
    CUDA_CHECK(cudaStreamDestroy(upper_prefetch_stream));
    CUDA_CHECK(cudaStreamDestroy(lower_prefetch_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    return 0;
}
