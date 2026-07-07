#include <algorithm>
#include <chrono>
#include <cmath>
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
    int layers = 20;
    int resident_experts = 80;
    int offloaded_experts = 10;
    int m = 512;
    int k = 1024;
    unsigned int pool_id = 0;
    size_t slot_bytes = 0;
    bool verbose_layers = true;
    bool compare_native = false;
};

struct LayerSample {
    double prefetch_ms = 0.0;
    float compute_ms = 0.0f;
    double invalidate_ms = 0.0;
    double validate_ms = 0.0;
    double total_ms = 0.0;
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
        else if (std::strcmp(argv[i], "--resident-experts") == 0 && i + 1 < argc)
            args.resident_experts = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--offloaded-experts") == 0 && i + 1 < argc)
            args.offloaded_experts = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc)
            args.m = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            args.k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pool-id") == 0 && i + 1 < argc)
            args.pool_id = (unsigned int)std::strtoul(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--slot-bytes") == 0 && i + 1 < argc)
            args.slot_bytes = (size_t)parse_ull(argv[++i]);
        else if (std::strcmp(argv[i], "--quiet-layers") == 0)
            args.verbose_layers = false;
        else if (std::strcmp(argv[i], "--compare-native") == 0)
            args.compare_native = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--layers 20] "
                         "[--resident-experts 80] [--offloaded-experts 10] "
                         "[--m 512] [--k 1024] [--pool-id 0] "
                         "[--slot-bytes N] [--quiet-layers] [--compare-native]\n",
                         argv[0]);
            std::exit(2);
        }
    }

    if (args.layers <= 0 ||
        args.resident_experts <= 0 ||
        args.offloaded_experts <= 0 ||
        args.offloaded_experts > 128 ||
        args.m <= 0 ||
        args.k <= 0) {
        std::fprintf(stderr, "invalid arguments\n");
        std::exit(2);
    }

    return args;
}

static double sample_average(const std::vector<LayerSample> &samples, double LayerSample::*field)
{
    double sum = 0.0;
    for (const LayerSample &sample : samples)
        sum += sample.*field;
    return sum / (double)samples.size();
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

static void print_wall_summary(const char *label,
                               const std::vector<LayerSample> &samples,
                               double LayerSample::*field,
                               size_t bytes_per_sample)
{
    std::vector<double> values;
    values.reserve(samples.size());
    for (const LayerSample &sample : samples)
        values.push_back(sample.*field);

    std::sort(values.begin(), values.end());
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double avg = sum / (double)values.size();
    double p50 = values[values.size() / 2];
    double p95 = values[(size_t)((values.size() - 1) * 0.95)];
    double gib = (double)bytes_per_sample / (1024.0 * 1024.0 * 1024.0);
    double gibps = bytes_per_sample ? gib / (avg / 1000.0) : 0.0;

    std::printf("%s samples=%zu avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f "
                "min_ms=%.3f max_ms=%.3f avg_GiBps=%.3f\n",
                label,
                values.size(),
                avg,
                p50,
                p95,
                values.front(),
                values.back(),
                gibps);
}

static void print_compute_summary(const char *label,
                                  const std::vector<LayerSample> &samples,
                                  double flops_per_layer)
{
    std::vector<double> values;
    values.reserve(samples.size());
    for (const LayerSample &sample : samples)
        values.push_back((double)sample.compute_ms);

    std::sort(values.begin(), values.end());
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double avg = sum / (double)values.size();
    double p50 = values[values.size() / 2];
    double p95 = values[(size_t)((values.size() - 1) * 0.95)];
    double tflops = (flops_per_layer / 1.0e12) / (avg / 1000.0);

    std::printf("%s samples=%zu avg_ms=%.3f p50_ms=%.3f "
                "p95_ms=%.3f min_ms=%.3f max_ms=%.3f avg_TFLOPs=%.6f\n",
                label,
                values.size(),
                avg,
                p50,
                p95,
                values.front(),
                values.back(),
                tflops);
}

__global__ void moe_layer_matmul_kernel(float **weights,
                                        const float *input,
                                        float *output,
                                        int layer,
                                        int experts_per_layer,
                                        int m,
                                        int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = experts_per_layer * m;
    if (idx >= total)
        return;

    int expert = idx / m;
    int row = idx - expert * m;
    const float *w = weights[layer * experts_per_layer + expert] + (size_t)row * (size_t)k;
    float sum = 0.0f;

    for (int col = 0; col < k; ++col)
        sum += w[col] * input[col];

    output[idx] = sum;
}

__global__ void validate_layer_outputs_kernel(const float *output,
                                              int layer,
                                              int experts_per_layer,
                                              int m,
                                              int k,
                                              unsigned int *errors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = experts_per_layer * m;
    if (idx >= total)
        return;

    int expert = idx / m;
    int global_expert = layer * experts_per_layer + expert;
    float expected = (float)(global_expert + 1) * (float)k;
    float diff = fabsf(output[idx] - expected);

    if (diff > 0.01f)
        atomicAdd(errors, 1u);
}

static void prefetch_offloaded_layer_to_cpu(const std::vector<float *> &weights,
                                            int layer,
                                            int experts_per_layer,
                                            int resident_experts,
                                            int offloaded_experts,
                                            size_t expert_bytes)
{
    for (int local = 0; local < offloaded_experts; ++local) {
        int expert = resident_experts + local;
        int global = layer * experts_per_layer + expert;
        CUDA_CHECK(cudaMemPrefetchAsync(weights[global], expert_bytes, cudaCpuDeviceId));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

static LayerSample run_native_layer(Args args,
                                    const std::vector<float *> &weights,
                                    float **d_weights,
                                    const float *input,
                                    float *output,
                                    unsigned int *errors,
                                    int layer,
                                    int experts_per_layer,
                                    size_t expert_bytes,
                                    bool prefetch_back_to_cpu,
                                    unsigned long long *checked_outputs_out)
{
    LayerSample sample;
    int threads = 256;
    int output_count = experts_per_layer * args.m;
    int grid = (output_count + threads - 1) / threads;
    cudaEvent_t compute_start, compute_end, layer_done;
    double layer_start;

    CUDA_CHECK(cudaEventCreate(&compute_start));
    CUDA_CHECK(cudaEventCreate(&compute_end));
    CUDA_CHECK(cudaEventCreateWithFlags(&layer_done, cudaEventDisableTiming));

    layer_start = now_ms();
    prefetch_offloaded_layer_to_cpu(weights,
                                    layer,
                                    experts_per_layer,
                                    args.resident_experts,
                                    args.offloaded_experts,
                                    expert_bytes);

    double prefetch_start = now_ms();
    for (int local = 0; local < args.offloaded_experts; ++local) {
        int expert = args.resident_experts + local;
        int global = layer * experts_per_layer + expert;
        CUDA_CHECK(cudaMemPrefetchAsync(weights[global], expert_bytes, args.device));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    sample.prefetch_ms = now_ms() - prefetch_start;

    CUDA_CHECK(cudaEventRecord(compute_start));
    moe_layer_matmul_kernel<<<grid, threads>>>(d_weights,
                                               input,
                                               output,
                                               layer,
                                               experts_per_layer,
                                               args.m,
                                               args.k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(compute_end));
    CUDA_CHECK(cudaEventRecord(layer_done));
    CUDA_CHECK(cudaEventSynchronize(layer_done));
    CUDA_CHECK(cudaEventElapsedTime(&sample.compute_ms, compute_start, compute_end));

    double cleanup_start = now_ms();
    if (prefetch_back_to_cpu) {
        for (int local = 0; local < args.offloaded_experts; ++local) {
            int expert = args.resident_experts + local;
            int global = layer * experts_per_layer + expert;
            CUDA_CHECK(cudaMemPrefetchAsync(weights[global], expert_bytes, cudaCpuDeviceId));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    sample.invalidate_ms = now_ms() - cleanup_start;

    double validate_start = now_ms();
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned int)));
    validate_layer_outputs_kernel<<<grid, threads>>>(output,
                                                     layer,
                                                     experts_per_layer,
                                                     args.m,
                                                     args.k,
                                                     errors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    sample.validate_ms = now_ms() - validate_start;
    if (*errors != 0) {
        std::fprintf(stderr, "native validation failed layer=%d errors=%u\n", layer, *errors);
        std::exit(2);
    }

    if (checked_outputs_out)
        *checked_outputs_out += (unsigned long long)output_count;
    sample.total_ms = now_ms() - layer_start;

    CUDA_CHECK(cudaEventDestroy(compute_start));
    CUDA_CHECK(cudaEventDestroy(compute_end));
    CUDA_CHECK(cudaEventDestroy(layer_done));
    return sample;
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    const int experts_per_layer = args.resident_experts + args.offloaded_experts;
    const int total_experts = args.layers * experts_per_layer;
    const size_t weight_elements = (size_t)args.m * (size_t)args.k;
    const size_t expert_bytes = weight_elements * sizeof(float);
    const size_t slot_bytes = args.slot_bytes == 0 ? expert_bytes : args.slot_bytes;
    const size_t offloaded_bytes_per_layer = (size_t)args.offloaded_experts * expert_bytes;
    const unsigned long long region_base_id = 0xABC300000001ull;

    if (slot_bytes < expert_bytes) {
        std::fprintf(stderr, "slot_bytes=%zu is smaller than expert_bytes=%zu\n", slot_bytes, expert_bytes);
        return 2;
    }

    CUDA_CHECK(cudaSetDevice(args.device));
    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
    CUuuid cuda_uuid{};
    CU_CHECK(cuDeviceGetUuid(&cuda_uuid, cu_device));

    std::printf("expert_layer_bench_begin layers=%d resident_experts=%d offloaded_experts=%d "
                "experts_per_layer=%d m=%d k=%d expert_bytes=%zu slot_count=%d "
                "slot_bytes=%zu gpu=%s\n",
                args.layers,
                args.resident_experts,
                args.offloaded_experts,
                experts_per_layer,
                args.m,
                args.k,
                expert_bytes,
                args.offloaded_experts,
                slot_bytes,
                prop.name);

    std::vector<float *> weights(total_experts, nullptr);
    std::vector<unsigned long long> region_ids(total_experts, 0);

    double alloc_start = now_ms();
    for (int global = 0; global < total_experts; ++global) {
        CUDA_CHECK(cudaMallocManaged(&weights[global], expert_bytes));
        std::fill(weights[global], weights[global] + weight_elements, (float)(global + 1));
        region_ids[global] = region_base_id + (unsigned long long)global;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double alloc_ms = now_ms() - alloc_start;
    std::printf("expert_alloc_and_cpu_fill experts=%d total_bytes=%zu ms=%.3f\n",
                total_experts,
                (size_t)total_experts * expert_bytes,
                alloc_ms);

    double resident_start = now_ms();
    for (int layer = 0; layer < args.layers; ++layer) {
        for (int expert = 0; expert < args.resident_experts; ++expert) {
            int global = layer * experts_per_layer + expert;
            CUDA_CHECK(cudaMemPrefetchAsync(weights[global], expert_bytes, args.device));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double resident_ms = now_ms() - resident_start;
    std::printf("resident_prefetch_init experts=%d bytes=%zu ms=%.3f avg_GiBps=%.3f\n",
                args.layers * args.resident_experts,
                (size_t)args.layers * (size_t)args.resident_experts * expert_bytes,
                resident_ms,
                (((double)args.layers * (double)args.resident_experts * (double)expert_bytes) /
                 (1024.0 * 1024.0 * 1024.0)) / (resident_ms / 1000.0));

    float **d_weights = nullptr;
    float *input = nullptr;
    float *output = nullptr;
    unsigned int *errors = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weights, (size_t)total_experts * sizeof(float *)));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), (size_t)total_experts * sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocManaged(&input, (size_t)args.k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, (size_t)experts_per_layer * (size_t)args.m * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&errors, sizeof(unsigned int)));
    std::fill(input, input + args.k, 1.0f);
    CUDA_CHECK(cudaMemPrefetchAsync(input, (size_t)args.k * sizeof(float), args.device));
    CUDA_CHECK(cudaDeviceSynchronize());

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
    pool.count = (unsigned int)args.offloaded_experts;
    pool.gpu_uuid = gpu_uuid;
    for (int slot = 0; slot < args.offloaded_experts; ++slot) {
        pool.entries[slot].pool_id = args.pool_id;
        pool.entries[slot].slot_id = (unsigned int)slot;
        pool.entries[slot].slot_bytes = slot_bytes;
    }
    IOCTL_CHECK(fd, kExpertReservedPoolIoctl, pool, "reserved_pool_prealloc");
    std::printf("reserved_pool_prealloc slots=%u success=%u failure=%u backing_bytes=%llu logical_bytes=%llu\n",
                pool.count,
                pool.success_count,
                pool.failure_count,
                pool.reserved_backing_bytes,
                pool.logical_slot_bytes);

    int registered = 0;
    for (int layer = 0; layer < args.layers; ++layer) {
        for (int local = 0; local < args.offloaded_experts; ++local) {
            int expert = args.resident_experts + local;
            int global = layer * experts_per_layer + expert;
            PhaseFRegisterRegionParams reg{};
            reg.region.region_id = region_ids[global];
            reg.region.base = reinterpret_cast<unsigned long long>(weights[global]);
            reg.region.length = expert_bytes;
            reg.region.pool = kPoolWeight;
            reg.region.layer_id = (unsigned int)layer;
            reg.region.expert_id = (unsigned int)expert;
            reg.region.role = kRoleW13;
            reg.region.flags = kRegionFlagPrefetchable |
                               kRegionFlagGpuReadOnly |
                               kRegionFlagStreamingSubpool |
                               kRegionFlagCpuOffloadedReserved;
            IOCTL_CHECK(fd, kUvmPhaseFRegisterRegionIoctl, reg, "register_offloaded_region");
            ++registered;
        }
    }
    std::printf("offloaded_regions_registered count=%d strict_flag=1\n", registered);

    cudaEvent_t compute_start, compute_end, layer_done;
    CUDA_CHECK(cudaEventCreate(&compute_start));
    CUDA_CHECK(cudaEventCreate(&compute_end));
    CUDA_CHECK(cudaEventCreateWithFlags(&layer_done, cudaEventDisableTiming));

    std::vector<LayerSample> samples;
    samples.reserve(args.layers);
    unsigned long long checked_outputs = 0;
    unsigned int total_errors = 0;
    int threads = 256;
    int output_count = experts_per_layer * args.m;
    int grid = (output_count + threads - 1) / threads;

    for (int layer = 0; layer < args.layers; ++layer) {
        LayerSample sample;
        double layer_start = now_ms();

        ExpertReservedPrefetchParams prefetch{};
        prefetch.mode = kReservedPrefetchModeReservedPrefetch;
        prefetch.count = (unsigned int)args.offloaded_experts;
        prefetch.gpu_uuid = gpu_uuid;
        prefetch.async = 1;
        for (int local = 0; local < args.offloaded_experts; ++local) {
            int expert = args.resident_experts + local;
            int global = layer * experts_per_layer + expert;
            prefetch.entries[local].region_id = region_ids[global];
            prefetch.entries[local].pool_id = args.pool_id;
            prefetch.entries[local].slot_id = (unsigned int)local;
            prefetch.entries[local].flags = kStrictFootprint;
            prefetch.entries[local].bytes_override = expert_bytes;
        }

        double prefetch_start = now_ms();
        IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, prefetch, "reserved_prefetch_layer");
        sample.prefetch_ms = now_ms() - prefetch_start;
        if (prefetch.success_count != (unsigned int)args.offloaded_experts ||
            prefetch.failure_count != 0 ||
            prefetch.accepted_bytes != offloaded_bytes_per_layer ||
            prefetch.assigned_slots != (unsigned long long)args.offloaded_experts ||
            prefetch.ordinary_resident_bytes != 0 ||
            prefetch.pool_external_alloc_count != 0 ||
            prefetch.footprint_contract_broken != 0) {
            std::fprintf(stderr,
                         "bad prefetch layer=%d success=%u failure=%u accepted=%llu "
                         "assigned=%llu ordinary=%llu external=%llu broken=%llu\n",
                         layer,
                         prefetch.success_count,
                         prefetch.failure_count,
                         prefetch.accepted_bytes,
                         prefetch.assigned_slots,
                         prefetch.ordinary_resident_bytes,
                         prefetch.pool_external_alloc_count,
                         prefetch.footprint_contract_broken);
            std::exit(1);
        }

        CUDA_CHECK(cudaEventRecord(compute_start));
        moe_layer_matmul_kernel<<<grid, threads>>>(d_weights,
                                                   input,
                                                   output,
                                                   layer,
                                                   experts_per_layer,
                                                   args.m,
                                                   args.k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(compute_end));
        CUDA_CHECK(cudaEventRecord(layer_done));
        CUDA_CHECK(cudaEventSynchronize(layer_done));
        CUDA_CHECK(cudaEventElapsedTime(&sample.compute_ms, compute_start, compute_end));

        double invalidate_start = now_ms();
        ExpertReservedPrefetchParams invalidate{};
        invalidate.mode = kReservedPrefetchModeInvalidate;
        invalidate.count = (unsigned int)args.offloaded_experts;
        invalidate.gpu_uuid = gpu_uuid;
        for (int local = 0; local < args.offloaded_experts; ++local) {
            invalidate.entries[local].pool_id = args.pool_id;
            invalidate.entries[local].slot_id = (unsigned int)local;
            invalidate.entries[local].flags = kStrictFootprint;
        }
        IOCTL_CHECK(fd, kExpertReservedPrefetchIoctl, invalidate, "reserved_prefetch_invalidate_layer");
        sample.invalidate_ms = now_ms() - invalidate_start;
        if (invalidate.success_count != (unsigned int)args.offloaded_experts ||
            invalidate.failure_count != 0 ||
            invalidate.invalidated_slots != (unsigned long long)args.offloaded_experts) {
            std::fprintf(stderr,
                         "bad invalidate layer=%d success=%u failure=%u invalidated=%llu\n",
                         layer,
                         invalidate.success_count,
                         invalidate.failure_count,
                         invalidate.invalidated_slots);
            std::exit(1);
        }

        double validate_start = now_ms();
        CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned int)));
        validate_layer_outputs_kernel<<<grid, threads>>>(output,
                                                         layer,
                                                         experts_per_layer,
                                                         args.m,
                                                         args.k,
                                                         errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        sample.validate_ms = now_ms() - validate_start;

        if (*errors != 0) {
            std::fprintf(stderr, "validation failed layer=%d errors=%u\n", layer, *errors);
            total_errors += *errors;
            std::exit(2);
        }

        checked_outputs += (unsigned long long)output_count;
        sample.total_ms = now_ms() - layer_start;
        samples.push_back(sample);

        if (args.verbose_layers) {
            std::printf("layer_result layer=%d prefetch_ms=%.3f compute_ms=%.3f "
                        "invalidate_ms=%.3f validate_ms=%.3f total_ms=%.3f errors=%u\n",
                        layer,
                        sample.prefetch_ms,
                        sample.compute_ms,
                        sample.invalidate_ms,
                        sample.validate_ms,
                        sample.total_ms,
                        *errors);
        }
    }

    print_wall_summary("reserved_prefetch_layers", samples, &LayerSample::prefetch_ms, offloaded_bytes_per_layer);
    print_compute_summary("reserved_moe_matmul_compute",
                          samples,
                          (double)experts_per_layer * (double)args.m * (double)args.k * 2.0);
    print_wall_summary("reserved_invalidate_layers", samples, &LayerSample::invalidate_ms, 0);
    print_wall_summary("reserved_validate_layers", samples, &LayerSample::validate_ms, 0);
    print_wall_summary("reserved_e2e_layer", samples, &LayerSample::total_ms, offloaded_bytes_per_layer);
    std::printf("reserved_correctness layers=%d checked_outputs=%llu kernel_errors=%u\n",
                args.layers,
                checked_outputs,
                total_errors);

    for (int layer = 0; layer < args.layers; ++layer) {
        for (int local = 0; local < args.offloaded_experts; ++local) {
            int expert = args.resident_experts + local;
            int global = layer * experts_per_layer + expert;
            PhaseFUnregisterRegionParams unreg{};
            unreg.region_id = region_ids[global];
            (void)ioctl(fd, kUvmPhaseFUnregisterRegionIoctl, &unreg);
        }
    }

    if (args.compare_native) {
        std::vector<LayerSample> native_gpu_only_samples;
        std::vector<LayerSample> native_cpu_cleanup_samples;
        unsigned long long native_gpu_only_checked = 0;
        unsigned long long native_cpu_cleanup_checked = 0;

        native_gpu_only_samples.reserve(args.layers);
        native_cpu_cleanup_samples.reserve(args.layers);

        std::printf("native_baseline_begin mode=gpu_only_cleanup_none\n");
        for (int layer = 0; layer < args.layers; ++layer) {
            LayerSample sample = run_native_layer(args,
                                                  weights,
                                                  d_weights,
                                                  input,
                                                  output,
                                                  errors,
                                                  layer,
                                                  experts_per_layer,
                                                  expert_bytes,
                                                  false,
                                                  &native_gpu_only_checked);
            native_gpu_only_samples.push_back(sample);
        }
        print_wall_summary("native_prefetch_gpu_layers",
                           native_gpu_only_samples,
                           &LayerSample::prefetch_ms,
                           offloaded_bytes_per_layer);
        print_compute_summary("native_gpu_moe_matmul_compute",
                              native_gpu_only_samples,
                              (double)experts_per_layer * (double)args.m * (double)args.k * 2.0);
        print_wall_summary("native_gpu_cleanup_layers",
                           native_gpu_only_samples,
                           &LayerSample::invalidate_ms,
                           0);
        print_wall_summary("native_gpu_e2e_layer",
                           native_gpu_only_samples,
                           &LayerSample::total_ms,
                           offloaded_bytes_per_layer);
        std::printf("native_gpu_correctness layers=%d checked_outputs=%llu kernel_errors=0\n",
                    args.layers,
                    native_gpu_only_checked);

        std::printf("native_baseline_begin mode=prefetch_back_to_cpu_cleanup\n");
        for (int layer = 0; layer < args.layers; ++layer) {
            LayerSample sample = run_native_layer(args,
                                                  weights,
                                                  d_weights,
                                                  input,
                                                  output,
                                                  errors,
                                                  layer,
                                                  experts_per_layer,
                                                  expert_bytes,
                                                  true,
                                                  &native_cpu_cleanup_checked);
            native_cpu_cleanup_samples.push_back(sample);
        }
        print_wall_summary("native_prefetch_cpu_cleanup_layers",
                           native_cpu_cleanup_samples,
                           &LayerSample::prefetch_ms,
                           offloaded_bytes_per_layer);
        print_compute_summary("native_cpu_cleanup_moe_matmul_compute",
                              native_cpu_cleanup_samples,
                              (double)experts_per_layer * (double)args.m * (double)args.k * 2.0);
        print_wall_summary("native_prefetch_back_cpu_layers",
                           native_cpu_cleanup_samples,
                           &LayerSample::invalidate_ms,
                           offloaded_bytes_per_layer);
        print_wall_summary("native_cpu_cleanup_e2e_layer",
                           native_cpu_cleanup_samples,
                           &LayerSample::total_ms,
                           offloaded_bytes_per_layer * 2);
        std::printf("native_cpu_cleanup_correctness layers=%d checked_outputs=%llu kernel_errors=0\n",
                    args.layers,
                    native_cpu_cleanup_checked);

        std::printf("comparison reserved_prefetch_avg_ms=%.3f native_gpu_prefetch_avg_ms=%.3f "
                    "native_cpu_cleanup_prefetch_avg_ms=%.3f reserved_vs_native_gpu_speedup=%.3f "
                    "reserved_vs_native_cpu_cleanup_speedup=%.3f\n",
                    sample_average(samples, &LayerSample::prefetch_ms),
                    sample_average(native_gpu_only_samples, &LayerSample::prefetch_ms),
                    sample_average(native_cpu_cleanup_samples, &LayerSample::prefetch_ms),
                    sample_average(native_gpu_only_samples, &LayerSample::prefetch_ms) /
                        sample_average(samples, &LayerSample::prefetch_ms),
                    sample_average(native_cpu_cleanup_samples, &LayerSample::prefetch_ms) /
                        sample_average(samples, &LayerSample::prefetch_ms));

        std::printf("comparison_e2e reserved_avg_ms=%.3f native_gpu_avg_ms=%.3f "
                    "native_cpu_cleanup_avg_ms=%.3f\n",
                    sample_average(samples, &LayerSample::total_ms),
                    sample_average(native_gpu_only_samples, &LayerSample::total_ms),
                    sample_average(native_cpu_cleanup_samples, &LayerSample::total_ms));
    }

    ExpertReservedPoolParams reset_after{};
    reset_after.mode = kReservedPoolModeReset;
    reset_after.gpu_uuid = gpu_uuid;
    (void)ioctl(fd, kExpertReservedPoolIoctl, &reset_after);

    CUDA_CHECK(cudaEventDestroy(compute_start));
    CUDA_CHECK(cudaEventDestroy(compute_end));
    CUDA_CHECK(cudaEventDestroy(layer_done));
    close(fd);

    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(errors));
    for (float *ptr : weights)
        CUDA_CHECK(cudaFree(ptr));

    return total_errors == 0 ? 0 : 2;
}
