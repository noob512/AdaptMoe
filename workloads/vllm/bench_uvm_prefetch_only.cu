#include <algorithm>
#include <chrono>
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

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t err__ = (expr);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                         \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

struct Args {
    int device = 0;
    int regions = 10;
    size_t mib_per_region = 2;
    int repeats = 20;
    int warmups = 3;
    bool single_alloc = false;
    bool read_mostly = false;
    bool initial_touch_before_advice = false;
    bool iter_touch = true;
    bool tag_expert_regions = false;
    bool reset_uvm_stats_before_prefetch = false;
    const char *uvm_stats_reset_path = "/proc/driver/nvidia-uvm/latency_stats_reset";
    const char *uvm_stats_dump_path = nullptr;
    const char *uvm_stats_path = "/proc/driver/nvidia-uvm/latency_stats";
};

static Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            args.device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--regions") == 0 && i + 1 < argc)
            args.regions = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mib-per-region") == 0 && i + 1 < argc)
            args.mib_per_region = (size_t)std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            args.repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmups") == 0 && i + 1 < argc)
            args.warmups = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--single-alloc") == 0)
            args.single_alloc = true;
        else if (std::strcmp(argv[i], "--read-mostly") == 0)
            args.read_mostly = true;
        else if (std::strcmp(argv[i], "--initial-touch-before-advice") == 0)
            args.initial_touch_before_advice = true;
        else if (std::strcmp(argv[i], "--no-iter-touch") == 0)
            args.iter_touch = false;
        else if (std::strcmp(argv[i], "--tag-expert-regions") == 0)
            args.tag_expert_regions = true;
        else if (std::strcmp(argv[i], "--reset-uvm-stats-before-prefetch") == 0)
            args.reset_uvm_stats_before_prefetch = true;
        else if (std::strcmp(argv[i], "--uvm-stats-reset-path") == 0 && i + 1 < argc)
            args.uvm_stats_reset_path = argv[++i];
        else if (std::strcmp(argv[i], "--dump-uvm-stats") == 0 && i + 1 < argc)
            args.uvm_stats_dump_path = argv[++i];
        else if (std::strcmp(argv[i], "--uvm-stats-path") == 0 && i + 1 < argc)
            args.uvm_stats_path = argv[++i];
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--regions N] [--mib-per-region N] "
                         "[--repeats N] [--warmups N] [--single-alloc] [--read-mostly] "
                         "[--initial-touch-before-advice] [--no-iter-touch] [--tag-expert-regions] "
                         "[--reset-uvm-stats-before-prefetch] [--uvm-stats-reset-path PATH] "
                         "[--dump-uvm-stats PATH] [--uvm-stats-path PATH]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    return args;
}

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

static std::vector<TaggedRegion> tag_expert_regions(const std::vector<unsigned char *> &regions,
                                                    size_t bytes_per_region)
{
    std::vector<TaggedRegion> tagged;
    int fd = open_registration_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr, "failed to open initialized /dev/nvidia-uvm for region tagging\n");
        std::exit(1);
    }

    const unsigned long long base_id = 0xEF00000000ull;
    tagged.reserve(regions.size());

    for (size_t i = 0; i < regions.size(); ++i) {
        unsigned long long region_id = base_id + (unsigned long long)i;
        unsigned int role = (i % 2) == 0 ? kRoleW13 : kRoleW2;
        unsigned int expert_id = (unsigned int)(i / 2);

        if (!register_region(fd, region_id, regions[i], bytes_per_region, expert_id, role)) {
            std::fprintf(stderr, "failed to register expert region %zu\n", i);
            for (const TaggedRegion &region : tagged)
                unregister_region(fd, region.id);
            close(fd);
            std::exit(1);
        }

        tagged.push_back({region_id});
    }

    close(fd);
    std::printf("tagged_expert_regions count=%zu pool=WEIGHT roles=alternating_W13_W2\n", tagged.size());
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

static void touch_cpu(unsigned char *ptr, size_t bytes, unsigned seed)
{
    for (size_t i = 0; i < bytes; i += 4096)
        ptr[i] = (unsigned char)(seed + i);
    if (bytes)
        ptr[bytes - 1] = (unsigned char)(seed ^ 0x5a);
}

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void reset_uvm_stats(const char *path)
{
    FILE *f = std::fopen(path, "w");
    if (!f) {
        std::perror(path);
        std::exit(1);
    }

    std::fputs("1\n", f);
    std::fclose(f);
}

static void dump_uvm_stats(const char *stats_path, const char *dump_path)
{
    FILE *in = std::fopen(stats_path, "r");
    if (!in) {
        std::perror(stats_path);
        std::exit(1);
    }

    FILE *out = std::fopen(dump_path, "w");
    if (!out) {
        std::perror(dump_path);
        std::fclose(in);
        std::exit(1);
    }

    char buffer[4096];
    size_t n;
    while ((n = std::fread(buffer, 1, sizeof(buffer), in)) > 0)
        std::fwrite(buffer, 1, n, out);

    std::fclose(out);
    std::fclose(in);
}

static float elapsed(cudaEvent_t start, cudaEvent_t end)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

struct PrefetchSample {
    float event_ms;
    double submit_ms;
    double sync_ms;
    double wall_ms;
};

static PrefetchSample prefetch_regions(const std::vector<unsigned char *> &ptrs,
                                       size_t bytes,
                                       int device,
                                       cudaStream_t stream)
{
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    const double wall_start = now_ms();
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (unsigned char *ptr : ptrs)
        CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    const double submit_ms = now_ms() - wall_start;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double wall_ms = now_ms() - wall_start;
    const double sync_ms = wall_ms - submit_ms;
    const float event_ms = elapsed(start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return {event_ms, submit_ms, sync_ms, wall_ms};
}

template <typename T>
static double percentile(std::vector<T> values, double p)
{
    std::sort(values.begin(), values.end());
    size_t idx = (size_t)((values.size() - 1) * p);
    return values[idx];
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    const size_t bytes_per_region = args.mib_per_region * 1024ull * 1024ull;
    const size_t total_bytes = bytes_per_region * (size_t)args.regions;
    std::vector<void *> allocations;
    std::vector<unsigned char *> regions;

    if (args.single_alloc) {
        void *raw = nullptr;
        CUDA_CHECK(cudaMallocManaged(&raw, total_bytes, cudaMemAttachGlobal));
        allocations.push_back(raw);
        for (int i = 0; i < args.regions; ++i)
            regions.push_back(static_cast<unsigned char *>(raw) + (size_t)i * bytes_per_region);
    }
    else {
        for (int i = 0; i < args.regions; ++i) {
            void *raw = nullptr;
            CUDA_CHECK(cudaMallocManaged(&raw, bytes_per_region, cudaMemAttachGlobal));
            allocations.push_back(raw);
            regions.push_back(static_cast<unsigned char *>(raw));
        }
    }

    if (args.initial_touch_before_advice) {
        for (int i = 0; i < args.regions; ++i)
            touch_cpu(regions[i], bytes_per_region, (unsigned)i);
    }

    if (args.read_mostly) {
        for (unsigned char *ptr : regions)
            CUDA_CHECK(cudaMemAdvise(ptr, bytes_per_region, cudaMemAdviseSetReadMostly, args.device));
    }

    std::vector<TaggedRegion> tagged_regions;
    if (args.tag_expert_regions)
        tagged_regions = tag_expert_regions(regions, bytes_per_region);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<float> event_samples;
    std::vector<double> submit_samples;
    std::vector<double> sync_samples;
    std::vector<double> wall_samples;
    const int total_iters = args.warmups + args.repeats;
    for (int iter = 0; iter < total_iters; ++iter) {
        for (int i = 0; i < args.regions; ++i)
            CUDA_CHECK(cudaMemPrefetchAsync(regions[i], bytes_per_region, cudaCpuDeviceId, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (args.iter_touch) {
            for (int i = 0; i < args.regions; ++i)
                touch_cpu(regions[i], bytes_per_region, (unsigned)(iter * 17 + i));
        }

        if (args.reset_uvm_stats_before_prefetch)
            reset_uvm_stats(args.uvm_stats_reset_path);

        PrefetchSample sample = prefetch_regions(regions, bytes_per_region, args.device, stream);
        if (iter >= args.warmups) {
            event_samples.push_back(sample.event_ms);
            submit_samples.push_back(sample.submit_ms);
            sync_samples.push_back(sample.sync_ms);
            wall_samples.push_back(sample.wall_ms);
            std::printf("sample iter=%d event_ms=%.3f submit_ms=%.3f sync_ms=%.3f wall_ms=%.3f "
                        "event_gib_s=%.3f wall_gib_s=%.3f\n",
                        iter - args.warmups,
                        sample.event_ms,
                        sample.submit_ms,
                        sample.sync_ms,
                        sample.wall_ms,
                        (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (sample.event_ms / 1000.0),
                        (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (sample.wall_ms / 1000.0));
        }
    }

    const double event_p50 = percentile(event_samples, 0.50);
    const double submit_p50 = percentile(submit_samples, 0.50);
    const double sync_p50 = percentile(sync_samples, 0.50);
    const double wall_p50 = percentile(wall_samples, 0.50);
    const double event_mean = std::accumulate(event_samples.begin(), event_samples.end(), 0.0) / event_samples.size();
    const double submit_mean = std::accumulate(submit_samples.begin(), submit_samples.end(), 0.0) / submit_samples.size();
    const double sync_mean = std::accumulate(sync_samples.begin(), sync_samples.end(), 0.0) / sync_samples.size();
    const double wall_mean = std::accumulate(wall_samples.begin(), wall_samples.end(), 0.0) / wall_samples.size();
    std::printf("summary device=%s regions=%d mib_per_region=%zu total_mib=%.2f single_alloc=%d read_mostly=%d "
                "initial_touch_before_advice=%d iter_touch=%d tag_expert_regions=%d "
                "event_p50_ms=%.3f event_mean_ms=%.3f event_p90_ms=%.3f "
                "submit_p50_ms=%.3f submit_mean_ms=%.3f sync_p50_ms=%.3f sync_mean_ms=%.3f "
                "wall_p50_ms=%.3f wall_mean_ms=%.3f wall_p90_ms=%.3f "
                "event_p50_gib_s=%.3f wall_p50_gib_s=%.3f\n",
                prop.name,
                args.regions,
                args.mib_per_region,
                (double)total_bytes / (1024.0 * 1024.0),
                args.single_alloc ? 1 : 0,
                args.read_mostly ? 1 : 0,
                args.initial_touch_before_advice ? 1 : 0,
                args.iter_touch ? 1 : 0,
                args.tag_expert_regions ? 1 : 0,
                event_p50,
                event_mean,
                percentile(event_samples, 0.90),
                submit_p50,
                submit_mean,
                sync_p50,
                sync_mean,
                wall_p50,
                wall_mean,
                percentile(wall_samples, 0.90),
                (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (event_p50 / 1000.0),
                (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / (wall_p50 / 1000.0));

    if (args.uvm_stats_dump_path)
        dump_uvm_stats(args.uvm_stats_path, args.uvm_stats_dump_path);

    CUDA_CHECK(cudaStreamDestroy(stream));
    untag_expert_regions(tagged_regions);
    for (void *ptr : allocations)
        CUDA_CHECK(cudaFree(ptr));

    return 0;
}
