#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

#include "uvm_linux_ioctl.h"

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr,                                             \
                         "CUDA error %s:%d: %s\n",                         \
                         __FILE__,                                           \
                         __LINE__,                                           \
                         cudaGetErrorString(err__));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

struct Options {
    std::string mechanism = "uvm_pure_remote";
    size_t pages = 8192;
    int passes = 2;
    int ops_per_page = 128;
    int repeat = 0;
    int device = 0;
    int threads = 256;
    int blocks_per_sm = 4;
    bool print_header = true;
};

namespace {

struct PrepareRemoteResult {
    double wall_ms = 0.0;
    uint64_t mapped_bytes = 0;
};

} // namespace

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static size_t parse_size(const char *s)
{
    char *end = nullptr;
    unsigned long long value = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "Invalid integer: %s\n", s);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(value);
}

static void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "Usage: %s --mechanism NAME [--pages N] [--passes N]\n"
                 "          [--ops-per-page N] [--repeat N] [--device N]\n"
                 "          [--threads N] [--blocks-per-sm N] [--no-header]\n"
                 "\n"
                 "Mechanisms:\n"
                 "  uvm_pure_remote    Managed, CPU preferred, GPU AccessedBy\n"
                 "  uva_pinned_remote  vLLM-style pinned CPU memory + UVA CUDA view\n"
                 "  readmostly_prepare_remote Managed, ReadMostly; manual driver remote READ prepare\n"
                 "  readmostly_prepare_remote_bg Managed, ReadMostly; manual remote READ prepare + bg read-dup promotion\n"
                 "  readmostly_prepare_remote_bg_after_first Managed, ReadMostly; prepare remote + defer bg promotion until pass0 completes\n"
                 "  hybrid_remote_first_bg Managed, ReadMostly; driver remote-first + bg promotion\n"
                 "  hybrid_remote_first_only Managed, ReadMostly; driver remote-first only\n"
                 "  readonly_dup       Managed, CPU resident, ReadMostly read-dup\n"
                 "  direct_migrate     Managed, CPU resident, normal on-demand migration\n"
                 "  counter_migrate    Managed, CPU preferred + AccessedBy; high ops trigger counters\n",
                 argv0);
}

static bool valid_mechanism(const std::string &mechanism)
{
    return mechanism == "uvm_pure_remote" ||
           mechanism == "uva_pinned_remote" ||
           mechanism == "readmostly_prepare_remote" ||
           mechanism == "readmostly_prepare_remote_bg" ||
           mechanism == "readmostly_prepare_remote_bg_after_first" ||
           mechanism == "hybrid_remote_first_bg" ||
           mechanism == "hybrid_remote_first_only" ||
           mechanism == "readonly_dup" ||
           mechanism == "direct_migrate" ||
           mechanism == "counter_migrate";
}

static Options parse_args(int argc, char **argv)
{
    Options opt;

    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--mechanism") == 0)
            opt.mechanism = need_value(argv[i]);
        else if (std::strcmp(argv[i], "--pages") == 0)
            opt.pages = parse_size(need_value(argv[i]));
        else if (std::strcmp(argv[i], "--passes") == 0)
            opt.passes = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--ops-per-page") == 0)
            opt.ops_per_page = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--repeat") == 0)
            opt.repeat = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--device") == 0)
            opt.device = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--threads") == 0)
            opt.threads = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--blocks-per-sm") == 0)
            opt.blocks_per_sm = static_cast<int>(parse_size(need_value(argv[i])));
        else if (std::strcmp(argv[i], "--no-header") == 0)
            opt.print_header = false;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (!valid_mechanism(opt.mechanism)) {
        std::fprintf(stderr, "Unknown mechanism: %s\n", opt.mechanism.c_str());
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    if (opt.pages == 0 || opt.passes <= 0 || opt.ops_per_page <= 0) {
        std::fprintf(stderr, "pages, passes, and ops_per_page must be positive\n");
        std::exit(EXIT_FAILURE);
    }

    return opt;
}

__global__ void readonly_probe_kernel(const uint64_t *data,
                                      uint64_t *partials,
                                      size_t pages,
                                      size_t elems_per_page,
                                      int ops_per_page,
                                      uint64_t salt)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid = gridDim.x * blockDim.x;
    const volatile uint64_t *volatile_data = data;
    uint64_t acc = salt + tid;

    for (size_t page = tid; page < pages; page += grid) {
        const size_t base = page * elems_per_page;
        for (int op = 0; op < ops_per_page; ++op) {
            const size_t offset = (static_cast<size_t>(op) * 131u + tid) & (elems_per_page - 1u);
            acc += volatile_data[base + offset];
        }
    }

    partials[tid] = acc;
}

__global__ void warmup_kernel(uint64_t *partials, size_t count)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count)
        partials[tid] = tid;
}

static void init_host_data(uint64_t *data, size_t elems)
{
    for (size_t i = 0; i < elems; ++i)
        data[i] = 0x9e3779b97f4a7c15ull ^ static_cast<uint64_t>(i * 11400714819323198485ull);
}

static bool get_device_uuid(int device, NvProcessorUuid *uuid)
{
    cudaDeviceProp prop{};
    cudaError_t err;

    if (!uuid)
        return false;

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
        return false;

    static_assert(sizeof(prop.uuid.bytes) == sizeof(uuid->uuid), "Unexpected CUDA UUID size");
    std::memcpy(uuid->uuid, prop.uuid.bytes, sizeof(uuid->uuid));
    return true;
}

static bool is_nvidia_uvm_fd_target(const char *target)
{
    const char *name;
    size_t len;

    if (!target)
        return false;

    name = std::strrchr(target, '/');
    name = name ? name + 1 : target;
    len = std::strlen("nvidia-uvm");

    return std::strncmp(name, "nvidia-uvm", len) == 0 && (name[len] == '\0' || name[len] == ' ');
}

static bool try_phase_f_ioctl(int fd,
                              void *ptr,
                              size_t bytes,
                              int device,
                              uint32_t action,
                              PrepareRemoteResult *result,
                              int *saved_errno,
                              uint32_t *rm_status,
                              uint32_t *range_status);

static bool try_existing_uvm_fds(void *ptr,
                                 size_t bytes,
                                 int device,
                                 uint32_t action,
                                 PrepareRemoteResult *result,
                                 int *saved_errno,
                                 uint32_t *rm_status,
                                 uint32_t *range_status)
{
    DIR *dir = opendir("/proc/self/fd");
    struct dirent *entry;
    const bool debug = std::getenv("UVM_PREPARE_REMOTE_DEBUG") != nullptr;

    if (!dir)
        return false;

    while ((entry = readdir(dir)) != nullptr) {
        char *end = nullptr;
        long fd_long = std::strtol(entry->d_name, &end, 10);
        char link_path[64];
        char target[PATH_MAX];
        ssize_t len;
        int dup_fd;

        if (end == entry->d_name || *end != '\0')
            continue;

        std::snprintf(link_path, sizeof(link_path), "/proc/self/fd/%ld", fd_long);
        len = readlink(link_path, target, sizeof(target) - 1);
        if (len < 0)
            continue;

        target[len] = '\0';
        if (!is_nvidia_uvm_fd_target(target))
            continue;

        if (debug)
            std::fprintf(stderr, "Trying existing UVM fd=%ld target=%s\n", fd_long, target);

        dup_fd = fcntl(static_cast<int>(fd_long), F_DUPFD_CLOEXEC, 3);
        if (dup_fd < 0)
            continue;

        if (try_phase_f_ioctl(dup_fd,
                              ptr,
                              bytes,
                              device,
                              action,
                              result,
                              saved_errno,
                              rm_status,
                              range_status)) {
            close(dup_fd);
            closedir(dir);
            return true;
        }

        close(dup_fd);
    }

    closedir(dir);
    return false;
}

static int open_initialized_uvm_fd()
{
    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);

    if (fd < 0)
        return -1;

    UVM_INITIALIZE_PARAMS init{};
    if (ioctl(fd, UVM_INITIALIZE, &init) != 0 || init.rmStatus != 0) {
        close(fd);
        return -1;
    }

    return fd;
}

static bool try_phase_f_ioctl(int fd,
                              void *ptr,
                              size_t bytes,
                              int device,
                              uint32_t action,
                              PrepareRemoteResult *result,
                              int *saved_errno,
                              uint32_t *rm_status,
                              uint32_t *range_status)
{
    UVM_PHASE_F_BATCH_EXECUTOR_PARAMS params{};

    params.action = action;
    params.count = 1;
    params.async = NV_FALSE;
    params.ranges[0].base = reinterpret_cast<uint64_t>(ptr);
    params.ranges[0].length = static_cast<uint64_t>(bytes);

    if (!get_device_uuid(device, &params.gpu_uuid)) {
        if (range_status)
            *range_status = 0xffffffffu;
        return false;
    }

    const double start_ms = now_ms();
    int ret = ioctl(fd, UVM_PHASE_F_BATCH_EXECUTOR, &params);
    if (saved_errno)
        *saved_errno = errno;
    if (rm_status)
        *rm_status = params.rmStatus;
    if (range_status)
        *range_status = params.results[0].status;

    if (ret != 0 || params.rmStatus != 0 || params.failure_count != 0 || params.results[0].status != 0)
        return false;

    if (result) {
        result->wall_ms = now_ms() - start_ms;
        result->mapped_bytes = params.results[0].resident_gpu_bytes;
    }

    return true;
}

static PrepareRemoteResult phase_f_action_or_die(void *ptr,
                                                 size_t bytes,
                                                 int device,
                                                 uint32_t action,
                                                 const char *action_name)
{
    PrepareRemoteResult result{};
    int saved_errno = 0;
    uint32_t rm_status = 0;
    uint32_t range_status = 0;

    if (try_existing_uvm_fds(ptr,
                             bytes,
                             device,
                             action,
                             &result,
                             &saved_errno,
                             &rm_status,
                             &range_status))
        return result;

    int fd = open_initialized_uvm_fd();
    if (fd >= 0) {
        if (try_phase_f_ioctl(fd,
                              ptr,
                              bytes,
                              device,
                              action,
                              &result,
                              &saved_errno,
                              &rm_status,
                              &range_status)) {
            close(fd);
            return result;
        }
        close(fd);
    }

    std::fprintf(stderr,
                 "%s failed: errno=%d rmStatus=%u rangeStatus=%u\n",
                 action_name,
                 saved_errno,
                 rm_status,
                 range_status);
    std::exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);

    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    CHECK_CUDA(cudaSetDevice(opt.device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, opt.device));
    if (opt.mechanism == "uva_pinned_remote" && !prop.canMapHostMemory) {
        std::fprintf(stderr, "Device %d cannot map host memory\n", opt.device);
        return EXIT_FAILURE;
    }

    const long os_page_size = sysconf(_SC_PAGESIZE);
    if (os_page_size <= 0 || (os_page_size & (os_page_size - 1)) != 0) {
        std::fprintf(stderr, "Unsupported OS page size: %ld\n", os_page_size);
        return EXIT_FAILURE;
    }

    const size_t page_size = static_cast<size_t>(os_page_size);
    const size_t elems_per_page = page_size / sizeof(uint64_t);
    if (elems_per_page == 0 || (elems_per_page & (elems_per_page - 1)) != 0) {
        std::fprintf(stderr, "Unsupported elems_per_page: %zu\n", elems_per_page);
        return EXIT_FAILURE;
    }

    const size_t elems = opt.pages * elems_per_page;
    const size_t bytes = elems * sizeof(uint64_t);
    const int blocks = prop.multiProcessorCount * opt.blocks_per_sm;
    const size_t partial_count = static_cast<size_t>(blocks) * opt.threads;

    uint64_t *data = nullptr;
    uint64_t *host_data = nullptr;
    uint64_t *partials = nullptr;
    PrepareRemoteResult prepare_remote{};
    const bool should_prepare_remote =
        opt.mechanism == "readmostly_prepare_remote" ||
        opt.mechanism == "readmostly_prepare_remote_bg" ||
        opt.mechanism == "readmostly_prepare_remote_bg_after_first";
    const bool should_start_bg_after_first =
        opt.mechanism == "readmostly_prepare_remote_bg_after_first";

    const double setup_start = now_ms();

    if (opt.mechanism == "uva_pinned_remote") {
        CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&host_data), bytes, cudaHostAllocMapped));
        init_host_data(host_data, elems);
        CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void **>(&data), host_data, 0));
    }
    else {
        CHECK_CUDA(cudaMallocManaged(&data, bytes));
        init_host_data(data, elems);
        CHECK_CUDA(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
        CHECK_CUDA(cudaDeviceSynchronize());

        if (opt.mechanism == "uvm_pure_remote" || opt.mechanism == "counter_migrate") {
            CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
            CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, opt.device));
        }

        if (opt.mechanism == "readonly_dup" ||
            opt.mechanism == "readmostly_prepare_remote" ||
            opt.mechanism == "readmostly_prepare_remote_bg" ||
            opt.mechanism == "readmostly_prepare_remote_bg_after_first" ||
            opt.mechanism == "hybrid_remote_first_bg" ||
            opt.mechanism == "hybrid_remote_first_only")
            CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, opt.device));

    }

    CHECK_CUDA(cudaMalloc(&partials, partial_count * sizeof(uint64_t)));
    warmup_kernel<<<blocks, opt.threads>>>(partials, partial_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (should_prepare_remote) {
        uint32_t action = UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ;
        const char *action_name = "PREPARE_REMOTE_READ";

        if (opt.mechanism == "readmostly_prepare_remote_bg") {
            action = UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG;
            action_name = "PREPARE_REMOTE_READ_BG";
        }
        else if (opt.mechanism == "readmostly_prepare_remote_bg_after_first") {
            action = UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED;
            action_name = "PREPARE_REMOTE_READ_BG_DEFERRED";
        }

        prepare_remote = phase_f_action_or_die(data, bytes, opt.device, action, action_name);
    }

    const double setup_ms = now_ms() - setup_start;

    if (opt.print_header) {
        std::printf("mechanism,repeat,pass,pages,ops_per_page,bytes,setup_ms,prepare_ms,prepare_mapped_mib,wall_ms,event_ms,mloads_s,gib_s\n");
    }

    const uint64_t logical_loads =
        static_cast<uint64_t>(opt.pages) * static_cast<uint64_t>(opt.ops_per_page);
    const double logical_gib =
        (static_cast<double>(logical_loads) * sizeof(uint64_t)) / (1024.0 * 1024.0 * 1024.0);

    for (int pass = 0; pass < opt.passes; ++pass) {
        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        const double wall_start = now_ms();
        CHECK_CUDA(cudaEventRecord(ev_start));
        readonly_probe_kernel<<<blocks, opt.threads>>>(
            data,
            partials,
            opt.pages,
            elems_per_page,
            opt.ops_per_page,
            0x56780000ull + static_cast<uint64_t>(opt.repeat * 257 + pass));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaDeviceSynchronize());
        const double wall_ms = now_ms() - wall_start;

        float event_ms_f = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&event_ms_f, ev_start, ev_stop));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));

        if (pass == 0 && should_start_bg_after_first) {
            phase_f_action_or_die(data,
                                  bytes,
                                  opt.device,
                                  UVM_PHASE_F_ACTION_START_BG_PROMOTION,
                                  "START_BG_PROMOTION");
        }

        const double mloads_s =
            wall_ms > 0.0 ? static_cast<double>(logical_loads) / (wall_ms * 1000.0) : 0.0;
        const double gib_s = wall_ms > 0.0 ? logical_gib / (wall_ms / 1000.0) : 0.0;

        std::printf("%s,%d,%d,%zu,%d,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    opt.mechanism.c_str(),
                    opt.repeat,
                    pass,
                    opt.pages,
                    opt.ops_per_page,
                    bytes,
                    setup_ms,
                    prepare_remote.wall_ms,
                    static_cast<double>(prepare_remote.mapped_bytes) / (1024.0 * 1024.0),
                    wall_ms,
                    static_cast<double>(event_ms_f),
                    mloads_s,
                    gib_s);
    }

    CHECK_CUDA(cudaFree(partials));
    if (host_data)
        CHECK_CUDA(cudaFreeHost(host_data));
    else
        CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
