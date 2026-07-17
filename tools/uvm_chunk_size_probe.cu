#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "uvm_linux_ioctl.h"
#include "uvm_test_ioctl.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                         "CUDA error %s:%d: %s\n",                            \
                         __FILE__,                                             \
                         __LINE__,                                             \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kBlockSize = 2ull * 1024ull * 1024ull;

__global__ void touch_sparse_pages(unsigned char *ptr, size_t bytes)
{
    const size_t block = blockIdx.x;
    const size_t offset = block * kBlockSize;

    if (offset < bytes)
        ptr[offset] = static_cast<unsigned char>(ptr[offset] + 1);
}

static void warmup_managed_fault_path()
{
    unsigned char *warm = nullptr;
    const size_t warm_bytes = kBlockSize;

    CHECK_CUDA(cudaMallocManaged(&warm, warm_bytes, cudaMemAttachGlobal));
    for (size_t offset = 0; offset < warm_bytes; offset += kPageSize)
        warm[offset] = static_cast<unsigned char>(offset / kPageSize);

    CHECK_CUDA(cudaMemPrefetchAsync(warm, warm_bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
    touch_sparse_pages<<<1, 1>>>(warm, warm_bytes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(warm));
    CHECK_CUDA(cudaDeviceSynchronize());
}

static bool fd_target_is_uvm(const char *target)
{
    const char *base;

    if (!target)
        return false;

    base = std::strrchr(target, '/');
    base = base ? base + 1 : target;

    return std::strcmp(base, "nvidia-uvm") == 0 ||
           std::strncmp(base, "nvidia-uvm ", std::strlen("nvidia-uvm ")) == 0;
}

static int duplicate_existing_uvm_fd()
{
    DIR *dir = opendir("/proc/self/fd");
    int result = -1;

    if (!dir)
        return -1;

    const int dir_fd = dirfd(dir);
    while (dirent *entry = readdir(dir)) {
        char *end = nullptr;
        long fd_long = std::strtol(entry->d_name, &end, 10);
        char fd_path[64];
        char link_target[PATH_MAX];
        ssize_t link_len;
        int dup_fd;

        if (end == entry->d_name || *end != '\0' || fd_long < 0 || fd_long > INT_MAX)
            continue;

        if (static_cast<int>(fd_long) == dir_fd)
            continue;

        std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%ld", fd_long);
        link_len = readlink(fd_path, link_target, sizeof(link_target) - 1);
        if (link_len < 0)
            continue;

        link_target[link_len] = '\0';
        if (!fd_target_is_uvm(link_target))
            continue;

        dup_fd = fcntl(static_cast<int>(fd_long), F_DUPFD_CLOEXEC, 3);
        if (dup_fd < 0)
            continue;

        result = dup_fd;
        break;
    }

    closedir(dir);
    return result;
}

static int query_block_info(int fd, unsigned char *addr)
{
    UVM_TEST_VA_BLOCK_INFO_PARAMS params = {};

    params.lookup_address = reinterpret_cast<NvU64>(addr);

    if (ioctl(fd, UVM_TEST_VA_BLOCK_INFO, &params) != 0) {
        std::fprintf(stderr,
                     "UVM_TEST_VA_BLOCK_INFO ioctl failed errno=%d (%s)\n",
                     errno,
                     std::strerror(errno));
        return -1;
    }

    std::printf("  va_block=[0x%llx, 0x%llx] size=%llu rmStatus=0x%x\n",
                static_cast<unsigned long long>(params.va_block_start),
                static_cast<unsigned long long>(params.va_block_end),
                static_cast<unsigned long long>(params.va_block_end - params.va_block_start + 1),
                params.rmStatus);

    return params.rmStatus == NV_OK ? 0 : -1;
}

static int query_residency(int fd, const char *label, unsigned char *addr)
{
    UVM_TEST_VA_RESIDENCY_INFO_PARAMS params = {};

    params.lookup_address = reinterpret_cast<NvU64>(addr);
    params.is_async = NV_FALSE;

    if (ioctl(fd, UVM_TEST_VA_RESIDENCY_INFO, &params) != 0) {
        std::fprintf(stderr,
                     "%s: UVM_TEST_VA_RESIDENCY_INFO ioctl failed errno=%d (%s)\n",
                     label,
                     errno,
                     std::strerror(errno));
        return -1;
    }

    std::printf("%s addr=0x%llx rmStatus=0x%x resident_on_count=%u mapped_on_count=%u populated_on_count=%u\n",
                label,
                static_cast<unsigned long long>(reinterpret_cast<NvU64>(addr)),
                params.rmStatus,
                params.resident_on_count,
                params.mapped_on_count,
                params.populated_on_count);

    for (NvU32 i = 0; i < params.resident_on_count; ++i) {
        std::printf("  resident[%u]: physical_size=%u physical_address=0x%llx\n",
                    i,
                    params.resident_physical_size[i],
                    static_cast<unsigned long long>(params.resident_physical_address[i]));
    }

    for (NvU32 i = 0; i < params.mapped_on_count; ++i) {
        std::printf("  mapped[%u]: page_size=%llu mapping_type=%u physical_address=0x%llx\n",
                    i,
                    static_cast<unsigned long long>(params.page_size[i]),
                    params.mapping_type[i],
                    static_cast<unsigned long long>(params.mapping_physical_address[i]));
    }

    return params.rmStatus == NV_OK ? 0 : -1;
}

static void print_mem_info(const char *label)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("%s cudaMemGetInfo free=%llu total=%llu used=%llu\n",
                label,
                static_cast<unsigned long long>(free_bytes),
                static_cast<unsigned long long>(total_bytes),
                static_cast<unsigned long long>(total_bytes - free_bytes));
}

} // namespace

int main(int argc, char **argv)
{
    size_t bytes = 4ull * 1024ull * 1024ull;
    int device = 0;

    if (argc > 1)
        bytes = std::strtoull(argv[1], nullptr, 0);
    if (argc > 2)
        device = std::atoi(argv[2]);

    CHECK_CUDA(cudaSetDevice(device));
    warmup_managed_fault_path();
    print_mem_info("initial");

    unsigned char *ptr = nullptr;
    CHECK_CUDA(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
    print_mem_info("after cudaMallocManaged");

    // CPU-touch the allocation first, then migrate only one 4K page per 2MiB
    // block to the GPU via faults from the sparse kernel.
    for (size_t offset = 0; offset < bytes; offset += kPageSize)
        ptr[offset] = static_cast<unsigned char>(offset / kPageSize);
    CHECK_CUDA(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
    print_mem_info("after CPU touch + CPU prefetch");

    const int fd = duplicate_existing_uvm_fd();
    if (fd < 0) {
        std::fprintf(stderr, "Could not find CUDA's existing /dev/nvidia-uvm fd\n");
        return EXIT_FAILURE;
    }

    std::printf("managed ptr=0x%llx bytes=%llu fd=%d\n",
                static_cast<unsigned long long>(reinterpret_cast<NvU64>(ptr)),
                static_cast<unsigned long long>(bytes),
                fd);

    std::puts("before GPU sparse touch:");
    (void)query_block_info(fd, ptr);
    (void)query_residency(fd, "block0+0", ptr);
    if (bytes > kBlockSize)
        (void)query_residency(fd, "block1+0", ptr + kBlockSize);

    const unsigned int num_blocks = static_cast<unsigned int>((bytes + kBlockSize - 1) / kBlockSize);
    touch_sparse_pages<<<num_blocks, 1>>>(ptr, bytes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    print_mem_info("after GPU sparse touch");

    std::puts("after GPU sparse touch:");
    (void)query_block_info(fd, ptr);
    (void)query_residency(fd, "block0+0", ptr);
    if (bytes > kPageSize)
        (void)query_residency(fd, "block0+4K", ptr + kPageSize);
    if (bytes > kBlockSize)
        (void)query_residency(fd, "block1+0", ptr + kBlockSize);
    if (bytes > kBlockSize + kPageSize)
        (void)query_residency(fd, "block1+4K", ptr + kBlockSize + kPageSize);

    close(fd);
    CHECK_CUDA(cudaFree(ptr));
    return EXIT_SUCCESS;
}
