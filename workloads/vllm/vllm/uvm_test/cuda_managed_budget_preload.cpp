/*
 * LD_PRELOAD helper for early UVM residency/budget experiments.
 *
 * It supports two independent knobs:
 *
 *   UVM_HBM_RESERVE_BYTES / UVM_HBM_RESERVE_MB / UVM_HBM_RESERVE_GB
 *     Lazily cudaMalloc() a raw device allocation before the first managed
 *     allocation. This consumes HBM and reduces the physical GPU residency
 *     headroom available to UVM migrations.
 *
 *   UVM_MANAGED_POOL_BYTES / UVM_MANAGED_POOL_MB / UVM_MANAGED_POOL_GB
 *     Lazily create one fixed cudaMallocManaged() arena and serve intercepted
 *     cudaMallocManaged() / cuMemAllocManaged() requests by suballocation.
 *     This limits logical managed allocation capacity for process-level smoke
 *     experiments.
 *
 * Important: a cudaMalloc() device pointer is not a valid managed allocation,
 * so cudaMallocManaged() cannot literally return slices from a raw HBM pool.
 * The raw HBM reserve models the physical-residency cap; the managed arena
 * models the logical managed-capacity cap.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#undef cuMemAllocManaged
#undef cuMemFree
#undef cuMemFree_v2

namespace {

struct PoolBlock {
    size_t offset;
    size_t size;
    bool free;
};

static cudaError_t (*real_cudaMalloc)(void **, size_t) = nullptr;
static cudaError_t (*real_cudaMallocManaged)(void **, size_t, unsigned int) = nullptr;
static cudaError_t (*real_cudaFree)(void *) = nullptr;
static cudaError_t (*real_cudaMemPrefetchAsync)(const void *, size_t, int, cudaStream_t) = nullptr;
static cudaError_t (*real_cudaDeviceSynchronize)() = nullptr;
static cudaError_t (*real_cudaGetDevice)(int *) = nullptr;
static cudaError_t (*real_cudaSetDevice)(int) = nullptr;
static cudaError_t (*real_cudaMemGetInfo)(size_t *, size_t *) = nullptr;

static CUresult (*real_cuMemAllocManaged)(CUdeviceptr *, size_t, unsigned int) = nullptr;
static CUresult (*real_cuMemFree)(CUdeviceptr) = nullptr;
static CUresult (*real_cuMemFree_v2)(CUdeviceptr) = nullptr;

static std::once_flag init_once;
static std::mutex pool_mutex;

static bool initialized = false;
static bool disabled = false;
static bool verbose = false;
static bool strict = true;
static bool prefetch_pool = false;
static bool touch_hbm_reserve = false;
static bool device_override_set = false;
static int target_device = 0;

static size_t hbm_reserve_bytes = 0;
static size_t managed_pool_bytes = 0;
static size_t pool_alignment = 256;

static void *hbm_reserve_ptr = nullptr;
static void *managed_pool_base = nullptr;
static std::vector<PoolBlock> pool_blocks;

static std::atomic<size_t> intercepted_allocs{0};
static std::atomic<size_t> pool_allocs{0};
static std::atomic<size_t> pool_frees{0};
static std::atomic<size_t> pool_live_bytes{0};
static std::atomic<size_t> pool_peak_live_bytes{0};

static bool env_truthy(const char *name, bool default_value)
{
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0')
        return default_value;
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
}

static size_t parse_size_env_one(const char *name, size_t multiplier)
{
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0')
        return 0;

    errno = 0;
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 0);
    if (errno != 0 || end == value)
        return 0;

    if (parsed > static_cast<unsigned long long>(SIZE_MAX / multiplier))
        return SIZE_MAX;

    return static_cast<size_t>(parsed) * multiplier;
}

static size_t parse_size_env(const char *bytes_name,
                             const char *mb_name,
                             const char *gb_name)
{
    size_t bytes = parse_size_env_one(bytes_name, 1);
    if (bytes)
        return bytes;

    bytes = parse_size_env_one(mb_name, 1024ULL * 1024ULL);
    if (bytes)
        return bytes;

    return parse_size_env_one(gb_name, 1024ULL * 1024ULL * 1024ULL);
}

static size_t align_up(size_t value, size_t alignment)
{
    if (alignment == 0)
        return value;

    const size_t remainder = value % alignment;
    if (remainder == 0)
        return value;

    if (value > SIZE_MAX - (alignment - remainder))
        return SIZE_MAX;

    return value + alignment - remainder;
}

static void update_peak_live(size_t current)
{
    size_t peak = pool_peak_live_bytes.load(std::memory_order_relaxed);
    while (current > peak) {
        if (pool_peak_live_bytes.compare_exchange_weak(peak, current))
            break;
    }
}

static void *lookup_symbol(const char *name, bool required = false)
{
    void *symbol = dlsym(RTLD_NEXT, name);
    if (symbol)
        return symbol;

    const char *error = dlerror();
    if (required && error)
        std::fprintf(stderr, "[UVM_BUDGET] dlsym(%s) failed: %s\n", name, error);
    return nullptr;
}

static bool load_real_functions()
{
    real_cudaMalloc = reinterpret_cast<cudaError_t (*)(void **, size_t)>(
        lookup_symbol("cudaMalloc", hbm_reserve_bytes != 0));
    real_cudaMallocManaged = reinterpret_cast<cudaError_t (*)(void **, size_t, unsigned int)>(
        lookup_symbol("cudaMallocManaged", true));
    real_cudaFree = reinterpret_cast<cudaError_t (*)(void *)>(
        lookup_symbol("cudaFree", true));
    real_cudaMemPrefetchAsync =
        reinterpret_cast<cudaError_t (*)(const void *, size_t, int, cudaStream_t)>(
            lookup_symbol("cudaMemPrefetchAsync"));
    real_cudaDeviceSynchronize = reinterpret_cast<cudaError_t (*)()>(
        lookup_symbol("cudaDeviceSynchronize"));
    real_cudaGetDevice = reinterpret_cast<cudaError_t (*)(int *)>(
        lookup_symbol("cudaGetDevice"));
    real_cudaSetDevice = reinterpret_cast<cudaError_t (*)(int)>(
        lookup_symbol("cudaSetDevice"));
    real_cudaMemGetInfo = reinterpret_cast<cudaError_t (*)(size_t *, size_t *)>(
        lookup_symbol("cudaMemGetInfo"));

    real_cuMemAllocManaged = reinterpret_cast<CUresult (*)(CUdeviceptr *, size_t, unsigned int)>(
        lookup_symbol("cuMemAllocManaged", false));
    real_cuMemFree = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
        lookup_symbol("cuMemFree", false));
    real_cuMemFree_v2 = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
        lookup_symbol("cuMemFree_v2", false));

    return real_cudaMallocManaged && real_cudaFree;
}

static void parse_config()
{
    disabled = env_truthy("UVM_MANAGED_BUDGET_DISABLE", false);
    verbose = env_truthy("UVM_MANAGED_BUDGET_VERBOSE", false);
    strict = env_truthy("UVM_MANAGED_BUDGET_STRICT", true);
    prefetch_pool = env_truthy("UVM_MANAGED_POOL_PREFETCH", false);
    touch_hbm_reserve = env_truthy("UVM_HBM_RESERVE_TOUCH", false);

    hbm_reserve_bytes = parse_size_env("UVM_HBM_RESERVE_BYTES",
                                       "UVM_HBM_RESERVE_MB",
                                       "UVM_HBM_RESERVE_GB");
    managed_pool_bytes = parse_size_env("UVM_MANAGED_POOL_BYTES",
                                        "UVM_MANAGED_POOL_MB",
                                        "UVM_MANAGED_POOL_GB");

    size_t requested_alignment = parse_size_env_one("UVM_MANAGED_POOL_ALIGNMENT", 1);
    if (requested_alignment)
        pool_alignment = std::max<size_t>(requested_alignment, 16);

    const char *device_env = std::getenv("UVM_MANAGED_BUDGET_DEVICE");
    if (!device_env || device_env[0] == '\0')
        device_env = std::getenv("UVM_HBM_GUARD_DEVICE");
    if (device_env && device_env[0] != '\0') {
        target_device = std::atoi(device_env);
        device_override_set = true;
    }
}

static void log_mem_info(const char *where)
{
    if (!verbose || !real_cudaMemGetInfo)
        return;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = real_cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        std::fprintf(stderr,
                     "[UVM_BUDGET] %s mem_free=%.2f MiB mem_total=%.2f MiB\n",
                     where,
                     free_bytes / 1048576.0,
                     total_bytes / 1048576.0);
    }
}

static bool maybe_select_device()
{
    if (device_override_set && real_cudaSetDevice) {
        cudaError_t err = real_cudaSetDevice(target_device);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "[UVM_BUDGET] cudaSetDevice(%d) failed before reserve: %s\n",
                         target_device,
                         cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    if (real_cudaGetDevice) {
        int current = 0;
        cudaError_t err = real_cudaGetDevice(&current);
        if (err == cudaSuccess)
            target_device = current;
    }
    return true;
}

static bool reserve_hbm()
{
    if (hbm_reserve_bytes == 0)
        return true;

    if (!real_cudaMalloc)
        return false;

    log_mem_info("before_hbm_reserve");
    cudaError_t err = real_cudaMalloc(&hbm_reserve_ptr, hbm_reserve_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[UVM_BUDGET] cudaMalloc HBM reserve %.2f MiB failed: %s\n",
                     hbm_reserve_bytes / 1048576.0,
                     cudaGetErrorString(err));
        return false;
    }

    if (touch_hbm_reserve) {
        cudaError_t memset_err = cudaMemset(hbm_reserve_ptr, 0, hbm_reserve_bytes);
        if (memset_err == cudaSuccess && real_cudaDeviceSynchronize)
            memset_err = real_cudaDeviceSynchronize();
        if (memset_err != cudaSuccess) {
            std::fprintf(stderr,
                         "[UVM_BUDGET] touching HBM reserve failed: %s\n",
                         cudaGetErrorString(memset_err));
            return false;
        }
    }

    std::fprintf(stderr,
                 "[UVM_BUDGET] reserved %.2f MiB raw HBM on device=%d ptr=%p\n",
                 hbm_reserve_bytes / 1048576.0,
                 target_device,
                 hbm_reserve_ptr);
    log_mem_info("after_hbm_reserve");
    return true;
}

static bool create_managed_pool()
{
    if (managed_pool_bytes == 0)
        return true;

    if (!real_cudaMallocManaged)
        return false;

    const size_t aligned_pool = align_up(managed_pool_bytes, pool_alignment);
    if (aligned_pool == SIZE_MAX)
        return false;

    log_mem_info("before_managed_pool");
    cudaError_t err = real_cudaMallocManaged(&managed_pool_base,
                                             aligned_pool,
                                             cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[UVM_BUDGET] cudaMallocManaged arena %.2f MiB failed: %s\n",
                     aligned_pool / 1048576.0,
                     cudaGetErrorString(err));
        return false;
    }

    managed_pool_bytes = aligned_pool;
    pool_blocks.clear();
    pool_blocks.push_back(PoolBlock{0, managed_pool_bytes, true});

    if (prefetch_pool && real_cudaMemPrefetchAsync) {
        err = real_cudaMemPrefetchAsync(managed_pool_base,
                                        managed_pool_bytes,
                                        target_device,
                                        0);
        if (err == cudaSuccess && real_cudaDeviceSynchronize)
            err = real_cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "[UVM_BUDGET] prefetch managed arena to device=%d failed: %s\n",
                         target_device,
                         cudaGetErrorString(err));
            return false;
        }
    }

    std::fprintf(stderr,
                 "[UVM_BUDGET] created managed arena %.2f MiB base=%p device=%d prefetch=%d\n",
                 managed_pool_bytes / 1048576.0,
                 managed_pool_base,
                 target_device,
                 prefetch_pool ? 1 : 0);
    log_mem_info("after_managed_pool");
    return true;
}

static void initialize()
{
    parse_config();
    if (disabled) {
        initialized = load_real_functions();
        return;
    }

    if (!load_real_functions()) {
        std::fprintf(stderr, "[UVM_BUDGET] failed to load CUDA symbols\n");
        initialized = false;
        return;
    }

    if (!maybe_select_device()) {
        initialized = false;
        return;
    }

    bool ok = reserve_hbm();
    if (ok)
        ok = create_managed_pool();

    initialized = ok || !strict;
    if (!ok && strict)
        std::fprintf(stderr, "[UVM_BUDGET] strict mode enabled; managed allocations will fail\n");
}

static bool ensure_initialized()
{
    std::call_once(init_once, initialize);
    return initialized;
}

static bool pointer_in_pool(const void *ptr)
{
    if (!managed_pool_base || !ptr)
        return false;

    const auto begin = reinterpret_cast<uintptr_t>(managed_pool_base);
    const auto end = begin + managed_pool_bytes;
    const auto value = reinterpret_cast<uintptr_t>(ptr);
    return value >= begin && value < end;
}

static cudaError_t pool_alloc(void **dev_ptr, size_t size)
{
    if (!managed_pool_base)
        return cudaErrorInvalidValue;

    const size_t aligned_size = align_up(size, pool_alignment);
    if (aligned_size == 0 || aligned_size == SIZE_MAX)
        return cudaErrorInvalidValue;

    std::lock_guard<std::mutex> lock(pool_mutex);
    for (size_t i = 0; i < pool_blocks.size(); ++i) {
        PoolBlock &block = pool_blocks[i];
        if (!block.free || block.size < aligned_size)
            continue;

        const size_t offset = block.offset;
        const size_t leftover = block.size - aligned_size;
        block.size = aligned_size;
        block.free = false;

        if (leftover > 0) {
            pool_blocks.insert(pool_blocks.begin() + static_cast<std::ptrdiff_t>(i + 1),
                               PoolBlock{offset + aligned_size, leftover, true});
        }

        *dev_ptr = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(managed_pool_base) + offset);

        const size_t live = pool_live_bytes.fetch_add(aligned_size) + aligned_size;
        pool_allocs.fetch_add(1);
        update_peak_live(live);

        if (verbose) {
            std::fprintf(stderr,
                         "[UVM_BUDGET] pool alloc request=%zu aligned=%zu ptr=%p live=%.2f MiB/%.2f MiB\n",
                         size,
                         aligned_size,
                         *dev_ptr,
                         live / 1048576.0,
                         managed_pool_bytes / 1048576.0);
        }

        return cudaSuccess;
    }

    if (verbose) {
        std::fprintf(stderr,
                     "[UVM_BUDGET] pool alloc %.2f MiB failed live=%.2f MiB pool=%.2f MiB\n",
                     aligned_size / 1048576.0,
                     pool_live_bytes.load() / 1048576.0,
                     managed_pool_bytes / 1048576.0);
    }
    *dev_ptr = nullptr;
    return cudaErrorMemoryAllocation;
}

static cudaError_t pool_free(void *ptr)
{
    if (!pointer_in_pool(ptr))
        return cudaErrorInvalidValue;

    const size_t offset = reinterpret_cast<uintptr_t>(ptr) -
                          reinterpret_cast<uintptr_t>(managed_pool_base);

    std::lock_guard<std::mutex> lock(pool_mutex);
    for (size_t i = 0; i < pool_blocks.size(); ++i) {
        PoolBlock &block = pool_blocks[i];
        if (block.offset != offset || block.free)
            continue;

        block.free = true;
        pool_live_bytes.fetch_sub(block.size);
        pool_frees.fetch_add(1);

        if (i + 1 < pool_blocks.size() && pool_blocks[i + 1].free) {
            block.size += pool_blocks[i + 1].size;
            pool_blocks.erase(pool_blocks.begin() + static_cast<std::ptrdiff_t>(i + 1));
        }
        if (i > 0 && pool_blocks[i - 1].free) {
            pool_blocks[i - 1].size += block.size;
            pool_blocks.erase(pool_blocks.begin() + static_cast<std::ptrdiff_t>(i));
        }

        if (verbose) {
            std::fprintf(stderr,
                         "[UVM_BUDGET] pool free ptr=%p live=%.2f MiB\n",
                         ptr,
                         pool_live_bytes.load() / 1048576.0);
        }
        return cudaSuccess;
    }

    return cudaErrorInvalidValue;
}

}  // namespace

extern "C" {

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    intercepted_allocs.fetch_add(1);

    if (!ensure_initialized()) {
        if (devPtr)
            *devPtr = nullptr;
        return cudaErrorMemoryAllocation;
    }

    if (disabled || managed_pool_bytes == 0)
        return real_cudaMallocManaged(devPtr, size, flags);

    return pool_alloc(devPtr, size);
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    intercepted_allocs.fetch_add(1);

    if (!ensure_initialized()) {
        if (dptr)
            *dptr = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    if (disabled || managed_pool_bytes == 0) {
        if (!real_cuMemAllocManaged)
            return CUDA_ERROR_NOT_SUPPORTED;
        return real_cuMemAllocManaged(dptr, bytesize, flags);
    }

    void *ptr = nullptr;
    cudaError_t err = pool_alloc(&ptr, bytesize);
    if (err != cudaSuccess) {
        if (dptr)
            *dptr = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr = reinterpret_cast<CUdeviceptr>(ptr);
    return CUDA_SUCCESS;
}

cudaError_t cudaFree(void *devPtr)
{
    if (!ensure_initialized())
        return cudaErrorMemoryAllocation;

    if (managed_pool_base && pointer_in_pool(devPtr))
        return pool_free(devPtr);

    return real_cudaFree(devPtr);
}

CUresult cuMemFree(CUdeviceptr dptr)
{
    if (!ensure_initialized())
        return CUDA_ERROR_NOT_INITIALIZED;

    void *ptr = reinterpret_cast<void *>(dptr);
    if (managed_pool_base && pointer_in_pool(ptr))
        return pool_free(ptr) == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;

    if (real_cuMemFree_v2)
        return real_cuMemFree_v2(dptr);
    if (real_cuMemFree)
        return real_cuMemFree(dptr);
    return CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    return cuMemFree(dptr);
}

size_t uvm_managed_budget_pool_live_bytes()
{
    return pool_live_bytes.load();
}

size_t uvm_managed_budget_pool_peak_live_bytes()
{
    return pool_peak_live_bytes.load();
}

size_t uvm_managed_budget_pool_capacity_bytes()
{
    return managed_pool_bytes;
}

void uvm_managed_budget_print_stats()
{
    std::fprintf(stderr,
                 "[UVM_BUDGET] stats intercepted=%zu pool_allocs=%zu pool_frees=%zu "
                 "pool_live=%.2f MiB pool_peak=%.2f MiB pool_capacity=%.2f MiB "
                 "hbm_reserve=%.2f MiB\n",
                 intercepted_allocs.load(),
                 pool_allocs.load(),
                 pool_frees.load(),
                 pool_live_bytes.load() / 1048576.0,
                 pool_peak_live_bytes.load() / 1048576.0,
                 managed_pool_bytes / 1048576.0,
                 hbm_reserve_bytes / 1048576.0);
}

}  // extern "C"

__attribute__((destructor))
static void on_unload()
{
    if (!initialized)
        return;

    if (verbose || pool_allocs.load() > 0 || hbm_reserve_ptr)
        uvm_managed_budget_print_stats();

    if (managed_pool_base && real_cudaFree) {
        real_cudaFree(managed_pool_base);
        managed_pool_base = nullptr;
    }

    if (hbm_reserve_ptr && real_cudaFree) {
        real_cudaFree(hbm_reserve_ptr);
        hbm_reserve_ptr = nullptr;
    }
}
