#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

__global__ void touch_pages_kernel(unsigned char *ptr, size_t bytes, size_t stride)
{
    const size_t page = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t offset = page * stride;
    if (offset < bytes)
        ptr[offset] = static_cast<unsigned char>(ptr[offset] + 1);
}

namespace {

void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--device N] [--alloc-mb N ...] [--touch-gpu 0|1] "
                 "[--prefetch-gpu 0|1] [--alloc-kind managed|device]\n",
                 argv0);
}

void print_mem_info(const char *tag)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        std::printf("probe_mem tag=%s free_mb=%.2f total_mb=%.2f\n",
                    tag,
                    free_bytes / 1048576.0,
                    total_bytes / 1048576.0);
    }
    else {
        std::printf("probe_mem tag=%s error=%s\n", tag, cudaGetErrorString(err));
    }
    std::fflush(stdout);
}

int touch_gpu(void *ptr, size_t bytes)
{
    constexpr size_t stride = 4096;
    const size_t pages = (bytes + stride - 1) / stride;
    const int threads = 256;
    const int blocks = static_cast<int>((pages + threads - 1) / threads);

    touch_pages_kernel<<<blocks, threads>>>(static_cast<unsigned char *>(ptr), bytes, stride);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "probe_touch_gpu failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    int device = 0;
    bool do_touch_gpu = true;
    bool do_prefetch_gpu = false;
    std::string alloc_kind = "managed";
    std::vector<size_t> alloc_bytes;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--alloc-mb") == 0 && i + 1 < argc) {
            const size_t mb = std::strtoull(argv[++i], nullptr, 0);
            alloc_bytes.push_back(mb * 1024ULL * 1024ULL);
        }
        else if (std::strcmp(argv[i], "--touch-gpu") == 0 && i + 1 < argc) {
            do_touch_gpu = std::atoi(argv[++i]) != 0;
        }
        else if (std::strcmp(argv[i], "--prefetch-gpu") == 0 && i + 1 < argc) {
            do_prefetch_gpu = std::atoi(argv[++i]) != 0;
        }
        else if (std::strcmp(argv[i], "--alloc-kind") == 0 && i + 1 < argc) {
            alloc_kind = argv[++i];
            if (alloc_kind != "managed" && alloc_kind != "device") {
                usage(argv[0]);
                return 2;
            }
        }
        else if (std::strcmp(argv[i], "-h") == 0 ||
                 std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
        else {
            usage(argv[0]);
            return 2;
        }
    }

    if (alloc_bytes.empty())
        alloc_bytes.push_back(64ULL * 1024ULL * 1024ULL);

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
                     device,
                     cudaGetErrorString(err));
        return 1;
    }

    std::printf("probe_begin device=%d alloc_count=%zu touch_gpu=%d prefetch_gpu=%d alloc_kind=%s\n",
                device,
                alloc_bytes.size(),
                do_touch_gpu ? 1 : 0,
                do_prefetch_gpu ? 1 : 0,
                alloc_kind.c_str());
    print_mem_info("begin");

    std::vector<void *> ptrs;
    int failed_allocs = 0;

    for (size_t i = 0; i < alloc_bytes.size(); ++i) {
        void *ptr = nullptr;
        const size_t bytes = alloc_bytes[i];

        if (alloc_kind == "device")
            err = cudaMalloc(&ptr, bytes);
        else
            err = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
        std::printf("probe_alloc index=%zu bytes=%zu mb=%.2f status=%s ptr=%p\n",
                    i,
                    bytes,
                    bytes / 1048576.0,
                    cudaGetErrorString(err),
                    ptr);
        std::fflush(stdout);

        if (err != cudaSuccess) {
            ++failed_allocs;
            continue;
        }

        ptrs.push_back(ptr);

        if (do_prefetch_gpu) {
            err = cudaMemPrefetchAsync(ptr, bytes, device, 0);
            if (err == cudaSuccess)
                err = cudaDeviceSynchronize();
            std::printf("probe_prefetch_gpu index=%zu status=%s\n",
                        i,
                        cudaGetErrorString(err));
            std::fflush(stdout);
            if (err != cudaSuccess)
                return 1;
        }

        if (do_touch_gpu && touch_gpu(ptr, bytes) != 0)
            return 1;

        print_mem_info(("after_alloc_" + std::to_string(i)).c_str());
    }

    for (size_t i = 0; i < ptrs.size(); ++i) {
        err = cudaFree(ptrs[i]);
        std::printf("probe_free index=%zu status=%s ptr=%p\n",
                    i,
                    cudaGetErrorString(err),
                    ptrs[i]);
        std::fflush(stdout);
        if (err != cudaSuccess)
            return 1;
    }

    print_mem_info("after_free");
    std::printf("probe_done failed_allocs=%d successful_allocs=%zu\n",
                failed_allocs,
                ptrs.size());
    std::fflush(stdout);

    return failed_allocs == 0 ? 0 : 3;
}
