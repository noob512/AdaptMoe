#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
int uvm_driver_phase_f_ping();
int uvm_driver_batch_prefetch_cpu(const uintptr_t* ptrs,
                                  const size_t* sizes,
                                  size_t count,
                                  int async);
int uvm_driver_batch_prefetch_gpu(const uintptr_t* ptrs,
                                  const size_t* sizes,
                                  size_t count,
                                  int device,
                                  int async);
int uvm_driver_batch_advise_cpu(const uintptr_t* ptrs,
                                const size_t* sizes,
                                size_t count);
int uvm_driver_batch_query_residency(const uintptr_t* ptrs,
                                     const size_t* sizes,
                                     size_t count,
                                     int device);
}

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--step all|ping|query|prefetch-gpu|advise-cpu|prefetch-cpu] "
                 "[--bytes N] [--device N] [--async 0|1]\n",
                 argv0);
}

bool want_step(const std::string& requested, const char* step) {
    return requested == "all" || requested == step;
}

int run_step(const char* name, int result) {
    std::printf("phase_f_smoke step=%s result=%d\n", name, result);
    std::fflush(stdout);
    return result < 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string step = "all";
    size_t bytes = 1 << 20;
    int device = 0;
    int async = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            step = argv[++i];
        }
        else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            bytes = std::strtoull(argv[++i], nullptr, 0);
        }
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--async") == 0 && i + 1 < argc) {
            async = std::atoi(argv[++i]);
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

    std::printf("phase_f_smoke begin step=%s bytes=%zu device=%d async=%d\n",
                step.c_str(), bytes, device, async);
    std::fflush(stdout);

    if (want_step(step, "ping")) {
        int failed = run_step("ping", uvm_driver_phase_f_ping());
        if (failed || step == "ping") return failed;
    }

    void* ptr = nullptr;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 3;
    }

    err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 4;
    }

    std::memset(ptr, 0x5a, bytes < 4096 ? bytes : 4096);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize after memset failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(ptr);
        return 5;
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    size_t size = bytes;
    std::printf("phase_f_smoke managed_ptr=0x%llx bytes=%zu\n",
                static_cast<unsigned long long>(base), size);
    std::fflush(stdout);

    if (want_step(step, "query")) {
        int failed = run_step(
            "query", uvm_driver_batch_query_residency(&base, &size, 1, device));
        if (failed || step == "query") {
            cudaFree(ptr);
            return failed;
        }
    }

    if (want_step(step, "prefetch-gpu")) {
        int failed = run_step(
            "prefetch-gpu",
            uvm_driver_batch_prefetch_gpu(&base, &size, 1, device, async));
        cudaDeviceSynchronize();
        if (failed || step == "prefetch-gpu") {
            cudaFree(ptr);
            return failed;
        }
    }

    if (want_step(step, "advise-cpu")) {
        int failed = run_step(
            "advise-cpu", uvm_driver_batch_advise_cpu(&base, &size, 1));
        if (failed || step == "advise-cpu") {
            cudaFree(ptr);
            return failed;
        }
    }

    if (want_step(step, "prefetch-cpu")) {
        int failed = run_step(
            "prefetch-cpu",
            uvm_driver_batch_prefetch_cpu(&base, &size, 1, async));
        cudaDeviceSynchronize();
        if (failed || step == "prefetch-cpu") {
            cudaFree(ptr);
            return failed;
        }
    }

    cudaFree(ptr);
    std::printf("phase_f_smoke done\n");
    return 0;
}
