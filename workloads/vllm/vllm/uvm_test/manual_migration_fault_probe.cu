#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

__global__ void read_managed_page(const unsigned char *data,
                                  unsigned long long *sum)
{
    unsigned long long value = data[0];
    atomicAdd(sum, value);
}

bool check_cuda(cudaError_t status, const char *what)
{
    if (status == cudaSuccess)
        return true;

    std::printf("manual_migration_fault_probe cuda_error step=%s code=%d name=%s message=%s\n",
                what,
                static_cast<int>(status),
                cudaGetErrorName(status),
                cudaGetErrorString(status));
    return false;
}

} // namespace

int main(int argc, char **argv)
{
    bool expect_failure = false;
    size_t bytes = 4 * 1024 * 1024;
    int device = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect-failure") == 0) {
            expect_failure = true;
        }
        else if (std::strcmp(argv[i], "--expect-success") == 0) {
            expect_failure = false;
        }
        else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            bytes = std::strtoull(argv[++i], nullptr, 0);
        }
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        }
        else {
            std::fprintf(stderr,
                         "usage: %s [--expect-success|--expect-failure] [--bytes N] [--device N]\n",
                         argv[0]);
            return 2;
        }
    }

    if (!check_cuda(cudaSetDevice(device), "set_device"))
        return expect_failure ? 0 : 1;

    unsigned char *managed = nullptr;
    unsigned long long *device_sum = nullptr;

    if (!check_cuda(cudaMallocManaged(&managed, bytes), "malloc_managed"))
        return 1;
    if (!check_cuda(cudaMalloc(&device_sum, sizeof(*device_sum)), "malloc_device_sum"))
        return 1;

    for (size_t i = 0; i < bytes; i += 4096)
        managed[i] = static_cast<unsigned char>((i / 4096) & 0xff);

    if (!check_cuda(cudaMemset(device_sum, 0, sizeof(*device_sum)), "memset_device_sum"))
        return 1;
    if (!check_cuda(cudaMemPrefetchAsync(managed, bytes, cudaCpuDeviceId, 0), "prefetch_to_cpu"))
        return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "sync_after_cpu_prefetch"))
        return 1;

    std::printf("manual_migration_fault_probe launch ptr=%p bytes=%zu expect_failure=%u\n",
                managed,
                bytes,
                expect_failure ? 1 : 0);

    read_managed_page<<<1, 1>>>(managed, device_sum);
    cudaError_t launch_status = cudaGetLastError();
    cudaError_t sync_status = cudaDeviceSynchronize();

    if (launch_status == cudaSuccess && sync_status == cudaSuccess) {
        unsigned long long host_sum = 0;
        cudaError_t copy_status = cudaMemcpy(&host_sum,
                                             device_sum,
                                             sizeof(host_sum),
                                             cudaMemcpyDeviceToHost);
        if (copy_status == cudaSuccess) {
            std::printf("manual_migration_fault_probe kernel_success sum=%llu\n", host_sum);
            cudaFree(device_sum);
            cudaFree(managed);
            return expect_failure ? 1 : 0;
        }

        std::printf("manual_migration_fault_probe cuda_error step=copy_sum code=%d name=%s message=%s\n",
                    static_cast<int>(copy_status),
                    cudaGetErrorName(copy_status),
                    cudaGetErrorString(copy_status));
    }
    else {
        if (launch_status != cudaSuccess) {
            std::printf("manual_migration_fault_probe cuda_error step=launch code=%d name=%s message=%s\n",
                        static_cast<int>(launch_status),
                        cudaGetErrorName(launch_status),
                        cudaGetErrorString(launch_status));
        }
        if (sync_status != cudaSuccess) {
            std::printf("manual_migration_fault_probe cuda_error step=sync code=%d name=%s message=%s\n",
                        static_cast<int>(sync_status),
                        cudaGetErrorName(sync_status),
                        cudaGetErrorString(sync_status));
        }
    }

    if (expect_failure) {
        std::printf("manual_migration_fault_probe expected_failure_observed=1\n");
        return 0;
    }

    std::printf("manual_migration_fault_probe unexpected_failure=1\n");
    return 1;
}
