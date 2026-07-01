#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void stage_read_kernel(const unsigned char *src,
                                  unsigned char *dst,
                                  size_t bytes,
                                  int iters)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    volatile const unsigned char *vsrc = src;

    if (iters < 1)
        iters = 1;

    for (; index < bytes; index += stride) {
        unsigned int acc = 0;
        for (int iter = 0; iter < iters; ++iter)
            acc += vsrc[index];
        dst[index] = static_cast<unsigned char>(acc);
    }
}

static int cuda_error_to_int(cudaError_t err)
{
    return err == cudaSuccess ? 0 : -static_cast<int>(err);
}

}  // namespace

extern "C" {

int uvm_stage_probe_set_device(int device)
{
    return cuda_error_to_int(cudaSetDevice(device));
}

int uvm_stage_probe_create_stream(void **stream_out)
{
    if (stream_out == nullptr)
        return -1;

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
        return cuda_error_to_int(err);

    *stream_out = reinterpret_cast<void *>(stream);
    return 0;
}

int uvm_stage_probe_destroy_stream(void *stream)
{
    if (stream == nullptr)
        return 0;
    return cuda_error_to_int(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

int uvm_stage_probe_create_event(void **event_out, int enable_timing)
{
    if (event_out == nullptr)
        return -1;

    cudaEvent_t event = nullptr;
    unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
    cudaError_t err = cudaEventCreateWithFlags(&event, flags);
    if (err != cudaSuccess)
        return cuda_error_to_int(err);

    *event_out = reinterpret_cast<void *>(event);
    return 0;
}

int uvm_stage_probe_destroy_event(void *event)
{
    if (event == nullptr)
        return 0;
    return cuda_error_to_int(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
}

int uvm_stage_probe_record_event(void *event, void *stream)
{
    if (event == nullptr)
        return -1;
    return cuda_error_to_int(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                            reinterpret_cast<cudaStream_t>(stream)));
}

int uvm_stage_probe_sync_event(void *event)
{
    if (event == nullptr)
        return -1;
    return cuda_error_to_int(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
}

int uvm_stage_probe_query_event(void *event)
{
    if (event == nullptr)
        return -1;

    cudaError_t err = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
    if (err == cudaSuccess)
        return 0;
    if (err == cudaErrorNotReady)
        return 1;
    return cuda_error_to_int(err);
}

int uvm_stage_probe_elapsed_ms(void *start_event, void *end_event, float *ms_out)
{
    if (start_event == nullptr || end_event == nullptr || ms_out == nullptr)
        return -1;

    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms,
                                           reinterpret_cast<cudaEvent_t>(start_event),
                                           reinterpret_cast<cudaEvent_t>(end_event));
    if (err != cudaSuccess)
        return cuda_error_to_int(err);

    *ms_out = ms;
    return 0;
}

int uvm_stage_probe_alloc_device(void **ptr_out, size_t bytes, int device)
{
    if (ptr_out == nullptr || bytes == 0)
        return -1;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
        return cuda_error_to_int(err);

    void *ptr = nullptr;
    err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
        return cuda_error_to_int(err);

    *ptr_out = ptr;
    return 0;
}

int uvm_stage_probe_free_device(void *ptr)
{
    if (ptr == nullptr)
        return 0;
    return cuda_error_to_int(cudaFree(ptr));
}

int uvm_stage_probe_copy_device_to_host(void *host_dst, const void *device_src, size_t bytes)
{
    if (host_dst == nullptr || device_src == nullptr || bytes == 0)
        return -1;

    cudaError_t err = cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost);
    return cuda_error_to_int(err);
}

int uvm_stage_probe_alloc_pinned_host(void **ptr_out, size_t bytes)
{
    if (ptr_out == nullptr || bytes == 0)
        return -1;

    void *ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess)
        return cuda_error_to_int(err);

    *ptr_out = ptr;
    return 0;
}

int uvm_stage_probe_free_pinned_host(void *ptr)
{
    if (ptr == nullptr)
        return 0;
    return cuda_error_to_int(cudaFreeHost(ptr));
}

int uvm_stage_probe_copy_host_to_device_async(void *device_dst,
                                              const void *host_src,
                                              size_t bytes,
                                              void *stream)
{
    if (device_dst == nullptr || host_src == nullptr || bytes == 0)
        return -1;

    cudaError_t err = cudaMemcpyAsync(device_dst,
                                      host_src,
                                      bytes,
                                      cudaMemcpyHostToDevice,
                                      reinterpret_cast<cudaStream_t>(stream));
    return cuda_error_to_int(err);
}

int uvm_stage_probe_launch_read(const void *src,
                                void *dst,
                                size_t bytes,
                                int iters,
                                void *stream)
{
    if (src == nullptr || dst == nullptr || bytes == 0)
        return -1;

    int threads = 256;
    int blocks = static_cast<int>((bytes + threads - 1) / threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 2048)
        blocks = 2048;

    stage_read_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const unsigned char *>(src),
        static_cast<unsigned char *>(dst),
        bytes,
        iters);

    return cuda_error_to_int(cudaGetLastError());
}

int uvm_stage_probe_device_synchronize()
{
    return cuda_error_to_int(cudaDeviceSynchronize());
}

}  // extern "C"
