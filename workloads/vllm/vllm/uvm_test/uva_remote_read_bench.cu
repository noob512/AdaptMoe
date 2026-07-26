#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void read_split_kernel(const uint4* __restrict__ data,
                                  std::size_t elements,
                                  unsigned long long* __restrict__ out) {
  const std::size_t blocks = gridDim.x;
  const std::size_t block_id = blockIdx.x;
  const std::size_t begin = (elements * block_id) / blocks;
  const std::size_t end = (elements * (block_id + 1)) / blocks;
  unsigned long long acc = 0;
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    uint4 v = data[i];
    acc += static_cast<unsigned long long>(v.x);
    acc += static_cast<unsigned long long>(v.y);
    acc += static_cast<unsigned long long>(v.z);
    acc += static_cast<unsigned long long>(v.w);
  }
  atomicAdd(out, acc);
}

__global__ void read_repeated_kernel(const uint4* __restrict__ data,
                                     std::size_t elements,
                                     unsigned long long* __restrict__ out) {
  unsigned long long acc = 0;
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
    uint4 v = data[i];
    acc += static_cast<unsigned long long>(v.x);
    acc += static_cast<unsigned long long>(v.y);
    acc += static_cast<unsigned long long>(v.z);
    acc += static_cast<unsigned long long>(v.w);
  }
  atomicAdd(out, acc);
}

enum class Backend {
  kUvaPinned,
  kUvmPreferredCpuAccessedBy,
  kDeviceLocal,
  kSyncH2DThenLocal,
};

struct Buffer {
  Backend backend;
  uint4* host_ptr;
  uint4* device_ptr;
  unsigned long long expected_sum;
};

struct Timing {
  float copy_ms;
  float kernel_ms;
  float total_ms;
};

static const char* backend_name(Backend backend) {
  switch (backend) {
    case Backend::kUvaPinned:
      return "uva_pinned";
    case Backend::kUvmPreferredCpuAccessedBy:
      return "uvm_preferred_cpu_accessed_by";
    case Backend::kDeviceLocal:
      return "device_local";
    case Backend::kSyncH2DThenLocal:
      return "sync_h2d_then_local";
  }
  return "unknown";
}

static Buffer allocate_buffer(Backend backend, std::size_t alloc_bytes,
                              std::size_t elems, int device) {
  Buffer buffer{backend, nullptr, nullptr, 0};
  if (backend == Backend::kUvaPinned) {
    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&buffer.host_ptr),
                             alloc_bytes, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&buffer.device_ptr), buffer.host_ptr, 0));
  } else if (backend == Backend::kUvmPreferredCpuAccessedBy) {
    CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&buffer.device_ptr),
                                 alloc_bytes));
    buffer.host_ptr = buffer.device_ptr;
    CHECK_CUDA(cudaMemAdvise(buffer.device_ptr, alloc_bytes,
                             cudaMemAdviseSetPreferredLocation,
                             cudaCpuDeviceId));
    CHECK_CUDA(cudaMemAdvise(buffer.device_ptr, alloc_bytes,
                             cudaMemAdviseSetAccessedBy, device));
  } else {
    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&buffer.host_ptr),
                             alloc_bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&buffer.device_ptr),
                          alloc_bytes));
  }

  for (std::size_t i = 0; i < elems; ++i) {
    const uint4 value = make_uint4(
        static_cast<unsigned>(i), static_cast<unsigned>(i >> 8),
        static_cast<unsigned>(i >> 16), static_cast<unsigned>(i >> 24));
    buffer.host_ptr[i] = value;
    buffer.expected_sum += static_cast<unsigned long long>(value.x);
    buffer.expected_sum += static_cast<unsigned long long>(value.y);
    buffer.expected_sum += static_cast<unsigned long long>(value.z);
    buffer.expected_sum += static_cast<unsigned long long>(value.w);
  }

  if (backend == Backend::kUvmPreferredCpuAccessedBy) {
    CHECK_CUDA(cudaMemPrefetchAsync(buffer.device_ptr, alloc_bytes,
                                    cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
  } else if (backend == Backend::kDeviceLocal ||
             backend == Backend::kSyncH2DThenLocal) {
    CHECK_CUDA(cudaMemcpy(buffer.device_ptr, buffer.host_ptr, alloc_bytes,
                          cudaMemcpyHostToDevice));
  }
  return buffer;
}

static void free_buffer(Buffer buffer) {
  if (buffer.backend == Backend::kUvaPinned) {
    CHECK_CUDA(cudaFreeHost(buffer.host_ptr));
  } else if (buffer.backend == Backend::kUvmPreferredCpuAccessedBy) {
    CHECK_CUDA(cudaFree(buffer.device_ptr));
  } else {
    CHECK_CUDA(cudaFree(buffer.device_ptr));
    CHECK_CUDA(cudaFreeHost(buffer.host_ptr));
  }
}

static void prepare_for_remote_read(Backend backend, const uint4* d_data,
                                    std::size_t alloc_bytes) {
  if (backend == Backend::kUvmPreferredCpuAccessedBy) {
    // Keep the measured kernel starting from CPU-resident managed pages. If the
    // driver migrates pages during a previous launch, this prevents later runs
    // from silently turning into HBM reads.
    CHECK_CUDA(cudaMemPrefetchAsync(const_cast<uint4*>(d_data), alloc_bytes,
                                    cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

static Timing time_iteration(const Buffer& buffer, bool repeated,
                             std::size_t elems, std::size_t alloc_bytes,
                             int blocks, int threads,
                             unsigned long long* d_out) {
  cudaEvent_t start, copy_done, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&copy_done));
  CHECK_CUDA(cudaEventCreate(&stop));
  prepare_for_remote_read(buffer.backend, buffer.device_ptr, alloc_bytes);
  CHECK_CUDA(cudaMemset(d_out, 0, sizeof(unsigned long long)));
  CHECK_CUDA(cudaEventRecord(start));

  const bool copy_before_read =
      buffer.backend == Backend::kSyncH2DThenLocal;
  if (copy_before_read) {
    // The copy and kernel use the same stream, so the read cannot begin until
    // the complete host-to-device transfer has finished.
    CHECK_CUDA(cudaMemcpyAsync(buffer.device_ptr, buffer.host_ptr, alloc_bytes,
                               cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(copy_done));
  }

  if (repeated) {
    read_repeated_kernel<<<blocks, threads>>>(buffer.device_ptr, elems, d_out);
  } else {
    read_split_kernel<<<blocks, threads>>>(buffer.device_ptr, elems, d_out);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));

  Timing timing{0.0f, 0.0f, 0.0f};
  if (copy_before_read) {
    CHECK_CUDA(cudaEventElapsedTime(&timing.copy_ms, start, copy_done));
    CHECK_CUDA(cudaEventElapsedTime(&timing.kernel_ms, copy_done, stop));
  } else {
    CHECK_CUDA(cudaEventElapsedTime(&timing.kernel_ms, start, stop));
  }
  CHECK_CUDA(cudaEventElapsedTime(&timing.total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(copy_done));
  CHECK_CUDA(cudaEventDestroy(stop));
  return timing;
}

static void validate_checksum(unsigned long long expected,
                              unsigned long long* d_out,
                              const char* backend, const char* mode,
                              int blocks) {
  unsigned long long actual = 0;
  CHECK_CUDA(cudaMemcpy(&actual, d_out, sizeof(actual),
                        cudaMemcpyDeviceToHost));
  if (actual != expected) {
    std::fprintf(stderr,
                 "checksum mismatch backend=%s mode=%s blocks=%d "
                 "expected=%llu actual=%llu\n",
                 backend, mode, blocks, expected, actual);
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  double gib = 5.0;
  int threads = 256;
  int warmups = 1;
  int iters = 3;
  if (argc > 1) {
    gib = std::atof(argv[1]);
  }
  if (argc > 2) {
    threads = std::atoi(argv[2]);
  }
  if (argc > 3) {
    iters = std::atoi(argv[3]);
  }
  if (argc > 4) {
    warmups = std::atoi(argv[4]);
  }

  CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
  CHECK_CUDA(cudaFree(0));

  const std::size_t bytes =
      static_cast<std::size_t>(gib * 1024.0 * 1024.0 * 1024.0);
  const std::size_t elems = bytes / sizeof(uint4);
  const std::size_t alloc_bytes = elems * sizeof(uint4);

  unsigned long long* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(unsigned long long)));

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::printf("device=%s sm_count=%d buffer=%.3fGiB bytes=%zu threads=%d "
              "iters=%d warmups=%d\n",
              prop.name, prop.multiProcessorCount,
              static_cast<double>(alloc_bytes) / (1024.0 * 1024.0 * 1024.0),
              alloc_bytes, threads, iters, warmups);
  std::printf("backend,mode,blocks,unique_gib,kernel_read_gib,copy_gib,"
              "copy_ms,kernel_ms,total_ms,copy_gib_s,kernel_gib_s,"
              "effective_read_gib_s\n");
  std::fflush(stdout);

  std::vector<int> block_counts = {1, 2, 4, 8, 16, 32, 64};
  std::vector<Backend> backends = {
      Backend::kUvaPinned,
      Backend::kUvmPreferredCpuAccessedBy,
      Backend::kDeviceLocal,
      Backend::kSyncH2DThenLocal,
  };

  for (Backend backend : backends) {
    Buffer buffer = allocate_buffer(backend, alloc_bytes, elems, device);
    for (bool repeated : {false, true}) {
      const char* mode = repeated ? "repeated" : "split";
      for (int blocks : block_counts) {
        for (int i = 0; i < warmups; ++i) {
          (void)time_iteration(buffer, repeated, elems, alloc_bytes, blocks,
                               threads, d_out);
        }
        Timing best{0.0f, 0.0f, 1.0e30f};
        for (int i = 0; i < iters; ++i) {
          const Timing current = time_iteration(
              buffer, repeated, elems, alloc_bytes, blocks, threads, d_out);
          if (current.total_ms < best.total_ms) {
            best = current;
          }
        }

        const unsigned long long expected =
            repeated ? buffer.expected_sum *
                           static_cast<unsigned long long>(blocks)
                     : buffer.expected_sum;
        validate_checksum(expected, d_out, backend_name(backend), mode,
                          blocks);

        const double unique_gib = static_cast<double>(alloc_bytes) /
                                  (1024.0 * 1024.0 * 1024.0);
        const double kernel_read_gib =
            unique_gib * (repeated ? static_cast<double>(blocks) : 1.0);
        const double copy_gib =
            backend == Backend::kSyncH2DThenLocal ? unique_gib : 0.0;
        const double copy_gib_s =
            best.copy_ms > 0.0f
                ? copy_gib / (static_cast<double>(best.copy_ms) / 1000.0)
                : 0.0;
        const double kernel_gib_s =
            kernel_read_gib /
            (static_cast<double>(best.kernel_ms) / 1000.0);
        const double effective_read_gib_s =
            kernel_read_gib /
            (static_cast<double>(best.total_ms) / 1000.0);
        std::printf(
            "%s,%s,%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            backend_name(backend), mode, blocks, unique_gib,
            kernel_read_gib, copy_gib, best.copy_ms, best.kernel_ms,
            best.total_ms, copy_gib_s, kernel_gib_s, effective_read_gib_s);
        std::fflush(stdout);
      }
    }
    free_buffer(buffer);
  }

  std::fprintf(stderr, "all_checksums_valid=true\n");
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
