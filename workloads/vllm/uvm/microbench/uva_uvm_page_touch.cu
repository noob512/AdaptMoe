#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
  do {                                                                         \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                             \
    }                                                                          \
  } while (0)

__global__ void touch_kernel(const uint8_t *data, uint64_t *out,
                             size_t page_count, size_t stride,
                             int touches_per_page) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = page_count * static_cast<size_t>(touches_per_page);
  if (idx >= total) {
    return;
  }
  size_t page = idx / static_cast<size_t>(touches_per_page);
  size_t rep = idx % static_cast<size_t>(touches_per_page);
  size_t offset = page * stride + ((rep * 128) & (stride - 1));
  out[idx] = static_cast<uint64_t>(data[offset]) + static_cast<uint64_t>(idx);
}

struct Args {
  std::string mode = "uvm";
  size_t pages = 262144;
  size_t stride = 4096;
  int touches = 1;
  int warmups = 1;
  int iters = 5;
  int device = 0;
  bool advise_cpu = false;
  bool prefetch_cpu = false;
  bool prefetch_gpu = false;
};

static size_t parse_size(const char *value) {
  char *end = nullptr;
  double number = std::strtod(value, &end);
  size_t multiplier = 1;
  if (end != nullptr) {
    if (std::strcmp(end, "K") == 0 || std::strcmp(end, "k") == 0) {
      multiplier = 1024ull;
    } else if (std::strcmp(end, "M") == 0 || std::strcmp(end, "m") == 0) {
      multiplier = 1024ull * 1024ull;
    } else if (std::strcmp(end, "G") == 0 || std::strcmp(end, "g") == 0) {
      multiplier = 1024ull * 1024ull * 1024ull;
    }
  }
  return static_cast<size_t>(number * static_cast<double>(multiplier));
}

static Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", key.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (key == "--mode") {
      args.mode = need_value();
    } else if (key == "--pages") {
      args.pages = parse_size(need_value());
    } else if (key == "--bytes") {
      args.pages = parse_size(need_value()) / args.stride;
    } else if (key == "--stride") {
      args.stride = parse_size(need_value());
    } else if (key == "--touches") {
      args.touches = std::atoi(need_value());
    } else if (key == "--warmups") {
      args.warmups = std::atoi(need_value());
    } else if (key == "--iters") {
      args.iters = std::atoi(need_value());
    } else if (key == "--device") {
      args.device = std::atoi(need_value());
    } else if (key == "--advise-cpu") {
      args.advise_cpu = true;
    } else if (key == "--prefetch-cpu") {
      args.prefetch_cpu = true;
    } else if (key == "--prefetch-gpu") {
      args.prefetch_gpu = true;
    } else if (key == "--help") {
      std::printf(
          "Usage: %s --mode uvm|uva [--pages N|--bytes N] [--touches N]\n"
          "          [--warmups N] [--iters N] [--advise-cpu]\n"
          "          [--prefetch-cpu|--prefetch-gpu]\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", key.c_str());
      std::exit(2);
    }
  }
  if (args.mode != "uvm" && args.mode != "uva") {
    std::fprintf(stderr, "--mode must be uvm or uva\n");
    std::exit(2);
  }
  if ((args.stride & (args.stride - 1)) != 0 || args.stride < 128) {
    std::fprintf(stderr, "--stride must be a power of two >= 128\n");
    std::exit(2);
  }
  args.pages = std::max<size_t>(1, args.pages);
  args.touches = std::max(1, args.touches);
  args.iters = std::max(1, args.iters);
  return args;
}

static double median_ms(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
  CHECK_CUDA(cudaSetDevice(args.device));

  size_t bytes = args.pages * args.stride;
  uint8_t *ptr = nullptr;
  uint8_t *kernel_ptr = nullptr;
  uint64_t *out = nullptr;
  size_t total_touches = args.pages * static_cast<size_t>(args.touches);
  CHECK_CUDA(cudaMalloc(&out, total_touches * sizeof(uint64_t)));

  if (args.mode == "uva") {
    CHECK_CUDA(cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostGetDevicePointer(&kernel_ptr, ptr, 0));
  } else {
    CHECK_CUDA(cudaMallocManaged(&ptr, bytes));
    kernel_ptr = ptr;
  }

  for (size_t offset = 0; offset < bytes; offset += 4096) {
    ptr[offset] = static_cast<uint8_t>(offset / 4096);
  }

  if (args.mode == "uvm" && args.advise_cpu) {
    CHECK_CUDA(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation,
                             cudaCpuDeviceId));
  }
  if (args.mode == "uvm" && args.prefetch_cpu) {
    CHECK_CUDA(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  if (args.mode == "uvm" && args.prefetch_gpu) {
    CHECK_CUDA(cudaMemPrefetchAsync(ptr, bytes, args.device));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  int block = 256;
  int grid = static_cast<int>((total_touches + block - 1) / block);
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < args.warmups; ++i) {
    CHECK_CUDA(cudaMemset(out, 0, total_touches * sizeof(uint64_t)));
    touch_kernel<<<grid, block>>>(kernel_ptr, out, args.pages, args.stride,
                                  args.touches);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    if (args.mode == "uvm" && args.prefetch_cpu) {
      CHECK_CUDA(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  }

  std::vector<float> times_ms;
  times_ms.reserve(args.iters);
  for (int i = 0; i < args.iters; ++i) {
    if (args.mode == "uvm" && args.prefetch_cpu) {
      CHECK_CUDA(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemset(out, 0, total_touches * sizeof(uint64_t)));
    CHECK_CUDA(cudaEventRecord(start));
    touch_kernel<<<grid, block>>>(kernel_ptr, out, args.pages, args.stride,
                                  args.touches);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    times_ms.push_back(ms);
  }

  double med = median_ms(times_ms);
  double per_page_us = med * 1000.0 / static_cast<double>(args.pages);
  double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  double bw_gib_s = (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) /
                    (med / 1000.0);

  std::printf(
      "{"
      "\"mode\":\"%s\","
      "\"device\":%d,"
      "\"pages\":%zu,"
      "\"bytes\":%zu,"
      "\"mib\":%.3f,"
      "\"stride\":%zu,"
      "\"touches_per_page\":%d,"
      "\"median_ms\":%.6f,"
      "\"per_page_us\":%.6f,"
      "\"effective_gib_s\":%.6f,"
      "\"advise_cpu\":%s,"
      "\"prefetch_cpu\":%s,"
      "\"prefetch_gpu\":%s"
      "}\n",
      args.mode.c_str(), args.device, args.pages, bytes, mib, args.stride,
      args.touches, med, per_page_us, bw_gib_s,
      args.advise_cpu ? "true" : "false",
      args.prefetch_cpu ? "true" : "false",
      args.prefetch_gpu ? "true" : "false");

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  if (args.mode == "uva") {
    CHECK_CUDA(cudaFreeHost(ptr));
  } else {
    CHECK_CUDA(cudaFree(ptr));
  }
  CHECK_CUDA(cudaFree(out));
  return 0;
}
