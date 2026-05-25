#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void repeat_touch_kernel(const uint8_t *data, uint64_t *out,
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

__global__ void partial_touch_kernel(const uint8_t *remote, const uint8_t *local,
                                     uint64_t *out, size_t remote_pages,
                                     size_t local_pages, size_t stride,
                                     int touches_per_page) {
  size_t total_pages = remote_pages + local_pages;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = total_pages * static_cast<size_t>(touches_per_page);
  if (idx >= total) {
    return;
  }
  size_t page = idx / static_cast<size_t>(touches_per_page);
  size_t rep = idx % static_cast<size_t>(touches_per_page);
  size_t offset = 0;
  uint8_t value = 0;
  if (page < remote_pages) {
    offset = page * stride + ((rep * 128) & (stride - 1));
    value = remote[offset];
  } else {
    size_t local_page = page - remote_pages;
    offset = local_page * stride + ((rep * 128) & (stride - 1));
    value = local[offset];
  }
  out[idx] = static_cast<uint64_t>(value) + static_cast<uint64_t>(idx);
}

struct Args {
  std::string scenario = "repeat";
  std::string remote_kind = "uvm_cpu";
  std::string label = "";
  size_t pages = 262144;
  size_t remote_pages = 0;
  double remote_fraction = 0.10;
  size_t stride = 4096;
  int touches = 1;
  int passes = 8;
  int device = 0;
  bool reset_uvm_each_pass = false;
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
    if (key == "--scenario") {
      args.scenario = need_value();
    } else if (key == "--remote-kind") {
      args.remote_kind = need_value();
    } else if (key == "--label") {
      args.label = need_value();
    } else if (key == "--pages") {
      args.pages = parse_size(need_value());
    } else if (key == "--bytes") {
      args.pages = parse_size(need_value()) / args.stride;
    } else if (key == "--remote-pages") {
      args.remote_pages = parse_size(need_value());
    } else if (key == "--remote-fraction") {
      args.remote_fraction = std::strtod(need_value(), nullptr);
    } else if (key == "--stride") {
      args.stride = parse_size(need_value());
    } else if (key == "--touches") {
      args.touches = std::atoi(need_value());
    } else if (key == "--passes") {
      args.passes = std::atoi(need_value());
    } else if (key == "--device") {
      args.device = std::atoi(need_value());
    } else if (key == "--reset-uvm-each-pass") {
      args.reset_uvm_each_pass = true;
    } else if (key == "--help") {
      std::printf(
          "Usage: %s --scenario repeat|partial --remote-kind uva|uvm_cpu|uvm_cpu_no_advise|uvm_gpu\n"
          "          [--pages N|--bytes N] [--remote-fraction F|--remote-pages N]\n"
          "          [--touches N] [--passes N] [--reset-uvm-each-pass]\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", key.c_str());
      std::exit(2);
    }
  }
  if (args.scenario != "repeat" && args.scenario != "partial") {
    std::fprintf(stderr, "--scenario must be repeat or partial\n");
    std::exit(2);
  }
  if (args.remote_kind != "uva" && args.remote_kind != "uvm_cpu" &&
      args.remote_kind != "uvm_cpu_no_advise" &&
      args.remote_kind != "uvm_gpu") {
    std::fprintf(stderr,
                 "--remote-kind must be uva, uvm_cpu, uvm_cpu_no_advise, or uvm_gpu\n");
    std::exit(2);
  }
  if ((args.stride & (args.stride - 1)) != 0 || args.stride < 128) {
    std::fprintf(stderr, "--stride must be a power of two >= 128\n");
    std::exit(2);
  }
  args.pages = std::max<size_t>(1, args.pages);
  args.touches = std::max(1, args.touches);
  args.passes = std::max(1, args.passes);
  if (args.remote_pages == 0) {
    args.remote_pages = static_cast<size_t>(args.remote_fraction * args.pages);
  }
  args.remote_pages = std::max<size_t>(1, std::min(args.remote_pages, args.pages));
  if (args.label.empty()) {
    args.label = args.scenario + "_" + args.remote_kind;
    if (args.reset_uvm_each_pass) {
      args.label += "_cold_each_pass";
    }
  }
  return args;
}

static double median_ms(std::vector<float> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

static void init_host_pages(uint8_t *ptr, size_t bytes) {
  for (size_t offset = 0; offset < bytes; offset += 4096) {
    ptr[offset] = static_cast<uint8_t>(offset / 4096);
  }
}

static void alloc_remote(const Args &args, size_t bytes, uint8_t **host_ptr,
                         uint8_t **kernel_ptr) {
  *host_ptr = nullptr;
  *kernel_ptr = nullptr;
  if (args.remote_kind == "uva") {
    CHECK_CUDA(cudaHostAlloc(host_ptr, bytes, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostGetDevicePointer(kernel_ptr, *host_ptr, 0));
  } else {
    CHECK_CUDA(cudaMallocManaged(host_ptr, bytes));
    *kernel_ptr = *host_ptr;
  }
  init_host_pages(*host_ptr, bytes);
  if (args.remote_kind == "uvm_cpu") {
    CHECK_CUDA(cudaMemAdvise(*host_ptr, bytes, cudaMemAdviseSetPreferredLocation,
                             cudaCpuDeviceId));
    CHECK_CUDA(cudaMemPrefetchAsync(*host_ptr, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
  } else if (args.remote_kind == "uvm_cpu_no_advise") {
    CHECK_CUDA(cudaMemPrefetchAsync(*host_ptr, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());
  } else if (args.remote_kind == "uvm_gpu") {
    CHECK_CUDA(cudaMemPrefetchAsync(*host_ptr, bytes, args.device));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

static void free_remote(const std::string &kind, uint8_t *host_ptr) {
  if (kind == "uva") {
    CHECK_CUDA(cudaFreeHost(host_ptr));
  } else {
    CHECK_CUDA(cudaFree(host_ptr));
  }
}

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
  CHECK_CUDA(cudaSetDevice(args.device));

  size_t remote_pages = args.scenario == "repeat" ? args.pages : args.remote_pages;
  size_t local_pages = args.scenario == "repeat" ? 0 : args.pages - remote_pages;
  size_t remote_bytes = remote_pages * args.stride;
  size_t local_bytes = local_pages * args.stride;
  size_t total_pages = remote_pages + local_pages;
  size_t total_touches = total_pages * static_cast<size_t>(args.touches);

  uint8_t *remote_host = nullptr;
  uint8_t *remote_kernel = nullptr;
  uint8_t *local_gpu = nullptr;
  uint64_t *out = nullptr;
  alloc_remote(args, remote_bytes, &remote_host, &remote_kernel);
  if (local_pages > 0) {
    CHECK_CUDA(cudaMalloc(&local_gpu, local_bytes));
    CHECK_CUDA(cudaMemset(local_gpu, 7, local_bytes));
  }
  CHECK_CUDA(cudaMalloc(&out, total_touches * sizeof(uint64_t)));

  int block = 256;
  int grid = static_cast<int>((total_touches + block - 1) / block);
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> pass_ms;
  pass_ms.reserve(args.passes);
  for (int pass = 0; pass < args.passes; ++pass) {
    if (args.reset_uvm_each_pass &&
        (args.remote_kind == "uvm_cpu" ||
         args.remote_kind == "uvm_cpu_no_advise")) {
      CHECK_CUDA(cudaMemPrefetchAsync(remote_host, remote_bytes, cudaCpuDeviceId));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemset(out, 0, total_touches * sizeof(uint64_t)));
    CHECK_CUDA(cudaEventRecord(start));
    if (args.scenario == "repeat") {
      repeat_touch_kernel<<<grid, block>>>(remote_kernel, out, remote_pages,
                                           args.stride, args.touches);
    } else {
      partial_touch_kernel<<<grid, block>>>(remote_kernel, local_gpu, out,
                                            remote_pages, local_pages,
                                            args.stride, args.touches);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    pass_ms.push_back(ms);
  }

  std::vector<float> later_ms;
  if (pass_ms.size() > 1) {
    later_ms.assign(pass_ms.begin() + 1, pass_ms.end());
  }
  double first = pass_ms.empty() ? 0.0 : pass_ms.front();
  double med_all = median_ms(pass_ms);
  double med_later = median_ms(later_ms);
  double logical_mib = static_cast<double>(total_pages * args.stride) /
                       (1024.0 * 1024.0);
  double remote_logical_mib = static_cast<double>(remote_bytes) /
                              (1024.0 * 1024.0);
  double actual_read_mib = static_cast<double>(total_touches) /
                           (1024.0 * 1024.0);
  double first_per_page_us = first * 1000.0 / static_cast<double>(total_pages);
  double later_per_page_us = med_later > 0.0
                                 ? med_later * 1000.0 /
                                       static_cast<double>(total_pages)
                                 : 0.0;

  std::printf(
      "{"
      "\"label\":\"%s\","
      "\"scenario\":\"%s\","
      "\"remote_kind\":\"%s\","
      "\"device\":%d,"
      "\"pages\":%zu,"
      "\"remote_pages\":%zu,"
      "\"local_pages\":%zu,"
      "\"stride\":%zu,"
      "\"touches_per_page\":%d,"
      "\"passes\":%d,"
      "\"reset_uvm_each_pass\":%s,"
      "\"logical_span_mib\":%.3f,"
      "\"remote_logical_span_mib\":%.3f,"
      "\"actual_read_mib_per_pass\":%.6f,"
      "\"first_ms\":%.6f,"
      "\"median_all_ms\":%.6f,"
      "\"median_later_ms\":%.6f,"
      "\"first_per_page_us\":%.6f,"
      "\"median_later_per_page_us\":%.6f,"
      "\"pass_ms\":[",
      args.label.c_str(), args.scenario.c_str(), args.remote_kind.c_str(),
      args.device, total_pages, remote_pages, local_pages, args.stride,
      args.touches, args.passes, args.reset_uvm_each_pass ? "true" : "false",
      logical_mib, remote_logical_mib, actual_read_mib, first, med_all,
      med_later, first_per_page_us, later_per_page_us);
  for (size_t i = 0; i < pass_ms.size(); ++i) {
    std::printf("%s%.6f", i == 0 ? "" : ",", pass_ms[i]);
  }
  std::printf("]}\n");

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(out));
  if (local_gpu != nullptr) {
    CHECK_CUDA(cudaFree(local_gpu));
  }
  free_remote(args.remote_kind, remote_host);
  return 0;
}
