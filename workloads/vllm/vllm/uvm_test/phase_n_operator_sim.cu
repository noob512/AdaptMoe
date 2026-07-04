#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
int uvm_driver_phase_f_register_region(uint64_t region_id,
                                       uintptr_t base,
                                       size_t length,
                                       uint32_t pool,
                                       uint32_t layer_id,
                                       uint32_t expert_id,
                                       uint32_t role,
                                       uint32_t flags);
int uvm_driver_phase_f_unregister_region(uint64_t region_id);
int uvm_driver_phase_n_get_pool_state(uint32_t pool,
                                      uint32_t gpu_id,
                                      uint64_t *values_out,
                                      size_t values_count);
int uvm_driver_phase_n_set_dynamic_budgets(uint32_t gpu_id,
                                           uint64_t epoch,
                                           uint64_t managed_hbm_budget_bytes,
                                           uint64_t safety_margin_bytes,
                                           uint64_t scratch_workspace_bytes,
                                           uint64_t kv_budget_bytes,
                                           uint64_t expert_hot_budget_bytes,
                                           uint64_t expert_streaming_budget_bytes,
                                           uint32_t mode,
                                           uint32_t flags,
                                           uint64_t *values_out,
                                           size_t values_count);
int uvm_driver_phase_n_protect_active_kv(uint32_t gpu_id,
                                         uint64_t epoch,
                                         uint64_t active_kv_bytes,
                                         uint64_t managed_hbm_budget_bytes,
                                         uint64_t safety_margin_bytes,
                                         uint64_t scratch_workspace_bytes,
                                         uint64_t region_id,
                                         uint32_t mode,
                                         uint32_t flags,
                                         uint64_t *values_out,
                                         size_t values_count);
int uvm_driver_phase_n_update_region_state(uint64_t region_id,
                                           uint32_t state,
                                           uint32_t flags_set,
                                           uint32_t flags_clear,
                                           uint32_t priority,
                                           uint64_t last_access_epoch);
}

namespace {

constexpr uint32_t kPoolKv = 1;
constexpr uint32_t kPoolExpert = 2;
constexpr uint64_t kKvRegionId = 4001;
constexpr uint64_t kExpertRegionId = 4002;

struct Config {
    int device = 0;
    size_t block_bytes = 2 * 1024 * 1024;
    size_t logical_blocks = 6;
    size_t resident_blocks = 4;
    int batch = 4;
    int heads = 4;
    int head_dim = 64;
    int hidden = 128;
    int ffn = 512;
    int layers = 4;
    int iterations = 1;
};

void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--device N] [--block-bytes N] [--logical-blocks N] "
                 "[--resident-blocks N] [--batch N] [--heads N] [--head-dim N] "
                 "[--hidden N] [--ffn N] [--layers N] [--iterations N]\n",
                 argv0);
}

bool parse_size(const char *arg, size_t *out)
{
    char *end = nullptr;
    unsigned long long value = std::strtoull(arg, &end, 0);
    if (end == arg || *end != '\0')
        return false;
    *out = static_cast<size_t>(value);
    return true;
}

bool parse_int(const char *arg, int *out)
{
    char *end = nullptr;
    long value = std::strtol(arg, &end, 0);
    if (end == arg || *end != '\0')
        return false;
    *out = static_cast<int>(value);
    return true;
}

int run_step(const char *name, int result)
{
    std::printf("phase_n_operator_ioctl step=%s result=%d\n", name, result);
    std::fflush(stdout);
    return result < 0 ? 1 : 0;
}

int check_cuda(cudaError_t err, const char *what)
{
    if (err == cudaSuccess)
        return 0;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return 1;
}

void fill_managed_pattern(float *ptr, size_t elements, float scale)
{
    for (size_t i = 0; i < elements; ++i)
        ptr[i] = scale * static_cast<float>((i % 251) + 1);
}

__global__ void init_vector_kernel(float *ptr, size_t elements, float scale)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = idx; i < elements; i += stride)
        ptr[i] = scale * static_cast<float>((i % 97) + 1);
}

__global__ void decode_attention_kernel(const float *kv,
                                        const float *query,
                                        float *out,
                                        int batch,
                                        int heads,
                                        int head_dim,
                                        int context_tokens)
{
    __shared__ float partial[256];

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    int tid = threadIdx.x;
    const float *q = query + static_cast<size_t>(bh) * head_dim;
    float acc = 0.0f;

    for (int t = tid; t < context_tokens; t += blockDim.x) {
        size_t base = (static_cast<size_t>(t) * heads + h) *
                      static_cast<size_t>(head_dim) * 2;
        const float *key = kv + base;
        const float *value = kv + base + head_dim;
        float dot = 0.0f;

        for (int d = 0; d < head_dim; ++d)
            dot += q[d] * key[d];

        float weight = 1.0f / (1.0f + fabsf(dot) * 0.001f);
        for (int d = 0; d < head_dim; d += 8)
            acc += weight * value[d] * (0.125f + 0.001f * b);
    }

    partial[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        out[bh] = partial[0];
}

__global__ void moe_expert_kernel(const float *weights,
                                  const float *input,
                                  float *output,
                                  int batch,
                                  int hidden,
                                  int ffn,
                                  size_t block_elements,
                                  int expert_block,
                                  int layer)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= batch || col >= hidden)
        return;

    size_t base = static_cast<size_t>(expert_block) * block_elements;
    const float *in = input + static_cast<size_t>(row) * hidden;
    float acc = 0.0f;

    for (int k = 0; k < ffn; ++k) {
        size_t w1_idx = base + (static_cast<size_t>(col * 131 + k * 17 + layer * 8191) %
                                block_elements);
        size_t w2_idx = base + (static_cast<size_t>(k * hidden + col + layer * 4099) %
                                block_elements);
        float x = in[(k + col + layer) % hidden];
        float gate = fmaxf(weights[w1_idx] * x + 0.01f, 0.0f);
        acc += gate * weights[w2_idx];
    }

    output[static_cast<size_t>(row) * hidden + col] = acc * 0.0001f;
}

int timed_attention(float *kv,
                    float *query,
                    float *out,
                    const Config &cfg,
                    size_t active_kv_blocks,
                    cudaStream_t stream,
                    float *elapsed_ms,
                    float *checksum)
{
    const size_t token_bytes = static_cast<size_t>(2) * cfg.heads * cfg.head_dim * sizeof(float);
    const int tokens_per_block = static_cast<int>(cfg.block_bytes / token_bytes);
    const int context_tokens = tokens_per_block * static_cast<int>(active_kv_blocks);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<float> host_out(static_cast<size_t>(cfg.batch) * cfg.heads, 0.0f);

    if (context_tokens <= 0)
        return 1;

    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate attention start") ||
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate attention stop"))
        return 1;

    if (check_cuda(cudaEventRecord(start, stream), "cudaEventRecord attention start"))
        return 1;

    decode_attention_kernel<<<cfg.batch * cfg.heads, 256, 0, stream>>>(
        kv, query, out, cfg.batch, cfg.heads, cfg.head_dim, context_tokens);

    if (check_cuda(cudaGetLastError(), "decode_attention_kernel launch") ||
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord attention stop") ||
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize attention") ||
        check_cuda(cudaEventElapsedTime(elapsed_ms, start, stop), "cudaEventElapsedTime attention") ||
        check_cuda(cudaMemcpy(host_out.data(),
                              out,
                              host_out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy attention output"))
        return 1;

    *checksum = 0.0f;
    for (float value : host_out)
        *checksum += value;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

int timed_experts(float *weights,
                  float *input,
                  float *output,
                  const Config &cfg,
                  size_t expert_blocks,
                  cudaStream_t stream,
                  float *elapsed_ms,
                  float *checksum)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<float> host_out(static_cast<size_t>(cfg.batch) * cfg.hidden, 0.0f);
    size_t block_elements = cfg.block_bytes / sizeof(float);
    dim3 block(128);
    dim3 grid((cfg.hidden + block.x - 1) / block.x, cfg.batch);

    if (expert_blocks == 0 || block_elements == 0)
        return 1;

    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate expert start") ||
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate expert stop"))
        return 1;

    if (check_cuda(cudaEventRecord(start, stream), "cudaEventRecord expert start"))
        return 1;

    for (int iter = 0; iter < cfg.iterations; ++iter) {
        for (int layer = 0; layer < cfg.layers; ++layer) {
            int expert_block = layer % static_cast<int>(expert_blocks);
            moe_expert_kernel<<<grid, block, 0, stream>>>(
                weights,
                input,
                output,
                cfg.batch,
                cfg.hidden,
                cfg.ffn,
                block_elements,
                expert_block,
                layer + iter * cfg.layers);
        }
    }

    if (check_cuda(cudaGetLastError(), "moe_expert_kernel launch") ||
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord expert stop") ||
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize expert") ||
        check_cuda(cudaEventElapsedTime(elapsed_ms, start, stop), "cudaEventElapsedTime expert") ||
        check_cuda(cudaMemcpy(host_out.data(),
                              output,
                              host_out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy expert output"))
        return 1;

    *checksum = 0.0f;
    for (float value : host_out)
        *checksum += value;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

int get_states(uint64_t *kv_state, uint64_t *expert_state)
{
    std::memset(kv_state, 0, sizeof(uint64_t) * 38);
    std::memset(expert_state, 0, sizeof(uint64_t) * 38);
    int failed = run_step("operator-kv-state", uvm_driver_phase_n_get_pool_state(kPoolKv, 0, kv_state, 38));
    if (!failed)
        failed = run_step("operator-expert-state", uvm_driver_phase_n_get_pool_state(kPoolExpert, 0, expert_state, 38));
    return failed;
}

void print_step(const char *name,
                const char *action,
                const Config &cfg,
                size_t active_kv_blocks,
                float attention_ms,
                float expert_ms,
                float checksum,
                uint64_t *kv_state,
                uint64_t *expert_state,
                uint64_t managed_budget)
{
    size_t token_bytes = static_cast<size_t>(2) * cfg.heads * cfg.head_dim * sizeof(float);
    size_t tokens_per_block = cfg.block_bytes / token_bytes;
    size_t context_tokens = tokens_per_block * active_kv_blocks;

    std::printf("phase_n_operator_step name=%s action=%s batch=%d heads=%d head_dim=%d context_tokens=%zu active_kv_blocks=%zu attention_ms=%.3f expert_ms=%.3f checksum=%.6e kv_budget=%llu kv_resident=%llu kv_pending=%llu kv_over=%llu expert_budget=%llu expert_resident=%llu expert_pending=%llu expert_over=%llu managed_budget=%llu\n",
                name,
                action,
                cfg.batch,
                cfg.heads,
                cfg.head_dim,
                context_tokens,
                active_kv_blocks,
                attention_ms,
                expert_ms,
                checksum,
                static_cast<unsigned long long>(kv_state[0]),
                static_cast<unsigned long long>(kv_state[1]),
                static_cast<unsigned long long>(kv_state[36]),
                static_cast<unsigned long long>(kv_state[6]),
                static_cast<unsigned long long>(expert_state[0]),
                static_cast<unsigned long long>(expert_state[1]),
                static_cast<unsigned long long>(expert_state[36]),
                static_cast<unsigned long long>(expert_state[6]),
                static_cast<unsigned long long>(managed_budget));
}

bool validate_config(const Config &cfg)
{
    if (cfg.block_bytes == 0 ||
        cfg.logical_blocks < 4 ||
        cfg.resident_blocks < 3 ||
        cfg.resident_blocks > cfg.logical_blocks ||
        cfg.batch <= 0 ||
        cfg.heads <= 0 ||
        cfg.head_dim <= 0 ||
        cfg.hidden <= 0 ||
        cfg.ffn <= 0 ||
        cfg.layers <= 0 ||
        cfg.iterations <= 0)
        return false;

    size_t token_bytes = static_cast<size_t>(2) * cfg.heads * cfg.head_dim * sizeof(float);
    if (token_bytes == 0 || cfg.block_bytes < token_bytes)
        return false;

    size_t block_elements = cfg.block_bytes / sizeof(float);
    if (block_elements == 0)
        return false;

    return true;
}

} // namespace

int main(int argc, char **argv)
{
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.device))
                return 2;
        }
        else if (std::strcmp(argv[i], "--block-bytes") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &cfg.block_bytes))
                return 2;
        }
        else if (std::strcmp(argv[i], "--logical-blocks") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &cfg.logical_blocks))
                return 2;
        }
        else if (std::strcmp(argv[i], "--resident-blocks") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &cfg.resident_blocks))
                return 2;
        }
        else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.batch))
                return 2;
        }
        else if (std::strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.heads))
                return 2;
        }
        else if (std::strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.head_dim))
                return 2;
        }
        else if (std::strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.hidden))
                return 2;
        }
        else if (std::strcmp(argv[i], "--ffn") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.ffn))
                return 2;
        }
        else if (std::strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.layers))
                return 2;
        }
        else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.iterations))
                return 2;
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
        else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!validate_config(cfg)) {
        usage(argv[0]);
        std::fprintf(stderr, "invalid configuration\n");
        return 2;
    }

    if (cfg.logical_blocks > static_cast<size_t>(-1) / cfg.block_bytes) {
        std::fprintf(stderr, "allocation size overflow\n");
        return 2;
    }

    const size_t pool_bytes = cfg.logical_blocks * cfg.block_bytes;
    const size_t pool_elements = pool_bytes / sizeof(float);
    const size_t kv_low_blocks = 1;
    const size_t kv_high_blocks = cfg.resident_blocks - 1;
    const size_t expert_high_blocks = cfg.resident_blocks - kv_low_blocks;
    const uint64_t block_bytes_u64 = static_cast<uint64_t>(cfg.block_bytes);
    const uint64_t managed_budget = static_cast<uint64_t>(cfg.resident_blocks) * block_bytes_u64;
    const uint64_t kv_low_budget = static_cast<uint64_t>(kv_low_blocks) * block_bytes_u64;
    const uint64_t kv_high_budget = static_cast<uint64_t>(kv_high_blocks) * block_bytes_u64;
    const uint64_t expert_high_budget = static_cast<uint64_t>(expert_high_blocks) * block_bytes_u64;

    float *kv = nullptr;
    float *expert = nullptr;
    float *query = nullptr;
    float *attention_out = nullptr;
    float *expert_input = nullptr;
    float *expert_out = nullptr;
    cudaStream_t compute_stream = nullptr;
    uint64_t dyn[8] = {};
    uint64_t protect_values[12] = {};
    uint64_t kv_state[38] = {};
    uint64_t expert_state[38] = {};
    int failed = 0;

    std::printf("phase_n_operator_begin device=%d block_bytes=%zu logical_blocks=%zu resident_blocks=%zu batch=%d heads=%d head_dim=%d hidden=%d ffn=%d layers=%d iterations=%d\n",
                cfg.device,
                cfg.block_bytes,
                cfg.logical_blocks,
                cfg.resident_blocks,
                cfg.batch,
                cfg.heads,
                cfg.head_dim,
                cfg.hidden,
                cfg.ffn,
                cfg.layers,
                cfg.iterations);
    std::fflush(stdout);

    failed = check_cuda(cudaSetDevice(cfg.device), "cudaSetDevice");
    if (!failed)
        failed = check_cuda(cudaMallocManaged(&kv, pool_bytes), "cudaMallocManaged kv");
    if (!failed)
        failed = check_cuda(cudaMallocManaged(&expert, pool_bytes), "cudaMallocManaged expert");

    if (!failed) {
        fill_managed_pattern(kv, pool_elements, 0.001f);
        fill_managed_pattern(expert, pool_elements, 0.0005f);
    }

    if (!failed)
        failed = check_cuda(cudaMalloc(&query, static_cast<size_t>(cfg.batch) * cfg.heads * cfg.head_dim * sizeof(float)), "cudaMalloc query");
    if (!failed)
        failed = check_cuda(cudaMalloc(&attention_out, static_cast<size_t>(cfg.batch) * cfg.heads * sizeof(float)), "cudaMalloc attention_out");
    if (!failed)
        failed = check_cuda(cudaMalloc(&expert_input, static_cast<size_t>(cfg.batch) * cfg.hidden * sizeof(float)), "cudaMalloc expert_input");
    if (!failed)
        failed = check_cuda(cudaMalloc(&expert_out, static_cast<size_t>(cfg.batch) * cfg.hidden * sizeof(float)), "cudaMalloc expert_out");
    if (!failed)
        failed = check_cuda(cudaStreamCreate(&compute_stream), "cudaStreamCreate compute_stream");

    if (!failed) {
        init_vector_kernel<<<64, 256, 0, compute_stream>>>(
            query,
            static_cast<size_t>(cfg.batch) * cfg.heads * cfg.head_dim,
            0.0007f);
        init_vector_kernel<<<64, 256, 0, compute_stream>>>(
            expert_input,
            static_cast<size_t>(cfg.batch) * cfg.hidden,
            0.002f);
        failed = check_cuda(cudaGetLastError(), "init_vector_kernel launch") ||
                 check_cuda(cudaStreamSynchronize(compute_stream), "cudaStreamSynchronize init");
    }

    uintptr_t kv_base = reinterpret_cast<uintptr_t>(kv);
    uintptr_t expert_base = reinterpret_cast<uintptr_t>(expert);
    std::printf("phase_n_operator_alloc kv=0x%llx expert=0x%llx pool_bytes=%zu block_bytes=%zu kv_low_blocks=%zu kv_high_blocks=%zu expert_high_blocks=%zu managed_budget=%llu\n",
                static_cast<unsigned long long>(kv_base),
                static_cast<unsigned long long>(expert_base),
                pool_bytes,
                cfg.block_bytes,
                kv_low_blocks,
                kv_high_blocks,
                expert_high_blocks,
                static_cast<unsigned long long>(managed_budget));

    if (!failed) {
        failed = run_step("operator-register-kv",
                          uvm_driver_phase_f_register_region(kKvRegionId,
                                                             kv_base,
                                                             pool_bytes,
                                                             kPoolKv,
                                                             0,
                                                             0,
                                                             1,
                                                             0x1));
    }
    if (!failed) {
        failed = run_step("operator-register-expert",
                          uvm_driver_phase_f_register_region(kExpertRegionId,
                                                             expert_base,
                                                             pool_bytes,
                                                             kPoolExpert,
                                                             0,
                                                             0,
                                                             2,
                                                             0x1));
    }
    if (!failed) {
        failed = run_step("operator-initial-budget",
                          uvm_driver_phase_n_set_dynamic_budgets(0,
                                                                 1,
                                                                 managed_budget,
                                                                 0,
                                                                 0,
                                                                 kv_low_budget,
                                                                 0,
                                                                 0,
                                                                 2,
                                                                 1,
                                                                 dyn,
                                                                 8));
    }

    float attn_ms = 0.0f;
    float expert_ms = 0.0f;
    float attn_checksum = 0.0f;
    float expert_checksum = 0.0f;

    if (!failed)
        failed = timed_attention(kv, query, attention_out, cfg, kv_low_blocks, compute_stream, &attn_ms, &attn_checksum);
    if (!failed)
        failed = timed_experts(expert, expert_input, expert_out, cfg, expert_high_blocks, compute_stream, &expert_ms, &expert_checksum);
    if (!failed)
        failed = get_states(kv_state, expert_state);
    if (!failed)
        print_step("expert_heavy_initial",
                   "small_kv_large_expert_hot",
                   cfg,
                   kv_low_blocks,
                   attn_ms,
                   expert_ms,
                   attn_checksum + expert_checksum,
                   kv_state,
                   expert_state,
                   managed_budget);

    if (!failed) {
        std::memset(protect_values, 0, sizeof(protect_values));
        failed = run_step("operator-protect-kv-heavy",
                          uvm_driver_phase_n_protect_active_kv(0,
                                                               2,
                                                               kv_high_budget,
                                                               managed_budget,
                                                               0,
                                                               0,
                                                               kKvRegionId,
                                                               2,
                                                               0,
                                                               protect_values,
                                                               12));
        std::printf("phase_n_operator_signal name=kv_heavy_decode active_kv=%llu old_kv_budget=%llu new_kv_budget=%llu old_expert_budget=%llu new_expert_budget=%llu expert_over=%llu\n",
                    static_cast<unsigned long long>(kv_high_budget),
                    static_cast<unsigned long long>(protect_values[0]),
                    static_cast<unsigned long long>(protect_values[1]),
                    static_cast<unsigned long long>(protect_values[2]),
                    static_cast<unsigned long long>(protect_values[3]),
                    static_cast<unsigned long long>(protect_values[10]));
    }

    if (!failed)
        failed = timed_attention(kv, query, attention_out, cfg, kv_high_blocks, compute_stream, &attn_ms, &attn_checksum);
    if (!failed)
        failed = get_states(kv_state, expert_state);
    if (!failed)
        print_step("kv_heavy_decode",
                   "protect_active_kv_before_attention",
                   cfg,
                   kv_high_blocks,
                   attn_ms,
                   0.0f,
                   attn_checksum,
                   kv_state,
                   expert_state,
                   managed_budget);

    if (!failed) {
        failed = run_step("operator-unprotect-kv",
                          uvm_driver_phase_n_update_region_state(kKvRegionId,
                                                                 5,
                                                                 0x1,
                                                                 0x6,
                                                                 9,
                                                                 3));
    }

    if (!failed) {
        std::memset(dyn, 0, sizeof(dyn));
        failed = run_step("operator-expert-budget-restore",
                          uvm_driver_phase_n_set_dynamic_budgets(0,
                                                                 3,
                                                                 managed_budget,
                                                                 0,
                                                                 0,
                                                                 kv_low_budget,
                                                                 0,
                                                                 0,
                                                                 2,
                                                                 1,
                                                                 dyn,
                                                                 8));
        std::printf("phase_n_operator_signal name=expert_heavy_recovery kv_budget=%llu expert_budget=%llu kv_over=%llu expert_over=%llu\n",
                    static_cast<unsigned long long>(kv_low_budget),
                    static_cast<unsigned long long>(expert_high_budget),
                    static_cast<unsigned long long>(dyn[5]),
                    static_cast<unsigned long long>(dyn[6]));
    }

    if (!failed)
        failed = timed_experts(expert, expert_input, expert_out, cfg, expert_high_blocks, compute_stream, &expert_ms, &expert_checksum);

    for (int attempt = 0; !failed && attempt < 20; ++attempt) {
        failed = get_states(kv_state, expert_state);
        if (failed)
            break;
        if (kv_state[1] <= kv_state[0] && kv_state[6] == 0 && expert_state[6] == 0)
            break;
        usleep(50000);
    }

    if (!failed)
        print_step("expert_heavy_recovery",
                   "restore_expert_budget",
                   cfg,
                   kv_low_blocks,
                   0.0f,
                   expert_ms,
                   expert_checksum,
                   kv_state,
                   expert_state,
                   managed_budget);

    if (!failed) {
        bool kv_phase_ok = protect_values[1] >= kv_high_budget &&
                           protect_values[3] <= managed_budget - kv_high_budget;
        bool final_kv_ok = kv_state[0] == kv_low_budget &&
                           kv_state[1] <= kv_state[0] &&
                           kv_state[6] == 0;
        bool final_expert_ok = expert_state[0] == expert_high_budget &&
                               expert_state[1] >= expert_high_budget &&
                               expert_state[6] == 0;
        bool total_ok = kv_state[1] + expert_state[1] <= managed_budget;

        std::printf("phase_n_operator_final kv_phase_ok=%u final_kv_ok=%u final_expert_ok=%u total_ok=%u final_kv_budget=%llu final_kv_resident=%llu final_expert_budget=%llu final_expert_resident=%llu managed_budget=%llu checksum=%.6e\n",
                    kv_phase_ok ? 1 : 0,
                    final_kv_ok ? 1 : 0,
                    final_expert_ok ? 1 : 0,
                    total_ok ? 1 : 0,
                    static_cast<unsigned long long>(kv_state[0]),
                    static_cast<unsigned long long>(kv_state[1]),
                    static_cast<unsigned long long>(expert_state[0]),
                    static_cast<unsigned long long>(expert_state[1]),
                    static_cast<unsigned long long>(managed_budget),
                    attn_checksum + expert_checksum);
        if (!kv_phase_ok || !final_kv_ok || !final_expert_ok || !total_ok)
            failed = 1;
    }

    (void)uvm_driver_phase_f_unregister_region(kKvRegionId);
    (void)uvm_driver_phase_f_unregister_region(kExpertRegionId);
    if (compute_stream)
        cudaStreamDestroy(compute_stream);
    if (query)
        cudaFree(query);
    if (attention_out)
        cudaFree(attention_out);
    if (expert_input)
        cudaFree(expert_input);
    if (expert_out)
        cudaFree(expert_out);
    if (kv)
        cudaFree(kv);
    if (expert)
        cudaFree(expert);

    return failed;
}
