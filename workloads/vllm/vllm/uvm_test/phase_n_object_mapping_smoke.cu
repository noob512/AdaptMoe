#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
int uvm_driver_phase_n_get_pool_state(uint32_t pool,
                                      uint32_t gpu_id,
                                      uint64_t *values_out,
                                      size_t values_count);
int uvm_driver_phase_n_query_region_blocks(uint64_t region_id,
                                           uint32_t gpu_id,
                                           uint64_t *values_out,
                                           size_t values_count);
int uvm_driver_phase_n_reclaim_region(uint64_t region_id,
                                      uint32_t gpu_id,
                                      uint32_t async,
                                      uint32_t flags,
                                      uint32_t max_blocks,
                                      uint64_t current_epoch,
                                      uint64_t *values_out,
                                      size_t values_count);
}

namespace {

constexpr uint32_t kPoolKv = 1;
constexpr uint32_t kPoolExpert = 2;
constexpr uint32_t kModeEnforce = 2;
constexpr uint32_t kFlagEvictable = 1u << 0;
constexpr uint64_t kKvRegionBase = 0x510000000000ULL;
constexpr uint64_t kExpertRegionBase = 0x520000000000ULL;
constexpr size_t kQueryValues = 87;
constexpr size_t kReclaimValues = 79;
constexpr size_t kPoolValues = 38;

struct Config {
    int device = 0;
    size_t object_bytes = 2 * 1024 * 1024;
    size_t kv_objects = 2;
    size_t expert_objects = 2;
    int async = 0;
};

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

void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--device N] [--object-bytes N] [--kv-objects N] "
                 "[--expert-objects N] [--async 0|1]\n",
                 argv0);
}

int check_cuda(cudaError_t err, const char *what)
{
    if (err == cudaSuccess)
        return 0;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return 1;
}

int run_step(const char *name, int result)
{
    std::printf("phase_n_object_ioctl step=%s result=%d\n", name, result);
    std::fflush(stdout);
    return result < 0 ? 1 : 0;
}

uintptr_t align_up(uintptr_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
}

bool is_power_of_two(size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

int touch_object(uintptr_t base, size_t object_bytes, size_t index, int value, const char *label)
{
    void *ptr = reinterpret_cast<void *>(base + index * object_bytes);
    size_t touch_bytes = std::min<size_t>(object_bytes, 4096);

    cudaError_t err = cudaMemset(ptr, value, touch_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMemset label=%s object=%zu failed: %s\n",
                     label,
                     index,
                     cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaDeviceSynchronize label=%s object=%zu failed: %s\n",
                     label,
                     index,
                     cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

void print_query(const char *label, const uint64_t *q)
{
    std::printf("phase_n_object_query label=%s region_id=%llu pool=%llu layer=%llu object_id=%llu role=%llu flags=0x%llx logical=%llu populated=%llu resident=%llu pending=%llu materialized=%llu mixed=%llu resident_bytes=%llu pending_bytes=%llu first=0x%llx last=0x%llx sampled=%llu block0_start=0x%llx block0_end=0x%llx block0_state=0x%llx\n",
                label,
                static_cast<unsigned long long>(q[0]),
                static_cast<unsigned long long>(q[3]),
                static_cast<unsigned long long>(q[4]),
                static_cast<unsigned long long>(q[5]),
                static_cast<unsigned long long>(q[6]),
                static_cast<unsigned long long>(q[7]),
                static_cast<unsigned long long>(q[11]),
                static_cast<unsigned long long>(q[12]),
                static_cast<unsigned long long>(q[13]),
                static_cast<unsigned long long>(q[14]),
                static_cast<unsigned long long>(q[15]),
                static_cast<unsigned long long>(q[16]),
                static_cast<unsigned long long>(q[17]),
                static_cast<unsigned long long>(q[18]),
                static_cast<unsigned long long>(q[19]),
                static_cast<unsigned long long>(q[20]),
                static_cast<unsigned long long>(q[21]),
                static_cast<unsigned long long>(q[23]),
                static_cast<unsigned long long>(q[39]),
                static_cast<unsigned long long>(q[71]));
}

void print_reclaim(const char *label, const uint64_t *r)
{
    std::printf("phase_n_object_reclaim label=%s region_id=%llu pool=%llu skipped_protected=%llu block_count=%llu resident_before=%llu resident_after=%llu pending=%llu attempted=%llu success=%llu failure=%llu traced=%llu attempted_bytes=%llu reclaimed_bytes=%llu failed_bytes=%llu status=%llu block0_start=0x%llx block0_end=0x%llx block0_status=%llu\n",
                label,
                static_cast<unsigned long long>(r[0]),
                static_cast<unsigned long long>(r[1]),
                static_cast<unsigned long long>(r[2]),
                static_cast<unsigned long long>(r[3]),
                static_cast<unsigned long long>(r[4]),
                static_cast<unsigned long long>(r[5]),
                static_cast<unsigned long long>(r[6]),
                static_cast<unsigned long long>(r[7]),
                static_cast<unsigned long long>(r[8]),
                static_cast<unsigned long long>(r[9]),
                static_cast<unsigned long long>(r[10]),
                static_cast<unsigned long long>(r[11]),
                static_cast<unsigned long long>(r[12]),
                static_cast<unsigned long long>(r[13]),
                static_cast<unsigned long long>(r[14]),
                static_cast<unsigned long long>(r[15]),
                static_cast<unsigned long long>(r[31]),
                static_cast<unsigned long long>(r[63]));
}

void print_pool(const char *label, const uint64_t *kv, const uint64_t *expert)
{
    std::printf("phase_n_object_pool label=%s kv_budget=%llu kv_resident=%llu kv_pending=%llu kv_over=%llu expert_budget=%llu expert_resident=%llu expert_pending=%llu expert_over=%llu\n",
                label,
                static_cast<unsigned long long>(kv[0]),
                static_cast<unsigned long long>(kv[1]),
                static_cast<unsigned long long>(kv[36]),
                static_cast<unsigned long long>(kv[6]),
                static_cast<unsigned long long>(expert[0]),
                static_cast<unsigned long long>(expert[1]),
                static_cast<unsigned long long>(expert[36]),
                static_cast<unsigned long long>(expert[6]));
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
        else if (std::strcmp(argv[i], "--object-bytes") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &cfg.object_bytes))
                return 2;
        }
        else if (std::strcmp(argv[i], "--kv-objects") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &cfg.kv_objects))
                return 2;
        }
        else if (std::strcmp(argv[i], "--expert-objects") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &cfg.expert_objects))
                return 2;
        }
        else if (std::strcmp(argv[i], "--async") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &cfg.async))
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

    if (!is_power_of_two(cfg.object_bytes) ||
        cfg.object_bytes < 2 * 1024 * 1024 ||
        cfg.kv_objects < 2 ||
        cfg.expert_objects < 2) {
        std::fprintf(stderr, "object_bytes must be power-of-two >= 2MB, and each pool needs at least 2 objects\n");
        return 2;
    }

    const size_t kv_span = cfg.kv_objects * cfg.object_bytes;
    const size_t expert_span = cfg.expert_objects * cfg.object_bytes;
    const size_t kv_alloc_bytes = kv_span + cfg.object_bytes;
    const size_t expert_alloc_bytes = expert_span + cfg.object_bytes;
    const uint64_t kv_budget = static_cast<uint64_t>(kv_span);
    const uint64_t expert_budget = static_cast<uint64_t>(expert_span);
    const uint64_t managed_budget = kv_budget + expert_budget;

    void *kv_raw = nullptr;
    void *expert_raw = nullptr;
    uintptr_t kv_base = 0;
    uintptr_t expert_base = 0;
    uint64_t dyn[8] = {};
    uint64_t kv_state_before[kPoolValues] = {};
    uint64_t expert_state_before[kPoolValues] = {};
    uint64_t kv_state_after[kPoolValues] = {};
    uint64_t expert_state_after[kPoolValues] = {};
    uint64_t kv0_query[kQueryValues] = {};
    uint64_t kv1_query[kQueryValues] = {};
    uint64_t expert0_query[kQueryValues] = {};
    uint64_t expert1_query[kQueryValues] = {};
    uint64_t kv0_after[kQueryValues] = {};
    uint64_t expert0_after[kQueryValues] = {};
    uint64_t kv_reclaim[kReclaimValues] = {};
    uint64_t expert_reclaim[kReclaimValues] = {};
    std::vector<uint64_t> region_ids;
    int failed = 0;

    std::printf("phase_n_object_begin device=%d object_bytes=%zu kv_objects=%zu expert_objects=%zu async=%d\n",
                cfg.device,
                cfg.object_bytes,
                cfg.kv_objects,
                cfg.expert_objects,
                cfg.async);

    failed = check_cuda(cudaSetDevice(cfg.device), "cudaSetDevice");
    if (!failed)
        failed = check_cuda(cudaMallocManaged(&kv_raw, kv_alloc_bytes), "cudaMallocManaged kv");
    if (!failed)
        failed = check_cuda(cudaMallocManaged(&expert_raw, expert_alloc_bytes), "cudaMallocManaged expert");
    if (failed)
        goto cleanup;

    kv_base = align_up(reinterpret_cast<uintptr_t>(kv_raw), cfg.object_bytes);
    expert_base = align_up(reinterpret_cast<uintptr_t>(expert_raw), cfg.object_bytes);
    if (kv_base + kv_span > reinterpret_cast<uintptr_t>(kv_raw) + kv_alloc_bytes ||
        expert_base + expert_span > reinterpret_cast<uintptr_t>(expert_raw) + expert_alloc_bytes) {
        std::fprintf(stderr, "aligned managed subrange overflow\n");
        failed = 1;
        goto cleanup;
    }

    std::printf("phase_n_object_alloc kv_raw=0x%llx kv_base=0x%llx expert_raw=0x%llx expert_base=0x%llx kv_span=%zu expert_span=%zu\n",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(kv_raw)),
                static_cast<unsigned long long>(kv_base),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(expert_raw)),
                static_cast<unsigned long long>(expert_base),
                kv_span,
                expert_span);

    for (size_t i = 0; !failed && i < cfg.kv_objects; ++i) {
        uint64_t region_id = kKvRegionBase + i;
        region_ids.push_back(region_id);
        failed = run_step("register-kv-object",
                          uvm_driver_phase_f_register_region(region_id,
                                                             kv_base + i * cfg.object_bytes,
                                                             cfg.object_bytes,
                                                             kPoolKv,
                                                             0,
                                                             static_cast<uint32_t>(i),
                                                             1,
                                                             kFlagEvictable));
    }

    for (size_t i = 0; !failed && i < cfg.expert_objects; ++i) {
        uint64_t region_id = kExpertRegionBase + i;
        region_ids.push_back(region_id);
        failed = run_step("register-expert-object",
                          uvm_driver_phase_f_register_region(region_id,
                                                             expert_base + i * cfg.object_bytes,
                                                             cfg.object_bytes,
                                                             kPoolExpert,
                                                             static_cast<uint32_t>(i),
                                                             static_cast<uint32_t>(i),
                                                             2,
                                                             kFlagEvictable));
    }

    if (!failed) {
        failed = run_step("set-dynamic-budget",
                          uvm_driver_phase_n_set_dynamic_budgets(0,
                                                                 1,
                                                                 managed_budget,
                                                                 0,
                                                                 0,
                                                                 kv_budget,
                                                                 expert_budget,
                                                                 0,
                                                                 kModeEnforce,
                                                                 0,
                                                                 dyn,
                                                                 8));
    }

    if (!failed)
        failed = touch_object(kv_base, cfg.object_bytes, 0, 0x31, "kv0");
    if (!failed)
        failed = touch_object(kv_base, cfg.object_bytes, 1, 0x32, "kv1");
    if (!failed)
        failed = touch_object(expert_base, cfg.object_bytes, 0, 0x41, "expert0");
    if (!failed)
        failed = touch_object(expert_base, cfg.object_bytes, 1, 0x42, "expert1");

    if (!failed)
        failed = run_step("pool-kv-before", uvm_driver_phase_n_get_pool_state(kPoolKv, 0, kv_state_before, kPoolValues));
    if (!failed)
        failed = run_step("pool-expert-before", uvm_driver_phase_n_get_pool_state(kPoolExpert, 0, expert_state_before, kPoolValues));
    if (!failed)
        print_pool("before_reclaim", kv_state_before, expert_state_before);

    if (!failed)
        failed = run_step("query-kv0", uvm_driver_phase_n_query_region_blocks(kKvRegionBase + 0, 0, kv0_query, kQueryValues));
    if (!failed)
        failed = run_step("query-kv1", uvm_driver_phase_n_query_region_blocks(kKvRegionBase + 1, 0, kv1_query, kQueryValues));
    if (!failed)
        failed = run_step("query-expert0", uvm_driver_phase_n_query_region_blocks(kExpertRegionBase + 0, 0, expert0_query, kQueryValues));
    if (!failed)
        failed = run_step("query-expert1", uvm_driver_phase_n_query_region_blocks(kExpertRegionBase + 1, 0, expert1_query, kQueryValues));

    if (!failed) {
        print_query("kv0_before", kv0_query);
        print_query("kv1_before", kv1_query);
        print_query("expert0_before", expert0_query);
        print_query("expert1_before", expert1_query);
    }

    if (!failed) {
        failed = run_step("reclaim-expert0",
                          uvm_driver_phase_n_reclaim_region(kExpertRegionBase + 0,
                                                            0,
                                                            cfg.async ? 1 : 0,
                                                            0,
                                                            0,
                                                            2,
                                                            expert_reclaim,
                                                            kReclaimValues));
    }
    if (!failed)
        print_reclaim("expert0", expert_reclaim);

    if (!failed) {
        failed = run_step("reclaim-kv0",
                          uvm_driver_phase_n_reclaim_region(kKvRegionBase + 0,
                                                            0,
                                                            cfg.async ? 1 : 0,
                                                            0,
                                                            0,
                                                            3,
                                                            kv_reclaim,
                                                            kReclaimValues));
    }
    if (!failed)
        print_reclaim("kv0", kv_reclaim);

    if (!failed)
        failed = run_step("query-expert0-after", uvm_driver_phase_n_query_region_blocks(kExpertRegionBase + 0, 0, expert0_after, kQueryValues));
    if (!failed)
        failed = run_step("query-kv0-after", uvm_driver_phase_n_query_region_blocks(kKvRegionBase + 0, 0, kv0_after, kQueryValues));
    if (!failed) {
        print_query("expert0_after", expert0_after);
        print_query("kv0_after", kv0_after);
    }

    if (!failed)
        failed = run_step("pool-kv-after", uvm_driver_phase_n_get_pool_state(kPoolKv, 0, kv_state_after, kPoolValues));
    if (!failed)
        failed = run_step("pool-expert-after", uvm_driver_phase_n_get_pool_state(kPoolExpert, 0, expert_state_after, kPoolValues));
    if (!failed)
        print_pool("after_reclaim", kv_state_after, expert_state_after);

    {
        const bool kv_map_ok = kv0_query[3] == kPoolKv &&
                               kv0_query[5] == 0 &&
                               kv0_query[11] >= 1 &&
                               kv0_query[12] >= 1 &&
                               kv0_query[13] >= 1 &&
                               kv1_query[3] == kPoolKv &&
                               kv1_query[5] == 1 &&
                               kv1_query[13] >= 1;
        const bool expert_map_ok = expert0_query[3] == kPoolExpert &&
                                   expert0_query[5] == 0 &&
                                   expert0_query[11] >= 1 &&
                                   expert0_query[12] >= 1 &&
                                   expert0_query[13] >= 1 &&
                                   expert1_query[3] == kPoolExpert &&
                                   expert1_query[5] == 1 &&
                                   expert1_query[13] >= 1;
        const bool expert_reclaim_ok = expert_reclaim[4] >= 1 &&
                                       expert_reclaim[5] == 0 &&
                                       expert_reclaim[8] >= 1 &&
                                       expert_reclaim[9] == 0 &&
                                       expert0_after[13] == 0;
        const bool kv_reclaim_ok = kv_reclaim[4] >= 1 &&
                                   kv_reclaim[5] == 0 &&
                                   kv_reclaim[8] >= 1 &&
                                   kv_reclaim[9] == 0 &&
                                   kv0_after[13] == 0;
        const bool pool_ok = kv_state_after[1] < kv_state_before[1] &&
                             expert_state_after[1] < expert_state_before[1];
        const bool total_ok = kv_map_ok && expert_map_ok &&
                              expert_reclaim_ok && kv_reclaim_ok && pool_ok && !failed;

        std::printf("phase_n_object_final kv_map_ok=%u expert_map_ok=%u expert_reclaim_ok=%u kv_reclaim_ok=%u pool_ok=%u total_ok=%u kv_resident_before=%llu kv_resident_after=%llu expert_resident_before=%llu expert_resident_after=%llu\n",
                    kv_map_ok ? 1 : 0,
                    expert_map_ok ? 1 : 0,
                    expert_reclaim_ok ? 1 : 0,
                    kv_reclaim_ok ? 1 : 0,
                    pool_ok ? 1 : 0,
                    total_ok ? 1 : 0,
                    static_cast<unsigned long long>(kv_state_before[1]),
                    static_cast<unsigned long long>(kv_state_after[1]),
                    static_cast<unsigned long long>(expert_state_before[1]),
                    static_cast<unsigned long long>(expert_state_after[1]));

        if (!total_ok)
            failed = 1;
    }

cleanup:
    for (uint64_t region_id : region_ids)
        (void)uvm_driver_phase_f_unregister_region(region_id);
    if (kv_raw)
        cudaFree(kv_raw);
    if (expert_raw)
        cudaFree(expert_raw);

    return failed ? 1 : 0;
}
