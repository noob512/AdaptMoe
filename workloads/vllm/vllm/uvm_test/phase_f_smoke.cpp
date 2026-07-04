#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

// 这些函数由 uvm_allocator.cpp 以 extern "C" 导出。
// phase_f_smoke.cpp 本身只负责：
//   1. 解析命令行参数；
//   2. 按 --step 选择测试阶段；
//   3. 在需要真实 UVM range 的阶段创建 cudaMallocManaged buffer；
//   4. 调用 uvm_allocator.cpp 中的用户态 driver/ioctl helper。
//
// 重要分层：
//   - ping-open / ping-init / ping-small / batch-mini0 / ping-ioctl / ping
//     都不需要 cudaMallocManaged buffer；
//   - batch-mini1 / query / prefetch-gpu / advise-cpu / prefetch-cpu
//     需要先分配 managed memory，再把地址和长度传给 driver helper。
extern "C" {
// 只打开 /dev/nvidia-uvm，然后关闭。用于验证设备节点和权限。
int uvm_driver_phase_f_open_only();

// 打开 /dev/nvidia-uvm 后执行 UVM_INITIALIZE_IOCTL。
int uvm_driver_phase_f_init_only();

// 走最小 Phase F ping ioctl。参数结构很小，只用于验证 test ioctl 路由。
int uvm_driver_phase_f_small_ping();

// 走 Phase F mini batch ioctl，count=0，不传真实地址。
int uvm_driver_phase_f_batch_mini0();

// 走 Phase F mini batch ioctl，count=1，传入一个真实 managed range。
int uvm_driver_phase_f_batch_mini1(uintptr_t base, size_t length);

// 走正式 Phase F batch executor ioctl 的空请求路径，count=0。
// 当前 --step ping-ioctl 调用的就是它。
int uvm_driver_phase_f_ioctl_only();

// 也是正式 Phase F batch executor 的 ping 路径，目前和 ping-ioctl 同类。
int uvm_driver_phase_f_ping();

// 以下 batch helper 会对真实 managed ranges 执行 Phase F 操作：
//   prefetch_cpu / prefetch_gpu -> driver 内部迁移；
//   advise_cpu                 -> 设置 preferred location；
//   query_residency            -> 查询 range 当前 resident 位置。
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
int uvm_driver_batch_query_residency_bytes(const uintptr_t* ptrs,
                                           const size_t* sizes,
                                           size_t count,
                                           int device,
                                           uint64_t* gpu_bytes_out,
                                           uint64_t* cpu_bytes_out,
                                           uint64_t* any_bytes_out,
                                           uint64_t* unknown_bytes_out,
                                           uint64_t* queried_bytes_out);
int uvm_driver_phase_f_register_region(uint64_t region_id,
                                       uintptr_t base,
                                       size_t length,
                                       uint32_t pool,
                                       uint32_t layer_id,
                                       uint32_t expert_id,
                                       uint32_t role,
                                       uint32_t flags);
int uvm_driver_phase_f_unregister_region(uint64_t region_id);
int uvm_driver_batch_query_residency_regions(const uint64_t* region_ids,
                                             size_t count,
                                             int device);
int uvm_driver_phase_f_register_query_region(uint64_t region_id,
                                             uintptr_t base,
                                             size_t length,
                                             int device);
int uvm_driver_phase_n_set_pool_budget(uint32_t pool,
                                       uint32_t gpu_id,
                                       uint64_t budget_bytes,
                                       uint32_t mode,
                                       uint32_t flags);
int uvm_driver_phase_n_get_pool_state(uint32_t pool,
                                      uint32_t gpu_id,
                                      uint64_t* values_out,
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
                                           uint64_t* values_out,
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
                                         uint64_t* values_out,
                                         size_t values_count);
int uvm_driver_phase_n_metadata_state_smoke(uint32_t pool,
                                            uint64_t budget_bytes,
                                            uint64_t* values_out,
                                            size_t values_count);
int uvm_driver_phase_n_reclaim_pool(uint32_t pool,
                                    uint32_t gpu_id,
                                    uint64_t target_bytes,
                                    uint64_t current_epoch,
                                    uint64_t cooldown_epochs,
                                    uint32_t max_victims,
                                    uint32_t async,
                                    uint32_t flags,
                                    uint64_t* values_out,
                                    size_t values_count);
int uvm_driver_phase_n_update_region_state(uint64_t region_id,
                                           uint32_t state,
                                           uint32_t flags_set,
                                           uint32_t flags_clear,
                                           uint32_t priority,
                                           uint64_t last_access_epoch);
int uvm_driver_phase_n_query_region(uint64_t region_id,
                                    uint64_t* values_out,
                                    size_t values_count);
int uvm_driver_phase_n_query_region_by_va(uintptr_t address,
                                          uint64_t* values_out,
                                          size_t values_count);
int uvm_driver_phase_f_probe_existing_uvm_fd();
}

namespace {

// 打印命令行帮助。这里只展示 phase_f_smoke 支持的 step，
// shell 脚本 run_stage_m_phase_f_smoke.sh 会把同名 --step 继续传进来。
void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--step all|ping-open|ping-init|ping-small|batch-mini0|batch-mini1|ping-ioctl|ping|probe-uvm-fd|register-region|query-region|phase-n-region|phase-n-metadata-state|phase-n-reclaim|phase-n-block-quota|phase-n-dynamic-budget|phase-n-dynamic-budget-auto|phase-n-kv-protect|phase-n-synthetic-planner|query|prefetch-gpu|prefetch-gpu-query|advise-cpu|prefetch-cpu] "
                 "[--bytes N] [--device N] [--async 0|1] [--pool N] [--blocks N] [--budget-blocks N]\n",
                 argv0);
}

// 判断当前是否应该执行某个 step。
// requested == "all" 时，每个 step 都会被执行；
// requested == step 时，只执行对应单步。
bool want_step(const std::string& requested, const char* step) {
    return requested == "all" || requested == step;
}

// 统一打印单个 step 的返回值，并把负数转换成进程失败。
// uvm_allocator.cpp 中的 helper 约定：
//   >= 0 表示成功或成功数量；
//   < 0  表示错误。
int run_step(const char* name, int result) {
    std::printf("phase_f_smoke step=%s result=%d\n", name, result);
    std::fflush(stdout);
    return result < 0 ? 1 : 0;
}

// 用户态 trace。stdout 已经被 shell 脚本重定向到主日志，
// 所以每个 trace 都会落到 latest_<step>.log 中。
void trace(const char* msg) {
    std::printf("phase_f_smoke trace=%s\n", msg);
    std::fflush(stdout);
    fsync(fileno(stdout));
}

}  // namespace

int main(int argc, char** argv) {
    // 默认跑完整 all 流程；脚本通常会显式传 --step ping-ioctl 进行单步定位。
    std::string step = "all";

    // bytes/device/async 只有部分 step 真正使用：
    //   - bytes: managed allocation 大小；
    //   - device: CUDA device id；
    //   - async: prefetch/migrate 类操作是否设置 async 标志。
    //
    // 对 --step ping-ioctl 来说，这三个参数只会被解析和打印，
    // 不会触发 cudaSetDevice/cudaMallocManaged。
    size_t bytes = 1 << 20;
    int device = 0;
    int async = 0;
    uint32_t pool = 1;
    size_t blocks = 4;
    size_t budget_blocks = 2;

    // 简单命令行解析。未知参数或缺少值时打印 usage 并返回 2。
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
        else if (std::strcmp(argv[i], "--pool") == 0 && i + 1 < argc) {
            pool = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
        }
        else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            blocks = std::strtoull(argv[++i], nullptr, 0);
        }
        else if (std::strcmp(argv[i], "--budget-blocks") == 0 && i + 1 < argc) {
            budget_blocks = std::strtoull(argv[++i], nullptr, 0);
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

    // 打印本次配置。若日志中看不到这一行，说明进程可能还没进入 main，
    // 或者动态链接/程序启动阶段已经失败。
    std::printf("phase_f_smoke begin step=%s bytes=%zu device=%d async=%d pool=%u blocks=%zu budget_blocks=%zu\n",
                step.c_str(), bytes, device, async, pool, blocks, budget_blocks);
    std::fflush(stdout);
    trace("main_after_parse");

    // Step 1: 只打开 /dev/nvidia-uvm。
    // 失败通常指向设备节点不存在、权限问题、driver 未加载等。
    if (want_step(step, "ping-open")) {
        trace("before_ping_open");
        int failed = run_step("ping-open", uvm_driver_phase_f_open_only());
        trace("after_ping_open");
        if (failed || step == "ping-open") return failed;
    }

    // Step 2: open + UVM_INITIALIZE_IOCTL。
    // 用来验证当前进程能初始化 UVM VA space/file state。
    if (want_step(step, "ping-init")) {
        trace("before_ping_init");
        int failed = run_step("ping-init", uvm_driver_phase_f_init_only());
        trace("after_ping_init");
        if (failed || step == "ping-init") return failed;
    }

    // Step 3: 最小 Phase F ping ioctl。
    // 这个路径不走 batch executor，只验证新增 test ioctl 的最小路由和参数 copy。
    if (want_step(step, "ping-small")) {
        trace("before_ping_small");
        int failed = run_step("ping-small", uvm_driver_phase_f_small_ping());
        trace("after_ping_small");
        if (failed || step == "ping-small") return failed;
    }

    // Step 4: mini batch ioctl，count=0。
    // 不触碰真实 managed range，适合验证 mini batch ioctl 的空请求路径。
    if (want_step(step, "batch-mini0")) {
        trace("before_batch_mini0");
        int failed = run_step("batch-mini0", uvm_driver_phase_f_batch_mini0());
        trace("after_batch_mini0");
        if (failed || step == "batch-mini0") return failed;
    }

    // Step 5: 正式 Phase F batch executor ioctl，count=0。
    // 这是当前 crash 定位里最重要的单步：
    //   phase_f_smoke --step ping-ioctl
    //
    // 它不会调用 cudaSetDevice，也不会分配 cudaMallocManaged。
    // 内部只会：
    //   1. open /dev/nvidia-uvm；
    //   2. UVM_INITIALIZE_IOCTL；
    //   3. UVM_TEST_PHASE_F_BATCH_EXECUTOR ioctl，count=0；
    //   4. close fd。
    //
    // 如果日志停在 before_ping_ioctl 之后，重点怀疑 uvm_allocator.cpp
    // 中 uvm_driver_phase_f_ping_internal() 的 open/init/ioctl 阶段。
    if (want_step(step, "ping-ioctl")) {
        trace("before_ping_ioctl");
        sleep(3);  // 给足够时间 attach profiler 或者 strace 定位 ioctl 阶段的 crash。
        int failed = run_step("ping-ioctl", uvm_driver_phase_f_ioctl_only());
        trace("after_ping_ioctl");
        if (failed || step == "ping-ioctl") return failed;
    }

    // Step 6: batch executor ping 路径。
    // 目前它和 ping-ioctl 的定位价值相近，保留为另一个语义入口。
    if (want_step(step, "ping")) {
        trace("before_ping");
        int failed = run_step("ping", uvm_driver_phase_f_ping());
        trace("after_ping");
        if (failed || step == "ping") return failed;
    }

    if (want_step(step, "phase-n-metadata-state")) {
        uint64_t values[36] = {};
        trace("before_phase_n_metadata_state");
        int failed = run_step(
            "phase-n-metadata-state",
            uvm_driver_phase_n_metadata_state_smoke(
                pool,
                bytes / 2,
                values,
                36));
        std::printf("phase_n_metadata_state pool=%u budget=%llu resident_gpu=%llu resident_cpu=%llu reclaim=%llu protected=%llu evictable=%llu over=%llu target=%llu candidate_bytes=%llu candidate_regions=%llu selected_id=%llu selected_bytes=%llu victim_count=%llu metadata_only=%llu corrections=%llu mode=%llu flags=0x%llx\n",
                    pool,
                    static_cast<unsigned long long>(values[0]),
                    static_cast<unsigned long long>(values[1]),
                    static_cast<unsigned long long>(values[2]),
                    static_cast<unsigned long long>(values[3]),
                    static_cast<unsigned long long>(values[4]),
                    static_cast<unsigned long long>(values[5]),
                    static_cast<unsigned long long>(values[6]),
                    static_cast<unsigned long long>(values[7]),
                    static_cast<unsigned long long>(values[8]),
                    static_cast<unsigned long long>(values[9]),
                    static_cast<unsigned long long>(values[10]),
                    static_cast<unsigned long long>(values[11]),
                    static_cast<unsigned long long>(values[12]),
                    static_cast<unsigned long long>(values[13]),
                    static_cast<unsigned long long>(values[33]),
                    static_cast<unsigned long long>(values[34]),
                    static_cast<unsigned long long>(values[35]));
        trace("after_phase_n_metadata_state");
        if (failed || step == "phase-n-metadata-state") return failed;
    }

    if (step == "phase-n-dynamic-budget" ||
        step == "phase-n-dynamic-budget-auto" ||
        step == "phase-n-kv-protect") {
        const bool auto_budget = step == "phase-n-dynamic-budget-auto";
        const bool kv_protect = step == "phase-n-kv-protect";
        if (bytes == 0 || blocks < 2 || budget_blocks == 0 || budget_blocks >= blocks) {
            std::fprintf(stderr, "%s requires --bytes > 0 and --blocks > --budget-blocks > 0\n",
                         step.c_str());
            return 2;
        }
        if (blocks > static_cast<size_t>(-1) / bytes) {
            std::fprintf(stderr, "%s allocation size overflow\n", step.c_str());
            return 2;
        }

        trace("before_dynamic_cuda_set_device");
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
            return 3;
        }

        const size_t pool_bytes = bytes * blocks;
        void* kv_ptr = nullptr;
        void* expert_ptr = nullptr;
        trace("before_dynamic_cuda_malloc_managed");
        err = cudaMallocManaged(&kv_ptr, pool_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMallocManaged kv failed: %s\n", cudaGetErrorString(err));
            return 4;
        }
        err = cudaMallocManaged(&expert_ptr, pool_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMallocManaged expert failed: %s\n", cudaGetErrorString(err));
            cudaFree(kv_ptr);
            return 4;
        }

        uintptr_t kv_base = reinterpret_cast<uintptr_t>(kv_ptr);
        uintptr_t expert_base = reinterpret_cast<uintptr_t>(expert_ptr);
        const uint64_t kv_region_id = 2001;
        const uint64_t expert_region_id = 2002;
        const uint64_t kv_initial_budget = static_cast<uint64_t>(budget_blocks) * static_cast<uint64_t>(bytes);
        const uint64_t expert_initial_budget = static_cast<uint64_t>(blocks) * static_cast<uint64_t>(bytes);
        const uint64_t kv_after_budget = static_cast<uint64_t>(blocks) * static_cast<uint64_t>(bytes);
        const uint64_t expert_after_budget = static_cast<uint64_t>(budget_blocks) * static_cast<uint64_t>(bytes);
        const uint64_t managed_budget = kv_after_budget + expert_after_budget;
        uint64_t dyn[8] = {};
        uint64_t kv_state[38] = {};
        uint64_t expert_state[38] = {};
        uint64_t reclaim_values[45] = {};
        uint64_t protect_values[12] = {};

        std::printf("phase_n_dynamic_budget_alloc kv=0x%llx expert=0x%llx pool_bytes=%zu block_bytes=%zu blocks=%zu budget_blocks=%zu\n",
                    static_cast<unsigned long long>(kv_base),
                    static_cast<unsigned long long>(expert_base),
                    pool_bytes,
                    bytes,
                    blocks,
                    budget_blocks);

        auto cleanup_dynamic = [&]() {
            (void)uvm_driver_phase_f_unregister_region(kv_region_id);
            (void)uvm_driver_phase_f_unregister_region(expert_region_id);
            cudaFree(kv_ptr);
            cudaFree(expert_ptr);
        };

        auto touch_block = [&](void* ptr, size_t block_index, int value, const char* label) -> int {
            char* block_ptr = static_cast<char*>(ptr) + block_index * bytes;
            size_t touch_bytes = bytes < 4096 ? bytes : 4096;
            cudaError_t local_err = cudaMemset(block_ptr, value, touch_bytes);
            if (local_err != cudaSuccess) {
                std::fprintf(stderr, "cudaMemset %s block=%zu failed: %s\n",
                             label, block_index, cudaGetErrorString(local_err));
                return 6;
            }
            local_err = cudaDeviceSynchronize();
            if (local_err != cudaSuccess) {
                std::fprintf(stderr, "cudaDeviceSynchronize %s block=%zu failed: %s\n",
                             label, block_index, cudaGetErrorString(local_err));
                return 7;
            }
            return 0;
        };

        trace("before_phase_n_dynamic_budget");
        int failed = run_step(
            "phase-n-dynamic-register-kv",
            uvm_driver_phase_f_register_region(kv_region_id,
                                               kv_base,
                                               pool_bytes,
                                               1,
                                               0,
                                               0,
                                               1,
                                               0x1));
        if (!failed) {
            failed = run_step(
                "phase-n-dynamic-register-expert",
                uvm_driver_phase_f_register_region(expert_region_id,
                                                   expert_base,
                                                   pool_bytes,
                                                   2,
                                                   0,
                                                   0,
                                                   2,
                                                   0x1));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-dynamic-initial-budget",
                uvm_driver_phase_n_set_dynamic_budgets(0,
                                                       1,
                                                       managed_budget,
                                                       0,
                                                       0,
                                                       kv_initial_budget,
                                                       expert_initial_budget,
                                                       0,
                                                       2,
                                                       0,
                                                       dyn,
                                                       8));
            std::printf("phase_n_dynamic_budget_initial kv_budget=%llu expert_budget=%llu managed=%llu kv_resident=%llu kv_pending=%llu expert_resident=%llu expert_pending=%llu shared_over=%llu kv_over=%llu expert_over=%llu\n",
                        static_cast<unsigned long long>(kv_initial_budget),
                        static_cast<unsigned long long>(expert_initial_budget),
                        static_cast<unsigned long long>(managed_budget),
                        static_cast<unsigned long long>(dyn[0]),
                        static_cast<unsigned long long>(dyn[1]),
                        static_cast<unsigned long long>(dyn[2]),
                        static_cast<unsigned long long>(dyn[3]),
                        static_cast<unsigned long long>(dyn[4]),
                        static_cast<unsigned long long>(dyn[5]),
                        static_cast<unsigned long long>(dyn[6]));
        }

        for (size_t i = 0; !failed && i < budget_blocks; ++i)
            failed = touch_block(kv_ptr, i, 0x41, "kv_initial");
        for (size_t i = 0; !failed && i < blocks; ++i)
            failed = touch_block(expert_ptr, i, 0x51, "expert_initial");

        if (!failed) {
            failed = run_step(
                "phase-n-dynamic-kv-state-before",
                uvm_driver_phase_n_get_pool_state(1, 0, kv_state, 38));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-dynamic-expert-state-before",
                uvm_driver_phase_n_get_pool_state(2, 0, expert_state, 38));
        }
        if (!failed) {
            std::printf("phase_n_dynamic_budget_before kv_budget=%llu kv_resident=%llu kv_pending=%llu kv_over=%llu expert_budget=%llu expert_resident=%llu expert_pending=%llu expert_over=%llu\n",
                        static_cast<unsigned long long>(kv_state[0]),
                        static_cast<unsigned long long>(kv_state[1]),
                        static_cast<unsigned long long>(kv_state[36]),
                        static_cast<unsigned long long>(kv_state[6]),
                        static_cast<unsigned long long>(expert_state[0]),
                        static_cast<unsigned long long>(expert_state[1]),
                        static_cast<unsigned long long>(expert_state[36]),
                        static_cast<unsigned long long>(expert_state[6]));
        }

        if (!failed && kv_protect) {
            failed = run_step(
                "phase-n-kv-protect-active",
                uvm_driver_phase_n_protect_active_kv(0,
                                                     2,
                                                     kv_after_budget,
                                                     managed_budget,
                                                     0,
                                                     0,
                                                     kv_region_id,
                                                     2,
                                                     0,
                                                     protect_values,
                                                     12));
            std::printf("phase_n_kv_protect_signal active_kv=%llu managed=%llu old_kv_budget=%llu new_kv_budget=%llu old_expert_budget=%llu new_expert_budget=%llu kv_resident=%llu kv_pending=%llu expert_resident=%llu expert_pending=%llu shared_over=%llu kv_over=%llu expert_over=%llu\n",
                        static_cast<unsigned long long>(kv_after_budget),
                        static_cast<unsigned long long>(managed_budget),
                        static_cast<unsigned long long>(protect_values[0]),
                        static_cast<unsigned long long>(protect_values[1]),
                        static_cast<unsigned long long>(protect_values[2]),
                        static_cast<unsigned long long>(protect_values[3]),
                        static_cast<unsigned long long>(protect_values[4]),
                        static_cast<unsigned long long>(protect_values[5]),
                        static_cast<unsigned long long>(protect_values[6]),
                        static_cast<unsigned long long>(protect_values[7]),
                        static_cast<unsigned long long>(protect_values[8]),
                        static_cast<unsigned long long>(protect_values[9]),
                        static_cast<unsigned long long>(protect_values[10]));
        }
        else if (!failed) {
            std::memset(dyn, 0, sizeof(dyn));
            failed = run_step(
                auto_budget ? "phase-n-dynamic-auto-shift-budget" : "phase-n-dynamic-shift-budget",
                uvm_driver_phase_n_set_dynamic_budgets(0,
                                                       2,
                                                       managed_budget,
                                                       0,
                                                       0,
                                                       kv_after_budget,
                                                       auto_budget ? 0 : expert_after_budget,
                                                       0,
                                                       2,
                                                       auto_budget ? 1 : 0,
                                                       dyn,
                                                       8));
            std::printf("%s kv_budget=%llu expert_budget=%llu managed=%llu kv_resident=%llu kv_pending=%llu expert_resident=%llu expert_pending=%llu shared_over=%llu kv_over=%llu expert_over=%llu flags=0x%x\n",
                        auto_budget ? "phase_n_dynamic_budget_auto_shift" : "phase_n_dynamic_budget_shift",
                        static_cast<unsigned long long>(kv_after_budget),
                        static_cast<unsigned long long>(expert_after_budget),
                        static_cast<unsigned long long>(managed_budget),
                        static_cast<unsigned long long>(dyn[0]),
                        static_cast<unsigned long long>(dyn[1]),
                        static_cast<unsigned long long>(dyn[2]),
                        static_cast<unsigned long long>(dyn[3]),
                        static_cast<unsigned long long>(dyn[4]),
                        static_cast<unsigned long long>(dyn[5]),
                        static_cast<unsigned long long>(dyn[6]),
                        auto_budget ? 1 : 0);
        }

        if (!failed && !auto_budget && !kv_protect) {
            failed = run_step(
                "phase-n-dynamic-reclaim-expert",
                uvm_driver_phase_n_reclaim_pool(2, 0, 0, 2, 0, 8, async, 0, reclaim_values, 45));
            std::printf("phase_n_dynamic_budget_reclaim_expert attempted=%llu success=%llu failure=%llu victim_count=%llu budget=%llu resident_before=%llu over=%llu target=%llu attempted_bytes=%llu reclaimed=%llu failed_bytes=%llu\n",
                        static_cast<unsigned long long>(reclaim_values[0]),
                        static_cast<unsigned long long>(reclaim_values[1]),
                        static_cast<unsigned long long>(reclaim_values[2]),
                        static_cast<unsigned long long>(reclaim_values[3]),
                        static_cast<unsigned long long>(reclaim_values[4]),
                        static_cast<unsigned long long>(reclaim_values[5]),
                        static_cast<unsigned long long>(reclaim_values[6]),
                        static_cast<unsigned long long>(reclaim_values[7]),
                        static_cast<unsigned long long>(reclaim_values[8]),
                        static_cast<unsigned long long>(reclaim_values[9]),
                        static_cast<unsigned long long>(reclaim_values[10]));
        }
        else if (!failed) {
            std::printf("%s manual_reclaim=0 reason=auto_reclaim_expected\n",
                        kv_protect ? "phase_n_kv_protect" : "phase_n_dynamic_budget_auto");
        }

        for (size_t i = budget_blocks; !failed && i < blocks; ++i)
            failed = touch_block(kv_ptr, i, 0x42, "kv_after_shift");

        if (!failed) {
            std::memset(kv_state, 0, sizeof(kv_state));
            std::memset(expert_state, 0, sizeof(expert_state));
            failed = run_step(
                "phase-n-dynamic-kv-state-after",
                uvm_driver_phase_n_get_pool_state(1, 0, kv_state, 38));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-dynamic-expert-state-after",
                uvm_driver_phase_n_get_pool_state(2, 0, expert_state, 38));
        }
        if (!failed) {
            std::printf("%s kv_budget=%llu kv_resident=%llu kv_pending=%llu kv_over=%llu expert_budget=%llu expert_resident=%llu expert_pending=%llu expert_over=%llu\n",
                        kv_protect ? "phase_n_kv_protect_after" :
                                     (auto_budget ? "phase_n_dynamic_budget_auto_after" :
                                                    "phase_n_dynamic_budget_after"),
                        static_cast<unsigned long long>(kv_state[0]),
                        static_cast<unsigned long long>(kv_state[1]),
                        static_cast<unsigned long long>(kv_state[36]),
                        static_cast<unsigned long long>(kv_state[6]),
                        static_cast<unsigned long long>(expert_state[0]),
                        static_cast<unsigned long long>(expert_state[1]),
                        static_cast<unsigned long long>(expert_state[36]),
                        static_cast<unsigned long long>(expert_state[6]));
        }

        cleanup_dynamic();
        trace("after_phase_n_dynamic_budget");
        return failed;
    }

    if (step == "phase-n-synthetic-planner") {
        if (bytes == 0 || blocks < 4 || budget_blocks < 3 || budget_blocks > blocks) {
            std::fprintf(stderr,
                         "phase-n-synthetic-planner requires --bytes > 0 and --blocks >= --budget-blocks >= 3, with --blocks >= 4\n");
            return 2;
        }
        if (blocks > static_cast<size_t>(-1) / bytes) {
            std::fprintf(stderr, "phase-n-synthetic-planner allocation size overflow\n");
            return 2;
        }

        trace("before_synthetic_planner_cuda_set_device");
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
            return 3;
        }

        const size_t pool_bytes = bytes * blocks;
        const uint64_t block_bytes = static_cast<uint64_t>(bytes);
        const uint64_t managed_budget = static_cast<uint64_t>(budget_blocks) * block_bytes;
        const size_t kv_low_blocks = 1;
        const size_t kv_high_blocks = budget_blocks > 1 ? budget_blocks - 1 : 1;
        const size_t expert_high_blocks = budget_blocks - kv_low_blocks;
        void* kv_ptr = nullptr;
        void* expert_ptr = nullptr;
        uintptr_t kv_base = 0;
        uintptr_t expert_base = 0;
        const uint64_t kv_region_id = 3001;
        const uint64_t expert_region_id = 3002;
        uint64_t dyn[8] = {};
        uint64_t protect_values[12] = {};
        uint64_t kv_state[38] = {};
        uint64_t expert_state[38] = {};

        trace("before_synthetic_planner_cuda_malloc_managed");
        err = cudaMallocManaged(&kv_ptr, pool_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMallocManaged kv failed: %s\n", cudaGetErrorString(err));
            return 4;
        }
        err = cudaMallocManaged(&expert_ptr, pool_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMallocManaged expert failed: %s\n", cudaGetErrorString(err));
            cudaFree(kv_ptr);
            return 4;
        }

        kv_base = reinterpret_cast<uintptr_t>(kv_ptr);
        expert_base = reinterpret_cast<uintptr_t>(expert_ptr);

        std::printf("phase_n_synthetic_planner_alloc kv=0x%llx expert=0x%llx logical_blocks=%zu managed_blocks=%zu block_bytes=%llu managed_budget=%llu\n",
                    static_cast<unsigned long long>(kv_base),
                    static_cast<unsigned long long>(expert_base),
                    blocks,
                    budget_blocks,
                    static_cast<unsigned long long>(block_bytes),
                    static_cast<unsigned long long>(managed_budget));

        auto cleanup_synthetic = [&]() {
            (void)uvm_driver_phase_f_unregister_region(kv_region_id);
            (void)uvm_driver_phase_f_unregister_region(expert_region_id);
            cudaFree(kv_ptr);
            cudaFree(expert_ptr);
        };

        auto touch_block = [&](void* ptr, size_t block_index, int value, const char* label) -> int {
            char* block_ptr = static_cast<char*>(ptr) + block_index * bytes;
            size_t touch_bytes = bytes < 4096 ? bytes : 4096;
            cudaError_t local_err = cudaMemset(block_ptr, value, touch_bytes);
            if (local_err != cudaSuccess) {
                std::fprintf(stderr, "cudaMemset %s block=%zu failed: %s\n",
                             label, block_index, cudaGetErrorString(local_err));
                return 6;
            }
            local_err = cudaDeviceSynchronize();
            if (local_err != cudaSuccess) {
                std::fprintf(stderr, "cudaDeviceSynchronize %s block=%zu failed: %s\n",
                             label, block_index, cudaGetErrorString(local_err));
                return 7;
            }
            return 0;
        };

        auto get_states = [&]() -> int {
            std::memset(kv_state, 0, sizeof(kv_state));
            std::memset(expert_state, 0, sizeof(expert_state));
            int failed = run_step("phase-n-synthetic-kv-state",
                                  uvm_driver_phase_n_get_pool_state(1, 0, kv_state, 38));
            if (!failed) {
                failed = run_step("phase-n-synthetic-expert-state",
                                  uvm_driver_phase_n_get_pool_state(2, 0, expert_state, 38));
            }
            return failed;
        };

        auto print_step = [&](const char* name,
                              uint32_t batch,
                              uint32_t context_tokens,
                              size_t active_kv_blocks,
                              uint32_t expert_load_us,
                              const char* action) {
            std::printf("phase_n_synthetic_planner_step name=%s batch=%u context_tokens=%u active_kv_blocks=%zu expert_load_us=%u action=%s kv_budget=%llu kv_resident=%llu kv_pending=%llu kv_over=%llu expert_budget=%llu expert_resident=%llu expert_pending=%llu expert_over=%llu managed_budget=%llu\n",
                        name,
                        batch,
                        context_tokens,
                        active_kv_blocks,
                        expert_load_us,
                        action,
                        static_cast<unsigned long long>(kv_state[0]),
                        static_cast<unsigned long long>(kv_state[1]),
                        static_cast<unsigned long long>(kv_state[36]),
                        static_cast<unsigned long long>(kv_state[6]),
                        static_cast<unsigned long long>(expert_state[0]),
                        static_cast<unsigned long long>(expert_state[1]),
                        static_cast<unsigned long long>(expert_state[36]),
                        static_cast<unsigned long long>(expert_state[6]),
                        static_cast<unsigned long long>(managed_budget));
        };

        trace("before_phase_n_synthetic_planner");
        int failed = run_step(
            "phase-n-synthetic-register-kv",
            uvm_driver_phase_f_register_region(kv_region_id,
                                               kv_base,
                                               pool_bytes,
                                               1,
                                               0,
                                               0,
                                               1,
                                               0x1));
        if (!failed) {
            failed = run_step(
                "phase-n-synthetic-register-expert",
                uvm_driver_phase_f_register_region(expert_region_id,
                                                   expert_base,
                                                   pool_bytes,
                                                   2,
                                                   0,
                                                   0,
                                                   2,
                                                   0x1));
        }

        if (!failed) {
            failed = run_step(
                "phase-n-synthetic-initial-budget",
                uvm_driver_phase_n_set_dynamic_budgets(0,
                                                       1,
                                                       managed_budget,
                                                       0,
                                                       0,
                                                       kv_low_blocks * block_bytes,
                                                       0,
                                                       0,
                                                       2,
                                                       1,
                                                       dyn,
                                                       8));
        }

        for (size_t i = 0; !failed && i < kv_low_blocks; ++i)
            failed = touch_block(kv_ptr, i, 0x31, "synthetic_kv_initial");
        for (size_t i = 0; !failed && i < expert_high_blocks; ++i)
            failed = touch_block(expert_ptr, i, 0x41, "synthetic_expert_initial");

        if (!failed)
            failed = get_states();
        if (!failed)
            print_step("expert_heavy_initial", 1, 512, kv_low_blocks, 700, "expert_hot_budget_high");

        if (!failed) {
            std::memset(protect_values, 0, sizeof(protect_values));
            failed = run_step(
                "phase-n-synthetic-protect-kv-heavy",
                uvm_driver_phase_n_protect_active_kv(0,
                                                     2,
                                                     kv_high_blocks * block_bytes,
                                                     managed_budget,
                                                     0,
                                                     0,
                                                     kv_region_id,
                                                     2,
                                                     0,
                                                     protect_values,
                                                     12));
            std::printf("phase_n_synthetic_planner_signal name=kv_heavy_long_context active_kv=%llu old_kv_budget=%llu new_kv_budget=%llu old_expert_budget=%llu new_expert_budget=%llu expert_over=%llu\n",
                        static_cast<unsigned long long>(kv_high_blocks * block_bytes),
                        static_cast<unsigned long long>(protect_values[0]),
                        static_cast<unsigned long long>(protect_values[1]),
                        static_cast<unsigned long long>(protect_values[2]),
                        static_cast<unsigned long long>(protect_values[3]),
                        static_cast<unsigned long long>(protect_values[10]));
        }

        for (size_t i = kv_low_blocks; !failed && i < kv_high_blocks; ++i)
            failed = touch_block(kv_ptr, i, 0x51, "synthetic_kv_heavy");

        if (!failed)
            failed = get_states();
        if (!failed)
            print_step("kv_heavy_long_context", 4, 4096, kv_high_blocks, 2300, "protect_active_kv");

        if (!failed) {
            failed = run_step(
                "phase-n-synthetic-unprotect-kv",
                uvm_driver_phase_n_update_region_state(kv_region_id,
                                                       5,
                                                       0x1,
                                                       0x6,
                                                       9,
                                                       3));
        }

        if (!failed) {
            std::memset(dyn, 0, sizeof(dyn));
            failed = run_step(
                "phase-n-synthetic-expert-budget-restore",
                uvm_driver_phase_n_set_dynamic_budgets(0,
                                                       3,
                                                       managed_budget,
                                                       0,
                                                       0,
                                                       kv_low_blocks * block_bytes,
                                                       0,
                                                       0,
                                                       2,
                                                       1,
                                                       dyn,
                                                       8));
            std::printf("phase_n_synthetic_planner_signal name=expert_heavy_recovery kv_budget=%llu expert_budget=%llu kv_over=%llu expert_over=%llu\n",
                        static_cast<unsigned long long>(kv_low_blocks * block_bytes),
                        static_cast<unsigned long long>(managed_budget - kv_low_blocks * block_bytes),
                        static_cast<unsigned long long>(dyn[5]),
                        static_cast<unsigned long long>(dyn[6]));
        }

        for (size_t i = 0; !failed && i < expert_high_blocks; ++i)
            failed = touch_block(expert_ptr, i, 0x61, "synthetic_expert_recovery");

        if (!failed)
            failed = get_states();
        if (!failed)
            print_step("expert_heavy_recovery", 1, 1024, kv_low_blocks, 650, "restore_expert_budget");

        if (!failed) {
            const uint64_t kv_high_bytes = kv_high_blocks * block_bytes;
            const uint64_t expert_high_bytes = expert_high_blocks * block_bytes;
            const bool kv_phase_ok = protect_values[1] >= kv_high_bytes &&
                                     protect_values[3] <= managed_budget - kv_high_bytes;
            const bool final_kv_ok = kv_state[0] == kv_low_blocks * block_bytes &&
                                     kv_state[1] <= kv_state[0] &&
                                     kv_state[6] == 0;
            const bool final_expert_ok = expert_state[0] == expert_high_bytes &&
                                         expert_state[1] >= expert_high_bytes &&
                                         expert_state[6] == 0;
            const bool total_ok = kv_state[1] + expert_state[1] <= managed_budget;

            std::printf("phase_n_synthetic_planner_final kv_phase_ok=%u final_kv_ok=%u final_expert_ok=%u total_ok=%u final_kv_budget=%llu final_kv_resident=%llu final_expert_budget=%llu final_expert_resident=%llu managed_budget=%llu\n",
                        kv_phase_ok ? 1 : 0,
                        final_kv_ok ? 1 : 0,
                        final_expert_ok ? 1 : 0,
                        total_ok ? 1 : 0,
                        static_cast<unsigned long long>(kv_state[0]),
                        static_cast<unsigned long long>(kv_state[1]),
                        static_cast<unsigned long long>(expert_state[0]),
                        static_cast<unsigned long long>(expert_state[1]),
                        static_cast<unsigned long long>(managed_budget));
            if (!kv_phase_ok || !final_kv_ok || !final_expert_ok || !total_ok)
                failed = 1;
        }

        cleanup_synthetic();
        trace("after_phase_n_synthetic_planner");
        return failed;
    }

    // 从这里开始，后续 step 都需要真实 managed allocation。
    // 因此只有 requested == all 或 requested 属于后续 range step 时才会走到这里。
    // 对 ping-ioctl 单步来说，前面已经 return，不会执行以下 CUDA 代码。
    void* ptr = nullptr;
    size_t allocation_bytes = bytes;
    if (want_step(step, "phase-n-block-quota")) {
        if (blocks == 0 || budget_blocks == 0 || bytes == 0) {
            std::fprintf(stderr, "phase-n-block-quota requires positive --bytes, --blocks, and --budget-blocks\n");
            return 2;
        }
        if (blocks > static_cast<size_t>(-1) / bytes) {
            std::fprintf(stderr, "phase-n-block-quota allocation size overflow\n");
            return 2;
        }
        allocation_bytes = bytes * blocks;
    }

    // 选择 CUDA device。失败时返回 3。
    trace("before_cuda_set_device");
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 3;
    }

    // 分配一个 managed memory buffer。后续 Phase F range 操作会使用它的地址。
    // 失败时返回 4。
    trace("before_cuda_malloc_managed");
    err = cudaMallocManaged(&ptr, allocation_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 4;
    }

    // 触摸 buffer 前 4KB，让这个 allocation 有实际访问痕迹。
    // 这能帮助后续 residency/query/prefetch 测试更接近真实 UVM 使用场景。
    trace("before_memset");
    std::memset(ptr, 0x5a, allocation_bytes < 4096 ? allocation_bytes : 4096);

    // 等待 CUDA 侧状态稳定。失败时释放 buffer 并返回 5。
    trace("before_cuda_device_synchronize_after_memset");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize after memset failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(ptr);
        return 5;
    }

    // 把 managed pointer 转成整数地址，传给 uvm_allocator.cpp 的 ioctl helper。
    uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    size_t size = allocation_bytes;
    std::printf("phase_f_smoke managed_ptr=0x%llx bytes=%zu\n",
                static_cast<unsigned long long>(base), size);
    std::fflush(stdout);
    trace("after_managed_allocation_ready");

    // Step 6.5: 探测 CUDA runtime 是否已经在当前进程打开 /dev/nvidia-uvm。
    // 返回 1 表示找到并成功 dup；返回 0 表示当前进程 fd 表中没有可采用的 UVM fd。
    if (want_step(step, "probe-uvm-fd")) {
        trace("before_probe_uvm_fd");
        int failed = run_step("probe-uvm-fd", uvm_driver_phase_f_probe_existing_uvm_fd());
        trace("after_probe_uvm_fd");
        cudaFree(ptr);
        return failed;
    }

    // Step 7: mini batch ioctl，count=1，传真实 managed range。
    // 这一步用来区分“空 ioctl 路径没问题”与“带 range 参数后出问题”。
    if (want_step(step, "batch-mini1")) {
        trace("before_batch_mini1");
        int failed = run_step("batch-mini1", uvm_driver_phase_f_batch_mini1(base, size));
        trace("after_batch_mini1");
        if (failed || step == "batch-mini1") {
            cudaFree(ptr);
            return failed;
        }
    }

    // Step 8: 查询 managed range 的 residency。
    // 会调用 Phase F batch executor，action=QUERY_RESIDENCY，count=1。
    if (want_step(step, "query")) {
        trace("before_query");
        uint64_t gpu_bytes = 0;
        uint64_t cpu_bytes = 0;
        uint64_t any_bytes = 0;
        uint64_t unknown_bytes = 0;
        uint64_t queried_bytes = 0;
        int query_result = uvm_driver_batch_query_residency_bytes(&base,
                                                                  &size,
                                                                  1,
                                                                  device,
                                                                  &gpu_bytes,
                                                                  &cpu_bytes,
                                                                  &any_bytes,
                                                                  &unknown_bytes,
                                                                  &queried_bytes);
        std::printf("phase_f_smoke residency_bytes queried=%llu gpu=%llu cpu=%llu any=%llu unknown=%llu\n",
                    static_cast<unsigned long long>(queried_bytes),
                    static_cast<unsigned long long>(gpu_bytes),
                    static_cast<unsigned long long>(cpu_bytes),
                    static_cast<unsigned long long>(any_bytes),
                    static_cast<unsigned long long>(unknown_bytes));
        int failed = run_step("query", query_result);
        trace("after_query");
        if (failed || step == "query") {
            cudaFree(ptr);
            return failed;
        }
    }

    const uint64_t region_id = 1;

    // Step 8.1: 按 region id 查询 residency。
    // helper 会在同一个 UVM fd 生命周期内完成 register -> query -> unregister，
    // 避免 region 随 fd close 立刻消失。
    if (want_step(step, "query-region")) {
        trace("before_query_region");
        int failed = run_step(
            "query-region",
            uvm_driver_phase_f_register_query_region(region_id, base, size, device));
        trace("after_query_region");
        if (failed || step == "query-region") {
            cudaFree(ptr);
            return failed;
        }
    }

    // Step 8.2: 只注册一个 Phase F policy region。
    // 单独跑 register-region 时只验证注册 ioctl；由于 helper 会关闭 fd，
    // 这个 region 的生命周期也会随该 VA space 结束。
    if (want_step(step, "register-region")) {
        trace("before_register_region");
        int failed = run_step(
            "register-region",
            uvm_driver_phase_f_register_region(region_id, base, size, 1, 0, 0, 0, 0));
        trace("after_register_region");
        if (failed || step == "register-region") {
            cudaFree(ptr);
            return failed;
        }
    }

    // Step 8.3: Stage N 控制面闭环。
    // 注册 region tag，设置 KV pool budget，更新 region state，再按 id/VA 查询。
    if (want_step(step, "phase-n-region")) {
        uint64_t values[36] = {};
        trace("before_phase_n_region");
        int failed = run_step(
            "phase-n-set-pool-budget",
            uvm_driver_phase_n_set_pool_budget(1, 0, bytes / 2, 0, 0));
        if (!failed) {
            failed = run_step(
                "phase-n-register-region",
                uvm_driver_phase_f_register_region(region_id, base, size, 1, 0, 0, 1, 1));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-update-region-state",
                uvm_driver_phase_n_update_region_state(region_id, 4, 0x6, 0x1, 7, 42));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-query-region",
                uvm_driver_phase_n_query_region(region_id, values, 11));
            std::printf("phase_n_region id=%llu base=0x%llx length=%llu pool=%llu role=%llu flags=0x%llx state=%llu priority=%llu epoch=%llu\n",
                        static_cast<unsigned long long>(values[0]),
                        static_cast<unsigned long long>(values[1]),
                        static_cast<unsigned long long>(values[2]),
                        static_cast<unsigned long long>(values[3]),
                        static_cast<unsigned long long>(values[6]),
                        static_cast<unsigned long long>(values[7]),
                        static_cast<unsigned long long>(values[8]),
                        static_cast<unsigned long long>(values[9]),
                        static_cast<unsigned long long>(values[10]));
        }
        if (!failed) {
            std::memset(values, 0, sizeof(values));
            failed = run_step(
                "phase-n-query-region-by-va",
                uvm_driver_phase_n_query_region_by_va(base, values, 11));
            std::printf("phase_n_region_by_va id=%llu base=0x%llx length=%llu pool=%llu role=%llu flags=0x%llx state=%llu priority=%llu epoch=%llu\n",
                        static_cast<unsigned long long>(values[0]),
                        static_cast<unsigned long long>(values[1]),
                        static_cast<unsigned long long>(values[2]),
                        static_cast<unsigned long long>(values[3]),
                        static_cast<unsigned long long>(values[6]),
                        static_cast<unsigned long long>(values[7]),
                        static_cast<unsigned long long>(values[8]),
                        static_cast<unsigned long long>(values[9]),
                        static_cast<unsigned long long>(values[10]));
        }
        if (!failed) {
            std::memset(values, 0, sizeof(values));
            failed = run_step(
                "phase-n-get-pool-state",
                uvm_driver_phase_n_get_pool_state(1, 0, values, 36));
            std::printf("phase_n_pool_state budget=%llu resident_gpu=%llu resident_cpu=%llu reclaim=%llu protected=%llu evictable=%llu over=%llu target=%llu candidate_bytes=%llu candidate_regions=%llu selected_id=%llu selected_bytes=%llu victim_count=%llu corrections=%llu mode=%llu flags=0x%llx\n",
                        static_cast<unsigned long long>(values[0]),
                        static_cast<unsigned long long>(values[1]),
                        static_cast<unsigned long long>(values[2]),
                        static_cast<unsigned long long>(values[3]),
                        static_cast<unsigned long long>(values[4]),
                        static_cast<unsigned long long>(values[5]),
                        static_cast<unsigned long long>(values[6]),
                        static_cast<unsigned long long>(values[7]),
                        static_cast<unsigned long long>(values[8]),
                        static_cast<unsigned long long>(values[9]),
                        static_cast<unsigned long long>(values[10]),
                        static_cast<unsigned long long>(values[11]),
                        static_cast<unsigned long long>(values[12]),
                        static_cast<unsigned long long>(values[33]),
                        static_cast<unsigned long long>(values[34]),
                        static_cast<unsigned long long>(values[35]));
            std::printf("phase_n_victims:");
            for (int i = 0; i < 8; ++i) {
                std::printf(" [%d]=%llu:%llu",
                            i,
                            static_cast<unsigned long long>(values[14 + i]),
                            static_cast<unsigned long long>(values[22 + i]));
            }
            std::printf("\n");
        }
        (void)uvm_driver_phase_f_unregister_region(region_id);
        trace("after_phase_n_region");
        if (failed || step == "phase-n-region") {
            cudaFree(ptr);
            return failed;
        }
    }

    // Step 8.4: Stage N P6 soft reclaim 控制面。
    // 先把 region 迁到 GPU，再设置 SOFT budget，最后让 driver 选择 victim 并迁回 CPU。
    if (want_step(step, "phase-n-reclaim")) {
        uint64_t values[45] = {};
        uint64_t post_state[36] = {};
        uint32_t role = pool == 3 ? 5 : (pool == 2 ? 2 : 1);
        uint32_t flags = pool == 2 ? 0x2 : 0x1;
        if (pool == 3)
            flags |= 0x20;
        trace("before_phase_n_reclaim");
        int failed = run_step(
            "phase-n-reclaim-register-region",
            uvm_driver_phase_f_register_region(region_id,
                                               base,
                                               size,
                                               pool,
                                               0,
                                               0,
                                               role,
                                               flags));
        if (!failed) {
            failed = run_step(
                "phase-n-reclaim-update-region-state",
                uvm_driver_phase_n_update_region_state(region_id, 5, 0x1, 0x6, 7, 1));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-reclaim-prefetch-gpu",
                uvm_driver_batch_prefetch_gpu(&base, &size, 1, device, 0));
            cudaDeviceSynchronize();
        }
        if (!failed) {
            failed = run_step(
                "phase-n-reclaim-set-pool-budget",
                uvm_driver_phase_n_set_pool_budget(pool, 0, bytes / 2, 1, 0));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-reclaim-pool",
                uvm_driver_phase_n_reclaim_pool(pool, 0, 0, 100, 0, 8, async, 0, values, 45));
            std::printf("phase_n_reclaim pool=%u attempted=%llu success=%llu failure=%llu victim_count=%llu budget=%llu resident_before=%llu over=%llu target=%llu attempted_bytes=%llu reclaimed=%llu failed_bytes=%llu candidate_bytes=%llu candidate_regions=%llu cooldown_bytes=%llu cooldown_regions=%llu success_events=%llu failed_events=%llu mode=%llu flags=0x%llx\n",
                        pool,
                        static_cast<unsigned long long>(values[0]),
                        static_cast<unsigned long long>(values[1]),
                        static_cast<unsigned long long>(values[2]),
                        static_cast<unsigned long long>(values[3]),
                        static_cast<unsigned long long>(values[4]),
                        static_cast<unsigned long long>(values[5]),
                        static_cast<unsigned long long>(values[6]),
                        static_cast<unsigned long long>(values[7]),
                        static_cast<unsigned long long>(values[8]),
                        static_cast<unsigned long long>(values[9]),
                        static_cast<unsigned long long>(values[10]),
                        static_cast<unsigned long long>(values[11]),
                        static_cast<unsigned long long>(values[12]),
                        static_cast<unsigned long long>(values[13]),
                        static_cast<unsigned long long>(values[14]),
                        static_cast<unsigned long long>(values[39]),
                        static_cast<unsigned long long>(values[40]),
                        static_cast<unsigned long long>(values[41]),
                        static_cast<unsigned long long>(values[42]));
            std::printf("phase_n_reclaim_victims:");
            for (int i = 0; i < 8; ++i) {
                std::printf(" [%d]=%llu:%llu:0x%llx",
                            i,
                            static_cast<unsigned long long>(values[15 + i]),
                            static_cast<unsigned long long>(values[23 + i]),
                            static_cast<unsigned long long>(values[31 + i]));
            }
            std::printf("\n");
        }
        if (!failed) {
            failed = run_step(
                "phase-n-reclaim-post-pool-state",
                uvm_driver_phase_n_get_pool_state(pool, 0, post_state, 36));
            std::printf("phase_n_reclaim_post_state pool=%u budget=%llu resident_gpu=%llu resident_cpu=%llu over=%llu target=%llu candidate_bytes=%llu candidate_regions=%llu selected_id=%llu selected_bytes=%llu victim_count=%llu success_events=%llu failed_events=%llu mode=%llu flags=0x%llx\n",
                        pool,
                        static_cast<unsigned long long>(post_state[0]),
                        static_cast<unsigned long long>(post_state[1]),
                        static_cast<unsigned long long>(post_state[2]),
                        static_cast<unsigned long long>(post_state[6]),
                        static_cast<unsigned long long>(post_state[7]),
                        static_cast<unsigned long long>(post_state[8]),
                        static_cast<unsigned long long>(post_state[9]),
                        static_cast<unsigned long long>(post_state[10]),
                        static_cast<unsigned long long>(post_state[11]),
                        static_cast<unsigned long long>(post_state[12]),
                        static_cast<unsigned long long>(post_state[31]),
                        static_cast<unsigned long long>(post_state[32]),
                        static_cast<unsigned long long>(post_state[34]),
                        static_cast<unsigned long long>(post_state[35]));
        }
        (void)uvm_driver_phase_f_unregister_region(region_id);
        trace("after_phase_n_reclaim");
        if (failed || step == "phase-n-reclaim") {
            cudaFree(ptr);
            return failed;
        }
    }

    // Step 8.5: Stage N P8c block-quota synthetic proof.
    // The region is registered before GPU touches. Each cudaMemset hits the
    // first page of one native-size block so the replayable-fault outer hook can
    // materialize the Stage N block wrapper and optionally auto-reclaim.
    if (want_step(step, "phase-n-block-quota")) {
        uint64_t state[36] = {};
        const uint64_t block_region_id = 1000 + pool;
        const uint32_t role = pool == 2 ? 2 : (pool == 3 ? 5 : 1);
        uint32_t flags = 0x1;
        if (pool == 3)
            flags |= 0x20;
        size_t touch_bytes = bytes < 4096 ? bytes : 4096;
        uint64_t budget_bytes = static_cast<uint64_t>(budget_blocks) * static_cast<uint64_t>(bytes);

        trace("before_phase_n_block_quota");
        int failed = run_step(
            "phase-n-block-quota-set-pool-budget",
            uvm_driver_phase_n_set_pool_budget(pool, 0, budget_bytes, 1, 0));
        if (!failed) {
            failed = run_step(
                "phase-n-block-quota-register-region",
                uvm_driver_phase_f_register_region(block_region_id,
                                                   base,
                                                   size,
                                                   pool,
                                                   0,
                                                   0,
                                                   role,
                                                   flags));
        }
        if (!failed) {
            failed = run_step(
                "phase-n-block-quota-initial-pool-state",
                uvm_driver_phase_n_get_pool_state(pool, 0, state, 36));
            std::printf("phase_n_block_quota_initial pool=%u budget=%llu resident_gpu=%llu over=%llu mode=%llu flags=0x%llx\n",
                        pool,
                        static_cast<unsigned long long>(state[0]),
                        static_cast<unsigned long long>(state[1]),
                        static_cast<unsigned long long>(state[6]),
                        static_cast<unsigned long long>(state[34]),
                        static_cast<unsigned long long>(state[35]));
        }

        for (size_t i = 0; !failed && i < blocks; ++i) {
            char* block_ptr = static_cast<char*>(ptr) + i * bytes;
            err = cudaMemset(block_ptr, static_cast<int>(0x30 + (i & 0x3f)), touch_bytes);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "cudaMemset block=%zu failed: %s\n",
                             i, cudaGetErrorString(err));
                failed = 6;
                break;
            }
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::fprintf(stderr, "cudaDeviceSynchronize block=%zu failed: %s\n",
                             i, cudaGetErrorString(err));
                failed = 7;
                break;
            }

            std::memset(state, 0, sizeof(state));
            failed = run_step(
                "phase-n-block-quota-touch-pool-state",
                uvm_driver_phase_n_get_pool_state(pool, 0, state, 36));
            std::printf("phase_n_block_quota_touch pool=%u block=%zu offset=%zu touch_bytes=%zu budget=%llu resident_gpu=%llu over=%llu victim_count=%llu success_events=%llu failed_events=%llu mode=%llu flags=0x%llx\n",
                        pool,
                        i,
                        i * bytes,
                        touch_bytes,
                        static_cast<unsigned long long>(state[0]),
                        static_cast<unsigned long long>(state[1]),
                        static_cast<unsigned long long>(state[6]),
                        static_cast<unsigned long long>(state[12]),
                        static_cast<unsigned long long>(state[31]),
                        static_cast<unsigned long long>(state[32]),
                        static_cast<unsigned long long>(state[34]),
                        static_cast<unsigned long long>(state[35]));
        }

        if (!failed) {
            std::memset(state, 0, sizeof(state));
            failed = run_step(
                "phase-n-block-quota-final-pool-state",
                uvm_driver_phase_n_get_pool_state(pool, 0, state, 36));
            std::printf("phase_n_block_quota_final pool=%u budget=%llu resident_gpu=%llu over=%llu victim_count=%llu success_events=%llu failed_events=%llu mode=%llu flags=0x%llx\n",
                        pool,
                        static_cast<unsigned long long>(state[0]),
                        static_cast<unsigned long long>(state[1]),
                        static_cast<unsigned long long>(state[6]),
                        static_cast<unsigned long long>(state[12]),
                        static_cast<unsigned long long>(state[31]),
                        static_cast<unsigned long long>(state[32]),
                        static_cast<unsigned long long>(state[34]),
                        static_cast<unsigned long long>(state[35]));
        }

        (void)uvm_driver_phase_f_unregister_region(block_region_id);
        trace("after_phase_n_block_quota");
        if (failed || step == "phase-n-block-quota") {
            cudaFree(ptr);
            return failed;
        }
    }

    // Step 9: prefetch managed range 到 GPU。
    // 会调用 Phase F batch executor，action=PREFETCH_GPU，count=1。
    // 之后执行 cudaDeviceSynchronize，让同步 prefetch 或相关 CUDA work 完成。
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

    // Step 9.1: 在同一个 managed allocation 上先 prefetch 到 GPU，再查询 residency bytes。
    // 这是验证真实 GPU resident bytes 的最小闭环；不能拆成两个脚本运行，
    // 因为两次运行会创建不同的 managed allocation。
    if (want_step(step, "prefetch-gpu-query")) {
        trace("before_prefetch_gpu_query_prefetch");
        int failed = run_step(
            "prefetch-gpu",
            uvm_driver_batch_prefetch_gpu(&base, &size, 1, device, async));
        cudaDeviceSynchronize();
        trace("after_prefetch_gpu_query_prefetch");
        if (failed) {
            cudaFree(ptr);
            return failed;
        }

        uint64_t gpu_bytes = 0;
        uint64_t cpu_bytes = 0;
        uint64_t any_bytes = 0;
        uint64_t unknown_bytes = 0;
        uint64_t queried_bytes = 0;
        trace("before_prefetch_gpu_query_query");
        int query_result = uvm_driver_batch_query_residency_bytes(&base,
                                                                  &size,
                                                                  1,
                                                                  device,
                                                                  &gpu_bytes,
                                                                  &cpu_bytes,
                                                                  &any_bytes,
                                                                  &unknown_bytes,
                                                                  &queried_bytes);
        std::printf("phase_f_smoke residency_bytes_after_prefetch_gpu queried=%llu gpu=%llu cpu=%llu any=%llu unknown=%llu\n",
                    static_cast<unsigned long long>(queried_bytes),
                    static_cast<unsigned long long>(gpu_bytes),
                    static_cast<unsigned long long>(cpu_bytes),
                    static_cast<unsigned long long>(any_bytes),
                    static_cast<unsigned long long>(unknown_bytes));
        failed = run_step("prefetch-gpu-query", query_result);
        trace("after_prefetch_gpu_query_query");
        cudaFree(ptr);
        return failed;
    }

    // Step 10: advise managed range 的 preferred location 为 CPU。
    // 会调用 Phase F batch executor，action=ADVISE_CPU，count=1。
    if (want_step(step, "advise-cpu")) {
        int failed = run_step(
            "advise-cpu", uvm_driver_batch_advise_cpu(&base, &size, 1));
        if (failed || step == "advise-cpu") {
            cudaFree(ptr);
            return failed;
        }
    }

    // Step 11: prefetch managed range 回 CPU。
    // 会调用 Phase F batch executor，action=PREFETCH_CPU，count=1。
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

    // 正常收尾。所有需要 managed memory 的 step 执行完后释放 buffer。
    cudaFree(ptr);
    std::printf("phase_f_smoke done\n");
    return 0;
}
