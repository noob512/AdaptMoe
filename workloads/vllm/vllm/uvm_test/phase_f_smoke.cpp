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
int uvm_driver_phase_f_probe_existing_uvm_fd();
}

namespace {

// 打印命令行帮助。这里只展示 phase_f_smoke 支持的 step，
// shell 脚本 run_stage_m_phase_f_smoke.sh 会把同名 --step 继续传进来。
void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--step all|ping-open|ping-init|ping-small|batch-mini0|batch-mini1|ping-ioctl|ping|probe-uvm-fd|register-region|query-region|query|prefetch-gpu|prefetch-gpu-query|advise-cpu|prefetch-cpu] "
                 "[--bytes N] [--device N] [--async 0|1]\n",
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
    std::printf("phase_f_smoke begin step=%s bytes=%zu device=%d async=%d\n",
                step.c_str(), bytes, device, async);
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

    // 从这里开始，后续 step 都需要真实 managed allocation。
    // 因此只有 requested == all 或 requested 属于后续 range step 时才会走到这里。
    // 对 ping-ioctl 单步来说，前面已经 return，不会执行以下 CUDA 代码。
    void* ptr = nullptr;

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
    err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 4;
    }

    // 触摸 buffer 前 4KB，让这个 allocation 有实际访问痕迹。
    // 这能帮助后续 residency/query/prefetch 测试更接近真实 UVM 使用场景。
    trace("before_memset");
    std::memset(ptr, 0x5a, bytes < 4096 ? bytes : 4096);

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
    size_t size = bytes;
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
