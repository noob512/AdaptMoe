/*******************************************************************************
    Copyright (c) 2015-2025 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_global.h"
#include "uvm_procfs.h"
#include "uvm_gpu.h"

#include "nv-procfs.h"
#include "uvm_linux.h"
#include <linux/delay.h>
#include <linux/mutex.h>
#include <linux/seq_file.h>
#include <linux/timekeeping.h>

#define UVM_PROC_DIR_NAME "driver/nvidia-uvm"
#define UVM_PROC_GPUS_DIR_NAME "gpus"
#define UVM_PROC_CPU_DIR_NAME "cpu"
#define UVM_PHASE_F_ROUTE_LOG_SIZE (64 * 1024)

#if defined(CONFIG_PROC_FS)
  // This parameter enables additional debug procfs entries. It's enabled by
  // default for debug and develop builds and disabled for release builds.
  static int uvm_enable_debug_procfs = UVM_IS_DEBUG() || UVM_IS_DEVELOP();
  module_param(uvm_enable_debug_procfs, int, S_IRUGO);
  MODULE_PARM_DESC(uvm_enable_debug_procfs, "Enable debug procfs entries in /proc/" UVM_PROC_DIR_NAME);
#else
  static int uvm_enable_debug_procfs = 0;
#endif

// Optional debug-only delay after each Phase F route log write. This gives a
// userspace watcher more time to copy /proc/driver/nvidia-uvm/phase_f_route_log
// into a real file before a suspected hard crash. The Phase F crash diagnostic
// run uses 500ms by default to make each route step visible before continuing.
static unsigned int uvm_phase_f_route_log_delay_ms = 500;
module_param(uvm_phase_f_route_log_delay_ms, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_phase_f_route_log_delay_ms,
                 "Delay after each Phase F route log write, in milliseconds");

static struct proc_dir_entry *uvm_proc_dir;
static struct proc_dir_entry *uvm_proc_gpus;
static struct proc_dir_entry *uvm_proc_cpu;
static struct proc_dir_entry *uvm_proc_phase_f_route_log;

static DEFINE_MUTEX(g_phase_f_route_log_lock);
static char g_phase_f_route_log[UVM_PHASE_F_ROUTE_LOG_SIZE];
static size_t g_phase_f_route_log_pos;
static bool g_phase_f_route_log_wrapped;

bool uvm_procfs_is_debug_enabled(void)
{
    return uvm_enable_debug_procfs != 0;
}

void uvm_phase_f_route_log(const char *fmt, ...)
{
    char line[512];
    va_list args;
    size_t len;
    struct timespec64 ts;

    ktime_get_real_ts64(&ts);
    len = scnprintf(line,
                    sizeof(line),
                    "[%lld.%09ld] ",
                    (long long)ts.tv_sec,
                    ts.tv_nsec);

    va_start(args, fmt);
    len += vscnprintf(line + len, sizeof(line) - len, fmt, args);
    va_end(args);

    if (len >= sizeof(line))
        len = sizeof(line) - 1;

    if (len == 0 || line[len - 1] != '\n') {
        if (len < sizeof(line) - 1)
            line[len++] = '\n';
        else
            line[sizeof(line) - 1] = '\n';
    }

    mutex_lock(&g_phase_f_route_log_lock);
    if (len >= sizeof(g_phase_f_route_log)) {
        memcpy(g_phase_f_route_log, line + len - sizeof(g_phase_f_route_log), sizeof(g_phase_f_route_log));
        g_phase_f_route_log_pos = 0;
        g_phase_f_route_log_wrapped = true;
    }
    else {
        size_t first = min(len, sizeof(g_phase_f_route_log) - g_phase_f_route_log_pos);
        memcpy(g_phase_f_route_log + g_phase_f_route_log_pos, line, first);
        if (len > first)
            memcpy(g_phase_f_route_log, line + first, len - first);
        g_phase_f_route_log_pos = (g_phase_f_route_log_pos + len) % sizeof(g_phase_f_route_log);
        if (g_phase_f_route_log_pos == 0 || len > first)
            g_phase_f_route_log_wrapped = true;
    }
    mutex_unlock(&g_phase_f_route_log_lock);

    if (uvm_phase_f_route_log_delay_ms != 0)
        msleep(uvm_phase_f_route_log_delay_ms);
}

static int nv_procfs_read_phase_f_route_log(struct seq_file *s, void *v)
{
    mutex_lock(&g_phase_f_route_log_lock);
    if (g_phase_f_route_log_wrapped) {
        seq_write(s,
                  g_phase_f_route_log + g_phase_f_route_log_pos,
                  sizeof(g_phase_f_route_log) - g_phase_f_route_log_pos);
        seq_write(s, g_phase_f_route_log, g_phase_f_route_log_pos);
    }
    else {
        seq_write(s, g_phase_f_route_log, g_phase_f_route_log_pos);
    }
    mutex_unlock(&g_phase_f_route_log_lock);

    return 0;
}

UVM_DEFINE_SINGLE_PROCFS_FILE(phase_f_route_log);

NV_STATUS uvm_procfs_init(void)
{
    if (!uvm_procfs_is_enabled())
        return NV_OK;

    uvm_proc_dir = NV_CREATE_PROC_DIR(UVM_PROC_DIR_NAME, NULL);
    if (uvm_proc_dir == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    uvm_proc_gpus = NV_CREATE_PROC_DIR(UVM_PROC_GPUS_DIR_NAME, uvm_proc_dir);
    if (uvm_proc_gpus == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    uvm_proc_cpu = NV_CREATE_PROC_DIR(UVM_PROC_CPU_DIR_NAME, uvm_proc_dir);
    if (uvm_proc_cpu == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    uvm_proc_phase_f_route_log = NV_CREATE_PROC_FILE("phase_f_route_log",
                                                     uvm_proc_dir,
                                                     phase_f_route_log,
                                                     NULL);
    if (uvm_proc_phase_f_route_log == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    return NV_OK;
}

void uvm_procfs_exit(void)
{
    proc_remove(uvm_proc_dir);
}

struct proc_dir_entry *uvm_procfs_get_gpu_base_dir(void)
{
    return uvm_proc_gpus;
}

struct proc_dir_entry *uvm_procfs_get_cpu_base_dir(void)
{
    return uvm_proc_cpu;
}
