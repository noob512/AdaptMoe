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

#include "uvm_api.h"
#include "uvm_global.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_tools_init.h"
#include "uvm_lock.h"
#include "uvm_test.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_va_range.h"
#include "uvm_va_block.h"
#include "uvm_tools.h"
#include "uvm_common.h"
#include "uvm_fd_type.h"
#include "uvm_linux_ioctl.h"
#include "uvm_hmm.h"
#include "uvm_hal.h"
#include "uvm_mem.h"
#include "uvm_migrate.h"
#include "uvm_kvmalloc.h"
#include "uvm_test_file.h"
#include "uvm_push.h"
#include "uvm_mmu.h"

#include <linux/mutex.h>
#include <linux/uaccess.h>

#define NVIDIA_UVM_DEVICE_NAME          "nvidia-uvm"

static dev_t g_uvm_base_dev;
static struct cdev g_uvm_cdev;
static const struct file_operations uvm_fops;

static uvm_phase_f_region_t *uvm_phase_f_region_from_node(uvm_range_tree_node_t *node)
{
    if (!node)
        return NULL;

    return container_of(node, uvm_phase_f_region_t, va_node);
}

static bool uvm_phase_f_registered_region_is_valid(const uvm_phase_f_region_t *region);


#define UVM_MOE_SLOT_POOL_MAX_SLOTS 8192
#define UVM_MOE_SLOT_POOL_GPU_COPY_STAGING_CHUNK_BYTES (1ULL * 1024 * 1024)
#define UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT 16


typedef struct
{
    NvU64 va_base;
    NvU64 bytes;
} uvm_moe_slot_pool_pte_range_t;

typedef struct
{
    NvBool valid;
    NvU32 layer_id;
    NvU32 slot_id;
    NvU32 owner_expert_id;
    NvU32 state;
    NvU32 range_count;
    NvU32 last_role;
    NvU32 flags;
    NvU32 last_status;
    NvU64 epoch;
    NvU64 batch_id;
    NvU64 first_va_base;
    NvU64 last_va_base;
    NvU64 last_bytes;
    NvU64 total_bytes;
    uvm_mem_t *slot_mem;
    uvm_gpu_t *slot_gpu;
    NvU64 slot_bytes;
    uvm_mem_t *async_copy_staging;
    uvm_tracker_t async_copy_tracker;
    NvBool async_copy_pending;
    NvU64 async_copy_bytes;
    NvU64 async_copy_slot_offset;
    uvm_page_table_range_vec_t *pte_range_vec;
    uvm_gpu_t *pte_gpu;
    NvU64 pte_va_base;
    NvU64 pte_bytes;
    uvm_moe_slot_pool_pte_range_t pte_ranges[UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT];
    NvU32 pte_range_count;
    uvm_va_space_t *pte_va_space;
    NvU64 pte_slot_offset;
    NvU32 backing_kind;
    NvU32 copy_status;
    NvU32 pte_status;
    NvU64 copy_count;
    NvU64 copy_bytes;
    NvU64 copy_failed_count;
    NvU64 pte_remap_count;
    NvU64 pte_remap_bytes;
    NvU64 pte_clear_count;
    NvU64 pte_block_unmap_count;
    NvU64 pte_block_tracker_wait_count;
    NvU64 pte_rollback_count;
    NvU64 pte_rollback_bytes;
    NvU64 pte_failure_count;
    NvU64 pte_fault_clear_count;
    NvU64 pte_migrate_clear_count;
    NvU64 pte_unmap_clear_count;
    NvU64 pte_split_clear_count;
    NvU64 pte_release_clear_count;
    NvU64 pte_evict_clear_count;
    NvU64 pte_dangerous_op_reject_count;
    NvU32 pte_mode;
    NvBool pte_va_block_tracked;
} uvm_moe_slot_pool_slot_t;

static NV_STATUS uvm_moe_slot_pool_finish_async_copy_locked(uvm_moe_slot_pool_slot_t *slot,
                                                            NvBool wait_for_completion,
                                                            NvBool *completed_out);

static NV_STATUS uvm_moe_slot_pool_record_pte_range_locked(uvm_moe_slot_pool_slot_t *slot, NvU64 va_base, NvU64 bytes)
{
    NvU64 va_end = va_base + bytes;

    if (bytes == 0 || va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    if (slot->pte_range_count >= UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT)
        return NV_ERR_INSUFFICIENT_RESOURCES;

    uvm_phase_f_route_log("moe_pte_range action=record layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu range_index=%u existing_ranges=%u mode=%u va_space=%px\n",
                          slot->layer_id,
                          slot->owner_expert_id,
                          slot->slot_id,
                          slot->last_role,
                          va_base,
                          va_end,
                          bytes,
                          slot->pte_range_count,
                          slot->pte_range_count,
                          slot->pte_mode,
                          slot->pte_va_space);

    slot->pte_ranges[slot->pte_range_count].va_base = va_base;
    slot->pte_ranges[slot->pte_range_count].bytes = bytes;
    ++slot->pte_range_count;

    if (slot->pte_bytes == 0 || va_base < slot->pte_va_base)
        slot->pte_va_base = va_base;

    slot->pte_bytes += bytes;

    return NV_OK;
}

typedef struct
{
    uvm_moe_slot_pool_slot_t *installed_slots[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 installed_indices[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 installed_count;
    NvBool rollback_batch;
} uvm_moe_slot_pool_pte_txn_t;

typedef struct
{
    NvBool installed[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU64 block_unmap_count[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU64 tracker_wait_count[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
} uvm_moe_slot_pool_user_backing_group_state_t;

static DEFINE_MUTEX(g_uvm_moe_slot_pool_lock);
static uvm_moe_slot_pool_slot_t g_uvm_moe_slot_pool[UVM_MOE_SLOT_POOL_MAX_SLOTS];
static NvU64 g_uvm_moe_slot_pool_current_epoch;
static NvU64 g_uvm_moe_slot_pool_current_batch_id;
static NvU64 g_uvm_moe_slot_pool_validated_entries;
static NvU64 g_uvm_moe_slot_pool_overwrite_count;
static NvU64 g_uvm_moe_slot_pool_range_update_count;
static NvU64 g_uvm_moe_slot_pool_full_count;
static NvU64 g_uvm_moe_slot_pool_copy_count;
static NvU64 g_uvm_moe_slot_pool_copy_bytes;
static NvU64 g_uvm_moe_slot_pool_copy_failed_count;
static NvU64 g_uvm_moe_slot_pool_vidmem_evict_count;
static NvU64 g_uvm_moe_slot_pool_pte_remap_count;
static NvU64 g_uvm_moe_slot_pool_pte_remap_bytes;
static NvU64 g_uvm_moe_slot_pool_pte_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_block_unmap_count;
static NvU64 g_uvm_moe_slot_pool_pte_block_tracker_wait_count;
static NvU64 g_uvm_moe_slot_pool_pte_rollback_count;
static NvU64 g_uvm_moe_slot_pool_pte_rollback_bytes;
static NvU64 g_uvm_moe_slot_pool_pte_failure_count;
static NvU64 g_uvm_moe_slot_pool_pte_fault_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_migrate_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_unmap_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_split_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_release_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_evict_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_dangerous_op_reject_count;

static unsigned long long uvm_moe_slot_pool_vidmem_budget_bytes __read_mostly = 512ULL * 1024 * 1024;
module_param(uvm_moe_slot_pool_vidmem_budget_bytes, ullong, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_pool_vidmem_budget_bytes,
                 "MoE gpu_copy_only driver-owned vidmem backing budget in bytes; 0 disables budget eviction");

static int uvm_moe_slot_pte_remap_enable __read_mostly = 0;
module_param(uvm_moe_slot_pte_remap_enable, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_pte_remap_enable,
                 "Enable experimental smoke-only MoE slot-backed GPU PTE remap. Default: 0.");

static unsigned long long uvm_moe_slot_pte_remap_max_bytes __read_mostly = 64ULL * 1024;
module_param(uvm_moe_slot_pte_remap_max_bytes, ullong, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_pte_remap_max_bytes,
                 "Maximum bytes accepted by one experimental MoE slot PTE remap entry. Default: 64KiB.");

static int uvm_moe_slot_user_backing_defer_tlb_invalidate __read_mostly = 0;
module_param(uvm_moe_slot_user_backing_defer_tlb_invalidate, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_user_backing_defer_tlb_invalidate,
                 "Experimentally defer mode=7 user-backing map-side TLB invalidates and issue one batch invalidate. Default: 0.");

static int uvm_moe_slot_user_backing_grouped_remap __read_mostly = 0;
module_param(uvm_moe_slot_user_backing_grouped_remap, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_user_backing_grouped_remap,
                 "Experimentally group mode=7 user-backing remap by target VA block. Default: 0.");

typedef enum
{
    UVM_MOE_SLOT_POOL_COPY_NONE = 0,
    UVM_MOE_SLOT_POOL_COPY_SYSMEM = 1,
    UVM_MOE_SLOT_POOL_COPY_GPU = 2,
    UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM = 3,
    UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC = 4,
} uvm_moe_slot_pool_copy_mode_t;

typedef enum
{
    UVM_MOE_SLOT_PTE_CLEAR_REASON_FAULT = 0,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_SPLIT,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_EVICT,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT,
} uvm_moe_slot_pte_clear_reason_t;

static void uvm_moe_slot_pool_release_va_space_mappings(uvm_va_space_t *va_space);
static NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_release_locked(uvm_va_space_t *va_space,
                                                                           struct mm_struct *mm,
                                                                           NvU64 base,
                                                                           NvU64 bytes);

static void uvm_copy_region_to_query(uvm_phase_f_region_t *region,
                                     UVM_QUERY_REGION_PARAMS *params,
                                     NvU32 found_by_va)
{
    params->region_id = region->region_id;
    params->base = region->base;
    params->length = region->length;
    params->state = region->state;
    params->found_by_va = found_by_va;
}

static void uvm_copy_region_to_query_by_va(uvm_phase_f_region_t *region,
                                           UVM_QUERY_REGION_BY_VA_PARAMS *params)
{
    params->region_id = region->region_id;
    params->base = region->base;
    params->length = region->length;
    params->state = region->state;
    params->found_by_va = 1;
}

bool uvm_file_is_nvidia_uvm(struct file *filp)
{
    return (filp != NULL) && (filp->f_op == &uvm_fops);
}

bool uvm_file_is_nvidia_uvm_va_space(struct file *filp)
{
    return uvm_file_is_nvidia_uvm(filp) && uvm_fd_type(filp, NULL) == UVM_FD_VA_SPACE;
}

static NV_STATUS uvm_api_mm_initialize(UVM_MM_INITIALIZE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space;
    uvm_va_space_mm_t *va_space_mm;
    struct file *uvm_file;
    struct mm_struct *mm;
    NV_STATUS status;

    uvm_file = fget(params->uvmFd);
    if (!uvm_file_is_nvidia_uvm(uvm_file)) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto err;
    }

    if (uvm_fd_type(uvm_file, (void **)&va_space) != UVM_FD_VA_SPACE) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto err;
    }

    // Tell userspace the MM FD is not required and it may be released
    // with no loss of functionality.
    if (!uvm_va_space_mm_enabled(va_space)) {
        status = NV_WARN_NOTHING_TO_DO;
        goto err;
    }

    status = uvm_fd_type_init(filp);
    if (status != NV_OK)
        goto err;

    va_space_mm = &va_space->va_space_mm;
    uvm_spin_lock(&va_space_mm->lock);
    switch (va_space->va_space_mm.state) {
        // We only allow the va_space_mm to be initialised once. If
        // userspace passed the UVM FD to another process it is up to
        // userspace to ensure it also passes the UVM MM FD that
        // initialised the va_space_mm or arranges some other way to keep
        // a reference on the FD.
        case UVM_VA_SPACE_MM_STATE_ALIVE:
            status = NV_ERR_INVALID_STATE;
            goto err_release_unlock;
            break;

        // Once userspace has released the va_space_mm the GPU is
        // effectively dead and no new work can be started. We don't
        // support re-initializing once userspace has closed the FD.
        case UVM_VA_SPACE_MM_STATE_RELEASED:
            status = NV_ERR_PAGE_TABLE_NOT_AVAIL;
            goto err_release_unlock;
            break;

        // Keep the warnings at bay
        case UVM_VA_SPACE_MM_STATE_UNINITIALIZED:
            mm = va_space->va_space_mm.mm;
            if (!mm || !mmget_not_zero(mm)) {
                status = NV_ERR_PAGE_TABLE_NOT_AVAIL;
                goto err_release_unlock;
            }

            va_space_mm->state = UVM_VA_SPACE_MM_STATE_ALIVE;
            break;

        default:
            UVM_ASSERT(0);
            break;
    }
    uvm_spin_unlock(&va_space_mm->lock);
    uvm_fd_type_set(filp, UVM_FD_MM, uvm_file);

    return NV_OK;

err_release_unlock:
    uvm_spin_unlock(&va_space_mm->lock);
    uvm_fd_type_set(filp, UVM_FD_UNINITIALIZED, NULL);

err:
    if (uvm_file)
        fput(uvm_file);

    return status;
}

// Called when opening /dev/nvidia-uvm. This code doesn't take any UVM locks, so
// there's no need to acquire g_uvm_global.pm.lock, but if that changes the PM
// lock will need to be taken.
static int uvm_open(struct inode *inode, struct file *filp)
{
    struct address_space *mapping;
    NV_STATUS status = uvm_global_get_status();

    if (status != NV_OK)
        return -nv_status_to_errno(status);

    mapping = uvm_kvmalloc(sizeof(*mapping));
    if (!mapping)
        return -ENOMEM;

    // By default all struct files on the same inode share the same
    // address_space structure (the inode's) across all processes. This means
    // unmap_mapping_range would unmap virtual mappings across all processes on
    // that inode.
    //
    // Since the UVM driver uses the mapping offset as the VA of the file's
    // process, we need to isolate the mappings to each process.
    address_space_init_once(mapping);
    mapping->host = inode;

    // Some paths in the kernel, for example force_page_cache_readahead which
    // can be invoked from user-space via madvise MADV_WILLNEED and fadvise
    // POSIX_FADV_WILLNEED, check the function pointers within
    // file->f_mapping->a_ops for validity. However, those paths assume that a_ops
    // itself is always valid. Handle that by using the inode's a_ops pointer,
    // which is what f_mapping->a_ops would point to anyway if we weren't re-
    // assigning f_mapping.
    mapping->a_ops = inode->i_mapping->a_ops;

    filp->private_data = NULL;
    filp->f_mapping = mapping;

    return NV_OK;
}

static int uvm_open_entry(struct inode *inode, struct file *filp)
{
   UVM_ENTRY_RET(uvm_open(inode, filp));
}

static void uvm_release_deferred(void *data)
{
    uvm_va_space_t *va_space = data;

    // Since this function is only scheduled to run when uvm_release() fails
    // to trylock-acquire the pm.lock, the following acquisition attempt
    // is expected to block this thread, and cause it to remain blocked until
    // uvm_resume() releases the lock. As a result, the deferred release
    // kthread queue may stall for long periods of time.
    uvm_down_read(&g_uvm_global.pm.lock);

    uvm_moe_slot_pool_release_va_space_mappings(va_space);
    uvm_va_space_destroy(va_space);

    uvm_up_read(&g_uvm_global.pm.lock);
}

static void uvm_release_va_space(struct file *filp, uvm_va_space_t *va_space)
{
    int ret;

    filp->private_data = NULL;
    filp->f_mapping = NULL;

    // Because the kernel discards the status code returned from this release
    // callback, early exit in case of a pm.lock acquisition failure is not
    // an option. Instead, the teardown work normally performed synchronously
    // needs to be scheduled to run after uvm_resume() releases the lock.
    if (uvm_down_read_trylock(&g_uvm_global.pm.lock)) {
        uvm_moe_slot_pool_release_va_space_mappings(va_space);
        uvm_va_space_destroy(va_space);
        uvm_up_read(&g_uvm_global.pm.lock);
    }
    else {
        // Remove references to this inode from the address_space. This isn't
        // strictly necessary, as any CPU mappings of this file have already
        // been destroyed, and va_space->mapping won't be used again. Still,
        // the va_space survives the inode if its destruction is deferred, in
        // which case the references are rendered stale.
        address_space_init_once(va_space->mapping);

        nv_kthread_q_item_init(&va_space->deferred_release_q_item, uvm_release_deferred, va_space);
        ret = nv_kthread_q_schedule_q_item(&g_uvm_global.deferred_release_q, &va_space->deferred_release_q_item);
        UVM_ASSERT(ret != 0);
    }
}

static void uvm_release_mm(struct file *filp, struct file *uvm_file)
{
    uvm_va_space_t *va_space = uvm_va_space_get(uvm_file);
    uvm_va_space_mm_t *va_space_mm = &va_space->va_space_mm;
    struct mm_struct *mm = va_space_mm->mm;

    uvm_kvfree(filp->f_mapping);

    if (uvm_va_space_mm_enabled(va_space)) {
        uvm_va_space_mm_unregister(va_space);

        if (uvm_va_space_mm_enabled(va_space))
            uvm_mmput(mm);

        va_space_mm->mm = NULL;
        fput(uvm_file);
    }
}

static int uvm_release(struct inode *inode, struct file *filp)
{
    void *ptr;
    uvm_fd_type_t fd_type = uvm_fd_type(filp, &ptr);

    switch (fd_type) {
        case UVM_FD_UNINITIALIZED:
            uvm_kvfree(filp->f_mapping);
            break;

        case UVM_FD_VA_SPACE:
            uvm_release_va_space(filp, (uvm_va_space_t *)ptr);
            break;

        case UVM_FD_MM:
            uvm_release_mm(filp, (struct file *)ptr);
            break;

        case UVM_FD_TEST:
            uvm_test_file_release(filp, (uvm_test_file_t *)ptr);
            break;

        default:
            UVM_ASSERT_MSG(0, "Unexpected fd type: %d\n", fd_type);
    }

    return 0;
}

static int uvm_release_entry(struct inode *inode, struct file *filp)
{
   UVM_ENTRY_RET(uvm_release(inode, filp));
}

static void uvm_destroy_vma_managed(struct vm_area_struct *vma, bool make_zombie)
{
    uvm_va_range_managed_t *managed_range, *managed_range_next;
    NvU64 size = 0;

    uvm_assert_rwsem_locked_write(&uvm_va_space_get(vma->vm_file)->lock);
    uvm_for_each_va_range_managed_in_vma_safe(managed_range, managed_range_next, vma) {
        // On exit_mmap (process teardown), current->mm is cleared so
        // uvm_va_range_vma_current would return NULL.
        UVM_ASSERT(uvm_va_range_vma(managed_range) == vma);
        UVM_ASSERT(managed_range->va_range.node.start >= vma->vm_start);
        UVM_ASSERT(managed_range->va_range.node.end   <  vma->vm_end);
        size += uvm_va_range_size(&managed_range->va_range);
        if (make_zombie)
            uvm_va_range_zombify(managed_range);
        else
            uvm_va_range_destroy(&managed_range->va_range, NULL);
    }

    if (vma->vm_private_data) {
        uvm_vma_wrapper_destroy(vma->vm_private_data);
        vma->vm_private_data = NULL;
    }
    UVM_ASSERT(size == vma->vm_end - vma->vm_start);
}

static void uvm_destroy_vma_semaphore_pool(struct vm_area_struct *vma)
{
    uvm_va_range_semaphore_pool_t *semaphore_pool_range;
    uvm_va_space_t *va_space;

    va_space = uvm_va_space_get(vma->vm_file);
    uvm_assert_rwsem_locked(&va_space->lock);
    semaphore_pool_range = uvm_va_range_semaphore_pool_find(va_space, vma->vm_start);
    UVM_ASSERT(semaphore_pool_range &&
               semaphore_pool_range->va_range.node.start   == vma->vm_start &&
               semaphore_pool_range->va_range.node.end + 1 == vma->vm_end);

    uvm_mem_unmap_cpu_user(semaphore_pool_range->mem);
}

// The kernel will also SIGBUS faults to vmas with valid ops but no fault
// handler, but it didn't always do that. Make it explicit so we don't rely on
// the kernel's implementation.
static vm_fault_t uvm_vm_fault_sigbus(struct vm_fault *vmf)
{
    UVM_DBG_PRINT_RL("Fault to address 0x%lx in disabled vma\n", vmf->address);
    return VM_FAULT_SIGBUS;
}

static vm_fault_t uvm_vm_fault_sigbus_entry(struct vm_fault *vmf)
{
    UVM_ENTRY_RET(uvm_vm_fault_sigbus(vmf));
}

static struct vm_operations_struct uvm_vm_ops_disabled =
{
    .fault = uvm_vm_fault_sigbus_entry,
};

static void uvm_disable_vma(struct vm_area_struct *vma)
{
    // In the case of fork, the kernel has already copied the old PTEs over to
    // the child process, so an access in the child might succeed instead of
    // causing a fault. To force a fault we'll unmap it directly here.
    //
    // Note that since the unmap works on file offset, not virtual address, this
    // unmaps both the old and new vmas.
    //
    // In the case of a move (mremap), the kernel will copy the PTEs over later,
    // so it doesn't matter if we unmap here. However, the new vma's open will
    // immediately be followed by a close on the old vma. We call
    // unmap_mapping_range for the close, which also unmaps the new vma because
    // they have the same file offset.
    unmap_mapping_range(vma->vm_file->f_mapping,
                        vma->vm_pgoff << PAGE_SHIFT,
                        vma->vm_end - vma->vm_start,
                        1);

    vma->vm_ops = &uvm_vm_ops_disabled;

    if (vma->vm_private_data) {
        uvm_vma_wrapper_destroy(vma->vm_private_data);
        vma->vm_private_data = NULL;
    }
}

// We can't return an error from uvm_vm_open so on failed splits
// we'll disable *both* vmas. This isn't great behavior for the
// user, but we don't have many options. We could leave the old VA
// range in place but that breaks the model of vmas always
// completely covering VA ranges. We'd have to be very careful
// handling later splits and closes of both that partially-covered
// VA range, and of the vmas which might or might not cover it any
// more.
//
// A failure likely means we're in OOM territory, so this should not
// be common by any means, and the process might die anyway.
static void uvm_vm_open_failure(struct vm_area_struct *original,
                                struct vm_area_struct *new)
{
    uvm_va_space_t *va_space = uvm_va_space_get(new->vm_file);
    static const bool make_zombie = false;

    UVM_ASSERT(va_space == uvm_va_space_get(original->vm_file));
    uvm_assert_rwsem_locked_write(&va_space->lock);

    uvm_destroy_vma_managed(original, make_zombie);
    uvm_disable_vma(original);
    uvm_disable_vma(new);
}

// vm_ops->open cases:
//
// 1) Parent vma is dup'd (fork)
//    This is undefined behavior in the UVM Programming Model. For convenience
//    the parent will continue operating properly, but the child is not
//    guaranteed access to the range.
//
// 2) Original vma is split (munmap, mprotect, mremap, mbind, etc)
//    The UVM Programming Model supports mbind always and supports mprotect if
//    HMM is present. Supporting either of those means all such splitting cases
//    must be handled. This involves splitting the va_range covering the split
//    location. Note that the kernel will never merge us back on two counts: we
//    set VM_MIXEDMAP and we have a ->close callback.
//
// 3) Original vma is moved (mremap)
//    This is undefined behavior in the UVM Programming Model. We'll get an open
//    on the new vma in which we disable operations on the new vma, then a close
//    on the old vma.
//
// Note that since we set VM_DONTEXPAND on the vma we're guaranteed that the vma
// will never increase in size, only shrink/split.
static void uvm_vm_open_managed(struct vm_area_struct *vma)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);
    uvm_va_range_managed_t *managed_range;
    struct vm_area_struct *original;
    NV_STATUS status;
    NvU64 new_end;

    // This is slightly ugly. We need to know the parent vma of this new one,
    // but we can't use the range tree to look up the original because that
    // doesn't handle a vma move operation.
    //
    // However, all of the old vma's fields have been copied into the new vma,
    // and open of the new vma is always called before close of the old (in
    // cases where close will be called immediately afterwards, like move).
    // vma->vm_private_data will thus still point to the original vma that we
    // set in mmap or open.
    //
    // Things to watch out for here:
    // - For splits, the old vma hasn't been adjusted yet so its vm_start and
    //   vm_end region will overlap with this vma's start and end.
    //
    // - For splits and moves, the new vma has not yet been inserted into the
    //   mm's list so vma->vm_prev and vma->vm_next cannot be used, nor will
    //   the new vma show up in find_vma and friends.
    original = ((uvm_vma_wrapper_t*)vma->vm_private_data)->vma;
    vma->vm_private_data = NULL;
    // On fork or move we want to simply disable the new vma
    if (vma->vm_mm != original->vm_mm ||
        (vma->vm_start != original->vm_start && vma->vm_end != original->vm_end)) {
        uvm_disable_vma(vma);
        return;
    }

    // At this point we are guaranteed that the mmap_lock is held in write
    // mode.
    uvm_record_lock_mmap_lock_write(current->mm);

    // Split vmas should always fall entirely within the old one, and be on one
    // side.
    UVM_ASSERT(vma->vm_start >= original->vm_start && vma->vm_end <= original->vm_end);
    UVM_ASSERT(vma->vm_start == original->vm_start || vma->vm_end == original->vm_end);

    // The vma is splitting, so create a new range under this vma if necessary.
    // The kernel handles splits in the middle of the vma by doing two separate
    // splits so we just have to handle one vma splitting in two here.
    if (vma->vm_start == original->vm_start)
        new_end = vma->vm_end - 1; // Left split (new_end is inclusive)
    else
        new_end = vma->vm_start - 1; // Right split (new_end is inclusive)

    uvm_va_space_down_write(va_space);

    vma->vm_private_data = uvm_vma_wrapper_alloc(vma);
    if (!vma->vm_private_data) {
        uvm_vm_open_failure(original, vma);
        goto out;
    }

    // There can be multiple ranges under the vma already. Check if one spans
    // the new split boundary. If so, split it.
    managed_range = uvm_va_range_managed_find(va_space, new_end);
    UVM_ASSERT(managed_range);
    UVM_ASSERT(uvm_va_range_vma_current(managed_range) == original);
    if (managed_range->va_range.node.end != new_end) {
        status = uvm_va_range_split(managed_range, new_end, NULL);
        if (status != NV_OK) {
            UVM_DBG_PRINT("Failed to split VA range, destroying both: %s. "
                          "original vma [0x%lx, 0x%lx) new vma [0x%lx, 0x%lx)\n",
                          nvstatusToString(status),
                          original->vm_start, original->vm_end,
                          vma->vm_start, vma->vm_end);
            uvm_vm_open_failure(original, vma);
            goto out;
        }
    }

    // Point managed_ranges to the new vma
    uvm_for_each_va_range_managed_in_vma(managed_range, vma) {
        UVM_ASSERT(uvm_va_range_vma_current(managed_range) == original);
        managed_range->vma_wrapper = vma->vm_private_data;
    }

out:
    uvm_va_space_up_write(va_space);
    uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_open_managed_entry(struct vm_area_struct *vma)
{
   UVM_ENTRY_VOID(uvm_vm_open_managed(vma));
}

static void uvm_vm_close_managed(struct vm_area_struct *vma)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);
    bool make_zombie = false;
    NV_STATUS status;

    if (current->mm != NULL)
        uvm_record_lock_mmap_lock_write(current->mm);

    // current->mm will be NULL on process teardown, in which case we have
    // special handling.
    if (current->mm == NULL) {
        make_zombie = (va_space->initialization_flags & UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE);
        if (!make_zombie) {
            // If we're not in multi-process mode, then we want to stop all user
            // channels before unmapping the managed allocations to avoid
            // spurious MMU faults in the system log. If we have a va_space_mm
            // then this must've already happened as part of
            // uvm_va_space_mm_shutdown. Otherwise we need to handle it here.
            if (uvm_va_space_mm_enabled(va_space) && current->mm == va_space->va_space_mm.mm) {
                UVM_ASSERT(atomic_read(&va_space->user_channels_stopped));
            }
            else {
                // Stopping channels involves making RM calls, so we have to do
                // that with the VA space lock in read mode.
                uvm_va_space_down_read_rm(va_space);
                if (!atomic_read(&va_space->user_channels_stopped))
                    uvm_va_space_stop_all_user_channels(va_space);
                uvm_va_space_up_read_rm(va_space);
            }
        }
    }

    // See uvm_mmap for why we need this in addition to mmap_lock
    uvm_va_space_down_write(va_space);

    if (current->mm == NULL)
        status = uvm_moe_slot_pool_clear_range_mappings_for_release_locked(va_space,
                                                                           current->mm,
                                                                           vma->vm_start,
                                                                           vma->vm_end - vma->vm_start);
    else
        status = uvm_moe_slot_pool_clear_range_mappings_for_unmap_locked(va_space,
                                                                          current->mm,
                                                                          vma->vm_start,
                                                                          vma->vm_end - vma->vm_start);

    (void)status;

    uvm_destroy_vma_managed(vma, make_zombie);

    uvm_va_space_up_write(va_space);

    if (current->mm != NULL)
        uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_close_managed_entry(struct vm_area_struct *vma)
{
    UVM_ENTRY_VOID(uvm_vm_close_managed(vma));
}

static vm_fault_t uvm_vm_fault(struct vm_fault *vmf)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vmf->vma->vm_file);
    return uvm_va_space_cpu_fault_managed(va_space, vmf);
}

static vm_fault_t uvm_vm_fault_entry(struct vm_fault *vmf)
{
    UVM_ENTRY_RET(uvm_vm_fault(vmf));
}

static struct vm_operations_struct uvm_vm_ops_managed =
{
    .open         = uvm_vm_open_managed_entry,
    .close        = uvm_vm_close_managed_entry,
    .fault        = uvm_vm_fault_entry,
    .page_mkwrite = uvm_vm_fault_entry,
};

// vm operations on semaphore pool allocations only control CPU mappings. Unmapping GPUs,
// freeing the allocation, and destroying the range are handled by UVM_FREE.
static void uvm_vm_open_semaphore_pool(struct vm_area_struct *vma)
{
    struct vm_area_struct *origin_vma = (struct vm_area_struct *)vma->vm_private_data;
    uvm_va_space_t *va_space = uvm_va_space_get(origin_vma->vm_file);
    uvm_va_range_semaphore_pool_t *semaphore_pool_range;
    bool is_fork = (vma->vm_mm != origin_vma->vm_mm);
    NV_STATUS status;

    uvm_record_lock_mmap_lock_write(current->mm);

    uvm_va_space_down_write(va_space);

    semaphore_pool_range = uvm_va_range_semaphore_pool_find(va_space, origin_vma->vm_start);
    UVM_ASSERT(semaphore_pool_range);
    UVM_ASSERT_MSG(semaphore_pool_range &&
                   semaphore_pool_range->va_range.node.start == origin_vma->vm_start &&
                   semaphore_pool_range->va_range.node.end + 1 == origin_vma->vm_end,
                   "origin vma [0x%llx, 0x%llx); va_range [0x%llx, 0x%llx) type %d\n",
                   (NvU64)origin_vma->vm_start,
                   (NvU64)origin_vma->vm_end,
                   semaphore_pool_range->va_range.node.start,
                   semaphore_pool_range->va_range.node.end + 1,
                   semaphore_pool_range->va_range.type);

    // Semaphore pool vmas do not have vma wrappers, but some functions will
    // assume vm_private_data is a wrapper.
    vma->vm_private_data = NULL;
#if defined(VM_WIPEONFORK)
    nv_vm_flags_set(vma, VM_WIPEONFORK);
#endif

    if (is_fork) {
        // If we forked, leave the parent vma alone.
        uvm_disable_vma(vma);

        // uvm_disable_vma unmaps in the parent as well; clear the uvm_mem CPU
        // user mapping metadata and then remap.
        uvm_mem_unmap_cpu_user(semaphore_pool_range->mem);

        status = uvm_mem_map_cpu_user(semaphore_pool_range->mem, semaphore_pool_range->va_range.va_space, origin_vma);
        if (status != NV_OK) {
            UVM_DBG_PRINT("Failed to remap semaphore pool to CPU for parent after fork; status = %d (%s)",
                    status, nvstatusToString(status));
            origin_vma->vm_ops = &uvm_vm_ops_disabled;
        }
    }
    else {
        origin_vma->vm_private_data = NULL;
        origin_vma->vm_ops = &uvm_vm_ops_disabled;
        vma->vm_ops = &uvm_vm_ops_disabled;
        uvm_mem_unmap_cpu_user(semaphore_pool_range->mem);
    }

    uvm_va_space_up_write(va_space);

    uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_open_semaphore_pool_entry(struct vm_area_struct *vma)
{
   UVM_ENTRY_VOID(uvm_vm_open_semaphore_pool(vma));
}

// vm operations on semaphore pool allocations only control CPU mappings. Unmapping GPUs,
// freeing the allocation, and destroying the va_range are handled by UVM_FREE.
static void uvm_vm_close_semaphore_pool(struct vm_area_struct *vma)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);

    if (current->mm != NULL)
        uvm_record_lock_mmap_lock_write(current->mm);

    uvm_va_space_down_read(va_space);

    uvm_destroy_vma_semaphore_pool(vma);

    uvm_va_space_up_read(va_space);

    if (current->mm != NULL)
        uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_close_semaphore_pool_entry(struct vm_area_struct *vma)
{
   UVM_ENTRY_VOID(uvm_vm_close_semaphore_pool(vma));
}

static struct vm_operations_struct uvm_vm_ops_semaphore_pool =
{
    .open         = uvm_vm_open_semaphore_pool_entry,
    .close        = uvm_vm_close_semaphore_pool_entry,
    .fault        = uvm_vm_fault_sigbus_entry,
};

static void uvm_vm_open_device_p2p(struct vm_area_struct *vma)
{
    struct vm_area_struct *origin_vma = (struct vm_area_struct *)vma->vm_private_data;
    uvm_va_space_t *va_space = uvm_va_space_get(origin_vma->vm_file);
    uvm_va_range_t *va_range;
    bool is_fork = (vma->vm_mm != origin_vma->vm_mm);

    uvm_record_lock_mmap_lock_write(current->mm);

    uvm_va_space_down_write(va_space);

    va_range = uvm_va_range_find(va_space, origin_vma->vm_start);
    UVM_ASSERT(va_range);
    UVM_ASSERT_MSG(va_range->type == UVM_VA_RANGE_TYPE_DEVICE_P2P &&
                   va_range->node.start == origin_vma->vm_start &&
                   va_range->node.end + 1 == origin_vma->vm_end,
                   "origin vma [0x%llx, 0x%llx); va_range [0x%llx, 0x%llx) type %d\n",
                   (NvU64)origin_vma->vm_start, (NvU64)origin_vma->vm_end, va_range->node.start,
                   va_range->node.end + 1, va_range->type);

    // Device P2P vmas do not have vma wrappers, but some functions will
    // assume vm_private_data is a wrapper.
    vma->vm_private_data = NULL;
#if defined(VM_WIPEONFORK)
    nv_vm_flags_set(vma, VM_WIPEONFORK);
#endif

    if (is_fork) {
        // If we forked, leave the parent vma alone.
        uvm_disable_vma(vma);

        // uvm_disable_vma unmaps in the parent as well so remap the parent
        uvm_va_range_device_p2p_map_cpu(va_range->va_space, origin_vma, uvm_va_range_to_device_p2p(va_range));
    }
    else {
        // mremap will free the backing pages via unmap so we can't support it.
        origin_vma->vm_private_data = NULL;
        origin_vma->vm_ops = &uvm_vm_ops_disabled;
        vma->vm_ops = &uvm_vm_ops_disabled;
        unmap_mapping_range(va_space->mapping, va_range->node.start, uvm_va_range_size(va_range), 1);
    }

    uvm_va_space_up_write(va_space);

    uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_open_device_p2p_entry(struct vm_area_struct *vma)
{
    UVM_ENTRY_VOID(uvm_vm_open_device_p2p(vma));
}

// Device P2P pages are only mapped on the CPU. Pages are allocated externally
// to UVM but destroying the range must unpin the RM object.
static void uvm_vm_close_device_p2p(struct vm_area_struct *vma)
{
}

static void uvm_vm_close_device_p2p_entry(struct vm_area_struct *vma)
{
    UVM_ENTRY_VOID(uvm_vm_close_device_p2p(vma));
}

static struct vm_operations_struct uvm_vm_ops_device_p2p =
{
    .open         = uvm_vm_open_device_p2p_entry,
    .close        = uvm_vm_close_device_p2p_entry,
    .fault        = uvm_vm_fault_sigbus_entry,
};

static bool va_range_type_expects_mmap(uvm_va_range_type_t type)
{
    switch (type) {
        case UVM_VA_RANGE_TYPE_SEMAPHORE_POOL:
        case UVM_VA_RANGE_TYPE_DEVICE_P2P:
            return true;

        // Although UVM_VA_RANGE_TYPE_MANAGED does support mmap, it doesn't
        // expect mmap to be called on a pre-existing range. mmap itself creates
        // the managed va range.

        default:
            return false;
    }
}

static int uvm_mmap(struct file *filp, struct vm_area_struct *vma)
{
    void *fd_type_ptr;
    uvm_va_space_t *va_space;
    NV_STATUS status = uvm_global_get_status();
    int ret = 0;
    bool vma_wrapper_allocated = false;

    if (status != NV_OK)
        return -nv_status_to_errno(status);

    switch (uvm_fd_type(filp, &fd_type_ptr)) {
        case UVM_FD_VA_SPACE:
            va_space = (uvm_va_space_t *)fd_type_ptr;
            break;

        case UVM_FD_TEST:
            return uvm_test_file_mmap((uvm_test_file_t *)fd_type_ptr, vma);

        default:
            return -EBADFD;
    }

    // When the VA space is associated with an mm, all vmas under the VA space
    // must come from that mm.
    if (uvm_va_space_mm_enabled(va_space)) {
        UVM_ASSERT(va_space->va_space_mm.mm);
        if (va_space->va_space_mm.mm != current->mm)
            return -EOPNOTSUPP;
    }

    // UVM mappings are required to set offset == VA. This simplifies things
    // since we don't have to worry about address aliasing (except for fork,
    // handled separately) and it makes unmap_mapping_range simpler.
    if (vma->vm_start != (vma->vm_pgoff << PAGE_SHIFT)) {
        UVM_DBG_PRINT_RL("vm_start 0x%lx != vm_pgoff 0x%lx\n", vma->vm_start, vma->vm_pgoff << PAGE_SHIFT);
        return -EINVAL;
    }

    // Enforce shared read/writable mappings so we get all fault callbacks
    // without the kernel doing COW behind our backs. The user can still call
    // mprotect to change protections, but that will only hurt user space.
    if ((vma->vm_flags & (VM_SHARED|VM_READ|VM_WRITE)) !=
                         (VM_SHARED|VM_READ|VM_WRITE)) {
        UVM_DBG_PRINT_RL("User requested non-shared or non-writable mapping\n");
        return -EINVAL;
    }

    // If the PM lock cannot be acquired, disable the VMA and report success
    // to the caller. The caller is expected to determine whether the
    // map operation succeeded via an ioctl() call. This is necessary to
    // safely handle MAP_FIXED, which needs to complete atomically to prevent
    // the loss of the virtual address range.
    if (!uvm_down_read_trylock(&g_uvm_global.pm.lock)) {
        uvm_disable_vma(vma);
        return 0;
    }

    uvm_record_lock_mmap_lock_write(current->mm);

    // VM_MIXEDMAP      Required to use vm_insert_page
    //
    // VM_DONTEXPAND    mremap can grow a vma in place without giving us any
    //                  callback. We need to prevent this so our ranges stay
    //                  up-to-date with the vma. This flag doesn't prevent
    //                  mremap from moving the mapping elsewhere, nor from
    //                  shrinking it. We can detect both of those cases however
    //                  with vm_ops->open() and vm_ops->close() callbacks.
    //
    // Using VM_DONTCOPY would be nice, but madvise(MADV_DOFORK) can reset that
    // so we have to handle vm_open on fork anyway. We could disable MADV_DOFORK
    // with VM_IO, but that causes other mapping issues.
    // Make the default behavior be VM_DONTCOPY to avoid the performance impact
    // of removing CPU mappings in the parent on fork()+exec(). Users can call
    // madvise(MDV_DOFORK) if the child process requires access to the
    // allocation.
    nv_vm_flags_set(vma, VM_MIXEDMAP | VM_DONTEXPAND | VM_DONTCOPY);

    vma->vm_ops = &uvm_vm_ops_managed;

    // This identity assignment is needed so uvm_vm_open can find its parent vma
    vma->vm_private_data = uvm_vma_wrapper_alloc(vma);
    if (!vma->vm_private_data) {
        ret = -ENOMEM;
        goto out;
    }
    vma_wrapper_allocated = true;

    // The kernel has taken mmap_lock in write mode, but that doesn't prevent
    // this va_space from being modified by the GPU fault path or from the ioctl
    // path where we don't have this mm for sure, so we have to lock the VA
    // space directly.
    uvm_va_space_down_write(va_space);

    // uvm_va_range_create_mmap will catch collisions. Below are some example
    // cases which can cause collisions. There may be others.
    // 1) An overlapping range was previously created with an ioctl, for example
    //    for an external mapping.
    // 2) This file was passed to another process via a UNIX domain socket
    status = uvm_va_range_create_mmap(va_space, current->mm, vma->vm_private_data, NULL);

    if (status == NV_ERR_UVM_ADDRESS_IN_USE) {
        uvm_va_range_t *existing_range = uvm_va_range_find(va_space, vma->vm_start);

        // Does the existing range exactly match the vma and expects mmap?
        if (existing_range &&
            existing_range->node.start == vma->vm_start &&
            existing_range->node.end + 1 == vma->vm_end &&
            va_range_type_expects_mmap(existing_range->type)) {

            // We speculatively initialized the managed vma before checking for
            // collisions because we expect successful insertion to be the
            // common case. Undo that.
            uvm_vma_wrapper_destroy(vma->vm_private_data);
            vma_wrapper_allocated = false;
            vma->vm_private_data = vma;

            switch (existing_range->type) {
                case UVM_VA_RANGE_TYPE_SEMAPHORE_POOL:
                    vma->vm_ops = &uvm_vm_ops_semaphore_pool;
                    status = uvm_mem_map_cpu_user(uvm_va_range_to_semaphore_pool(existing_range)->mem, va_space, vma);
                    break;

                case UVM_VA_RANGE_TYPE_DEVICE_P2P:
                    vma->vm_ops = &uvm_vm_ops_device_p2p;
                    status = uvm_va_range_device_p2p_map_cpu(va_space,
                                                             vma,
                                                             uvm_va_range_to_device_p2p(existing_range));
                    break;

                default:
                    UVM_ASSERT(0);
                    break;
            }
        }
    }

    if (status != NV_OK) {
        UVM_DBG_PRINT_RL("Failed to create or map VA range for vma [0x%lx, 0x%lx): %s\n",
                         vma->vm_start, vma->vm_end, nvstatusToString(status));
        ret = -nv_status_to_errno(status);
    }

    uvm_va_space_up_write(va_space);

out:
    if (ret != 0 && vma_wrapper_allocated)
        uvm_vma_wrapper_destroy(vma->vm_private_data);

    uvm_record_unlock_mmap_lock_write(current->mm);

    uvm_up_read(&g_uvm_global.pm.lock);

    return ret;
}

bool uvm_vma_is_managed(struct vm_area_struct *vma)
{
    return vma->vm_ops == &uvm_vm_ops_disabled ||
           vma->vm_ops == &uvm_vm_ops_managed ||
           vma->vm_ops == &uvm_vm_ops_semaphore_pool;
}

static int uvm_mmap_entry(struct file *filp, struct vm_area_struct *vma)
{
   UVM_ENTRY_RET(uvm_mmap(filp, vma));
}

static NV_STATUS uvm_api_initialize(UVM_INITIALIZE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space;
    NV_STATUS status;
    uvm_fd_type_t old_fd_type;

    // Normally we expect private_data == UVM_FD_UNINITIALIZED. However multiple
    // threads may call this ioctl concurrently so we have to be careful to
    // avoid initializing multiple va_spaces and/or leaking memory. To do this
    // we do an atomic compare and swap. Only one thread will observe
    // UVM_FD_UNINITIALIZED and that thread will allocate and setup the
    // va_space.
    //
    // Other threads will either see UVM_FD_INITIALIZING or UVM_FD_VA_SPACE. In
    // the case of UVM_FD_VA_SPACE we return success if and only if the
    // initialization flags match. If another thread is still initializing the
    // va_space we return NV_ERR_BUSY_RETRY.
    //
    // If va_space initialization fails we return the failure code and reset the
    // FD state back to UVM_FD_UNINITIALIZED to allow another initialization
    // attempt to be made. This is safe because other threads will have only had
    // a chance to observe UVM_FD_INITIALIZING and not UVM_FD_VA_SPACE in this
    // case.
    old_fd_type = uvm_fd_type_init_cas(filp);
    switch (old_fd_type) {
        case UVM_FD_UNINITIALIZED:
            status = uvm_va_space_create(filp->f_mapping, &va_space, params->flags);
            if (status != NV_OK) {
                uvm_fd_type_set(filp, UVM_FD_UNINITIALIZED, NULL);
                return status;
            }

            uvm_fd_type_set(filp, UVM_FD_VA_SPACE, va_space);
            break;

        case UVM_FD_VA_SPACE:
            va_space = uvm_va_space_get(filp);
            if (params->flags != va_space->initialization_flags)
                status = NV_ERR_INVALID_ARGUMENT;
            else
                status = NV_OK;

            break;

        case UVM_FD_MM:
        case UVM_FD_TEST:
            status = NV_ERR_INVALID_ARGUMENT;
            break;

        case UVM_FD_INITIALIZING:
            status = NV_ERR_BUSY_RETRY;
            break;

        default:
            UVM_ASSERT(0);
            status = NV_ERR_INVALID_STATE; // Quiet compiler warnings
            break;
    }

    return status;
}

static NV_STATUS uvm_api_pageable_mem_access(UVM_PAGEABLE_MEM_ACCESS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    params->pageableMemAccess = uvm_va_space_pageable_mem_access_supported(va_space) ? NV_TRUE : NV_FALSE;
    return NV_OK;
}

static NvU64 uvm_phase_f_page_overlap_bytes(NvU64 page_start, NvU64 query_start, NvU64 query_end)
{
    NvU64 page_end = page_start + PAGE_SIZE;
    NvU64 overlap_start = max(page_start, query_start);
    NvU64 overlap_end = min(page_end, query_end);

    if (overlap_start >= overlap_end)
        return 0;

    return overlap_end - overlap_start;
}

static void uvm_phase_f_add_resident_bytes_from_block(uvm_va_block_t *block,
                                                      uvm_gpu_t *gpu,
                                                      NvU64 query_start,
                                                      NvU64 query_end,
                                                      UVM_PHASE_F_RANGE_RESULT *result)
{
    uvm_page_mask_t *gpu_resident = uvm_va_block_resident_mask_get(block, gpu->id, NUMA_NO_NODE);
    uvm_page_mask_t *cpu_resident = uvm_va_block_resident_mask_get(block, UVM_ID_CPU, NUMA_NO_NODE);
    NvU64 first_addr;
    NvU64 last_addr;
    uvm_page_index_t first_page_index;
    uvm_page_index_t last_page_index;
    uvm_page_index_t page_index;

    if (query_start >= query_end)
        return;

    first_addr = UVM_ALIGN_DOWN(max(query_start, block->start), PAGE_SIZE);
    last_addr = UVM_ALIGN_DOWN(min(query_end - 1, block->end), PAGE_SIZE);
    first_page_index = uvm_va_block_cpu_page_index(block, first_addr);
    last_page_index = uvm_va_block_cpu_page_index(block, last_addr);

    for (page_index = first_page_index; page_index <= last_page_index; ++page_index) {
        NvU64 page_start = block->start + (NvU64)page_index * PAGE_SIZE;
        NvU64 bytes = uvm_phase_f_page_overlap_bytes(page_start, query_start, query_end);
        bool on_gpu = gpu_resident && uvm_page_mask_test(gpu_resident, page_index);
        bool on_cpu = cpu_resident && uvm_page_mask_test(cpu_resident, page_index);

        if (on_gpu)
            result->resident_gpu_bytes += bytes;
        if (on_cpu)
            result->resident_cpu_bytes += bytes;
        if (on_gpu || on_cpu)
            result->resident_any_bytes += bytes;
        else
            result->resident_unknown_bytes += bytes;
    }
}

static NV_STATUS uvm_phase_f_query_resident_bytes(const NvProcessorUuid *gpu_uuid,
                                                  NvBool async,
                                                  NvU64 base,
                                                  NvU64 length,
                                                  UVM_PHASE_F_RANGE_RESULT *result,
                                                  uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    bool retained_mm = false;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    if (uvm_va_space_mm_enabled(va_space)) {
        mm = uvm_va_space_mm_retain(va_space);
        retained_mm = mm != NULL;
    }
    else if (!(current->flags & PF_KTHREAD)) {
        mm = current->mm;
    }

    if (mm)
        uvm_down_read_mmap_lock(mm);

    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(query_end - 1, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block = uvm_va_range_block(managed_range, block_index);
            const NvU64 block_start = block ? max(query_start, block->start) : 0;
            const NvU64 block_end = block ? min(query_end, block->end + 1) : 0;

            if (!block) {
                NvU64 managed_aligned_start = UVM_ALIGN_DOWN(managed_range->va_range.node.start, UVM_VA_BLOCK_SIZE);
                NvU64 missing_block_start = max(query_start,
                                                max(managed_range->va_range.node.start,
                                                    managed_aligned_start + (NvU64)block_index * UVM_VA_BLOCK_SIZE));
                NvU64 missing_block_end = min(query_end,
                                              min(managed_range->va_range.node.end + 1,
                                                  managed_aligned_start +
                                                      (NvU64)(block_index + 1) * UVM_VA_BLOCK_SIZE));
                if (missing_block_start < missing_block_end)
                    result->resident_unknown_bytes += missing_block_end - missing_block_start;
                continue;
            }

            uvm_mutex_lock(&block->lock);
            if (!async)
                status = uvm_tracker_wait(&block->tracker);
            if (status == NV_OK)
                uvm_phase_f_add_resident_bytes_from_block(block, gpu, block_start, block_end, result);
            uvm_mutex_unlock(&block->lock);

            if (status != NV_OK)
                goto out;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    if (mm)
        uvm_up_read_mmap_lock(mm);
    if (retained_mm)
        uvm_va_space_mm_release(va_space);

    uvm_phase_f_route_log("phase_f_resident_bytes base=0x%llx length=%llu status=0x%x queried=%llu gpu=%llu cpu=%llu any=%llu unknown=%llu\n",
                          base,
                          length,
                          status,
                          result->queried_bytes,
                          result->resident_gpu_bytes,
                          result->resident_cpu_bytes,
                          result->resident_any_bytes,
                          result->resident_unknown_bytes);
    return status;
}

static NV_STATUS uvm_phase_f_prepare_remote_read(const NvProcessorUuid *gpu_uuid,
                                                 NvBool async,
                                                 NvU64 base,
                                                 NvU64 length,
                                                 bool enqueue_bg,
                                                 bool schedule_bg,
                                                 UVM_PHASE_F_RANGE_RESULT *result,
                                                 uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 mapped_pages = 0;
    NvU64 bg_enqueued_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_mapped_pages = 0;
            NvU32 block_bg_enqueued_pages = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_premap_readmostly_remote_read(block,
                                                                                        block_context,
                                                                                        gpu->id,
                                                                                        block_region,
                                                                                        !async,
                                                                                        &block_mapped_pages));
            if (status != NV_OK)
                goto out;

            mapped_pages += block_mapped_pages;

            if (enqueue_bg) {
                status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                                 NULL,
                                                 schedule_bg ?
                                                    uvm_va_block_enqueue_readmostly_bg_promote(block,
                                                                                               block_context,
                                                                                               gpu->id,
                                                                                               block_region,
                                                                                               &block_bg_enqueued_pages) :
                                                    uvm_va_block_enqueue_readmostly_bg_promote_deferred(block,
                                                                                                        block_context,
                                                                                                        gpu->id,
                                                                                                        block_region,
                                                                                                        &block_bg_enqueued_pages));
                if (status != NV_OK)
                    goto out;

                bg_enqueued_pages += block_bg_enqueued_pages;
            }
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = mapped_pages * PAGE_SIZE;
    result->resident_cpu_bytes = candidate_pages * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_prepare_remote_read base=0x%llx length=%llu status=0x%x gpu=%s async=%u enqueue_bg=%u schedule_bg=%u covered=%llu candidate_pages=%llu mapped_pages=%llu bg_enqueued_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          enqueue_bg,
                          schedule_bg,
                          covered_bytes,
                          candidate_pages,
                          mapped_pages,
                          bg_enqueued_pages);

    return status;
}

static NV_STATUS uvm_phase_f_start_bg_promotion(const NvProcessorUuid *gpu_uuid,
                                                NvBool async,
                                                NvU64 base,
                                                NvU64 length,
                                                UVM_PHASE_F_RANGE_RESULT *result,
                                                uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 started_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_started_pages = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_start_readmostly_bg_promote(block,
                                                                                      block_context,
                                                                                      gpu->id,
                                                                                      block_region,
                                                                                      &block_started_pages));
            if (status != NV_OK)
                goto out;

            started_pages += block_started_pages;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = started_pages * PAGE_SIZE;
    result->resident_cpu_bytes = candidate_pages * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_start_bg_promotion base=0x%llx length=%llu status=0x%x gpu=%s async=%u covered=%llu candidate_pages=%llu started_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          covered_bytes,
                          candidate_pages,
                          started_pages);

    return status;
}

static NV_STATUS uvm_phase_f_set_bg_promotion_paused(NvU64 base,
                                                     NvU64 length,
                                                     bool pause,
                                                     UVM_PHASE_F_RANGE_RESULT *result,
                                                     uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_blocks = 0;
    NvU64 touched_blocks = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            NvU32 block_touched = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            covered_bytes += block_end - block_start + 1;
            candidate_blocks++;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_set_bg_promote_paused(block,
                                                                                pause,
                                                                                &block_touched));
            if (status != NV_OK)
                goto out;

            touched_blocks += block_touched;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = touched_blocks;
    result->resident_cpu_bytes = candidate_blocks;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    if (pause)
        uvm_phase_f_route_log("phase_f_pause_bg_promotion base=0x%llx length=%llu status=0x%x covered=%llu blocks=%llu touched=%llu\n",
                              base,
                              length,
                              status,
                              covered_bytes,
                              candidate_blocks,
                              touched_blocks);
    else
        uvm_phase_f_route_log("phase_f_resume_bg_promotion base=0x%llx length=%llu status=0x%x covered=%llu blocks=%llu touched=%llu\n",
                              base,
                              length,
                              status,
                              covered_bytes,
                              candidate_blocks,
                              touched_blocks);

    return status;
}

static NV_STATUS uvm_phase_f_gpu_only_soft(const NvProcessorUuid *gpu_uuid,
                                           NvBool async,
                                           NvU64 base,
                                           NvU64 length,
                                           bool read_only,
                                           bool clear,
                                           bool preserve_cpu_backing,
                                           UVM_PHASE_F_RANGE_RESULT *result,
                                           uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 applied_pages = 0;
    NvU64 cpu_unmapped_pages = 0;
    NvU64 gpu_mapped_pages = 0;
    NvU64 cpu_preserved_pages = 0;
    NvU64 cleared_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    if (!clear) {
        gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
        if (!gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto out;
        }
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_applied_pages = 0;
            NvU32 block_cpu_unmapped_pages = 0;
            NvU32 block_gpu_mapped_pages = 0;
            NvU32 block_cpu_preserved_pages = 0;
            NvU32 block_cleared_pages = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            if (clear) {
                status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                                 NULL,
                                                 uvm_va_block_clear_gpu_only_soft(block,
                                                                                  block_context,
                                                                                  block_region,
                                                                                  &block_cleared_pages));
            }
            else {
                uvm_va_block_retry_t block_retry;

                status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                                 &block_retry,
                                                 uvm_va_block_apply_gpu_only_soft(block,
                                                                                  &block_retry,
                                                                                  block_context,
                                                                                  gpu->id,
                                                                                  block_region,
                                                                                  read_only,
                                                                                  preserve_cpu_backing,
                                                                                  !async,
                                                                                  &block_applied_pages,
                                                                                  &block_cpu_unmapped_pages,
                                                                                  &block_gpu_mapped_pages,
                                                                                  &block_cpu_preserved_pages));
            }

            if (status != NV_OK)
                goto out;

            applied_pages += block_applied_pages;
            cpu_unmapped_pages += block_cpu_unmapped_pages;
            gpu_mapped_pages += block_gpu_mapped_pages;
            cpu_preserved_pages += block_cpu_preserved_pages;
            cleared_pages += block_cleared_pages;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = (clear ? cleared_pages : applied_pages) * PAGE_SIZE;
    result->resident_cpu_bytes = (preserve_cpu_backing ? cpu_preserved_pages : cpu_unmapped_pages) * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_gpu_only_soft base=0x%llx length=%llu status=0x%x gpu=%s async=%u read_only=%u clear=%u preserve_cpu_backing=%u covered=%llu candidate_pages=%llu applied_pages=%llu cpu_unmapped_pages=%llu gpu_mapped_pages=%llu cpu_preserved_pages=%llu cleared_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          read_only ? 1 : 0,
                          clear ? 1 : 0,
                          preserve_cpu_backing ? 1 : 0,
                          covered_bytes,
                          candidate_pages,
                          applied_pages,
                          cpu_unmapped_pages,
                          gpu_mapped_pages,
                          cpu_preserved_pages,
                          cleared_pages);

    return status;
}

static bool uvm_phase_f_managed_range_is_expert_weight(uvm_va_range_managed_t *managed_range)
{
    /* The caller already supplied an explicit VA range/region id. */
    return managed_range != NULL;
}

static NV_STATUS uvm_phase_f_expert_read_dup_fast(const NvProcessorUuid *gpu_uuid,
                                                  NvBool async,
                                                  NvU64 base,
                                                  NvU64 length,
                                                  UVM_PHASE_F_RANGE_RESULT *result,
                                                  uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 applied_pages = 0;
    NvU64 cpu_preserved_pages = 0;
    NvU64 gpu_mapped_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        if (!uvm_phase_f_managed_range_is_expert_weight(managed_range)) {
            status = NV_ERR_INVALID_ADDRESS;
            goto out;
        }

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_committed_pages = 0;
            uvm_va_block_retry_t block_retry;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             &block_retry,
                                             uvm_va_block_promote_read_duplicate_now(block,
                                                                                     &block_retry,
                                                                                     block_context,
                                                                                     gpu->id,
                                                                                     block_region,
                                                                                     !async,
                                                                                     &block_committed_pages));
            if (status != NV_OK)
                goto out;

            applied_pages += block_committed_pages;
            gpu_mapped_pages += block_committed_pages;
            cpu_preserved_pages += block_committed_pages;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = applied_pages * PAGE_SIZE;
    result->resident_cpu_bytes = cpu_preserved_pages * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_expert_read_dup_fast base=0x%llx length=%llu status=0x%x gpu=%s async=%u covered=%llu candidate_pages=%llu applied_pages=%llu gpu_mapped_pages=%llu cpu_preserved_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          covered_bytes,
                          candidate_pages,
                          applied_pages,
                          gpu_mapped_pages,
                          cpu_preserved_pages);

    return status;
}

static NV_STATUS uvm_phase_f_execute_range(NvU32 action,
                                           const NvProcessorUuid *gpu_uuid,
                                           NvBool async,
                                           NvU64 base,
                                           NvU64 length,
                                           UVM_PHASE_F_RANGE_RESULT *result,
                                           struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm = NULL;
    uvm_va_range_t *va_range = NULL;
    uvm_api_range_type_t range_type = UVM_API_RANGE_TYPE_INVALID;
    const char *range_type_name = "INVALID";
    NvU64 last_address = 0;
    bool range_invalid = true;
    bool range_empty = true;
    bool skip_range_diag = action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST ||
                           action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS;

    result->status = NV_OK;
    result->resident_on_count = 0;
    result->resident_on_cpu = NV_FALSE;
    result->resident_on_gpu = NV_FALSE;
    result->queried_bytes = 0;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (!skip_range_diag && base != 0 && length != 0) {
        range_invalid = uvm_api_range_invalid(base, length);
        if (!range_invalid)
            last_address = base + length - 1;

        mm = uvm_va_space_mm_or_current_retain_lock(va_space);
        uvm_va_space_down_read(va_space);

        if (!range_invalid) {
            va_range = uvm_va_range_find(va_space, base);
            range_empty = uvm_va_space_range_empty(va_space, base, last_address);
            range_type = uvm_api_range_type_check(va_space, mm, base, length);
        }

        if (range_type == UVM_API_RANGE_TYPE_MANAGED)
            range_type_name = "MANAGED";
        else if (range_type == UVM_API_RANGE_TYPE_HMM)
            range_type_name = "HMM";
        else if (range_type == UVM_API_RANGE_TYPE_ATS)
            range_type_name = "ATS";

        uvm_phase_f_route_log("phase_f_range_diag action=%u base=0x%llx length=%llu last=0x%llx range_invalid=%u base_page_aligned=%u length_page_aligned=%u type=%s range_empty=%u va_range=%p va_range_type=%d va_start=0x%llx va_end=0x%llx va_space=%p mm=%p current_mm=%p\n",
                              action,
                              base,
                              length,
                              last_address,
                              range_invalid,
                              IS_ALIGNED(base, PAGE_SIZE),
                              IS_ALIGNED(length, PAGE_SIZE),
                              range_type_name,
                              range_empty,
                              va_range,
                              va_range ? va_range->type : UVM_VA_RANGE_TYPE_INVALID,
                              va_range ? va_range->node.start : 0,
                              va_range ? va_range->node.end : 0,
                              va_space,
                              mm,
                              current->mm);

        uvm_va_space_up_read(va_space);
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    }

    if (base == 0 || length == 0) {
        status = NV_ERR_INVALID_ARGUMENT;
    }
    else if (action == UVM_PHASE_F_ACTION_ADVISE_CPU) {
        UVM_SET_PREFERRED_LOCATION_PARAMS *advise_params =
            uvm_kvmalloc_zero(sizeof(*advise_params));

        if (!advise_params) {
            status = NV_ERR_NO_MEMORY;
        }
        else {
            advise_params->requestedBase = base;
            advise_params->length = length;
            advise_params->preferredLocation = NV_PROCESSOR_UUID_CPU_DEFAULT;
            advise_params->preferredCpuNumaNode = NUMA_NO_NODE;

            status = uvm_api_set_preferred_location(advise_params, filp);
            uvm_kvfree(advise_params);
        }
    }
    else if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU ||
             action == UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP ||
             action == UVM_PHASE_F_ACTION_PREFETCH_GPU) {
        UVM_MIGRATE_PARAMS *migrate_params =
            uvm_kvmalloc_zero(sizeof(*migrate_params));

        if (!migrate_params) {
            status = NV_ERR_NO_MEMORY;
        }
        else if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP) {
            status = uvm_migrate_managed_range_to_cpu(va_space,
                                                      base,
                                                      length,
                                                      (async ? UVM_MIGRATE_FLAG_ASYNC : 0) |
                                                      UVM_MIGRATE_FLAG_SKIP_CPU_MAP);
            uvm_kvfree(migrate_params);
        }
        else {
            migrate_params->base = base;
            migrate_params->length = length;
            migrate_params->destinationUuid =
                action == UVM_PHASE_F_ACTION_PREFETCH_CPU ?
                    NV_PROCESSOR_UUID_CPU_DEFAULT : *gpu_uuid;
            migrate_params->flags = async ? UVM_MIGRATE_FLAG_ASYNC : 0;
            migrate_params->semaphoreAddress = 0;
            migrate_params->semaphorePayload = 0;
            migrate_params->cpuNumaNode = NUMA_NO_NODE;

            status = uvm_api_migrate(migrate_params, filp);
            uvm_kvfree(migrate_params);
        }
    }
    else if (action == UVM_PHASE_F_ACTION_QUERY_RESIDENCY) {
        UVM_TEST_VA_RESIDENCY_INFO_PARAMS *query_params =
            uvm_kvmalloc_zero(sizeof(*query_params));
        NvU32 resident_index;

        if (!query_params) {
            status = NV_ERR_NO_MEMORY;
        }
        else {
            query_params->lookup_address = base;
            query_params->is_async = async;

            status = uvm_test_va_residency_info(query_params, filp);
            if (status == NV_OK) {
                result->resident_on_count = query_params->resident_on_count;

                for (resident_index = 0;
                     resident_index < query_params->resident_on_count &&
                     resident_index < UVM_MAX_PROCESSORS;
                     ++resident_index) {
                    if (uvm_uuid_eq(&query_params->resident_on[resident_index],
                                    &NV_PROCESSOR_UUID_CPU_DEFAULT))
                        result->resident_on_cpu = NV_TRUE;

                    if (uvm_uuid_eq(&query_params->resident_on[resident_index],
                                    gpu_uuid))
                        result->resident_on_gpu = NV_TRUE;
                }
            }

            uvm_kvfree(query_params);

            if (status == NV_OK) {
                status = uvm_phase_f_query_resident_bytes(gpu_uuid,
                                                          async,
                                                          base,
                                                          length,
                                                          result,
                                                          va_space);
            }
        }
    }
    else if (action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ ||
             action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG ||
             action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED) {
        status = uvm_phase_f_prepare_remote_read(gpu_uuid,
                                                 async,
                                                 base,
                                                 length,
                                                 action != UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ,
                                                 action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG,
                                                 result,
                                                 va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_START_BG_PROMOTION) {
        status = uvm_phase_f_start_bg_promotion(gpu_uuid,
                                                async,
                                                base,
                                                length,
                                                result,
                                                va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION ||
             action == UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION) {
        bool pause = action == UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION;

        uvm_va_block_note_bg_promote_gate_action(pause);
        status = uvm_phase_f_set_bg_promotion_paused(base,
                                                     length,
                                                     pause,
                                                     result,
                                                     va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT ||
             action == UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT ||
             action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT ||
             action == UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY) {
        status = uvm_phase_f_gpu_only_soft(gpu_uuid,
                                           async,
                                           base,
                                           length,
                                           action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT ||
                                           action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT,
                                           action == UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY,
                                           action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT,
                                           result,
                                           va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST ||
             action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS) {
        status = uvm_phase_f_expert_read_dup_fast(gpu_uuid,
                                                  async,
                                                  base,
                                                  length,
                                                  result,
                                                  va_space);
    }
    else {
        status = NV_ERR_INVALID_ARGUMENT;
    }

    result->status = status;
    uvm_phase_f_route_log("phase_f_range_action_end action=%u base=0x%llx end=0x%llx length=%llu status=0x%x gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu unknown_bytes=%llu async=%u\n",
                          action,
                          base,
                          base + length,
                          length,
                          status,
                          result->resident_gpu_bytes,
                          result->resident_cpu_bytes,
                          result->resident_any_bytes,
                          result->resident_unknown_bytes,
                          async);
    return status;
}

static NV_STATUS uvm_phase_f_note_status(NV_STATUS status,
                                         NvU32 *success_count,
                                         NvU32 *failure_count,
                                         NV_STATUS *first_error)
{
    if (status == NV_OK) {
        ++*success_count;
    }
    else {
        ++*failure_count;
        if (*first_error == NV_OK)
            *first_error = status;
    }

    return status;
}

typedef struct
{
    NvU64 covered_bytes;
    NvU64 candidate_pages;
    NvU64 committed_pages;
} uvm_phase_f_expert_batch_range_state_t;

typedef struct
{
    uvm_phase_f_expert_batch_range_state_t range_states[UVM_PHASE_F_MAX_RANGES];
    uvm_va_block_expert_batch_entry_t entries[UVM_PHASE_F_MAX_RANGES];
    NvU32 entry_range_indices[UVM_PHASE_F_MAX_RANGES];
    uvm_va_range_managed_t *single_managed_range;
    size_t single_first_block_index;
    size_t single_last_block_index;
    DECLARE_BITMAP(single_block_seen, UVM_PHASE_F_MAX_RANGES);
    NvU32 single_block_owner[UVM_PHASE_F_MAX_RANGES];
} uvm_phase_f_expert_batch_region_work_t;

static void uvm_phase_f_result_init(UVM_PHASE_F_RANGE_RESULT *result, NvU64 queried_bytes)
{
    result->status = NV_OK;
    result->resident_on_count = 0;
    result->resident_on_cpu = NV_FALSE;
    result->resident_on_gpu = NV_FALSE;
    result->queried_bytes = queried_bytes;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;
}

static NV_STATUS uvm_phase_f_expert_batch_count_blocks(uvm_va_space_t *va_space,
                                                       UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params,
                                                       uvm_phase_f_expert_batch_range_state_t *range_states,
                                                       size_t *entry_count_out)
{
    NvU32 range_index;

    *entry_count_out = 0;

    for (range_index = 0; range_index < params->count; ++range_index) {
        UVM_PHASE_F_RANGE *range = &params->ranges[range_index];
        uvm_phase_f_expert_batch_range_state_t *range_state = &range_states[range_index];
        uvm_va_range_managed_t *managed_range = NULL;
        uvm_va_range_managed_t *managed_range_last = NULL;
        NvU64 last_address;

        uvm_phase_f_result_init(&params->results[range_index], range->length);

        if (range->base == 0 || range->length == 0 || uvm_api_range_invalid(range->base, range->length))
            return NV_WARN_MORE_PROCESSING_REQUIRED;

        last_address = range->base + range->length - 1;

        uvm_for_each_va_range_managed_in_contig(managed_range, va_space, range->base, last_address) {
            NvU64 range_start = max(range->base, managed_range->va_range.node.start);
            NvU64 range_end = min(last_address, managed_range->va_range.node.end);
            size_t first_block_index;
            size_t last_block_index;
            size_t block_count;

            managed_range_last = managed_range;

            if (!uvm_phase_f_managed_range_is_expert_weight(managed_range))
                return NV_WARN_MORE_PROCESSING_REQUIRED;

            first_block_index = uvm_va_range_block_index(managed_range, range_start);
            last_block_index = uvm_va_range_block_index(managed_range, range_end);
            block_count = last_block_index - first_block_index + 1;

            if (block_count > UVM_PHASE_F_MAX_RANGES || *entry_count_out > UVM_PHASE_F_MAX_RANGES - block_count)
                return NV_WARN_MORE_PROCESSING_REQUIRED;

            *entry_count_out += block_count;
            range_state->covered_bytes += range_end - range_start + 1;
            range_state->candidate_pages += (range_end - range_start + 1) / PAGE_SIZE;
        }

        if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
            return NV_WARN_MORE_PROCESSING_REQUIRED;

        if (range_state->covered_bytes != range->length)
            return NV_WARN_MORE_PROCESSING_REQUIRED;
    }

    if (*entry_count_out == 0)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_fill_entries(uvm_va_space_t *va_space,
                                                       UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params,
                                                       uvm_va_block_expert_batch_entry_t *entries,
                                                       NvU32 *entry_range_indices,
                                                       size_t entry_count)
{
    size_t entry_index = 0;
    NvU32 range_index;

    for (range_index = 0; range_index < params->count; ++range_index) {
        UVM_PHASE_F_RANGE *range = &params->ranges[range_index];
        uvm_va_range_managed_t *managed_range = NULL;
        NvU64 last_address = range->base + range->length - 1;

        uvm_for_each_va_range_managed_in_contig(managed_range, va_space, range->base, last_address) {
            NvU64 range_start = max(range->base, managed_range->va_range.node.start);
            NvU64 range_end = min(last_address, managed_range->va_range.node.end);
            size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
            size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
            size_t block_index;

            for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
                uvm_va_block_t *block;
                NvU64 block_start;
                NvU64 block_end;
                NV_STATUS status;

                UVM_ASSERT(entry_index < entry_count);

                status = uvm_va_range_block_create(managed_range, block_index, &block);
                if (status != NV_OK)
                    return status;

                block_start = max(range->base, block->start);
                block_end = min(last_address, block->end);
                if (block_start > block_end)
                    continue;

                entries[entry_index].va_block = block;
                entries[entry_index].region = uvm_va_block_region_from_start_end(block, block_start, block_end);
                entries[entry_index].committed_pages = 0;
                entry_range_indices[entry_index] = range_index;
                ++entry_index;
            }
        }
    }

    UVM_ASSERT(entry_index == entry_count);
    return NV_OK;
}

static bool uvm_phase_f_registered_region_is_valid(const uvm_phase_f_region_t *region)
{
    return region && region->region_id != 0 && region->base != 0 && region->length != 0;
}

static NV_STATUS uvm_phase_f_expert_batch_collect_region(uvm_va_space_t *va_space,
                                                         NvU64 region_id,
                                                         NvU32 range_index,
                                                         uvm_phase_f_expert_batch_range_state_t *range_state,
                                                         uvm_va_block_expert_batch_entry_t *entries,
                                                         NvU32 *entry_range_indices,
                                                         size_t *entry_count_inout)
{
    uvm_phase_f_region_t *region;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    NvU64 base;
    NvU64 length;
    NvU64 last_address;

    if (region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    if (!uvm_phase_f_registered_region_is_valid(region))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    base = region->base;
    length = region->length;
    if (base == 0 || length == 0 || uvm_api_range_invalid(base, length))
        return NV_WARN_MORE_PROCESSING_REQUIRED;
    if (!IS_ALIGNED(base, UVM_VA_BLOCK_SIZE) || !IS_ALIGNED(length, UVM_VA_BLOCK_SIZE))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    last_address = base + length - 1;
    range_state->covered_bytes = 0;
    range_state->candidate_pages = 0;
    range_state->committed_pages = 0;

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        NvU64 range_start = max(base, managed_range->va_range.node.start);
        NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        size_t first_block_index;
        size_t last_block_index;
        size_t block_index;

        managed_range_last = managed_range;

        first_block_index = uvm_va_range_block_index(managed_range, range_start);
        last_block_index = uvm_va_range_block_index(managed_range, range_end);

        if (last_block_index - first_block_index + 1 > UVM_PHASE_F_MAX_RANGES ||
            *entry_count_inout > UVM_PHASE_F_MAX_RANGES - (last_block_index - first_block_index + 1)) {
            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }

        range_state->covered_bytes += range_end - range_start + 1;
        range_state->candidate_pages += (range_end - range_start + 1) / PAGE_SIZE;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            NV_STATUS status;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                return status;

            block_start = max(base, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            entries[*entry_count_inout].va_block = block;
            entries[*entry_count_inout].region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            entries[*entry_count_inout].committed_pages = 0;
            entry_range_indices[*entry_count_inout] = range_index;
            ++*entry_count_inout;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (range_state->covered_bytes != length)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_probe_single_va_range_region(uvm_va_space_t *va_space,
                                                                       NvU64 region_id,
                                                                       NvU32 range_index,
                                                                       uvm_phase_f_expert_batch_region_work_t *work)
{
    uvm_phase_f_region_t *region;
    uvm_va_range_t *va_range;
    uvm_va_range_managed_t *managed_range;
    uvm_phase_f_expert_batch_range_state_t *range_state = &work->range_states[range_index];
    NvU64 base;
    NvU64 length;
    NvU64 last_address;
    size_t first_block_index;
    size_t last_block_index;
    size_t block_count;

    if (region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    if (!uvm_phase_f_registered_region_is_valid(region))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    base = region->base;
    length = region->length;
    if (base == 0 || length == 0 || uvm_api_range_invalid(base, length))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    last_address = base + length - 1;
    va_range = uvm_va_range_find(va_space, base);
    managed_range = uvm_va_range_to_managed_or_null(va_range);
    if (!managed_range || managed_range->va_range.node.end < last_address)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (work->single_managed_range && work->single_managed_range != managed_range)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    first_block_index = uvm_va_range_block_index(managed_range, base);
    last_block_index = uvm_va_range_block_index(managed_range, last_address);
    block_count = last_block_index - first_block_index + 1;
    if (block_count > UVM_PHASE_F_MAX_RANGES)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (!work->single_managed_range) {
        work->single_managed_range = managed_range;
        work->single_first_block_index = first_block_index;
        work->single_last_block_index = last_block_index;
    }
    else {
        work->single_first_block_index = min(work->single_first_block_index, first_block_index);
        work->single_last_block_index = max(work->single_last_block_index, last_block_index);
        if (work->single_last_block_index - work->single_first_block_index + 1 > UVM_PHASE_F_MAX_RANGES)
            return NV_WARN_MORE_PROCESSING_REQUIRED;
    }

    range_state->covered_bytes = length;
    range_state->candidate_pages = length / PAGE_SIZE;
    range_state->committed_pages = 0;
    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_mark_single_va_range_region(uvm_va_space_t *va_space,
                                                                      NvU64 region_id,
                                                                      NvU32 range_index,
                                                                      uvm_phase_f_expert_batch_region_work_t *work)
{
    uvm_phase_f_region_t *region;
    uvm_va_range_managed_t *managed_range = work->single_managed_range;
    NvU64 base;
    NvU64 last_address;
    size_t first_block_index;
    size_t last_block_index;
    size_t block_index;

    UVM_ASSERT(managed_range);

    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    base = region->base;
    last_address = base + region->length - 1;
    first_block_index = uvm_va_range_block_index(managed_range, base);
    last_block_index = uvm_va_range_block_index(managed_range, last_address);

    for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
        size_t seen_index = block_index - work->single_first_block_index;

        if (!test_and_set_bit(seen_index, work->single_block_seen))
            work->single_block_owner[seen_index] = range_index;
    }

    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_emit_single_va_range_entries(uvm_phase_f_expert_batch_region_work_t *work,
                                                                       UVM_PHASE_F_BATCH_REGIONS_PARAMS *params,
                                                                       size_t *entry_count_out)
{
    uvm_va_range_managed_t *managed_range = work->single_managed_range;
    size_t relative_index;
    size_t span;

    *entry_count_out = 0;
    UVM_ASSERT(managed_range);

    span = work->single_last_block_index - work->single_first_block_index + 1;
    for (relative_index = 0; relative_index < span; ++relative_index) {
        size_t block_index;
        NvU32 owner;
        uvm_phase_f_region_t *region;
        uvm_va_block_t *block;
        NvU64 last_address;
        NvU64 block_start;
        NvU64 block_end;
        NV_STATUS status;

        if (!test_bit(relative_index, work->single_block_seen))
            continue;

        if (*entry_count_out >= UVM_PHASE_F_MAX_RANGES)
            return NV_WARN_MORE_PROCESSING_REQUIRED;

        block_index = work->single_first_block_index + relative_index;
        owner = work->single_block_owner[relative_index];
        region = radix_tree_lookup(&managed_range->va_range.va_space->phase_f_regions, params->region_ids[owner]);
        if (!region)
            return NV_ERR_OBJECT_NOT_FOUND;

        status = uvm_va_range_block_create(managed_range, block_index, &block);
        if (status != NV_OK)
            return status;

        last_address = region->base + region->length - 1;
        block_start = max(region->base, block->start);
        block_end = min(last_address, block->end);
        if (block_start > block_end)
            continue;

        work->entries[*entry_count_out].va_block = block;
        work->entries[*entry_count_out].region = uvm_va_block_region_from_start_end(block, block_start, block_end);
        work->entries[*entry_count_out].committed_pages = 0;
        work->entry_range_indices[*entry_count_out] = owner;
        ++*entry_count_out;
    }

    return NV_OK;
}

static void uvm_phase_f_expert_batch_sort_entries(uvm_va_block_expert_batch_entry_t *entries,
                                                  NvU32 *entry_range_indices,
                                                  size_t entry_count)
{
    size_t i;

    for (i = 1; i < entry_count; ++i) {
        uvm_va_block_expert_batch_entry_t entry = entries[i];
        NvU32 range_index = entry_range_indices[i];
        size_t j = i;

        while (j > 0 && entries[j - 1].va_block->start > entry.va_block->start) {
            entries[j] = entries[j - 1];
            entry_range_indices[j] = entry_range_indices[j - 1];
            --j;
        }

        entries[j] = entry;
        entry_range_indices[j] = range_index;
    }
}

static bool uvm_phase_f_expert_batch_has_duplicate_blocks(uvm_va_block_expert_batch_entry_t *entries,
                                                          size_t entry_count)
{
    size_t entry_index;

    for (entry_index = 1; entry_index < entry_count; ++entry_index) {
        if (entries[entry_index - 1].va_block == entries[entry_index].va_block)
            return true;
    }

    return false;
}

static NV_STATUS uvm_phase_f_expert_read_dup_fast_batch(const NvProcessorUuid *gpu_uuid,
                                                        NvBool async,
                                                        UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params,
                                                        struct file *filp)
{
    NV_STATUS status = NV_OK;
    NV_STATUS first_error = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm = NULL;
    uvm_gpu_t *gpu = NULL;
    uvm_phase_f_expert_batch_range_state_t *range_states = NULL;
    uvm_va_block_expert_batch_entry_t *entries = NULL;
    NvU32 *entry_range_indices = NULL;
    uvm_processor_mask_t *involved_processors = NULL;
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    size_t entry_count = 0;
    size_t entry_index;
    NvU32 range_index;
    NvU64 committed_pages_total = 0;
    bool mmap_lock_dropped = false;

    UVM_ASSERT(params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST);
    UVM_ASSERT(params->count > 1);

    range_states = uvm_kvmalloc_zero(sizeof(*range_states) * params->count);
    if (!range_states)
        return NV_ERR_NO_MEMORY;

    for (range_index = 0; range_index < params->count; ++range_index)
        uvm_phase_f_result_init(&params->results[range_index], params->ranges[range_index].length);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    status = uvm_phase_f_expert_batch_count_blocks(va_space, params, range_states, &entry_count);
    if (status != NV_OK)
        goto out;

    entries = uvm_kvmalloc_zero(sizeof(*entries) * entry_count);
    entry_range_indices = uvm_kvmalloc_zero(sizeof(*entry_range_indices) * entry_count);
    involved_processors = uvm_processor_mask_cache_alloc();
    if (!entries || !entry_range_indices || !involved_processors) {
        status = NV_ERR_NO_MEMORY;
        goto out;
    }

    status = uvm_phase_f_expert_batch_fill_entries(va_space,
                                                   params,
                                                   entries,
                                                   entry_range_indices,
                                                   entry_count);
    if (status != NV_OK)
        goto out;

    uvm_phase_f_expert_batch_sort_entries(entries, entry_range_indices, entry_count);
    if (uvm_phase_f_expert_batch_has_duplicate_blocks(entries, entry_count)) {
        status = NV_WARN_MORE_PROCESSING_REQUIRED;
        goto out;
    }

    status = uvm_va_block_expert_promote_read_duplicate_batch(va_space,
                                                              mm,
                                                              gpu,
                                                              entries,
                                                              entry_count,
                                                              &batch_tracker,
                                                              involved_processors);
    if (status != NV_OK)
        goto out;

    if (!async) {
        if (mm) {
            uvm_up_read_mmap_lock_out_of_order(mm);
            mmap_lock_dropped = true;
        }

        status = uvm_tracker_wait(&batch_tracker);
        if (status != NV_OK)
            goto out;
    }

    for (entry_index = 0; entry_index < entry_count; ++entry_index) {
        NvU32 owner = entry_range_indices[entry_index];

        range_states[owner].committed_pages += entries[entry_index].committed_pages;
        committed_pages_total += entries[entry_index].committed_pages;
    }

out:
    if (status != NV_WARN_MORE_PROCESSING_REQUIRED) {
        params->success_count = 0;
        params->failure_count = 0;

        for (range_index = 0; range_index < params->count; ++range_index) {
            UVM_PHASE_F_RANGE_RESULT *result = &params->results[range_index];
            NV_STATUS range_status = status == NV_OK ? NV_OK : status;

            result->status = range_status;
            if (range_status == NV_OK) {
                result->resident_gpu_bytes = range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_cpu_bytes = range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_any_bytes = range_states[range_index].covered_bytes;
                result->resident_unknown_bytes =
                    params->ranges[range_index].length > range_states[range_index].covered_bytes ?
                        params->ranges[range_index].length - range_states[range_index].covered_bytes : 0;
            }

            uvm_phase_f_note_status(range_status,
                                    &params->success_count,
                                    &params->failure_count,
                                    &first_error);
        }
    }

    uvm_phase_f_route_log("phase_f_expert_read_dup_fast_batch count=%u entries=%zu status=0x%x gpu=%s async=%u committed_pages=%llu\n",
                          params->count,
                          entry_count,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          committed_pages_total);

    uvm_tracker_deinit(&batch_tracker);

    if (involved_processors)
        uvm_processor_mask_cache_free(involved_processors);
    uvm_kvfree(entry_range_indices);
    uvm_kvfree(entries);

    uvm_va_space_up_read(va_space);
    if (mmap_lock_dropped)
        uvm_va_space_mm_or_current_release(va_space, mm);
    else
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    uvm_kvfree(range_states);

    return status == NV_WARN_MORE_PROCESSING_REQUIRED ? status : first_error;
}

static NV_STATUS uvm_phase_f_expert_read_dup_fast_batch_regions(const NvProcessorUuid *gpu_uuid,
                                                                NvBool async,
                                                                UVM_PHASE_F_BATCH_REGIONS_PARAMS *params,
                                                                struct file *filp)
{
    NV_STATUS status = NV_OK;
    NV_STATUS first_error = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm = NULL;
    uvm_gpu_t *gpu = NULL;
    uvm_phase_f_expert_batch_region_work_t *work = NULL;
    uvm_processor_mask_t *involved_processors = NULL;
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    size_t entry_count = 0;
    size_t entry_index;
    NvU32 range_index;
    NvU64 committed_pages_total = 0;
    bool mmap_lock_dropped = false;
    bool single_va_range_fast = false;

    UVM_ASSERT(params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST);
    UVM_ASSERT(params->count > 1);

    work = uvm_kvmalloc_zero(sizeof(*work));
    if (!work)
        return NV_ERR_NO_MEMORY;

    for (range_index = 0; range_index < params->count; ++range_index)
        uvm_phase_f_result_init(&params->results[range_index], 0);

    involved_processors = uvm_processor_mask_cache_alloc();
    if (!involved_processors) {
        uvm_kvfree(work);
        return NV_ERR_NO_MEMORY;
    }

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (false) {
        NV_STATUS fast_status = NV_OK;

        for (range_index = 0; range_index < params->count; ++range_index) {
            fast_status = uvm_phase_f_expert_batch_probe_single_va_range_region(va_space,
                                                                                params->region_ids[range_index],
                                                                                range_index,
                                                                                work);
            if (fast_status != NV_OK)
                break;
        }

        if (fast_status == NV_OK) {
            for (range_index = 0; range_index < params->count; ++range_index) {
                fast_status = uvm_phase_f_expert_batch_mark_single_va_range_region(va_space,
                                                                                   params->region_ids[range_index],
                                                                                   range_index,
                                                                                   work);
                if (fast_status != NV_OK)
                    break;
            }
        }

        if (fast_status == NV_OK)
            fast_status = uvm_phase_f_expert_batch_emit_single_va_range_entries(work, params, &entry_count);

        if (fast_status == NV_OK) {
            single_va_range_fast = true;
            for (range_index = 0; range_index < params->count; ++range_index)
                params->results[range_index].queried_bytes = work->range_states[range_index].covered_bytes;
        }
        else {
            memset(work, 0, sizeof(*work));
            entry_count = 0;
        }
    }

    if (!single_va_range_fast) {
        for (range_index = 0; range_index < params->count; ++range_index) {
            status = uvm_phase_f_expert_batch_collect_region(va_space,
                                                             params->region_ids[range_index],
                                                             range_index,
                                                             &work->range_states[range_index],
                                                             work->entries,
                                                             work->entry_range_indices,
                                                             &entry_count);
            if (status != NV_OK)
                goto out;

            params->results[range_index].queried_bytes = work->range_states[range_index].covered_bytes;
        }
    }

    if (entry_count == 0) {
        status = NV_WARN_MORE_PROCESSING_REQUIRED;
        goto out;
    }

    if (!single_va_range_fast) {
        uvm_phase_f_expert_batch_sort_entries(work->entries, work->entry_range_indices, entry_count);
        if (uvm_phase_f_expert_batch_has_duplicate_blocks(work->entries, entry_count)) {
            status = NV_WARN_MORE_PROCESSING_REQUIRED;
            goto out;
        }
    }

    if (single_va_range_fast) {
        status = uvm_va_block_expert_promote_read_duplicate_batch_single_lock(va_space,
                                                                              mm,
                                                                              gpu,
                                                                              work->entries,
                                                                              entry_count,
                                                                              &batch_tracker,
                                                                              involved_processors);
    }
    else if (false) {
        status = uvm_va_block_expert_promote_read_duplicate_batch_trusted(va_space,
                                                                          mm,
                                                                          gpu,
                                                                          work->entries,
                                                                          entry_count,
                                                                          &batch_tracker,
                                                                          involved_processors);
    }
    else {
        status = uvm_va_block_expert_promote_read_duplicate_batch(va_space,
                                                                  mm,
                                                                  gpu,
                                                                  work->entries,
                                                                  entry_count,
                                                                  &batch_tracker,
                                                                  involved_processors);
    }
    if (status != NV_OK)
        goto out;

    if (!async) {
        if (mm) {
            uvm_up_read_mmap_lock_out_of_order(mm);
            mmap_lock_dropped = true;
        }

        status = uvm_tracker_wait(&batch_tracker);
        if (status != NV_OK)
            goto out;
    }

    for (entry_index = 0; entry_index < entry_count; ++entry_index) {
        NvU32 owner = work->entry_range_indices[entry_index];

        work->range_states[owner].committed_pages += work->entries[entry_index].committed_pages;
        committed_pages_total += work->entries[entry_index].committed_pages;
    }

out:
    if (status != NV_WARN_MORE_PROCESSING_REQUIRED) {
        params->success_count = 0;
        params->failure_count = 0;

        for (range_index = 0; range_index < params->count; ++range_index) {
            UVM_PHASE_F_RANGE_RESULT *result = &params->results[range_index];
            NV_STATUS range_status = status == NV_OK ? NV_OK : status;

            result->status = range_status;
            if (range_status == NV_OK) {
                result->resident_gpu_bytes = work->range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_cpu_bytes = work->range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_any_bytes = work->range_states[range_index].covered_bytes;
                result->resident_unknown_bytes = 0;
            }

            uvm_phase_f_note_status(range_status,
                                    &params->success_count,
                                    &params->failure_count,
                                    &first_error);
        }
    }

    uvm_phase_f_route_log("phase_f_expert_read_dup_fast_batch_regions count=%u entries=%zu status=0x%x gpu=%s async=%u single_va_range=%u committed_pages=%llu\n",
                          params->count,
                          entry_count,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          single_va_range_fast,
                          committed_pages_total);

    uvm_tracker_deinit(&batch_tracker);

    uvm_va_space_up_read(va_space);
    if (mmap_lock_dropped)
        uvm_va_space_mm_or_current_release(va_space, mm);
    else
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    uvm_processor_mask_cache_free(involved_processors);
    uvm_kvfree(work);

    return status == NV_WARN_MORE_PROCESSING_REQUIRED ? status : first_error;
}

static NV_STATUS uvm_api_phase_f_batch_executor(UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params, struct file *filp)
{
    NvU32 i;
    NV_STATUS first_error = NV_OK;

    if (params->count > UVM_PHASE_F_MAX_RANGES)
        return NV_ERR_INVALID_ARGUMENT;

    params->success_count = 0;
    params->failure_count = 0;

    UVM_INFO_PRINT("phase_f_policy_batch begin action=%u count=%u async=%u first_base=0x%llx first_len=%llu\n",
                   params->action,
                   params->count,
                   params->async,
                   params->count > 0 ? params->ranges[0].base : 0,
                   params->count > 0 ? params->ranges[0].length : 0);
    uvm_phase_f_route_log("phase_f_policy_batch begin action=%u count=%u async=%u first_base=0x%llx first_len=%llu\n",
                          params->action,
                          params->count,
                          params->async,
                          params->count > 0 ? params->ranges[0].base : 0,
                          params->count > 0 ? params->ranges[0].length : 0);

    if (params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST && params->count > 1) {
        NV_STATUS batch_status = uvm_phase_f_expert_read_dup_fast_batch(&params->gpu_uuid,
                                                                        params->async,
                                                                        params,
                                                                        filp);
        if (batch_status != NV_WARN_MORE_PROCESSING_REQUIRED) {
            first_error = batch_status;
            goto out;
        }

        params->success_count = 0;
        params->failure_count = 0;
        first_error = NV_OK;
        uvm_phase_f_route_log("phase_f_policy_batch expert_fast_batch_fallback count=%u\n", params->count);
    }

    for (i = 0; i < params->count; ++i) {
        NV_STATUS status = uvm_phase_f_execute_range(params->action,
                                                     &params->gpu_uuid,
                                                     params->async,
                                                     params->ranges[i].base,
                                                     params->ranges[i].length,
                                                     &params->results[i],
                                                     filp);
        uvm_phase_f_note_status(status,
                                &params->success_count,
                                &params->failure_count,
                                &first_error);
    }

out:
    UVM_INFO_PRINT("phase_f_policy_batch end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                   params->action,
                   params->count,
                   params->success_count,
                   params->failure_count,
                   first_error);
    uvm_phase_f_route_log("phase_f_policy_batch end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                          params->action,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          first_error);

    return first_error;
}

static NV_STATUS uvm_api_phase_f_register_region(UVM_PHASE_F_REGISTER_REGION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;
    NV_STATUS status;
    int ret;

    if (params->region.region_id == 0 ||
        params->region.base == 0 ||
        params->region.length == 0 ||
        params->region.base + params->region.length - 1 < params->region.base)
        return NV_ERR_INVALID_ARGUMENT;

    region = uvm_kvmalloc_zero(sizeof(*region));
    if (!region)
        return NV_ERR_NO_MEMORY;

    region->region_id = params->region.region_id;
    region->base = params->region.base;
    region->length = params->region.length;
    region->state = UVM_REGION_STATE_UNKNOWN;
    region->va_node.start = region->base;
    region->va_node.end = region->base + region->length - 1;

    uvm_va_space_down_write(va_space);
    status = uvm_range_tree_add(&va_space->phase_f_region_ranges, &region->va_node);
    if (status == NV_OK) {
        ret = radix_tree_insert(&va_space->phase_f_regions, region->region_id, region);
        status = errno_to_nv_status(ret);
        if (status != NV_OK)
            uvm_range_tree_remove(&va_space->phase_f_region_ranges, &region->va_node);
    }
    uvm_va_space_up_write(va_space);

    if (status != NV_OK) {
        uvm_kvfree(region);
        return status;
    }

    uvm_phase_f_route_log("phase_f_register_region id=%llu base=0x%llx length=%llu state=%u\n",
                          region->region_id,
                          region->base,
                          region->length,
                          region->state);

    return NV_OK;
}

static NV_STATUS uvm_api_phase_f_unregister_region(UVM_PHASE_F_UNREGISTER_REGION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;

    if (params->region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_space_down_write(va_space);
    region = radix_tree_delete(&va_space->phase_f_regions, params->region_id);
    if (region) {
        uvm_range_tree_remove(&va_space->phase_f_region_ranges, &region->va_node);
    }
    uvm_va_space_up_write(va_space);

    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    uvm_phase_f_route_log("phase_f_unregister_region id=%llu base=0x%llx length=%llu\n",
                          region->region_id,
                          region->base,
                          region->length);
    uvm_kvfree(region);

    return NV_OK;
}

// Batch region execution is the generic restore/release path. Keep the
// residency state machine tied to the operation result, including an explicit
// in-flight state for asynchronous submissions.
static void uvm_phase_f_update_region_state_after_action(uvm_va_space_t *va_space,
                                                          NvU64 region_id,
                                                          NvU32 action,
                                                          NvBool async,
                                                          NV_STATUS status)
{
    NvU32 new_state;
    uvm_phase_f_region_t *region;

    if (action == UVM_PHASE_F_ACTION_PREFETCH_GPU) {
        new_state = status == NV_OK ?
                        (async ? UVM_REGION_STATE_PREFETCHING_GPU :
                                 UVM_REGION_STATE_GPU) :
                        UVM_REGION_STATE_UNKNOWN;
    }
    else if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU ||
             action == UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP) {
        new_state = status == NV_OK ?
                        (async ? UVM_REGION_STATE_PREFETCHING_CPU :
                                 UVM_REGION_STATE_CPU) :
                        UVM_REGION_STATE_UNKNOWN;
    }
    else {
        return;
    }

    uvm_va_space_down_write(va_space);
    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (region)
        region->state = new_state;
    uvm_va_space_up_write(va_space);
}

static bool uvm_phase_f_region_action_is_idempotent(NvU32 action,
                                                     NvU32 state)
{
    if (action == UVM_PHASE_F_ACTION_PREFETCH_GPU)
        return state == UVM_REGION_STATE_GPU;

    if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU)
        return state == UVM_REGION_STATE_CPU;

    return false;
}

static void uvm_phase_f_result_set_idempotent(UVM_PHASE_F_RANGE_RESULT *result,
                                               NvU32 action,
                                               NvU64 length)
{
    uvm_phase_f_result_init(result, length);
    result->resident_on_count = 1;
    result->resident_any_bytes = length;

    if (action == UVM_PHASE_F_ACTION_PREFETCH_GPU) {
        result->resident_on_gpu = NV_TRUE;
        result->resident_gpu_bytes = length;
    }
    else {
        result->resident_on_cpu = NV_TRUE;
        result->resident_cpu_bytes = length;
    }
}

static NV_STATUS uvm_api_phase_f_batch_regions(UVM_PHASE_F_BATCH_REGIONS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    NvU32 i;
    NV_STATUS first_error = NV_OK;

    if (params->count > UVM_PHASE_F_MAX_RANGES)
        return NV_ERR_INVALID_ARGUMENT;

    params->success_count = 0;
    params->failure_count = 0;

    uvm_phase_f_route_log("phase_f_policy_batch_regions begin action=%u count=%u async=%u first_region=%llu\n",
                          params->action,
                          params->count,
                          params->async,
                          params->count > 0 ? params->region_ids[0] : 0);

    if (params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST && params->count > 1) {
        NV_STATUS batch_status = uvm_phase_f_expert_read_dup_fast_batch_regions(&params->gpu_uuid,
                                                                                params->async,
                                                                                params,
                                                                                filp);
        if (batch_status != NV_WARN_MORE_PROCESSING_REQUIRED) {
            first_error = batch_status;
            goto out;
        }

        params->success_count = 0;
        params->failure_count = 0;
        first_error = NV_OK;
        uvm_phase_f_route_log("phase_f_policy_batch_regions expert_fast_region_batch_fallback count=%u\n",
                              params->count);
    }

    for (i = 0; i < params->count; ++i) {
        uvm_phase_f_region_t *region;
        NvU64 base = 0;
        NvU64 length = 0;
        NvU32 state = UVM_REGION_STATE_UNKNOWN;
        bool idempotent = false;
        NV_STATUS status = NV_OK;

        params->results[i].status = NV_OK;
        params->results[i].resident_on_count = 0;
        params->results[i].resident_on_cpu = NV_FALSE;
        params->results[i].resident_on_gpu = NV_FALSE;
        params->results[i].queried_bytes = 0;
        params->results[i].resident_gpu_bytes = 0;
        params->results[i].resident_cpu_bytes = 0;
        params->results[i].resident_any_bytes = 0;
        params->results[i].resident_unknown_bytes = 0;

        if (params->region_ids[i] == 0) {
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else {
            uvm_va_space_down_read(va_space);
            region = radix_tree_lookup(&va_space->phase_f_regions, params->region_ids[i]);
            if (region) {
                base = region->base;
                length = region->length;
                state = region->state;
            }
            uvm_va_space_up_read(va_space);

            if (!region)
                status = NV_ERR_OBJECT_NOT_FOUND;
        }

        if (status == NV_OK) {
            idempotent = uvm_phase_f_region_action_is_idempotent(params->action, state);
            if (idempotent) {
                uvm_phase_f_result_set_idempotent(&params->results[i],
                                                   params->action,
                                                   length);
                goto note_status;
            }

            uvm_phase_f_route_log("phase_f_policy_batch_region_entry index=%u action=%u region=%llu base=0x%llx end=0x%llx length=%llu async=%u\n",
                                  i,
                                  params->action,
                                  params->region_ids[i],
                                  base,
                                  base + length,
                                  length,
                                  params->async);
            status = uvm_phase_f_execute_range(params->action,
                                               &params->gpu_uuid,
                                               params->async,
                                               base,
                                               length,
                                               &params->results[i],
                                               filp);
            uvm_phase_f_update_region_state_after_action(va_space,
                                                         params->region_ids[i],
                                                         params->action,
                                                         params->async,
                                                         status);
        }
        else {
            params->results[i].status = status;
        }

note_status:
        uvm_phase_f_note_status(status,
                                &params->success_count,
                                &params->failure_count,
                                &first_error);
    }

out:
    uvm_phase_f_route_log("phase_f_policy_batch_regions end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                          params->action,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          first_error);

    return first_error;
}

static bool uvm_moe_slot_remap_mode_valid(NvU32 mode)
{
    return mode == UVM_MOE_SLOT_REMAP_MODE_DRY_RUN ||
           mode == UVM_MOE_SLOT_REMAP_MODE_VALIDATE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY ||
           mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT;
}

static bool uvm_moe_slot_remap_entry_range_invalid(const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    if (entry->va_base == 0 || entry->bytes == 0)
        return true;

    if (entry->va_base + entry->bytes - 1 < entry->va_base)
        return true;

    if (entry->slot_offset + entry->bytes - 1 < entry->slot_offset)
        return true;

    if (!IS_ALIGNED(entry->va_base, PAGE_SIZE) ||
        !IS_ALIGNED(entry->bytes, PAGE_SIZE) ||
        !IS_ALIGNED(entry->slot_offset, PAGE_SIZE))
        return true;

    return false;
}

static bool uvm_moe_slot_remap_entry_excluded_from_conflict_check(const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    return entry->status == UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT ||
           entry->status == UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ADDRESS ||
           entry->status == UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
}

static bool uvm_moe_slot_remap_entry_target_ranges_overlap(const UVM_MOE_SLOT_REMAP_ENTRY *a,
                                                           const UVM_MOE_SLOT_REMAP_ENTRY *b)
{
    NvU64 a_end = a->slot_offset + a->bytes;
    NvU64 b_end = b->slot_offset + b->bytes;

    return a->slot_offset < b_end && b->slot_offset < a_end;
}

static bool uvm_moe_slot_remap_has_slot_conflict(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params, NvU32 index)
{
    UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[index];
    NvU32 i;

    for (i = 0; i < index; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *prev = &params->entries[i];

        if (uvm_moe_slot_remap_entry_excluded_from_conflict_check(prev))
            continue;

        if (prev->layer_id != entry->layer_id || prev->slot_id != entry->slot_id)
            continue;

        if (prev->expert_id != entry->expert_id)
            return true;

        if (uvm_moe_slot_remap_entry_target_ranges_overlap(prev, entry))
            return true;
    }

    return false;
}

static NV_STATUS uvm_moe_slot_pool_uninstall_va_block_slot_mapping(uvm_va_space_t *va_space,
                                                                   uvm_gpu_t *gpu,
                                                                   struct mm_struct *mm,
                                                                   NvU64 va_base,
                                                                   NvU64 bytes,
                                                                   NvU64 *block_unmap_count_out,
                                                                   NvU64 *tracker_wait_count_out);

static void uvm_moe_slot_pool_note_pte_failure_locked(uvm_moe_slot_pool_slot_t *slot)
{
    if (slot)
        ++slot->pte_failure_count;

    ++g_uvm_moe_slot_pool_pte_failure_count;
}

static void uvm_moe_slot_pool_note_pte_clear_reason_locked(uvm_moe_slot_pool_slot_t *slot,
                                                           uvm_moe_slot_pte_clear_reason_t reason)
{
    if (!slot)
        return;

    switch (reason) {
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_FAULT:
            ++slot->pte_fault_clear_count;
            ++g_uvm_moe_slot_pool_pte_fault_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE:
            ++slot->pte_migrate_clear_count;
            ++g_uvm_moe_slot_pool_pte_migrate_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP:
            ++slot->pte_unmap_clear_count;
            ++g_uvm_moe_slot_pool_pte_unmap_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_SPLIT:
            ++slot->pte_split_clear_count;
            ++g_uvm_moe_slot_pool_pte_split_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE:
            ++slot->pte_release_clear_count;
            ++g_uvm_moe_slot_pool_pte_release_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_EVICT:
            ++slot->pte_evict_clear_count;
            ++g_uvm_moe_slot_pool_pte_evict_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT:
            ++slot->pte_dangerous_op_reject_count;
            ++g_uvm_moe_slot_pool_pte_dangerous_op_reject_count;
            break;
    }
}

static bool uvm_moe_slot_pool_pte_mapping_overlaps(const uvm_moe_slot_pool_slot_t *slot,
                                                   uvm_va_space_t *va_space,
                                                   NvU64 va_base,
                                                   NvU64 bytes)
{
    NvU64 va_end = va_base + bytes;
    NvU32 i;

    if (!slot->valid || slot->pte_va_space != va_space || slot->pte_bytes == 0)
        return false;

    if (va_end <= va_base)
        return false;

    if (!slot->pte_range_vec &&
        !((slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
           slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP) &&
          slot->pte_gpu))
        return false;

    if (slot->pte_range_count == 0) {
        NvU64 slot_end = slot->pte_va_base + slot->pte_bytes;

        return slot_end > slot->pte_va_base &&
               slot->pte_va_base < va_end &&
               va_base < slot_end;
    }

    for (i = 0; i < slot->pte_range_count; ++i) {
        const uvm_moe_slot_pool_pte_range_t *range = &slot->pte_ranges[i];
        NvU64 range_end = range->va_base + range->bytes;

        if (range->bytes == 0 || range_end <= range->va_base)
            continue;

        if (range->va_base < va_end && va_base < range_end)
            return true;
    }

    return false;
}

static NV_STATUS uvm_moe_slot_pool_release_pte_mapping_locked(uvm_moe_slot_pool_slot_t *slot,
                                                              uvm_va_space_t *locked_va_space,
                                                              struct mm_struct *mm)
{
    NV_STATUS status = NV_OK;
    NvU64 old_pte_va_base = slot->pte_va_base;
    NvU64 old_pte_bytes = slot->pte_bytes;
    NvU64 old_pte_end = old_pte_va_base + old_pte_bytes;
    NvU32 old_pte_range_count = slot->pte_range_count;
    NvU32 old_pte_mode = slot->pte_mode;

    if (old_pte_bytes != 0 || slot->pte_range_vec || old_pte_range_count != 0) {
        uvm_phase_f_route_log("moe_pte_range action=release_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu ranges=%u mode=%u va_space=%px locked_va_space=%px mm=%px gpu=%s\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              old_pte_va_base,
                              old_pte_end,
                              old_pte_bytes,
                              old_pte_range_count,
                              old_pte_mode,
                              slot->pte_va_space,
                              locked_va_space,
                              mm,
                              slot->pte_gpu ? uvm_gpu_name(slot->pte_gpu) : "unknown");
    }

    if (slot->pte_range_vec) {
        uvm_membar_t tlb_membar = UVM_MEMBAR_SYS;

        if (slot->pte_gpu && slot->slot_mem)
            tlb_membar = uvm_hal_downgrade_membar_type(slot->pte_gpu,
                                                       uvm_mem_is_local_vidmem(slot->slot_mem, slot->pte_gpu));

        uvm_phase_f_route_log("moe_pte_range action=clear_range_vec_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu ranges=%u mode=%u membar=%u\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              slot->pte_va_base,
                              slot->pte_va_base + slot->pte_bytes,
                              slot->pte_bytes,
                              slot->pte_range_count,
                              slot->pte_mode,
                              tlb_membar);
        status = uvm_page_table_range_vec_clear_ptes(slot->pte_range_vec, tlb_membar);
        uvm_phase_f_route_log("moe_pte_range action=clear_range_vec_end layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              slot->pte_va_base,
                              slot->pte_va_base + slot->pte_bytes,
                              slot->pte_bytes,
                              status);
        if (status != NV_OK)
            UVM_ERR_PRINT("MoE slot PTE clear failed: %s, GPU %s, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          slot->pte_gpu ? uvm_gpu_name(slot->pte_gpu) : "unknown",
                          slot->pte_va_base,
                          slot->pte_va_base + slot->pte_bytes);

        uvm_page_table_range_vec_destroy(slot->pte_range_vec);
        slot->pte_range_vec = NULL;
        ++slot->pte_clear_count;
        ++g_uvm_moe_slot_pool_pte_clear_count;
    }

    if ((slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
         slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP) &&
        slot->pte_bytes != 0) {
        NvU64 block_unmap_count = 0;
        NvU64 tracker_wait_count = 0;
        NvU32 i;

        if (!locked_va_space || locked_va_space != slot->pte_va_space || !slot->pte_gpu)
            return NV_ERR_INVALID_STATE;

        if (slot->pte_range_count == 0) {
            slot->pte_ranges[0].va_base = slot->pte_va_base;
            slot->pte_ranges[0].bytes = slot->pte_bytes;
            slot->pte_range_count = 1;
        }

        for (i = 0; i < slot->pte_range_count; ++i) {
            uvm_moe_slot_pool_pte_range_t *range = &slot->pte_ranges[i];
            NvU64 range_block_unmap_count = 0;
            NvU64 range_tracker_wait_count = 0;

            if (range->bytes == 0)
                continue;

            uvm_phase_f_route_log("moe_pte_range action=clear_va_block_begin layer=%u expert=%u slot=%u role=%u range_index=%u va=0x%llx end=0x%llx bytes=%llu full_va=0x%llx full_end=0x%llx full_bytes=%llu mode=%u\n",
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  slot->last_role,
                                  i,
                                  range->va_base,
                                  range->va_base + range->bytes,
                                  range->bytes,
                                  slot->pte_va_base,
                                  slot->pte_va_base + slot->pte_bytes,
                                  slot->pte_bytes,
                                  slot->pte_mode);
            status = uvm_moe_slot_pool_uninstall_va_block_slot_mapping(locked_va_space,
                                                                       slot->pte_gpu,
                                                                       mm,
                                                                       range->va_base,
                                                                       range->bytes,
                                                                       &range_block_unmap_count,
                                                                       &range_tracker_wait_count);
            uvm_phase_f_route_log("moe_pte_range action=clear_va_block_end layer=%u expert=%u slot=%u role=%u range_index=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x block_unmaps=%llu tracker_waits=%llu\n",
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  slot->last_role,
                                  i,
                                  range->va_base,
                                  range->va_base + range->bytes,
                                  range->bytes,
                                  status,
                                  range_block_unmap_count,
                                  range_tracker_wait_count);
            if (status != NV_OK) {
                UVM_ERR_PRINT("MoE slot VA-block PTE clear failed: %s, GPU %s, VA [0x%llx, 0x%llx)\n",
                              nvstatusToString(status),
                              slot->pte_gpu ? uvm_gpu_name(slot->pte_gpu) : "unknown",
                              range->va_base,
                              range->va_base + range->bytes);
                return status;
            }

            block_unmap_count += range_block_unmap_count;
            tracker_wait_count += range_tracker_wait_count;
        }

        slot->pte_block_unmap_count += block_unmap_count;
        slot->pte_block_tracker_wait_count += tracker_wait_count;
        g_uvm_moe_slot_pool_pte_block_unmap_count += block_unmap_count;
        g_uvm_moe_slot_pool_pte_block_tracker_wait_count += tracker_wait_count;
        ++slot->pte_clear_count;
        ++g_uvm_moe_slot_pool_pte_clear_count;
    }

    if (old_pte_bytes != 0 || old_pte_range_count != 0) {
        uvm_phase_f_route_log("moe_pte_range action=release_end layer=%u expert=%u slot=%u role=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu ranges=%u mode=%u status=0x%x\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              old_pte_va_base,
                              old_pte_end,
                              old_pte_bytes,
                              old_pte_range_count,
                              old_pte_mode,
                              status);
    }

    if (slot->pte_gpu) {
        uvm_gpu_release(slot->pte_gpu);
        slot->pte_gpu = NULL;
    }

    slot->pte_va_base = 0;
    slot->pte_bytes = 0;
    slot->pte_range_count = 0;
    memset(slot->pte_ranges, 0, sizeof(slot->pte_ranges));
    slot->pte_va_space = NULL;
    slot->pte_slot_offset = 0;
    slot->pte_mode = 0;
    slot->pte_va_block_tracked = NV_FALSE;

    return status;
}

static void uvm_moe_slot_pool_forget_release_pte_mapping_locked(uvm_moe_slot_pool_slot_t *slot)
{
    if (slot->pte_gpu) {
        uvm_gpu_release(slot->pte_gpu);
        slot->pte_gpu = NULL;
    }

    slot->pte_range_vec = NULL;
    slot->pte_va_base = 0;
    slot->pte_bytes = 0;
    slot->pte_range_count = 0;
    memset(slot->pte_ranges, 0, sizeof(slot->pte_ranges));
    slot->pte_va_space = NULL;
    slot->pte_slot_offset = 0;
    slot->pte_mode = 0;
    slot->pte_va_block_tracked = NV_FALSE;
}

static NV_STATUS uvm_moe_slot_pool_rollback_pte_mapping_locked(uvm_moe_slot_pool_slot_t *slot,
                                                               uvm_va_space_t *locked_va_space,
                                                               struct mm_struct *mm)
{
    NvU64 rollback_bytes = slot->pte_bytes;
    NV_STATUS status;

    if (slot->async_copy_pending)
        (void)uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_TRUE, NULL);

    status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, locked_va_space, mm);
    if (status != NV_OK)
        return status;

    ++slot->pte_rollback_count;
    slot->pte_rollback_bytes += rollback_bytes;
    ++g_uvm_moe_slot_pool_pte_rollback_count;
    g_uvm_moe_slot_pool_pte_rollback_bytes += rollback_bytes;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_ROLLED_BACK;

    return NV_OK;
}

static void uvm_moe_slot_pool_note_partial_rollback_locked(uvm_moe_slot_pool_slot_t *slot,
                                                           NvU64 rollback_bytes,
                                                           NvU64 block_unmap_count,
                                                           NvU64 tracker_wait_count)
{
    if (rollback_bytes == 0)
        return;

    ++slot->pte_rollback_count;
    slot->pte_rollback_bytes += rollback_bytes;
    ++slot->pte_clear_count;
    slot->pte_block_unmap_count += block_unmap_count;
    slot->pte_block_tracker_wait_count += tracker_wait_count;
    ++g_uvm_moe_slot_pool_pte_rollback_count;
    g_uvm_moe_slot_pool_pte_rollback_bytes += rollback_bytes;
    ++g_uvm_moe_slot_pool_pte_clear_count;
    g_uvm_moe_slot_pool_pte_block_unmap_count += block_unmap_count;
    g_uvm_moe_slot_pool_pte_block_tracker_wait_count += tracker_wait_count;
}

static NV_STATUS uvm_moe_slot_pool_release_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                          uvm_va_space_t *locked_va_space,
                                                          struct mm_struct *mm)
{
    NV_STATUS status;

    if (slot->async_copy_pending) {
        status = uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_TRUE, NULL);
        if (status != NV_OK)
            return status;
    }

    status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, locked_va_space, mm);
    if (status != NV_OK)
        return status;

    if (slot->slot_mem)
        uvm_mem_free(slot->slot_mem);

    slot->slot_mem = NULL;
    slot->slot_bytes = 0;
    slot->backing_kind = UVM_MOE_SLOT_POOL_BACKING_NONE;
    slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;

    if (slot->slot_gpu) {
        uvm_gpu_release(slot->slot_gpu);
        slot->slot_gpu = NULL;
    }

    return NV_OK;
}

static NvBool uvm_moe_slot_pool_slot_has_vidmem_backing(const uvm_moe_slot_pool_slot_t *slot,
                                                        const uvm_gpu_t *gpu)
{
    return slot->slot_mem &&
           slot->slot_gpu == gpu &&
           slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_VIDMEM;
}

static NvBool uvm_moe_slot_pool_slot_has_active_pte(const uvm_moe_slot_pool_slot_t *slot)
{
    return slot->pte_range_vec || slot->pte_bytes != 0;
}

static NV_STATUS uvm_moe_slot_pool_finish_async_copy_locked(uvm_moe_slot_pool_slot_t *slot,
                                                            NvBool wait_for_completion,
                                                            NvBool *completed_out)
{
    NV_STATUS status;

    if (completed_out)
        *completed_out = NV_TRUE;

    if (!slot->async_copy_pending)
        return NV_OK;

    status = wait_for_completion ?
                 uvm_tracker_wait(&slot->async_copy_tracker) :
                 uvm_tracker_query(&slot->async_copy_tracker);
    if (!wait_for_completion && status == NV_WARN_MORE_PROCESSING_REQUIRED) {
        if (completed_out)
            *completed_out = NV_FALSE;
        return NV_OK;
    }

    uvm_tracker_deinit(&slot->async_copy_tracker);

    if (slot->async_copy_staging) {
        uvm_mem_free(slot->async_copy_staging);
        slot->async_copy_staging = NULL;
    }

    slot->async_copy_pending = NV_FALSE;

    if (status == NV_OK) {
        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        slot->copy_count++;
        slot->copy_bytes += slot->async_copy_bytes;
        g_uvm_moe_slot_pool_copy_count++;
        g_uvm_moe_slot_pool_copy_bytes += slot->async_copy_bytes;
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
    }
    else {
        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;
        slot->copy_failed_count++;
        g_uvm_moe_slot_pool_copy_failed_count++;
        slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
    }

    slot->async_copy_bytes = 0;
    slot->async_copy_slot_offset = 0;

    return status;
}

static NvU64 uvm_moe_slot_pool_vidmem_bytes_locked(const uvm_gpu_t *gpu,
                                                   const uvm_moe_slot_pool_slot_t *skip)
{
    NvU64 bytes = 0;
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        const uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot == skip)
            continue;

        if (uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu))
            bytes += slot->slot_bytes;
    }

    return bytes;
}

static NvBool uvm_moe_slot_pool_slot_can_evict_locked(const uvm_moe_slot_pool_slot_t *slot,
                                                      uvm_va_space_t *locked_va_space,
                                                      NvBool allow_active)
{
    if (!uvm_moe_slot_pool_slot_has_active_pte(slot))
        return NV_TRUE;

    if (!allow_active || !locked_va_space || slot->pte_va_space != locked_va_space)
        return NV_FALSE;

    return NV_TRUE;
}

static uvm_moe_slot_pool_slot_t *uvm_moe_slot_pool_oldest_vidmem_slot_locked(const uvm_gpu_t *gpu,
                                                                             const uvm_moe_slot_pool_slot_t *skip,
                                                                             uvm_va_space_t *locked_va_space,
                                                                             NvBool allow_active,
                                                                             NvBool *blocked_active_out)
{
    uvm_moe_slot_pool_slot_t *victim = NULL;
    NvU32 i;

    if (blocked_active_out)
        *blocked_active_out = NV_FALSE;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot == skip)
            continue;

        if (!uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu))
            continue;

        if (!uvm_moe_slot_pool_slot_can_evict_locked(slot, locked_va_space, allow_active)) {
            if (blocked_active_out && uvm_moe_slot_pool_slot_has_active_pte(slot))
                *blocked_active_out = NV_TRUE;
            continue;
        }

        if (!victim ||
            slot->epoch < victim->epoch ||
            (slot->epoch == victim->epoch && slot->batch_id < victim->batch_id))
            victim = slot;
    }

    return victim;
}

static void uvm_moe_slot_pool_note_blocked_active_eviction_locked(const uvm_gpu_t *gpu,
                                                                  const uvm_moe_slot_pool_slot_t *skip,
                                                                  uvm_va_space_t *locked_va_space)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot == skip)
            continue;

        if (!uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu))
            continue;

        if (!uvm_moe_slot_pool_slot_has_active_pte(slot))
            continue;

        if (locked_va_space && slot->pte_va_space == locked_va_space)
            continue;

        uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT);
    }
}

static NV_STATUS uvm_moe_slot_pool_evict_slot_locked(uvm_moe_slot_pool_slot_t *slot,
                                                     uvm_va_space_t *locked_va_space,
                                                     struct mm_struct *mm)
{
    NvBool had_active_pte = uvm_moe_slot_pool_slot_has_active_pte(slot);
    NV_STATUS status = uvm_moe_slot_pool_release_backing_locked(slot, locked_va_space, mm);

    if (status != NV_OK)
        return status;

    if (had_active_pte)
        uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_EVICT);

    memset(slot, 0, sizeof(*slot));
    ++g_uvm_moe_slot_pool_vidmem_evict_count;
    return NV_OK;
}

static NvU64 uvm_moe_slot_pool_enforce_vidmem_budget_locked(uvm_gpu_t *gpu,
                                                            const uvm_moe_slot_pool_slot_t *skip,
                                                            NvU64 allocation_bytes,
                                                            NvBool force_one,
                                                            uvm_va_space_t *locked_va_space,
                                                            struct mm_struct *mm)
{
    NvU64 budget = (NvU64)uvm_moe_slot_pool_vidmem_budget_bytes;
    NvU64 bytes;
    NvU64 evicted = 0;

    if (!gpu)
        return 0;

    bytes = uvm_moe_slot_pool_vidmem_bytes_locked(gpu, skip);

    while ((budget != 0 && bytes + allocation_bytes > budget) || (force_one && evicted == 0)) {
        NvBool blocked_active = NV_FALSE;
        uvm_moe_slot_pool_slot_t *victim = uvm_moe_slot_pool_oldest_vidmem_slot_locked(gpu,
                                                                                      skip,
                                                                                      locked_va_space,
                                                                                      NV_FALSE,
                                                                                      &blocked_active);
        NvU64 victim_bytes;

        if (!victim)
            victim = uvm_moe_slot_pool_oldest_vidmem_slot_locked(gpu,
                                                                 skip,
                                                                 locked_va_space,
                                                                 NV_TRUE,
                                                                 &blocked_active);

        if (!victim) {
            if (blocked_active)
                uvm_moe_slot_pool_note_blocked_active_eviction_locked(gpu, skip, locked_va_space);
            break;
        }

        victim_bytes = victim->slot_bytes;
        uvm_phase_f_route_log("moe_slot_pool_vidmem_evict layer=%u slot=%u expert=%u epoch=%llu bytes=%llu budget=%llu current=%llu pending=%llu force=%u active_pte=%u\n",
                              victim->layer_id,
                              victim->slot_id,
                              victim->owner_expert_id,
                              victim->epoch,
                              victim_bytes,
                              budget,
                              bytes,
                              allocation_bytes,
                              force_one,
                              uvm_moe_slot_pool_slot_has_active_pte(victim));

        if (uvm_moe_slot_pool_evict_slot_locked(victim, locked_va_space, mm) != NV_OK)
            break;

        bytes = (bytes > victim_bytes) ? bytes - victim_bytes : 0;
        ++evicted;
    }

    return evicted;
}

static NV_STATUS uvm_moe_slot_pool_clear_metadata_keep_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                                      uvm_va_space_t *locked_va_space,
                                                                      struct mm_struct *mm)
{
    uvm_mem_t *slot_mem = slot->slot_mem;
    uvm_gpu_t *slot_gpu = slot->slot_gpu;
    NvU64 slot_bytes = slot->slot_bytes;
    NvU32 backing_kind = slot->backing_kind;
    NvBool had_async_copy = slot->async_copy_pending;
    NV_STATUS status;

    if (slot->async_copy_pending) {
        status = uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_TRUE, NULL);
        if (status != NV_OK)
            return status;
    }

    status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, locked_va_space, mm);
    if (status != NV_OK)
        return status;

    memset(slot, 0, sizeof(*slot));
    slot->slot_mem = slot_mem;
    slot->slot_gpu = slot_gpu;
    slot->slot_bytes = slot_bytes;
    slot->backing_kind = backing_kind;
    slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    if (had_async_copy)
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;

    if (slot_mem && slot_bytes && backing_kind == UVM_MOE_SLOT_POOL_BACKING_SYSMEM)
        memset(uvm_mem_get_cpu_addr_kernel(slot_mem), 0, slot_bytes);

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_reset_locked(uvm_va_space_t *locked_va_space, struct mm_struct *mm)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        NV_STATUS status = uvm_moe_slot_pool_release_backing_locked(&g_uvm_moe_slot_pool[i],
                                                                    locked_va_space,
                                                                    mm);
        if (status != NV_OK)
            return status;
    }

    memset(g_uvm_moe_slot_pool, 0, sizeof(g_uvm_moe_slot_pool));
    g_uvm_moe_slot_pool_current_epoch = 0;
    g_uvm_moe_slot_pool_current_batch_id = 0;
    g_uvm_moe_slot_pool_validated_entries = 0;
    g_uvm_moe_slot_pool_overwrite_count = 0;
    g_uvm_moe_slot_pool_range_update_count = 0;
    g_uvm_moe_slot_pool_full_count = 0;
    g_uvm_moe_slot_pool_copy_count = 0;
    g_uvm_moe_slot_pool_copy_bytes = 0;
    g_uvm_moe_slot_pool_copy_failed_count = 0;
    g_uvm_moe_slot_pool_vidmem_evict_count = 0;
    g_uvm_moe_slot_pool_pte_remap_count = 0;
    g_uvm_moe_slot_pool_pte_remap_bytes = 0;
    g_uvm_moe_slot_pool_pte_clear_count = 0;
    g_uvm_moe_slot_pool_pte_block_unmap_count = 0;
    g_uvm_moe_slot_pool_pte_block_tracker_wait_count = 0;
    g_uvm_moe_slot_pool_pte_rollback_count = 0;
    g_uvm_moe_slot_pool_pte_rollback_bytes = 0;
    g_uvm_moe_slot_pool_pte_failure_count = 0;
    g_uvm_moe_slot_pool_pte_fault_clear_count = 0;
    g_uvm_moe_slot_pool_pte_migrate_clear_count = 0;
    g_uvm_moe_slot_pool_pte_unmap_clear_count = 0;
    g_uvm_moe_slot_pool_pte_split_clear_count = 0;
    g_uvm_moe_slot_pool_pte_release_clear_count = 0;
    g_uvm_moe_slot_pool_pte_evict_clear_count = 0;
    g_uvm_moe_slot_pool_pte_dangerous_op_reject_count = 0;

    return NV_OK;
}

static void uvm_moe_slot_pool_reset(void)
{
    NV_STATUS status;

    mutex_lock(&g_uvm_moe_slot_pool_lock);
    status = uvm_moe_slot_pool_reset_locked(NULL, NULL);
    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    if (status != NV_OK)
        UVM_ERR_PRINT("MoE slot pool reset failed: %s\n", nvstatusToString(status));
}

static void uvm_moe_slot_pool_release_va_space_mappings(uvm_va_space_t *va_space)
{
    struct mm_struct *mm;
    NvU32 i;

    if (!va_space)
        return;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NV_STATUS status;

        if (slot->pte_va_space != va_space)
            continue;

        if (!mm) {
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
            uvm_phase_f_route_log("moe_slot_pte_forget_va_space_release layer=%u expert=%u slot=%u old_va=0x%llx old_bytes=%llu\n",
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  slot->pte_va_base,
                                  slot->pte_bytes);
            uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
            continue;
        }

        status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status == NV_OK)
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            UVM_ERR_PRINT("MoE slot VA-space PTE cleanup failed: %s, VA space %px, layer %u, slot %u\n",
                          nvstatusToString(status),
                          va_space,
                          slot->layer_id,
                          slot->slot_id);
        }

        if (status != NV_OK)
            uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
}

NV_STATUS uvm_moe_slot_pool_clear_gpu_va_space_mappings_locked(uvm_va_space_t *va_space,
                                                               uvm_gpu_t *gpu,
                                                               struct mm_struct *mm)
{
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    if (!va_space || !gpu)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;
        NV_STATUS status;

        if (slot->pte_va_space != va_space || slot->pte_gpu != gpu || !uvm_moe_slot_pool_slot_has_active_pte(slot))
            continue;

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status == NV_OK) {
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
            uvm_phase_f_route_log("moe_slot_pte_clear_gpu_va_space_release layer=%u expert=%u slot=%u gpu=%s old_va=0x%llx old_bytes=%llu\n",
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  uvm_gpu_name(gpu),
                                  old_pte_va_base,
                                  old_pte_bytes);
        }
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
            UVM_ERR_PRINT("MoE slot GPU VA-space PTE cleanup failed: %s, GPU %s, VA space %px, layer %u, slot %u, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          uvm_gpu_name(gpu),
                          va_space,
                          slot->layer_id,
                          slot->slot_id,
                          old_pte_va_base,
                          old_pte_va_base + old_pte_bytes);
            uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
        }
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return first_error;
}

static NV_STATUS uvm_moe_slot_pool_clear_range_mappings(uvm_va_space_t *va_space,
                                                        NvU64 va_base,
                                                        NvU64 bytes,
                                                        uvm_moe_slot_pte_clear_reason_t reason)
{
    struct mm_struct *mm;
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_scan_begin reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu va_space=%px mm=%px\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          va_space,
                          mm);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;
        NV_STATUS status;

        if (!uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes))
            continue;

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status == NV_OK) {
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, reason);
            uvm_phase_f_route_log("moe_slot_pte_clear_guard reason=%u layer=%u expert=%u slot=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                                  reason,
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  old_pte_va_base,
                                  old_pte_va_base + old_pte_bytes,
                                  old_pte_bytes,
                                  va_base,
                                  va_base + bytes,
                                  bytes);
        }
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
            UVM_ERR_PRINT("MoE slot PTE clear guard failed: %s, VA space %px, layer %u, slot %u, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          va_space,
                          slot->layer_id,
                          slot->slot_id,
                          old_pte_va_base,
                          old_pte_va_base + old_pte_bytes);
        }
    }

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_scan_end reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu status=0x%x\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          first_error);

    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return first_error;
}

static NV_STATUS uvm_moe_slot_pool_clear_range_mappings_locked(uvm_va_space_t *va_space,
                                                               struct mm_struct *mm,
                                                               NvU64 va_base,
                                                               NvU64 bytes,
                                                               uvm_moe_slot_pte_clear_reason_t reason)
{
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_locked_scan_begin reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu va_space=%px mm=%px\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          va_space,
                          mm);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;
        NV_STATUS status;

        if (!uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes))
            continue;

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status == NV_OK) {
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, reason);
            uvm_phase_f_route_log("moe_slot_pte_clear_guard_locked reason=%u layer=%u expert=%u slot=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                                  reason,
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  old_pte_va_base,
                                  old_pte_va_base + old_pte_bytes,
                                  old_pte_bytes,
                                  va_base,
                                  va_base + bytes,
                                  bytes);
        }
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
            UVM_ERR_PRINT("MoE slot PTE locked clear guard failed: %s, VA space %px, layer %u, slot %u, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          va_space,
                          slot->layer_id,
                          slot->slot_id,
                          old_pte_va_base,
                          old_pte_va_base + old_pte_bytes);
        }
    }

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_locked_scan_end reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu status=0x%x\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          first_error);

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return first_error;
}

static NV_STATUS uvm_moe_slot_pool_forget_range_mappings_locked(uvm_va_space_t *va_space,
                                                                NvU64 va_base,
                                                                NvU64 bytes,
                                                                uvm_moe_slot_pte_clear_reason_t reason)
{
    NvU32 i;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    uvm_phase_f_route_log("moe_pte_range action=forget_guard_scan_begin reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu va_space=%px\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          va_space);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;

        if (!uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes))
            continue;

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, reason);
        uvm_phase_f_route_log("moe_slot_pte_forget_guard reason=%u layer=%u expert=%u slot=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                              reason,
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              old_pte_va_base,
                              old_pte_va_base + old_pte_bytes,
                              old_pte_bytes,
                              va_base,
                              va_base + bytes,
                              bytes);
        uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
    }

    uvm_phase_f_route_log("moe_pte_range action=forget_guard_scan_end reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes);

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return NV_OK;
}

bool uvm_moe_slot_pool_range_has_mappings(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    NvU32 i;
    bool found = false;

    if (!va_space || bytes == 0 || base + bytes <= base)
        return false;

    uvm_assert_rwsem_locked(&va_space->lock);

    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (uvm_moe_slot_pool_pte_mapping_overlaps(&g_uvm_moe_slot_pool[i], va_space, base, bytes)) {
            found = true;
            break;
        }
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return found;
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_fault(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    struct mm_struct *mm;
    NV_STATUS status;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (base + bytes <= base)
        return NV_ERR_INVALID_ARGUMENT;

    mm = uvm_va_space_mm_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    status = uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                           mm,
                                                           base,
                                                           bytes,
                                                           UVM_MOE_SLOT_PTE_CLEAR_REASON_FAULT);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_release_unlock(va_space, mm);

    return status;
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_migrate(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings(va_space,
                                                  base,
                                                  bytes,
                                                  UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE);
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_discard(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings(va_space,
                                                  base,
                                                  bytes,
                                                  UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP);
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_unmap_locked(uvm_va_space_t *va_space,
                                                                  struct mm_struct *mm,
                                                                  NvU64 base,
                                                                  NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                         mm,
                                                         base,
                                                         bytes,
                                                         UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP);
}

static NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_release_locked(uvm_va_space_t *va_space,
                                                                           struct mm_struct *mm,
                                                                           NvU64 base,
                                                                           NvU64 bytes)
{
    if (!mm)
        return uvm_moe_slot_pool_forget_range_mappings_locked(va_space,
                                                              base,
                                                              bytes,
                                                              UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);

    return uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                         mm,
                                                         base,
                                                         bytes,
                                                         UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_native_invalidate(uvm_va_space_t *va_space,
                                                                       struct mm_struct *mm,
                                                                       NvU64 base,
                                                                       NvU64 bytes)
{
    NV_STATUS status;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_space_down_write(va_space);
    status = uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                           mm,
                                                           base,
                                                           bytes,
                                                           UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP);
    uvm_va_space_up_write(va_space);

    return status;
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_split_locked(uvm_va_space_t *va_space,
                                                                  struct mm_struct *mm,
                                                                  NvU64 base,
                                                                  NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                         mm,
                                                         base,
                                                         bytes,
                                                         UVM_MOE_SLOT_PTE_CLEAR_REASON_SPLIT);
}

static bool uvm_moe_slot_pool_entry_matches(const uvm_moe_slot_pool_slot_t *slot,
                                            NvU32 layer_id,
                                            NvU32 slot_id)
{
    return slot->valid && slot->layer_id == layer_id && slot->slot_id == slot_id;
}

static uvm_moe_slot_pool_slot_t *uvm_moe_slot_pool_find_locked(NvU32 layer_id, NvU32 slot_id)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (uvm_moe_slot_pool_entry_matches(&g_uvm_moe_slot_pool[i], layer_id, slot_id))
            return &g_uvm_moe_slot_pool[i];
    }

    return NULL;
}

static uvm_moe_slot_pool_slot_t *uvm_moe_slot_pool_find_free_locked(void)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (!g_uvm_moe_slot_pool[i].valid)
            return &g_uvm_moe_slot_pool[i];
    }

    return NULL;
}

static NV_STATUS uvm_moe_slot_pool_gpu_memset(uvm_gpu_t *gpu,
                                              uvm_mem_t *mem,
                                              NvU64 offset,
                                              NvU64 bytes)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_gpu_address_t dst;

    if (bytes == 0)
        return NV_OK;

    if (bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_GPU_INTERNAL,
                            &push,
                            "MoE slot pool zero");
    if (status != NV_OK)
        return status;

    dst = uvm_mem_gpu_address_virtual_kernel(mem, gpu);
    dst.address += offset;
    gpu->parent->ce_hal->memset_1(&push, dst, 0, (size_t)bytes);

    return uvm_push_end_and_wait(&push);
}

static NV_STATUS uvm_moe_slot_pool_gpu_memcopy(uvm_gpu_t *gpu,
                                               uvm_channel_type_t channel_type,
                                               uvm_mem_t *dst_mem,
                                               NvU64 dst_offset,
                                               uvm_mem_t *src_mem,
                                               NvU64 src_offset,
                                               NvU64 bytes)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_gpu_address_t dst;
    uvm_gpu_address_t src;

    if (bytes == 0)
        return NV_OK;

    if (bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    status = uvm_push_begin(gpu->channel_manager,
                            channel_type,
                            &push,
                            "MoE slot pool copy");
    if (status != NV_OK)
        return status;

    dst = uvm_mem_gpu_address_virtual_kernel(dst_mem, gpu);
    dst.address += dst_offset;
    src = uvm_mem_gpu_address_virtual_kernel(src_mem, gpu);
    src.address += src_offset;
    gpu->parent->ce_hal->memcopy(&push, dst, src, (size_t)bytes);

    return uvm_push_end_and_wait(&push);
}

static NV_STATUS uvm_moe_slot_pool_ensure_sysmem_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                                NvU64 required_bytes)
{
    NV_STATUS status;
    uvm_mem_t *new_mem = NULL;
    void *new_addr;
    NvU64 old_bytes;

    required_bytes = UVM_ALIGN_UP(required_bytes, PAGE_SIZE);
    if (required_bytes == 0)
        return NV_ERR_INVALID_ARGUMENT;

    if (slot->slot_mem &&
        slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_SYSMEM &&
        slot->slot_bytes >= required_bytes)
        return NV_OK;

    if (slot->pte_range_vec || slot->pte_bytes != 0)
        return NV_ERR_INVALID_STATE;

    status = uvm_mem_alloc_sysmem_and_map_cpu_kernel(required_bytes, current->mm, &new_mem);
    if (status != NV_OK)
        return status;

    new_addr = uvm_mem_get_cpu_addr_kernel(new_mem);
    if (!new_addr) {
        uvm_mem_free(new_mem);
        return NV_ERR_INVALID_STATE;
    }

    memset(new_addr, 0, required_bytes);

    if (slot->slot_mem && slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_SYSMEM) {
        void *old_addr = uvm_mem_get_cpu_addr_kernel(slot->slot_mem);

        old_bytes = min(slot->slot_bytes, required_bytes);
        if (old_addr && old_bytes)
            memcpy(new_addr, old_addr, old_bytes);
    }

    status = uvm_moe_slot_pool_release_backing_locked(slot, NULL, NULL);
    if (status != NV_OK) {
        uvm_mem_free(new_mem);
        return status;
    }

    slot->slot_mem = new_mem;
    slot->slot_bytes = required_bytes;
    slot->backing_kind = UVM_MOE_SLOT_POOL_BACKING_SYSMEM;
    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_ensure_vidmem_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                                uvm_gpu_t *gpu,
                                                                NvU64 required_bytes,
                                                                uvm_va_space_t *locked_va_space,
                                                                struct mm_struct *mm)
{
    NV_STATUS status;
    uvm_mem_t *new_mem = NULL;
    NvU64 old_bytes = 0;
    NvU64 allocation_bytes;
    NvBool preserve_old = NV_FALSE;

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    required_bytes = UVM_ALIGN_UP(required_bytes, PAGE_SIZE);
    if (required_bytes == 0)
        return NV_ERR_INVALID_ARGUMENT;

    if (slot->slot_mem &&
        slot->slot_gpu == gpu &&
        slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_VIDMEM &&
        slot->slot_bytes >= required_bytes)
        return NV_OK;

    if (slot->pte_range_vec || slot->pte_bytes != 0)
        return NV_ERR_INVALID_STATE;

    preserve_old = slot->slot_mem &&
                   slot->slot_gpu == gpu &&
                   slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_VIDMEM;

    allocation_bytes = required_bytes;
    if (preserve_old)
        allocation_bytes += slot->slot_bytes;

    uvm_moe_slot_pool_enforce_vidmem_budget_locked(gpu,
                                                   slot,
                                                   allocation_bytes,
                                                   NV_FALSE,
                                                   locked_va_space,
                                                   mm);

    status = uvm_mem_alloc_vidmem(required_bytes, gpu, &new_mem);
    if (status == NV_ERR_NO_MEMORY) {
        NvU64 evicted = uvm_moe_slot_pool_enforce_vidmem_budget_locked(gpu,
                                                                       slot,
                                                                       allocation_bytes,
                                                                       NV_TRUE,
                                                                       locked_va_space,
                                                                       mm);

        if (evicted) {
            uvm_phase_f_route_log("moe_slot_pool_vidmem_alloc_retry required=%llu evicted=%llu budget=%llu\n",
                                  required_bytes,
                                  evicted,
                                  (NvU64)uvm_moe_slot_pool_vidmem_budget_bytes);
            status = uvm_mem_alloc_vidmem(required_bytes, gpu, &new_mem);
        }
    }
    if (status != NV_OK)
        return status;

    status = uvm_mem_map_gpu_kernel(new_mem, gpu);
    if (status != NV_OK)
        goto error;

    status = uvm_moe_slot_pool_gpu_memset(gpu, new_mem, 0, required_bytes);
    if (status != NV_OK)
        goto error;

    if (preserve_old) {
        status = uvm_mem_map_gpu_kernel(slot->slot_mem, gpu);
        if (status != NV_OK)
            goto error;

        old_bytes = min(slot->slot_bytes, required_bytes);
        status = uvm_moe_slot_pool_gpu_memcopy(gpu,
                                               UVM_CHANNEL_TYPE_GPU_INTERNAL,
                                               new_mem,
                                               0,
                                               slot->slot_mem,
                                               0,
                                               old_bytes);
        if (status != NV_OK)
            goto error;
    }

    status = uvm_moe_slot_pool_release_backing_locked(slot, NULL, NULL);
    if (status != NV_OK)
        goto error;

    slot->slot_mem = new_mem;
    slot->slot_bytes = required_bytes;
    slot->backing_kind = UVM_MOE_SLOT_POOL_BACKING_VIDMEM;
    uvm_gpu_retain(gpu);
    slot->slot_gpu = gpu;

    return NV_OK;

error:
    uvm_mem_free(new_mem);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_copy_sysmem_entry_locked(uvm_moe_slot_pool_slot_t *slot,
                                                            const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    NV_STATUS status;
    void *dst;
    const void __user *src = (const void __user *)(unsigned long)entry->va_base;
    NvU64 copy_end = entry->slot_offset + entry->bytes;

    status = uvm_moe_slot_pool_ensure_sysmem_backing_locked(slot, copy_end);
    if (status != NV_OK)
        return status;

    dst = (char *)uvm_mem_get_cpu_addr_kernel(slot->slot_mem) + entry->slot_offset;
    if (!access_ok(src, entry->bytes))
        return NV_ERR_INVALID_ADDRESS;

    if (copy_from_user(dst, src, entry->bytes) != 0)
        return NV_ERR_INVALID_ADDRESS;

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_copy_gpu_entry_locked(uvm_moe_slot_pool_slot_t *slot,
                                                         const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                         uvm_gpu_t *gpu,
                                                         uvm_va_space_t *locked_va_space,
                                                         struct mm_struct *mm)
{
    NV_STATUS status;
    const char __user *src = (const char __user *)(unsigned long)entry->va_base;
    NvU64 copy_end = entry->slot_offset + entry->bytes;
    NvU64 copied = 0;
    NvU64 max_chunk_bytes = UVM_MOE_SLOT_POOL_GPU_COPY_STAGING_CHUNK_BYTES;

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    if (entry->bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    if (!access_ok(src, entry->bytes))
        return NV_ERR_INVALID_ADDRESS;

    status = uvm_moe_slot_pool_ensure_vidmem_backing_locked(slot, gpu, copy_end, locked_va_space, mm);
    if (status != NV_OK)
        return status;

    max_chunk_bytes = UVM_ALIGN_DOWN(max_chunk_bytes, PAGE_SIZE);
    if (max_chunk_bytes == 0)
        max_chunk_bytes = PAGE_SIZE;

    while (copied < entry->bytes) {
        uvm_mem_t *staging = NULL;
        void *staging_addr;
        NvU64 remaining = entry->bytes - copied;
        NvU64 chunk_bytes = min(remaining, max_chunk_bytes);

        chunk_bytes = UVM_ALIGN_DOWN(chunk_bytes, PAGE_SIZE);
        if (chunk_bytes == 0)
            chunk_bytes = remaining;

        status = uvm_mem_alloc_sysmem_dma_and_map_cpu_kernel(chunk_bytes, gpu, current->mm, &staging);
        if (status != NV_OK) {
            if (status == NV_ERR_NO_MEMORY && max_chunk_bytes > PAGE_SIZE) {
                max_chunk_bytes = UVM_ALIGN_DOWN(max_chunk_bytes / 2, PAGE_SIZE);
                if (max_chunk_bytes < PAGE_SIZE)
                    max_chunk_bytes = PAGE_SIZE;
                uvm_phase_f_route_log("moe_slot_pool_gpu_copy_staging_retry layer=%u expert=%u slot=%u offset=%llu remaining=%llu next_chunk=%llu status=0x%x\n",
                                      entry->layer_id,
                                      entry->expert_id,
                                      entry->slot_id,
                                      copied,
                                      remaining,
                                      max_chunk_bytes,
                                      status);
                continue;
            }
            return status;
        }

        staging_addr = uvm_mem_get_cpu_addr_kernel(staging);
        if (!staging_addr) {
            status = NV_ERR_INVALID_STATE;
            goto done;
        }

        if (copy_from_user(staging_addr, src + copied, chunk_bytes) != 0) {
            status = NV_ERR_INVALID_ADDRESS;
            goto done;
        }

        status = uvm_mem_map_gpu_kernel(staging, gpu);
        if (status != NV_OK)
            goto done;

        status = uvm_moe_slot_pool_gpu_memcopy(gpu,
                                               UVM_CHANNEL_TYPE_CPU_TO_GPU,
                                               slot->slot_mem,
                                               entry->slot_offset + copied,
                                               staging,
                                               0,
                                               chunk_bytes);

done:
        uvm_mem_free(staging);
        if (status != NV_OK)
            return status;

        copied += chunk_bytes;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_copy_gpu_entry_async_locked(uvm_moe_slot_pool_slot_t *slot,
                                                               const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                               uvm_gpu_t *gpu,
                                                               uvm_va_space_t *locked_va_space,
                                                               struct mm_struct *mm)
{
    NV_STATUS status;
    const char __user *src = (const char __user *)(unsigned long)entry->va_base;
    NvU64 copy_end = entry->slot_offset + entry->bytes;
    uvm_mem_t *staging = NULL;
    void *staging_addr;
    uvm_push_t push;
    uvm_gpu_address_t dst;
    uvm_gpu_address_t staging_src;
    NvBool completed = NV_FALSE;

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    if (entry->bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    if (!access_ok(src, entry->bytes))
        return NV_ERR_INVALID_ADDRESS;

    if (slot->async_copy_pending) {
        status = uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_FALSE, &completed);
        if (status != NV_OK)
            return status;
        if (!completed)
            return NV_ERR_INVALID_STATE;
    }

    status = uvm_moe_slot_pool_ensure_vidmem_backing_locked(slot, gpu, copy_end, locked_va_space, mm);
    if (status != NV_OK)
        return status;

    status = uvm_mem_alloc_sysmem_dma_and_map_cpu_kernel(entry->bytes, gpu, current->mm, &staging);
    if (status != NV_OK)
        return status;

    staging_addr = uvm_mem_get_cpu_addr_kernel(staging);
    if (!staging_addr) {
        status = NV_ERR_INVALID_STATE;
        goto error;
    }

    if (copy_from_user(staging_addr, src, entry->bytes) != 0) {
        status = NV_ERR_INVALID_ADDRESS;
        goto error;
    }

    status = uvm_mem_map_gpu_kernel(staging, gpu);
    if (status != NV_OK)
        goto error;

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_CPU_TO_GPU,
                            &push,
                            "Async MoE slot pool copy");
    if (status != NV_OK)
        goto error;

    dst = uvm_mem_gpu_address_virtual_kernel(slot->slot_mem, gpu);
    dst.address += entry->slot_offset;
    staging_src = uvm_mem_gpu_address_virtual_kernel(staging, gpu);
    gpu->parent->ce_hal->memcopy(&push, dst, staging_src, (size_t)entry->bytes);
    uvm_push_end(&push);

    uvm_tracker_init(&slot->async_copy_tracker);
    uvm_tracker_overwrite_with_push(&slot->async_copy_tracker, &push);
    slot->async_copy_staging = staging;
    slot->async_copy_pending = NV_TRUE;
    slot->async_copy_bytes = entry->bytes;
    slot->async_copy_slot_offset = entry->slot_offset;
    slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->state = UVM_MOE_SLOT_POOL_STATE_LOADING;

    return NV_OK;

error:
    if (staging)
        uvm_mem_free(staging);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_copy_entry_locked(uvm_moe_slot_pool_slot_t *slot,
                                                     const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                     uvm_moe_slot_pool_copy_mode_t copy_mode,
                                                     uvm_gpu_t *gpu,
                                                     uvm_va_space_t *locked_va_space,
                                                     struct mm_struct *mm)
{
    if (copy_mode == UVM_MOE_SLOT_POOL_COPY_SYSMEM)
        return uvm_moe_slot_pool_copy_sysmem_entry_locked(slot, entry);

    if (copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU)
        return uvm_moe_slot_pool_copy_gpu_entry_locked(slot, entry, gpu, locked_va_space, mm);

    if (copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC)
        return uvm_moe_slot_pool_copy_gpu_entry_async_locked(slot, entry, gpu, locked_va_space, mm);

    return NV_OK;
}

typedef struct
{
    uvm_mem_t *mem;
    uvm_gpu_t *gpu;
    NvU64 slot_offset;
} uvm_moe_slot_pte_maker_data_t;

static NvU64 uvm_moe_slot_pte_maker(uvm_page_table_range_vec_t *range_vec, NvU64 offset, void *caller_data)
{
    uvm_moe_slot_pte_maker_data_t *data = (uvm_moe_slot_pte_maker_data_t *)caller_data;
    uvm_page_tree_t *tree = range_vec->tree;
    uvm_gpu_phys_address_t phys = uvm_mem_gpu_physical(data->mem,
                                                       data->gpu,
                                                       data->slot_offset + offset,
                                                       range_vec->page_size);

    return tree->hal->make_pte(phys.aperture,
                               phys.address,
                               UVM_PROT_READ_ONLY,
                               uvm_mem_is_vidmem(data->mem) ? UVM_MMU_PTE_FLAGS_CACHED : UVM_MMU_PTE_FLAGS_NONE);
}

static NV_STATUS uvm_moe_slot_pool_install_pte_mapping_locked(uvm_va_space_t *va_space,
                                                              uvm_moe_slot_pool_slot_t *slot,
                                                              const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                              uvm_gpu_t *gpu,
                                                              struct mm_struct *mm,
                                                              NvU32 pte_mode)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_page_table_range_vec_t *range_vec = NULL;
    uvm_page_tree_t *page_tables;
    uvm_membar_t tlb_membar;
    NvU64 map_end = entry->slot_offset + entry->bytes;
    NvU64 page_size;
    uvm_moe_slot_pte_maker_data_t pte_maker_data;

    if (!uvm_moe_slot_pte_remap_enable)
        return NV_ERR_NOT_SUPPORTED;

    if (!gpu || !slot->slot_mem || slot->backing_kind != UVM_MOE_SLOT_POOL_BACKING_VIDMEM)
        return NV_ERR_INVALID_STATE;

    if (slot->slot_gpu != gpu)
        return NV_ERR_INVALID_DEVICE;

    if (entry->bytes > (NvU64)uvm_moe_slot_pte_remap_max_bytes)
        return NV_ERR_INVALID_ARGUMENT;

    if (map_end < entry->slot_offset || map_end > slot->slot_bytes)
        return NV_ERR_INVALID_ARGUMENT;

    if (!uvm_gpu_can_address(gpu, entry->va_base, entry->bytes))
        return NV_ERR_OUT_OF_RANGE;

    gpu_va_space = va_space->gpu_va_spaces[uvm_id_gpu_index(gpu->id)];
    if (!gpu_va_space)
        return NV_ERR_INVALID_DEVICE;

    page_tables = &gpu_va_space->page_tables;
    page_size = pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ?
                    UVM_PAGE_SIZE_4K :
                    uvm_mmu_biggest_page_size_up_to(page_tables, min(entry->bytes, UVM_PAGE_SIZE_64K));
    if (page_size == 0)
        return NV_ERR_NOT_SUPPORTED;

    if (!IS_ALIGNED(entry->va_base, page_size) ||
        !IS_ALIGNED(entry->bytes, page_size) ||
        !IS_ALIGNED(entry->slot_offset, page_size))
        return NV_ERR_INVALID_ARGUMENT;

    uvm_phase_f_route_log("moe_pte_range action=install_range_vec_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu slot_offset=%llu slot_end=%llu page_size=%llu mode=%u va_space=%px mm=%px gpu=%s\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry->va_base + entry->bytes,
                          entry->bytes,
                          entry->slot_offset,
                          map_end,
                          page_size,
                          pte_mode,
                          va_space,
                          mm,
                          uvm_gpu_name(gpu));

    status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_pte_range action=install_range_vec_release_failed layer=%u expert=%u slot=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->va_base + entry->bytes,
                              entry->bytes,
                              status);
        return status;
    }

    status = uvm_page_table_range_vec_create(page_tables,
                                             entry->va_base,
                                             entry->bytes,
                                             page_size,
                                             UVM_PMM_ALLOC_FLAGS_EVICT,
                                             &range_vec);
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_pte_range action=install_range_vec_create_failed layer=%u expert=%u slot=%u va=0x%llx end=0x%llx bytes=%llu page_size=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->va_base + entry->bytes,
                              entry->bytes,
                              page_size,
                              status);
        return status;
    }

    pte_maker_data.mem = slot->slot_mem;
    pte_maker_data.gpu = gpu;
    pte_maker_data.slot_offset = entry->slot_offset;

    tlb_membar = uvm_hal_downgrade_membar_type(gpu, uvm_mem_is_local_vidmem(slot->slot_mem, gpu));
    status = uvm_page_table_range_vec_write_ptes(range_vec,
                                                 tlb_membar,
                                                 uvm_moe_slot_pte_maker,
                                                 &pte_maker_data);
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_pte_range action=install_range_vec_write_failed layer=%u expert=%u slot=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->va_base + entry->bytes,
                              entry->bytes,
                              status);
        (void)uvm_page_table_range_vec_clear_ptes(range_vec, tlb_membar);
        uvm_page_table_range_vec_destroy(range_vec);
        return status;
    }

    slot->pte_range_vec = range_vec;
    slot->pte_gpu = gpu;
    uvm_gpu_retain(gpu);
    slot->pte_va_base = entry->va_base;
    slot->pte_bytes = entry->bytes;
    slot->pte_va_space = va_space;
    slot->pte_slot_offset = entry->slot_offset;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->pte_mode = pte_mode;
    slot->pte_va_block_tracked = pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP;
    slot->pte_remap_count++;
    slot->pte_remap_bytes += entry->bytes;
    ++g_uvm_moe_slot_pool_pte_remap_count;
    g_uvm_moe_slot_pool_pte_remap_bytes += entry->bytes;

    uvm_phase_f_route_log("moe_pte_range action=install_range_vec_end layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu slot_offset=%llu page_size=%llu mode=%u status=0x%x remap_count=%llu remap_bytes=%llu\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry->va_base + entry->bytes,
                          entry->bytes,
                          entry->slot_offset,
                          page_size,
                          pte_mode,
                          status,
                          slot->pte_remap_count,
                          slot->pte_remap_bytes);

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_unmap_va_block_gpu_locked(uvm_va_block_t *block,
                                                             uvm_va_block_context_t *block_context,
                                                             const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                             uvm_gpu_t *gpu)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(block, entry->va_base, entry->bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    return uvm_va_block_unmap(block, block_context, gpu->id, region, page_mask, &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_map_va_block_slot_locked(uvm_va_block_t *block,
                                                            uvm_va_block_context_t *block_context,
                                                            const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                            uvm_gpu_t *gpu,
                                                            uvm_mem_t *slot_mem)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(block, entry->va_base, entry->bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    return uvm_va_block_map_moe_slot_readonly(block,
                                              block_context,
                                              gpu,
                                              region,
                                              page_mask,
                                              slot_mem,
                                              entry->slot_offset,
                                              &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_unmap_va_block_slot_locked(uvm_va_block_t *block,
                                                              uvm_va_block_context_t *block_context,
                                                              NvU64 va_base,
                                                              NvU64 bytes,
                                                              uvm_gpu_t *gpu)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(block, va_base, bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    return uvm_va_block_unmap_moe_slot(block,
                                       block_context,
                                       gpu,
                                       region,
                                       page_mask,
                                       &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_prepare_va_block_remap(uvm_va_space_t *va_space,
                                                          const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                          uvm_gpu_t *gpu,
                                                          struct mm_struct *mm,
                                                          NvU64 *block_unmap_count_out,
                                                          NvU64 *tracker_wait_count_out)
{
    NV_STATUS status;
    uvm_va_block_t *block = NULL;
    uvm_va_block_context_t *block_context;
    uvm_va_block_retry_t block_retry;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NvU64 entry_end = entry->va_base + entry->bytes - 1;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (entry_end < entry->va_base)
        return NV_ERR_INVALID_ARGUMENT;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    status = uvm_va_block_find_create_managed(va_space, entry->va_base, &block);
    if (status != NV_OK)
        goto out;

    if (entry_end > block->end) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                     &block_retry,
                                     uvm_moe_slot_pool_unmap_va_block_gpu_locked(block,
                                                                                 block_context,
                                                                                 entry,
                                                                                 gpu));
    if (status != NV_OK)
        goto out;

    if (block_unmap_count_out)
        *block_unmap_count_out = 1;

    uvm_mutex_lock(&block->lock);
    status = uvm_tracker_init_from(&local_tracker, &block->tracker);
    uvm_mutex_unlock(&block->lock);
    if (status != NV_OK)
        goto out;

    status = uvm_tracker_wait_deinit(&local_tracker);
    if (status != NV_OK)
        goto out;

    if (tracker_wait_count_out)
        *tracker_wait_count_out = 1;

out:
    uvm_va_block_context_free(block_context);
    return status;
}

static void uvm_moe_slot_pool_log_va_block_find_failed(uvm_va_space_t *va_space,
                                                       const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                       struct mm_struct *mm,
                                                       NvU64 cursor,
                                                       NvU64 remaining,
                                                       NV_STATUS status,
                                                       uvm_api_range_type_t range_type)
{
    struct mm_struct *va_space_mm = va_space ? va_space->va_space_mm.mm : NULL;
    struct vm_area_struct *vma = mm ? find_vma(mm, cursor) : NULL;
    NvU64 vma_start = vma ? (NvU64)vma->vm_start : 0;
    NvU64 vma_end = vma ? (NvU64)vma->vm_end : 0;
    unsigned long vma_flags = vma ? vma->vm_flags : 0;
    NvU32 vma_covers = vma && cursor >= (NvU64)vma->vm_start && cursor < (NvU64)vma->vm_end;
    NvU32 vma_read = vma && (vma->vm_flags & VM_READ);
    NvU32 vma_io = vma && (vma->vm_flags & VM_IO);
    NvU32 vma_pfnmap = vma && (vma->vm_flags & VM_PFNMAP);
    NvU32 vma_managed = vma && uvm_vma_is_managed(vma);

    uvm_phase_f_route_log("moe_slot_va_block_install_find_failed layer=%u expert=%u slot=%u cursor=0x%llx remaining=%llu status=0x%x range_type=%u hmm_enabled=%u mm=%px current_mm=%px va_space_mm=%px vma=%px vma=[0x%llx,0x%llx) vma_flags=0x%lx vma_covers=%u vma_read=%u vma_io=%u vma_pfnmap=%u vma_managed=%u vma_ops=%px\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          cursor,
                          remaining,
                          status,
                          range_type,
                          va_space ? (NvU32)uvm_hmm_is_enabled(va_space) : 0,
                          mm,
                          current->mm,
                          va_space_mm,
                          vma,
                          vma_start,
                          vma_end,
                          vma_flags,
                          vma_covers,
                          vma_read,
                          vma_io,
                          vma_pfnmap,
                          vma_managed,
                          vma ? vma->vm_ops : NULL);
}

static NV_STATUS uvm_moe_slot_pool_install_va_block_slot_mapping(uvm_va_space_t *va_space,
                                                                 const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                                 uvm_gpu_t *gpu,
                                                                 struct mm_struct *mm,
                                                                 uvm_mem_t *slot_mem,
                                                                 NvU64 *block_unmap_count_out,
                                                                 NvU64 *tracker_wait_count_out,
                                                                 NvU64 *rollback_bytes_out,
                                                                 NvU64 *rollback_block_unmap_count_out,
                                                                 NvU64 *rollback_tracker_wait_count_out)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_context_t *block_context;
    NvU64 entry_end = entry->va_base + entry->bytes;
    NvU64 cursor = entry->va_base;
    NvU64 installed_bytes = 0;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;
    if (rollback_bytes_out)
        *rollback_bytes_out = 0;
    if (rollback_block_unmap_count_out)
        *rollback_block_unmap_count_out = 0;
    if (rollback_tracker_wait_count_out)
        *rollback_tracker_wait_count_out = 0;

    if (!slot_mem)
        return NV_ERR_INVALID_STATE;

    if (entry_end <= entry->va_base)
        return NV_ERR_INVALID_ARGUMENT;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    uvm_phase_f_route_log("moe_pte_range action=install_va_block_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu slot_offset=%llu va_space=%px mm=%px gpu=%s\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry_end,
                          entry->bytes,
                          entry->slot_offset,
                          va_space,
                          mm,
                          uvm_gpu_name(gpu));

    while (cursor < entry_end) {
        UVM_MOE_SLOT_REMAP_ENTRY slice = *entry;
        uvm_va_block_t *block = NULL;
        uvm_va_block_retry_t block_retry;
        uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
        NvU64 slice_end;
        uvm_api_range_type_t range_type;

        block_context->hmm.vma = NULL;
        status = mm ?
                    uvm_va_block_find_create(va_space, cursor, &block_context->hmm.vma, &block) :
                    uvm_va_block_find_create_managed(va_space, cursor, &block);
        if (status != NV_OK) {
            range_type = uvm_api_range_type_check(va_space, mm, cursor, min(entry_end - cursor, (NvU64)PAGE_SIZE));
            uvm_moe_slot_pool_log_va_block_find_failed(va_space,
                                                       entry,
                                                       mm,
                                                       cursor,
                                                       entry_end - cursor,
                                                       status,
                                                       range_type);
            break;
        }

        if (cursor > block->end) {
            status = NV_ERR_INVALID_ARGUMENT;
            uvm_phase_f_route_log("moe_slot_va_block_install_cursor_after_block layer=%u expert=%u slot=%u cursor=0x%llx block=[0x%llx,0x%llx]\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  cursor,
                                  block->start,
                                  block->end);
            break;
        }

        slice_end = min(entry_end, block->end + 1);
        if (block_context->hmm.vma)
            slice_end = min(slice_end, (NvU64)block_context->hmm.vma->vm_end);
        if (slice_end <= cursor) {
            status = NV_ERR_INVALID_ARGUMENT;
            uvm_phase_f_route_log("moe_slot_va_block_install_empty_slice layer=%u expert=%u slot=%u cursor=0x%llx block=[0x%llx,0x%llx] entry_end=0x%llx\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  cursor,
                                  block->start,
                                  block->end,
                                  entry_end);
            break;
        }

        slice.va_base = cursor;
        slice.bytes = slice_end - cursor;
        slice.slot_offset = entry->slot_offset + (cursor - entry->va_base);

        uvm_phase_f_route_log("moe_pte_range action=install_va_block_slice_begin layer=%u expert=%u slot=%u role=%u slice_va=0x%llx slice_end=0x%llx slice_bytes=%llu slot_offset=%llu block_start=0x%llx block_end=0x%llx vma=%px\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->role,
                              slice.va_base,
                              slice_end,
                              slice.bytes,
                              slice.slot_offset,
                              block->start,
                              block->end,
                              block_context->hmm.vma);

        status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                         &block_retry,
                                         uvm_moe_slot_pool_unmap_va_block_gpu_locked(block,
                                                                                     block_context,
                                                                                     &slice,
                                                                                     gpu));
        if (status != NV_OK) {
            uvm_phase_f_route_log("moe_slot_va_block_install_unmap_failed layer=%u expert=%u slot=%u slice_va=0x%llx slice_bytes=%llu block=[0x%llx,0x%llx] status=0x%x\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  slice.va_base,
                                  slice.bytes,
                                  block->start,
                                  block->end,
                                  status);
            break;
        }

        if (block_unmap_count_out)
            ++*block_unmap_count_out;

        status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                         &block_retry,
                                         uvm_moe_slot_pool_map_va_block_slot_locked(block,
                                                                                   block_context,
                                                                                   &slice,
                                                                                   gpu,
                                                                                   slot_mem));
        if (status != NV_OK) {
            uvm_phase_f_route_log("moe_slot_va_block_install_map_failed layer=%u expert=%u slot=%u slice_va=0x%llx slice_bytes=%llu slot_offset=%llu block=[0x%llx,0x%llx] status=0x%x\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  slice.va_base,
                                  slice.bytes,
                                  slice.slot_offset,
                                  block->start,
                                  block->end,
                                  status);
            break;
        }

        installed_bytes += slice.bytes;

        uvm_mutex_lock(&block->lock);
        status = uvm_tracker_init_from(&local_tracker, &block->tracker);
        uvm_mutex_unlock(&block->lock);
        if (status != NV_OK)
            break;

        status = uvm_tracker_wait_deinit(&local_tracker);
        if (status != NV_OK)
            break;

        if (tracker_wait_count_out)
            ++*tracker_wait_count_out;

        uvm_phase_f_route_log("moe_pte_range action=install_va_block_slice_end layer=%u expert=%u slot=%u role=%u slice_va=0x%llx slice_end=0x%llx slice_bytes=%llu installed_bytes=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->role,
                              slice.va_base,
                              slice_end,
                              slice.bytes,
                              installed_bytes,
                              status);

        cursor = slice_end;
    }

    uvm_va_block_context_free(block_context);

    if (status != NV_OK && installed_bytes != 0) {
        NvU64 rollback_block_unmap_count = 0;
        NvU64 rollback_tracker_wait_count = 0;
        NV_STATUS rollback_status;

        rollback_status = uvm_moe_slot_pool_uninstall_va_block_slot_mapping(va_space,
                                                                            gpu,
                                                                            mm,
                                                                            entry->va_base,
                                                                            installed_bytes,
                                                                            &rollback_block_unmap_count,
                                                                            &rollback_tracker_wait_count);
        if (rollback_status == NV_OK) {
            if (rollback_bytes_out)
                *rollback_bytes_out = installed_bytes;
            if (rollback_block_unmap_count_out)
                *rollback_block_unmap_count_out = rollback_block_unmap_count;
            if (rollback_tracker_wait_count_out)
                *rollback_tracker_wait_count_out = rollback_tracker_wait_count;
        }
        else {
            UVM_ERR_PRINT("MoE slot partial VA-block rollback failed: %s, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(rollback_status),
                          entry->va_base,
                          entry->va_base + installed_bytes);
        }
    }

    uvm_phase_f_route_log("moe_pte_range action=install_va_block_end layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu installed=%llu status=0x%x\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry_end,
                          entry->bytes,
                          installed_bytes,
                          status);

    return status;
}

static NV_STATUS uvm_moe_slot_pool_uninstall_va_block_slot_mapping(uvm_va_space_t *va_space,
                                                                   uvm_gpu_t *gpu,
                                                                   struct mm_struct *mm,
                                                                   NvU64 va_base,
                                                                   NvU64 bytes,
                                                                   NvU64 *block_unmap_count_out,
                                                                   NvU64 *tracker_wait_count_out)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_context_t *block_context;
    NvU64 va_end = va_base + bytes;
    NvU64 cursor = va_base;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (!va_space || !gpu || bytes == 0)
        return NV_ERR_INVALID_ARGUMENT;

    if (va_end <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_begin va=0x%llx end=0x%llx bytes=%llu va_space=%px mm=%px gpu=%s\n",
                          va_base,
                          va_end,
                          bytes,
                          va_space,
                          mm,
                          uvm_gpu_name(gpu));

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    while (cursor < va_end) {
        uvm_va_block_t *block = NULL;
        uvm_va_block_retry_t block_retry;
        uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
        NvU64 slice_end;

        block_context->hmm.vma = NULL;
        status = mm ?
                    uvm_va_block_find_create(va_space, cursor, &block_context->hmm.vma, &block) :
                    uvm_va_block_find(va_space, cursor, &block);
        if (status == NV_ERR_OBJECT_NOT_FOUND || status == NV_ERR_INVALID_ADDRESS) {
            NvU64 next_block = UVM_VA_BLOCK_ALIGN_UP(cursor + 1);

            uvm_phase_f_route_log("moe_slot_uninstall_skip va=0x%llx next=0x%llx end=0x%llx status=0x%x mm=%px\n",
                                  cursor,
                                  next_block,
                                  va_end,
                                  status,
                                  mm);
            cursor = min(va_end, next_block);
            status = NV_OK;
            continue;
        }
        if (status != NV_OK)
            break;

        if (cursor > block->end) {
            status = NV_ERR_INVALID_ARGUMENT;
            break;
        }

        slice_end = min(va_end, block->end + 1);
        if (block_context->hmm.vma)
            slice_end = min(slice_end, (NvU64)block_context->hmm.vma->vm_end);
        if (slice_end <= cursor) {
            status = NV_ERR_INVALID_ARGUMENT;
            break;
        }

        uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_slice_begin block_start=0x%llx block_end=0x%llx slice_va=0x%llx slice_end=0x%llx bytes=%llu mm=%px vma=%px\n",
                              block->start,
                              block->end,
                              cursor,
                              slice_end,
                              slice_end - cursor,
                              mm,
                              block_context->hmm.vma);

        status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                         &block_retry,
                                         uvm_moe_slot_pool_unmap_va_block_slot_locked(block,
                                                                                      block_context,
                                                                                      cursor,
                                                                                      slice_end - cursor,
                                                                                      gpu));
        uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_slice_end block_start=0x%llx block_end=0x%llx slice_va=0x%llx slice_end=0x%llx status=0x%x\n",
                              block->start,
                              block->end,
                              cursor,
                              slice_end,
                              status);
        if (status != NV_OK)
            break;

        if (block_unmap_count_out)
            ++*block_unmap_count_out;

        uvm_mutex_lock(&block->lock);
        status = uvm_tracker_init_from(&local_tracker, &block->tracker);
        uvm_mutex_unlock(&block->lock);
        if (status != NV_OK)
            break;

        status = uvm_tracker_wait_deinit(&local_tracker);
        if (status != NV_OK)
            break;

        if (tracker_wait_count_out)
            ++*tracker_wait_count_out;

        cursor = slice_end;
    }

    uvm_va_block_context_free(block_context);
    uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_end va=0x%llx end=0x%llx bytes=%llu status=0x%x block_unmaps=%llu tracker_waits=%llu\n",
                          va_base,
                          va_end,
                          bytes,
                          status,
                          block_unmap_count_out ? *block_unmap_count_out : 0,
                          tracker_wait_count_out ? *tracker_wait_count_out : 0);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_map_va_block_user_backing_locked(uvm_va_block_t *target_block,
                                                                    uvm_va_block_t *backing_block,
                                                                    uvm_va_block_context_t *block_context,
                                                                    const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                                    uvm_gpu_t *gpu,
                                                                    NvBool defer_tlb_invalidate)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(target_block, entry->va_base, entry->bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    return uvm_va_block_map_moe_user_backing_readonly(target_block,
                                                      block_context,
                                                      gpu,
                                                      region,
                                                      page_mask,
                                                      backing_block,
                                                      entry->slot_offset,
                                                      defer_tlb_invalidate,
                                                      &target_block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_install_va_block_user_backing_mapping(uvm_va_space_t *va_space,
                                                                         const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                                         uvm_gpu_t *gpu,
                                                                         struct mm_struct *mm,
                                                                         NvBool defer_tlb_invalidate,
                                                                         NvU64 *block_unmap_count_out,
                                                                         NvU64 *tracker_wait_count_out)
{
    NV_STATUS status;
    uvm_va_block_context_t *block_context;
    uvm_va_block_t *target_block = NULL;
    uvm_va_block_t *backing_block = NULL;
    uvm_va_block_retry_t target_retry;
    uvm_va_block_retry_t backing_retry;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NvU64 entry_end = entry->va_base + entry->bytes - 1;
    NvU64 backing_end = entry->slot_offset + entry->bytes - 1;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (!va_space || !gpu || entry_end < entry->va_base || backing_end < entry->slot_offset)
        return NV_ERR_INVALID_ARGUMENT;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    status = uvm_va_block_find_create_managed(va_space, entry->va_base, &target_block);
    if (status != NV_OK)
        goto out;

    if (entry_end > target_block->end) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    status = uvm_va_block_find_create_managed(va_space, entry->slot_offset, &backing_block);
    if (status != NV_OK)
        goto out;

    if (backing_end > backing_block->end) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    status = UVM_VA_BLOCK_LOCK_RETRY(target_block,
                                     &target_retry,
                                     uvm_moe_slot_pool_unmap_va_block_gpu_locked(target_block,
                                                                                 block_context,
                                                                                 entry,
                                                                                 gpu));
    if (status != NV_OK)
        goto out;

    if (block_unmap_count_out)
        *block_unmap_count_out = 1;

    uvm_va_block_retry_init(&target_retry);
    uvm_va_block_retry_init(&backing_retry);

    // Lock in address order to avoid deadlock if target/backing blocks differ.
    if (target_block == backing_block) {
        uvm_mutex_lock(&target_block->lock);
    }
    else if (target_block->start < backing_block->start) {
        uvm_mutex_lock(&target_block->lock);
        uvm_mutex_lock(&backing_block->lock);
    }
    else {
        uvm_mutex_lock(&backing_block->lock);
        uvm_mutex_lock(&target_block->lock);
    }

    do {
        status = uvm_moe_slot_pool_map_va_block_user_backing_locked(target_block,
                                                                    backing_block,
                                                                    block_context,
                                                                    entry,
                                                                    gpu,
                                                                    defer_tlb_invalidate);
    } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);

    if (target_block != backing_block) {
        uvm_mutex_unlock(&backing_block->lock);
        uvm_mutex_unlock(&target_block->lock);
    }
    else {
        uvm_mutex_unlock(&target_block->lock);
    }

    uvm_va_block_retry_deinit(&backing_retry, backing_block);
    uvm_va_block_retry_deinit(&target_retry, target_block);

    if (status != NV_OK)
        goto out;

    uvm_mutex_lock(&target_block->lock);
    status = uvm_tracker_init_from(&local_tracker, &target_block->tracker);
    uvm_mutex_unlock(&target_block->lock);
    if (status != NV_OK)
        goto out;

    status = uvm_tracker_wait_deinit(&local_tracker);
    if (status != NV_OK)
        goto out;

    if (tracker_wait_count_out)
        *tracker_wait_count_out = 1;

out:
    uvm_va_block_context_free(block_context);
    return status;
}

static NvBool uvm_moe_slot_pool_user_backing_entry_in_block(uvm_va_block_t *target_block,
                                                            const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    NvU64 entry_end = entry->va_base + entry->bytes - 1;

    return entry_end >= entry->va_base &&
           entry->va_base >= target_block->start &&
           entry_end <= target_block->end;
}

static NV_STATUS uvm_moe_slot_pool_unmap_va_block_gpu_mask_locked(uvm_va_block_t *block,
                                                                  uvm_va_block_context_t *block_context,
                                                                  uvm_gpu_t *gpu,
                                                                  const uvm_page_mask_t *page_mask)
{
    return uvm_va_block_unmap(block,
                              block_context,
                              gpu->id,
                              uvm_va_block_region_from_block(block),
                              page_mask,
                              &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_map_one_user_backing_slice(uvm_va_space_t *va_space,
                                                              uvm_va_block_t *target_block,
                                                              uvm_va_block_context_t *block_context,
                                                              const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                              uvm_gpu_t *gpu,
                                                              NvBool defer_tlb_invalidate)
{
    NV_STATUS status;
    uvm_va_block_t *backing_block = NULL;
    uvm_va_block_retry_t target_retry;
    uvm_va_block_retry_t backing_retry;
    NvU64 backing_end = entry->slot_offset + entry->bytes - 1;

    if (backing_end < entry->slot_offset)
        return NV_ERR_INVALID_ARGUMENT;

    status = uvm_va_block_find_create_managed(va_space, entry->slot_offset, &backing_block);
    if (status != NV_OK)
        return status;

    if (backing_end > backing_block->end)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_block_retry_init(&target_retry);
    uvm_va_block_retry_init(&backing_retry);

    if (target_block == backing_block) {
        uvm_mutex_lock(&target_block->lock);
    }
    else if (target_block->start < backing_block->start) {
        uvm_mutex_lock(&target_block->lock);
        uvm_mutex_lock(&backing_block->lock);
    }
    else {
        uvm_mutex_lock(&backing_block->lock);
        uvm_mutex_lock(&target_block->lock);
    }

    do {
        status = uvm_moe_slot_pool_map_va_block_user_backing_locked(target_block,
                                                                    backing_block,
                                                                    block_context,
                                                                    entry,
                                                                    gpu,
                                                                    defer_tlb_invalidate);
    } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);

    if (target_block != backing_block) {
        uvm_mutex_unlock(&backing_block->lock);
        uvm_mutex_unlock(&target_block->lock);
    }
    else {
        uvm_mutex_unlock(&target_block->lock);
    }

    uvm_va_block_retry_deinit(&backing_retry, backing_block);
    uvm_va_block_retry_deinit(&target_retry, target_block);

    return status;
}

static NV_STATUS uvm_moe_slot_pool_install_user_backing_group(uvm_va_space_t *va_space,
                                                              UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                              NvU32 current_index,
                                                              uvm_gpu_t *gpu,
                                                              struct mm_struct *mm,
                                                              NvBool defer_tlb_invalidate,
                                                              uvm_moe_slot_pool_user_backing_group_state_t *group_state,
                                                              NvU64 *block_unmap_count_out,
                                                              NvU64 *tracker_wait_count_out)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_context_t *block_context = NULL;
    uvm_va_block_t *target_block = NULL;
    uvm_va_block_retry_t target_retry;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    uvm_page_mask_t *combined_mask;
    NvU32 candidate_indices[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 candidate_count = 0;
    NvU32 i;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (!group_state || group_state->installed[current_index])
        return NV_OK;

    status = uvm_va_block_find_create_managed(va_space, params->entries[current_index].va_base, &target_block);
    if (status != NV_OK)
        return status;

    if (!uvm_moe_slot_pool_user_backing_entry_in_block(target_block, &params->entries[current_index]))
        return NV_WARN_NOTHING_TO_DO;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    combined_mask = &block_context->caller_page_mask;
    uvm_page_mask_zero(combined_mask);

    for (i = current_index; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        uvm_va_block_region_t region;

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK || group_state->installed[i])
            continue;

        if (!uvm_moe_slot_pool_user_backing_entry_in_block(target_block, entry))
            continue;

        slot = uvm_moe_slot_pool_find_locked(entry->layer_id, entry->slot_id);
        if (!slot || slot->owner_expert_id != entry->expert_id)
            continue;

        if (slot->pte_range_count >= UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT)
            continue;

        if (slot->pte_bytes != 0 &&
            (slot->pte_va_space != va_space ||
             slot->pte_gpu != gpu ||
             slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP ||
             slot->batch_id != g_uvm_moe_slot_pool_current_batch_id))
            continue;

        region = uvm_va_block_region_from_start_size(target_block, entry->va_base, entry->bytes);
        uvm_page_mask_region_fill(combined_mask, region);
        candidate_indices[candidate_count++] = i;
    }

    if (candidate_count <= 1 || candidate_indices[0] != current_index) {
        status = NV_WARN_NOTHING_TO_DO;
        goto out;
    }

    status = UVM_VA_BLOCK_LOCK_RETRY(target_block,
                                     &target_retry,
                                     uvm_moe_slot_pool_unmap_va_block_gpu_mask_locked(target_block,
                                                                                      block_context,
                                                                                      gpu,
                                                                                      combined_mask));
    if (status != NV_OK)
        goto out;

    for (i = 0; i < candidate_count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[candidate_indices[i]];

        status = uvm_moe_slot_pool_map_one_user_backing_slice(va_space,
                                                              target_block,
                                                              block_context,
                                                              entry,
                                                              gpu,
                                                              defer_tlb_invalidate);
        if (status != NV_OK)
            goto rollback;
    }

    uvm_mutex_lock(&target_block->lock);
    status = uvm_tracker_init_from(&local_tracker, &target_block->tracker);
    uvm_mutex_unlock(&target_block->lock);
    if (status != NV_OK)
        goto rollback;

    status = uvm_tracker_wait_deinit(&local_tracker);
    if (status != NV_OK)
        goto rollback;

    for (i = 0; i < candidate_count; ++i)
        group_state->installed[candidate_indices[i]] = NV_TRUE;

    group_state->block_unmap_count[current_index] = 1;
    group_state->tracker_wait_count[current_index] = 1;

    if (block_unmap_count_out)
        *block_unmap_count_out = 1;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 1;

    goto out;

rollback:
    for (i = 0; i < candidate_count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[candidate_indices[i]];

        (void)uvm_moe_slot_pool_uninstall_va_block_slot_mapping(va_space,
                                                                gpu,
                                                                mm,
                                                                entry->va_base,
                                                                entry->bytes,
                                                                NULL,
                                                                NULL);
    }

out:
    uvm_va_block_context_free(block_context);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_user_backing_batch_tlb_invalidate(uvm_va_space_t *va_space, uvm_gpu_t *gpu)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_page_tree_t *tree;
    uvm_push_t push;

    if (!va_space || !gpu)
        return NV_ERR_INVALID_ARGUMENT;

    gpu_va_space = va_space->gpu_va_spaces[uvm_id_gpu_index(gpu->id)];
    if (!gpu_va_space || uvm_gpu_va_space_state(gpu_va_space) != UVM_GPU_VA_SPACE_STATE_ACTIVE)
        return NV_ERR_INVALID_DEVICE;

    tree = &gpu_va_space->page_tables;
    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_MEMOPS,
                            &push,
                            "Batch TLB invalidate for MoE user backing remap");
    if (status != NV_OK)
        return status;

    gpu->parent->host_hal->tlb_invalidate_all(&push,
                                              uvm_page_tree_pdb_address(tree),
                                              tree->hal->page_table_depth(UVM_PAGE_SIZE_4K),
                                              UVM_MEMBAR_GPU);

    uvm_push_end(&push);
    return uvm_push_wait(&push);
}

static NV_STATUS uvm_moe_slot_pool_apply_entry_locked(UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                      NvU64 batch_id,
                                                      NvU64 epoch,
                                                      uvm_moe_slot_pool_copy_mode_t copy_mode,
                                                      uvm_gpu_t *gpu,
                                                      uvm_va_space_t *locked_va_space,
                                                      struct mm_struct *mm)
{
    uvm_moe_slot_pool_slot_t *slot;
    NV_STATUS status = NV_OK;
    NvBool new_slot = NV_FALSE;
    NvBool overwrite = NV_FALSE;
    NvBool refresh = NV_FALSE;
    NvBool same_owner;

    slot = uvm_moe_slot_pool_find_locked(entry->layer_id, entry->slot_id);
    if (!slot) {
        slot = uvm_moe_slot_pool_find_free_locked();
        if (!slot) {
            ++g_uvm_moe_slot_pool_full_count;
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_NO_MEMORY;
        }
        new_slot = NV_TRUE;
    }

    same_owner = slot->valid && slot->owner_expert_id == entry->expert_id;
    overwrite = slot->valid && !same_owner;
    refresh = !slot->valid || !same_owner || slot->batch_id != batch_id;

    if (new_slot || refresh) {
        status = uvm_moe_slot_pool_clear_metadata_keep_backing_locked(slot, locked_va_space, mm);
        if (status != NV_OK) {
            entry->status = copy_mode == UVM_MOE_SLOT_POOL_COPY_NONE ?
                                UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED :
                                UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;
            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_status = entry->status;
            slot->pte_status = entry->status;
            return status;
        }

        slot->valid = NV_TRUE;
        slot->layer_id = entry->layer_id;
        slot->slot_id = entry->slot_id;
        slot->owner_expert_id = entry->expert_id;
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
        slot->first_va_base = entry->va_base;
        if (overwrite)
            ++g_uvm_moe_slot_pool_overwrite_count;
    }
    else if (!slot->valid) {
        slot->valid = NV_TRUE;
        slot->layer_id = entry->layer_id;
        slot->slot_id = entry->slot_id;
        slot->owner_expert_id = entry->expert_id;
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
        slot->first_va_base = entry->va_base;
    }

    if (copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM) {
        status = uvm_moe_slot_pool_ensure_vidmem_backing_locked(slot,
                                                                gpu,
                                                                entry->slot_offset + entry->bytes,
                                                                locked_va_space,
                                                                mm);
        if (status != NV_OK) {
            if (status == NV_ERR_NO_MEMORY)
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            else
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_role = entry->role;
            slot->flags = entry->flags;
            slot->last_status = entry->status;
            slot->copy_status = entry->status;
            slot->epoch = epoch;
            slot->batch_id = batch_id;
            slot->last_va_base = entry->va_base;
            slot->last_bytes = entry->bytes;
            return status;
        }

        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    }

    if (copy_mode != UVM_MOE_SLOT_POOL_COPY_NONE &&
        copy_mode != UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM) {
        status = uvm_moe_slot_pool_copy_entry_locked(slot, entry, copy_mode, gpu, locked_va_space, mm);
        if (status != NV_OK) {
            if (status == NV_ERR_NO_MEMORY)
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            else
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_role = entry->role;
            slot->flags = entry->flags;
            slot->last_status = entry->status;
            slot->copy_status = entry->status;
            slot->epoch = epoch;
            slot->batch_id = batch_id;
            slot->last_va_base = entry->va_base;
            slot->last_bytes = entry->bytes;
            slot->copy_failed_count++;
            g_uvm_moe_slot_pool_copy_failed_count++;
            return status;
        }

        if (copy_mode != UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC) {
            slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
            slot->copy_count++;
            slot->copy_bytes += entry->bytes;
            g_uvm_moe_slot_pool_copy_count++;
            g_uvm_moe_slot_pool_copy_bytes += entry->bytes;
        }
    }

    slot->range_count++;
    slot->last_role = entry->role;
    slot->flags = entry->flags;
    slot->last_status = entry->status;
    slot->epoch = epoch;
    slot->batch_id = batch_id;
    slot->last_va_base = entry->va_base;
    slot->last_bytes = entry->bytes;
    slot->total_bytes += entry->bytes;
    ++g_uvm_moe_slot_pool_range_update_count;

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_apply_batch(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                               uvm_moe_slot_pool_copy_mode_t copy_mode,
                                               uvm_gpu_t *gpu,
                                               uvm_va_space_t *va_space)
{
    NV_STATUS first_error = NV_OK;
    NvU64 batch_id;
    NvU32 i;
    NvBool has_candidate = NV_FALSE;
    struct mm_struct *mm = NULL;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_SLOT_REMAP_ENTRY_OK) {
            has_candidate = NV_TRUE;
            break;
        }
    }

    if (!has_candidate)
        return NV_OK;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    mutex_lock(&g_uvm_moe_slot_pool_lock);
    batch_id = ++g_uvm_moe_slot_pool_current_batch_id;
    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        NvU64 epoch;
        NV_STATUS status;

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        epoch = g_uvm_moe_slot_pool_current_epoch + 1;
        status = uvm_moe_slot_pool_apply_entry_locked(entry,
                                                      batch_id,
                                                      epoch,
                                                      copy_mode,
                                                      gpu,
                                                      va_space,
                                                      mm);
        if (status == NV_OK) {
            g_uvm_moe_slot_pool_current_epoch = epoch;
            ++g_uvm_moe_slot_pool_validated_entries;
            ++params->success_count;
            params->accepted_bytes += entry->bytes;
        }
        else {
            ++params->failure_count;
            if (first_error == NV_OK)
                first_error = status;
        }
    }
    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return first_error;
}

static void uvm_moe_slot_remap_fail_ok_entries(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                               NvU32 entry_status);

static NvU32 uvm_moe_slot_pte_status_from_nv_status(NV_STATUS status)
{
    if (status == NV_ERR_NOT_SUPPORTED)
        return UVM_MOE_SLOT_REMAP_ENTRY_REMAP_NOT_SUPPORTED;

    if (status == NV_ERR_INVALID_ARGUMENT || status == NV_ERR_OUT_OF_RANGE)
        return UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;

    if (status == NV_ERR_INVALID_ADDRESS)
        return UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ADDRESS;

    return UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
}

static NV_STATUS uvm_moe_slot_pool_apply_slot_pte_batch(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                        uvm_va_space_t *va_space,
                                                        uvm_gpu_t *gpu,
                                                        NvBool use_va_block_path,
                                                        NvBool use_user_backing_path)
{
    NV_STATUS first_error = NV_OK;
    NvU64 batch_id;
    NvU32 i;
    NvBool has_candidate = NV_FALSE;
    struct mm_struct *mm = NULL;
    uvm_moe_slot_pool_pte_txn_t *txn = NULL;
    uvm_moe_slot_pool_user_backing_group_state_t *group_state = NULL;
    NvBool defer_user_backing_tlb_invalidate =
        use_user_backing_path && uvm_moe_slot_user_backing_defer_tlb_invalidate != 0;
    NvBool grouped_user_backing_remap =
        use_user_backing_path && uvm_moe_slot_user_backing_grouped_remap != 0;

    if (!uvm_moe_slot_pte_remap_enable) {
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_NOT_SUPPORTED);
        return NV_ERR_NOT_SUPPORTED;
    }

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_SLOT_REMAP_ENTRY_OK) {
            has_candidate = NV_TRUE;
            break;
        }
    }

    if (!has_candidate)
        return NV_OK;

    if (use_va_block_path) {
        txn = uvm_kvmalloc_zero(sizeof(*txn));
        if (!txn) {
            uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            return NV_ERR_NO_MEMORY;
        }
    }

    if (grouped_user_backing_remap) {
        group_state = uvm_kvmalloc_zero(sizeof(*group_state));
        if (!group_state) {
            uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            uvm_kvfree(txn);
            return NV_ERR_NO_MEMORY;
        }
    }

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    mutex_lock(&g_uvm_moe_slot_pool_lock);
    batch_id = ++g_uvm_moe_slot_pool_current_batch_id;


    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        NV_STATUS status;
        NvU64 epoch;
        NvU64 block_unmap_count = 0;
        NvU64 tracker_wait_count = 0;
        NvU64 rollback_bytes = 0;
        NvU64 rollback_block_unmap_count = 0;
        NvU64 rollback_tracker_wait_count = 0;
        NvBool release_existing_mapping = NV_TRUE;

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        slot = uvm_moe_slot_pool_find_locked(entry->layer_id, entry->slot_id);
        if (!slot || (!slot->slot_mem && !use_user_backing_path)) {
            status = NV_ERR_OBJECT_NOT_FOUND;
        }
        else if (slot->owner_expert_id != entry->expert_id) {
            status = NV_ERR_INVALID_STATE;
        }
        else if (use_va_block_path) {
            release_existing_mapping = slot->pte_bytes != 0 &&
                                       (slot->pte_va_space != va_space ||
                                        slot->pte_gpu != gpu ||
                                        slot->pte_mode != (use_user_backing_path ?
                                                              UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP :
                                                              UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP) ||
                                        slot->batch_id != batch_id);

            if (group_state && !group_state->installed[i]) {
                status = uvm_moe_slot_pool_install_user_backing_group(va_space,
                                                                      params,
                                                                      i,
                                                                      gpu,
                                                                      mm,
                                                                      defer_user_backing_tlb_invalidate,
                                                                      group_state,
                                                                      &block_unmap_count,
                                                                      &tracker_wait_count);
                if (status == NV_WARN_NOTHING_TO_DO)
                    status = NV_OK;
            }
            else {
                status = NV_OK;
            }

            if (status == NV_OK && (!group_state || !group_state->installed[i])) {
                status = release_existing_mapping ?
                             uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm) :
                             NV_OK;
            }
            if (status == NV_OK) {
                if (group_state && group_state->installed[i]) {
                    block_unmap_count = group_state->block_unmap_count[i];
                    tracker_wait_count = group_state->tracker_wait_count[i];
                }
                else if (use_user_backing_path) {
                    status = uvm_moe_slot_pool_install_va_block_user_backing_mapping(va_space,
	                                                                                    entry,
	                                                                                    gpu,
	                                                                                    mm,
	                                                                                    defer_user_backing_tlb_invalidate,
	                                                                                    &block_unmap_count,
	                                                                                    &tracker_wait_count);
                }
                else {
                    status = uvm_moe_slot_pool_install_va_block_slot_mapping(va_space,
                                                                             entry,
                                                                             gpu,
                                                                             mm,
                                                                             slot->slot_mem,
                                                                             &block_unmap_count,
                                                                             &tracker_wait_count,
                                                                             &rollback_bytes,
                                                                             &rollback_block_unmap_count,
                                                                             &rollback_tracker_wait_count);
                }
            }
            if (status == NV_OK) {
                if (slot->pte_gpu != gpu) {
                    slot->pte_gpu = gpu;
                    uvm_gpu_retain(gpu);
                }
                slot->pte_va_space = va_space;
                slot->pte_slot_offset = entry->slot_offset;
                slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
                slot->pte_mode = use_user_backing_path ?
                                      UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP :
                                      UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP;
                slot->pte_va_block_tracked = NV_TRUE;
                status = uvm_moe_slot_pool_record_pte_range_locked(slot, entry->va_base, entry->bytes);
                if (status != NV_OK) {
                    (void)uvm_moe_slot_pool_uninstall_va_block_slot_mapping(va_space,
                                                                            gpu,
                                                                            mm,
                                                                            entry->va_base,
                                                                            entry->bytes,
                                                                            NULL,
                                                                            NULL);
                }
            }

            if (status == NV_OK) {
                slot->pte_remap_count++;
                slot->pte_remap_bytes += entry->bytes;
                ++g_uvm_moe_slot_pool_pte_remap_count;
                g_uvm_moe_slot_pool_pte_remap_bytes += entry->bytes;
                g_uvm_moe_slot_pool_pte_block_unmap_count += block_unmap_count;
                g_uvm_moe_slot_pool_pte_block_tracker_wait_count += tracker_wait_count;
            }
            else if (rollback_bytes != 0) {
                uvm_moe_slot_pool_note_partial_rollback_locked(slot,
                                                               rollback_bytes,
                                                               rollback_block_unmap_count,
                                                               rollback_tracker_wait_count);
            }
        }
        else {
            status = uvm_moe_slot_pool_install_pte_mapping_locked(
                va_space,
                slot,
                entry,
                gpu,
                mm,
                UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP);
        }

        if (status == NV_OK) {
            UVM_ASSERT(slot);
            epoch = g_uvm_moe_slot_pool_current_epoch + 1;
            slot->range_count++;
            slot->last_role = entry->role;
            slot->flags = entry->flags;
            slot->last_status = entry->status;
            slot->epoch = epoch;
            slot->batch_id = batch_id;
            slot->last_va_base = entry->va_base;
            slot->last_bytes = entry->bytes;
            slot->total_bytes += entry->bytes;
            ++g_uvm_moe_slot_pool_range_update_count;
            g_uvm_moe_slot_pool_current_epoch = epoch;
            ++g_uvm_moe_slot_pool_validated_entries;
            ++params->success_count;
            params->accepted_bytes += entry->bytes;
            ++params->remap_count;
            if (!defer_user_backing_tlb_invalidate)
                ++params->tlb_invalidate_count;
            if (use_va_block_path) {
                slot->pte_block_unmap_count += block_unmap_count;
                slot->pte_block_tracker_wait_count += tracker_wait_count;
                txn->installed_slots[txn->installed_count] = slot;
                txn->installed_indices[txn->installed_count] = i;
                ++txn->installed_count;
            }
        }
        else {
            entry->status = uvm_moe_slot_pte_status_from_nv_status(status);
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (slot) {
                slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
                slot->last_status = entry->status;
                slot->pte_status = entry->status;
            }
            ++params->failure_count;
            if (first_error == NV_OK)
                first_error = status;

            if (use_va_block_path) {
                NvU32 rollback_index;

                txn->rollback_batch = NV_TRUE;
                for (rollback_index = txn->installed_count; rollback_index > 0; --rollback_index) {
                    NvU32 installed_pos = rollback_index - 1;
                    NvU32 entry_index = txn->installed_indices[installed_pos];
                    uvm_moe_slot_pool_slot_t *installed_slot = txn->installed_slots[installed_pos];
                    NV_STATUS rollback_status;

                    rollback_status = uvm_moe_slot_pool_rollback_pte_mapping_locked(installed_slot, va_space, mm);
                    if (rollback_status != NV_OK) {
                        uvm_moe_slot_pool_note_pte_failure_locked(installed_slot);
                        if (first_error == NV_OK)
                            first_error = rollback_status;
                        continue;
                    }

                    params->entries[entry_index].status = UVM_MOE_SLOT_REMAP_ENTRY_ROLLED_BACK;
                    if (params->success_count > 0)
                        --params->success_count;
                    if (params->remap_count > 0)
                        --params->remap_count;
                    if (params->tlb_invalidate_count > 0)
                        --params->tlb_invalidate_count;
                    params->accepted_bytes -= min(params->accepted_bytes, params->entries[entry_index].bytes);
                    ++params->failure_count;
                }
                txn->installed_count = 0;
            }
        }

        uvm_phase_f_route_log("moe_slot_pte_remap_apply index=%u mode=%u layer=%u expert=%u slot=%u va=0x%llx bytes=%llu status=%u nv_status=0x%x remap_count=%u tlb_invalidate_count=%u\n",
                              i,
                              use_user_backing_path ?
                                  UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP :
                                  (use_va_block_path ? UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP :
                                                       UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP),
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->bytes,
                              entry->status,
                              status,
                              params->remap_count,
                              params->tlb_invalidate_count);

        if (txn && txn->rollback_batch)
            break;
    }

    if (txn && txn->rollback_batch) {
        NvU32 skipped;

        for (skipped = i + 1; skipped < params->count; ++skipped) {
            if (params->entries[skipped].status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
                continue;

            params->entries[skipped].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            ++params->failure_count;
        }
    }

    if (defer_user_backing_tlb_invalidate && params->success_count > 0 && first_error == NV_OK) {
        NV_STATUS invalidate_status = uvm_moe_slot_pool_user_backing_batch_tlb_invalidate(va_space, gpu);

        if (invalidate_status == NV_OK) {
            params->tlb_invalidate_count = 1;
        }
        else {
            NvU32 failed_index;

            if (first_error == NV_OK)
                first_error = invalidate_status;

            for (failed_index = 0; failed_index < params->count; ++failed_index) {
                if (params->entries[failed_index].status == UVM_MOE_SLOT_REMAP_ENTRY_OK)
                    params->entries[failed_index].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            }
        }
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    uvm_kvfree(group_state);
    uvm_kvfree(txn);

    return first_error;
}

static NV_STATUS uvm_moe_slot_remap_migrate_entry(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                  UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                  struct file *filp)
{
    NV_STATUS status;
    UVM_MIGRATE_PARAMS *migrate_params = uvm_kvmalloc_zero(sizeof(*migrate_params));

    if (!migrate_params)
        return NV_ERR_NO_MEMORY;

    migrate_params->base = entry->va_base;
    migrate_params->length = entry->bytes;
    migrate_params->destinationUuid = params->gpu_uuid;
    migrate_params->flags = 0;
    migrate_params->semaphoreAddress = 0;
    migrate_params->semaphorePayload = 0;
    migrate_params->cpuNumaNode = NUMA_NO_NODE;

    status = uvm_api_migrate(migrate_params, filp);
    uvm_kvfree(migrate_params);

    return status;
}

static NV_STATUS uvm_moe_slot_remap_validate_managed_range(uvm_va_space_t *va_space,
                                                           NvU64 base,
                                                           NvU64 length)
{
    NV_STATUS status = NV_OK;
    struct mm_struct *mm = NULL;
    bool retained_mm = false;

    if (uvm_va_space_mm_enabled(va_space)) {
        mm = uvm_va_space_mm_retain(va_space);
        retained_mm = mm != NULL;
    }
    else if (!(current->flags & PF_KTHREAD)) {
        mm = current->mm;
    }

    if (mm)
        uvm_down_read_mmap_lock(mm);

    uvm_va_space_down_read(va_space);
    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED)
        status = NV_ERR_INVALID_ADDRESS;
    uvm_va_space_up_read(va_space);

    if (mm)
        uvm_up_read_mmap_lock(mm);
    if (retained_mm)
        uvm_va_space_mm_release(va_space);

    return status;
}

static bool uvm_moe_slot_remap_has_ok_entries(const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_SLOT_REMAP_ENTRY_OK)
            return true;
    }

    return false;
}

static void uvm_moe_slot_remap_fail_ok_entries(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                               NvU32 entry_status)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        entry->status = entry_status;
        ++params->failure_count;
    }
}

static NV_STATUS uvm_moe_slot_pool_retain_gpu_by_uuid(const NvProcessorUuid *gpu_uuid,
                                                      uvm_gpu_t **gpu_out)
{
    uvm_rm_user_object_t user_rm_device = { 0 };
    uvm_test_parent_gpu_inject_error_t parent_gpu_error = { 0 };

    return uvm_gpu_retain_by_uuid(gpu_uuid,
                                  &user_rm_device,
                                  &parent_gpu_error,
                                  false,
                                  gpu_out);
}

static void uvm_expert_offload_release_zero(UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    NvU64 region_id = params->region_id;
    NvProcessorUuid gpu_uuid = params->gpu_uuid;
    NvU32 flags = params->flags;

    memset(params, 0, sizeof(*params));
    params->region_id = region_id;
    params->gpu_uuid = gpu_uuid;
    params->flags = flags;
}

static void uvm_expert_offload_release_record_failure(UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params,
                                                       NvU64 failed_address,
                                                       NvU64 failed_bytes,
                                                       NV_STATUS status)
{
    ++params->failure_blocks;
    params->failed_bytes += failed_bytes;

    if (params->first_error == NV_OK) {
        params->first_error = status;
        params->first_failed_block = failed_address;
    }
}

static NV_STATUS uvm_expert_offload_release_block(uvm_va_space_t *va_space,
                                                  uvm_gpu_t *gpu,
                                                  uvm_va_block_t *block,
                                                  const uvm_phase_f_region_t *region,
                                                  UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    uvm_va_block_gpu_state_t *gpu_state;
    uvm_gpu_chunk_t *chunk;
    NvU64 released_bytes = 0;
    NV_STATUS status;

    if (uvm_va_block_size(block) != UVM_CHUNK_SIZE_MAX) {
        ++params->failure_blocks;
        params->failed_bytes += uvm_va_block_size(block);
        if (params->first_error == NV_OK) {
            params->first_error = NV_ERR_NOT_SUPPORTED;
            params->first_failed_block = block->start;
        }
        return NV_ERR_NOT_SUPPORTED;
    }

    gpu_state = uvm_va_block_gpu_state_get(block, gpu->id);
    if (!gpu_state || !gpu_state->chunks || !gpu_state->chunks[0])
        return NV_WARN_NOTHING_TO_DO;

    chunk = gpu_state->chunks[0];
    if (uvm_gpu_chunk_get_size(chunk) != UVM_CHUNK_SIZE_MAX) {
        ++params->failure_blocks;
        params->failed_bytes += UVM_CHUNK_SIZE_MAX;
        if (params->first_error == NV_OK) {
            params->first_error = NV_ERR_NOT_SUPPORTED;
            params->first_failed_block = block->start;
        }
        return NV_ERR_NOT_SUPPORTED;
    }

    ++params->attempted_blocks;
    params->attempted_bytes += UVM_CHUNK_SIZE_MAX;

    status = uvm_pmm_gpu_evict_root_chunk_to_free(&gpu->pmm, chunk, &released_bytes);
    if (status == NV_OK) {
        ++params->success_blocks;
        params->released_bytes += released_bytes;
    }
    else {
        ++params->failure_blocks;
        params->failed_bytes += UVM_CHUNK_SIZE_MAX;
        if (params->first_error == NV_OK) {
            params->first_error = status;
            params->first_failed_block = block->start;
        }
    }

    return status;
}

static NV_STATUS uvm_expert_offload_release_subchunk_block(uvm_va_space_t *va_space,
                                                           struct mm_struct *mm,
                                                           uvm_gpu_t *gpu,
                                                           uvm_va_block_t *block,
                                                           const uvm_phase_f_region_t *region,
                                                           UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    NvU64 region_end = region->base + region->length - 1;
    NvU64 release_start = max(region->base, block->start);
    NvU64 release_end = min(region_end, block->end);
    uvm_va_block_region_t block_region;
    NvU64 release_bytes;
    NvU64 released_bytes = 0;
    NV_STATUS status;

    if (release_end < release_start)
        return NV_WARN_NOTHING_TO_DO;

    block_region = uvm_va_block_region_from_start_end(block, release_start, release_end);
    release_bytes = uvm_va_block_region_size(block_region);

    if (release_bytes == 0)
        return NV_WARN_NOTHING_TO_DO;

    ++params->attempted_blocks;
    params->attempted_bytes += release_bytes;

    status = uvm_va_block_release_gpu_subchunks_to_cpu(block,
                                                       mm,
                                                       gpu,
                                                       block_region,
                                                       UVM_CHUNK_SIZE_64K,
                                                       &block->tracker,
                                                       &released_bytes);
    if (status == NV_OK) {
        ++params->success_blocks;
        params->released_bytes += released_bytes;
    }
    else {
        ++params->failure_blocks;
        params->failed_bytes += release_bytes;
        if (params->first_error == NV_OK) {
            params->first_error = status;
            params->first_failed_block = release_start;
        }
    }

    return status;
}

static NV_STATUS uvm_expert_offload_remote_map_block(uvm_gpu_t *gpu,
                                                     uvm_va_block_t *block,
                                                     uvm_va_block_context_t *block_context,
                                                     uvm_va_block_region_t block_region,
                                                     UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    uvm_va_block_gpu_state_t *gpu_state;
    uvm_page_index_t page_index;
    NvU32 expected_pages = uvm_va_block_region_num_pages(block_region);
    NvU32 cpu_pages = 0;
    NvU32 remote_pages = 0;
    NvU32 newly_mapped_pages = 0;
    NV_STATUS status;

    status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                     NULL,
                                     uvm_va_block_premap_readmostly_remote_read(block,
                                                                                block_context,
                                                                                gpu->id,
                                                                                block_region,
                                                                                true,
                                                                                &newly_mapped_pages));
    if (status != NV_OK)
        goto failed;

    uvm_mutex_lock(&block->lock);
    gpu_state = uvm_va_block_gpu_state_get(block, gpu->id);
    for_each_va_block_page_in_region(page_index, block_region) {
        bool cpu_resident = uvm_page_mask_test(&block->cpu.resident, page_index);
        bool gpu_resident = gpu_state &&
                            uvm_page_mask_test(&gpu_state->resident, page_index);
        bool gpu_read_mapped = gpu_state &&
                               uvm_page_mask_test(&gpu_state->pte_bits[UVM_PTE_BITS_GPU_READ],
                                                  page_index);

        if (cpu_resident)
            ++cpu_pages;
        if (cpu_resident && !gpu_resident && gpu_read_mapped)
            ++remote_pages;
    }
    uvm_mutex_unlock(&block->lock);

    params->cpu_resident_bytes += (NvU64)cpu_pages * PAGE_SIZE;
    params->remote_mapped_bytes += (NvU64)remote_pages * PAGE_SIZE;
    params->newly_remote_mapped_bytes += (NvU64)newly_mapped_pages * PAGE_SIZE;

    if (cpu_pages != expected_pages || remote_pages != expected_pages) {
        status = NV_ERR_INVALID_STATE;
        goto failed;
    }

    ++params->remote_map_blocks;
    return NV_OK;

failed:
    ++params->remote_map_failure_blocks;
    if (params->first_error == NV_OK) {
        params->first_error = status;
        params->first_failed_block = block->start +
                                     (NvU64)block_region.first * PAGE_SIZE;
    }
    return status;
}

static NV_STATUS uvm_api_expert_offload_release(UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params,
                                                struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    uvm_phase_f_region_t *region = NULL;
    uvm_va_block_context_t *block_context = NULL;
    NvU64 region_end;
    NvU64 addr;
    bool placement_complete = false;
    bool remote_map_only = false;
    NV_STATUS status = NV_OK;

    if (params->flags & ~(UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT |
                          UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K |
                          UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ)) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    status = uvm_moe_slot_pool_retain_gpu_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK) {
        uvm_expert_offload_release_zero(params);
        params->rmStatus = status;
        return status;
    }

    uvm_expert_offload_release_zero(params);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    if (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ)
        block_context = uvm_va_space_block_context(va_space, mm);

    region = radix_tree_lookup(&va_space->phase_f_regions, params->region_id);
    if (!region) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto done_unlock;
    }

    params->base = region->base;
    params->length = region->length;

    if (region->state == UVM_REGION_STATE_CPU) {
        if (!(params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ)) {
            placement_complete = true;
            goto done_unlock;
        }

        /* CPU placement does not prove that GPU remote-read PTEs exist. Avoid
         * re-running the release primitive, which rejects already-installed
         * remote PTEs, and instead establish and validate those PTEs below. */
        remote_map_only = true;
    }
    /* UNKNOWN is not an idempotent success, but it is safe to converge by
     * executing the explicitly requested release. */
    region->state = UVM_REGION_STATE_PREFETCHING_CPU;

    if (region->length == 0) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done_unlock;
    }

    region_end = region->base + region->length - 1;
    if (region_end < region->base) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done_unlock;
    }

    if (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K) {
        if (!IS_ALIGNED(region->base, UVM_CHUNK_SIZE_64K) ||
            !IS_ALIGNED(region->length, UVM_CHUNK_SIZE_64K)) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto done_unlock;
        }
    }

    addr = region->base;
    while (addr <= region_end) {
        uvm_va_range_t *va_range = uvm_va_range_find(va_space, addr);
        uvm_va_range_managed_t *managed_range;
        size_t first_block_index;
        size_t last_block_index;
        size_t block_index;
        NvU64 managed_start;
        NvU64 managed_end;
        NvU64 covered_end;

        if (!va_range) {
            uvm_expert_offload_release_record_failure(params,
                                                       addr,
                                                       region_end - addr + 1,
                                                       NV_ERR_INVALID_ADDRESS);
            break;
        }

        covered_end = min(region_end, va_range->node.end);
        managed_range = uvm_va_range_to_managed_or_null(va_range);
        if (!managed_range) {
            uvm_expert_offload_release_record_failure(params,
                                                       addr,
                                                       covered_end - addr + 1,
                                                       NV_ERR_INVALID_ADDRESS);
            goto next_range;
        }

        managed_start = max(region->base, managed_range->va_range.node.start);
        managed_end = min(region_end, managed_range->va_range.node.end);
        first_block_index = uvm_va_range_block_index(managed_range, managed_start);
        last_block_index = uvm_va_range_block_index(managed_range, managed_end);

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            uvm_va_block_region_t block_region;
            NV_STATUS block_status;

            block_status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (block_status != NV_OK) {
                ++params->failure_blocks;
                if (params->first_error == NV_OK) {
                    params->first_error = block_status;
                    params->first_failed_block = managed_start;
                }
                break;
            }

            ++params->block_count;

            block_region = uvm_va_block_region_from_start_end(block,
                                                               max(region->base, block->start),
                                                               min(region_end, block->end));

            if (remote_map_only) {
                block_status = uvm_expert_offload_remote_map_block(gpu,
                                                                   block,
                                                                   block_context,
                                                                   block_region,
                                                                   params);
            }
            else if (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K) {
                block_status = uvm_expert_offload_release_subchunk_block(va_space,
                                                                         mm,
                                                                         gpu,
                                                                         block,
                                                                         region,
                                                                         params);
            }
            else {
                block_status = uvm_expert_offload_release_block(va_space,
                                                                gpu,
                                                                block,
                                                                region,
                                                                params);
            }

            if (!remote_map_only &&
                (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ) &&
                (block_status == NV_OK || block_status == NV_WARN_NOTHING_TO_DO)) {
                block_status = uvm_expert_offload_remote_map_block(gpu,
                                                                   block,
                                                                   block_context,
                                                                   block_region,
                                                                   params);
            }

            if (block_status == NV_WARN_NOTHING_TO_DO) {
                ++params->skipped_blocks;
            }
        }

next_range:
        if (covered_end == region_end) {
            placement_complete = true;
            break;
        }

        addr = covered_end + 1;
    }

done_unlock:
    if (status == NV_OK && !placement_complete) {
        status = params->first_error != NV_OK ?
                 params->first_error :
                 NV_ERR_INVALID_ADDRESS;
    }
    if (status == NV_OK && params->first_error != NV_OK)
        status = params->first_error;
    if (region) {
        region->state = status == NV_OK && placement_complete ?
                        UVM_REGION_STATE_CPU :
                        UVM_REGION_STATE_UNKNOWN;
    }

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    uvm_gpu_release(gpu);

    params->rmStatus = status;

    uvm_phase_f_route_log("region_release region_id=%llu base=0x%llx length=%llu blocks=%llu attempted=%u success=%u failure=%u skipped=%u attempted_bytes=%llu released_bytes=%llu failed_bytes=%llu remote_blocks=%u remote_failures=%u cpu_bytes=%llu remote_bytes=%llu newly_remote_bytes=%llu first_failed=0x%llx status=0x%x\n",
                          params->region_id,
                          params->base,
                          params->length,
                          params->block_count,
                          params->attempted_blocks,
                          params->success_blocks,
                          params->failure_blocks,
                          params->skipped_blocks,
                          params->attempted_bytes,
                          params->released_bytes,
                          params->failed_bytes,
                          params->remote_map_blocks,
                          params->remote_map_failure_blocks,
                          params->cpu_resident_bytes,
                          params->remote_mapped_bytes,
                          params->newly_remote_mapped_bytes,
                          params->first_failed_block,
                          params->rmStatus);

    return status;
}

static NV_STATUS uvm_api_moe_slot_remap_batch(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_gpu_t *gpu = NULL;
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    if (params->count > UVM_MOE_SLOT_REMAP_MAX_ENTRIES ||
        !uvm_moe_slot_remap_mode_valid(params->mode))
        return NV_ERR_INVALID_ARGUMENT;

    params->success_count = 0;
    params->failure_count = 0;
    params->conflict_count = 0;
    params->remap_count = 0;
    params->tlb_invalidate_count = 0;
    params->reserved = 0;
    params->requested_bytes = 0;
    params->accepted_bytes = 0;

    uvm_phase_f_route_log("moe_slot_remap_batch begin mode=%u count=%u validate_managed=%u first_layer=%u first_expert=%u first_slot=%u first_va=0x%llx first_bytes=%llu\n",
                          params->mode,
                          params->count,
                          params->validate_managed,
                          params->count > 0 ? params->entries[0].layer_id : 0,
                          params->count > 0 ? params->entries[0].expert_id : 0,
                          params->count > 0 ? params->entries[0].slot_id : 0,
                          params->count > 0 ? params->entries[0].va_base : 0,
                          params->count > 0 ? params->entries[0].bytes : 0);

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        NV_STATUS status = NV_OK;

        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;

        if (entry->bytes > 0)
            params->requested_bytes += entry->bytes;

        if (uvm_moe_slot_remap_entry_range_invalid(entry)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (uvm_moe_slot_remap_has_slot_conflict(params, i)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
            ++params->conflict_count;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (params->validate_managed &&
                 params->mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP) {
            status = uvm_moe_slot_remap_validate_managed_range(va_space, entry->va_base, entry->bytes);
            if (status != NV_OK)
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ADDRESS;
        }

        if (status == NV_OK && params->mode == UVM_MOE_SLOT_REMAP_MODE_DRY_RUN) {
            ++params->success_count;
            params->accepted_bytes += entry->bytes;
        }
        else if (status != NV_OK) {
            ++params->failure_count;
            if (first_error == NV_OK)
                first_error = status;
        }

        uvm_phase_f_route_log("moe_slot_remap_entry index=%u mode=%u layer=%u expert=%u slot=%u role=%u va=0x%llx bytes=%llu slot_offset=%llu flags=0x%x status=%u nv_status=0x%x\n",
                              i,
                              params->mode,
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->role,
                              entry->va_base,
                              entry->bytes,
                              entry->slot_offset,
                              entry->flags,
                              entry->status,
                              status);
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_REMAP) {
        NV_STATUS apply_status = NV_OK;

        for (i = 0; i < params->count; ++i) {
            UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
            NV_STATUS status;

            if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
                continue;

            status = uvm_moe_slot_remap_migrate_entry(params, entry, filp);
            if (status == NV_OK) {
                ++params->remap_count;
                ++params->tlb_invalidate_count;
            }
            else {
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
                ++params->failure_count;
                if (first_error == NV_OK)
                    first_error = status;
            }

            uvm_phase_f_route_log("moe_slot_remap_apply index=%u layer=%u expert=%u slot=%u va=0x%llx bytes=%llu status=%u nv_status=0x%x remap_count=%u tlb_invalidate_count=%u\n",
                                  i,
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  entry->va_base,
                                  entry->bytes,
                                  entry->status,
                                  status,
                                  params->remap_count,
                                  params->tlb_invalidate_count);
        }

        if (uvm_moe_slot_remap_has_ok_entries(params))
            apply_status = uvm_moe_slot_pool_apply_batch(params,
                                                         UVM_MOE_SLOT_POOL_COPY_NONE,
                                                         NULL,
                                                         va_space);

        if (first_error == NV_OK && apply_status != NV_OK)
            first_error = apply_status;
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP) {
        NV_STATUS apply_status = NV_OK;
        NvBool use_va_block_path = params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
                                   params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP;
        NvBool use_user_backing_path =
            params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP;

        if (uvm_moe_slot_remap_has_ok_entries(params)) {
            apply_status = uvm_moe_slot_pool_retain_gpu_by_uuid(&params->gpu_uuid, &gpu);
            if (apply_status != NV_OK) {
                uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            }
        }

        if (apply_status == NV_OK &&
            use_va_block_path &&
            first_error != NV_OK) {
            uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            apply_status = first_error;
        }

        if (apply_status == NV_OK)
            apply_status = uvm_moe_slot_pool_apply_slot_pte_batch(
                params,
                va_space,
                gpu,
                use_va_block_path,
                use_user_backing_path);

        if (gpu) {
            uvm_gpu_release(gpu);
            gpu = NULL;
        }

        if (first_error == NV_OK && apply_status != NV_OK)
            first_error = apply_status;
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_VALIDATE ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT) {
        NV_STATUS apply_status = NV_OK;
        uvm_moe_slot_pool_copy_mode_t copy_mode = UVM_MOE_SLOT_POOL_COPY_NONE;

        if (params->mode == UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY)
            copy_mode = UVM_MOE_SLOT_POOL_COPY_SYSMEM;
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY)
            copy_mode = UVM_MOE_SLOT_POOL_COPY_GPU;
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC)
            copy_mode = UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM;
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT)
            copy_mode = UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC;

        if ((copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU ||
             copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM ||
             copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC) &&
            uvm_moe_slot_remap_has_ok_entries(params)) {
            apply_status = uvm_moe_slot_pool_retain_gpu_by_uuid(&params->gpu_uuid, &gpu);
            if (apply_status != NV_OK) {
                uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED);
            }
        }

        if (apply_status == NV_OK) {
            apply_status = uvm_moe_slot_pool_apply_batch(params,
                                                         copy_mode,
                                                         gpu,
                                                         va_space);
        }

        if (gpu) {
            uvm_gpu_release(gpu);
            gpu = NULL;
        }

        if (first_error == NV_OK && apply_status != NV_OK)
            first_error = apply_status;
    }

    uvm_phase_f_route_log("moe_slot_remap_batch end mode=%u count=%u success=%u failure=%u conflicts=%u requested_bytes=%llu accepted_bytes=%llu remap_count=%u tlb_invalidate_count=%u first_error=0x%x\n",
                          params->mode,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          params->conflict_count,
                          params->requested_bytes,
                          params->accepted_bytes,
                          params->remap_count,
                          params->tlb_invalidate_count,
                          first_error);

    return first_error;
}

static NV_STATUS uvm_api_update_region_state(UVM_UPDATE_REGION_STATE_PARAMS *params,
                                             struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;

    if (params->region_id == 0 || params->state > UVM_REGION_STATE_PREFETCHING_GPU)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_space_down_write(va_space);
    region = radix_tree_lookup(&va_space->phase_f_regions, params->region_id);
    if (region)
        region->state = params->state;
    uvm_va_space_up_write(va_space);

    return region ? NV_OK : NV_ERR_OBJECT_NOT_FOUND;
}

static NV_STATUS uvm_api_query_region(UVM_QUERY_REGION_PARAMS *params,
                                      struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;

    if (params->region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    params->base = 0;
    params->length = 0;
    params->state = UVM_REGION_STATE_UNKNOWN;
    params->found_by_va = 0;

    uvm_va_space_down_read(va_space);
    region = radix_tree_lookup(&va_space->phase_f_regions, params->region_id);
    if (region)
        uvm_copy_region_to_query(region, params, 0);
    uvm_va_space_up_read(va_space);

    return region ? NV_OK : NV_ERR_OBJECT_NOT_FOUND;
}

static NV_STATUS uvm_api_query_region_by_va(UVM_QUERY_REGION_BY_VA_PARAMS *params,
                                            struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;
    uvm_range_tree_node_t *node;

    if (params->address == 0)
        return NV_ERR_INVALID_ARGUMENT;

    params->region_id = 0;
    params->base = 0;
    params->length = 0;
    params->state = UVM_REGION_STATE_UNKNOWN;
    params->found_by_va = 0;

    uvm_va_space_down_read(va_space);
    node = uvm_range_tree_find(&va_space->phase_f_region_ranges, params->address);
    region = uvm_phase_f_region_from_node(node);
    if (region)
        uvm_copy_region_to_query_by_va(region, params);
    uvm_va_space_up_read(va_space);

    return region ? NV_OK : NV_ERR_OBJECT_NOT_FOUND;
}

static long uvm_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    switch (cmd)
    {
        case UVM_DEINITIALIZE:
            return 0;

        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_INITIALIZE,                  uvm_api_initialize);
        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_MM_INITIALIZE,               uvm_api_mm_initialize);

        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PAGEABLE_MEM_ACCESS,            uvm_api_pageable_mem_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PAGEABLE_MEM_ACCESS_ON_GPU,     uvm_api_pageable_mem_access_on_gpu);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_REGISTER_GPU,                   uvm_api_register_gpu);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNREGISTER_GPU,                 uvm_api_unregister_gpu);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CREATE_RANGE_GROUP,             uvm_api_create_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DESTROY_RANGE_GROUP,            uvm_api_destroy_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ENABLE_PEER_ACCESS,             uvm_api_enable_peer_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DISABLE_PEER_ACCESS,            uvm_api_disable_peer_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_SET_RANGE_GROUP,                uvm_api_set_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CREATE_EXTERNAL_RANGE,          uvm_api_create_external_range);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MAP_EXTERNAL_ALLOCATION,        uvm_api_map_external_allocation);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MAP_EXTERNAL_SPARSE,            uvm_api_map_external_sparse);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_FREE,                           uvm_api_free);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DISCARD,                        uvm_api_discard);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PREVENT_MIGRATION_RANGE_GROUPS, uvm_api_prevent_migration_range_groups);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ALLOW_MIGRATION_RANGE_GROUPS,   uvm_api_allow_migration_range_groups);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_SET_PREFERRED_LOCATION,         uvm_api_set_preferred_location);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNSET_PREFERRED_LOCATION,       uvm_api_unset_preferred_location);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_SET_ACCESSED_BY,                uvm_api_set_accessed_by);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNSET_ACCESSED_BY,              uvm_api_unset_accessed_by);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_REGISTER_GPU_VASPACE,           uvm_api_register_gpu_va_space);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNREGISTER_GPU_VASPACE,         uvm_api_unregister_gpu_va_space);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_REGISTER_CHANNEL,               uvm_api_register_channel);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNREGISTER_CHANNEL,             uvm_api_unregister_channel);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ENABLE_READ_DUPLICATION,        uvm_api_enable_read_duplication);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DISABLE_READ_DUPLICATION,       uvm_api_disable_read_duplication);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MIGRATE,                        uvm_api_migrate);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_READ_PROCESS_MEMORY,      uvm_api_tools_read_process_memory);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_WRITE_PROCESS_MEMORY,     uvm_api_tools_write_process_memory);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_GET_PROCESSOR_UUID_TABLE, uvm_api_tools_get_processor_uuid_table);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MAP_DYNAMIC_PARALLELISM_REGION, uvm_api_map_dynamic_parallelism_region);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNMAP_EXTERNAL,                 uvm_api_unmap_external);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MIGRATE_RANGE_GROUP,            uvm_api_migrate_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_FLUSH_EVENTS,             uvm_api_tools_flush_events);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_ALLOC_SEMAPHORE_POOL,           uvm_api_alloc_semaphore_pool);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CLEAN_UP_ZOMBIE_RESOURCES,      uvm_api_clean_up_zombie_resources);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_POPULATE_PAGEABLE,              uvm_api_populate_pageable);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_VALIDATE_VA_RANGE,              uvm_api_validate_va_range);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2,uvm_api_tools_get_processor_uuid_table_v2);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ALLOC_DEVICE_P2P,               uvm_api_alloc_device_p2p);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CLEAR_ALL_ACCESS_COUNTERS,      uvm_api_clear_all_access_counters);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_PHASE_F_BATCH_EXECUTOR,         uvm_api_phase_f_batch_executor);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PHASE_F_REGISTER_REGION,        uvm_api_phase_f_register_region);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PHASE_F_UNREGISTER_REGION,      uvm_api_phase_f_unregister_region);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_PHASE_F_BATCH_REGIONS,          uvm_api_phase_f_batch_regions);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UPDATE_REGION_STATE,             uvm_api_update_region_state);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_QUERY_REGION,                    uvm_api_query_region);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_QUERY_REGION_BY_VA,              uvm_api_query_region_by_va);
        UVM_ROUTE_CMD_HEAP_INIT_CHECK(UVM_EXPERT_OFFLOAD_RELEASE,            uvm_api_expert_offload_release);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MOE_SLOT_REMAP_BATCH,            uvm_api_moe_slot_remap_batch);
    }

    // Try the test ioctls if none of the above matched
    return uvm_test_ioctl(filp, cmd, arg);
}

static long uvm_unlocked_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    long ret;

    if (!uvm_down_read_trylock(&g_uvm_global.pm.lock))
        return -EAGAIN;

    ret = uvm_ioctl(filp, cmd, arg);

    uvm_up_read(&g_uvm_global.pm.lock);

    uvm_thread_assert_all_unlocked();

    return ret;
}

static long uvm_unlocked_ioctl_entry(struct file *filp, unsigned int cmd, unsigned long arg)
{
   UVM_ENTRY_RET(uvm_unlocked_ioctl(filp, cmd, arg));
}

static const struct file_operations uvm_fops =
{
    .open            = uvm_open_entry,
    .release         = uvm_release_entry,
    .mmap            = uvm_mmap_entry,
    .unlocked_ioctl  = uvm_unlocked_ioctl_entry,
#if NVCPU_IS_X86_64
    .compat_ioctl    = uvm_unlocked_ioctl_entry,
#endif
    .owner           = THIS_MODULE,
};

NV_STATUS uvm_test_register_unload_state_buffer(UVM_TEST_REGISTER_UNLOAD_STATE_BUFFER_PARAMS *params, struct file *filp)
{
    long ret;
    struct page *page;
    NV_STATUS status = NV_OK;

    if (!IS_ALIGNED(params->unload_state_buf, sizeof(NvU64)))
        return NV_ERR_INVALID_ADDRESS;

    // Hold mmap_lock to call get_user_pages(), the UVM locking helper functions
    // are not used because unload_state_buf may be a managed memory pointer and
    // therefore a locking assertion from the CPU fault handler could be fired.
    nv_mmap_read_lock(current->mm);
    ret = NV_PIN_USER_PAGES(params->unload_state_buf, 1, FOLL_WRITE, &page);
    nv_mmap_read_unlock(current->mm);

    if (ret < 0)
        return errno_to_nv_status(ret);
    UVM_ASSERT(ret == 1);

    uvm_mutex_lock(&g_uvm_global.global_lock);

    if (g_uvm_global.unload_state.ptr) {
        NV_UNPIN_USER_PAGE(page);
        status = NV_ERR_IN_USE;
        goto error;
    }

    g_uvm_global.unload_state.page = page;
    g_uvm_global.unload_state.ptr = (NvU64 *)((char *)kmap(page) + (params->unload_state_buf & ~PAGE_MASK));
    *g_uvm_global.unload_state.ptr = 0;

error:
    uvm_mutex_unlock(&g_uvm_global.global_lock);

    return status;
}

static void uvm_test_unload_state_exit(void)
{
    if (g_uvm_global.unload_state.ptr) {
        kunmap(g_uvm_global.unload_state.page);
        NV_UNPIN_USER_PAGE(g_uvm_global.unload_state.page);
    }
}

static int uvm_chardev_create(void)
{
    dev_t uvm_dev;

    int ret = alloc_chrdev_region(&g_uvm_base_dev,
                                  0,
                                  NVIDIA_UVM_NUM_MINOR_DEVICES,
                                  NVIDIA_UVM_DEVICE_NAME);
    if (ret != 0) {
        UVM_ERR_PRINT("alloc_chrdev_region failed: %d\n", ret);
        return ret;
    }
    uvm_dev = MKDEV(MAJOR(g_uvm_base_dev), NVIDIA_UVM_PRIMARY_MINOR_NUMBER);

    uvm_init_character_device(&g_uvm_cdev, &uvm_fops);
    ret = cdev_add(&g_uvm_cdev, uvm_dev, 1);
    if (ret != 0) {
        UVM_ERR_PRINT("cdev_add (major %u, minor %u) failed: %d\n", MAJOR(uvm_dev), MINOR(uvm_dev), ret);
        unregister_chrdev_region(g_uvm_base_dev, NVIDIA_UVM_NUM_MINOR_DEVICES);
        return ret;
    }

    return 0;
}

static void uvm_chardev_exit(void)
{
    cdev_del(&g_uvm_cdev);
    unregister_chrdev_region(g_uvm_base_dev, NVIDIA_UVM_NUM_MINOR_DEVICES);
}

static int uvm_init(void)
{
    bool initialized_globals = false;
    bool added_device = false;
    int ret;

    NV_STATUS status = uvm_global_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_global_init() failed: %s\n", nvstatusToString(status));
        ret = -ENODEV;
        goto error;
    }
    initialized_globals = true;

    ret = uvm_chardev_create();
    if (ret != 0) {
        UVM_ERR_PRINT("uvm_chardev_create failed: %d\n", ret);
        goto error;
    }
    added_device = true;

    ret = uvm_tools_init(g_uvm_base_dev);
    if (ret != 0) {
        UVM_ERR_PRINT("uvm_tools_init() failed: %d\n", ret);
        goto error;
    }

    if (uvm_enable_builtin_tests)
        UVM_INFO_PRINT("Built-in UVM tests are enabled. This is a security risk.\n");

    return 0;

error:
    if (added_device)
        uvm_chardev_exit();

    if (initialized_globals)
        uvm_global_exit();

    UVM_ERR_PRINT("uvm init failed: %d\n", ret);

    return ret;
}

static int __init uvm_init_entry(void)
{
   UVM_ENTRY_RET(uvm_init());
}

static void uvm_exit(void)
{
    uvm_tools_exit();
    uvm_chardev_exit();

    uvm_moe_slot_pool_reset();
    uvm_global_exit();

    uvm_test_unload_state_exit();
}

static void __exit uvm_exit_entry(void)
{
   UVM_ENTRY_VOID(uvm_exit());
}

module_init(uvm_init_entry);
module_exit(uvm_exit_entry);

MODULE_LICENSE("Dual MIT/GPL");
MODULE_INFO(supported, "external");
MODULE_VERSION(NV_VERSION_STRING);
MODULE_DESCRIPTION("NVIDIA Unified Virtual Memory kernel module");
