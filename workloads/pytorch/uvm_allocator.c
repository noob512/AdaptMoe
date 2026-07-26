/*
 * CUDA UVM Allocator for PyTorch — V1 (Plain cudaMallocManaged)
 *
 * !!!! DO NOT MODIFY THIS FILE !!!!
 *
 * This allocator MUST remain as plain cudaMallocManaged WITHOUT any
 * cudaMemPrefetchAsync or cudaMemAdvise calls. This is the canonical
 * "without-user-prefetch" baseline used in all paper experiments.
 *
 * - cudaMemPrefetchAsync changes the lazy-migration baseline
 * - cudaMemAdviseSetPreferredLocation=CPU causes 2x regression by forcing
 *   evicted pages back to CPU, doubling migration traffic
 *
 * Paper baseline: 10M nodes = ~70s/epoch (UVM lazy migration)
 * If you need a different allocator variant, create a SEPARATE file.
 *
 * !!!! DO NOT MODIFY THIS FILE !!!!
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdatomic.h>

// Global memory statistics
static atomic_size_t total_allocated = 0;
static atomic_size_t peak_allocated = 0;
static atomic_size_t num_allocs = 0;
static atomic_size_t num_frees = 0;

// Allocate CUDA managed memory
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    void* ptr = NULL;
    cudaError_t err;

    // Use cudaMallocManaged for UVM — NO prefetch, NO advise
    err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);

    if (err != cudaSuccess) {
        fprintf(stderr, "[UVM] cudaMallocManaged failed: %s\n",
                cudaGetErrorString(err));
        return NULL;
    }

    // Update statistics
    size_t current = atomic_fetch_add(&total_allocated, size) + size;
    size_t alloc_count = atomic_fetch_add(&num_allocs, 1) + 1;

    // Update peak if needed
    size_t peak = atomic_load(&peak_allocated);
    while (current > peak) {
        if (atomic_compare_exchange_weak(&peak_allocated, &peak, current)) {
            break;
        }
    }

    // Log large allocations
    if (size > 1000 * 1024 * 1024) { // > 1GB
        fprintf(stderr, "[UVM] Alloc #%zu: %.2f GB (total: %.2f GB, peak: %.2f GB)\n",
                alloc_count, size / 1e9, current / 1e9, atomic_load(&peak_allocated) / 1e9);
    }

    return ptr;
}

// Free CUDA managed memory
void uvm_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr != NULL) {
        cudaFree(ptr);

        // Update statistics
        atomic_fetch_sub(&total_allocated, size);
        atomic_fetch_add(&num_frees, 1);
    }
}

// Get current allocated bytes
size_t uvm_get_allocated_bytes(void) {
    return atomic_load(&total_allocated);
}

// Get peak allocated bytes
size_t uvm_get_peak_allocated_bytes(void) {
    return atomic_load(&peak_allocated);
}

// Get allocation count
size_t uvm_get_num_allocs(void) {
    return atomic_load(&num_allocs);
}

// Get free count
size_t uvm_get_num_frees(void) {
    return atomic_load(&num_frees);
}

// Reset peak statistics
void uvm_reset_peak_stats(void) {
    atomic_store(&peak_allocated, atomic_load(&total_allocated));
}
