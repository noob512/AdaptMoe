# Makefile for UVM allocator and LD_PRELOAD interceptor
#
# Usage:
#   make              # Build all
#   make uvm          # Build PyTorch pluggable allocator
#   make preload      # Build LD_PRELOAD interceptor
#   make budget       # Build managed-budget LD_PRELOAD helper and probe
#   make clean        # Clean build artifacts
#
# To use the LD_PRELOAD interceptor:
#   LD_PRELOAD=./libcudamalloc_managed.so python your_script.py

NVCC ?= nvcc
CXX ?= g++

# CUDA paths
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib64

# Compiler flags
CXXFLAGS = -fPIC -O2 -std=c++17
NVCCFLAGS = -Xcompiler -fPIC -O2 -std=c++17
LDFLAGS = -shared -L$(CUDA_LIB) -lcudart -ldl

# Targets
all: uvm_allocator.so libcudamalloc_managed.so libuvm_managed_budget.so phase_f_smoke managed_pool_probe

# PyTorch pluggable allocator (same as before)
uvm: uvm_allocator.so

uvm_allocator.so: uvm_allocator.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -shared -o $@ $< -lcudart

phase_f_smoke: phase_f_smoke.cpp uvm_allocator.so
	$(CXX) -O2 -std=c++17 -I$(CUDA_INCLUDE) -L. -Wl,-rpath,'$$ORIGIN' -o $@ $< -l:uvm_allocator.so -L$(CUDA_LIB) -lcudart -ldl

phase_n_operator_sim: phase_n_operator_sim.cu uvm_allocator.so
	$(NVCC) -O2 -std=c++17 -I$(CUDA_INCLUDE) -L. -Xlinker -rpath,'$$ORIGIN' -o $@ $< -l:uvm_allocator.so -L$(CUDA_LIB) -lcudart -ldl

phase_n_object_mapping_smoke: phase_n_object_mapping_smoke.cu uvm_allocator.so
	$(NVCC) -O2 -std=c++17 -I$(CUDA_INCLUDE) -L. -Xlinker -rpath,'$$ORIGIN' -o $@ $< -l:uvm_allocator.so -L$(CUDA_LIB) -lcudart -ldl

phase_n_3gb_rebalance_smoke: phase_n_3gb_rebalance_smoke.cu uvm_allocator.so
	$(NVCC) -O2 -std=c++17 -I$(CUDA_INCLUDE) -L. -Xlinker -rpath,'$$ORIGIN' -o $@ $< -l:uvm_allocator.so -L$(CUDA_LIB) -lcudart -ldl

# LD_PRELOAD library to intercept all cudaMalloc calls
preload: libcudamalloc_managed.so

libcudamalloc_managed.so: cuda_malloc_managed_preload.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I$(CUDA_INCLUDE) -o $@ $<

budget: libuvm_managed_budget.so managed_pool_probe

libuvm_managed_budget.so: cuda_managed_budget_preload.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I$(CUDA_INCLUDE) -o $@ $<

managed_pool_probe: managed_pool_probe.cu
	$(NVCC) -O2 -std=c++17 -o $@ $< -lcudart

clean:
	rm -f uvm_allocator.so libcudamalloc_managed.so libuvm_managed_budget.so phase_f_smoke phase_n_operator_sim phase_n_object_mapping_smoke phase_n_3gb_rebalance_smoke managed_pool_probe

.PHONY: all uvm preload budget clean
