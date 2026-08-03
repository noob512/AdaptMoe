NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib64

NVCCFLAGS = -Xcompiler -fPIC -O2 -std=c++17

all: uvm uvm_to_uvm_reuse_probe

uvm: uvm_allocator.so

uvm_allocator.so: uvm_allocator.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -shared -o $@ $< -lcudart

uvm_to_uvm_reuse_probe: uvm_to_uvm_reuse_probe.cu uvm_allocator.so
	$(NVCC) -O2 -std=c++17 -I$(CUDA_INCLUDE) -L. \
		-Xlinker -rpath,'$$ORIGIN' -o $@ $< \
		-l:uvm_allocator.so -L$(CUDA_LIB) -lcudart -ldl

clean:
	rm -f uvm_allocator.so uvm_to_uvm_reuse_probe

.PHONY: all uvm clean
