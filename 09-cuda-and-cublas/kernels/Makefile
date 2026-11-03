# Build the CUDA exercises.  `make` builds everything; `make run` builds and runs.
#
# ARCH: set to your GPU's compute capability to avoid JIT on first launch, e.g.
#   make ARCH=sm_86     (Ampere consumer)     make ARCH=sm_90   (Hopper)
# `native` asks nvcc to detect it, and needs CUDA 11.5 or newer.
ARCH    ?= native
NVCC    ?= nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo

BINS = 01_vector_add 02_sgemm 03_softmax 04_attention

all: $(BINS)

02_sgemm: 02_sgemm.cu common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcublas

%: %.cu common.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: all
	@for b in $(BINS); do echo "=== $$b ==="; ./$$b || exit 1; echo; done

clean:
	rm -f $(BINS) *.exe *.exp *.lib *.pdb

.PHONY: all run clean
