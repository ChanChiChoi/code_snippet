NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
INCLUDE_PATH = -I../common/inc


OBJ = simpleAtomicIntrinsics

all:$(OBJ)


$(OBJ):simpleAtomicIntrinsics.cu simpleAtomicIntrinsics_cpu.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) $^ -o $@



clean:
	rm -f $(OBJ)

