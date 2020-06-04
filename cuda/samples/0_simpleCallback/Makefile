NVCC = /usr/local/cuda/bin/nvcc
# 假设当前机器是p40，那就不要设成arch=sm_70,否则运行结果是错的
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11 -arch=sm_50
INCLUDE_PATH = -I../common/inc

OBJ = simpleCallback

all:$(OBJ)

$(OBJ):simpleCallback.o multithreading.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@


simpleCallback.o:simpleCallback.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $^ -o $@

multithreading.o:multithreading.cpp
	$(NVCC)  $(NVCC_FLAGS)  -c $^ -o $@


clean:
	rm -f *.o
	rm -f $(OBJ)

