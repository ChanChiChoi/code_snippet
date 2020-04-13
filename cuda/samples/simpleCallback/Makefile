NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
INCLUDE_PATH = -I../common/inc

OBJ = simpleCallback

all:$(OBJ)

$(OBJ):simpleCallback.o multithreading.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@


simpleCallback.o:simpleCallback.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $^ -o $@

multithreading.o:multithreading.cpp
	$(NVCC) -ccbin g++ -std=c++11 -c $^ -o $@


clean:
	rm -f *.o
	rm -f $(OBJ)

