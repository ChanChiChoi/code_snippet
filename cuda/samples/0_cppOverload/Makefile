CC = g++
NVCC = /usr/local/cuda/bin/nvcc
INCLUDE_PATH = -I../common/inc #-I/usr/local/cuda/include
NVCC_FLAGS = -std=c++11 -g -G -Xcompiler -Wall -arch=sm_60

OBJ = cppOverload
all:$(OBJ)

$(OBJ):cppOverload.cu kernel_overload.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH)  $< -o $@

run:
	cuda-memcheck --leak-check full ./$(OBJ)

clean:
	rm -f *.o
	rm -f $(OBJ)
