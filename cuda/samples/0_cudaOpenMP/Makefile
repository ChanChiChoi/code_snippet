NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
OPENMP = -Xcompiler -fopenmp
INCLUDE_PATH = -I../common/inc

OBJ = cudaOpenMP

all:$(OBJ)


$(OBJ):%:%.cu
	$(NVCC) $(NVCC_FLAGS) $(OPENMP) $(INCLUDE_PATH) $^ -o $@


run:
	cuda-memcheck --leak-check full ./$(OBJ)

clean:
	rm -f $(OBJ)
