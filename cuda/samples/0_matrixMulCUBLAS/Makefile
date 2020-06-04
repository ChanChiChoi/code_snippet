NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
INCLUDE_PATH = -I../common/inc
LDLIBRARY_PATH = -lcublas

OBJ = matrixMulCUBLAS

all:$(OBJ)

$(OBJ):%:%.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) $(LDLIBRARY_PATH)  $^ -o $@





clean:
	rm -f $(OBJ)
