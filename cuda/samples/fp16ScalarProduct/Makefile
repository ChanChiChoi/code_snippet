NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
EXTRA_FLSGS = -arch=sm_53
INCLUDE_PATH = -I../common/inc -I/usr/local/cuda/include

OBJ = fp16ScalarProduct


all:$(OBJ)

$(OBJ):%:%.cu
	$(NVCC) $(NVCC_FLAGS) $(EXTRA_FLSGS) $(INCLUDE_PATH) $^ -o $@

clean:
	rm -f *.o
	rm -f $(OBJ)
