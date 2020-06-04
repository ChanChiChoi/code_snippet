NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11

INC = -I../common/inc
LD_LIB = -lcuda

all: clock


clock:clock.cu
	$(NVCC) $(NVCC_FLAGS) $(INC)  $^ -o $@
