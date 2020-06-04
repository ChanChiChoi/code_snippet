NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS =  -std=c++11 -arch=sm_35
INC = -I../common/inc
EXTRA_FLAGS = -dc  -m64
LD_FLAGS = -lcudadevrt

OBJ = cdpSimpleQuicksort

all:$(OBJ)

%:%.o
	$(NVCC) $(NVCC_FLAGS) $(LD_FLAGS) $< -o $@

%.o:%.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) $(EXTRA_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJ)
	rm -f *.o

