NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -ccbin g++ -std=c++11 -gencode arch=compute_60,code=sm_60
INC = -I../common/inc
EXTRA_FLAGS = -dc 
LD_FLAGS = -lcudadevrt

all:cdpSimplePrint

%:%.o
	$(NVCC) $(NVCC_FLAGS) $(LD_FLAGS) $< -o $@

%.o:%.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) $(EXTRA_FLAGS) -c $< -o $@

clean:
	rm -f cdpSimplePrint
	rm -f *.o

