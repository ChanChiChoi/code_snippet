NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
INCLUDE_PATH = -I../common/inc
CC = mpicxx
CC_FLAG = -std=c++11
LD_PATH = -L/usr/local/cuda/lib64 -lcudart


OBJ = simpleMPI
NP = 4


all:$(OBJ)


$(OBJ):$(addsuffix .o,$(OBJ)) $(addsuffix _mpi.o,$(OBJ))
	$(CC) $(CC_FLAG) $(LD_PATH) $^ -o $@ 

$(addsuffix .o,$(OBJ)):$(addsuffix .cu,$(OBJ))
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $^ -o $@

$(addsuffix _mpi.o,$(OBJ)):$(addsuffix .cpp,$(OBJ))
	$(CC) $(CC_FLAG) -c $^ -o $@

run:
	mpirun -np $(NP) ./simpleMPI


clean:
	rm -f $(OBJ)
	rm -f *.o



