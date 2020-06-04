NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11 -arch sm_60 -Xcompiler -fopenmp
INCLUDE_PATH = -I../common/inc
CC = g++
CC_FLAG = -std=c++11
LD_PATH = -L/usr/local/cuda/lib64 -lcudart -lcublas


OBJ = UnifiedMemoryStreams


all:$(OBJ)

${OBJ}:%:%.cu
	${NVCC} ${NVCC_FLAGS} ${INCLUDE_PATH} ${LD_PATH} $^ -o $@

run:
	 @./${OBJ}


clean:
	rm -f $(OBJ)
	rm -f *.o



