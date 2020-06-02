NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
INCLUDE_PATH = -I../common/inc
CC = g++
CC_FLAG = -std=c++11
LD_PATH = -L/usr/local/cuda/lib64 -lcudart


OBJ = simpleZeroCopy


all:$(OBJ)

${OBJ}:%:%.cu
	${NVCC} ${NVCC_FLAGS} ${INCLUDE_PATH} $^ -o $@

run:
	 @./${OBJ}


clean:
	rm -f $(OBJ)
	rm -f *.o



