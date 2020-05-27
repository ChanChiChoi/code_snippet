NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
NVCC_EXTRA_FLAGS = -dc
INCLUDE_PATH = -I../common/inc
CC = g++
CC_FLAG = -std=c++11
LD_PATH = -L/usr/local/cuda/lib64 -lcudart


OBJ = simpleSeparateCompilation
SRC = simpleDeviceLibrary simpleSeparateCompilation
MID = $(addsuffix .o,${SRC})

all:${OBJ}

${OBJ}:${MID}
	${NVCC} $^ -o $@


%.o:%.cu
	${NVCC} ${NVCC_FLAGS} ${INCLUDE_PATH} ${NVCC_EXTRA_FLAGS} $< -o $@

run:
	 ./${OBJ}


clean:
	rm -f $(OBJ)
	rm -f *.o



