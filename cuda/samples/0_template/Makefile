NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11 -arch sm_60
INCLUDE_PATH = -I../common/inc
CC = g++
CC_FLAG = -std=c++11
LD_PATH = -L/usr/local/cuda/lib64 -lcudart


OBJ = template


all:$(OBJ)

${OBJ}:template_cpu.o template.o
	${NVCC} $^ -o $@

#template_cpu.o:template_cpu.cpp
#	${NVCC} ${NVCC_FLAGS} -c $^ -o $@

%.o:%.cu
	${NVCC} ${NVCC_FLAGS} ${INCLUDE_PATH} -c $^ -o $@

run:
	 @./${OBJ}


clean:
	rm -f $(OBJ)
	rm -f *.o



