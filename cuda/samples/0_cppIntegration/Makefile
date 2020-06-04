CC = g++
NVCC = /usr/local/cuda/bin/nvcc
INCLUDE_PATH = -I../common/inc #-I/usr/local/cuda/include
NVCC_FLAGS = -std=c++11 -g -G -Xcompiler -Wall -arch=sm_60

OBJ = cppIntegration
all:$(OBJ)

$(OBJ):cppIntegration.o cppIntegration_gold.o main.o
	$(NVCC) $^ -o $@

main.o:main.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $< -o $@

cppIntegration.o:cppIntegration.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $< -o $@

cppIntegration_gold.o:cppIntegration_gold.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


clean:
	rm -f *.o
	rm -f $(OBJ)
