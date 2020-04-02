CC = g++
NVCC = /usr/local/cuda/bin/nvcc
INCLUDE_PATH = -I/usr/local/cuda/include -I../common/inc
NVCC_FLAGS = -std=c++11 -g -G -Xcompiler -Wall

OBJ = cppIntegration
all:$(OBJ)

$(OBJ):cppIntegration.o cppIntegration_gold.o main.o
	$(NVCC) $^ -o $@

main.o:main.cpp
	$(NVCC) $(INCLUDE_PATH) -c $< -o $@

cppIntegration.o:cppIntegration.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $< -o $@

cppIntegration_gold.o:cppIntegration_gold.cpp
	$(NVCC) -c $< -o $@


clean:
	rm -f *.o
	rm -f $(OBJ)
