NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all:unique normal


unique: uniquePtr.cu
	$(NVCC) $(NVCC_FLAGS) -D UNIQUE $^ -o $@

normal:uniquePtr.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@


run:
	cuda-memcheck --leak-check full ./unique
	cuda-memcheck --leak-check full ./normal

clean:
	rm -f unique normal
