CC = g++
INCLUDE_PATH = -I../common/inc -I/usr/local/cuda/include
NVCC = /usr/local/cuda/bin/nvcc
LD_PATH =  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs
LD_FLAGS = -lcuda -lnvrtc

OBJ = clock_nvrtc
MID = $(addsuffix .o, $(OBJ))
all: $(OBJ)

$(OBJ):$(MID)
	$(CC) $(LD_PATH) $(LD_FLAGS) $< -o $@
$(MID):clock_nvrtc.cpp
	$(CC) $(INCLUDE_PATH) -c $< -o $@


clean:
	rm -f *.o
	rm -f $(OBJ)
