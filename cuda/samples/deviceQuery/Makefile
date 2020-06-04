NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std=c++11
INC = ../common/inc

INC := $(addprefix -I,$(INC))

TARGETS = deviceQuery

all:$(TARGETS)

$(TARGETS):%:%.o
	$(NVCC) $(NVCC_LDFLAGS) $< -o $@

%.o:%.cpp
	$(NVCC) $(NVCC_FLAGS) $(INC) $< -c -o $@


clean:
	rm -f $(TARGETS)
	rm -f *.o
