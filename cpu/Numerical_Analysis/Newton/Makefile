CC = g++
CCFLAGS = -g -Wall -std=c++11

OBJ = Newton

all:$(OBJ)

$(OBJ):%:%.cpp
	$(CC) $(CCFLAGS) $^ -o $@


clean:
	rm -f $(OBJ)
