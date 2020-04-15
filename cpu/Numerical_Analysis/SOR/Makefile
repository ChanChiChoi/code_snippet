CC = g++
CC_FLAGS = -g -Wall -std=c++11
INCLUDE_PATH = -I../../eigen-3.3.7

OBJ = SOR

all:$(OBJ)


$(OBJ):%:%.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_PATH) $^ -o $@



clean:
	rm -f $(OBJ)
